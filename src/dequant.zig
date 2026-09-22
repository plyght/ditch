//! Dequantisation of quantised safetensors checkpoints on read.
//!
//! A quantised weight is stored as several tensors (codes plus scales, zero
//! points or shapes) that no kernel can use directly. `register` finds every
//! such group in the opened files and replaces it with one virtual BF16 tensor
//! (`safetensors.Overlay.dequant`) under the name the model expects, so the
//! loader, the weight store, the kernels and the exporter see an ordinary
//! bf16 checkpoint. The bytes of a virtual tensor are produced on demand:
//! in streamed mode a positional read of a row range decodes just those rows
//! (chunk-wise, bounded by `chunk_bytes` of source data or one expert slab),
//! in mapped mode the tensor is decoded once at load into a resident buffer.
//!
//! Formats:
//! * FP8 (DeepSeek V3, Kimi K2, `quant_method = "fp8"`; also compressed-tensors
//!   `float-quantized` and `fbgemm_fp8`): `weight` F8_E4M3 or F8_E5M2 with a
//!   float `weight_scale_inv` (or `weight_scale`) holding one scale per block
//!   of `weight_block_size` elements (per row or per tensor when the scale has
//!   that shape). `w = code * scale`.
//! * MXFP4 (gpt-oss, `quant_method = "mxfp4"`): `*_blocks` U8 `[..., R, C/32, 16]`
//!   holding E2M1 nibble pairs (low nibble first) and `*_scales` U8
//!   `[..., R, C/32]` E8M0 exponents (bias 127). The bf16 tensor is the
//!   `[..., C, R]` transpose, the layout of the bf16 gpt-oss checkpoints.
//! * compressed-tensors `pack-quantized` (Kimi K2.5): `weight_packed` I32 with
//!   `num_bits`-wide fields packed densely from the low end, `weight_scale`
//!   per group of `group_size` columns (or per row / per tensor),
//!   `weight_zero_point` packed the same way along the rows (asymmetric
//!   schemes only) and `weight_shape` I64. `w = (q - zp) * scale`.
//!
//! Every value is computed in f32 and rounded to bf16, which is what the
//! Hugging Face integrations produce when they dequantise to `bfloat16`.

const std = @import("std");
const Io = std.Io;
const tensor = @import("tensor.zig");
const safetensors = @import("safetensors.zig");

const Allocator = std.mem.Allocator;
const DType = tensor.DType;

/// Source bytes decoded per step in streamed mode (plus their scales).
pub const chunk_bytes: usize = 4 << 20;

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

pub const Method = enum { none, fp8, mxfp4, int_packed };

/// The `quantization_config` of a config.json, reduced to what the decoders need.
pub const QuantConfig = struct {
    method: Method = .none,
    /// FP8: elements per scale along rows and columns (`weight_block_size`); 0 = from the scale shape.
    block_rows: usize = 0,
    block_cols: usize = 0,
    /// pack-quantized: field width and columns per scale (0 = from the scale shape).
    num_bits: u8 = 4,
    group_size: usize = 0,
    /// Human-readable format name for messages.
    label: []const u8 = "none",
};

fn objStr(obj: std.json.ObjectMap, key: []const u8) ?[]const u8 {
    const v = obj.get(key) orelse return null;
    return if (v == .string) v.string else null;
}

fn objInt(obj: std.json.ObjectMap, key: []const u8) ?usize {
    const v = obj.get(key) orelse return null;
    return switch (v) {
        .integer => |i| if (i >= 0) @as(usize, @intCast(i)) else null,
        .float => |f| if (f >= 0) @as(usize, @intFromFloat(f)) else null,
        else => null,
    };
}

/// Parses `quantization_config` (the object under that key of a config.json,
/// or null when absent). Unknown formats are an error naming the format:
/// ditch never guesses at a checkpoint's encoding.
pub fn parseQuantConfig(qc: ?std.json.ObjectMap) !QuantConfig {
    const obj = qc orelse return .{};
    const method = objStr(obj, "quant_method") orelse "";
    const format = objStr(obj, "format") orelse "";
    if (std.mem.eql(u8, method, "fp8") or std.mem.eql(u8, method, "fbgemm_fp8")) {
        var out = QuantConfig{ .method = .fp8, .label = "fp8" };
        if (obj.get("weight_block_size")) |bs| {
            if (bs == .array and bs.array.items.len == 2 and bs.array.items[0] == .integer and bs.array.items[1] == .integer) {
                out.block_rows = @intCast(bs.array.items[0].integer);
                out.block_cols = @intCast(bs.array.items[1].integer);
            }
        }
        return out;
    }
    if (std.mem.eql(u8, method, "mxfp4")) return .{ .method = .mxfp4, .label = "mxfp4" };
    if (std.mem.eql(u8, method, "compressed-tensors")) {
        if (std.mem.eql(u8, format, "float-quantized")) return .{ .method = .fp8, .label = "compressed-tensors float-quantized" };
        if (std.mem.eql(u8, format, "dense") or format.len == 0) return .{};
        if (std.mem.eql(u8, format, "pack-quantized")) {
            var out = QuantConfig{ .method = .int_packed, .label = "compressed-tensors pack-quantized" };
            if (obj.get("config_groups")) |groups| {
                if (groups == .object) {
                    var it = groups.object.iterator();
                    while (it.next()) |g| {
                        if (g.value_ptr.* != .object) continue;
                        const w = g.value_ptr.object.get("weights") orelse continue;
                        if (w != .object) continue;
                        const t = objStr(w.object, "type") orelse "int";
                        if (!std.mem.eql(u8, t, "int")) {
                            std.log.err("unsupported model: compressed-tensors pack-quantized weights of type '{s}' (only int is supported)", .{t});
                            return error.UnsupportedArchitecture;
                        }
                        if (objInt(w.object, "num_bits")) |b| {
                            if (b == 0 or b > 8) {
                                std.log.err("unsupported model: {d}-bit pack-quantized weights (1 to 8 bits are supported)", .{b});
                                return error.UnsupportedArchitecture;
                            }
                            out.num_bits = @intCast(b);
                        }
                        if (objInt(w.object, "group_size")) |g_| out.group_size = g_;
                        break;
                    }
                }
            }
            return out;
        }
        std.log.err("unsupported model: compressed-tensors format '{s}' cannot be dequantised (pack-quantized and float-quantized are supported)", .{format});
        return error.UnsupportedArchitecture;
    }
    const shown = if (method.len > 0) method else if (format.len > 0) format else "unknown";
    std.log.err("unsupported model: '{s}' quantised weights cannot be dequantised (fp8, mxfp4 and compressed-tensors pack-quantized/float-quantized are supported)", .{shown});
    return error.UnsupportedArchitecture;
}

/// Whether an `expert_dtype` config value names a storage format ditch reads.
pub fn expertDtypeSupported(dt: []const u8) bool {
    const known = [_][]const u8{ "bfloat16", "float32", "float16", "bf16", "fp32", "fp16", "fp8", "float8", "fp8_e4m3", "float8_e4m3fn", "e4m3", "mxfp4", "int4", "pack-quantized" };
    for (known) |k| if (std.ascii.eqlIgnoreCase(dt, k)) return true;
    return false;
}

// ---------------------------------------------------------------------------
// Element decoders
// ---------------------------------------------------------------------------

/// `2^e` as f32 for normal exponents.
inline fn pow2(e: i32) f32 {
    return @bitCast(@as(u32, @intCast(e + 127)) << 23);
}

fn fp8Table(comptime exp_bits: u4) [256]f32 {
    @setEvalBranchQuota(100_000);
    const man_bits = 7 - exp_bits;
    const bias = (1 << (exp_bits - 1)) - 1;
    var t: [256]f32 = undefined;
    for (0..256) |code| {
        const sign = code >> 7;
        const exp: i32 = @intCast((code >> man_bits) & ((1 << exp_bits) - 1));
        const man: f32 = @floatFromInt(code & ((1 << man_bits) - 1));
        const man_scale: f32 = 1.0 / @as(f32, @floatFromInt(1 << man_bits));
        var v: f32 = undefined;
        if (exp == 0) {
            v = man * man_scale * pow2(1 - bias);
        } else if (exp_bits == 4 and exp == 15 and man == 7) {
            v = std.math.nan(f32); // E4M3 (fn): the all-ones code is NaN, there are no infinities.
        } else if (exp_bits == 5 and exp == 31) {
            v = if (man == 0) std.math.inf(f32) else std.math.nan(f32);
        } else {
            v = (1.0 + man * man_scale) * pow2(exp - bias);
        }
        t[code] = if (sign == 1) -v else v;
    }
    return t;
}

pub const e4m3_table = fp8Table(4);
pub const e5m2_table = fp8Table(5);

/// E2M1 values by nibble (sign in bit 3).
pub const e2m1_table = [16]f32{ 0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6 };

/// Field `i` of `bits` bits in a little-endian word stream (`pack_to_int32`).
pub inline fn unpackField(words: []const u32, i: usize, bits: u8) u32 {
    const start = i * bits;
    const w = start / 32;
    const off: u5 = @intCast(start % 32);
    const lo: u8 = @min(@as(u8, @intCast(32 - @as(u32, off))), bits);
    var v = (words[w] >> off) & fieldMask(lo);
    if (lo < bits) v |= (words[w + 1] & fieldMask(bits - lo)) << @intCast(lo);
    return v;
}

inline fn fieldMask(bits: u8) u32 {
    return if (bits >= 32) 0xffff_ffff else (@as(u32, 1) << @intCast(bits)) - 1;
}

// ---------------------------------------------------------------------------
// Virtual tensors
// ---------------------------------------------------------------------------

/// A byte range of a source file.
const Piece = struct {
    file: *safetensors.File,
    offset: u64,
    byte_len: usize,
    dtype: DType = .f32,

    fn read(self: Piece, io: Io, rel: u64, out: []u8) !void {
        std.debug.assert(rel + out.len <= self.byte_len);
        try self.file.readRange(io, self.offset + rel, out);
    }
};

/// One virtual bf16 tensor: `slabs` matrices of `rows` x `cols` (a slab is
/// one expert of a stacked expert tensor; plain matrices have one slab).
pub const Dequant = struct {
    name: []const u8,
    method: Method,
    slabs: usize,
    rows: usize,
    cols: usize,
    /// Source matrix per slab. MXFP4 stores the transpose: `src_rows == cols`, `src_cols == rows`.
    src_rows: usize,
    src_cols: usize,
    data: Piece,
    scale: Piece,
    zero: ?Piece = null,
    /// FP8: scale grid per slab (`scale_rows` x `scale_cols`, `block_rows` x `block_cols` elements each).
    scale_rows: usize = 0,
    scale_cols: usize = 0,
    block_rows: usize = 0,
    block_cols: usize = 0,
    e5m2: bool = false,
    /// pack-quantized: field width, columns per scale, words per packed row.
    bits: u8 = 4,
    group: usize = 0,
    words_per_row: usize = 0,
    /// Mapped mode: the whole tensor, decoded at load.
    materialized: ?[]u8 = null,
    /// Streamed mode, MXFP4: the last decoded slab, so per-row reads of a
    /// transposed tensor do not decode the slab once per row.
    cache_lock: std.atomic.Value(bool) = .init(false),
    cache_slab: ?usize = null,
    cache_buf: []u8 = &.{},

    pub fn rowBytes(self: *const Dequant) usize {
        return self.cols * 2;
    }

    pub fn slabBytes(self: *const Dequant) usize {
        return self.rows * self.rowBytes();
    }

    pub fn byteLen(self: *const Dequant) usize {
        return self.slabs * self.slabBytes();
    }

    pub fn deinit(self: *Dequant, gpa: Allocator) void {
        if (self.materialized) |m| gpa.free(m);
        if (self.cache_buf.len > 0) std.heap.page_allocator.free(self.cache_buf);
        gpa.destroy(self);
    }

    /// Bytes `[rel, rel + out.len)` of the virtual tensor. Whole rows only
    /// (every reader of the weight store works in rows).
    /// (`anyerror`: the source reads go through `File.readRange`, which calls back here.)
    pub fn readRange(self: *Dequant, io: Io, rel: usize, out: []u8) anyerror!void {
        if (out.len == 0) return;
        if (self.materialized) |m| {
            if (rel + out.len > m.len) return error.UnexpectedEndOfFile;
            @memcpy(out, m[rel..][0..out.len]);
            return;
        }
        const rb = self.rowBytes();
        if (rel % rb != 0 or out.len % rb != 0 or rel + out.len > self.byteLen()) return error.UnexpectedEndOfFile;
        var r = rel / rb;
        var done: usize = 0;
        while (done < out.len) {
            const slab = r / self.rows;
            const a = r % self.rows;
            const n = @min(self.rows - a, (out.len - done) / rb);
            try self.readSlabRows(io, slab, a, n, out[done..][0 .. n * rb]);
            r += n;
            done += n * rb;
        }
    }

    /// Decodes the whole tensor into a resident buffer (mapped mode).
    pub fn materialize(self: *Dequant, gpa: Allocator, io: Io) ![]const u8 {
        if (self.materialized) |m| return m;
        const buf = try gpa.alloc(u8, self.byteLen());
        errdefer gpa.free(buf);
        try self.readRange(io, 0, buf);
        self.materialized = buf;
        return buf;
    }

    fn readSlabRows(self: *Dequant, io: Io, slab: usize, a: usize, n: usize, out: []u8) !void {
        switch (self.method) {
            .fp8 => try self.readFp8(io, slab, a, n, out),
            .int_packed => try self.readPacked(io, slab, a, n, out),
            .mxfp4 => try self.readMxfp4(io, slab, a, n, out),
            .none => unreachable,
        }
    }

    // -- FP8 ----------------------------------------------------------------

    fn readFp8(self: *Dequant, io: Io, slab: usize, a: usize, n: usize, out: []u8) !void {
        const pa = std.heap.page_allocator;
        const src_rb = self.src_cols; // one byte per element
        const per_chunk = @max(1, chunk_bytes / src_rb);
        const table = if (self.e5m2) &e5m2_table else &e4m3_table;
        var r = a;
        while (r < a + n) {
            const cn = @min(per_chunk, a + n - r);
            const codes = try pa.alloc(u8, cn * src_rb);
            defer pa.free(codes);
            try self.data.read(io, (@as(u64, slab) * self.src_rows + r) * src_rb, codes);
            // Scale rows covering [r, r + cn).
            const s0 = r / self.block_rows;
            const s1 = (r + cn - 1) / self.block_rows + 1;
            const es = self.scale.dtype.size();
            const sraw = try pa.alloc(u8, (s1 - s0) * self.scale_cols * es);
            defer pa.free(sraw);
            try self.scale.read(io, (@as(u64, slab) * self.scale_rows + s0) * self.scale_cols * es, sraw);
            const scales = try pa.alloc(f32, (s1 - s0) * self.scale_cols);
            defer pa.free(scales);
            tensor.convertToF32(self.scale.dtype, sraw, scales);
            const dst = std.mem.bytesAsSlice(u16, out[(r - a) * self.rowBytes() ..][0 .. cn * self.rowBytes()]);
            for (0..cn) |i| {
                const srow = scales[((r + i) / self.block_rows - s0) * self.scale_cols ..][0..self.scale_cols];
                const crow = codes[i * src_rb ..][0..self.cols];
                const drow = dst[i * self.cols ..][0..self.cols];
                var c: usize = 0;
                while (c < self.cols) {
                    const s = srow[c / self.block_cols];
                    const end = @min(self.cols, (c / self.block_cols + 1) * self.block_cols);
                    while (c < end) : (c += 1) drow[c] = tensor.f32ToBf16(table[crow[c]] * s);
                }
            }
            r += cn;
        }
    }

    // -- pack-quantized -----------------------------------------------------

    fn readPacked(self: *Dequant, io: Io, slab: usize, a: usize, n: usize, out: []u8) !void {
        const pa = std.heap.page_allocator;
        const src_rb = self.words_per_row * 4;
        const per_chunk = @max(1, chunk_bytes / src_rb);
        const groups = self.scale_cols;
        const es = self.scale.dtype.size();
        const offset: i32 = @as(i32, 1) << @intCast(self.bits - 1);
        var r = a;
        while (r < a + n) {
            const cn = @min(per_chunk, a + n - r);
            const raw = try pa.alloc(u8, cn * src_rb);
            defer pa.free(raw);
            try self.data.read(io, (@as(u64, slab) * self.src_rows + r) * src_rb, raw);
            const sraw = try pa.alloc(u8, cn * groups * es);
            defer pa.free(sraw);
            try self.scale.read(io, (@as(u64, slab) * self.rows + r) * groups * es, sraw);
            const scales = try pa.alloc(f32, cn * groups);
            defer pa.free(scales);
            tensor.convertToF32(self.scale.dtype, sraw, scales);
            // Zero points: packed along the rows, `[ceil(rows * bits / 32)][groups]` words per slab.
            var zwords: []u32 = &.{};
            defer if (zwords.len > 0) pa.free(zwords);
            var z0: usize = 0;
            if (self.zero) |z| {
                z0 = r * self.bits / 32;
                const z1 = ((r + cn - 1) * self.bits + self.bits - 1) / 32 + 1;
                const zrows_slab = (self.rows * self.bits + 31) / 32;
                const zraw = try pa.alloc(u8, (z1 - z0) * groups * 4);
                defer pa.free(zraw);
                try z.read(io, (@as(u64, slab) * zrows_slab + z0) * groups * 4, zraw);
                zwords = try pa.alloc(u32, (z1 - z0) * groups);
                for (zwords, 0..) |*w, i| w.* = std.mem.readInt(u32, zraw[i * 4 ..][0..4], .little);
            }
            const words = try pa.alloc(u32, cn * self.words_per_row);
            defer pa.free(words);
            for (words, 0..) |*w, i| w.* = std.mem.readInt(u32, raw[i * 4 ..][0..4], .little);
            const dst = std.mem.bytesAsSlice(u16, out[(r - a) * self.rowBytes() ..][0 .. cn * self.rowBytes()]);
            for (0..cn) |i| {
                const row_words = words[i * self.words_per_row ..][0..self.words_per_row];
                const srow = scales[i * groups ..][0..groups];
                const drow = dst[i * self.cols ..][0..self.cols];
                for (0..groups) |g| {
                    var zp: i32 = 0;
                    if (zwords.len > 0) {
                        // Field `r + i` of column `g`'s word stream (row-major `[words][groups]`).
                        const start = (r + i) * self.bits;
                        const w = start / 32 - z0;
                        const off: u5 = @intCast(start % 32);
                        const lo: u8 = @min(@as(u8, @intCast(32 - @as(u32, off))), self.bits);
                        var v = (zwords[w * groups + g] >> off) & fieldMask(lo);
                        if (lo < self.bits) v |= (zwords[(w + 1) * groups + g] & fieldMask(self.bits - lo)) << @intCast(lo);
                        zp = @as(i32, @intCast(v)) - offset;
                    }
                    const s = srow[g];
                    const c0 = g * self.group;
                    const c1 = @min(self.cols, c0 + self.group);
                    var c = c0;
                    while (c < c1) : (c += 1) {
                        const q = @as(i32, @intCast(unpackField(row_words, c, self.bits))) - offset;
                        drow[c] = tensor.f32ToBf16(@as(f32, @floatFromInt(q - zp)) * s);
                    }
                }
            }
            r += cn;
        }
    }

    // -- MXFP4 --------------------------------------------------------------

    /// Decodes one whole slab (source `[src_rows][src_cols]`) into its
    /// transpose `out` (`[rows][cols]` bf16, i.e. `[src_cols][src_rows]`).
    fn decodeMxfp4Slab(self: *Dequant, io: Io, slab: usize, out: []u8) !void {
        const pa = std.heap.page_allocator;
        const nblk = self.src_cols / 32;
        const blk_bytes = self.src_rows * nblk * 16;
        const raw = try pa.alloc(u8, blk_bytes + self.src_rows * nblk);
        defer pa.free(raw);
        const blocks = raw[0..blk_bytes];
        const scales = raw[blk_bytes..];
        try self.data.read(io, @as(u64, slab) * blk_bytes, blocks);
        try self.scale.read(io, @as(u64, slab) * self.src_rows * nblk, scales);
        const dst = std.mem.bytesAsSlice(u16, out);
        const cols = self.cols; // == src_rows
        // Tiles of 32 source rows x one block (32 columns): the tile is decoded
        // into `tmp` and written out as 32 runs of up to 32 contiguous elements.
        var tmp: [32][32]f32 = undefined;
        var j0: usize = 0;
        while (j0 < self.src_rows) : (j0 += 32) {
            const jn = @min(32, self.src_rows - j0);
            for (0..nblk) |k| {
                for (0..jn) |jj| {
                    const b = blocks[((j0 + jj) * nblk + k) * 16 ..][0..16];
                    const e: i32 = @as(i32, scales[(j0 + jj) * nblk + k]) - 127;
                    for (0..16) |i| {
                        tmp[jj][2 * i] = std.math.ldexp(e2m1_table[b[i] & 0x0f], e);
                        tmp[jj][2 * i + 1] = std.math.ldexp(e2m1_table[b[i] >> 4], e);
                    }
                }
                for (0..32) |i| {
                    const h = k * 32 + i;
                    const drow = dst[h * cols + j0 ..][0..jn];
                    for (0..jn) |jj| drow[jj] = tensor.f32ToBf16(tmp[jj][i]);
                }
            }
        }
    }

    fn readMxfp4(self: *Dequant, io: Io, slab: usize, a: usize, n: usize, out: []u8) !void {
        if (a == 0 and n == self.rows) return self.decodeMxfp4Slab(io, slab, out);
        // A row range needs the whole slab (the source is the transpose): keep
        // the last decoded slab so consecutive row reads share one decode.
        while (self.cache_lock.cmpxchgWeak(false, true, .acquire, .monotonic) != null) std.atomic.spinLoopHint();
        defer self.cache_lock.store(false, .release);
        if (self.cache_slab != slab) {
            if (self.cache_buf.len == 0) self.cache_buf = try std.heap.page_allocator.alloc(u8, self.slabBytes());
            self.cache_slab = null;
            try self.decodeMxfp4Slab(io, slab, self.cache_buf);
            self.cache_slab = slab;
        }
        @memcpy(out, self.cache_buf[a * self.rowBytes() ..][0 .. n * self.rowBytes()]);
    }
};

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

const Found = struct { file: *safetensors.File, info: safetensors.TensorInfo };
const FoundRaw = struct { file: *safetensors.File, info: safetensors.RawInfo };

fn findFloat(files: []const *safetensors.File, name: []const u8) ?Found {
    for (files) |f| if (f.get(name)) |t| return .{ .file = f, .info = t };
    return null;
}

fn findRaw(files: []const *safetensors.File, name: []const u8) ?FoundRaw {
    for (files) |f| if (f.raw.get(name)) |t| return .{ .file = f, .info = t };
    return null;
}

fn removeFloat(files: []const *safetensors.File, name: []const u8) void {
    for (files) |f| _ = f.tensors.orderedRemove(name);
}

fn removeRaw(files: []const *safetensors.File, name: []const u8) void {
    for (files) |f| _ = f.raw.orderedRemove(name);
}

fn piece(f: *safetensors.File, offset: u64, byte_len: usize, dtype: DType) Piece {
    return .{ .file = f, .offset = offset, .byte_len = byte_len, .dtype = dtype };
}

fn leading(shape: []const usize, keep: usize) usize {
    var n: usize = 1;
    for (shape[0 .. shape.len - keep]) |s| n *= s;
    return n;
}

fn withPrefix(gpa: Allocator, base: []const u8, suffix: []const u8) ![]u8 {
    return std.fmt.allocPrint(gpa, "{s}{s}", .{ base, suffix });
}

/// Names of a module's auxiliary quantisation tensors that have no place in a bf16 export.
const module_aux = [_][]const u8{ ".weight_scale", ".weight_scale_inv", ".weight_zero_point", ".weight_shape", ".input_scale", ".input_zero_point", ".weight_global_scale", ".input_global_scale" };

fn dropModuleAux(gpa: Allocator, files: []const *safetensors.File, module: []const u8) !void {
    for (module_aux) |suffix| {
        const n = try withPrefix(gpa, module, suffix);
        defer gpa.free(n);
        removeFloat(files, n);
        removeRaw(files, n);
    }
}

pub const Registered = struct {
    count: usize = 0,
    fp8: usize = 0,
    mxfp4: usize = 0,
    int_packed: usize = 0,
};

/// Finds every quantised tensor group in `files`, registers a virtual bf16
/// tensor for each (materialised at once when its file is memory-mapped) and
/// removes the storage tensors from the files' indexes. Returns the counts.
pub fn register(gpa: Allocator, io: Io, files: []const *safetensors.File, cfg: QuantConfig) !Registered {
    var out = Registered{};
    // Collect the candidates first: registering mutates the raw indexes.
    var names = std.ArrayList([]const u8).empty;
    defer {
        for (names.items) |n| gpa.free(n);
        names.deinit(gpa);
    }
    for (files) |f| {
        var it = f.raw.iterator();
        while (it.next()) |kv| try names.append(gpa, try gpa.dupe(u8, kv.key_ptr.*));
    }
    for (names.items) |name| {
        const found = findRaw(files, name) orelse continue; // consumed by an earlier group
        const dt = found.info.dtype;
        if (std.mem.eql(u8, dt, "F8_E4M3") or std.mem.eql(u8, dt, "F8_E5M2")) {
            if (try registerFp8(gpa, io, files, found, cfg)) out.fp8 += 1;
        } else if (std.mem.eql(u8, dt, "U8") and std.mem.endsWith(u8, name, "_blocks")) {
            if (try registerMxfp4(gpa, io, files, found)) out.mxfp4 += 1;
        } else if (std.mem.eql(u8, dt, "I32") and std.mem.endsWith(u8, name, ".weight_packed")) {
            if (try registerPacked(gpa, io, files, found, cfg)) out.int_packed += 1;
        }
    }
    out.count = out.fp8 + out.mxfp4 + out.int_packed;
    return out;
}

/// Registers `dq` as the tensor `name` of `f` with the given shape, decoding it now when `f` is mapped.
fn install(gpa: Allocator, io: Io, f: *safetensors.File, dq: *Dequant, shape: []const usize) !void {
    var data: []const u8 = &.{};
    if (f.isMapped()) data = try dq.materialize(gpa, io);
    try f.addDequant(dq, shape, data);
}

fn registerFp8(gpa: Allocator, io: Io, files: []const *safetensors.File, w: FoundRaw, cfg: QuantConfig) !bool {
    const name = w.info.name;
    if (w.info.shape.len < 2) return false;
    const scale_inv_name = try withPrefix(gpa, name, "_scale_inv");
    defer gpa.free(scale_inv_name);
    const scale_name = try withPrefix(gpa, name, "_scale");
    defer gpa.free(scale_name);
    const sc = findFloat(files, scale_inv_name) orelse findFloat(files, scale_name) orelse {
        std.log.err("{s}: {s} weights without a weight_scale_inv/weight_scale tensor", .{ name, w.info.dtype });
        return error.UnsupportedArchitecture;
    };
    const rows = w.info.shape[w.info.shape.len - 2];
    const cols = w.info.shape[w.info.shape.len - 1];
    const slabs = leading(w.info.shape, 2);
    // Scale grid: `[..., sr, sc]`, `[..., sr]` (per row) or `[]`/`[1]` (per tensor).
    var scale_rows: usize = 1;
    var scale_cols: usize = 1;
    const ss = sc.info.shape;
    const scale_numel = sc.info.numel();
    if (ss.len >= 2 and scale_numel == slabs * ss[ss.len - 2] * ss[ss.len - 1] and leading(ss, 2) == slabs) {
        scale_rows = ss[ss.len - 2];
        scale_cols = ss[ss.len - 1];
    } else if (scale_numel == slabs * rows) {
        scale_rows = rows;
    } else if (scale_numel != slabs) {
        std.log.err("{s}: scale tensor has {d} elements for a [{d}][{d}] weight", .{ name, scale_numel, rows, cols });
        return error.InvalidConfig;
    }
    var block_rows = (rows + scale_rows - 1) / scale_rows;
    var block_cols = (cols + scale_cols - 1) / scale_cols;
    if (cfg.block_rows > 0 and cfg.block_cols > 0 and (rows + cfg.block_rows - 1) / cfg.block_rows == scale_rows and (cols + cfg.block_cols - 1) / cfg.block_cols == scale_cols) {
        block_rows = cfg.block_rows;
        block_cols = cfg.block_cols;
    }
    const dq = try gpa.create(Dequant);
    errdefer gpa.destroy(dq);
    dq.* = .{
        .name = name,
        .method = .fp8,
        .slabs = slabs,
        .rows = rows,
        .cols = cols,
        .src_rows = rows,
        .src_cols = cols,
        .data = piece(w.file, w.info.offset, w.info.byte_len, .f32),
        .scale = piece(sc.file, sc.info.offset, sc.info.byte_len, sc.info.dtype),
        .scale_rows = scale_rows,
        .scale_cols = scale_cols,
        .block_rows = block_rows,
        .block_cols = block_cols,
        .e5m2 = std.mem.eql(u8, w.info.dtype, "F8_E5M2"),
    };
    const shape = try w.file.arena.allocator().dupe(usize, w.info.shape);
    removeRaw(files, name);
    if (std.mem.endsWith(u8, name, ".weight")) try dropModuleAux(gpa, files, name[0 .. name.len - ".weight".len]);
    removeFloat(files, scale_inv_name);
    removeFloat(files, scale_name);
    try install(gpa, io, w.file, dq, shape);
    return true;
}

fn registerMxfp4(gpa: Allocator, io: Io, files: []const *safetensors.File, b: FoundRaw) !bool {
    const bname = b.info.name;
    const base = bname[0 .. bname.len - "_blocks".len];
    const sname = try withPrefix(gpa, base, "_scales");
    defer gpa.free(sname);
    const sc = findRaw(files, sname) orelse {
        std.log.err("{s}: MXFP4 blocks without a {s} tensor", .{ bname, sname });
        return error.UnsupportedArchitecture;
    };
    const bs = b.info.shape;
    if (bs.len < 3 or bs[bs.len - 1] != 16 or !std.mem.eql(u8, sc.info.dtype, "U8")) {
        std.log.err("{s}: unexpected MXFP4 layout (blocks {any}, scales {s})", .{ bname, bs, sc.info.dtype });
        return error.InvalidConfig;
    }
    const src_rows = bs[bs.len - 3];
    const src_cols = bs[bs.len - 2] * 32;
    const slabs = leading(bs, 3);
    if (sc.info.numel() != slabs * src_rows * (src_cols / 32)) {
        std.log.err("{s}: scales do not match the blocks", .{bname});
        return error.InvalidConfig;
    }
    const dq = try gpa.create(Dequant);
    errdefer gpa.destroy(dq);
    dq.* = .{
        .name = try b.file.arena.allocator().dupe(u8, base),
        .method = .mxfp4,
        .slabs = slabs,
        .rows = src_cols,
        .cols = src_rows,
        .src_rows = src_rows,
        .src_cols = src_cols,
        .data = piece(b.file, b.info.offset, b.info.byte_len, .f32),
        .scale = piece(sc.file, sc.info.offset, sc.info.byte_len, .f32),
    };
    // Virtual shape: the leading dims, then the transpose of the matrix.
    const shape = try b.file.arena.allocator().alloc(usize, bs.len - 1);
    @memcpy(shape[0 .. bs.len - 3], bs[0 .. bs.len - 3]);
    shape[bs.len - 3] = src_cols;
    shape[bs.len - 2] = src_rows;
    removeRaw(files, bname);
    removeRaw(files, sname);
    try install(gpa, io, b.file, dq, shape);
    return true;
}

fn registerPacked(gpa: Allocator, io: Io, files: []const *safetensors.File, p: FoundRaw, cfg: QuantConfig) !bool {
    const pname = p.info.name;
    const module = pname[0 .. pname.len - ".weight_packed".len];
    const scale_name = try withPrefix(gpa, module, ".weight_scale");
    defer gpa.free(scale_name);
    const zero_name = try withPrefix(gpa, module, ".weight_zero_point");
    defer gpa.free(zero_name);
    const shape_name = try withPrefix(gpa, module, ".weight_shape");
    defer gpa.free(shape_name);
    const sc = findFloat(files, scale_name) orelse {
        std.log.err("{s}: packed weights without a {s} tensor", .{ pname, scale_name });
        return error.UnsupportedArchitecture;
    };
    const ps = p.info.shape;
    if (ps.len < 2) return false;
    const bits: u8 = if (cfg.method == .int_packed) cfg.num_bits else 4;
    const rows = ps[ps.len - 2];
    const words_per_row = ps[ps.len - 1];
    const slabs = leading(ps, 2);
    // Columns: the stored shape when present, else what the scale grid implies.
    var cols: usize = 0;
    if (findRaw(files, shape_name)) |sh| {
        if (std.mem.eql(u8, sh.info.dtype, "I64") and sh.info.numel() >= 2) {
            var buf: [8]u8 = undefined;
            try sh.file.readRange(io, sh.info.offset + @as(u64, sh.info.numel() - 1) * 8, &buf);
            cols = @intCast(std.mem.readInt(i64, &buf, .little));
        }
    }
    const ss = sc.info.shape;
    const scale_numel = sc.info.numel();
    var groups: usize = 1;
    if (ss.len >= 2 and leading(ss, 2) == slabs and ss[ss.len - 2] == rows) {
        groups = ss[ss.len - 1];
    } else if (scale_numel == slabs * rows) {
        groups = 1;
    } else if (scale_numel != slabs) {
        std.log.err("{s}: scale tensor has {d} elements for {d} rows", .{ pname, scale_numel, rows });
        return error.InvalidConfig;
    }
    if (scale_numel == slabs and rows != 1) {
        std.log.err("{s}: per-tensor scales are not supported for packed weights", .{pname});
        return error.UnsupportedArchitecture;
    }
    var group: usize = if (cfg.method == .int_packed) cfg.group_size else 0;
    if (cols == 0) {
        if (group == 0) {
            std.log.err("{s}: neither weight_shape nor a group_size gives the column count", .{pname});
            return error.InvalidConfig;
        }
        cols = groups * group;
    }
    if (group == 0 or groups * group < cols) group = (cols + groups - 1) / groups;
    if (words_per_row * 32 < cols * bits) {
        std.log.err("{s}: {d} packed words per row cannot hold {d} {d}-bit columns", .{ pname, words_per_row, cols, bits });
        return error.InvalidConfig;
    }
    var zero: ?Piece = null;
    if (findRaw(files, zero_name)) |z| {
        if (!std.mem.eql(u8, z.info.dtype, "I32") or z.info.numel() != slabs * ((rows * bits + 31) / 32) * groups) {
            std.log.err("{s}: unexpected zero-point layout", .{pname});
            return error.InvalidConfig;
        }
        zero = piece(z.file, z.info.offset, z.info.byte_len, .f32);
    }
    const dq = try gpa.create(Dequant);
    errdefer gpa.destroy(dq);
    dq.* = .{
        .name = try withPrefix(p.file.arena.allocator(), module, ".weight"),
        .method = .int_packed,
        .slabs = slabs,
        .rows = rows,
        .cols = cols,
        .src_rows = rows,
        .src_cols = cols,
        .data = piece(p.file, p.info.offset, p.info.byte_len, .f32),
        .scale = piece(sc.file, sc.info.offset, sc.info.byte_len, sc.info.dtype),
        .zero = zero,
        .scale_cols = groups,
        .bits = bits,
        .group = group,
        .words_per_row = words_per_row,
    };
    const shape = try p.file.arena.allocator().alloc(usize, ps.len);
    @memcpy(shape[0 .. ps.len - 1], ps[0 .. ps.len - 1]);
    shape[ps.len - 1] = cols;
    removeRaw(files, pname);
    try dropModuleAux(gpa, files, module);
    try install(gpa, io, p.file, dq, shape);
    return true;
}

// ---------------------------------------------------------------------------
// Config rewriting
// ---------------------------------------------------------------------------

/// Removes every `"quantization_config": ...` member from a JSON text (at any
/// nesting level, e.g. inside `text_config`), leaving the rest byte for byte.
pub fn stripQuantizationConfig(gpa: Allocator, json: []const u8) ![]u8 {
    var text = try gpa.dupe(u8, json);
    errdefer gpa.free(text);
    while (findKey(text, "quantization_config")) |span| {
        var start = span.start;
        var end = span.end;
        // Remove the separating comma: the following one, else the preceding one.
        var j = end;
        while (j < text.len and std.ascii.isWhitespace(text[j])) : (j += 1) {}
        if (j < text.len and text[j] == ',') {
            end = j + 1;
            while (end < text.len and std.ascii.isWhitespace(text[end])) : (end += 1) {}
        } else {
            var i = start;
            while (i > 0 and std.ascii.isWhitespace(text[i - 1])) : (i -= 1) {}
            if (i > 0 and text[i - 1] == ',') start = i - 1;
        }
        const next = try std.mem.concat(gpa, u8, &.{ text[0..start], text[end..] });
        gpa.free(text);
        text = next;
    }
    return text;
}

const Span = struct { start: usize, end: usize };

/// Locates `"key": value` (key start to value end) at any depth of a JSON text.
fn findKey(text: []const u8, key: []const u8) ?Span {
    var i: usize = 0;
    while (i < text.len) : (i += 1) {
        if (text[i] != '"') continue;
        const s_end = skipString(text, i) orelse return null;
        const s = text[i + 1 .. s_end - 1];
        var j = s_end;
        while (j < text.len and std.ascii.isWhitespace(text[j])) : (j += 1) {}
        const is_key = j < text.len and text[j] == ':';
        if (is_key and std.mem.eql(u8, s, key)) {
            j += 1;
            while (j < text.len and std.ascii.isWhitespace(text[j])) : (j += 1) {}
            const v_end = skipValue(text, j) orelse return null;
            return .{ .start = i, .end = v_end };
        }
        // Other strings (keys and values) are skipped whole; the values of
        // other keys are scanned since nested objects may hold the key.
        i = s_end - 1;
    }
    return null;
}

/// Index just past the string starting at `i` (which must be a quote).
fn skipString(text: []const u8, i: usize) ?usize {
    var j = i + 1;
    while (j < text.len) : (j += 1) {
        if (text[j] == '\\') {
            j += 1;
        } else if (text[j] == '"') return j + 1;
    }
    return null;
}

/// Index just past the JSON value starting at `i`.
fn skipValue(text: []const u8, i: usize) ?usize {
    if (i >= text.len) return null;
    switch (text[i]) {
        '"' => return skipString(text, i),
        '{', '[' => {
            var depth: usize = 0;
            var j = i;
            while (j < text.len) : (j += 1) {
                switch (text[j]) {
                    '"' => j = (skipString(text, j) orelse return null) - 1,
                    '{', '[' => depth += 1,
                    '}', ']' => {
                        depth -= 1;
                        if (depth == 0) return j + 1;
                    },
                    else => {},
                }
            }
            return null;
        },
        else => {
            var j = i;
            while (j < text.len and text[j] != ',' and text[j] != '}' and text[j] != ']' and !std.ascii.isWhitespace(text[j])) : (j += 1) {}
            return j;
        },
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

test "fp8 code tables" {
    try std.testing.expectEqual(@as(f32, 448), e4m3_table[0x7e]);
    try std.testing.expectEqual(@as(f32, -448), e4m3_table[0xfe]);
    try std.testing.expect(std.math.isNan(e4m3_table[0x7f]));
    try std.testing.expectEqual(@as(f32, 1.0), e4m3_table[0x38]);
    try std.testing.expectEqual(@as(f32, 0.001953125), e4m3_table[0x01]); // smallest subnormal 2^-9
    try std.testing.expectEqual(@as(f32, 57344), e5m2_table[0x7b]);
    try std.testing.expect(std.math.isInf(e5m2_table[0x7c]));
    try std.testing.expectEqual(@as(f32, 1.0), e5m2_table[0x3c]);
}

test "dense field unpacking" {
    // 4-bit fields 0..15 fill exactly two words.
    var words = [_]u32{ 0x7654_3210, 0xfedc_ba98 };
    for (0..16) |i| try std.testing.expectEqual(@as(u32, @intCast(i)), unpackField(&words, i, 4));
    // 3-bit fields straddle word boundaries: field 10 spans bits 30..32.
    words = .{ 0, 0 };
    words[0] = @as(u32, 0b01) << 30; // low two bits of the field
    words[1] = 1; // its high bit
    try std.testing.expectEqual(@as(u32, 0b101), unpackField(&words, 10, 3));
}

test "strip quantization_config" {
    const gpa = std.testing.allocator;
    const a = try stripQuantizationConfig(gpa, "{\"a\": 1, \"quantization_config\": {\"quant_method\": \"fp8\", \"weight_block_size\": [128, 128]}, \"b\": \"x\"}");
    defer gpa.free(a);
    try std.testing.expectEqualStrings("{\"a\": 1, \"b\": \"x\"}", a);
    const b = try stripQuantizationConfig(gpa, "{\n  \"a\": 1,\n  \"quantization_config\": {\"m\": \"}\"}\n}");
    defer gpa.free(b);
    try std.testing.expectEqualStrings("{\n  \"a\": 1\n}", b);
    const c = try stripQuantizationConfig(gpa, "{\"text_config\": {\"quantization_config\": [1, 2], \"x\": 2}, \"quantization_config\": null}");
    defer gpa.free(c);
    try std.testing.expectEqualStrings("{\"text_config\": {\"x\": 2}}", c);
    const d = try stripQuantizationConfig(gpa, "{\"no_quantization_config\": 1}");
    defer gpa.free(d);
    try std.testing.expectEqualStrings("{\"no_quantization_config\": 1}", d);
}

test "quant config parsing" {
    const gpa = std.testing.allocator;
    var p = try std.json.parseFromSlice(std.json.Value, gpa, "{\"quant_method\": \"fp8\", \"weight_block_size\": [128, 64]}", .{});
    defer p.deinit();
    const fp8 = try parseQuantConfig(p.value.object);
    try std.testing.expectEqual(Method.fp8, fp8.method);
    try std.testing.expectEqual(@as(usize, 64), fp8.block_cols);
    var q = try std.json.parseFromSlice(std.json.Value, gpa, "{\"quant_method\": \"compressed-tensors\", \"format\": \"pack-quantized\", \"config_groups\": {\"group_0\": {\"weights\": {\"num_bits\": 4, \"group_size\": 32, \"type\": \"int\"}}}}", .{});
    defer q.deinit();
    const packed_ = try parseQuantConfig(q.value.object);
    try std.testing.expectEqual(Method.int_packed, packed_.method);
    try std.testing.expectEqual(@as(usize, 32), packed_.group_size);
    var r = try std.json.parseFromSlice(std.json.Value, gpa, "{\"quant_method\": \"compressed-tensors\", \"format\": \"dense\"}", .{});
    defer r.deinit();
    try std.testing.expectEqual(Method.none, (try parseQuantConfig(r.value.object)).method);
    try std.testing.expectEqual(Method.none, (try parseQuantConfig(null)).method);
}
