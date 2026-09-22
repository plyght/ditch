//! Reading and writing of the safetensors format (https://github.com/huggingface/safetensors).
//! Files are memory-mapped for reading by default so weights are never copied
//! unless needed; `File.openOptions` with `.map = false` parses only the header
//! and leaves the tensor bytes on disk (see stream.zig for positional reads).
//! `File.openRemote` does the same over a remote shard (remote.zig): the
//! header comes through range requests and so do the tensor bytes later.

const std = @import("std");
const Io = std.Io;
const tensor = @import("tensor.zig");
const remote = @import("remote.zig");
const DType = tensor.DType;

pub const TensorInfo = struct {
    name: []const u8,
    dtype: DType,
    shape: []const usize,
    /// Raw bytes inside the mapped file (empty when the file was opened without mapping).
    data: []const u8,
    /// Absolute byte offset of the tensor data within the file.
    offset: u64,
    /// Length of the tensor data in bytes.
    byte_len: usize,

    pub fn numel(self: TensorInfo) usize {
        var n: usize = 1;
        for (self.shape) |s| n *= s;
        return n;
    }

    pub fn rows(self: TensorInfo) usize {
        std.debug.assert(self.shape.len >= 1);
        return self.shape[0];
    }

    pub fn cols(self: TensorInfo) usize {
        return if (self.shape.len == 1) 1 else self.numel() / self.rows();
    }

    /// A view over the mapped bytes. Only valid for files opened with `.map = true`.
    pub fn asWeight(self: TensorInfo) tensor.Weight {
        std.debug.assert(self.data.len == self.byte_len);
        return .{ .data = self.data, .dtype = self.dtype, .rows = self.rows(), .cols = self.cols() };
    }
};

pub const OpenOptions = struct {
    /// Memory-map the whole file. When false only the header is read and
    /// `TensorInfo.data` is empty; tensor bytes must be read positionally.
    map: bool = true,
};

/// Where a file's bytes come from.
pub const Source = union(enum) {
    local: Io.File,
    remote: *remote.RemoteFile,
};

/// Data served in place of a file range: a synthetic file (see
/// `File.initSynthetic`, used for GGUF sources) addresses tensors that are not
/// stored verbatim on disk (llama.cpp's permuted q/k undone, gemma norms
/// without their `+1`) through virtual offsets beyond the file length.
pub const Overlay = struct {
    /// Virtual byte offset (>= `File.len`) of the first byte.
    offset: u64,
    byte_len: usize,
    kind: union(enum) {
        bytes: []const u8,
        /// Rows `[rows][row_bytes]` at `src_offset`, stored in llama.cpp's
        /// permuted order and read back in Hugging Face order (streamed mode).
        permuted_rows: struct { src_offset: u64, rows: usize, row_bytes: usize, n_head: usize },
    },
};

/// Row `hf_row` of a Hugging Face q/k matrix with `head_dim` rows per head is
/// stored at this row by llama.cpp (`permute`: `[2][head_dim/2]` -> `[head_dim/2][2]` per head).
pub fn llamaPermutedRow(hf_row: usize, head_dim: usize) usize {
    const half = head_dim / 2;
    const h = hf_row / head_dim;
    const r = hf_row % head_dim;
    return h * head_dim + 2 * (r % half) + (r / half);
}

pub const File = struct {
    path: []const u8,
    source: Source,
    map: ?Io.File.MemoryMap,
    header_len: usize,
    /// Total file length in bytes.
    len: u64,
    tensors: std.StringArrayHashMapUnmanaged(TensorInfo),
    arena: std.heap.ArenaAllocator,
    /// Synthetic data (see `Overlay`), addressed above `len`.
    overlays: std.ArrayList(Overlay) = .empty,
    next_virtual: u64 = 0,
    /// The dtype of the first tensor the header parser skipped (unsupported
    /// storage type such as FP8), so a loader can name it instead of failing
    /// on a missing tensor.
    skipped_dtype: ?[]const u8 = null,

    pub fn open(gpa: std.mem.Allocator, io: Io, dir: Io.Dir, sub_path: []const u8) !*File {
        return openOptions(gpa, io, dir, sub_path, .{});
    }

    pub fn openOptions(gpa: std.mem.Allocator, io: Io, dir: Io.Dir, sub_path: []const u8, options: OpenOptions) !*File {
        const self = try gpa.create(File);
        errdefer gpa.destroy(self);
        self.* = .{
            .path = undefined,
            .source = undefined,
            .map = null,
            .header_len = 0,
            .len = 0,
            .tensors = .{},
            .arena = std.heap.ArenaAllocator.init(gpa),
        };
        errdefer self.arena.deinit();
        const arena = self.arena.allocator();
        self.path = try arena.dupe(u8, sub_path);
        const file = try dir.openFile(io, sub_path, .{});
        self.source = .{ .local = file };
        errdefer file.close(io);
        const len: usize = @intCast(try file.length(io));
        self.len = len;
        if (len < 8) return error.InvalidSafetensors;
        var header_owned: ?[]u8 = null;
        defer if (header_owned) |h| gpa.free(h);
        var header: []const u8 = undefined;
        var data: []const u8 = &.{};
        if (options.map) {
            self.map = try Io.File.MemoryMap.create(io, file, .{
                .len = len,
                .protection = .{ .read = true, .write = false },
                .populate = false,
            });
            const bytes = self.map.?.memory[0..len];
            const n = std.mem.readInt(u64, bytes[0..8], .little);
            if (n > len - 8) return error.InvalidSafetensors;
            self.header_len = @intCast(n);
            header = bytes[8 .. 8 + self.header_len];
            data = bytes[8 + self.header_len ..];
        } else {
            var len_buf: [8]u8 = undefined;
            if (try file.readPositionalAll(io, &len_buf, 0) != 8) return error.InvalidSafetensors;
            const n = std.mem.readInt(u64, &len_buf, .little);
            if (n > len - 8) return error.InvalidSafetensors;
            self.header_len = @intCast(n);
            const h = try gpa.alloc(u8, self.header_len);
            header_owned = h;
            if (try file.readPositionalAll(io, h, 8) != h.len) return error.InvalidSafetensors;
            header = h;
        }
        errdefer if (self.map) |*m| m.destroy(io);
        const data_start: u64 = 8 + @as(u64, self.header_len);
        const data_len: usize = len - 8 - self.header_len;
        try self.parseHeader(gpa, header, data, data_start, data_len);
        return self;
    }

    /// Opens a remote shard: the 8-byte length and the JSON header are read
    /// through range requests, tensor bytes stay remote until read. The
    /// source keeps ownership of `rf`.
    pub fn openRemote(gpa: std.mem.Allocator, io: Io, rf: *remote.RemoteFile) !*File {
        const self = try gpa.create(File);
        errdefer gpa.destroy(self);
        self.* = .{
            .path = undefined,
            .source = .{ .remote = rf },
            .map = null,
            .header_len = 0,
            .len = 0,
            .tensors = .{},
            .arena = std.heap.ArenaAllocator.init(gpa),
        };
        errdefer self.arena.deinit();
        self.path = try self.arena.allocator().dupe(u8, rf.name);
        var len_buf: [8]u8 = undefined;
        try rf.readRange(io, 0, &len_buf);
        const n = std.mem.readInt(u64, &len_buf, .little);
        if (n == 0 or n > (1 << 30)) return error.InvalidSafetensors;
        self.header_len = @intCast(n);
        const header = try gpa.alloc(u8, self.header_len);
        defer gpa.free(header);
        try rf.readRange(io, 8, header);
        const data_start: u64 = 8 + @as(u64, self.header_len);
        // The total length is not known without a HEAD request; it is derived from the header.
        try self.parseHeader(gpa, header, &.{}, data_start, std.math.maxInt(usize));
        var end: u64 = data_start;
        var it = self.tensors.iterator();
        while (it.next()) |kv| end = @max(end, kv.value_ptr.offset + kv.value_ptr.byte_len);
        self.len = end;
        return self;
    }

    fn parseHeader(self: *File, gpa: std.mem.Allocator, header: []const u8, data: []const u8, data_start: u64, data_len: usize) !void {
        const arena = self.arena.allocator();
        var parsed = try std.json.parseFromSlice(std.json.Value, gpa, header, .{});
        defer parsed.deinit();
        if (parsed.value != .object) return error.InvalidSafetensors;
        var it = parsed.value.object.iterator();
        while (it.next()) |entry| {
            const name = entry.key_ptr.*;
            if (std.mem.eql(u8, name, "__metadata__")) continue;
            const obj = entry.value_ptr.*;
            if (obj != .object) return error.InvalidSafetensors;
            const dtype_str = (obj.object.get("dtype") orelse return error.InvalidSafetensors).string;
            const dtype = DType.fromSafetensors(dtype_str) orelse {
                std.log.warn("skipping tensor {s} with unsupported dtype {s}", .{ name, dtype_str });
                if (self.skipped_dtype == null) self.skipped_dtype = try arena.dupe(u8, dtype_str);
                continue;
            };
            const shape_val = (obj.object.get("shape") orelse return error.InvalidSafetensors).array;
            const shape = try arena.alloc(usize, shape_val.items.len);
            for (shape_val.items, 0..) |v, i| shape[i] = @intCast(v.integer);
            const offs = (obj.object.get("data_offsets") orelse return error.InvalidSafetensors).array;
            const start: usize = @intCast(offs.items[0].integer);
            const end: usize = @intCast(offs.items[1].integer);
            if (end > data_len or start > end) return error.InvalidSafetensors;
            try self.tensors.put(arena, try arena.dupe(u8, name), .{
                .name = try arena.dupe(u8, name),
                .dtype = dtype,
                .shape = shape,
                .data = if (data.len > 0) data[start..end] else &.{},
                .offset = data_start + start,
                .byte_len = end - start,
            });
        }
    }

    /// Opens `sub_path` as an empty tensor container (no header is parsed);
    /// tensors are registered with `addTensor` at explicit file offsets or as
    /// overlays. Used to present a GGUF file to the weight store.
    pub fn initSynthetic(gpa: std.mem.Allocator, io: Io, dir: Io.Dir, sub_path: []const u8, map: bool) !*File {
        const self = try gpa.create(File);
        errdefer gpa.destroy(self);
        self.* = .{
            .path = undefined,
            .source = undefined,
            .map = null,
            .header_len = 0,
            .len = 0,
            .tensors = .{},
            .arena = std.heap.ArenaAllocator.init(gpa),
        };
        errdefer self.arena.deinit();
        self.path = try self.arena.allocator().dupe(u8, sub_path);
        const file = try dir.openFile(io, sub_path, .{});
        self.source = .{ .local = file };
        errdefer file.close(io);
        self.len = try file.length(io);
        if (map) {
            self.map = try Io.File.MemoryMap.create(io, file, .{
                .len = @intCast(self.len),
                .protection = .{ .read = true, .write = false },
                .populate = false,
            });
        }
        self.next_virtual = (self.len + 63) / 64 * 64;
        return self;
    }

    /// Registers a tensor of a synthetic file (`info.name` is the key).
    pub fn addTensor(self: *File, info: TensorInfo) !void {
        try self.tensors.put(self.arena.allocator(), info.name, info);
    }

    /// Registers in-memory bytes (owned by the caller for the file's lifetime)
    /// and returns the virtual offset under which they are read.
    pub fn addOverlayBytes(self: *File, bytes: []const u8) !u64 {
        const off = self.next_virtual;
        try self.overlays.append(self.arena.allocator(), .{ .offset = off, .byte_len = bytes.len, .kind = .{ .bytes = bytes } });
        self.next_virtual += (bytes.len + 63) / 64 * 64;
        return off;
    }

    /// Registers a permuted-row view over the file range at `src_offset`.
    pub fn addOverlayPermuted(self: *File, src_offset: u64, rows: usize, row_bytes: usize, n_head: usize) !u64 {
        const off = self.next_virtual;
        const byte_len = rows * row_bytes;
        try self.overlays.append(self.arena.allocator(), .{ .offset = off, .byte_len = byte_len, .kind = .{ .permuted_rows = .{ .src_offset = src_offset, .rows = rows, .row_bytes = row_bytes, .n_head = n_head } } });
        self.next_virtual += (byte_len + 63) / 64 * 64;
        return off;
    }

    fn overlayAt(self: *const File, offset: u64) ?*const Overlay {
        for (self.overlays.items) |*o| {
            if (offset >= o.offset and offset < o.offset + o.byte_len) return o;
        }
        return null;
    }

    /// Bytes `[offset, offset + len)` of a mapped file (or of a byte overlay).
    pub fn mappedSlice(self: *const File, offset: u64, len: usize) []const u8 {
        if (offset < self.len) return self.map.?.memory[@intCast(offset)..][0..len];
        const o = self.overlayAt(offset).?;
        const rel: usize = @intCast(offset - o.offset);
        return switch (o.kind) {
            .bytes => |b| b[rel..][0..len],
            .permuted_rows => unreachable, // only registered for unmapped files
        };
    }

    pub fn close(self: *File, gpa: std.mem.Allocator, io: Io) void {
        if (self.map) |*m| m.destroy(io);
        switch (self.source) {
            .local => |f| f.close(io),
            .remote => {},
        }
        self.arena.deinit();
        gpa.destroy(self);
    }

    pub fn isRemote(self: *const File) bool {
        return self.source == .remote;
    }

    pub fn get(self: *const File, name: []const u8) ?TensorInfo {
        return self.tensors.get(name);
    }

    pub fn isMapped(self: *const File) bool {
        return self.map != null;
    }

    /// Reads `info`'s bytes from disk into `out` (which must be at least `byte_len` long).
    pub fn readTensor(self: *const File, io: Io, info: TensorInfo, out: []u8) !void {
        try self.readRange(io, info.offset, out[0..info.byte_len]);
    }

    /// Positional read of `out.len` bytes at absolute file `offset`.
    pub fn readRange(self: *const File, io: Io, offset: u64, out: []u8) !void {
        if (offset >= self.len) {
            const o = self.overlayAt(offset) orelse return error.UnexpectedEndOfFile;
            const rel: usize = @intCast(offset - o.offset);
            if (rel + out.len > o.byte_len) return error.UnexpectedEndOfFile;
            switch (o.kind) {
                .bytes => |b| @memcpy(out, b[rel..][0..out.len]),
                .permuted_rows => |p| {
                    // Whole rows only: the store reads row slices.
                    if (rel % p.row_bytes != 0 or out.len % p.row_bytes != 0) return error.UnexpectedEndOfFile;
                    const head_dim = p.rows / p.n_head;
                    var r = rel / p.row_bytes;
                    var done: usize = 0;
                    while (done < out.len) : ({
                        done += p.row_bytes;
                        r += 1;
                    }) {
                        const src_row = llamaPermutedRow(r, head_dim);
                        // Permuted views are only registered for local (GGUF) files.
                        const local = switch (self.source) {
                            .local => |f| f,
                            .remote => return error.UnexpectedEndOfFile,
                        };
                        const n = try local.readPositionalAll(io, out[done..][0..p.row_bytes], p.src_offset + @as(u64, src_row) * p.row_bytes);
                        if (n != p.row_bytes) return error.UnexpectedEndOfFile;
                    }
                },
            }
            return;
        }
        if (self.map) |m| {
            @memcpy(out, m.memory[@intCast(offset)..][0..out.len]);
            return;
        }
        switch (self.source) {
            .local => |f| {
                const n = try f.readPositionalAll(io, out, offset);
                if (n != out.len) return error.UnexpectedEndOfFile;
            },
            .remote => |rf| try rf.readRange(io, offset, out),
        }
    }
};

// ---------------------------------------------------------------------------
// Writing
// ---------------------------------------------------------------------------

pub const OutTensor = struct {
    name: []const u8,
    dtype: DType,
    shape: []const usize,
    data: []const u8,

    pub fn byteLen(self: OutTensor) usize {
        return self.data.len;
    }
};

/// Writes a single safetensors file with the given tensors (in order).
pub fn writeFile(gpa: std.mem.Allocator, io: Io, dir: Io.Dir, sub_path: []const u8, tensors: []const OutTensor, metadata: ?[]const u8) !void {
    var header: Io.Writer.Allocating = .init(gpa);
    defer header.deinit();
    const w = &header.writer;
    try w.writeAll("{");
    var first = true;
    if (metadata) |m| {
        try w.print("\"__metadata__\":{{\"format\":\"pt\",\"producer\":\"{s}\"}}", .{m});
        first = false;
    }
    var offset: usize = 0;
    for (tensors) |t| {
        if (!first) try w.writeAll(",");
        first = false;
        try w.writeAll("\"");
        try std.json.Stringify.encodeJsonStringChars(t.name, .{}, w);
        try w.print("\":{{\"dtype\":\"{s}\",\"shape\":[", .{t.dtype.safetensorsName()});
        for (t.shape, 0..) |s, i| {
            if (i > 0) try w.writeAll(",");
            try w.print("{d}", .{s});
        }
        try w.print("],\"data_offsets\":[{d},{d}]}}", .{ offset, offset + t.byteLen() });
        offset += t.byteLen();
    }
    try w.writeAll("}");
    // Pad header to a multiple of 8 with spaces, as the reference implementation does.
    while (header.written().len % 8 != 0) try w.writeAll(" ");
    const header_bytes = header.written();

    const file = try dir.createFile(io, sub_path, .{});
    defer file.close(io);
    var buf: [1 << 16]u8 = undefined;
    var fw = file.writer(io, &buf);
    const out = &fw.interface;
    var len_buf: [8]u8 = undefined;
    std.mem.writeInt(u64, &len_buf, header_bytes.len, .little);
    try out.writeAll(&len_buf);
    try out.writeAll(header_bytes);
    for (tensors) |t| try out.writeAll(t.data);
    try out.flush();
}

test "safetensors round trip" {
    const gpa = std.testing.allocator;
    var threaded: Io.Threaded = .init_single_threaded;
    const io = threaded.io();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const data = [_]f32{ 1, 2, 3, 4, 5, 6 };
    const shape = [_]usize{ 2, 3 };
    try writeFile(gpa, io, tmp.dir, "t.safetensors", &.{.{ .name = "w", .dtype = .f32, .shape = &shape, .data = std.mem.sliceAsBytes(&data) }}, "ditch");
    const f = try File.open(gpa, io, tmp.dir, "t.safetensors");
    defer f.close(gpa, io);
    const t = f.get("w").?;
    try std.testing.expectEqual(@as(usize, 2), t.shape[0]);
    const w = t.asWeight();
    var row: [3]f32 = undefined;
    w.row(1, &row);
    try std.testing.expectEqual(@as(f32, 5), row[1]);

    // Header-only open + positional read gives the same bytes.
    const f2 = try File.openOptions(gpa, io, tmp.dir, "t.safetensors", .{ .map = false });
    defer f2.close(gpa, io);
    const t2 = f2.get("w").?;
    try std.testing.expect(!f2.isMapped());
    try std.testing.expectEqual(@as(usize, 0), t2.data.len);
    try std.testing.expectEqual(t.byte_len, t2.byte_len);
    try std.testing.expectEqual(t.offset, t2.offset);
    const buf = try gpa.alloc(u8, t2.byte_len);
    defer gpa.free(buf);
    try f2.readTensor(io, t2, buf);
    try std.testing.expectEqualSlices(u8, t.data, buf);
}
