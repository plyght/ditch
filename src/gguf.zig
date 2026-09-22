//! The GGUF container format (https://github.com/ggml-org/ggml/blob/master/docs/gguf.md),
//! version 3, little endian: a reader over a memory-mapped file and a writer
//! that emits the header first so tensor data can be streamed afterwards.
//! Model-level conversion lives in gguf_model.zig (input) and gguf_export.zig (output).

const std = @import("std");
const Io = std.Io;
const tensor = @import("tensor.zig");
const DType = tensor.DType;

const Allocator = std.mem.Allocator;

pub const magic = "GGUF";
pub const version: u32 = 3;
pub const default_alignment: usize = 32;

/// Metadata value types (`gguf_metadata_value_type`).
pub const ValueType = enum(u32) {
    uint8 = 0,
    int8 = 1,
    uint16 = 2,
    int16 = 3,
    uint32 = 4,
    int32 = 5,
    float32 = 6,
    bool = 7,
    string = 8,
    array = 9,
    uint64 = 10,
    int64 = 11,
    float64 = 12,
};

pub const Value = union(ValueType) {
    uint8: u8,
    int8: i8,
    uint16: u16,
    int16: i16,
    uint32: u32,
    int32: i32,
    float32: f32,
    bool: bool,
    string: []const u8,
    array: Array,
    uint64: u64,
    int64: i64,
    float64: f64,

    pub const Array = struct {
        elem: ValueType,
        items: []const Value,
    };

    pub fn asInt(self: Value) ?i64 {
        return switch (self) {
            .uint8 => |v| v,
            .int8 => |v| v,
            .uint16 => |v| v,
            .int16 => |v| v,
            .uint32 => |v| v,
            .int32 => |v| v,
            .uint64 => |v| if (v <= std.math.maxInt(i64)) @intCast(v) else null,
            .int64 => |v| v,
            .bool => |v| @intFromBool(v),
            .float32 => |v| @intFromFloat(v),
            .float64 => |v| @intFromFloat(v),
            else => null,
        };
    }

    pub fn asFloat(self: Value) ?f64 {
        return switch (self) {
            .float32 => |v| v,
            .float64 => |v| v,
            .uint8, .int8, .uint16, .int16, .uint32, .int32, .uint64, .int64 => @floatFromInt(self.asInt().?),
            else => null,
        };
    }

    pub fn asString(self: Value) ?[]const u8 {
        return switch (self) {
            .string => |s| s,
            else => null,
        };
    }

    pub fn asBool(self: Value) ?bool {
        return switch (self) {
            .bool => |b| b,
            else => if (self.asInt()) |i| i != 0 else null,
        };
    }
};

/// ggml_type ids used in tensor infos.
pub const GgmlType = enum(u32) {
    f32 = 0,
    f16 = 1,
    q4_0 = 2,
    q4_1 = 3,
    q5_0 = 6,
    q5_1 = 7,
    q8_0 = 8,
    q8_1 = 9,
    q2_k = 10,
    q3_k = 11,
    q4_k = 12,
    q5_k = 13,
    q6_k = 14,
    q8_k = 15,
    iq2_xxs = 16,
    iq2_xs = 17,
    iq3_xxs = 18,
    iq1_s = 19,
    iq4_nl = 20,
    iq3_s = 21,
    iq2_s = 22,
    iq4_xs = 23,
    i8 = 24,
    i16 = 25,
    i32 = 26,
    i64 = 27,
    f64 = 28,
    iq1_m = 29,
    bf16 = 30,
    _,

    pub fn name(self: GgmlType) []const u8 {
        return switch (self) {
            .f32 => "F32",
            .f16 => "F16",
            .q4_0 => "Q4_0",
            .q4_1 => "Q4_1",
            .q5_0 => "Q5_0",
            .q5_1 => "Q5_1",
            .q8_0 => "Q8_0",
            .q8_1 => "Q8_1",
            .q2_k => "Q2_K",
            .q3_k => "Q3_K",
            .q4_k => "Q4_K",
            .q5_k => "Q5_K",
            .q6_k => "Q6_K",
            .q8_k => "Q8_K",
            .bf16 => "BF16",
            else => "?",
        };
    }
};

pub fn dtypeFromGgml(t: GgmlType) ?DType {
    return switch (t) {
        .f32 => .f32,
        .f16 => .f16,
        .bf16 => .bf16,
        .q8_0 => .q8_0,
        .q4_0 => .q4_0,
        .q4_1 => .q4_1,
        .q5_0 => .q5_0,
        .q5_1 => .q5_1,
        .q4_k => .q4_k,
        .q6_k => .q6_k,
        .q8_k => .q8_k,
        else => null,
    };
}

pub fn ggmlFromDtype(d: DType) GgmlType {
    return switch (d) {
        .f32 => .f32,
        .f16 => .f16,
        .bf16 => .bf16,
        .i64 => .i64,
        .q8_0 => .q8_0,
        .q4_0 => .q4_0,
        .q4_1 => .q4_1,
        .q5_0 => .q5_0,
        .q5_1 => .q5_1,
        .q4_k => .q4_k,
        .q6_k => .q6_k,
        .q8_k => .q8_k,
    };
}

/// `general.file_type` values (`LlamaFileType`).
pub fn fileType(d: DType) u32 {
    return switch (d) {
        .f32 => 0,
        .f16 => 1,
        .q4_0 => 2,
        .q4_1 => 3,
        .q8_0 => 7,
        .q5_0 => 8,
        .q5_1 => 9,
        .q4_k => 15,
        .q6_k => 18,
        .bf16 => 32,
        .i64 => 0,
        .q8_k => 1024, // "guessed"
    };
}

pub fn alignOffset(offset: usize, alignment: usize) usize {
    return offset + (alignment - offset % alignment) % alignment;
}

/// One tensor of a GGUF file.
pub const TensorInfo = struct {
    name: []const u8,
    /// Dimensions as stored: `dims[0]` is the fastest-varying (the row length).
    dims: []const u64,
    ggml_type: GgmlType,
    offset: u64,
    /// Raw bytes inside the mapped file.
    data: []const u8,

    pub fn dtype(self: TensorInfo) ?DType {
        return dtypeFromGgml(self.ggml_type);
    }

    pub fn numel(self: TensorInfo) usize {
        var n: usize = 1;
        for (self.dims) |d| n *= @intCast(d);
        return n;
    }

    /// Shape in row-major (Hugging Face) order, i.e. `dims` reversed.
    pub fn shape(self: TensorInfo, a: Allocator) ![]usize {
        const out = try a.alloc(usize, self.dims.len);
        for (out, 0..) |*s, i| s.* = @intCast(self.dims[self.dims.len - 1 - i]);
        return out;
    }

    /// Byte length of the tensor data for its type and dimensions.
    pub fn byteLen(self: TensorInfo) ?usize {
        const d = self.dtype() orelse return null;
        const cols: usize = if (self.dims.len > 0) @intCast(self.dims[0]) else 1;
        if (cols % d.blockSize() != 0) return null;
        return d.byteLen(self.numel(), cols);
    }
};

// ---------------------------------------------------------------------------
// Reading
// ---------------------------------------------------------------------------

const Cursor = struct {
    bytes: []const u8,
    pos: usize = 0,
    /// `bytes` is a prefix of the file: running out means the prefix was too short.
    partial: bool = false,

    fn need(self: *Cursor, n: usize) ![]const u8 {
        if (self.pos + n > self.bytes.len) return if (self.partial) error.HeaderTruncated else error.InvalidGguf;
        const s = self.bytes[self.pos..][0..n];
        self.pos += n;
        return s;
    }

    fn int(self: *Cursor, comptime T: type) !T {
        const b = try self.need(@sizeOf(T));
        return std.mem.readInt(T, b[0..@sizeOf(T)], .little);
    }

    fn string(self: *Cursor, arena: Allocator) ![]const u8 {
        const len = try self.int(u64);
        if (len > self.bytes.len) return error.InvalidGguf;
        return arena.dupe(u8, try self.need(@intCast(len)));
    }

    fn value(self: *Cursor, arena: Allocator, t: ValueType) !Value {
        return switch (t) {
            .uint8 => .{ .uint8 = try self.int(u8) },
            .int8 => .{ .int8 = @bitCast(try self.int(u8)) },
            .uint16 => .{ .uint16 = try self.int(u16) },
            .int16 => .{ .int16 = @bitCast(try self.int(u16)) },
            .uint32 => .{ .uint32 = try self.int(u32) },
            .int32 => .{ .int32 = @bitCast(try self.int(u32)) },
            .float32 => .{ .float32 = @bitCast(try self.int(u32)) },
            .bool => .{ .bool = (try self.int(u8)) != 0 },
            .string => .{ .string = try self.string(arena) },
            .uint64 => .{ .uint64 = try self.int(u64) },
            .int64 => .{ .int64 = @bitCast(try self.int(u64)) },
            .float64 => .{ .float64 = @bitCast(try self.int(u64)) },
            .array => blk: {
                const elem = try self.valueType();
                const n = try self.int(u64);
                if (n > self.bytes.len) return error.InvalidGguf;
                const items = try arena.alloc(Value, @intCast(n));
                for (items) |*it| it.* = try self.value(arena, elem);
                break :blk .{ .array = .{ .elem = elem, .items = items } };
            },
        };
    }

    fn valueType(self: *Cursor) !ValueType {
        const raw = try self.int(u32);
        if (raw > @intFromEnum(ValueType.float64)) return error.InvalidGguf;
        return @enumFromInt(raw);
    }
};

pub const OpenOptions = struct {
    /// Memory-map the whole file. When false only the header is read (with
    /// positional reads) and `TensorInfo.data` stays empty; tensor bytes are
    /// read with `readRange`.
    map: bool = true,
};

pub const File = struct {
    path: []const u8,
    file: Io.File,
    map: ?Io.File.MemoryMap,
    /// Total file length in bytes.
    len: u64,
    arena: std.heap.ArenaAllocator,
    kv: std.StringArrayHashMapUnmanaged(Value),
    tensors: std.StringArrayHashMapUnmanaged(TensorInfo),
    alignment: usize,
    data_offset: usize,

    pub fn open(gpa: Allocator, io: Io, dir: Io.Dir, sub_path: []const u8) !*File {
        return openOptions(gpa, io, dir, sub_path, .{});
    }

    pub fn openOptions(gpa: Allocator, io: Io, dir: Io.Dir, sub_path: []const u8, options: OpenOptions) !*File {
        const self = try gpa.create(File);
        errdefer gpa.destroy(self);
        self.* = .{
            .path = undefined,
            .file = undefined,
            .map = null,
            .len = 0,
            .arena = std.heap.ArenaAllocator.init(gpa),
            .kv = .{},
            .tensors = .{},
            .alignment = default_alignment,
            .data_offset = 0,
        };
        errdefer self.arena.deinit();
        const arena = self.arena.allocator();
        self.path = try arena.dupe(u8, sub_path);
        self.file = try dir.openFile(io, sub_path, .{});
        errdefer self.file.close(io);
        const len: usize = @intCast(try self.file.length(io));
        self.len = len;
        if (len < 24) return error.InvalidGguf;
        if (options.map) {
            self.map = try Io.File.MemoryMap.create(io, self.file, .{
                .len = len,
                .protection = .{ .read = true, .write = false },
                .populate = false,
            });
            errdefer self.map.?.destroy(io);
            try self.parse(arena, self.map.?.memory[0..len], true);
        } else {
            // The header length is not stored: read a growing prefix until it parses.
            var want: usize = @min(len, 1 << 20);
            while (true) {
                const buf = try gpa.alloc(u8, want);
                defer gpa.free(buf);
                const n = try self.file.readPositionalAll(io, buf, 0);
                if (n != want) return error.InvalidGguf;
                self.parse(arena, buf[0..n], false) catch |err| switch (err) {
                    error.HeaderTruncated => {
                        if (want >= len) return error.InvalidGguf;
                        self.kv = .{};
                        self.tensors = .{};
                        want = @min(len, want * 4);
                        continue;
                    },
                    else => return err,
                };
                break;
            }
        }
        return self;
    }

    /// Positional read of `out.len` bytes at absolute file `offset`.
    pub fn readRange(self: *const File, io: Io, offset: u64, out: []u8) !void {
        if (self.map) |m| {
            @memcpy(out, m.memory[@intCast(offset)..][0..out.len]);
            return;
        }
        const n = try self.file.readPositionalAll(io, out, offset);
        if (n != out.len) return error.UnexpectedEndOfFile;
    }

    /// Absolute file offset of a tensor's first byte.
    pub fn tensorOffset(self: *const File, t: TensorInfo) u64 {
        return @as(u64, self.data_offset) + t.offset;
    }

    fn parse(self: *File, arena: Allocator, bytes: []const u8, whole: bool) !void {
        var c = Cursor{ .bytes = bytes, .partial = !whole };
        if (!std.mem.eql(u8, try c.need(4), magic)) return error.InvalidGguf;
        const ver = try c.int(u32);
        if (ver != 2 and ver != 3) {
            std.log.err("unsupported GGUF version {d} (expected 2 or 3)", .{ver});
            return error.UnsupportedGguf;
        }
        const n_tensors = try c.int(u64);
        const n_kv = try c.int(u64);
        if (n_tensors > self.len or n_kv > self.len) return error.InvalidGguf;
        var i: usize = 0;
        while (i < n_kv) : (i += 1) {
            const key = try c.string(arena);
            const t = try c.valueType();
            const v = try c.value(arena, t);
            try self.kv.put(arena, key, v);
        }
        if (self.kv.get("general.alignment")) |a| {
            const al = a.asInt() orelse return error.InvalidGguf;
            if (al <= 0 or @rem(al, 8) != 0) return error.InvalidGguf;
            self.alignment = @intCast(al);
        }
        i = 0;
        var infos = try arena.alloc(TensorInfo, @intCast(n_tensors));
        while (i < n_tensors) : (i += 1) {
            const name = try c.string(arena);
            const n_dims = try c.int(u32);
            if (n_dims > 8) return error.InvalidGguf;
            const dims = try arena.alloc(u64, n_dims);
            for (dims) |*d| d.* = try c.int(u64);
            const t: GgmlType = @enumFromInt(try c.int(u32));
            const offset = try c.int(u64);
            infos[i] = .{ .name = name, .dims = dims, .ggml_type = t, .offset = offset, .data = &.{} };
        }
        self.data_offset = alignOffset(c.pos, self.alignment);
        if (self.data_offset > self.len) return error.InvalidGguf;
        const data_len: usize = @intCast(self.len - self.data_offset);
        for (infos) |*info| {
            const start: usize = @intCast(info.offset);
            if (info.byteLen()) |bl| {
                if (start > data_len or bl > data_len - start) return error.InvalidGguf;
                if (whole) info.data = bytes[self.data_offset + start ..][0..bl];
            } else {
                std.log.warn("skipping tensor {s} with unsupported ggml type {d}", .{ info.name, @intFromEnum(info.ggml_type) });
                continue;
            }
            try self.tensors.put(arena, info.name, info.*);
        }
    }

    pub fn close(self: *File, gpa: Allocator, io: Io) void {
        if (self.map) |*m| m.destroy(io);
        self.file.close(io);
        self.arena.deinit();
        gpa.destroy(self);
    }

    pub fn get(self: *const File, key: []const u8) ?Value {
        return self.kv.get(key);
    }

    pub fn getTensor(self: *const File, name: []const u8) ?TensorInfo {
        return self.tensors.get(name);
    }

    pub fn getStr(self: *const File, key: []const u8) ?[]const u8 {
        const v = self.get(key) orelse return null;
        return v.asString();
    }

    pub fn getInt(self: *const File, key: []const u8) ?i64 {
        const v = self.get(key) orelse return null;
        return v.asInt();
    }

    pub fn getFloat(self: *const File, key: []const u8) ?f64 {
        const v = self.get(key) orelse return null;
        return v.asFloat();
    }

    pub fn getBool(self: *const File, key: []const u8) ?bool {
        const v = self.get(key) orelse return null;
        return v.asBool();
    }

    pub fn getArray(self: *const File, key: []const u8) ?Value.Array {
        const v = self.get(key) orelse return null;
        return switch (v) {
            .array => |a| a,
            else => null,
        };
    }

    pub fn architecture(self: *const File) ?[]const u8 {
        return self.getStr("general.architecture");
    }

    /// Looks up `<architecture>.<suffix>`.
    pub fn archValue(self: *const File, suffix: []const u8) ?Value {
        const arch = self.architecture() orelse return null;
        var buf: [256]u8 = undefined;
        const key = std.fmt.bufPrint(&buf, "{s}.{s}", .{ arch, suffix }) catch return null;
        return self.get(key);
    }

    pub fn archInt(self: *const File, suffix: []const u8) ?i64 {
        const v = self.archValue(suffix) orelse return null;
        return v.asInt();
    }

    pub fn archFloat(self: *const File, suffix: []const u8) ?f64 {
        const v = self.archValue(suffix) orelse return null;
        return v.asFloat();
    }

    pub fn archStr(self: *const File, suffix: []const u8) ?[]const u8 {
        const v = self.archValue(suffix) orelse return null;
        return v.asString();
    }
};

// ---------------------------------------------------------------------------
// Writing
// ---------------------------------------------------------------------------

/// Describes one tensor to be written; the data follows the header in order.
pub const OutTensor = struct {
    name: []const u8,
    /// Row-major (Hugging Face) shape; written reversed.
    shape: []const usize,
    dtype: DType,
    byte_len: usize,
    offset: usize,
};

const KvEntry = struct { key: []const u8, value: Value };

/// Collects metadata and tensor infos, then writes the header. Tensor bytes are
/// written by the caller afterwards, each followed by `padding(byte_len)` zero bytes.
pub const Writer = struct {
    arena: std.heap.ArenaAllocator,
    kv: std.ArrayList(KvEntry) = .empty,
    tensors: std.ArrayList(OutTensor) = .empty,
    alignment: usize = default_alignment,
    data_len: usize = 0,

    pub fn init(gpa: Allocator) Writer {
        return .{ .arena = std.heap.ArenaAllocator.init(gpa) };
    }

    pub fn deinit(self: *Writer) void {
        self.arena.deinit();
    }

    fn a(self: *Writer) Allocator {
        return self.arena.allocator();
    }

    pub fn add(self: *Writer, key: []const u8, value: Value) !void {
        const arena = self.a();
        const owned: Value = switch (value) {
            .string => |s| .{ .string = try arena.dupe(u8, s) },
            .array => |arr| blk: {
                const items = try arena.alloc(Value, arr.items.len);
                for (arr.items, 0..) |it, i| items[i] = switch (it) {
                    .string => |s| .{ .string = try arena.dupe(u8, s) },
                    else => it,
                };
                break :blk .{ .array = .{ .elem = arr.elem, .items = items } };
            },
            else => value,
        };
        // Replace an existing key.
        for (self.kv.items) |*e| {
            if (std.mem.eql(u8, e.key, key)) {
                e.value = owned;
                return;
            }
        }
        try self.kv.append(arena, .{ .key = try arena.dupe(u8, key), .value = owned });
    }

    pub fn addString(self: *Writer, key: []const u8, s: []const u8) !void {
        try self.add(key, .{ .string = s });
    }
    pub fn addU32(self: *Writer, key: []const u8, v: u32) !void {
        try self.add(key, .{ .uint32 = v });
    }
    pub fn addI32(self: *Writer, key: []const u8, v: i32) !void {
        try self.add(key, .{ .int32 = v });
    }
    pub fn addU64(self: *Writer, key: []const u8, v: u64) !void {
        try self.add(key, .{ .uint64 = v });
    }
    pub fn addF32(self: *Writer, key: []const u8, v: f32) !void {
        try self.add(key, .{ .float32 = v });
    }
    pub fn addBool(self: *Writer, key: []const u8, v: bool) !void {
        try self.add(key, .{ .bool = v });
    }
    pub fn addStringArray(self: *Writer, key: []const u8, items: []const []const u8) !void {
        const vals = try self.a().alloc(Value, items.len);
        for (items, 0..) |s, i| vals[i] = .{ .string = s };
        try self.add(key, .{ .array = .{ .elem = .string, .items = vals } });
    }
    pub fn addI32Array(self: *Writer, key: []const u8, items: []const i32) !void {
        const vals = try self.a().alloc(Value, items.len);
        for (items, 0..) |v, i| vals[i] = .{ .int32 = v };
        try self.add(key, .{ .array = .{ .elem = .int32, .items = vals } });
    }
    pub fn addF32Array(self: *Writer, key: []const u8, items: []const f32) !void {
        const vals = try self.a().alloc(Value, items.len);
        for (items, 0..) |v, i| vals[i] = .{ .float32 = v };
        try self.add(key, .{ .array = .{ .elem = .float32, .items = vals } });
    }

    /// Registers a tensor; its data must be written in registration order.
    pub fn addTensor(self: *Writer, name: []const u8, shape: []const usize, dtype: DType) !void {
        const arena = self.a();
        var numel: usize = 1;
        for (shape) |s| numel *= s;
        const cols = if (shape.len > 0) shape[shape.len - 1] else 1;
        if (cols % dtype.blockSize() != 0) return error.InvalidShapeForDType;
        const byte_len = dtype.byteLen(numel, cols);
        try self.tensors.append(arena, .{
            .name = try arena.dupe(u8, name),
            .shape = try arena.dupe(usize, shape),
            .dtype = dtype,
            .byte_len = byte_len,
            .offset = self.data_len,
        });
        self.data_len += alignOffset(byte_len, self.alignment);
    }

    /// Zero bytes to append after a tensor of `byte_len` bytes.
    pub fn padding(self: *const Writer, byte_len: usize) usize {
        return alignOffset(byte_len, self.alignment) - byte_len;
    }

    fn writeString(w: *Io.Writer, s: []const u8) !void {
        try w.writeInt(u64, s.len, .little);
        try w.writeAll(s);
    }

    fn writeValue(w: *Io.Writer, v: Value) !void {
        switch (v) {
            .uint8 => |x| try w.writeInt(u8, x, .little),
            .int8 => |x| try w.writeInt(i8, x, .little),
            .uint16 => |x| try w.writeInt(u16, x, .little),
            .int16 => |x| try w.writeInt(i16, x, .little),
            .uint32 => |x| try w.writeInt(u32, x, .little),
            .int32 => |x| try w.writeInt(i32, x, .little),
            .float32 => |x| try w.writeInt(u32, @bitCast(x), .little),
            .bool => |x| try w.writeInt(u8, @intFromBool(x), .little),
            .string => |s| try writeString(w, s),
            .uint64 => |x| try w.writeInt(u64, x, .little),
            .int64 => |x| try w.writeInt(i64, x, .little),
            .float64 => |x| try w.writeInt(u64, @bitCast(x), .little),
            .array => |arr| {
                try w.writeInt(u32, @intFromEnum(arr.elem), .little);
                try w.writeInt(u64, arr.items.len, .little);
                for (arr.items) |it| {
                    if (@as(ValueType, it) != arr.elem) return error.MixedArray;
                    try writeValue(w, it);
                }
            },
        }
    }

    /// Writes magic, version, counts, metadata and tensor infos, padded to the alignment.
    pub fn writeHeader(self: *Writer, w: *Io.Writer) !void {
        var header: Io.Writer.Allocating = .init(self.a());
        const h = &header.writer;
        try h.writeAll(magic);
        try h.writeInt(u32, version, .little);
        try h.writeInt(u64, self.tensors.items.len, .little);
        try h.writeInt(u64, self.kv.items.len, .little);
        for (self.kv.items) |e| {
            try writeString(h, e.key);
            try h.writeInt(u32, @intFromEnum(@as(ValueType, e.value)), .little);
            try writeValue(h, e.value);
        }
        for (self.tensors.items) |t| {
            try writeString(h, t.name);
            try h.writeInt(u32, @intCast(t.shape.len), .little);
            var i: usize = t.shape.len;
            while (i > 0) : (i -= 1) try h.writeInt(u64, t.shape[i - 1], .little);
            try h.writeInt(u32, @intFromEnum(ggmlFromDtype(t.dtype)), .little);
            try h.writeInt(u64, t.offset, .little);
        }
        const pad = self.padding(header.written().len);
        try h.splatByteAll(0, pad);
        try w.writeAll(header.written());
    }
};

/// Writes a complete small GGUF file from in-memory tensors (tests and fixtures).
pub fn writeFile(gpa: Allocator, io: Io, dir: Io.Dir, sub_path: []const u8, w: *Writer, datas: []const []const u8) !void {
    std.debug.assert(datas.len == w.tensors.items.len);
    const file = try dir.createFile(io, sub_path, .{});
    defer file.close(io);
    var buf: [1 << 16]u8 = undefined;
    var fw = file.writer(io, &buf);
    const out = &fw.interface;
    try w.writeHeader(out);
    for (datas, 0..) |d, i| {
        std.debug.assert(d.len == w.tensors.items[i].byte_len);
        try out.writeAll(d);
        try out.splatByteAll(0, w.padding(d.len));
    }
    try out.flush();
    _ = gpa;
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

test "gguf header and metadata round trip" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    var w = Writer.init(gpa);
    defer w.deinit();
    try w.addString("general.architecture", "llama");
    try w.addU32("llama.block_count", 3);
    try w.addF32("llama.attention.layer_norm_rms_epsilon", 1e-6);
    try w.addBool("tokenizer.ggml.add_bos_token", true);
    try w.addU64("big", 1 << 40);
    try w.addI32("neg", -5);
    try w.add("f64", .{ .float64 = 2.5 });
    try w.add("i8", .{ .int8 = -3 });
    try w.add("u16", .{ .uint16 = 65535 });
    try w.addStringArray("tokenizer.ggml.tokens", &.{ "a", "bc", "" });
    try w.addI32Array("tokenizer.ggml.token_type", &.{ 1, 3, 4 });
    try w.addF32Array("tokenizer.ggml.scores", &.{ 0, -1.5 });
    const shape2 = [_]usize{ 2, 32 };
    try w.addTensor("blk.0.attn_q.weight", &shape2, .f16);
    const shape1 = [_]usize{5};
    try w.addTensor("output_norm.weight", &shape1, .f32);
    try w.addTensor("blk.0.ffn_down.weight", &.{ 3, 64 }, .q8_0);
    try std.testing.expectEqual(@as(usize, 128), w.tensors.items[0].byte_len);
    try std.testing.expectEqual(@as(usize, 128), w.tensors.items[1].offset);
    try std.testing.expectEqual(@as(usize, 160), w.tensors.items[2].offset);
    try std.testing.expectEqual(@as(usize, 3 * 2 * 34), w.tensors.items[2].byte_len);

    var t0: [128]u8 = undefined;
    for (&t0, 0..) |*b, i| b.* = @intCast(i);
    var t1: [20]u8 = undefined;
    for (&t1, 0..) |*b, i| b.* = @intCast(200 + i);
    var t2: [3 * 2 * 34]u8 = undefined;
    for (&t2, 0..) |*b, i| b.* = @intCast(i % 7);
    try writeFile(gpa, io, tmp.dir, "t.gguf", &w, &.{ &t0, &t1, &t2 });

    const f = try File.open(gpa, io, tmp.dir, "t.gguf");
    defer f.close(gpa, io);
    try std.testing.expectEqualStrings("llama", f.architecture().?);
    try std.testing.expectEqual(@as(i64, 3), f.archInt("block_count").?);
    try std.testing.expectApproxEqAbs(@as(f64, 1e-6), f.archFloat("attention.layer_norm_rms_epsilon").?, 1e-12);
    try std.testing.expect(f.getBool("tokenizer.ggml.add_bos_token").?);
    try std.testing.expectEqual(@as(i64, 1 << 40), f.getInt("big").?);
    try std.testing.expectEqual(@as(i64, -5), f.getInt("neg").?);
    try std.testing.expectEqual(@as(f64, 2.5), f.getFloat("f64").?);
    try std.testing.expectEqual(@as(i64, -3), f.getInt("i8").?);
    try std.testing.expectEqual(@as(i64, 65535), f.getInt("u16").?);
    const toks = f.getArray("tokenizer.ggml.tokens").?;
    try std.testing.expectEqual(ValueType.string, toks.elem);
    try std.testing.expectEqual(@as(usize, 3), toks.items.len);
    try std.testing.expectEqualStrings("bc", toks.items[1].string);
    try std.testing.expectEqualStrings("", toks.items[2].string);
    try std.testing.expectEqual(@as(i64, 4), f.getArray("tokenizer.ggml.token_type").?.items[2].asInt().?);
    try std.testing.expectEqual(@as(f64, -1.5), f.getArray("tokenizer.ggml.scores").?.items[1].asFloat().?);
    try std.testing.expectEqual(@as(usize, 0), f.data_offset % default_alignment);

    const q = f.getTensor("blk.0.attn_q.weight").?;
    try std.testing.expectEqual(GgmlType.f16, q.ggml_type);
    try std.testing.expectEqual(@as(u64, 32), q.dims[0]);
    try std.testing.expectEqual(@as(u64, 2), q.dims[1]);
    try std.testing.expectEqualSlices(u8, &t0, q.data);
    const n = f.getTensor("output_norm.weight").?;
    try std.testing.expectEqual(@as(usize, 1), n.dims.len);
    try std.testing.expectEqualSlices(u8, &t1, n.data);
    const dn = f.getTensor("blk.0.ffn_down.weight").?;
    try std.testing.expectEqual(DType.q8_0, dn.dtype().?);
    try std.testing.expectEqualSlices(u8, &t2, dn.data);
    const shape = try dn.shape(gpa);
    defer gpa.free(shape);
    try std.testing.expectEqualSlices(usize, &.{ 3, 64 }, shape);

    // Header-only open with positional reads sees the same metadata and bytes.
    const h = try File.openOptions(gpa, io, tmp.dir, "t.gguf", .{ .map = false });
    defer h.close(gpa, io);
    try std.testing.expectEqualStrings("llama", h.architecture().?);
    try std.testing.expectEqual(f.data_offset, h.data_offset);
    const hq = h.getTensor("blk.0.attn_q.weight").?;
    try std.testing.expectEqual(@as(usize, 0), hq.data.len);
    var buf: [128]u8 = undefined;
    try h.readRange(io, h.tensorOffset(hq), &buf);
    try std.testing.expectEqualSlices(u8, &t0, &buf);
    var buf2: [20]u8 = undefined;
    try h.readRange(io, h.tensorOffset(h.getTensor("output_norm.weight").?), &buf2);
    try std.testing.expectEqualSlices(u8, &t1, &buf2);
}
