//! Reading and writing of the safetensors format (https://github.com/huggingface/safetensors).
//! Files are memory-mapped for reading so weights are never copied unless needed.

const std = @import("std");
const Io = std.Io;
const tensor = @import("tensor.zig");
const DType = tensor.DType;

pub const TensorInfo = struct {
    name: []const u8,
    dtype: DType,
    shape: []const usize,
    /// Raw bytes inside the mapped file.
    data: []const u8,

    pub fn numel(self: TensorInfo) usize {
        var n: usize = 1;
        for (self.shape) |s| n *= s;
        return n;
    }

    pub fn asWeight(self: TensorInfo) tensor.Weight {
        std.debug.assert(self.shape.len >= 1);
        const rows = self.shape[0];
        const cols = if (self.shape.len == 1) 1 else self.numel() / rows;
        return .{ .data = self.data, .dtype = self.dtype, .rows = rows, .cols = cols };
    }
};

pub const File = struct {
    path: []const u8,
    file: Io.File,
    map: Io.File.MemoryMap,
    header_len: usize,
    tensors: std.StringArrayHashMapUnmanaged(TensorInfo),
    arena: std.heap.ArenaAllocator,

    pub fn open(gpa: std.mem.Allocator, io: Io, dir: Io.Dir, sub_path: []const u8) !*File {
        const self = try gpa.create(File);
        errdefer gpa.destroy(self);
        self.* = .{
            .path = undefined,
            .file = undefined,
            .map = undefined,
            .header_len = 0,
            .tensors = .{},
            .arena = std.heap.ArenaAllocator.init(gpa),
        };
        errdefer self.arena.deinit();
        const arena = self.arena.allocator();
        self.path = try arena.dupe(u8, sub_path);
        self.file = try dir.openFile(io, sub_path, .{});
        errdefer self.file.close(io);
        const len: usize = @intCast(try self.file.length(io));
        if (len < 8) return error.InvalidSafetensors;
        self.map = try Io.File.MemoryMap.create(io, self.file, .{
            .len = len,
            .protection = .{ .read = true, .write = false },
            .populate = false,
        });
        errdefer self.map.destroy(io);
        const bytes = self.map.memory[0..len];
        const n = std.mem.readInt(u64, bytes[0..8], .little);
        if (n > len - 8) return error.InvalidSafetensors;
        self.header_len = @intCast(n);
        const header = bytes[8 .. 8 + self.header_len];
        const data = bytes[8 + self.header_len ..];

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
                continue;
            };
            const shape_val = (obj.object.get("shape") orelse return error.InvalidSafetensors).array;
            const shape = try arena.alloc(usize, shape_val.items.len);
            for (shape_val.items, 0..) |v, i| shape[i] = @intCast(v.integer);
            const offs = (obj.object.get("data_offsets") orelse return error.InvalidSafetensors).array;
            const start: usize = @intCast(offs.items[0].integer);
            const end: usize = @intCast(offs.items[1].integer);
            if (end > data.len or start > end) return error.InvalidSafetensors;
            try self.tensors.put(arena, try arena.dupe(u8, name), .{
                .name = try arena.dupe(u8, name),
                .dtype = dtype,
                .shape = shape,
                .data = data[start..end],
            });
        }
        return self;
    }

    pub fn close(self: *File, gpa: std.mem.Allocator, io: Io) void {
        self.map.destroy(io);
        self.file.close(io);
        self.arena.deinit();
        gpa.destroy(self);
    }

    pub fn get(self: *const File, name: []const u8) ?TensorInfo {
        return self.tensors.get(name);
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
}
