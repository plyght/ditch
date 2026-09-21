//! Exports a model with its abliteration deltas merged into the weights.

const std = @import("std");
const Io = std.Io;
const tensor = @import("tensor.zig");
const safetensors = @import("safetensors.zig");
const model_mod = @import("model.zig");

const Allocator = std.mem.Allocator;
const Model = model_mod.Model;

const Entry = struct {
    name: []const u8,
    info: safetensors.TensorInfo,
    out_dtype: tensor.DType,
    byte_len: usize,
    delta: ?tensor.Delta,
};

fn modifiedDelta(model: *const Model, name: []const u8) ?tensor.Delta {
    if (!std.mem.startsWith(u8, name, model.prefix)) return null;
    const rest = name[model.prefix.len..];
    if (!std.mem.startsWith(u8, rest, "layers.")) return null;
    const after = rest["layers.".len..];
    const dot = std.mem.indexOfScalar(u8, after, '.') orelse return null;
    const layer = std.fmt.parseInt(usize, after[0..dot], 10) catch return null;
    if (layer >= model.layers.len) return null;
    const suffix = after[dot + 1 ..];
    if (std.mem.eql(u8, suffix, "self_attn.o_proj.weight")) return model.getDelta(layer, .attn_o_proj);
    if (std.mem.eql(u8, suffix, "mlp.down_proj.weight")) return model.getDelta(layer, .mlp_down_proj);
    return null;
}

/// Produces the bytes of one tensor in the export dtype, merging the delta if present.
fn materialize(gpa: Allocator, e: Entry) ![]u8 {
    const w = e.info.asWeight();
    const f = try w.toF32(gpa);
    defer gpa.free(f);
    if (e.delta) |d| {
        for (0..w.rows) |i| {
            const row = f[i * w.cols ..][0..w.cols];
            for (0..d.rank) |k| tensor.axpy(row, d.b[i * d.rank + k], d.a[k * w.cols ..][0..w.cols]);
        }
    }
    const out = try gpa.alloc(u8, e.byte_len);
    tensor.convertFromF32(e.out_dtype, f, out);
    return out;
}

fn writeShard(gpa: Allocator, io: Io, dir: Io.Dir, name: []const u8, entries: []const Entry) !void {
    var header: Io.Writer.Allocating = .init(gpa);
    defer header.deinit();
    const w = &header.writer;
    try w.writeAll("{\"__metadata__\":{\"format\":\"pt\",\"producer\":\"ditch\"}");
    var offset: usize = 0;
    for (entries) |e| {
        try w.writeAll(",\"");
        try std.json.Stringify.encodeJsonStringChars(e.name, .{}, w);
        try w.print("\":{{\"dtype\":\"{s}\",\"shape\":[", .{e.out_dtype.safetensorsName()});
        for (e.info.shape, 0..) |s, i| {
            if (i > 0) try w.writeAll(",");
            try w.print("{d}", .{s});
        }
        try w.print("],\"data_offsets\":[{d},{d}]}}", .{ offset, offset + e.byte_len });
        offset += e.byte_len;
    }
    try w.writeAll("}");
    while (header.written().len % 8 != 0) try w.writeAll(" ");
    const hb = header.written();

    const file = try dir.createFile(io, name, .{});
    defer file.close(io);
    var buf: [1 << 18]u8 = undefined;
    var fw = file.writer(io, &buf);
    const out = &fw.interface;
    var len_buf: [8]u8 = undefined;
    std.mem.writeInt(u64, &len_buf, hb.len, .little);
    try out.writeAll(&len_buf);
    try out.writeAll(hb);
    for (entries) |e| {
        if (e.delta == null and e.out_dtype == e.info.dtype) {
            try out.writeAll(e.info.data);
        } else {
            const bytes = try materialize(gpa, e);
            defer gpa.free(bytes);
            try out.writeAll(bytes);
        }
    }
    try out.flush();
}

fn copyIfExists(io: Io, src: Io.Dir, dst: Io.Dir, name: []const u8) void {
    src.copyFile(name, dst, name, io, .{}) catch {};
}

pub const SaveOptions = struct {
    max_shard_size: u64 = 5 * 1024 * 1024 * 1024,
    export_dtype: ?tensor.DType = null,
    /// Markdown appended to the generated README.md (parameters, scores).
    readme_body: ?[]const u8 = null,
};

/// Saves the model (with merged deltas) to `out_dir` in Hugging Face format.
pub fn saveModel(gpa: Allocator, io: Io, model: *const Model, out_dir: []const u8, opts: SaveOptions, out: *Io.Writer) !void {
    const cwd = Io.Dir.cwd();
    try cwd.createDirPath(io, out_dir);
    var dir = try cwd.openDir(io, out_dir, .{});
    defer dir.close(io);

    var entries = std.ArrayList(Entry).empty;
    defer entries.deinit(gpa);
    var total: u64 = 0;
    for (model.files) |f| {
        var it = f.tensors.iterator();
        while (it.next()) |kv| {
            const info = kv.value_ptr.*;
            const out_dtype = opts.export_dtype orelse info.dtype;
            const byte_len = info.numel() * out_dtype.size();
            try entries.append(gpa, .{ .name = info.name, .info = info, .out_dtype = out_dtype, .byte_len = byte_len, .delta = modifiedDelta(model, info.name) });
            total += byte_len;
        }
    }

    // Group into shards.
    var shards = std.ArrayList([]const Entry).empty;
    defer shards.deinit(gpa);
    var start: usize = 0;
    var acc: u64 = 0;
    for (entries.items, 0..) |e, i| {
        if (i > start and acc + e.byte_len > opts.max_shard_size) {
            try shards.append(gpa, entries.items[start..i]);
            start = i;
            acc = 0;
        }
        acc += e.byte_len;
    }
    try shards.append(gpa, entries.items[start..]);

    if (shards.items.len == 1) {
        try out.writeAll("* Writing model.safetensors...\n");
        try out.flush();
        try writeShard(gpa, io, dir, "model.safetensors", shards.items[0]);
    } else {
        var index: Io.Writer.Allocating = .init(gpa);
        defer index.deinit();
        try index.writer.print("{{\n  \"metadata\": {{\n    \"total_size\": {d}\n  }},\n  \"weight_map\": {{\n", .{total});
        var first = true;
        for (shards.items, 0..) |shard, si| {
            const name = try std.fmt.allocPrint(gpa, "model-{d:0>5}-of-{d:0>5}.safetensors", .{ si + 1, shards.items.len });
            defer gpa.free(name);
            try out.print("* Writing {s}...\n", .{name});
            try out.flush();
            try writeShard(gpa, io, dir, name, shard);
            for (shard) |e| {
                if (!first) try index.writer.writeAll(",\n");
                first = false;
                try index.writer.writeAll("    \"");
                try std.json.Stringify.encodeJsonStringChars(e.name, .{}, &index.writer);
                try index.writer.print("\": \"{s}\"", .{name});
            }
        }
        try index.writer.writeAll("\n  }\n}\n");
        try dir.writeFile(io, .{ .sub_path = "model.safetensors.index.json", .data = index.written() });
    }

    // Config and tokenizer files.
    try dir.writeFile(io, .{ .sub_path = "config.json", .data = model.config_json });
    try dir.writeFile(io, .{ .sub_path = "tokenizer.json", .data = model.tokenizer_json });
    if (model.generation_config_json) |g| try dir.writeFile(io, .{ .sub_path = "generation_config.json", .data = g });
    if (model.tokenizer_config_json) |t| try dir.writeFile(io, .{ .sub_path = "tokenizer_config.json", .data = t });
    var src = cwd.openDir(io, model.source_dir, .{}) catch null;
    if (src) |*s| {
        defer s.close(io);
        copyIfExists(io, s.*, dir, "special_tokens_map.json");
        copyIfExists(io, s.*, dir, "chat_template.jinja");
        copyIfExists(io, s.*, dir, "added_tokens.json");
        copyIfExists(io, s.*, dir, "preprocessor_config.json");
    }
    if (opts.readme_body) |body| try dir.writeFile(io, .{ .sub_path = "README.md", .data = body });
}

test "export round trip merges deltas" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    var threaded: Io.Threaded = .init_single_threaded;
    _ = &threaded;
    const pool = tensor.Pool.init(io, 1);
    const model = try Model.load(gpa, io, &pool, "tests/fixtures/qwen2");
    defer model.deinit();
    // Rank-1 delta on layer 0 o_proj: W' = W + b aᵀ with a = e0, b = ones.
    const w = model.componentWeight(0, .attn_o_proj);
    const a = try gpa.alloc(f32, w.cols);
    @memset(a, 0);
    a[0] = 1;
    const b = try gpa.alloc(f32, w.rows);
    @memset(b, 1);
    model.setDelta(0, .attn_o_proj, .{ .rank = 1, .a = a, .b = b });

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const n = try tmp.dir.realPath(io, &path_buf);
    const out_dir = try std.fs.path.join(gpa, &.{ path_buf[0..n], "exported" });
    defer gpa.free(out_dir);
    var sink: Io.Writer.Allocating = .init(gpa);
    defer sink.deinit();
    try saveModel(gpa, io, model, out_dir, .{ .max_shard_size = 4096 }, &sink.writer);

    const reloaded = try Model.load(gpa, io, &pool, out_dir);
    defer reloaded.deinit();
    const w2 = reloaded.componentWeight(0, .attn_o_proj);
    const r0 = try gpa.alloc(f32, w.cols);
    defer gpa.free(r0);
    const r1 = try gpa.alloc(f32, w.cols);
    defer gpa.free(r1);
    w.row(3, r0);
    w2.row(3, r1);
    try std.testing.expectApproxEqAbs(r0[0] + 1.0, r1[0], 0.02);
    try std.testing.expectApproxEqAbs(r0[1], r1[1], 1e-6);
    // Untouched tensors are byte-identical.
    const e0 = model.embed;
    const e1 = reloaded.embed;
    try std.testing.expectEqualSlices(u8, e0.data, e1.data);
    try std.testing.expect(reloaded.files.len > 1);
}
