//! Exports a model with its abliteration deltas merged into the weights.
//!
//! Tensors are written one at a time through the model's weight store, so peak
//! memory is one tensor (plus its f32 copy when a delta is merged) and a
//! bounded conversion chunk, regardless of model size. A `.incomplete` marker
//! is created before the first shard and removed only after everything was
//! written, so a directory never looks complete when it is not.

const std = @import("std");
const Io = std.Io;
const tensor = @import("tensor.zig");
const safetensors = @import("safetensors.zig");
const model_mod = @import("model.zig");
const stream = @import("stream.zig");

const Allocator = std.mem.Allocator;
const Model = model_mod.Model;

/// How a tensor is modified on export.
const Edit = union(enum) {
    /// A 2-D matrix with one delta: `W' = W + B A`.
    whole: tensor.Delta,
    /// A fused expert down tensor of this layer; per-expert deltas are merged slice by slice.
    fused_down: usize,
};

const Entry = struct {
    name: []const u8,
    info: safetensors.TensorInfo,
    ref: stream.WeightRef,
    out_dtype: tensor.DType,
    byte_len: usize,
    edit: ?Edit,
};

/// Conversion chunk: rows are converted into a buffer of about this size before writing.
pub const convert_chunk_bytes: usize = 1 << 18;

fn modifiedDelta(model: *const Model, name: []const u8) ?Edit {
    const layer = model_mod.layerIndexOf(model.prefix, name) orelse return null;
    if (layer >= model.layers.len) return null;
    const rest = name[model.prefix.len + "layers.".len ..];
    const dot = std.mem.indexOfScalar(u8, rest, '.') orelse return null;
    const suffix = rest[dot + 1 ..];
    if (std.mem.eql(u8, suffix, "self_attn.o_proj.weight")) return whole(model.getDelta(layer, .attn_o_proj));
    if (std.mem.eql(u8, suffix, "mlp.down_proj.weight")) return whole(model.getDelta(layer, .mlp_down_proj));
    if (model.layers[layer].moe) |*m| {
        const target = m.exportTarget(suffix) orelse return null;
        return switch (target) {
            .expert => |e| whole(m.getDownDelta(e)),
            .fused_down => if (m.anyExpertDelta()) .{ .fused_down = layer } else null,
        };
    }
    return null;
}

fn whole(delta: ?tensor.Delta) ?Edit {
    return if (delta) |d| .{ .whole = d } else null;
}

/// Rows of `ref` processed per step: for a fused expert tensor one expert
/// slice (its delta is merged as a block), otherwise about
/// `convert_chunk_bytes` (less under a tight memory budget: the chunk, its
/// f32 copy and the output buffer must stay well inside the limit).
fn rowsPerChunk(model: *const Model, e: Entry) usize {
    if (e.edit) |edit| if (edit == .fused_down) return 1;
    var chunk_bytes: u64 = convert_chunk_bytes;
    if (model.budget) |b| {
        if (b.limited()) chunk_bytes = @min(chunk_bytes, b.limitBytes() / 16);
    }
    const row_bytes = @max(1, e.ref.cols * @max(e.ref.dtype.size(), e.out_dtype.size()));
    return @intCast(@max(1, @min(@as(u64, e.ref.rows), chunk_bytes / row_bytes)));
}

/// Peak bytes `writeTensor` holds for the largest tensor of `model` (the
/// resident chunk, its f32 copy and the output chunk), for the feasibility estimate.
pub fn peakBytes(model: *const Model, export_dtype: ?tensor.DType) u64 {
    var peak: u64 = 0;
    for (model.files, 0..) |f, fi| {
        var it = f.tensors.iterator();
        while (it.next()) |kv| {
            const info = kv.value_ptr.*;
            const out_dtype = export_dtype orelse info.dtype;
            const e = Entry{ .name = info.name, .info = info, .ref = model.store.refFor(fi, info), .out_dtype = out_dtype, .byte_len = info.numel() * out_dtype.size(), .edit = modifiedDelta(model, info.name) };
            const rows = rowsPerChunk(model, e);
            const elems: u64 = @as(u64, rows) * e.ref.cols;
            peak = @max(peak, elems * (e.ref.dtype.size() + 4 + out_dtype.size()));
        }
    }
    return peak;
}

/// Writes one tensor in row chunks: acquire the chunk from the store, merge
/// the edit (if any) into an f32 copy, convert, write, release. Peak memory
/// is one chunk regardless of the tensor's size.
fn writeTensor(gpa: Allocator, model: *const Model, out: *Io.Writer, e: Entry) !void {
    const store: *stream.WeightStore = @constCast(&model.store);
    const rows_per_chunk = rowsPerChunk(model, e);
    const cols = e.ref.cols;
    const es = e.out_dtype.size();
    const plain = e.edit == null and e.out_dtype == e.ref.dtype;
    const chunk: []u8 = if (plain) &.{} else try gpa.alloc(u8, rows_per_chunk * cols * es);
    defer if (chunk.len > 0) gpa.free(chunk);
    const f: []f32 = if (plain) &.{} else try gpa.alloc(f32, rows_per_chunk * cols);
    defer if (f.len > 0) gpa.free(f);
    var r: usize = 0;
    while (r < e.ref.rows) : (r += rows_per_chunk) {
        if (model.budget) |b| try b.checkTime();
        const n = @min(rows_per_chunk, e.ref.rows - r);
        const lease = try store.acquire(e.ref.rowSlice(r, n));
        defer store.release(lease);
        const w = lease.weight;
        if (plain) {
            try out.writeAll(w.data);
            continue;
        }
        const fc = f[0 .. n * cols];
        tensor.convertToF32(w.dtype, w.data, fc);
        if (e.edit) |edit| switch (edit) {
            .whole => |d| for (0..n) |i| {
                const row = fc[i * cols ..][0..cols];
                for (0..d.rank) |k| tensor.axpy(row, d.b[(r + i) * d.rank + k], d.a[k * cols ..][0..cols]);
            },
            .fused_down => |layer| model.layers[layer].moe.?.mergeExpertDown(r, fc),
        };
        tensor.convertFromF32(e.out_dtype, fc, chunk);
        try out.writeAll(chunk[0 .. n * cols * es]);
    }
}

fn writeShard(gpa: Allocator, io: Io, model: *const Model, dir: Io.Dir, name: []const u8, entries: []const Entry) !void {
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
    for (entries) |e| try writeTensor(gpa, model, out, e);
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
/// On any error the `.incomplete` marker is left behind (and `Model.load`
/// refuses the directory); the source model is never touched.
pub fn saveModel(gpa: Allocator, io: Io, model: *const Model, out_dir: []const u8, opts: SaveOptions, out: *Io.Writer) !void {
    const cwd = Io.Dir.cwd();
    try cwd.createDirPath(io, out_dir);
    var dir = try cwd.openDir(io, out_dir, .{});
    defer dir.close(io);
    try dir.writeFile(io, .{ .sub_path = model_mod.export_incomplete_marker, .data = "export in progress\n" });
    try saveModelInner(gpa, io, model, dir, opts, out);
    try dir.deleteFile(io, model_mod.export_incomplete_marker);
}

fn saveModelInner(gpa: Allocator, io: Io, model: *const Model, dir: Io.Dir, opts: SaveOptions, out: *Io.Writer) !void {
    const cwd = Io.Dir.cwd();
    var entries = std.ArrayList(Entry).empty;
    defer entries.deinit(gpa);
    var total: u64 = 0;
    for (model.files, 0..) |f, fi| {
        var it = f.tensors.iterator();
        while (it.next()) |kv| {
            const info = kv.value_ptr.*;
            const out_dtype = opts.export_dtype orelse info.dtype;
            const byte_len = info.numel() * out_dtype.size();
            try entries.append(gpa, .{ .name = info.name, .info = info, .ref = model.store.refFor(fi, info), .out_dtype = out_dtype, .byte_len = byte_len, .edit = modifiedDelta(model, info.name) });
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
        try writeShard(gpa, io, model, dir, "model.safetensors", shards.items[0]);
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
            try writeShard(gpa, io, model, dir, name, shard);
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
    // The marker is gone after a successful export.
    var od = try Io.Dir.cwd().openDir(io, out_dir, .{});
    defer od.close(io);
    try std.testing.expectError(error.FileNotFound, od.access(io, model_mod.export_incomplete_marker, .{}));
}
