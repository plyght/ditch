//! Exports a model with its abliteration deltas merged into the weights.
//!
//! Tensors are written one at a time through the model's weight store, so peak
//! memory is one tensor (plus its f32 copy when a delta is merged) and a
//! bounded conversion chunk, regardless of model size. A `.incomplete` marker
//! is created before the first shard and removed only after everything was
//! written, so a directory never looks complete when it is not.
//!
//! A checkpoint that was dequantised on load (FP8, MXFP4, pack-quantized INT4;
//! see dequant.zig) is exported as a plain bf16 checkpoint: the store hands
//! out the decoded tensors, so every one of them is written in bf16 (or the
//! requested `export_dtype`) and `config.json` loses its `quantization_config`.

const std = @import("std");
const Io = std.Io;
const tensor = @import("tensor.zig");
const safetensors = @import("safetensors.zig");
const model_mod = @import("model.zig");
const stream = @import("stream.zig");
const budget_mod = @import("budget.zig");

const Allocator = std.mem.Allocator;
const Model = model_mod.Model;

/// How a tensor is modified on export (decided by the model, which knows the family's tensor names).
const Edit = model_mod.ExportEdit;

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
    return model.exportEdit(name);
}

/// Storage dtype of a tensor in a Hugging Face export when none is requested:
/// its own for floating-point sources, f16 for quantised (GGUF) sources.
pub fn hfDtype(source: tensor.DType) tensor.DType {
    return if (source.isQuantized()) .f16 else source;
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
    const plain = e.edit == null and e.out_dtype == e.ref.dtype;
    const cols = e.ref.cols;
    // A plain copy holds one chunk; a conversion also holds its f32 copy and the output chunk.
    const row_bytes = @max(1, if (plain) e.ref.dtype.rowBytes(cols) else e.ref.dtype.rowBytes(cols) + 4 * cols + e.out_dtype.rowBytes(cols));
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
            const out_dtype = if (!info.dtype.isFloat() and !info.dtype.isQuantized()) info.dtype else export_dtype orelse hfDtype(info.dtype);
            const e = Entry{ .name = info.name, .info = info, .ref = model.store.refFor(fi, info), .out_dtype = out_dtype, .byte_len = info.numel() * out_dtype.size(), .edit = modifiedDelta(model, info.name) };
            const rows = rowsPerChunk(model, e);
            const cols = e.ref.cols;
            peak = @max(peak, @as(u64, rows) * (e.ref.dtype.rowBytes(cols) + 4 * cols + out_dtype.rowBytes(cols)));
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
            // Conv1D `[in][out]`: row `r + i` is input `r + i`, column `j` output `j`.
            .whole_transposed => |d| for (0..n) |i| {
                const row = fc[i * cols ..][0..cols];
                const in_cols = d.a.len / d.rank;
                for (row, 0..) |*v, j| {
                    var acc: f32 = 0;
                    for (0..d.rank) |k| acc += d.b[j * d.rank + k] * d.a[k * in_cols + r + i];
                    v.* += acc;
                }
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
    for (entries) |e| {
        if (budget_mod.interrupted()) return error.Interrupted;
        try writeTensor(gpa, model, out, e);
    }
    try out.flush();
}

fn copyIfExists(io: Io, src: Io.Dir, dst: Io.Dir, name: []const u8) void {
    src.copyFile(name, dst, name, io, .{}) catch {};
}

/// The files beside the weights that the export needs to load the way the
/// source did: tokenizer and processor files, a tiktoken vocabulary (with the
/// tokenizer.json synthesised from it), and every top-level `*.py`, which is
/// the code a `trust_remote_code` config's `auto_map` names (configuration,
/// modeling, tokenization) and the modules it imports. `src` must be opened
/// with `.iterate = true`.
fn copySideFiles(io: Io, src: Io.Dir, dst: Io.Dir, tiktoken: bool) void {
    copyIfExists(io, src, dst, "special_tokens_map.json");
    copyIfExists(io, src, dst, "chat_template.jinja");
    copyIfExists(io, src, dst, "chat_template.json");
    copyIfExists(io, src, dst, "added_tokens.json");
    copyIfExists(io, src, dst, "preprocessor_config.json");
    if (tiktoken) {
        copyIfExists(io, src, dst, "tiktoken.model");
        copyIfExists(io, src, dst, "tokenizer.model");
    }
    var it = src.iterate();
    while (it.next(io) catch null) |entry| {
        if (entry.kind == .file and std.mem.endsWith(u8, entry.name, ".py")) copyIfExists(io, src, dst, entry.name);
    }
}

test "saveModel carries the remote code and names the export dtype" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.createDirPath(io, "src");
    var fixture = try Io.Dir.cwd().openDir(io, "tests/fixtures/qwen2", .{});
    defer fixture.close(io);
    var src = try tmp.dir.openDir(io, "src", .{});
    defer src.close(io);
    for ([_][]const u8{ "config.json", "generation_config.json", "model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors", "model.safetensors.index.json", "tokenizer.json", "tokenizer_config.json" }) |name| {
        try fixture.copyFile(name, src, name, io, .{});
    }
    for ([_][]const u8{ "modeling_kimi_k3.py", "configuration_kimi_k3.py", "tokenization_kimi.py", "media_utils.py", "chat_template.jinja" }) |name| {
        try src.writeFile(io, .{ .sub_path = name, .data = name });
    }
    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const n = try tmp.dir.realPath(io, &path_buf);
    const src_dir = try std.fs.path.join(gpa, &.{ path_buf[0..n], "src" });
    defer gpa.free(src_dir);
    const out_dir = try std.fs.path.join(gpa, &.{ path_buf[0..n], "out" });
    defer gpa.free(out_dir);
    const pool = tensor.Pool.init(io, 1);
    const model = try Model.load(gpa, io, &pool, src_dir);
    defer model.deinit();
    var sink: Io.Writer.Allocating = .init(gpa);
    defer sink.deinit();
    try saveModel(gpa, io, model, out_dir, .{ .export_dtype = .f32 }, &sink.writer);
    var out = try Io.Dir.cwd().openDir(io, out_dir, .{});
    defer out.close(io);
    // The config names the dtype the weights were written in.
    const config = try out.readFileAlloc(io, "config.json", gpa, .limited(1 << 20));
    defer gpa.free(config);
    try std.testing.expect(std.mem.indexOf(u8, config, "\"torch_dtype\": \"float32\"") != null);
    for ([_][]const u8{ "modeling_kimi_k3.py", "configuration_kimi_k3.py", "tokenization_kimi.py", "media_utils.py", "chat_template.jinja" }) |name| {
        try out.access(io, name, .{});
    }
}

/// `config.json` with every floating-point `dtype` / `torch_dtype` (top level
/// and nested text / vision configs) naming `dtype`: transformers loads with
/// `dtype="auto"` from it, so a float32 export that still said bfloat16 would
/// be rounded back to bf16 on load.
fn withConfigDtype(gpa: Allocator, json: []const u8, dtype: tensor.DType) ![]u8 {
    const name: []const u8 = switch (dtype) {
        .f32 => "float32",
        .f16 => "float16",
        .bf16 => "bfloat16",
        else => return gpa.dupe(u8, json),
    };
    var out: Io.Writer.Allocating = .init(gpa);
    errdefer out.deinit();
    var i: usize = 0;
    while (i < json.len) {
        const at = std.mem.indexOfScalarPos(u8, json, i, '"') orelse break;
        const key_end = stringEnd(json, at) orelse break;
        const key = json[at + 1 .. key_end];
        var j = key_end + 1;
        while (j < json.len and std.ascii.isWhitespace(json[j])) : (j += 1) {}
        if (j < json.len and json[j] == ':' and (std.mem.eql(u8, key, "dtype") or std.mem.eql(u8, key, "torch_dtype"))) {
            j += 1;
            while (j < json.len and std.ascii.isWhitespace(json[j])) : (j += 1) {}
            if (j < json.len and json[j] == '"') {
                const v_end = stringEnd(json, j) orelse break;
                const value = json[j + 1 .. v_end];
                for ([_][]const u8{ "float32", "float16", "bfloat16" }) |f| {
                    if (std.mem.eql(u8, value, f)) {
                        try out.writer.writeAll(json[i .. j + 1]);
                        try out.writer.writeAll(name);
                        try out.writer.writeByte('"');
                        i = v_end + 1;
                        break;
                    }
                } else {
                    try out.writer.writeAll(json[i .. v_end + 1]);
                    i = v_end + 1;
                }
                continue;
            }
        }
        // Not a dtype key: copy through the string (a key or a value).
        try out.writer.writeAll(json[i .. key_end + 1]);
        i = key_end + 1;
    }
    try out.writer.writeAll(json[i..]);
    return out.toOwnedSlice();
}

/// The index of the quote closing the JSON string that opens at `start`.
fn stringEnd(json: []const u8, start: usize) ?usize {
    var k = start + 1;
    while (k < json.len) : (k += 1) switch (json[k]) {
        '\\' => k += 1,
        '"' => return k,
        else => {},
    };
    return null;
}

test "withConfigDtype names the export dtype" {
    const gpa = std.testing.allocator;
    const json =
        \\{"dtype": "bfloat16", "text_config": {"torch_dtype":"bfloat16", "model_type": "x"},
        \\ "quantization_config": {"dtype": "int8"}, "name": "dtype", "note": "a \"dtype\": \"bfloat16\" in a string"}
    ;
    const got = try withConfigDtype(gpa, json, .f32);
    defer gpa.free(got);
    try std.testing.expectEqualStrings(
        \\{"dtype": "float32", "text_config": {"torch_dtype":"float32", "model_type": "x"},
        \\ "quantization_config": {"dtype": "int8"}, "name": "dtype", "note": "a \"dtype\": \"bfloat16\" in a string"}
    , got);
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
            // Integer tables (DeepSeek V4's `tid2eid`) are never converted.
            const out_dtype = if (!info.dtype.isFloat() and !info.dtype.isQuantized()) info.dtype else opts.export_dtype orelse hfDtype(info.dtype);
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
    if (opts.export_dtype) |dt| {
        const config = try withConfigDtype(gpa, model.export_config_json, dt);
        defer gpa.free(config);
        try dir.writeFile(io, .{ .sub_path = "config.json", .data = config });
    } else try dir.writeFile(io, .{ .sub_path = "config.json", .data = model.export_config_json });
    try dir.writeFile(io, .{ .sub_path = "tokenizer.json", .data = model.tokenizer_json });
    if (model.generation_config_json) |g| try dir.writeFile(io, .{ .sub_path = "generation_config.json", .data = g });
    if (model.tokenizer_config_json) |t| try dir.writeFile(io, .{ .sub_path = "tokenizer_config.json", .data = t });
    var src = cwd.openDir(io, model.source_dir, .{ .iterate = true }) catch null;
    if (src) |*s| {
        defer s.close(io);
        copySideFiles(io, s.*, dir, model.tokenizer.tiktoken_kind != null);
    }
    if (opts.readme_body) |body| try dir.writeFile(io, .{ .sub_path = "README.md", .data = body });
}

test "export round trip merges deltas" {
    if (tensor.mmapDisabled()) return error.SkipZigTest; // this test is about the mapped path
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
