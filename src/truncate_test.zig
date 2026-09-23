//! Tests for `ditch truncate` (truncate.zig): the config.json edits of
//! tools/truncate_checkpoint.py on small configs of several families, the
//! layer-kind picker, and end-to-end cuts of local fixtures (and of one served
//! over HTTP by tools/range_server.py) that must load with ditch's model
//! loader: a cut keeping every layer gives the same logits as the source.

const std = @import("std");
const Io = std.Io;
const hf = @import("hf.zig");
const truncate = @import("truncate.zig");
const model_mod = @import("model.zig");
const tensor = @import("tensor.zig");

const Model = model_mod.Model;
const Value = std.json.Value;

/// `editConfigText` on `text`, parsed back.
fn edit(a: std.mem.Allocator, text: []const u8, layers: []const usize, drop: []const []const u8) !Value {
    const out = try truncate.editConfigText(a, text, layers, drop);
    return std.json.parseFromSliceLeaky(Value, a, out, .{});
}

fn expectInts(want: []const i64, v: ?Value) !void {
    const arr = (v orelse return error.TestExpectedEqual).array.items;
    try std.testing.expectEqual(want.len, arr.len);
    for (want, arr) |w, x| try std.testing.expectEqual(w, x.integer);
}

fn expectStrings(want: []const []const u8, v: ?Value) !void {
    const arr = (v orelse return error.TestExpectedEqual).array.items;
    try std.testing.expectEqual(want.len, arr.len);
    for (want, arr) |w, x| try std.testing.expectEqualStrings(w, x.string);
}

test "truncate config: per-layer lists, key order and the Python tool's formatting" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const got = try truncate.editConfigText(a,
        \\{"name": "café", "num_hidden_layers": 4, "layer_types": ["x", "y", "x", "y"], "intermediate_size": [8, 16, 24, 32], "mlp_only_layers": [], "rope_theta": 1e-06, "big": 12345678901234567890, "f": 10000.0}
    , &.{ 1, 3 }, &.{});
    try std.testing.expectEqualStrings(
        \\{
        \\ "name": "caf\u00e9",
        \\ "num_hidden_layers": 2,
        \\ "layer_types": [
        \\  "y",
        \\  "y"
        \\ ],
        \\ "intermediate_size": [
        \\  16,
        \\  32
        \\ ],
        \\ "mlp_only_layers": [],
        \\ "rope_theta": 1e-06,
        \\ "big": 12345678901234567890,
        \\ "f": 10000.0
        \\}
    , got);
}

test "truncate config: MiMo layer kinds written out from its patterns" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const v = try edit(a,
        \\{"model_type": "mimo_v2", "num_hidden_layers": 6, "hybrid_layer_pattern": [0, 1, 1, 1, 1, 0], "moe_layer_freq": [0, 1, 1, 1, 1, 1], "first_k_dense_replace": 1}
    , &.{ 0, 1, 5 }, &.{});
    const o = v.object;
    try std.testing.expectEqual(@as(i64, 3), o.get("num_hidden_layers").?.integer);
    try expectInts(&.{ 0, 1, 0 }, o.get("hybrid_layer_pattern"));
    try expectInts(&.{ 0, 1, 1 }, o.get("moe_layer_freq"));
    try expectStrings(&.{ "full_attention", "sliding_attention", "full_attention" }, o.get("layer_types"));
    try expectStrings(&.{ "dense", "sparse", "sparse" }, o.get("mlp_layer_types"));
    try std.testing.expectEqual(@as(i64, 1), o.get("first_k_dense_replace").?.integer);
    // The derived lists come last, as Python appends them.
    const keys = o.keys();
    try std.testing.expectEqualStrings("layer_types", keys[keys.len - 2]);
    try std.testing.expectEqualStrings("mlp_layer_types", keys[keys.len - 1]);
    // An explicit layer_types is cut, not derived again.
    const w = try edit(a,
        \\{"num_hidden_layers": 3, "hybrid_layer_pattern": [0, 1, 1], "layer_types": ["full_attention", "sliding_attention", "sliding_attention"]}
    , &.{2}, &.{});
    try expectStrings(&.{"sliding_attention"}, w.object.get("layer_types"));
    try std.testing.expect(w.object.get("mlp_layer_types") == null);
}

test "truncate config: linear_attn_config ids, 1-based (Kimi) and 0-based (GLM)" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    // Kimi-Linear: 1-based ids, layer 4 (id 4) the full-attention layer.
    const kimi = try edit(a,
        \\{"num_hidden_layers": 4, "first_k_dense_replace": 1, "linear_attn_config": {"kda_layers": [1, 2, 3], "full_attn_layers": [4], "head_dim": 4}}
    , &.{ 0, 3 }, &.{});
    const lk = kimi.object.get("linear_attn_config").?.object;
    try expectInts(&.{1}, lk.get("kda_layers"));
    try expectInts(&.{2}, lk.get("full_attn_layers"));
    try std.testing.expectEqual(@as(i64, 4), lk.get("head_dim").?.integer);
    try std.testing.expectEqual(@as(i64, 1), kimi.object.get("first_k_dense_replace").?.integer);
    // GLM-5.3-Flash: 0-based (the lists contain a 0).
    const glm = try edit(a,
        \\{"num_hidden_layers": 4, "first_k_dense_replace": 3, "linear_attn_config": {"kda_layers": [0, 1, 2], "full_attn_layers": [3]}}
    , &.{ 1, 3 }, &.{});
    const lg = glm.object.get("linear_attn_config").?.object;
    try expectInts(&.{0}, lg.get("kda_layers"));
    try expectInts(&.{1}, lg.get("full_attn_layers"));
    try std.testing.expectEqual(@as(i64, 1), glm.object.get("first_k_dense_replace").?.integer);
}

test "truncate config: DeepSeek V4.1 id lists, engram pairing, candidate source and MTP" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const text =
        \\{"model_type": "deepseek_v41", "text_config": {"num_hidden_layers": 4, "compress_ratios": [0, 2, 1, 1, 0, 0, 0],
        \\ "kv_source_layer_ids": [1, 2], "index_source_layer_ids": [1, 2], "candidate_source_layer_id": 2,
        \\ "engram_layer_ids": [1, 3], "engram_num_embeddings": [156, 200], "num_nextn_predict_layers": 3},
        \\ "vision_config": {"num_hidden_layers": 1}, "num_nextn_predict_layers": 5}
    ;
    const v = try edit(a, text, &.{ 0, 2, 3 }, &.{"mtp."});
    const tc = v.object.get("text_config").?.object;
    try std.testing.expectEqual(@as(i64, 3), tc.get("num_hidden_layers").?.integer);
    try expectInts(&.{ 0, 1, 1 }, tc.get("compress_ratios"));
    try expectInts(&.{1}, tc.get("kv_source_layer_ids"));
    try expectInts(&.{1}, tc.get("index_source_layer_ids"));
    try std.testing.expectEqual(@as(i64, 1), tc.get("candidate_source_layer_id").?.integer);
    try expectInts(&.{2}, tc.get("engram_layer_ids"));
    try expectInts(&.{200}, tc.get("engram_num_embeddings"));
    try std.testing.expectEqual(@as(i64, 0), tc.get("num_nextn_predict_layers").?.integer);
    // Only the text config is edited.
    try std.testing.expectEqual(@as(i64, 1), v.object.get("vision_config").?.object.get("num_hidden_layers").?.integer);
    try std.testing.expectEqual(@as(i64, 5), v.object.get("num_nextn_predict_layers").?.integer);
    // A candidate source that is cut away becomes -1; no MTP drop, no change.
    const w = try edit(a, text, &.{ 0, 1 }, &.{});
    const wc = w.object.get("text_config").?.object;
    try std.testing.expectEqual(@as(i64, -1), wc.get("candidate_source_layer_id").?.integer);
    try expectInts(&.{1}, wc.get("engram_layer_ids"));
    try expectInts(&.{156}, wc.get("engram_num_embeddings"));
    try std.testing.expectEqual(@as(i64, 3), wc.get("num_nextn_predict_layers").?.integer);
}

test "truncate config: Gemma 4 KV-shared count, Nemotron-H pattern, MiniMax M3 lists" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const g = try edit(a,
        \\{"model_type": "gemma4", "text_config": {"num_hidden_layers": 5, "num_kv_shared_layers": 2, "hidden_size_per_layer_input": 4,
        \\ "layer_types": ["sliding_attention", "full_attention", "sliding_attention", "sliding_attention", "full_attention"]}}
    , &.{ 0, 1, 4 }, &.{});
    const gt = g.object.get("text_config").?.object;
    try std.testing.expectEqual(@as(i64, 1), gt.get("num_kv_shared_layers").?.integer);
    try expectStrings(&.{ "sliding_attention", "full_attention", "full_attention" }, gt.get("layer_types"));

    const n = try edit(a,
        \\{"model_type": "nemotron_h", "num_hidden_layers": 6, "hybrid_override_pattern": "M*M-ME"}
    , &.{ 1, 3, 5 }, &.{});
    try std.testing.expectEqualStrings("*-E", n.object.get("hybrid_override_pattern").?.string);

    const m = try edit(a,
        \\{"num_hidden_layers": 3, "sparse_attention_config": {"use_sparse": [true, false, true], "pair": [1, 2], "block": 64}}
    , &.{2}, &.{});
    const sac = m.object.get("sparse_attention_config").?.object;
    try std.testing.expectEqual(@as(usize, 1), sac.get("use_sparse").?.array.items.len);
    try std.testing.expect(sac.get("use_sparse").?.array.items[0].bool);
    try expectInts(&.{ 1, 2 }, sac.get("pair"));
    try std.testing.expectEqual(@as(i64, 64), sac.get("block").?.integer);

    // Llama 4: no_rope_layers cut, moe_layers renumbered.
    const l = try edit(a,
        \\{"num_hidden_layers": 4, "no_rope_layers": [1, 1, 1, 0], "moe_layers": [1, 3]}
    , &.{ 2, 3 }, &.{});
    try expectInts(&.{ 1, 0 }, l.object.get("no_rope_layers"));
    try expectInts(&.{1}, l.object.get("moe_layers"));
}

fn kinds(a: std.mem.Allocator, text: []const u8) ![]usize {
    const root = try truncate.parseConfig(a, text);
    return truncate.kindLayers(a, root);
}

test "truncate kinds: the first layer of every kind, plus the layers they read" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    // A plain model: layer 0 alone.
    try std.testing.expectEqualSlices(usize, &.{0}, try kinds(a,
        \\{"num_hidden_layers": 24}
    ));
    // DeepSeek V4.1: ratio 0 (with and without engram), the ratio-2 source and
    // a layer reading it, the ratio-1 source and a reader; the candidate source.
    try std.testing.expectEqualSlices(usize, &.{ 0, 1, 2, 3, 20, 21 }, try kinds(a,
        \\{"text_config": {"num_hidden_layers": 24, "compress_ratios": [0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1],
        \\ "kv_source_layer_ids": [2, 20], "index_source_layer_ids": [2, 20], "candidate_source_layer_id": 20, "engram_layer_ids": [1]}}
    ));
    // Kimi-Linear (1-based ids): KDA dense, KDA MoE, full MoE.
    try std.testing.expectEqualSlices(usize, &.{ 0, 1, 3 }, try kinds(a,
        \\{"num_hidden_layers": 6, "first_k_dense_replace": 1, "linear_attn_config": {"kda_layers": [1, 2, 3, 5, 6], "full_attn_layers": [4]}}
    ));
    // Nemotron-H: one layer per pattern character.
    try std.testing.expectEqualSlices(usize, &.{ 0, 1, 3, 5 }, try kinds(a,
        \\{"num_hidden_layers": 6, "hybrid_override_pattern": "M*M-ME"}
    ));
    // MiMo: full dense, sliding MoE, full MoE.
    try std.testing.expectEqualSlices(usize, &.{ 0, 1, 5 }, try kinds(a,
        \\{"num_hidden_layers": 7, "hybrid_layer_pattern": [0, 1, 1, 1, 1, 0, 1], "moe_layer_freq": [0, 1, 1, 1, 1, 1, 1]}
    ));
    // Gemma 4: a KV-shared sliding layer reads the last non-shared sliding
    // layer (3), which is kept although its kind is already covered by 0.
    try std.testing.expectEqualSlices(usize, &.{ 0, 2, 3, 4, 5 }, try kinds(a,
        \\{"text_config": {"num_hidden_layers": 6, "num_kv_shared_layers": 2, "layer_types":
        \\ ["sliding_attention", "sliding_attention", "full_attention", "sliding_attention", "full_attention", "sliding_attention"]}}
    ));
    // A kind given by the layer index (Qwen3-Next without layer_types) only
    // survives a prefix: layers 0..3 for the first full-attention layer 3.
    try std.testing.expectEqualSlices(usize, &.{ 0, 1, 2, 3 }, try kinds(a,
        \\{"num_hidden_layers": 48, "full_attention_interval": 4}
    ));
    // With layer_types the interval is not used, and the cut can skip.
    try std.testing.expectEqualSlices(usize, &.{ 0, 3 }, try kinds(a,
        \\{"num_hidden_layers": 8, "full_attention_interval": 4, "layer_types": ["linear_attention", "linear_attention", "linear_attention", "full_attention",
        \\ "linear_attention", "linear_attention", "linear_attention", "full_attention"]}
    ));
    // The candidate source is kept even when its kind is covered.
    try std.testing.expectEqualSlices(usize, &.{ 0, 3 }, try kinds(a,
        \\{"num_hidden_layers": 4, "candidate_source_layer_id": 3}
    ));
}

test "truncate names: layer tensors in both spellings, drops" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const layers = [_]usize{ 20, 3 };
    try std.testing.expectEqualStrings("model.layers.0.mlp.w", (try truncate.cutName(a, "model.layers.20.mlp.w", &layers, &.{})).?);
    try std.testing.expectEqualStrings("layers.1.attn.wq", (try truncate.cutName(a, "layers.3.attn.wq", &layers, &.{})).?);
    try std.testing.expect(try truncate.cutName(a, "model.layers.2.mlp.w", &layers, &.{}) == null);
    try std.testing.expectEqualStrings("model.norm.weight", (try truncate.cutName(a, "model.norm.weight", &layers, &.{})).?);
    // `sublayers.3.` is not a layer; the first real `layers.N.` counts.
    try std.testing.expectEqualStrings("x.sublayers.3.y", (try truncate.cutName(a, "x.sublayers.3.y", &layers, &.{})).?);
    try std.testing.expectEqualStrings("a.layers.x.layers.0.b", (try truncate.cutName(a, "a.layers.x.layers.20.b", &layers, &.{})).?);
    try std.testing.expect(try truncate.cutName(a, "mtp.0.norm", &layers, &.{"mtp."}) == null);
    try std.testing.expect(truncate.isSmallFile("tokenizer.json") and truncate.isSmallFile("tiktoken.model") and truncate.isSmallFile("merges.txt"));
    try std.testing.expect(!truncate.isSmallFile("model.safetensors.index.json") and !truncate.isSmallFile(".gitattributes") and !truncate.isSmallFile("original/params.json") and !truncate.isSmallFile("model.safetensors"));
}

// ---------------------------------------------------------------------------
// End to end
// ---------------------------------------------------------------------------

fn firstTokenLogits(gpa: std.mem.Allocator, model: *Model, ids: []const u32) ![]f32 {
    const c = &model.config;
    var ws = try model_mod.Workspace.init(model.gpa, c, 16, 1);
    defer ws.deinit();
    var cache = try model_mod.KvCache.initFor(model, model.gpa, 1, 16);
    defer cache.deinit();
    const logits = try gpa.alloc(f32, c.vocab_size);
    errdefer gpa.free(logits);
    try model_mod.prefill(model, &ws, &cache, &.{ids}, logits, null);
    return logits;
}

/// The header of `dir/model.safetensors` (in `a`) and the offset of its data.
fn readHeader(a: std.mem.Allocator, io: Io, dir: []const u8) !struct { header: std.json.ObjectMap, base: u64, bytes: []u8 } {
    const path = try std.fs.path.join(a, &.{ dir, "model.safetensors" });
    const bytes = try Io.Dir.cwd().readFileAlloc(io, path, a, .unlimited);
    const n = std.mem.readInt(u64, bytes[0..8], .little);
    try std.testing.expect((8 + n) % 8 == 0);
    const v = try std.json.parseFromSliceLeaky(Value, a, bytes[8..][0..n], .{});
    return .{ .header = v.object, .base = 8 + n, .bytes = bytes };
}

fn tensorBytes(h: anytype, name: []const u8) ![]const u8 {
    const t = h.header.get(name) orelse return error.TestUnexpectedResult;
    const offs = t.object.get("data_offsets").?.array.items;
    const s: usize = @intCast(offs[0].integer);
    const e: usize = @intCast(offs[1].integer);
    return h.bytes[@intCast(h.base + s)..@intCast(h.base + e)];
}

fn tmpPath(a: std.mem.Allocator, io: Io, tmp: *std.testing.TmpDir, sub: []const u8) ![]const u8 {
    var buf: [std.fs.max_path_bytes]u8 = undefined;
    const n = try tmp.dir.realPath(io, &buf);
    return std.fs.path.join(a, &.{ buf[0..n], sub });
}

test "truncate end to end: sharded qwen2, every layer kept gives the same logits" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    const pool = tensor.Pool.init(io, 2);
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const ids = [_]u32{ 40, 100, 7 };

    const source = try Model.load(gpa, io, &pool, "tests/fixtures/qwen2");
    defer source.deinit();
    const want = try firstTokenLogits(gpa, source, &ids);
    defer gpa.free(want);

    // Every layer, from two shards into one file: the same model.
    const full = try tmpPath(a, io, &tmp, "full");
    var r = try truncate.truncate(gpa, io, null, .{ .model = "tests/fixtures/qwen2", .out_dir = full, .layers = .{ .first = 3 }, .request_size = 1 << 16, .connections = 3 });
    defer r.deinit(gpa);
    try std.testing.expectEqual(@as(usize, 3), r.source_layers);
    try std.testing.expectEqual(r.logical_bytes, r.fetched_bytes);
    {
        const m = try Model.load(gpa, io, &pool, full);
        defer m.deinit();
        try std.testing.expectEqual(@as(usize, 3), m.config.num_layers);
        const got = try firstTokenLogits(gpa, m, &ids);
        defer gpa.free(got);
        try std.testing.expectEqualSlices(f32, want, got);
    }
    // No index is written, the small files are copied.
    var fdir = try Io.Dir.cwd().openDir(io, full, .{});
    defer fdir.close(io);
    try std.testing.expectError(error.FileNotFound, fdir.access(io, "model.safetensors.index.json", .{}));
    try fdir.access(io, "tokenizer.json", .{});
    try fdir.access(io, "generation_config.json", .{});

    // Layers 2 and 0, in that order: renamed 0 and 1, the bytes of the source's layers.
    const cut = try tmpPath(a, io, &tmp, "cut");
    var r2 = try truncate.truncate(gpa, io, null, .{ .model = "tests/fixtures/qwen2", .out_dir = cut, .layers = .{ .list = &.{ 2, 0 } } });
    defer r2.deinit(gpa);
    try std.testing.expectEqualSlices(usize, &.{ 2, 0 }, r2.layers);
    {
        const m = try Model.load(gpa, io, &pool, cut);
        defer m.deinit();
        try std.testing.expectEqual(@as(usize, 2), m.config.num_layers);
        const logits = try firstTokenLogits(gpa, m, &ids);
        defer gpa.free(logits);
        for (logits) |x| try std.testing.expect(std.math.isFinite(x));
    }
    const hf_full = try readHeader(a, io, full);
    const hc = try readHeader(a, io, cut);
    try std.testing.expect(hc.header.get("model.layers.2.mlp.up_proj.weight") == null);
    try std.testing.expectEqualSlices(u8, try tensorBytes(hf_full, "model.layers.2.mlp.up_proj.weight"), try tensorBytes(hc, "model.layers.0.mlp.up_proj.weight"));
    try std.testing.expectEqualSlices(u8, try tensorBytes(hf_full, "model.layers.0.self_attn.k_proj.bias"), try tensorBytes(hc, "model.layers.1.self_attn.k_proj.bias"));
    try std.testing.expectEqualSlices(u8, try tensorBytes(hf_full, "model.embed_tokens.weight"), try tensorBytes(hc, "model.embed_tokens.weight"));
    const cfg = try std.json.parseFromSliceLeaky(Value, a, try Io.Dir.cwd().readFileAlloc(io, try std.fs.path.join(a, &.{ cut, "config.json" }), a, .unlimited), .{});
    try std.testing.expectEqual(@as(i64, 2), cfg.object.get("num_hidden_layers").?.integer);
    try std.testing.expectEqual(@as(i64, 32), cfg.object.get("hidden_size").?.integer);

    // --rows: the table keeps its shape; only the listed rows hold data.
    const rows = try tmpPath(a, io, &tmp, "rows");
    var r3 = try truncate.truncate(gpa, io, null, .{
        .model = "tests/fixtures/qwen2",
        .out_dir = rows,
        .layers = .{ .first = 1 },
        .rows = &.{.{ .name = "model.embed_tokens.weight", .rows = &.{ 7, 5, 6, 100 } }},
    });
    defer r3.deinit(gpa);
    try std.testing.expect(r3.fetched_bytes < r3.logical_bytes);
    const hr = try readHeader(a, io, rows);
    const table = try tensorBytes(hr, "model.embed_tokens.weight");
    const orig = try tensorBytes(hf_full, "model.embed_tokens.weight");
    try std.testing.expectEqual(orig.len, table.len);
    const row = orig.len / 271;
    for (0..271) |i| {
        const got = table[i * row ..][0..row];
        if (i == 5 or i == 6 or i == 7 or i == 100) {
            try std.testing.expectEqualSlices(u8, orig[i * row ..][0..row], got);
        } else {
            for (got) |b| try std.testing.expectEqual(@as(u8, 0), b);
        }
    }
}

test "truncate end to end: hybrid and MoE families, per-layer tables, drops" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    const pool = tensor.Pool.init(io, 2);
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();

    // MiMo V2 (hybrid sliding / full, dense / MoE), by kinds: 0 full dense,
    // 1 sliding MoE, 3 full MoE.
    {
        const out = try tmpPath(a, io, &tmp, "mimo");
        var r = try truncate.truncate(gpa, io, null, .{ .model = "tests/fixtures/mimo_v2", .out_dir = out, .layers = .kinds });
        defer r.deinit(gpa);
        try std.testing.expectEqualSlices(usize, &.{ 0, 1, 3 }, r.layers);
        const m = try Model.load(gpa, io, &pool, out);
        defer m.deinit();
        try std.testing.expectEqual(@as(usize, 3), m.config.num_layers);
        const h = try readHeader(a, io, out);
        try std.testing.expect(h.header.get("model.layers.2.mlp.gate.weight") != null);
        try std.testing.expect(h.header.get("model.layers.3.input_layernorm.weight") == null);
    }
    // Gemma 4: the per-layer embedding table and its projection cut to the kept layers' blocks.
    {
        const out = try tmpPath(a, io, &tmp, "gemma4");
        var r = try truncate.truncate(gpa, io, null, .{ .model = "tests/fixtures/gemma4", .out_dir = out, .layers = .{ .list = &.{ 0, 1, 4 } } });
        defer r.deinit(gpa);
        const m = try Model.load(gpa, io, &pool, out);
        defer m.deinit();
        try std.testing.expectEqual(@as(usize, 3), m.config.num_layers);
        const src = try readHeader(a, io, "tests/fixtures/gemma4");
        const h = try readHeader(a, io, out);
        var it = h.header.iterator();
        var name_table: ?[]const u8 = null;
        var name_proj: ?[]const u8 = null;
        while (it.next()) |kv| {
            if (std.mem.endsWith(u8, kv.key_ptr.*, "embed_tokens_per_layer.weight")) name_table = kv.key_ptr.*;
            if (std.mem.endsWith(u8, kv.key_ptr.*, "per_layer_model_projection.weight")) name_proj = kv.key_ptr.*;
        }
        const shape = h.header.get(name_table.?).?.object.get("shape").?.array.items;
        try std.testing.expectEqual(@as(i64, 3 * 4), shape[1].integer);
        try std.testing.expectEqual(@as(i64, 3 * 4), h.header.get(name_proj.?).?.object.get("shape").?.array.items[0].integer);
        // Row 5 of the table: the columns of layers 0, 1 and 4.
        const st = try tensorBytes(src, name_table.?);
        const ct = try tensorBytes(h, name_table.?);
        const elem = st.len / (374 * 5 * 4);
        const seg = 4 * elem;
        const r5 = st[5 * 5 * seg ..][0 .. 5 * seg];
        const c5 = ct[5 * 3 * seg ..][0 .. 3 * seg];
        try std.testing.expectEqualSlices(u8, r5[0 .. 2 * seg], c5[0 .. 2 * seg]);
        try std.testing.expectEqualSlices(u8, r5[4 * seg ..][0..seg], c5[2 * seg ..][0..seg]);
        const sp = try tensorBytes(src, name_proj.?);
        const cp = try tensorBytes(h, name_proj.?);
        const blk = sp.len / 5;
        try std.testing.expectEqualSlices(u8, sp[4 * blk ..][0..blk], cp[2 * blk ..][0..blk]);
    }
    // DeepSeek V4.1 with the MTP layers dropped: ids renumbered, loads.
    {
        const out = try tmpPath(a, io, &tmp, "dsv41");
        var r = try truncate.truncate(gpa, io, null, .{ .model = "tests/fixtures/deepseek_v41", .out_dir = out, .layers = .{ .list = &.{ 0, 2, 3 } }, .drop = &.{"mtp."} });
        defer r.deinit(gpa);
        const m = try Model.load(gpa, io, &pool, out);
        defer m.deinit();
        try std.testing.expectEqual(@as(usize, 3), m.config.num_layers);
        const h = try readHeader(a, io, out);
        var it = h.header.iterator();
        while (it.next()) |kv| try std.testing.expect(!std.mem.startsWith(u8, kv.key_ptr.*, "mtp."));
    }
}

// ---------------------------------------------------------------------------
// Remote
// ---------------------------------------------------------------------------

const Server = struct {
    child: std.process.Child,
    port: u16,

    fn start(io: Io, dir: []const u8) !Server {
        var child = try std.process.spawn(io, .{
            .argv = &.{ "python3", "tools/range_server.py", dir },
            .stdin = .ignore,
            .stdout = .pipe,
            .stderr = .inherit,
        });
        errdefer child.kill(io);
        var buf: [256]u8 = undefined;
        var fr = child.stdout.?.readerStreaming(io, &buf);
        const line = try fr.interface.takeDelimiterExclusive('\n');
        if (!std.mem.startsWith(u8, line, "PORT ")) return error.ServerDidNotStart;
        const port = try std.fmt.parseInt(u16, std.mem.trim(u8, line["PORT ".len..], " \r"), 10);
        return .{ .child = child, .port = port };
    }

    fn stop(self: *Server, io: Io) void {
        self.child.kill(io);
    }
};

test "truncate end to end: over HTTP range requests, the same bytes as from disk" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var environ = std.process.Environ.Map.init(gpa);
    defer environ.deinit();
    var http = try hf.Http.init(gpa, io, a, &environ);
    defer http.deinit();

    for ([_][]const u8{ "tests/fixtures/qwen3_moe_big", "tests/fixtures/gemma4" }) |fixture| {
        var server = try Server.start(io, fixture);
        defer server.stop(io);
        const base_url = try std.fmt.allocPrint(a, "http://127.0.0.1:{d}/", .{server.port});
        const local_out = try tmpPath(a, io, &tmp, try std.fmt.allocPrint(a, "local-{s}", .{std.fs.path.basename(fixture)}));
        const remote_out = try tmpPath(a, io, &tmp, try std.fmt.allocPrint(a, "remote-{s}", .{std.fs.path.basename(fixture)}));
        var progress: Io.Writer.Allocating = .init(gpa);
        defer progress.deinit();
        var rl = try truncate.truncate(gpa, io, null, .{ .model = fixture, .out_dir = local_out, .layers = .{ .list = &.{ 1, 3 } } });
        defer rl.deinit(gpa);
        var rr = try truncate.truncate(gpa, io, &http, .{ .model = base_url, .out_dir = remote_out, .layers = .{ .list = &.{ 1, 3 } }, .request_size = 1 << 16, .connections = 4, .progress = &progress.writer });
        defer rr.deinit(gpa);
        try std.testing.expect(std.mem.indexOf(u8, progress.written(), " GB of ") != null);
        try std.testing.expectEqual(rl.tensors, rr.tensors);
        try std.testing.expectEqual(rl.logical_bytes, rr.logical_bytes);
        try std.testing.expectEqual(rl.fetched_bytes, rr.fetched_bytes);
        for ([_][]const u8{ "model.safetensors", "config.json", "tokenizer.json", "tokenizer_config.json" }) |name| {
            const x = try Io.Dir.cwd().readFileAlloc(io, try std.fs.path.join(a, &.{ local_out, name }), a, .unlimited);
            const y = try Io.Dir.cwd().readFileAlloc(io, try std.fs.path.join(a, &.{ remote_out, name }), a, .unlimited);
            try std.testing.expectEqualSlices(u8, x, y);
        }
        // Small files the server does not have leave nothing behind.
        var dir = try Io.Dir.cwd().openDir(io, remote_out, .{ .iterate = true });
        defer dir.close(io);
        var it = dir.iterate();
        while (try it.next(io)) |e| try std.testing.expect(!std.mem.endsWith(u8, e.name, ".part"));
    }
}
