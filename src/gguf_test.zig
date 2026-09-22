//! GGUF round trips: the NumPy-written GGUF fixtures (mapped and streamed),
//! Hugging Face fixtures exported to GGUF and reloaded, the tokenizer rebuilt
//! from the ggml vocabulary, and GGUF re-exported to Hugging Face format.
const std = @import("std");
const Io = std.Io;
const model_mod = @import("model.zig");
const model_test = @import("model_test.zig");
const tensor = @import("tensor.zig");
const gguf = @import("gguf.zig");
const gguf_export = @import("gguf_export.zig");
const export_mod = @import("export.zig");

const Model = model_mod.Model;

const Case = struct { text: []const u8, ids: []u32, last_logits: []f32 };
const Reference = struct { cases: []Case };

test "qwen2 GGUF fixture (Q8_0 feed-forward, qwen2 vocabulary)" {
    try model_test.checkFixture("qwen2_gguf");
}

test "llama GGUF fixture (permuted q/k, rope_freqs, llama-bpe vocabulary)" {
    try model_test.checkFixture("llama_gguf");
}

/// Last-position logits `[texts][vocab]` of `model` for `texts`; also returns the ids.
fn lastLogits(gpa: std.mem.Allocator, model: *Model, texts: []const []const u8, ids_out: ?[][]u32) ![]f32 {
    const c = &model.config;
    var ws = try model_mod.Workspace.init(gpa, c, 64, 8);
    defer ws.deinit();
    var cache = try model_mod.KvCache.init(gpa, c.num_layers, texts.len, 64, c.kvDim());
    defer cache.deinit();
    var prompts = std.ArrayList([]const u32).empty;
    defer {
        for (prompts.items) |p| gpa.free(p);
        prompts.deinit(gpa);
    }
    for (texts, 0..) |t, i| {
        const ids = try model.tokenizer.encode(gpa, t, true);
        if (ids_out) |o| o[i] = try gpa.dupe(u32, ids);
        try prompts.append(gpa, ids);
    }
    const logits = try gpa.alloc(f32, texts.len * c.vocab_size);
    errdefer gpa.free(logits);
    try model_mod.prefill(model, &ws, &cache, prompts.items, logits, null);
    return logits;
}

fn expectClose(a: []const f32, b: []const f32, n: usize, vocab: usize, rel_tol: f32, label: []const u8) !void {
    for (0..n) |i| {
        const x = a[i * vocab ..][0..vocab];
        const y = b[i * vocab ..][0..vocab];
        var max_abs: f32 = 0;
        var max_err: f32 = 0;
        for (x, 0..) |v, j| {
            max_abs = @max(max_abs, @abs(v));
            max_err = @max(max_err, @abs(v - y[j]));
        }
        if (max_err > rel_tol * max_abs) {
            std.debug.print("{s}: prompt {d}: max err {d} vs max |logit| {d}\n", .{ label, i, max_err, max_abs });
            return error.LogitsMismatch;
        }
    }
}

/// The streamed store reads the GGUF file positionally (permuted rows through the overlay).
fn checkStreamed(comptime fixture: []const u8) !void {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    const dir = "tests/fixtures/" ++ fixture;
    const pool = tensor.Pool.init(io, 2);
    const model = try Model.loadWithOptions(gpa, io, &pool, dir, .{ .store = .streamed });
    defer model.deinit();
    try std.testing.expect(model.streamed());
    const ref_text = try Io.Dir.cwd().readFileAlloc(io, dir ++ "/reference.json", gpa, .unlimited);
    defer gpa.free(ref_text);
    const ref = try std.json.parseFromSlice(Reference, gpa, ref_text, .{ .ignore_unknown_fields = true });
    defer ref.deinit();
    const texts = try gpa.alloc([]const u8, ref.value.cases.len);
    defer gpa.free(texts);
    for (ref.value.cases, 0..) |case, i| texts[i] = case.text;
    const logits = try lastLogits(gpa, model, texts, null);
    defer gpa.free(logits);
    const vocab = model.config.vocab_size;
    for (ref.value.cases, 0..) |case, i| {
        var max_err: f32 = 0;
        for (case.last_logits, 0..) |e, j| max_err = @max(max_err, @abs(e - logits[i * vocab + j]));
        try std.testing.expect(max_err <= 2e-3);
    }
}

test "GGUF fixtures load through the streamed store" {
    try checkStreamed("qwen2_gguf");
    try checkStreamed("llama_gguf");
}

const sample_texts = [_][]const u8{ "the ant or you", "an era in the", "hello" };

/// Puts a rank-1 delta on layer 0's o_proj and on a down projection (layer 1,
/// or expert 0 of layer 0 for MoE models).
fn addDeltas(gpa: std.mem.Allocator, model: *Model) !void {
    const w = model.componentWeight(0, .attn_o_proj);
    const a = try gpa.alloc(f32, w.cols);
    @memset(a, 0);
    a[0] = 0.5;
    a[1] = -0.25;
    const b = try gpa.alloc(f32, w.rows);
    for (b, 0..) |*v, i| v.* = if (i % 2 == 0) 0.3 else -0.2;
    model.setDelta(0, .attn_o_proj, .{ .rank = 1, .a = a, .b = b });
    if (model.layers[0].moe != null) {
        const d = model.expertDownWeight(0, 0);
        const a2 = try gpa.alloc(f32, d.cols);
        for (a2, 0..) |*v, i| v.* = if (i % 3 == 0) 0.4 else 0;
        const b2 = try gpa.alloc(f32, d.rows);
        @memset(b2, 0.25);
        model.setExpertDelta(0, 0, .{ .rank = 1, .a = a2, .b = b2 });
    } else {
        const d = model.componentWeight(1, .mlp_down_proj);
        const a2 = try gpa.alloc(f32, d.cols);
        for (a2, 0..) |*v, i| v.* = if (i % 3 == 0) 0.4 else 0;
        const b2 = try gpa.alloc(f32, d.rows);
        @memset(b2, 0.25);
        model.setDelta(1, .mlp_down_proj, .{ .rank = 1, .a = a2, .b = b2 });
    }
}

fn tmpPath(gpa: std.mem.Allocator, io: Io, tmp: *std.testing.TmpDir, name: []const u8) ![]const u8 {
    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const n = try tmp.dir.realPath(io, &path_buf);
    return std.fs.path.join(gpa, &.{ path_buf[0..n], name });
}

/// HF fixture (+ deltas) -> GGUF -> reload (optionally ignoring the embedded
/// HF files so config and tokenizer come from the ggml metadata) -> the same
/// ids and logits within `tol` of the largest logit -> HF re-export -> same again.
fn roundTrip(comptime fixture: []const u8, dtype: ?tensor.DType, tol: f32, ignore_embedded: bool) !void {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    const pool = tensor.Pool.init(io, 2);
    const model = try Model.load(gpa, io, &pool, "tests/fixtures/" ++ fixture);
    defer model.deinit();
    try addDeltas(gpa, model);
    var ids_a: [sample_texts.len][]u32 = undefined;
    const logits_a = try lastLogits(gpa, model, &sample_texts, &ids_a);
    defer gpa.free(logits_a);
    defer for (ids_a) |x| gpa.free(x);

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const out_dir = try tmpPath(gpa, io, &tmp, "gguf");
    defer gpa.free(out_dir);
    var sink: Io.Writer.Allocating = .init(gpa);
    defer sink.deinit();
    try gguf_export.saveGguf(gpa, io, model, out_dir, .{ .dtype = dtype, .name = fixture, .readme_body = "# test\n" }, &sink.writer);

    const reloaded = try Model.loadWithOptions(gpa, io, &pool, out_dir, .{ .gguf_ignore_embedded = ignore_embedded });
    defer reloaded.deinit();
    try std.testing.expect(reloaded.gguf != null);
    try std.testing.expectEqual(!ignore_embedded, reloaded.gguf.?.embedded_tokenizer);
    try std.testing.expectEqual(model.config.num_layers, reloaded.config.num_layers);
    try std.testing.expectEqual(model.config.head_dim, reloaded.config.head_dim);
    try std.testing.expectEqual(model.config.num_experts, reloaded.config.num_experts);
    try std.testing.expectEqual(model.eos_ids.len, reloaded.eos_ids.len);
    try std.testing.expectEqualStrings(model.chat_template.?, reloaded.chat_template.?);
    var ids_b: [sample_texts.len][]u32 = undefined;
    const logits_b = try lastLogits(gpa, reloaded, &sample_texts, &ids_b);
    defer gpa.free(logits_b);
    defer for (ids_b) |x| gpa.free(x);
    for (ids_a, ids_b) |x, y| try std.testing.expectEqualSlices(u32, x, y);
    const vocab = model.config.vocab_size;
    try expectClose(logits_a, logits_b, sample_texts.len, vocab, tol, fixture ++ " gguf");

    // Back to Hugging Face format (quantised tensors dequantised to f16).
    const hf_dir = try tmpPath(gpa, io, &tmp, "hf");
    defer gpa.free(hf_dir);
    try export_mod.saveModel(gpa, io, reloaded, hf_dir, .{}, &sink.writer);
    const again = try Model.load(gpa, io, &pool, hf_dir);
    defer again.deinit();
    try std.testing.expect(again.gguf == null);
    const logits_c = try lastLogits(gpa, again, &sample_texts, null);
    defer gpa.free(logits_c);
    try expectClose(logits_b, logits_c, sample_texts.len, vocab, 1e-2, fixture ++ " hf re-export");
}

test "qwen2 -> GGUF f16 -> reload" {
    try roundTrip("qwen2", .f16, 1e-2, false);
}

test "qwen2 -> GGUF q8_0 -> reload" {
    try roundTrip("qwen2", .q8_0, 5e-2, false);
}

test "llama -> GGUF f16 -> reload from the ggml metadata (permutation, rope_freqs, llama-bpe vocab)" {
    try roundTrip("llama", .f16, 1e-2, true);
}

test "gemma3 -> GGUF f16 -> reload from the ggml metadata (SentencePiece scores, +1 norms)" {
    try roundTrip("gemma3", .f16, 1e-2, true);
}

test "qwen3 -> GGUF bf16 -> reload (q/k norms)" {
    try roundTrip("qwen3", .bf16, 1e-2, true);
}

test "qwen3_moe -> GGUF q8_0 -> reload (stacked experts with per-expert delta)" {
    try roundTrip("qwen3_moe_fused_t", .q8_0, 5e-2, true);
}

test "GGUF input re-exported as GGUF keeps quantisation and copies untouched tensors verbatim" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    const pool = tensor.Pool.init(io, 2);
    inline for ([_][]const u8{ "qwen2_gguf", "llama_gguf" }) |fixture| {
        const src_dir = "tests/fixtures/" ++ fixture;
        const model = try Model.load(gpa, io, &pool, src_dir);
        defer model.deinit();
        try addDeltas(gpa, model);
        const logits_a = try lastLogits(gpa, model, &sample_texts, null);
        defer gpa.free(logits_a);
        var tmp = std.testing.tmpDir(.{});
        defer tmp.cleanup();
        const out_dir = try tmpPath(gpa, io, &tmp, "out");
        defer gpa.free(out_dir);
        var sink: Io.Writer.Allocating = .init(gpa);
        defer sink.deinit();
        try gguf_export.saveGguf(gpa, io, model, out_dir, .{}, &sink.writer);

        var src_d = try Io.Dir.cwd().openDir(io, src_dir, .{});
        defer src_d.close(io);
        var out_d = try Io.Dir.cwd().openDir(io, out_dir, .{});
        defer out_d.close(io);
        const in_f = try gguf.File.open(gpa, io, src_d, "model.gguf");
        defer in_f.close(gpa, io);
        const out_f = try gguf.File.open(gpa, io, out_d, "model.gguf");
        defer out_f.close(gpa, io);
        try std.testing.expectEqual(in_f.tensors.count(), out_f.tensors.count());
        var it = in_f.tensors.iterator();
        while (it.next()) |kv| {
            const a = kv.value_ptr.*;
            const b = out_f.getTensor(a.name) orelse return error.MissingTensor;
            try std.testing.expectEqual(a.ggml_type, b.ggml_type);
            try std.testing.expectEqualSlices(u64, a.dims, b.dims);
            const edited = std.mem.eql(u8, a.name, "blk.0.attn_output.weight") or std.mem.eql(u8, a.name, "blk.1.ffn_down.weight");
            if (edited) {
                try std.testing.expect(!std.mem.eql(u8, a.data, b.data));
            } else {
                try std.testing.expectEqualSlices(u8, a.data, b.data);
            }
        }
        try std.testing.expectEqual(gguf.GgmlType.q8_0, out_f.getTensor("blk.1.ffn_down.weight").?.ggml_type);
        try std.testing.expectEqual(@as(i64, 7), out_f.getInt("general.file_type").?);

        const reloaded = try Model.load(gpa, io, &pool, out_dir);
        defer reloaded.deinit();
        const logits_b = try lastLogits(gpa, reloaded, &sample_texts, null);
        defer gpa.free(logits_b);
        try expectClose(logits_a, logits_b, sample_texts.len, model.config.vocab_size, 5e-2, fixture ++ " gguf->gguf");
    }
}
