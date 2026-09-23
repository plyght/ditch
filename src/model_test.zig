//! End-to-end checks of tokenizer + forward pass against NumPy reference fixtures.
const std = @import("std");
const model_mod = @import("model.zig");
const tensor = @import("tensor.zig");

const Case = struct {
    text: []const u8,
    ids: []u32,
    last_logits: []f32,
    argmax: u32,
    last_hidden: [][]f32,
};
const Reference = struct { family: []const u8, cases: []Case };

pub fn checkFixture(comptime family: []const u8) !void {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    const dir = "tests/fixtures/" ++ family;
    const pool = tensor.Pool.init(io, 2);
    const model = try model_mod.Model.load(gpa, io, &pool, dir);
    defer model.deinit();

    const ref_text = try std.Io.Dir.cwd().readFileAlloc(io, dir ++ "/reference.json", gpa, .unlimited);
    defer gpa.free(ref_text);
    const ref = try std.json.parseFromSlice(Reference, gpa, ref_text, .{ .ignore_unknown_fields = true });
    defer ref.deinit();

    const c = &model.config;
    var ws = try model_mod.Workspace.init(gpa, c, 64, 8);
    defer ws.deinit();
    var cache = try model_mod.KvCache.init(gpa, c.num_layers, ref.value.cases.len, 64, c.kvDim());
    defer cache.deinit();
    if (c.hasRecurrent()) {
        cache.linear = try model_mod.LinearCache.init(gpa, c, ref.value.cases.len);
    }
    if (c.dsv4 != null) {
        cache.compress = try model_mod.dsv4.CompressCache.init(gpa, c, ref.value.cases.len, 64);
    }
    if (c.ngram_ple != null) {
        cache.ngram = try model_mod.qwen4.NgramCache.init(gpa, c, ref.value.cases.len);
    }

    var prompts = std.ArrayList([]const u32).empty;
    defer {
        for (prompts.items) |p| gpa.free(p);
        prompts.deinit(gpa);
    }
    for (ref.value.cases) |case| {
        const ids = try model.tokenizer.encode(gpa, case.text, true);
        try std.testing.expectEqualSlices(u32, case.ids, ids);
        try prompts.append(gpa, ids);
    }
    const logits = try gpa.alloc(f32, prompts.items.len * c.vocab_size);
    defer gpa.free(logits);
    const residuals = try gpa.alloc(f32, (c.num_layers + 1) * prompts.items.len * c.hidden_size);
    defer gpa.free(residuals);
    try model_mod.prefill(model, &ws, &cache, prompts.items, logits, residuals);
    for (ref.value.cases, 0..) |case, b| {
        const got = logits[b * c.vocab_size ..][0..c.vocab_size];
        var max_err: f32 = 0;
        for (case.last_logits, 0..) |e, i| max_err = @max(max_err, @abs(e - got[i]));
        if (max_err > 2e-3) {
            std.debug.print("{s} case {d}: logits max err {d}\n", .{ family, b, max_err });
            return error.LogitsMismatch;
        }
        for (case.last_hidden, 0..) |hl, l| {
            const g = residuals[(l * prompts.items.len + b) * c.hidden_size ..][0..c.hidden_size];
            var herr: f32 = 0;
            for (hl, 0..) |e, i| herr = @max(herr, @abs(e - g[i]));
            if (herr > 2e-3) {
                std.debug.print("{s} case {d} layer entry {d}: hidden max err {d}\n", .{ family, b, l, herr });
                return error.HiddenMismatch;
            }
        }
    }
    // Greedy generation must start with the reference argmax token.
    const outs = try model_mod.generate(model, &ws, &cache, prompts.items, 4);
    defer {
        for (outs) |o| gpa.free(o);
        gpa.free(outs);
    }
    for (ref.value.cases, 0..) |case, b| {
        try std.testing.expectEqual(case.argmax, outs[b][0]);
    }
}

test "llama fixture" {
    try checkFixture("llama");
}
test "qwen2 fixture" {
    try checkFixture("qwen2");
}
test "qwen3 fixture" {
    try checkFixture("qwen3");
}
test "gemma3 fixture" {
    try checkFixture("gemma3");
}
test "gemma2 fixture (alternating local layers, attention and logit softcapping)" {
    try checkFixture("gemma2");
}
test "qwen3_moe fixture (separate expert tensors)" {
    try checkFixture("qwen3_moe");
}
test "qwen3_moe fixture (fused expert tensors)" {
    try checkFixture("qwen3_moe_fused");
}
test "qwen3_moe fixture (transposed fused expert tensors)" {
    try checkFixture("qwen3_moe_fused_t");
}
test "qwen3_moe_big fixture (16 experts, 4 routed layers)" {
    try checkFixture("qwen3_moe_big");
}

// Registry families (tools/make_fixture.py `SPECS`): each fixture exercises the
// tensor names, layouts and config keys of one Hugging Face model_type.
test "phi3 fixture" {
    try checkFixture("phi3");
}
test "phi fixture" {
    try checkFixture("phi");
}
test "gpt_neox fixture" {
    try checkFixture("gpt_neox");
}
test "gpt2 fixture" {
    try checkFixture("gpt2");
}
test "falcon fixture" {
    try checkFixture("falcon");
}

test "falcon_rw fixture (sequential, interleaved qkv, scaled ALiBi)" {
    try checkFixture("falcon_rw");
}
test "stablelm fixture" {
    try checkFixture("stablelm");
}
test "internlm2 fixture" {
    try checkFixture("internlm2");
}
test "olmo2 fixture" {
    try checkFixture("olmo2");
}
test "olmo fixture" {
    try checkFixture("olmo");
}
test "cohere fixture" {
    try checkFixture("cohere");
}
test "glm4 fixture" {
    try checkFixture("glm4");
}
test "chatglm fixture" {
    try checkFixture("chatglm");
}
test "granite fixture" {
    try checkFixture("granite");
}
test "deepseek_v2 fixture" {
    try checkFixture("deepseek_v2");
}
test "deepseek_v3 fixture" {
    try checkFixture("deepseek_v3");
}
test "llama4 fixture" {
    try checkFixture("llama4");
}
test "gpt_oss fixture" {
    try checkFixture("gpt_oss");
}
test "minicpm fixture" {
    try checkFixture("minicpm");
}
test "exaone fixture" {
    try checkFixture("exaone");
}
test "exaone4 fixture" {
    try checkFixture("exaone4");
}
test "nemotron fixture" {
    try checkFixture("nemotron");
}
test "smollm3 fixture" {
    try checkFixture("smollm3");
}
test "bloom fixture" {
    try checkFixture("bloom");
}
test "opt fixture" {
    try checkFixture("opt");
}
test "mpt fixture" {
    try checkFixture("mpt");
}
test "starcoder2 fixture" {
    try checkFixture("starcoder2");
}
test "gpt_bigcode fixture" {
    try checkFixture("gpt_bigcode");
}
test "baichuan fixture" {
    try checkFixture("baichuan");
}
test "mistral fixture" {
    try checkFixture("mistral");
}
test "mixtral fixture" {
    try checkFixture("mixtral");
}
test "qwen2_moe fixture" {
    try checkFixture("qwen2_moe");
}
test "qwen3_next fixture" {
    try checkFixture("qwen3_next");
}
test "qwen3_5 fixture" {
    try checkFixture("qwen3_5");
}
test "qwen3_5_moe fixture" {
    try checkFixture("qwen3_5_moe");
}
test "glm4_moe fixture" {
    try checkFixture("glm4_moe");
}
test "glm_moe_dsa fixture" {
    try checkFixture("glm_moe_dsa");
}
test "glm4_moe_lite fixture (GLM-4.7-Flash)" {
    try checkFixture("glm4_moe_lite");
}
test "glm5_next fixture (GLM-5.3-Flash)" {
    try checkFixture("glm5_next");
}
test "qwen4_exp fixture (Qwen3.8-Flash-Next)" {
    try checkFixture("qwen4_exp");
}
test "seed_oss fixture" {
    try checkFixture("seed_oss");
}
test "lfm2 fixture" {
    try checkFixture("lfm2");
}
test "mistral4 fixture" {
    try checkFixture("mistral4");
}
test "deepseek_v32 fixture (indexed_attention layers as dense, indexer tensors unread)" {
    try checkFixture("deepseek_v32");
}

// The lightning indexer of DeepSeek V3.2 keeps the best `index_topk` keys, so
// running the layer as dense attention is exactly the reference only while the
// context fits that many tokens. The fixture prompts (and the tokens generated
// from them) stay inside the bound; one token past it the forward pass refuses
// rather than silently approximating.
test "deepseek_v32 indexed_attention layers carry the index_topk bound" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    const pool = tensor.Pool.init(io, 2);
    const model = try model_mod.Model.load(gpa, io, &pool, "tests/fixtures/deepseek_v32");
    defer model.deinit();
    const c = &model.config;
    const bound = c.index_bound orelse return error.NoIndexBound;
    // One key per block, `index_topk` of them: exact up to 16 tokens.
    try std.testing.expectEqual(@as(usize, 1), bound.block);
    try std.testing.expectEqual(@as(usize, 16), bound.max_blocks);
    try std.testing.expect(bound.fits(15));
    try std.testing.expect(!bound.fits(16));
    // Every layer is an `indexed_attention` layer, so none of them is sliding,
    // linear or convolutional, and the indexer's own tensors are present in the
    // checkpoint but bound to no weight the forward pass reads.
    for (0..c.num_layers) |li| {
        try std.testing.expect(!c.sliding_layers[li]);
        try std.testing.expect(!c.linear_layers[li]);
        try std.testing.expect(!c.conv_layers[li]);
    }
    for ([_][]const u8{ "wq_b.weight", "wk.weight", "k_norm.weight", "k_norm.bias", "weights_proj.weight" }) |suffix| {
        const name = try std.fmt.allocPrint(gpa, "model.layers.0.self_attn.indexer.{s}", .{suffix});
        defer gpa.free(name);
        try std.testing.expect(model.find(name) != null);
    }
}

test "gemma4 fixture" {
    try checkFixture("gemma4");
}
test "gemma3n fixture" {
    try checkFixture("gemma3n");
}
// q_a_layernorm / kv_a_layernorm use the RMSNorm default 1e-6, not
// rms_norm_eps (1e-5 here), as in DeepSeek's code and transformers; the
// latent projections are scaled down so the difference is plain.
test "deepseek_v3 fixture with rms_norm_eps 1e-5 and small latents (latent norms at 1e-6)" {
    try checkFixture("deepseek_v3_latent_eps");
}
test "deepseek_v4 fixture" {
    try checkFixture("deepseek_v4");
}
test "deepseek_v41 fixture" {
    try checkFixture("deepseek_v41");
}
// The released DeepSeek V4 / V4.1 and GLM-5.3-Flash checkpoints flatten the
// hyper-connection tensors (`layers.N.hc_attn_fn`, `hc_head_fn`) where
// transformers spells them as submodules (`attn_hc.fn`, `hc_head.hc_fn`).
// Same weights, same reference outputs, the other spelling.
test "deepseek_v4 fixture with the released checkpoints' hyper-connection names" {
    try checkFixture("deepseek_v4_hubnames");
}
// DeepSeek V4 as released: DeepSeek's own tensor names (`embed.weight`,
// `layers.N.attn.wq_a.weight`, `layers.N.ffn.experts.E.w1.weight`), FP8
// matrices with F8_E8M0 block scales under `<module>.scale`, FP4 experts (I8
// e2m1 nibble pairs, an F8_E8M0 scale per 32 columns), `expert_dtype: fp4` at
// the top of config.json. Reference outputs from the dequantised weights.
test "deepseek_v4 fixture in DeepSeek's own names with fp8 and fp4 weights" {
    try checkFixture("deepseek_v4_native");
}
test "deepseek_v4_native edit, export and streamed reload (renamed and dequantised on load)" {
    try checkEditExportStream("deepseek_v4_native");
}
test "minimax_m2 fixture" {
    try checkFixture("minimax_m2");
}
test "minimax fixture (lightning attention, renormalised residual)" {
    try checkFixture("minimax");
}
test "minimax_m3 fixture" {
    try checkFixture("minimax_m3");
}
// The released MiniMax-M3 checkpoints keep the dense and shared MLPs split
// (`gate_proj` / `up_proj`) where the reference implementation fuses them.
test "minimax_m3 fixture with split gate/up projections" {
    try checkFixture("minimax_m3_split");
}
test "ernie4_5_moe fixture" {
    try checkFixture("ernie4_5_moe");
}
test "hunyuan_v1_moe fixture" {
    try checkFixture("hunyuan_v1_moe");
}
test "granitemoe fixture" {
    try checkFixture("granitemoe");
}
test "granitemoehybrid fixture (attention only, fused shared_mlp)" {
    try checkFixture("granitemoehybrid_attn");
}
test "kimi_linear fixture (checkpoint layout: split convolutions, block_sparse_moe)" {
    try checkFixture("kimi_linear");
}
test "kimi_linear fixture (Hugging Face layout: forget_gate, fused conv1d, stacked experts)" {
    try checkFixture("kimi_linear_hf");
}
test "mimo_v2_flash fixture (transformers spelling: layer_types, rope_parameters, stacked experts)" {
    try checkFixture("mimo_v2_flash");
}
test "mimo_v2 fixture (hub spelling: hybrid_layer_pattern, chunked qkv_proj, omni and MTP pass-through)" {
    try checkFixture("mimo_v2");
}
test "kimi_k25 fixture (DeepSeek V3 text config under a multimodal wrapper)" {
    try checkFixture("kimi_k25");
}
test "kimi_k3 fixture (Attention Residual, full-rank and safe KDA gates, MLA output gate, latent MoE, SiTU)" {
    try checkFixture("kimi_k3");
}

// Families swept from transformers' MODEL_FOR_CAUSAL_LM mapping.
test "arcee fixture (dense relu² MLP)" {
    try checkFixture("arcee");
}
test "jais2 fixture (LayerNorm, dense relu² MLP with biases)" {
    try checkFixture("jais2");
}
test "ernie4_5 fixture (interleaved rotary, use_bias)" {
    try checkFixture("ernie4_5");
}
test "helium fixture (interleaved pairs against the duplicated rotary table)" {
    try checkFixture("helium");
}
test "hunyuan_v1_dense fixture (q/k norm after rope, NTK-alpha base)" {
    try checkFixture("hunyuan_v1_dense");
}
test "bitnet fixture (attention and MLP sub-norms)" {
    try checkFixture("bitnet");
}
test "apertus fixture (xIELU activation, attention/feedforward norms)" {
    try checkFixture("apertus");
}
test "nanochat fixture (non-parametric norms, weightless q/k norm, logit softcap)" {
    try checkFixture("nanochat");
}
test "persimmon fixture (head-interleaved qkv, per-head q/k LayerNorm)" {
    try checkFixture("persimmon");
}
test "gptj fixture (parallel residual, interleaved partial rotary)" {
    try checkFixture("gptj");
}
test "codegen fixture (tensor-parallel [q | v | k] blocks)" {
    try checkFixture("codegen");
}
test "gpt_neo fixture (learned positions, unscaled logits, local layers)" {
    try checkFixture("gpt_neo");
}
test "xglm fixture (sinusoidal positions, embedding scale)" {
    try checkFixture("xglm");
}
test "biogpt fixture (learned positions with offset 2, output_projection)" {
    try checkFixture("biogpt");
}
test "ministral3 fixture (query scaling on every layer)" {
    try checkFixture("ministral3");
}
test "granite_swa fixture (attention sinks, sliding layers with their own rope base)" {
    try checkFixture("granite_swa");
}
test "olmoe fixture (full-projection q/k norm, clip_qkv, softmax routing)" {
    try checkFixture("olmoe");
}
test "flex_olmo fixture (post-norms, fused experts)" {
    try checkFixture("flex_olmo");
}
test "dots1 fixture (DeepSeek-V3 routing, shared experts, sliding layers)" {
    try checkFixture("dots1");
}
test "exaone_moe fixture (sliding_window_pattern, mlp_layer_types)" {
    try checkFixture("exaone_moe");
}
test "solar_open fixture (partial rotary, fused experts with a shared expert)" {
    try checkFixture("solar_open");
}
test "afmoe fixture (sigmoid attention gate, four norms, route_scale)" {
    try checkFixture("afmoe");
}
test "mellum fixture (per-layer-type rope parameters, softmax routing)" {
    try checkFixture("mellum");
}
test "laguna fixture (per-head softplus gate, router logit softcapping)" {
    try checkFixture("laguna");
}
test "hy_v3 fixture (checkpoint MoE names, shared_mlp)" {
    try checkFixture("hy_v3");
}

// Quantised checkpoints (dequantised on load, see dequant.zig): the reference
// logits were computed on the dequantised weights.
test "qwen2_fp8 fixture (FP8 E4M3 with block scales)" {
    try checkFixture("qwen2_fp8");
}
test "qwen2_int4 fixture (compressed-tensors pack-quantized, zero points)" {
    try checkFixture("qwen2_int4");
}
// `actorder`: `weight_g_idx` names each column's scale group instead of
// `column / group_size`. Ignoring it decodes every column with the wrong
// scale and the model still loads, so the reference pins the mapping.
test "qwen2_int4 fixture with actorder (weight_g_idx groups)" {
    try checkFixture("qwen2_int4_actorder");
}
test "gpt_oss_mxfp4 fixture (MXFP4 expert blocks and scales)" {
    try checkFixture("gpt_oss_mxfp4");
}
test "kimi_k3_mxfp4 fixture (compressed-tensors mxfp4-pack-quantized experts)" {
    try checkFixture("kimi_k3_mxfp4");
}
test "mimo_v2_fp8 fixture (MiMo V2.5 / V2.6 fp8 qkv_proj blocked per tensor-parallel shard)" {
    try checkFixture("mimo_v2_fp8");
}

test "mimo_v2_split_fp8 fixture (MiMo-V2-Flash fp8 q/k/v blocked per tensor-parallel shard)" {
    try checkFixture("mimo_v2_split_fp8");
}

test "mimo_v2_mxfp4 fixture (MiMo V2.6 store_dtype mxfp4 experts, bf16 MoE router)" {
    try checkFixture("mimo_v2_mxfp4");
}

// Mamba families (selective state-space blocks with a per-sequence recurrent state).
test "mamba2 fixture" {
    try checkFixture("mamba2");
}
test "nemotron_h fixture" {
    try checkFixture("nemotron_h");
}
test "falcon_h1 fixture" {
    try checkFixture("falcon_h1");
}
test "falcon_h1 fixture (no gated norm)" {
    try checkFixture("falcon_h1_nonorm");
}
test "jamba fixture" {
    try checkFixture("jamba");
}
test "granitemoehybrid fixture" {
    try checkFixture("granitemoehybrid");
}
test "granitemoehybrid fixture (no experts, rope)" {
    try checkFixture("granitemoehybrid_dense");
}

/// Decoding one token at a time from the recurrent state must give the
/// logits a fresh prefill of the longer prompt gives: the Mamba state of a
/// sequence survives across forward calls exactly like the KV cache.
fn checkDecodeMatchesPrefill(comptime family: []const u8) !void {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    const dir = "tests/fixtures/" ++ family;
    const pool = tensor.Pool.init(io, 2);
    const model = try model_mod.Model.load(gpa, io, &pool, dir);
    defer model.deinit();
    const c = &model.config;
    try std.testing.expect(c.has_ssm);
    const prompt = [_]u32{ 40, 100, 200, 7, 3, 66 };
    const extra = [_]u32{ 12, 90 };
    var ws = try model_mod.Workspace.init(gpa, c, 16, 2);
    defer ws.deinit();
    // Two slots: the second sequence starts later and restarts once, so the
    // per-slot bookkeeping (reset at position 0) is exercised too.
    var cache = try model_mod.KvCache.initFor(model, gpa, 2, 16);
    defer cache.deinit();
    const step_logits = try gpa.alloc(f32, extra.len * c.vocab_size);
    defer gpa.free(step_logits);
    try model_mod.prefill(model, &ws, &cache, &.{ &prompt, prompt[0..3] }, null, null);
    for (extra, 0..) |t, k| {
        const rows = [_]model_mod.Row{.{ .b = 0, .pos = prompt.len + k }};
        try model_mod.forward(model, &ws, &cache, &.{t}, &rows, .{ .logit_rows = &.{0} });
        @memcpy(step_logits[k * c.vocab_size ..][0..c.vocab_size], ws.logits[0..c.vocab_size]);
    }
    for (0..extra.len) |k| {
        const full = prompt ++ extra;
        var fresh = try model_mod.KvCache.initFor(model, gpa, 1, 16);
        defer fresh.deinit();
        const logits = try gpa.alloc(f32, c.vocab_size);
        defer gpa.free(logits);
        try model_mod.prefill(model, &ws, &fresh, &.{full[0 .. prompt.len + k + 1]}, logits, null);
        var max_err: f32 = 0;
        for (logits, step_logits[k * c.vocab_size ..][0..c.vocab_size]) |a, b| max_err = @max(max_err, @abs(a - b));
        try std.testing.expect(max_err < 1e-4);
    }
    // A slot that restarts at position 0 forgets its state; a gap is an error.
    try model_mod.prefill(model, &ws, &cache, &.{ prompt[0..2], prompt[0..4] }, null, null);
    const gap = [_]model_mod.Row{.{ .b = 1, .pos = 6 }};
    try std.testing.expectError(error.NonContiguousRows, model_mod.forward(model, &ws, &cache, &.{1}, &gap, .{}));
}

// Warp mode (streamed weights with an expert cache) on the non-gated
// Nemotron-H experts: two matrices per cache entry, same logits as mapped.
test "nemotron_h warp mode matches mapped mode" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    const pool = tensor.Pool.init(io, 2);
    const dir = "tests/fixtures/nemotron_h";
    const mapped = try model_mod.Model.load(gpa, io, &pool, dir);
    defer mapped.deinit();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const n = try tmp.dir.realPath(io, &path_buf);
    const scratch = try std.fs.path.join(gpa, &.{ path_buf[0..n], "scratch" });
    defer gpa.free(scratch);
    const warp = try model_mod.Model.loadWithOptions(gpa, io, &pool, dir, .{ .store = .streamed, .scratch_dir = scratch });
    defer warp.deinit();
    try std.testing.expect(warp.warp());
    const ids = [_]u32{ 40, 100, 200, 7, 3 };
    const a = try runLogits(mapped, gpa, &ids);
    defer gpa.free(a);
    const b = try runLogits(warp, gpa, &ids);
    defer gpa.free(b);
    try std.testing.expectEqualSlices(f32, a, b);
    try std.testing.expect(warp.expert_cache.?.anyVisited());
}

// A Hugging Face hub snapshot directory (`~/.cache/huggingface/hub/
// models--o--n/snapshots/<sha>/`) is symbolic links into `blobs/`, so a
// directory scan that only accepts regular files finds no weights there.
test "a model directory of symbolic links loads (hub snapshot layout)" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    const pool = tensor.Pool.init(io, 2);
    const src = "tests/fixtures/llama";
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const tmp_len = try tmp.dir.realPath(io, &path_buf);
    const tmp_path = path_buf[0..tmp_len];
    var src_dir = try std.Io.Dir.cwd().openDir(io, src, .{ .iterate = true });
    defer src_dir.close(io);
    var src_buf: [std.fs.max_path_bytes]u8 = undefined;
    const src_len = try src_dir.realPath(io, &src_buf);
    const src_abs = src_buf[0..src_len];
    var it = src_dir.iterate();
    while (try it.next(io)) |entry| {
        if (entry.kind != .file) continue;
        const abs = try std.fs.path.join(gpa, &.{ src_abs, entry.name });
        defer gpa.free(abs);
        try tmp.dir.symLink(io, abs, entry.name, .{});
    }
    const linked = try model_mod.Model.load(gpa, io, &pool, tmp_path);
    defer linked.deinit();
    const direct = try model_mod.Model.load(gpa, io, &pool, src);
    defer direct.deinit();
    const ids = [_]u32{ 40, 100, 200, 7, 3 };
    const a = try runLogits(direct, gpa, &ids);
    defer gpa.free(a);
    const b = try runLogits(linked, gpa, &ids);
    defer gpa.free(b);
    try std.testing.expectEqualSlices(f32, a, b);
}

test "mamba2 decode from the recurrent state matches prefill" {
    try checkDecodeMatchesPrefill("mamba2");
}
test "jamba decode from the recurrent state matches prefill" {
    try checkDecodeMatchesPrefill("jamba");
}
test "falcon_h1 decode from the recurrent state matches prefill" {
    try checkDecodeMatchesPrefill("falcon_h1");
}

// ---------------------------------------------------------------------------
// Abliteration, export and streaming on the registry layouts
// ---------------------------------------------------------------------------

const abliterate = @import("abliterate.zig");
const export_mod = @import("export.zig");

fn randomDirs(gpa: std.mem.Allocator, entries: usize, hidden: usize, seed: u64) ![]f32 {
    const dirs = try gpa.alloc(f32, entries * hidden);
    var prng = std.Random.DefaultPrng.init(seed);
    const rand = prng.random();
    for (dirs) |*x| x.* = rand.floatNorm(f32);
    for (0..entries) |e| tensor.normalize(dirs[e * hidden ..][0..hidden]);
    return dirs;
}

fn runLogits(model: *const model_mod.Model, gpa: std.mem.Allocator, ids: []const u32) ![]f32 {
    const c = &model.config;
    var ws = try model_mod.Workspace.init(gpa, c, 16, 1);
    defer ws.deinit();
    var cache = try model_mod.KvCache.initFor(model, gpa, 1, 16);
    defer cache.deinit();
    const logits = try gpa.alloc(f32, c.vocab_size);
    errdefer gpa.free(logits);
    try model_mod.prefill(model, &ws, &cache, &.{ids}, logits, null);
    return logits;
}

fn bothComponents() std.EnumMap(model_mod.Component, abliterate.Params) {
    var params = std.EnumMap(model_mod.Component, abliterate.Params){};
    // The kernel reaches every layer of every fixture (the deepest has 5).
    params.put(.attn_o_proj, .{ .max_weight = 1.0, .max_weight_position = 1, .min_weight = 0.5, .min_weight_distance = 4 });
    params.put(.mlp_down_proj, .{ .max_weight = 1.0, .max_weight_position = 1, .min_weight = 0.5, .min_weight_distance = 4 });
    return params;
}

/// Loads a fixture, abliterates both components on every layer, checks that
/// the edit changed the logits, exports the merged weights (f32, so nothing
/// is re-rounded) and reloads them in mapped and streamed mode: both must
/// reproduce the delta model, and the streamed reload must match the mapped
/// one bit for bit. Covers the family's output/down projection mapping
/// (`Component`), the export edit (Conv1D transposes, fused expert tensors)
/// and the streamed acquisition of every layout.
fn checkEditExportStream(comptime family: []const u8) !void {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    const pool = tensor.Pool.init(io, 2);
    const model = try model_mod.Model.load(gpa, io, &pool, "tests/fixtures/" ++ family);
    defer model.deinit();
    const ids = [_]u32{ 40, 100, 200, 7, 3 };
    const base = try runLogits(model, gpa, &ids);
    defer gpa.free(base);
    const hidden = model.config.hidden_size;
    const dirs = try randomDirs(gpa, model.config.num_layers + 1, hidden, 7);
    defer gpa.free(dirs);
    try abliterate.apply(model, dirs, null, bothComponents(), .{ .row_normalization = .full, .lora_rank = 2 });
    var n_edits: usize = 0;
    for (model.layers, 0..) |*layer, li| {
        // Single-block layers (Mamba2, Nemotron-H) hold only one of the two components.
        if (model.hasComponent(li, .attn_o_proj)) {
            try std.testing.expect(model.getDelta(li, .attn_o_proj) != null);
            n_edits += 1;
        } else try std.testing.expect(model.getDelta(li, .attn_o_proj) == null);
        if (model.hasSsmOut(li)) {
            try std.testing.expect(model.getSsmOutDelta(li) != null);
            n_edits += 1;
        }
        if (layer.moe) |*m| {
            if (m.routedInLatent()) {
                // Latent MoE: the routed write matrix carries the edit, the latent-space experts none.
                try std.testing.expect(model.getLatentDelta(li) != null);
                for (0..m.experts.len) |e| try std.testing.expect(model.getExpertDelta(li, e) == null);
                if (m.sharedIndex()) |si| try std.testing.expect(model.getExpertDelta(li, si) != null);
            } else {
                for (0..m.experts.len) |e| try std.testing.expect(model.getExpertDelta(li, e) != null);
            }
            n_edits += 1;
        } else if (model.hasComponent(li, .mlp_down_proj)) {
            try std.testing.expect(model.getDelta(li, .mlp_down_proj) != null);
            n_edits += 1;
        } else try std.testing.expect(model.getDelta(li, .mlp_down_proj) == null);
    }
    try std.testing.expect(n_edits >= model.layers.len);
    const edited = try runLogits(model, gpa, &ids);
    defer gpa.free(edited);
    var moved: f32 = 0;
    for (base, edited) |a, b| moved = @max(moved, @abs(a - b));
    try std.testing.expect(moved > 1e-4);

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const n = try tmp.dir.realPath(io, &path_buf);
    const out_dir = try std.fs.path.join(gpa, &.{ path_buf[0..n], "exported" });
    defer gpa.free(out_dir);
    var sink: std.Io.Writer.Allocating = .init(gpa);
    defer sink.deinit();
    try export_mod.saveModel(gpa, io, model, out_dir, .{ .export_dtype = .f32 }, &sink.writer);

    const reloaded = try model_mod.Model.load(gpa, io, &pool, out_dir);
    defer reloaded.deinit();
    const merged = try runLogits(reloaded, gpa, &ids);
    defer gpa.free(merged);
    var scale: f32 = 0;
    for (edited) |x| scale = @max(scale, @abs(x));
    for (edited, merged) |a, b| try std.testing.expectApproxEqAbs(a, b, 2e-3 * @max(scale, 1));
    // Every tensor name and shape survives the round trip.
    for (model.files) |f| {
        var it = f.tensors.iterator();
        while (it.next()) |kv| {
            const info = reloaded.find(kv.key_ptr.*) orelse return error.TensorLost;
            try std.testing.expectEqualSlices(usize, kv.value_ptr.shape, info.shape);
        }
    }
    const scratch = try std.fs.path.join(gpa, &.{ path_buf[0..n], "scratch" });
    defer gpa.free(scratch);
    const streamed = try model_mod.Model.loadWithOptions(gpa, io, &pool, out_dir, .{ .store = .streamed, .scratch_dir = scratch, .expert_cache = 0 });
    defer streamed.deinit();
    try std.testing.expect(streamed.streamed());
    const via_stream = try runLogits(streamed, gpa, &ids);
    defer gpa.free(via_stream);
    try std.testing.expectEqualSlices(f32, merged, via_stream);
}

test "gpt2 edit, export and streamed reload (Conv1D transposes)" {
    try checkEditExportStream("gpt2");
}
test "phi3 edit, export and streamed reload (fused qkv and gate_up)" {
    try checkEditExportStream("phi3");
}
test "gpt_neox edit, export and streamed reload (interleaved qkv, biases)" {
    try checkEditExportStream("gpt_neox");
}
test "bloom edit, export and streamed reload (ALiBi, embedding norm)" {
    try checkEditExportStream("bloom");
}
test "deepseek_v3 edit, export and streamed reload (MLA, sigmoid MoE)" {
    try checkEditExportStream("deepseek_v3");
}
test "gpt_oss edit, export and streamed reload (sinks, interleaved fused experts)" {
    try checkEditExportStream("gpt_oss");
}
test "llama4 edit, export and streamed reload (transposed fused experts, NoPE layers)" {
    try checkEditExportStream("llama4");
}
test "chatglm edit, export and streamed reload (remote-code layout)" {
    try checkEditExportStream("chatglm");
}
test "cohere edit, export and streamed reload (parallel residual, tied head)" {
    try checkEditExportStream("cohere");
}
test "qwen3_next edit, export and streamed reload (linear attention, gated full attention)" {
    try checkEditExportStream("qwen3_next");
}
test "qwen3_5_moe edit, export and streamed reload (split linear projections, swish gate)" {
    try checkEditExportStream("qwen3_5_moe");
}
test "glm4_moe edit, export and streamed reload (per-head q/k norms, sigmoid MoE)" {
    try checkEditExportStream("glm4_moe");
}
test "glm_moe_dsa edit, export and streamed reload (MLA sparse indexer as dense)" {
    try checkEditExportStream("glm_moe_dsa");
}
test "glm4_moe_lite edit, export and streamed reload (stacked experts, group-limited sigmoid MoE)" {
    try checkEditExportStream("glm4_moe_lite");
}
test "glm5_next edit, export and streamed reload (mHC streams, KDA, NoPE MLA behind the indexer)" {
    try checkEditExportStream("glm5_next");
}
test "qwen4_exp edit, export and streamed reload (gated streams, sharded n-gram tables)" {
    try checkEditExportStream("qwen4_exp");
}
test "mamba2 edit, export and streamed reload (Mamba out_proj as the attention output)" {
    try checkEditExportStream("mamba2");
}
test "nemotron_h edit, export and streamed reload (single-block layers, non-gated experts)" {
    try checkEditExportStream("nemotron_h");
}
test "falcon_h1 edit, export and streamed reload (o_proj and Mamba out_proj side by side)" {
    try checkEditExportStream("falcon_h1");
}
test "jamba edit, export and streamed reload (Mamba1, separate expert tensors)" {
    try checkEditExportStream("jamba");
}
test "granitemoehybrid edit, export and streamed reload (fused input_linear experts, shared_mlp)" {
    try checkEditExportStream("granitemoehybrid");
}
test "seed_oss edit, export and streamed reload (biased q/k/v, explicit head_dim)" {
    try checkEditExportStream("seed_oss");
}
test "lfm2 edit, export and streamed reload (short-conv out_proj as the attention output)" {
    try checkEditExportStream("lfm2");
}
test "mistral4 edit, export and streamed reload (MLA, fused softmax MoE, shared experts)" {
    try checkEditExportStream("mistral4");
}
test "mistral edit, export and streamed reload (sliding window on every layer, explicit head_dim)" {
    try checkEditExportStream("mistral");
}
test "gemma2 edit, export and streamed reload (alternating local layers, softcapping)" {
    try checkEditExportStream("gemma2");
}
test "mixtral edit, export and streamed reload (block_sparse_moe w1/w2/w3 experts)" {
    try checkEditExportStream("mixtral");
}
test "qwen2_moe edit, export and streamed reload (gated shared expert, dense mlp_only layer)" {
    try checkEditExportStream("qwen2_moe");
}
test "deepseek_v32 edit, export and streamed reload (indexer tensors pass through)" {
    try checkEditExportStream("deepseek_v32");
}
test "gemma4 edit, export and streamed reload (per-layer head sizes, KV sharing, per-layer inputs)" {
    try checkEditExportStream("gemma4");
}
test "gemma3n edit, export and streamed reload (AltUp streams, Laurel, per-layer inputs)" {
    try checkEditExportStream("gemma3n");
}
test "mimo_v2_flash edit, export and streamed reload (sinks on sliding layers, doubled kv heads, stacked experts, MTP pass-through)" {
    try checkEditExportStream("mimo_v2_flash");
}
test "mimo_v2 edit, export and streamed reload (chunked qkv_proj, per-expert tensors, omni tensors pass-through)" {
    try checkEditExportStream("mimo_v2");
}
test "mimo_v2_mxfp4 edit, export and streamed reload (store_dtype mxfp4 experts export as bf16)" {
    try checkEditExportStream("mimo_v2_mxfp4");
}
test "deepseek_v4 edit, export and streamed reload (hyper-connections, hash routing, MTP pass-through)" {
    try checkEditExportStream("deepseek_v4");
}
test "deepseek_v41 edit, export and streamed reload (shared compressed KV, engram tables)" {
    try checkEditExportStream("deepseek_v41");
}
test "minimax_m2 edit, export and streamed reload (full-projection q/k norm, Mixtral expert names)" {
    try checkEditExportStream("minimax_m2");
}
test "minimax edit, export and streamed reload (lightning attention out_proj)" {
    try checkEditExportStream("minimax");
}
test "minimax_m3 edit, export and streamed reload (fused shared expert, pass-through indexer tensors)" {
    try checkEditExportStream("minimax_m3");
}
test "minimax_m3 split edit, export and streamed reload" {
    try checkEditExportStream("minimax_m3_split");
}
test "ernie4_5_moe edit, export and streamed reload (moe_statics bias, shared experts)" {
    try checkEditExportStream("ernie4_5_moe");
}
test "kimi_k3 edit, export and streamed reload (latent MoE up projection, attention residual scorers pass through)" {
    try checkEditExportStream("kimi_k3");
}
test "hunyuan_v1_moe edit, export and streamed reload (q/k norm after rope, shared_mlp)" {
    try checkEditExportStream("hunyuan_v1_moe");
}
test "granitemoe edit, export and streamed reload (fused input_linear / output_linear experts)" {
    try checkEditExportStream("granitemoe");
}
test "granitemoehybrid edit, export and streamed reload (attention only, fused shared_mlp)" {
    try checkEditExportStream("granitemoehybrid_attn");
}
test "kimi_linear edit, export and streamed reload (KDA, block_sparse_moe experts)" {
    try checkEditExportStream("kimi_linear");
}
test "kimi_linear_hf edit, export and streamed reload (fused conv1d, stacked experts)" {
    try checkEditExportStream("kimi_linear_hf");
}
test "kimi_k25 edit, export and streamed reload (language_model prefix)" {
    try checkEditExportStream("kimi_k25");
}
test "bitnet edit, export and streamed reload (attention and MLP sub-norms)" {
    try checkEditExportStream("bitnet");
}
test "apertus edit, export and streamed reload (xIELU parameters)" {
    try checkEditExportStream("apertus");
}
test "persimmon edit, export and streamed reload (head-interleaved qkv)" {
    try checkEditExportStream("persimmon");
}
test "codegen edit, export and streamed reload (tensor-parallel qkv blocks)" {
    try checkEditExportStream("codegen");
}
test "gpt_neo edit, export and streamed reload (learned positions, local layers)" {
    try checkEditExportStream("gpt_neo");
}
test "xglm edit, export and streamed reload (sinusoidal positions)" {
    try checkEditExportStream("xglm");
}
test "granite_swa edit, export and streamed reload (sinks, per-layer-type rope)" {
    try checkEditExportStream("granite_swa");
}
test "olmoe edit, export and streamed reload (separate experts, clip_qkv)" {
    try checkEditExportStream("olmoe");
}
test "dots1 edit, export and streamed reload (shared experts, correction bias)" {
    try checkEditExportStream("dots1");
}
test "afmoe edit, export and streamed reload (attention gate, four norms)" {
    try checkEditExportStream("afmoe");
}
test "laguna edit, export and streamed reload (per-head gate, router softcap)" {
    try checkEditExportStream("laguna");
}
test "hy_v3 edit, export and streamed reload (checkpoint MoE names)" {
    try checkEditExportStream("hy_v3");
}
test "qwen2_fp8 edit, export and streamed reload (dequantised on load)" {
    try checkEditExportStream("qwen2_fp8");
}
test "qwen2_int4 edit, export and streamed reload (dequantised on load)" {
    try checkEditExportStream("qwen2_int4");
}
test "qwen2_int4 actorder edit, export and streamed reload" {
    try checkEditExportStream("qwen2_int4_actorder");
}
test "gpt_oss_mxfp4 edit, export and streamed reload (dequantised experts)" {
    try checkEditExportStream("gpt_oss_mxfp4");
}

/// A quantised checkpoint exports as a plain bf16 one: the storage tensors are
/// gone, the dequantised weights are written in bf16 under their model names,
/// `config.json` has no `quantization_config`, and the export reloads (mapped
/// and streamed) to the same logits as the source.
fn checkQuantisedExport(comptime family: []const u8, comptime weight: []const u8, comptime storage: []const u8) !void {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    const pool = tensor.Pool.init(io, 2);
    const model = try model_mod.Model.load(gpa, io, &pool, "tests/fixtures/" ++ family);
    defer model.deinit();
    try std.testing.expect(model.dequantised > 0);
    try std.testing.expect(model.find(weight) != null);
    try std.testing.expect(model.find(storage) == null);
    try std.testing.expect(std.mem.indexOf(u8, model.config_json, "quantization_config") != null);
    try std.testing.expect(std.mem.indexOf(u8, model.export_config_json, "quantization_config") == null);
    const ids = [_]u32{ 40, 100, 200, 7, 3 };
    const base = try runLogits(model, gpa, &ids);
    defer gpa.free(base);

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const n = try tmp.dir.realPath(io, &path_buf);
    const out_dir = try std.fs.path.join(gpa, &.{ path_buf[0..n], "exported" });
    defer gpa.free(out_dir);
    var sink: std.Io.Writer.Allocating = .init(gpa);
    defer sink.deinit();
    try export_mod.saveModel(gpa, io, model, out_dir, .{}, &sink.writer);

    const reloaded = try model_mod.Model.load(gpa, io, &pool, out_dir);
    defer reloaded.deinit();
    try std.testing.expectEqual(@as(usize, 0), reloaded.dequantised);
    const info = reloaded.find(weight) orelse return error.TensorLost;
    try std.testing.expectEqual(tensor.DType.bf16, info.dtype);
    try std.testing.expect(reloaded.find(storage) == null);
    try std.testing.expect(std.mem.indexOf(u8, reloaded.config_json, "quantization_config") == null);
    // The dequantised values are bf16 already, so the export is exact.
    const again = try runLogits(reloaded, gpa, &ids);
    defer gpa.free(again);
    try std.testing.expectEqualSlices(f32, base, again);
    const scratch = try std.fs.path.join(gpa, &.{ path_buf[0..n], "scratch" });
    defer gpa.free(scratch);
    const streamed = try model_mod.Model.loadWithOptions(gpa, io, &pool, "tests/fixtures/" ++ family, .{ .store = .streamed, .scratch_dir = scratch, .expert_cache = 0 });
    defer streamed.deinit();
    const via_stream = try runLogits(streamed, gpa, &ids);
    defer gpa.free(via_stream);
    try std.testing.expectEqualSlices(f32, base, via_stream);
}

test "qwen2_fp8 exports as bf16 without quantization_config" {
    try checkQuantisedExport("qwen2_fp8", "model.layers.0.self_attn.q_proj.weight", "model.layers.0.self_attn.q_proj.weight_scale_inv");
}
test "qwen2_int4 exports as bf16 without quantization_config" {
    try checkQuantisedExport("qwen2_int4", "model.layers.1.mlp.down_proj.weight", "model.layers.1.mlp.down_proj.weight_packed");
}
test "gpt_oss_mxfp4 exports as bf16 without quantization_config" {
    try checkQuantisedExport("gpt_oss_mxfp4", "model.layers.2.mlp.experts.gate_up_proj", "model.layers.2.mlp.experts.gate_up_proj_blocks");
}
