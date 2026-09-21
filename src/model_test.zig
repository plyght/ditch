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
    var cache = try model_mod.KvCache.init(gpa, c.num_layers, ref.value.cases.len, 64, c.num_kv_heads * c.head_dim);
    defer cache.deinit();

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
test "qwen3_moe fixture (separate expert tensors)" {
    try checkFixture("qwen3_moe");
}
test "qwen3_moe fixture (fused expert tensors)" {
    try checkFixture("qwen3_moe_fused");
}
test "qwen3_moe fixture (transposed fused expert tensors)" {
    try checkFixture("qwen3_moe_fused_t");
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
