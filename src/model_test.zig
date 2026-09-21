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

fn checkFixture(comptime family: []const u8) !void {
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
    var ws = try model_mod.Workspace.init(gpa, c, 64);
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
