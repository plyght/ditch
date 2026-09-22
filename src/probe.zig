//! `ditch probe <model> --prompt TEXT`: shows what the model sees and says
//! for a prompt, so the tokenizer, the chat template and the forward pass
//! can be checked against a reference implementation (see
//! tools/probe_reference.py). Prints the rendered prompt, its token ids,
//! the highest first-token logits and the greedy continuation; `--json`
//! also includes the full first-token logit vector, and `--residuals` the
//! last token's residual at every layer (layer L is the vector layer L
//! reads, so entry 0 is the embedding and entry `num_layers` is what the
//! final norm reads — the same series as transformers' `hidden_states`).

const std = @import("std");
const Io = std.Io;
const Allocator = std.mem.Allocator;

const config = @import("config.zig");
const tensor = @import("tensor.zig");
const model_mod = @import("model.zig");
const engine_mod = @import("engine.zig");
const chat = @import("chat.zig");
const hf = @import("hf.zig");

const Model = model_mod.Model;
const Engine = engine_mod.Engine;

const top_k = 10;

const Top = struct { id: u32, logit: f32 };

fn topK(logits: []const f32, out: []Top) []Top {
    var n: usize = 0;
    for (logits, 0..) |v, i| {
        if (n < out.len) {
            out[n] = .{ .id = @intCast(i), .logit = v };
            n += 1;
        } else if (v > out[n - 1].logit) {
            out[n - 1] = .{ .id = @intCast(i), .logit = v };
        } else continue;
        // Keep the filled prefix sorted, highest first.
        var j = n - 1;
        while (j > 0 and out[j].logit > out[j - 1].logit) : (j -= 1) std.mem.swap(Top, &out[j], &out[j - 1]);
    }
    return out[0..n];
}

/// Runs the probe: messages go to `out`, the report (text or JSON) to `result_out`.
pub fn run(gpa: Allocator, arena: Allocator, io: Io, settings: *config.Settings, http: *hf.Http, cache_root: []const u8, pool: *const tensor.Pool, out: *Io.Writer, result_out: *Io.Writer) !void {
    if (settings.probe_prompts.len == 0) {
        std.log.err("ditch probe needs at least one --prompt TEXT", .{});
        std.process.exit(2);
    }
    try out.print("\nProbing {s}...\n", .{settings.model});
    try out.flush();
    const model_dir = try hf.resolveModel(arena, http, cache_root, settings.model, settings.model_commit, out);
    const model = try Model.load(gpa, io, pool, model_dir);
    defer model.deinit();
    const c = &model.config;
    const template = if (settings.chat_template) |name| (chat.Template.parse(name) orelse return error.InvalidChatTemplate) else chat.detect(model.chat_template, c.model_type);
    var engine = Engine.init(gpa, model, settings, template);
    defer engine.deinit();
    if (settings.response_prefix == null) settings.response_prefix = "";
    engine.batch_size = 1;
    try out.print("* Architecture: {s} ({d} layers, vocabulary {d}, {s} weights), chat template {s}\n", .{ c.model_type, c.num_layers, c.vocab_size, model.dtype.safetensorsName(), @tagName(template) });
    try out.flush();

    var js: std.json.Stringify = .{ .writer = result_out };
    if (settings.json) {
        try js.beginObject();
        try js.objectField("model");
        try js.write(settings.model);
        try js.objectField("model_type");
        try js.write(c.model_type);
        try js.objectField("chat_template");
        try js.write(@tagName(template));
        try js.objectField("prompts");
        try js.beginArray();
    }
    for (settings.probe_prompts) |user| {
        const text = if (settings.probe_raw) try gpa.dupe(u8, user) else try engine.formatPrompt(gpa, .{ .system = settings.system_prompt, .user = user });
        defer gpa.free(text);
        const ids = try model.tokenizer.encode(gpa, text, !settings.probe_raw);
        defer gpa.free(ids);
        const prompts = [_][]u32{ids};
        const generated = try engine.generateBatch(gpa, &prompts, @max(settings.max_response_length, 1));
        defer {
            for (generated) |g| model.gpa.free(g);
            model.gpa.free(generated);
        }
        const response = try model.tokenizer.decode(gpa, generated[0], false);
        defer gpa.free(response);
        // First-token logits: a separate prefill so the vector is exact
        // (`generate` only keeps the argmax).
        const ws = try engine.ensureWorkspace(ids.len, 1, 0);
        var cache = try model_mod.KvCache.initFor(model, gpa, 1, ids.len + 1);
        defer cache.deinit();
        const logits = try gpa.alloc(f32, c.vocab_size);
        defer gpa.free(logits);
        const residuals: ?[]f32 = if (settings.probe_residuals) try gpa.alloc(f32, (c.num_layers + 1) * c.hidden_size) else null;
        defer if (residuals) |r| gpa.free(r);
        try model_mod.prefill(model, ws, &cache, &prompts, logits, residuals);
        var top_buf: [top_k]Top = undefined;
        const top = topK(logits, &top_buf);

        if (settings.json) {
            try js.beginObject();
            try js.objectField("user");
            try js.write(user);
            try js.objectField("text");
            try js.write(text);
            try js.objectField("ids");
            try js.write(ids);
            try js.objectField("top");
            try js.beginArray();
            for (top) |t| {
                try js.beginObject();
                try js.objectField("id");
                try js.write(t.id);
                try js.objectField("token");
                try js.write(model.tokenizer.id_to_token[t.id]);
                try js.objectField("logit");
                try js.write(t.logit);
                try js.endObject();
            }
            try js.endArray();
            try js.objectField("generated_ids");
            try js.write(generated[0]);
            try js.objectField("response");
            try js.write(response);
            try js.objectField("logits");
            try js.write(logits);
            if (residuals) |r| {
                try js.objectField("residuals");
                try js.beginArray();
                var l: usize = 0;
                while (l <= c.num_layers) : (l += 1) try js.write(r[l * c.hidden_size ..][0..c.hidden_size]);
                try js.endArray();
            }
            try js.endObject();
        } else {
            try result_out.print("\nPrompt: {s}\nRendered ({d} tokens): {s}\nIds:", .{ user, ids.len, text });
            for (ids) |id| try result_out.print(" {d}", .{id});
            try result_out.writeAll("\nTop first-token logits:\n");
            for (top) |t| try result_out.print("  {d:>8}  {d:>10.4}  {s}\n", .{ t.id, t.logit, model.tokenizer.id_to_token[t.id] });
            try result_out.print("Greedy ({d} tokens): {s}\n", .{ generated[0].len, response });
            if (residuals) |r| {
                try result_out.writeAll("Residual norm per layer (last token):\n");
                var l: usize = 0;
                while (l <= c.num_layers) : (l += 1) {
                    var ss: f64 = 0;
                    for (r[l * c.hidden_size ..][0..c.hidden_size]) |v| ss += @as(f64, v) * v;
                    try result_out.print("  {d:>3}  {d:>12.4}\n", .{ l, @sqrt(ss) });
                }
            }
        }
        try result_out.flush();
    }
    if (settings.json) {
        try js.endArray();
        try js.endObject();
        try result_out.writeAll("\n");
    }
    try result_out.flush();
}

test "topK keeps the largest logits in order" {
    const logits = [_]f32{ 1, 5, 3, 9, 2, 8 };
    var buf: [3]Top = undefined;
    const top = topK(&logits, &buf);
    try std.testing.expectEqual(@as(usize, 3), top.len);
    try std.testing.expectEqual(@as(u32, 3), top[0].id);
    try std.testing.expectEqual(@as(u32, 5), top[1].id);
    try std.testing.expectEqual(@as(u32, 1), top[2].id);
}
