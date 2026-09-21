//! `ditch bench <model>`: measures inference throughput, the cost of the
//! abliteration building blocks and memory use, and prints a Markdown table.
//! The same model loading, prompt datasets and scorer configuration as a
//! normal run are used, so the numbers describe what a study would cost.

const std = @import("std");
const Io = std.Io;
const Allocator = std.mem.Allocator;

const config = @import("config.zig");
const tensor = @import("tensor.zig");
const model_mod = @import("model.zig");
const engine_mod = @import("engine.zig");
const chat = @import("chat.zig");
const hf = @import("hf.zig");
const abliterate = @import("abliterate.zig");
const search = @import("search.zig");
const scorers = @import("scorers.zig");

const Model = model_mod.Model;
const Engine = engine_mod.Engine;
const Prompt = hf.Prompt;

pub const Result = struct {
    model: []const u8,
    architecture: []const u8,
    num_layers: usize,
    dtype: []const u8,
    threads: usize,
    batch_size: usize,
    bench_prompts: usize,
    bench_tokens: usize,
    prefill_tokens: usize,
    prefill_seconds: f64,
    decode_tokens: usize,
    decode_seconds: f64,
    residual_prompts: usize,
    residual_seconds: f64,
    /// Apply time per row normalisation mode, in `abliterate.RowNormalization` order.
    apply_seconds: [3]f64,
    refusal_prompts: usize,
    kl_prompts: usize,
    trial_seconds: f64,
    peak_rss_bytes: ?u64,
    weight_bytes: u64,

    pub fn prefillTokensPerSecond(self: *const Result) f64 {
        return @as(f64, @floatFromInt(self.prefill_tokens)) / @max(self.prefill_seconds, 1e-9);
    }

    pub fn decodeTokensPerSecond(self: *const Result) f64 {
        return @as(f64, @floatFromInt(self.decode_tokens)) / @max(self.decode_seconds, 1e-9);
    }
};

fn secondsSince(io: Io, start: Io.Timestamp) f64 {
    const now = Io.Timestamp.now(io, .awake);
    return @as(f64, @floatFromInt(start.durationTo(now).nanoseconds)) / 1e9;
}

fn argmax(x: []const f32) u32 {
    var best: usize = 0;
    for (x, 0..) |v, i| if (v > x[best]) {
        best = i;
    };
    return @intCast(best);
}

/// Peak resident set size of this process from /proc/self/status (Linux only).
pub fn peakRss(io: Io) ?u64 {
    // /proc files report a size of 0, so the file is streamed into a fixed buffer.
    const file = Io.Dir.cwd().openFile(io, "/proc/self/status", .{}) catch return null;
    defer file.close(io);
    var buf: [4096]u8 = undefined;
    var fr = file.reader(io, &buf);
    var text: [1 << 16]u8 = undefined;
    var len: usize = 0;
    while (len < text.len) {
        const n = fr.interface.readSliceShort(text[len..]) catch return null;
        if (n == 0) break;
        len += n;
    }
    return parseVmHwm(text[0..len]);
}

fn parseVmHwm(text: []const u8) ?u64 {
    var it = std.mem.splitScalar(u8, text, '\n');
    while (it.next()) |line| {
        if (!std.mem.startsWith(u8, line, "VmHWM:")) continue;
        const rest = std.mem.trim(u8, line["VmHWM:".len..], " \t");
        const end = std.mem.indexOfScalar(u8, rest, ' ') orelse rest.len;
        const kb = std.fmt.parseInt(u64, rest[0..end], 10) catch return null;
        return kb * 1024;
    }
    return null;
}

fn formatBytes(buf: []u8, bytes: u64) []const u8 {
    const b: f64 = @floatFromInt(bytes);
    if (bytes >= 1 << 30) return std.fmt.bufPrint(buf, "{d:.2} GiB", .{b / (1 << 30)}) catch "?";
    if (bytes >= 1 << 20) return std.fmt.bufPrint(buf, "{d:.1} MiB", .{b / (1 << 20)}) catch "?";
    return std.fmt.bufPrint(buf, "{d:.1} KiB", .{b / 1024}) catch "?";
}

fn formatSeconds(buf: []u8, s: f64) []const u8 {
    if (s < 1.0) return std.fmt.bufPrint(buf, "{d:.1} ms", .{s * 1000}) catch "?";
    return std.fmt.bufPrint(buf, "{d:.2} s", .{s}) catch "?";
}

/// Writes the results as a Markdown table.
pub fn writeTable(r: *const Result, w: *Io.Writer) !void {
    var b1: [64]u8 = undefined;
    var b2: [64]u8 = undefined;
    try w.writeAll("| Metric | Value |\n| :--- | ---: |\n");
    try w.print("| Model | `{s}` ({s}, {d} layers, {s} weights) |\n", .{ r.model, r.architecture, r.num_layers, r.dtype });
    try w.print("| Threads | {d} |\n", .{r.threads});
    try w.print("| Batch size | {d} |\n", .{r.batch_size});
    try w.print("| Prefill tokens/s | {d:.1} ({d} tokens in {s}) |\n", .{ r.prefillTokensPerSecond(), r.prefill_tokens, formatSeconds(&b1, r.prefill_seconds) });
    try w.print("| Decode tokens/s | {d:.1} ({d} prompts x {d} tokens, greedy, {s}) |\n", .{ r.decodeTokensPerSecond(), r.bench_prompts, r.bench_tokens, formatSeconds(&b1, r.decode_seconds) });
    try w.print("| Residual-mean pass | {s} ({d} prompts) |\n", .{ formatSeconds(&b1, r.residual_seconds), r.residual_prompts });
    for (r.apply_seconds, 0..) |s, i| {
        const mode: abliterate.RowNormalization = @enumFromInt(i);
        try w.print("| Apply time (row_normalization = {s}) | {s} |\n", .{ @tagName(mode), formatSeconds(&b1, s) });
    }
    try w.print("| Time per trial | {s} (apply + {d} refusal prompts + {d} KL prompts) |\n", .{ formatSeconds(&b1, r.trial_seconds), r.refusal_prompts, r.kl_prompts });
    if (r.peak_rss_bytes) |p| {
        try w.print("| Peak RSS | {s} ({d} bytes) |\n", .{ formatBytes(&b1, p), p });
    } else {
        try w.writeAll("| Peak RSS | n/a (not Linux) |\n");
    }
    try w.print("| Total weight bytes | {d} ({s}) |\n", .{ r.weight_bytes, formatBytes(&b2, r.weight_bytes) });
}

fn cycle(a: Allocator, prompts: []const Prompt, n: usize) ![]Prompt {
    const out = try a.alloc(Prompt, n);
    for (out, 0..) |*p, i| p.* = prompts[i % prompts.len];
    return out;
}

/// Runs the benchmark and prints the table (also to `settings.bench_output` when set).
pub fn run(gpa: Allocator, arena: Allocator, io: Io, settings: *config.Settings, http: *hf.Http, cache_root: []const u8, pool: *const tensor.Pool, out: *Io.Writer) !void {
    try out.print("\nBenchmarking {s}...\n", .{settings.model});
    try out.flush();
    const model_dir = try hf.resolveModel(arena, http, cache_root, settings.model, settings.model_commit, out);
    const model = try Model.load(gpa, io, pool, model_dir);
    defer model.deinit();
    const c = &model.config;
    const template = if (settings.chat_template) |name| (chat.Template.parse(name) orelse return error.InvalidChatTemplate) else chat.detect(model.chat_template, c.model_type);
    var engine = Engine.init(gpa, model, settings, template);
    defer engine.deinit();
    if (settings.response_prefix == null) settings.response_prefix = "";

    const n = @max(settings.bench_prompts, 1);
    const m = @max(settings.bench_tokens, 1);
    engine.batch_size = if (settings.batch_size > 0) settings.batch_size else n;

    var result = Result{
        .model = settings.model,
        .architecture = c.model_type,
        .num_layers = c.num_layers,
        .dtype = model.dtype.safetensorsName(),
        .threads = pool.threads,
        .batch_size = engine.batch_size,
        .bench_prompts = n,
        .bench_tokens = m,
        .prefill_tokens = 0,
        .prefill_seconds = 0,
        .decode_tokens = 0,
        .decode_seconds = 0,
        .residual_prompts = 0,
        .residual_seconds = 0,
        .apply_seconds = .{ 0, 0, 0 },
        .refusal_prompts = 0,
        .kl_prompts = 0,
        .trial_seconds = 0,
        .peak_rss_bytes = null,
        .weight_bytes = 0,
    };
    for (model.files) |f| {
        var it = f.tensors.iterator();
        while (it.next()) |kv| result.weight_bytes += kv.value_ptr.data.len;
    }

    // Prompts.
    try out.print("\nLoading good prompts from {s}...\n", .{settings.good_prompts.dataset});
    try out.flush();
    const good_prompts = try hf.loadPrompts(arena, http, cache_root, settings, settings.good_prompts, out);
    try out.print("\nLoading bad prompts from {s}...\n", .{settings.bad_prompts.dataset});
    try out.flush();
    const bad_prompts = try hf.loadPrompts(arena, http, cache_root, settings, settings.bad_prompts, out);
    if (good_prompts.len == 0 or bad_prompts.len == 0) return error.NoPrompts;

    // Prefill and decode throughput on a batch of n prompts.
    try out.print("\nMeasuring prefill and decode ({d} prompts, {d} tokens)...\n", .{ n, m });
    try out.flush();
    {
        const batch = try cycle(arena, good_prompts, n);
        const ids = try gpa.alloc([]u32, n);
        defer {
            for (ids) |x| gpa.free(x);
            gpa.free(ids);
        }
        var total: usize = 0;
        var longest: usize = 0;
        for (batch, 0..) |p, i| {
            ids[i] = try engine.encodePrompt(gpa, p);
            total += ids[i].len;
            longest = @max(longest, ids[i].len);
        }
        var ws = try model_mod.Workspace.init(gpa, c, @max(total, 1), n);
        defer ws.deinit();
        var cache = try model_mod.KvCache.init(gpa, c.num_layers, n, longest + m + 1, c.num_kv_heads * c.head_dim);
        defer cache.deinit();
        const logits = try gpa.alloc(f32, n * c.vocab_size);
        defer gpa.free(logits);
        // Warm-up (allocations, page faults on the mapped weights), then the timed runs.
        try model_mod.prefill(model, &ws, &cache, ids, logits, null);
        var start = Io.Timestamp.now(io, .awake);
        try model_mod.prefill(model, &ws, &cache, ids, logits, null);
        result.prefill_seconds = secondsSince(io, start);
        result.prefill_tokens = total;

        const tokens = try gpa.alloc(u32, n);
        defer gpa.free(tokens);
        const rows = try gpa.alloc(model_mod.Row, n);
        defer gpa.free(rows);
        const logit_rows = try gpa.alloc(usize, n);
        defer gpa.free(logit_rows);
        for (0..n) |b| {
            tokens[b] = argmax(logits[b * c.vocab_size ..][0..c.vocab_size]);
            rows[b] = .{ .b = b, .pos = ids[b].len };
            logit_rows[b] = b;
        }
        start = Io.Timestamp.now(io, .awake);
        for (0..m) |_| {
            try model_mod.forward(model, &ws, &cache, tokens, rows, .{ .logit_rows = logit_rows });
            for (0..n) |b| {
                tokens[b] = argmax(ws.logits[b * c.vocab_size ..][0..c.vocab_size]);
                rows[b].pos += 1;
            }
        }
        result.decode_seconds = secondsSince(io, start);
        result.decode_tokens = n * m;
    }

    // Residual means and directions.
    try out.print("\nMeasuring the residual-mean pass ({d} prompts)...\n", .{good_prompts.len});
    try out.flush();
    var start = Io.Timestamp.now(io, .awake);
    const good_means = try engine.getResidualMean(gpa, good_prompts, null);
    defer gpa.free(good_means);
    result.residual_seconds = secondsSince(io, start);
    result.residual_prompts = good_prompts.len;
    const bad_means = try engine.getResidualMean(gpa, bad_prompts, null);
    defer gpa.free(bad_means);
    const dirs = try abliterate.computeDirections(gpa, good_means, bad_means, c.num_layers + 1, c.hidden_size, settings.orthogonalize_direction);
    defer gpa.free(dirs);

    // Apply time per row normalisation mode, at the centre of the search space.
    try out.writeAll("\nMeasuring abliteration apply time...\n");
    try out.flush();
    var space = try search.buildSpace(gpa, model);
    defer space.deinit();
    const vector = try arena.alloc(f64, space.dims());
    for (space.space.specs, 0..) |spec, i| vector[i] = switch (spec) {
        .categorical => 0,
        .float => |f| (f.low + f.high) / 2,
    };
    const cfg = search.decode(&space, vector);
    for ([_]abliterate.RowNormalization{ .none, .pre, .full }) |mode| {
        model.resetDeltas();
        start = Io.Timestamp.now(io, .awake);
        try search.applyTrial(model, dirs, cfg, .{
            .row_normalization = mode,
            .lora_rank = settings.full_normalization_lora_rank,
            .seed = settings.seed orelse 0,
            .expert_selection = settings.expert_selection,
        });
        result.apply_seconds[@intFromEnum(mode)] = secondsSince(io, start);
    }
    model.resetDeltas();

    // A full trial: apply with the configured mode, then score.
    var evaluator = try scorers.Evaluator.init(gpa, arena, &engine, settings, http, cache_root, out);
    defer evaluator.deinit();
    for (evaluator.entries) |e| switch (e.scorer) {
        .keyword_rate => |k| result.refusal_prompts += k.prompts.len,
        .kl_divergence => |k| result.kl_prompts += k.prompts.len,
    };
    try out.writeAll("\nMeasuring one full trial...\n");
    try out.flush();
    start = Io.Timestamp.now(io, .awake);
    try search.applyTrial(model, dirs, cfg, .{
        .row_normalization = settings.row_normalization,
        .lora_rank = settings.full_normalization_lora_rank,
        .seed = settings.seed orelse 0,
        .expert_selection = settings.expert_selection,
    });
    const scores = try evaluator.scores(arena, &engine, out);
    result.trial_seconds = secondsSince(io, start);
    for (scores) |s| try out.print("  * {s}: {s}\n", .{ s.name, s.score.display });
    model.resetDeltas();

    result.peak_rss_bytes = peakRss(io);

    try out.writeAll("\n");
    try writeTable(&result, out);
    try out.flush();
    if (settings.bench_output) |path| {
        var buf: Io.Writer.Allocating = .init(gpa);
        defer buf.deinit();
        try buf.writer.print("# ditch bench: {s}\n\n", .{settings.model});
        try writeTable(&result, &buf.writer);
        try Io.Dir.cwd().writeFile(io, .{ .sub_path = path, .data = buf.written() });
        try out.print("\nBenchmark table written to {s}.\n", .{path});
    }
}

test "VmHWM parsing" {
    try std.testing.expectEqual(@as(?u64, 12345 * 1024), parseVmHwm("Name:\tditch\nVmPeak:\t  99 kB\nVmHWM:\t   12345 kB\nVmRSS:\t 1 kB\n"));
    try std.testing.expectEqual(@as(?u64, null), parseVmHwm("Name:\tditch\n"));
}

test "benchmark table rows" {
    const gpa = std.testing.allocator;
    const r = Result{
        .model = "m",
        .architecture = "qwen2",
        .num_layers = 3,
        .dtype = "BF16",
        .threads = 4,
        .batch_size = 4,
        .bench_prompts = 4,
        .bench_tokens = 4,
        .prefill_tokens = 100,
        .prefill_seconds = 0.5,
        .decode_tokens = 16,
        .decode_seconds = 0.25,
        .residual_prompts = 6,
        .residual_seconds = 0.1,
        .apply_seconds = .{ 0.001, 0.002, 0.003 },
        .refusal_prompts = 6,
        .kl_prompts = 6,
        .trial_seconds = 2.5,
        .peak_rss_bytes = 3 << 20,
        .weight_bytes = 64256,
    };
    var sink: Io.Writer.Allocating = .init(gpa);
    defer sink.deinit();
    try writeTable(&r, &sink.writer);
    const text = sink.written();
    try std.testing.expect(std.mem.indexOf(u8, text, "| Prefill tokens/s | 200.0") != null);
    try std.testing.expect(std.mem.indexOf(u8, text, "| Decode tokens/s | 64.0") != null);
    try std.testing.expect(std.mem.indexOf(u8, text, "row_normalization = full) | 3.0 ms") != null);
    try std.testing.expect(std.mem.indexOf(u8, text, "| Peak RSS | 3.0 MiB") != null);
    try std.testing.expect(std.mem.indexOf(u8, text, "| Total weight bytes | 64256") != null);
}
