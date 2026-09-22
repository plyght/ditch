//! `ditch bench <model>`: measures inference throughput, the cost of the
//! abliteration building blocks and memory use, and prints a Markdown table.
//! The same model loading, prompt datasets and scorer configuration as a
//! normal run are used, so the numbers describe what a study would cost.

const std = @import("std");
const builtin = @import("builtin");
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

/// Writes the results as one JSON object.
pub fn writeJson(r: *const Result, w: *Io.Writer) !void {
    var js: std.json.Stringify = .{ .writer = w };
    try js.beginObject();
    inline for (std.meta.fields(Result)) |f| {
        try js.objectField(f.name);
        try js.write(@field(r, f.name));
    }
    try js.objectField("prefill_tokens_per_second");
    try js.write(r.prefillTokensPerSecond());
    try js.objectField("decode_tokens_per_second");
    try js.write(r.decodeTokensPerSecond());
    try js.endObject();
    try w.writeAll("\n");
}

/// Writes the results as `key: value` lines (--plain).
pub fn writePlain(r: *const Result, w: *Io.Writer) !void {
    try w.print("model: {s}\narchitecture: {s}\nlayers: {d}\ndtype: {s}\nthreads: {d}\nbatch_size: {d}\n", .{ r.model, r.architecture, r.num_layers, r.dtype, r.threads, r.batch_size });
    try w.print("prefill_tokens_per_second: {d:.1}\ndecode_tokens_per_second: {d:.1}\nresidual_seconds: {d:.3}\n", .{ r.prefillTokensPerSecond(), r.decodeTokensPerSecond(), r.residual_seconds });
    for (r.apply_seconds, 0..) |s, i| try w.print("apply_seconds_{s}: {d:.4}\n", .{ @tagName(@as(abliterate.RowNormalization, @enumFromInt(i))), s });
    try w.print("trial_seconds: {d:.3}\npeak_rss_bytes: {?d}\nweight_bytes: {d}\n", .{ r.trial_seconds, r.peak_rss_bytes, r.weight_bytes });
}

/// Runs the benchmark: messages go to `out`, the table (or JSON / plain
/// lines) to `result`, and also to `settings.bench_output` when set.
pub fn run(gpa: Allocator, arena: Allocator, io: Io, settings: *config.Settings, http: *hf.Http, cache_root: []const u8, pool: *const tensor.Pool, out: *Io.Writer, result_out: *Io.Writer) !void {
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
        while (it.next()) |kv| result.weight_bytes += kv.value_ptr.byte_len;
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
        var cache = try model_mod.KvCache.initFor(model, gpa, n, longest + m + 1);
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
        .refusal_logit => |r| result.refusal_prompts += r.prompts.len,
    };
    try out.writeAll("\nMeasuring one full trial...\n");
    try out.flush();
    start = Io.Timestamp.now(io, .awake);
    try search.applyTrial(model, dirs, cfg, .{
        .row_normalization = settings.row_normalization,
        .lora_rank = settings.full_normalization_lora_rank,
        .seed = settings.seed orelse 0,
        .expert_selection = settings.expert_selection,
        .ablate_inputs = settings.ablate_inputs,
    });
    const scores = try evaluator.scores(arena, &engine, out);
    result.trial_seconds = secondsSince(io, start);
    for (scores) |s| try out.print("  * {s}: {s}\n", .{ s.name, s.score.display });
    model.resetDeltas();

    result.peak_rss_bytes = peakRss(io);

    try out.flush();
    if (settings.json) try writeJson(&result, result_out) else if (settings.plain) try writePlain(&result, result_out) else try writeTable(&result, result_out);
    try result_out.flush();
    if (settings.bench_output) |path| {
        var buf: Io.Writer.Allocating = .init(gpa);
        defer buf.deinit();
        try buf.writer.print("# ditch bench: {s}\n\n", .{settings.model});
        try writeTable(&result, &buf.writer);
        try Io.Dir.cwd().writeFile(io, .{ .sub_path = path, .data = buf.written() });
        try out.print("\nBenchmark table written to {s}.\n", .{path});
    }
}

// ---------------------------------------------------------------------------
// Per-kernel microbenchmark (`ditch bench --kernels`)
// ---------------------------------------------------------------------------

/// One measured kernel. `flops` counts useful floating-point operations and
/// `bytes` the weight traffic, so a memory-bound kernel can be read as GB/s
/// and a compute-bound one as GFLOP/s.
pub const KernelRow = struct {
    name: []const u8,
    shape: []const u8,
    seconds: f64,
    iterations: usize,
    flops: f64,
    bytes: f64,

    pub fn gflops(self: *const KernelRow) f64 {
        return self.flops * @as(f64, @floatFromInt(self.iterations)) / @max(self.seconds, 1e-9) / 1e9;
    }

    pub fn gbytes(self: *const KernelRow) f64 {
        return self.bytes * @as(f64, @floatFromInt(self.iterations)) / @max(self.seconds, 1e-9) / 1e9;
    }
};

/// Build and machine configuration the kernel numbers depend on.
pub const KernelSetup = struct {
    vector_lanes: usize,
    tile_inputs: usize,
    tile_rows: usize,
    threads: usize,
    cpus: usize,
    performance_cores: ?usize,
    efficiency_cores: ?usize,
    accelerate_built: bool,
    accelerate_active: bool,
    cpu_arch: []const u8,
};

pub fn kernelSetup(pool: *const tensor.Pool) KernelSetup {
    return .{
        .vector_lanes = tensor.VL,
        .tile_inputs = tensor.tile_inputs,
        .tile_rows = tensor.tile_rows,
        .threads = pool.threads,
        .cpus = std.Thread.getCpuCount() catch 1,
        .performance_cores = tensor.performanceCores(),
        .efficiency_cores = tensor.efficiencyCores(),
        .accelerate_built = tensor.have_accelerate,
        .accelerate_active = tensor.accelerateActive(),
        .cpu_arch = @tagName(builtin.cpu.arch),
    };
}

/// Deterministic pseudo-random values in [-1, 1); the numbers must not depend
/// on the machine, so the benchmark is comparable across runs.
fn fillRandom(x: []f32, seed: u64) void {
    var rng = std.Random.DefaultPrng.init(seed);
    const r = rng.random();
    for (x) |*v| v.* = r.float(f32) * 2.0 - 1.0;
}

fn fillRandomBf16(bytes: []u8, seed: u64) void {
    var rng = std.Random.DefaultPrng.init(seed);
    const r = rng.random();
    const dst = std.mem.bytesAsSlice(u16, bytes);
    for (dst) |*v| v.* = tensor.f32ToBf16(r.float(f32) * 2.0 - 1.0);
}

/// Runs `body` until at least `min_seconds` have passed (at least three
/// times), and records the per-iteration cost.
fn measure(io: Io, rows: *std.ArrayList(KernelRow), gpa: Allocator, name: []const u8, shape: []const u8, flops: f64, bytes: f64, ctx: anytype, comptime body: fn (@TypeOf(ctx)) anyerror!void) !void {
    const min_seconds = 0.25;
    try body(ctx); // warm-up: first-touch page faults and the pool's scratch
    var iterations: usize = 0;
    const start = Io.Timestamp.now(io, .awake);
    var seconds: f64 = 0;
    while (iterations < 3 or seconds < min_seconds) {
        try body(ctx);
        iterations += 1;
        seconds = secondsSince(io, start);
        if (iterations >= 1 << 20) break;
    }
    try rows.append(gpa, .{ .name = name, .shape = shape, .seconds = seconds, .iterations = iterations, .flops = flops, .bytes = bytes });
}

/// Measures every compute kernel on synthetic data of transformer-like
/// shapes: prefill matmul, decode matvec, the abliteration transpose product,
/// attention and the gated activation.
pub fn kernelRows(gpa: Allocator, io: Io, pool: *const tensor.Pool) ![]KernelRow {
    const cols: usize = 2048;
    const rows_n: usize = 2048;
    const prefill_n: usize = 64;

    var out: std.ArrayList(KernelRow) = .empty;
    errdefer out.deinit(gpa);

    const wbytes = try gpa.alloc(u8, rows_n * cols * 2);
    defer gpa.free(wbytes);
    fillRandomBf16(wbytes, 1);
    const w = tensor.Weight{ .data = wbytes, .dtype = .bf16, .rows = rows_n, .cols = cols };

    const wf32 = try gpa.alloc(f32, rows_n * cols);
    defer gpa.free(wf32);
    tensor.convertToF32(.bf16, wbytes, wf32);
    const wf = tensor.Weight{ .data = std.mem.sliceAsBytes(wf32), .dtype = .f32, .rows = rows_n, .cols = cols };

    const x = try gpa.alloc(f32, prefill_n * cols);
    defer gpa.free(x);
    fillRandom(x, 2);
    const y = try gpa.alloc(f32, prefill_n * rows_n);
    defer gpa.free(y);
    fillRandom(y, 3);

    const Mm = struct {
        gpa: Allocator,
        pool: *const tensor.Pool,
        out: []f32,
        x: []const f32,
        n: usize,
        w: tensor.Weight,
        fn call(c: *const @This()) anyerror!void {
            try tensor.matmulT(c.pool, c.gpa, c.out, c.x, c.n, c.w, null);
        }
    };
    const mm_prefill = Mm{ .gpa = gpa, .pool = pool, .out = y, .x = x, .n = prefill_n, .w = w };
    try measure(io, &out, gpa, "matmul prefill (bf16 weights)", "64 x 2048 x 2048", 2.0 * @as(f64, @floatFromInt(prefill_n * rows_n * cols)), @floatFromInt(rows_n * cols * 2), &mm_prefill, Mm.call);

    const mm_prefill_f32 = Mm{ .gpa = gpa, .pool = pool, .out = y, .x = x, .n = prefill_n, .w = wf };
    try measure(io, &out, gpa, "matmul prefill (f32 weights)", "64 x 2048 x 2048", 2.0 * @as(f64, @floatFromInt(prefill_n * rows_n * cols)), @floatFromInt(rows_n * cols * 4), &mm_prefill_f32, Mm.call);

    const mm_decode = Mm{ .gpa = gpa, .pool = pool, .out = y, .x = x, .n = 1, .w = w };
    try measure(io, &out, gpa, "matvec decode (bf16 weights)", "1 x 2048 x 2048", 2.0 * @as(f64, @floatFromInt(rows_n * cols)), @floatFromInt(rows_n * cols * 2), &mm_decode, Mm.call);

    const mm_batch4 = Mm{ .gpa = gpa, .pool = pool, .out = y, .x = x, .n = 4, .w = w };
    try measure(io, &out, gpa, "matvec decode, batch 4", "4 x 2048 x 2048", 2.0 * @as(f64, @floatFromInt(4 * rows_n * cols)), @floatFromInt(rows_n * cols * 2), &mm_batch4, Mm.call);

    const Mv = struct {
        gpa: Allocator,
        pool: *const tensor.Pool,
        out: []f32,
        w: tensor.Weight,
        y: []const f32,
        fn call(c: *const @This()) anyerror!void {
            try tensor.matvecT(c.pool, c.gpa, c.out, c.w, c.y);
        }
    };
    const mv = Mv{ .gpa = gpa, .pool = pool, .out = x, .w = w, .y = y };
    try measure(io, &out, gpa, "matvecT (abliteration apply)", "2048 x 2048", 2.0 * @as(f64, @floatFromInt(rows_n * cols)), @floatFromInt(rows_n * cols * 2), &mv, Mv.call);

    // Attention: one head of 128 dimensions over 1024 cached positions.
    const hd: usize = 128;
    const keys: usize = 1024;
    const kv = try gpa.alloc(f32, keys * hd);
    defer gpa.free(kv);
    fillRandom(kv, 4);
    const q = try gpa.alloc(f32, hd);
    defer gpa.free(q);
    fillRandom(q, 5);
    const scores = try gpa.alloc(f32, keys);
    defer gpa.free(scores);
    const acc = try gpa.alloc(f32, hd);
    defer gpa.free(acc);

    const At = struct {
        q: []const f32,
        kv: []const f32,
        scores: []f32,
        acc: []f32,
        hd: usize,
        fn scoresCall(c: *const @This()) anyerror!void {
            tensor.attentionScores(c.scores, c.q, c.kv.ptr, c.hd, 0.08838835);
            std.mem.doNotOptimizeAway(c.scores[0]);
        }
        fn valuesCall(c: *const @This()) anyerror!void {
            tensor.attentionValues(c.acc, c.scores, c.kv.ptr, c.hd);
            std.mem.doNotOptimizeAway(c.acc[0]);
        }
        fn softmaxCall(c: *const @This()) anyerror!void {
            tensor.softmaxInPlace(c.scores);
            std.mem.doNotOptimizeAway(c.scores[0]);
        }
    };
    const at = At{ .q = q, .kv = kv, .scores = scores, .acc = acc, .hd = hd };
    try measure(io, &out, gpa, "attention scores", "1024 keys x 128", 2.0 * @as(f64, @floatFromInt(keys * hd)), @floatFromInt(keys * hd * 4), &at, At.scoresCall);
    try measure(io, &out, gpa, "attention values", "1024 keys x 128", 2.0 * @as(f64, @floatFromInt(keys * hd)), @floatFromInt(keys * hd * 4), &at, At.valuesCall);
    try measure(io, &out, gpa, "softmax", "1024", @floatFromInt(keys), @floatFromInt(keys * 4), &at, At.softmaxCall);

    // Gated activation over a transformer-sized MLP block.
    const act_rows: usize = 32;
    const act_len: usize = 8192;
    const gate = try gpa.alloc(f32, act_rows * act_len);
    defer gpa.free(gate);
    fillRandom(gate, 6);
    const up = try gpa.alloc(f32, act_rows * act_len);
    defer gpa.free(up);
    fillRandom(up, 7);
    const act_out = try gpa.alloc(f32, act_rows * act_len);
    defer gpa.free(act_out);
    const Ac = struct {
        pool: *const tensor.Pool,
        out: []f32,
        gate: []const f32,
        up: []const f32,
        rows: usize,
        len: usize,
        fn call(c: *const @This()) anyerror!void {
            tensor.gatedActivation(c.pool, .silu, c.out, c.gate, c.up, c.rows, c.len, c.len, c.len);
            std.mem.doNotOptimizeAway(c.out[0]);
        }
    };
    const ac = Ac{ .pool = pool, .out = act_out, .gate = gate, .up = up, .rows = act_rows, .len = act_len };
    // A silu plus the up-projection product is about ten operations per element.
    try measure(io, &out, gpa, "gated activation (silu)", "32 x 8192", 10.0 * @as(f64, @floatFromInt(act_rows * act_len)), @floatFromInt(act_rows * act_len * 12), &ac, Ac.call);

    // Weight conversion, the path every f32 tile goes through.
    const Cv = struct {
        src: []const u8,
        dst: []f32,
        fn call(c: *const @This()) anyerror!void {
            tensor.convertToF32(.bf16, c.src, c.dst);
            std.mem.doNotOptimizeAway(c.dst[0]);
        }
    };
    const cv = Cv{ .src = wbytes, .dst = wf32 };
    try measure(io, &out, gpa, "bf16 to f32 conversion", "2048 x 2048", @floatFromInt(rows_n * cols), @floatFromInt(rows_n * cols * 6), &cv, Cv.call);

    return out.toOwnedSlice(gpa);
}

/// Writes the kernel table (Markdown, or `key: value` lines with --plain, or
/// one JSON object with --json).
pub fn writeKernels(setup: *const KernelSetup, rows: []const KernelRow, w: *Io.Writer, mode: enum { table, plain, json }) !void {
    switch (mode) {
        .json => {
            var js: std.json.Stringify = .{ .writer = w };
            try js.beginObject();
            inline for (std.meta.fields(KernelSetup)) |f| {
                try js.objectField(f.name);
                try js.write(@field(setup, f.name));
            }
            try js.objectField("kernels");
            try js.beginArray();
            for (rows) |r| {
                try js.beginObject();
                try js.objectField("name");
                try js.write(r.name);
                try js.objectField("shape");
                try js.write(r.shape);
                try js.objectField("gflops");
                try js.write(r.gflops());
                try js.objectField("gbytes_per_second");
                try js.write(r.gbytes());
                try js.objectField("iterations");
                try js.write(r.iterations);
                try js.endObject();
            }
            try js.endArray();
            try js.endObject();
            try w.writeAll("\n");
        },
        .plain => {
            try w.print("cpu_arch: {s}\nvector_lanes: {d}\ntile: {d}x{d}\nthreads: {d}\ncpus: {d}\nperformance_cores: {?d}\nefficiency_cores: {?d}\naccelerate_built: {}\naccelerate_active: {}\n", .{
                setup.cpu_arch, setup.vector_lanes,      setup.tile_inputs,      setup.tile_rows,        setup.threads,
                setup.cpus,     setup.performance_cores, setup.efficiency_cores, setup.accelerate_built, setup.accelerate_active,
            });
            for (rows) |r| try w.print("kernel {s} ({s}): {d:.2} GFLOP/s, {d:.2} GB/s\n", .{ r.name, r.shape, r.gflops(), r.gbytes() });
        },
        .table => {
            try w.print("| Kernel | Shape | GFLOP/s | GB/s |\n| :--- | :--- | ---: | ---: |\n", .{});
            for (rows) |r| try w.print("| {s} | {s} | {d:.1} | {d:.1} |\n", .{ r.name, r.shape, r.gflops(), r.gbytes() });
            try w.print("\n{d} f32 lanes per vector, {d}x{d} register tile, {d} threads of {d} CPUs", .{ setup.vector_lanes, setup.tile_inputs, setup.tile_rows, setup.threads, setup.cpus });
            if (setup.performance_cores) |p| try w.print(", {d} performance and {?d} efficiency cores", .{ p, setup.efficiency_cores });
            if (setup.accelerate_built) {
                try w.print(", Accelerate {s}", .{if (setup.accelerate_active) "active" else "built in but off"});
            }
            try w.print(" ({s}).\n", .{setup.cpu_arch});
        },
    }
}

/// Runs one prefill-shaped matrix product through both the Accelerate path
/// and the Zig kernel and returns the largest difference between them,
/// relative to the largest output magnitude. Null when Accelerate is not
/// active. The two sum in a different order, so they are close but not
/// bit-identical; anything above `accelerate_tolerance` means the BLAS call
/// is being handed the wrong shape, which must fail the build, not print.
pub fn accelerateDifference(gpa: Allocator, pool: *const tensor.Pool) !?f64 {
    if (!tensor.accelerateActive()) return null;
    const cols: usize = 2048;
    const rows: usize = 512;
    const n: usize = 16;
    const wbytes = try gpa.alloc(u8, rows * cols * 2);
    defer gpa.free(wbytes);
    fillRandomBf16(wbytes, 11);
    const w = tensor.Weight{ .data = wbytes, .dtype = .bf16, .rows = rows, .cols = cols };
    const x = try gpa.alloc(f32, n * cols);
    defer gpa.free(x);
    fillRandom(x, 12);
    const a = try gpa.alloc(f32, n * rows);
    defer gpa.free(a);
    const b = try gpa.alloc(f32, n * rows);
    defer gpa.free(b);

    try tensor.matmulT(pool, gpa, a, x, n, w, null);
    tensor.accelerate_enabled = false;
    defer tensor.accelerate_enabled = true;
    try tensor.matmulT(pool, gpa, b, x, n, w, null);

    var worst: f64 = 0;
    var scale: f64 = 1e-30;
    for (a, b) |va, vb| {
        worst = @max(worst, @abs(@as(f64, va) - @as(f64, vb)));
        scale = @max(scale, @abs(@as(f64, va)));
    }
    return worst / scale;
}

/// Largest relative difference tolerated between Accelerate and the Zig
/// kernel: the two differ only in summation order over 2048 terms.
pub const accelerate_tolerance: f64 = 1e-4;

/// `ditch bench --kernels`: per-kernel throughput, no model needed.
pub fn runKernels(gpa: Allocator, io: Io, settings: *const config.Settings, pool: *const tensor.Pool, out: *Io.Writer, result_out: *Io.Writer) !void {
    try out.writeAll("\nMeasuring the compute kernels...\n");
    try out.flush();
    const rows = try kernelRows(gpa, io, pool);
    defer gpa.free(rows);
    const setup = kernelSetup(pool);
    // With Accelerate active, the same kernels are measured again with it
    // switched off, and the two paths are compared numerically.
    var zig_rows: []KernelRow = &.{};
    defer if (zig_rows.len > 0) gpa.free(zig_rows);
    var difference: ?f64 = null;
    if (tensor.accelerateActive()) {
        difference = try accelerateDifference(gpa, pool);
        tensor.accelerate_enabled = false;
        zig_rows = try kernelRows(gpa, io, pool);
        tensor.accelerate_enabled = true;
    }
    if (settings.json) {
        try writeKernels(&setup, rows, result_out, .json);
    } else if (settings.plain) {
        try writeKernels(&setup, rows, result_out, .plain);
    } else {
        try writeKernels(&setup, rows, result_out, .table);
    }
    try result_out.flush();
    if (zig_rows.len > 0) {
        var off = setup;
        off.accelerate_active = false;
        try result_out.writeAll("\nThe same kernels with Accelerate switched off:\n\n");
        if (settings.plain) try writeKernels(&off, zig_rows, result_out, .plain) else try writeKernels(&off, zig_rows, result_out, .table);
        try result_out.flush();
    }
    if (difference) |d| {
        try out.print("\nAccelerate vs the built-in kernel: largest relative difference {e:.3} (tolerance {e:.3}).\n", .{ d, accelerate_tolerance });
        try out.flush();
        if (!(d <= accelerate_tolerance)) return error.AccelerateMismatch;
    }
    if (settings.bench_output) |path| {
        var buf: Io.Writer.Allocating = .init(gpa);
        defer buf.deinit();
        try buf.writer.writeAll("# ditch bench --kernels\n\n");
        try writeKernels(&setup, rows, &buf.writer, .table);
        try Io.Dir.cwd().writeFile(io, .{ .sub_path = path, .data = buf.written() });
        try out.print("\nKernel table written to {s}.\n", .{path});
        try out.flush();
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
