//! Tests for budgeted, layer-streamed execution: mapped/streamed equivalence,
//! an end-to-end calibration + search + export + reload under a memory limit,
//! feasibility refusal, time-limit / disk-full handling and a decode-speed
//! measurement (streamed decode re-reads every layer per step).

const std = @import("std");
const Io = std.Io;
const model_mod = @import("model.zig");
const tensor = @import("tensor.zig");
const stream = @import("stream.zig");
const budget_mod = @import("budget.zig");
const engine_mod = @import("engine.zig");
const config = @import("config.zig");
const abliterate = @import("abliterate.zig");
const export_mod = @import("export.zig");
const hf = @import("hf.zig");

const Model = model_mod.Model;
const fixture = "tests/fixtures/qwen2";

const Reference = struct { family: []const u8, cases: []struct { text: []const u8, ids: []u32 } };

fn loadPrompts(gpa: std.mem.Allocator, io: Io, dir: []const u8) !std.json.Parsed(Reference) {
    const path = try std.fs.path.join(gpa, &.{ dir, "reference.json" });
    defer gpa.free(path);
    const text = try Io.Dir.cwd().readFileAlloc(io, path, gpa, .unlimited);
    defer gpa.free(text);
    return std.json.parseFromSlice(Reference, gpa, text, .{ .ignore_unknown_fields = true });
}

const Run = struct {
    logits: []f32,
    residuals: []f32,
    tokens: [][]u32,

    fn deinit(self: *Run, gpa: std.mem.Allocator) void {
        gpa.free(self.logits);
        gpa.free(self.residuals);
        for (self.tokens) |t| gpa.free(t);
        gpa.free(self.tokens);
    }
};

/// Prefill logits + residuals and a short greedy generation for `prompts`.
fn runModel(gpa: std.mem.Allocator, model: *Model, prompts: []const []const u32, ws_rows: usize) !Run {
    const c = &model.config;
    var ws = try model_mod.Workspace.init(model.gpa, c, ws_rows, prompts.len);
    defer ws.deinit();
    var max_len: usize = 0;
    for (prompts) |p| max_len = @max(max_len, p.len);
    var cache = try model_mod.KvCache.initFor(model, model.gpa, prompts.len, max_len + 8);
    defer cache.deinit();
    const logits = try gpa.alloc(f32, prompts.len * c.vocab_size);
    errdefer gpa.free(logits);
    const residuals = try gpa.alloc(f32, (c.num_layers + 1) * prompts.len * c.hidden_size);
    errdefer gpa.free(residuals);
    try model_mod.prefill(model, &ws, &cache, prompts, logits, residuals);
    const toks = try model_mod.generate(model, &ws, &cache, prompts, 6);
    // generate allocates with model.gpa; move to the test allocator for uniform freeing.
    const out = try gpa.alloc([]u32, toks.len);
    for (toks, 0..) |t, i| {
        out[i] = try gpa.dupe(u32, t);
        model.gpa.free(t);
    }
    model.gpa.free(toks);
    return .{ .logits = logits, .residuals = residuals, .tokens = out };
}

fn expectSameRun(a: *const Run, b: *const Run) !void {
    try std.testing.expectEqualSlices(f32, a.logits, b.logits);
    try std.testing.expectEqualSlices(f32, a.residuals, b.residuals);
    try std.testing.expectEqual(a.tokens.len, b.tokens.len);
    for (a.tokens, b.tokens) |x, y| try std.testing.expectEqualSlices(u32, x, y);
}

const TmpPaths = struct {
    tmp: std.testing.TmpDir,
    root: []u8,
    scratch: []u8,

    fn init(gpa: std.mem.Allocator, io: Io) !TmpPaths {
        var tmp = std.testing.tmpDir(.{});
        errdefer tmp.cleanup();
        var buf: [std.fs.max_path_bytes]u8 = undefined;
        const n = try tmp.dir.realPath(io, &buf);
        const root = try gpa.dupe(u8, buf[0..n]);
        errdefer gpa.free(root);
        const scratch = try std.fs.path.join(gpa, &.{ root, "scratch" });
        return .{ .tmp = tmp, .root = root, .scratch = scratch };
    }

    fn deinit(self: *TmpPaths, gpa: std.mem.Allocator) void {
        gpa.free(self.scratch);
        gpa.free(self.root);
        self.tmp.cleanup();
    }
};

test "streamed forward matches mapped forward bitwise (with chunking, prefetch and spill)" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    const pool = tensor.Pool.init(io, 2);
    var paths = try TmpPaths.init(gpa, io);
    defer paths.deinit(gpa);

    inline for (.{ "qwen2", "gemma3", "qwen3_next", "kimi_linear", "qwen2_fp8", "qwen2_int4", "gpt_oss_mxfp4", "lfm2", "gemma3n", "gemma4" }) |family| {
        const dir = "tests/fixtures/" ++ family;
        const ref = try loadPrompts(gpa, io, dir);
        defer ref.deinit();
        var prompts = std.ArrayList([]const u32).empty;
        defer prompts.deinit(gpa);
        for (ref.value.cases) |case| try prompts.append(gpa, case.ids);

        const mapped = try Model.load(gpa, io, &pool, dir);
        defer mapped.deinit();
        var a = try runModel(gpa, mapped, prompts.items, 64);
        defer a.deinit(gpa);

        // Streamed with prefetch, workspace large enough for a single chunk.
        const streamed = try Model.loadWithOptions(gpa, io, &pool, dir, .{ .store = .streamed, .scratch_dir = paths.scratch });
        defer streamed.deinit();
        try std.testing.expect(streamed.streamed());
        var b = try runModel(gpa, streamed, prompts.items, 64);
        defer b.deinit(gpa);
        try expectSameRun(&a, &b);
        try std.testing.expect(streamed.store.weight_bytes_read.load(.monotonic) > 0);

        // Streamed, tiny workspace (3 rows -> chunked per layer), activations in RAM.
        var c = try runModel(gpa, streamed, prompts.items, 3);
        defer c.deinit(gpa);
        try expectSameRun(&a, &c);

        // Streamed, everything spilled to scratch (activations + KV cache), no prefetch.
        const spilled = try Model.loadWithOptions(gpa, io, &pool, dir, .{ .store = .streamed, .scratch_dir = paths.scratch, .spill_always = true, .prefetch = false });
        defer spilled.deinit();
        var d = try runModel(gpa, spilled, prompts.items, 2);
        defer d.deinit(gpa);
        try expectSameRun(&a, &d);
    }
}

fn makePrompts(gpa: std.mem.Allocator, texts: []const []const u8) ![]hf.Prompt {
    const out = try gpa.alloc(hf.Prompt, texts.len);
    for (texts, 0..) |t, i| out[i] = .{ .system = "", .user = t };
    return out;
}

test "calibration, two trials, export and reload stay under the memory budget" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    const pool = tensor.Pool.init(io, 2);
    var paths = try TmpPaths.init(gpa, io);
    defer paths.deinit(gpa);

    // Pick the limit from the fixture: below all weights, above one layer.
    const probe = try Model.loadWithOptions(gpa, io, &pool, fixture, .{ .store = .streamed });
    const est0 = budget_mod.estimate(probe, .{ .batch_size = 1, .max_prompt_tokens = 16, .max_response_length = 4, .threads = 2 });
    probe.deinit();
    const limit: u64 = @min(est0.total_weight_bytes - 1, 3 * est0.largest_layer_bytes + est0.largest_tensor_bytes / 2);
    try std.testing.expect(limit < est0.total_weight_bytes);
    try std.testing.expect(limit > est0.largest_layer_bytes);

    var budget = try budget_mod.Budget.init(gpa, io, .{ .max_ram = limit, .headroom = 0, .scratch_dir = paths.scratch });
    defer budget.deinit();
    const model = try Model.loadWithOptions(gpa, io, &pool, fixture, .{ .store = .streamed, .budget = &budget });
    defer model.deinit();

    var sink: Io.Writer.Allocating = .init(gpa);
    defer sink.deinit();
    const est = budget_mod.estimate(model, .{ .batch_size = 1, .max_prompt_tokens = 16, .max_response_length = 4, .threads = 2 });
    try est.print(&sink.writer);
    try budget_mod.check(est, &budget, &sink.writer);

    var settings = config.Settings{ .batch_size = 1, .max_response_length = 4, .threads = 2, .max_ram = limit };
    var engine = engine_mod.Engine.init(budget.allocator(), model, &settings, .raw);
    defer engine.deinit();
    const good = try makePrompts(gpa, &.{ "hello world", "the cat sat", "one two three four five" });
    defer gpa.free(good);
    const bad = try makePrompts(gpa, &.{ "goodbye moon", "a dog ran", "six seven eight nine" });
    defer gpa.free(bad);

    // Calibration.
    const good_mean = try engine.getResidualMean(budget.allocator(), good, null);
    defer budget.allocator().free(good_mean);
    const bad_mean = try engine.getResidualMean(budget.allocator(), bad, null);
    defer budget.allocator().free(bad_mean);
    const c = &model.config;
    const dirs = try abliterate.computeDirections(budget.allocator(), good_mean, bad_mean, c.num_layers + 1, c.hidden_size, true);
    defer budget.allocator().free(dirs);

    // Two trials: apply (resetting the previous deltas), generate, logits.
    var params = std.EnumMap(model_mod.Component, abliterate.Params){};
    params.put(.attn_o_proj, .{ .max_weight = 1.0, .max_weight_position = 1, .min_weight = 0.5, .min_weight_distance = 2 });
    params.put(.mlp_down_proj, .{ .max_weight = 0.8, .max_weight_position = 1, .min_weight = 0.2, .min_weight_distance = 2 });
    var trial: usize = 0;
    while (trial < 2) : (trial += 1) {
        try abliterate.apply(model, dirs, if (trial == 0) null else 0.5, params, .{ .row_normalization = .full, .lora_rank = 3, .seed = trial });
        try std.testing.expect(model.getDelta(1, .attn_o_proj) != null);
        const responses = try engine.getResponses(budget.allocator(), bad, true);
        defer {
            for (responses) |r| budget.allocator().free(r);
            budget.allocator().free(responses);
        }
        const logits = try engine.getLogits(budget.allocator(), good);
        defer budget.allocator().free(logits);
        try std.testing.expect(budget.alloc.peakBytes() <= budget.limitBytes());
        try budget.report().print(&sink.writer, "trial");
    }
    engine.deinit();
    engine.ws = null;

    // Export under budget, then reload through the streamed path and compare logits.
    const out_dir = try std.fs.path.join(gpa, &.{ paths.root, "exported" });
    defer gpa.free(out_dir);
    try export_mod.saveModel(budget.allocator(), io, model, out_dir, .{ .max_shard_size = 16384 }, &sink.writer);
    var ids = std.ArrayList([]const u32).empty;
    defer {
        for (ids.items) |x| gpa.free(x);
        ids.deinit(gpa);
    }
    for (good) |p| try ids.append(gpa, try model.tokenizer.encode(gpa, p.user, true));
    const v = try stream.validateExport(gpa, io, &pool, model, out_dir, ids.items, &budget);
    try std.testing.expectEqual(ids.items.len, v.prompts);
    // Merged weights are re-quantised to bf16, so logits differ slightly but the argmax must not.
    try std.testing.expect(v.max_abs_diff > 0 and v.max_abs_diff < 0.1);
    try std.testing.expectEqual(@as(f32, 1.0), v.argmax_match);
    // Without deltas the export is byte-identical and the reload bit-exact.
    model.resetDeltas();
    const out_dir0 = try std.fs.path.join(gpa, &.{ paths.root, "exported0" });
    defer gpa.free(out_dir0);
    try export_mod.saveModel(budget.allocator(), io, model, out_dir0, .{}, &sink.writer);
    const v0 = try stream.validateExport(gpa, io, &pool, model, out_dir0, ids.items, &budget);
    try std.testing.expectEqual(@as(f32, 0), v0.max_abs_diff);

    const r = budget.report();
    try std.testing.expect(r.peak <= r.limit);
    try std.testing.expectEqual(@as(u64, 0), r.refused);
    try std.testing.expect(r.weight_bytes_read > est.total_weight_bytes);
    try r.print(&sink.writer, "final");
    try std.testing.expect(std.mem.indexOf(u8, sink.written(), "Memory estimate") != null);
    std.debug.print("\n[budgeted qwen2 fixture] weights {d} B, limit {d} B, budgeted peak {d} B, RSS peak {d} B, weights read {d} B, scratch written {d} B / read {d} B, pressure {d}\n", .{
        est.total_weight_bytes, r.limit, r.peak, r.rss.peak, r.weight_bytes_read, r.scratch_written, r.scratch_read, r.pressure,
    });
}

test "feasibility check refuses a budget below one layer" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    const pool = tensor.Pool.init(io, 2);
    var budget = try budget_mod.Budget.init(gpa, io, .{ .max_ram = 8 * 1024, .headroom = 0 });
    defer budget.deinit();
    const model = try Model.loadWithOptions(gpa, io, &pool, fixture, .{ .store = .streamed, .budget = &budget });
    defer model.deinit();
    const est = budget_mod.estimate(model, .{ .batch_size = 4, .max_prompt_tokens = 32, .max_response_length = 16, .threads = 2 });
    try std.testing.expect(est.largest_layer_bytes > 8 * 1024);
    try std.testing.expect(est.total_weight_bytes > est.largest_layer_bytes);
    try std.testing.expect(est.largest_tensor_bytes >= est.largest_layer_bytes / 7);
    var sink: Io.Writer.Allocating = .init(gpa);
    defer sink.deinit();
    try std.testing.expectError(error.BudgetTooSmall, budget_mod.check(est, &budget, &sink.writer));
    try std.testing.expect(std.mem.indexOf(u8, sink.written(), "Memory budget too small") != null);
    try std.testing.expect(std.mem.indexOf(u8, sink.written(), "does not fit") != null);
    // A generous budget passes silently.
    var big = try budget_mod.Budget.init(gpa, io, .{ .max_ram = 64 << 20 });
    defer big.deinit();
    sink.clearRetainingCapacity();
    try budget_mod.check(est, &big, &sink.writer);
    try std.testing.expectEqual(@as(usize, 0), sink.written().len);
}

test "time limit and disk-full stop cleanly and leave the source untouched" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    const pool = tensor.Pool.init(io, 2);
    var paths = try TmpPaths.init(gpa, io);
    defer paths.deinit(gpa);
    const ref = try loadPrompts(gpa, io, fixture);
    defer ref.deinit();
    const prompts = [_][]const u32{ref.value.cases[0].ids};

    const src_shard = fixture ++ "/model-00001-of-00002.safetensors";
    const before = try Io.Dir.cwd().readFileAlloc(io, src_shard, gpa, .unlimited);
    defer gpa.free(before);

    // Time limit already expired: forward and export fail with TimeLimitExceeded.
    var expired = try budget_mod.Budget.init(gpa, io, .{ .time_limit = Io.Duration.fromNanoseconds(0), .scratch_dir = paths.scratch });
    defer expired.deinit();
    const model = try Model.loadWithOptions(gpa, io, &pool, fixture, .{ .store = .streamed, .budget = &expired });
    defer model.deinit();
    try std.testing.expectError(error.TimeLimitExceeded, runModel(gpa, model, &prompts, 16));
    const out_dir = try std.fs.path.join(gpa, &.{ paths.root, "partial" });
    defer gpa.free(out_dir);
    var sink: Io.Writer.Allocating = .init(gpa);
    defer sink.deinit();
    try std.testing.expectError(error.TimeLimitExceeded, export_mod.saveModel(gpa, io, model, out_dir, .{}, &sink.writer));
    // The output is marked incomplete and refused on load.
    var od = try Io.Dir.cwd().openDir(io, out_dir, .{});
    defer od.close(io);
    try od.access(io, model_mod.export_incomplete_marker, .{});
    try std.testing.expectError(error.IncompleteModel, Model.load(gpa, io, &pool, out_dir));

    // "Disk full" / unwritable scratch: a regular file where the scratch directory should be.
    try paths.tmp.dir.writeFile(io, .{ .sub_path = "notadir", .data = "x" });
    const bad_scratch = try std.fs.path.join(gpa, &.{ paths.root, "notadir", "scratch" });
    defer gpa.free(bad_scratch);
    const spill = try Model.loadWithOptions(gpa, io, &pool, fixture, .{ .store = .streamed, .scratch_dir = bad_scratch, .spill_always = true });
    defer spill.deinit();
    try std.testing.expectError(error.NotDir, runModel(gpa, spill, &prompts, 16));

    // The source model is byte-identical.
    const after = try Io.Dir.cwd().readFileAlloc(io, src_shard, gpa, .unlimited);
    defer gpa.free(after);
    try std.testing.expectEqualSlices(u8, before, after);
}

test "decode throughput: streamed vs mapped (measurement)" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    const pool = tensor.Pool.init(io, 2);
    var paths = try TmpPaths.init(gpa, io);
    defer paths.deinit(gpa);
    const ref = try loadPrompts(gpa, io, fixture);
    defer ref.deinit();
    var prompts = std.ArrayList([]const u32).empty;
    defer prompts.deinit(gpa);
    for (ref.value.cases) |case| try prompts.append(gpa, case.ids);
    const new_tokens: usize = 32;

    const Result = struct { tok_per_s: f64, bytes_read: u64, peak: u64, per_step_weights: u64 };
    const Bench = struct {
        fn run(a: std.mem.Allocator, iox: Io, p: *const tensor.Pool, dir: []const u8, ps: []const []const u32, n_new: usize, streamed: bool, scratch: []const u8) !Result {
            var b = try budget_mod.Budget.init(a, iox, .{ .scratch_dir = scratch });
            defer b.deinit();
            const m = try Model.loadWithOptions(a, iox, p, dir, .{ .store = if (streamed) .streamed else .mapped, .budget = &b });
            defer m.deinit();
            const c = &m.config;
            var ws = try model_mod.Workspace.init(m.gpa, c, 64, ps.len);
            defer ws.deinit();
            var max_len: usize = 0;
            for (ps) |x| max_len = @max(max_len, x.len);
            var cache = try model_mod.KvCache.initFor(m, m.gpa, ps.len, max_len + n_new + 1);
            defer cache.deinit();
            // Do not count prefill: prefill first, then time the decode loop alone.
            const logits = try a.alloc(f32, ps.len * c.vocab_size);
            defer a.free(logits);
            try model_mod.prefill(m, &ws, &cache, ps, logits, null);
            const read0 = m.store.weight_bytes_read.load(.monotonic);
            const t0 = Io.Clock.awake.now(iox);
            var toks = try a.alloc(u32, ps.len);
            defer a.free(toks);
            var rows = try a.alloc(model_mod.Row, ps.len);
            defer a.free(rows);
            var lr = try a.alloc(usize, ps.len);
            defer a.free(lr);
            for (0..ps.len) |i| {
                toks[i] = 1;
                rows[i] = .{ .b = i, .pos = ps[i].len };
                lr[i] = i;
            }
            var step: usize = 0;
            while (step < n_new) : (step += 1) {
                try model_mod.forward(m, &ws, &cache, toks, rows, .{ .logit_rows = lr });
                for (0..ps.len) |i| rows[i].pos += 1;
            }
            const dt = t0.durationTo(Io.Clock.awake.now(iox));
            const secs = @as(f64, @floatFromInt(dt.nanoseconds)) / 1e9;
            var per_step: u64 = m.lm_head_ref.byteLen();
            for (m.layers) |*l| per_step += l.refs.bytes();
            return .{
                .tok_per_s = @as(f64, @floatFromInt(n_new * ps.len)) / @max(secs, 1e-9),
                .bytes_read = m.store.weight_bytes_read.load(.monotonic) - read0,
                .peak = b.alloc.peakBytes(),
                .per_step_weights = per_step,
            };
        }
    };
    const mapped = try Bench.run(gpa, io, &pool, fixture, prompts.items, new_tokens, false, paths.scratch);
    const streamed = try Bench.run(gpa, io, &pool, fixture, prompts.items, new_tokens, true, paths.scratch);
    std.debug.print("\n[decode qwen2 fixture, {d} prompts x {d} tokens] mapped: {d:.0} tok/s, budgeted peak {d} B; streamed: {d:.0} tok/s, budgeted peak {d} B, weights read {d} B ({d} B/step)\n", .{
        prompts.items.len, new_tokens, mapped.tok_per_s, mapped.peak, streamed.tok_per_s, streamed.peak, streamed.bytes_read, streamed.bytes_read / new_tokens,
    });
    // Every decode step re-reads every layer plus the LM head (plus the embedding rows).
    try std.testing.expect(streamed.bytes_read >= new_tokens * streamed.per_step_weights);
    try std.testing.expect(streamed.bytes_read < new_tokens * (streamed.per_step_weights + 4096));
    try std.testing.expectEqual(@as(u64, 0), mapped.bytes_read);
}
