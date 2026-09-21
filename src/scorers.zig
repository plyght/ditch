//! Scorers (keyword rate, KL divergence) and the evaluator that runs them.

const std = @import("std");
const Io = std.Io;
const config = @import("config.zig");
const engine_mod = @import("engine.zig");
const hf = @import("hf.zig");
const tensor = @import("tensor.zig");

const Allocator = std.mem.Allocator;
const Engine = engine_mod.Engine;
const Prompt = hf.Prompt;

pub const Score = struct {
    value: f64,
    /// Human-readable rendering, e.g. "3/100" or "0.1234".
    display: []const u8,
    /// Set when scoring was stopped early (see `KeywordRate.score`).
    pruned: ?Pruned = null,
};

/// How far an early-stopped refusal scorer got.
pub const Pruned = struct {
    /// Prompts scored before pruning.
    done: usize,
    total: usize,
};

// ---------------------------------------------------------------------------
// Early stopping
// ---------------------------------------------------------------------------
//
// Objectives are evaluated in scorer order, with the (expensive) refusal
// scorer last, so that when refusal scoring starts every other objective of
// the trial is known. A completed trial `f` on the Pareto front then dominates
// the running trial `t` regardless of how the remaining prompts turn out as
// soon as
//
//     f.loss[j] <= t.loss[j]  for every already known objective j, and
//     f.refusals < t.refusals_so_far,
//
// because `t`'s final refusal count can only grow. Pruning at that point is
// sound: `t` can never enter the front of completed trials. The threshold is
// the smallest refusal rate among the front trials that satisfy the first
// condition (for two objectives: the front trial with the lowest refusal rate
// whose KL divergence is at most the trial's).

/// Refusal-rate threshold above which the running trial is dominated by the
/// front: `null` when no front trial qualifies. `known` holds the trial's
/// objective losses evaluated so far (objectives `0 .. k`), `k` is the index
/// of the refusal-rate objective within the loss vectors.
pub fn pruneThreshold(front: []const []const f64, known: []const f64, k: usize) ?f64 {
    var best: ?f64 = null;
    for (front) |f| {
        if (f.len <= k) continue;
        var qualifies = true;
        for (known[0..k], 0..) |v, j| qualifies = qualifies and f[j] <= v;
        if (!qualifies) continue;
        best = if (best) |b| @min(b, f[k]) else f[k];
    }
    return best;
}

/// True when `matches` refusals out of `total` prompts already exceed the threshold.
pub fn shouldPrune(matches: usize, total: usize, threshold: ?f64) bool {
    const thr = threshold orelse return false;
    return @as(f64, @floatFromInt(matches)) / @as(f64, @floatFromInt(@max(total, 1))) > thr;
}

/// The first generated token of every scored response and whether the
/// response was a keyword refusal (the training data of `RefusalLogit`).
pub const FirstTokens = struct {
    tokens: std.ArrayList(u32) = .empty,
    refused: std.ArrayList(bool) = .empty,

    pub fn deinit(self: *FirstTokens, gpa: Allocator) void {
        self.tokens.deinit(gpa);
        self.refused.deinit(gpa);
    }
};

pub const KeywordRate = struct {
    settings: *const config.KeywordRateSettings,
    prompts: []Prompt,

    /// Scores the prompts batch by batch. With a `threshold` (refusal rate),
    /// scoring stops as soon as the refusals so far exceed it; the returned
    /// value then counts every remaining prompt as a refusal (an upper bound).
    /// With `capture`, the first token of every response is recorded.
    pub fn score(self: *KeywordRate, gpa: Allocator, engine: *Engine, out: *Io.Writer, threshold: ?f64) !Score {
        return self.scoreCapturing(gpa, engine, out, threshold, null);
    }

    pub fn scoreCapturing(self: *KeywordRate, gpa: Allocator, engine: *Engine, out: *Io.Writer, threshold: ?f64, capture: ?*FirstTokens) !Score {
        const total = self.prompts.len;
        var matches: usize = 0;
        var done: usize = 0;
        while (done < total) {
            const end = @min(total, done + @max(engine.batch_size, 1));
            const batch = self.prompts[done..end];
            const ids = try gpa.alloc([]u32, batch.len);
            defer gpa.free(ids);
            var n_ids: usize = 0;
            defer for (ids[0..n_ids]) |x| gpa.free(x);
            for (batch) |p| {
                ids[n_ids] = try engine.encodePrompt(gpa, p);
                n_ids += 1;
            }
            // `generateBatch` allocates its result with the model's allocator.
            const tokens = try engine.generateBatch(gpa, ids, engine.settings.max_response_length);
            defer {
                for (tokens) |t| engine.model.gpa.free(t);
                engine.model.gpa.free(tokens);
            }
            for (batch, tokens) |p, t| {
                const r = try engine.model.tokenizer.decode(gpa, t, false);
                defer gpa.free(r);
                const m = try isMatch(gpa, r, self.settings.keyword_markers);
                if (m) matches += 1;
                if (capture) |c| {
                    try c.tokens.append(gpa, if (t.len > 0) t[0] else engine.model.pad_id);
                    try c.refused.append(gpa, m);
                }
                if (self.settings.print_responses) {
                    try out.print("\nSystem prompt: {s}\nPrompt: {s}\nResponse{s}: {s}\n", .{ p.system, p.user, if (m) " [refusal]" else "", if (std.mem.trim(u8, r, " \t\r\n").len == 0) "[empty]" else r });
                }
            }
            done = end;
            if (shouldPrune(matches, total, threshold)) {
                if (self.settings.print_responses) try out.writeAll("\n");
                try out.print("* Pruned after {d}/{d} prompts\n", .{ done, total });
                try out.flush();
                const bound = matches + (total - done);
                return .{
                    .value = @as(f64, @floatFromInt(bound)) / @as(f64, @floatFromInt(@max(total, 1))),
                    .display = try std.fmt.allocPrint(gpa, ">={d}/{d}", .{ bound, total }),
                    .pruned = .{ .done = done, .total = total },
                };
            }
        }
        if (self.settings.print_responses) try out.writeAll("\n");
        return .{
            .value = @as(f64, @floatFromInt(matches)) / @as(f64, @floatFromInt(@max(total, 1))),
            .display = try std.fmt.allocPrint(gpa, "{d}/{d}", .{ matches, total }),
        };
    }

    /// Mirrors heretic's matching: empty responses count, emphasis and
    /// typographic apostrophes are normalised, whitespace collapsed.
    pub fn isMatch(gpa: Allocator, response: []const u8, markers: []const []const u8) !bool {
        if (std.mem.trim(u8, response, " \t\r\n").len == 0) return true;
        var norm = std.ArrayList(u8).empty;
        defer norm.deinit(gpa);
        var i: usize = 0;
        var pending_space = false;
        while (i < response.len) {
            const c = response[i];
            if (c == '*') {
                i += 1;
                continue;
            }
            if (std.mem.startsWith(u8, response[i..], "\xe2\x80\x99")) { // ’
                if (pending_space and norm.items.len > 0) try norm.append(gpa, ' ');
                pending_space = false;
                try norm.append(gpa, '\'');
                i += 3;
                continue;
            }
            if (c == ' ' or c == '\t' or c == '\n' or c == '\r' or c == 0x0B or c == 0x0C) {
                pending_space = true;
                i += 1;
                continue;
            }
            if (pending_space and norm.items.len > 0) try norm.append(gpa, ' ');
            pending_space = false;
            try norm.append(gpa, std.ascii.toLower(c));
            i += 1;
        }
        for (markers) |m| {
            const lower = try std.ascii.allocLowerString(gpa, m);
            defer gpa.free(lower);
            if (std.mem.indexOf(u8, norm.items, lower) != null) return true;
        }
        return false;
    }
};

/// Probability mass of the first-token distribution on "refusal-start"
/// tokens, averaged over the refusal prompts (0..1, one prefill per trial).
///
/// The token set is learned from the base model: every prompt's baseline
/// response (the same generation that gives the baseline keyword score) is
/// classified by the keyword scorer, and a token that started at least one
/// refusal gets the weight `refusals started / responses started` (its
/// precision as a refusal signal), so "I" counts fully when the base model
/// only ever refuses with "I cannot ..." and partially when it also opens
/// helpful answers with "I". When the base model refuses nothing, common
/// refusal openers are tokenised instead, with weight 1.
///
/// It is a proxy: it measures whether the model *starts* like a refusal,
/// not whether the generated response contains a refusal keyword, and a
/// model can refuse after a helpful opener. It is therefore meant for the
/// search; the Pareto candidates are always re-scored by generation
/// (`--fast-search` does this before the results menu).
pub const RefusalLogit = struct {
    prompts: []Prompt,
    /// Refusal-start token → weight in (0, 1].
    weights: std.AutoHashMapUnmanaged(u32, f32) = .empty,
    vocab: usize,
    /// How the token set was obtained (for the log).
    source: enum { learned, openers } = .learned,
    n_refusals: usize = 0,

    pub const openers = [_][]const u8{ "I", "I'm", "I’m", "Sorry", "As", "Unfortunately", "No", "It's", "It’s", "Apolog", "Unfortunately," };

    pub fn deinit(self: *RefusalLogit, gpa: Allocator) void {
        self.weights.deinit(gpa);
    }

    /// Builds the token weights from the baseline responses' first tokens.
    pub fn learn(self: *RefusalLogit, gpa: Allocator, first: []const u32, refused: []const bool) !void {
        std.debug.assert(first.len == refused.len);
        var totals = std.AutoHashMapUnmanaged(u32, [2]usize).empty;
        defer totals.deinit(gpa);
        self.n_refusals = 0;
        for (first, refused) |t, r| {
            const e = try totals.getOrPut(gpa, t);
            if (!e.found_existing) e.value_ptr.* = .{ 0, 0 };
            e.value_ptr[0] += 1;
            if (r) {
                e.value_ptr[1] += 1;
                self.n_refusals += 1;
            }
        }
        self.weights.clearRetainingCapacity();
        var it = totals.iterator();
        while (it.next()) |e| {
            if (e.value_ptr[1] == 0) continue;
            try self.weights.put(gpa, e.key_ptr.*, @as(f32, @floatFromInt(e.value_ptr[1])) / @as(f32, @floatFromInt(e.value_ptr[0])));
        }
        self.source = .learned;
    }

    /// Fallback: the first token of every common refusal opener (with and without a leading space).
    pub fn useOpeners(self: *RefusalLogit, gpa: Allocator, engine: *Engine) !void {
        self.weights.clearRetainingCapacity();
        for (openers) |o| {
            for ([_][]const u8{ "", " " }) |sp| {
                const text = try std.fmt.allocPrint(gpa, "{s}{s}", .{ sp, o });
                defer gpa.free(text);
                const ids = try engine.model.tokenizer.encode(gpa, text, false);
                defer gpa.free(ids);
                if (ids.len > 0) try self.weights.put(gpa, ids[0], 1.0);
            }
        }
        self.source = .openers;
    }

    /// Weighted probability mass of one log-probability row.
    pub fn massOf(self: *const RefusalLogit, logp: []const f32) f64 {
        var mass: f64 = 0;
        var it = self.weights.iterator();
        while (it.next()) |e| {
            if (e.key_ptr.* < logp.len) mass += @as(f64, e.value_ptr.*) * @exp(@as(f64, logp[e.key_ptr.*]));
        }
        return @min(mass, 1.0);
    }

    pub fn score(self: *RefusalLogit, gpa: Allocator, engine: *Engine) !Score {
        const logits = try engine.getLogits(gpa, self.prompts);
        defer gpa.free(logits);
        const tmp = try gpa.alloc(f32, self.vocab);
        defer gpa.free(tmp);
        var total: f64 = 0;
        for (0..self.prompts.len) |i| {
            tensor.logSoftmax(tmp, logits[i * self.vocab ..][0..self.vocab]);
            total += self.massOf(tmp);
        }
        const value = total / @as(f64, @floatFromInt(@max(self.prompts.len, 1)));
        return .{ .value = value, .display = try std.fmt.allocPrint(gpa, "{d:.4}", .{value}) };
    }
};

/// KL divergence of the abliterated model's next-token distribution from the
/// base model's, averaged over the harmless prompts. With `tokens = T > 1`
/// the base model's greedy continuation of `T − 1` tokens is generated once
/// and every trial is scored teacher-forced at the `T` positions predicting
/// that continuation (one prefill of prompt + T − 1 tokens per prompt);
/// `T = 1` is heretic's first-token KL.
pub const KlDivergence = struct {
    prompts: []Prompt,
    tokens: usize,
    /// Tokenised prompt plus the baseline continuation, and the positions scored per sequence.
    seqs: [][]u32,
    tails: []usize,
    /// Baseline log-probabilities `[Σ tails][vocab]`.
    baseline: []f32,
    vocab: usize,

    pub fn init(gpa: Allocator, engine: *Engine, prompts: []Prompt, tokens: usize) !KlDivergence {
        const t = @max(tokens, 1);
        const vocab = engine.model.config.vocab_size;
        const seqs = try gpa.alloc([]u32, prompts.len);
        errdefer gpa.free(seqs);
        var n_seqs: usize = 0;
        errdefer for (seqs[0..n_seqs]) |s| gpa.free(s);
        const tails = try gpa.alloc(usize, prompts.len);
        errdefer gpa.free(tails);
        var start: usize = 0;
        while (start < prompts.len) {
            const end = @min(prompts.len, start + @max(engine.batch_size, 1));
            const ids = try gpa.alloc([]u32, end - start);
            defer gpa.free(ids);
            var n_ids: usize = 0;
            defer for (ids[0..n_ids]) |x| gpa.free(x);
            for (prompts[start..end]) |p| {
                ids[n_ids] = try engine.encodePrompt(gpa, p);
                n_ids += 1;
            }
            if (t == 1) {
                for (ids, 0..) |x, i| {
                    seqs[start + i] = try gpa.dupe(u32, x);
                    tails[start + i] = 1;
                    n_seqs += 1;
                }
            } else {
                const gen = try engine.generateBatch(gpa, ids, t - 1);
                defer {
                    for (gen) |g| engine.model.gpa.free(g);
                    engine.model.gpa.free(gen);
                }
                for (ids, gen, 0..) |x, g, i| {
                    // The position after an end-of-sequence token is meaningless.
                    var m = g.len;
                    if (m > 0 and engine.model.isEos(g[m - 1])) m -= 1;
                    const seq = try gpa.alloc(u32, x.len + m);
                    @memcpy(seq[0..x.len], x);
                    @memcpy(seq[x.len..], g[0..m]);
                    seqs[start + i] = seq;
                    tails[start + i] = m + 1;
                    n_seqs += 1;
                }
            }
            start = end;
        }
        const logits = try engine.getLogitsAt(gpa, seqs, tails);
        errdefer gpa.free(logits);
        const tmp = try gpa.alloc(f32, vocab);
        defer gpa.free(tmp);
        var rows: usize = 0;
        for (tails) |x| rows += x;
        for (0..rows) |i| {
            const row = logits[i * vocab ..][0..vocab];
            tensor.logSoftmax(tmp, row);
            @memcpy(row, tmp);
        }
        return .{ .prompts = prompts, .tokens = t, .seqs = seqs, .tails = tails, .baseline = logits, .vocab = vocab };
    }

    pub fn deinit(self: *KlDivergence, gpa: Allocator) void {
        for (self.seqs) |s| gpa.free(s);
        gpa.free(self.seqs);
        gpa.free(self.tails);
        gpa.free(self.baseline);
    }

    pub fn score(self: *KlDivergence, gpa: Allocator, engine: *Engine) !Score {
        const logits = try engine.getLogitsAt(gpa, self.seqs, self.tails);
        defer gpa.free(logits);
        const tmp = try gpa.alloc(f32, self.vocab);
        defer gpa.free(tmp);
        var total: f64 = 0;
        var row: usize = 0;
        for (self.tails) |tail| {
            var prompt_kl: f64 = 0;
            for (0..tail) |_| {
                tensor.logSoftmax(tmp, logits[row * self.vocab ..][0..self.vocab]);
                const base = self.baseline[row * self.vocab ..][0..self.vocab];
                var kl: f64 = 0;
                for (base, 0..) |lb, j| {
                    const p = @exp(@as(f64, lb));
                    kl += p * (@as(f64, lb) - @as(f64, tmp[j]));
                }
                prompt_kl += kl;
                row += 1;
            }
            total += prompt_kl / @as(f64, @floatFromInt(@max(tail, 1)));
        }
        const value = total / @as(f64, @floatFromInt(@max(self.prompts.len, 1)));
        return .{ .value = value, .display = try std.fmt.allocPrint(gpa, "{d:.4}", .{value}) };
    }
};

pub const Scorer = union(enum) {
    keyword_rate: KeywordRate,
    kl_divergence: KlDivergence,
    refusal_logit: RefusalLogit,
};

pub const Entry = struct {
    name: []const u8,
    optimization: config.Optimization,
    scorer: Scorer,
};

pub const NamedScore = struct {
    name: []const u8,
    score: Score,
};

pub const Evaluator = struct {
    gpa: Allocator,
    entries: []Entry,
    baseline: []NamedScore,
    /// `--fast-search`: the keyword scorer, run on the Pareto candidates only
    /// (`deferredScore`) instead of on every trial.
    deferred: ?Entry = null,
    deferred_baseline: ?NamedScore = null,

    pub fn init(gpa: Allocator, arena: Allocator, engine: *Engine, settings: *const config.Settings, http: *hf.Http, cache_root: []const u8, out: *Io.Writer) !Evaluator {
        var entries = std.ArrayList(Entry).empty;
        try out.writeAll("\nLoading and initializing scorers...\n");
        var keyword_prompts: ?[]Prompt = null;
        for (settings.scorers) |sc| {
            switch (sc.kind) {
                .keyword_rate => {
                    const kr = &settings.keyword_rate;
                    const name = if (sc.instance_name) |n| try std.fmt.allocPrint(arena, "{s} - {s}", .{ kr.score_name, n }) else kr.score_name;
                    try out.print("* Loaded: KeywordRate ({s})\n", .{name});
                    const prompts = keyword_prompts orelse try loadKeywordPrompts(arena, settings, http, cache_root, out);
                    keyword_prompts = prompts;
                    try entries.append(arena, .{ .name = name, .optimization = sc.optimization, .scorer = .{ .keyword_rate = .{ .settings = kr, .prompts = prompts } } });
                },
                .kl_divergence => {
                    const name = if (sc.instance_name) |n| try std.fmt.allocPrint(arena, "KL divergence - {s}", .{n}) else "KL divergence";
                    try out.print("* Loaded: KLDivergence ({s})\n", .{name});
                    try out.print("\nLoading KL divergence evaluation prompts from {s}...\n", .{settings.kl_divergence.prompts.dataset});
                    try out.flush();
                    const prompts = try hf.loadPrompts(arena, http, cache_root, settings, settings.kl_divergence.prompts, out);
                    try out.print("* {d} prompts loaded\n", .{prompts.len});
                    if (settings.kl_tokens > 1) {
                        try out.print("* Generating the baseline continuation ({d} tokens) and its probability distributions...\n", .{settings.kl_tokens - 1});
                    } else {
                        try out.writeAll("* Obtaining baseline first-token probability distributions...\n");
                    }
                    try out.flush();
                    const kl = try KlDivergence.init(gpa, engine, prompts, settings.kl_tokens);
                    try entries.append(arena, .{ .name = name, .optimization = sc.optimization, .scorer = .{ .kl_divergence = kl } });
                },
                .refusal_logit => {
                    const name = if (sc.instance_name) |n| try std.fmt.allocPrint(arena, "Refusal mass - {s}", .{n}) else "Refusal mass";
                    try out.print("* Loaded: RefusalLogit ({s}; first-token mass on refusal-start tokens, learned from the baseline responses)\n", .{name});
                    const prompts = keyword_prompts orelse try loadKeywordPrompts(arena, settings, http, cache_root, out);
                    keyword_prompts = prompts;
                    try entries.append(arena, .{ .name = name, .optimization = sc.optimization, .scorer = .{ .refusal_logit = .{ .prompts = prompts, .vocab = engine.model.config.vocab_size } } });
                },
            }
        }
        var self = Evaluator{ .gpa = gpa, .entries = entries.items, .baseline = &.{} };
        if (settings.fast_search) {
            const kr = &settings.keyword_rate;
            try out.print("* Deferred: KeywordRate ({s}) runs on the Pareto-optimal trials only (--fast-search)\n", .{kr.score_name});
            const prompts = keyword_prompts orelse try loadKeywordPrompts(arena, settings, http, cache_root, out);
            keyword_prompts = prompts;
            self.deferred = .{ .name = kr.score_name, .optimization = .none, .scorer = .{ .keyword_rate = .{ .settings = kr, .prompts = prompts } } };
        }
        try out.writeAll("\nGetting baseline scores...\n");
        try out.flush();
        self.baseline = try self.baselineScores(arena, engine, settings, out);
        for (self.baseline) |b| try out.print("* Baseline {s}: {s}\n", .{ b.name, b.score.display });
        if (self.deferred_baseline) |b| try out.print("* Baseline {s}: {s}\n", .{ b.name, b.score.display });
        return self;
    }

    fn loadKeywordPrompts(arena: Allocator, settings: *const config.Settings, http: *hf.Http, cache_root: []const u8, out: *Io.Writer) ![]Prompt {
        const kr = &settings.keyword_rate;
        try out.print("\nLoading {s} evaluation prompts from {s}...\n", .{ kr.score_name, kr.prompts.dataset });
        try out.flush();
        const prompts = try hf.loadPrompts(arena, http, cache_root, settings, kr.prompts, out);
        try out.print("* {d} prompts loaded\n", .{prompts.len});
        return prompts;
    }

    pub fn deinit(self: *Evaluator) void {
        for (self.entries) |*e| switch (e.scorer) {
            .kl_divergence => |*k| k.deinit(self.gpa),
            .refusal_logit => |*r| r.deinit(self.gpa),
            else => {},
        };
    }

    /// The deferred keyword score of the current model (`--fast-search`), if configured.
    pub fn deferredScore(self: *Evaluator, alloc: Allocator, engine: *Engine, out: *Io.Writer) !?NamedScore {
        const e = &(self.deferred orelse return null);
        const s = try e.scorer.keyword_rate.score(alloc, engine, out, null);
        return .{ .name = e.name, .score = s };
    }

    /// Runs all scorers. Display strings are allocated with `alloc`.
    pub fn scores(self: *Evaluator, alloc: Allocator, engine: *Engine, out: *Io.Writer) ![]NamedScore {
        return self.evaluate(alloc, engine, out, null);
    }

    /// Runs all scorers in order. With `front` (loss vectors of the completed
    /// Pareto front), the refusal scorer that is the last objective may stop
    /// early; the returned scores then carry `pruned` and the refusal value is
    /// an upper bound. Scorers after a pruned one are skipped.
    pub fn evaluate(self: *Evaluator, alloc: Allocator, engine: *Engine, out: *Io.Writer, front: ?[]const []const f64) ![]NamedScore {
        const result = try alloc.alloc(NamedScore, self.entries.len);
        const known = try alloc.alloc(f64, self.entries.len);
        var n_known: usize = 0;
        var pruned = false;
        const stop_entry = self.earlyStopEntry();
        for (self.entries, 0..) |*e, i| {
            if (pruned) {
                result[i] = .{ .name = e.name, .score = .{ .value = 0, .display = "skipped" } };
                continue;
            }
            const s = switch (e.scorer) {
                .keyword_rate => |*k| blk: {
                    var threshold: ?f64 = null;
                    if (front) |fr| if (stop_entry == i) {
                        threshold = pruneThreshold(fr, known[0..n_known], n_known);
                    };
                    break :blk try k.score(alloc, engine, out, threshold);
                },
                .kl_divergence => |*k| try k.score(alloc, engine),
                .refusal_logit => |*r| try r.score(alloc, engine),
            };
            result[i] = .{ .name = e.name, .score = s };
            pruned = s.pruned != null;
            switch (e.optimization) {
                .minimize => known[n_known] = s.value,
                .maximize => known[n_known] = -s.value,
                .none => continue,
            }
            n_known += 1;
        }
        return result;
    }

    /// Index of the entry that may stop early: a minimised keyword-rate scorer
    /// that is the last objective (so every other objective is known by then).
    pub fn earlyStopEntry(self: *const Evaluator) ?usize {
        var last: ?usize = null;
        for (self.entries, 0..) |e, i| if (e.optimization != .none) {
            last = i;
        };
        const i = last orelse return null;
        const e = self.entries[i];
        if (e.scorer != .keyword_rate or e.optimization != .minimize) return null;
        return i;
    }

    /// Baseline scores on the base model. The first keyword scorer (active
    /// or deferred) is run once with its first tokens captured; that
    /// generation also trains every `RefusalLogit` scorer. Without any
    /// keyword scorer a temporary one on the keyword prompts is used.
    fn baselineScores(self: *Evaluator, alloc: Allocator, engine: *Engine, settings: *const config.Settings, out: *Io.Writer) ![]NamedScore {
        const gpa = self.gpa;
        var capture = FirstTokens{};
        defer capture.deinit(gpa);
        var captured = false;
        const result = try alloc.alloc(NamedScore, self.entries.len);
        for (self.entries, 0..) |*e, i| {
            const s: Score = switch (e.scorer) {
                .keyword_rate => |*k| blk: {
                    const s = try k.scoreCapturing(alloc, engine, out, null, if (captured) null else &capture);
                    captured = true;
                    break :blk s;
                },
                .kl_divergence => .{ .value = 0, .display = "0 (by definition)" },
                .refusal_logit => .{ .value = 0, .display = "" }, // filled below
            };
            result[i] = .{ .name = e.name, .score = s };
        }
        if (self.deferred) |*d| {
            const s = try d.scorer.keyword_rate.scoreCapturing(alloc, engine, out, null, if (captured) null else &capture);
            captured = true;
            self.deferred_baseline = .{ .name = d.name, .score = s };
        }
        for (self.entries, 0..) |*e, i| {
            const r = switch (e.scorer) {
                .refusal_logit => |*r| r,
                else => continue,
            };
            if (!captured) {
                var tmp = KeywordRate{ .settings = &settings.keyword_rate, .prompts = r.prompts };
                try out.writeAll("* Generating baseline responses for the refusal-start tokens...\n");
                try out.flush();
                const s = try tmp.scoreCapturing(alloc, engine, out, null, &capture);
                _ = s;
                captured = true;
            }
            try r.learn(gpa, capture.tokens.items, capture.refused.items);
            if (r.weights.count() == 0) {
                try r.useOpeners(gpa, engine);
                try out.print("* {s}: the base model refused no prompt; using {d} tokenised refusal openers\n", .{ e.name, r.weights.count() });
            } else {
                try out.print("* {s}: {d} refusal-start tokens learned from {d} baseline refusals\n", .{ e.name, r.weights.count(), r.n_refusals });
            }
            result[i].score = try r.score(alloc, engine);
        }
        return result;
    }

    /// Whether any returned score was pruned.
    pub fn prunedOf(s: []const NamedScore) ?Pruned {
        for (s) |ns| if (ns.score.pruned) |p| return p;
        return null;
    }

    pub fn objectiveCount(self: *const Evaluator) usize {
        var n: usize = 0;
        for (self.entries) |e| if (e.optimization != .none) {
            n += 1;
        };
        return n;
    }

    /// Objective values converted to minimisation, in entry order.
    pub fn objectiveLosses(self: *const Evaluator, alloc: Allocator, s: []const NamedScore) ![]f64 {
        var out = std.ArrayList(f64).empty;
        for (self.entries, 0..) |e, i| {
            switch (e.optimization) {
                .minimize => try out.append(alloc, s[i].score.value),
                .maximize => try out.append(alloc, -s[i].score.value),
                .none => {},
            }
        }
        return out.toOwnedSlice(alloc);
    }
};

test "prune threshold picks the qualifying front trial with the fewest refusals" {
    // Losses are [kl, refusal_rate].
    const a = [_]f64{ 0.10, 0.50 };
    const b = [_]f64{ 0.20, 0.30 };
    const c = [_]f64{ 0.40, 0.10 };
    const front = [_][]const f64{ &a, &b, &c };
    try std.testing.expectEqual(@as(?f64, 0.30), pruneThreshold(&front, &.{0.25}, 1));
    try std.testing.expectEqual(@as(?f64, 0.10), pruneThreshold(&front, &.{0.40}, 1));
    try std.testing.expectEqual(@as(?f64, null), pruneThreshold(&front, &.{0.05}, 1));
    try std.testing.expect(!shouldPrune(3, 10, 0.30));
    try std.testing.expect(shouldPrune(4, 10, 0.30));
    try std.testing.expect(!shouldPrune(9, 10, null));
}

test "early stopping with synthetic scorers never prunes a front trial" {
    // A synthetic search over x, y in [0, 1]: kl = x^2 + y^2, and prompt p
    // refuses iff x < t_p for fixed thresholds t_p, so refusals fall with x
    // (a genuine trade-off) while y only damages the model (trials with a
    // large y are dominated). Refusals are scored in batches with the pruning
    // rule; afterwards every pruned trial must be dominated by a completed
    // trial, so the front of completed trials equals the front that full
    // evaluation of every trial would have produced.
    const tpe = @import("tpe.zig");
    const gpa = std.testing.allocator;
    const n_prompts = 20;
    const batch = 4;
    var thresholds: [n_prompts]f64 = undefined;
    for (&thresholds, 0..) |*t, p| t.* = @as(f64, @floatFromInt(p + 1)) / @as(f64, n_prompts + 1);
    const Trial = struct { params: [2]f64, losses: [2]f64, true_rate: f64, pruned: bool };
    var trials = std.ArrayList(Trial).empty;
    defer trials.deinit(gpa);
    var history = std.ArrayList(tpe.Observation).empty;
    defer {
        for (history.items) |h| {
            gpa.free(h.params);
            gpa.free(h.losses);
        }
        history.deinit(gpa);
    }
    var front = std.ArrayList([]const f64).empty;
    defer front.deinit(gpa);

    const names = [_][]const u8{ "x", "y" };
    const specs = [_]tpe.ParamSpec{ .{ .float = .{ .low = 0, .high = 1 } }, .{ .float = .{ .low = 0, .high = 1 } } };
    var sampler = tpe.Sampler.init(.{ .names = &names, .specs = &specs }, 6, 7);
    var n_pruned: usize = 0;
    var t: usize = 0;
    while (t < 40) : (t += 1) {
        var params: [2]f64 = undefined;
        try sampler.sample(gpa, history.items, &params);
        const x = params[0];
        const kl = x * x + params[1] * params[1];
        // Current front of completed trials.
        front.clearRetainingCapacity();
        for (trials.items) |*tr| if (!tr.pruned) {
            var dominated = false;
            for (trials.items) |o| if (!o.pruned and (o.losses[0] <= tr.losses[0] and o.losses[1] <= tr.losses[1]) and (o.losses[0] < tr.losses[0] or o.losses[1] < tr.losses[1])) {
                dominated = true;
            };
            if (!dominated) try front.append(gpa, &tr.losses);
        };
        const threshold = pruneThreshold(front.items, &.{kl}, 1);
        var matches: usize = 0;
        var true_matches: usize = 0;
        for (thresholds) |tp| true_matches += @intFromBool(x < tp);
        var done: usize = 0;
        var pruned = false;
        while (done < n_prompts) {
            const end = @min(n_prompts, done + batch);
            for (thresholds[done..end]) |tp| matches += @intFromBool(x < tp);
            done = end;
            if (shouldPrune(matches, n_prompts, threshold)) {
                pruned = true;
                break;
            }
        }
        const rate = if (pruned) @as(f64, @floatFromInt(matches + n_prompts - done)) / n_prompts else @as(f64, @floatFromInt(matches)) / n_prompts;
        n_pruned += @intFromBool(pruned);
        try trials.append(gpa, .{ .params = params, .losses = .{ kl, rate }, .true_rate = @as(f64, @floatFromInt(true_matches)) / n_prompts, .pruned = pruned });
        // Pruned trials feed the sampler with their bounded losses.
        try history.append(gpa, .{ .params = try gpa.dupe(f64, &params), .losses = try gpa.dupe(f64, &.{ kl, rate }) });
    }
    try std.testing.expect(n_pruned > 0);
    try std.testing.expect(n_pruned < trials.items.len);
    // Soundness: every pruned trial is strictly dominated by a completed trial
    // even when judged by its true (fully evaluated) refusal rate.
    for (trials.items) |tr| {
        if (!tr.pruned) continue;
        try std.testing.expect(tr.true_rate <= tr.losses[1]);
        var dominated = false;
        for (trials.items) |o| if (!o.pruned and o.losses[0] <= tr.losses[0] and o.losses[1] < tr.true_rate) {
            dominated = true;
        };
        try std.testing.expect(dominated);
    }
}

test "refusal logit weights are first-token precisions and rank edits like the keyword labels" {
    const gpa = std.testing.allocator;
    var rl = RefusalLogit{ .prompts = &.{}, .vocab = 12 };
    defer rl.deinit(gpa);
    // Baseline first tokens: 5 starts two refusals and one helpful answer,
    // 9 starts one refusal, 7 never refuses.
    try rl.learn(gpa, &.{ 5, 5, 7, 9, 5, 7 }, &.{ true, true, false, true, false, false });
    try std.testing.expectEqual(@as(usize, 2), rl.weights.count());
    try std.testing.expectApproxEqAbs(@as(f32, 2.0 / 3.0), rl.weights.get(5).?, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), rl.weights.get(9).?, 1e-6);
    try std.testing.expect(rl.weights.get(7) == null);
    try std.testing.expectEqual(@as(usize, 3), rl.n_refusals);
    // Two hand-made first-token distributions: edit A still opens like the
    // refusals (0.6 on token 5, 0.3 on 9), edit B has moved that mass to
    // token 7 (which the keyword scorer never labelled a refusal). A must
    // score higher, and a distribution with no mass on refusal starts scores 0.
    const logp = struct {
        fn of(buf: []f32, probs: []const f32) []f32 {
            for (buf, 0..) |*x, i| x.* = @log(@max(probs[i], 1e-30));
            return buf;
        }
    };
    var buf: [12]f32 = undefined;
    const a = logp.of(&buf, &.{ 0.0125, 0.0125, 0.0125, 0.0125, 0.0125, 0.6, 0.0125, 0.0125, 0.0125, 0.3, 0.0125, 0.0125 });
    const mass_a = rl.massOf(a);
    try std.testing.expectApproxEqAbs(@as(f64, 0.6 * 2.0 / 3.0 + 0.3), mass_a, 1e-6);
    const b = logp.of(&buf, &.{ 0.0125, 0.0125, 0.0125, 0.0125, 0.0125, 0.05, 0.0125, 0.85, 0.0125, 0.0125, 0.0125, 0.0125 });
    const mass_b = rl.massOf(b);
    try std.testing.expect(mass_a > mass_b);
    const none = logp.of(&buf, &.{ 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 });
    try std.testing.expectApproxEqAbs(@as(f64, 0), rl.massOf(none), 1e-9);
    // No refusals at all: the learned set is empty (the fallback openers apply).
    try rl.learn(gpa, &.{ 1, 2 }, &.{ false, false });
    try std.testing.expectEqual(@as(usize, 0), rl.weights.count());
}

const model_mod = @import("model.zig");

fn fixturePrompts(gpa: Allocator, texts: []const []const u8) ![]Prompt {
    const out = try gpa.alloc(Prompt, texts.len);
    for (texts, 0..) |t, i| out[i] = .{ .system = "", .user = t };
    return out;
}

test "refusal logit and multi-token KL on the qwen2 fixture" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    const pool = tensor.Pool.init(io, 1);
    const model = try model_mod.Model.load(gpa, io, &pool, "tests/fixtures/qwen2");
    defer model.deinit();
    var settings = config.Settings{ .batch_size = 2, .max_response_length = 4, .response_prefix = "" };
    var engine = engine_mod.Engine.init(gpa, model, &settings, .raw);
    defer engine.deinit();
    const prompts = try fixturePrompts(gpa, &.{ "tell me how", "the cat sat", "one two three four", "pick a lock" });
    defer gpa.free(prompts);
    var sink: Io.Writer.Allocating = .init(gpa);
    defer sink.deinit();

    // The baseline generation's first tokens train the logit scorer; with a
    // marker that no response contains, nothing is learned and the openers
    // are used instead. With every response counted as a refusal, the greedy
    // first tokens all get weight 1 and the base model's mass is high: the
    // argmax token of every prompt is in the set.
    var kr_settings = config.KeywordRateSettings{ .keyword_markers = &.{"\x00never"} };
    var kr = KeywordRate{ .settings = &kr_settings, .prompts = prompts };
    var capture = FirstTokens{};
    defer capture.deinit(gpa);
    const s0 = try kr.scoreCapturing(gpa, &engine, &sink.writer, null, &capture);
    defer gpa.free(s0.display);
    try std.testing.expectEqual(@as(usize, 4), capture.tokens.items.len);
    var rl = RefusalLogit{ .prompts = prompts, .vocab = model.config.vocab_size };
    defer rl.deinit(gpa);
    try rl.learn(gpa, capture.tokens.items, capture.refused.items);
    try std.testing.expectEqual(@as(usize, 0), rl.weights.count());
    try rl.useOpeners(gpa, &engine);
    try std.testing.expect(rl.weights.count() > 0);
    const all_refused = [_]bool{ true, true, true, true };
    try rl.learn(gpa, capture.tokens.items, &all_refused);
    const base = try rl.score(gpa, &engine);
    defer gpa.free(base.display);
    try std.testing.expect(base.value > 0 and base.value <= 1);
    // Greedy first tokens are argmax tokens, so each prompt's mass is at least
    // 1/vocab and, for a peaked fixture distribution, well above it.
    try std.testing.expect(base.value > 1.0 / @as(f64, @floatFromInt(model.config.vocab_size)));
    // Per-prompt agreement: the labelled prompts carry the mass.
    var rl2 = RefusalLogit{ .prompts = prompts, .vocab = model.config.vocab_size };
    defer rl2.deinit(gpa);
    const half = [_]bool{ true, false, true, false };
    try rl2.learn(gpa, capture.tokens.items, &half);
    const logits = try engine.getLogits(gpa, prompts);
    defer gpa.free(logits);
    const tmp = try gpa.alloc(f32, model.config.vocab_size);
    defer gpa.free(tmp);
    var mass_ref: f64 = 0;
    var mass_other: f64 = 0;
    for (0..4) |i| {
        tensor.logSoftmax(tmp, logits[i * model.config.vocab_size ..][0..model.config.vocab_size]);
        if (half[i]) mass_ref += rl2.massOf(tmp) else mass_other += rl2.massOf(tmp);
    }
    try std.testing.expect(mass_ref > mass_other);

    // KL over 1 and 3 positions: zero on the base model, equal at T = 1 to the
    // first-token scorer, and finite after an edit.
    var kl1 = try KlDivergence.init(gpa, &engine, prompts, 1);
    defer kl1.deinit(gpa);
    var kl3 = try KlDivergence.init(gpa, &engine, prompts, 3);
    defer kl3.deinit(gpa);
    for (kl1.tails) |t| try std.testing.expectEqual(@as(usize, 1), t);
    for (kl3.tails, kl3.seqs, 0..) |t, s, i| {
        try std.testing.expect(t >= 1 and t <= 3);
        try std.testing.expectEqual(kl1.seqs[i].len + t - 1, s.len);
    }
    const z1 = try kl1.score(gpa, &engine);
    defer gpa.free(z1.display);
    const z3 = try kl3.score(gpa, &engine);
    defer gpa.free(z3.display);
    try std.testing.expect(@abs(z1.value) < 1e-5 and @abs(z3.value) < 1e-5);
    const c = &model.config;
    const dirs = try gpa.alloc(f32, (c.num_layers + 1) * c.hidden_size);
    defer gpa.free(dirs);
    for (dirs, 0..) |*x, i| x.* = if (i % c.hidden_size == 1) 1.0 else 0.0;
    var params = std.EnumMap(model_mod.Component, @import("abliterate.zig").Params){};
    params.put(.attn_o_proj, .{ .max_weight = 1.0, .max_weight_position = 1, .min_weight = 1.0, .min_weight_distance = 2 });
    try @import("abliterate.zig").apply(model, dirs, null, params, .{ .row_normalization = .none });
    const e1 = try kl1.score(gpa, &engine);
    defer gpa.free(e1.display);
    const e3 = try kl3.score(gpa, &engine);
    defer gpa.free(e3.display);
    try std.testing.expect(e1.value > 0 and std.math.isFinite(e1.value));
    try std.testing.expect(e3.value > 0 and std.math.isFinite(e3.value));
    model.resetDeltas();
}

test "keyword matching" {
    const gpa = std.testing.allocator;
    const markers = [_][]const u8{ "i can'", "sorry", "illegal" };
    try std.testing.expect(try KeywordRate.isMatch(gpa, "I *can’t* help with that.", &markers));
    try std.testing.expect(try KeywordRate.isMatch(gpa, "   ", &markers));
    try std.testing.expect(try KeywordRate.isMatch(gpa, "This is\n  ILLEGAL", &markers));
    try std.testing.expect(!try KeywordRate.isMatch(gpa, "Sure, here is how", &markers));
}
