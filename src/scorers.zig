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
};

pub const KeywordRate = struct {
    settings: *const config.KeywordRateSettings,
    prompts: []Prompt,

    pub fn score(self: *KeywordRate, gpa: Allocator, engine: *Engine, out: *Io.Writer) !Score {
        const responses = try engine.getResponses(gpa, self.prompts, false);
        defer {
            for (responses) |r| gpa.free(r);
            gpa.free(responses);
        }
        var matches: usize = 0;
        for (self.prompts, responses) |p, r| {
            const m = try isMatch(gpa, r, self.settings.keyword_markers);
            if (m) matches += 1;
            if (self.settings.print_responses) {
                try out.print("\nSystem prompt: {s}\nPrompt: {s}\nResponse{s}: {s}\n", .{ p.system, p.user, if (m) " [refusal]" else "", if (std.mem.trim(u8, r, " \t\r\n").len == 0) "[empty]" else r });
            }
        }
        if (self.settings.print_responses) try out.writeAll("\n");
        return .{
            .value = @as(f64, @floatFromInt(matches)) / @as(f64, @floatFromInt(@max(self.prompts.len, 1))),
            .display = try std.fmt.allocPrint(gpa, "{d}/{d}", .{ matches, self.prompts.len }),
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

pub const KlDivergence = struct {
    prompts: []Prompt,
    /// Baseline log-probabilities `[prompts][vocab]`.
    baseline: []f32,
    vocab: usize,

    pub fn init(gpa: Allocator, engine: *Engine, prompts: []Prompt) !KlDivergence {
        const logits = try engine.getLogits(gpa, prompts);
        const vocab = engine.model.config.vocab_size;
        for (0..prompts.len) |i| {
            const row = logits[i * vocab ..][0..vocab];
            const tmp = try gpa.alloc(f32, vocab);
            defer gpa.free(tmp);
            tensor.logSoftmax(tmp, row);
            @memcpy(row, tmp);
        }
        return .{ .prompts = prompts, .baseline = logits, .vocab = vocab };
    }

    pub fn deinit(self: *KlDivergence, gpa: Allocator) void {
        gpa.free(self.baseline);
    }

    pub fn score(self: *KlDivergence, gpa: Allocator, engine: *Engine) !Score {
        const logits = try engine.getLogits(gpa, self.prompts);
        defer gpa.free(logits);
        const tmp = try gpa.alloc(f32, self.vocab);
        defer gpa.free(tmp);
        var total: f64 = 0;
        for (0..self.prompts.len) |i| {
            tensor.logSoftmax(tmp, logits[i * self.vocab ..][0..self.vocab]);
            const base = self.baseline[i * self.vocab ..][0..self.vocab];
            var kl: f64 = 0;
            for (base, 0..) |lb, j| {
                const p = @exp(@as(f64, lb));
                kl += p * (@as(f64, lb) - @as(f64, tmp[j]));
            }
            total += kl;
        }
        const value = total / @as(f64, @floatFromInt(@max(self.prompts.len, 1)));
        return .{ .value = value, .display = try std.fmt.allocPrint(gpa, "{d:.4}", .{value}) };
    }
};

pub const Scorer = union(enum) {
    keyword_rate: KeywordRate,
    kl_divergence: KlDivergence,
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

    pub fn init(gpa: Allocator, arena: Allocator, engine: *Engine, settings: *const config.Settings, http: *hf.Http, cache_root: []const u8, out: *Io.Writer) !Evaluator {
        var entries = std.ArrayList(Entry).empty;
        try out.writeAll("\nLoading and initializing scorers...\n");
        for (settings.scorers) |sc| {
            switch (sc.kind) {
                .keyword_rate => {
                    const kr = &settings.keyword_rate;
                    const name = if (sc.instance_name) |n| try std.fmt.allocPrint(arena, "{s} - {s}", .{ kr.score_name, n }) else kr.score_name;
                    try out.print("* Loaded: KeywordRate ({s})\n", .{name});
                    try out.print("\nLoading {s} evaluation prompts from {s}...\n", .{ kr.score_name, kr.prompts.dataset });
                    try out.flush();
                    const prompts = try hf.loadPrompts(arena, http, cache_root, settings, kr.prompts, out);
                    try out.print("* {d} prompts loaded\n", .{prompts.len});
                    try entries.append(arena, .{ .name = name, .optimization = sc.optimization, .scorer = .{ .keyword_rate = .{ .settings = kr, .prompts = prompts } } });
                },
                .kl_divergence => {
                    const name = if (sc.instance_name) |n| try std.fmt.allocPrint(arena, "KL divergence - {s}", .{n}) else "KL divergence";
                    try out.print("* Loaded: KLDivergence ({s})\n", .{name});
                    try out.print("\nLoading KL divergence evaluation prompts from {s}...\n", .{settings.kl_divergence.prompts.dataset});
                    try out.flush();
                    const prompts = try hf.loadPrompts(arena, http, cache_root, settings, settings.kl_divergence.prompts, out);
                    try out.print("* {d} prompts loaded\n", .{prompts.len});
                    try out.writeAll("* Obtaining baseline first-token probability distributions...\n");
                    try out.flush();
                    const kl = try KlDivergence.init(gpa, engine, prompts);
                    try entries.append(arena, .{ .name = name, .optimization = sc.optimization, .scorer = .{ .kl_divergence = kl } });
                },
            }
        }
        var self = Evaluator{ .gpa = gpa, .entries = entries.items, .baseline = &.{} };
        try out.writeAll("\nGetting baseline scores...\n");
        try out.flush();
        self.baseline = try self.baselineScores(arena, engine, out);
        for (self.baseline) |b| try out.print("* Baseline {s}: {s}\n", .{ b.name, b.score.display });
        return self;
    }

    pub fn deinit(self: *Evaluator) void {
        for (self.entries) |*e| switch (e.scorer) {
            .kl_divergence => |*k| k.deinit(self.gpa),
            else => {},
        };
    }

    /// Runs all scorers. Display strings are allocated with `alloc`.
    pub fn scores(self: *Evaluator, alloc: Allocator, engine: *Engine, out: *Io.Writer) ![]NamedScore {
        const result = try alloc.alloc(NamedScore, self.entries.len);
        for (self.entries, 0..) |*e, i| {
            const s = switch (e.scorer) {
                .keyword_rate => |*k| try k.score(alloc, engine, out),
                .kl_divergence => |*k| try k.score(alloc, engine),
            };
            result[i] = .{ .name = e.name, .score = s };
        }
        return result;
    }

    fn baselineScores(self: *Evaluator, alloc: Allocator, engine: *Engine, out: *Io.Writer) ![]NamedScore {
        const result = try alloc.alloc(NamedScore, self.entries.len);
        for (self.entries, 0..) |*e, i| {
            const s: Score = switch (e.scorer) {
                .keyword_rate => |*k| try k.score(alloc, engine, out),
                .kl_divergence => .{ .value = 0, .display = "0 (by definition)" },
            };
            result[i] = .{ .name = e.name, .score = s };
        }
        return result;
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

test "keyword matching" {
    const gpa = std.testing.allocator;
    const markers = [_][]const u8{ "i can'", "sorry", "illegal" };
    try std.testing.expect(try KeywordRate.isMatch(gpa, "I *can’t* help with that.", &markers));
    try std.testing.expect(try KeywordRate.isMatch(gpa, "   ", &markers));
    try std.testing.expect(try KeywordRate.isMatch(gpa, "This is\n  ILLEGAL", &markers));
    try std.testing.expect(!try KeywordRate.isMatch(gpa, "Sure, here is how", &markers));
}
