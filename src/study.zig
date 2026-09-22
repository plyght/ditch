//! Study persistence (JSON lines journal) and Pareto-front selection.

const std = @import("std");
const Io = std.Io;
const tpe = @import("tpe.zig");
const abliterate = @import("abliterate.zig");
const model_mod = @import("model.zig");

const Allocator = std.mem.Allocator;

pub const ScoreRecord = struct {
    name: []const u8,
    value: f64,
    display: []const u8,
};

pub const State = enum {
    /// Fully evaluated; eligible for the results menu.
    complete,
    /// Early-stopped: the losses are an upper bound (all remaining prompts
    /// counted as refusals). Used by the sampler, never offered as a result.
    pruned,

    pub fn parse(s: []const u8) ?State {
        inline for (@typeInfo(State).@"enum".fields) |f| {
            if (std.mem.eql(u8, s, f.name)) return @enumFromInt(f.value);
        }
        return null;
    }
};

pub const Trial = struct {
    index: usize,
    /// Sampled parameter vector in the study's parameter space order.
    params: []const f64,
    /// Objective values converted to minimisation.
    losses: []const f64,
    direction_index: ?f32,
    parameters: std.EnumMap(model_mod.Component, abliterate.Params),
    scores: []const ScoreRecord,
    state: State = .complete,

    pub fn observation(self: *const Trial) tpe.Observation {
        return .{ .params = self.params, .losses = self.losses };
    }
};

pub const Study = struct {
    arena: std.heap.ArenaAllocator,
    io: Io,
    path: []const u8,
    trials: std.ArrayList(Trial),
    finished: bool,
    /// Settings snapshot stored with the study (JSON), if any.
    settings_json: ?[]const u8,
    /// The parsed settings snapshot (object fields), if any.
    settings: ?std.json.ObjectMap,

    /// Loads the journal at `path` (which need not exist). Nothing is written
    /// until `reset`, `addTrial` or `markFinished` is called, so a foreign
    /// journal (e.g. a warm-start study) can be opened read-only.
    pub fn open(gpa: Allocator, io: Io, path: []const u8) !Study {
        var self = Study{
            .arena = std.heap.ArenaAllocator.init(gpa),
            .io = io,
            .path = undefined,
            .trials = .empty,
            .finished = false,
            .settings_json = null,
            .settings = null,
        };
        errdefer self.arena.deinit();
        const a = self.arena.allocator();
        self.path = try a.dupe(u8, path);
        const text = Io.Dir.cwd().readFileAlloc(io, path, a, .unlimited) catch |err| switch (err) {
            error.FileNotFound => return self,
            else => return err,
        };
        var it = std.mem.splitScalar(u8, text, '\n');
        while (it.next()) |line| {
            const trimmed = std.mem.trim(u8, line, " \r\t");
            if (trimmed.len == 0) continue;
            self.parseLine(gpa, trimmed) catch |err| {
                std.log.warn("skipping malformed checkpoint line: {s}", .{@errorName(err)});
            };
        }
        return self;
    }

    pub fn deinit(self: *Study) void {
        self.arena.deinit();
    }

    pub fn exists(self: *const Study) bool {
        return self.trials.items.len > 0 or self.settings_json != null;
    }

    fn parseLine(self: *Study, gpa: Allocator, line: []const u8) !void {
        const a = self.arena.allocator();
        var parsed = try std.json.parseFromSlice(std.json.Value, gpa, line, .{});
        defer parsed.deinit();
        const obj = parsed.value.object;
        const kind = (obj.get("type") orelse return error.Invalid).string;
        if (std.mem.eql(u8, kind, "settings")) {
            self.settings_json = try a.dupe(u8, line);
            self.settings = (try std.json.parseFromSliceLeaky(std.json.Value, a, line, .{})).object;
            self.finished = false;
            self.trials.clearRetainingCapacity();
        } else if (std.mem.eql(u8, kind, "finished")) {
            self.finished = true;
        } else if (std.mem.eql(u8, kind, "rescore")) {
            // Scores added after the trial ran (the deferred keyword scorer).
            const index: usize = @intCast(obj.get("index").?.integer);
            const sv = obj.get("scores") orelse return error.Invalid;
            for (self.trials.items) |*t| if (t.index == index) {
                for (sv.array.items) |s| {
                    try self.mergeScore(t, .{
                        .name = try a.dupe(u8, s.object.get("name").?.string),
                        .value = numberOf(s.object.get("value").?),
                        .display = try a.dupe(u8, s.object.get("display").?.string),
                    });
                }
            };
        } else if (std.mem.eql(u8, kind, "trial")) {
            const params_v = obj.get("params").?.array;
            const params = try a.alloc(f64, params_v.items.len);
            for (params_v.items, 0..) |v, i| params[i] = numberOf(v);
            const losses_v = obj.get("losses").?.array;
            const losses = try a.alloc(f64, losses_v.items.len);
            for (losses_v.items, 0..) |v, i| losses[i] = numberOf(v);
            const di: ?f32 = if (obj.get("direction_index")) |d| (if (d == .null) null else @as(f32, @floatCast(numberOf(d)))) else null;
            var pmap = std.EnumMap(model_mod.Component, abliterate.Params){};
            if (obj.get("parameters")) |pv| {
                var pit = pv.object.iterator();
                while (pit.next()) |e| {
                    const comp = model_mod.Component.fromName(e.key_ptr.*) orelse continue;
                    const po = e.value_ptr.object;
                    pmap.put(comp, .{
                        .max_weight = @floatCast(numberOf(po.get("max_weight").?)),
                        .max_weight_position = @floatCast(numberOf(po.get("max_weight_position").?)),
                        .min_weight = @floatCast(numberOf(po.get("min_weight").?)),
                        .min_weight_distance = @floatCast(numberOf(po.get("min_weight_distance").?)),
                    });
                }
            }
            var scores = std.ArrayList(ScoreRecord).empty;
            if (obj.get("scores")) |sv| {
                for (sv.array.items) |s| {
                    try scores.append(a, .{
                        .name = try a.dupe(u8, s.object.get("name").?.string),
                        .value = numberOf(s.object.get("value").?),
                        .display = try a.dupe(u8, s.object.get("display").?.string),
                    });
                }
            }
            var state: State = .complete;
            if (obj.get("state")) |sv| {
                if (sv == .string) state = State.parse(sv.string) orelse return error.Invalid;
            }
            try self.trials.append(a, .{
                .index = @intCast(obj.get("index").?.integer),
                .params = params,
                .losses = losses,
                .direction_index = di,
                .parameters = pmap,
                .scores = scores.items,
                .state = state,
            });
        }
    }

    /// An integer field of the stored settings snapshot, if present.
    pub fn settingInteger(self: *const Study, key: []const u8) ?i64 {
        const s = self.settings orelse return null;
        const v = s.get(key) orelse return null;
        return switch (v) {
            .integer => |i| i,
            .float => |f| @intFromFloat(f),
            else => null,
        };
    }

    pub fn settingFloat(self: *const Study, key: []const u8) ?f64 {
        const s = self.settings orelse return null;
        const v = s.get(key) orelse return null;
        return switch (v) {
            .integer => |i| @floatFromInt(i),
            .float => |f| f,
            else => null,
        };
    }

    pub fn settingString(self: *const Study, key: []const u8) ?[]const u8 {
        const s = self.settings orelse return null;
        const v = s.get(key) orelse return null;
        return if (v == .string) v.string else null;
    }

    pub fn settingBool(self: *const Study, key: []const u8) ?bool {
        const s = self.settings orelse return null;
        const v = s.get(key) orelse return null;
        return if (v == .bool) v.bool else null;
    }

    fn numberOf(v: std.json.Value) f64 {
        return switch (v) {
            .integer => |i| @floatFromInt(i),
            .float => |f| f,
            .number_string => |s| std.fmt.parseFloat(f64, s) catch 0,
            else => 0,
        };
    }

    fn appendLine(self: *Study, line: []const u8) !void {
        const cwd = Io.Dir.cwd();
        if (std.fs.path.dirname(self.path)) |d| try cwd.createDirPath(self.io, d);
        const file = try cwd.createFile(self.io, self.path, .{ .truncate = false });
        defer file.close(self.io);
        const len = try file.length(self.io);
        try file.writePositionalAll(self.io, line, len);
        try file.writePositionalAll(self.io, "\n", len + line.len);
    }

    /// Starts a fresh journal (deleting any previous content).
    pub fn reset(self: *Study, settings_json: []const u8) !void {
        const cwd = Io.Dir.cwd();
        cwd.deleteFile(self.io, self.path) catch {};
        self.trials.clearRetainingCapacity();
        self.finished = false;
        const a = self.arena.allocator();
        self.settings_json = try a.dupe(u8, settings_json);
        self.settings = (try std.json.parseFromSliceLeaky(std.json.Value, a, settings_json, .{})).object;
        try self.appendLine(settings_json);
    }

    pub fn markFinished(self: *Study) !void {
        if (self.finished) return;
        self.finished = true;
        try self.appendLine("{\"type\":\"finished\"}");
    }

    pub fn markUnfinished(self: *Study) void {
        self.finished = false;
    }

    /// Records a completed trial (copying all data into the study arena) and appends it to the journal.
    pub fn addTrial(self: *Study, gpa: Allocator, trial: Trial) !void {
        const a = self.arena.allocator();
        var copy = trial;
        copy.params = try a.dupe(f64, trial.params);
        copy.losses = try a.dupe(f64, trial.losses);
        const scores_copy = try a.alloc(ScoreRecord, trial.scores.len);
        for (trial.scores, 0..) |s, i| scores_copy[i] = .{ .name = try a.dupe(u8, s.name), .value = s.value, .display = try a.dupe(u8, s.display) };
        copy.scores = scores_copy;
        try self.trials.append(a, copy);

        var buf: Io.Writer.Allocating = .init(gpa);
        defer buf.deinit();
        var js: std.json.Stringify = .{ .writer = &buf.writer };
        try js.beginObject();
        try js.objectField("type");
        try js.write("trial");
        try writeTrialFields(&js, &copy);
        try js.endObject();
        try self.appendLine(buf.written());
    }

    /// Writes a trial's fields into an open JSON object (the journal record
    /// and `--json-log` share this).
    pub fn writeTrialFields(js: *std.json.Stringify, copy: *const Trial) !void {
        try js.objectField("index");
        try js.write(copy.index);
        try js.objectField("params");
        try js.write(copy.params);
        try js.objectField("losses");
        try js.write(copy.losses);
        try js.objectField("direction_index");
        try js.write(copy.direction_index);
        if (copy.state != .complete) {
            try js.objectField("state");
            try js.write(@tagName(copy.state));
        }
        try js.objectField("parameters");
        try js.beginObject();
        for (model_mod.Component.all) |comp| {
            const p = copy.parameters.get(comp) orelse continue;
            try js.objectField(comp.name());
            try js.write(p);
        }
        try js.endObject();
        try js.objectField("scores");
        try js.beginArray();
        for (copy.scores) |s| {
            try js.beginObject();
            try js.objectField("name");
            try js.write(s.name);
            try js.objectField("value");
            try js.write(s.value);
            try js.objectField("display");
            try js.write(s.display);
            try js.endObject();
        }
        try js.endArray();
    }

    /// Replaces the trial's score of the same name or appends it (the trial's
    /// scores live in the study arena).
    fn mergeScore(self: *Study, trial: *Trial, score: ScoreRecord) !void {
        const a = self.arena.allocator();
        var replaced = false;
        for (trial.scores) |s| replaced = replaced or std.mem.eql(u8, s.name, score.name);
        const grown = try a.alloc(ScoreRecord, trial.scores.len + @intFromBool(!replaced));
        for (trial.scores, 0..) |s, i| grown[i] = if (std.mem.eql(u8, s.name, score.name)) score else s;
        if (!replaced) grown[trial.scores.len] = score;
        trial.scores = grown;
    }

    /// Adds scores to a recorded trial (for example the deferred keyword
    /// score of a Pareto candidate) and journals them; the trial's losses
    /// are unchanged, so the search is unaffected.
    pub fn addScores(self: *Study, gpa: Allocator, index: usize, scores: []const ScoreRecord) !void {
        const a = self.arena.allocator();
        for (self.trials.items) |*t| if (t.index == index) {
            for (scores) |s| try self.mergeScore(t, .{ .name = try a.dupe(u8, s.name), .value = s.value, .display = try a.dupe(u8, s.display) });
        };
        var buf: Io.Writer.Allocating = .init(gpa);
        defer buf.deinit();
        var js: std.json.Stringify = .{ .writer = &buf.writer };
        try js.beginObject();
        try js.objectField("type");
        try js.write("rescore");
        try js.objectField("index");
        try js.write(index);
        try js.objectField("scores");
        try js.beginArray();
        for (scores) |s| {
            try js.beginObject();
            try js.objectField("name");
            try js.write(s.name);
            try js.objectField("value");
            try js.write(s.value);
            try js.objectField("display");
            try js.write(s.display);
            try js.endObject();
        }
        try js.endArray();
        try js.endObject();
        try self.appendLine(buf.written());
    }

    /// The score of `name` recorded for a trial, if any.
    pub fn scoreOf(trial: *const Trial, name: []const u8) ?ScoreRecord {
        for (trial.scores) |s| if (std.mem.eql(u8, s.name, name)) return s;
        return null;
    }

    /// `--select auto`: among `best` (indices into `trials`), the trial
    /// minimising `refusals + lambda · kl`, where refusals is the score named
    /// `refusal_name` (a rate) or, when a trial lacks it, `fallback_name`
    /// (the refusal proxy); trials without either or without `kl_name` are
    /// skipped. Returns the position within `best`.
    pub fn selectScalarised(self: *const Study, best: []const usize, refusal_name: []const u8, fallback_name: ?[]const u8, kl_name: []const u8, lambda: f64) ?usize {
        var pick: ?usize = null;
        var pick_value: f64 = std.math.inf(f64);
        for (best, 0..) |ti, i| {
            const t = &self.trials.items[ti];
            const kl = scoreOf(t, kl_name) orelse continue;
            const r = scoreOf(t, refusal_name) orelse (if (fallback_name) |f| scoreOf(t, f) else null) orelse continue;
            const v = r.value + lambda * kl.value;
            if (v < pick_value) {
                pick_value = v;
                pick = i;
            }
        }
        return pick;
    }

    pub fn observations(self: *const Study, gpa: Allocator) ![]tpe.Observation {
        const out = try gpa.alloc(tpe.Observation, self.trials.items.len);
        for (self.trials.items, 0..) |*t, i| out[i] = t.observation();
        return out;
    }

    /// Number of pruned (early-stopped) trials.
    pub fn prunedCount(self: *const Study) usize {
        var n: usize = 0;
        for (self.trials.items) |t| n += @intFromBool(t.state == .pruned);
        return n;
    }

    /// Indices of Pareto-optimal completed trials, sorted by their losses.
    /// Pruned trials neither appear nor dominate: their losses are only bounds.
    pub fn bestTrials(self: *const Study, gpa: Allocator) ![]usize {
        var complete = std.ArrayList(usize).empty;
        defer complete.deinit(gpa);
        for (self.trials.items, 0..) |t, i| if (t.state == .complete) try complete.append(gpa, i);
        const losses = try gpa.alloc([]const f64, complete.items.len);
        defer gpa.free(losses);
        for (complete.items, 0..) |ti, i| losses[i] = self.trials.items[ti].losses;
        const ranks = try tpe.nonDominatedRanks(gpa, losses);
        defer gpa.free(ranks);
        var best = std.ArrayList(usize).empty;
        for (ranks, 0..) |r, i| if (r == 0) try best.append(gpa, complete.items[i]);
        const all = try gpa.alloc([]const f64, self.trials.items.len);
        defer gpa.free(all);
        for (self.trials.items, 0..) |t, i| all[i] = t.losses;
        self.sortFront(best.items, all);
        return best.toOwnedSlice(gpa);
    }

    /// A trial's refusal score (any score whose name contains "Refus", so both
    /// the generation "Refusals" count and the fast-search "Refusal mass"
    /// proxy match) and its KL score (name contains "KL").
    fn refusalAndKl(trial: *const Trial) struct { refusal: ?f64, kl: ?f64 } {
        var refusal: ?f64 = null;
        var kl: ?f64 = null;
        for (trial.scores) |s| {
            if (std.mem.indexOf(u8, s.name, "Refus") != null) refusal = s.value;
            if (std.mem.indexOf(u8, s.name, "KL") != null) kl = s.value;
        }
        return .{ .refusal = refusal, .kl = kl };
    }

    /// Orders the Pareto front for display and `--trial-index`, matching
    /// heretic: fewest refusals first, ties broken by KL divergence. The
    /// first trial is thus the most strongly abliterated one, so a
    /// non-interactive `--trial-index 1 --model-action save` saves an
    /// abliterated model rather than the near-baseline low-KL end. Trials that
    /// lack the named scores fall back to the raw loss order.
    fn sortFront(self: *const Study, best: []usize, losses: []const []const f64) void {
        std.mem.sort(usize, best, Ctx{ .study = self, .losses = losses }, Ctx.lt);
    }

    const Ctx = struct {
        study: *const Study,
        losses: []const []const f64,

        fn lt(ctx: Ctx, a: usize, b: usize) bool {
            const sa = refusalAndKl(&ctx.study.trials.items[a]);
            const sb = refusalAndKl(&ctx.study.trials.items[b]);
            if (sa.refusal != null and sb.refusal != null) {
                if (sa.refusal.? != sb.refusal.?) return sa.refusal.? < sb.refusal.?;
                if (sa.kl != null and sb.kl != null and sa.kl.? != sb.kl.?) return sa.kl.? < sb.kl.?;
            }
            // Break exact ties (and order trials with no named refusal score,
            // e.g. before deferred scoring) by the raw loss vector, lexicographic.
            const la = ctx.losses[a];
            const lb = ctx.losses[b];
            for (la, 0..) |x, i| {
                if (x < lb[i]) return true;
                if (x > lb[i]) return false;
            }
            return false;
        }
    };

    /// Loss vectors of the current Pareto front of completed trials (the
    /// reference set for early stopping). The outer slice is owned by the caller.
    pub fn frontLosses(self: *const Study, gpa: Allocator) ![]const []const f64 {
        const best = try self.bestTrials(gpa);
        defer gpa.free(best);
        const out = try gpa.alloc([]const f64, best.len);
        for (best, 0..) |bi, i| out[i] = self.trials.items[bi].losses;
        return out;
    }
};

/// Turns a model id into a checkpoint file name, as heretic does.
pub fn checkpointFileName(a: Allocator, dir: []const u8, model: []const u8) ![]u8 {
    var name = std.ArrayList(u8).empty;
    for (model) |c| {
        if (std.ascii.isAlphanumeric(c) or c == '_' or c == '-') try name.append(a, c) else try name.appendSlice(a, "--");
    }
    return std.fs.path.join(a, &.{ dir, try std.fmt.allocPrint(a, "{s}.jsonl", .{name.items}) });
}

test "study round trip" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const dir_len = try tmp.dir.realPath(io, &path_buf);
    const path = try std.fs.path.join(gpa, &.{ path_buf[0..dir_len], "study.jsonl" });
    defer gpa.free(path);

    var study = try Study.open(gpa, io, path);
    defer study.deinit();
    try std.testing.expect(!study.exists());
    try study.reset("{\"type\":\"settings\",\"n_trials\":3}");
    var pmap = std.EnumMap(model_mod.Component, abliterate.Params){};
    pmap.put(.attn_o_proj, .{ .max_weight = 1, .max_weight_position = 2, .min_weight = 0.5, .min_weight_distance = 3 });
    const params = [_]f64{ 0, 1.5 };
    const losses = [_]f64{ 0.5, 0.1 };
    const scores = [_]ScoreRecord{.{ .name = "Refusals", .value = 0.5, .display = "5/10" }};
    try study.addTrial(gpa, .{ .index = 1, .params = &params, .losses = &losses, .direction_index = 3.5, .parameters = pmap, .scores = &scores });
    const losses2 = [_]f64{ 0.2, 0.3 };
    try study.addTrial(gpa, .{ .index = 2, .params = &params, .losses = &losses2, .direction_index = null, .parameters = pmap, .scores = &scores });
    const losses3 = [_]f64{ 0.9, 0.9 };
    try study.addTrial(gpa, .{ .index = 3, .params = &params, .losses = &losses3, .direction_index = null, .parameters = pmap, .scores = &scores });
    // A pruned trial with (bounded) losses that would otherwise dominate everything.
    const losses4 = [_]f64{ 0.05, 0.05 };
    try study.addTrial(gpa, .{ .index = 4, .params = &params, .losses = &losses4, .direction_index = null, .parameters = pmap, .scores = &scores, .state = .pruned });
    try study.markFinished();

    var reloaded = try Study.open(gpa, io, path);
    defer reloaded.deinit();
    try std.testing.expect(reloaded.finished);
    try std.testing.expectEqual(@as(usize, 4), reloaded.trials.items.len);
    try std.testing.expectEqual(@as(?f32, 3.5), reloaded.trials.items[0].direction_index);
    try std.testing.expectEqual(@as(f32, 0.5), reloaded.trials.items[1].parameters.get(.attn_o_proj).?.min_weight);
    try std.testing.expectEqualStrings("5/10", reloaded.trials.items[2].scores[0].display);
    try std.testing.expectEqual(State.pruned, reloaded.trials.items[3].state);
    try std.testing.expectEqual(@as(usize, 1), reloaded.prunedCount());
    try std.testing.expectEqual(@as(?i64, 3), reloaded.settingInteger("n_trials"));
    try std.testing.expectEqual(@as(?i64, null), reloaded.settingInteger("n_directions"));
    // The pruned trial feeds the sampler but is excluded from the results.
    const obs = try reloaded.observations(gpa);
    defer gpa.free(obs);
    try std.testing.expectEqual(@as(usize, 4), obs.len);
    const best = try reloaded.bestTrials(gpa);
    defer gpa.free(best);
    try std.testing.expectEqualSlices(usize, &.{ 1, 0 }, best);
    const front = try reloaded.frontLosses(gpa);
    defer gpa.free(front);
    try std.testing.expectEqual(@as(usize, 2), front.len);
    try std.testing.expectEqual(@as(f64, 0.2), front[0][0]);

    // Rescoring adds a score to a trial, replaces one of the same name, is
    // journaled, and leaves the losses and front membership alone. The display
    // order, however, follows the refusal score (heretic order: fewest refusals
    // first), so once trial 1 is rescored to 1/10 refusals it sorts ahead of
    // trial 2 (3/10), the reverse of the loss order.
    try reloaded.addScores(gpa, 2, &.{ .{ .name = "KL divergence", .value = 0.2, .display = "0.2000" }, .{ .name = "Refusals", .value = 0.3, .display = "3/10" } });
    try reloaded.addScores(gpa, 1, &.{ .{ .name = "KL divergence", .value = 0.5, .display = "0.5000" }, .{ .name = "Refusals", .value = 0.1, .display = "1/10" } });
    var again = try Study.open(gpa, io, path);
    defer again.deinit();
    try std.testing.expectEqual(@as(usize, 2), again.trials.items[1].scores.len);
    try std.testing.expectEqualStrings("3/10", Study.scoreOf(&again.trials.items[1], "Refusals").?.display);
    try std.testing.expectEqualStrings("0.2000", Study.scoreOf(&again.trials.items[1], "KL divergence").?.display);
    try std.testing.expectEqualStrings("1/10", Study.scoreOf(&again.trials.items[0], "Refusals").?.display);
    const best2 = try again.bestTrials(gpa);
    defer gpa.free(best2);
    // items[0] (trial 1, 1/10 refusals) now sorts before items[1] (trial 2, 3/10).
    try std.testing.expectEqualSlices(usize, &.{ 0, 1 }, best2);
    // Scalarised selection returns the position within best2 = {0, 1}.
    // λ = 1: trial 2 (items[1]) wins with 0.3 + 0.2 = 0.5 vs trial 1's 0.1 + 0.5 = 0.6,
    // and items[1] is at position 1. λ = 0.1: trial 1 (items[0]) wins with 0.15 vs 0.32,
    // at position 0.
    try std.testing.expectEqual(@as(?usize, 1), again.selectScalarised(best2, "Refusals", null, "KL divergence", 1.0));
    try std.testing.expectEqual(@as(?usize, 0), again.selectScalarised(best2, "Refusals", null, "KL divergence", 0.1));
    try std.testing.expectEqual(@as(?usize, null), again.selectScalarised(best2, "Refusals", null, "missing", 1.0));
    // The fallback name is used when the primary score is missing.
    try std.testing.expectEqual(@as(?usize, 0), again.selectScalarised(best2, "nope", "Refusals", "KL divergence", 0.1));
}

test "the front is ordered fewest refusals first (heretic order)" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const dir_len = try tmp.dir.realPath(io, &path_buf);
    const path = try std.fs.path.join(gpa, &.{ path_buf[0..dir_len], "study.jsonl" });
    defer gpa.free(path);

    var study = try Study.open(gpa, io, path);
    defer study.deinit();
    try study.reset("{\"type\":\"settings\"}");
    const pmap = std.EnumMap(model_mod.Component, abliterate.Params){};
    const params = [_]f64{0};
    // Three Pareto-optimal trials: as KL grows, refusals fall (the abliteration
    // trade-off). Objective order is [KL, refusals], so the raw loss order is
    // KL-ascending, i.e. most refusals first — the opposite of what we want.
    try study.addTrial(gpa, .{ .index = 1, .params = &params, .losses = &.{ 0.001, 0.90 }, .direction_index = null, .parameters = pmap, .scores = &.{ .{ .name = "KL divergence", .value = 0.001, .display = "0.0010" }, .{ .name = "Refusals", .value = 0.90, .display = "90/100" } } });
    try study.addTrial(gpa, .{ .index = 2, .params = &params, .losses = &.{ 0.02, 0.15 }, .direction_index = null, .parameters = pmap, .scores = &.{ .{ .name = "KL divergence", .value = 0.02, .display = "0.0200" }, .{ .name = "Refusals", .value = 0.15, .display = "15/100" } } });
    try study.addTrial(gpa, .{ .index = 3, .params = &params, .losses = &.{ 0.08, 0.05 }, .direction_index = null, .parameters = pmap, .scores = &.{ .{ .name = "KL divergence", .value = 0.08, .display = "0.0800" }, .{ .name = "Refusals", .value = 0.05, .display = "5/100" } } });
    const best = try study.bestTrials(gpa);
    defer gpa.free(best);
    // Fewest refusals first: trial 3 (5/100), then trial 2 (15/100), then trial 1 (90/100).
    try std.testing.expectEqualSlices(usize, &.{ 2, 1, 0 }, best);
}
