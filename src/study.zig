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
        try js.endObject();
        try self.appendLine(buf.written());
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
        sortByLosses(best.items, all);
        return best.toOwnedSlice(gpa);
    }

    /// Loss vectors of the current Pareto front of completed trials (the
    /// reference set for early stopping). The outer slice is owned by the caller.
    pub fn frontLosses(self: *const Study, gpa: Allocator) ![]const []const f64 {
        const best = try self.bestTrials(gpa);
        defer gpa.free(best);
        const out = try gpa.alloc([]const f64, best.len);
        for (best, 0..) |bi, i| out[i] = self.trials.items[bi].losses;
        return out;
    }

    fn sortByLosses(best: []usize, losses: []const []const f64) void {
        std.mem.sort(usize, best, losses, struct {
            fn lt(l: []const []const f64, a: usize, b: usize) bool {
                for (l[a], 0..) |x, i| {
                    if (x < l[b][i]) return true;
                    if (x > l[b][i]) return false;
                }
                return false;
            }
        }.lt);
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
}
