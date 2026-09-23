//! ditch: fully automatic censorship removal for language models, in Zig.
//!
//! This file is the program flow (a port of heretic's `main.py`): settings,
//! model loading, batch size and response prefix detection, residual
//! direction extraction, the TPE optimisation loop with a resumable study
//! journal, and the interactive result / save / chat menus. A memory budget
//! (`--max-ram`) switches weight access to layer streaming and a time limit
//! (`--time-limit`) turns the optimisation into a clean, resumable stop.

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
const tpe = @import("tpe.zig");
const study_mod = @import("study.zig");
const scorers = @import("scorers.zig");
const export_mod = @import("export.zig");
const gguf_export = @import("gguf_export.zig");
const budget_mod = @import("budget.zig");
const stream = @import("stream.zig");
const reproduce = @import("reproduce.zig");
const bench = @import("bench.zig");
const probe = @import("probe.zig");
const compute = @import("compute.zig");
const selftest = @import("selftest.zig");
const directions = @import("directions.zig");
const remote = @import("remote.zig");
const logo = @import("logo.zig");
const wrap = @import("wrap.zig");

const Model = model_mod.Model;
const Engine = engine_mod.Engine;
const Prompt = hf.Prompt;

/// Colour of stderr messages (errors, warnings): decided in `run` from the
/// terminal, NO_COLOR / FORCE_COLOR / DITCH_NO_COLOR / TERM and --no-color.
var color_enabled: ?bool = null;

pub const std_options: std.Options = .{ .logFn = logFn };

fn logFn(comptime level: std.log.Level, comptime scope: @EnumLiteral(), comptime format: []const u8, args: anytype) void {
    var buffer: [64]u8 = undefined;
    const locked = std.debug.lockStderr(&buffer);
    defer std.debug.unlockStderr();
    var t = locked.terminal();
    if (color_enabled) |on| {
        if (!on) t.mode = .no_color else if (t.mode == .no_color) t.mode = .escape_codes;
    }
    std.log.defaultLogFileTerminal(level, scope, format, args, t) catch {};
    // The process may exit right after an error: do not leave it in the buffer.
    t.writer.flush() catch {};
}

// ---------------------------------------------------------------------------
// Ctrl+C handling
// ---------------------------------------------------------------------------

var interrupted = std.atomic.Value(bool).init(false);

fn onSigint(_: std.posix.SIG) callconv(.c) void {
    if (interrupted.load(.seq_cst)) {
        // Second Ctrl+C: give up immediately.
        std.process.exit(130);
    }
    interrupted.store(true, .seq_cst);
    budget_mod.interrupt_requested.store(true, .seq_cst);
    // Only async-signal-safe calls here.
    const msg = "\nInterrupted: finishing the current step (press Ctrl+C again to abort now)\n";
    if (builtin.os.tag == .linux) {
        _ = std.os.linux.write(2, msg.ptr, msg.len);
    } else if (builtin.link_libc) {
        _ = std.c.write(2, msg.ptr, msg.len);
    }
}

fn installSigint() void {
    if (builtin.os.tag == .windows or builtin.os.tag == .wasi) return;
    const act: std.posix.Sigaction = .{
        .handler = .{ .handler = onSigint },
        .mask = std.posix.sigemptyset(),
        .flags = 0,
    };
    std.posix.sigaction(.INT, &act, null);
}

fn takeInterrupt() bool {
    budget_mod.interrupt_requested.store(false, .seq_cst);
    return interrupted.swap(false, .seq_cst);
}

// ---------------------------------------------------------------------------
// Console helpers
// ---------------------------------------------------------------------------

/// Messaging (banner, progress, trial logs, prompts) goes to `out` = stderr;
/// primary output (results, scores, the benchmark table, JSON, chat
/// responses) to `result` = stdout, so pipes and redirections see only the
/// results. Prompts are only shown when `interactive` (stdin is a terminal or
/// --interactive, and not --no-input); otherwise the answer's flag is named.
const Console = struct {
    /// General messaging; a discarding writer under --quiet.
    out: *Io.Writer,
    /// Essential messaging (trial one-liners, menus, "model saved"): stderr, always.
    log: *Io.Writer,
    result: *Io.Writer,
    in: *Io.Reader,
    /// Whether stderr is a terminal (progress lines are overwritten in place).
    tty: bool = false,
    /// Whether stdout is a terminal (bold headings in the help).
    tty_out: bool = false,
    tty_in: bool = false,
    interactive: bool = false,
    color: bool = false,

    /// Reads one line from stdin (without the newline); null on end of input.
    fn readLine(self: *Console) !?[]const u8 {
        try self.out.flush();
        try self.log.flush();
        const line = self.in.takeDelimiter('\n') catch |err| switch (err) {
            error.StreamTooLong => return error.LineTooLong,
            error.ReadFailed => return null,
        };
        if (interrupted.load(.seq_cst)) return null;
        return std.mem.trimEnd(u8, line orelse return null, "\r");
    }

    /// The error for a prompt that cannot be shown: names the flag to pass instead.
    fn needInput(self: *Console, question: []const u8, flag: []const u8) error{NoInput} {
        self.out.flush() catch {};
        std.log.err("\"{s}\" needs an answer, but stdin is not a terminal: pass {s} (or --interactive to read answers from stdin)", .{ question, flag });
        return error.NoInput;
    }

    /// Prints a numbered menu and returns the chosen (0-based) option, or null to exit.
    /// `flag` is the option that answers the question non-interactively.
    fn menu(self: *Console, question: []const u8, options: []const []const u8, flag: []const u8) !?usize {
        if (!self.interactive) return self.needInput(question, flag);
        while (true) {
            try self.log.print("\n{s}\n", .{question});
            for (options, 0..) |o, i| try self.log.print("  [{d}] {s}\n", .{ i + 1, o });
            try self.log.print("Enter a number (1-{d}): ", .{options.len});
            const line = (try self.readLine()) orelse return null;
            const trimmed = std.mem.trim(u8, line, " \t");
            if (trimmed.len == 0) continue;
            const n = std.fmt.parseInt(usize, trimmed, 10) catch {
                try self.log.writeAll("Please enter a number.\n");
                continue;
            };
            if (n < 1 or n > options.len) {
                try self.log.writeAll("Please enter one of the listed numbers.\n");
                continue;
            }
            return n - 1;
        }
    }

    fn ask(self: *Console, question: []const u8, flag: []const u8) !?[]const u8 {
        if (!self.interactive) return self.needInput(question, flag);
        try self.log.print("{s} ", .{question});
        return self.readLine();
    }

    /// A yes/no question (default no). Non-interactive: false, with `flag` named.
    fn confirm(self: *Console, question: []const u8, flag: []const u8) !bool {
        if (!self.interactive) {
            std.log.err("{s} Pass {s} to proceed without asking.", .{ question, flag });
            return false;
        }
        try self.log.print("{s} [y/N] ", .{question});
        const line = (try self.readLine()) orelse return false;
        const t = std.mem.trim(u8, line, " \t");
        return t.len > 0 and (t[0] == 'y' or t[0] == 'Y');
    }

    /// Writes primary output; flushed at once so a pipe sees every line.
    fn emit(self: *Console, text: []const u8) !void {
        try self.result.writeAll(text);
        try self.result.flush();
    }
};

fn formatDuration(buf: []u8, seconds_f: f64) []const u8 {
    const total: u64 = @intFromFloat(@max(0, @round(seconds_f)));
    const hours = total / 3600;
    const minutes = (total % 3600) / 60;
    const seconds = total % 60;
    if (hours > 0) return std.fmt.bufPrint(buf, "{d}h {d}m", .{ hours, minutes }) catch "?";
    if (minutes > 0) return std.fmt.bufPrint(buf, "{d}m {d}s", .{ minutes, seconds }) catch "?";
    return std.fmt.bufPrint(buf, "{d}s", .{seconds}) catch "?";
}

fn secondsSince(io: Io, start: Io.Timestamp) f64 {
    const now = Io.Timestamp.now(io, .awake);
    return @as(f64, @floatFromInt(start.durationTo(now).nanoseconds)) / 1e9;
}

/// Python-style repr of a string for printing prefixes.
fn printRepr(out: *Io.Writer, s: []const u8) !void {
    try out.writeByte('\'');
    for (s) |c| switch (c) {
        '\n' => try out.writeAll("\\n"),
        '\r' => try out.writeAll("\\r"),
        '\t' => try out.writeAll("\\t"),
        '\'' => try out.writeAll("\\'"),
        '\\' => try out.writeAll("\\\\"),
        else => try out.writeByte(c),
    };
    try out.writeByte('\'');
}

fn printScores(out: *Io.Writer, scores: []const scorers.NamedScore) !void {
    for (scores) |s| try out.print("  * {s}: {s}\n", .{ s.name, s.score.display });
}

/// The primary output of --evaluate-model: the scores as text or JSON on stdout.
fn writeScores(con: *Console, a: Allocator, model: []const u8, scores: []const scorers.NamedScore, settings: *const config.Settings) !void {
    var w: Io.Writer.Allocating = .init(a);
    if (settings.json) {
        var js: std.json.Stringify = .{ .writer = &w.writer };
        try js.beginObject();
        try js.objectField("model");
        try js.write(model);
        try js.objectField("scores");
        try js.beginObject();
        for (scores) |s| {
            try js.objectField(s.name);
            try js.beginObject();
            try js.objectField("value");
            try js.write(s.score.value);
            try js.objectField("display");
            try js.write(s.score.display);
            try js.endObject();
        }
        try js.endObject();
        try js.endObject();
        try w.writer.writeAll("\n");
    } else {
        try printScores(&w.writer, scores);
    }
    try con.emit(w.written());
}

// ---------------------------------------------------------------------------
// Application state
// ---------------------------------------------------------------------------

const App = struct {
    gpa: Allocator,
    /// The budget's allocator: used for weights, workspaces, caches and exports.
    rt_gpa: Allocator,
    arena: Allocator,
    io: Io,
    con: *Console,
    settings: *config.Settings,
    http: *hf.Http,
    cache_root: []const u8,
    pool: *const tensor.Pool,
    budget: *budget_mod.Budget,
    model: *Model,
    engine: *Engine,
    template: chat.Template,
    evaluator: *scorers.Evaluator,
    dirs: []const f32,
    space: *search.Space,
    study: *study_mod.Study,
    sampler: *tpe.Sampler,
    optimization_start: Io.Timestamp,
    start_index: usize,
    /// Inputs recorded in the reproducibility manifest on export.
    model_dir: []const u8,
    good_prompts: []const Prompt,
    bad_prompts: []const Prompt,
    /// Warm-start observations from a previous study (sampled from, never counted or shown).
    warm: []const tpe.Observation = &.{},

    fn abliterateOptions(self: *App) abliterate.Options {
        return .{
            .row_normalization = self.settings.row_normalization,
            .lora_rank = self.settings.full_normalization_lora_rank,
            .seed = self.settings.seed orelse 0,
            .expert_selection = self.settings.expert_selection,
            .debug_writer = if (self.settings.print_debug_information) self.con.out else null,
            .n_directions = self.settings.n_directions,
            .visited_experts_only = self.settings.visited_experts_only orelse self.model.warp(),
            .ablate_inputs = self.settings.ablate_inputs,
        };
    }

    /// Warm-start observations followed by this study's own trials.
    fn allObservations(self: *App, gpa: Allocator) ![]tpe.Observation {
        const own = try self.study.observations(gpa);
        defer gpa.free(own);
        const out = try gpa.alloc(tpe.Observation, self.warm.len + own.len);
        @memcpy(out[0..self.warm.len], self.warm);
        @memcpy(out[self.warm.len..], own);
        return out;
    }

    const TrialsOutcome = enum { finished, interrupted, time_limit };

    /// Runs trials `study.trials.len .. settings.n_trials`. Stops early on
    /// Ctrl+C or when the time limit expires; completed trials are always in
    /// the study journal, so either stop can be resumed.
    fn runTrials(self: *App) !TrialsOutcome {
        const out = self.con.out;
        const gpa = self.gpa;
        const n_trials = self.settings.n_trials;
        const dims = self.space.dims();
        const vector = try gpa.alloc(f64, dims);
        defer gpa.free(vector);
        if (self.study.trials.items.len > 0 and self.study.trials.items.len < n_trials) {
            try out.writeAll("\nResuming existing study.\n");
        }
        self.optimization_start = Io.Timestamp.now(self.io, .awake);
        self.start_index = self.study.trials.items.len;
        while (self.study.trials.items.len < n_trials) {
            if (takeInterrupt()) return .interrupted;
            if (self.budget.expired()) return .time_limit;
            self.runTrial(vector) catch |err| switch (err) {
                error.TimeLimitExceeded => return .time_limit,
                else => return err,
            };
            if (takeInterrupt()) return .interrupted;
        }
        const n_pruned = self.study.prunedCount();
        if (n_pruned > 0) try out.print("\n{d} of {d} trials were pruned by early stopping.\n", .{ n_pruned, self.study.trials.items.len });
        try self.study.markFinished();
        return .finished;
    }

    fn runTrial(self: *App, vector: []f64) !void {
        const out = self.con.out;
        const gpa = self.gpa;
        const n_trials = self.settings.n_trials;
        const trial_index = self.study.trials.items.len + 1;
        const observations = try self.allObservations(gpa);
        defer gpa.free(observations);
        try self.sampler.sample(gpa, observations, vector);
        var resampled = false;
        if (search.isRepeat(self.space, vector, observations)) {
            // A repeat would cost a full evaluation and teach the sampler nothing.
            self.sampler.sampleRandom(vector);
            resampled = true;
        }
        const cfg = search.decode(self.space, vector);

        const quiet = self.settings.quiet;
        const trial_start = Io.Timestamp.now(self.io, .awake);
        if (!quiet) {
            try out.print("\nRunning trial {d} of {d}...\n", .{ trial_index, n_trials });
            if (resampled) try out.writeAll("* The proposed parameters repeated an evaluated trial; sampled a random point instead.\n");
            try out.writeAll("* Parameters:\n");
            try search.describe(self.space, vector, out);
            try out.writeAll("* Resetting model...\n");
            try out.flush();
        }
        self.model.resetDeltas();
        if (!quiet) {
            try out.writeAll("* Abliterating...\n");
            try out.flush();
        }
        try search.applyTrial(self.model, self.dirs, cfg, self.abliterateOptions());
        if (!quiet) {
            try out.writeAll("* Evaluating...\n");
            try out.flush();
        }
        var scratch = std.heap.ArenaAllocator.init(gpa);
        defer scratch.deinit();
        const sa = scratch.allocator();
        // Early stopping compares against the Pareto front of completed trials.
        var front: ?[]const []const f64 = null;
        if (self.settings.early_stop) front = try self.study.frontLosses(sa);
        const scores = try self.evaluator.evaluate(sa, self.engine, out, front);
        const pruned = scorers.Evaluator.prunedOf(scores);
        const trial_seconds = secondsSince(self.io, trial_start);
        const elapsed = secondsSince(self.io, self.optimization_start);
        const done: f64 = @floatFromInt(trial_index - self.start_index);
        const remaining = elapsed / done * @as(f64, @floatFromInt(n_trials - trial_index));
        var buf: [64]u8 = undefined;
        if (quiet) {
            // One line per trial: "trial 3/50: kl=0.0123 refusals=4/100 (12.3 s, 9 min left)".
            const log = self.con.log;
            try log.print("trial {d}/{d}:", .{ trial_index, n_trials });
            for (scores) |s| try log.print(" {s}={s}", .{ s.name, s.score.display });
            try log.print(" ({d:.1} s", .{trial_seconds});
            if (trial_index < n_trials) try log.print(", {s} left", .{formatDuration(&buf, remaining)});
            try log.writeAll(")\n");
            try log.flush();
        } else {
            try printScores(out, scores);
            try out.print("\nElapsed time: {s}\n", .{formatDuration(&buf, elapsed)});
            if (trial_index < n_trials) try out.print("Estimated remaining time: {s}\n", .{formatDuration(&buf, remaining)});
        }
        if (self.settings.print_debug_information) {
            try self.budget.report().print(out, "Memory");
            if (self.model.expert_cache) |c| try c.stats().print(out, "Expert cache");
        }
        try out.flush();

        const losses = try self.evaluator.objectiveLosses(sa, scores);
        const records = try sa.alloc(study_mod.ScoreRecord, scores.len);
        for (scores, 0..) |s, i| records[i] = .{ .name = s.name, .value = s.score.value, .display = s.score.display };
        const trial = study_mod.Trial{
            .index = trial_index,
            .params = vector,
            .losses = losses,
            .direction_index = cfg.direction_index,
            .parameters = cfg.parameters,
            .scores = records,
            .state = if (pruned != null) .pruned else .complete,
        };
        try self.study.addTrial(gpa, trial);
        if (self.settings.json_log) |path| try appendJsonLog(gpa, self.io, path, &trial, trial_seconds);
    }

    /// `--json-log`: appends the trial as one JSON object (the journal's fields
    /// plus the wall-clock time of the trial) to `path`.
    fn appendJsonLog(gpa: Allocator, io: Io, path: []const u8, trial: *const study_mod.Trial, seconds: f64) !void {
        var buf: Io.Writer.Allocating = .init(gpa);
        defer buf.deinit();
        var js: std.json.Stringify = .{ .writer = &buf.writer };
        try js.beginObject();
        try study_mod.Study.writeTrialFields(&js, trial);
        try js.objectField("seconds");
        try js.write(seconds);
        try js.endObject();
        try buf.writer.writeAll("\n");
        const file = try Io.Dir.cwd().createFile(io, path, .{ .truncate = false });
        defer file.close(io);
        try file.writePositionalAll(io, buf.written(), try file.length(io));
    }

    /// Prints the time-limit notice; the study journal holds every completed trial.
    fn printTimeLimit(self: *App) !void {
        const r = self.budget.report();
        try self.con.log.print("\nTime limit reached ({f} elapsed of {f}). The {d} completed trial(s) have been saved; run ditch again with --checkpoint-action continue to resume.\n", .{ budget_mod.fmtDuration(r.elapsed), budget_mod.fmtDuration(r.time_limit orelse r.elapsed), self.study.trials.items.len });
        try self.con.log.flush();
    }

    fn restoreTrial(self: *App, trial: *const study_mod.Trial) !void {
        const out = self.con.out;
        try out.print("\nRestoring model from trial {d}...\n", .{trial.index});
        try out.writeAll("* Parameters:\n");
        try search.describe(self.space, trial.params, out);
        try out.writeAll("* Resetting model...\n");
        try out.flush();
        self.model.resetDeltas();
        try out.writeAll("* Abliterating...\n");
        try out.flush();
        const cfg = search.decode(self.space, trial.params);
        try search.applyTrial(self.model, self.dirs, cfg, self.abliterateOptions());
    }

    fn trialTitle(self: *App, a: Allocator, trial: *const study_mod.Trial) ![]const u8 {
        _ = self;
        var w: Io.Writer.Allocating = .init(a);
        try w.writer.print("[Trial {d: >3}]", .{trial.index});
        for (trial.scores, 0..) |s, i| {
            try w.writer.print("{s} {s}: {s}", .{ if (i == 0) "" else ",", s.name, s.display });
        }
        return w.toOwnedSlice();
    }

    /// The results / trial selection loop. Returns when the user exits.
    fn resultsLoop(self: *App) !void {
        const out = self.con.out;
        const gpa = self.gpa;
        var trial_index_used = false;
        var additional_used = false;
        var results_shown = false;
        while (true) {
            if (self.study.trials.items.len == 0) {
                try out.writeAll("\nNo trials have been completed.\n");
                return;
            }
            var menu_arena = std.heap.ArenaAllocator.init(gpa);
            defer menu_arena.deinit();
            const ma = menu_arena.allocator();
            const best = try self.study.bestTrials(ma);
            try self.rescoreDeferred(best);
            var auto_pick: ?usize = null;
            if (self.settings.select == .auto) {
                auto_pick = self.study.selectScalarised(best, self.settings.keyword_rate.score_name, "Refusal mass", "KL divergence", self.settings.select_lambda);
                // The chosen trial is listed first; the rest keep their order.
                if (auto_pick) |p| if (p > 0) {
                    const chosen = best[p];
                    std.mem.copyBackwards(usize, best[1 .. p + 1], best[0..p]);
                    best[0] = chosen;
                };
            }
            var options = std.ArrayList([]const u8).empty;
            for (best) |bi| try options.append(ma, try self.trialTitle(ma, &self.study.trials.items[bi]));
            try options.append(ma, "Run additional trials");
            try options.append(ma, "Exit");

            try out.writeAll("\nOptimization finished!\n");
            if (!results_shown) {
                results_shown = true;
                try self.writeResults(ma, best);
            }
            if (self.settings.trial_index == null) {
                try out.writeAll("\nThe following trials resulted in Pareto optimal combinations of the optimization objectives. After selecting a trial, you will be able to save the model or chat with it to test how well it works. You can return to this menu later to select a different trial.\n");
            }
            if (self.settings.select == .auto) {
                if (auto_pick != null) {
                    try out.print("\nListed first: the trial minimising refusals + {d} x KL divergence (--select auto).\n", .{self.settings.select_lambda});
                } else {
                    try out.writeAll("\n--select auto needs a refusal score and a KL divergence per trial; none was found, keeping the Pareto order.\n");
                }
            }

            var choice: usize = undefined;
            if (self.settings.n_additional_trials != null and !additional_used) {
                additional_used = true;
                choice = best.len; // "Run additional trials"
            } else if (self.settings.trial_index) |ti| {
                if (trial_index_used) return;
                trial_index_used = true;
                if (ti < 1 or ti > best.len) {
                    std.log.err("trial index {d} is out of range (1-{d})", .{ ti, best.len });
                    return error.InvalidTrialIndex;
                }
                choice = ti - 1;
                try out.print("\nSelected: {s}\n", .{options.items[choice]});
            } else {
                choice = (try self.con.menu("Which trial do you want to use?", options.items, "--trial-index <n>")) orelse return;
            }

            if (choice == best.len + 1) return;
            if (choice == best.len) {
                var n_additional: usize = 0;
                if (self.settings.n_additional_trials) |n| {
                    n_additional = n;
                } else {
                    while (true) {
                        const line = (try self.con.ask("How many additional trials do you want to run?", "--n-additional-trials <n>")) orelse return;
                        const t = std.mem.trim(u8, line, " \t");
                        if (t.len == 0) break;
                        n_additional = std.fmt.parseInt(usize, t, 10) catch {
                            try out.writeAll("Please enter a number.\n");
                            continue;
                        };
                        if (n_additional > 0) break;
                        try out.writeAll("Please enter a number greater than 0.\n");
                    }
                }
                if (n_additional == 0) continue;
                self.settings.n_trials = self.study.trials.items.len + n_additional;
                self.study.markUnfinished();
                if (try self.runTrials() == .time_limit) {
                    try self.printTimeLimit();
                    return;
                }
                continue;
            }

            const trial = &self.study.trials.items[best[choice]];
            try self.restoreTrial(trial);
            const back = try self.modelLoop(trial);
            if (!back) return;
        }
    }

    /// The primary output of a study: the Pareto-optimal trials, as text
    /// (one per line) or, with --json, as one JSON document.
    fn writeResults(self: *App, a: Allocator, best: []const usize) !void {
        var w: Io.Writer.Allocating = .init(a);
        if (self.settings.json) {
            var js: std.json.Stringify = .{ .writer = &w.writer };
            try js.beginObject();
            try js.objectField("model");
            try js.write(self.settings.model);
            try js.objectField("trials");
            try js.write(self.study.trials.items.len);
            try js.objectField("parameters");
            try js.write(self.space.space.names);
            try js.objectField("pareto");
            try js.beginArray();
            for (best) |bi| {
                try js.beginObject();
                try study_mod.Study.writeTrialFields(&js, &self.study.trials.items[bi]);
                try js.endObject();
            }
            try js.endArray();
            try js.endObject();
            try w.writer.writeAll("\n");
        } else {
            if (!self.settings.plain) try w.writer.writeAll("Pareto-optimal trials:\n");
            const bold = self.con.color and self.con.tty_out and !self.settings.plain;
            for (best) |bi| {
                const t = &self.study.trials.items[bi];
                if (bold) try w.writer.print("\x1b[1mtrial {d}:\x1b[0m", .{t.index}) else try w.writer.print("trial {d}:", .{t.index});
                for (t.scores) |s| try w.writer.print(" {s}={s}", .{ s.name, s.display });
                try w.writer.writeAll("\n");
            }
        }
        try self.con.emit(w.written());
    }

    /// `--fast-search`: runs the deferred keyword scorer on every Pareto
    /// candidate that has not been scored by it yet, so the results menu
    /// shows real refusal counts, and journals the scores.
    fn rescoreDeferred(self: *App, best: []const usize) !void {
        const deferred = self.evaluator.deferred orelse return;
        const out = self.con.out;
        var pending: usize = 0;
        for (best) |bi| pending += @intFromBool(study_mod.Study.scoreOf(&self.study.trials.items[bi], deferred.name) == null);
        if (pending == 0) return;
        try out.print("\nScoring {d} Pareto-optimal trial(s) with the deferred {s} scorer (generation)...\n", .{ pending, deferred.name });
        try out.flush();
        for (best) |bi| {
            const trial = &self.study.trials.items[bi];
            if (study_mod.Study.scoreOf(trial, deferred.name) != null) continue;
            try self.restoreTrial(trial);
            var scratch = std.heap.ArenaAllocator.init(self.gpa);
            defer scratch.deinit();
            const sa = scratch.allocator();
            const s = (try self.evaluator.deferredScore(sa, self.engine, out)) orelse return;
            try out.print("  * {s}: {s}\n", .{ s.name, s.score.display });
            try out.flush();
            try self.study.addScores(self.gpa, trial.index, &.{.{ .name = s.name, .value = s.score.value, .display = s.score.display }});
        }
    }

    /// The "what do you want to do with the model" loop. Returns true to go back to trial selection.
    fn modelLoop(self: *App, trial: *const study_mod.Trial) !bool {
        const out = self.con.out;
        const options = [_][]const u8{ "Save the model to a local folder", "Chat with the model", "Return to the trial selection menu", "Exit" };
        var forced_done = false;
        while (true) {
            var action: usize = undefined;
            if (self.settings.model_action) |ma| {
                if (forced_done) return false;
                forced_done = true;
                if (std.mem.eql(u8, ma, "save")) action = 0 else if (std.mem.eql(u8, ma, "chat")) action = 1 else if (std.mem.eql(u8, ma, "exit")) action = 3 else {
                    std.log.err("unknown model action: {s} (expected save, chat or exit)", .{ma});
                    return error.InvalidModelAction;
                }
            } else {
                action = (try self.con.menu("What do you want to do with the decensored model?", &options, "--model-action save|chat|exit")) orelse return false;
            }
            switch (action) {
                0 => self.saveModel(trial) catch |err| {
                    try out.print("Error while saving the model: {s}\n", .{@errorName(err)});
                    if (err == error.TimeLimitExceeded) try out.print("The export directory is incomplete (it contains {s}) and will be refused on load.\n", .{model_mod.export_incomplete_marker});
                    if (self.settings.model_action != null) return if (err == error.TimeLimitExceeded) error.ExportIncomplete else err;
                },
                1 => try self.chatLoop(),
                2 => return true,
                else => return false,
            }
        }
    }

    const ExportFormat = enum { hf, gguf, both };

    fn exportFormat(self: *App) !ExportFormat {
        const s = self.settings.export_format orelse return if (self.model.gguf != null) .gguf else .hf;
        if (std.ascii.eqlIgnoreCase(s, "hf") or std.ascii.eqlIgnoreCase(s, "safetensors")) return .hf;
        if (std.ascii.eqlIgnoreCase(s, "gguf")) return .gguf;
        if (std.ascii.eqlIgnoreCase(s, "both")) return .both;
        std.log.err("unknown export format: {s} (expected hf, gguf or both)", .{s});
        return error.InvalidExportFormat;
    }

    /// The GGUF matrix dtype: null keeps the source types.
    fn ggufDtype(self: *App) !?tensor.DType {
        const s = self.settings.gguf_dtype orelse return if (self.model.gguf != null) null else .f16;
        if (std.ascii.eqlIgnoreCase(s, "source") or std.ascii.eqlIgnoreCase(s, "auto")) return null;
        const d = tensor.DType.parse(s) orelse {
            std.log.err("unknown gguf dtype: {s} (expected f16, bf16, f32, q8_0, q4_0, q4_1, q5_0, q5_1 or source)", .{s});
            return error.InvalidExportDtype;
        };
        if (d.isQuantized() and !@import("quant.zig").canQuantize(d)) {
            std.log.err("ditch cannot quantise to {s}; use q8_0, q4_0, q4_1, q5_0 or q5_1", .{d.safetensorsName()});
            return error.InvalidExportDtype;
        }
        return d;
    }

    fn parseExportDtype(s: []const u8) ?tensor.DType {
        if (std.ascii.eqlIgnoreCase(s, "bf16") or std.ascii.eqlIgnoreCase(s, "bfloat16")) return .bf16;
        if (std.ascii.eqlIgnoreCase(s, "f16") or std.ascii.eqlIgnoreCase(s, "float16") or std.ascii.eqlIgnoreCase(s, "fp16")) return .f16;
        if (std.ascii.eqlIgnoreCase(s, "f32") or std.ascii.eqlIgnoreCase(s, "float32") or std.ascii.eqlIgnoreCase(s, "fp32")) return .f32;
        return null;
    }

    fn modelCard(self: *App, a: Allocator, trial: *const study_mod.Trial) ![]const u8 {
        var w: Io.Writer.Allocating = .init(a);
        const o = &w.writer;
        const model_id = remote.hubId(self.settings.model);
        const is_hf = std.mem.indexOfScalar(u8, model_id, '/') != null and !remote.isRemoteId(model_id) and !hf.isLocalDir(self.io, model_id);
        try o.writeAll("---\ntags:\n- ditch\n- heretic\n- uncensored\n- decensored\n- abliterated\n---\n\n");
        try o.writeAll("# This is a decensored version of ");
        if (is_hf) try o.print("[{s}](https://huggingface.co/{s})", .{ model_id, model_id }) else try o.writeAll("a model");
        try o.print(", made using [ditch](https://github.com/plyght/ditch) v{s} (a Zig rebuild of [Heretic](https://heretic-project.org))\n\n", .{config.version});
        try o.writeAll("## Abliteration parameters\n\n| Parameter | Value |\n| :-------- | :---: |\n");
        if (trial.direction_index) |di| try o.print("| **direction_index** | {d:.2} |\n", .{di}) else try o.writeAll("| **direction_index** | per layer |\n");
        if (self.settings.n_directions > 1) try o.print("| **n_directions** | {d} |\n", .{self.settings.n_directions});
        if (self.settings.direction_method != .mean) try o.print("| **direction_method** | {s} |\n", .{@tagName(self.settings.direction_method)});
        if (self.settings.direction_token_window > 1) try o.print("| **direction_token_window** | {d} |\n", .{self.settings.direction_token_window});
        if (self.settings.ablate_inputs) try o.writeAll("| **ablate_inputs** | true |\n");
        if (self.settings.kl_tokens > 1) try o.print("| **kl_tokens** | {d} |\n", .{self.settings.kl_tokens});
        for (model_mod.Component.all) |comp| {
            const p = trial.parameters.get(comp) orelse continue;
            try o.print("| **{s}.max_weight** | {d:.2} |\n", .{ comp.name(), p.max_weight });
            try o.print("| **{s}.max_weight_position** | {d:.2} |\n", .{ comp.name(), p.max_weight_position });
            try o.print("| **{s}.min_weight** | {d:.2} |\n", .{ comp.name(), p.min_weight });
            try o.print("| **{s}.min_weight_distance** | {d:.2} |\n", .{ comp.name(), p.min_weight_distance });
        }
        try o.writeAll("\n## Performance\n\n| Metric | This model | Original model");
        if (is_hf) try o.print(" ([{s}](https://huggingface.co/{s}))", .{ model_id, model_id });
        try o.writeAll(" |\n| :----- | :--------: | :---------------------------: |\n");
        for (trial.scores) |s| {
            var baseline: []const u8 = "?";
            for (self.evaluator.baseline) |b| if (std.mem.eql(u8, b.name, s.name)) {
                baseline = b.score.display;
            };
            try o.print("| **{s}** | {s} | {s} |\n", .{ s.name, s.display, baseline });
        }
        try o.writeAll("\n-----\n\n");
        try o.writeAll("Generated with ditch. The abliteration directions were computed from the difference of mean residuals between \"harmful\" and \"harmless\" prompts and projected out of the attention output and MLP down-projection weights, with the weight kernel parameters above chosen by multi-objective TPE optimisation.\n");
        return w.toOwnedSlice();
    }

    fn saveModel(self: *App, trial: *const study_mod.Trial) !void {
        const out = self.con.out;
        var dir: []const u8 = undefined;
        if (self.settings.save_directory) |d| {
            dir = d;
        } else {
            const line = (try self.con.ask("Path to the folder:", "--save-directory <path> (-o)")) orelse return;
            const t = std.mem.trim(u8, line, " \t");
            if (t.len == 0) return;
            dir = try self.arena.dupe(u8, t);
        }
        var dtype: ?tensor.DType = null;
        if (self.settings.export_dtype) |d| {
            dtype = parseExportDtype(d) orelse {
                std.log.err("unknown export dtype: {s} (expected bf16, f16 or f32)", .{d});
                return error.InvalidExportDtype;
            };
        }
        if (!self.settings.force and try dirHasEntries(self.io, dir)) {
            var q: [std.fs.max_path_bytes + 96]u8 = undefined;
            const question = try std.fmt.bufPrint(&q, "{s} is not empty; overwrite its files?", .{dir});
            if (!try self.con.confirm(question, "--force (-f)")) return error.DirectoryNotEmpty;
        }
        try out.writeAll("Saving merged model...\n");
        try out.flush();
        var scratch = std.heap.ArenaAllocator.init(self.gpa);
        defer scratch.deinit();
        const sa = scratch.allocator();
        const manifest = try reproduce.build(sa, self.io, .{
            .settings = self.settings,
            .model = self.model,
            .template = @tagName(self.template),
            .good_prompts = self.good_prompts,
            .bad_prompts = self.bad_prompts,
            .keyword_rate_prompts = reproduce.scorerPrompts(self.evaluator, .keyword_rate),
            .kl_divergence_prompts = reproduce.scorerPrompts(self.evaluator, .kl_divergence),
            .space = self.space,
            .trial = trial,
            .baseline = self.evaluator.baseline,
        });
        var card: Io.Writer.Allocating = .init(sa);
        try card.writer.writeAll(try self.modelCard(sa, trial));
        try reproduce.markdown(&manifest, &card.writer);
        const format = try self.exportFormat();
        const gguf_dtype = try self.ggufDtype();
        if (format == .hf or format == .both) {
            try export_mod.saveModel(self.rt_gpa, self.io, self.model, dir, .{
                .max_shard_size = self.settings.max_shard_size,
                .export_dtype = dtype,
                .readme_body = card.written(),
            }, out);
        }
        if (format == .gguf or format == .both) {
            try gguf_export.saveGguf(self.rt_gpa, self.io, self.model, dir, .{
                .dtype = gguf_dtype,
                .name = std.fs.path.basename(std.mem.trimEnd(u8, dir, "/")),
                .readme_body = card.written(),
            }, out);
        }
        try out.print("* Writing {s}...\n", .{reproduce.file_name});
        try reproduce.writeFile(self.gpa, self.io, &manifest, dir);
        try self.con.log.print("Model saved to {s}.\n", .{dir});
        try self.con.log.flush();
        try out.flush();
        self.validateExport(dir) catch |err| {
            try out.print("* Validation failed: {s} (the model was saved; verify it by hand)\n", .{@errorName(err)});
            try out.flush();
        };
    }

    /// Reloads the export through the streamed path and compares first-token
    /// logits with the in-memory (delta) model on a few good prompts.
    fn validateExport(self: *App, dir: []const u8) !void {
        const out = self.con.out;
        // Metadata of the reloaded model (tokenizer, config) is not budgeted;
        // its weights and buffers are, through `self.budget`.
        const gpa = self.gpa;
        try out.writeAll("Validating the exported model...\n");
        try out.flush();
        const n = @min(self.good_prompts.len, 4);
        var ids = std.ArrayList([]const u32).empty;
        defer {
            for (ids.items) |x| gpa.free(x);
            ids.deinit(gpa);
        }
        for (self.good_prompts[0..n]) |p| try ids.append(gpa, try self.engine.encodePrompt(gpa, p));
        // Two models run under the budget for a moment: drop the cached workspace.
        self.engine.releaseWorkspace();
        const v = try stream.validateExport(gpa, self.io, self.pool, self.model, dir, ids.items, self.budget);
        try out.print("* {d} prompts: max |Δ| first-token logit {d:.4}, argmax agreement {d:.0}%\n", .{ v.prompts, v.max_abs_diff, 100.0 * v.argmax_match });
        if (v.argmax_match < 1.0) try out.writeAll("* Warning: the exported model does not reproduce the abliterated model on every prompt.\n");
        try out.flush();
    }

    fn chatLoop(self: *App) !void {
        const out = self.con.out;
        const gpa = self.gpa;
        if (!self.con.interactive) {
            std.log.err("chat needs messages from a terminal (or --interactive to read them from stdin)", .{});
            return error.NoInput;
        }
        try out.writeAll("\nType a message and press Enter. An empty line or /exit returns to the menu; Ctrl+C stops a response.\n");
        var history = std.ArrayList(chat.Message).empty;
        defer {
            for (history.items) |m| if (m.role != .system) gpa.free(m.content);
            history.deinit(gpa);
        }
        try history.append(gpa, .{ .role = .system, .content = self.settings.system_prompt });
        while (true) {
            try out.writeAll("\n> ");
            const line = (try self.con.readLine()) orelse break;
            const msg = std.mem.trim(u8, line, " \t");
            if (msg.len == 0 or std.mem.eql(u8, msg, "/exit") or std.mem.eql(u8, msg, "/quit")) break;
            try history.append(gpa, .{ .role = .user, .content = try gpa.dupe(u8, msg) });
            try self.con.emit("Assistant: ");
            const response = try self.streamResponse(history.items);
            try history.append(gpa, .{ .role = .assistant, .content = response });
            try self.con.emit("\n");
        }
        _ = takeInterrupt();
    }

    fn argmax(x: []const f32) u32 {
        var best: usize = 0;
        for (x, 0..) |v, i| if (v > x[best]) {
            best = i;
        };
        return @intCast(best);
    }

    /// Generates a response for the conversation, printing tokens (to stdout) as they are produced.
    fn streamResponse(self: *App, messages: []const chat.Message) ![]u8 {
        const out = self.con.result;
        const gpa = self.rt_gpa;
        const model = self.model;
        const c = &model.config;
        const max_new: usize = 1024;

        const body = try chat.render(gpa, self.template, messages);
        defer gpa.free(body);
        // The same BOS rules as the study's prompts (see `Engine`).
        const text = try std.mem.concat(gpa, u8, &.{ self.engine.bos_prefix, body });
        defer gpa.free(text);
        const ids = try model.tokenizer.encode(gpa, text, self.engine.add_special);
        defer gpa.free(ids);
        const max_len = ids.len + max_new + 1;
        const kv_bytes = model_mod.KvCache.bytesFor(c.num_layers, 1, max_len, c.kvDim());
        var ws = try model_mod.Workspace.init(gpa, c, stream.workspaceRows(model, @max(ids.len, 1), 1, kv_bytes), 1);
        defer ws.deinit();
        var cache = try model_mod.KvCache.initFor(model, gpa, 1, max_len);
        defer cache.deinit();
        const logits = try gpa.alloc(f32, c.vocab_size);
        defer gpa.free(logits);
        const prompts = [_][]const u32{ids};
        try model_mod.prefill(model, &ws, &cache, &prompts, logits, null);

        var generated = std.ArrayList(u32).empty;
        defer generated.deinit(gpa);
        var printed = std.ArrayList(u8).empty;
        defer printed.deinit(gpa);
        var pos = ids.len;
        var token = argmax(logits);
        const decode_start = Io.Timestamp.now(self.io, .awake);
        var step: usize = 0;
        while (step < max_new) : (step += 1) {
            if (model.isEos(token)) break;
            if (interrupted.load(.seq_cst)) {
                _ = takeInterrupt();
                break;
            }
            try generated.append(gpa, token);
            // Decode the whole response and print the new suffix, holding back incomplete UTF-8.
            const full = try model.tokenizer.decode(gpa, generated.items, true);
            defer gpa.free(full);
            if (std.mem.startsWith(u8, full, printed.items)) {
                var end = full.len;
                var back: usize = 0;
                while (end > printed.items.len and back < 4 and (full[end - 1] & 0xC0) == 0x80) : (back += 1) end -= 1;
                if (end > printed.items.len and back > 0 and (full[end - 1] & 0x80) != 0) {
                    const lead = full[end - 1];
                    const need: usize = if (lead & 0xE0 == 0xC0) 1 else if (lead & 0xF0 == 0xE0) 2 else if (lead & 0xF8 == 0xF0) 3 else 0;
                    if (need == back) end = full.len else end -= 1;
                }
                if (end > printed.items.len) {
                    try out.writeAll(full[printed.items.len..end]);
                    try out.flush();
                    try printed.appendSlice(gpa, full[printed.items.len..end]);
                }
            }
            if (pos >= cache.max_len) break;
            const toks = [_]u32{token};
            const rows = [_]model_mod.Row{.{ .b = 0, .pos = pos }};
            const lr = [_]usize{0};
            try model_mod.forward(model, &ws, &cache, &toks, &rows, .{ .logit_rows = &lr });
            if (model.expert_cache) |ec| ec.endDecodeStep();
            pos += 1;
            token = argmax(ws.logits[0..c.vocab_size]);
        }
        const full = try model.tokenizer.decode(gpa, generated.items, true);
        defer gpa.free(full);
        if (full.len > printed.items.len and std.mem.startsWith(u8, full, printed.items)) {
            try out.writeAll(full[printed.items.len..]);
        }
        try out.flush();
        const decode_seconds = secondsSince(self.io, decode_start);
        if (generated.items.len > 0) {
            try self.con.out.print("\n[{d} tokens, {d:.1} tokens/s]", .{ generated.items.len, @as(f64, @floatFromInt(generated.items.len)) / @max(decode_seconds, 1e-9) });
            try self.con.out.flush();
        }
        return gpa.dupe(u8, full);
    }
};

// ---------------------------------------------------------------------------
// Pipeline steps
// ---------------------------------------------------------------------------

fn detectBatchSize(gpa: Allocator, io: Io, engine: *Engine, settings: *config.Settings, good_prompts: []const Prompt, out: *Io.Writer) !void {
    try out.writeAll("\nDetermining optimal batch size...\n");
    var batch_size: usize = 1;
    var best_batch_size: usize = 1;
    var best_performance: f64 = -1;
    const longest = try maxPromptTokens(gpa, engine, good_prompts);
    while (batch_size <= settings.max_batch_size) {
        // Under a memory budget, stop before a batch whose KV cache and
        // workspace could not be resident (it would only spill and slow down).
        if (batch_size > 1) {
            const c = &engine.model.config;
            const kv = model_mod.KvCache.bytesFor(c.num_layers, batch_size, longest + settings.max_response_length + 1, c.kvDim());
            const rows = batch_size * longest;
            if (engine.workspaceRows(rows, batch_size, kv) < rows) {
                try out.print("* Batch size {d} would exceed the memory budget; keeping {d}\n", .{ batch_size, best_batch_size });
                break;
            }
        }
        try out.print("* Trying batch size {d}... ", .{batch_size});
        try out.flush();
        const prompts = try gpa.alloc(Prompt, batch_size);
        defer gpa.free(prompts);
        for (prompts, 0..) |*p, i| p.* = good_prompts[i % good_prompts.len];
        engine.batch_size = batch_size;
        // Warm-up run (allocates the workspace), then the timed run.
        const warm = engine.getResponses(gpa, prompts, false) catch |err| {
            if (batch_size == 1) return err;
            try out.print("Failed ({s})\n", .{@errorName(err)});
            break;
        };
        for (warm) |r| gpa.free(r);
        gpa.free(warm);
        const start = Io.Timestamp.now(io, .awake);
        const responses = engine.getResponses(gpa, prompts, false) catch |err| {
            if (batch_size == 1) return err;
            try out.print("Failed ({s})\n", .{@errorName(err)});
            break;
        };
        const seconds = secondsSince(io, start);
        var tokens: usize = 0;
        for (responses) |r| {
            const ids = try engine.model.tokenizer.encode(gpa, r, false);
            tokens += ids.len;
            gpa.free(ids);
            gpa.free(r);
        }
        gpa.free(responses);
        const performance = @as(f64, @floatFromInt(tokens)) / @max(seconds, 1e-9);
        try out.print("Ok ({d:.0} tokens/s)\n", .{performance});
        try out.flush();
        if (performance > best_performance) {
            best_performance = performance;
            best_batch_size = batch_size;
        } else {
            // Throughput stopped improving: larger batches only cost memory.
            break;
        }
        batch_size *= 2;
    }
    settings.batch_size = best_batch_size;
    engine.batch_size = best_batch_size;
    try out.print("* Chosen batch size: {d}\n", .{best_batch_size});
}

fn detectResponsePrefix(gpa: Allocator, arena: Allocator, engine: *Engine, settings: *config.Settings, good_prompts: []const Prompt, bad_prompts: []const Prompt, out: *Io.Writer) !void {
    try out.writeAll("\nChecking for common response prefix...\n");
    try out.flush();
    var check = std.ArrayList(Prompt).empty;
    defer check.deinit(gpa);
    try check.appendSlice(gpa, good_prompts[0..@min(100, good_prompts.len)]);
    try check.appendSlice(gpa, bad_prompts[0..@min(100, bad_prompts.len)]);

    // Does the chat template itself open a chain-of-thought block?
    const dummy_msgs = [_]chat.Message{.{ .role = .user, .content = "This is a dummy prompt." }};
    const dummy = try chat.render(gpa, engine.template, &dummy_msgs);
    defer gpa.free(dummy);
    const dummy_trimmed = std.mem.trimEnd(u8, dummy, " \t\r\n");
    var cot_skip_applied = false;
    for (settings.chain_of_thought_skips) |pair| {
        if (std.mem.endsWith(u8, dummy_trimmed, pair[0])) {
            settings.response_prefix = pair[1];
            try out.writeAll("* Closed Chain-of-Thought block: ");
            try printRepr(out, pair[1]);
            try out.writeAll("\n");
            cot_skip_applied = true;
            break;
        }
    }

    if (settings.response_prefix == null) {
        const responses = try engine.getResponses(gpa, check.items, false);
        defer {
            for (responses) |r| gpa.free(r);
            gpa.free(responses);
        }
        const prefix = std.mem.trimEnd(u8, engine_mod.commonPrefix(responses), " ");
        settings.response_prefix = try arena.dupe(u8, prefix);
        if (prefix.len > 0) {
            try out.writeAll("* Prefix found: ");
            try printRepr(out, prefix);
            try out.writeAll("\n");
            for (settings.chain_of_thought_skips) |pair| {
                if (std.mem.startsWith(u8, prefix, pair[0])) {
                    settings.response_prefix = pair[1];
                    try out.writeAll("* Closed Chain-of-Thought block: ");
                    try printRepr(out, pair[1]);
                    try out.writeAll("\n");
                    cot_skip_applied = true;
                    break;
                }
            }
        } else {
            try out.writeAll("* None found\n");
        }
    }

    if (cot_skip_applied) {
        try out.writeAll("* Rechecking with prefix...\n");
        try out.flush();
        const responses = try engine.getResponses(gpa, check.items, false);
        defer {
            for (responses) |r| gpa.free(r);
            gpa.free(responses);
        }
        const additional = std.mem.trimEnd(u8, engine_mod.commonPrefix(responses), " ");
        if (additional.len > 0) {
            settings.response_prefix = try std.fmt.allocPrint(arena, "{s}{s}", .{ settings.response_prefix.?, additional });
            try out.writeAll("* Extended prefix found: ");
            try printRepr(out, settings.response_prefix.?);
            try out.writeAll("\n");
        }
    }
    try out.flush();
}

fn printResidualGeometry(out: *Io.Writer, good: []const f32, bad: []const f32, entries: usize, hidden: usize, sep: ?directions.Separation) !void {
    try out.writeAll("\nResidual geometry (per layer entry; entry 0 = embeddings, entry L = output of layer L-1):\n");
    try out.writeAll("  entry   cos(good,bad)     |good|      |bad|   |bad-good|");
    if (sep != null) try out.writeAll("     AUROC        d'");
    try out.writeAll("\n");
    for (0..entries) |e| {
        const g = good[e * hidden ..][0..hidden];
        const b = bad[e * hidden ..][0..hidden];
        var dot: f64 = 0;
        var gn: f64 = 0;
        var bn: f64 = 0;
        var dn: f64 = 0;
        for (g, b) |x, y| {
            dot += @as(f64, x) * y;
            gn += @as(f64, x) * x;
            bn += @as(f64, y) * y;
            dn += (@as(f64, y) - x) * (@as(f64, y) - x);
        }
        const cos = dot / @max(@sqrt(gn) * @sqrt(bn), 1e-12);
        try out.print("  {d: >5}   {d: >13.4}   {d: >8.3}   {d: >8.3}   {d: >10.3}", .{ e, cos, @sqrt(gn), @sqrt(bn), @sqrt(dn) });
        if (sep) |s| try out.print("   {d: >7.4}   {d: >7.3}", .{ s.auroc[e], s.dprime[e] });
        try out.writeAll("\n");
    }
    if (sep != null) try out.writeAll("  AUROC / d' = separation of the good and bad prompts by their projection onto the entry's direction.\n");
}

/// The resolved `direction_index` bounds: heretic's fractions of the last
/// layer, or the layers whose separation is highest (`--direction-range auto`).
fn resolveDirectionRange(settings: *const config.Settings, model: *const Model, sep: ?directions.Separation) search.IndexRange {
    const last: f64 = @floatFromInt(model.config.num_layers - 1);
    return switch (settings.direction_range) {
        .fixed => |f| .{ .low = f.low * last, .high = f.high * last },
        .auto => blk: {
            const r = directions.autoRange(sep.?.auroc);
            break :blk .{ .low = r.low, .high = r.high };
        },
    };
}

/// Names and optimisation directions of the scorers, e.g. "kl_divergence:minimize,keyword_rate:minimize".
fn scorerSetString(a: Allocator, scorers_cfg: []const config.ScorerConfig) ![]const u8 {
    var w: Io.Writer.Allocating = .init(a);
    for (scorers_cfg, 0..) |sc, i| try w.writer.print("{s}{s}:{s}", .{ if (i == 0) "" else ",", @tagName(sc.kind), @tagName(sc.optimization) });
    return w.toOwnedSlice();
}

fn settingsSnapshot(a: Allocator, settings: *const config.Settings, model: *const Model, range: search.IndexRange) ![]const u8 {
    var w: Io.Writer.Allocating = .init(a);
    var js: std.json.Stringify = .{ .writer = &w.writer };
    try js.beginObject();
    try js.objectField("type");
    try js.write("settings");
    try js.objectField("model");
    try js.write(settings.model);
    // Architecture and direction settings that a study cannot be continued
    // or warm-started across.
    try js.objectField("num_layers");
    try js.write(model.config.num_layers);
    try js.objectField("n_components");
    try js.write(model_mod.Component.all.len);
    try js.objectField("n_directions");
    try js.write(settings.n_directions);
    // Settings that change the objectives or the directions: `continue` refuses when they differ.
    try js.objectField("scorers");
    try js.write(try scorerSetString(a, settings.scorers));
    try js.objectField("kl_tokens");
    try js.write(settings.kl_tokens);
    try js.objectField("direction_method");
    try js.write(@tagName(settings.direction_method));
    try js.objectField("direction_token_window");
    try js.write(settings.direction_token_window);
    try js.objectField("direction_index_low");
    try js.write(range.low);
    try js.objectField("direction_index_high");
    try js.write(range.high);
    try js.objectField("ablate_inputs");
    try js.write(settings.ablate_inputs);
    try js.objectField("n_trials");
    try js.write(settings.n_trials);
    try js.objectField("n_startup_trials");
    try js.write(settings.n_startup_trials);
    try js.objectField("seed");
    try js.write(settings.seed);
    try js.objectField("response_prefix");
    try js.write(settings.response_prefix);
    try js.objectField("row_normalization");
    try js.write(@tagName(settings.row_normalization));
    try js.objectField("orthogonalize_direction");
    try js.write(settings.orthogonalize_direction);
    try js.endObject();
    return w.toOwnedSlice();
}

/// Refuses to continue a study whose objectives or directions would differ
/// from the recorded ones (its trials would not be comparable). Fields
/// missing from an older journal are taken as the defaults.
fn checkStudySettings(a: Allocator, study: *const study_mod.Study, settings: *const config.Settings, range: search.IndexRange) !void {
    const stored: usize = @intCast(study.settingInteger("n_directions") orelse 1);
    if (stored != settings.n_directions) {
        std.log.err("the checkpoint was created with n_directions = {d}, but n_directions = {d} was requested; pass --n-directions {d} or restart the study", .{ stored, settings.n_directions, stored });
        return error.StudyDirectionsMismatch;
    }
    const defaults = config.Settings{};
    const stored_scorers = study.settingString("scorers") orelse try scorerSetString(a, defaults.scorers);
    const wanted_scorers = try scorerSetString(a, settings.scorers);
    if (!std.mem.eql(u8, stored_scorers, wanted_scorers)) {
        std.log.err("the checkpoint was created with the scorers {s}, but {s} was requested (the objectives would differ); use the same scorers / --fast-search setting or restart the study", .{ stored_scorers, wanted_scorers });
        return error.StudySettingsMismatch;
    }
    const kl_tokens: usize = @intCast(study.settingInteger("kl_tokens") orelse 1);
    if (kl_tokens != settings.kl_tokens) {
        std.log.err("the checkpoint was created with kl_tokens = {d}, but kl_tokens = {d} was requested; pass --kl-tokens {d} or restart the study", .{ kl_tokens, settings.kl_tokens, kl_tokens });
        return error.StudySettingsMismatch;
    }
    const method = study.settingString("direction_method") orelse "mean";
    if (!std.mem.eql(u8, method, @tagName(settings.direction_method))) {
        std.log.err("the checkpoint was created with direction_method = {s}, but {s} was requested; pass --direction-method {s} or restart the study", .{ method, @tagName(settings.direction_method), method });
        return error.StudySettingsMismatch;
    }
    const window: usize = @intCast(study.settingInteger("direction_token_window") orelse 1);
    if (window != settings.direction_token_window) {
        std.log.err("the checkpoint was created with direction_token_window = {d}, but {d} was requested; pass --direction-token-window {d} or restart the study", .{ window, settings.direction_token_window, window });
        return error.StudySettingsMismatch;
    }
    const ablate_inputs = study.settingBool("ablate_inputs") orelse false;
    if (ablate_inputs != settings.ablate_inputs) {
        std.log.err("the checkpoint was created with ablate_inputs = {}, but {} was requested; restart the study or use --ablate-inputs {}", .{ ablate_inputs, settings.ablate_inputs, ablate_inputs });
        return error.StudySettingsMismatch;
    }
    if (study.settingFloat("direction_index_low")) |low| {
        const high = study.settingFloat("direction_index_high") orelse range.high;
        if (@abs(low - range.low) > 1e-6 or @abs(high - range.high) > 1e-6) {
            std.log.err("the checkpoint searched direction_index in [{d:.3}, {d:.3}], but the current settings give [{d:.3}, {d:.3}] (--direction-range); use the same range or restart the study", .{ low, high, range.low, range.high });
            return error.StudySettingsMismatch;
        }
    }
}

/// Loads the trials of a previous study (read-only) as sampler observations.
/// The study must come from the same architecture and parameter space.
fn loadWarmStart(gpa: Allocator, arena: Allocator, io: Io, path: []const u8, model: *const Model, dims: usize, n_objectives: usize, out: *Io.Writer) ![]tpe.Observation {
    var warm = try study_mod.Study.open(gpa, io, path);
    defer warm.deinit();
    if (!warm.exists()) {
        std.log.err("warm-start study {s} does not exist or is empty", .{path});
        return error.WarmStartNotFound;
    }
    const layers = warm.settingInteger("num_layers") orelse {
        std.log.err("warm-start study {s} carries no architecture information (num_layers)", .{path});
        return error.WarmStartIncompatible;
    };
    if (layers != @as(i64, @intCast(model.config.num_layers))) {
        std.log.err("warm-start study {s} was run on a model with {d} layers, this model has {d}", .{ path, layers, model.config.num_layers });
        return error.WarmStartIncompatible;
    }
    var obs = std.ArrayList(tpe.Observation).empty;
    for (warm.trials.items) |t| {
        if (t.params.len != dims or t.losses.len != n_objectives) {
            std.log.err("warm-start study {s} has a different parameter space ({d} parameters, {d} objectives; expected {d} and {d})", .{ path, t.params.len, t.losses.len, dims, n_objectives });
            return error.WarmStartIncompatible;
        }
        try obs.append(arena, .{ .params = try arena.dupe(f64, t.params), .losses = try arena.dupe(f64, t.losses) });
    }
    try out.print("\nWarm start: {d} trials loaded from {s}\n", .{ obs.items.len, path });
    return obs.toOwnedSlice(arena);
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

pub fn main(init: std.process.Init) !void {
    const io = init.io;
    var err_buf: [8192]u8 = undefined;
    var ew = Io.File.Writer.initStreaming(.stderr(), io, &err_buf);
    const out = &ew.interface;
    defer out.flush() catch {};
    var out_buf: [8192]u8 = undefined;
    var rw = Io.File.Writer.initStreaming(.stdout(), io, &out_buf);
    const result = &rw.interface;
    defer result.flush() catch {};
    var in_buf: [4096]u8 = undefined;
    var fr = Io.File.stdin().readerStreaming(io, &in_buf);
    var discard_buf: [256]u8 = undefined;
    var discarding: Io.Writer.Discarding = .init(&discard_buf);
    var con = Console{
        .out = out,
        .log = out,
        .result = result,
        .in = &fr.interface,
        .tty = Io.File.stderr().isTty(io) catch false,
        .tty_out = Io.File.stdout().isTty(io) catch false,
        .tty_in = Io.File.stdin().isTty(io) catch false,
    };

    run(init, &con, &discarding.writer) catch |err| {
        out.flush() catch {};
        result.flush() catch {};
        switch (err) {
            error.TimeLimitExceeded => std.log.err("time limit reached before the optimisation could start; nothing to resume", .{}),
            error.ExportIncomplete => std.log.err("the export was cut short (time limit); the output directory is marked {s}", .{model_mod.export_incomplete_marker}),
            error.BudgetTooSmall => std.log.err("memory budget too small (see above)", .{}),
            error.Interrupted => std.log.err("interrupted", .{}),
            error.NoInput, error.DirectoryNotEmpty => {},
            else => std.log.err("{s}", .{@errorName(err)}),
        }
        std.process.exit(if (err == error.BudgetTooSmall) 2 else 1);
    };
}

/// Feasibility estimate for the loaded model under the current settings.
fn estimateFor(model: *const Model, settings: *const config.Settings, threads: usize, max_prompt_tokens: usize) budget_mod.Estimate {
    return budget_mod.estimate(model, .{
        .batch_size = if (settings.batch_size > 0) settings.batch_size else settings.max_batch_size,
        .max_prompt_tokens = max_prompt_tokens,
        .max_response_length = settings.max_response_length,
        .threads = threads,
        .lora_rank = settings.full_normalization_lora_rank,
        .export_dtype = if (settings.export_dtype) |d| App.parseExportDtype(d) else null,
    });
}

/// True when the environment variable is present and non-empty.
fn envSet(env: *std.process.Environ.Map, name: []const u8) bool {
    const v = env.get(name) orelse return false;
    return v.len > 0;
}

/// True when `path` is a directory with at least one entry.
fn dirHasEntries(io: Io, path: []const u8) !bool {
    var dir = Io.Dir.cwd().openDir(io, path, .{ .iterate = true }) catch |err| switch (err) {
        error.FileNotFound => return false,
        else => return err,
    };
    defer dir.close(io);
    var it = dir.iterate();
    return (try it.next(io)) != null;
}

/// Longest tokenised prompt (in tokens) among `prompts`.
fn maxPromptTokens(gpa: Allocator, engine: *Engine, prompts: []const Prompt) !usize {
    var max: usize = 0;
    for (prompts) |p| {
        const ids = try engine.encodePrompt(gpa, p);
        defer gpa.free(ids);
        max = @max(max, ids.len);
    }
    return max;
}

/// The width to wrap help to: the terminal's, or the widest help line when
/// stdout is not a terminal.
fn stdoutColumns(io: Io, con: Console) usize {
    if (!con.tty_out) return wrap.max_columns;
    return logo.terminalColumns(io, .stdout()) orelse 80;
}

fn run(init: std.process.Init, con: *Console, discarding: *Io.Writer) !void {
    const gpa = init.gpa;
    const arena = init.arena.allocator();
    const io = init.io;

    // Settings.
    const raw_args = try init.minimal.args.toSlice(arena);
    var args = try arena.alloc([]const u8, raw_args.len);
    for (raw_args, 0..) |a, i| args[i] = a;
    var loaded = try config.load(gpa, io, args, init.environ_map);
    defer loaded.deinit();
    const settings = &loaded.settings;
    // --quiet: general messaging is dropped; `con.log` keeps the essentials.
    if (settings.quiet) con.out = discarding;
    const out = con.out;

    // Terminal, colour and interactivity.
    const env = init.environ_map;
    const env_no_color = envSet(env, "NO_COLOR") or envSet(env, "DITCH_NO_COLOR") or (if (env.get("TERM")) |t| std.mem.eql(u8, t, "dumb") else false);
    const env_force_color = envSet(env, "FORCE_COLOR");
    con.color = if (settings.no_color or settings.plain) false else if (env_force_color) true else con.tty and !env_no_color;
    color_enabled = con.color;
    con.interactive = !settings.no_input and (con.tty_in or settings.interactive);
    if (settings.quiet) engine_mod.progress_style = .off else if (con.tty) engine_mod.progress_style = .tty;
    const bold_out = con.tty_out and con.color;

    if (settings.version) {
        try con.result.print("ditch {s} (zig {s}, {s}-{s}, {s}, {d}-lane {d}x{d} kernels{s})\n", .{
            config.version,
            builtin.zig_version_string,
            @tagName(builtin.cpu.arch),
            @tagName(builtin.os.tag),
            @tagName(builtin.mode),
            tensor.VL,
            tensor.tile_inputs,
            tensor.tile_rows,
            if (tensor.have_accelerate) ", Accelerate" else "",
        });
        return;
    }
    if (settings.help) {
        const bench_topic = settings.bench or (if (settings.help_topic) |t| std.mem.eql(u8, t, "bench") else false);
        if (settings.help_topic) |t| if (!std.mem.eql(u8, t, "bench")) {
            std.log.err("unknown help topic: {s} (try ditch help bench)", .{t});
            std.process.exit(2);
        };
        var help: Io.Writer.Allocating = .init(arena);
        if (bench_topic) try config.writeBenchHelp(&help.writer, bold_out) else try config.writeHelp(&help.writer, bold_out);
        try wrap.write(arena, con.result, help.written(), stdoutColumns(io, con.*));
        return;
    }
    if (args.len == 1) {
        if (con.tty_out) {
            try logo.write(con.result, bold_out, logo.terminalColumns(io, .stdout()));
            try con.result.writeAll("\n");
        }
        var help: Io.Writer.Allocating = .init(arena);
        try config.writeConciseHelp(&help.writer, bold_out);
        try wrap.write(arena, con.result, help.written(), stdoutColumns(io, con.*));
        try con.result.flush();
        std.process.exit(2);
    }
    if (loaded.errors.len > 0) {
        for (loaded.errors) |e| std.log.err("{s}", .{e});
        try out.writeAll("Run ditch --help for all options; config.default.lua documents every setting.\n");
        try out.flush();
        std.process.exit(2);
    }
    if (!settings.quiet) {
        try logo.write(out, con.color, if (con.tty) logo.terminalColumns(io, .stderr()) else null);
        try out.print("\n  v{s}  ditch censorship.  https://github.com/plyght/ditch\n", .{config.version});
        try out.writeAll("  Built on Heretic: https://github.com/p-e-w/heretic\n\n");
    }
    // Accelerate (macOS): resolved once, before any kernel runs.
    tensor.accelerate_enabled = settings.accelerate;
    if (tensor.have_accelerate and settings.accelerate) {
        if (!tensor.initAccelerate()) try out.writeAll("Accelerate could not be loaded; using the built-in kernels.\n");
    }

    // `ditch bench --kernels`: the compute kernels on synthetic data, no model.
    if (settings.bench and settings.bench_kernels) {
        const kpool = try arena.create(tensor.Pool);
        kpool.* = tensor.Pool.initPersistent(gpa, io, settings.threads);
        defer kpool.deinit();
        try bench.runKernels(gpa, io, settings, kpool, out, con.result);
        return;
    }

    // A reproducibility manifest replaces the recorded settings (model, seed, datasets, ...).
    var manifest: ?reproduce.Manifest = null;
    if (settings.reproduce) |path| {
        try out.print("Loading reproducibility manifest {s}...\n", .{path});
        try out.flush();
        manifest = try reproduce.load(gpa, arena, io, path);
        reproduce.applySettings(&manifest.?, settings);
        try out.print("* Recorded by ditch {s}: model {s}, trial {d}\n", .{ manifest.?.ditch_version, manifest.?.model, manifest.?.trial_index });
    }
    // `ditch selftest` checks the compute kernels themselves and needs no model.
    if (settings.model.len == 0 and !settings.selftest) {
        try out.writeAll("No model specified.\n\n");
        try out.writeAll(config.help_text);
        try out.flush();
        std.process.exit(1);
    }
    if (settings.config_path) |p| try out.print("Using configuration file {s}\n", .{p});
    if (settings.seed == null) {
        var b: [8]u8 = undefined;
        io.random(&b);
        settings.seed = std.mem.readInt(u64, &b, .little) & 0xffff_ffff;
    }
    try out.print("Random seed: {d}\n", .{settings.seed.?});

    // Threads.
    const cpu_count = std.Thread.getCpuCount() catch 1;
    const pool = try arena.create(tensor.Pool);
    pool.* = tensor.Pool.initPersistent(gpa, io, settings.threads);
    defer pool.deinit();
    if (tensor.performanceCores()) |p| {
        // Apple Silicon: the default is the performance cores only (see tensor.defaultThreads).
        try out.print("Using {d} threads ({d} CPUs available, {d} performance cores)\n", .{ pool.threads, cpu_count, p });
    } else {
        try out.print("Using {d} threads ({d} CPUs available)\n", .{ pool.threads, cpu_count });
    }
    if (tensor.accelerateActive()) try out.writeAll("Matrix products above the batch threshold use Accelerate.\n");
    try out.flush();
    installSigint();

    // Compute backend. The CPU backend is the default and the reference; any
    // other backend is checked against it by `ditch selftest`.
    const device_kind = settings.deviceKind() orelse {
        std.log.err("unknown device: {s} (expected {s})", .{ settings.device, compute.Kind.names });
        std.process.exit(2);
    };
    if (settings.selftest) {
        try selftest.run(gpa, settings, pool, out, con.result);
        return;
    }
    const device_note = compute.selectInto(gpa, device_kind, .{ .memory_budget = settings.gpu_memory, .io = pool.io, .min_macs = settings.device_min_macs }) catch {
        std.log.err("device {s} is not available on this build or machine (build with -Dmetal on Apple silicon, or use --device auto)", .{settings.device});
        std.process.exit(2);
    };
    defer compute.shutdown();
    // Runs before the shutdown above: say how much the device actually did, so
    // a small model that never crossed the dispatch threshold is visible as
    // "0 matrix products" instead of passing for a GPU run.
    defer if (!compute.active.isCpu()) {
        out.print("{d} matrix products ran on {s}\n", .{ compute.served.load(.monotonic), compute.active.name }) catch {};
        out.flush() catch {};
    };
    if (device_note) |n| try out.print("{s}\n", .{n});
    if (!compute.active.isCpu()) {
        try out.print("Compute device: {s}", .{compute.active.name});
        if (compute.active.memory_budget > 0) try out.print(" (up to {f} of resident weights)", .{budget_mod.fmtBytes(compute.active.memory_budget)});
        try out.writeAll("\n");
    }
    try out.flush();

    // Memory / time budget. Every large runtime buffer comes from the budget's
    // allocator (unlimited when no --max-ram is given, but still accounted).
    var budget = try budget_mod.Budget.fromSettingsEnv(gpa, io, settings, init.environ_map);
    defer budget.deinit();
    const rt_gpa = budget.allocator();
    // Remote weights and an explicit expert cache imply streaming.
    const is_remote = remote.isRemoteId(settings.model) or settings.remote_weights;
    const store_mode: stream.Mode = if (settings.max_ram > 0 or settings.expert_cache != null or is_remote) .streamed else .mapped;
    // Only memory-mapped weights keep a stable address for the whole run, so
    // only they may stay resident on a device (see compute.Residency).
    compute.setWeightsStable(store_mode == .mapped);
    if (budget.limited()) try out.print("Memory budget: {f} (headroom {f}, scratch directory {s})\n", .{ budget_mod.fmtBytes(budget.max_ram), budget_mod.fmtBytes(budget.headroom), budget.scratch_dir });
    if (budget.time_limit) |t| try out.print("Time limit: {f}\n", .{budget_mod.fmtDuration(t)});
    if (settings.max_vram > 0) try out.writeAll("Note: --max-vram is accepted for compatibility but unused; a GPU backend is bounded by --gpu-memory.\n");
    defer {
        if (budget.limited() or budget.time_limit != null or settings.print_debug_information) {
            budget.report().print(out, "\nMemory") catch {};
            out.flush() catch {};
        }
    }

    // Model.
    var http = try hf.Http.initWithOptions(gpa, io, arena, init.environ_map, .{ .token_file = settings.token_file, .timeout_seconds = settings.http_timeout_seconds });
    defer http.deinit();
    const cache_root = try hf.cacheDir(arena, settings, init.environ_map);
    if (settings.bench) {
        try bench.run(gpa, arena, io, settings, &http, cache_root, pool, out, con.result);
        return;
    }
    if (settings.probe) {
        // Loaded as the study would load it: streamed under --max-ram, warp
        // mode, and `hf://` weights read through the chunk cache.
        var probe_src: ?*remote.Source = null;
        defer if (probe_src) |s| s.deinit();
        if (is_remote) probe_src = try remote.Source.open(gpa, io, &http, cache_root, settings.model, .{ .revision = settings.model_commit, .chunk_size = settings.remote_chunk_size, .cache_size = settings.remote_cache_size, .connections = settings.remote_connections }, out);
        try probe.run(gpa, arena, io, settings, &http, cache_root, pool, .{ .store = store_mode, .budget = &budget, .expert_cache = settings.expert_cache, .remote = probe_src }, if (probe_src) |s| s.dir_path else null, out, con.result);
        if (probe_src) |s| printRemoteStats(out, s);
        return;
    }
    try out.print("\nLoading model {s}...\n", .{settings.model});
    try out.flush();
    var remote_src: ?*remote.Source = null;
    defer if (remote_src) |s| s.deinit();
    const model_dir = if (is_remote) blk: {
        remote_src = try remote.Source.open(gpa, io, &http, cache_root, settings.model, .{ .revision = settings.model_commit, .chunk_size = settings.remote_chunk_size, .cache_size = settings.remote_cache_size, .connections = settings.remote_connections }, out);
        break :blk remote_src.?.dir_path;
    } else try hf.resolveModel(arena, &http, cache_root, settings.model, settings.model_commit, out);
    const model = try Model.loadWithOptions(gpa, io, pool, model_dir, .{ .store = store_mode, .budget = &budget, .expert_cache = settings.expert_cache, .remote = remote_src });
    defer model.deinit();
    // Printed before the model goes away (and before the memory report).
    defer {
        if (remote_src) |s| printRemoteStats(out, s);
        if (model.expert_cache) |ec| {
            ec.stats().print(out, "\nExpert cache") catch {};
            if (settings.hotlist) {
                if (model.writeHotlist(gpa, settings.model)) |maybe| {
                    if (maybe) |path| {
                        out.print("Expert hotlist written to {s}\n", .{path}) catch {};
                        gpa.free(path);
                    }
                } else |err| out.print("Could not write the expert hotlist: {s}\n", .{@errorName(err)}) catch {};
            }
        }
        out.flush() catch {};
    }
    const c = &model.config;
    try out.print("* Architecture: {s} ({d} layers, hidden size {d}, vocabulary {d}, {s} weights)\n", .{ c.model_type, c.num_layers, c.hidden_size, c.vocab_size, model.dtype.safetensorsName() });
    // What the chunk cache must hold; also marks the trunk's chunks so that
    // eviction keeps them longest. Said once, here, when the trunk alone
    // does not fit the bound.
    const footprint: ?remote.Footprint = if (remote_src) |s| try remote.planModel(s, gpa, model) else null;
    if (footprint) |fp| try fp.warn(out);
    if (model.expert_cache) |ec| {
        try out.print("* Weights: trunk streamed layer by layer from {s}, routed experts through an expert cache of {f} (warp mode)\n", .{ if (remote_src != null) "the remote source" else "disk", budget_mod.fmtBytes(ec.capacity) });
        if (settings.hotlist) {
            const warmed = try model.warmExpertCache(gpa, settings.model);
            if (warmed > 0) try out.print("* hotlist: {d} experts warmed\n", .{warmed});
        }
    } else {
        try out.print("* Weights: {s}\n", .{if (model.streamed()) "streamed layer by layer from disk (memory budget)" else "memory-mapped"});
    }
    if (model.gguf) |g| try out.print("* Source: GGUF file {s} (architecture {s}, {s} tokenizer)\n", .{ g.file_name, g.arch, if (g.embedded_tokenizer) "embedded Hugging Face" else "rebuilt from the ggml vocabulary" });
    if (manifest) |*m| try reproduce.verifyModelFiles(arena, io, m, model, settings.ignore_mismatches, out);
    var template: chat.Template = undefined;
    if (settings.chat_template) |name| {
        template = chat.Template.parse(name) orelse {
            std.log.err("unknown chat template: {s} (expected chatml, llama3, llama2, mistral, gemma or raw)", .{name});
            return error.InvalidChatTemplate;
        };
        try out.print("* Chat template: {s} (from settings)\n", .{@tagName(template)});
    } else {
        template = chat.detect(model.chat_template, c.model_type);
        try out.print("* Chat template: {s} ({s})\n", .{ @tagName(template), if (model.chat_template != null) "detected from the model's chat template" else "inferred from the model type" });
    }
    var engine = Engine.init(rt_gpa, model, settings, template);
    defer engine.deinit();

    // Prompts.
    try out.print("\nLoading good prompts from {s}...\n", .{settings.good_prompts.dataset});
    try out.flush();
    const good_prompts = try hf.loadPrompts(arena, &http, cache_root, settings, settings.good_prompts, out);
    try out.print("* {d} prompts loaded\n", .{good_prompts.len});
    try out.print("\nLoading bad prompts from {s}...\n", .{settings.bad_prompts.dataset});
    try out.flush();
    const bad_prompts = try hf.loadPrompts(arena, &http, cache_root, settings, settings.bad_prompts, out);
    try out.print("* {d} prompts loaded\n", .{bad_prompts.len});
    if (good_prompts.len == 0 or bad_prompts.len == 0) {
        std.log.err("both prompt datasets must contain at least one prompt", .{});
        return error.NoPrompts;
    }
    if (manifest) |*m| {
        try out.writeAll("\nVerifying prompts against the manifest...\n");
        try reproduce.verifyPrompts(arena, m.good_prompts, good_prompts, "good prompts", settings.ignore_mismatches, out);
        try reproduce.verifyPrompts(arena, m.bad_prompts, bad_prompts, "bad prompts", settings.ignore_mismatches, out);
    }
    try out.flush();

    // Feasibility: what must be resident, and does it fit the budget?
    const max_prompt_tokens = @max(try maxPromptTokens(gpa, &engine, good_prompts), try maxPromptTokens(gpa, &engine, bad_prompts));
    const estimate = estimateFor(model, settings, pool.threads, max_prompt_tokens);
    if (budget.limited() or settings.print_debug_information) {
        try out.writeAll("\n");
        try estimate.print(out);
        if (footprint) |fp| try fp.print(out);
    }
    budget_mod.check(estimate, &budget, out) catch |err| switch (err) {
        error.BudgetTooSmall => {
            try out.flush();
            std.process.exit(2);
        },
        else => return err,
    };
    try out.flush();
    if (settings.dry_run) {
        if (!(budget.limited() or settings.print_debug_information)) {
            try out.writeAll("\n");
            try estimate.print(out);
            if (footprint) |fp| try fp.print(out);
        }
        try con.log.writeAll(if (footprint != null) "\nDry run: the model, prompts, memory and disk estimates are in order; stopping here (exit 0).\n" else "\nDry run: the model, prompts and memory estimate are in order; stopping here (exit 0).\n");
        return;
    }

    if (settings.batch_size == 0) {
        try detectBatchSize(gpa, io, &engine, settings, good_prompts, out);
    } else {
        engine.batch_size = settings.batch_size;
    }
    try out.flush();

    if (settings.response_prefix == null) {
        try detectResponsePrefix(gpa, arena, &engine, settings, good_prompts, bad_prompts, out);
    } else {
        try out.writeAll("\nUsing response prefix: ");
        try printRepr(out, settings.response_prefix.?);
        try out.writeAll("\n");
    }

    // Scorers and baseline (always computed on the base model).
    var evaluator = try scorers.Evaluator.init(gpa, arena, &engine, settings, &http, cache_root, out);
    defer evaluator.deinit();
    if (manifest) |*m| {
        try out.writeAll("\nVerifying scorer prompts against the manifest...\n");
        if (m.keyword_rate_prompts) |d| if (reproduce.scorerPrompts(&evaluator, .keyword_rate)) |p| try reproduce.verifyPrompts(arena, d, p, "refusal scoring prompts", settings.ignore_mismatches, out);
        if (m.kl_divergence_prompts) |d| if (reproduce.scorerPrompts(&evaluator, .kl_divergence)) |p| try reproduce.verifyPrompts(arena, d, p, "KL divergence prompts", settings.ignore_mismatches, out);
    }
    try out.flush();

    if (settings.evaluate_model) |eval_id| {
        try out.print("\nLoading model {s}...\n", .{eval_id});
        try out.flush();
        const eval_dir = try hf.resolveModel(arena, &http, cache_root, eval_id, null, out);
        const eval_model = try Model.loadWithOptions(gpa, io, pool, eval_dir, .{ .store = store_mode, .budget = &budget });
        defer eval_model.deinit();
        budget_mod.check(estimateFor(eval_model, settings, pool.threads, max_prompt_tokens), &budget, out) catch |err| switch (err) {
            error.BudgetTooSmall => {
                try out.flush();
                std.process.exit(2);
            },
            else => return err,
        };
        var eval_engine = Engine.init(rt_gpa, eval_model, settings, template);
        defer eval_engine.deinit();
        eval_engine.batch_size = engine.batch_size;
        try out.writeAll("* Evaluating...\n");
        try out.flush();
        const scores = try evaluator.scores(arena, &eval_engine, out);
        var all_scores = std.ArrayList(scorers.NamedScore).empty;
        try all_scores.appendSlice(arena, scores);
        if (try evaluator.deferredScore(arena, &eval_engine, out)) |d| try all_scores.append(arena, d);
        try writeScores(con, arena, eval_id, all_scores.items, settings);
        return;
    }

    if (evaluator.objectiveCount() == 0) {
        std.log.err("no optimization objectives configured: at least one scorer must set optimization to \"minimize\" or \"maximize\"", .{});
        return error.NoObjectives;
    }

    // Residual directions.
    try out.writeAll("\nCalculating per-layer residual directions...\n");
    const entries = c.num_layers + 1;
    const window = settings.direction_token_window;
    if (window > 1) try out.print("* Residuals are averaged over the last {d} prompt tokens\n", .{window});
    // The "separating" method needs the per-coordinate variances of both sets.
    var good_moments: ?directions.Moments = null;
    defer if (good_moments) |*m| m.deinit();
    var bad_moments: ?directions.Moments = null;
    defer if (bad_moments) |*m| m.deinit();
    if (settings.direction_method == .separating) {
        good_moments = try directions.Moments.init(rt_gpa, entries, c.hidden_size);
        bad_moments = try directions.Moments.init(rt_gpa, entries, c.hidden_size);
    }
    try out.writeAll("* Obtaining residual mean for good prompts...\n");
    try out.flush();
    const good_means = try engine.getResidualMeanObserved(rt_gpa, good_prompts, out, .{ .moments = if (good_moments) |*m| m else null, .window = window });
    defer rt_gpa.free(good_means);
    try out.writeAll("* Obtaining residual mean for bad prompts...\n");
    try out.flush();
    // With several directions per layer, the same pass over the bad prompts
    // also accumulates the covariance sketch (see directions.zig).
    var sketch: ?directions.Sketch = null;
    defer if (sketch) |*s| s.deinit();
    if (settings.n_directions > 1) sketch = try directions.Sketch.init(rt_gpa, entries, c.hidden_size, settings.n_directions, good_means, settings.seed.?);
    const bad_means = try engine.getResidualMeanObserved(rt_gpa, bad_prompts, out, .{ .sketch = if (sketch) |*s| s else null, .moments = if (bad_moments) |*m| m else null, .window = window });
    defer rt_gpa.free(bad_means);
    const first = switch (settings.direction_method) {
        .mean => try abliterate.computeDirections(rt_gpa, good_means, bad_means, entries, c.hidden_size, settings.orthogonalize_direction),
        .separating => blk: {
            try out.print("* Whitening the difference of means by the per-coordinate variance (shrinkage {d})...\n", .{settings.direction_shrinkage});
            break :blk try directions.computeSeparating(rt_gpa, good_means, bad_means, &good_moments.?, &bad_moments.?, entries, c.hidden_size, settings.direction_shrinkage, settings.orthogonalize_direction);
        },
    };
    defer rt_gpa.free(first);
    if (settings.n_directions > 1) try out.print("* Extracting {d} orthonormal directions per layer...\n", .{settings.n_directions});
    const dirs = try directions.computeBasisFrom(rt_gpa, first, good_means, entries, c.hidden_size, settings.n_directions, settings.orthogonalize_direction, if (sketch) |*s| s else null);
    defer rt_gpa.free(dirs);
    if (settings.dump_directions) |path| {
        try dumpDirections(rt_gpa, io, path, dirs, good_means, bad_means, entries, c.hidden_size, settings.n_directions);
        try out.print("* Wrote the directions and residual means to {s}\n", .{path});
    }
    // Separation scores need one more pass (the projections of every prompt
    // onto the final directions); only when something uses them.
    var sep: ?directions.Separation = null;
    defer if (sep) |*s| s.deinit(rt_gpa);
    if (settings.print_residual_geometry or (settings.direction_range == .auto and manifest == null)) {
        try out.writeAll("* Projecting the prompts onto the directions for the separation scores...\n");
        try out.flush();
        const stride = settings.n_directions * c.hidden_size;
        const good_proj = try engine.getProjections(rt_gpa, good_prompts, dirs, stride, window);
        defer rt_gpa.free(good_proj);
        const bad_proj = try engine.getProjections(rt_gpa, bad_prompts, dirs, stride, window);
        defer rt_gpa.free(bad_proj);
        sep = try directions.separation(rt_gpa, good_proj, bad_proj, entries);
    }
    if (settings.print_residual_geometry) try printResidualGeometry(out, good_means, bad_means, entries, c.hidden_size, sep);
    const index_range = if (manifest != null and sep == null) search.defaultIndexRange(model) else resolveDirectionRange(settings, model, sep);
    if (settings.direction_range == .auto and sep != null) {
        try out.print("* direction_index range: {d:.2} to {d:.2} (layers whose projection AUROC is within {d} of the best; heretic: {d:.2} to {d:.2})\n", .{ index_range.low, index_range.high, directions.auto_range_tolerance, search.defaultIndexRange(model).low, search.defaultIndexRange(model).high });
    }
    try out.flush();

    if (manifest) |*m| {
        try runReproduction(gpa, rt_gpa, arena, io, con, settings, &http, cache_root, pool, &budget, model, &engine, template, &evaluator, dirs, model_dir, good_prompts, bad_prompts, m);
        try out.flush();
        return;
    }

    // Study.
    const checkpoint_path = try study_mod.checkpointFileName(arena, settings.study_checkpoint_dir, settings.model);
    var study = try study_mod.Study.open(gpa, io, checkpoint_path);
    defer study.deinit();
    var show_results_only = false;
    if (study.exists()) {
        var action: []const u8 = undefined;
        if (settings.checkpoint_action) |a| {
            action = a;
        } else if (study.finished) {
            try out.writeAll("\nYou have already processed this model. You can show the results from the previous run, allowing you to export models or to run additional trials. Alternatively, you can ignore the previous run and start from scratch. This will delete the checkpoint file and all results from the previous run.\n");
            const opts = [_][]const u8{ "Show the results from the previous run", "Ignore the previous run and start from scratch", "Exit" };
            const choice = (try con.menu("How would you like to proceed?", &opts, "--checkpoint-action continue|restart")) orelse return;
            action = switch (choice) {
                0 => "continue",
                1 => if (try con.confirm("This deletes the checkpoint and every result of the previous run. Continue?", "--checkpoint-action restart")) "restart" else return,
                else => return,
            };
        } else {
            try out.writeAll("\nYou have already processed this model, but the run was interrupted. You can continue the previous run from where it stopped. Alternatively, you can ignore the previous run and start from scratch. This will delete the checkpoint file and all results from the previous run.\n");
            const opts = [_][]const u8{ "Continue the previous run", "Ignore the previous run and start from scratch", "Exit" };
            const choice = (try con.menu("How would you like to proceed?", &opts, "--checkpoint-action continue|restart")) orelse return;
            action = switch (choice) {
                0 => "continue",
                1 => if (try con.confirm("This deletes the checkpoint and every result of the previous run. Continue?", "--checkpoint-action restart")) "restart" else return,
                else => return,
            };
        }
        if (std.mem.eql(u8, action, "restart")) {
            try study.reset(try settingsSnapshot(arena, settings, model, index_range));
        } else if (std.mem.eql(u8, action, "continue")) {
            try checkStudySettings(arena, &study, settings, index_range);
            if (study.finished) show_results_only = true;
            if (study.trials.items.len > settings.n_trials) settings.n_trials = study.trials.items.len;
        } else if (std.mem.eql(u8, action, "exit")) {
            return;
        } else {
            std.log.err("unknown checkpoint action: {s} (expected continue or restart)", .{action});
            return error.InvalidCheckpointAction;
        }
    } else {
        try study.reset(try settingsSnapshot(arena, settings, model, index_range));
    }
    try out.print("\nStudy checkpoint: {s}\n", .{checkpoint_path});

    var space = try search.buildSpaceWithRange(gpa, model, index_range);
    defer space.deinit();
    var sampler = tpe.Sampler.init(space.space, settings.n_startup_trials, settings.seed.?);
    var warm: []const tpe.Observation = &.{};
    if (settings.warm_start) |p| warm = try loadWarmStart(gpa, arena, io, p, model, space.dims(), evaluator.objectiveCount(), out);
    if (settings.early_stop and evaluator.earlyStopEntry() == null and !settings.fast_search) {
        try out.writeAll("\nEarly stopping is disabled: it requires a minimised keyword-rate scorer as the last objective (list kl_divergence before keyword_rate).\n");
    }

    var app = App{
        .gpa = gpa,
        .rt_gpa = rt_gpa,
        .arena = arena,
        .io = io,
        .con = con,
        .settings = settings,
        .http = &http,
        .cache_root = cache_root,
        .pool = pool,
        .budget = &budget,
        .model = model,
        .engine = &engine,
        .template = template,
        .evaluator = &evaluator,
        .dirs = dirs,
        .space = &space,
        .study = &study,
        .sampler = &sampler,
        .optimization_start = Io.Timestamp.now(io, .awake),
        .start_index = 0,
        .model_dir = model_dir,
        .good_prompts = good_prompts,
        .bad_prompts = bad_prompts,
        .warm = warm,
    };

    if (!show_results_only) {
        switch (try app.runTrials()) {
            .finished => {},
            .interrupted => try out.writeAll("\nOptimization interrupted. The completed trials have been saved; run ditch again to continue.\n"),
            .time_limit => {
                // A clean stop: the journal is resumable, nothing else is attempted.
                try app.printTimeLimit();
                return;
            },
        }
    }
    try app.resultsLoop();
    try out.flush();
}

/// `--reproduce`: applies the recorded trial directly (no search), prints the
/// scores next to the recorded ones and offers the usual model menu.
fn runReproduction(
    gpa: Allocator,
    rt_gpa: Allocator,
    arena: Allocator,
    io: Io,
    con: *Console,
    settings: *config.Settings,
    http: *hf.Http,
    cache_root: []const u8,
    pool: *const tensor.Pool,
    budget: *budget_mod.Budget,
    model: *Model,
    engine: *Engine,
    template: chat.Template,
    evaluator: *scorers.Evaluator,
    dirs: []const f32,
    model_dir: []const u8,
    good_prompts: []const Prompt,
    bad_prompts: []const Prompt,
    m: *const reproduce.Manifest,
) !void {
    const out = con.out;
    var space = try search.buildSpace(gpa, model);
    defer space.deinit();
    const vector = try reproduce.vectorFor(arena, m, &space);
    // The study and sampler are never used: the trial comes from the manifest.
    var study = try study_mod.Study.open(gpa, io, try study_mod.checkpointFileName(arena, settings.study_checkpoint_dir, settings.model));
    defer study.deinit();
    var sampler = tpe.Sampler.init(space.space, settings.n_startup_trials, settings.seed.?);
    var app = App{
        .gpa = gpa,
        .rt_gpa = rt_gpa,
        .arena = arena,
        .io = io,
        .con = con,
        .settings = settings,
        .http = http,
        .cache_root = cache_root,
        .pool = pool,
        .budget = budget,
        .model = model,
        .engine = engine,
        .template = template,
        .evaluator = evaluator,
        .dirs = dirs,
        .space = &space,
        .study = &study,
        .sampler = &sampler,
        .optimization_start = Io.Timestamp.now(io, .awake),
        .start_index = 0,
        .model_dir = model_dir,
        .good_prompts = good_prompts,
        .bad_prompts = bad_prompts,
    };

    try out.print("\nApplying recorded trial {d}...\n", .{m.trial_index});
    try out.writeAll("* Parameters:\n");
    try search.describe(&space, vector, out);
    try out.writeAll("* Abliterating...\n");
    try out.flush();
    model.resetDeltas();
    const cfg = search.decode(&space, vector);
    try search.applyTrial(model, dirs, cfg, app.abliterateOptions());
    try out.writeAll("* Evaluating...\n");
    try out.flush();
    var scores = try evaluator.scores(arena, engine, out);
    if (try evaluator.deferredScore(arena, engine, out)) |d| {
        const grown = try arena.alloc(scorers.NamedScore, scores.len + 1);
        @memcpy(grown[0..scores.len], scores);
        grown[scores.len] = d;
        scores = grown;
    }
    _ = try reproduce.printComparison(m, scores, out);
    try out.flush();

    const losses = try evaluator.objectiveLosses(arena, scores);
    const records = try arena.alloc(study_mod.ScoreRecord, scores.len);
    for (scores, 0..) |s, i| records[i] = .{ .name = s.name, .value = s.score.value, .display = s.score.display };
    const trial = study_mod.Trial{
        .index = m.trial_index,
        .params = vector,
        .losses = losses,
        .direction_index = cfg.direction_index,
        .parameters = cfg.parameters,
        .scores = records,
    };
    _ = try app.modelLoop(&trial);
}

/// The end-of-run summary of a remote source: ranges fetched, and the chunk cache.
fn printRemoteStats(out: *Io.Writer, s: *remote.Source) void {
    const st = s.stats();
    out.print("\nRemote source: fetched {d} ranges ({f}), {d} chunk reads served from the disk cache\n", .{ st.ranges_fetched, budget_mod.fmtBytes(st.bytes_fetched), st.chunks_from_disk }) catch {};
    out.print("Chunk cache: {f} on disk of a {f} bound (peak {f}), {d} chunks evicted, {d} served from RAM without being kept\n", .{ budget_mod.fmtBytes(st.cache_bytes), budget_mod.fmtBytes(st.cache_limit), budget_mod.fmtBytes(st.peak_cache_bytes), st.chunks_evicted, st.chunks_unpersisted }) catch {};
    out.flush() catch {};
}

/// `--dump-directions`: `directions` `[entries][n_directions * hidden]`,
/// `good_means` and `bad_means` `[entries][hidden]`, all f32; entry 0 is the
/// embedding output and entry `l + 1` the output of layer `l`.
fn dumpDirections(gpa: std.mem.Allocator, io: std.Io, path: []const u8, dirs: []const f32, good: []const f32, bad: []const f32, entries: usize, hidden: usize, k: usize) !void {
    const st = @import("safetensors.zig");
    const shape_d = [_]usize{ entries, k * hidden };
    const shape_m = [_]usize{ entries, hidden };
    const tensors = [_]st.OutTensor{
        .{ .name = "directions", .dtype = .f32, .shape = &shape_d, .data = std.mem.sliceAsBytes(dirs) },
        .{ .name = "good_means", .dtype = .f32, .shape = &shape_m, .data = std.mem.sliceAsBytes(good) },
        .{ .name = "bad_means", .dtype = .f32, .shape = &shape_m, .data = std.mem.sliceAsBytes(bad) },
    };
    try st.writeFile(gpa, io, std.Io.Dir.cwd(), path, &tensors, "ditch");
}
