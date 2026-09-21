//! ditch: fully automatic censorship removal for language models, in Zig.
//!
//! This file is the program flow (a port of heretic's `main.py`): settings,
//! model loading, batch size and response prefix detection, residual
//! direction extraction, the TPE optimisation loop with a resumable study
//! journal, and the interactive result / save / chat menus.

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
const directions = @import("directions.zig");

const Model = model_mod.Model;
const Engine = engine_mod.Engine;
const Prompt = hf.Prompt;

const banner =
    \\    _ _ _       _
    \\ __| (_) |_ __| |_
    \\/ _` | |  _/ _| ' \
    \\\__,_|_|\__\__|_||_|
    \\
;

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
    return interrupted.swap(false, .seq_cst);
}

// ---------------------------------------------------------------------------
// Console helpers
// ---------------------------------------------------------------------------

const Console = struct {
    out: *Io.Writer,
    in: *Io.Reader,

    /// Reads one line from stdin (without the newline); null on end of input.
    fn readLine(self: *Console) !?[]const u8 {
        try self.out.flush();
        const line = self.in.takeDelimiter('\n') catch |err| switch (err) {
            error.StreamTooLong => return error.LineTooLong,
            error.ReadFailed => return null,
        };
        if (interrupted.load(.seq_cst)) return null;
        return std.mem.trimEnd(u8, line orelse return null, "\r");
    }

    /// Prints a numbered menu and returns the chosen (0-based) option, or null to exit.
    fn menu(self: *Console, question: []const u8, options: []const []const u8) !?usize {
        while (true) {
            try self.out.print("\n{s}\n", .{question});
            for (options, 0..) |o, i| try self.out.print("  [{d}] {s}\n", .{ i + 1, o });
            try self.out.print("Enter a number (1-{d}): ", .{options.len});
            const line = (try self.readLine()) orelse return null;
            const trimmed = std.mem.trim(u8, line, " \t");
            if (trimmed.len == 0) continue;
            const n = std.fmt.parseInt(usize, trimmed, 10) catch {
                try self.out.writeAll("Please enter a number.\n");
                continue;
            };
            if (n < 1 or n > options.len) {
                try self.out.writeAll("Please enter one of the listed numbers.\n");
                continue;
            }
            return n - 1;
        }
    }

    fn ask(self: *Console, question: []const u8) !?[]const u8 {
        try self.out.print("{s} ", .{question});
        return self.readLine();
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

// ---------------------------------------------------------------------------
// Application state
// ---------------------------------------------------------------------------

const App = struct {
    gpa: Allocator,
    arena: Allocator,
    io: Io,
    con: *Console,
    settings: *config.Settings,
    http: *hf.Http,
    cache_root: []const u8,
    pool: *const tensor.Pool,
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

    /// Runs trials `study.trials.len .. settings.n_trials`. Returns false if interrupted.
    fn runTrials(self: *App) !bool {
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
            if (takeInterrupt()) return false;
            const trial_index = self.study.trials.items.len + 1;
            const observations = try self.allObservations(gpa);
            defer gpa.free(observations);
            try self.sampler.sample(gpa, observations, vector);
            const cfg = search.decode(self.space, vector);

            try out.print("\nRunning trial {d} of {d}...\n", .{ trial_index, n_trials });
            try out.writeAll("* Parameters:\n");
            try search.describe(self.space, vector, out);
            try out.writeAll("* Resetting model...\n");
            try out.flush();
            self.model.resetDeltas();
            try out.writeAll("* Abliterating...\n");
            try out.flush();
            try search.applyTrial(self.model, self.dirs, cfg, self.abliterateOptions());
            try out.writeAll("* Evaluating...\n");
            try out.flush();
            var scratch = std.heap.ArenaAllocator.init(gpa);
            defer scratch.deinit();
            const sa = scratch.allocator();
            // Early stopping compares against the Pareto front of completed trials.
            var front: ?[]const []const f64 = null;
            if (self.settings.early_stop) front = try self.study.frontLosses(sa);
            const scores = try self.evaluator.evaluate(sa, self.engine, out, front);
            const pruned = scorers.Evaluator.prunedOf(scores);
            try printScores(out, scores);

            const elapsed = secondsSince(self.io, self.optimization_start);
            const done: f64 = @floatFromInt(trial_index - self.start_index);
            const remaining = elapsed / done * @as(f64, @floatFromInt(n_trials - trial_index));
            var buf: [64]u8 = undefined;
            try out.print("\nElapsed time: {s}\n", .{formatDuration(&buf, elapsed)});
            if (trial_index < n_trials) try out.print("Estimated remaining time: {s}\n", .{formatDuration(&buf, remaining)});
            try out.flush();

            const losses = try self.evaluator.objectiveLosses(sa, scores);
            const records = try sa.alloc(study_mod.ScoreRecord, scores.len);
            for (scores, 0..) |s, i| records[i] = .{ .name = s.name, .value = s.score.value, .display = s.score.display };
            try self.study.addTrial(gpa, .{
                .index = trial_index,
                .params = vector,
                .losses = losses,
                .direction_index = cfg.direction_index,
                .parameters = cfg.parameters,
                .scores = records,
                .state = if (pruned != null) .pruned else .complete,
            });
            if (takeInterrupt()) return false;
        }
        const n_pruned = self.study.prunedCount();
        if (n_pruned > 0) try out.print("\n{d} of {d} trials were pruned by early stopping.\n", .{ n_pruned, self.study.trials.items.len });
        try self.study.markFinished();
        return true;
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
        while (true) {
            if (self.study.trials.items.len == 0) {
                try out.writeAll("\nNo trials have been completed.\n");
                return;
            }
            var menu_arena = std.heap.ArenaAllocator.init(gpa);
            defer menu_arena.deinit();
            const ma = menu_arena.allocator();
            const best = try self.study.bestTrials(ma);
            var options = std.ArrayList([]const u8).empty;
            for (best) |bi| try options.append(ma, try self.trialTitle(ma, &self.study.trials.items[bi]));
            try options.append(ma, "Run additional trials");
            try options.append(ma, "Exit");

            try out.writeAll("\nOptimization finished!\n");
            if (self.settings.trial_index == null) {
                try out.writeAll("\nThe following trials resulted in Pareto optimal combinations of the optimization objectives. After selecting a trial, you will be able to save the model or chat with it to test how well it works. You can return to this menu later to select a different trial.\n");
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
                choice = (try self.con.menu("Which trial do you want to use?", options.items)) orelse return;
            }

            if (choice == best.len + 1) return;
            if (choice == best.len) {
                var n_additional: usize = 0;
                if (self.settings.n_additional_trials) |n| {
                    n_additional = n;
                } else {
                    while (true) {
                        const line = (try self.con.ask("How many additional trials do you want to run?")) orelse return;
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
                _ = try self.runTrials();
                continue;
            }

            const trial = &self.study.trials.items[best[choice]];
            try self.restoreTrial(trial);
            const back = try self.modelLoop(trial);
            if (!back) return;
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
                action = (try self.con.menu("What do you want to do with the decensored model?", &options)) orelse return false;
            }
            switch (action) {
                0 => self.saveModel(trial) catch |err| {
                    try out.print("Error while saving the model: {s}\n", .{@errorName(err)});
                    if (self.settings.model_action != null) return err;
                },
                1 => try self.chatLoop(),
                2 => return true,
                else => return false,
            }
        }
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
        const model_id = self.settings.model;
        const is_hf = std.mem.indexOfScalar(u8, model_id, '/') != null and !hf.isLocalDir(self.io, model_id);
        try o.writeAll("---\ntags:\n- ditch\n- heretic\n- uncensored\n- decensored\n- abliterated\n---\n\n");
        try o.writeAll("# This is a decensored version of ");
        if (is_hf) try o.print("[{s}](https://huggingface.co/{s})", .{ model_id, model_id }) else try o.writeAll("a model");
        try o.print(", made using [ditch](https://github.com/p-e-w/heretic) v{s} (a Zig port of [Heretic](https://heretic-project.org))\n\n", .{config.version});
        try o.writeAll("## Abliteration parameters\n\n| Parameter | Value |\n| :-------- | :---: |\n");
        if (trial.direction_index) |di| try o.print("| **direction_index** | {d:.2} |\n", .{di}) else try o.writeAll("| **direction_index** | per layer |\n");
        if (self.settings.n_directions > 1) try o.print("| **n_directions** | {d} |\n", .{self.settings.n_directions});
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
            const line = (try self.con.ask("Path to the folder:")) orelse return;
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
        try out.writeAll("Saving merged model...\n");
        try out.flush();
        var scratch = std.heap.ArenaAllocator.init(self.gpa);
        defer scratch.deinit();
        const card = try self.modelCard(scratch.allocator(), trial);
        try export_mod.saveModel(self.gpa, self.io, self.model, dir, .{
            .max_shard_size = self.settings.max_shard_size,
            .export_dtype = dtype,
            .readme_body = card,
        }, out);
        try out.print("Model saved to {s}.\n", .{dir});
        try out.flush();
    }

    fn chatLoop(self: *App) !void {
        const out = self.con.out;
        const gpa = self.gpa;
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
            try out.writeAll("Assistant: ");
            try out.flush();
            const response = try self.streamResponse(history.items);
            try history.append(gpa, .{ .role = .assistant, .content = response });
            try out.writeAll("\n");
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

    /// Generates a response for the conversation, printing tokens as they are produced.
    fn streamResponse(self: *App, messages: []const chat.Message) ![]u8 {
        const out = self.con.out;
        const gpa = self.gpa;
        const model = self.model;
        const c = &model.config;
        const max_new: usize = 1024;

        const text = try chat.render(gpa, self.template, messages);
        defer gpa.free(text);
        const ids = try model.tokenizer.encode(gpa, text, true);
        defer gpa.free(ids);
        const max_len = ids.len + max_new + 1;
        var ws = try model_mod.Workspace.init(gpa, c, @max(ids.len, 1), 1);
        defer ws.deinit();
        var cache = try model_mod.KvCache.init(gpa, c.num_layers, 1, max_len, c.num_kv_heads * c.head_dim);
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
            pos += 1;
            token = argmax(ws.logits[0..c.vocab_size]);
        }
        const full = try model.tokenizer.decode(gpa, generated.items, true);
        defer gpa.free(full);
        if (full.len > printed.items.len and std.mem.startsWith(u8, full, printed.items)) {
            try out.writeAll(full[printed.items.len..]);
        }
        try out.flush();
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
    while (batch_size <= settings.max_batch_size) {
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

fn printResidualGeometry(out: *Io.Writer, good: []const f32, bad: []const f32, entries: usize, hidden: usize) !void {
    try out.writeAll("\nResidual geometry (per layer entry; entry 0 = embeddings, entry L = output of layer L-1):\n");
    try out.writeAll("  entry   cos(good,bad)     |good|      |bad|   |bad-good|\n");
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
        try out.print("  {d: >5}   {d: >13.4}   {d: >8.3}   {d: >8.3}   {d: >10.3}\n", .{ e, cos, @sqrt(gn), @sqrt(bn), @sqrt(dn) });
    }
}

fn settingsSnapshot(a: Allocator, settings: *const config.Settings, model: *const Model) ![]const u8 {
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

/// Refuses to continue a study whose direction count differs from the settings.
fn checkStudyDirections(study: *const study_mod.Study, settings: *const config.Settings) !void {
    const stored: usize = @intCast(study.settingInteger("n_directions") orelse 1);
    if (stored != settings.n_directions) {
        std.log.err("the checkpoint was created with n_directions = {d}, but n_directions = {d} was requested; pass --n-directions {d} or restart the study", .{ stored, settings.n_directions, stored });
        return error.StudyDirectionsMismatch;
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
    var out_buf: [8192]u8 = undefined;
    var fw = Io.File.Writer.init(.stdout(), io, &out_buf);
    const out = &fw.interface;
    defer out.flush() catch {};
    var in_buf: [4096]u8 = undefined;
    var fr = Io.File.stdin().readerStreaming(io, &in_buf);
    var con = Console{ .out = out, .in = &fr.interface };

    run(init, &con) catch |err| {
        out.flush() catch {};
        std.log.err("{s}", .{@errorName(err)});
        std.process.exit(1);
    };
}

fn run(init: std.process.Init, con: *Console) !void {
    const gpa = init.gpa;
    const arena = init.arena.allocator();
    const io = init.io;
    const out = con.out;

    try out.print("{s}  v{s}  https://github.com/p-e-w/heretic (original)\n\n", .{ banner, config.version });

    // Settings.
    const raw_args = try init.minimal.args.toSlice(arena);
    const args = try arena.alloc([]const u8, raw_args.len);
    for (raw_args, 0..) |a, i| args[i] = a;
    var loaded = try config.load(gpa, io, args);
    defer loaded.deinit();
    const settings = &loaded.settings;
    if (settings.help) {
        try out.writeAll(config.help_text);
        return;
    }
    if (settings.version) {
        try out.print("ditch {s}\n", .{config.version});
        return;
    }
    if (loaded.errors.len > 0) {
        try out.print("Configuration contains {d} error(s):\n", .{loaded.errors.len});
        for (loaded.errors) |e| try out.print("  * {s}\n", .{e});
        try out.writeAll("\nRun ditch --help or see config.default.lua for details about configuration parameters.\n");
        try out.flush();
        std.process.exit(1);
    }
    if (settings.model.len == 0) {
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
    pool.* = tensor.Pool.init(io, settings.threads);
    try out.print("Using {d} threads ({d} CPUs available)\n", .{ pool.threads, cpu_count });
    try out.flush();
    installSigint();

    // Model.
    var http = try hf.Http.init(gpa, io, arena, init.environ_map);
    defer http.deinit();
    const cache_root = try hf.cacheDir(arena, settings, init.environ_map);
    try out.print("\nLoading model {s}...\n", .{settings.model});
    try out.flush();
    const model_dir = try hf.resolveModel(arena, &http, cache_root, settings.model, settings.model_commit, out);
    const model = try Model.load(gpa, io, pool, model_dir);
    defer model.deinit();
    const c = &model.config;
    try out.print("* Architecture: {s} ({d} layers, hidden size {d}, vocabulary {d}, {s} weights)\n", .{ c.model_type, c.num_layers, c.hidden_size, c.vocab_size, model.dtype.safetensorsName() });
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
    var engine = Engine.init(gpa, model, settings, template);
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
    try out.flush();

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
    try out.flush();

    if (settings.evaluate_model) |eval_id| {
        try out.print("\nLoading model {s}...\n", .{eval_id});
        try out.flush();
        const eval_dir = try hf.resolveModel(arena, &http, cache_root, eval_id, null, out);
        const eval_model = try Model.load(gpa, io, pool, eval_dir);
        defer eval_model.deinit();
        var eval_engine = Engine.init(gpa, eval_model, settings, template);
        defer eval_engine.deinit();
        eval_engine.batch_size = engine.batch_size;
        try out.writeAll("* Evaluating...\n");
        try out.flush();
        const scores = try evaluator.scores(arena, &eval_engine, out);
        try printScores(out, scores);
        return;
    }

    if (evaluator.objectiveCount() == 0) {
        std.log.err("no optimization objectives configured: at least one scorer must set optimization to \"minimize\" or \"maximize\"", .{});
        return error.NoObjectives;
    }

    // Residual directions.
    try out.writeAll("\nCalculating per-layer residual directions...\n");
    try out.writeAll("* Obtaining residual mean for good prompts...\n");
    try out.flush();
    const good_means = try engine.getResidualMean(gpa, good_prompts, out);
    defer gpa.free(good_means);
    try out.writeAll("* Obtaining residual mean for bad prompts...\n");
    try out.flush();
    const entries = c.num_layers + 1;
    // With several directions per layer, the same pass over the bad prompts
    // also accumulates the covariance sketch (see directions.zig).
    var sketch: ?directions.Sketch = null;
    defer if (sketch) |*s| s.deinit();
    if (settings.n_directions > 1) sketch = try directions.Sketch.init(gpa, entries, c.hidden_size, settings.n_directions, good_means, settings.seed.?);
    const bad_means = try engine.getResidualMeanSketched(gpa, bad_prompts, out, if (sketch) |*s| s else null);
    defer gpa.free(bad_means);
    if (settings.print_residual_geometry) try printResidualGeometry(out, good_means, bad_means, entries, c.hidden_size);
    if (settings.n_directions > 1) try out.print("* Extracting {d} orthonormal directions per layer...\n", .{settings.n_directions});
    const dirs = try directions.computeBasis(gpa, good_means, bad_means, entries, c.hidden_size, settings.n_directions, settings.orthogonalize_direction, if (sketch) |*s| s else null);
    defer gpa.free(dirs);
    try out.flush();

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
            const choice = (try con.menu("How would you like to proceed?", &opts)) orelse return;
            action = switch (choice) {
                0 => "continue",
                1 => "restart",
                else => return,
            };
        } else {
            try out.writeAll("\nYou have already processed this model, but the run was interrupted. You can continue the previous run from where it stopped. Alternatively, you can ignore the previous run and start from scratch. This will delete the checkpoint file and all results from the previous run.\n");
            const opts = [_][]const u8{ "Continue the previous run", "Ignore the previous run and start from scratch", "Exit" };
            const choice = (try con.menu("How would you like to proceed?", &opts)) orelse return;
            action = switch (choice) {
                0 => "continue",
                1 => "restart",
                else => return,
            };
        }
        if (std.mem.eql(u8, action, "restart")) {
            try study.reset(try settingsSnapshot(arena, settings, model));
        } else if (std.mem.eql(u8, action, "continue")) {
            try checkStudyDirections(&study, settings);
            if (study.finished) show_results_only = true;
            if (study.trials.items.len > settings.n_trials) settings.n_trials = study.trials.items.len;
        } else if (std.mem.eql(u8, action, "exit")) {
            return;
        } else {
            std.log.err("unknown checkpoint action: {s} (expected continue or restart)", .{action});
            return error.InvalidCheckpointAction;
        }
    } else {
        try study.reset(try settingsSnapshot(arena, settings, model));
    }
    try out.print("\nStudy checkpoint: {s}\n", .{checkpoint_path});

    var space = try search.buildSpace(gpa, model);
    defer space.deinit();
    var sampler = tpe.Sampler.init(space.space, settings.n_startup_trials, settings.seed.?);
    var warm: []const tpe.Observation = &.{};
    if (settings.warm_start) |p| warm = try loadWarmStart(gpa, arena, io, p, model, space.dims(), evaluator.objectiveCount(), out);
    if (settings.early_stop and evaluator.earlyStopEntry() == null) {
        try out.writeAll("\nEarly stopping is disabled: it requires a minimised keyword-rate scorer as the last objective (list kl_divergence before keyword_rate).\n");
    }

    var app = App{
        .gpa = gpa,
        .arena = arena,
        .io = io,
        .con = con,
        .settings = settings,
        .http = &http,
        .cache_root = cache_root,
        .pool = pool,
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
        .warm = warm,
    };

    if (!show_results_only) {
        const completed = try app.runTrials();
        if (!completed) try out.writeAll("\nOptimization interrupted. The completed trials have been saved; run ditch again to continue.\n");
    }
    try app.resultsLoop();
    try out.flush();
}
