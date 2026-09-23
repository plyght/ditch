//! High-level inference helpers used by the optimisation loop and scorers:
//! prompt formatting, tokenisation, batched generation, first-token logits
//! and residual means.

const std = @import("std");
const Io = std.Io;
const model_mod = @import("model.zig");
const chat = @import("chat.zig");
const config = @import("config.zig");
const hf = @import("hf.zig");
const tensor = @import("tensor.zig");
const stream = @import("stream.zig");
const directions = @import("directions.zig");

const Allocator = std.mem.Allocator;
const Model = model_mod.Model;
const Prompt = hf.Prompt;
const Tokenizer = @import("tokenizer.zig").Tokenizer;

/// How progress lines are shown: overwritten in place on a terminal, printed
/// at most a few times per phase otherwise, or not at all (`--quiet`).
pub const ProgressStyle = enum { tty, plain, off };
pub var progress_style: ProgressStyle = .plain;

/// A "done/total" progress line with elapsed and remaining time.
pub const Progress = struct {
    out: *Io.Writer,
    io: Io,
    label: []const u8,
    total: usize,
    start: Io.Timestamp,
    last_quarter: usize = 0,

    pub fn init(out: *Io.Writer, io: Io, label: []const u8, total: usize) Progress {
        return .{ .out = out, .io = io, .label = label, .total = total, .start = Io.Timestamp.now(io, .awake) };
    }

    /// Reports `done` of `total` items complete.
    pub fn update(self: *Progress, done: usize) void {
        const elapsed = @as(f64, @floatFromInt(self.start.durationTo(Io.Timestamp.now(self.io, .awake)).nanoseconds)) / 1e9;
        switch (progress_style) {
            .off => return,
            .tty => {},
            .plain => {
                // Without a terminal: one line per quarter, and only for phases that take a while.
                const quarter = done * 4 / @max(self.total, 1);
                if (quarter == self.last_quarter or elapsed < 2.0) return;
                self.last_quarter = quarter;
            },
        }
        const remaining = if (done > 0) elapsed / @as(f64, @floatFromInt(done)) * @as(f64, @floatFromInt(self.total - @min(done, self.total))) else 0;
        if (progress_style == .tty) self.out.writeAll("\r") catch {};
        self.out.print("  {d}/{d} {s} ({d:.0} s elapsed, {d:.0} s remaining)", .{ done, self.total, self.label, elapsed, remaining }) catch {};
        if (progress_style == .tty) self.out.print("{s: <6}", .{""}) catch {} else self.out.writeAll("\n") catch {};
        self.out.flush() catch {};
    }

    /// Clears the line on a terminal.
    pub fn finish(self: *Progress) void {
        if (progress_style != .tty) return;
        self.out.print("\r{s: <72}\r", .{""}) catch {};
        self.out.flush() catch {};
    }
};

/// The chat format for `model`: a family the `chat_template` setting names,
/// otherwise the model's own template, with the family `chat.detect` picks
/// as its fallback.
pub fn modelFormat(gpa: Allocator, model: *Model, setting: ?[]const u8) !chat.Format {
    const fallback = chat.detect(model.chat_template, model.config.model_type);
    if (setting) |name| if (!std.mem.eql(u8, name, "model")) {
        return chat.Format.named(chat.Template.parse(name) orelse return error.InvalidChatTemplate);
    };
    const tokens = try chat.specialTokens(model.arena.allocator(), model.tokenizer_config_json, model.special_tokens_map_json);
    return chat.Format.init(gpa, model.chat_template, tokens, fallback);
}

pub const Engine = struct {
    gpa: Allocator,
    model: *Model,
    settings: *config.Settings,
    format: chat.Format,
    batch_size: usize,
    /// BOS text the model's own template puts before the first turn but the
    /// rendered family and the tokenizer do not add (see `chat.templateBos`).
    bos_prefix: []const u8 = "",
    /// Whether a rendered chat prompt is encoded with the tokenizer's own
    /// special tokens (its BOS). transformers' `apply_chat_template` encodes
    /// without them, so a BOS appears only where the template writes one;
    /// see `chatAddsSpecial`.
    add_special: bool = true,
    ws: ?model_mod.Workspace = null,
    ws_rows: usize = 0,

    pub fn init(gpa: Allocator, model: *Model, settings: *config.Settings, format: chat.Format) Engine {
        // Templates that write today's date (Llama 3.2, gpt-oss, SmolLM3, Solar Open).
        const now = std.Io.Clock.real.now(model.io);
        const seconds: i64 = @intCast(@divFloor(now.nanoseconds, std.time.ns_per_s));
        chat.today = chat.Date.fromUnix(seconds);
        chat.now_seconds = seconds;
        const special = encodingFor(model.gpa, model.tokenizer, model.chat_template, format);
        return .{
            .gpa = gpa,
            .model = model,
            .settings = settings,
            .format = format,
            .batch_size = @max(settings.batch_size, 1),
            .bos_prefix = special.bos_prefix,
            .add_special = special.add_special,
        };
    }

    pub const Encoding = struct { bos_prefix: []const u8, add_special: bool };

    /// How a rendered prompt is encoded (see `bos_prefix`, `add_special`).
    /// The model's own template is encoded as `apply_chat_template` encodes
    /// it: without the tokenizer's special tokens, so a BOS is exactly where
    /// the template writes one.
    pub fn encodingFor(gpa: Allocator, tok: *const Tokenizer, chat_template: ?[]const u8, format: chat.Format) Encoding {
        if (format.template != null) return .{ .bos_prefix = "", .add_special = false };
        return .{ .bos_prefix = bosPrefix(gpa, tok, chat_template, format.family), .add_special = chatAddsSpecial(tok, chat_template) };
    }

    /// For a named family: false when the tokenizer would prepend a BOS but
    /// the model's template never writes one (arcee-ai/AFM-4.5B, MiniCPM4):
    /// transformers, vLLM and the model's own training render those prompts
    /// without a BOS. Templates that mention `bos_token` or the BOS text keep
    /// the tokenizer's, since the family cannot say where in the Jinja it lands.
    fn chatAddsSpecial(tok: *const Tokenizer, chat_template: ?[]const u8) bool {
        if (!tok.add_bos) return true;
        const t = chat_template orelse return true;
        const bos_id = tok.bos_id orelse return true;
        const bos = tok.id_to_token[bos_id];
        return std.mem.indexOf(u8, t, "bos_token") != null or (bos.len > 0 and std.mem.indexOf(u8, t, bos) != null);
    }

    /// The BOS the rendered prompt must carry itself: the model's template
    /// emits one, the family's rendering does not start with it, and the
    /// tokenizer will not prepend it either.
    fn bosPrefix(gpa: Allocator, tok: *const Tokenizer, chat_template: ?[]const u8, template: chat.Template) []const u8 {
        if (tok.add_bos) return "";
        const bos_id = tok.bos_id orelse return "";
        const bos = tok.id_to_token[bos_id];
        const want = chat.templateBos(chat_template, bos);
        if (want.len == 0) return "";
        const empty = chat.renderPrompt(gpa, template, "", "") catch return "";
        defer gpa.free(empty);
        return if (std.mem.startsWith(u8, empty, bos)) "" else want;
    }

    pub fn deinit(self: *Engine) void {
        self.releaseWorkspace();
    }

    /// Frees the cached forward workspace (it is recreated on the next call);
    /// used to give memory back to the budget before another model runs.
    pub fn releaseWorkspace(self: *Engine) void {
        if (self.ws) |*w| w.deinit();
        self.ws = null;
    }

    /// Rows the forward workspace may hold under the memory budget (see `stream.workspaceRows`).
    pub fn workspaceRows(self: *Engine, wanted: usize, logit_rows: usize, kv_bytes: u64) usize {
        return stream.workspaceRows(self.model, wanted, logit_rows, kv_bytes);
    }

    pub fn ensureWorkspace(self: *Engine, rows: usize, logit_rows: usize, kv_bytes: u64) !*model_mod.Workspace {
        if (self.ws) |*w| {
            if (w.max_rows >= rows and w.max_logit_rows >= logit_rows) return w;
            w.deinit();
            self.ws = null;
        }
        const capped = self.workspaceRows(rows, logit_rows, kv_bytes);
        self.ws = try model_mod.Workspace.init(self.gpa, &self.model.config, capped, logit_rows);
        return &self.ws.?;
    }

    pub fn kvBytes(self: *Engine, batch: usize, max_len: usize) u64 {
        const c = &self.model.config;
        return model_mod.KvCache.bytesFor(c.num_layers, batch, max_len, c.kvDim());
    }

    /// Renders the chat template and appends the response prefix.
    pub fn formatPrompt(self: *Engine, gpa: Allocator, prompt: Prompt) ![]u8 {
        const base = try self.format.prompt(gpa, prompt.system, prompt.user);
        const prefix = self.settings.response_prefix orelse "";
        if (prefix.len == 0 and self.bos_prefix.len == 0) return base;
        defer gpa.free(base);
        return std.fmt.allocPrint(gpa, "{s}{s}{s}", .{ self.bos_prefix, base, prefix });
    }

    pub fn encodePrompt(self: *Engine, gpa: Allocator, prompt: Prompt) ![]u32 {
        const text = try self.formatPrompt(gpa, prompt);
        defer gpa.free(text);
        return self.model.tokenizer.encode(gpa, text, self.add_special);
    }

    fn encodeBatch(self: *Engine, gpa: Allocator, prompts: []const Prompt) ![][]u32 {
        const ids = try gpa.alloc([]u32, prompts.len);
        errdefer gpa.free(ids);
        for (prompts, 0..) |p, i| ids[i] = try self.encodePrompt(gpa, p);
        return ids;
    }

    fn freeBatch(gpa: Allocator, ids: [][]u32) void {
        for (ids) |x| gpa.free(x);
        gpa.free(ids);
    }

    fn totalAndMax(ids: []const []const u32) struct { total: usize, max: usize } {
        var total: usize = 0;
        var max: usize = 0;
        for (ids) |x| {
            total += x.len;
            max = @max(max, x.len);
        }
        return .{ .total = total, .max = max };
    }

    /// Generates greedy responses (decoded text, special tokens kept as in heretic) for all prompts.
    pub fn getResponses(self: *Engine, gpa: Allocator, prompts: []const Prompt, skip_special: bool) ![][]u8 {
        const out = try gpa.alloc([]u8, prompts.len);
        errdefer gpa.free(out);
        var done: usize = 0;
        errdefer for (out[0..done]) |o| gpa.free(o);
        var start: usize = 0;
        while (start < prompts.len) {
            const end = @min(prompts.len, start + self.batch_size);
            const ids = try self.encodeBatch(gpa, prompts[start..end]);
            defer freeBatch(gpa, ids);
            // `model.generate` allocates its result with the model's allocator.
            const tokens = try self.generateBatch(gpa, ids, self.settings.max_response_length);
            defer {
                for (tokens) |t| self.model.gpa.free(t);
                self.model.gpa.free(tokens);
            }
            for (tokens, 0..) |t, i| {
                out[start + i] = try self.model.tokenizer.decode(gpa, t, skip_special);
                done += 1;
            }
            start = end;
        }
        return out;
    }

    /// Generates tokens for one batch of tokenised prompts. The returned
    /// slices are allocated with `self.model.gpa` (see `model_mod.generate`).
    pub fn generateBatch(self: *Engine, gpa: Allocator, ids: []const []u32, max_new_tokens: usize) ![][]u32 {
        const tm = totalAndMax(ids);
        const ws = try self.ensureWorkspace(@max(tm.total, 1), @max(ids.len, 1), self.kvBytes(ids.len, tm.max + max_new_tokens + 1));
        var cache = try model_mod.KvCache.initFor(self.model, gpa, ids.len, tm.max + max_new_tokens + 1);
        defer cache.deinit();
        return model_mod.generate(self.model, ws, &cache, ids, max_new_tokens);
    }

    /// First-token logits for every prompt: `[prompts][vocab]`.
    pub fn getLogits(self: *Engine, gpa: Allocator, prompts: []const Prompt) ![]f32 {
        const c = &self.model.config;
        const out = try gpa.alloc(f32, prompts.len * c.vocab_size);
        errdefer gpa.free(out);
        var start: usize = 0;
        while (start < prompts.len) {
            const end = @min(prompts.len, start + self.batch_size);
            const ids = try self.encodeBatch(gpa, prompts[start..end]);
            defer freeBatch(gpa, ids);
            const tm = totalAndMax(ids);
            const ws = try self.ensureWorkspace(@max(tm.total, 1), @max(ids.len, 1), self.kvBytes(ids.len, tm.max + 1));
            var cache = try model_mod.KvCache.initFor(self.model, gpa, ids.len, tm.max + 1);
            defer cache.deinit();
            try model_mod.prefill(self.model, ws, &cache, ids, out[start * c.vocab_size ..][0 .. ids.len * c.vocab_size], null);
            start = end;
        }
        return out;
    }

    /// Mean residual vector at the last prompt position for every layer entry:
    /// `[num_layers + 1][hidden]`, optionally winsorised per prompt and layer.
    pub fn getResidualMean(self: *Engine, gpa: Allocator, prompts: []const Prompt, progress: ?*Io.Writer) ![]f32 {
        return self.getResidualMeanSketched(gpa, prompts, progress, null);
    }

    /// Like `getResidualMean`; every (winsorised) per-prompt residual is also
    /// fed to `sketch` (see `directions.Sketch`) in the same pass.
    pub fn getResidualMeanSketched(self: *Engine, gpa: Allocator, prompts: []const Prompt, progress: ?*Io.Writer, sketch: ?*directions.Sketch) ![]f32 {
        return self.getResidualMeanObserved(gpa, prompts, progress, .{ .sketch = sketch });
    }

    /// What else the residual pass feeds, and which prompt positions it reads.
    pub const ResidualObservers = struct {
        sketch: ?*directions.Sketch = null,
        moments: ?*directions.Moments = null,
        /// The per-prompt residual is the mean over the last `window` prompt
        /// tokens (1 = the last token only, as in heretic).
        window: usize = 1,
    };

    /// Mean residual `[num_layers + 1][hidden]` over `prompts`; the same
    /// per-prompt residuals also feed the observers.
    pub fn getResidualMeanObserved(self: *Engine, gpa: Allocator, prompts: []const Prompt, progress: ?*Io.Writer, obs: ResidualObservers) ![]f32 {
        const c = &self.model.config;
        const entries = c.num_layers + 1;
        const hidden = c.hidden_size;
        const Acc = struct {
            sum: []f64,
            hidden: usize,
            obs: ResidualObservers,
            fn add(ctx: *@This(), entry: usize, v: []const f32) void {
                const acc = ctx.sum[entry * ctx.hidden ..][0..ctx.hidden];
                for (v, 0..) |x, i| acc[i] += x;
                if (ctx.obs.sketch) |s| s.add(entry, v);
                if (ctx.obs.moments) |m| m.add(entry, v);
            }
        };
        const sum = try gpa.alloc(f64, entries * hidden);
        defer gpa.free(sum);
        @memset(sum, 0);
        var acc = Acc{ .sum = sum, .hidden = hidden, .obs = obs };
        const count = try self.forEachResidual(gpa, prompts, progress, obs.window, &acc);
        const mean = try gpa.alloc(f32, entries * hidden);
        for (mean, 0..) |*m, i| m.* = @floatCast(sum[i] / @as(f64, @floatFromInt(@max(count, 1))));
        return mean;
    }

    /// Projection of every prompt's residual onto its entry's first direction:
    /// `[prompts][entries]`. `dirs` is `[entries][k][hidden]` (`stride = k * hidden`).
    pub fn getProjections(self: *Engine, gpa: Allocator, prompts: []const Prompt, dirs: []const f32, stride: usize, window: usize) ![]f32 {
        const c = &self.model.config;
        const entries = c.num_layers + 1;
        const hidden = c.hidden_size;
        std.debug.assert(dirs.len >= entries * stride);
        const Proj = struct {
            out: []f32,
            dirs: []const f32,
            stride: usize,
            hidden: usize,
            entries: usize,
            prompt: usize = 0,
            fn add(ctx: *@This(), entry: usize, v: []const f32) void {
                ctx.out[ctx.prompt * ctx.entries + entry] = tensor.dot(v, ctx.dirs[entry * ctx.stride ..][0..ctx.hidden]);
                if (entry + 1 == ctx.entries) ctx.prompt += 1;
            }
        };
        const out = try gpa.alloc(f32, prompts.len * entries);
        errdefer gpa.free(out);
        var proj = Proj{ .out = out, .dirs = dirs, .stride = stride, .hidden = hidden, .entries = entries };
        _ = try self.forEachResidual(gpa, prompts, null, window, &proj);
        return out;
    }

    /// Runs the model over `prompts` and calls `ctx.add(entry, residual)` for
    /// every prompt and layer entry (entries in order for each prompt) with
    /// the winsorised residual, averaged over the last `window` prompt tokens.
    /// Returns the number of prompts seen.
    fn forEachResidual(self: *Engine, gpa: Allocator, prompts: []const Prompt, progress: ?*Io.Writer, window: usize, ctx: anytype) !usize {
        var prog: ?Progress = if (progress) |pw| Progress.init(pw, self.model.io, "prompts", prompts.len) else null;
        const c = &self.model.config;
        const entries = c.num_layers + 1;
        const hidden = c.hidden_size;
        const w = @max(window, 1);
        const q = self.settings.winsorization_quantile;
        var sorted: ?[]f32 = null;
        defer if (sorted) |s| gpa.free(s);
        if (q >= 0 and q < 1) sorted = try gpa.alloc(f32, hidden);
        const v = try gpa.alloc(f32, hidden);
        defer gpa.free(v);
        var count: usize = 0;
        var start: usize = 0;
        while (start < prompts.len) {
            const end = @min(prompts.len, start + self.batch_size);
            const ids = try self.encodeBatch(gpa, prompts[start..end]);
            defer freeBatch(gpa, ids);
            const tm = totalAndMax(ids);
            const ws = try self.ensureWorkspace(@max(tm.total, 1), @max(ids.len, 1), self.kvBytes(ids.len, tm.max + 1));
            var cache = try model_mod.KvCache.initFor(self.model, gpa, ids.len, tm.max + 1);
            defer cache.deinit();
            // Flatten the batch; capture the last `w` rows of every prompt.
            const tokens = try gpa.alloc(u32, tm.total);
            defer gpa.free(tokens);
            const rows = try gpa.alloc(model_mod.Row, tm.total);
            defer gpa.free(rows);
            var capture = std.ArrayList(usize).empty;
            defer capture.deinit(gpa);
            const first_capture = try gpa.alloc(usize, ids.len);
            defer gpa.free(first_capture);
            var idx: usize = 0;
            for (ids, 0..) |p, b| {
                const take = @min(w, p.len);
                first_capture[b] = capture.items.len;
                for (p, 0..) |t, pos| {
                    tokens[idx] = t;
                    rows[idx] = .{ .b = b, .pos = pos };
                    if (pos + take >= p.len) try capture.append(gpa, idx);
                    idx += 1;
                }
            }
            const res = try gpa.alloc(f32, entries * capture.items.len * hidden);
            defer gpa.free(res);
            try model_mod.forward(self.model, ws, &cache, tokens, rows, .{ .capture_rows = capture.items, .residuals = res });
            for (ids, 0..) |p, b| {
                const take = @min(w, p.len);
                const inv: f32 = 1.0 / @as(f32, @floatFromInt(@max(take, 1)));
                for (0..entries) |l| {
                    @memset(v, 0);
                    for (0..take) |t| {
                        const ci = first_capture[b] + t;
                        tensor.axpy(v, inv, res[(l * capture.items.len + ci) * hidden ..][0..hidden]);
                    }
                    if (sorted) |s| {
                        for (s, 0..) |*x, i| x.* = @abs(v[i]);
                        std.mem.sort(f32, s, {}, std.sort.asc(f32));
                        const thr = quantile(s, q);
                        for (v) |*x| x.* = std.math.clamp(x.*, -thr, thr);
                    }
                    ctx.add(l, v);
                }
            }
            count += ids.len;
            if (prog) |*pg| pg.update(count);
            start = end;
        }
        if (prog) |*pg| pg.finish();
        return count;
    }

    /// Logits at the last `tails[i]` positions of every token sequence
    /// `seqs[i]`, packed in order: `[Σ tails][vocab]`. One prefill per batch.
    pub fn getLogitsAt(self: *Engine, gpa: Allocator, seqs: []const []const u32, tails: []const usize) ![]f32 {
        const c = &self.model.config;
        std.debug.assert(seqs.len == tails.len);
        var total_rows: usize = 0;
        for (tails) |t| total_rows += t;
        const out = try gpa.alloc(f32, total_rows * c.vocab_size);
        errdefer gpa.free(out);
        var written: usize = 0;
        var start: usize = 0;
        while (start < seqs.len) {
            const end = @min(seqs.len, start + self.batch_size);
            const ids = seqs[start..end];
            const tm = totalAndMax(ids);
            var n_logits: usize = 0;
            for (tails[start..end]) |t| n_logits += t;
            const ws = try self.ensureWorkspace(@max(tm.total, 1), @max(n_logits, 1), self.kvBytes(ids.len, tm.max + 1));
            var cache = try model_mod.KvCache.initFor(self.model, gpa, ids.len, tm.max + 1);
            defer cache.deinit();
            const tokens = try gpa.alloc(u32, tm.total);
            defer gpa.free(tokens);
            const rows = try gpa.alloc(model_mod.Row, tm.total);
            defer gpa.free(rows);
            const logit_rows = try gpa.alloc(usize, n_logits);
            defer gpa.free(logit_rows);
            var idx: usize = 0;
            var li: usize = 0;
            for (ids, 0..) |p, b| {
                const take = @min(tails[start + b], p.len);
                for (p, 0..) |t, pos| {
                    tokens[idx] = t;
                    rows[idx] = .{ .b = b, .pos = pos };
                    if (pos + take >= p.len) {
                        logit_rows[li] = idx;
                        li += 1;
                    }
                    idx += 1;
                }
            }
            try model_mod.forward(self.model, ws, &cache, tokens, rows, .{ .logit_rows = logit_rows[0..li] });
            @memcpy(out[written * c.vocab_size ..][0 .. li * c.vocab_size], ws.logits[0 .. li * c.vocab_size]);
            written += li;
            start = end;
        }
        // Every tail must fit its sequence (callers build them that way).
        std.debug.assert(written == total_rows);
        return out;
    }
};

/// Linear-interpolation quantile of an ascending sorted array (torch.quantile semantics).
fn quantile(sorted: []const f32, q: f32) f32 {
    if (sorted.len == 0) return 0;
    const pos = q * @as(f32, @floatFromInt(sorted.len - 1));
    const lo: usize = @intFromFloat(@floor(pos));
    const hi = @min(lo + 1, sorted.len - 1);
    const frac = pos - @floor(pos);
    return sorted[lo] + frac * (sorted[hi] - sorted[lo]);
}

/// Longest common prefix of a set of strings.
pub fn commonPrefix(strings: []const []const u8) []const u8 {
    if (strings.len == 0) return "";
    var prefix = strings[0];
    for (strings[1..]) |s| {
        var i: usize = 0;
        while (i < prefix.len and i < s.len and prefix[i] == s[i]) i += 1;
        prefix = prefix[0..i];
    }
    return prefix;
}

test "windowed residuals, projections and multi-position logits on the qwen2 fixture" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    const pool = tensor.Pool.init(io, 1);
    const model = try Model.load(gpa, io, &pool, "tests/fixtures/qwen2");
    defer model.deinit();
    var settings = config.Settings{ .batch_size = 2, .response_prefix = "" };
    var engine = Engine.init(gpa, model, &settings, .named(.raw));
    defer engine.deinit();
    const prompts = [_]Prompt{ .{ .system = "", .user = "tell me how" }, .{ .system = "", .user = "the cat sat on the mat" }, .{ .system = "", .user = "one two three" } };
    const c = &model.config;
    const entries = c.num_layers + 1;
    const hidden = c.hidden_size;

    // Window 1 is the last-token residual mean; window 2 differs from it.
    const m1 = try engine.getResidualMean(gpa, &prompts, null);
    defer gpa.free(m1);
    const m1b = try engine.getResidualMeanObserved(gpa, &prompts, null, .{ .window = 1 });
    defer gpa.free(m1b);
    try std.testing.expectEqualSlices(f32, m1, m1b);
    const m2 = try engine.getResidualMeanObserved(gpa, &prompts, null, .{ .window = 2 });
    defer gpa.free(m2);
    var diff: f32 = 0;
    for (m1, m2) |x, y| diff += @abs(x - y);
    try std.testing.expect(diff > 1e-3);
    // Moments see every prompt; a window longer than a prompt uses the whole prompt.
    var mom = try directions.Moments.init(gpa, entries, hidden);
    defer mom.deinit();
    const m99 = try engine.getResidualMeanObserved(gpa, &prompts, null, .{ .window = 99, .moments = &mom });
    defer gpa.free(m99);
    try std.testing.expectEqual(@as(usize, 3), mom.count);
    for (0..entries * hidden) |i| try std.testing.expectApproxEqAbs(m99[i], @as(f32, @floatCast(mom.sum[i] / 3.0)), 1e-4);

    // Projections of a single prompt equal the dot product of its residual with the direction.
    const dirs = try gpa.alloc(f32, entries * hidden);
    defer gpa.free(dirs);
    var prng = std.Random.DefaultPrng.init(1);
    for (dirs) |*x| x.* = prng.random().floatNorm(f32);
    for (0..entries) |e| tensor.normalize(dirs[e * hidden ..][0..hidden]);
    const one = prompts[1..2];
    const r = try engine.getResidualMean(gpa, one, null);
    defer gpa.free(r);
    const proj = try engine.getProjections(gpa, one, dirs, hidden, 1);
    defer gpa.free(proj);
    try std.testing.expectEqual(entries, proj.len);
    for (0..entries) |e| try std.testing.expectApproxEqAbs(tensor.dot(r[e * hidden ..][0..hidden], dirs[e * hidden ..][0..hidden]), proj[e], 1e-4);

    // Logits at the last position equal the first-token logits; two tails give two rows.
    const l1 = try engine.getLogits(gpa, &prompts);
    defer gpa.free(l1);
    var seqs: [3][]u32 = undefined;
    for (&prompts, 0..) |p, i| seqs[i] = try engine.encodePrompt(gpa, p);
    defer for (seqs) |s| gpa.free(s);
    const tails1 = [_]usize{ 1, 1, 1 };
    const la = try engine.getLogitsAt(gpa, &seqs, &tails1);
    defer gpa.free(la);
    for (l1, la) |x, y| try std.testing.expectApproxEqAbs(x, y, 1e-4);
    const tails2 = [_]usize{ 2, 1, 2 };
    const lb = try engine.getLogitsAt(gpa, &seqs, &tails2);
    defer gpa.free(lb);
    try std.testing.expectEqual(@as(usize, 5 * c.vocab_size), lb.len);
    // Row 1 (prompt 0, last position) and row 2 (prompt 1) match the single-tail rows.
    for (0..c.vocab_size) |j| {
        try std.testing.expectApproxEqAbs(la[j], lb[c.vocab_size + j], 1e-4);
        try std.testing.expectApproxEqAbs(la[c.vocab_size + j], lb[2 * c.vocab_size + j], 1e-4);
    }
}

test "quantile and common prefix" {
    const s = [_]f32{ 1, 2, 3, 4 };
    try std.testing.expectApproxEqAbs(@as(f32, 2.5), quantile(&s, 0.5), 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 4), quantile(&s, 1.0), 1e-6);
    const strs = [_][]const u8{ "<think>\nabc", "<think>\nxyz", "<think>\n" };
    try std.testing.expectEqualStrings("<think>\n", commonPrefix(&strs));
}
