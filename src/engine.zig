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

pub const Engine = struct {
    gpa: Allocator,
    model: *Model,
    settings: *config.Settings,
    template: chat.Template,
    batch_size: usize,
    ws: ?model_mod.Workspace = null,
    ws_rows: usize = 0,

    pub fn init(gpa: Allocator, model: *Model, settings: *config.Settings, template: chat.Template) Engine {
        return .{ .gpa = gpa, .model = model, .settings = settings, .template = template, .batch_size = @max(settings.batch_size, 1) };
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

    fn ensureWorkspace(self: *Engine, rows: usize, logit_rows: usize, kv_bytes: u64) !*model_mod.Workspace {
        if (self.ws) |*w| {
            if (w.max_rows >= rows and w.max_logit_rows >= logit_rows) return w;
            w.deinit();
            self.ws = null;
        }
        const capped = self.workspaceRows(rows, logit_rows, kv_bytes);
        self.ws = try model_mod.Workspace.init(self.gpa, &self.model.config, capped, logit_rows);
        return &self.ws.?;
    }

    fn kvBytes(self: *Engine, batch: usize, max_len: usize) u64 {
        const c = &self.model.config;
        return model_mod.KvCache.bytesFor(c.num_layers, batch, max_len, c.num_kv_heads * c.head_dim);
    }

    /// Renders the chat template and appends the response prefix.
    pub fn formatPrompt(self: *Engine, gpa: Allocator, prompt: Prompt) ![]u8 {
        const base = try chat.renderPrompt(gpa, self.template, prompt.system, prompt.user);
        const prefix = self.settings.response_prefix orelse return base;
        defer gpa.free(base);
        return std.fmt.allocPrint(gpa, "{s}{s}", .{ base, prefix });
    }

    pub fn encodePrompt(self: *Engine, gpa: Allocator, prompt: Prompt) ![]u32 {
        const text = try self.formatPrompt(gpa, prompt);
        defer gpa.free(text);
        return self.model.tokenizer.encode(gpa, text, true);
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

    fn totalAndMax(ids: []const []u32) struct { total: usize, max: usize } {
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
        const c = &self.model.config;
        const entries = c.num_layers + 1;
        const hidden = c.hidden_size;
        const sum = try gpa.alloc(f64, entries * hidden);
        defer gpa.free(sum);
        @memset(sum, 0);
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
            const res = try gpa.alloc(f32, entries * ids.len * hidden);
            defer gpa.free(res);
            try model_mod.prefill(self.model, ws, &cache, ids, null, res);
            const q = self.settings.winsorization_quantile;
            var sorted: ?[]f32 = null;
            defer if (sorted) |s| gpa.free(s);
            if (q >= 0 and q < 1) sorted = try gpa.alloc(f32, hidden);
            for (0..entries) |l| {
                for (0..ids.len) |b| {
                    const v = res[(l * ids.len + b) * hidden ..][0..hidden];
                    if (sorted) |s| {
                        for (s, 0..) |*x, i| x.* = @abs(v[i]);
                        std.mem.sort(f32, s, {}, std.sort.asc(f32));
                        const thr = quantile(s, q);
                        for (v) |*x| x.* = std.math.clamp(x.*, -thr, thr);
                    }
                    const acc = sum[l * hidden ..][0..hidden];
                    for (v, 0..) |x, i| acc[i] += x;
                    if (sketch) |s| s.add(l, v);
                }
            }
            count += ids.len;
            if (progress) |w| {
                w.print("\r  {d}/{d} prompts", .{ count, prompts.len }) catch {};
                w.flush() catch {};
            }
            start = end;
        }
        if (progress) |w| {
            w.writeAll("\r") catch {};
            w.print("{s: <40}\r", .{""}) catch {};
        }
        const mean = try gpa.alloc(f32, entries * hidden);
        for (mean, 0..) |*m, i| m.* = @floatCast(sum[i] / @as(f64, @floatFromInt(@max(count, 1))));
        return mean;
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

test "quantile and common prefix" {
    const s = [_]f32{ 1, 2, 3, 4 };
    try std.testing.expectApproxEqAbs(@as(f32, 2.5), quantile(&s, 0.5), 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 4), quantile(&s, 1.0), 1e-6);
    const strs = [_][]const u8{ "<think>\nabc", "<think>\nxyz", "<think>\n" };
    try std.testing.expectEqualStrings("<think>\n", commonPrefix(&strs));
}
