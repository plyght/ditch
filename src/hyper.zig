//! Hyper-connection residual streams (`Config.hc_mult > 1`): the residual of
//! a token is `hc_mult` parallel vectors. Around every attention and MLP
//! block a *site* collapses the streams into the single block input, and
//! expands the block output back into the streams. Two flavours exist:
//!
//! * Manifold-constrained hyper-connections (mHC; DeepSeek V4 / V4.1,
//!   GLM-5.3-Flash): one projection of the RMS-normalised flattened streams
//!   yields the collapse weights `pre`, the expansion weights `post` and a
//!   stream mixing matrix `comb` projected onto the doubly-stochastic
//!   manifold by Sinkhorn-Knopp. V4 and GLM collapse with the `pre` of the
//!   same site, V4.1 (single pass) with the `pre` computed by the previous
//!   site.
//! * Gated residuals (Qwen4-Exp): every stream is (1 + w)-RMS-normalised, a
//!   low-rank sigmoid mixer weights the normalised streams channel by channel
//!   and their mean is the block input (there is no further input norm); the
//!   block output is added to every stream with a sigmoid injection weight,
//!   and the streams are never mixed.
//!
//! The final collapse before the last norm is the family's head: V4's
//! weighted `hc_head`, V4.1's last collapse weights, GLM's unweighted mean or
//! Qwen4-Exp's mixer (whose output is already normalised; the family has no
//! final norm).
//!
//! For direction extraction ditch defines "the residual at layer L" as the
//! collapsed input of layer L's attention block (the single vector the block
//! reads, before its input norm where the family has one); entry
//! `num_layers` is the final collapse that enters the last norm. The weight
//! edits stay on the attention output projection (which writes into the
//! streams) and the down projections.

const std = @import("std");
const tensor = @import("tensor.zig");
const model_mod = @import("model.zig");
const stream = @import("stream.zig");
const arch = @import("arch.zig");
const dsv4 = @import("deepseek_v4.zig");
const qwen4 = @import("qwen4_exp.zig");

const Allocator = std.mem.Allocator;
const Weight = tensor.Weight;
const Model = model_mod.Model;
const Layer = model_mod.Layer;
const Workspace = model_mod.Workspace;
const KvCache = model_mod.KvCache;
const Row = model_mod.Row;

// ---------------------------------------------------------------------------
// Weights
// ---------------------------------------------------------------------------

/// One mHC site (`attn_hc` / `ffn_hc`).
pub const MhcSite = struct {
    /// `[(2 + hc) * hc][hc * hidden]` mix projection.
    fn_w: Weight,
    /// `[(2 + hc) * hc]` biases: pre, post, then comb rows.
    base: []const f32,
    /// `[3]` scales of pre, post and comb.
    scale: []const f32,
};

/// One gated site (`attn_hyper_connection` / `mlp_hyper_connection`, or the
/// final `hyper_connection_mixer` without `inject`).
pub const GatedSite = struct {
    /// `[hc * hidden]` (1 + w) weights of the per-stream norm.
    norm: []const f32,
    /// `[lowrank][hc * hidden]` and `[hc * hidden][lowrank]` input mixer.
    down: Weight,
    up: Weight,
    /// `[hc][hc * hidden]` injection weights (null on the final mixer).
    inject: ?Weight,
};

pub const Site = union(enum) {
    mhc: MhcSite,
    gated: GatedSite,
};

pub const LayerWeights = struct {
    attn: Site,
    ffn: Site,
};

/// Model-level weights of the final collapse.
pub const Head = union(enum) {
    /// The last site's collapse weights (V4.1) or the mean (GLM): nothing to load.
    none,
    /// DeepSeek V4 `hc_head`.
    weighted: struct {
        /// `[hc][hc * hidden]`
        fn_w: []const f32,
        base: []const f32,
        scale: f32,
    },
    /// Qwen4-Exp `hyper_connection_mixer`: the norm and the input mixer
    /// matrices, acquired for the duration of one forward call.
    gated: struct {
        norm: []const f32,
        down: stream.WeightRef,
        up: stream.WeightRef,
    },
};

/// The resident matrices of a gated head (nothing for the other heads).
pub const HeadLease = struct {
    leases: [2]stream.Lease = undefined,
    n: usize = 0,
    down: Weight = undefined,
    up: Weight = undefined,

    pub fn release(self: *HeadLease, model: *const Model) void {
        const store: *stream.WeightStore = @constCast(&model.store);
        store.releaseSet(self.leases[0..self.n]);
        self.n = 0;
    }
};

pub fn acquireHead(model: *const Model) !HeadLease {
    var lease = HeadLease{};
    switch (model.hyper_head) {
        .gated => |g| {
            const store: *stream.WeightStore = @constCast(&model.store);
            try store.acquireSet(0, &.{ g.down, g.up }, lease.leases[0..2]);
            lease.n = 2;
            lease.down = lease.leases[0].weight;
            lease.up = lease.leases[1].weight;
        },
        else => {},
    }
    return lease;
}

fn cat(arena: Allocator, a: []const u8, b: []const u8) ![]const u8 {
    return std.fmt.allocPrint(arena, "{s}{s}", .{ a, b });
}

const SiteSlots = struct { fn_w: model_mod.Slot, down: model_mod.Slot, up: model_mod.Slot, inject: model_mod.Slot };

/// The store's spelling of an mHC site: the module one (`attn_hc.` + `fn`) or
/// the flat one the released checkpoints use (`hc_attn_` + `fn`). Falls back to
/// the module spelling so its name is what a "missing tensor" error names.
fn sitePrefix(model: *Model, arena: Allocator, lp: []const u8, site: []const u8, flat: ?[]const u8) ![]const u8 {
    const dotted = try std.fmt.allocPrint(arena, "{s}{s}.", .{ lp, site });
    if (model.store.lookup(try cat(arena, dotted, "fn")) != null) return dotted;
    if (flat) |f| {
        const p = try std.fmt.allocPrint(arena, "{s}{s}", .{ lp, f });
        if (model.store.lookup(try cat(arena, p, "fn")) != null) return p;
    }
    return dotted;
}

fn loadSite(model: *Model, layer: *Layer, arena: Allocator, lp: []const u8, site: []const u8, flat: ?[]const u8, slots: SiteSlots) !Site {
    const c = &model.config;
    const hy = c.hyper.?;
    const hc = c.hc_mult;
    const sw = hc * c.hidden_size;
    switch (hy.kind) {
        .mhc, .mhc_single_pass => {
            const mix = (2 + hc) * hc;
            const p = try sitePrefix(model, arena, lp, site, flat);
            return .{ .mhc = .{
                .fn_w = try model_mod.loadMatChecked(model, layer, slots.fn_w, try cat(arena, p, "fn"), mix, sw),
                .base = try model_mod.loadVecChecked(model, try cat(arena, p, "base"), mix),
                .scale = try model_mod.loadVecChecked(model, try cat(arena, p, "scale"), 3),
            } };
        },
        .gated => {
            const p = try cat(arena, lp, site);
            return .{ .gated = .{
                .norm = try model_mod.loadVecChecked(model, try cat(arena, p, ".hc_norm.weight"), sw),
                .down = try model_mod.loadMatChecked(model, layer, slots.down, try cat(arena, p, ".input_mix_weight_down.weight"), hy.lowrank, sw),
                .up = try model_mod.loadMatChecked(model, layer, slots.up, try cat(arena, p, ".input_mix_weight_up.weight"), sw, hy.lowrank),
                .inject = try model_mod.loadMatChecked(model, layer, slots.inject, try cat(arena, p, ".block_inject_weight.weight"), hc, sw),
            } };
        },
    }
}

/// Loads the two sites of layer `li` (prefix `lp`) into `layer.hyper`.
pub fn loadLayer(model: *Model, layer: *Layer, arena: Allocator, lp: []const u8) !void {
    const names = &model.config.arch.names;
    layer.hyper = .{
        .attn = try loadSite(model, layer, arena, lp, names.hc_attn, names.hc_attn_flat, .{ .fn_w = .hc_attn_fn, .down = .hc_attn_down, .up = .hc_attn_up, .inject = .hc_attn_inject }),
        .ffn = try loadSite(model, layer, arena, lp, names.hc_ffn, names.hc_ffn_flat, .{ .fn_w = .hc_ffn_fn, .down = .hc_ffn_down, .up = .hc_ffn_up, .inject = .hc_ffn_inject }),
    };
}

/// Loads the family's head weights.
pub fn loadModel(model: *Model, arena: Allocator) !Head {
    const c = &model.config;
    const hy = c.hyper.?;
    const hc = c.hc_mult;
    const sw = hc * c.hidden_size;
    switch (hy.head) {
        .previous_pre, .mean => return .none,
        .weighted => {
            // `hc_head.hc_fn` (the module spelling) or the released
            // checkpoints' flat `hc_head_fn`.
            const dotted = try std.fmt.allocPrint(arena, "{s}hc_head.hc_", .{model.prefix});
            const p = if (model.store.lookup(try cat(arena, dotted, "fn")) != null)
                dotted
            else
                try std.fmt.allocPrint(arena, "{s}hc_head_", .{model.prefix});
            const scale = try model_mod.loadVecChecked(model, try cat(arena, p, "scale"), 1);
            return .{ .weighted = .{
                .fn_w = try model_mod.loadVecChecked(model, try cat(arena, p, "fn"), hc * sw),
                .base = try model_mod.loadVecChecked(model, try cat(arena, p, "base"), hc),
                .scale = scale[0],
            } };
        },
        .gated_mixer => {
            const p = try std.fmt.allocPrint(arena, "{s}hyper_connection_mixer.", .{model.prefix});
            const down_name = try cat(arena, p, "input_mix_weight_down.weight");
            const up_name = try cat(arena, p, "input_mix_weight_up.weight");
            const down = try model.ref(down_name);
            const up = try model.ref(up_name);
            if (!down.dtype.isFloat() or !up.dtype.isFloat()) {
                std.log.err("{s} is stored as {s}; ditch reads F32/F16/BF16 weights (dequantise the checkpoint first)", .{ down_name, down.dtype.safetensorsName() });
                return error.UnsupportedArchitecture;
            }
            if (down.rows != hy.lowrank or down.cols != sw or up.rows != sw or up.cols != hy.lowrank) {
                std.log.err("{s} is [{d}][{d}], expected [{d}][{d}] (hc_lowrank {d}, hc_count {d})", .{ down_name, down.rows, down.cols, hy.lowrank, sw, hy.lowrank, hc });
                return error.InvalidConfig;
            }
            return .{ .gated = .{
                .norm = try model_mod.loadVecChecked(model, try cat(arena, p, "hc_norm.weight"), sw),
                .down = down,
                .up = up,
            } };
        },
    }
}

// ---------------------------------------------------------------------------
// Sites
// ---------------------------------------------------------------------------

/// The mix of one site for `n` rows: `pre[n][hc]`, `post[n][hc]` and the
/// Sinkhorn-projected `comb[n][hc][hc]` of an mHC site; the normalised
/// streams `normed[n][hc * hidden]`, the mixer weights `w[n][hc * hidden]`
/// and the injection weights `inject[n][hc]` of a gated site.
const Mix = struct {
    pre: []f32 = &.{},
    post: []f32 = &.{},
    comb: []f32 = &.{},
    normed: []f32 = &.{},
    w: []f32 = &.{},
    inject: []f32 = &.{},

    fn init(gpa: Allocator, hy: arch.Hyper, n: usize, hc: usize, hidden: usize) !Mix {
        var m = Mix{};
        errdefer m.deinit(gpa);
        switch (hy.kind) {
            .mhc, .mhc_single_pass => {
                m.pre = try gpa.alloc(f32, n * hc);
                m.post = try gpa.alloc(f32, n * hc);
                m.comb = try gpa.alloc(f32, n * hc * hc);
            },
            .gated => {
                m.normed = try gpa.alloc(f32, n * hc * hidden);
                m.w = try gpa.alloc(f32, n * hc * hidden);
                m.inject = try gpa.alloc(f32, n * hc);
            },
        }
        return m;
    }

    fn deinit(self: *Mix, gpa: Allocator) void {
        if (self.pre.len > 0) gpa.free(self.pre);
        if (self.post.len > 0) gpa.free(self.post);
        if (self.comb.len > 0) gpa.free(self.comb);
        if (self.normed.len > 0) gpa.free(self.normed);
        if (self.w.len > 0) gpa.free(self.w);
        if (self.inject.len > 0) gpa.free(self.inject);
        self.* = .{};
    }
};

/// Computes the site's mix from the streams `x[n][hc * hidden]` (`tmp` holds `n * hc * hidden` floats).
fn siteMix(model: *const Model, site: *const Site, x: []const f32, n: usize, out: *Mix, tmp: []f32) !void {
    const c = &model.config;
    const hy = c.hyper.?;
    const gpa = model.gpa;
    const hc = c.hc_mult;
    const hidden = c.hidden_size;
    const sw = hc * hidden;
    switch (site.*) {
        .mhc => |s| {
            const mix = (2 + hc) * hc;
            // Unweighted RMSNorm over the whole flattened stream, then the projection.
            for (0..n) |t| {
                const src = x[t * sw ..][0..sw];
                const dst = tmp[t * sw ..][0..sw];
                var ss: f32 = 0;
                for (src) |v| ss += v * v;
                const inv = 1.0 / @sqrt(ss / @as(f32, @floatFromInt(sw)) + c.rms_norm_eps);
                for (dst, 0..) |*o, i| o.* = src[i] * inv;
            }
            const proj = try gpa.alloc(f32, n * mix);
            defer gpa.free(proj);
            try tensor.matmulT(model.pool, gpa, proj, tmp, n, s.fn_w, null);
            const eps = hy.eps;
            for (0..n) |t| {
                const p = proj[t * mix ..][0..mix];
                const pre = out.pre[t * hc ..][0..hc];
                const post = out.post[t * hc ..][0..hc];
                const comb = out.comb[t * hc * hc ..][0 .. hc * hc];
                for (0..hc) |j| {
                    pre[j] = sigmoid(p[j] * s.scale[0] + s.base[j]) + eps;
                    post[j] = 2.0 * sigmoid(p[hc + j] * s.scale[1] + s.base[hc + j]);
                }
                for (0..hc) |j| {
                    const row = comb[j * hc ..][0..hc];
                    for (0..hc) |k| row[k] = p[2 * hc + j * hc + k] * s.scale[2] + s.base[2 * hc + j * hc + k];
                    tensor.softmaxInPlace(row);
                    for (row) |*v| v.* += eps;
                }
                sinkhorn(comb, hc, hy.sinkhorn_iters, eps);
            }
        },
        .gated => |s| {
            // (1 + w) RMSNorm per stream, silu(down / hc), sigmoid(up), and
            // the injection weights 2 sigmoid(inject / hc).
            for (0..n) |t| {
                for (0..hc) |j| {
                    const src = x[t * sw + j * hidden ..][0..hidden];
                    tensor.rmsnorm(out.normed[t * sw + j * hidden ..][0..hidden], src, s.norm[j * hidden ..][0..hidden], c.rms_norm_eps, true);
                }
            }
            const lr = try gpa.alloc(f32, n * hy.lowrank);
            defer gpa.free(lr);
            try tensor.matmulT(model.pool, gpa, lr, out.normed[0 .. n * sw], n, s.down, null);
            const inv_hc = 1.0 / @as(f32, @floatFromInt(hc));
            for (lr) |*v| v.* = tensor.silu(v.* * inv_hc);
            try tensor.matmulT(model.pool, gpa, out.w[0 .. n * sw], lr, n, s.up, null);
            for (out.w[0 .. n * sw]) |*v| v.* = sigmoid(v.*);
            if (s.inject) |inj| {
                try tensor.matmulT(model.pool, gpa, out.inject[0 .. n * hc], out.normed[0 .. n * sw], n, inj, null);
                for (out.inject[0 .. n * hc]) |*v| v.* = 2.0 * sigmoid(v.* * inv_hc);
            }
        },
    }
}

/// Projects `m[hc][hc]` onto the doubly-stochastic manifold: columns first,
/// then `iters - 1` rounds of rows then columns (each sum floored by `eps`).
fn sinkhorn(m: []f32, hc: usize, iters: usize, eps: f32) void {
    normColumns(m, hc, eps);
    var i: usize = 1;
    while (i < iters) : (i += 1) {
        for (0..hc) |j| {
            const row = m[j * hc ..][0..hc];
            var s: f32 = 0;
            for (row) |v| s += v;
            const inv = 1.0 / (s + eps);
            for (row) |*v| v.* *= inv;
        }
        normColumns(m, hc, eps);
    }
}

fn normColumns(m: []f32, hc: usize, eps: f32) void {
    for (0..hc) |k| {
        var s: f32 = 0;
        for (0..hc) |j| s += m[j * hc + k];
        const inv = 1.0 / (s + eps);
        for (0..hc) |j| m[j * hc + k] *= inv;
    }
}

inline fn sigmoid(x: f32) f32 {
    return 1.0 / (1.0 + @exp(-x));
}

/// `out[n][hidden] = sum_j pre[j] * x[j]` per row.
fn collapseWeighted(x: []const f32, pre: []const f32, n: usize, hc: usize, hidden: usize, out: []f32) void {
    for (0..n) |t| {
        const dst = out[t * hidden ..][0..hidden];
        @memset(dst, 0);
        for (0..hc) |j| tensor.axpy(dst, pre[t * hc + j], x[t * hc * hidden + j * hidden ..][0..hidden]);
    }
}

/// `out[n][hidden] = mean_j w[j] * normed[j]` per row (gated sites and mixer).
fn collapseGated(normed: []const f32, w: []const f32, n: usize, hc: usize, hidden: usize, out: []f32) void {
    const sw = hc * hidden;
    const inv_hc = 1.0 / @as(f32, @floatFromInt(hc));
    for (0..n) |t| {
        const dst = out[t * hidden ..][0..hidden];
        @memset(dst, 0);
        for (0..hc) |j| {
            const nv = normed[t * sw + j * hidden ..][0..hidden];
            const wv = w[t * sw + j * hidden ..][0..hidden];
            for (dst, 0..) |*o, i| o.* += wv[i] * nv[i];
        }
        tensor.scale(dst, inv_hc);
    }
}

/// The block input of a site: its own or the previous site's collapse
/// weights over the streams (mHC), or the mixer's weighted mean (gated).
fn collapse(hy: arch.Hyper, mix: *const Mix, pre_mix: ?[]const f32, x: []const f32, n: usize, hc: usize, hidden: usize, out: []f32) void {
    switch (hy.kind) {
        .mhc => collapseWeighted(x, mix.pre, n, hc, hidden, out),
        .mhc_single_pass => collapseWeighted(x, pre_mix.?[0 .. n * hc], n, hc, hidden, out),
        .gated => collapseGated(mix.normed, mix.w, n, hc, hidden, out),
    }
}

/// Writes the block output `y[n][hidden]` back into the streams: mHC
/// `x[k] = post[k] * y + sum_j comb[j][k] * x[j]` (`tmp` holds one row of
/// streams), gated `x[k] += inject[k] * y`.
fn expand(hy: arch.Hyper, mix: *const Mix, x: []f32, y: []const f32, n: usize, hc: usize, hidden: usize, tmp: []f32) void {
    const sw = hc * hidden;
    for (0..n) |t| {
        const row = x[t * sw ..][0..sw];
        const yt = y[t * hidden ..][0..hidden];
        switch (hy.kind) {
            .mhc, .mhc_single_pass => {
                @memcpy(tmp[0..sw], row);
                const post = mix.post[t * hc ..][0..hc];
                const comb = mix.comb[t * hc * hc ..][0 .. hc * hc];
                for (0..hc) |k| {
                    const dst = row[k * hidden ..][0..hidden];
                    for (dst, 0..) |*v, i| v.* = post[k] * yt[i];
                    for (0..hc) |j| tensor.axpy(dst, comb[j * hc + k], tmp[j * hidden ..][0..hidden]);
                }
            },
            .gated => {
                for (0..hc) |k| tensor.axpy(row[k * hidden ..][0..hidden], mix.inject[t * hc + k], yt);
            },
        }
    }
}

// ---------------------------------------------------------------------------
// Layer and final collapse
// ---------------------------------------------------------------------------

/// One layer applied in place to the residual streams `x[n][hc * hidden]`.
/// `pre_mix[n][hc]` (single-pass mHC only) carries the collapse weights
/// from the previous site and receives this layer's last ones. When
/// `capture` is given, the collapsed attention input (this layer's residual)
/// is written to it (`[n][hidden]`).
pub fn layerBlock(model: *const Model, layer: *const Layer, li: usize, ws: *Workspace, cache: *KvCache, x: []f32, rows: []const Row, tokens: []const u32, pre_mix: ?[]f32, capture: ?[]f32) !void {
    const c = &model.config;
    const hy = c.hyper.?;
    const gpa = model.gpa;
    const n = rows.len;
    const hidden = c.hidden_size;
    const hc = c.hc_mult;
    const sw = hc * hidden;
    const hw = &layer.hyper.?;

    // Memories added to the streams before the attention site.
    if (layer.dsv4) |*dw| if (dw.engram) |*eg| {
        const cc = &(cache.compress orelse return error.MissingCompressCache);
        try dsv4.engramApply(model, eg, cc, x, rows, tokens);
    };
    if (layer.ngram) |*ng| try qwen4.pleApply(model, ng, cache, x, rows, tokens);

    var mix = try Mix.init(gpa, hy, n, hc, hidden);
    defer mix.deinit(gpa);
    const tmp = try gpa.alloc(f32, n * sw);
    defer gpa.free(tmp);
    const collapsed = try gpa.alloc(f32, n * hidden);
    defer gpa.free(collapsed);
    const h = ws.h[0 .. n * hidden];

    // Attention site.
    try siteMix(model, &hw.attn, x, n, &mix, tmp);
    collapse(hy, &mix, pre_mix, x, n, hc, hidden, collapsed);
    if (capture) |cap| @memcpy(cap[0 .. n * hidden], collapsed);
    model_mod.normRows(c, h, collapsed, n, hidden, layer.input_norm);
    if (c.dsv4 != null) try dsv4.attention(model, layer, li, ws, cache, h, rows) else try model_mod.mixer(model, layer, li, ws, cache, h, rows);
    expand(hy, &mix, x, ws.o[0 .. n * hidden], n, hc, hidden, tmp);
    if (hy.kind == .mhc_single_pass) @memcpy(pre_mix.?[0 .. n * hc], mix.pre);

    // MLP site.
    try siteMix(model, &hw.ffn, x, n, &mix, tmp);
    collapse(hy, &mix, pre_mix, x, n, hc, hidden, collapsed);
    model_mod.normRows(c, h, collapsed, n, hidden, layer.pre_ff_norm);
    try model_mod.mlpBlock(model, layer, li, ws, h, n, tokens);
    expand(hy, &mix, x, ws.m[0 .. n * hidden], n, hc, hidden, tmp);
    if (hy.kind == .mhc_single_pass) @memcpy(pre_mix.?[0 .. n * hc], mix.pre);
}

/// Collapses one row of streams `x[hc * hidden]` into `out[hidden]`: V4
/// through `hc_head`, V4.1 with the last site's `pre_mix[hc]`, GLM as the
/// mean, Qwen4-Exp through the (resident) mixer.
pub fn finalCollapse(model: *const Model, head: *const HeadLease, x: []const f32, pre_mix: ?[]const f32, out: []f32) !void {
    const c = &model.config;
    const hy = c.hyper.?;
    const hc = c.hc_mult;
    const hidden = c.hidden_size;
    const sw = hc * hidden;
    @memset(out, 0);
    switch (hy.head) {
        .previous_pre => for (0..hc) |j| tensor.axpy(out, pre_mix.?[j], x[j * hidden ..][0..hidden]),
        .mean => {
            for (0..hc) |j| tensor.axpy(out, 1.0, x[j * hidden ..][0..hidden]);
            tensor.scale(out, 1.0 / @as(f32, @floatFromInt(hc)));
        },
        .weighted => {
            const w = model.hyper_head.weighted;
            var ss: f32 = 0;
            for (x) |v| ss += v * v;
            const inv = 1.0 / @sqrt(ss / @as(f32, @floatFromInt(sw)) + c.rms_norm_eps);
            for (0..hc) |j| {
                var acc: f32 = 0;
                const wr = w.fn_w[j * sw ..][0..sw];
                for (x, 0..) |v, i| acc += v * inv * wr[i];
                const p = sigmoid(acc * w.scale + w.base[j]) + hy.eps;
                tensor.axpy(out, p, x[j * hidden ..][0..hidden]);
            }
        },
        .gated_mixer => {
            const g = model.hyper_head.gated;
            const gpa = model.gpa;
            const normed = try gpa.alloc(f32, sw);
            defer gpa.free(normed);
            for (0..hc) |j| tensor.rmsnorm(normed[j * hidden ..][0..hidden], x[j * hidden ..][0..hidden], g.norm[j * hidden ..][0..hidden], c.rms_norm_eps, true);
            const lr = try gpa.alloc(f32, hy.lowrank);
            defer gpa.free(lr);
            try tensor.matmulT(model.pool, gpa, lr, normed, 1, head.down, null);
            const inv_hc = 1.0 / @as(f32, @floatFromInt(hc));
            for (lr) |*v| v.* = tensor.silu(v.* * inv_hc);
            const w = try gpa.alloc(f32, sw);
            defer gpa.free(w);
            try tensor.matmulT(model.pool, gpa, w, lr, 1, head.up, null);
            for (w) |*v| v.* = sigmoid(v.*);
            collapseGated(normed, w, 1, hc, hidden, out);
        },
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

test "sinkhorn produces a doubly stochastic matrix" {
    var m = [_]f32{ 0.7, 0.2, 0.1, 0.1, 0.8, 0.1, 0.4, 0.4, 0.2 };
    sinkhorn(&m, 3, 20, 1e-6);
    for (0..3) |j| {
        var rs: f32 = 0;
        var cs: f32 = 0;
        for (0..3) |k| {
            rs += m[j * 3 + k];
            cs += m[k * 3 + j];
        }
        try std.testing.expectApproxEqAbs(@as(f32, 1.0), rs, 1e-4);
        try std.testing.expectApproxEqAbs(@as(f32, 1.0), cs, 1e-4);
    }
}
