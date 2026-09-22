//! DeepSeek V4 and V4.1 (`deepseek_v4`, `deepseek_v41`): the computations
//! these families add on top of the generic forward pass of model.zig.
//!
//! * Manifold-constrained hyper-connections (mHC): the residual is `hc_mult`
//!   parallel streams. Around every attention and MLP block a learned mix
//!   (one projection of the RMS-normalised, flattened streams) yields the
//!   collapse weights `pre`, the expansion weights `post` and a stream mixing
//!   matrix `comb` projected onto the doubly-stochastic manifold by
//!   Sinkhorn-Knopp. V4 collapses with the `pre` of the same site, V4.1
//!   (single pass) with the `pre` computed by the previous site. For
//!   direction extraction ditch defines "the residual at layer L" as the
//!   collapsed input of layer L's attention block (the single vector the
//!   block reads, before its input norm); entry `num_layers` is the final
//!   collapse that enters the last norm. The weight edits stay on the
//!   attention output projection (`o_b_proj`, which writes into the streams)
//!   and the expert down projections.
//! * Shared-KV sliding-window attention: a low-rank query (`q_a`/`q_b`), one
//!   KV latent per token used as key and value, partial interleaved RoPE on
//!   the trailing channels, per-head sinks, the query's rotation removed from
//!   the output, then a grouped low-rank output projection (`o_a`, `o_b`).
//! * Compressed KV branches. V4 CSA pools two overlapping series over
//!   `2 * ratio` slots with a position bias (then a Lightning Indexer picks
//!   `index_topk` entries per query); V4 HCA pools one series over `ratio`
//!   slots; V4.1 CSA2 pools `ratio` tokens with a softmax gate in the KV
//!   source layer of a group and shares the entries with the group. Entry
//!   `w` is reachable by the query at position `t` when `w < (t + 1) / ratio`.
//!   ditch runs the indexer's dense equivalent: every reachable entry is
//!   attended, which is exactly the reference's top-k while the reachable
//!   entries fit `index_topk` (and, for V4.1, the candidate blocks); longer
//!   contexts are refused rather than approximated. V4.1's quantisation-aware
//!   training rounds the window KV to block-scaled FP8 and the compressed
//!   latents to block-scaled FP4 in every forward pass; both are reproduced.
//! * Hash-routed MoE layers (V4, `tid2eid`) live in moe.zig; the engram
//!   n-gram memory (V4.1) is here: token ids are mapped to a tokenizer-
//!   normalised compressed vocabulary, hashed as 2..N-grams into prime-sized
//!   buckets of a huge embedding table (rows are read one at a time through
//!   the weight store) and gated into every residual stream.

const std = @import("std");
const tensor = @import("tensor.zig");
const model_mod = @import("model.zig");
const stream = @import("stream.zig");
const arch = @import("arch.zig");
const uni = @import("unicode_tables.zig");
const Tokenizer = @import("tokenizer.zig").Tokenizer;

const Allocator = std.mem.Allocator;
const Weight = tensor.Weight;
const Model = model_mod.Model;
const Layer = model_mod.Layer;
const Workspace = model_mod.Workspace;
const KvCache = model_mod.KvCache;
const Row = model_mod.Row;
const Config = arch.Config;

// ---------------------------------------------------------------------------
// Weights
// ---------------------------------------------------------------------------

/// One hyper-connection site (`attn_hc` / `ffn_hc`).
pub const HyperSite = struct {
    /// `[(2 + hc) * hc][hc * hidden]` mix projection.
    fn_w: Weight,
    /// `[(2 + hc) * hc]` biases: pre, post, then comb rows.
    base: []const f32,
    /// `[3]` scales of pre, post and comb.
    scale: []const f32,
};

/// The compressor of a layer that owns a compressed-KV branch.
pub const Compressor = struct {
    /// `[width][hidden]`: `width = 2 * head_dim` for V4 CSA, `head_dim` otherwise.
    kv: Weight,
    /// Absent for a V4.1 branch of ratio 1 (no pooling).
    gate: ?Weight,
    /// `[ratio][width]` added to the gate logits (V4 only).
    position_bias: ?[]const f32,
    /// RMSNorm weight of the pooled entry `[head_dim]`.
    norm: []const f32,
    width: usize,
};

pub const EngramWeights = struct {
    /// `[hidden * (hc + 1)][n_cols * engram_head_dim]`: keys per stream, then the value.
    wkv: Weight,
    /// `[hc][hidden]` each; only their product is used.
    q_weight: []const f32,
    k_weight: []const f32,
    /// The layer's hash table `[num_embeddings][engram_head_dim]`, read row by row.
    table: stream.WeightRef,
    /// Position of this layer in `engram_layer_ids`.
    index: usize,
};

pub const LayerWeights = struct {
    q_a: Weight,
    q_a_norm: []const f32,
    q_b: Weight,
    kv: Weight,
    kv_norm: []const f32,
    /// `[o_groups * o_lora_rank][heads * head_dim / o_groups]` block-diagonal projection.
    o_a: Weight,
    attn_hc: HyperSite,
    ffn_hc: HyperSite,
    compressor: ?Compressor,
    engram: ?EngramWeights,
};

/// V4's final stream collapse (`hc_head`).
pub const HyperHead = struct {
    /// `[hc][hc * hidden]`
    fn_w: []const f32,
    base: []const f32,
    scale: f32,
};

/// Tokenizer- and config-derived engram hash state (V4.1).
pub const EngramState = struct {
    /// Compressed id of every tokenizer id.
    token_map: []const u32,
    pad_cid: i64,
    max_ngram: usize,
    n_heads: usize,
    head_dim: usize,
    n_cols: usize,
    /// `[engram layer][max_ngram - 1][n_heads]` bucket sizes.
    primes: []const u64,
    /// `[engram layer][n_cols]` bucket starts.
    offsets: []const u64,
    /// `[engram layer][max_ngram]` odd hash multipliers.
    multipliers: []const i64,
};

fn cat(arena: Allocator, a: []const u8, b: []const u8) ![]const u8 {
    return std.fmt.allocPrint(arena, "{s}{s}", .{ a, b });
}

fn loadMatChecked(model: *Model, layer: *Layer, slot: model_mod.Slot, name: []const u8, rows: usize, cols: usize) !Weight {
    const r = try model.ref(name);
    if (!r.dtype.isFloat()) {
        std.log.err("{s} is stored as {s}; ditch reads F32/F16/BF16 weights (dequantise the checkpoint first)", .{ name, r.dtype.safetensorsName() });
        return error.UnsupportedArchitecture;
    }
    if (r.rows != rows or r.cols != cols) {
        std.log.err("{s} is [{d}][{d}], expected [{d}][{d}]", .{ name, r.rows, r.cols, rows, cols });
        return error.InvalidConfig;
    }
    layer.refs.add(slot, r, false);
    return model.loadMat(name);
}

fn loadVecChecked(model: *Model, name: []const u8, len: usize) ![]const f32 {
    const v = model.loadVecOpt(name) orelse {
        std.log.err("missing tensor: {s}", .{name});
        return error.MissingWeights;
    };
    if (v.len != len) {
        std.log.err("{s} has {d} elements, expected {d}", .{ name, v.len, len });
        return error.InvalidConfig;
    }
    return v;
}

fn loadSite(model: *Model, layer: *Layer, arena: Allocator, lp: []const u8, site: []const u8, slot: model_mod.Slot) !HyperSite {
    const c = &model.config;
    const hc = c.hc_mult;
    const mix = (2 + hc) * hc;
    const p = try cat(arena, lp, site);
    return .{
        .fn_w = try loadMatChecked(model, layer, slot, try cat(arena, p, ".fn"), mix, hc * c.hidden_size),
        .base = try loadVecChecked(model, try cat(arena, p, ".base"), mix),
        .scale = try loadVecChecked(model, try cat(arena, p, ".scale"), 3),
    };
}

/// Loads the family-specific tensors of layer `li` (prefix `lp`) into
/// `layer.dsv4`, plus the output projection into `layer.o`.
pub fn loadLayer(model: *Model, layer: *Layer, arena: Allocator, li: usize, lp: []const u8) !void {
    const c = &model.config;
    const d = c.dsv4.?;
    const hidden = c.hidden_size;
    const nh = c.num_heads;
    const hd = c.head_dim;
    var w: LayerWeights = undefined;
    w.q_a = try loadMatChecked(model, layer, .d_q_a, try cat(arena, lp, "self_attn.q_a_proj.weight"), d.q_lora_rank, hidden);
    w.q_a_norm = try loadVecChecked(model, try cat(arena, lp, "self_attn.q_a_norm.weight"), d.q_lora_rank);
    w.q_b = try loadMatChecked(model, layer, .d_q_b, try cat(arena, lp, "self_attn.q_b_proj.weight"), nh * hd, d.q_lora_rank);
    w.kv = try loadMatChecked(model, layer, .d_kv, try cat(arena, lp, "self_attn.kv_proj.weight"), hd, hidden);
    w.kv_norm = try loadVecChecked(model, try cat(arena, lp, "self_attn.kv_norm.weight"), hd);
    w.o_a = try loadMatChecked(model, layer, .d_o_a, try cat(arena, lp, "self_attn.o_a_proj.weight"), d.o_groups * d.o_lora_rank, nh * hd / d.o_groups);
    layer.o = try loadMatChecked(model, layer, .o, try cat(arena, lp, c.arch.names.o), hidden, d.o_groups * d.o_lora_rank);
    w.attn_hc = try loadSite(model, layer, arena, lp, "attn_hc", .d_hc_attn);
    w.ffn_hc = try loadSite(model, layer, arena, lp, "ffn_hc", .d_hc_ffn);
    w.compressor = null;
    if (d.branch[li] != .none and d.kv_source[li] == li) {
        const ratio = d.compress_ratio[li];
        const width = if (d.branch[li] == .csa) 2 * hd else hd;
        const cp = try cat(arena, lp, "self_attn.compressor.");
        var comp = Compressor{
            .kv = try loadMatChecked(model, layer, .d_comp_kv, try cat(arena, cp, "kv_proj.weight"), width, hidden),
            .gate = null,
            .position_bias = null,
            .norm = try loadVecChecked(model, try cat(arena, cp, "kv_norm.weight"), hd),
            .width = width,
        };
        if (!d.v41 or ratio > 1) comp.gate = try loadMatChecked(model, layer, .d_comp_gate, try cat(arena, cp, "gate_proj.weight"), width, hidden);
        if (!d.v41) comp.position_bias = try loadVecChecked(model, try cat(arena, cp, "position_bias"), ratio * width);
        w.compressor = comp;
    }
    w.engram = null;
    if (d.engram) |eg| {
        for (eg.layer_ids, 0..) |eli, k| {
            if (eli != li) continue;
            const hc = c.hc_mult;
            const n_cols = (eg.max_ngram - 1) * eg.n_heads;
            const ep = try cat(arena, lp, "engram.");
            const table_name = try std.fmt.allocPrint(arena, "{s}engram_tables.{d}.weight", .{ model.prefix, li });
            const table = try model.ref(table_name);
            if (!table.dtype.isFloat()) {
                std.log.err("{s} is stored as {s}; ditch reads F32/F16/BF16 tables (dequantise the checkpoint first)", .{ table_name, table.dtype.safetensorsName() });
                return error.UnsupportedArchitecture;
            }
            if (table.rows != eg.num_embeddings[k] or table.cols != eg.head_dim) {
                std.log.err("{s} is [{d}][{d}], expected [{d}][{d}]", .{ table_name, table.rows, table.cols, eg.num_embeddings[k], eg.head_dim });
                return error.InvalidConfig;
            }
            w.engram = .{
                .wkv = try loadMatChecked(model, layer, .d_engram_wkv, try cat(arena, ep, "wkv.weight"), hidden * (hc + 1), n_cols * eg.head_dim),
                .q_weight = try loadVecChecked(model, try cat(arena, ep, "q_weight"), hc * hidden),
                .k_weight = try loadVecChecked(model, try cat(arena, ep, "k_weight"), hc * hidden),
                .table = table,
                .index = k,
            };
        }
    }
    layer.dsv4 = w;
}

/// Model-level state: the checkpoint dtype guard, V4's `hc_head` and V4.1's
/// engram hash state (config primes and multipliers plus the compressed
/// token map derived from the tokenizer).
pub fn loadModel(model: *Model, arena: Allocator) !void {
    const c = &model.config;
    const d = c.dsv4.?;
    for (model.files) |f| {
        if (f.skipped_dtype) |dt| {
            std.log.err("{s} stores tensors as {s}; ditch reads F32/F16/BF16 weights (dequantise the checkpoint first)", .{ f.path, dt });
            return error.UnsupportedArchitecture;
        }
    }
    if (!model.embed_ref.dtype.isFloat() or !model.lm_head_ref.dtype.isFloat()) {
        std.log.err("embedding or lm_head stored as {s}; ditch reads F32/F16/BF16 weights", .{model.embed_ref.dtype.safetensorsName()});
        return error.UnsupportedArchitecture;
    }
    const hc = c.hc_mult;
    model.dsv4_head = null;
    model.engram = null;
    if (!d.v41) {
        const p = try std.fmt.allocPrint(arena, "{s}hc_head.", .{model.prefix});
        const scale = try loadVecChecked(model, try cat(arena, p, "hc_scale"), 1);
        model.dsv4_head = .{
            .fn_w = try loadVecChecked(model, try cat(arena, p, "hc_fn"), hc * hc * c.hidden_size),
            .base = try loadVecChecked(model, try cat(arena, p, "hc_base"), hc),
            .scale = scale[0],
        };
    }
    if (d.engram) |eg| model.engram = try buildEngramState(arena, model.tokenizer, eg);
}

// ---------------------------------------------------------------------------
// Compressed-KV cache
// ---------------------------------------------------------------------------

/// Per (compressor-owning layer, batch slot): the compressed entries emitted
/// so far, the source tokens of the partial group not yet pooled and, for V4
/// CSA, the previous window's first series (`Ca`). Also the engram look-back
/// of every batch slot. Slots reset when a sequence restarts at position 0;
/// any other gap in the positions is an error.
pub const CompressCache = struct {
    gpa: Allocator,
    batch: usize,
    hd: usize,
    /// Per model layer: index of the cache it owns, or null.
    index: []?u32,
    n: usize,
    ratio: []usize,
    width: []usize,
    branch: []arch.CompressBranch,
    max_entries: usize,
    max_ratio: usize,
    max_width: usize,
    /// `[n][batch][max_entries][hd]`
    entries: []f32,
    /// `[n][batch][max_ratio][max_width]`
    buf_kv: []f32,
    buf_gate: []f32,
    /// `[n][batch][max_ratio][hd]` (V4 CSA overlap: kv and biased gate of the last window's Ca).
    overlap_kv: []f32,
    overlap_gate: []f32,
    /// `[n][batch]`
    counts: []usize,
    buffered: []usize,
    has_overlap: []bool,
    next_pos: []usize,
    /// Engram look-back `[batch][ctx]` (-1 = nothing there) and the next expected position.
    hist: []i64,
    ctx: usize,
    engram_next: []usize,

    pub fn init(gpa: Allocator, c: *const Config, batch: usize, max_len: usize) !CompressCache {
        const d = c.dsv4.?;
        const index = try gpa.alloc(?u32, c.num_layers);
        errdefer gpa.free(index);
        var n: usize = 0;
        for (0..c.num_layers) |li| {
            if (d.branch[li] != .none and d.kv_source[li] == li) {
                index[li] = @intCast(n);
                n += 1;
            } else index[li] = null;
        }
        const ratio = try gpa.alloc(usize, n);
        errdefer gpa.free(ratio);
        const width = try gpa.alloc(usize, n);
        errdefer gpa.free(width);
        const branch = try gpa.alloc(arch.CompressBranch, n);
        errdefer gpa.free(branch);
        var max_ratio: usize = 1;
        var max_width: usize = 1;
        var max_entries: usize = 1;
        for (0..c.num_layers) |li| {
            const ci = index[li] orelse continue;
            ratio[ci] = d.compress_ratio[li];
            width[ci] = if (d.branch[li] == .csa) 2 * c.head_dim else c.head_dim;
            branch[ci] = d.branch[li];
            max_ratio = @max(max_ratio, ratio[ci]);
            max_width = @max(max_width, width[ci]);
            max_entries = @max(max_entries, max_len / ratio[ci] + 1);
        }
        const hd = c.head_dim;
        const entries = try gpa.alloc(f32, n * batch * max_entries * hd);
        errdefer gpa.free(entries);
        const buf_kv = try gpa.alloc(f32, n * batch * max_ratio * max_width);
        errdefer gpa.free(buf_kv);
        const buf_gate = try gpa.alloc(f32, n * batch * max_ratio * max_width);
        errdefer gpa.free(buf_gate);
        const overlap_kv = try gpa.alloc(f32, n * batch * max_ratio * hd);
        errdefer gpa.free(overlap_kv);
        const overlap_gate = try gpa.alloc(f32, n * batch * max_ratio * hd);
        errdefer gpa.free(overlap_gate);
        const counts = try gpa.alloc(usize, n * batch);
        errdefer gpa.free(counts);
        const buffered = try gpa.alloc(usize, n * batch);
        errdefer gpa.free(buffered);
        const has_overlap = try gpa.alloc(bool, n * batch);
        errdefer gpa.free(has_overlap);
        const next_pos = try gpa.alloc(usize, n * batch);
        errdefer gpa.free(next_pos);
        const ctx: usize = if (d.engram) |eg| eg.max_ngram - 1 else 0;
        const hist = try gpa.alloc(i64, batch * ctx);
        errdefer gpa.free(hist);
        const engram_next = try gpa.alloc(usize, batch);
        errdefer gpa.free(engram_next);
        @memset(counts, 0);
        @memset(buffered, 0);
        @memset(has_overlap, false);
        @memset(next_pos, 0);
        @memset(hist, -1);
        @memset(engram_next, 0);
        return .{
            .gpa = gpa,
            .batch = batch,
            .hd = hd,
            .index = index,
            .n = n,
            .ratio = ratio,
            .width = width,
            .branch = branch,
            .max_entries = max_entries,
            .max_ratio = max_ratio,
            .max_width = max_width,
            .entries = entries,
            .buf_kv = buf_kv,
            .buf_gate = buf_gate,
            .overlap_kv = overlap_kv,
            .overlap_gate = overlap_gate,
            .counts = counts,
            .buffered = buffered,
            .has_overlap = has_overlap,
            .next_pos = next_pos,
            .hist = hist,
            .ctx = ctx,
            .engram_next = engram_next,
        };
    }

    pub fn deinit(self: *CompressCache) void {
        const gpa = self.gpa;
        gpa.free(self.index);
        gpa.free(self.ratio);
        gpa.free(self.width);
        gpa.free(self.branch);
        gpa.free(self.entries);
        gpa.free(self.buf_kv);
        gpa.free(self.buf_gate);
        gpa.free(self.overlap_kv);
        gpa.free(self.overlap_gate);
        gpa.free(self.counts);
        gpa.free(self.buffered);
        gpa.free(self.has_overlap);
        gpa.free(self.next_pos);
        gpa.free(self.hist);
        gpa.free(self.engram_next);
    }

    fn slot(self: *const CompressCache, ci: usize, b: usize) usize {
        return ci * self.batch + b;
    }

    /// Compressed entries of cache `ci`, batch slot `b`: `[max_entries][hd]`.
    pub fn entriesOf(self: *CompressCache, ci: usize, b: usize) []f32 {
        return self.entries[self.slot(ci, b) * self.max_entries * self.hd ..][0 .. self.max_entries * self.hd];
    }

    fn bufKv(self: *CompressCache, ci: usize, b: usize) []f32 {
        return self.buf_kv[self.slot(ci, b) * self.max_ratio * self.max_width ..][0 .. self.max_ratio * self.max_width];
    }

    fn bufGate(self: *CompressCache, ci: usize, b: usize) []f32 {
        return self.buf_gate[self.slot(ci, b) * self.max_ratio * self.max_width ..][0 .. self.max_ratio * self.max_width];
    }

    fn overlapKv(self: *CompressCache, ci: usize, b: usize) []f32 {
        return self.overlap_kv[self.slot(ci, b) * self.max_ratio * self.hd ..][0 .. self.max_ratio * self.hd];
    }

    fn overlapGate(self: *CompressCache, ci: usize, b: usize) []f32 {
        return self.overlap_gate[self.slot(ci, b) * self.max_ratio * self.hd ..][0 .. self.max_ratio * self.hd];
    }
};

// ---------------------------------------------------------------------------
// Hyper-connections
// ---------------------------------------------------------------------------

/// Mix weights of one site for `n` rows: `pre[n][hc]`, `post[n][hc]`,
/// `comb[n][hc][hc]` (Sinkhorn-projected).
const Mix = struct {
    pre: []f32,
    post: []f32,
    comb: []f32,
};

/// Computes the site's mix weights from the streams `x[n][hc * hidden]`.
fn siteMix(model: *const Model, site: *const HyperSite, x: []const f32, n: usize, out: Mix, tmp: []f32) !void {
    const c = &model.config;
    const d = c.dsv4.?;
    const gpa = model.gpa;
    const hc = c.hc_mult;
    const sw = hc * c.hidden_size;
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
    try tensor.matmulT(model.pool, gpa, proj, tmp, n, site.fn_w, null);
    const eps = d.hc_eps;
    for (0..n) |t| {
        const p = proj[t * mix ..][0..mix];
        const pre = out.pre[t * hc ..][0..hc];
        const post = out.post[t * hc ..][0..hc];
        const comb = out.comb[t * hc * hc ..][0 .. hc * hc];
        for (0..hc) |j| {
            pre[j] = sigmoid(p[j] * site.scale[0] + site.base[j]) + eps;
            post[j] = 2.0 * sigmoid(p[hc + j] * site.scale[1] + site.base[hc + j]);
        }
        for (0..hc) |j| {
            const row = comb[j * hc ..][0..hc];
            for (0..hc) |k| row[k] = p[2 * hc + j * hc + k] * site.scale[2] + site.base[2 * hc + j * hc + k];
            tensor.softmaxInPlace(row);
            for (row) |*v| v.* += eps;
        }
        sinkhorn(comb, hc, d.hc_sinkhorn_iters, eps);
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
fn collapse(x: []const f32, pre: []const f32, n: usize, hc: usize, hidden: usize, out: []f32) void {
    for (0..n) |t| {
        const dst = out[t * hidden ..][0..hidden];
        @memset(dst, 0);
        for (0..hc) |j| tensor.axpy(dst, pre[t * hc + j], x[t * hc * hidden + j * hidden ..][0..hidden]);
    }
}

/// `x[k] = post[k] * y + sum_j comb[j][k] * x[j]` per row, in place (`tmp` holds one row of streams).
fn expand(x: []f32, y: []const f32, mix: Mix, n: usize, hc: usize, hidden: usize, tmp: []f32) void {
    const sw = hc * hidden;
    for (0..n) |t| {
        const row = x[t * sw ..][0..sw];
        @memcpy(tmp[0..sw], row);
        const post = mix.post[t * hc ..][0..hc];
        const comb = mix.comb[t * hc * hc ..][0 .. hc * hc];
        const yt = y[t * hidden ..][0..hidden];
        for (0..hc) |k| {
            const dst = row[k * hidden ..][0..hidden];
            for (dst, 0..) |*v, i| v.* = post[k] * yt[i];
            for (0..hc) |j| tensor.axpy(dst, comb[j * hc + k], tmp[j * hidden ..][0..hidden]);
        }
    }
}

/// One layer applied in place to the residual streams `x[n][hc * hidden]`.
/// `pre_mix[n][hc]` (V4.1 only) carries the collapse weights from the
/// previous site and receives this layer's last ones. When `capture` is
/// given, the collapsed attention input (this layer's residual) is written
/// to it (`[n][hidden]`).
pub fn layerBlock(model: *const Model, layer: *const Layer, li: usize, ws: *Workspace, cache: *KvCache, x: []f32, rows: []const Row, tokens: []const u32, pre_mix: ?[]f32, capture: ?[]f32) !void {
    const c = &model.config;
    const d = c.dsv4.?;
    const gpa = model.gpa;
    const n = rows.len;
    const hidden = c.hidden_size;
    const hc = c.hc_mult;
    const sw = hc * hidden;
    const w = &layer.dsv4.?;
    const cc = &(cache.compress orelse return error.MissingCompressCache);

    if (w.engram) |*eg| try engramApply(model, eg, cc, x, rows, tokens);

    const tmp = try gpa.alloc(f32, n * sw);
    defer gpa.free(tmp);
    const pre = try gpa.alloc(f32, n * hc);
    defer gpa.free(pre);
    const post = try gpa.alloc(f32, n * hc);
    defer gpa.free(post);
    const comb = try gpa.alloc(f32, n * hc * hc);
    defer gpa.free(comb);
    const mix = Mix{ .pre = pre, .post = post, .comb = comb };
    const collapsed = try gpa.alloc(f32, n * hidden);
    defer gpa.free(collapsed);
    const h = ws.h[0 .. n * hidden];

    // Attention site.
    try siteMix(model, &w.attn_hc, x, n, mix, tmp);
    collapse(x, if (d.v41) pre_mix.? else pre, n, hc, hidden, collapsed);
    if (capture) |cap| @memcpy(cap[0 .. n * hidden], collapsed);
    for (0..n) |t| tensor.rmsnorm(h[t * hidden ..][0..hidden], collapsed[t * hidden ..][0..hidden], layer.input_norm.?.w, c.rms_norm_eps, false);
    try attention(model, layer, li, ws, cache, h, rows);
    expand(x, ws.o[0 .. n * hidden], mix, n, hc, hidden, tmp);
    if (d.v41) @memcpy(pre_mix.?[0 .. n * hc], pre);

    // MLP site.
    try siteMix(model, &w.ffn_hc, x, n, mix, tmp);
    collapse(x, if (d.v41) pre_mix.? else pre, n, hc, hidden, collapsed);
    for (0..n) |t| tensor.rmsnorm(h[t * hidden ..][0..hidden], collapsed[t * hidden ..][0..hidden], layer.pre_ff_norm.?.w, c.rms_norm_eps, false);
    try model_mod.mlpBlock(model, layer, li, ws, h, n, tokens);
    expand(x, ws.m[0 .. n * hidden], mix, n, hc, hidden, tmp);
    if (d.v41) @memcpy(pre_mix.?[0 .. n * hc], pre);
}

/// Collapses one row of streams `x[hc * hidden]` into `out[hidden]`: V4
/// through `hc_head`, V4.1 with the last site's `pre_mix[hc]`.
pub fn finalCollapse(model: *const Model, x: []const f32, pre_mix: ?[]const f32, out: []f32) void {
    const c = &model.config;
    const d = c.dsv4.?;
    const hc = c.hc_mult;
    const hidden = c.hidden_size;
    @memset(out, 0);
    if (d.v41) {
        for (0..hc) |j| tensor.axpy(out, pre_mix.?[j], x[j * hidden ..][0..hidden]);
        return;
    }
    const head = model.dsv4_head.?;
    const sw = hc * hidden;
    var ss: f32 = 0;
    for (x) |v| ss += v * v;
    const inv = 1.0 / @sqrt(ss / @as(f32, @floatFromInt(sw)) + c.rms_norm_eps);
    for (0..hc) |j| {
        var acc: f32 = 0;
        const wr = head.fn_w[j * sw ..][0..sw];
        for (x, 0..) |v, i| acc += v * inv * wr[i];
        const p = sigmoid(acc * head.scale + head.base[j]) + d.hc_eps;
        tensor.axpy(out, p, x[j * hidden ..][0..hidden]);
    }
}

// ---------------------------------------------------------------------------
// Attention
// ---------------------------------------------------------------------------

/// Interleaved RoPE on the trailing `rd` channels of `x` (`sign = -1` undoes it).
fn ropeTail(x: []f32, rd: usize, cos_row: []const f32, sin_row: []const f32, sign: f32) void {
    const off = x.len - rd;
    var i: usize = 0;
    while (i < rd / 2) : (i += 1) {
        const a = x[off + 2 * i];
        const b = x[off + 2 * i + 1];
        const cs = cos_row[i];
        const sn = sin_row[i] * sign;
        x[off + 2 * i] = a * cs - b * sn;
        x[off + 2 * i + 1] = b * cs + a * sn;
    }
}

const CompRef = struct { ci: usize, ratio: usize };

const AttnCtx = struct {
    model: *const Model,
    cache: *KvCache,
    layer: usize,
    rows: []const Row,
    q: []f32,
    out: []f32,
    sinks: []const f32,
    window: usize,
    /// Compressed branch: cache index and ratio (null: sliding window only).
    comp: ?CompRef,
    n_tasks: usize,
    chunks: usize,
    per: usize,
    scores: []f32,
    max_keys: usize,
};

fn attentionWorker(ctx: *const AttnCtx, start: usize, end: usize) void {
    const model = ctx.model;
    const c = &model.config;
    const hd = c.head_dim;
    const nh = c.num_heads;
    const slot = start / ctx.per;
    const scores_buf = ctx.scores[slot * ctx.max_keys ..][0..ctx.max_keys];
    const stride = ctx.cache.kv_dim;
    var i = start;
    while (i < end) : (i += 1) {
        const task = (i % ctx.per) * ctx.chunks + slot;
        if (task >= ctx.n_tasks) continue;
        const r = task / nh;
        const h = task % nh;
        const row = ctx.rows[r];
        const q = ctx.q[r * nh * hd + h * hd ..][0..hd];
        const out = ctx.out[r * nh * hd + h * hd ..][0..hd];
        const lo: usize = if (row.pos + 1 > ctx.window) row.pos + 1 - ctx.window else 0;
        const n_win = row.pos + 1 - lo;
        var n_comp: usize = 0;
        var entries: []const f32 = &.{};
        if (ctx.comp) |cp| {
            n_comp = (row.pos + 1) / cp.ratio;
            entries = ctx.cache.compress.?.entriesOf(cp.ci, row.b);
        }
        const scores = scores_buf[0 .. n_win + n_comp];
        const kbase = ctx.cache.kSlot(ctx.layer, row.b, lo).ptr;
        for (0..n_win) |p| scores[p] = tensor.dot(q, kbase[p * stride ..][0..hd]) * c.attention_scale;
        for (0..n_comp) |e| scores[n_win + e] = tensor.dot(q, entries[e * hd ..][0..hd]) * c.attention_scale;
        softmaxWithSink(scores, ctx.sinks[h]);
        @memset(out, 0);
        for (0..n_win) |p| tensor.axpy(out, scores[p], kbase[p * stride ..][0..hd]);
        for (0..n_comp) |e| tensor.axpy(out, scores[n_win + e], entries[e * hd ..][0..hd]);
    }
}

fn softmaxWithSink(x: []f32, sink: f32) void {
    var m: f32 = sink;
    for (x) |v| m = @max(m, v);
    var s: f32 = @exp(sink - m);
    for (x) |*v| {
        v.* = @exp(v.* - m);
        s += v.*;
    }
    const inv = 1.0 / s;
    for (x) |*v| v.* *= inv;
}

/// Attention sublayer of layer `li`: reads the normalised block input `h`,
/// writes `ws.o`. Updates the sliding-window KV cache and, for a layer that
/// owns a compressor, its compressed entries before attending.
fn attention(model: *const Model, layer: *const Layer, li: usize, ws: *Workspace, cache: *KvCache, h: []const f32, rows: []const Row) !void {
    const c = &model.config;
    const d = c.dsv4.?;
    const w = &layer.dsv4.?;
    const gpa = model.gpa;
    const n = rows.len;
    const nh = c.num_heads;
    const hd = c.head_dim;
    const rd = c.rotary_dim;
    const half = rd / 2;
    const qlr = d.q_lora_rank;
    const compress_rope = d.branch[li] != .none;
    const cos = if (compress_rope) model.rope_cos_compress else model.rope_cos;
    const sin = if (compress_rope) model.rope_sin_compress else model.rope_sin;

    // Queries: low-rank projection, norm, expansion, (V4) unweighted head norm, rope on the tail.
    const q_res = try gpa.alloc(f32, n * qlr);
    defer gpa.free(q_res);
    try tensor.matmulT(model.pool, gpa, q_res, h, n, w.q_a, null);
    const tmp = try gpa.alloc(f32, @max(qlr, hd));
    defer gpa.free(tmp);
    for (0..n) |t| {
        const row = q_res[t * qlr ..][0..qlr];
        tensor.rmsnorm(tmp[0..qlr], row, w.q_a_norm, c.rms_norm_eps, false);
        @memcpy(row, tmp[0..qlr]);
    }
    try tensor.matmulT(model.pool, gpa, ws.q, q_res, n, w.q_b, null);
    // Keys/values: one latent per token, normed, roped, (V4.1) FP8-rounded.
    try tensor.matmulT(model.pool, gpa, ws.k, h, n, w.kv, null);
    for (0..n) |t| {
        const pos = @min(rows[t].pos, model.rope_len - 1);
        const cr = cos[pos * half ..][0..half];
        const sr = sin[pos * half ..][0..half];
        for (0..nh) |hh| {
            const q = ws.q[t * nh * hd + hh * hd ..][0..hd];
            if (!d.v41) {
                var ss: f32 = 0;
                for (q) |v| ss += v * v;
                const inv = 1.0 / @sqrt(ss / @as(f32, @floatFromInt(hd)) + c.rms_norm_eps);
                for (q) |*v| v.* *= inv;
            }
            ropeTail(q, rd, cr, sr, 1.0);
        }
        const kv = ws.k[t * hd ..][0..hd];
        tensor.rmsnorm(tmp[0..hd], kv, w.kv_norm, c.rms_norm_eps, false);
        @memcpy(kv, tmp[0..hd]);
        ropeTail(kv, rd, cr, sr, 1.0);
        if (d.fake_quant) fakeQuantFp8(kv, 32);
        @memcpy(cache.kSlot(li, rows[t].b, rows[t].pos), kv);
        cache.noteWrite(rows[t].pos);
    }

    var comp: ?CompRef = null;
    if (d.branch[li] != .none) {
        const cc = &(cache.compress orelse return error.MissingCompressCache);
        const src = d.kv_source[li].?;
        const ci = cc.index[src] orelse return error.MissingCompressCache;
        if (src == li) try compressorUpdate(model, w.compressor.?, li, cc, ci, h, rows);
        const ratio = d.compress_ratio[li];
        // The dense equivalent of the indexer is exact only while every
        // reachable entry would be selected.
        if (d.branch[li] != .hca) {
            var max_reach: usize = 0;
            for (rows) |row| max_reach = @max(max_reach, (row.pos + 1) / ratio);
            if (max_reach > d.index_topk) {
                std.log.err("layer {d}: {d} compressed entries are reachable but the indexer keeps index_topk = {d}; the dense equivalent is only exact for contexts up to {d} tokens", .{ li, max_reach, d.index_topk, d.index_topk * ratio });
                return error.ContextExceedsSparseIndexer;
            }
            if (d.candidate_source) |_| {
                const blocks = (max_reach + d.candidate_block_size - 1) / d.candidate_block_size;
                if (blocks > d.candidate_topk_blocks) {
                    std.log.err("layer {d}: {d} candidate blocks are reachable but candidate_topk_blocks = {d}; the dense equivalent is only exact for contexts up to {d} tokens", .{ li, blocks, d.candidate_topk_blocks, d.candidate_topk_blocks * d.candidate_block_size * ratio });
                    return error.ContextExceedsSparseIndexer;
                }
            }
        }
        for (rows) |row| {
            if ((row.pos + 1) / ratio > cc.counts[cc.slot(ci, row.b)]) return error.NonContiguousRows;
        }
        comp = .{ .ci = ci, .ratio = ratio };
    }

    const window = c.sliding_window orelse std.math.maxInt(usize);
    const n_tasks = n * nh;
    const chunks = @max(1, @min(model.pool.threads, n_tasks));
    const per = (n_tasks + chunks - 1) / chunks;
    var max_keys: usize = 1;
    for (rows) |row| {
        var keys = @min(row.pos + 1, window);
        if (comp) |cp| keys += (row.pos + 1) / cp.ratio;
        max_keys = @max(max_keys, keys);
    }
    const scores = try model.pool.allocScratch(gpa, chunks * max_keys);
    defer model.pool.freeScratch(gpa, scores);
    const actx = AttnCtx{
        .model = model,
        .cache = cache,
        .layer = li,
        .rows = rows,
        .q = ws.q,
        .out = ws.attn,
        .sinks = layer.sinks.?,
        .window = window,
        .comp = comp,
        .n_tasks = n_tasks,
        .chunks = chunks,
        .per = per,
        .scores = scores,
        .max_keys = max_keys,
    };
    model.pool.parallelFor(chunks * per, &actx, attentionWorker);

    // Undo the query rotation on the output, then the grouped output projection.
    const per_group = nh * hd / d.o_groups;
    const grouped = try gpa.alloc(f32, n * d.o_groups * d.o_lora_rank);
    defer gpa.free(grouped);
    const gin = try gpa.alloc(f32, n * per_group);
    defer gpa.free(gin);
    const gout = try gpa.alloc(f32, n * d.o_lora_rank);
    defer gpa.free(gout);
    for (0..n) |t| {
        const pos = @min(rows[t].pos, model.rope_len - 1);
        const cr = cos[pos * half ..][0..half];
        const sr = sin[pos * half ..][0..half];
        for (0..nh) |hh| ropeTail(ws.attn[t * nh * hd + hh * hd ..][0..hd], rd, cr, sr, -1.0);
    }
    const es = w.o_a.dtype.size();
    for (0..d.o_groups) |g| {
        for (0..n) |t| @memcpy(gin[t * per_group ..][0..per_group], ws.attn[t * nh * hd + g * per_group ..][0..per_group]);
        const block = Weight{
            .data = w.o_a.data[g * d.o_lora_rank * per_group * es ..][0 .. d.o_lora_rank * per_group * es],
            .dtype = w.o_a.dtype,
            .rows = d.o_lora_rank,
            .cols = per_group,
        };
        try tensor.matmulT(model.pool, gpa, gout, gin, n, block, null);
        for (0..n) |t| @memcpy(grouped[t * d.o_groups * d.o_lora_rank + g * d.o_lora_rank ..][0..d.o_lora_rank], gout[t * d.o_lora_rank ..][0..d.o_lora_rank]);
    }
    try tensor.matmulT(model.pool, gpa, ws.o, grouped, n, layer.o, if (layer.o_delta) |*dl| dl else null);
}

// ---------------------------------------------------------------------------
// Compressors
// ---------------------------------------------------------------------------

/// Projects the block inputs of `rows` and pools every completed group of
/// `ratio` tokens into a compressed entry of cache `ci` (per batch slot, in
/// position order).
fn compressorUpdate(model: *const Model, comp: Compressor, li: usize, cc: *CompressCache, ci: usize, h: []const f32, rows: []const Row) !void {
    const c = &model.config;
    const d = c.dsv4.?;
    const gpa = model.gpa;
    const n = rows.len;
    const hd = c.head_dim;
    const rd = c.rotary_dim;
    const half = rd / 2;
    const ratio = d.compress_ratio[li];
    const width = comp.width;
    const branch = d.branch[li];

    const kvp = try gpa.alloc(f32, n * width);
    defer gpa.free(kvp);
    try tensor.matmulT(model.pool, gpa, kvp, h, n, comp.kv, null);
    const gatep = try gpa.alloc(f32, n * width);
    defer gpa.free(gatep);
    if (comp.gate) |g| try tensor.matmulT(model.pool, gpa, gatep, h, n, g, null) else @memset(gatep, 0);

    const n_slots = if (branch == .csa) 2 * ratio else ratio;
    const slot_kv = try gpa.alloc(f32, n_slots * hd);
    defer gpa.free(slot_kv);
    const slot_gate = try gpa.alloc(f32, n_slots * hd);
    defer gpa.free(slot_gate);
    const entry = try gpa.alloc(f32, hd);
    defer gpa.free(entry);
    const tmp = try gpa.alloc(f32, hd);
    defer gpa.free(tmp);

    for (0..n) |t| {
        const b = rows[t].b;
        const pos = rows[t].pos;
        const s = cc.slot(ci, b);
        if (pos == 0) {
            cc.counts[s] = 0;
            cc.buffered[s] = 0;
            cc.has_overlap[s] = false;
            cc.next_pos[s] = 0;
        }
        if (pos != cc.next_pos[s]) return error.NonContiguousRows;
        cc.next_pos[s] = pos + 1;
        const buf_kv = cc.bufKv(ci, b);
        const buf_gate = cc.bufGate(ci, b);
        const i = cc.buffered[s];
        @memcpy(buf_kv[i * width ..][0..width], kvp[t * width ..][0..width]);
        @memcpy(buf_gate[i * width ..][0..width], gatep[t * width ..][0..width]);
        cc.buffered[s] = i + 1;
        if (cc.buffered[s] < ratio) continue;
        cc.buffered[s] = 0;
        const g = cc.counts[s];
        if (g >= cc.max_entries) return error.CacheFull;
        // Lay out the pooling slots.
        switch (branch) {
            .csa => {
                // Previous window's first series (Ca), or nothing for the first window.
                const okv = cc.overlapKv(ci, b);
                const ogate = cc.overlapGate(ci, b);
                for (0..ratio) |j| {
                    if (cc.has_overlap[s]) {
                        @memcpy(slot_kv[j * hd ..][0..hd], okv[j * hd ..][0..hd]);
                        @memcpy(slot_gate[j * hd ..][0..hd], ogate[j * hd ..][0..hd]);
                    } else {
                        @memset(slot_kv[j * hd ..][0..hd], 0);
                        @memset(slot_gate[j * hd ..][0..hd], -std.math.inf(f32));
                    }
                }
                // This window's second series (Cb); its Ca becomes the next overlap.
                const pb = comp.position_bias.?;
                for (0..ratio) |j| {
                    const kvj = buf_kv[j * width ..][0..width];
                    const gj = buf_gate[j * width ..][0..width];
                    const pbj = pb[j * width ..][0..width];
                    for (0..hd) |k| {
                        slot_kv[(ratio + j) * hd + k] = kvj[hd + k];
                        slot_gate[(ratio + j) * hd + k] = gj[hd + k] + pbj[hd + k];
                        okv[j * hd + k] = kvj[k];
                        ogate[j * hd + k] = gj[k] + pbj[k];
                    }
                }
                cc.has_overlap[s] = true;
            },
            .hca => {
                const pb = comp.position_bias.?;
                for (0..ratio) |j| {
                    @memcpy(slot_kv[j * hd ..][0..hd], buf_kv[j * width ..][0..hd]);
                    for (0..hd) |k| slot_gate[j * hd + k] = buf_gate[j * width + k] + pb[j * width + k];
                }
            },
            .shared => {
                for (0..ratio) |j| {
                    @memcpy(slot_kv[j * hd ..][0..hd], buf_kv[j * width ..][0..hd]);
                    @memcpy(slot_gate[j * hd ..][0..hd], buf_gate[j * width ..][0..hd]);
                }
            },
            .none => unreachable,
        }
        // Softmax over the slots per channel, weighted sum, norm, rope at the
        // group's first position, (V4.1) FP4 rounding.
        if (comp.gate == null) {
            @memcpy(entry, slot_kv[0..hd]);
        } else {
            for (0..hd) |k| {
                var m: f32 = -std.math.inf(f32);
                for (0..n_slots) |j| m = @max(m, slot_gate[j * hd + k]);
                var sum: f32 = 0;
                var acc: f32 = 0;
                for (0..n_slots) |j| {
                    const e = @exp(slot_gate[j * hd + k] - m);
                    sum += e;
                    acc += e * slot_kv[j * hd + k];
                }
                entry[k] = acc / sum;
            }
        }
        tensor.rmsnorm(tmp, entry, comp.norm, c.rms_norm_eps, false);
        const gpos = @min(g * ratio, model.rope_len - 1);
        ropeTail(tmp, rd, model.rope_cos_compress[gpos * half ..][0..half], model.rope_sin_compress[gpos * half ..][0..half], 1.0);
        if (d.fake_quant) fakeQuantFp4(tmp, 16, true);
        @memcpy(cc.entriesOf(ci, b)[g * hd ..][0..hd], tmp);
        cc.counts[s] = g + 1;
    }
}

// ---------------------------------------------------------------------------
// Fake quantisation (V4.1 quantisation-aware training semantics)
// ---------------------------------------------------------------------------

/// `2^ceil(log2(t))` for `t > 0` (the reference's ue8m0 scale rounding).
fn pow2Ceil(t: f32) f32 {
    const bits: u32 = @bitCast(t);
    const exponent: i32 = @intCast((bits >> 23) & 0xFF);
    const mantissa = bits & 0x7FFFFF;
    const k = exponent - 127 + @as(i32, if (mantissa != 0) 1 else 0);
    return std.math.ldexp(@as(f32, 1.0), k);
}

/// Round-to-nearest-even onto the float8 e4m3fn grid (3 mantissa bits, min
/// normal 2^-6, subnormal step 2^-9, max 448; inputs are clamped to ±448).
pub fn roundE4m3(v: f32) f32 {
    if (v == 0 or std.math.isNan(v)) return v;
    const a = @min(@abs(v), 448.0);
    const step: f32 = if (a < 0.015625) 0.001953125 else std.math.ldexp(@as(f32, 1.0), std.math.frexp(a).exponent - 1 - 3);
    const q = roundHalfEven(a / step) * step;
    return if (v < 0) -q else q;
}

fn roundHalfEven(x: f32) f32 {
    const f = @floor(x);
    const diff = x - f;
    if (diff < 0.5) return f;
    if (diff > 0.5) return f + 1;
    return if (@mod(f, 2.0) == 0) f else f + 1;
}

/// Block-wise FP8 rounding with power-of-two scales (`_fake_quant_fp8_block`).
/// Vectors whose length is not a multiple of `block` are left as they are, like the reference.
pub fn fakeQuantFp8(x: []f32, block: usize) void {
    if (x.len % block != 0) return;
    var i: usize = 0;
    while (i < x.len) : (i += block) {
        const blk = x[i..][0..block];
        var amax: f32 = 0;
        for (blk) |v| amax = @max(amax, @abs(v));
        amax = @max(amax, 1e-4);
        const scale = pow2Ceil(amax * (1.0 / 448.0));
        for (blk) |*v| v.* = roundE4m3(std.math.clamp(v.* / scale, -448.0, 448.0)) * scale;
    }
}

/// Code of `q` (pre-clamped to ±6) on the e2m1 grid with the reference's tie
/// rule: ties at even-code boundaries round down, at odd-code boundaries up.
fn e2m1Code(q: f32) u4 {
    const magnitude = @abs(q);
    const boundaries = [_]f32{ 0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0 };
    const ties_up = [_]bool{ false, true, false, true, false, true, false };
    var code: u4 = 0;
    for (boundaries, 0..) |bnd, i| {
        const threshold = if (ties_up[i]) bnd else std.math.nextAfter(f32, bnd, std.math.inf(f32));
        if (magnitude >= threshold) code += 1;
    }
    if (std.math.signbit(q)) code |= 8;
    return code;
}

const fp4_table = [16]f32{ 0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0 };

/// Block-wise FP4 (e2m1) rounding (`_fake_quant_fp4_block`) with e4m3 scales
/// (`e4m3_scales`) or power-of-two scales.
pub fn fakeQuantFp4(x: []f32, block: usize, e4m3_scales: bool) void {
    if (x.len % block != 0) return;
    var i: usize = 0;
    while (i < x.len) : (i += block) {
        const blk = x[i..][0..block];
        var amax: f32 = 0;
        for (blk) |v| amax = @max(amax, @abs(v));
        const scale = if (e4m3_scales)
            roundE4m3(@max(amax, 6.0 * 0.001953125) / 6.0)
        else
            pow2Ceil(@max(amax, 6.0 * std.math.ldexp(@as(f32, 1.0), -126)) * (1.0 / 6.0));
        for (blk) |*v| v.* = fp4_table[e2m1Code(std.math.clamp(v.* / scale, -6.0, 6.0))] * scale;
    }
}

// ---------------------------------------------------------------------------
// Engram (V4.1)
// ---------------------------------------------------------------------------

/// Adds the n-gram memory of `eg` to the streams `x[n][hc * hidden]` of `rows`.
fn engramApply(model: *const Model, eg: *const EngramWeights, cc: *CompressCache, x: []f32, rows: []const Row, tokens: []const u32) !void {
    const c = &model.config;
    const es = model.engram orelse return error.MissingEngramState;
    const gpa = model.gpa;
    const n = rows.len;
    const hidden = c.hidden_size;
    const hc = c.hc_mult;
    const sw = hc * hidden;
    const ehd = es.head_dim;
    const n_cols = es.n_cols;
    const ctx = es.max_ngram - 1;
    const store: *stream.WeightStore = @constCast(&model.store);

    // Gather the hashed rows of every token (per batch slot in position order).
    const rows_flat = try gpa.alloc(f32, n * n_cols * ehd);
    defer gpa.free(rows_flat);
    const ngram = try gpa.alloc(i64, es.max_ngram);
    defer gpa.free(ngram);
    for (0..n) |t| {
        const b = rows[t].b;
        const pos = rows[t].pos;
        const hist = cc.hist[b * ctx ..][0..ctx];
        if (pos == 0) {
            @memset(hist, -1);
            cc.engram_next[b] = 0;
        }
        if (pos != cc.engram_next[b]) return error.NonContiguousRows;
        cc.engram_next[b] = pos + 1;
        const tid = tokens[t];
        const cid: i64 = if (tid < es.token_map.len) es.token_map[tid] else es.pad_cid;
        // Look back: once a shift hits the sequence start, the longer n-grams hash the pad id.
        var blocked = false;
        for (0..es.max_ngram) |shift| {
            const src: i64 = if (shift == 0) cid else hist[ctx - shift];
            blocked = blocked or src < 0;
            ngram[shift] = if (blocked) es.pad_cid else src;
        }
        for (0..ctx) |k| {
            if (k + 1 < ctx) hist[k] = hist[k + 1];
        }
        if (ctx > 0) hist[ctx - 1] = cid;
        const mult = es.multipliers[eg.index * es.max_ngram ..][0..es.max_ngram];
        var rolling: u64 = @bitCast(ngram[0] * mult[0]);
        for (1..es.max_ngram) |i| {
            rolling ^= @as(u64, @bitCast(ngram[i] * mult[i]));
            for (0..es.n_heads) |hh| {
                const col = (i - 1) * es.n_heads + hh;
                const prime = es.primes[(eg.index * ctx + (i - 1)) * es.n_heads + hh];
                const row_index = rolling % prime + es.offsets[eg.index * n_cols + col];
                try store.readRow(eg.table, @intCast(row_index), rows_flat[(t * n_cols + col) * ehd ..][0..ehd]);
            }
        }
    }
    // Keys per stream and one value, then the normalised-dot gate per stream.
    const kv = try gpa.alloc(f32, n * hidden * (hc + 1));
    defer gpa.free(kv);
    try tensor.matmulT(model.pool, gpa, kv, rows_flat, n, eg.wkv, null);
    const inv_sqrt_h = 1.0 / @sqrt(@as(f32, @floatFromInt(hidden)));
    for (0..n) |t| {
        const streams = x[t * sw ..][0..sw];
        const kvt = kv[t * hidden * (hc + 1) ..][0 .. hidden * (hc + 1)];
        const value = kvt[hc * hidden ..][0..hidden];
        for (0..hc) |j| {
            const hj = streams[j * hidden ..][0..hidden];
            const key = kvt[j * hidden ..][0..hidden];
            var hh: f32 = 0;
            var kk: f32 = 0;
            var dot: f32 = 0;
            for (0..hidden) |i| {
                hh += hj[i] * hj[i];
                kk += key[i] * key[i];
                dot += hj[i] * (eg.q_weight[j * hidden + i] * eg.k_weight[j * hidden + i]) * key[i];
            }
            const hf: f32 = @floatFromInt(hidden);
            const rstd = (1.0 / @sqrt(hh / hf + c.rms_norm_eps)) * (1.0 / @sqrt(kk / hf + c.rms_norm_eps));
            dot = dot * rstd * inv_sqrt_h;
            const mag = @sqrt(@max(@abs(dot), 1e-6));
            const gate = sigmoid(if (dot < 0) -mag else mag);
            tensor.axpy(hj, gate, value);
        }
    }
}

fn isPrime(n: u64) bool {
    if (n < 2) return false;
    if (n % 2 == 0) return n == 2;
    var i: u64 = 3;
    while (i * i <= n) : (i += 2) {
        if (n % i == 0) return false;
    }
    return true;
}

/// Builds the hash state: prime buckets (drawn in order above
/// `engram_vocab_size`, never reused), per-layer offsets, per-layer
/// multipliers from numpy's `default_rng(10007 * layer_id)` and the
/// compressed token map derived from the tokenizer.
pub fn buildEngramState(arena: Allocator, tokenizer: *const Tokenizer, eg: arch.Engram) !EngramState {
    const n_layers = eg.layer_ids.len;
    const ctx = eg.max_ngram - 1;
    const n_cols = ctx * eg.n_heads;
    const primes = try arena.alloc(u64, n_layers * ctx * eg.n_heads);
    var seen = std.AutoHashMap(u64, void).init(arena);
    for (0..n_layers) |l| {
        for (0..ctx) |i| {
            var current: u64 = eg.vocab_size - 1;
            for (0..eg.n_heads) |hh| {
                current += 1;
                while (!isPrime(current) or seen.contains(current)) current += 1;
                try seen.put(current, {});
                primes[(l * ctx + i) * eg.n_heads + hh] = current;
            }
        }
    }
    const offsets = try arena.alloc(u64, n_layers * n_cols);
    for (0..n_layers) |l| {
        var total: u64 = 0;
        for (0..n_cols) |col| {
            offsets[l * n_cols + col] = total;
            total += primes[l * n_cols + col];
        }
        if (total > eg.num_embeddings[l]) {
            std.log.err("engram layer {d}: the hash buckets need {d} rows but engram_num_embeddings gives {d}", .{ eg.layer_ids[l], total, eg.num_embeddings[l] });
            return error.InvalidConfig;
        }
    }
    const multipliers = try arena.alloc(i64, n_layers * eg.max_ngram);
    const bound: u64 = @max(1, (@as(u64, std.math.maxInt(i64)) / eg.compressed_vocab_size) / 2);
    for (0..n_layers) |l| {
        var rng = NumpyRng.init(10007 * @as(u64, eg.layer_ids[l]));
        for (0..eg.max_ngram) |i| multipliers[l * eg.max_ngram + i] = @intCast(rng.integer(bound) * 2 + 1);
    }
    const map = try compressedTokenMap(arena, tokenizer);
    if (map.count != eg.compressed_vocab_size) {
        std.log.err("the tokenizer-derived compressed vocabulary has {d} entries but engram_compressed_vocab_size is {d}; the engram hashes would not match the tables", .{ map.count, eg.compressed_vocab_size });
        return error.InvalidConfig;
    }
    if (eg.pad_id >= map.ids.len) return error.InvalidConfig;
    return .{
        .token_map = map.ids,
        .pad_cid = map.ids[eg.pad_id],
        .max_ngram = eg.max_ngram,
        .n_heads = eg.n_heads,
        .head_dim = eg.head_dim,
        .n_cols = n_cols,
        .primes = primes,
        .offsets = offsets,
        .multipliers = multipliers,
    };
}

const TokenMap = struct { ids: []u32, count: usize };

/// Maps every tokenizer id onto the compressed vocabulary: tokens whose
/// decoded text normalises alike (NFKC, NFD, accents stripped, lowercased,
/// whitespace collapsed and trimmed) share an id; a token that does not
/// decode to valid UTF-8 is keyed by its raw vocabulary string.
pub fn compressedTokenMap(arena: Allocator, tokenizer: *const Tokenizer) !TokenMap {
    const n = tokenizer.vocabSize();
    const ids = try arena.alloc(u32, n);
    var keys = std.StringHashMap(u32).init(arena);
    var buf = std.ArrayList(u8).empty;
    var count: usize = 0;
    for (0..n) |id| {
        const text = try tokenizer.decode(arena, &.{@intCast(id)}, false);
        var key: []const u8 = undefined;
        if (!std.unicode.utf8ValidateSlice(text) or std.mem.indexOf(u8, text, "\xef\xbf\xbd") != null) {
            key = tokenizer.id_to_token[id];
        } else {
            buf.clearRetainingCapacity();
            try normalizeText(arena, text, &buf);
            key = if (buf.items.len > 0) try arena.dupe(u8, buf.items) else text;
        }
        const gop = try keys.getOrPut(key);
        if (!gop.found_existing) {
            gop.value_ptr.* = @intCast(count);
            count += 1;
        }
        ids[id] = gop.value_ptr.*;
    }
    return .{ .ids = ids, .count = count };
}

/// The engram normaliser: fold every code point (NFKD, drop nonspacing
/// marks, lowercase), collapse runs of ` \t\r\n` to one space, keep a lone
/// space, otherwise strip Unicode whitespace from both ends.
pub fn normalizeText(gpa: Allocator, text: []const u8, out: *std.ArrayList(u8)) !void {
    var folded = std.ArrayList(u8).empty;
    defer folded.deinit(gpa);
    var it = std.unicode.Utf8View.initUnchecked(text).iterator();
    while (it.nextCodepoint()) |cp| try foldCodepoint(gpa, cp, &folded);
    // Collapse ASCII whitespace runs.
    var collapsed = std.ArrayList(u8).empty;
    defer collapsed.deinit(gpa);
    var i: usize = 0;
    while (i < folded.items.len) {
        const ch = folded.items[i];
        if (ch == ' ' or ch == '\t' or ch == '\r' or ch == '\n') {
            while (i < folded.items.len and (folded.items[i] == ' ' or folded.items[i] == '\t' or folded.items[i] == '\r' or folded.items[i] == '\n')) i += 1;
            try collapsed.append(gpa, ' ');
        } else {
            try collapsed.append(gpa, ch);
            i += 1;
        }
    }
    if (std.mem.eql(u8, collapsed.items, " ")) {
        try out.append(gpa, ' ');
        return;
    }
    // Strip Unicode whitespace from both ends.
    var start: usize = 0;
    var end: usize = collapsed.items.len;
    while (start < end) {
        const len = std.unicode.utf8ByteSequenceLength(collapsed.items[start]) catch 1;
        const cp = std.unicode.utf8Decode(collapsed.items[start..][0..len]) catch break;
        if (!isUnicodeWhitespace(cp)) break;
        start += len;
    }
    while (end > start) {
        var s = end - 1;
        while (s > start and (collapsed.items[s] & 0xC0) == 0x80) s -= 1;
        const cp = std.unicode.utf8Decode(collapsed.items[s..end]) catch break;
        if (!isUnicodeWhitespace(cp)) break;
        end = s;
    }
    try out.appendSlice(gpa, collapsed.items[start..end]);
}

fn isUnicodeWhitespace(cp: u21) bool {
    return switch (cp) {
        0x09...0x0D, 0x20, 0x85, 0xA0, 0x1680, 0x2000...0x200A, 0x2028, 0x2029, 0x202F, 0x205F, 0x3000 => true,
        else => false,
    };
}

/// Appends the fold of `cp` (see `normalizeText`) to `out`.
pub fn foldCodepoint(gpa: Allocator, cp: u21, out: *std.ArrayList(u8)) !void {
    if (cp >= 0xAC00 and cp <= 0xD7A3) {
        // Hangul syllable: algorithmic decomposition into jamo.
        const s = cp - 0xAC00;
        try appendCodepoint(gpa, out, 0x1100 + s / 588);
        try appendCodepoint(gpa, out, 0x1161 + (s % 588) / 28);
        if (s % 28 != 0) try appendCodepoint(gpa, out, 0x11A7 + s % 28);
        return;
    }
    if (findRange(&uni.fold_drop, cp)) return;
    if (findRun(cp)) |delta| {
        try appendCodepoint(gpa, out, @intCast(@as(i32, @intCast(cp)) + delta));
        return;
    }
    var lo: usize = 0;
    var hi: usize = uni.fold_multi.len;
    while (lo < hi) {
        const mid = (lo + hi) / 2;
        const e = uni.fold_multi[mid];
        if (e.cp == cp) {
            try out.appendSlice(gpa, e.out);
            return;
        }
        if (e.cp < cp) lo = mid + 1 else hi = mid;
    }
    try appendCodepoint(gpa, out, cp);
}

fn appendCodepoint(gpa: Allocator, out: *std.ArrayList(u8), cp: u21) !void {
    var buf: [4]u8 = undefined;
    const len = std.unicode.utf8Encode(cp, &buf) catch return;
    try out.appendSlice(gpa, buf[0..len]);
}

fn findRange(ranges: []const uni.Range, cp: u21) bool {
    var lo: usize = 0;
    var hi: usize = ranges.len;
    while (lo < hi) {
        const mid = (lo + hi) / 2;
        const r = ranges[mid];
        if (cp < r.lo) hi = mid else if (cp > r.hi) lo = mid + 1 else return true;
    }
    return false;
}

fn findRun(cp: u21) ?i32 {
    var lo: usize = 0;
    var hi: usize = uni.fold_runs.len;
    while (lo < hi) {
        const mid = (lo + hi) / 2;
        const r = uni.fold_runs[mid];
        if (cp < r.lo) hi = mid else if (cp > r.hi) lo = mid + 1 else return r.delta;
    }
    return null;
}

// ---------------------------------------------------------------------------
// numpy's default generator (SeedSequence + PCG64), for the hash multipliers
// ---------------------------------------------------------------------------

pub const NumpyRng = struct {
    state: u128,
    inc: u128,

    const mult: u128 = (@as(u128, 2549297995355413924) << 64) | 4865540595714422341;

    /// `np.random.default_rng(seed)` for a non-negative integer seed.
    pub fn init(seed: u64) NumpyRng {
        // SeedSequence: the entropy as 32-bit words, mixed into a pool of 4.
        var words_buf: [2]u32 = .{ @truncate(seed), @truncate(seed >> 32) };
        const words: []const u32 = if (seed >> 32 == 0) words_buf[0..1] else words_buf[0..2];
        var hash_const: u32 = 0x43b0d7e5;
        var pool: [4]u32 = undefined;
        for (0..4) |i| pool[i] = hashmix(if (i < words.len) words[i] else 0, &hash_const);
        for (0..4) |src| {
            for (0..4) |dst| {
                if (src != dst) pool[dst] = mix(pool[dst], hashmix(pool[src], &hash_const));
            }
        }
        var i_src: usize = 4;
        while (i_src < words.len) : (i_src += 1) {
            for (0..4) |dst| pool[dst] = mix(pool[dst], hashmix(words[i_src], &hash_const));
        }
        // generate_state(4, uint64): 8 words, little-endian pairs.
        var out_const: u32 = 0x8b51f9dd;
        var state_words: [8]u32 = undefined;
        for (0..8) |i| {
            var v = pool[i % 4];
            v ^= out_const;
            out_const *%= 0x58f38ded;
            v *%= out_const;
            v ^= v >> 16;
            state_words[i] = v;
        }
        const w64 = [4]u64{
            @as(u64, state_words[0]) | (@as(u64, state_words[1]) << 32),
            @as(u64, state_words[2]) | (@as(u64, state_words[3]) << 32),
            @as(u64, state_words[4]) | (@as(u64, state_words[5]) << 32),
            @as(u64, state_words[6]) | (@as(u64, state_words[7]) << 32),
        };
        const initstate: u128 = (@as(u128, w64[0]) << 64) | w64[1];
        const initseq: u128 = (@as(u128, w64[2]) << 64) | w64[3];
        var self = NumpyRng{ .state = 0, .inc = (initseq << 1) | 1 };
        self.step();
        self.state +%= initstate;
        self.step();
        return self;
    }

    fn hashmix(value_in: u32, hash_const: *u32) u32 {
        var value = value_in ^ hash_const.*;
        hash_const.* *%= 0x931e8875;
        value *%= hash_const.*;
        value ^= value >> 16;
        return value;
    }

    fn mix(x: u32, y: u32) u32 {
        var result = (0xca01f9dd *% x) -% (0x4973f715 *% y);
        result ^= result >> 16;
        return result;
    }

    fn step(self: *NumpyRng) void {
        self.state = self.state *% mult +% self.inc;
    }

    /// `random_raw` / `next_uint64` (PCG XSL RR 128/64).
    pub fn next(self: *NumpyRng) u64 {
        self.step();
        const hi: u64 = @truncate(self.state >> 64);
        const lo: u64 = @truncate(self.state);
        const rot: u6 = @truncate(self.state >> 122);
        return std.math.rotr(u64, hi ^ lo, rot);
    }

    /// `integers(0, high, dtype=int64)` for `high > 2^32 + 1` (Lemire's method, as numpy).
    pub fn integer(self: *NumpyRng, high: u64) u64 {
        const rng = high - 1;
        std.debug.assert(rng > 0xFFFFFFFF and rng != std.math.maxInt(u64));
        const rng_excl = rng + 1;
        var m: u128 = @as(u128, self.next()) * rng_excl;
        var leftover: u64 = @truncate(m);
        if (leftover < rng_excl) {
            const threshold = (std.math.maxInt(u64) - rng) % rng_excl;
            while (leftover < threshold) {
                m = @as(u128, self.next()) * rng_excl;
                leftover = @truncate(m);
            }
        }
        return @truncate(m >> 64);
    }
};

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

test "numpy default_rng matches the reference stream and bounded integers" {
    var rng = NumpyRng.init(10007);
    try std.testing.expectEqual(@as(u64, 15187255322829266483), rng.next());
    try std.testing.expectEqual(@as(u64, 959186003677246517), rng.next());
    try std.testing.expectEqual(@as(u64, 7126631698937927467), rng.next());
    // compute_hash_multipliers for compressed_vocab_size 99092, layer 14.
    const bound: u64 = (@as(u64, std.math.maxInt(i64)) / 99092) / 2;
    try std.testing.expectEqual(@as(u64, 46539438283891), bound);
    var r14 = NumpyRng.init(10007 * 14);
    const want = [_]u64{ 67716810739261, 51510806800915, 30921347202721, 82619226485591 };
    for (want) |w| try std.testing.expectEqual(w, r14.integer(bound) * 2 + 1);
    var r1 = NumpyRng.init(10007);
    try std.testing.expectEqual(@as(u64, 76632096046245), r1.integer(bound) * 2 + 1);
    var r0 = NumpyRng.init(0);
    const bound7: u64 = (@as(u64, std.math.maxInt(i64)) / 7) / 2;
    try std.testing.expectEqual(@as(u64, 839276373626933875), r0.integer(bound7) * 2 + 1);
}

test "engram text normalisation" {
    const gpa = std.testing.allocator;
    var out = std.ArrayList(u8).empty;
    defer out.deinit(gpa);
    const cases = [_][2][]const u8{
        .{ " The", "the" },
        .{ "Éte\u{0301}", "ete" },
        .{ "ﬁnal  \tWORD\n", "final word" },
        .{ " ", " " },
        .{ "\u{0130}", "i" },
        .{ "한", "\u{1112}\u{1161}\u{11AB}" },
        .{ "Σ", "σ" },
        .{ "\u{3000}x\u{3000}", "x" },
    };
    for (cases) |cs| {
        out.clearRetainingCapacity();
        try normalizeText(gpa, cs[0], &out);
        try std.testing.expectEqualStrings(cs[1], out.items);
    }
}

test "fp8 and fp4 fake quantisation" {
    // e4m3: 3 mantissa bits, ties to even; subnormals in steps of 2^-9.
    try std.testing.expectEqual(@as(f32, 1.0), roundE4m3(1.0));
    try std.testing.expectEqual(@as(f32, 1.0), roundE4m3(1.0625)); // tie -> even (1.0)
    try std.testing.expectEqual(@as(f32, 1.25), roundE4m3(1.1875)); // tie -> even (1.25 = 1.010b)
    try std.testing.expectEqual(@as(f32, 448.0), roundE4m3(450.0));
    try std.testing.expectEqual(@as(f32, 0.001953125), roundE4m3(0.002));
    try std.testing.expectEqual(@as(f32, -0.875), roundE4m3(-0.9));
    var v = [_]f32{ 0.1, -0.2, 0.3, 0.4 };
    fakeQuantFp8(&v, 4);
    // scale = 2^ceil(log2(0.4 / 448)) = 2^-10; 0.1 / 2^-10 = 102.4 -> 104 (e4m3 step 8 there).
    try std.testing.expectApproxEqAbs(@as(f32, 104.0 / 1024.0), v[0], 1e-7);
    try std.testing.expectApproxEqAbs(@as(f32, -208.0 / 1024.0), v[1], 1e-7);
    try std.testing.expectEqual(@as(u4, 0), e2m1Code(0.25));
    try std.testing.expectEqual(@as(u4, 1), e2m1Code(0.26));
    try std.testing.expectEqual(@as(u4, 2), e2m1Code(0.75));
    try std.testing.expectEqual(@as(u4, 2), e2m1Code(1.25));
    try std.testing.expectEqual(@as(u4, 6), e2m1Code(5.0)); // tie at an even code boundary rounds down
    try std.testing.expectEqual(@as(u4, 7), e2m1Code(5.01));
    try std.testing.expectEqual(@as(u4, 6), e2m1Code(4.99));
    try std.testing.expectEqual(@as(u4, 11), e2m1Code(-1.6));
    var w = [_]f32{ 3.0, -1.0, 0.2, 0.7 };
    fakeQuantFp4(&w, 4, true);
    // scale = e4m3(3 / 6) = 0.5: 3 -> 6*0.5, -1 -> -2*0.5, 0.2 -> 0.5*0.5 (0.4 >= 0.25), 0.7 -> 1.4 -> 1.5*0.5.
    try std.testing.expectEqual(@as(f32, 3.0), w[0]);
    try std.testing.expectEqual(@as(f32, -1.0), w[1]);
    try std.testing.expectEqual(@as(f32, 0.25), w[2]);
    try std.testing.expectEqual(@as(f32, 0.75), w[3]);
}

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
