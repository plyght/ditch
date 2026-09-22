//! Transformer model loading and inference for decoder-only Hugging Face
//! checkpoints. The family-specific knowledge (tensor names, norm and
//! residual layout, attention/MLP layouts, positional encoding, MoE routing)
//! lives in the architecture registry (`arch.zig`); this module is the
//! generic loader and forward pass driven by a `Config`.
//!
//! Weights are accessed through a `stream.WeightStore`: memory-mapped by
//! default, or streamed layer by layer from disk under a memory budget
//! (`LoadOptions`), in which case `forward` keeps one layer (plus a prefetched
//! one) resident at a time. A streamed mixture-of-experts model additionally
//! gets an expert cache ("warp mode", expert_cache.zig): layers acquire only
//! their trunk and the routed experts are fetched on demand. Weights may
//! also come from a remote safetensors source (remote.zig) instead of a
//! local directory.

const std = @import("std");
const Io = std.Io;
const tensor = @import("tensor.zig");
const safetensors = @import("safetensors.zig");
const stream = @import("stream.zig");
const budget_mod = @import("budget.zig");
const gguf_model = @import("gguf_model.zig");
const Tokenizer = @import("tokenizer.zig").Tokenizer;
const moe = @import("moe.zig");
const abliterate = @import("abliterate.zig");
const search = @import("search.zig");
pub const arch = @import("arch.zig");
const expert_cache = @import("expert_cache.zig");
const remote = @import("remote.zig");

const Allocator = std.mem.Allocator;
const Weight = tensor.Weight;
const Delta = tensor.Delta;
const WeightRef = stream.WeightRef;

pub const Config = arch.Config;
pub const RopeScaling = arch.RopeScaling;
pub const parseConfig = arch.parseConfig;

// ---------------------------------------------------------------------------
// Weights
// ---------------------------------------------------------------------------

/// A normalisation layer's parameters (`w` is empty for the non-parametric kind).
pub const Norm = struct {
    w: []const f32,
    b: ?[]const f32 = null,
};

/// Which matrix of a layer a `LayerRefs` entry describes.
pub const Slot = enum { q, k, v, qkv, o, gate, up, gate_up, down, router, q_a, q_b, kv_a, kv_b, lin_qkvz, lin_qkv, lin_z, lin_b, lin_a, lin_ba, lin_conv, lin_q, lin_k, lin_v, lin_conv_q, lin_conv_k, lin_conv_v, lin_f_a, lin_f_b, lin_g_a, lin_g_b };

/// Where a layer's matrices live on disk. `Model.acquireLayer` turns these
/// into resident `Weight` views for the duration of one layer's compute.
/// Dense layers carry a `gate`/`up`/`gate_up`/`down` set; mixture-of-experts
/// layers carry the `router` instead (the expert matrices are described by
/// `Layer.moe`). A `transposed` entry is stored `[in][out]` on disk (GPT-2
/// Conv1D) and made resident as `[out][in]`.
pub const LayerRefs = struct {
    refs: [max]WeightRef = undefined,
    slots: [max]Slot = undefined,
    transposed: [max]bool = [_]bool{false} ** max,
    n: usize = 0,

    pub const max = 16;

    pub fn add(self: *LayerRefs, slot: Slot, ref: WeightRef, transposed: bool) void {
        std.debug.assert(self.n < max);
        self.refs[self.n] = ref;
        self.slots[self.n] = slot;
        self.transposed[self.n] = transposed;
        self.n += 1;
    }

    pub fn get(self: *const LayerRefs, slot: Slot) ?WeightRef {
        for (self.slots[0..self.n], 0..) |s, i| if (s == slot) return self.refs[i];
        return null;
    }

    pub fn isTransposed(self: *const LayerRefs, slot: Slot) bool {
        for (self.slots[0..self.n], 0..) |s, i| if (s == slot) return self.transposed[i];
        return false;
    }

    /// Every ref, in insertion order.
    pub fn all(self: *const LayerRefs, buf: *[max]WeightRef) []WeightRef {
        @memcpy(buf[0..self.n], self.refs[0..self.n]);
        return buf[0..self.n];
    }

    /// The refs that are read as they are (not transposed), in insertion order.
    pub fn plain(self: *const LayerRefs, buf: *[max]WeightRef) []WeightRef {
        var n: usize = 0;
        for (0..self.n) |i| {
            if (self.transposed[i]) continue;
            buf[n] = self.refs[i];
            n += 1;
        }
        return buf[0..n];
    }

    /// Bytes of the attention / dense-MLP / router matrices (expert matrices excluded).
    pub fn bytes(self: *const LayerRefs) u64 {
        var total: u64 = 0;
        for (self.refs[0..self.n]) |r| total += r.byteLen();
        return total;
    }
};

/// Multi-head latent attention projections (DeepSeek V2/V3).
pub const MlaWeights = struct {
    /// Low-rank query down projection (null when `q_lora_rank` is unset: `q_b` is the full query projection).
    q_a: ?Weight,
    q_a_norm: ?[]const f32,
    q_b: Weight,
    /// `[kv_lora_rank + qk_rope_head_dim][hidden]`.
    kv_a: Weight,
    kv_a_norm: []const f32,
    /// `[heads * (qk_nope_head_dim + v_head_dim)][kv_lora_rank]`.
    kv_b: Weight,
};

/// Linear-attention projections. Gated DeltaNet (Qwen hybrids) carries
/// either the fused `qkvz`/`ba` pair (Qwen3-Next) or the split
/// `qkv`/`z`/`b`/`a` set (Qwen3.5); Kimi Delta Attention carries separate
/// `q`/`k`/`v`, the low-rank forget gate `f_a`/`f_b`, beta `b` and the
/// low-rank output gate `g_a`/`g_b`. The output projection lives in
/// `Layer.o` so that abliteration treats it like any attention output.
pub const LinearWeights = struct {
    qkvz: ?Weight = null,
    qkv: ?Weight = null,
    z: ?Weight = null,
    b: ?Weight = null,
    a: ?Weight = null,
    ba: ?Weight = null,
    q: ?Weight = null,
    k: ?Weight = null,
    v: ?Weight = null,
    f_a: ?Weight = null,
    f_b: ?Weight = null,
    g_a: ?Weight = null,
    g_b: ?Weight = null,
    /// Depthwise causal convolution `[conv_dim][1][kernel]` (read as
    /// `[conv_dim][kernel]`), or one tensor per q/k/v projection (original
    /// Kimi Linear checkpoints).
    conv: ?Weight = null,
    conv_q: ?Weight = null,
    conv_k: ?Weight = null,
    conv_v: ?Weight = null,
    /// Gated DeltaNet: `[v_heads]` each; KDA: `dt_bias[heads * head_dim]`, `a_log[heads]`.
    dt_bias: []const f32,
    a_log: []const f32,
    /// Gated norm weight `[v_dim]`.
    norm: []const f32,
};

pub const Layer = struct {
    input_norm: ?Norm,
    /// Norm on the attention output (Gemma, OLMo 2, GLM-4).
    post_attn_norm: ?Norm,
    /// Norm on the MLP input (sequential layouts).
    pre_ff_norm: ?Norm,
    post_ff_norm: ?Norm,
    /// Separate MLP input norm (parallel residual with two norms).
    mlp_norm: ?Norm,
    q_norm: ?Norm,
    k_norm: ?Norm,
    /// Separate projections, or one fused `qkv`.
    q: ?Weight,
    k: ?Weight,
    v: ?Weight,
    qkv: ?Weight,
    o: Weight,
    q_bias: ?[]const f32,
    k_bias: ?[]const f32,
    v_bias: ?[]const f32,
    qkv_bias: ?[]const f32,
    o_bias: ?[]const f32,
    /// Per-head attention sink logits (gpt-oss).
    sinks: ?[]const f32,
    mla: ?MlaWeights,
    /// Gated DeltaNet weights (null for full-attention layers).
    linear: ?LinearWeights = null,
    /// Dense MLP (null for mixture-of-experts layers).
    ///
    /// In mapped mode the `Weight` fields of a layer view the mapping (or an
    /// arena-owned transposed copy) and are always valid. In streamed mode
    /// they carry only the shape (empty data) and the resident views live in
    /// the `Layer` copy returned by `Model.acquireLayer`.
    gate: ?Weight,
    up: ?Weight,
    gate_up: ?Weight,
    down: ?Weight,
    gate_bias: ?[]const f32,
    up_bias: ?[]const f32,
    down_bias: ?[]const f32,
    /// Routed mixture of experts (null for dense layers).
    moe: ?moe.MoeLayer = null,
    refs: LayerRefs,
    /// Abliteration deltas (null = identity). Expert deltas live in `moe`.
    o_delta: ?Delta = null,
    down_delta: ?Delta = null,

    /// Bytes that must be resident to run this layer (attention, MLP or
    /// router plus every expert, and the transient needed to transpose fused
    /// expert blocks in streamed mode). In warp mode (`warp`) the routed
    /// experts live in the expert cache and only one token's selection counts.
    pub fn residentBytes(self: *const Layer, warp: bool) u64 {
        var total = self.refs.bytes();
        if (self.moe) |*m| total += m.residentBytes(warp);
        return total;
    }

    /// Bytes of the layer's trunk: everything except the routed experts.
    pub fn trunkBytes(self: *const Layer) u64 {
        var total = self.refs.bytes();
        if (self.moe) |*m| total += m.trunkBytes();
        return total;
    }

    fn setSlot(self: *Layer, slot: Slot, w: Weight) void {
        switch (slot) {
            .q => self.q = w,
            .k => self.k = w,
            .v => self.v = w,
            .qkv => self.qkv = w,
            .o => self.o = w,
            .gate => self.gate = w,
            .up => self.up = w,
            .gate_up => self.gate_up = w,
            .down => self.down = w,
            .router => self.moe.?.router = w,
            .q_a => self.mla.?.q_a = w,
            .q_b => self.mla.?.q_b = w,
            .kv_a => self.mla.?.kv_a = w,
            .kv_b => self.mla.?.kv_b = w,
            .lin_qkvz => self.linear.?.qkvz = w,
            .lin_qkv => self.linear.?.qkv = w,
            .lin_z => self.linear.?.z = w,
            .lin_b => self.linear.?.b = w,
            .lin_a => self.linear.?.a = w,
            .lin_ba => self.linear.?.ba = w,
            .lin_conv => self.linear.?.conv = w,
            .lin_q => self.linear.?.q = w,
            .lin_k => self.linear.?.k = w,
            .lin_v => self.linear.?.v = w,
            .lin_conv_q => self.linear.?.conv_q = w,
            .lin_conv_k => self.linear.?.conv_k = w,
            .lin_conv_v => self.linear.?.conv_v = w,
            .lin_f_a => self.linear.?.f_a = w,
            .lin_f_b => self.linear.?.f_b = w,
            .lin_g_a => self.linear.?.g_a = w,
            .lin_g_b => self.linear.?.g_b = w,
        }
    }
};

/// A layer whose matrices are resident. `layer` is a copy of the model's
/// `Layer` with valid `Weight` views; release it when done with the layer.
pub const LayerLease = struct {
    model: *const Model,
    layer: Layer,
    leases: [LayerRefs.max]stream.Lease,
    n_leases: usize,
    experts: ?moe.MoeLease = null,

    pub fn release(self: *LayerLease) void {
        const store: *stream.WeightStore = @constCast(&self.model.store);
        if (self.experts) |*e| e.release(store);
        store.releaseSet(self.leases[0..self.n_leases]);
    }
};

/// Options for `Model.loadWithOptions`. The default preserves the historical
/// behaviour (memory-mapped weights, no budget).
pub const LoadOptions = struct {
    store: stream.Mode = .mapped,
    /// When set, weight buffers, deltas and runtime buffers allocated through
    /// `model.gpa` come from the budget's allocator.
    budget: ?*budget_mod.Budget = null,
    /// Directory for spilled activations / KV caches (default: the budget's, else "scratch").
    scratch_dir: ?[]const u8 = null,
    /// Read the next layer's weights while the current one computes (streamed mode).
    prefetch: bool = true,
    /// Always spill activations and KV caches to scratch (testing aid).
    spill_always: bool = false,
    /// Expert cache capacity in bytes for streamed mixture-of-experts models
    /// ("warp mode"): null = automatic (what remains of the budget after the
    /// trunk and a reserve for workspaces), 0 = no cache (every expert of a
    /// layer is acquired with the layer, as in plain streamed mode).
    expert_cache: ?u64 = null,
    /// Remote safetensors source; `dir_path` then holds only the small files
    /// (config, tokenizer) and the shards are read through the source. Implies
    /// streamed mode.
    remote: ?*remote.Source = null,
    /// GGUF input: ignore the embedded Hugging Face config/tokenizer copies
    /// and rebuild them from the ggml metadata (testing aid).
    gguf_ignore_embedded: bool = false,
};

/// The two abliterable components, named as in heretic. `attn_o_proj` is the
/// family's attention output projection (`o_proj`, `dense`, `wo`, `c_proj`,
/// ...) and `mlp_down_proj` its MLP down projection (`down_proj`, `fc2`,
/// `dense_4h_to_h`, `w2`, ...; per expert on MoE layers).
pub const Component = enum {
    attn_o_proj,
    mlp_down_proj,

    pub fn name(self: Component) []const u8 {
        return switch (self) {
            .attn_o_proj => "attn.o_proj",
            .mlp_down_proj => "mlp.down_proj",
        };
    }

    pub fn fromName(s: []const u8) ?Component {
        if (std.mem.eql(u8, s, "attn.o_proj")) return .attn_o_proj;
        if (std.mem.eql(u8, s, "mlp.down_proj")) return .mlp_down_proj;
        return null;
    }

    pub const all = [_]Component{ .attn_o_proj, .mlp_down_proj };
};

/// How a tensor is modified on export.
pub const ExportEdit = union(enum) {
    /// A 2-D `[out][in]` matrix with one delta: `W' = W + B A`.
    whole: Delta,
    /// A `[in][out]` (Conv1D) matrix: `W' = W + (B A)ᵀ`.
    whole_transposed: Delta,
    /// A fused expert down tensor of this layer; per-expert deltas are merged slice by slice.
    fused_down: usize,
};

/// Expands a name template: `{p}` → prefix, `{i}` → layer index, `{e}` → expert index.
pub fn resolveName(arena: Allocator, template: []const u8, prefix: []const u8, layer: usize, expert: usize) ![]const u8 {
    var out: Io.Writer.Allocating = .init(arena);
    const w = &out.writer;
    var i: usize = 0;
    while (i < template.len) {
        if (template[i] == '{' and i + 2 < template.len and template[i + 2] == '}') {
            switch (template[i + 1]) {
                'p' => try w.writeAll(prefix),
                'i' => try w.print("{d}", .{layer}),
                'e' => try w.print("{d}", .{expert}),
                else => try w.writeAll(template[i .. i + 3]),
            }
            i += 3;
        } else {
            try w.writeByte(template[i]);
            i += 1;
        }
    }
    return out.toOwnedSlice();
}

/// `x.weight` → `x.bias` (a bare parameter name gets `_bias`).
pub fn biasName(arena: Allocator, name: []const u8) ![]const u8 {
    if (std.mem.endsWith(u8, name, ".weight")) return std.fmt.allocPrint(arena, "{s}.bias", .{name[0 .. name.len - ".weight".len]});
    return std.fmt.allocPrint(arena, "{s}_bias", .{name});
}

pub const Model = struct {
    /// Runtime allocator (the budget's allocator when loaded with a budget).
    gpa: Allocator,
    /// Allocator for metadata (tokenizer, config, rope tables, this struct).
    meta_gpa: Allocator,
    io: Io,
    pool: *const tensor.Pool,
    arena: std.heap.ArenaAllocator,
    config: Config,
    tokenizer: *Tokenizer,
    files: []*safetensors.File,
    /// Set when the model was loaded from a GGUF file (see gguf_model.zig);
    /// `files` then holds one synthetic file over it.
    gguf: ?*gguf_model.Source,
    /// Weight access (mapped or streamed) over `files`. The forward pass takes
    /// `*const Model` but acquiring weights mutates the store (buffer pool,
    /// prefetch state); since a `Model` always lives on the heap this is done
    /// through `@constCast` in `acquireLayer` & co.
    store: stream.WeightStore,
    /// Warp mode: the bounded cache of resident routed experts (streamed MoE models).
    expert_cache: ?*expert_cache.ExpertCache = null,
    budget: ?*budget_mod.Budget,
    scratch_dir: []const u8,
    /// Tensor name prefix for the language model (e.g. "model." or "transformer.").
    prefix: []const u8,
    /// Shape-valid always; data valid only in mapped mode (use `embedRow` / `acquireLmHead`).
    embed: Weight,
    lm_head: Weight,
    embed_ref: WeightRef,
    lm_head_ref: WeightRef,
    lm_head_bias: ?[]const f32,
    /// Learned absolute position table (GPT-2, OPT).
    pos_embed_ref: ?WeightRef,
    /// LayerNorm on the embeddings (BLOOM).
    embed_norm: ?Norm,
    largest_layer_bytes: u64,
    /// Largest layer without its routed experts (attention, norms, router, shared expert).
    largest_trunk_layer_bytes: u64,
    /// Largest routed expert (0 for dense models).
    largest_expert_bytes: u64,
    /// Bytes of all routed experts of all layers.
    total_expert_bytes: u64,
    largest_tensor_bytes: u64,
    spill_always: bool,
    final_norm: Norm,
    layers: []Layer,
    eos_ids: []u32,
    pad_id: u32,
    /// Directory the model was loaded from.
    source_dir: []const u8,
    /// Raw JSON texts kept for export.
    config_json: []const u8,
    tokenizer_json: []const u8,
    generation_config_json: ?[]const u8,
    tokenizer_config_json: ?[]const u8,
    chat_template: ?[]const u8,
    dtype: tensor.DType,
    rope_cos: []f32, // [max_pos][rotary_dim/2]
    rope_sin: []f32,
    rope_cos_local: []f32,
    rope_sin_local: []f32,
    rope_len: usize,
    /// ALiBi slope per head (empty unless `config.positional == .alibi`).
    alibi_slopes: []f32,

    pub fn deinit(self: *Model) void {
        self.resetDeltas();
        if (self.expert_cache) |c| {
            c.deinit();
            self.meta_gpa.destroy(c);
            self.expert_cache = null;
        }
        self.store.deinit();
        for (self.files) |f| f.close(self.meta_gpa, self.io);
        if (self.gguf) |g| g.close(self.meta_gpa, self.io);
        self.tokenizer.deinit();
        self.arena.deinit();
        self.meta_gpa.destroy(self);
    }

    pub fn streamed(self: *const Model) bool {
        return self.store.mode == .streamed;
    }

    /// True when routed experts go through the expert cache.
    pub fn warp(self: *const Model) bool {
        return self.expert_cache != null;
    }

    /// Bytes the expert cache could give back right now (unpinned entries).
    pub fn expertCacheEvictable(self: *const Model) u64 {
        const c = self.expert_cache orelse return 0;
        return c.evictable();
    }

    /// Whether routed expert `expert` of `layer` was used by a forward pass of
    /// this run. Without an expert cache nothing is tracked and every expert
    /// counts as visited.
    pub fn expertVisited(self: *const Model, layer: usize, expert: usize) bool {
        const c = self.expert_cache orelse return true;
        return c.visited(layer, expert);
    }

    /// True when "visited experts only" can be applied: the cache tracks
    /// uses and at least one forward pass has run.
    pub fn visitedExpertsKnown(self: *const Model) bool {
        const c = self.expert_cache orelse return false;
        return c.anyVisited();
    }

    /// Loads the hotlist written by a previous run on `model_id` (from the
    /// scratch directory) into the expert cache. Returns the number of
    /// experts warmed; 0 when there is no cache or no (readable) hotlist.
    pub fn warmExpertCache(self: *Model, gpa: Allocator, model_id: []const u8) !usize {
        const c = self.expert_cache orelse return 0;
        const path = try expert_cache.hotlistPath(gpa, self.scratch_dir, model_id);
        defer gpa.free(path);
        const hot = expert_cache.ExpertCache.readHotlist(gpa, self.io, path) catch |err| switch (err) {
            error.FileNotFound => return 0,
            error.InvalidHotlist => {
                std.log.warn("ignoring malformed hotlist {s}", .{path});
                return 0;
            },
            else => return err,
        };
        defer gpa.free(hot);
        return c.warm(self, hot);
    }

    /// Writes the expert cache's hotlist for `model_id` (no-op without a cache).
    pub fn writeHotlist(self: *Model, gpa: Allocator, model_id: []const u8) !?[]u8 {
        const c = self.expert_cache orelse return null;
        const path = try expert_cache.hotlistPath(gpa, self.scratch_dir, model_id);
        errdefer gpa.free(path);
        try c.writeHotlist(gpa, self.io, path, model_id);
        return path;
    }

    /// Bytes of weights that must be resident at once in streamed mode
    /// (one layer plus a prefetched one, or the LM head, whichever is larger).
    pub fn residentWeightNeed(self: *const Model) u64 {
        if (!self.streamed()) return 0;
        const layers = if (self.store.prefetch_enabled) 2 * self.largest_layer_bytes else self.largest_layer_bytes;
        return @max(layers, self.lm_head_ref.byteLen());
    }

    /// Loads a model from a local directory containing config.json, tokenizer.json and safetensors files.
    pub fn load(gpa: Allocator, io: Io, pool: *const tensor.Pool, dir_path: []const u8) !*Model {
        return loadWithOptions(gpa, io, pool, dir_path, .{});
    }

    /// Loads a model with an explicit weight store mode and optional memory budget.
    /// `gpa` is used for metadata; runtime buffers use `opts.budget` when given.
    pub fn loadWithOptions(gpa: Allocator, io: Io, pool: *const tensor.Pool, dir_path: []const u8, opts: LoadOptions) !*Model {
        const self = try gpa.create(Model);
        errdefer gpa.destroy(self);
        self.* = undefined;
        self.expert_cache = null;
        self.meta_gpa = gpa;
        self.gpa = if (opts.budget) |b| b.allocator() else gpa;
        self.budget = opts.budget;
        self.spill_always = opts.spill_always;
        self.io = io;
        self.pool = pool;
        self.arena = std.heap.ArenaAllocator.init(gpa);
        errdefer self.arena.deinit();
        const arena = self.arena.allocator();

        self.scratch_dir = try arena.dupe(u8, opts.scratch_dir orelse (if (opts.budget) |b| b.scratch_dir else "scratch"));
        // A GGUF file (or a directory holding one) is loaded through gguf_model.zig.
        self.gguf = null;
        const gguf_path = try gguf_model.locate(io, arena, dir_path);
        const dir_name = if (gguf_path) |p| (std.fs.path.dirname(p) orelse ".") else dir_path;
        var dir = try Io.Dir.cwd().openDir(io, dir_name, .{ .iterate = true });
        defer dir.close(io);
        if (dir.access(io, export_incomplete_marker, .{})) |_| {
            std.log.warn("{s} contains {s}: the export did not finish; refusing to load it", .{ dir_path, export_incomplete_marker });
            return error.IncompleteModel;
        } else |_| {}
        if (gguf_path) |p| {
            try gguf_model.attach(self, p, opts.store == .mapped, .{ .ignore_embedded = opts.gguf_ignore_embedded });
        } else try self.loadHfFiles(dir, dir_path, opts);
        errdefer self.tokenizer.deinit();
        errdefer for (self.files) |f| f.close(gpa, io);
        errdefer if (self.gguf) |g| g.close(gpa, io);

        const store_mode: stream.Mode = if (opts.remote != null) .streamed else opts.store;
        self.store = stream.WeightStore.init(self.gpa, io, self.files, store_mode, .{ .budget = opts.budget, .prefetch = opts.prefetch });
        self.store.registerReclaim();
        errdefer self.store.deinit();

        try self.loadWeights();
        // Warp mode: streamed mixture-of-experts models get an expert cache
        // unless it was explicitly disabled (`expert_cache = 0`).
        const use_cache = self.streamed() and self.isMoe() and (opts.expert_cache orelse 1) != 0;
        self.largest_layer_bytes = 0;
        self.largest_trunk_layer_bytes = 0;
        self.largest_expert_bytes = 0;
        self.total_expert_bytes = 0;
        for (self.layers) |*layer| {
            self.largest_layer_bytes = @max(self.largest_layer_bytes, layer.residentBytes(use_cache));
            self.largest_trunk_layer_bytes = @max(self.largest_trunk_layer_bytes, layer.trunkBytes());
            if (layer.moe) |*m| {
                self.largest_expert_bytes = @max(self.largest_expert_bytes, m.maxExpertBytes());
                for (m.experts) |*ex| self.total_expert_bytes += moe.expertBytes(ex);
            }
        }
        if (opts.budget) |b| {
            // Prefetching needs two layers resident; disable it up front when that cannot fit.
            if (b.limited() and b.limitBytes() < 2 * self.largest_layer_bytes + self.lm_head_ref.byteLen()) self.store.prefetch_enabled = false;
        }
        if (use_cache) {
            const cap = opts.expert_cache orelse self.defaultExpertCacheBytes();
            const cache = try gpa.create(expert_cache.ExpertCache);
            errdefer gpa.destroy(cache);
            cache.* = expert_cache.ExpertCache.init(self.gpa, io, &self.store, cap);
            if (opts.budget) |b| cache.registerReclaim(b);
            self.expert_cache = cache;
        }
        errdefer if (self.expert_cache) |cache| {
            cache.deinit();
            gpa.destroy(cache);
        };
        try self.buildRope();
        return self;
    }

    /// Loads a Gated DeltaNet linear-attention layer: its projections (the
    /// fused qkvz/ba pair or the split qkv/z/b/a set), convolution, time-step
    /// vectors and gated norm, plus the output projection into `layer.o`.
    fn loadLinear(self: *Model, layer: *Layer, arena: Allocator, li: usize, lp: []const u8) !void {
        const c = &self.config;
        const names = &c.arch.names;
        const hidden = c.hidden_size;
        const kd = c.linear_k_heads * c.linear_k_dim;
        const vd = c.linear_v_heads * c.linear_v_dim;
        const vh = c.linear_v_heads;
        var lin = LinearWeights{
            .dt_bias = &.{},
            .a_log = &.{},
            .norm = &.{},
        };
        if (names.lin_qkvz) |t| {
            const n = try cat(arena, lp, t);
            lin.qkvz = try self.loadMat(n);
            layer.refs.add(.lin_qkvz, try self.ref(n), false);
            if (lin.qkvz.?.rows != 2 * kd + 2 * vd or lin.qkvz.?.cols != hidden) {
                std.log.err("layer {d}: fused linear projection is [{d}][{d}], expected [{d}][{d}]", .{ li, lin.qkvz.?.rows, lin.qkvz.?.cols, 2 * kd + 2 * vd, hidden });
                return error.InvalidConfig;
            }
        }
        if (names.lin_ba) |t| {
            const n = try cat(arena, lp, t);
            lin.ba = try self.loadMat(n);
            layer.refs.add(.lin_ba, try self.ref(n), false);
            if (lin.ba.?.rows != 2 * vh or lin.ba.?.cols != hidden) {
                std.log.err("layer {d}: fused linear b/a projection has wrong shape", .{li});
                return error.InvalidConfig;
            }
        }
        if (names.lin_qkv) |t| {
            const n = try cat(arena, lp, t);
            lin.qkv = try self.loadMat(n);
            layer.refs.add(.lin_qkv, try self.ref(n), false);
            if (lin.qkv.?.rows != 2 * kd + vd or lin.qkv.?.cols != hidden) {
                std.log.err("layer {d}: linear qkv projection has wrong shape", .{li});
                return error.InvalidConfig;
            }
        }
        const split_names = .{ names.lin_z, names.lin_b, names.lin_a };
        const split_slots = .{ Slot.lin_z, Slot.lin_b, Slot.lin_a };
        const split_rows = .{ vd, vh, vh };
        inline for (split_names, split_slots, split_rows) |t, slot, want| {
            if (t) |template| {
                const n = try cat(arena, lp, template);
                const w = try self.loadMat(n);
                layer.refs.add(slot, try self.ref(n), false);
                if (w.rows != want or w.cols != hidden) {
                    std.log.err("layer {d}: linear projection {s} has wrong shape", .{ li, n });
                    return error.InvalidConfig;
                }
                switch (slot) {
                    .lin_z => lin.z = w,
                    .lin_b => lin.b = w,
                    .lin_a => lin.a = w,
                    else => unreachable,
                }
            }
        }
        if (lin.qkvz == null and lin.qkv == null) {
            std.log.err("layer {d}: no linear-attention projections found", .{li});
            return error.MissingWeights;
        }
        const conv_name = try cat(arena, lp, names.lin_conv orelse return error.InvalidConfig);
        lin.conv = try self.loadMat(conv_name);
        layer.refs.add(.lin_conv, try self.ref(conv_name), false);
        if (lin.conv.?.rows != 2 * kd + vd or lin.conv.?.cols != c.linear_conv_kernel) {
            std.log.err("layer {d}: linear convolution has wrong shape", .{li});
            return error.InvalidConfig;
        }
        lin.dt_bias = try self.loadVec(try self.requireFirst(arena, lp, names.lin_dt_bias));
        lin.a_log = try self.loadVec(try self.requireFirst(arena, lp, names.lin_a_log));
        lin.norm = try self.loadVec(try cat(arena, lp, names.lin_norm orelse return error.InvalidConfig));
        if (lin.dt_bias.len != vh or lin.a_log.len != vh or lin.norm.len != c.linear_v_dim) {
            std.log.err("layer {d}: linear time-step vectors have wrong shapes", .{li});
            return error.InvalidConfig;
        }
        const o_name = try cat(arena, lp, names.lin_out orelse return error.InvalidConfig);
        layer.o = try self.loadMat(o_name);
        layer.refs.add(.o, try self.ref(o_name), false);
        if (layer.o.rows != hidden or layer.o.cols != vd) {
            std.log.err("layer {d}: linear output projection is [{d}][{d}], expected [{d}][{d}]", .{ li, layer.o.rows, layer.o.cols, hidden, vd });
            return error.InvalidConfig;
        }
        layer.linear = lin;
    }

    /// The first of `templates` (relative to `lp`) that names a tensor of the
    /// store; an error naming the first alternative when none is present.
    fn requireFirst(self: *Model, arena: Allocator, lp: []const u8, templates: []const []const u8) ![]const u8 {
        for (templates) |t| {
            const n = try cat(arena, lp, t);
            if (self.store.lookup(n) != null) return n;
        }
        if (templates.len == 0) return error.InvalidConfig;
        std.log.err("missing tensor: {s}{s}", .{ lp, templates[0] });
        return error.MissingWeights;
    }

    /// Loads a Kimi Delta Attention layer (`KimiLinearDeltaAttention`): the
    /// q/k/v projections, their short convolution (one fused tensor, or one
    /// per projection in the original checkpoints), the low-rank forget gate
    /// with its time-step bias and decay, the beta projection, the low-rank
    /// output gate, the gated norm, and the output projection into `layer.o`.
    fn loadKda(self: *Model, layer: *Layer, arena: Allocator, li: usize, lp: []const u8) !void {
        const c = &self.config;
        const names = &c.arch.names;
        const hidden = c.hidden_size;
        const heads = c.linear_v_heads;
        const hd = c.linear_v_dim;
        const dim = heads * hd;
        const kc = c.linear_conv_kernel;
        var lin = LinearWeights{
            .dt_bias = &.{},
            .a_log = &.{},
            .norm = &.{},
        };
        const proj_names = .{ names.lin_q, names.lin_k, names.lin_v, names.lin_b, names.lin_g_a, names.lin_g_b };
        const proj_slots = .{ Slot.lin_q, Slot.lin_k, Slot.lin_v, Slot.lin_b, Slot.lin_g_a, Slot.lin_g_b };
        const proj_rows = .{ dim, dim, dim, heads, hd, dim };
        const proj_cols = .{ hidden, hidden, hidden, hidden, hidden, hd };
        inline for (proj_names, proj_slots, proj_rows, proj_cols) |t, slot, rows, cols| {
            const n = try cat(arena, lp, t orelse return error.InvalidConfig);
            const w = try self.loadMat(n);
            layer.refs.add(slot, try self.ref(n), false);
            if (w.rows != rows or w.cols != cols) {
                std.log.err("layer {d}: {s} is [{d}][{d}], expected [{d}][{d}]", .{ li, n, w.rows, w.cols, rows, cols });
                return error.InvalidConfig;
            }
            switch (slot) {
                .lin_q => lin.q = w,
                .lin_k => lin.k = w,
                .lin_v => lin.v = w,
                .lin_b => lin.b = w,
                .lin_g_a => lin.g_a = w,
                .lin_g_b => lin.g_b = w,
                else => unreachable,
            }
        }
        const fa_name = try self.requireFirst(arena, lp, names.lin_f_a);
        lin.f_a = try self.loadMat(fa_name);
        layer.refs.add(.lin_f_a, try self.ref(fa_name), false);
        const fb_name = try self.requireFirst(arena, lp, names.lin_f_b);
        lin.f_b = try self.loadMat(fb_name);
        layer.refs.add(.lin_f_b, try self.ref(fb_name), false);
        if (lin.f_a.?.rows != hd or lin.f_a.?.cols != hidden or lin.f_b.?.rows != dim or lin.f_b.?.cols != hd) {
            std.log.err("layer {d}: forget gate projections do not match linear_head_dim {d} / linear_num_heads {d}", .{ li, hd, heads });
            return error.InvalidConfig;
        }
        const fused_conv: ?[]const u8 = if (names.lin_conv) |t| blk: {
            const n = try cat(arena, lp, t);
            break :blk if (self.store.lookup(n) != null) n else null;
        } else null;
        if (fused_conv) |n| {
            lin.conv = try self.loadMat(n);
            layer.refs.add(.lin_conv, try self.ref(n), false);
            if (lin.conv.?.rows != 3 * dim or lin.conv.?.cols != kc) {
                std.log.err("layer {d}: {s} is [{d}][{d}], expected [{d}][{d}]", .{ li, n, lin.conv.?.rows, lin.conv.?.cols, 3 * dim, kc });
                return error.InvalidConfig;
            }
        } else if (names.lin_conv_split.len == 3) {
            const conv_slots = .{ Slot.lin_conv_q, Slot.lin_conv_k, Slot.lin_conv_v };
            inline for (conv_slots, 0..) |slot, i| {
                const n = try cat(arena, lp, names.lin_conv_split[i]);
                const w = try self.loadMat(n);
                layer.refs.add(slot, try self.ref(n), false);
                if (w.rows != dim or w.cols != kc) {
                    std.log.err("layer {d}: {s} is [{d}][{d}], expected [{d}][{d}]", .{ li, n, w.rows, w.cols, dim, kc });
                    return error.InvalidConfig;
                }
                switch (slot) {
                    .lin_conv_q => lin.conv_q = w,
                    .lin_conv_k => lin.conv_k = w,
                    .lin_conv_v => lin.conv_v = w,
                    else => unreachable,
                }
            }
        } else {
            std.log.err("layer {d}: no linear-attention convolution found", .{li});
            return error.MissingWeights;
        }
        lin.dt_bias = try self.loadVec(try self.requireFirst(arena, lp, names.lin_dt_bias));
        lin.a_log = try self.loadVec(try self.requireFirst(arena, lp, names.lin_a_log));
        lin.norm = try self.loadVec(try cat(arena, lp, names.lin_norm orelse return error.InvalidConfig));
        if (lin.dt_bias.len != dim or lin.a_log.len != heads or lin.norm.len != hd) {
            std.log.err("layer {d}: dt_bias / A_log / o_norm have wrong shapes (expected [{d}], [{d}], [{d}])", .{ li, dim, heads, hd });
            return error.InvalidConfig;
        }
        const o_name = try cat(arena, lp, names.lin_out orelse return error.InvalidConfig);
        layer.o = try self.loadMat(o_name);
        layer.o_bias = self.loadVecOpt(try biasName(arena, o_name));
        layer.refs.add(.o, try self.ref(o_name), false);
        if (layer.o.rows != hidden or layer.o.cols != dim) {
            std.log.err("layer {d}: linear output projection is [{d}][{d}], expected [{d}][{d}]", .{ li, layer.o.rows, layer.o.cols, hidden, dim });
            return error.InvalidConfig;
        }
        layer.linear = lin;
    }

    /// Resolves the tensor names of the family and builds the layer table.
    fn loadWeights(self: *Model) !void {
        const arena = self.arena.allocator();
        const c = &self.config;
        const names = &c.arch.names;

        // Detect prefix.
        var found_prefix = false;
        for (names.prefixes) |p| {
            const key = try resolveName(arena, names.embed, p, 0, 0);
            if (self.store.lookup(key) != null) {
                self.prefix = p;
                found_prefix = true;
                break;
            }
        }
        if (!found_prefix) {
            std.log.err("embedding tensor '{s}' not found under any known prefix", .{names.embed});
            return error.MissingWeights;
        }

        const embed_name = try self.name(names.embed);
        self.embed_ref = self.store.lookup(embed_name).?;
        self.embed = try self.loadMat(embed_name);
        self.dtype = self.embed_ref.dtype;
        if (c.vocab_size == 0 or c.vocab_size > self.embed.rows) c.vocab_size = self.embed.rows;
        self.pos_embed_ref = if (names.pos_embed) |t| self.store.lookup(try self.name(t)) else null;
        if (c.positional == .learned and self.pos_embed_ref == null) {
            std.log.err("missing position embedding tensor", .{});
            return error.MissingWeights;
        }
        self.embed_norm = if (names.embed_norm) |t| try self.loadNormOpt(try self.name(t)) else null;
        self.final_norm = if (c.norm == .none) Norm{ .w = &.{} } else try self.loadNorm(try self.name(names.final_norm));
        self.lm_head_bias = null;
        var lm: ?WeightRef = null;
        for (names.lm_head) |t| {
            const n = try self.name(t);
            if (self.store.lookup(n)) |r| {
                lm = r;
                self.lm_head_bias = self.loadVecOpt(try biasName(arena, n));
                break;
            }
        }
        if (lm == null) {
            // Some multimodal exports keep the head next to the language model
            // (`language_model.lm_head.weight` beside `language_model.model.`).
            const trimmed = if (std.mem.endsWith(u8, self.prefix, "model.")) self.prefix[0 .. self.prefix.len - "model.".len] else self.prefix;
            if (self.store.lookup(try std.fmt.allocPrint(arena, "{s}lm_head.weight", .{trimmed}))) |r| lm = r;
        }
        self.lm_head_ref = lm orelse self.embed_ref;
        self.lm_head = try self.loadMat(self.lm_head_ref.name);

        self.largest_tensor_bytes = 0;
        for (self.files) |f| {
            var it = f.tensors.iterator();
            while (it.next()) |kv| self.largest_tensor_bytes = @max(self.largest_tensor_bytes, kv.value_ptr.byte_len);
        }

        self.layers = try arena.alloc(Layer, c.num_layers);
        for (self.layers, 0..) |*layer, i| {
            const lp = try resolveName(arena, names.layer, self.prefix, i, 0);
            layer.* = .{
                .input_norm = null,
                .post_attn_norm = try self.normSlot(lp, names.post_attn_norm, true),
                .pre_ff_norm = try self.normSlot(lp, names.pre_ff_norm, !c.parallel_residual),
                .post_ff_norm = try self.normSlot(lp, names.post_ff_norm, true),
                .mlp_norm = try self.normSlot(lp, names.mlp_norm, false),
                .q_norm = if (names.q_norm) |t| try self.loadNormOpt(try cat(arena, lp, t)) else null,
                .k_norm = if (names.k_norm) |t| try self.loadNormOpt(try cat(arena, lp, t)) else null,
                .q = null,
                .k = null,
                .v = null,
                .qkv = null,
                .o = undefined,
                .q_bias = null,
                .k_bias = null,
                .v_bias = null,
                .qkv_bias = null,
                .o_bias = null,
                .sinks = if (names.sinks) |t| self.loadVecOpt(try cat(arena, lp, t)) else null,
                .mla = null,
                .gate = null,
                .up = null,
                .gate_up = null,
                .down = null,
                .gate_bias = null,
                .up_bias = null,
                .down_bias = null,
                .refs = .{},
            };
            if (c.norm == .none) {
                if (names.input_norm.len > 0) layer.input_norm = Norm{ .w = &.{} };
            } else {
                for (names.input_norm) |t| {
                    if (try self.loadNormOpt(try cat(arena, lp, t))) |nm| {
                        layer.input_norm = nm;
                        break;
                    }
                }
                if (layer.input_norm == null and names.input_norm.len > 0) {
                    std.log.err("missing tensor: {s}{s}", .{ lp, names.input_norm[0] });
                    return error.MissingWeights;
                }
            }
            if (c.qk_norm == .l2) {
                layer.q_norm = Norm{ .w = &.{} };
                layer.k_norm = Norm{ .w = &.{} };
            } else if (c.qk_norm == .none) {
                layer.q_norm = null;
                layer.k_norm = null;
            }
            if (c.sinks and layer.sinks == null) {
                std.log.err("missing attention sinks in layer {d}", .{i});
                return error.MissingWeights;
            }

            // Attention projections.
            if (c.linear_layers[i]) {
                switch (c.linear_kind) {
                    .gated_deltanet => try self.loadLinear(layer, arena, i, lp),
                    .kda => try self.loadKda(layer, arena, i, lp),
                }
            } else if (c.mla) |m| {
                const kv_a_name = try cat(arena, lp, names.kv_a orelse return error.InvalidConfig);
                const kv_b_name = try cat(arena, lp, names.kv_b orelse return error.InvalidConfig);
                var mla = MlaWeights{
                    .q_a = null,
                    .q_a_norm = null,
                    .q_b = undefined,
                    .kv_a = try self.loadMat(kv_a_name),
                    .kv_a_norm = try self.loadVec(try cat(arena, lp, names.kv_a_norm orelse return error.InvalidConfig)),
                    .kv_b = try self.loadMat(kv_b_name),
                };
                layer.refs.add(.kv_a, try self.ref(kv_a_name), false);
                layer.refs.add(.kv_b, try self.ref(kv_b_name), false);
                if (m.q_lora_rank != null) {
                    const q_a_name = try cat(arena, lp, names.q_a orelse return error.InvalidConfig);
                    const q_b_name = try cat(arena, lp, names.q_b orelse return error.InvalidConfig);
                    mla.q_a = try self.loadMat(q_a_name);
                    mla.q_a_norm = try self.loadVec(try cat(arena, lp, names.q_a_norm orelse return error.InvalidConfig));
                    mla.q_b = try self.loadMat(q_b_name);
                    layer.refs.add(.q_a, try self.ref(q_a_name), false);
                    layer.refs.add(.q_b, try self.ref(q_b_name), false);
                } else {
                    const q_name = try cat(arena, lp, names.q orelse return error.InvalidConfig);
                    mla.q_b = try self.loadMat(q_name);
                    layer.refs.add(.q_b, try self.ref(q_name), false);
                }
                if (mla.kv_a.rows != m.kv_lora_rank + m.qk_rope_head_dim or mla.kv_b.rows != c.num_heads * (m.qk_nope_head_dim + m.v_head_dim)) {
                    std.log.err("layer {d}: MLA projection shapes do not match the config", .{i});
                    return error.InvalidConfig;
                }
                layer.mla = mla;
            } else if (c.qkv_layout != .separate) {
                const qkv_name = try cat(arena, lp, names.qkv orelse return error.InvalidConfig);
                layer.qkv = try self.loadMatT(qkv_name, c.arch.conv1d);
                layer.qkv_bias = self.loadVecOpt(try biasName(arena, qkv_name));
                layer.refs.add(.qkv, try self.ref(qkv_name), c.arch.conv1d);
                const want = c.num_heads * c.head_dim + 2 * c.num_kv_heads * c.head_dim;
                if (layer.qkv.?.rows != want or layer.qkv.?.cols != c.hidden_size) {
                    std.log.err("layer {d}: fused qkv tensor is [{d}][{d}], expected [{d}][{d}]", .{ i, layer.qkv.?.rows, layer.qkv.?.cols, want, c.hidden_size });
                    return error.InvalidConfig;
                }
            } else {
                const q_name = try cat(arena, lp, names.q orelse return error.InvalidConfig);
                const k_name = try cat(arena, lp, names.k orelse return error.InvalidConfig);
                const v_name = try cat(arena, lp, names.v orelse return error.InvalidConfig);
                layer.q = try self.loadMat(q_name);
                layer.k = try self.loadMat(k_name);
                layer.v = try self.loadMat(v_name);
                layer.q_bias = self.loadVecOpt(try biasName(arena, q_name));
                layer.k_bias = self.loadVecOpt(try biasName(arena, k_name));
                layer.v_bias = self.loadVecOpt(try biasName(arena, v_name));
                layer.refs.add(.q, try self.ref(q_name), false);
                layer.refs.add(.k, try self.ref(k_name), false);
                layer.refs.add(.v, try self.ref(v_name), false);
                const qwant = if (c.gated_attention) 2 * c.num_heads * c.head_dim else c.num_heads * c.head_dim;
                if (layer.q.?.rows != qwant or layer.k.?.rows != c.num_kv_heads * c.head_dim) {
                    std.log.err("layer {d}: q/k projection shapes do not match the config (heads {d}, kv heads {d}, head_dim {d})", .{ i, c.num_heads, c.num_kv_heads, c.head_dim });
                    return error.InvalidConfig;
                }
            }
            const o_name = try cat(arena, lp, names.o);
            if (!c.linear_layers[i]) {
                layer.o = try self.loadMatT(o_name, c.arch.conv1d);
                layer.o_bias = self.loadVecOpt(try biasName(arena, o_name));
                layer.refs.add(.o, try self.ref(o_name), c.arch.conv1d);
                if (layer.o.rows != c.hidden_size or layer.o.cols != c.num_heads * c.v_head_dim) {
                    std.log.err("layer {d}: output projection is [{d}][{d}], expected [{d}][{d}]", .{ i, layer.o.rows, layer.o.cols, c.hidden_size, c.num_heads * c.v_head_dim });
                    return error.InvalidConfig;
                }
            }

            // MLP.
            if (c.moe_layers[i]) {
                layer.moe = try moe.loadLayer(self, arena, i, lp);
                layer.refs.add(.router, layer.moe.?.router_ref, false);
            } else {
                const down_name = try cat(arena, lp, names.down);
                switch (c.mlp) {
                    .gated => {
                        const gate_name = try cat(arena, lp, names.gate orelse return error.InvalidConfig);
                        const up_name = try cat(arena, lp, names.up orelse return error.InvalidConfig);
                        layer.gate = try self.loadMat(gate_name);
                        layer.up = try self.loadMat(up_name);
                        layer.gate_bias = self.loadVecOpt(try biasName(arena, gate_name));
                        layer.up_bias = self.loadVecOpt(try biasName(arena, up_name));
                        layer.refs.add(.gate, try self.ref(gate_name), false);
                        layer.refs.add(.up, try self.ref(up_name), false);
                    },
                    .gated_fused => {
                        const gu_name = try cat(arena, lp, names.gate_up orelse return error.InvalidConfig);
                        layer.gate_up = try self.loadMat(gu_name);
                        layer.up_bias = self.loadVecOpt(try biasName(arena, gu_name));
                        layer.refs.add(.gate_up, try self.ref(gu_name), false);
                    },
                    .dense => {
                        const up_name = try cat(arena, lp, names.up orelse return error.InvalidConfig);
                        layer.up = try self.loadMatT(up_name, c.arch.conv1d);
                        layer.up_bias = self.loadVecOpt(try biasName(arena, up_name));
                        layer.refs.add(.up, try self.ref(up_name), c.arch.conv1d);
                    },
                }
                layer.down = try self.loadMatT(down_name, c.arch.conv1d);
                layer.down_bias = self.loadVecOpt(try biasName(arena, down_name));
                layer.refs.add(.down, try self.ref(down_name), c.arch.conv1d);
                const inter = if (layer.gate_up) |g| g.rows / 2 else layer.up.?.rows;
                if (i == 0 and inter != c.intermediate_size) {
                    std.log.warn("intermediate_size {d} in config, {d} in the weights; using the weights", .{ c.intermediate_size, inter });
                }
                if (inter > c.intermediate_size) c.intermediate_size = inter;
                if (layer.down.?.rows != c.hidden_size or layer.down.?.cols != inter) {
                    std.log.err("layer {d}: down projection is [{d}][{d}], expected [{d}][{d}]", .{ i, layer.down.?.rows, layer.down.?.cols, c.hidden_size, inter });
                    return error.InvalidConfig;
                }
            }
        }
        self.alibi_slopes = &.{};
        if (c.positional == .alibi) self.alibi_slopes = try alibiSlopes(arena, c.num_heads);
    }

    /// Resolves a model-level name template with the detected prefix.
    fn name(self: *Model, template: []const u8) ![]const u8 {
        return resolveName(self.arena.allocator(), template, self.prefix, 0, 0);
    }

    /// Automatic expert cache capacity: what the budget leaves after the
    /// resident trunk (two layers when prefetching, or the LM head) and a
    /// quarter of the limit reserved for workspaces, KV caches, deltas and
    /// kernel scratch; at least the experts one token selects and at most
    /// every expert of the model. Without a limit, a quarter of the
    /// machine's memory (or 2GB when unknown).
    pub fn defaultExpertCacheBytes(self: *const Model) u64 {
        var top_k: u64 = 1;
        for (self.layers) |l| if (l.moe) |m| {
            top_k = @max(top_k, m.top_k);
        };
        const one_set = top_k * self.largest_expert_bytes;
        var cap: u64 = 0;
        var limited = false;
        if (self.budget) |b| {
            if (b.limited()) {
                limited = true;
                const limit = b.limitBytes();
                const trunk = @max(if (self.store.prefetch_enabled) 2 * self.largest_trunk_layer_bytes else self.largest_trunk_layer_bytes, self.lm_head_ref.byteLen());
                const reserve = trunk + limit / 4;
                cap = if (limit > reserve) limit - reserve else 0;
            }
        }
        if (!limited) {
            const total = std.process.totalSystemMemory() catch (8 << 30);
            cap = total / 4;
        }
        return @min(@max(cap, one_set), @max(self.total_expert_bytes, one_set));
    }

    /// Bytes that must be resident to compute one layer, without the routed experts.
    pub fn trunkResidentNeed(self: *const Model) u64 {
        if (!self.streamed()) return 0;
        const layers = if (self.store.prefetch_enabled) 2 * self.largest_trunk_layer_bytes else self.largest_trunk_layer_bytes;
        return @max(layers, self.lm_head_ref.byteLen());
    }

    /// Locates a tensor by name in the store.
    pub fn ref(self: *const Model, name_: []const u8) !WeightRef {
        return self.store.lookup(name_) orelse {
            std.log.err("missing tensor: {s}", .{name_});
            return error.MissingWeights;
        };
    }

    /// Makes layer `li`'s matrices resident (view in mapped mode, budgeted
    /// buffers in streamed mode) and starts prefetching layer `li + 1`.
    ///
    /// For a mixture-of-experts layer every expert matrix is acquired as well
    /// (one lease per matrix, sliced out of the stacked tensor for the fused
    /// layouts), so a layer is read from disk once per forward call; the
    /// experts are released together with the layer.
    pub fn acquireLayer(self: *const Model, li: usize) !LayerLease {
        const store: *stream.WeightStore = @constCast(&self.store);
        var lease = LayerLease{ .model = self, .layer = self.layers[li], .leases = undefined, .n_leases = 0 };
        const lr = &self.layers[li].refs;
        var buf: [LayerRefs.max]WeightRef = undefined;
        const plain = lr.plain(&buf);
        try store.acquireSet(li, plain, lease.leases[0..plain.len]);
        lease.n_leases = plain.len;
        errdefer store.releaseSet(lease.leases[0..lease.n_leases]);
        const l = &lease.layer;
        var pi: usize = 0;
        for (0..lr.n) |i| {
            if (lr.transposed[i]) {
                // Mapped mode keeps an arena-owned transposed copy in the layer already.
                if (!self.streamed()) continue;
                var out: [1]stream.Lease = undefined;
                try store.acquireTransposed(lr.refs[i], &.{.{ 0, lr.refs[i].cols }}, &out);
                lease.leases[lease.n_leases] = out[0];
                lease.n_leases += 1;
                l.setSlot(lr.slots[i], out[0].weight);
            } else {
                l.setSlot(lr.slots[i], lease.leases[pi].weight);
                pi += 1;
            }
        }
        if (l.moe) |*m| {
            lease.experts = try moe.acquireLayer(self, m);
            l.moe = lease.experts.?.layer;
        }
        if (li + 1 < self.layers.len) {
            var next_buf: [LayerRefs.max]WeightRef = undefined;
            store.prefetch(li + 1, self.layers[li + 1].refs.plain(&next_buf));
        }
        return lease;
    }

    /// Makes one abliterable matrix resident; release with `self.store.release`.
    /// For `.mlp_down_proj` on a mixture-of-experts layer use `acquireExpertDown`.
    pub fn acquireComponent(self: *const Model, layer: usize, comp: Component) !stream.Lease {
        const store: *stream.WeightStore = @constCast(&self.store);
        const slot: Slot = switch (comp) {
            .attn_o_proj => .o,
            .mlp_down_proj => .down,
        };
        const lr = &self.layers[layer].refs;
        const r = lr.get(slot) orelse return error.NotDenseLayer;
        if (lr.isTransposed(slot)) {
            if (!self.streamed()) return .{ .weight = self.componentWeight(layer, comp) };
            var out: [1]stream.Lease = undefined;
            try store.acquireTransposed(r, &.{.{ 0, r.cols }}, &out);
            return out[0];
        }
        return store.acquire(r);
    }

    /// Makes the down projection of expert `expert` of an MoE layer resident
    /// (the shared expert, if any, is index `experts.len`); release with `DownLease.release`.
    pub fn acquireExpertDown(self: *const Model, layer: usize, expert: usize) !moe.DownLease {
        return self.layers[layer].moe.?.acquireDown(self, expert);
    }

    pub fn acquireLmHead(self: *const Model) !stream.Lease {
        const store: *stream.WeightStore = @constCast(&self.store);
        return store.acquire(self.lm_head_ref);
    }

    /// Reads embedding row `t` as f32 (a positional read in streamed mode).
    pub fn embedRow(self: *const Model, t: u32, out: []f32) !void {
        const store: *stream.WeightStore = @constCast(&self.store);
        try store.readRow(self.embed_ref, @min(t, self.embed_ref.rows - 1), out);
    }

    /// Reads the learned position embedding for `pos` (with the family's offset).
    fn posEmbedRow(self: *const Model, pos: usize, out: []f32) !void {
        const r = self.pos_embed_ref orelse return;
        const store: *stream.WeightStore = @constCast(&self.store);
        try store.readRow(r, @min(pos + self.config.position_offset, r.rows - 1), out);
    }

    /// Reads config.json, tokenizer files and opens the safetensors files of a
    /// Hugging Face model directory.
    fn loadHfFiles(self: *Model, dir: Io.Dir, dir_path: []const u8, opts: LoadOptions) !void {
        const gpa = self.meta_gpa;
        const io = self.io;
        const arena = self.arena.allocator();
        self.source_dir = try arena.dupe(u8, dir_path);
        self.config_json = try dir.readFileAlloc(io, "config.json", arena, .unlimited);
        self.config = try parseConfig(arena, self.config_json);
        self.generation_config_json = dir.readFileAlloc(io, "generation_config.json", arena, .unlimited) catch null;
        self.tokenizer_config_json = dir.readFileAlloc(io, "tokenizer_config.json", arena, .unlimited) catch null;
        const tok_json = dir.readFileAlloc(io, "tokenizer.json", arena, .unlimited) catch {
            std.log.err("tokenizer.json not found in {s} (only fast tokenizers are supported)", .{dir_path});
            return error.MissingTokenizer;
        };
        self.tokenizer_json = tok_json;
        self.tokenizer = try Tokenizer.parse(gpa, tok_json, self.tokenizer_config_json);
        errdefer self.tokenizer.deinit();
        self.chat_template = null;
        if (self.tokenizer_config_json) |tc| {
            var parsed = try std.json.parseFromSlice(std.json.Value, gpa, tc, .{});
            defer parsed.deinit();
            if (parsed.value == .object) {
                if (parsed.value.object.get("chat_template")) |ct| {
                    switch (ct) {
                        .string => |s| self.chat_template = try arena.dupe(u8, s),
                        .array => |a| {
                            for (a.items) |item| {
                                if (item == .object) {
                                    if (item.object.get("template")) |t| {
                                        if (t == .string) self.chat_template = try arena.dupe(u8, t.string);
                                    }
                                    if (item.object.get("name")) |n| {
                                        if (n == .string and std.mem.eql(u8, n.string, "default")) break;
                                    }
                                }
                            }
                        },
                        else => {},
                    }
                }
            }
        }
        if (self.chat_template == null) {
            const ct = dir.readFileAlloc(io, "chat_template.jinja", arena, .unlimited) catch null;
            self.chat_template = ct;
        }

        // EOS ids: generation_config eos_token_id (int or list) + tokenizer eos.
        var eos = std.ArrayList(u32).empty;
        if (self.generation_config_json) |gc| {
            var parsed = try std.json.parseFromSlice(std.json.Value, gpa, gc, .{});
            defer parsed.deinit();
            if (parsed.value == .object) {
                if (parsed.value.object.get("eos_token_id")) |e| {
                    switch (e) {
                        .integer => |i| try eos.append(arena, @intCast(i)),
                        .array => |a| for (a.items) |x| {
                            if (x == .integer) try eos.append(arena, @intCast(x.integer));
                        },
                        else => {},
                    }
                }
            }
        }
        if (self.tokenizer.eos_id) |e| {
            var found = false;
            for (eos.items) |x| found = found or x == e;
            if (!found) try eos.append(arena, e);
        }
        self.eos_ids = eos.items;
        self.pad_id = if (eos.items.len > 0) eos.items[0] else 0;

        // Safetensors files.
        var files = std.ArrayList(*safetensors.File).empty;
        errdefer for (files.items) |f| f.close(gpa, io);
        if (opts.remote) |src| {
            // Shards come from the remote source: headers now, tensor bytes on demand.
            for (src.shards) |n| try files.append(arena, try safetensors.File.openRemote(gpa, io, try src.openFile(n)));
        } else {
            var names = std.ArrayList([]const u8).empty;
            var it = dir.iterate();
            while (try it.next(io)) |entry| {
                if (entry.kind != .file) continue;
                if (std.mem.endsWith(u8, entry.name, ".safetensors")) try names.append(arena, try arena.dupe(u8, entry.name));
            }
            std.mem.sort([]const u8, names.items, {}, struct {
                fn lt(_: void, a: []const u8, b: []const u8) bool {
                    return std.mem.lessThan(u8, a, b);
                }
            }.lt);
            if (names.items.len == 0) {
                std.log.err("no .safetensors files found in {s}", .{dir_path});
                return error.MissingWeights;
            }
            for (names.items) |n| try files.append(arena, try safetensors.File.openOptions(gpa, io, dir, n, .{ .map = opts.store == .mapped }));
        }
        self.files = files.items;
    }

    fn cat(arena: Allocator, a: []const u8, b: []const u8) ![]const u8 {
        return std.fmt.allocPrint(arena, "{s}{s}", .{ a, b });
    }

    pub fn find(self: *const Model, name_: []const u8) ?safetensors.TensorInfo {
        for (self.files) |f| {
            if (f.get(name_)) |t| return t;
        }
        return null;
    }

    /// Matrix view of a named tensor: the mapping in mapped mode, the shape
    /// only (empty data) in streamed mode.
    pub fn loadMat(self: *Model, name_: []const u8) !Weight {
        const r = try self.ref(name_);
        return switch (self.store.mode) {
            .mapped => self.find(name_).?.asWeight(),
            .streamed => r.shapeOnly(),
        };
    }

    /// Like `loadMat`; a `transposed` (Conv1D `[in][out]`) tensor is returned
    /// as `[out][in]`: an arena-owned copy in mapped mode, the shape in streamed mode.
    fn loadMatT(self: *Model, name_: []const u8, transposed: bool) !Weight {
        if (!transposed) return self.loadMat(name_);
        const r = try self.ref(name_);
        if (self.streamed()) return .{ .data = &.{}, .dtype = r.dtype, .rows = r.cols, .cols = r.rows };
        const src = self.find(name_).?.asWeight();
        const es = src.dtype.size();
        const out = try self.arena.allocator().alloc(u8, src.data.len);
        var i: usize = 0;
        while (i < src.rows) : (i += 1) {
            var j: usize = 0;
            while (j < src.cols) : (j += 1) {
                @memcpy(out[(j * src.rows + i) * es ..][0..es], src.data[(i * src.cols + j) * es ..][0..es]);
            }
        }
        return .{ .data = out, .dtype = src.dtype, .rows = src.cols, .cols = src.rows };
    }

    fn loadVec(self: *Model, name_: []const u8) ![]f32 {
        return self.loadVecOpt(name_) orelse {
            std.log.err("missing tensor: {s}", .{name_});
            return error.MissingWeights;
        };
    }

    /// Reads a small tensor as an f32 vector (null if absent).
    pub fn loadVecOpt(self: *Model, name_: []const u8) ?[]f32 {
        const r = self.store.lookup(name_) orelse return null;
        return self.store.readVecF32(self.arena.allocator(), r) catch null;
    }

    /// A layer norm at `lp ++ template`: null when the family has no such
    /// norm, a weightless slot for the non-parametric family, otherwise the
    /// loaded parameters (`required` decides whether absence is an error).
    fn normSlot(self: *Model, lp: []const u8, template: ?[]const u8, required: bool) !?Norm {
        const t = template orelse return null;
        if (self.config.norm == .none) return Norm{ .w = &.{} };
        const name_ = try cat(self.arena.allocator(), lp, t);
        return if (required) try self.loadNorm(name_) else try self.loadNormOpt(name_);
    }

    fn loadNormOpt(self: *Model, name_: []const u8) !?Norm {
        const w = self.loadVecOpt(name_) orelse return null;
        return Norm{ .w = w, .b = self.loadVecOpt(try biasName(self.arena.allocator(), name_)) };
    }

    fn loadNorm(self: *Model, name_: []const u8) !Norm {
        return (try self.loadNormOpt(name_)) orelse {
            std.log.err("missing tensor: {s}", .{name_});
            return error.MissingWeights;
        };
    }

    fn buildRope(self: *Model) !void {
        const c = &self.config;
        const arena = self.arena.allocator();
        const half = c.rotary_dim / 2;
        self.rope_len = @min(c.max_position_embeddings, 8192);
        self.rope_cos = try arena.alloc(f32, self.rope_len * half);
        self.rope_sin = try arena.alloc(f32, self.rope_len * half);
        try self.fillRope(self.rope_cos, self.rope_sin, c.rope_theta, c.rope_scaling);
        if (c.arch.norm == .rms_gemma and std.mem.startsWith(u8, c.model_type, "gemma3")) {
            self.rope_cos_local = try arena.alloc(f32, self.rope_len * half);
            self.rope_sin_local = try arena.alloc(f32, self.rope_len * half);
            try self.fillRope(self.rope_cos_local, self.rope_sin_local, c.rope_local_theta, .none);
        } else {
            self.rope_cos_local = self.rope_cos;
            self.rope_sin_local = self.rope_sin;
        }
    }

    fn fillRope(self: *Model, cos: []f32, sin: []f32, theta: f32, scaling: RopeScaling) !void {
        const c = &self.config;
        const dim = c.rotary_dim;
        const half = dim / 2;
        if (half == 0) return;
        const inv_freq = try self.gpa.alloc(f64, half);
        defer self.gpa.free(inv_freq);
        for (inv_freq, 0..) |*f, i| {
            const exponent: f64 = @as(f64, @floatFromInt(2 * i)) / @as(f64, @floatFromInt(dim));
            f.* = 1.0 / std.math.pow(f64, theta, exponent);
        }
        var attention_factor: f64 = 1.0;
        switch (scaling) {
            .none => {},
            .linear => |factor| for (inv_freq) |*f| {
                f.* /= factor;
            },
            .factors => |factors| for (inv_freq, 0..) |*f, i| {
                if (i < factors.len) f.* /= factors[i];
            },
            .llama3 => |s| {
                const low_wavelen = s.original_max_position / s.low_freq_factor;
                const high_wavelen = s.original_max_position / s.high_freq_factor;
                for (inv_freq) |*f| {
                    const wavelen = 2.0 * std.math.pi / f.*;
                    if (wavelen < high_wavelen) {
                        // keep
                    } else if (wavelen > low_wavelen) {
                        f.* /= s.factor;
                    } else {
                        const smooth = (s.original_max_position / wavelen - s.low_freq_factor) / (s.high_freq_factor - s.low_freq_factor);
                        f.* = (1.0 - smooth) * f.* / s.factor + smooth * f.*;
                    }
                }
            },
            .yarn => |y| {
                // Blend of interpolated (low frequency) and extrapolated (high
                // frequency) frequencies with a linear ramp between the
                // correction dims, as in Hugging Face `_compute_yarn_parameters`.
                const base: f64 = theta;
                const dimf: f64 = @floatFromInt(dim);
                const corr = struct {
                    fn dimFor(rot: f64, d: f64, b: f64, max_pos: f64) f64 {
                        return (d * @log(max_pos / (rot * 2.0 * std.math.pi))) / (2.0 * @log(b));
                    }
                };
                var low = corr.dimFor(y.beta_fast, dimf, base, y.original_max_position);
                var high = corr.dimFor(y.beta_slow, dimf, base, y.original_max_position);
                if (y.truncate) {
                    low = @floor(low);
                    high = @ceil(high);
                }
                low = @max(low, 0);
                high = @min(high, dimf - 1);
                if (low == high) high += 0.001;
                for (inv_freq, 0..) |*f, i| {
                    const ramp = std.math.clamp((@as(f64, @floatFromInt(i)) - low) / (high - low), 0.0, 1.0);
                    const extrapolation_factor = 1.0 - ramp;
                    const interp = f.* / y.factor;
                    f.* = interp * (1.0 - extrapolation_factor) + f.* * extrapolation_factor;
                }
                attention_factor = y.attention_factor;
            },
            .longrope => |l| {
                for (inv_freq, 0..) |*f, i| f.* /= l.factors[i];
                attention_factor = l.attention_factor;
            },
        }
        var pos: usize = 0;
        while (pos < self.rope_len) : (pos += 1) {
            for (inv_freq, 0..) |f, i| {
                const angle = @as(f64, @floatFromInt(pos)) * f;
                cos[pos * half + i] = @floatCast(@cos(angle) * attention_factor);
                sin[pos * half + i] = @floatCast(@sin(angle) * attention_factor);
            }
        }
    }

    pub fn isEos(self: *const Model, id: u32) bool {
        for (self.eos_ids) |e| if (e == id) return true;
        return false;
    }

    /// Layer index encoded in a tensor name of this model, if any.
    pub fn layerIndex(self: *const Model, name_: []const u8) ?usize {
        return layerIndexOfTemplate(self.config.arch.names.layer, self.prefix, name_);
    }

    /// The tensor-name suffix after the layer prefix (`self_attn.o_proj.weight`), if `name_` is a layer tensor.
    pub fn layerSuffix(self: *const Model, name_: []const u8) ?struct { layer: usize, suffix: []const u8 } {
        const li = self.layerIndex(name_) orelse return null;
        var buf: [96]u8 = undefined;
        const lp = layerPrefixBuf(&buf, self.config.arch.names.layer, self.prefix, li) orelse return null;
        if (!std.mem.startsWith(u8, name_, lp)) return null;
        return .{ .layer = li, .suffix = name_[lp.len..] };
    }

    /// How the export must modify tensor `name_` (null: copy as is).
    pub fn exportEdit(self: *const Model, name_: []const u8) ?ExportEdit {
        const ls = self.layerSuffix(name_) orelse return null;
        if (ls.layer >= self.layers.len) return null;
        const layer = &self.layers[ls.layer];
        const names = &self.config.arch.names;
        if (std.mem.eql(u8, ls.suffix, names.o)) return wholeEdit(layer.o_delta, layer.refs.isTransposed(.o));
        if (names.lin_out) |lo| if (std.mem.eql(u8, ls.suffix, lo)) return wholeEdit(layer.o_delta, false);
        if (layer.moe) |*m| {
            const target = m.exportTarget(ls.suffix) orelse return null;
            return switch (target) {
                .expert => |e| wholeEdit(m.getDownDelta(e), false),
                .fused_down => if (m.anyExpertDelta()) .{ .fused_down = ls.layer } else null,
            };
        }
        if (std.mem.eql(u8, ls.suffix, names.down)) return wholeEdit(layer.down_delta, layer.refs.isTransposed(.down));
        return null;
    }

    fn wholeEdit(delta: ?Delta, transposed: bool) ?ExportEdit {
        const d = delta orelse return null;
        return if (transposed) .{ .whole_transposed = d } else .{ .whole = d };
    }

    /// Removes all abliteration deltas.
    pub fn resetDeltas(self: *Model) void {
        for (self.layers) |*l| {
            if (l.o_delta) |d| {
                self.gpa.free(d.a);
                self.gpa.free(d.b);
                l.o_delta = null;
            }
            if (l.down_delta) |d| {
                self.gpa.free(d.a);
                self.gpa.free(d.b);
                l.down_delta = null;
            }
            if (l.moe) |*m| m.resetDeltas(self.gpa);
        }
    }

    /// The matrix view for a dense abliterable component. Valid data only in
    /// mapped mode; in streamed mode use `acquireComponent`. For
    /// `.mlp_down_proj` on an MoE layer use `expertDownWeight`.
    pub fn componentWeight(self: *const Model, layer: usize, comp: Component) Weight {
        return switch (comp) {
            .attn_o_proj => self.layers[layer].o,
            .mlp_down_proj => self.layers[layer].down.?,
        };
    }

    /// Down projection of routed expert `expert` of an MoE layer (data valid in
    /// mapped mode only); the shared expert (if any) is addressed by index `experts.len`.
    pub fn expertDownWeight(self: *const Model, layer: usize, expert: usize) Weight {
        return self.layers[layer].moe.?.downWeight(expert);
    }

    pub fn setExpertDelta(self: *Model, layer: usize, expert: usize, delta: Delta) void {
        const slot = self.layers[layer].moe.?.downDelta(expert);
        if (slot.*) |d| {
            self.gpa.free(d.a);
            self.gpa.free(d.b);
        }
        slot.* = delta;
    }

    pub fn getExpertDelta(self: *const Model, layer: usize, expert: usize) ?Delta {
        return self.layers[layer].moe.?.getDownDelta(expert);
    }

    pub fn setDelta(self: *Model, layer: usize, comp: Component, delta: Delta) void {
        const l = &self.layers[layer];
        switch (comp) {
            .attn_o_proj => {
                if (l.o_delta) |d| {
                    self.gpa.free(d.a);
                    self.gpa.free(d.b);
                }
                l.o_delta = delta;
            },
            .mlp_down_proj => {
                if (l.down_delta) |d| {
                    self.gpa.free(d.a);
                    self.gpa.free(d.b);
                }
                l.down_delta = delta;
            },
        }
    }

    /// True if any layer is a routed mixture of experts.
    pub fn isMoe(self: *const Model) bool {
        for (self.layers) |l| if (l.moe != null) return true;
        return false;
    }

    /// Largest number of routed experts in any layer (0 for dense models).
    pub fn numExpertsPerLayer(self: *const Model) usize {
        var n: usize = 0;
        for (self.layers) |l| if (l.moe) |m| {
            n = @max(n, m.experts.len);
        };
        return n;
    }

    pub fn getDelta(self: *const Model, layer: usize, comp: Component) ?Delta {
        return switch (comp) {
            .attn_o_proj => self.layers[layer].o_delta,
            .mlp_down_proj => self.layers[layer].down_delta,
        };
    }
};

/// Marker file written by the exporter before any shard and removed only when
/// the export completed; a directory containing it is refused by `Model.load`.
pub const export_incomplete_marker = ".incomplete";

/// Writes the layer prefix (`<prefix>layers.<n>.`) for `layer` into `buf`.
fn layerPrefixBuf(buf: []u8, template: []const u8, prefix: []const u8, layer: usize) ?[]const u8 {
    var fbs = Io.Writer.fixed(buf);
    var i: usize = 0;
    while (i < template.len) {
        if (template[i] == '{' and i + 2 < template.len and template[i + 2] == '}') {
            switch (template[i + 1]) {
                'p' => fbs.writeAll(prefix) catch return null,
                'i' => fbs.print("{d}", .{layer}) catch return null,
                else => return null,
            }
            i += 3;
        } else {
            fbs.writeByte(template[i]) catch return null;
            i += 1;
        }
    }
    return fbs.buffered();
}

/// Layer index encoded in a tensor name according to a layer template
/// (`{p}layers.{i}.`), if any.
pub fn layerIndexOfTemplate(template: []const u8, prefix: []const u8, name_: []const u8) ?usize {
    const marker = std.mem.indexOf(u8, template, "{i}") orelse return null;
    var buf: [96]u8 = undefined;
    const head = layerPrefixBuf(&buf, template[0..marker], prefix, 0) orelse return null;
    if (!std.mem.startsWith(u8, name_, head)) return null;
    const after = name_[head.len..];
    const tail = template[marker + 3 ..];
    const end = std.mem.indexOf(u8, after, tail) orelse return null;
    if (end == 0) return null;
    return std.fmt.parseInt(usize, after[0..end], 10) catch null;
}

/// Layer index encoded in a tensor name (`<prefix>layers.<n>.…`), if any.
pub fn layerIndexOf(prefix: []const u8, name_: []const u8) ?usize {
    return layerIndexOfTemplate("{p}layers.{i}.", prefix, name_);
}

/// ALiBi slopes for `n` heads (the BLOOM / MPT construction, identical for both).
fn alibiSlopes(arena: Allocator, n: usize) ![]f32 {
    const out = try arena.alloc(f32, n);
    var pow2: usize = 1;
    while (pow2 < n) pow2 *= 2;
    // Slopes for the next power of two, then interleaved like MPT/BLOOM for odd counts.
    const full = try arena.alloc(f32, pow2);
    for (full, 0..) |*s, i| s.* = @floatCast(std.math.pow(f64, 2.0, -8.0 * @as(f64, @floatFromInt(i + 1)) / @as(f64, @floatFromInt(pow2))));
    if (pow2 == n) {
        @memcpy(out, full);
    } else {
        var k: usize = 0;
        var i: usize = 1;
        while (i < pow2 and k < n) : (i += 2) {
            out[k] = full[i];
            k += 1;
        }
        i = 0;
        while (i < pow2 and k < n) : (i += 2) {
            out[k] = full[i];
            k += 1;
        }
    }
    return out;
}

test "layer index from templates" {
    try std.testing.expectEqual(@as(?usize, 12), layerIndexOfTemplate("{p}layers.{i}.", "model.", "model.layers.12.self_attn.o_proj.weight"));
    try std.testing.expectEqual(@as(?usize, 3), layerIndexOfTemplate("{p}h.{i}.", "transformer.", "transformer.h.3.attn.c_proj.weight"));
    try std.testing.expectEqual(@as(?usize, null), layerIndexOfTemplate("{p}h.{i}.", "transformer.", "transformer.wte.weight"));
    try std.testing.expectEqual(@as(?usize, 7), layerIndexOfTemplate("{p}encoder.layers.{i}.", "transformer.", "transformer.encoder.layers.7.mlp.dense_4h_to_h.weight"));
}

test "alibi slopes" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const s8 = try alibiSlopes(arena.allocator(), 8);
    try std.testing.expectApproxEqRel(@as(f32, 0.5), s8[0], 1e-6);
    try std.testing.expectApproxEqRel(@as(f32, 1.0 / 256.0), s8[7], 1e-6);
    const s12 = try alibiSlopes(arena.allocator(), 12);
    try std.testing.expectApproxEqRel(@as(f32, 0.5), s12[0], 1e-6);
    try std.testing.expectApproxEqRel(@as(f32, 1.0 / 256.0), s12[7], 1e-6);
    try std.testing.expectApproxEqRel(@as(f32, std.math.pow(f32, 2.0, -0.5)), s12[8], 1e-6);
    try std.testing.expectApproxEqRel(@as(f32, std.math.pow(f32, 2.0, -3.5)), s12[11], 1e-6);
}

// ---------------------------------------------------------------------------
// Inference
// ---------------------------------------------------------------------------

/// Key/value cache `[layer][batch][pos][kv_dim]`. Either fully in RAM or, when
/// the budget does not allow that, in a scratch file with only the current
/// layer's slice resident (`beginLayer` loads it, `endLayer` writes it back).
/// In scratch mode every layer visit costs one read + one write of the
/// layer's `[batch][high_water][kv_dim]` region, on top of the weight reads.
pub const KvCache = struct {
    gpa: Allocator,
    batch: usize,
    max_len: usize,
    kv_dim: usize,
    layers: usize,
    /// RAM mode: [layer][batch][pos][kv_dim]. Scratch mode: [batch][pos][kv_dim] of `current_layer`.
    k: []f32,
    v: []f32,
    scratch: ?stream.ScratchFile = null,
    current_layer: usize = 0,
    /// Highest `pos + 1` ever written (bounds scratch traffic).
    high_water: usize = 0,
    /// Scratch mode: per layer, how many positions have been written to the file.
    layer_written: []usize = &.{},
    /// Recurrent state of linear-attention layers (null for dense models).
    linear: ?LinearCache = null,

    pub fn init(gpa: Allocator, layers: usize, batch: usize, max_len: usize, kv_dim: usize) !KvCache {
        const n = layers * batch * max_len * kv_dim;
        const k = try gpa.alloc(f32, n);
        errdefer gpa.free(k);
        const v = try gpa.alloc(f32, n);
        return .{ .gpa = gpa, .batch = batch, .max_len = max_len, .kv_dim = kv_dim, .layers = layers, .k = k, .v = v };
    }

    /// A cache backed by a scratch file; only one layer is resident at a time.
    pub fn initScratch(gpa: Allocator, io: Io, scratch_dir: []const u8, budget: ?*budget_mod.Budget, layers: usize, batch: usize, max_len: usize, kv_dim: usize) !KvCache {
        const n = batch * max_len * kv_dim;
        const k = try gpa.alloc(f32, n);
        errdefer gpa.free(k);
        const v = try gpa.alloc(f32, n);
        errdefer gpa.free(v);
        @memset(k, 0);
        @memset(v, 0);
        const lw = try gpa.alloc(usize, layers);
        errdefer gpa.free(lw);
        @memset(lw, 0);
        const sf = try stream.ScratchFile.create(io, scratch_dir, budget);
        return .{ .gpa = gpa, .batch = batch, .max_len = max_len, .kv_dim = kv_dim, .layers = layers, .k = k, .v = v, .scratch = sf, .layer_written = lw };
    }

    pub fn bytesFor(layers: usize, batch: usize, max_len: usize, kv_dim: usize) u64 {
        return @as(u64, layers) * batch * max_len * kv_dim * 4 * 2;
    }

    /// Picks RAM or scratch backing depending on the model's budget.
    pub fn initFor(model: *const Model, gpa: Allocator, batch: usize, max_len: usize) !KvCache {
        const c = &model.config;
        const kvd = c.num_kv_heads * c.head_dim;
        var use_scratch = model.spill_always;
        if (model.budget) |b| {
            const need = bytesFor(c.num_layers, batch, max_len, kvd) + LinearCache.bytesFor(c, batch) + model.residentWeightNeed();
            if (b.limited() and b.available() + model.expertCacheEvictable() < need) use_scratch = true;
        }
        var cache: KvCache = undefined;
        if (use_scratch) {
            cache = try initScratch(gpa, model.io, model.scratch_dir, model.budget, c.num_layers, batch, max_len, kvd);
        } else {
            cache = init(gpa, c.num_layers, batch, max_len, kvd) catch |err| blk: {
                if (err != error.OutOfMemory or !model.streamed()) return err;
                break :blk try initScratch(gpa, model.io, model.scratch_dir, model.budget, c.num_layers, batch, max_len, kvd);
            };
        }
        if (c.has_linear) cache.linear = try LinearCache.init(gpa, c, batch);
        return cache;
    }

    pub fn deinit(self: *KvCache) void {
        self.gpa.free(self.k);
        self.gpa.free(self.v);
        if (self.scratch) |*s| s.deinit();
        if (self.layer_written.len > 0) self.gpa.free(self.layer_written);
        if (self.linear) |*l| l.deinit();
    }

    pub fn spilled(self: *const KvCache) bool {
        return self.scratch != null;
    }

    inline fn index(self: *const KvCache, layer: usize, b: usize, pos: usize) usize {
        const l = if (self.scratch != null) 0 else layer;
        return ((l * self.batch + b) * self.max_len + pos) * self.kv_dim;
    }

    fn scratchIndex(self: *const KvCache, layer: usize, b: usize, which: usize) u64 {
        // Layout in the scratch file: [layer][k/v][batch][max_len][kv_dim] floats.
        return ((@as(u64, layer) * 2 + which) * self.batch + b) * self.max_len * self.kv_dim;
    }

    /// Loads `layer`'s slice into RAM (scratch mode only).
    pub fn beginLayer(self: *KvCache, layer: usize) !void {
        const s = &(self.scratch orelse return);
        self.current_layer = layer;
        const n = self.layer_written[layer] * self.kv_dim;
        if (n == 0) return;
        for (0..self.batch) |b| {
            const base = (b * self.max_len) * self.kv_dim;
            try s.readF32(self.scratchIndex(layer, b, 0), self.k[base..][0..n]);
            try s.readF32(self.scratchIndex(layer, b, 1), self.v[base..][0..n]);
        }
    }

    /// Writes `layer`'s slice back (scratch mode only).
    pub fn endLayer(self: *KvCache, layer: usize) !void {
        const s = &(self.scratch orelse return);
        std.debug.assert(layer == self.current_layer);
        const n = self.high_water * self.kv_dim;
        if (n == 0) return;
        for (0..self.batch) |b| {
            const base = (b * self.max_len) * self.kv_dim;
            try s.writeF32(self.scratchIndex(layer, b, 0), self.k[base..][0..n]);
            try s.writeF32(self.scratchIndex(layer, b, 1), self.v[base..][0..n]);
        }
        self.layer_written[layer] = self.high_water;
    }

    pub fn kSlot(self: *KvCache, layer: usize, b: usize, pos: usize) []f32 {
        std.debug.assert(self.scratch == null or layer == self.current_layer);
        return self.k[self.index(layer, b, pos)..][0..self.kv_dim];
    }
    pub fn vSlot(self: *KvCache, layer: usize, b: usize, pos: usize) []f32 {
        std.debug.assert(self.scratch == null or layer == self.current_layer);
        return self.v[self.index(layer, b, pos)..][0..self.kv_dim];
    }

    /// Records that position `pos` was written (any layer).
    pub fn noteWrite(self: *KvCache, pos: usize) void {
        if (pos + 1 > self.high_water) self.high_water = pos + 1;
    }
};

/// Recurrent state of Gated DeltaNet linear-attention layers (Qwen hybrids):
/// per (linear layer, batch slot) a `[v_heads][k_dim][v_dim]` f32 state plus
/// the causal-convolution history, with the next expected position. Slots
/// reset when a sequence restarts at position 0, so caches stay reusable
/// across batch operations like the KV cache; anything else discontinuous is
/// an error rather than a silent wrong result.
pub const LinearCache = struct {
    gpa: Allocator,
    batch: usize,
    /// Per model layer: linear-layer index or null.
    lin_index: []?u32,
    state_len: usize,
    conv_len: usize,
    buf: []f32,
    next_pos: []usize,

    pub fn bytesFor(c: *const Config, batch: usize) u64 {
        if (!c.has_linear) return 0;
        const per: u64 = @intCast(c.linear_v_heads * c.linear_k_dim * c.linear_v_dim + linearConvDim(c) * (c.linear_conv_kernel - 1));
        var n_lin: u64 = 0;
        for (c.linear_layers) |is_lin| {
            if (is_lin) n_lin += 1;
        }
        return n_lin * @as(u64, batch) * per * 4 + n_lin * @as(u64, batch) * 8;
    }

    pub fn init(gpa: Allocator, c: *const Config, batch: usize) !LinearCache {
        var n_lin: usize = 0;
        const lin_index = try gpa.alloc(?u32, c.num_layers);
        errdefer gpa.free(lin_index);
        for (c.linear_layers, 0..) |l, li| {
            if (l) {
                lin_index[li] = @intCast(n_lin);
                n_lin += 1;
            } else lin_index[li] = null;
        }
        const state_len: usize = c.linear_v_heads * c.linear_k_dim * c.linear_v_dim;
        const conv_len: usize = linearConvDim(c) * (c.linear_conv_kernel - 1);
        const buf = try gpa.alloc(f32, n_linear_buflen(n_lin, batch, state_len, conv_len));
        errdefer gpa.free(buf);
        @memset(buf, 0);
        const next_pos = try gpa.alloc(usize, n_lin * batch);
        errdefer gpa.free(next_pos);
        @memset(next_pos, 0);
        return .{ .gpa = gpa, .batch = batch, .lin_index = lin_index, .state_len = state_len, .conv_len = conv_len, .buf = buf, .next_pos = next_pos };
    }

    fn n_linear_buflen(n_lin: usize, batch: usize, state_len: usize, conv_len: usize) usize {
        return n_lin * batch * (state_len + conv_len);
    }

    pub fn deinit(self: *LinearCache) void {
        self.gpa.free(self.buf);
        self.gpa.free(self.next_pos);
        self.gpa.free(self.lin_index);
    }

    fn slotLen(self: *const LinearCache) usize {
        return self.state_len + self.conv_len;
    }

    pub fn state(self: *LinearCache, li: usize, b: usize) []f32 {
        const idx = self.lin_index[li].?;
        return self.buf[(idx * self.batch + b) * self.slotLen() ..][0..self.state_len];
    }

    pub fn conv(self: *LinearCache, li: usize, b: usize) []f32 {
        const idx = self.lin_index[li].?;
        return self.buf[(idx * self.batch + b) * self.slotLen() + self.state_len ..][0..self.conv_len];
    }

    pub fn nextPos(self: *LinearCache, li: usize, b: usize) *usize {
        const idx = self.lin_index[li].?;
        return &self.next_pos[idx * self.batch + b];
    }
};

fn linearConvDim(c: *const Config) usize {
    return 2 * c.linear_k_heads * c.linear_k_dim + c.linear_v_heads * c.linear_v_dim;
}

/// Describes one row of a batched forward call.
pub const Row = struct {
    /// Batch slot (index into the KV cache).
    b: usize,
    /// Absolute position of this token within its sequence.
    pos: usize,
};

const AttnCtx = struct {
    model: *const Model,
    cache: *KvCache,
    layer: usize,
    rows: []const Row,
    q: []f32, // [n][heads*head_dim] (already roped)
    out: []f32, // [n][heads*head_dim]; per head the first v_head_dim entries are written
    sliding: bool,
    sinks: ?[]const f32,
    /// Tasks are `(row, head)` pairs; the pool's chunk `c` handles tasks
    /// `c, c + chunks, c + 2 chunks, ...` so that a long causal prefix (many
    /// keys for late rows) is spread over every thread.
    n_tasks: usize,
    chunks: usize,
    per: usize,
    /// One `max_keys` score buffer per chunk.
    scores: []f32,
    max_keys: usize,
};

fn attentionWorker(ctx: *const AttnCtx, start: usize, end: usize) void {
    const model = ctx.model;
    const c = &model.config;
    const hd = c.head_dim;
    const vd = c.v_head_dim;
    const groups = c.num_heads / c.num_kv_heads;
    const slot = start / ctx.per;
    const scores_buf = ctx.scores[slot * ctx.max_keys ..][0..ctx.max_keys];
    const stride = ctx.cache.kv_dim;
    // The vector path needs whole vectors along the head dimension.
    const vectorised = hd % tensor.VL == 0 and vd % tensor.VL == 0;
    var i = start;
    while (i < end) : (i += 1) {
        const task = (i % ctx.per) * ctx.chunks + slot;
        if (task >= ctx.n_tasks) continue;
        const r = task / c.num_heads;
        const h = task % c.num_heads;
        const row = ctx.rows[r];
        const kvh = h / groups;
        const q = ctx.q[r * c.num_heads * hd + h * hd ..][0..hd];
        const out = ctx.out[r * c.num_heads * hd + h * hd ..][0..vd];
        var lo: usize = 0;
        if (ctx.sliding) {
            if (c.sliding_window) |w| {
                if (row.pos + 1 > w) lo = row.pos + 1 - w;
            }
        }
        const n_keys = row.pos + 1 - lo;
        const scores = scores_buf[0..n_keys];
        const slope: f32 = if (model.alibi_slopes.len > 0) model.alibi_slopes[h] else 0;
        const kbase = ctx.cache.kSlot(ctx.layer, row.b, lo).ptr + kvh * hd;
        const vbase = ctx.cache.vSlot(ctx.layer, row.b, lo).ptr + kvh * hd;
        if (vectorised) {
            tensor.attentionScores(scores, q, kbase, stride, c.attention_scale);
        } else {
            for (scores, 0..) |*s, p| s.* = tensor.dot(q, kbase[p * stride ..][0..hd]) * c.attention_scale;
        }
        if (c.attn_logit_softcapping) |cap| {
            for (scores) |*s| s.* = cap * std.math.tanh(s.* / cap);
        }
        if (slope != 0) {
            for (scores, 0..) |*s, p| s.* += slope * @as(f32, @floatFromInt(lo + p));
        }
        if (ctx.sinks) |sk| softmaxWithSink(scores, sk[h]) else tensor.softmaxInPlace(scores);
        if (vectorised) {
            tensor.attentionValues(out, scores, vbase, stride);
        } else {
            @memset(out, 0);
            for (scores, 0..) |s, p| tensor.axpy(out, s, vbase[p * stride ..][0..vd]);
        }
    }
}

/// Softmax over `x` with an extra sink logit that takes probability mass but
/// contributes no value (gpt-oss).
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

/// Workspace for forward passes, sized for `max_rows` tokens per call.
pub const Workspace = struct {
    gpa: Allocator,
    x: []f32,
    h: []f32,
    h2: []f32,
    q: []f32,
    k: []f32,
    v: []f32,
    /// Fused qkv product `[rows][q + 2 kv]` (empty for separate projections).
    qkv: []f32,
    attn: []f32,
    o: []f32,
    /// MLP output.
    m: []f32,
    gate: []f32,
    up: []f32,
    /// Fused gate/up product `[rows][2 I]` (empty unless the MLP is `gated_fused`).
    gate_up: []f32,
    logits: []f32,
    max_rows: usize,
    max_logit_rows: usize,

    /// Bytes `init` allocates per row (everything but the logits).
    pub fn bytesPerRow(c: *const Config) u64 {
        const hidden: u64 = c.hidden_size;
        const qd: u64 = c.num_heads * c.head_dim;
        const kvd: u64 = c.num_kv_heads * c.head_dim;
        const inter: u64 = c.intermediate_size;
        return (5 * hidden + 2 * qd + 2 * kvd + fusedQkvRows(c) + 2 * inter + fusedGateUpCols(c) + linearCols(c)) * 4;
    }

    fn fusedQkvRows(c: *const Config) usize {
        return if (c.qkv_layout != .separate and c.mla == null) (c.num_heads + 2 * c.num_kv_heads) * c.head_dim else 0;
    }

    fn fusedGateUpCols(c: *const Config) usize {
        return if (c.mlp == .gated_fused) 2 * c.intermediate_size else 0;
    }

    fn linearCols(c: *const Config) usize {
        if (!c.has_linear) return 0;
        return 2 * c.linear_k_heads * c.linear_k_dim + 3 * c.linear_v_heads * c.linear_v_dim + 2 * c.linear_v_heads;
    }

    pub fn init(gpa: Allocator, c: *const Config, max_rows: usize, max_logit_rows: usize) !Workspace {
        const hidden = c.hidden_size;
        const qd = c.num_heads * c.head_dim;
        const kvd = c.num_kv_heads * c.head_dim;
        const qkv_rows = fusedQkvRows(c);
        const gu = fusedGateUpCols(c);
        var self = Workspace{
            .gpa = gpa,
            .x = &.{},
            .h = &.{},
            .h2 = &.{},
            .q = &.{},
            .k = &.{},
            .v = &.{},
            .qkv = &.{},
            .attn = &.{},
            .o = &.{},
            .m = &.{},
            .gate = &.{},
            .up = &.{},
            .gate_up = &.{},
            .logits = &.{},
            .max_rows = max_rows,
            .max_logit_rows = max_logit_rows,
        };
        errdefer self.deinit();
        self.x = try gpa.alloc(f32, max_rows * hidden);
        self.h = try gpa.alloc(f32, max_rows * hidden);
        self.h2 = try gpa.alloc(f32, max_rows * hidden);
        self.q = try gpa.alloc(f32, max_rows * qd);
        self.k = try gpa.alloc(f32, max_rows * kvd);
        self.v = try gpa.alloc(f32, max_rows * kvd);
        self.qkv = try gpa.alloc(f32, max_rows * qkv_rows);
        self.attn = try gpa.alloc(f32, max_rows * qd);
        self.o = try gpa.alloc(f32, max_rows * hidden);
        self.m = try gpa.alloc(f32, max_rows * hidden);
        self.gate = try gpa.alloc(f32, max_rows * c.intermediate_size);
        self.up = try gpa.alloc(f32, max_rows * c.intermediate_size);
        self.gate_up = try gpa.alloc(f32, max_rows * gu);
        self.logits = try gpa.alloc(f32, max_logit_rows * c.vocab_size);
        return self;
    }

    pub fn deinit(self: *Workspace) void {
        self.gpa.free(self.x);
        self.gpa.free(self.h);
        self.gpa.free(self.h2);
        self.gpa.free(self.q);
        self.gpa.free(self.k);
        self.gpa.free(self.v);
        self.gpa.free(self.qkv);
        self.gpa.free(self.attn);
        self.gpa.free(self.o);
        self.gpa.free(self.m);
        self.gpa.free(self.gate);
        self.gpa.free(self.up);
        self.gpa.free(self.gate_up);
        self.gpa.free(self.logits);
    }
};

/// Options for a forward call.
pub const ForwardOptions = struct {
    /// If set, hidden states (residual stream) for the rows listed in
    /// `capture_rows` are written to `residuals[layer_entry][i][hidden]` where
    /// layer_entry 0 is the embedding output and entry L is the output of layer L-1.
    capture_rows: []const usize = &.{},
    residuals: ?[]f32 = null,
    /// Rows for which logits should be computed (indexes into `rows`).
    logit_rows: []const usize = &.{},
};

/// Applies the family's normalisation to one vector.
fn applyNorm(c: *const Config, out: []f32, x: []const f32, nm: Norm) void {
    switch (c.norm) {
        .rms => tensor.rmsnorm(out, x, nm.w, c.rms_norm_eps, false),
        .rms_gemma => tensor.rmsnorm(out, x, nm.w, c.rms_norm_eps, true),
        .layer => tensor.layernorm(out, x, nm.w, nm.b, c.rms_norm_eps, false),
        .layer_1p => tensor.layernorm(out, x, nm.w, nm.b, c.rms_norm_eps, true),
        .none => tensor.layernorm(out, x, &.{}, null, c.rms_norm_eps, false),
    }
}

/// `out[i] = norm(x[i])` for `n` rows, or a copy when there is no norm.
fn normRows(c: *const Config, out: []f32, x: []const f32, n: usize, hidden: usize, nm: ?Norm) void {
    if (nm) |norm| {
        var i: usize = 0;
        while (i < n) : (i += 1) applyNorm(c, out[i * hidden ..][0..hidden], x[i * hidden ..][0..hidden], norm);
    } else {
        @memcpy(out[0 .. n * hidden], x[0 .. n * hidden]);
    }
}

fn normRowsInPlace(c: *const Config, buf: []f32, n: usize, hidden: usize, nm: Norm, tmp: []f32) void {
    normRows(c, tmp, buf, n, hidden, nm);
    @memcpy(buf[0 .. n * hidden], tmp[0 .. n * hidden]);
}

/// Normalises a query/key vector in place: RMS or LayerNorm per the family
/// (weightless RMS when `w` is empty, e.g. the Llama 4 L2 norm).
fn normVecInPlace(c: *const Config, x: []f32, w: []const f32, b: ?[]const f32) void {
    var tmp: [1024]f32 = undefined;
    const out = tmp[0..x.len];
    if (w.len == 0) {
        var ss: f32 = 0;
        for (x) |v| ss += v * v;
        const inv = 1.0 / @sqrt(ss / @as(f32, @floatFromInt(x.len)) + c.rms_norm_eps);
        for (x) |*v| v.* *= inv;
        return;
    }
    switch (c.norm) {
        .rms, .none => tensor.rmsnorm(out, x, w, c.rms_norm_eps, false),
        .rms_gemma => tensor.rmsnorm(out, x, w, c.rms_norm_eps, true),
        .layer, .layer_1p => tensor.layernorm(out, x, w, b, c.rms_norm_eps, false),
    }
    @memcpy(x, out);
}

/// Applies the q/k norm to head `h` of a projection row (`.head`, `.heads`, `.l2`).
fn qkNormHead(c: *const Config, head: []f32, nm: Norm, h: usize) void {
    const hd = head.len;
    switch (c.qk_norm) {
        .head => normVecInPlace(c, head, nm.w[0..hd], if (nm.b) |b| b[0..hd] else null),
        .heads => normVecInPlace(c, head, nm.w[h * hd ..][0..hd], if (nm.b) |b| b[h * hd ..][0..hd] else null),
        .l2 => normVecInPlace(c, head, &.{}, null),
        .none, .full => {},
    }
}

fn ropeHead(c: *const Config, x: []f32, cos_row: []const f32, sin_row: []const f32) void {
    const d = c.rotary_dim;
    if (d == 0) return;
    switch (c.rope_style) {
        .neox => tensor.applyRope(x[0..d], cos_row, sin_row),
        .gptj => tensor.applyRopeInterleaved(x[0..d], cos_row, sin_row),
    }
}

fn addBias(buf: []f32, n: usize, width: usize, bias: []const f32) void {
    var i: usize = 0;
    while (i < n) : (i += 1) tensor.axpy(buf[i * width ..][0..width], 1.0, bias[0..width]);
}

fn clampAll(buf: []f32, limit: f32) void {
    for (buf) |*v| v.* = std.math.clamp(v.*, -limit, limit);
}

/// Splits one fused qkv row into q, k and v according to the family layout.
fn scatterQkv(c: *const Config, fused: []const f32, q: []f32, k: []f32, v: []f32) void {
    const hd = c.head_dim;
    const nh = c.num_heads;
    const nkv = c.num_kv_heads;
    const qd = nh * hd;
    const kvd = nkv * hd;
    switch (c.qkv_layout) {
        .separate => unreachable,
        .concat => {
            @memcpy(q, fused[0..qd]);
            @memcpy(k, fused[qd..][0..kvd]);
            @memcpy(v, fused[qd + kvd ..][0..kvd]);
        },
        .heads_interleaved => {
            std.debug.assert(nkv == nh);
            for (0..nh) |h| {
                @memcpy(q[h * hd ..][0..hd], fused[h * 3 * hd ..][0..hd]);
                @memcpy(k[h * hd ..][0..hd], fused[h * 3 * hd + hd ..][0..hd]);
                @memcpy(v[h * hd ..][0..hd], fused[h * 3 * hd + 2 * hd ..][0..hd]);
            }
        },
        .grouped => {
            const groups = nh / nkv;
            for (0..nkv) |g| {
                const base = g * (groups + 2) * hd;
                for (0..groups) |j| @memcpy(q[(g * groups + j) * hd ..][0..hd], fused[base + j * hd ..][0..hd]);
                @memcpy(k[g * hd ..][0..hd], fused[base + groups * hd ..][0..hd]);
                @memcpy(v[g * hd ..][0..hd], fused[base + (groups + 1) * hd ..][0..hd]);
            }
        },
    }
}

/// Multi-head latent attention projections: fills `ws.q` (`[n][heads][nope | rope]`),
/// `ws.k` (same layout, the rope part shared by every head) and `ws.v`
/// (`[n][heads][v_head_dim]` at `head_dim` stride).
fn mlaProject(model: *const Model, layer: *const Layer, ws: *Workspace, h: []const f32, n: usize) !void {
    const c = &model.config;
    const m = c.mla.?;
    const mw = layer.mla.?;
    const gpa = model.gpa;
    const nh = c.num_heads;
    const hd = c.head_dim;
    const nope = m.qk_nope_head_dim;
    const rd = m.qk_rope_head_dim;
    const vd = m.v_head_dim;
    const eps = c.rms_norm_eps;
    if (mw.q_a) |qa| {
        const qlr = m.q_lora_rank.?;
        const qa_out = try gpa.alloc(f32, n * qlr);
        defer gpa.free(qa_out);
        try tensor.matmulT(model.pool, gpa, qa_out, h, n, qa, null);
        const tmp = try gpa.alloc(f32, qlr);
        defer gpa.free(tmp);
        for (0..n) |t| {
            const row = qa_out[t * qlr ..][0..qlr];
            tensor.rmsnorm(tmp, row, mw.q_a_norm.?, eps, false);
            @memcpy(row, tmp);
        }
        try tensor.matmulT(model.pool, gpa, ws.q, qa_out, n, mw.q_b, null);
    } else {
        try tensor.matmulT(model.pool, gpa, ws.q, h, n, mw.q_b, null);
    }
    const kvr = m.kv_lora_rank + rd;
    const kva = try gpa.alloc(f32, n * kvr);
    defer gpa.free(kva);
    try tensor.matmulT(model.pool, gpa, kva, h, n, mw.kv_a, null);
    const ckv = try gpa.alloc(f32, n * m.kv_lora_rank);
    defer gpa.free(ckv);
    for (0..n) |t| tensor.rmsnorm(ckv[t * m.kv_lora_rank ..][0..m.kv_lora_rank], kva[t * kvr ..][0..m.kv_lora_rank], mw.kv_a_norm, eps, false);
    const kvb_rows = nh * (nope + vd);
    const kvb = try gpa.alloc(f32, n * kvb_rows);
    defer gpa.free(kvb);
    try tensor.matmulT(model.pool, gpa, kvb, ckv, n, mw.kv_b, null);
    for (0..n) |t| {
        const k_pe = kva[t * kvr + m.kv_lora_rank ..][0..rd];
        for (0..nh) |hh| {
            const src = kvb[t * kvb_rows + hh * (nope + vd) ..][0 .. nope + vd];
            const k = ws.k[t * nh * hd + hh * hd ..][0..hd];
            const v = ws.v[t * nh * hd + hh * hd ..][0..hd];
            @memcpy(k[0..nope], src[0..nope]);
            @memcpy(k[nope..][0..rd], k_pe);
            @memset(v, 0);
            @memcpy(v[0..vd], src[nope..][0..vd]);
        }
    }
}

/// Gated DeltaNet linear-attention sublayer (Qwen hybrids): projections, the
/// depthwise causal convolution, the per-sequence delta-rule recurrence and
/// the gated output projection (with its delta). Reads `h`, writes `ws.o`.
/// Projections are order independent and run over every row at once; the
/// recurrence runs each batch slot's rows in order against `cache.linear`.
fn linearForward(model: *const Model, layer: *const Layer, li: usize, ws: *Workspace, cache: *KvCache, h: []const f32, rows: []const Row) !void {
    const c = &model.config;
    const gpa = model.gpa;
    const lin = layer.linear.?;
    const lcache = &(cache.linear orelse return error.MissingLinearCache);
    const n = rows.len;
    const hidden = c.hidden_size;
    const kh = c.linear_k_heads;
    const kd = c.linear_k_dim;
    const vh = c.linear_v_heads;
    const vd = c.linear_v_dim;
    const kd_tot = kh * kd;
    const vd_tot = vh * vd;
    const kc = c.linear_conv_kernel;
    const conv_dim = 2 * kd_tot + vd_tot;
    const rep = vh / kh;
    const eps = c.rms_norm_eps;
    const qkv_cols = 2 * kd_tot + vd_tot;

    const mixed = try gpa.alloc(f32, n * qkv_cols);
    defer gpa.free(mixed);
    const zbuf = try gpa.alloc(f32, n * vd_tot);
    defer gpa.free(zbuf);
    const bbuf = try gpa.alloc(f32, n * vh);
    defer gpa.free(bbuf);
    const abuf = try gpa.alloc(f32, n * vh);
    defer gpa.free(abuf);
    if (lin.qkvz) |w| {
        const cols = 2 * kd_tot + 2 * vd_tot;
        const p = try gpa.alloc(f32, n * cols);
        defer gpa.free(p);
        try tensor.matmulT(model.pool, gpa, p, h, n, w, null);
        const sub = vd_tot / kh;
        for (0..n) |t| {
            const pr = p[t * cols ..][0..cols];
            const mx = mixed[t * qkv_cols ..][0..qkv_cols];
            const zz = zbuf[t * vd_tot ..][0..vd_tot];
            for (0..kh) |g| {
                const gr = pr[g * (2 * kd + 2 * sub) ..][0 .. 2 * kd + 2 * sub];
                @memcpy(mx[g * kd ..][0..kd], gr[0..kd]);
                @memcpy(mx[kd_tot + g * kd ..][0..kd], gr[kd .. 2 * kd]);
                @memcpy(mx[2 * kd_tot + g * sub ..][0..sub], gr[2 * kd .. 2 * kd + sub]);
                @memcpy(zz[g * sub ..][0..sub], gr[2 * kd + sub ..][0..sub]);
            }
        }
    } else {
        try tensor.matmulT(model.pool, gpa, mixed, h, n, lin.qkv.?, null);
        try tensor.matmulT(model.pool, gpa, zbuf, h, n, lin.z.?, null);
    }
    if (lin.ba) |w| {
        const p = try gpa.alloc(f32, n * 2 * vh);
        defer gpa.free(p);
        try tensor.matmulT(model.pool, gpa, p, h, n, w, null);
        const sub = vh / kh;
        for (0..n) |t| {
            const pr = p[t * 2 * vh ..][0 .. 2 * vh];
            const bb = bbuf[t * vh ..][0..vh];
            const aa = abuf[t * vh ..][0..vh];
            for (0..kh) |g| {
                @memcpy(bb[g * sub ..][0..sub], pr[g * 2 * sub ..][0..sub]);
                @memcpy(aa[g * sub ..][0..sub], pr[g * 2 * sub + sub ..][0..sub]);
            }
        }
    } else {
        try tensor.matmulT(model.pool, gpa, bbuf, h, n, lin.b.?, null);
        try tensor.matmulT(model.pool, gpa, abuf, h, n, lin.a.?, null);
    }

    const core = try gpa.alloc(f32, n * vd_tot);
    defer gpa.free(core);
    const qh = try gpa.alloc(f32, vh * kd);
    defer gpa.free(qh);
    const khb = try gpa.alloc(f32, vh * kd);
    defer gpa.free(khb);
    const vv = try gpa.alloc(f32, vh * vd);
    defer gpa.free(vv);
    const y = try gpa.alloc(f32, conv_dim);
    defer gpa.free(y);
    const convw = lin.conv.?.data;
    const cdt = lin.conv.?.dtype;
    for (0..n) |t| {
        const b = rows[t].b;
        const pos = rows[t].pos;
        const S = lcache.state(li, b);
        const C = lcache.conv(li, b);
        const next = lcache.nextPos(li, b);
        if (pos == 0) {
            @memset(S, 0);
            @memset(C, 0);
            next.* = 0;
        }
        if (pos != next.*) return error.NonContiguousRows;
        const m = mixed[t * qkv_cols ..][0..qkv_cols];
        for (0..conv_dim) |j| {
            var acc: f32 = elemAt(cdt, convw, j * kc + kc - 1) * m[j];
            if (kc > 1) {
                for (1..kc) |i| acc += elemAt(cdt, convw, j * kc + kc - 1 - i) * C[j * (kc - 1) + kc - 1 - i];
            }
            y[j] = c.activation.apply(acc);
            if (kc > 1) {
                @memmove(C[j * (kc - 1) ..][0 .. kc - 2], C[j * (kc - 1) + 1 ..][0 .. kc - 2]);
                C[j * (kc - 1) + kc - 2] = m[j];
            }
        }
        const q0 = y[0..kd_tot];
        const k0 = y[kd_tot .. 2 * kd_tot];
        const v0 = y[2 * kd_tot ..][0..vd_tot];
        const bb = bbuf[t * vh ..][0..vh];
        const aa = abuf[t * vh ..][0..vh];
        for (0..vh) |vhi| {
            const khi = vhi / rep;
            @memcpy(qh[vhi * kd ..][0..kd], q0[khi * kd ..][0..kd]);
            @memcpy(khb[vhi * kd ..][0..kd], k0[khi * kd ..][0..kd]);
            @memcpy(vv[vhi * vd ..][0..vd], v0[vhi * vd ..][0..vd]);
        }
        l2normRows(qh, vh, kd);
        l2normRows(khb, vh, kd);
        const qscale = 1.0 / @sqrt(@as(f32, @floatFromInt(kd)));
        tensor.scale(qh, qscale);
        for (0..vh) |vhi| {
            const beta = 1.0 / (1.0 + @exp(-bb[vhi]));
            const gv = -@exp(lin.a_log[vhi]) * softplus(aa[vhi] + lin.dt_bias[vhi]);
            const dec = @exp(gv);
            const St = S[vhi * kd * vd ..][0 .. kd * vd];
            const q = qh[vhi * kd ..][0..kd];
            const k = khb[vhi * kd ..][0..kd];
            const v = vv[vhi * vd ..][0..vd];
            for (St) |*s| s.* *= dec;
            const out = core[t * vd_tot + vhi * vd ..][0..vd];
            for (0..vd) |jj| {
                var mem: f32 = 0;
                for (0..kd) |i| mem += St[i * vd + jj] * k[i];
                const delta = (v[jj] - mem) * beta;
                for (0..kd) |i| St[i * vd + jj] += k[i] * delta;
            }
            for (0..vd) |jj| {
                var s: f32 = 0;
                for (0..kd) |i| s += St[i * vd + jj] * q[i];
                out[jj] = s;
            }
        }
        const zz = zbuf[t * vd_tot ..][0..vd_tot];
        const oo = core[t * vd_tot ..][0..vd_tot];
        for (0..vh) |vhi| {
            const o = oo[vhi * vd ..][0..vd];
            const z = zz[vhi * vd ..][0..vd];
            var variance: f32 = 0;
            for (o) |v| variance += v * v;
            variance = variance / @as(f32, @floatFromInt(vd)) + eps;
            const inv = 1.0 / @sqrt(variance);
            for (o, 0..) |*v, jj| v.* = v.* * inv * lin.norm[jj] * tensor.silu(z[jj]);
        }
        next.* = pos + 1;
    }
    try tensor.matmulT(model.pool, gpa, ws.o, core, n, layer.o, if (layer.o_delta) |*d| d else null);
    if (layer.o_bias) |bias| addBias(ws.o, n, hidden, bias);
}

/// Kimi Delta Attention sublayer (`KimiLinearDeltaAttention`): the q/k/v
/// projections, the depthwise causal convolution over them, the delta-rule
/// recurrence with a per-channel decay from the low-rank forget gate, the
/// sigmoid-gated output norm and the output projection (with its delta).
/// Reads `h`, writes `ws.o`. Like `linearForward`, the projections run over
/// every row at once and the recurrence runs each batch slot's rows in order
/// against `cache.linear` (the chunked reference kernel computes the same
/// recurrence in blocks).
fn kdaForward(model: *const Model, layer: *const Layer, li: usize, ws: *Workspace, cache: *KvCache, h: []const f32, rows: []const Row) !void {
    const c = &model.config;
    const gpa = model.gpa;
    const lin = layer.linear.?;
    const lcache = &(cache.linear orelse return error.MissingLinearCache);
    const n = rows.len;
    const hidden = c.hidden_size;
    const nh = c.linear_v_heads;
    const hd = c.linear_v_dim;
    const dim = nh * hd;
    const kc = c.linear_conv_kernel;
    const conv_dim = 3 * dim;
    const eps = c.rms_norm_eps;

    // `mixed` holds q | k | v per row (the convolution's channels).
    const mixed = try gpa.alloc(f32, n * conv_dim);
    defer gpa.free(mixed);
    {
        const tmp = try gpa.alloc(f32, n * dim);
        defer gpa.free(tmp);
        const projs = .{ lin.q.?, lin.k.?, lin.v.? };
        inline for (projs, 0..) |w, s| {
            try tensor.matmulT(model.pool, gpa, tmp, h, n, w, null);
            for (0..n) |t| @memcpy(mixed[t * conv_dim + s * dim ..][0..dim], tmp[t * dim ..][0..dim]);
        }
    }
    // Forget gate: g = -exp(A_log[head]) * softplus(f_b(f_a(h)) + dt_bias), per channel.
    const low = try gpa.alloc(f32, n * hd);
    defer gpa.free(low);
    const decay = try gpa.alloc(f32, n * dim);
    defer gpa.free(decay);
    try tensor.matmulT(model.pool, gpa, low, h, n, lin.f_a.?, null);
    try tensor.matmulT(model.pool, gpa, decay, low, n, lin.f_b.?, null);
    for (0..n) |t| {
        const g = decay[t * dim ..][0..dim];
        for (0..nh) |hh| {
            const rate = -@exp(lin.a_log[hh]);
            for (0..hd) |i| g[hh * hd + i] = rate * softplus(g[hh * hd + i] + lin.dt_bias[hh * hd + i]);
        }
    }
    const bbuf = try gpa.alloc(f32, n * nh);
    defer gpa.free(bbuf);
    try tensor.matmulT(model.pool, gpa, bbuf, h, n, lin.b.?, null);
    // Output gate: g_b(g_a(h)).
    const gate = try gpa.alloc(f32, n * dim);
    defer gpa.free(gate);
    try tensor.matmulT(model.pool, gpa, low, h, n, lin.g_a.?, null);
    try tensor.matmulT(model.pool, gpa, gate, low, n, lin.g_b.?, null);

    const core = try gpa.alloc(f32, n * dim);
    defer gpa.free(core);
    const y = try gpa.alloc(f32, conv_dim);
    defer gpa.free(y);
    const qscale = 1.0 / @sqrt(@as(f32, @floatFromInt(hd)));
    for (0..n) |t| {
        const b = rows[t].b;
        const pos = rows[t].pos;
        const S = lcache.state(li, b);
        const C = lcache.conv(li, b);
        const next = lcache.nextPos(li, b);
        if (pos == 0) {
            @memset(S, 0);
            @memset(C, 0);
            next.* = 0;
        }
        if (pos != next.*) return error.NonContiguousRows;
        const m = mixed[t * conv_dim ..][0..conv_dim];
        for (0..conv_dim) |j| {
            var acc: f32 = kdaConvTap(lin, dim, kc, j, kc - 1) * m[j];
            if (kc > 1) {
                for (1..kc) |i| acc += kdaConvTap(lin, dim, kc, j, kc - 1 - i) * C[j * (kc - 1) + kc - 1 - i];
            }
            y[j] = c.activation.apply(acc);
            if (kc > 1) {
                @memmove(C[j * (kc - 1) ..][0 .. kc - 2], C[j * (kc - 1) + 1 ..][0 .. kc - 2]);
                C[j * (kc - 1) + kc - 2] = m[j];
            }
        }
        const q0 = y[0..dim];
        const k0 = y[dim .. 2 * dim];
        const v0 = y[2 * dim ..][0..dim];
        l2normRows(q0, nh, hd);
        l2normRows(k0, nh, hd);
        tensor.scale(q0, qscale);
        const g = decay[t * dim ..][0..dim];
        for (0..nh) |hh| {
            const beta = 1.0 / (1.0 + @exp(-bbuf[t * nh + hh]));
            const St = S[hh * hd * hd ..][0 .. hd * hd];
            const q = q0[hh * hd ..][0..hd];
            const k = k0[hh * hd ..][0..hd];
            const v = v0[hh * hd ..][0..hd];
            const gh = g[hh * hd ..][0..hd];
            // Per-channel decay: row i of the [k_dim][v_dim] state decays by exp(g[i]).
            for (0..hd) |i| {
                const dec = @exp(gh[i]);
                for (St[i * hd ..][0..hd]) |*sv| sv.* *= dec;
            }
            const out = core[t * dim + hh * hd ..][0..hd];
            for (0..hd) |jj| {
                var mem: f32 = 0;
                for (0..hd) |i| mem += St[i * hd + jj] * k[i];
                const delta = (v[jj] - mem) * beta;
                for (0..hd) |i| St[i * hd + jj] += k[i] * delta;
            }
            for (0..hd) |jj| {
                var sum: f32 = 0;
                for (0..hd) |i| sum += St[i * hd + jj] * q[i];
                out[jj] = sum;
            }
        }
        // Gated RMSNorm per head: w * rms(o) * sigmoid(gate).
        const gg = gate[t * dim ..][0..dim];
        const oo = core[t * dim ..][0..dim];
        for (0..nh) |hh| {
            const o = oo[hh * hd ..][0..hd];
            const z = gg[hh * hd ..][0..hd];
            var variance: f32 = 0;
            for (o) |v| variance += v * v;
            variance = variance / @as(f32, @floatFromInt(hd)) + eps;
            const inv = 1.0 / @sqrt(variance);
            for (o, 0..) |*v, jj| v.* = v.* * inv * lin.norm[jj] / (1.0 + @exp(-z[jj]));
        }
        next.* = pos + 1;
    }
    try tensor.matmulT(model.pool, gpa, ws.o, core, n, layer.o, if (layer.o_delta) |*d| d else null);
    if (layer.o_bias) |bias| addBias(ws.o, n, hidden, bias);
}

/// Tap `i` of channel `j`'s convolution kernel: from the fused `[3 dim][kernel]`
/// tensor, or from the q/k/v tensor the channel belongs to.
fn kdaConvTap(lin: LinearWeights, dim: usize, kc: usize, j: usize, i: usize) f32 {
    if (lin.conv) |w| return elemAt(w.dtype, w.data, j * kc + i);
    const s = j / dim;
    const w = switch (s) {
        0 => lin.conv_q.?,
        1 => lin.conv_k.?,
        else => lin.conv_v.?,
    };
    return elemAt(w.dtype, w.data, (j - s * dim) * kc + i);
}

fn elemAt(dtype: tensor.DType, data: []const u8, idx: usize) f32 {
    return switch (dtype) {
        .f32 => @bitCast(std.mem.readInt(u32, data[idx * 4 ..][0..4], .little)),
        .bf16 => tensor.bf16ToF32(std.mem.readInt(u16, data[idx * 2 ..][0..2], .little)),
        .f16 => tensor.f16ToF32(std.mem.readInt(u16, data[idx * 2 ..][0..2], .little)),
        else => unreachable,
    };
}

fn l2normRows(x: []f32, heads: usize, dim: usize) void {
    for (0..heads) |h| {
        const row = x[h * dim ..][0..dim];
        var s: f32 = 0;
        for (row) |v| s += v * v;
        const inv = 1.0 / @sqrt(s + 1e-6);
        tensor.scale(row, inv);
    }
}

fn softplus(x: f32) f32 {
    if (x > 20.0) return x;
    return @log(1.0 + @exp(x));
}

/// Attention sublayer: projections, q/k norms, RoPE, KV cache update,
/// attention and the output projection (with its delta). Reads `h`, writes `ws.o`.
fn attention(model: *const Model, layer: *const Layer, li: usize, ws: *Workspace, cache: *KvCache, h: []const f32, rows: []const Row) !void {
    const c = &model.config;
    const gpa = model.gpa;
    const n = rows.len;
    const hd = c.head_dim;
    const qd = c.num_heads * hd;
    const kvd = c.num_kv_heads * hd;

    if (layer.mla != null) {
        try mlaProject(model, layer, ws, h, n);
    } else if (layer.qkv) |w| {
        const qkv_rows = qd + 2 * kvd;
        try tensor.matmulT(model.pool, gpa, ws.qkv, h, n, w, null);
        if (layer.qkv_bias) |b| addBias(ws.qkv, n, qkv_rows, b);
        var i: usize = 0;
        while (i < n) : (i += 1) scatterQkv(c, ws.qkv[i * qkv_rows ..][0..qkv_rows], ws.q[i * qd ..][0..qd], ws.k[i * kvd ..][0..kvd], ws.v[i * kvd ..][0..kvd]);
    } else if (c.gated_attention) {
        const tmp = try gpa.alloc(f32, n * 2 * qd);
        defer gpa.free(tmp);
        const gate = try gpa.alloc(f32, n * qd);
        defer gpa.free(gate);
        try tensor.matmulT(model.pool, gpa, tmp, h, n, layer.q.?, null);
        if (layer.q_bias) |b| {
            if (b.len == 2 * qd) addBias(tmp, n, 2 * qd, b);
        }
        for (0..n) |i| {
            const qr = tmp[i * 2 * qd ..][0 .. 2 * qd];
            const qo = ws.q[i * qd ..][0..qd];
            const go = gate[i * qd ..][0..qd];
            for (0..c.num_heads) |hh| {
                @memcpy(qo[hh * hd ..][0..hd], qr[hh * 2 * hd ..][0..hd]);
                @memcpy(go[hh * hd ..][0..hd], qr[hh * 2 * hd + hd ..][0..hd]);
            }
        }
        if (layer.q_bias) |b| {
            if (b.len == qd) addBias(ws.q, n, qd, b);
        }
        try tensor.matmulT(model.pool, gpa, ws.k, h, n, layer.k.?, null);
        try tensor.matmulT(model.pool, gpa, ws.v, h, n, layer.v.?, null);
        if (layer.k_bias) |b| addBias(ws.k, n, kvd, b);
        if (layer.v_bias) |b| addBias(ws.v, n, kvd, b);
        try attentionTail(model, layer, li, ws, cache, rows, gate);
        return;
    } else {
        try tensor.matmulT(model.pool, gpa, ws.q, h, n, layer.q.?, null);
        try tensor.matmulT(model.pool, gpa, ws.k, h, n, layer.k.?, null);
        try tensor.matmulT(model.pool, gpa, ws.v, h, n, layer.v.?, null);
        if (layer.q_bias) |b| addBias(ws.q, n, qd, b);
        if (layer.k_bias) |b| addBias(ws.k, n, kvd, b);
        if (layer.v_bias) |b| addBias(ws.v, n, kvd, b);
    }
    try attentionTail(model, layer, li, ws, cache, rows, null);
}

/// Norms, RoPE, KV cache update, attention and the output projection shared
/// by the projection layouts. `gate` (gated full attention) multiplies the
/// attention output before the output projection.
fn attentionTail(model: *const Model, layer: *const Layer, li: usize, ws: *Workspace, cache: *KvCache, rows: []const Row, gate: ?[]const f32) !void {
    const c = &model.config;
    const gpa = model.gpa;
    const n = rows.len;
    const hidden = c.hidden_size;
    const hd = c.head_dim;
    const qd = c.num_heads * hd;
    const kvd = c.num_kv_heads * hd;
    const half = c.rotary_dim / 2;
    if (c.clip_qkv) |clip| {
        clampAll(ws.q[0 .. n * qd], clip);
        clampAll(ws.k[0 .. n * kvd], clip);
        clampAll(ws.v[0 .. n * kvd], clip);
    }

    const use_rope = c.rope_layers[li];
    const sliding = c.sliding_layers[li];
    const local = sliding and model.rope_cos_local.ptr != model.rope_cos.ptr;
    const cos = if (local) model.rope_cos_local else model.rope_cos;
    const sin = if (local) model.rope_sin_local else model.rope_sin;
    const rope_off: usize = if (c.mla) |m| m.qk_nope_head_dim else 0;
    const do_qk_norm = !(c.qk_norm_rope_only and !use_rope);
    var i: usize = 0;
    while (i < n) : (i += 1) {
        const pos = @min(rows[i].pos, model.rope_len - 1);
        const cr = cos[pos * half ..][0..half];
        const sr = sin[pos * half ..][0..half];
        const qrow = ws.q[i * qd ..][0..qd];
        const krow = ws.k[i * kvd ..][0..kvd];
        if (do_qk_norm and c.qk_norm == .full) {
            if (layer.q_norm) |nm| normVecInPlace(c, qrow, nm.w, nm.b);
            if (layer.k_norm) |nm| normVecInPlace(c, krow, nm.w, nm.b);
        }
        var temp: ?f32 = null;
        if (!use_rope) {
            if (c.attn_temperature) |t| {
                const p: f32 = @floatFromInt(rows[i].pos);
                temp = @log(@floor((p + 1.0) / t.floor_scale) + 1.0) * t.attn_scale + 1.0;
            }
        }
        var hh: usize = 0;
        while (hh < c.num_heads) : (hh += 1) {
            const q = qrow[hh * hd ..][0..hd];
            if (do_qk_norm) if (layer.q_norm) |nm| qkNormHead(c, q, nm, hh);
            if (use_rope) ropeHead(c, q[rope_off..], cr, sr);
            if (temp) |t| tensor.scale(q, t);
        }
        hh = 0;
        while (hh < c.num_kv_heads) : (hh += 1) {
            const k = krow[hh * hd ..][0..hd];
            if (do_qk_norm) if (layer.k_norm) |nm| qkNormHead(c, k, nm, hh);
            if (use_rope) ropeHead(c, k[rope_off..], cr, sr);
        }
        @memcpy(cache.kSlot(li, rows[i].b, rows[i].pos), krow);
        @memcpy(cache.vSlot(li, rows[i].b, rows[i].pos), ws.v[i * kvd ..][0..kvd]);
        cache.noteWrite(rows[i].pos);
    }
    const n_tasks = n * c.num_heads;
    const chunks = @max(1, @min(model.pool.threads, n_tasks));
    const per = (n_tasks + chunks - 1) / chunks;
    var max_keys: usize = 1;
    for (rows) |row| max_keys = @max(max_keys, row.pos + 1);
    const scores = try model.pool.allocScratch(gpa, chunks * max_keys);
    defer model.pool.freeScratch(gpa, scores);
    const actx = AttnCtx{
        .model = model,
        .cache = cache,
        .layer = li,
        .rows = rows,
        .q = ws.q,
        .out = ws.attn,
        .sliding = sliding,
        .sinks = layer.sinks,
        .n_tasks = n_tasks,
        .chunks = chunks,
        .per = per,
        .scores = scores,
        .max_keys = max_keys,
    };
    model.pool.parallelFor(chunks * per, &actx, attentionWorker);
    if (gate) |g| {
        const swish = c.gate_swish;
        for (0..n) |r| {
            const a = ws.attn[r * qd ..][0..qd];
            const gg = g[r * qd ..][0..qd];
            for (a, 0..) |*v, j| {
                const s = 1.0 / (1.0 + @exp(-gg[j]));
                v.* *= if (swish) gg[j] * s else s;
            }
        }
    }
    const vd = c.v_head_dim;
    if (vd != hd) {
        // Compact `[n][heads][head_dim]` (v_head_dim valid per head) to `[n][heads * v_head_dim]`.
        var dst: usize = 0;
        for (0..n * c.num_heads) |hi| {
            std.mem.copyForwards(f32, ws.attn[dst..][0..vd], ws.attn[hi * hd ..][0..vd]);
            dst += vd;
        }
    }
    try tensor.matmulT(model.pool, gpa, ws.o, ws.attn, n, layer.o, if (layer.o_delta) |*d| d else null);
    if (layer.o_bias) |b| addBias(ws.o, n, hidden, b);
}

/// MLP sublayer (dense or mixture of experts): reads `h_in`, writes `ws.m`.
fn mlpBlock(model: *const Model, layer: *const Layer, li: usize, ws: *Workspace, h_in: []const f32, n: usize) !void {
    const c = &model.config;
    const gpa = model.gpa;
    const hidden = c.hidden_size;
    if (layer.moe) |*m| return moe.forward(model, m, li, ws.m, h_in, n);
    const down = layer.down.?;
    const inter = down.cols;
    var din: []f32 = ws.gate;
    switch (c.mlp) {
        .gated => {
            try tensor.matmulT(model.pool, gpa, ws.gate, h_in, n, layer.gate.?, null);
            try tensor.matmulT(model.pool, gpa, ws.up, h_in, n, layer.up.?, null);
            if (layer.gate_bias) |b| addBias(ws.gate, n, inter, b);
            if (layer.up_bias) |b| addBias(ws.up, n, inter, b);
            tensor.gatedActivation(model.pool, c.activation, ws.gate, ws.gate, ws.up, n, inter, inter, inter);
        },
        .gated_fused => {
            const gu = layer.gate_up.?;
            try tensor.matmulT(model.pool, gpa, ws.gate_up, h_in, n, gu, null);
            if (layer.up_bias) |b| addBias(ws.gate_up, n, 2 * inter, b);
            tensor.gatedActivation(model.pool, c.activation, ws.gate, ws.gate_up, ws.gate_up[inter..], n, inter, 2 * inter, inter);
        },
        .dense => {
            try tensor.matmulT(model.pool, gpa, ws.up, h_in, n, layer.up.?, null);
            if (layer.up_bias) |b| addBias(ws.up, n, inter, b);
            tensor.gatedActivation(model.pool, c.activation, ws.up, ws.up, null, n, inter, inter, inter);
            din = ws.up;
        },
    }
    try tensor.matmulT(model.pool, gpa, ws.m, din, n, down, if (layer.down_delta) |*d| d else null);
    if (layer.down_bias) |b| addBias(ws.m, n, hidden, b);
}

/// Runs the transformer over `tokens`/`rows` (n tokens). Logits for the
/// requested rows are written to `ws.logits[i * vocab ..]`.
///
/// Layers are the outer loop and tokens the inner one: each layer's weights
/// are acquired once (from the mapping, or read from disk into a budgeted
/// buffer in streamed mode) and applied to every token of the call before
/// being released. When `n` exceeds the workspace the residual stream is kept
/// in an `Activations` buffer (RAM, or a scratch file under a tight budget) and
/// processed in chunks of `ws.max_rows` rows per layer, so a layer is still
/// loaded only once per call. In streamed mode a decode step therefore costs
/// one full read of every layer plus the LM head: decode throughput is bounded
/// by storage bandwidth, not compute.
pub fn forward(model: *const Model, ws: *Workspace, cache: *KvCache, tokens: []const u32, rows: []const Row, opts: ForwardOptions) !void {
    const c = &model.config;
    const gpa = model.gpa;
    const n = tokens.len;
    std.debug.assert(n == rows.len);
    const hidden = c.hidden_size;
    const chunk_rows = ws.max_rows;
    const single = n <= chunk_rows;
    var act: ?stream.Activations = null;
    defer if (act) |*a| a.deinit();
    if (!single) act = try stream.Activations.init(gpa, model.io, n, hidden, .{
        .budget = model.budget,
        .scratch_dir = model.scratch_dir,
        .reserve = model.residentWeightNeed(),
        .evictable = model.expertCacheEvictable(),
        .force_scratch = model.spill_always,
    });

    // Embeddings.
    var start: usize = 0;
    while (start < n) : (start += chunk_rows) {
        const cn = @min(chunk_rows, n - start);
        const xs = if (single) ws.x[0 .. n * hidden] else act.?.chunkUninit(start, cn, ws.x);
        for (tokens[start..][0..cn], 0..) |t, i| {
            const x = xs[i * hidden ..][0..hidden];
            try model.embedRow(t, x);
            if (c.embed_scale != 1.0) tensor.scale(x, c.embed_scale);
            if (model.pos_embed_ref != null) {
                const pe = ws.h[0..hidden];
                try model.posEmbedRow(rows[start + i].pos, pe);
                tensor.axpy(x, 1.0, pe);
            }
            if (model.embed_norm) |nm| {
                applyNorm(c, ws.h[0..hidden], x, nm);
                @memcpy(x, ws.h[0..hidden]);
            }
        }
        if (!single) try act.?.commit(start, cn, xs);
        captureResiduals(opts, 0, hidden, start, cn, xs);
    }

    for (0..c.num_layers) |li| {
        if (model.budget) |b| try b.checkTime();
        var lease = try model.acquireLayer(li);
        defer lease.release();
        const layer = &lease.layer;
        try cache.beginLayer(li);
        start = 0;
        while (start < n) : (start += chunk_rows) {
            const cn = @min(chunk_rows, n - start);
            const xs = if (single) ws.x[0 .. n * hidden] else try act.?.chunk(start, cn, ws.x);
            try layerBlock(model, layer, li, ws, cache, xs, rows[start..][0..cn]);
            if (!single) try act.?.commit(start, cn, xs);
            captureResiduals(opts, li + 1, hidden, start, cn, xs);
        }
        try cache.endLayer(li);
    }

    // Final norm + logits for requested rows.
    if (opts.logit_rows.len > 0) {
        std.debug.assert(opts.logit_rows.len <= ws.max_logit_rows);
        // `ws.h` holds `max_rows` rows, which may be fewer than the logit rows under chunking.
        const h = try gpa.alloc(f32, opts.logit_rows.len * hidden);
        defer gpa.free(h);
        for (opts.logit_rows, 0..) |r, i| {
            const dst = h[i * hidden ..][0..hidden];
            if (single) {
                applyNorm(c, dst, ws.x[r * hidden ..][0..hidden], model.final_norm);
            } else {
                const tmp = ws.o[0..hidden];
                try act.?.readRow(r, tmp);
                applyNorm(c, dst, tmp, model.final_norm);
            }
        }
        const lm = try model.acquireLmHead();
        defer @constCast(&model.store).release(lm);
        try tensor.matmulT(model.pool, gpa, ws.logits, h, opts.logit_rows.len, lm.weight, null);
        const logits = ws.logits[0 .. opts.logit_rows.len * c.vocab_size];
        if (model.lm_head_bias) |b| addBias(logits, opts.logit_rows.len, c.vocab_size, b);
        if (c.logit_scale != 1.0) tensor.scale(logits, c.logit_scale);
        if (c.final_logit_softcapping) |cap| tensor.softcap(logits, cap);
    }
}

/// Copies the residual of every capture row inside `[start, start + cn)` into `residuals[entry]`.
fn captureResiduals(opts: ForwardOptions, entry: usize, hidden: usize, start: usize, cn: usize, xs: []const f32) void {
    const res = opts.residuals orelse return;
    for (opts.capture_rows, 0..) |r, ci| {
        if (r < start or r >= start + cn) continue;
        @memcpy(res[(entry * opts.capture_rows.len + ci) * hidden ..][0..hidden], xs[(r - start) * hidden ..][0..hidden]);
    }
}

/// One transformer layer (attention + MLP) applied in place to the residual
/// rows `x` (`rows.len` tokens). `layer` must hold resident weights.
///
/// Sequential layout: `x += attn(norm(x)); x += mlp(norm(x))`, with optional
/// norms on the sublayer outputs (Gemma, OLMo 2, GLM-4) and no input norm for
/// the post-norm families. Parallel layout (GPT-NeoX, Falcon, Phi, Cohere):
/// `x += attn(h) + mlp(h)` with `h = norm(x)` (or a second norm for the MLP).
fn layerBlock(model: *const Model, layer: *const Layer, li: usize, ws: *Workspace, cache: *KvCache, x: []f32, rows: []const Row) !void {
    const c = &model.config;
    const n = rows.len;
    const hidden = c.hidden_size;
    const h = ws.h[0 .. n * hidden];
    const rm = c.residual_multiplier;

    normRows(c, h, x, n, hidden, layer.input_norm);
    if (c.linear_layers[li]) {
        switch (c.linear_kind) {
            .gated_deltanet => try linearForward(model, layer, li, ws, cache, h, rows),
            .kda => try kdaForward(model, layer, li, ws, cache, h, rows),
        }
    } else {
        try attention(model, layer, li, ws, cache, h, rows);
    }
    const attn_out = ws.o[0 .. n * hidden];
    if (layer.post_attn_norm) |nm| normRowsInPlace(c, attn_out, n, hidden, nm, ws.h2);

    if (c.parallel_residual) {
        var mlp_in: []const f32 = h;
        if (layer.mlp_norm) |nm| {
            normRows(c, ws.h2, x, n, hidden, nm);
            mlp_in = ws.h2[0 .. n * hidden];
        }
        try mlpBlock(model, layer, li, ws, mlp_in, n);
        const m = ws.m[0 .. n * hidden];
        if (layer.post_ff_norm) |nm| normRowsInPlace(c, m, n, hidden, nm, ws.h2);
        tensor.axpy(x[0 .. n * hidden], rm, attn_out);
        tensor.axpy(x[0 .. n * hidden], rm, m);
    } else {
        tensor.axpy(x[0 .. n * hidden], rm, attn_out);
        normRows(c, h, x, n, hidden, layer.pre_ff_norm);
        try mlpBlock(model, layer, li, ws, h, n);
        const m = ws.m[0 .. n * hidden];
        if (layer.post_ff_norm) |nm| normRowsInPlace(c, m, n, hidden, nm, ws.h2);
        tensor.axpy(x[0 .. n * hidden], rm, m);
    }
}

// ---------------------------------------------------------------------------
// High-level batched API
// ---------------------------------------------------------------------------

pub const GenerateResult = struct {
    /// Generated token ids per prompt (caller frees each and the outer slice).
    tokens: [][]u32,
};

fn argmax(x: []const f32) u32 {
    var best: usize = 0;
    var bv: f32 = -std.math.inf(f32);
    for (x, 0..) |v, i| {
        if (v > bv) {
            bv = v;
            best = i;
        }
    }
    return @intCast(best);
}

/// Runs prefill over a batch of tokenised prompts. Fills the KV cache and
/// returns the next-token logits for each prompt in `logits_out` ([batch][vocab]).
/// Optionally captures residuals for the last prompt token ([layer+1][batch][hidden]).
pub fn prefill(model: *const Model, ws: *Workspace, cache: *KvCache, prompts: []const []const u32, logits_out: ?[]f32, residuals_out: ?[]f32) !void {
    const gpa = model.gpa;
    const c = &model.config;
    var total: usize = 0;
    for (prompts) |p| total += p.len;
    const tokens = try gpa.alloc(u32, total);
    defer gpa.free(tokens);
    const rows = try gpa.alloc(Row, total);
    defer gpa.free(rows);
    const last_rows = try gpa.alloc(usize, prompts.len);
    defer gpa.free(last_rows);
    var idx: usize = 0;
    for (prompts, 0..) |p, b| {
        for (p, 0..) |t, pos| {
            tokens[idx] = t;
            rows[idx] = .{ .b = b, .pos = pos };
            idx += 1;
        }
        last_rows[b] = idx - 1;
    }
    // Process in chunks that fit the workspace, keeping whole prompts together where possible.
    var start: usize = 0;
    var res_tmp: ?[]f32 = null;
    defer if (res_tmp) |r| gpa.free(r);
    if (residuals_out != null) res_tmp = try gpa.alloc(f32, (c.num_layers + 1) * prompts.len * c.hidden_size);
    while (start < total) {
        // In streamed mode `forward` chunks internally so that every layer is read
        // once for the whole batch; otherwise chunk to the workspace here.
        var end = if (model.streamed()) total else @min(total, start + ws.max_rows);
        // Do not split a prompt across chunks unless it is longer than the workspace.
        if (end < total) {
            var e = end;
            while (e > start and rows[e].pos != 0) e -= 1;
            if (e > start) end = e;
        }
        var lr = std.ArrayList(usize).empty;
        defer lr.deinit(gpa);
        var lr_batch = std.ArrayList(usize).empty;
        defer lr_batch.deinit(gpa);
        for (last_rows, 0..) |r, b| {
            if (r >= start and r < end) {
                try lr.append(gpa, r - start);
                try lr_batch.append(gpa, b);
            }
        }
        var residual_chunk: ?[]f32 = null;
        defer if (residual_chunk) |r| gpa.free(r);
        if (residuals_out != null and lr.items.len > 0) residual_chunk = try gpa.alloc(f32, (c.num_layers + 1) * lr.items.len * c.hidden_size);
        try forward(model, ws, cache, tokens[start..end], rows[start..end], .{
            .logit_rows = lr.items,
            .capture_rows = lr.items,
            .residuals = residual_chunk,
        });
        for (lr_batch.items, 0..) |b, i| {
            if (logits_out) |lo| @memcpy(lo[b * c.vocab_size ..][0..c.vocab_size], ws.logits[i * c.vocab_size ..][0..c.vocab_size]);
            if (residuals_out) |ro| {
                var l: usize = 0;
                while (l <= c.num_layers) : (l += 1) {
                    @memcpy(ro[(l * prompts.len + b) * c.hidden_size ..][0..c.hidden_size], residual_chunk.?[(l * lr.items.len + i) * c.hidden_size ..][0..c.hidden_size]);
                }
            }
        }
        start = end;
    }
}

/// Greedy generation for a batch of prompts.
pub fn generate(model: *const Model, ws: *Workspace, cache: *KvCache, prompts: []const []const u32, max_new_tokens: usize) ![][]u32 {
    const gpa = model.gpa;
    const c = &model.config;
    const b = prompts.len;
    const logits = try gpa.alloc(f32, b * c.vocab_size);
    defer gpa.free(logits);
    try prefill(model, ws, cache, prompts, logits, null);

    var outputs = try gpa.alloc(std.ArrayList(u32), b);
    defer gpa.free(outputs);
    for (outputs) |*o| o.* = .empty;
    errdefer for (outputs) |*o| o.deinit(gpa);
    var active = try gpa.alloc(bool, b);
    defer gpa.free(active);
    @memset(active, true);
    var positions = try gpa.alloc(usize, b);
    defer gpa.free(positions);
    for (prompts, 0..) |p, i| positions[i] = p.len;

    var tokens = try gpa.alloc(u32, b);
    defer gpa.free(tokens);
    var rows = try gpa.alloc(Row, b);
    defer gpa.free(rows);
    var logit_rows = try gpa.alloc(usize, b);
    defer gpa.free(logit_rows);

    // First token from prefill logits.
    var n_active: usize = 0;
    for (0..b) |i| {
        const t = argmax(logits[i * c.vocab_size ..][0..c.vocab_size]);
        try outputs[i].append(gpa, t);
        if (model.isEos(t) or max_new_tokens <= 1 or positions[i] >= cache.max_len) active[i] = false else n_active += 1;
    }
    // Prefill misses are not decode misses.
    if (model.expert_cache) |ec| ec.resetStep();
    var step: usize = 1;
    while (step < max_new_tokens and n_active > 0) : (step += 1) {
        var n: usize = 0;
        for (0..b) |i| {
            if (!active[i]) continue;
            tokens[n] = outputs[i].items[outputs[i].items.len - 1];
            rows[n] = .{ .b = i, .pos = positions[i] };
            logit_rows[n] = n;
            n += 1;
        }
        try forward(model, ws, cache, tokens[0..n], rows[0..n], .{ .logit_rows = logit_rows[0..n] });
        if (model.expert_cache) |ec| ec.endDecodeStep();
        var j: usize = 0;
        for (0..b) |i| {
            if (!active[i]) continue;
            const t = argmax(ws.logits[j * c.vocab_size ..][0..c.vocab_size]);
            j += 1;
            positions[i] += 1;
            try outputs[i].append(gpa, t);
            if (model.isEos(t) or positions[i] >= cache.max_len) {
                active[i] = false;
                n_active -= 1;
            }
        }
    }
    const result = try gpa.alloc([]u32, b);
    for (outputs, 0..) |*o, i| result[i] = try o.toOwnedSlice(gpa);
    return result;
}

/// Expert-selective abliteration entry point (see `moe.applyExpertSelective`).
pub fn applyExpertSelective(model: *Model, dirs: []const f32, cfg: search.TrialConfig, opts: abliterate.Options) !void {
    return moe.applyExpertSelective(model, dirs, cfg, opts);
}
