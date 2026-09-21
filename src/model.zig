//! Transformer model loading and inference for Llama-family models
//! (Llama 2/3, Mistral, Qwen 2/2.5/3, Gemma 2/3 text) and their
//! mixture-of-experts variants (Mixtral, Qwen2-MoE, Qwen3-MoE; see moe.zig).
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
const Tokenizer = @import("tokenizer.zig").Tokenizer;
const moe = @import("moe.zig");
const abliterate = @import("abliterate.zig");
const search = @import("search.zig");
const expert_cache = @import("expert_cache.zig");
const remote = @import("remote.zig");

const Allocator = std.mem.Allocator;
const Weight = tensor.Weight;
const Delta = tensor.Delta;
const WeightRef = stream.WeightRef;

pub const Family = enum {
    llama,
    mistral,
    qwen2,
    qwen3,
    gemma2,
    gemma3,
    qwen2_moe,
    qwen3_moe,
    mixtral,

    pub fn isGemma(self: Family) bool {
        return self == .gemma2 or self == .gemma3;
    }
};

pub const RopeScaling = union(enum) {
    none,
    linear: f32,
    llama3: struct { factor: f32, low_freq_factor: f32, high_freq_factor: f32, original_max_position: f32 },
};

pub const Config = struct {
    family: Family,
    model_type: []const u8,
    hidden_size: usize,
    intermediate_size: usize,
    num_layers: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    vocab_size: usize,
    rms_norm_eps: f32,
    rope_theta: f32,
    rope_local_theta: f32,
    rope_scaling: RopeScaling,
    tie_word_embeddings: bool,
    activation: tensor.Activation,
    max_position_embeddings: usize,
    sliding_window: ?usize,
    /// Per layer: true if the layer uses sliding-window (local) attention.
    sliding_layers: []bool,
    attention_scale: f32,
    attn_logit_softcapping: ?f32,
    final_logit_softcapping: ?f32,
    attention_bias: bool,
    embed_scale: f32,
    /// Mixture-of-experts settings (num_experts == 0 for dense models).
    num_experts: usize,
    num_experts_per_tok: usize,
    norm_topk_prob: bool,
    moe_intermediate_size: usize,
    /// Per layer: true if the layer's MLP is a routed mixture of experts.
    moe_layers: []bool,
};

fn getNum(obj: std.json.ObjectMap, key: []const u8) ?f64 {
    const v = obj.get(key) orelse return null;
    return switch (v) {
        .integer => |i| @floatFromInt(i),
        .float => |f| f,
        else => null,
    };
}

fn getInt(obj: std.json.ObjectMap, key: []const u8, default: usize) usize {
    const v = getNum(obj, key) orelse return default;
    return @intFromFloat(v);
}

fn getF32(obj: std.json.ObjectMap, key: []const u8, default: f32) f32 {
    const v = getNum(obj, key) orelse return default;
    return @floatCast(v);
}

fn getBool(obj: std.json.ObjectMap, key: []const u8, default: bool) bool {
    const v = obj.get(key) orelse return default;
    return if (v == .bool) v.bool else default;
}

pub fn parseConfig(arena: Allocator, json_text: []const u8) !Config {
    var parsed = try std.json.parseFromSlice(std.json.Value, arena, json_text, .{});
    defer parsed.deinit();
    var obj = parsed.value.object;
    var model_type = if (obj.get("model_type")) |m| m.string else "llama";
    // Multimodal wrappers keep the text config nested.
    if (obj.get("text_config")) |tc| {
        if (tc == .object) {
            obj = tc.object;
            if (obj.get("model_type")) |m| model_type = m.string;
        }
    }
    const family: Family = if (std.mem.eql(u8, model_type, "llama"))
        .llama
    else if (std.mem.eql(u8, model_type, "mistral"))
        .mistral
    else if (std.mem.eql(u8, model_type, "qwen2"))
        .qwen2
    else if (std.mem.eql(u8, model_type, "qwen3"))
        .qwen3
    else if (std.mem.eql(u8, model_type, "gemma2"))
        .gemma2
    else if (std.mem.eql(u8, model_type, "gemma3") or std.mem.eql(u8, model_type, "gemma3_text"))
        .gemma3
    else if (std.mem.eql(u8, model_type, "qwen2_moe"))
        .qwen2_moe
    else if (std.mem.eql(u8, model_type, "qwen3_moe"))
        .qwen3_moe
    else if (std.mem.eql(u8, model_type, "mixtral"))
        .mixtral
    else {
        std.log.err("unsupported model_type: {s}", .{model_type});
        return error.UnsupportedArchitecture;
    };

    const hidden = getInt(obj, "hidden_size", 0);
    const heads = getInt(obj, "num_attention_heads", 0);
    const kv_heads = getInt(obj, "num_key_value_heads", heads);
    const head_dim = getInt(obj, "head_dim", if (heads > 0) hidden / heads else 0);
    const layers = getInt(obj, "num_hidden_layers", 0);
    if (hidden == 0 or heads == 0 or layers == 0) return error.InvalidConfig;

    var act: tensor.Activation = .silu;
    const act_name = if (obj.get("hidden_activation")) |a| (if (a == .string) a.string else "") else if (obj.get("hidden_act")) |a| (if (a == .string) a.string else "") else "";
    if (std.mem.eql(u8, act_name, "gelu_pytorch_tanh") or std.mem.eql(u8, act_name, "gelu_tanh")) act = .gelu_tanh else if (std.mem.eql(u8, act_name, "gelu")) act = .gelu else act = .silu;
    if (family.isGemma() and act_name.len == 0) act = .gelu_tanh;

    var rope_scaling: RopeScaling = .none;
    if (obj.get("rope_scaling")) |rs| {
        if (rs == .object) {
            const t = if (rs.object.get("rope_type")) |t| t.string else if (rs.object.get("type")) |t| t.string else "";
            if (std.mem.eql(u8, t, "llama3")) {
                rope_scaling = .{ .llama3 = .{
                    .factor = getF32(rs.object, "factor", 8),
                    .low_freq_factor = getF32(rs.object, "low_freq_factor", 1),
                    .high_freq_factor = getF32(rs.object, "high_freq_factor", 4),
                    .original_max_position = getF32(rs.object, "original_max_position_embeddings", 8192),
                } };
            } else if (std.mem.eql(u8, t, "linear")) {
                rope_scaling = .{ .linear = getF32(rs.object, "factor", 1) };
            } else if (t.len > 0 and !std.mem.eql(u8, t, "default")) {
                std.log.warn("rope scaling type '{s}' is not supported; using unscaled RoPE", .{t});
            }
        }
    }

    const sliding_window: ?usize = blk: {
        const v = obj.get("sliding_window") orelse break :blk null;
        break :blk switch (v) {
            .integer => |i| @intCast(i),
            else => null,
        };
    };

    const sliding_layers = try arena.alloc(bool, layers);
    @memset(sliding_layers, false);
    if (obj.get("layer_types")) |lt| {
        if (lt == .array) {
            for (lt.array.items, 0..) |v, i| {
                if (i < layers and v == .string) sliding_layers[i] = std.mem.eql(u8, v.string, "sliding_attention");
            }
        }
    } else if (family == .gemma2) {
        for (sliding_layers, 0..) |*s, i| s.* = (i % 2 == 0);
    } else if (family == .gemma3) {
        const pattern = getInt(obj, "sliding_window_pattern", 6);
        for (sliding_layers, 0..) |*s, i| s.* = ((i + 1) % pattern != 0);
    } else if (sliding_window != null and family == .mistral) {
        @memset(sliding_layers, true);
    }

    // Mixture of experts: `num_experts` (Qwen) or `num_local_experts` (Mixtral);
    // Qwen additionally allows dense layers via `mlp_only_layers` / `decoder_sparse_step`.
    const num_experts = getInt(obj, "num_experts", getInt(obj, "num_local_experts", 0));
    const moe_layers = try arena.alloc(bool, layers);
    @memset(moe_layers, false);
    if (num_experts > 0) {
        const sparse_step = @max(getInt(obj, "decoder_sparse_step", 1), 1);
        for (moe_layers, 0..) |*m, i| m.* = ((i + 1) % sparse_step == 0);
        if (obj.get("mlp_only_layers")) |ml| {
            if (ml == .array) for (ml.array.items) |v| {
                if (v == .integer and v.integer >= 0 and v.integer < layers) moe_layers[@intCast(v.integer)] = false;
            };
        }
    }
    const intermediate_size = getInt(obj, "intermediate_size", 4 * hidden);

    const query_pre_attn_scalar = getNum(obj, "query_pre_attn_scalar");
    const attention_scale: f32 = if (query_pre_attn_scalar) |q| @floatCast(1.0 / @sqrt(q)) else 1.0 / @sqrt(@as(f32, @floatFromInt(head_dim)));

    return .{
        .family = family,
        .model_type = try arena.dupe(u8, model_type),
        .hidden_size = hidden,
        .intermediate_size = intermediate_size,
        .num_layers = layers,
        .num_heads = heads,
        .num_kv_heads = kv_heads,
        .head_dim = head_dim,
        .vocab_size = getInt(obj, "vocab_size", 0),
        .rms_norm_eps = getF32(obj, "rms_norm_eps", 1e-6),
        .rope_theta = getF32(obj, "rope_theta", 10000.0),
        .rope_local_theta = getF32(obj, "rope_local_base_freq", 10000.0),
        .rope_scaling = rope_scaling,
        .tie_word_embeddings = getBool(obj, "tie_word_embeddings", family.isGemma()),
        .activation = act,
        .max_position_embeddings = getInt(obj, "max_position_embeddings", 4096),
        .sliding_window = sliding_window,
        .sliding_layers = sliding_layers,
        .attention_scale = attention_scale,
        .attn_logit_softcapping = if (getNum(obj, "attn_logit_softcapping")) |v| @as(f32, @floatCast(v)) else null,
        .final_logit_softcapping = if (getNum(obj, "final_logit_softcapping")) |v| @as(f32, @floatCast(v)) else null,
        .attention_bias = getBool(obj, "attention_bias", family == .qwen2 or family == .qwen2_moe),
        .embed_scale = if (family.isGemma()) @sqrt(@as(f32, @floatFromInt(hidden))) else 1.0,
        .num_experts = num_experts,
        .num_experts_per_tok = getInt(obj, "num_experts_per_tok", 2),
        .norm_topk_prob = getBool(obj, "norm_topk_prob", family == .mixtral),
        .moe_intermediate_size = getInt(obj, "moe_intermediate_size", intermediate_size),
        .moe_layers = moe_layers,
    };
}

// ---------------------------------------------------------------------------
// Weights
// ---------------------------------------------------------------------------

/// Where a layer's matrices live on disk. `Model.acquireLayer` turns these
/// into resident `Weight` views for the duration of one layer's compute.
/// Dense layers carry `gate`/`up`/`down`; mixture-of-experts layers carry the
/// `router` instead (the expert matrices are described by `Layer.moe`).
pub const LayerRefs = struct {
    q: WeightRef,
    k: WeightRef,
    v: WeightRef,
    o: WeightRef,
    gate: ?WeightRef = null,
    up: ?WeightRef = null,
    down: ?WeightRef = null,
    router: ?WeightRef = null,

    pub const max = 8;

    /// The refs in a fixed order (q, k, v, o, then gate, up, down or router).
    pub fn all(self: *const LayerRefs, buf: *[max]WeightRef) []WeightRef {
        buf[0] = self.q;
        buf[1] = self.k;
        buf[2] = self.v;
        buf[3] = self.o;
        var n: usize = 4;
        inline for (.{ self.gate, self.up, self.down, self.router }) |opt| {
            if (opt) |r| {
                buf[n] = r;
                n += 1;
            }
        }
        return buf[0..n];
    }

    /// Bytes of the attention / dense-MLP / router matrices (expert matrices excluded).
    pub fn bytes(self: *const LayerRefs) u64 {
        var buf: [max]WeightRef = undefined;
        var total: u64 = 0;
        for (self.all(&buf)) |r| total += r.byteLen();
        return total;
    }
};

pub const Layer = struct {
    input_norm: []f32,
    post_attn_norm: []f32,
    pre_ff_norm: ?[]f32, // gemma
    post_ff_norm: ?[]f32, // gemma
    q_norm: ?[]f32, // qwen3
    k_norm: ?[]f32,
    q: Weight,
    k: Weight,
    v: Weight,
    o: Weight,
    q_bias: ?[]f32,
    k_bias: ?[]f32,
    v_bias: ?[]f32,
    /// Dense MLP (null for mixture-of-experts layers).
    ///
    /// In mapped mode the `Weight` fields of a layer view the mapping and are
    /// always valid. In streamed mode they carry only the shape (empty data)
    /// and the resident views live in the `Layer` copy returned by
    /// `Model.acquireLayer`.
    gate: ?Weight,
    up: ?Weight,
    down: ?Weight,
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
};

/// The two abliterable components, named as in heretic.
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
    /// Weight access (mapped or streamed) over `files`. The forward pass takes
    /// `*const Model` but acquiring weights mutates the store (buffer pool,
    /// prefetch state); since a `Model` always lives on the heap this is done
    /// through `@constCast` in `acquireLayer` & co.
    store: stream.WeightStore,
    /// Warp mode: the bounded cache of resident routed experts (streamed MoE models).
    expert_cache: ?*expert_cache.ExpertCache = null,
    budget: ?*budget_mod.Budget,
    scratch_dir: []const u8,
    /// Tensor name prefix for the language model (e.g. "model." or "language_model.model.").
    prefix: []const u8,
    /// Shape-valid always; data valid only in mapped mode (use `embedRow` / `acquireLmHead`).
    embed: Weight,
    lm_head: Weight,
    embed_ref: WeightRef,
    lm_head_ref: WeightRef,
    largest_layer_bytes: u64,
    /// Largest layer without its routed experts (attention, norms, router, shared expert).
    largest_trunk_layer_bytes: u64,
    /// Largest routed expert (0 for dense models).
    largest_expert_bytes: u64,
    /// Bytes of all routed experts of all layers.
    total_expert_bytes: u64,
    largest_tensor_bytes: u64,
    spill_always: bool,
    final_norm: []f32,
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
    rope_cos: []f32, // [max_pos][head_dim/2]
    rope_sin: []f32,
    rope_cos_local: []f32,
    rope_sin_local: []f32,
    rope_len: usize,

    pub fn deinit(self: *Model) void {
        self.resetDeltas();
        if (self.expert_cache) |c| {
            c.deinit();
            self.meta_gpa.destroy(c);
            self.expert_cache = null;
        }
        self.store.deinit();
        for (self.files) |f| f.close(self.meta_gpa, self.io);
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

        var dir = try Io.Dir.cwd().openDir(io, dir_path, .{ .iterate = true });
        defer dir.close(io);
        if (dir.access(io, export_incomplete_marker, .{})) |_| {
            std.log.warn("{s} contains {s}: the export did not finish; refusing to load it", .{ dir_path, export_incomplete_marker });
            return error.IncompleteModel;
        } else |_| {}

        self.source_dir = try arena.dupe(u8, dir_path);
        self.scratch_dir = try arena.dupe(u8, opts.scratch_dir orelse (if (opts.budget) |b| b.scratch_dir else "scratch"));
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
        const store_mode: stream.Mode = if (opts.remote != null) .streamed else opts.store;
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
            for (names.items) |n| try files.append(arena, try safetensors.File.openOptions(gpa, io, dir, n, .{ .map = store_mode == .mapped }));
        }
        self.files = files.items;
        self.store = stream.WeightStore.init(self.gpa, io, self.files, store_mode, .{ .budget = opts.budget, .prefetch = opts.prefetch });
        self.store.registerReclaim();
        errdefer self.store.deinit();

        // Detect prefix.
        const prefixes = [_][]const u8{ "model.", "language_model.model.", "model.language_model.", "" };
        self.prefix = "";
        var found_prefix = false;
        for (prefixes) |p| {
            const key = try std.fmt.allocPrint(arena, "{s}embed_tokens.weight", .{p});
            if (self.find(key) != null) {
                self.prefix = p;
                found_prefix = true;
                break;
            }
        }
        if (!found_prefix) return error.MissingWeights;

        const embed_name = try std.fmt.allocPrint(arena, "{s}embed_tokens.weight", .{self.prefix});
        self.embed_ref = self.store.lookup(embed_name).?;
        self.embed = try self.loadMat(embed_name);
        self.dtype = self.embed_ref.dtype;
        if (self.config.vocab_size == 0) self.config.vocab_size = self.embed.rows;
        self.final_norm = try self.loadVec(try std.fmt.allocPrint(arena, "{s}norm.weight", .{self.prefix}));
        if (self.store.lookup("lm_head.weight")) |lm| {
            self.lm_head_ref = lm;
        } else if (self.store.lookup(try std.fmt.allocPrint(arena, "{s}lm_head.weight", .{std.mem.trimEnd(u8, self.prefix, "model.")}))) |lm| {
            self.lm_head_ref = lm;
        } else {
            self.lm_head_ref = self.embed_ref;
        }
        self.lm_head = try self.loadMat(self.lm_head_ref.name);

        self.largest_tensor_bytes = 0;
        for (self.files) |f| {
            var it = f.tensors.iterator();
            while (it.next()) |kv| self.largest_tensor_bytes = @max(self.largest_tensor_bytes, kv.value_ptr.byte_len);
        }

        const c = &self.config;
        self.layers = try arena.alloc(Layer, c.num_layers);
        for (self.layers, 0..) |*layer, i| {
            const lp = try std.fmt.allocPrint(arena, "{s}layers.{d}.", .{ self.prefix, i });
            layer.* = .{
                .input_norm = try self.loadVec(try cat(arena, lp, "input_layernorm.weight")),
                .post_attn_norm = try self.loadVec(try cat(arena, lp, "post_attention_layernorm.weight")),
                .pre_ff_norm = self.loadVecOpt(try cat(arena, lp, "pre_feedforward_layernorm.weight")),
                .post_ff_norm = self.loadVecOpt(try cat(arena, lp, "post_feedforward_layernorm.weight")),
                .q_norm = self.loadVecOpt(try cat(arena, lp, "self_attn.q_norm.weight")),
                .k_norm = self.loadVecOpt(try cat(arena, lp, "self_attn.k_norm.weight")),
                .q = try self.loadMat(try cat(arena, lp, "self_attn.q_proj.weight")),
                .k = try self.loadMat(try cat(arena, lp, "self_attn.k_proj.weight")),
                .v = try self.loadMat(try cat(arena, lp, "self_attn.v_proj.weight")),
                .o = try self.loadMat(try cat(arena, lp, "self_attn.o_proj.weight")),
                .q_bias = self.loadVecOpt(try cat(arena, lp, "self_attn.q_proj.bias")),
                .k_bias = self.loadVecOpt(try cat(arena, lp, "self_attn.k_proj.bias")),
                .v_bias = self.loadVecOpt(try cat(arena, lp, "self_attn.v_proj.bias")),
                .gate = null,
                .up = null,
                .down = null,
                .refs = .{
                    .q = try self.ref(try cat(arena, lp, "self_attn.q_proj.weight")),
                    .k = try self.ref(try cat(arena, lp, "self_attn.k_proj.weight")),
                    .v = try self.ref(try cat(arena, lp, "self_attn.v_proj.weight")),
                    .o = try self.ref(try cat(arena, lp, "self_attn.o_proj.weight")),
                },
            };
            if (c.moe_layers[i]) {
                layer.moe = try moe.loadLayer(self, arena, i, lp);
                layer.refs.router = layer.moe.?.router_ref;
            } else {
                layer.gate = try self.loadMat(try cat(arena, lp, "mlp.gate_proj.weight"));
                layer.up = try self.loadMat(try cat(arena, lp, "mlp.up_proj.weight"));
                layer.down = try self.loadMat(try cat(arena, lp, "mlp.down_proj.weight"));
                layer.refs.gate = try self.ref(try cat(arena, lp, "mlp.gate_proj.weight"));
                layer.refs.up = try self.ref(try cat(arena, lp, "mlp.up_proj.weight"));
                layer.refs.down = try self.ref(try cat(arena, lp, "mlp.down_proj.weight"));
            }
        }
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
    pub fn ref(self: *const Model, name: []const u8) !WeightRef {
        return self.store.lookup(name) orelse {
            std.log.err("missing tensor: {s}", .{name});
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
        var buf: [LayerRefs.max]WeightRef = undefined;
        const refs = self.layers[li].refs.all(&buf);
        try store.acquireSet(li, refs, lease.leases[0..refs.len]);
        lease.n_leases = refs.len;
        errdefer store.releaseSet(lease.leases[0..lease.n_leases]);
        const l = &lease.layer;
        l.q = lease.leases[0].weight;
        l.k = lease.leases[1].weight;
        l.v = lease.leases[2].weight;
        l.o = lease.leases[3].weight;
        if (l.moe) |*m| {
            m.router = lease.leases[4].weight;
            lease.experts = try moe.acquireLayer(self, m);
            l.moe = lease.experts.?.layer;
        } else {
            l.gate = lease.leases[4].weight;
            l.up = lease.leases[5].weight;
            l.down = lease.leases[6].weight;
        }
        if (li + 1 < self.layers.len) {
            var next_buf: [LayerRefs.max]WeightRef = undefined;
            store.prefetch(li + 1, self.layers[li + 1].refs.all(&next_buf));
        }
        return lease;
    }

    /// Makes one abliterable matrix resident; release with `self.store.release`.
    /// For `.mlp_down_proj` on a mixture-of-experts layer use `acquireExpertDown`.
    pub fn acquireComponent(self: *const Model, layer: usize, comp: Component) !stream.Lease {
        const store: *stream.WeightStore = @constCast(&self.store);
        return store.acquire(switch (comp) {
            .attn_o_proj => self.layers[layer].refs.o,
            .mlp_down_proj => self.layers[layer].refs.down.?,
        });
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

    fn cat(arena: Allocator, a: []const u8, b: []const u8) ![]const u8 {
        return std.fmt.allocPrint(arena, "{s}{s}", .{ a, b });
    }

    pub fn find(self: *const Model, name: []const u8) ?safetensors.TensorInfo {
        for (self.files) |f| {
            if (f.get(name)) |t| return t;
        }
        return null;
    }

    /// Matrix view of a named tensor: the mapping in mapped mode, the shape
    /// only (empty data) in streamed mode.
    pub fn loadMat(self: *Model, name: []const u8) !Weight {
        const r = try self.ref(name);
        return switch (self.store.mode) {
            .mapped => self.find(name).?.asWeight(),
            .streamed => r.shapeOnly(),
        };
    }

    fn loadVec(self: *Model, name: []const u8) ![]f32 {
        return self.loadVecOpt(name) orelse {
            std.log.err("missing tensor: {s}", .{name});
            return error.MissingWeights;
        };
    }

    /// Reads a small tensor as an f32 vector (null if absent).
    pub fn loadVecOpt(self: *Model, name: []const u8) ?[]f32 {
        const r = self.store.lookup(name) orelse return null;
        return self.store.readVecF32(self.arena.allocator(), r) catch null;
    }

    fn buildRope(self: *Model) !void {
        const c = &self.config;
        const arena = self.arena.allocator();
        const half = c.head_dim / 2;
        self.rope_len = @min(c.max_position_embeddings, 8192);
        self.rope_cos = try arena.alloc(f32, self.rope_len * half);
        self.rope_sin = try arena.alloc(f32, self.rope_len * half);
        try self.fillRope(self.rope_cos, self.rope_sin, c.rope_theta, c.rope_scaling);
        if (c.family == .gemma3) {
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
        const half = c.head_dim / 2;
        const inv_freq = try self.gpa.alloc(f64, half);
        defer self.gpa.free(inv_freq);
        for (inv_freq, 0..) |*f, i| {
            const exponent: f64 = @as(f64, @floatFromInt(2 * i)) / @as(f64, @floatFromInt(c.head_dim));
            f.* = 1.0 / std.math.pow(f64, theta, exponent);
        }
        switch (scaling) {
            .none => {},
            .linear => |factor| for (inv_freq) |*f| {
                f.* /= factor;
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
        }
        var pos: usize = 0;
        while (pos < self.rope_len) : (pos += 1) {
            for (inv_freq, 0..) |f, i| {
                const angle = @as(f64, @floatFromInt(pos)) * f;
                cos[pos * half + i] = @floatCast(@cos(angle));
                sin[pos * half + i] = @floatCast(@sin(angle));
            }
        }
    }

    pub fn isEos(self: *const Model, id: u32) bool {
        for (self.eos_ids) |e| if (e == id) return true;
        return false;
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

/// Layer index encoded in a tensor name (`<prefix>layers.<n>.…`), if any.
pub fn layerIndexOf(prefix: []const u8, name: []const u8) ?usize {
    if (!std.mem.startsWith(u8, name, prefix)) return null;
    const rest = name[prefix.len..];
    if (!std.mem.startsWith(u8, rest, "layers.")) return null;
    const after = rest["layers.".len..];
    const dot = std.mem.indexOfScalar(u8, after, '.') orelse return null;
    return std.fmt.parseInt(usize, after[0..dot], 10) catch null;
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
        if (model.spill_always) return initScratch(gpa, model.io, model.scratch_dir, model.budget, c.num_layers, batch, max_len, kvd);
        if (model.budget) |b| {
            const need = bytesFor(c.num_layers, batch, max_len, kvd) + model.residentWeightNeed();
            if (b.limited() and b.available() + model.expertCacheEvictable() < need) {
                return initScratch(gpa, model.io, model.scratch_dir, b, c.num_layers, batch, max_len, kvd);
            }
        }
        return init(gpa, c.num_layers, batch, max_len, kvd) catch |err| switch (err) {
            error.OutOfMemory => if (model.streamed()) initScratch(gpa, model.io, model.scratch_dir, model.budget, c.num_layers, batch, max_len, kvd) else err,
        };
    }

    pub fn deinit(self: *KvCache) void {
        self.gpa.free(self.k);
        self.gpa.free(self.v);
        if (self.scratch) |*s| s.deinit();
        if (self.layer_written.len > 0) self.gpa.free(self.layer_written);
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
    out: []f32, // [n][heads*head_dim]
    sliding: bool,
};

fn attentionWorker(ctx: *const AttnCtx, start: usize, end: usize) void {
    const c = &ctx.model.config;
    const hd = c.head_dim;
    const groups = c.num_heads / c.num_kv_heads;
    var scores_buf: [8192]f32 = undefined;
    var task = start;
    while (task < end) : (task += 1) {
        const r = task / c.num_heads;
        const h = task % c.num_heads;
        const row = ctx.rows[r];
        const kvh = h / groups;
        const q = ctx.q[r * c.num_heads * hd + h * hd ..][0..hd];
        const out = ctx.out[r * c.num_heads * hd + h * hd ..][0..hd];
        var lo: usize = 0;
        if (ctx.sliding) {
            if (c.sliding_window) |w| {
                if (row.pos + 1 > w) lo = row.pos + 1 - w;
            }
        }
        const n_keys = row.pos + 1 - lo;
        const scores = scores_buf[0..n_keys];
        var p: usize = 0;
        while (p < n_keys) : (p += 1) {
            const k = ctx.cache.kSlot(ctx.layer, row.b, lo + p)[kvh * hd ..][0..hd];
            var s = tensor.dot(q, k) * c.attention_scale;
            if (c.attn_logit_softcapping) |cap| s = cap * std.math.tanh(s / cap);
            scores[p] = s;
        }
        tensor.softmaxInPlace(scores);
        @memset(out, 0);
        p = 0;
        while (p < n_keys) : (p += 1) {
            const v = ctx.cache.vSlot(ctx.layer, row.b, lo + p)[kvh * hd ..][0..hd];
            tensor.axpy(out, scores[p], v);
        }
    }
}

/// Workspace for forward passes, sized for `max_rows` tokens per call.
pub const Workspace = struct {
    gpa: Allocator,
    x: []f32,
    h: []f32,
    q: []f32,
    k: []f32,
    v: []f32,
    attn: []f32,
    o: []f32,
    gate: []f32,
    up: []f32,
    logits: []f32,
    max_rows: usize,
    max_logit_rows: usize,

    pub fn init(gpa: Allocator, c: *const Config, max_rows: usize, max_logit_rows: usize) !Workspace {
        const hidden = c.hidden_size;
        const qd = c.num_heads * c.head_dim;
        const kvd = c.num_kv_heads * c.head_dim;
        return .{
            .gpa = gpa,
            .x = try gpa.alloc(f32, max_rows * hidden),
            .h = try gpa.alloc(f32, max_rows * hidden),
            .q = try gpa.alloc(f32, max_rows * qd),
            .k = try gpa.alloc(f32, max_rows * kvd),
            .v = try gpa.alloc(f32, max_rows * kvd),
            .attn = try gpa.alloc(f32, max_rows * qd),
            .o = try gpa.alloc(f32, max_rows * hidden),
            .gate = try gpa.alloc(f32, max_rows * c.intermediate_size),
            .up = try gpa.alloc(f32, max_rows * c.intermediate_size),
            .logits = try gpa.alloc(f32, max_logit_rows * c.vocab_size),
            .max_rows = max_rows,
            .max_logit_rows = max_logit_rows,
        };
    }

    pub fn deinit(self: *Workspace) void {
        self.gpa.free(self.x);
        self.gpa.free(self.h);
        self.gpa.free(self.q);
        self.gpa.free(self.k);
        self.gpa.free(self.v);
        self.gpa.free(self.attn);
        self.gpa.free(self.o);
        self.gpa.free(self.gate);
        self.gpa.free(self.up);
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
            try model.embedRow(t, xs[i * hidden ..][0..hidden]);
            if (c.embed_scale != 1.0) tensor.scale(xs[i * hidden ..][0..hidden], c.embed_scale);
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
                tensor.rmsnorm(dst, ws.x[r * hidden ..][0..hidden], model.final_norm, c.rms_norm_eps, c.family.isGemma());
            } else {
                const tmp = ws.o[0..hidden];
                try act.?.readRow(r, tmp);
                tensor.rmsnorm(dst, tmp, model.final_norm, c.rms_norm_eps, c.family.isGemma());
            }
        }
        const lm = try model.acquireLmHead();
        defer @constCast(&model.store).release(lm);
        try tensor.matmulT(model.pool, gpa, ws.logits, h, opts.logit_rows.len, lm.weight, null);
        if (c.final_logit_softcapping) |cap| tensor.softcap(ws.logits[0 .. opts.logit_rows.len * c.vocab_size], cap);
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
fn layerBlock(model: *const Model, layer: *const Layer, li: usize, ws: *Workspace, cache: *KvCache, x: []f32, rows: []const Row) !void {
    const c = &model.config;
    const gpa = model.gpa;
    const n = rows.len;
    const hidden = c.hidden_size;
    const hd = c.head_dim;
    const qd = c.num_heads * hd;
    const kvd = c.num_kv_heads * hd;
    const half = hd / 2;
    const h = ws.h[0 .. n * hidden];

    // Attention block.
    var i: usize = 0;
    while (i < n) : (i += 1) tensor.rmsnorm(h[i * hidden ..][0..hidden], x[i * hidden ..][0..hidden], layer.input_norm, c.rms_norm_eps, c.family.isGemma());
    try tensor.matmulT(model.pool, gpa, ws.q, h, n, layer.q, null);
    try tensor.matmulT(model.pool, gpa, ws.k, h, n, layer.k, null);
    try tensor.matmulT(model.pool, gpa, ws.v, h, n, layer.v, null);
    if (layer.q_bias) |b| {
        i = 0;
        while (i < n) : (i += 1) tensor.axpy(ws.q[i * qd ..][0..qd], 1.0, b);
    }
    if (layer.k_bias) |b| {
        i = 0;
        while (i < n) : (i += 1) tensor.axpy(ws.k[i * kvd ..][0..kvd], 1.0, b);
    }
    if (layer.v_bias) |b| {
        i = 0;
        while (i < n) : (i += 1) tensor.axpy(ws.v[i * kvd ..][0..kvd], 1.0, b);
    }
    const sliding = c.sliding_layers[li];
    const cos = if (sliding and c.family == .gemma3) model.rope_cos_local else model.rope_cos;
    const sin = if (sliding and c.family == .gemma3) model.rope_sin_local else model.rope_sin;
    i = 0;
    while (i < n) : (i += 1) {
        const pos = @min(rows[i].pos, model.rope_len - 1);
        const cr = cos[pos * half ..][0..half];
        const sr = sin[pos * half ..][0..half];
        var hh: usize = 0;
        while (hh < c.num_heads) : (hh += 1) {
            const q = ws.q[i * qd + hh * hd ..][0..hd];
            if (layer.q_norm) |qn| {
                var tmp: [512]f32 = undefined;
                tensor.rmsnorm(tmp[0..hd], q, qn, c.rms_norm_eps, false);
                @memcpy(q, tmp[0..hd]);
            }
            tensor.applyRope(q, cr, sr);
        }
        hh = 0;
        while (hh < c.num_kv_heads) : (hh += 1) {
            const k = ws.k[i * kvd + hh * hd ..][0..hd];
            if (layer.k_norm) |kn| {
                var tmp: [512]f32 = undefined;
                tensor.rmsnorm(tmp[0..hd], k, kn, c.rms_norm_eps, false);
                @memcpy(k, tmp[0..hd]);
            }
            tensor.applyRope(k, cr, sr);
        }
        @memcpy(cache.kSlot(li, rows[i].b, rows[i].pos), ws.k[i * kvd ..][0..kvd]);
        @memcpy(cache.vSlot(li, rows[i].b, rows[i].pos), ws.v[i * kvd ..][0..kvd]);
        cache.noteWrite(rows[i].pos);
    }
    const actx = AttnCtx{ .model = model, .cache = cache, .layer = li, .rows = rows, .q = ws.q, .out = ws.attn, .sliding = sliding };
    model.pool.parallelFor(n * c.num_heads, &actx, attentionWorker);
    try tensor.matmulT(model.pool, gpa, ws.o, ws.attn, n, layer.o, if (layer.o_delta) |*d| d else null);
    i = 0;
    while (i < n) : (i += 1) {
        const o = ws.o[i * hidden ..][0..hidden];
        if (c.family.isGemma()) {
            const tmp = h[i * hidden ..][0..hidden];
            tensor.rmsnorm(tmp, o, layer.post_attn_norm, c.rms_norm_eps, true);
            tensor.axpy(x[i * hidden ..][0..hidden], 1.0, tmp);
        } else {
            tensor.axpy(x[i * hidden ..][0..hidden], 1.0, o);
        }
    }

    // MLP block.
    const ff_norm = if (c.family.isGemma()) layer.pre_ff_norm.? else layer.post_attn_norm;
    i = 0;
    while (i < n) : (i += 1) tensor.rmsnorm(h[i * hidden ..][0..hidden], x[i * hidden ..][0..hidden], ff_norm, c.rms_norm_eps, c.family.isGemma());
    if (layer.moe) |*m| {
        try moe.forward(model, m, li, ws.o, h, n);
    } else {
        try tensor.matmulT(model.pool, gpa, ws.gate, h, n, layer.gate.?, null);
        try tensor.matmulT(model.pool, gpa, ws.up, h, n, layer.up.?, null);
        const inter = c.intermediate_size;
        for (ws.gate[0 .. n * inter], 0..) |*g, j| g.* = c.activation.apply(g.*) * ws.up[j];
        try tensor.matmulT(model.pool, gpa, ws.o, ws.gate, n, layer.down.?, if (layer.down_delta) |*d| d else null);
    }
    i = 0;
    while (i < n) : (i += 1) {
        const o = ws.o[i * hidden ..][0..hidden];
        if (c.family.isGemma()) {
            const tmp = h[i * hidden ..][0..hidden];
            tensor.rmsnorm(tmp, o, layer.post_ff_norm.?, c.rms_norm_eps, true);
            tensor.axpy(x[i * hidden ..][0..hidden], 1.0, tmp);
        } else {
            tensor.axpy(x[i * hidden ..][0..hidden], 1.0, o);
        }
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
