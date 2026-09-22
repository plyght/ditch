//! Architecture registry: one descriptor per Hugging Face `model_type`
//! capturing everything the loader and the forward pass need to know about a
//! family (tensor-name templates, norm and residual layout, attention and MLP
//! layouts, positional encoding, MoE routing) plus the generalised `Config`
//! and its parser. Adding a family is a table entry here and, only when the
//! family needs a genuinely new computation, a small code path in model.zig
//! or moe.zig.
//!
//! Every entry documents what was verified against a NumPy reference fixture
//! (`tools/make_fixture.py`, `src/model_test.zig`); entries marked
//! `verified = false` are implemented from the Hugging Face reference
//! implementation but have no fixture.

const std = @import("std");
const tensor = @import("tensor.zig");

const Allocator = std.mem.Allocator;

/// Layer normalisation flavour.
pub const NormKind = enum {
    /// `x / rms(x) * w`
    rms,
    /// `x / rms(x) * (1 + w)` (Gemma).
    rms_gemma,
    /// `(x - mean) / std * w + b` (bias optional).
    layer,
    /// LayerNorm with `(1 + w)` scaling (Nemotron).
    layer_1p,
    /// Non-parametric LayerNorm (OLMo 1).
    none,
};

/// Which pairs of coordinates a rotary embedding rotates.
pub const RopeStyle = enum {
    /// `(x[i], x[i + d/2])` (Hugging Face `rotate_half`).
    neox,
    /// `(x[2i], x[2i + 1])` (GPT-J, GLM, Llama 4, DeepSeek).
    gptj,
};

pub const Positional = enum { rope, learned, alibi, none };

/// Row layout of a fused query/key/value projection `[rows][hidden]`.
pub const QkvLayout = enum {
    separate,
    /// `[q (all heads) | k (all kv heads) | v (all kv heads)]`.
    concat,
    /// `[head][q | k | v]` (GPT-NeoX, BLOOM, Falcon multi-head).
    heads_interleaved,
    /// `[kv group][q heads of the group | k | v]` (InternLM2, Falcon new decoder).
    grouped,
};

pub const MlpKind = enum {
    /// `down(act(gate(x)) * up(x))`.
    gated,
    /// Same with one `[2I][H]` tensor holding gate rows then up rows.
    gated_fused,
    /// `down(act(up(x)))`.
    dense,
};

/// Normalisation of query/key vectors before RoPE.
pub const QkNorm = enum {
    none,
    /// One `[head_dim]` weight shared by every head, normalised per head.
    head,
    /// `[heads * head_dim]` weights, normalised per head (Cohere).
    heads,
    /// `[heads * head_dim]` weight, normalised over the whole projection (OLMo 2).
    full,
    /// Per-head L2 normalisation without weights (Llama 4).
    l2,
};

pub const RouterScoring = enum { softmax, sigmoid };

/// Linear-attention recurrence of a hybrid family's `linear_attention` layers.
pub const LinearKind = enum {
    /// Gated DeltaNet (Qwen3-Next, Qwen3.5): one decay per head, a
    /// silu-gated output norm.
    gated_deltanet,
    /// Kimi Delta Attention (Kimi Linear): per-channel decay from a low-rank
    /// forget gate, a sigmoid-gated output norm; conv over q/k/v only.
    kda,
};

pub const TopkMethod = enum {
    /// Plain top-k over the scores.
    greedy,
    /// Group-limited: keep the best `topk_group` groups of experts, then top-k.
    group_limited,
};

pub const RopeScaling = union(enum) {
    none,
    linear: f32,
    llama3: struct { factor: f32, low_freq_factor: f32, high_freq_factor: f32, original_max_position: f32 },
    yarn: struct { factor: f32, original_max_position: f32, beta_fast: f32, beta_slow: f32, attention_factor: f32, truncate: bool },
    /// Phi-3 LongRoPE; the short factors are used for every position.
    longrope: struct { factors: []const f32, attention_factor: f32 },
    /// Per-frequency divisors (`rope_freqs.weight` of a GGUF file, llama.cpp's
    /// precomputed llama3 scaling): `inv_freq[i] /= factors[i]`.
    factors: []const f32,
};

/// Multi-head latent attention (DeepSeek V2/V3).
pub const Mla = struct {
    q_lora_rank: ?usize,
    kv_lora_rank: usize,
    qk_nope_head_dim: usize,
    qk_rope_head_dim: usize,
    v_head_dim: usize,
};

/// Llama 4 attention temperature tuning on layers without RoPE.
pub const AttnTemperature = struct { floor_scale: f32, attn_scale: f32 };

/// gpt-oss gated activation: `(clamp(up) + 1) * clamp(gate) * sigmoid(alpha * gate)`.
pub const Swiglu = struct { alpha: f32, limit: f32 };

pub const MoeConfig = struct {
    scoring: RouterScoring = .softmax,
    topk_method: TopkMethod = .greedy,
    n_group: usize = 1,
    topk_group: usize = 1,
    routed_scaling_factor: f32 = 1.0,
    /// Scale the expert input by the routing weight instead of its output (Llama 4).
    scale_input: bool = false,
    /// Gate and up columns of the fused `[E][H][2I]` expert tensor are interleaved (gpt-oss).
    gate_up_interleaved: bool = false,
    /// Expert projections carry biases (gpt-oss).
    expert_bias: bool = false,
    swiglu: ?Swiglu = null,
    /// Group scores are the sum of the two best selection scores even without
    /// a correction bias (Kimi Linear's router).
    group_score_top2: bool = false,
};

/// Tensor-name templates. `{p}` is the model prefix, `{i}` the layer index and
/// `{e}` an expert index; layer-level names are relative to `layer`. Biases
/// are always optional and named by replacing a trailing `.weight` with
/// `.bias`; norm biases likewise.
pub const Names = struct {
    prefixes: []const []const u8 = &.{ "model.", "language_model.model.", "model.language_model.", "" },
    embed: []const u8 = "{p}embed_tokens.weight",
    /// Learned absolute position table `[positions][hidden]`.
    pos_embed: ?[]const u8 = null,
    /// LayerNorm applied to the embeddings (BLOOM).
    embed_norm: ?[]const u8 = null,
    final_norm: []const u8 = "{p}norm.weight",
    lm_head: []const []const u8 = &.{"lm_head.weight"},
    layer: []const u8 = "{p}layers.{i}.",
    /// Norm on the attention input (null: attention reads the residual directly).
    input_norm: []const []const u8 = &.{"input_layernorm.weight"},
    /// Norm on the attention output before the residual add.
    post_attn_norm: ?[]const u8 = null,
    /// Norm on the MLP input in sequential layouts.
    pre_ff_norm: ?[]const u8 = "post_attention_layernorm.weight",
    /// Norm on the MLP output before the residual add.
    post_ff_norm: ?[]const u8 = null,
    /// Separate MLP input norm in parallel-residual layouts (Falcon 40B `ln_mlp`).
    mlp_norm: ?[]const u8 = null,
    q_norm: ?[]const u8 = null,
    k_norm: ?[]const u8 = null,
    q: ?[]const u8 = "self_attn.q_proj.weight",
    k: ?[]const u8 = "self_attn.k_proj.weight",
    v: ?[]const u8 = "self_attn.v_proj.weight",
    qkv: ?[]const u8 = null,
    o: []const u8 = "self_attn.o_proj.weight",
    sinks: ?[]const u8 = null,
    /// Gated DeltaNet linear-attention projections (Qwen hybrids). Qwen3-Next
    /// fuses q/k/v/z into `lin_qkvz` and b/a into `lin_ba`; Qwen3.5 splits
    /// them into `lin_qkv`, `lin_z`, `lin_b`, `lin_a`.
    lin_qkvz: ?[]const u8 = null,
    lin_qkv: ?[]const u8 = null,
    lin_z: ?[]const u8 = null,
    lin_b: ?[]const u8 = null,
    lin_a: ?[]const u8 = null,
    lin_ba: ?[]const u8 = null,
    /// Kimi Delta Attention keeps separate q/k/v projections, a low-rank
    /// forget gate (`lin_f_a`, `lin_f_b`) and a low-rank output gate
    /// (`lin_g_a`, `lin_g_b`); `lin_b` is its beta projection.
    lin_q: ?[]const u8 = null,
    lin_k: ?[]const u8 = null,
    lin_v: ?[]const u8 = null,
    lin_f_a: []const []const u8 = &.{},
    lin_f_b: []const []const u8 = &.{},
    lin_g_a: ?[]const u8 = null,
    lin_g_b: ?[]const u8 = null,
    /// Depthwise causal convolution, time-step bias, decay and gated norm of a
    /// linear-attention layer; its output projection is `lin_out`. Lists hold
    /// the alternatives of families whose checkpoints and Hugging Face
    /// modules name a tensor differently (first present wins).
    lin_conv: ?[]const u8 = null,
    /// Per-projection q/k/v convolutions (original Kimi Linear checkpoints),
    /// used when `lin_conv` is absent.
    lin_conv_split: []const []const u8 = &.{},
    lin_dt_bias: []const []const u8 = &.{},
    lin_a_log: []const []const u8 = &.{},
    lin_norm: ?[]const u8 = null,
    lin_out: ?[]const u8 = null,
    // MLA
    q_a: ?[]const u8 = null,
    q_a_norm: ?[]const u8 = null,
    q_b: ?[]const u8 = null,
    kv_a: ?[]const u8 = null,
    kv_a_norm: ?[]const u8 = null,
    kv_b: ?[]const u8 = null,
    gate: ?[]const u8 = "mlp.gate_proj.weight",
    up: ?[]const u8 = "mlp.up_proj.weight",
    gate_up: ?[]const u8 = null,
    down: []const u8 = "mlp.down_proj.weight",
    // MoE
    router: []const u8 = "mlp.gate.weight",
    router_correction_bias: ?[]const u8 = null,
    expert: []const u8 = "mlp.experts.{e}.",
    expert_gate: []const u8 = "gate_proj.weight",
    expert_up: []const u8 = "up_proj.weight",
    expert_down: []const u8 = "down_proj.weight",
    fused_gate_up: []const []const u8 = &.{ "mlp.experts.gate_up_proj", "mlp.experts.gate_up_proj.weight" },
    fused_down: []const []const u8 = &.{ "mlp.experts.down_proj", "mlp.experts.down_proj.weight" },
    /// Shared expert prefix (its projections use the expert_* names unless
    /// `shared_gate`/`shared_up`/`shared_down` override them).
    shared_expert: ?[]const u8 = null,
    shared_expert_gate: ?[]const u8 = null,
    shared_gate: ?[]const u8 = null,
    shared_up: ?[]const u8 = null,
    shared_down: ?[]const u8 = null,
    /// Alternative MoE names (router, experts, shared expert) of checkpoints
    /// that predate the family's Hugging Face module layout; picked when the
    /// primary router tensor is absent and this one's is present.
    moe_alt: ?*const Names = null,
};

/// One architecture family.
pub const Arch = struct {
    model_type: []const u8,
    aliases: []const []const u8 = &.{},
    /// llama.cpp architecture name (for a GGUF writer), if there is one.
    llama_cpp: ?[]const u8,
    /// Chat template family used when the model's Jinja template is not recognised.
    chat: []const u8 = "raw",
    /// Verified against a NumPy fixture (see `notes`).
    verified: bool = false,
    /// What the fixture covers, or what is unverified.
    notes: []const u8 = "",
    names: Names = .{},
    norm: NormKind = .rms,
    parallel_residual: bool = false,
    positional: Positional = .rope,
    rope_style: RopeStyle = .neox,
    qkv: QkvLayout = .separate,
    mlp: MlpKind = .gated,
    /// Weights are stored `[in][out]` (GPT-2 Conv1D) and transposed on load.
    conv1d: bool = false,
    /// Activation used when the config does not name one.
    activation: tensor.Activation = .silu,
    attention_bias: bool = false,
    tie_word_embeddings: bool = false,
    embed_scale_sqrt: bool = false,
    qk_norm: QkNorm = .none,
    /// Recurrence of the family's `linear_attention` layers, if it has any.
    linear: LinearKind = .gated_deltanet,
    /// Family-specific config keys.
    extra: ?*const fn (*Config, Allocator, std.json.ObjectMap) anyerror!void = null,
};

pub const Config = struct {
    arch: *const Arch,
    model_type: []const u8,
    hidden_size: usize,
    intermediate_size: usize,
    num_layers: usize,
    num_heads: usize,
    num_kv_heads: usize,
    /// Query/key head size (the KV cache stride; `v_head_dim <= head_dim`).
    head_dim: usize,
    v_head_dim: usize,
    vocab_size: usize,
    /// Norm epsilon (RMSNorm or LayerNorm).
    rms_norm_eps: f32,
    rope_theta: f32,
    rope_local_theta: f32,
    rope_scaling: RopeScaling,
    /// Rotated coordinates per head (`<= head_dim`).
    rotary_dim: usize,
    rope_style: RopeStyle,
    /// Per layer: whether rotary embeddings are applied.
    rope_layers: []bool,
    positional: Positional,
    /// Per layer: true for Gated DeltaNet linear-attention layers (Qwen
    /// hybrids); false for full-attention layers.
    linear_layers: []bool,
    /// Any linear-attention layer present.
    has_linear: bool,
    linear_kind: LinearKind,
    linear_k_heads: usize,
    linear_k_dim: usize,
    linear_v_heads: usize,
    linear_v_dim: usize,
    linear_conv_kernel: usize,
    /// Full-attention `q_proj` carries q rows then gate rows; the gate
    /// (sigmoid, or silu when `gate_swish`) multiplies the attention output.
    gated_attention: bool,
    gate_swish: bool,
    /// Offset added to positions when indexing the learned table (OPT: 2).
    position_offset: usize,
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
    norm: NormKind,
    parallel_residual: bool,
    qkv_layout: QkvLayout,
    mlp: MlpKind,
    qk_norm: QkNorm,
    /// Apply the q/k norm only on layers with RoPE (Llama 4).
    qk_norm_rope_only: bool,
    clip_qkv: ?f32,
    /// Scale on every sublayer output before the residual add (Granite, MiniCPM).
    residual_multiplier: f32,
    /// Multiplier on the final logits.
    logit_scale: f32,
    attn_temperature: ?AttnTemperature,
    sinks: bool,
    mla: ?Mla,
    /// Mixture-of-experts settings (num_experts == 0 for dense models).
    num_experts: usize,
    num_experts_per_tok: usize,
    norm_topk_prob: bool,
    moe_intermediate_size: usize,
    /// Per layer: true if the layer's MLP is a routed mixture of experts.
    moe_layers: []bool,
    moe: MoeConfig,

    pub fn isGemma(self: *const Config) bool {
        return self.norm == .rms_gemma;
    }

    pub fn kvDim(self: *const Config) usize {
        return self.num_kv_heads * self.head_dim;
    }
};

// ---------------------------------------------------------------------------
// JSON helpers
// ---------------------------------------------------------------------------

pub fn getNum(obj: std.json.ObjectMap, key: []const u8) ?f64 {
    const v = obj.get(key) orelse return null;
    return switch (v) {
        .integer => |i| @floatFromInt(i),
        .float => |f| f,
        else => null,
    };
}

pub fn getInt(obj: std.json.ObjectMap, key: []const u8, default: usize) usize {
    const v = getNum(obj, key) orelse return default;
    if (v < 0) return default;
    return @intFromFloat(v);
}

pub fn getF32(obj: std.json.ObjectMap, key: []const u8, default: f32) f32 {
    const v = getNum(obj, key) orelse return default;
    return @floatCast(v);
}

pub fn getBool(obj: std.json.ObjectMap, key: []const u8, default: bool) bool {
    const v = obj.get(key) orelse return default;
    return switch (v) {
        .bool => |b| b,
        .integer => |i| i != 0,
        else => default,
    };
}

pub fn getStr(obj: std.json.ObjectMap, key: []const u8) ?[]const u8 {
    const v = obj.get(key) orelse return null;
    return if (v == .string) v.string else null;
}

fn getObj(obj: std.json.ObjectMap, key: []const u8) ?std.json.ObjectMap {
    const v = obj.get(key) orelse return null;
    return if (v == .object) v.object else null;
}

/// First present key of `keys`.
fn getIntAny(obj: std.json.ObjectMap, keys: []const []const u8, default: usize) usize {
    for (keys) |k| if (getNum(obj, k) != null) return getInt(obj, k, default);
    return default;
}

fn getF32Any(obj: std.json.ObjectMap, keys: []const []const u8, default: f32) f32 {
    for (keys) |k| if (getNum(obj, k) != null) return getF32(obj, k, default);
    return default;
}

/// Reads a JSON array of 0/1 or bools into a per-layer table (missing entries keep `default`).
fn layerFlags(arena: Allocator, obj: std.json.ObjectMap, key: []const u8, layers: usize, default: bool) !?[]bool {
    const v = obj.get(key) orelse return null;
    if (v != .array) return null;
    const out = try arena.alloc(bool, layers);
    @memset(out, default);
    for (v.array.items, 0..) |item, i| {
        if (i >= layers) break;
        out[i] = switch (item) {
            .bool => |b| b,
            .integer => |n| n != 0,
            else => default,
        };
    }
    return out;
}

pub fn parseActivation(name: []const u8) ?tensor.Activation {
    const table = .{
        .{ "silu", tensor.Activation.silu },           .{ "swish", tensor.Activation.silu },
        .{ "swiglu", tensor.Activation.silu },         .{ "gelu", tensor.Activation.gelu },
        .{ "gelu_new", tensor.Activation.gelu_tanh },  .{ "gelu_pytorch_tanh", tensor.Activation.gelu_tanh },
        .{ "gelu_tanh", tensor.Activation.gelu_tanh }, .{ "gelu_fast", tensor.Activation.gelu_tanh },
        .{ "relu", tensor.Activation.relu },           .{ "relu2", tensor.Activation.relu2 },
        .{ "relu_squared", tensor.Activation.relu2 },  .{ "quick_gelu", tensor.Activation.quick_gelu },
    };
    inline for (table) |e| {
        if (std.mem.eql(u8, name, e[0])) return e[1];
    }
    return null;
}

// ---------------------------------------------------------------------------
// Config parsing
// ---------------------------------------------------------------------------

/// Finds the descriptor for a `model_type`.
pub fn lookup(model_type: []const u8) ?*const Arch {
    for (&registry) |*a| {
        if (std.mem.eql(u8, a.model_type, model_type)) return a;
        for (a.aliases) |alias| if (std.mem.eql(u8, alias, model_type)) return a;
    }
    return null;
}

fn rejectKnownHybrid(model_type: []const u8) !void {
    const table = .{
        .{ "kimi_k3", "Kimi K3 (its weights format and AttnRes are not implemented yet)" },
        .{ "kimi_k2", "Kimi K2 (FP8 E4M3 block-quantised weights, no tokenizer.json)" },
        .{ "qwen4_exp", "Qwen3.8-Flash-Next hybrid (linear attention with sparse indexer and hyper-connections)" },
        .{ "qwen4_exp_text", "Qwen3.8-Flash-Next hybrid (linear attention with sparse indexer and hyper-connections)" },
        .{ "glm5_next", "GLM-5.3-Flash hybrid (linear attention with sparse indexer and hyper-connections)" },
        .{ "glm5_next_text", "GLM-5.3-Flash hybrid (linear attention with sparse indexer and hyper-connections)" },
        .{ "glm_moe_lite", "GLM-4.7-Flash (not yet verified against a reference forward pass)" },
        .{ "deepseek_v4", "DeepSeek V4 (sparse indexer, hash layers, hyper-connections, FP4/FP8 weights)" },
        .{ "deepseek_v41", "DeepSeek V4.1 (sparse indexer, hash layers, hyper-connections, FP4/FP8 weights)" },
    };
    inline for (table) |e| {
        if (std.mem.eql(u8, model_type, e[0])) {
            std.log.err("unsupported model: {s}", .{e[1]});
            return error.UnsupportedArchitecture;
        }
    }
}

fn rejectUnsupportedMath(top: std.json.ObjectMap, obj: std.json.ObjectMap) !void {
    if (getObj(top, "quantization_config")) |qc| {
        const method = getStr(qc, "quant_method") orelse "";
        const fmt = getStr(qc, "format") orelse getStr(qc, "fmt") orelse "";
        if (std.mem.eql(u8, method, "compressed-tensors") or std.mem.indexOf(u8, fmt, "mxfp4") != null or std.mem.indexOf(u8, fmt, "pack-quantized") != null) {
            std.log.err("unsupported model: compressed/MXFP4 quantised weights cannot be dequantised", .{});
            return error.UnsupportedArchitecture;
        }
        if (std.mem.eql(u8, method, "fp8")) {
            std.log.err("unsupported model: FP8 block-quantised weights cannot be dequantised yet", .{});
            return error.UnsupportedArchitecture;
        }
    }
    if (getStr(obj, "expert_dtype")) |dt| {
        if (!std.mem.eql(u8, dt, "bfloat16") and !std.mem.eql(u8, dt, "float32")) {
            std.log.err("unsupported model: '{s}' expert dtype cannot be dequantised yet", .{dt});
            return error.UnsupportedArchitecture;
        }
    }
    if (getNum(obj, "num_hash_layers") != null or getBool(obj, "mhc", false) or getNum(obj, "hc_mult") != null) {
        std.log.err("unsupported model: hyper-connections and hash layers are not implemented", .{});
        return error.UnsupportedArchitecture;
    }
    if (getStr(obj, "scoring_func")) |s| {
        if (std.mem.eql(u8, s, "sqrtsoftplus")) {
            std.log.err("unsupported model: 'sqrtsoftplus' MoE routing is not implemented", .{});
            return error.UnsupportedArchitecture;
        }
    }
    const act = getStr(obj, "hidden_activation") orelse getStr(obj, "hidden_act") orelse getStr(obj, "activation_function") orelse "";
    if (std.mem.eql(u8, act, "situ")) {
        std.log.err("unsupported model: 'situ' activation is not implemented", .{});
        return error.UnsupportedArchitecture;
    }
}

pub fn parseConfig(arena: Allocator, json_text: []const u8) !Config {
    var parsed = try std.json.parseFromSlice(std.json.Value, arena, json_text, .{});
    defer parsed.deinit();
    if (parsed.value != .object) return error.InvalidConfig;
    var obj = parsed.value.object;
    const top_type = getStr(obj, "model_type") orelse "llama";
    var model_type = top_type;
    // Multimodal wrappers keep the text config nested.
    if (getObj(obj, "text_config")) |tc| {
        obj = tc;
        if (getStr(obj, "model_type")) |m| model_type = m;
    }
    const arch = lookup(model_type) orelse lookup(top_type) orelse {
        try rejectKnownHybrid(model_type);
        try rejectKnownHybrid(top_type);
        try rejectUnsupportedMath(parsed.value.object, obj);
        var names: std.Io.Writer.Allocating = .init(arena);
        for (&registry, 0..) |*a, i| {
            if (i > 0) try names.writer.writeAll(", ");
            try names.writer.writeAll(a.model_type);
        }
        std.log.err("unsupported model_type: {s} (supported: {s})", .{ model_type, names.written() });
        return error.UnsupportedArchitecture;
    };
    // Nested attention config (MPT).
    const attn_cfg: std.json.ObjectMap = getObj(obj, "attn_config") orelse obj;

    const hidden = getIntAny(obj, &.{ "hidden_size", "n_embd", "d_model" }, 0);
    const heads = getIntAny(obj, &.{ "num_attention_heads", "n_head", "n_heads" }, 0);
    const layers = getIntAny(obj, &.{ "num_hidden_layers", "n_layer", "n_layers", "num_layers" }, 0);
    if (hidden == 0 or heads == 0 or layers == 0) return error.InvalidConfig;
    var kv_heads = getIntAny(obj, &.{ "num_key_value_heads", "num_kv_heads", "n_head_kv", "multi_query_group_num" }, heads);
    if (attn_cfg.get("kv_n_heads") != null) kv_heads = getInt(attn_cfg, "kv_n_heads", heads);
    if (getBool(obj, "multi_query", false)) kv_heads = 1;
    var head_dim = getIntAny(obj, &.{ "head_dim", "kv_channels" }, hidden / heads);
    var v_head_dim = head_dim;

    var mla: ?Mla = null;
    if (getNum(obj, "kv_lora_rank") != null and getNum(obj, "qk_rope_head_dim") != null) {
        const m = Mla{
            .q_lora_rank = if (getNum(obj, "q_lora_rank")) |_| getInt(obj, "q_lora_rank", 0) else null,
            .kv_lora_rank = getInt(obj, "kv_lora_rank", 0),
            .qk_nope_head_dim = getInt(obj, "qk_nope_head_dim", 0),
            .qk_rope_head_dim = getInt(obj, "qk_rope_head_dim", 0),
            .v_head_dim = getInt(obj, "v_head_dim", 0),
        };
        if (m.q_lora_rank == 0) return error.InvalidConfig;
        head_dim = m.qk_nope_head_dim + m.qk_rope_head_dim;
        v_head_dim = m.v_head_dim;
        if (v_head_dim > head_dim) return error.InvalidConfig;
        kv_heads = heads;
        mla = m;
    }

    const act_name = getStr(obj, "hidden_activation") orelse getStr(obj, "hidden_act") orelse getStr(obj, "activation_function") orelse "";
    var act = arch.activation;
    if (act_name.len > 0) {
        if (parseActivation(act_name)) |a| act = a else std.log.warn("unknown activation '{s}'; using {s}", .{ act_name, @tagName(act) });
    }

    const max_pos = getIntAny(obj, &.{ "max_position_embeddings", "n_positions", "max_seq_len", "seq_length" }, 4096);
    const rope_theta = getF32Any(obj, &.{ "rope_theta", "rotary_emb_base", "rope_base" }, 10000.0);
    var rotary_dim = head_dim;
    if (getNum(obj, "partial_rotary_factor")) |f| rotary_dim = @intFromFloat(@as(f64, @floatFromInt(head_dim)) * f);
    if (getNum(obj, "rotary_pct")) |f| rotary_dim = @intFromFloat(@as(f64, @floatFromInt(head_dim)) * f);
    if (mla) |m| rotary_dim = m.qk_rope_head_dim;
    rotary_dim -= rotary_dim % 2;

    var rope_scaling: RopeScaling = .none;
    if (getObj(obj, "rope_scaling")) |rs| {
        const t = getStr(rs, "rope_type") orelse getStr(rs, "type") orelse "";
        if (std.mem.eql(u8, t, "llama3")) {
            rope_scaling = .{ .llama3 = .{
                .factor = getF32(rs, "factor", 8),
                .low_freq_factor = getF32(rs, "low_freq_factor", 1),
                .high_freq_factor = getF32(rs, "high_freq_factor", 4),
                .original_max_position = getF32(rs, "original_max_position_embeddings", 8192),
            } };
        } else if (std.mem.eql(u8, t, "linear")) {
            rope_scaling = .{ .linear = getF32(rs, "factor", 1) };
        } else if (std.mem.eql(u8, t, "yarn")) {
            const factor = getF32(rs, "factor", 1);
            const mscale = getF32(rs, "mscale", 1);
            const mscale_all_dim = getF32(rs, "mscale_all_dim", 0);
            const attention_factor: f32 = if (getNum(rs, "attention_factor")) |af| @floatCast(af) else if (mscale != 0 and mscale_all_dim != 0)
                yarnMscale(factor, mscale) / yarnMscale(factor, mscale_all_dim)
            else
                yarnMscale(factor, 1);
            rope_scaling = .{ .yarn = .{
                .factor = factor,
                .original_max_position = getF32(rs, "original_max_position_embeddings", getF32(obj, "original_max_position_embeddings", 4096)),
                .beta_fast = getF32(rs, "beta_fast", 32),
                .beta_slow = getF32(rs, "beta_slow", 1),
                .attention_factor = attention_factor,
                .truncate = getBool(rs, "truncate", true),
            } };
        } else if (std.mem.eql(u8, t, "longrope")) {
            const half = rotary_dim / 2;
            const factors = try arena.alloc(f32, half);
            @memset(factors, 1);
            if (rs.get("short_factor")) |sf| {
                if (sf == .array) for (sf.array.items, 0..) |v, i| {
                    if (i < half) factors[i] = switch (v) {
                        .float => |f| @floatCast(f),
                        .integer => |n| @floatFromInt(n),
                        else => 1,
                    };
                };
            }
            const original: f32 = getF32(obj, "original_max_position_embeddings", getF32(rs, "original_max_position_embeddings", @floatFromInt(max_pos)));
            const factor: f32 = if (getNum(rs, "factor")) |f| @floatCast(f) else @as(f32, @floatFromInt(max_pos)) / original;
            const attention_factor: f32 = if (getNum(rs, "attention_factor")) |af| @floatCast(af) else if (factor <= 1) 1.0 else @sqrt(1.0 + @log(factor) / @log(original));
            rope_scaling = .{ .longrope = .{ .factors = factors, .attention_factor = attention_factor } };
            std.log.warn("longrope: using the short rotary factors for every position (prompts beyond {d} tokens use the wrong table)", .{@as(usize, @intFromFloat(original))});
        } else if (std.mem.eql(u8, t, "ditch_factors")) {
            // Per-frequency divisors (a GGUF `rope_freqs.weight` that is not a
            // standard llama3 scaling); written by gguf_model.zig, read only by ditch.
            if (rs.get("factors")) |fa| {
                if (fa == .array) {
                    const factors = try arena.alloc(f32, fa.array.items.len);
                    for (fa.array.items, 0..) |v, i| factors[i] = switch (v) {
                        .float => |x| @floatCast(x),
                        .integer => |x| @floatFromInt(x),
                        else => 1.0,
                    };
                    rope_scaling = .{ .factors = factors };
                }
            }
        } else if (std.mem.eql(u8, t, "dynamic")) {
            std.log.warn("dynamic NTK rope scaling is treated as unscaled RoPE (exact below the original context length)", .{});
        } else if (std.mem.eql(u8, t, "mrope") or std.mem.eql(u8, t, "default") or t.len == 0) {
            // mrope over text-only positions equals plain RoPE.
        } else {
            std.log.warn("rope scaling type '{s}' is not supported; using unscaled RoPE", .{t});
        }
    }

    const sliding_window: ?usize = blk: {
        const v = obj.get("sliding_window") orelse break :blk null;
        break :blk switch (v) {
            .integer => |i| if (i > 0) @as(?usize, @intCast(i)) else null,
            else => null,
        };
    };
    const sliding_layers = try arena.alloc(bool, layers);
    @memset(sliding_layers, false);
    const linear_layers = try arena.alloc(bool, layers);
    @memset(linear_layers, false);
    var has_linear = false;
    if (obj.get("layer_types")) |lt| {
        if (lt == .array) {
            for (lt.array.items, 0..) |v, i| {
                if (i < layers and v == .string) {
                    const t = v.string;
                    if (std.mem.eql(u8, t, "sliding_attention") or std.mem.eql(u8, t, "chunked_attention")) {
                        sliding_layers[i] = true;
                    } else if (std.mem.eql(u8, t, "linear_attention")) {
                        linear_layers[i] = true;
                        has_linear = true;
                    } else if (std.mem.eql(u8, t, "full_attention")) {} else if (std.mem.eql(u8, t, "deepseek_sparse_attention")) {
                        // Zhipu's sparse indexer selects a top-k over the keys; for
                        // the short calibration contexts ditch scores, the top-k
                        // covers the whole context, so dense attention is exact.
                        std.log.warn("deepseek_sparse_attention runs as dense attention (exact for short contexts)", .{});
                    } else {
                        std.log.err("unsupported layer type '{s}' (only full/sliding/linear attention are implemented)", .{t});
                        return error.UnsupportedArchitecture;
                    }
                }
            }
        }
    } else if (getNum(obj, "full_attention_interval")) |_| {
        // Qwen3-Next style hybrid: every Nth layer is full attention, the rest linear.
        const interval = getInt(obj, "full_attention_interval", 4);
        if (interval > 0) for (linear_layers, 0..) |*l, i| {
            l.* = ((i + 1) % interval != 0);
            has_linear = has_linear or l.*;
        };
    } else if (sliding_window != null and getNum(obj, "sliding_window_pattern") == null) {
        @memset(sliding_layers, true);
    }
    if (getNum(obj, "sliding_window_pattern")) |_| {
        const pattern = getInt(obj, "sliding_window_pattern", 6);
        if (pattern > 0) for (sliding_layers, 0..) |*s, i| {
            s.* = ((i + 1) % pattern != 0);
        };
    }

    // Mixture of experts: `num_experts` (Qwen), `num_local_experts` (Mixtral,
    // Llama 4, gpt-oss) or `n_routed_experts` (DeepSeek).
    const num_experts = getIntAny(obj, &.{ "num_experts", "num_local_experts", "n_routed_experts" }, 0);
    const moe_layers = try arena.alloc(bool, layers);
    @memset(moe_layers, false);
    if (num_experts > 0) {
        const sparse_step = @max(getIntAny(obj, &.{ "decoder_sparse_step", "interleave_moe_layer_step", "moe_layer_freq" }, 1), 1);
        const first_dense = getInt(obj, "first_k_dense_replace", 0);
        for (moe_layers, 0..) |*m, i| m.* = (i >= first_dense) and ((i + 1) % sparse_step == 0);
        if (getNum(obj, "moe_layer_freq") != null) {
            for (moe_layers, 0..) |*m, i| m.* = (i >= first_dense) and (i % sparse_step == 0);
        }
        if (obj.get("mlp_only_layers")) |ml| {
            if (ml == .array) for (ml.array.items) |v| {
                if (v == .integer and v.integer >= 0 and v.integer < layers) moe_layers[@intCast(v.integer)] = false;
            };
        }
        if (obj.get("moe_layers")) |ml| {
            if (ml == .array) {
                @memset(moe_layers, false);
                for (ml.array.items) |v| {
                    if (v == .integer and v.integer >= 0 and v.integer < layers) moe_layers[@intCast(v.integer)] = true;
                }
            }
        }
    }
    const intermediate_size = getIntAny(obj, &.{ "intermediate_size", "n_inner", "ffn_dim", "ffn_hidden_size" }, 0);

    const linear_k_heads = getInt(obj, "linear_num_key_heads", 0);
    const linear_k_dim = getInt(obj, "linear_key_head_dim", 0);
    const linear_v_heads = getInt(obj, "linear_num_value_heads", 0);
    const linear_v_dim = getInt(obj, "linear_value_head_dim", 0);
    const linear_conv_kernel = getInt(obj, "linear_conv_kernel_dim", 0);
    const gate_swish = blk: {
        const t = getStr(obj, "output_gate_type") orelse break :blk false;
        if (std.mem.eql(u8, t, "swish") or std.mem.eql(u8, t, "silu")) break :blk true;
        if (std.mem.eql(u8, t, "sigmoid")) break :blk false;
        std.log.err("unsupported output_gate_type '{s}'", .{t});
        return error.UnsupportedArchitecture;
    };

    const rope_layers = try arena.alloc(bool, layers);
    @memset(rope_layers, arch.positional == .rope);

    var c = Config{
        .arch = arch,
        .model_type = try arena.dupe(u8, model_type),
        .hidden_size = hidden,
        .intermediate_size = if (intermediate_size > 0) intermediate_size else 4 * hidden,
        .num_layers = layers,
        .num_heads = heads,
        .num_kv_heads = kv_heads,
        .head_dim = head_dim,
        .v_head_dim = v_head_dim,
        .vocab_size = getIntAny(obj, &.{ "vocab_size", "padded_vocab_size" }, 0),
        .rms_norm_eps = getF32Any(obj, &.{ "rms_norm_eps", "layer_norm_eps", "layer_norm_epsilon", "layernorm_epsilon", "norm_eps", "norm_epsilon" }, if (arch.norm == .rms or arch.norm == .rms_gemma) 1e-6 else 1e-5),
        .rope_theta = rope_theta,
        .rope_local_theta = getF32(obj, "rope_local_base_freq", 10000.0),
        .rope_scaling = rope_scaling,
        .rotary_dim = rotary_dim,
        .rope_style = arch.rope_style,
        .rope_layers = rope_layers,
        .positional = arch.positional,
        .linear_layers = linear_layers,
        .has_linear = has_linear,
        .linear_kind = arch.linear,
        .linear_k_heads = linear_k_heads,
        .linear_k_dim = linear_k_dim,
        .linear_v_heads = linear_v_heads,
        .linear_v_dim = linear_v_dim,
        .linear_conv_kernel = linear_conv_kernel,
        .gated_attention = false,
        .gate_swish = gate_swish,
        .position_offset = 0,
        .tie_word_embeddings = getBool(obj, "tie_word_embeddings", arch.tie_word_embeddings),
        .activation = act,
        .max_position_embeddings = max_pos,
        .sliding_window = sliding_window,
        .sliding_layers = sliding_layers,
        .attention_scale = 1.0 / @sqrt(@as(f32, @floatFromInt(head_dim))),
        .attn_logit_softcapping = if (getNum(obj, "attn_logit_softcapping")) |v| @as(f32, @floatCast(v)) else null,
        .final_logit_softcapping = if (getNum(obj, "final_logit_softcapping")) |v| @as(f32, @floatCast(v)) else null,
        .attention_bias = getBool(obj, "attention_bias", arch.attention_bias),
        .embed_scale = if (arch.embed_scale_sqrt) @sqrt(@as(f32, @floatFromInt(hidden))) else 1.0,
        .norm = arch.norm,
        .parallel_residual = arch.parallel_residual,
        .qkv_layout = arch.qkv,
        .mlp = arch.mlp,
        .qk_norm = arch.qk_norm,
        .qk_norm_rope_only = false,
        .clip_qkv = null,
        .residual_multiplier = 1.0,
        .logit_scale = 1.0,
        .attn_temperature = null,
        .sinks = false,
        .mla = mla,
        .num_experts = num_experts,
        .num_experts_per_tok = getInt(obj, "num_experts_per_tok", 2),
        .norm_topk_prob = getBool(obj, "norm_topk_prob", false),
        .moe_intermediate_size = getInt(obj, "moe_intermediate_size", if (intermediate_size > 0) intermediate_size else 4 * hidden),
        .moe_layers = moe_layers,
        .moe = .{},
    };
    if (getNum(obj, "query_pre_attn_scalar")) |q| c.attention_scale = @floatCast(1.0 / @sqrt(q));
    if (getNum(attn_cfg, "clip_qkv")) |v| c.clip_qkv = @floatCast(v);
    if (obj.get("clip_qkv")) |v| {
        if (v == .float or v == .integer) c.clip_qkv = @floatCast(getNum(obj, "clip_qkv").?);
    }
    if (mla) |m| {
        c.attention_scale = 1.0 / @sqrt(@as(f32, @floatFromInt(m.qk_nope_head_dim + m.qk_rope_head_dim)));
        if (rope_scaling == .yarn) {
            // DeepSeek scales the softmax by mscale² (mscale_all_dim).
            if (getObj(obj, "rope_scaling")) |rs| {
                const all_dim = getF32(rs, "mscale_all_dim", 0);
                if (all_dim != 0) {
                    const ms = yarnMscale(rope_scaling.yarn.factor, all_dim);
                    c.attention_scale *= ms * ms;
                }
            }
        }
    }
    if (getBool(obj, "use_parallel_residual", false) or getBool(obj, "parallel_attn", false)) c.parallel_residual = true;
    if (obj.get("use_parallel_residual")) |v| {
        if (v == .bool) c.parallel_residual = v.bool;
    }
    if (obj.get("parallel_attn")) |v| {
        if (v == .bool) c.parallel_residual = v.bool;
    }
    if (getBool(attn_cfg, "alibi", false)) c.positional = .alibi;
    if (arch.extra) |f| try f(&c, arena, obj);
    if (c.has_linear and (c.linear_k_heads == 0 or c.linear_k_dim == 0 or c.linear_v_heads == 0 or c.linear_v_dim == 0 or c.linear_conv_kernel == 0)) {
        std.log.err("linear_attention layers need linear_num_key_heads/key_head_dim/num_value_heads/value_head_dim/conv_kernel_dim", .{});
        return error.InvalidConfig;
    }
    if (c.positional != .rope) @memset(c.rope_layers, false);
    if (getNum(obj, "sliding_window_pattern") == null and c.arch.norm == .rms_gemma and obj.get("layer_types") == null and c.sliding_window != null) {
        // Gemma 2: even layers are local.
        for (c.sliding_layers, 0..) |*s, i| s.* = (i % 2 == 0);
    }
    return c;
}

fn yarnMscale(scale: f32, mscale: f32) f32 {
    if (scale <= 1) return 1.0;
    return 0.1 * mscale * @log(scale) + 1.0;
}

// ---------------------------------------------------------------------------
// Family-specific config hooks
// ---------------------------------------------------------------------------

fn extraGemma(c: *Config, _: Allocator, obj: std.json.ObjectMap) !void {
    if (getStr(obj, "hidden_activation") == null and getStr(obj, "hidden_act") == null) c.activation = .gelu_tanh;
    if (getNum(obj, "sliding_window_pattern") == null and obj.get("layer_types") == null and c.sliding_window != null) {
        // Gemma 3 defaults to a pattern of 6 (5 local, 1 global); Gemma 2 alternates.
        if (std.mem.startsWith(u8, c.model_type, "gemma3")) {
            for (c.sliding_layers, 0..) |*s, i| s.* = ((i + 1) % 6 != 0);
        }
    }
}

fn extraMixtral(c: *Config, _: Allocator, obj: std.json.ObjectMap) !void {
    c.norm_topk_prob = getBool(obj, "norm_topk_prob", true);
}

fn extraPhi3(c: *Config, _: Allocator, _: std.json.ObjectMap) !void {
    if (c.rope_scaling == .yarn) {
        // Phi-3-small style yarn keeps the default attention factor.
    }
}

fn extraPhi(c: *Config, _: Allocator, obj: std.json.ObjectMap) !void {
    if (getBool(obj, "qk_layernorm", false)) c.qk_norm = .head;
}

fn extraNeox(c: *Config, _: Allocator, obj: std.json.ObjectMap) !void {
    if (obj.get("use_parallel_residual") == null) c.parallel_residual = true;
}

fn extraFalcon(c: *Config, _: Allocator, obj: std.json.ObjectMap) !void {
    if (obj.get("parallel_attn") == null) c.parallel_residual = true;
    if (getBool(obj, "new_decoder_architecture", false)) {
        c.qkv_layout = .grouped;
        c.num_kv_heads = getInt(obj, "num_kv_heads", c.num_heads);
    } else if (getBool(obj, "multi_query", true)) {
        c.qkv_layout = .concat;
        c.num_kv_heads = 1;
    } else {
        c.qkv_layout = .heads_interleaved;
        c.num_kv_heads = c.num_heads;
    }
    if (getBool(obj, "alibi", false)) c.positional = .alibi;
}

fn extraStableLm(c: *Config, _: Allocator, obj: std.json.ObjectMap) !void {
    c.attention_bias = getBool(obj, "use_qkv_bias", false);
    if (getBool(obj, "qk_layernorm", false)) c.qk_norm = .head;
}

fn extraOlmo2(c: *Config, _: Allocator, _: std.json.ObjectMap) !void {
    c.qk_norm = .full;
}

fn extraCohere(c: *Config, _: Allocator, obj: std.json.ObjectMap) !void {
    c.logit_scale = getF32(obj, "logit_scale", 1.0);
    if (getBool(obj, "use_qk_norm", false)) c.qk_norm = .heads;
    if (std.mem.eql(u8, c.model_type, "cohere2")) {
        // Command R7B: local layers use RoPE, global layers none.
        if (getNum(obj, "sliding_window_pattern") == null) {
            for (c.sliding_layers, 0..) |*s, i| s.* = ((i + 1) % 4 != 0);
        }
        for (c.rope_layers, 0..) |*r, i| r.* = c.sliding_layers[i];
    }
}

fn extraGlm4(c: *Config, _: Allocator, obj: std.json.ObjectMap) !void {
    if (getNum(obj, "partial_rotary_factor") == null) c.rotary_dim = c.head_dim / 2;
    c.attention_bias = getBool(obj, "attention_bias", true);
}

fn extraChatGlm(c: *Config, _: Allocator, obj: std.json.ObjectMap) !void {
    // Rotary embeddings cover half of `kv_channels`, interleaved; theta is scaled by rope_ratio.
    c.rotary_dim = c.head_dim / 2;
    c.rope_theta = 10000.0 * getF32(obj, "rope_ratio", 1.0);
    c.attention_bias = getBool(obj, "add_qkv_bias", true);
    if (!getBool(obj, "rmsnorm", true)) c.norm = .layer;
    if (getBool(obj, "apply_residual_connection_post_layernorm", false)) return error.UnsupportedArchitecture;
    if (getBool(obj, "post_layer_norm", true) == false) return error.UnsupportedArchitecture;
    if (getBool(obj, "multi_query_attention", false)) c.num_kv_heads = getInt(obj, "multi_query_group_num", c.num_heads) else c.num_kv_heads = c.num_heads;
    c.tie_word_embeddings = getBool(obj, "tie_word_embeddings", false);
}

fn extraGranite(c: *Config, _: Allocator, obj: std.json.ObjectMap) !void {
    c.embed_scale = getF32(obj, "embedding_multiplier", 1.0);
    c.residual_multiplier = getF32(obj, "residual_multiplier", 1.0);
    if (getNum(obj, "attention_multiplier")) |m| c.attention_scale = @floatCast(m);
    const ls = getF32(obj, "logits_scaling", 1.0);
    if (ls != 0) c.logit_scale = 1.0 / ls;
}

fn extraDeepseek(c: *Config, _: Allocator, obj: std.json.ObjectMap) !void {
    // The original checkpoints pair rotary coordinates as (2i, 2i+1) (`rope_interleave`).
    c.rope_style = if (getBool(obj, "rope_interleave", true)) .gptj else .neox;
    const scoring = getStr(obj, "scoring_func") orelse "softmax";
    c.moe.scoring = if (std.mem.eql(u8, scoring, "sigmoid")) .sigmoid else .softmax;
    const method = getStr(obj, "topk_method") orelse "greedy";
    c.moe.topk_method = if (std.mem.eql(u8, method, "group_limited_greedy") or std.mem.eql(u8, method, "noaux_tc")) .group_limited else .greedy;
    c.moe.n_group = @max(1, getInt(obj, "n_group", 1));
    c.moe.topk_group = @max(1, getInt(obj, "topk_group", 1));
    c.moe.routed_scaling_factor = getF32(obj, "routed_scaling_factor", 1.0);
    if (c.moe.topk_method == .group_limited and c.num_experts % c.moe.n_group != 0) return error.InvalidConfig;
}

fn extraLlama4(c: *Config, arena: Allocator, obj: std.json.ObjectMap) !void {
    if (try layerFlags(arena, obj, "no_rope_layers", c.num_layers, true)) |flags| {
        for (c.rope_layers, 0..) |*r, i| r.* = flags[i];
    }
    if (getBool(obj, "use_qk_norm", true)) {
        c.qk_norm = .l2;
        c.qk_norm_rope_only = true;
    }
    if (getBool(obj, "attn_temperature_tuning", false)) {
        c.attn_temperature = .{ .floor_scale = getF32(obj, "floor_scale", 8192), .attn_scale = getF32(obj, "attn_scale", 0.1) };
    }
    c.num_experts_per_tok = getInt(obj, "num_experts_per_tok", 1);
    c.moe.scoring = .sigmoid;
    c.moe.scale_input = true;
    // Routed and shared experts use `intermediate_size`; dense layers `intermediate_size_mlp`.
    c.moe_intermediate_size = c.intermediate_size;
    c.intermediate_size = getInt(obj, "intermediate_size_mlp", c.intermediate_size);
    if (getNum(obj, "attention_chunk_size")) |_| {
        std.log.warn("llama4: chunked local attention is run as full attention (exact for prompts shorter than attention_chunk_size)", .{});
        @memset(c.sliding_layers, false);
    }
}

fn extraGptOss(c: *Config, _: Allocator, obj: std.json.ObjectMap) !void {
    c.sinks = true;
    c.attention_bias = getBool(obj, "attention_bias", true);
    c.moe.gate_up_interleaved = true;
    c.moe.expert_bias = true;
    c.moe.swiglu = .{ .alpha = getF32(obj, "swiglu_alpha", 1.702), .limit = getF32(obj, "swiglu_limit", 7.0) };
    c.norm_topk_prob = true;
    c.num_experts_per_tok = getInt(obj, "num_experts_per_tok", 4);
    if (c.num_experts > 0) @memset(c.moe_layers, true);
}

fn extraMiniCpm(c: *Config, _: Allocator, obj: std.json.ObjectMap) !void {
    c.embed_scale = getF32(obj, "scale_emb", 1.0);
    c.residual_multiplier = getF32(obj, "scale_depth", 1.0) / @sqrt(@as(f32, @floatFromInt(c.num_layers)));
    const base = getF32(obj, "dim_model_base", @floatFromInt(c.hidden_size));
    c.logit_scale = base / @as(f32, @floatFromInt(c.hidden_size));
    if (getNum(obj, "kv_lora_rank") != null) return error.UnsupportedArchitecture; // MiniCPM3 (MLA) is not covered
}

fn extraExaone4(c: *Config, _: Allocator, obj: std.json.ObjectMap) !void {
    // Sliding (local) layers use RoPE; global layers have no positional encoding.
    if (c.sliding_window != null and obj.get("layer_types") == null) {
        if (getStr(obj, "sliding_window_pattern")) |pat| {
            // e.g. "LLLG": L = local (sliding), G = global.
            if (pat.len > 0) for (c.sliding_layers, 0..) |*s, i| {
                s.* = pat[i % pat.len] == 'L';
            };
        } else {
            const pattern = getInt(obj, "sliding_window_pattern", 4);
            for (c.sliding_layers, 0..) |*s, i| s.* = ((i + 1) % pattern != 0);
        }
    }
    if (c.sliding_window != null) {
        for (c.rope_layers, 0..) |*r, i| r.* = c.sliding_layers[i];
    }
}

fn extraNemotron(c: *Config, _: Allocator, obj: std.json.ObjectMap) !void {
    if (getStr(obj, "hidden_act") == null) c.activation = .relu2;
    c.attention_bias = getBool(obj, "attention_bias", false);
}

fn extraSmolLm3(c: *Config, arena: Allocator, obj: std.json.ObjectMap) !void {
    if (try layerFlags(arena, obj, "no_rope_layers", c.num_layers, true)) |flags| {
        for (c.rope_layers, 0..) |*r, i| r.* = flags[i];
    } else if (getNum(obj, "no_rope_layer_interval")) |_| {
        const interval = getInt(obj, "no_rope_layer_interval", 4);
        if (interval > 0) for (c.rope_layers, 0..) |*r, i| {
            r.* = ((i + 1) % interval != 0);
        };
    }
}

fn extraOpt(c: *Config, _: Allocator, obj: std.json.ObjectMap) !void {
    c.position_offset = 2;
    if (!getBool(obj, "do_layer_norm_before", true)) return error.UnsupportedArchitecture;
    if (getInt(obj, "word_embed_proj_dim", c.hidden_size) != c.hidden_size) return error.UnsupportedArchitecture;
    c.attention_bias = getBool(obj, "enable_bias", true);
}

fn extraMpt(c: *Config, _: Allocator, obj: std.json.ObjectMap) !void {
    const ratio = getF32(obj, "expansion_ratio", 4.0);
    if (getNum(obj, "intermediate_size") == null) c.intermediate_size = @intFromFloat(@as(f32, @floatFromInt(c.hidden_size)) * ratio);
    if (getObj(obj, "attn_config")) |ac| {
        if (getBool(ac, "qk_ln", false)) c.qk_norm = .full;
        if (getBool(ac, "alibi", false)) c.positional = .alibi else c.positional = .none;
    }
    if (getObj(obj, "ffn_config")) |fc| {
        if (getStr(fc, "ffn_type")) |t| if (!std.mem.eql(u8, t, "mptmlp")) return error.UnsupportedArchitecture;
    }
}

fn extraStarcoder2(c: *Config, _: Allocator, obj: std.json.ObjectMap) !void {
    c.attention_bias = getBool(obj, "use_bias", true);
    if (getStr(obj, "norm_type")) |t| if (!std.mem.eql(u8, t, "layer_norm")) return error.UnsupportedArchitecture;
}

fn extraGptBigcode(c: *Config, _: Allocator, obj: std.json.ObjectMap) !void {
    c.num_kv_heads = if (getBool(obj, "multi_query", true)) 1 else c.num_heads;
}

fn extraBaichuan(c: *Config, _: Allocator, obj: std.json.ObjectMap) !void {
    // Baichuan 13B checkpoints carry `model_max_length` instead of
    // `max_position_embeddings` and use ALiBi; 7B uses RoPE.
    if (obj.get("max_position_embeddings") == null and getNum(obj, "model_max_length") != null) {
        c.positional = .alibi;
        c.max_position_embeddings = getInt(obj, "model_max_length", 4096);
    }
}

fn extraQwenVl(c: *Config, _: Allocator, obj: std.json.ObjectMap) !void {
    if (std.mem.startsWith(u8, c.model_type, "qwen2")) c.attention_bias = getBool(obj, "attention_bias", true);
}

fn extraQwenHybrid(c: *Config, _: Allocator, obj: std.json.ObjectMap) !void {
    c.qk_norm = .head;
    c.gated_attention = true;
    if (c.num_experts > 0 and getNum(obj, "norm_topk_prob") == null and std.mem.startsWith(u8, c.model_type, "qwen3_5_moe")) {
        c.norm_topk_prob = true;
    }
}

fn extraGlm4Moe(c: *Config, _: Allocator, obj: std.json.ObjectMap) !void {
    if (getBool(obj, "use_qk_norm", false)) c.qk_norm = .head;
    c.moe.scoring = .sigmoid;
    c.moe.topk_method = .group_limited;
    c.moe.n_group = @max(1, getInt(obj, "n_group", 1));
    c.moe.topk_group = @max(1, getInt(obj, "topk_group", 1));
    c.moe.routed_scaling_factor = getF32(obj, "routed_scaling_factor", 1.0);
}

fn extraGlmMoeDsa(c: *Config, _: Allocator, obj: std.json.ObjectMap) !void {
    c.rope_style = if (getBool(obj, "rope_interleave", true)) .gptj else .neox;
    c.moe.scoring = .sigmoid;
    c.moe.topk_method = .group_limited;
    c.moe.n_group = @max(1, getInt(obj, "n_group", 1));
    c.moe.topk_group = @max(1, getInt(obj, "topk_group", 1));
    c.moe.routed_scaling_factor = getF32(obj, "routed_scaling_factor", 1.0);
    if (getNum(obj, "index_topk") != null) {
        std.log.warn("sparse indexer runs as dense attention (exact for short contexts)", .{});
    }
}

/// Kimi Linear (`KimiLinearConfig`): the original checkpoints keep the KDA
/// settings in a `linear_attn_config` sub-dict with 1-indexed layer lists and
/// use the `attribute_map` spellings (`num_experts_per_token`,
/// `moe_renormalize`, `num_expert_group`, `model_max_length`); the Hugging
/// Face module spells them out (`linear_num_heads`, `layer_types`,
/// `mlp_layer_types`). Full-attention layers are MLA without RoPE.
fn extraKimiLinear(c: *Config, _: Allocator, obj: std.json.ObjectMap) !void {
    const lac: ?std.json.ObjectMap = getObj(obj, "linear_attn_config");
    var heads = getInt(obj, "linear_num_heads", 32);
    var head_dim = getInt(obj, "linear_head_dim", 128);
    var kernel = getInt(obj, "linear_conv_kernel_dim", 4);
    if (lac) |l| {
        heads = getInt(l, "num_heads", heads);
        head_dim = getInt(l, "head_dim", head_dim);
        kernel = getInt(l, "short_conv_kernel_size", kernel);
    }
    c.linear_k_heads = heads;
    c.linear_v_heads = heads;
    c.linear_k_dim = head_dim;
    c.linear_v_dim = head_dim;
    c.linear_conv_kernel = kernel;
    if (obj.get("layer_types") == null) {
        var from_lists = false;
        if (lac) |l| {
            if (l.get("full_attn_layers") != null and l.get("kda_layers") != null) {
                from_lists = true;
                @memset(c.linear_layers, false);
                if (l.get("kda_layers")) |ka| if (ka == .array) for (ka.array.items) |v| {
                    if (v == .integer and v.integer >= 1 and v.integer <= c.num_layers) c.linear_layers[@intCast(v.integer - 1)] = true;
                };
            }
        }
        if (!from_lists) {
            for (c.linear_layers, 0..) |*is_lin, i| is_lin.* = !(i > 0 and i % 4 == 0);
        }
    }
    c.has_linear = false;
    for (c.linear_layers) |l| c.has_linear = c.has_linear or l;
    if (c.mla == null) {
        std.log.err("kimi_linear: full-attention layers need the MLA keys (kv_lora_rank, qk_rope_head_dim, ...)", .{});
        return error.InvalidConfig;
    }
    // MLA layers carry no positional encoding (positions come from the KDA layers).
    @memset(c.rope_layers, false);
    if (getNum(obj, "rms_norm_eps") == null) c.rms_norm_eps = 1e-5;
    if (getNum(obj, "model_max_length")) |_| c.max_position_embeddings = getInt(obj, "model_max_length", c.max_position_embeddings);
    // Mixture of experts: sigmoid scores, correction bias, top-2 group scores.
    c.num_experts_per_tok = getIntAny(obj, &.{ "num_experts_per_tok", "num_experts_per_token" }, 8);
    c.norm_topk_prob = getBool(obj, "norm_topk_prob", getBool(obj, "moe_renormalize", true));
    c.moe.scoring = .sigmoid;
    c.moe.topk_method = .group_limited;
    c.moe.n_group = @max(1, getIntAny(obj, &.{ "n_group", "num_expert_group" }, 1));
    c.moe.topk_group = @max(1, getInt(obj, "topk_group", 1));
    c.moe.routed_scaling_factor = getF32(obj, "routed_scaling_factor", 2.446);
    c.moe.group_score_top2 = true;
    if (c.num_experts % c.moe.n_group != 0) return error.InvalidConfig;
    if (c.num_experts > 0) {
        if (obj.get("mlp_layer_types")) |ml| {
            if (ml == .array) {
                for (ml.array.items, 0..) |v, i| {
                    if (i >= c.num_layers) break;
                    if (v == .string) c.moe_layers[i] = std.mem.eql(u8, v.string, "sparse");
                }
            }
        } else {
            const first_dense = getInt(obj, "first_k_dense_replace", 1);
            for (c.moe_layers, 0..) |*m, i| m.* = i >= first_dense;
        }
    }
}

// ---------------------------------------------------------------------------
// The registry
// ---------------------------------------------------------------------------

/// Original Kimi Linear checkpoints keep the experts under
/// `block_sparse_moe` with Mixtral's w1/w3/w2 names; the Hugging Face module
/// renames them to `mlp` and stacks the experts.
const kimi_linear_checkpoint_moe = Names{
    .router = "block_sparse_moe.gate.weight",
    .router_correction_bias = "block_sparse_moe.gate.e_score_correction_bias",
    .expert = "block_sparse_moe.experts.{e}.",
    .expert_gate = "w1.weight",
    .expert_up = "w3.weight",
    .expert_down = "w2.weight",
    .fused_gate_up = &.{},
    .fused_down = &.{},
    .shared_expert = "block_sparse_moe.shared_experts.",
    .shared_gate = "gate_proj.weight",
    .shared_up = "up_proj.weight",
    .shared_down = "down_proj.weight",
};

const deepseek_v3_names = Names{
    .q_a = "self_attn.q_a_proj.weight",
    .q_a_norm = "self_attn.q_a_layernorm.weight",
    .q_b = "self_attn.q_b_proj.weight",
    .kv_a = "self_attn.kv_a_proj_with_mqa.weight",
    .kv_a_norm = "self_attn.kv_a_layernorm.weight",
    .kv_b = "self_attn.kv_b_proj.weight",
    .router_correction_bias = "mlp.gate.e_score_correction_bias",
    .shared_expert = "mlp.shared_experts.",
};

const gemma_names = Names{
    .post_attn_norm = "post_attention_layernorm.weight",
    .pre_ff_norm = "pre_feedforward_layernorm.weight",
    .post_ff_norm = "post_feedforward_layernorm.weight",
};

const neox_style_names = Names{
    .prefixes = &.{ "gpt_neox.", "" },
    .embed = "{p}embed_in.weight",
    .final_norm = "{p}final_layer_norm.weight",
    .lm_head = &.{"embed_out.weight"},
    .q = null,
    .k = null,
    .v = null,
    .qkv = "attention.query_key_value.weight",
    .o = "attention.dense.weight",
    .gate = null,
    .up = "mlp.dense_h_to_4h.weight",
    .down = "mlp.dense_4h_to_h.weight",
};

pub const registry = [_]Arch{
    .{
        .model_type = "llama",
        .aliases = &.{ "mistral3_text", "smollm" },
        .llama_cpp = "llama",
        .chat = "llama3",
        .verified = true,
        .notes = "fixture: GQA, llama3 rope scaling, untied lm_head, byte-level BPE. Yi, SOLAR, TinyLlama, SmolLM 1/2 and Mistral 3 text configs are plain llama.",
    },
    .{
        .model_type = "mistral",
        .llama_cpp = "llama",
        .chat = "mistral",
        .notes = "llama layout with sliding window on every layer and an explicit head_dim; covered by the llama fixture (no sliding-window fixture).",
    },
    .{
        .model_type = "qwen2",
        .aliases = &.{ "qwen2_5_vl", "qwen2_5_vl_text", "qwen2_vl", "qwen2_vl_text" },
        .llama_cpp = "qwen2",
        .chat = "chatml",
        .verified = true,
        .attention_bias = true,
        .notes = "fixture: q/k/v biases, tied embeddings. Qwen2-VL / Qwen2.5-VL text configs (nested text_config, mrope over text positions) use the same path.",
        .extra = extraQwenVl,
    },
    .{
        .model_type = "qwen3",
        .aliases = &.{ "qwen3_vl", "qwen3_vl_text" },
        .llama_cpp = "qwen3",
        .chat = "chatml",
        .verified = true,
        .qk_norm = .head,
        .names = .{ .q_norm = "self_attn.q_norm.weight", .k_norm = "self_attn.k_norm.weight" },
        .notes = "fixture: per-head q/k RMSNorm. Qwen3-VL text config uses the same path.",
    },
    .{
        .model_type = "gemma2",
        .llama_cpp = "gemma2",
        .chat = "gemma",
        .norm = .rms_gemma,
        .activation = .gelu_tanh,
        .tie_word_embeddings = true,
        .embed_scale_sqrt = true,
        .names = gemma_names,
        .notes = "gemma3 layout with alternating local layers and logit softcapping; covered by the gemma3 fixture except softcapping.",
        .extra = extraGemma,
    },
    .{
        .model_type = "gemma3",
        .aliases = &.{"gemma3_text"},
        .llama_cpp = "gemma3",
        .chat = "gemma",
        .verified = true,
        .norm = .rms_gemma,
        .activation = .gelu_tanh,
        .tie_word_embeddings = true,
        .embed_scale_sqrt = true,
        .qk_norm = .head,
        .names = .{
            .post_attn_norm = "post_attention_layernorm.weight",
            .pre_ff_norm = "pre_feedforward_layernorm.weight",
            .post_ff_norm = "post_feedforward_layernorm.weight",
            .q_norm = "self_attn.q_norm.weight",
            .k_norm = "self_attn.k_norm.weight",
        },
        .notes = "fixture: (1+w) norms, pre/post norms, per-head (1+w) q/k norms, sqrt(H) embedding scale, sliding layers with a local rope base, query_pre_attn_scalar, linear rope scaling.",
        .extra = extraGemma,
    },
    .{
        .model_type = "qwen2_moe",
        .llama_cpp = "qwen2moe",
        .chat = "chatml",
        .attention_bias = true,
        .names = .{ .shared_expert = "mlp.shared_expert.", .shared_expert_gate = "mlp.shared_expert_gate.weight" },
        .notes = "qwen3_moe routing plus a sigmoid-gated shared expert (no fixture for the shared expert).",
    },
    .{
        .model_type = "qwen3_moe",
        .llama_cpp = "qwen3moe",
        .chat = "chatml",
        .verified = true,
        .qk_norm = .head,
        .names = .{ .q_norm = "self_attn.q_norm.weight", .k_norm = "self_attn.k_norm.weight" },
        .notes = "fixtures: softmax top-k with renormalisation, dense layers via mlp_only_layers, separate / fused / transposed-fused expert tensors.",
    },
    .{
        .model_type = "mixtral",
        .llama_cpp = "llama",
        .chat = "mistral",
        .names = .{
            .router = "block_sparse_moe.gate.weight",
            .expert = "block_sparse_moe.experts.{e}.",
            .expert_gate = "w1.weight",
            .expert_up = "w3.weight",
            .expert_down = "w2.weight",
            .fused_gate_up = &.{},
            .fused_down = &.{},
        },
        .notes = "qwen3_moe routing (softmax, top-k renormalised) with Mixtral tensor names; no fixture.",
        .extra = extraMixtral,
    },
    .{
        .model_type = "phi3",
        .aliases = &.{"phi4"},
        .llama_cpp = "phi3",
        .chat = "phi3",
        .verified = true,
        .qkv = .concat,
        .mlp = .gated_fused,
        .names = .{ .q = null, .k = null, .v = null, .qkv = "self_attn.qkv_proj.weight", .gate = null, .up = null, .gate_up = "mlp.gate_up_proj.weight" },
        .notes = "fixture: fused qkv_proj and gate_up_proj, longrope (short factors + attention factor). Phi-3 / 3.5 / 4-mini.",
        .extra = extraPhi3,
    },
    .{
        .model_type = "phi",
        .llama_cpp = "phi2",
        .verified = true,
        .norm = .layer,
        .parallel_residual = true,
        .mlp = .dense,
        .activation = .gelu_tanh,
        .attention_bias = true,
        .names = .{
            .final_norm = "{p}final_layernorm.weight",
            .pre_ff_norm = null,
            .o = "self_attn.dense.weight",
            .gate = null,
            .up = "mlp.fc1.weight",
            .down = "mlp.fc2.weight",
            .q_norm = "self_attn.q_layernorm.weight",
            .k_norm = "self_attn.k_layernorm.weight",
        },
        .notes = "fixture: LayerNorm with biases, parallel residual, partial rotary, fc1/fc2 with biases, lm_head bias. Phi-1 / 1.5 / 2.",
        .extra = extraPhi,
    },
    .{
        .model_type = "gpt_neox",
        .llama_cpp = "gptneox",
        .verified = true,
        .norm = .layer,
        .parallel_residual = true,
        .qkv = .heads_interleaved,
        .mlp = .dense,
        .activation = .gelu,
        .attention_bias = true,
        .names = neox_style_names,
        .notes = "fixture: head-interleaved query_key_value, parallel residual, rotary_pct, LayerNorm biases, embed_out. Pythia / GPT-NeoX.",
        .extra = extraNeox,
    },
    .{
        .model_type = "gpt2",
        .llama_cpp = "gpt2",
        .verified = true,
        .norm = .layer,
        .positional = .learned,
        .qkv = .concat,
        .mlp = .dense,
        .conv1d = true,
        .activation = .gelu_tanh,
        .attention_bias = true,
        .tie_word_embeddings = true,
        .names = .{
            .prefixes = &.{ "transformer.", "" },
            .embed = "{p}wte.weight",
            .pos_embed = "{p}wpe.weight",
            .final_norm = "{p}ln_f.weight",
            .layer = "{p}h.{i}.",
            .input_norm = &.{"ln_1.weight"},
            .pre_ff_norm = "ln_2.weight",
            .q = null,
            .k = null,
            .v = null,
            .qkv = "attn.c_attn.weight",
            .o = "attn.c_proj.weight",
            .gate = null,
            .up = "mlp.c_fc.weight",
            .down = "mlp.c_proj.weight",
        },
        .notes = "fixture: Conv1D ([in][out]) weights transposed on load and export, learned positions (wpe), fused c_attn, gelu_new, tied lm_head.",
    },
    .{
        .model_type = "falcon",
        .aliases = &.{"RefinedWebModel"},
        .llama_cpp = "falcon",
        .verified = true,
        .norm = .layer,
        .parallel_residual = true,
        .qkv = .concat,
        .mlp = .dense,
        .activation = .gelu,
        .tie_word_embeddings = true,
        .names = .{
            .prefixes = &.{ "transformer.", "" },
            .embed = "{p}word_embeddings.weight",
            .final_norm = "{p}ln_f.weight",
            .layer = "{p}h.{i}.",
            .input_norm = &.{ "input_layernorm.weight", "ln_attn.weight" },
            .mlp_norm = "ln_mlp.weight",
            .pre_ff_norm = null,
            .q = null,
            .k = null,
            .v = null,
            .qkv = "self_attention.query_key_value.weight",
            .o = "self_attention.dense.weight",
            .gate = null,
            .up = "mlp.dense_h_to_4h.weight",
            .down = "mlp.dense_4h_to_h.weight",
        },
        .notes = "fixture: multi-query fused qkv (7B layout), parallel attention with one LayerNorm. The 40B/180B grouped layout with ln_attn/ln_mlp and the ALiBi variant are implemented but unverified. Falcon-H1 (Mamba hybrid) is unsupported.",
        .extra = extraFalcon,
    },
    .{
        .model_type = "stablelm",
        .llama_cpp = "stablelm",
        .chat = "zephyr",
        .verified = true,
        .norm = .layer,
        .names = .{ .q_norm = "self_attn.q_layernorm.weight", .k_norm = "self_attn.k_layernorm.weight" },
        .notes = "fixture: LayerNorm with biases, partial rotary, qkv biases (use_qkv_bias). Parallel residual and qk_layernorm (StableLM 2 12B) are implemented but unverified.",
        .extra = extraStableLm,
    },
    .{
        .model_type = "internlm2",
        .llama_cpp = "internlm2",
        .chat = "chatml",
        .verified = true,
        .qkv = .grouped,
        .names = .{
            .embed = "{p}tok_embeddings.weight",
            .lm_head = &.{"output.weight"},
            .input_norm = &.{"attention_norm.weight"},
            .pre_ff_norm = "ffn_norm.weight",
            .q = null,
            .k = null,
            .v = null,
            .qkv = "attention.wqkv.weight",
            .o = "attention.wo.weight",
            .gate = "feed_forward.w1.weight",
            .up = "feed_forward.w3.weight",
            .down = "feed_forward.w2.weight",
        },
        .notes = "fixture: grouped wqkv layout, attention.wo, feed_forward.w1/w2/w3, output.weight.",
    },
    .{
        .model_type = "olmo2",
        .aliases = &.{"olmo3"},
        .llama_cpp = "olmo2",
        .chat = "olmo",
        .verified = true,
        .qk_norm = .full,
        .names = .{
            .input_norm = &.{},
            .post_attn_norm = "post_attention_layernorm.weight",
            .pre_ff_norm = null,
            .post_ff_norm = "post_feedforward_layernorm.weight",
            .q_norm = "self_attn.q_norm.weight",
            .k_norm = "self_attn.k_norm.weight",
        },
        .notes = "fixture: post-norms on the sublayer outputs (no input norm), q/k RMSNorm over the full projection.",
        .extra = extraOlmo2,
    },
    .{
        .model_type = "olmo",
        .llama_cpp = "olmo",
        .chat = "olmo",
        .verified = true,
        .norm = .none,
        .notes = "fixture: non-parametric LayerNorm (the default norm names exist only as slots), clip_qkv.",
    },
    .{
        .model_type = "cohere",
        .aliases = &.{"cohere2"},
        .llama_cpp = "command-r",
        .chat = "cohere",
        .verified = true,
        .norm = .layer,
        .parallel_residual = true,
        .tie_word_embeddings = true,
        .names = .{ .pre_ff_norm = null, .q_norm = "self_attn.q_norm.weight", .k_norm = "self_attn.k_norm.weight" },
        .notes = "fixture: LayerNorm without bias, parallel residual, logit_scale, tied embeddings, per-head q/k LayerNorm (use_qk_norm). cohere2 (Command R7B: sliding layers with RoPE, global layers without) is unverified.",
        .extra = extraCohere,
    },
    .{
        .model_type = "glm4",
        .aliases = &.{"glm"},
        .llama_cpp = "glm4",
        .chat = "glm4",
        .verified = true,
        .rope_style = .gptj,
        .mlp = .gated_fused,
        .attention_bias = true,
        .names = .{
            .post_attn_norm = "post_self_attn_layernorm.weight",
            .pre_ff_norm = "post_attention_layernorm.weight",
            .post_ff_norm = "post_mlp_layernorm.weight",
            .gate = null,
            .up = null,
            .gate_up = "mlp.gate_up_proj.weight",
        },
        .notes = "fixture: post_self_attn / post_mlp norms, fused gate_up_proj, interleaved half rotary, q/k/v biases. GLM-4 (0414) and the `glm` model_type (GLM-4-9B HF port).",
        .extra = extraGlm4,
    },
    .{
        .model_type = "chatglm",
        .llama_cpp = "chatglm",
        .chat = "glm4",
        .verified = true,
        .rope_style = .gptj,
        .qkv = .concat,
        .mlp = .gated_fused,
        .attention_bias = true,
        .names = .{
            .prefixes = &.{"transformer."},
            .embed = "{p}embedding.word_embeddings.weight",
            .final_norm = "{p}encoder.final_layernorm.weight",
            .lm_head = &.{"transformer.output_layer.weight"},
            .layer = "{p}encoder.layers.{i}.",
            .q = null,
            .k = null,
            .v = null,
            .qkv = "self_attention.query_key_value.weight",
            .o = "self_attention.dense.weight",
            .gate = null,
            .up = null,
            .gate_up = "mlp.dense_h_to_4h.weight",
            .down = "mlp.dense_4h_to_h.weight",
        },
        .notes = "fixture: ChatGLM3 / GLM-4 (remote-code layout): concatenated query_key_value with bias, fused dense_h_to_4h, interleaved half rotary with rope_ratio, output_layer.",
        .extra = extraChatGlm,
    },
    .{
        .model_type = "granite",
        .llama_cpp = "granite",
        .chat = "granite",
        .verified = true,
        .notes = "fixture: embedding, attention and residual multipliers, logits scaling. Granite 3.x dense (GraniteMoE is unsupported).",
        .extra = extraGranite,
    },
    .{
        .model_type = "deepseek_v2",
        .llama_cpp = "deepseek2",
        .chat = "deepseek",
        .verified = true,
        .names = deepseek_v3_names,
        .notes = "fixture: MLA with q_lora_rank (q_a/q_b) and without, softmax routing with group-limited top-k, shared experts, first_k_dense_replace, yarn with mscale. Checkpoints must be BF16/F16 (no FP8).",
        .extra = extraDeepseek,
    },
    .{
        .model_type = "deepseek_v3",
        .llama_cpp = "deepseek2",
        .chat = "deepseek",
        .verified = true,
        .names = deepseek_v3_names,
        .notes = "fixture: MLA, sigmoid routing with e_score_correction_bias, group-limited (noaux_tc) top-k, routed_scaling_factor, shared experts. Checkpoints must be BF16/F16 (no FP8).",
        .extra = extraDeepseek,
    },
    .{
        .model_type = "llama4",
        .aliases = &.{"llama4_text"},
        .llama_cpp = "llama4",
        .chat = "llama4",
        .verified = true,
        .rope_style = .gptj,
        .names = .{
            .gate = "feed_forward.gate_proj.weight",
            .up = "feed_forward.up_proj.weight",
            .down = "feed_forward.down_proj.weight",
            .router = "feed_forward.router.weight",
            .expert = "feed_forward.experts.{e}.",
            .fused_gate_up = &.{"feed_forward.experts.gate_up_proj"},
            .fused_down = &.{"feed_forward.experts.down_proj"},
            .shared_expert = "feed_forward.shared_expert.",
        },
        .notes = "fixture (text): top-1 sigmoid routing scaling the expert input, shared expert, transposed fused experts, no_rope_layers with attention temperature tuning, L2 qk norm, interleaved rope, dense layers with intermediate_size_mlp. Chunked attention runs as full attention.",
        .extra = extraLlama4,
    },
    .{
        .model_type = "gpt_oss",
        .llama_cpp = "gpt-oss",
        .chat = "harmony",
        .verified = true,
        .attention_bias = true,
        .names = .{
            .sinks = "self_attn.sinks",
            .router = "mlp.router.weight",
            .fused_gate_up = &.{"mlp.experts.gate_up_proj"},
            .fused_down = &.{"mlp.experts.down_proj"},
        },
        .notes = "fixture: attention sinks, alternating sliding layers, yarn, router bias with top-k softmax, interleaved fused experts with biases and the clamped swiglu. BF16 checkpoints only (MXFP4 must be dequantised first).",
        .extra = extraGptOss,
    },
    .{
        .model_type = "minicpm",
        .llama_cpp = "minicpm",
        .chat = "chatml",
        .verified = true,
        .tie_word_embeddings = true,
        .notes = "fixture: scale_emb, scale_depth residual scaling, dim_model_base logit scaling. MiniCPM 1/2 (MiniCPM3 with MLA is unsupported).",
        .extra = extraMiniCpm,
    },
    .{
        .model_type = "exaone",
        .llama_cpp = "exaone",
        .chat = "exaone",
        .verified = true,
        .names = .{
            .prefixes = &.{ "transformer.", "" },
            .embed = "{p}wte.weight",
            .final_norm = "{p}ln_f.weight",
            .layer = "{p}h.{i}.",
            .input_norm = &.{"ln_1.weight"},
            .pre_ff_norm = "ln_2.weight",
            .q = "attn.attention.q_proj.weight",
            .k = "attn.attention.k_proj.weight",
            .v = "attn.attention.v_proj.weight",
            .o = "attn.attention.out_proj.weight",
            .gate = "mlp.c_fc_0.weight",
            .up = "mlp.c_fc_1.weight",
            .down = "mlp.c_proj.weight",
        },
        .notes = "fixture: EXAONE 3.x tensor names (transformer.h, attn.attention, c_fc_0/c_fc_1).",
    },
    .{
        .model_type = "exaone4",
        .llama_cpp = "exaone4",
        .chat = "exaone",
        .verified = true,
        .qk_norm = .head,
        .names = .{
            .input_norm = &.{},
            .post_attn_norm = "post_attention_layernorm.weight",
            .pre_ff_norm = null,
            .post_ff_norm = "post_feedforward_layernorm.weight",
            .q_norm = "self_attn.q_norm.weight",
            .k_norm = "self_attn.k_norm.weight",
        },
        .notes = "fixture: post-norms, per-head q/k norm, hybrid sliding layers with RoPE and global layers without.",
        .extra = extraExaone4,
    },
    .{
        .model_type = "nemotron",
        .llama_cpp = "nemotron",
        .verified = true,
        .norm = .layer_1p,
        .mlp = .dense,
        .activation = .relu2,
        .names = .{ .gate = null, .up = "mlp.up_proj.weight", .down = "mlp.down_proj.weight" },
        .notes = "fixture: LayerNorm1p with bias, relu² dense MLP, partial rotary. Nemotron-H (Mamba hybrid) is unsupported.",
        .extra = extraNemotron,
    },
    .{
        .model_type = "smollm3",
        .llama_cpp = "smollm3",
        .chat = "chatml",
        .verified = true,
        .tie_word_embeddings = true,
        .notes = "fixture: llama layout with no_rope_layers.",
        .extra = extraSmolLm3,
    },
    .{
        .model_type = "bloom",
        .llama_cpp = "bloom",
        .verified = true,
        .norm = .layer,
        .positional = .alibi,
        .qkv = .heads_interleaved,
        .mlp = .dense,
        .activation = .gelu_tanh,
        .attention_bias = true,
        .tie_word_embeddings = true,
        .names = .{
            .prefixes = &.{ "transformer.", "" },
            .embed = "{p}word_embeddings.weight",
            .embed_norm = "{p}word_embeddings_layernorm.weight",
            .final_norm = "{p}ln_f.weight",
            .layer = "{p}h.{i}.",
            .q = null,
            .k = null,
            .v = null,
            .qkv = "self_attention.query_key_value.weight",
            .o = "self_attention.dense.weight",
            .gate = null,
            .up = "mlp.dense_h_to_4h.weight",
            .down = "mlp.dense_4h_to_h.weight",
        },
        .notes = "fixture: ALiBi, embedding LayerNorm, head-interleaved fused qkv with biases.",
    },
    .{
        .model_type = "opt",
        .llama_cpp = null,
        .verified = true,
        .norm = .layer,
        .positional = .learned,
        .mlp = .dense,
        .activation = .relu,
        .attention_bias = true,
        .tie_word_embeddings = true,
        .names = .{
            .prefixes = &.{ "model.decoder.", "decoder." },
            .pos_embed = "{p}embed_positions.weight",
            .final_norm = "{p}final_layer_norm.weight",
            .input_norm = &.{"self_attn_layer_norm.weight"},
            .pre_ff_norm = "final_layer_norm.weight",
            .o = "self_attn.out_proj.weight",
            .gate = null,
            .up = "fc1.weight",
            .down = "fc2.weight",
        },
        .notes = "fixture: learned positions with offset 2, ReLU, LayerNorm biases. Pre-norm variants only (OPT-350m's projection layers are unsupported).",
        .extra = extraOpt,
    },
    .{
        .model_type = "mpt",
        .llama_cpp = "mpt",
        .verified = true,
        .norm = .layer,
        .positional = .alibi,
        .qkv = .concat,
        .mlp = .dense,
        .activation = .gelu,
        .tie_word_embeddings = true,
        .names = .{
            .prefixes = &.{ "transformer.", "" },
            .embed = "{p}wte.weight",
            .final_norm = "{p}norm_f.weight",
            .layer = "{p}blocks.{i}.",
            .input_norm = &.{"norm_1.weight"},
            .pre_ff_norm = "norm_2.weight",
            .q = null,
            .k = null,
            .v = null,
            .qkv = "attn.Wqkv.weight",
            .o = "attn.out_proj.weight",
            .gate = null,
            .up = "ffn.up_proj.weight",
            .down = "ffn.down_proj.weight",
        },
        .notes = "fixture: ALiBi (alibi_bias_max), concatenated Wqkv, LayerNorm without bias, expansion_ratio.",
        .extra = extraMpt,
    },
    .{
        .model_type = "starcoder2",
        .llama_cpp = "starcoder2",
        .verified = true,
        .norm = .layer,
        .mlp = .dense,
        .activation = .gelu_tanh,
        .attention_bias = true,
        .names = .{ .gate = null, .up = "mlp.c_fc.weight", .down = "mlp.c_proj.weight" },
        .notes = "fixture: LayerNorm with biases, biased projections, c_fc/c_proj dense MLP, sliding window.",
        .extra = extraStarcoder2,
    },
    .{
        .model_type = "gpt_bigcode",
        .llama_cpp = "starcoder",
        .verified = true,
        .norm = .layer,
        .positional = .learned,
        .qkv = .concat,
        .mlp = .dense,
        .activation = .gelu_tanh,
        .attention_bias = true,
        .tie_word_embeddings = true,
        .names = .{
            .prefixes = &.{ "transformer.", "" },
            .embed = "{p}wte.weight",
            .pos_embed = "{p}wpe.weight",
            .final_norm = "{p}ln_f.weight",
            .layer = "{p}h.{i}.",
            .input_norm = &.{"ln_1.weight"},
            .pre_ff_norm = "ln_2.weight",
            .q = null,
            .k = null,
            .v = null,
            .qkv = "attn.c_attn.weight",
            .o = "attn.c_proj.weight",
            .gate = null,
            .up = "mlp.c_fc.weight",
            .down = "mlp.c_proj.weight",
        },
        .notes = "fixture: multi-query c_attn (nn.Linear layout), learned positions, LayerNorm biases. StarCoder 1 / SantaCoder.",
        .extra = extraGptBigcode,
    },
    .{
        .model_type = "baichuan",
        .llama_cpp = "baichuan",
        .verified = true,
        .qkv = .concat,
        .names = .{ .q = null, .k = null, .v = null, .qkv = "self_attn.W_pack.weight" },
        .notes = "fixture: fused W_pack (7B, RoPE). The 13B ALiBi variant (detected by model_max_length) is unverified. Needs a tokenizer.json (the SentencePiece-only checkpoints must be converted).",
        .extra = extraBaichuan,
    },
    .{
        .model_type = "qwen3_next",
        .llama_cpp = null,
        .chat = "chatml",
        .verified = true,
        .qk_norm = .head,
        .names = .{
            .q_norm = "self_attn.q_norm.weight",
            .k_norm = "self_attn.k_norm.weight",
            .lin_qkvz = "linear_attn.in_proj_qkvz.weight",
            .lin_ba = "linear_attn.in_proj_ba.weight",
            .lin_conv = "linear_attn.conv1d.weight",
            .lin_dt_bias = &.{"linear_attn.dt_bias"},
            .lin_a_log = &.{"linear_attn.A_log"},
            .lin_norm = "linear_attn.norm.weight",
            .lin_out = "linear_attn.out_proj.weight",
            .shared_expert = "mlp.shared_expert.",
            .shared_expert_gate = "mlp.shared_expert_gate.weight",
        },
        .notes = "fixture: Gated DeltaNet linear layers (fused projections, full_attention_interval), sigmoid-gated full attention with per-head q/k norms, partial rotary, softmax MoE with shared expert.",
        .extra = extraQwenHybrid,
    },
    .{
        .model_type = "qwen3_5",
        .aliases = &.{"qwen3_5_text"},
        .llama_cpp = null,
        .chat = "chatml",
        .verified = true,
        .qk_norm = .head,
        .names = .{
            .q_norm = "self_attn.q_norm.weight",
            .k_norm = "self_attn.k_norm.weight",
            .lin_qkv = "linear_attn.in_proj_qkv.weight",
            .lin_z = "linear_attn.in_proj_z.weight",
            .lin_b = "linear_attn.in_proj_b.weight",
            .lin_a = "linear_attn.in_proj_a.weight",
            .lin_conv = "linear_attn.conv1d.weight",
            .lin_dt_bias = &.{"linear_attn.dt_bias"},
            .lin_a_log = &.{"linear_attn.A_log"},
            .lin_norm = "linear_attn.norm.weight",
            .lin_out = "linear_attn.out_proj.weight",
        },
        .notes = "fixture: Gated DeltaNet linear layers (split projections, explicit layer_types), gated full attention, partial rotary. Multimodal wrappers keep the text config nested; vision weights pass through exports untouched.",
        .extra = extraQwenHybrid,
    },
    .{
        .model_type = "qwen3_5_moe",
        .aliases = &.{"qwen3_5_moe_text"},
        .llama_cpp = null,
        .chat = "chatml",
        .verified = true,
        .qk_norm = .head,
        .names = .{
            .q_norm = "self_attn.q_norm.weight",
            .k_norm = "self_attn.k_norm.weight",
            .lin_qkv = "linear_attn.in_proj_qkv.weight",
            .lin_z = "linear_attn.in_proj_z.weight",
            .lin_b = "linear_attn.in_proj_b.weight",
            .lin_a = "linear_attn.in_proj_a.weight",
            .lin_conv = "linear_attn.conv1d.weight",
            .lin_dt_bias = &.{"linear_attn.dt_bias"},
            .lin_a_log = &.{"linear_attn.A_log"},
            .lin_norm = "linear_attn.norm.weight",
            .lin_out = "linear_attn.out_proj.weight",
            .shared_expert = "mlp.shared_expert.",
            .shared_expert_gate = "mlp.shared_expert_gate.weight",
        },
        .notes = "fixture: split-projection linear layers with a swish output gate, fused softmax MoE with shared expert.",
        .extra = extraQwenHybrid,
    },
    .{
        .model_type = "glm4_moe",
        .aliases = &.{"glm4v_moe_text"},
        .llama_cpp = null,
        .chat = "glm4",
        .verified = true,
        .attention_bias = true,
        .names = .{
            .prefixes = &.{ "model.", "model.language_model.", "language_model.model.", "" },
            .q_norm = "self_attn.q_norm.weight",
            .k_norm = "self_attn.k_norm.weight",
            .router_correction_bias = "mlp.gate.e_score_correction_bias",
            .shared_expert = "mlp.shared_experts.",
        },
        .notes = "fixture: dense attention with per-head q/k norms (use_qk_norm), partial rotary, sigmoid MoE with correction bias and shared experts. The glm4v_moe image/video wrapper runs its text config; vision weights pass through exports untouched.",
        .extra = extraGlm4Moe,
    },
    .{
        .model_type = "glm_moe_dsa",
        .llama_cpp = null,
        .chat = "glm4",
        .verified = true,
        .names = .{
            .q_a = "self_attn.q_a_proj.weight",
            .q_a_norm = "self_attn.q_a_layernorm.weight",
            .q_b = "self_attn.q_b_proj.weight",
            .kv_a = "self_attn.kv_a_proj_with_mqa.weight",
            .kv_a_norm = "self_attn.kv_a_layernorm.weight",
            .kv_b = "self_attn.kv_b_proj.weight",
            .router_correction_bias = "mlp.gate.e_score_correction_bias",
            .shared_expert = "mlp.shared_experts.",
        },
        .notes = "fixture: MLA with a sparse indexer (run as dense attention: exact for short contexts), sigmoid MoE with correction bias and shared experts.",
        .extra = extraGlmMoeDsa,
    },
    .{
        .model_type = "kimi_linear",
        .llama_cpp = "kimi-linear",
        .verified = true,
        .linear = .kda,
        .names = .{
            .q_a = "self_attn.q_a_proj.weight",
            .q_a_norm = "self_attn.q_a_layernorm.weight",
            .q_b = "self_attn.q_b_proj.weight",
            .kv_a = "self_attn.kv_a_proj_with_mqa.weight",
            .kv_a_norm = "self_attn.kv_a_layernorm.weight",
            .kv_b = "self_attn.kv_b_proj.weight",
            .lin_q = "self_attn.q_proj.weight",
            .lin_k = "self_attn.k_proj.weight",
            .lin_v = "self_attn.v_proj.weight",
            .lin_conv = "self_attn.conv1d.weight",
            .lin_conv_split = &.{ "self_attn.q_conv1d.weight", "self_attn.k_conv1d.weight", "self_attn.v_conv1d.weight" },
            .lin_f_a = &.{ "self_attn.forget_gate.f_a_proj.weight", "self_attn.f_a_proj.weight" },
            .lin_f_b = &.{ "self_attn.forget_gate.f_b_proj.weight", "self_attn.f_b_proj.weight" },
            .lin_dt_bias = &.{ "self_attn.forget_gate.dt_bias", "self_attn.dt_bias" },
            .lin_a_log = &.{ "self_attn.forget_gate.A_log", "self_attn.A_log" },
            .lin_b = "self_attn.b_proj.weight",
            .lin_g_a = "self_attn.g_a_proj.weight",
            .lin_g_b = "self_attn.g_b_proj.weight",
            .lin_norm = "self_attn.o_norm.weight",
            .lin_out = "self_attn.o_proj.weight",
            .router_correction_bias = "mlp.gate.e_score_correction_bias",
            .shared_expert = "mlp.shared_experts.",
            .moe_alt = &kimi_linear_checkpoint_moe,
        },
        .notes = "fixtures: Kimi Delta Attention layers (per-channel decay from the low-rank forget gate, q/k/v short convolution, sigmoid-gated output norm) in the original checkpoint layout (linear_attn_config, split q/k/v convolutions, block_sparse_moe with w1/w3/w2 experts) and in the Hugging Face module layout (layer_types, fused conv1d, stacked experts); MLA full-attention layers without RoPE; sigmoid MoE with correction bias, top-2 group scores, routed_scaling_factor and shared experts. Kimi-Linear-48B-A3B. Needs BF16 weights and a tokenizer.json.",
        .extra = extraKimiLinear,
    },
    .{
        .model_type = "kimi_k25",
        .llama_cpp = "deepseek2",
        .chat = "deepseek",
        .verified = true,
        .names = deepseek_v3_names,
        .notes = "fixture: the Kimi K2.5 / K2.6 image-video wrapper (Kimi_K25ForConditionalGeneration) around a DeepSeek V3 text config (model_type kimi_k2 or deepseek_v3 under text_config): MLA, sigmoid routing with correction bias and group-limited top-k, shared experts, language_model prefix. The vision tower and projector pass through exports untouched. Needs BF16 weights (compressed-tensors INT4 must be dequantised first) and a tokenizer.json.",
        .extra = extraDeepseek,
    },
};

test "registry lookup and aliases" {
    try std.testing.expect(lookup("llama") != null);
    try std.testing.expectEqualStrings("gemma3", lookup("gemma3_text").?.model_type);
    try std.testing.expectEqualStrings("qwen2", lookup("qwen2_5_vl").?.model_type);
    try std.testing.expect(lookup("mamba") == null);
    // Every entry has a unique model_type.
    for (registry, 0..) |a, i| {
        for (registry[i + 1 ..]) |b| try std.testing.expect(!std.mem.eql(u8, a.model_type, b.model_type));
    }
}

test "parseConfig picks family knobs" {
    const arena_gpa = std.testing.allocator;
    var arena = std.heap.ArenaAllocator.init(arena_gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const neox = try parseConfig(a,
        \\{"model_type":"gpt_neox","hidden_size":64,"num_attention_heads":4,"num_hidden_layers":2,"rotary_pct":0.25,"rotary_emb_base":20000,"layer_norm_eps":1e-5,"vocab_size":100}
    );
    try std.testing.expectEqual(NormKind.layer, neox.norm);
    try std.testing.expect(neox.parallel_residual);
    try std.testing.expectEqual(@as(usize, 4), neox.rotary_dim);
    try std.testing.expectEqual(@as(f32, 20000), neox.rope_theta);
    try std.testing.expectEqual(QkvLayout.heads_interleaved, neox.qkv_layout);
    const gpt2 = try parseConfig(a,
        \\{"model_type":"gpt2","n_embd":64,"n_head":4,"n_layer":2,"n_positions":128,"vocab_size":100,"activation_function":"gelu_new"}
    );
    try std.testing.expectEqual(Positional.learned, gpt2.positional);
    try std.testing.expectEqual(@as(usize, 256), gpt2.intermediate_size);
    try std.testing.expect(!gpt2.rope_layers[0]);
    const ds = try parseConfig(a,
        \\{"model_type":"deepseek_v3","hidden_size":64,"num_attention_heads":4,"num_hidden_layers":3,"vocab_size":100,"q_lora_rank":32,"kv_lora_rank":16,"qk_nope_head_dim":8,"qk_rope_head_dim":4,"v_head_dim":8,"n_routed_experts":8,"num_experts_per_tok":2,"n_group":2,"topk_group":1,"scoring_func":"sigmoid","topk_method":"noaux_tc","first_k_dense_replace":1,"moe_layer_freq":1,"routed_scaling_factor":2.5,"norm_topk_prob":true,"moe_intermediate_size":16,"rope_scaling":{"type":"yarn","factor":40,"mscale":1.0,"mscale_all_dim":1.0,"original_max_position_embeddings":4096}}
    );
    try std.testing.expectEqual(@as(usize, 12), ds.head_dim);
    try std.testing.expectEqual(@as(usize, 8), ds.v_head_dim);
    try std.testing.expectEqual(@as(usize, 4), ds.rotary_dim);
    try std.testing.expect(!ds.moe_layers[0] and ds.moe_layers[1] and ds.moe_layers[2]);
    try std.testing.expectEqual(RouterScoring.sigmoid, ds.moe.scoring);
    try std.testing.expectEqual(TopkMethod.group_limited, ds.moe.topk_method);
    try std.testing.expect(ds.rope_scaling == .yarn);
    const ms = 0.1 * @log(@as(f32, 40)) + 1.0;
    try std.testing.expectApproxEqRel(ms * ms / @sqrt(@as(f32, 12)), ds.attention_scale, 1e-5);
    // Kimi Linear: original checkpoint spellings, 1-indexed layer lists, NoPE MLA layers.
    const kimi = try parseConfig(a,
        \\{"model_type":"kimi_linear","hidden_size":64,"num_attention_heads":4,"num_hidden_layers":8,"vocab_size":100,"q_lora_rank":null,"kv_lora_rank":16,"qk_nope_head_dim":8,"qk_rope_head_dim":4,"v_head_dim":8,"num_experts":8,"num_experts_per_token":2,"num_expert_group":2,"topk_group":1,"moe_renormalize":true,"routed_scaling_factor":2.446,"first_k_dense_replace":1,"moe_intermediate_size":16,"model_max_length":1024,"linear_attn_config":{"kda_layers":[1,2,3,5,6,7],"full_attn_layers":[4,8],"head_dim":16,"num_heads":2,"short_conv_kernel_size":4}}
    );
    try std.testing.expectEqual(LinearKind.kda, kimi.linear_kind);
    try std.testing.expect(kimi.has_linear and kimi.linear_layers[0] and !kimi.linear_layers[3] and kimi.linear_layers[6] and !kimi.linear_layers[7]);
    try std.testing.expectEqual(@as(usize, 2), kimi.linear_k_heads);
    try std.testing.expectEqual(@as(usize, 16), kimi.linear_v_dim);
    try std.testing.expectEqual(@as(usize, 4), kimi.linear_conv_kernel);
    try std.testing.expect(!kimi.rope_layers[3]);
    try std.testing.expectEqual(@as(usize, 12), kimi.head_dim);
    try std.testing.expectEqual(@as(usize, 1024), kimi.max_position_embeddings);
    try std.testing.expectEqual(@as(f32, 1e-5), kimi.rms_norm_eps);
    try std.testing.expectEqual(@as(usize, 2), kimi.num_experts_per_tok);
    try std.testing.expect(kimi.norm_topk_prob and kimi.moe.group_score_top2);
    try std.testing.expectEqual(RouterScoring.sigmoid, kimi.moe.scoring);
    try std.testing.expectEqual(@as(usize, 2), kimi.moe.n_group);
    try std.testing.expect(!kimi.moe_layers[0] and kimi.moe_layers[1] and kimi.moe_layers[7]);
    // Hugging Face spellings and the default 1-in-4 layer pattern.
    const kimi_hf = try parseConfig(a,
        \\{"model_type":"kimi_linear","hidden_size":64,"num_attention_heads":4,"num_hidden_layers":8,"vocab_size":100,"kv_lora_rank":16,"qk_nope_head_dim":8,"qk_rope_head_dim":4,"v_head_dim":8,"num_local_experts":8,"num_experts_per_tok":2,"moe_intermediate_size":16,"linear_num_heads":2,"linear_head_dim":16,"linear_conv_kernel_dim":4,"mlp_layer_types":["dense","dense","sparse","sparse","sparse","sparse","sparse","sparse"]}
    );
    try std.testing.expect(kimi_hf.linear_layers[0] and kimi_hf.linear_layers[3] and !kimi_hf.linear_layers[4] and kimi_hf.linear_layers[5]);
    try std.testing.expect(!kimi_hf.moe_layers[1] and kimi_hf.moe_layers[2]);
    try std.testing.expectEqual(@as(usize, 1), kimi_hf.moe.n_group);
    // Kimi K2.5: the text config nests a DeepSeek V3 layout under model_type kimi_k2.
    const k25 = try parseConfig(a,
        \\{"model_type":"kimi_k25","text_config":{"model_type":"kimi_k2","hidden_size":64,"num_attention_heads":4,"num_hidden_layers":3,"vocab_size":100,"q_lora_rank":32,"kv_lora_rank":16,"qk_nope_head_dim":8,"qk_rope_head_dim":4,"v_head_dim":8,"n_routed_experts":8,"num_experts_per_tok":2,"scoring_func":"sigmoid","topk_method":"noaux_tc","first_k_dense_replace":1,"moe_layer_freq":1,"moe_intermediate_size":16}}
    );
    try std.testing.expectEqualStrings("kimi_k25", k25.arch.model_type);
    try std.testing.expect(k25.mla != null and !k25.moe_layers[0] and k25.moe_layers[1]);
    try std.testing.expectEqual(RouterScoring.sigmoid, k25.moe.scoring);
}
