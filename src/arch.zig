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
const dequant = @import("dequant.zig");

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
    /// MiniMax lightning attention: `S = exp(-s_h) S + kᵀv`, `o = q S` per head,
    /// with a silu on the fused qkv projection, an RMSNorm and a sigmoid output gate.
    lightning,
};

/// Where the residual stream is normalised.
pub const ResidualLayout = enum {
    /// `x += f(norm(x))` (pre-norm; parallel or sequential).
    pre,
    /// MiniMax-01: `h = norm(x); x = α h + β f(h)` for both sublayers, so the
    /// residual stream itself is renormalised twice per layer.
    minimax,
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

/// Selective state-space block flavour of a family's Mamba layers.
pub const SsmKind = enum {
    none,
    /// Mamba2 (SSD): per-head scalar decay, grouped B/C, gated RMSNorm.
    mamba2,
    /// Mamba1 (Jamba): per-channel `[N]` decay from `A_log`, `x_proj` /
    /// `dt_proj` time-step path, RMS-normalised dt/B/C.
    mamba1,
};

/// Dimensions of a family's Mamba blocks (all Mamba layers of a model share them).
pub const SsmDims = struct {
    kind: SsmKind = .none,
    /// Mamba2: heads and head size; the inner size is `heads * head_dim`.
    heads: usize = 0,
    head_dim: usize = 0,
    /// Inner (expanded) size of the block.
    inter: usize = 0,
    /// State size per head (Mamba2) or per channel (Mamba1).
    state: usize = 0,
    /// Mamba2: groups of the B / C projections.
    groups: usize = 1,
    conv_kernel: usize = 4,
    /// Mamba1: rank of the time-step projection.
    dt_rank: usize = 0,
    /// Groups of the gated RMSNorm (Nemotron-H, Falcon-H1); 1 normalises the whole vector.
    norm_groups: usize = 1,
    /// Whether the gated RMSNorm exists (Falcon-H1 `mamba_rms_norm`); without
    /// it the scan output is only multiplied by `silu(gate)`.
    rms_norm: bool = true,
    /// Normalise before multiplying by the gate (Falcon-H1 `mamba_norm_before_gate`).
    norm_before_gate: bool = false,
    /// Clamp of the discretised time step (`time_step_limit`).
    dt_min: f32 = 0,
    dt_max: f32 = std.math.inf(f32),
    /// Activation after the causal convolution.
    act: tensor.Activation = .silu,
};

/// Falcon-H1 muP multipliers (all 1 for the other families).
pub const Multipliers = struct {
    ssm_in: f32 = 1,
    ssm_out: f32 = 1,
    attn_in: f32 = 1,
    attn_out: f32 = 1,
    /// Multiplies the key projection.
    key: f32 = 1,
    /// Multiplies the MLP gate before the activation, and the down projection output.
    mlp_gate: f32 = 1,
    mlp_down: f32 = 1,
    /// Per-section multipliers on the SSM input projection: gate, x, B, C, dt.
    ssm_proj: [5]f32 = .{ 1, 1, 1, 1, 1 },
};

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
    /// Experts (and the shared expert) are plain `down(act(up(x)))` MLPs
    /// without a gate projection (Nemotron-H).
    dense_experts: bool = false,
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
    /// Prefix of a Mamba block inside the layer (`mixer.`, `mamba.`). The
    /// tensor names below it are the Hugging Face ones shared by every
    /// family: `in_proj`, `conv1d`, `dt_bias`, `A_log`, `D`, `norm`,
    /// `out_proj` and, for Mamba1, `x_proj`, `dt_proj`, `dt_layernorm`,
    /// `b_layernorm`, `c_layernorm`.
    ssm: ?[]const u8 = null,
    /// MiniMax lightning attention: fused `[heads][q | k | v]` projection,
    /// sigmoid output gate, RMSNorm over `heads * head_dim` and the output
    /// projection (loaded into `Layer.o`).
    light_qkv: ?[]const u8 = null,
    light_gate: ?[]const u8 = null,
    light_norm: ?[]const u8 = null,
    light_out: ?[]const u8 = null,
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
    /// `shared_gate`/`shared_up`/`shared_down` (or the fused `shared_gate_up`)
    /// override them).
    shared_expert: ?[]const u8 = null,
    shared_expert_gate: ?[]const u8 = null,
    shared_gate: ?[]const u8 = null,
    shared_up: ?[]const u8 = null,
    /// Down projection of the shared expert, relative to `shared_expert`
    /// (default: `expert_down`).
    shared_down: ?[]const u8 = null,
    /// Fused `[2I][H]` gate/up tensor of the shared expert (GraniteMoeShared
    /// `input_linear`, MiniMax M3 `gate_up_proj`), relative to `shared_expert`.
    shared_gate_up: ?[]const u8 = null,
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
    /// Mamba block flavour of the family's state-space layers.
    ssm: SsmKind = .none,
    /// Every layer holds exactly one block (Mamba, attention, MLP or MoE)
    /// behind one norm (Mamba2, Nemotron-H); `layer_types` names the block.
    single_mixer: bool = false,
    /// Every layer runs a Mamba block and attention side by side on the same
    /// normalised input and sums them (Falcon-H1).
    parallel_ssm: bool = false,
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
    /// The checkpoint's `quantization_config` (dequantised on load, see dequant.zig).
    quant: dequant.QuantConfig = .{},
    linear_k_heads: usize,
    linear_k_dim: usize,
    linear_v_heads: usize,
    linear_v_dim: usize,
    linear_conv_kernel: usize,
    /// Per layer: true for Mamba (selective state-space) layers; their block
    /// dimensions are in `ssm`.
    ssm_layers: []bool,
    has_ssm: bool,
    ssm: SsmDims,
    /// Per layer: true if the layer has a full-attention block (false for
    /// Mamba / linear-attention layers and for the MLP-only layers of
    /// single-block families).
    attn_layers: []bool,
    /// Per layer: true if the layer has an MLP or MoE block.
    mlp_layers: []bool,
    /// Mamba block and attention run in parallel on the same input (Falcon-H1).
    parallel_ssm: bool,
    mult: Multipliers,
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
    /// Apply the q/k norm after the rotary embedding (HunYuan).
    qk_norm_after_rope: bool,
    residual_layout: ResidualLayout,
    /// MiniMax-01 residual scales `[full attention, linear attention, mlp]`
    /// as `[α (residual), β (sublayer output)]`.
    minimax_scales: [3][2]f32,
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

    /// Any layer keeps a recurrent state (Gated DeltaNet or Mamba).
    pub fn hasRecurrent(self: *const Config) bool {
        return self.has_linear or self.has_ssm;
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
        .{ "kimi_k3", "Kimi K3 (AttnRes is not implemented yet)" },
        .{ "kimi_k2", "Kimi K2 (use the kimi_k25 wrapper config or a deepseek_v3 config; the standalone kimi_k2 model_type is untested)" },
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

/// Rejects what no family can run, whatever its `model_type`: quantisation
/// formats ditch cannot decode (the supported ones become `Config.quant`),
/// expert storage dtypes it cannot read, and unimplemented layer maths.
fn rejectUnsupportedMath(top: std.json.ObjectMap, obj: std.json.ObjectMap) !dequant.QuantConfig {
    const quant = try dequant.parseQuantConfig(getObj(top, "quantization_config") orelse getObj(obj, "quantization_config"));
    if (getStr(obj, "expert_dtype")) |dt| {
        if (!dequant.expertDtypeSupported(dt)) {
            std.log.err("unsupported model: '{s}' expert dtype cannot be dequantised (bf16/f16/f32, fp8, mxfp4 and pack-quantized int4 are supported)", .{dt});
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
    return quant;
}

/// Replaces the bare `Infinity`, `-Infinity` and `NaN` literals Python's
/// json module writes (Mamba `time_step_limit`) with `null`, which JSON and
/// std.json accept; string contents are left alone.
pub fn sanitizeJson(arena: Allocator, text: []const u8) ![]const u8 {
    if (std.mem.indexOf(u8, text, "Infinity") == null and std.mem.indexOf(u8, text, "NaN") == null) return text;
    var out: std.Io.Writer.Allocating = .init(arena);
    var i: usize = 0;
    var in_string = false;
    while (i < text.len) {
        const ch = text[i];
        if (in_string) {
            try out.writer.writeByte(ch);
            if (ch == '\\' and i + 1 < text.len) {
                try out.writer.writeByte(text[i + 1]);
                i += 2;
                continue;
            }
            if (ch == '"') in_string = false;
            i += 1;
            continue;
        }
        if (ch == '"') {
            in_string = true;
        } else if (std.mem.startsWith(u8, text[i..], "-Infinity")) {
            try out.writer.writeAll("null");
            i += "-Infinity".len;
            continue;
        } else if (std.mem.startsWith(u8, text[i..], "Infinity")) {
            try out.writer.writeAll("null");
            i += "Infinity".len;
            continue;
        } else if (std.mem.startsWith(u8, text[i..], "NaN")) {
            try out.writer.writeAll("null");
            i += "NaN".len;
            continue;
        }
        try out.writer.writeByte(ch);
        i += 1;
    }
    return out.toOwnedSlice();
}

pub fn parseConfig(arena: Allocator, json_text: []const u8) !Config {
    var parsed = try std.json.parseFromSlice(std.json.Value, arena, try sanitizeJson(arena, json_text), .{});
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
        _ = try rejectUnsupportedMath(parsed.value.object, obj);
        var names: std.Io.Writer.Allocating = .init(arena);
        for (&registry, 0..) |*a, i| {
            if (i > 0) try names.writer.writeAll(", ");
            try names.writer.writeAll(a.model_type);
        }
        std.log.err("unsupported model_type: {s} (supported: {s})", .{ model_type, names.written() });
        return error.UnsupportedArchitecture;
    };
    const quant = try rejectUnsupportedMath(parsed.value.object, obj);
    // Nested attention config (MPT).
    const attn_cfg: std.json.ObjectMap = getObj(obj, "attn_config") orelse obj;

    const hidden = getIntAny(obj, &.{ "hidden_size", "n_embd", "d_model" }, 0);
    var heads = getIntAny(obj, &.{ "num_attention_heads", "n_head", "n_heads" }, 0);
    var layers = getIntAny(obj, &.{ "num_hidden_layers", "n_layer", "n_layers", "num_layers" }, 0);
    // Block-type lists (Nemotron-H `layers_block_type`) or a hybrid pattern
    // string define the depth when there is no explicit layer count.
    const block_types: ?std.json.Value = obj.get("layer_types") orelse obj.get("layers_block_type");
    if (layers == 0) {
        if (block_types) |bt| {
            if (bt == .array) layers = bt.array.items.len;
        } else if (getStr(obj, "hybrid_override_pattern")) |pat| layers = pat.len;
    }
    // A pure state-space model has no attention heads; the attention
    // dimensions are then placeholders (the KV cache stays empty).
    if (heads == 0 and arch.ssm != .none) heads = 1;
    if (hidden == 0 or heads == 0 or layers == 0) return error.InvalidConfig;
    var kv_heads = getIntAny(obj, &.{ "num_key_value_heads", "num_kv_heads", "n_head_kv", "multi_query_group_num" }, heads);
    if (attn_cfg.get("kv_n_heads") != null) kv_heads = getInt(attn_cfg, "kv_n_heads", heads);
    if (getBool(obj, "multi_query", false)) kv_heads = 1;
    var head_dim = getIntAny(obj, &.{ "head_dim", "attention_head_dim", "kv_channels" }, hidden / heads);
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
    // Newer configs nest the base frequency in `rope_parameters`.
    const rope_params: std.json.ObjectMap = getObj(obj, "rope_parameters") orelse obj;
    const rope_theta = getF32Any(obj, &.{ "rope_theta", "rotary_emb_base", "rope_base" }, getF32(rope_params, "rope_theta", 10000.0));
    var rotary_dim = head_dim;
    if (getNum(obj, "partial_rotary_factor")) |f| rotary_dim = @intFromFloat(@as(f64, @floatFromInt(head_dim)) * f);
    if (getNum(obj, "rotary_pct")) |f| rotary_dim = @intFromFloat(@as(f64, @floatFromInt(head_dim)) * f);
    // MiniMax checkpoints express partial rotary as an absolute `rotary_dim`.
    if (getNum(obj, "rotary_dim") != null) rotary_dim = @min(getInt(obj, "rotary_dim", head_dim), head_dim);
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
    // Mamba families: `linear_attention` (or the legacy `mamba`) marks a
    // state-space layer; single-block families also name `mlp` / `moe` layers.
    const ssm_layers = try arena.alloc(bool, layers);
    @memset(ssm_layers, false);
    const mlp_only = try arena.alloc(bool, layers);
    @memset(mlp_only, false);
    const moe_only = try arena.alloc(bool, layers);
    @memset(moe_only, false);
    var has_block_types = false;
    if (block_types) |lt| {
        if (lt == .array) {
            has_block_types = true;
            for (lt.array.items, 0..) |v, i| {
                if (i < layers and v == .string) {
                    const t = v.string;
                    if (std.mem.eql(u8, t, "sliding_attention") or std.mem.eql(u8, t, "chunked_attention")) {
                        sliding_layers[i] = true;
                    } else if (std.mem.eql(u8, t, "linear_attention") or std.mem.eql(u8, t, "mamba")) {
                        if (arch.ssm != .none) {
                            ssm_layers[i] = true;
                        } else if (std.mem.eql(u8, t, "mamba")) {
                            std.log.err("unsupported layer type 'mamba' in a {s} model", .{model_type});
                            return error.UnsupportedArchitecture;
                        } else {
                            linear_layers[i] = true;
                            has_linear = true;
                        }
                    } else if (std.mem.eql(u8, t, "full_attention") or std.mem.eql(u8, t, "attention")) {} else if (std.mem.eql(u8, t, "mlp") or std.mem.eql(u8, t, "moe")) {
                        if (!arch.single_mixer) {
                            std.log.err("unsupported layer type '{s}' in a {s} model", .{ t, model_type });
                            return error.UnsupportedArchitecture;
                        }
                        if (t[1] == 'l') mlp_only[i] = true else moe_only[i] = true;
                    } else if (std.mem.eql(u8, t, "deepseek_sparse_attention")) {
                        // Zhipu's sparse indexer selects a top-k over the keys; for
                        // the short calibration contexts ditch scores, the top-k
                        // covers the whole context, so dense attention is exact.
                        std.log.warn("deepseek_sparse_attention runs as dense attention (exact for short contexts)", .{});
                    } else if (std.mem.eql(u8, t, "minimax_m3_sparse")) {
                        // MiniMax Sparse Attention selects the top-k key blocks per
                        // query (plus the local blocks); with fewer blocks than the
                        // top-k in the context every block is selected and the
                        // layer is plain causal attention (see extraMiniMaxM3).
                        const prev = if (i > 0 and lt.array.items[i - 1] == .string) lt.array.items[i - 1].string else "";
                        if (!std.mem.eql(u8, prev, "minimax_m3_sparse")) {
                            std.log.warn("minimax_m3_sparse runs as dense attention (exact while the context fits index_topk_blocks blocks)", .{});
                        }
                    } else {
                        std.log.err("unsupported layer type '{s}' (only full/sliding/linear attention, Mamba, mlp and moe blocks are implemented)", .{t});
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

    // Which blocks each layer holds. Single-block families take the MLP /
    // MoE layers from the block-type list (the sparse-step rule above does
    // not apply to them); families without a list fill these in their hook.
    const attn_layers = try arena.alloc(bool, layers);
    const mlp_layers = try arena.alloc(bool, layers);
    for (0..layers) |i| {
        attn_layers[i] = !linear_layers[i] and !ssm_layers[i] and !mlp_only[i] and !moe_only[i];
        mlp_layers[i] = if (arch.single_mixer) (mlp_only[i] or moe_only[i]) else true;
        if (arch.single_mixer and has_block_types) moe_layers[i] = moe_only[i];
    }
    if (arch.parallel_ssm) {
        @memset(ssm_layers, true);
        @memset(attn_layers, true);
    }

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
        .quant = quant,
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
        .ssm_layers = ssm_layers,
        .has_ssm = false,
        .ssm = .{},
        .attn_layers = attn_layers,
        .mlp_layers = mlp_layers,
        .parallel_ssm = arch.parallel_ssm,
        .mult = .{},
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
        .qk_norm_after_rope = false,
        .residual_layout = .pre,
        .minimax_scales = .{ .{ 1, 1 }, .{ 1, 1 }, .{ 1, 1 } },
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
    // Hooks may mark linear / Mamba layers after the fact (Kimi Linear's
    // kda_layers, the Jamba periods): those layers hold no attention block
    // unless the family runs both side by side.
    for (0..layers) |i| c.attn_layers[i] = c.attn_layers[i] and !c.linear_layers[i] and (!c.ssm_layers[i] or c.parallel_ssm);
    for (c.ssm_layers) |s| c.has_ssm = c.has_ssm or s;
    if (c.has_ssm) {
        if (c.ssm.kind != arch.ssm) return error.InvalidConfig;
        const d = &c.ssm;
        if (d.inter == 0 or d.state == 0 or d.conv_kernel == 0 or (d.kind == .mamba2 and (d.heads == 0 or d.head_dim == 0 or d.groups == 0 or d.heads % d.groups != 0 or d.heads * d.head_dim != d.inter)) or (d.kind == .mamba1 and d.dt_rank == 0)) {
            std.log.err("inconsistent Mamba block dimensions in config.json", .{});
            return error.InvalidConfig;
        }
        if (d.norm_groups == 0 or d.inter % d.norm_groups != 0) return error.InvalidConfig;
    }
    if (c.has_linear and c.linear_kind != .lightning and (c.linear_k_heads == 0 or c.linear_k_dim == 0 or c.linear_v_heads == 0 or c.linear_v_dim == 0 or c.linear_conv_kernel == 0)) {
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

// --- Mamba families ---------------------------------------------------------

fn jsonF32(v: std.json.Value) ?f32 {
    return switch (v) {
        .integer => |i| @floatFromInt(i),
        .float => |f| @floatCast(f),
        else => null,
    };
}

/// Fills `out` from a JSON array of numbers (entries beyond the array keep their value).
fn floatList(obj: std.json.ObjectMap, key: []const u8, out: []f32) void {
    const v = obj.get(key) orelse return;
    if (v != .array) return;
    for (v.array.items, 0..) |item, i| {
        if (i >= out.len) break;
        if (jsonF32(item)) |f| out[i] = f;
    }
}

/// `time_step_limit`: a `[min, max]` clamp of the discretised time step. A
/// missing or null entry (JSON has no infinity) leaves that bound open.
fn dtLimit(obj: std.json.ObjectMap, d: *SsmDims) void {
    const v = obj.get("time_step_limit") orelse return;
    if (v != .array or v.array.items.len != 2) return;
    if (jsonF32(v.array.items[0])) |lo| d.dt_min = lo;
    if (jsonF32(v.array.items[1])) |hi| d.dt_max = hi;
}

/// Mamba2 dimensions from the `mamba_*` keys of the Falcon-H1 / Granite configs.
fn mambaDims(c: *Config, obj: std.json.ObjectMap) void {
    const d = &c.ssm;
    d.kind = .mamba2;
    const expand = getF32(obj, "mamba_expand", 2);
    d.inter = if (getNum(obj, "mamba_d_ssm")) |_| getInt(obj, "mamba_d_ssm", 0) else @intFromFloat(expand * @as(f32, @floatFromInt(c.hidden_size)));
    d.heads = getInt(obj, "mamba_n_heads", 128);
    // `mamba_d_head` may be the string "auto".
    d.head_dim = if (getNum(obj, "mamba_d_head")) |_| getInt(obj, "mamba_d_head", 0) else if (d.heads > 0) d.inter / d.heads else 0;
    d.state = getInt(obj, "mamba_d_state", 256);
    d.groups = getInt(obj, "mamba_n_groups", 1);
    d.conv_kernel = getInt(obj, "mamba_d_conv", 4);
    d.act = c.activation;
    dtLimit(obj, d);
}

fn extraMamba2(c: *Config, _: Allocator, obj: std.json.ObjectMap) !void {
    const d = &c.ssm;
    d.kind = .mamba2;
    d.heads = getInt(obj, "num_heads", 128);
    d.head_dim = getInt(obj, "head_dim", 64);
    d.inter = d.heads * d.head_dim;
    d.state = getInt(obj, "state_size", 128);
    d.groups = getInt(obj, "n_groups", 8);
    d.conv_kernel = getInt(obj, "conv_kernel", 4);
    d.act = c.activation;
    dtLimit(obj, d);
    const want: usize = @intFromFloat(getF32(obj, "expand", 2) * @as(f32, @floatFromInt(c.hidden_size)));
    if (want != d.inter) {
        std.log.err("mamba2: expand * hidden_size ({d}) must equal num_heads * head_dim ({d})", .{ want, d.inter });
        return error.InvalidConfig;
    }
    // No attention anywhere: placeholder attention dimensions keep the KV
    // cache (one float per position) out of the way.
    c.num_heads = 1;
    c.num_kv_heads = 1;
    c.head_dim = 1;
    c.v_head_dim = 1;
    c.rotary_dim = 0;
    c.positional = .none;
    @memset(c.ssm_layers, true);
    @memset(c.attn_layers, false);
    @memset(c.mlp_layers, false);
    @memset(c.moe_layers, false);
}

fn extraNemotronH(c: *Config, _: Allocator, obj: std.json.ObjectMap) !void {
    const d = &c.ssm;
    d.kind = .mamba2;
    d.heads = getInt(obj, "mamba_num_heads", 128);
    d.head_dim = getInt(obj, "mamba_head_dim", 64);
    d.inter = d.heads * d.head_dim;
    d.state = getInt(obj, "ssm_state_size", 128);
    d.groups = getInt(obj, "n_groups", 8);
    d.conv_kernel = getInt(obj, "conv_kernel", 4);
    // The gated RMSNorm normalises each B/C group of channels separately,
    // and the time step is floored at `time_step_min` (the chunked scan's
    // `dt_limit`; ditch applies it on every token).
    d.norm_groups = d.groups;
    d.dt_min = getF32(obj, "time_step_min", 0.001);
    d.act = .silu;
    if (getStr(obj, "mamba_hidden_act")) |a| d.act = parseActivation(a) orelse return error.UnsupportedArchitecture;
    // MLP and expert activation: `mlp_hidden_act` (relu²), not `hidden_act`.
    c.activation = .relu2;
    if (getStr(obj, "mlp_hidden_act")) |a| c.activation = parseActivation(a) orelse return error.UnsupportedArchitecture;
    // Attention layers carry no positional encoding.
    c.positional = .none;
    if (obj.get("layer_types") == null and obj.get("layers_block_type") == null) {
        const pat = getStr(obj, "hybrid_override_pattern") orelse {
            std.log.err("nemotron_h: config.json needs layers_block_type or hybrid_override_pattern", .{});
            return error.InvalidConfig;
        };
        if (pat.len != c.num_layers) {
            std.log.err("nemotron_h: hybrid_override_pattern has {d} entries for {d} layers", .{ pat.len, c.num_layers });
            return error.InvalidConfig;
        }
        for (pat, 0..) |ch, i| {
            c.ssm_layers[i] = ch == 'M';
            c.attn_layers[i] = ch == '*';
            c.mlp_layers[i] = ch == '-' or ch == 'E';
            c.moe_layers[i] = ch == 'E';
            if (ch != 'M' and ch != '*' and ch != '-' and ch != 'E') {
                std.log.err("nemotron_h: unknown block '{c}' in hybrid_override_pattern", .{ch});
                return error.UnsupportedArchitecture;
            }
        }
    }
    var any_moe = false;
    for (c.moe_layers) |m| any_moe = any_moe or m;
    if (any_moe and c.num_experts == 0) {
        std.log.err("nemotron_h: moe layers without n_routed_experts", .{});
        return error.InvalidConfig;
    }
    if (getNum(obj, "moe_latent_size") != null) {
        std.log.err("unsupported model: nemotron_h latent expert projections (moe_latent_size) are not implemented", .{});
        return error.UnsupportedArchitecture;
    }
    c.moe.scoring = .sigmoid;
    c.moe.topk_method = .group_limited;
    c.moe.n_group = @max(1, getInt(obj, "n_group", 1));
    c.moe.topk_group = @max(1, getInt(obj, "topk_group", 1));
    c.moe.routed_scaling_factor = getF32(obj, "routed_scaling_factor", 1.0);
    c.moe.dense_experts = true;
    if (c.moe.topk_method == .group_limited and c.num_experts > 0 and c.num_experts % c.moe.n_group != 0) return error.InvalidConfig;
}

fn extraFalconH1(c: *Config, _: Allocator, obj: std.json.ObjectMap) !void {
    mambaDims(c, obj);
    const d = &c.ssm;
    d.norm_groups = d.groups;
    d.rms_norm = getBool(obj, "mamba_rms_norm", false);
    d.norm_before_gate = getBool(obj, "mamba_norm_before_gate", true);
    const m = &c.mult;
    m.ssm_in = getF32(obj, "ssm_in_multiplier", 1);
    m.ssm_out = getF32(obj, "ssm_out_multiplier", 1);
    m.attn_in = getF32(obj, "attention_in_multiplier", 1);
    m.attn_out = getF32(obj, "attention_out_multiplier", 1);
    m.key = getF32(obj, "key_multiplier", 1);
    var mlp = [2]f32{ 1, 1 };
    floatList(obj, "mlp_multipliers", &mlp);
    m.mlp_gate = mlp[0];
    m.mlp_down = mlp[1];
    floatList(obj, "ssm_multipliers", &m.ssm_proj);
    c.embed_scale = getF32(obj, "embedding_multiplier", 1);
    c.logit_scale = getF32(obj, "lm_head_multiplier", 1);
    @memset(c.mlp_layers, true);
}

fn extraJamba(c: *Config, _: Allocator, obj: std.json.ObjectMap) !void {
    const d = &c.ssm;
    d.kind = .mamba1;
    d.inter = @intFromFloat(getF32(obj, "mamba_expand", 2) * @as(f32, @floatFromInt(c.hidden_size)));
    d.state = getInt(obj, "mamba_d_state", 16);
    d.conv_kernel = getInt(obj, "mamba_d_conv", 4);
    // `mamba_dt_rank` may be the string "auto": ceil(hidden / 16).
    d.dt_rank = if (getNum(obj, "mamba_dt_rank")) |_| getInt(obj, "mamba_dt_rank", 0) else (c.hidden_size + 15) / 16;
    d.act = c.activation;
    // Attention layers carry no positional encoding.
    c.positional = .none;
    if (obj.get("layer_types") == null and obj.get("layers_block_type") == null) {
        const period = @max(1, getInt(obj, "attn_layer_period", 8));
        const offset = getInt(obj, "attn_layer_offset", 4);
        for (0..c.num_layers) |i| {
            c.attn_layers[i] = i % period == offset;
            c.ssm_layers[i] = !c.attn_layers[i];
        }
    }
    // Every layer has an MLP; every `expert_layer_period`-th one is a mixture.
    if (c.num_experts <= 1) c.num_experts = 0;
    const eperiod = @max(1, getInt(obj, "expert_layer_period", 2));
    const eoffset = getInt(obj, "expert_layer_offset", 1);
    for (0..c.num_layers) |i| c.moe_layers[i] = c.num_experts > 0 and i % eperiod == eoffset;
    @memset(c.mlp_layers, true);
}

fn extraGraniteHybrid(c: *Config, arena: Allocator, obj: std.json.ObjectMap) !void {
    try extraGraniteMoe(c, arena, obj);
    mambaDims(c, obj);
    // Attention layers use RoPE only when the config asks for it.
    c.positional = .none;
    if (getStr(obj, "position_embedding_type")) |t| {
        if (std.mem.eql(u8, t, "rope")) {
            c.positional = .rope;
            @memset(c.rope_layers, true);
        }
    }
    if (obj.get("layer_types") == null and obj.get("layers_block_type") == null) {
        @memset(c.ssm_layers, true);
        @memset(c.attn_layers, false);
    }
    // Softmax over the top-k router logits equals a renormalised softmax top-k.
    c.norm_topk_prob = true;
    @memset(c.mlp_layers, true);
}

/// First element of an integer array config value (or the scalar), requiring
/// every element to agree: ditch has one value per model, not per layer.
fn uniformInt(obj: std.json.ObjectMap, key: []const u8, default: usize) !usize {
    const v = obj.get(key) orelse return default;
    switch (v) {
        .integer, .float => return getInt(obj, key, default),
        .array => |a| {
            if (a.items.len == 0) return default;
            var first: ?i64 = null;
            for (a.items) |item| {
                if (item != .integer) return error.InvalidConfig;
                if (first == null) first = item.integer;
                if (item.integer != first.?) {
                    std.log.err("per-layer values of '{s}' differ; only uniform values are supported", .{key});
                    return error.UnsupportedArchitecture;
                }
            }
            if (first.? < 0) return error.InvalidConfig;
            return @intCast(first.?);
        },
        else => return default,
    }
}

fn extraMiniMaxM2(c: *Config, _: Allocator, obj: std.json.ObjectMap) !void {
    // Sigmoid routing, top-k on the biased scores, renormalised routing weights.
    c.moe.scoring = .sigmoid;
    c.norm_topk_prob = true;
    c.num_experts_per_tok = getInt(obj, "num_experts_per_tok", 8);
    if (getNum(obj, "rope_theta") == null) c.rope_theta = 5000000.0;
    if (c.num_experts > 0) @memset(c.moe_layers, true);
}

fn extraMiniMax(c: *Config, _: Allocator, obj: std.json.ObjectMap) !void {
    // MiniMax-01 / M1: lightning (linear) attention on every layer but each
    // `full_attention` one, a renormalised residual stream with per-sublayer
    // α/β scales, softmax top-k routing renormalised over the selected experts.
    c.residual_layout = .minimax;
    c.norm_topk_prob = true;
    if (getNum(obj, "rope_theta") == null) c.rope_theta = 1000000.0;
    if (obj.get("layer_types") == null) {
        // Remote-code checkpoints carry `attn_type_list` (0 = lightning, 1 =
        // softmax attention); the Hugging Face default alternates, starting linear.
        if (obj.get("attn_type_list")) |al| {
            if (al == .array) for (al.array.items, 0..) |v, i| {
                if (i < c.num_layers) c.linear_layers[i] = (v == .integer and v.integer == 0);
            };
        } else {
            for (c.linear_layers, 0..) |*l, i| l.* = ((i + 1) % 2 != 0);
        }
        c.has_linear = false;
        for (c.linear_layers) |l| c.has_linear = c.has_linear or l;
    }
    const keys = [3][2][2][]const u8{
        .{ .{ "full_attn_alpha_factor", "layernorm_full_attention_alpha" }, .{ "full_attn_beta_factor", "layernorm_full_attention_beta" } },
        .{ .{ "linear_attn_alpha_factor", "layernorm_linear_attention_alpha" }, .{ "linear_attn_beta_factor", "layernorm_linear_attention_beta" } },
        .{ .{ "mlp_alpha_factor", "layernorm_mlp_alpha" }, .{ "mlp_beta_factor", "layernorm_mlp_beta" } },
    };
    for (&c.minimax_scales, keys) |*s, k| {
        s[0] = getF32Any(obj, &k[0], 1.0);
        s[1] = getF32Any(obj, &k[1], 1.0);
    }
    if (getBool(obj, "postnorm", false)) {
        std.log.err("unsupported model: MiniMax 'postnorm' residual layout", .{});
        return error.UnsupportedArchitecture;
    }
    if (c.num_experts > 0) @memset(c.moe_layers, true);
}

fn extraMiniMaxM3(c: *Config, _: Allocator, obj: std.json.ObjectMap) !void {
    // MiniMax M3 (transformers `minimax_m3_vl_text`): Gemma-style (1 + w)
    // norms everywhere, per-head q/k norm before a partial rotary, sigmoid
    // routing with a correction bias and a routed scaling factor, a shared
    // expert and the clamped gpt-oss swiglu in every MLP. Attention on
    // `minimax_m3_sparse` layers is MiniMax Sparse Attention: every query
    // scores the key blocks of `index_block_size` tokens with a small indexer
    // and attends to the best `index_topk_blocks` plus `index_local_blocks`
    // (the top-k always includes every block once the context is shorter
    // than `index_topk_blocks * index_block_size` tokens, 2048 with the
    // released config), so ditch runs those layers as dense causal attention
    // and never reads the indexer weights; longer prompts diverge from the
    // reference and are rejected below.
    c.moe.scoring = .sigmoid;
    c.norm_topk_prob = true;
    c.moe.routed_scaling_factor = getF32(obj, "routed_scaling_factor", 2.0);
    c.moe.swiglu = .{ .alpha = getF32(obj, "swiglu_alpha", 1.702), .limit = getF32(obj, "swiglu_limit", 7.0) };
    c.num_experts_per_tok = getInt(obj, "num_experts_per_tok", 4);
    if (getNum(obj, "rope_theta") == null) c.rope_theta = 5000000.0;
    if (getNum(obj, "rotary_dim") == null and getNum(obj, "partial_rotary_factor") == null) c.rotary_dim = @min(64, c.head_dim);
    // Experts use `intermediate_size`; dense layers `dense_intermediate_size`.
    c.moe_intermediate_size = c.intermediate_size;
    c.intermediate_size = getInt(obj, "dense_intermediate_size", c.intermediate_size);
    if (c.num_experts > 0) {
        @memset(c.moe_layers, true);
        if (obj.get("mlp_layer_types")) |ml| {
            if (ml == .array) for (ml.array.items, 0..) |v, i| {
                if (i < c.num_layers and v == .string) c.moe_layers[i] = !std.mem.eql(u8, v.string, "dense");
            };
        } else if (obj.get("moe_layer_freq")) |mf| {
            if (mf == .array) for (mf.array.items, 0..) |v, i| {
                if (i < c.num_layers) c.moe_layers[i] = !(v == .integer and v.integer == 0);
            };
        }
    }
    var block: usize = getInt(obj, "index_block_size", 128);
    var topk: usize = getInt(obj, "index_topk_blocks", 16);
    if (getObj(obj, "sparse_attention_config")) |sc| {
        block = getInt(sc, "sparse_block_size", block);
        topk = getInt(sc, "sparse_topk_blocks", topk);
    }
    var any_sparse = false;
    if (obj.get("layer_types")) |lt| {
        if (lt == .array) for (lt.array.items) |v| {
            if (v == .string and std.mem.eql(u8, v.string, "minimax_m3_sparse")) any_sparse = true;
        };
    } else if (getObj(obj, "sparse_attention_config")) |sc| {
        if (sc.get("sparse_attention_freq")) |f| {
            if (f == .array) for (f.array.items) |v| {
                if (v == .integer and v.integer != 0) any_sparse = true;
            };
        }
    }
    if (any_sparse) {
        // Dense attention is exact only while every key block is selected.
        std.log.warn("minimax_m3: sparse attention runs as dense attention, exact for prompts up to {d} tokens (index_block_size * index_topk_blocks)", .{block * topk});
    }
}

fn extraErnieMoe(c: *Config, _: Allocator, obj: std.json.ObjectMap) !void {
    // Softmax routing; the correction bias steers the choice only; the
    // selected probabilities are renormalised (clamped at moe_norm_min).
    c.norm_topk_prob = true;
    c.attention_bias = getBool(obj, "use_bias", false);
    if (getNum(obj, "rope_theta") == null) c.rope_theta = 500000.0;
    c.num_experts = getInt(obj, "moe_num_experts", c.num_experts);
    c.num_experts_per_tok = getInt(obj, "moe_k", getInt(obj, "num_experts_per_tok", 6));
    @memset(c.moe_layers, false);
    if (c.num_experts > 0) {
        const start = getInt(obj, "moe_layer_start_index", 1);
        var end = c.num_layers - 1;
        if (getNum(obj, "moe_layer_end_index")) |e| {
            if (e >= 0) end = @intFromFloat(e);
        }
        const interval = @max(getInt(obj, "moe_layer_interval", 1), 1);
        for (c.moe_layers, 0..) |*m, i| m.* = ((i + 1) % interval == 0) and i >= start and i <= end;
    }
}

fn extraHunyuanMoe(c: *Config, _: Allocator, obj: std.json.ObjectMap) !void {
    // Softmax top-k renormalised; per-head q/k RMSNorm applied after RoPE;
    // NTK-alpha "dynamic" rope scaling folded into the base.
    c.norm_topk_prob = true;
    c.qk_norm_after_rope = true;
    c.attention_bias = getBool(obj, "attention_bias", false);
    if (getBool(obj, "use_cla", false)) {
        std.log.err("unsupported model: HunYuan cross-layer attention (use_cla)", .{});
        return error.UnsupportedArchitecture;
    }
    c.num_experts = try uniformInt(obj, "num_experts", c.num_experts);
    c.num_experts_per_tok = try uniformInt(obj, "moe_topk", getInt(obj, "num_experts_per_tok", 1));
    c.moe_intermediate_size = try uniformInt(obj, "moe_intermediate_size", c.intermediate_size);
    @memset(c.moe_layers, c.num_experts > 0);
    const rs = getObj(obj, "rope_scaling") orelse getObj(obj, "rope_parameters");
    if (rs) |r| {
        const t = getStr(r, "rope_type") orelse getStr(r, "type") orelse "";
        if (std.mem.eql(u8, t, "dynamic")) {
            if (getNum(r, "alpha")) |alpha| {
                const d: f64 = @floatFromInt(c.head_dim);
                c.rope_theta = @floatCast(@as(f64, c.rope_theta) * std.math.pow(f64, alpha, d / (d - 2.0)));
            }
        }
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

fn extraGraniteMoe(c: *Config, arena: Allocator, obj: std.json.ObjectMap) !void {
    try extraGranite(c, arena, obj);
    // Top-k over the router logits, softmax over the selected ones (equal
    // to a renormalised softmax over every expert).
    c.norm_topk_prob = true;
    if (c.num_experts > 0) @memset(c.moe_layers, true);
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
        .notes = "fixture: multi-query fused qkv (7B layout), parallel attention with one LayerNorm. The 40B/180B grouped layout with ln_attn/ln_mlp and the ALiBi variant are implemented but unverified. Falcon-H1 is the `falcon_h1` entry.",
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
        .notes = "fixture: embedding, attention and residual multipliers, logits scaling. Granite 3.x dense (GraniteMoE and Granite 4.0 H `granitemoehybrid` have their own entries).",
        .extra = extraGranite,
    },
    .{
        .model_type = "deepseek_v2",
        .llama_cpp = "deepseek2",
        .chat = "deepseek",
        .verified = true,
        .names = deepseek_v3_names,
        .notes = "fixture: MLA with q_lora_rank (q_a/q_b) and without, softmax routing with group-limited top-k, shared experts, first_k_dense_replace, yarn with mscale. BF16/F16 and FP8 block-quantised checkpoints.",
        .extra = extraDeepseek,
    },
    .{
        .model_type = "deepseek_v3",
        .llama_cpp = "deepseek2",
        .chat = "deepseek",
        .verified = true,
        .names = deepseek_v3_names,
        .notes = "fixture: MLA, sigmoid routing with e_score_correction_bias, group-limited (noaux_tc) top-k, routed_scaling_factor, shared experts. BF16/F16 and FP8 block-quantised checkpoints (dequantised on load).",
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
        .notes = "fixture: attention sinks, alternating sliding layers, yarn, router bias with top-k softmax, interleaved fused experts with biases and the clamped swiglu. BF16 and MXFP4 checkpoints (experts dequantised on load).",
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
        .notes = "fixture: LayerNorm1p with bias, relu² dense MLP, partial rotary. Nemotron-H is the `nemotron_h` entry.",
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
        .model_type = "mamba2",
        .llama_cpp = "mamba2",
        .verified = true,
        .positional = .none,
        .ssm = .mamba2,
        .single_mixer = true,
        .names = .{
            .prefixes = &.{ "backbone.", "" },
            .embed = "{p}embeddings.weight",
            .final_norm = "{p}norm_f.weight",
            .input_norm = &.{"norm.weight"},
            .pre_ff_norm = null,
            .ssm = "mixer.",
        },
        .notes = "fixture: pure Mamba2 (SSD) blocks: in_proj split into gate / conv channels / dt, biased causal conv1d, grouped B/C, per-head decay, D skip, gated RMSNorm, out_proj; no attention, no MLP. Mamba-Codestral, state-spaces/mamba2-*-hf.",
        .extra = extraMamba2,
    },
    .{
        .model_type = "nemotron_h",
        .llama_cpp = "nemotron_h",
        .verified = true,
        .positional = .none,
        .mlp = .dense,
        .activation = .relu2,
        .ssm = .mamba2,
        .single_mixer = true,
        .names = .{
            .prefixes = &.{ "backbone.", "" },
            .embed = "{p}embeddings.weight",
            .final_norm = "{p}norm_f.weight",
            .input_norm = &.{"norm.weight"},
            .pre_ff_norm = null,
            .q = "mixer.q_proj.weight",
            .k = "mixer.k_proj.weight",
            .v = "mixer.v_proj.weight",
            .o = "mixer.o_proj.weight",
            .gate = null,
            .up = "mixer.up_proj.weight",
            .down = "mixer.down_proj.weight",
            .router = "mixer.gate.weight",
            .router_correction_bias = "mixer.gate.e_score_correction_bias",
            .expert = "mixer.experts.{e}.",
            .fused_gate_up = &.{},
            .fused_down = &.{},
            .shared_expert = "mixer.shared_experts.",
            .ssm = "mixer.",
        },
        .notes = "fixture: one block per layer from hybrid_override_pattern / layers_block_type (M: Mamba2 with grouped gated norm and dt floor, *: attention without positional encoding, -: relu² MLP, E: non-gated experts with sigmoid group-limited routing, correction bias and a shared expert). Nemotron-H, Nemotron 3 Nano (llama.cpp: nemotron_h_moe).",
        .extra = extraNemotronH,
    },
    .{
        .model_type = "falcon_h1",
        .llama_cpp = "falcon-h1",
        .chat = "chatml",
        .verified = true,
        .ssm = .mamba2,
        .parallel_ssm = true,
        .names = .{
            .prefixes = &.{ "model.", "" },
            .final_norm = "{p}final_layernorm.weight",
            .pre_ff_norm = "pre_ff_layernorm.weight",
            .gate = "feed_forward.gate_proj.weight",
            .up = "feed_forward.up_proj.weight",
            .down = "feed_forward.down_proj.weight",
            .ssm = "mamba.",
        },
        .notes = "fixtures: Mamba2 and attention in parallel on one input norm with the muP multipliers (ssm/attention in and out, key, mlp, per-section in_proj, embedding, lm_head), grouped gated RMSNorm with either gate order, and the norm-free variant. Both out projections are abliterated.",
        .extra = extraFalconH1,
    },
    .{
        .model_type = "jamba",
        .llama_cpp = "jamba",
        .verified = true,
        .positional = .none,
        .ssm = .mamba1,
        .names = .{
            .prefixes = &.{ "model.", "" },
            .final_norm = "{p}final_layernorm.weight",
            .pre_ff_norm = "pre_ff_layernorm.weight",
            .gate = "feed_forward.gate_proj.weight",
            .up = "feed_forward.up_proj.weight",
            .down = "feed_forward.down_proj.weight",
            .router = "feed_forward.router.weight",
            .expert = "feed_forward.experts.{e}.",
            .fused_gate_up = &.{"feed_forward.experts.gate_up_proj"},
            .fused_down = &.{"feed_forward.experts.down_proj"},
            .ssm = "mamba.",
        },
        .notes = "fixture: Mamba1 layers (in_proj, conv1d, x_proj with RMS-normalised dt/B/C, dt_proj, per-channel A_log, D, silu(z) gate) at attn_layer_period / offset, attention without positional encoding, softmax MoE at expert_layer_period / offset (separate expert tensors) and dense MLPs.",
        .extra = extraJamba,
    },
    .{
        .model_type = "minimax_m2",
        .llama_cpp = "minimax-m2",
        .verified = true,
        .qk_norm = .full,
        .names = .{
            .q_norm = "self_attn.q_norm.weight",
            .k_norm = "self_attn.k_norm.weight",
            .router = "block_sparse_moe.gate.weight",
            .router_correction_bias = "block_sparse_moe.e_score_correction_bias",
            .expert = "block_sparse_moe.experts.{e}.",
            .expert_gate = "w1.weight",
            .expert_up = "w3.weight",
            .expert_down = "w2.weight",
            .fused_gate_up = &.{},
            .fused_down = &.{},
        },
        .notes = "fixture: q/k RMSNorm over the whole projection, partial rotary (rotary_dim), sigmoid routing with e_score_correction_bias and renormalised top-k, Mixtral-style expert tensors.",
        .extra = extraMiniMaxM2,
    },
    .{
        .model_type = "minimax",
        .aliases = &.{ "minimax_text_01", "minimax_m1", "MiniMaxText01", "MiniMaxM1" },
        .llama_cpp = "minimax-01",
        .verified = true,
        .linear = .lightning,
        .names = .{
            .light_qkv = "self_attn.qkv_proj.weight",
            .light_gate = "self_attn.output_gate.weight",
            .light_norm = "self_attn.norm.weight",
            .light_out = "self_attn.out_proj.weight",
            .router = "block_sparse_moe.gate.weight",
            .expert = "block_sparse_moe.experts.{e}.",
            .expert_gate = "w1.weight",
            .expert_up = "w3.weight",
            .expert_down = "w2.weight",
            .fused_gate_up = &.{},
            .fused_down = &.{},
        },
        .notes = "fixture: lightning attention layers (silu qkv, per-head decay recurrence, RMSNorm, sigmoid output gate) alternating with softmax attention with partial rotary, the renormalised residual layout with α/β scales, softmax top-k MoE. MiniMax-Text-01 / M1 (`layer_types` or `attn_type_list`). The recurrence runs sequentially.",
        .extra = extraMiniMax,
    },
    .{
        .model_type = "minimax_m3_vl_text",
        .aliases = &.{ "minimax_m3_vl", "minimax_m3" },
        .llama_cpp = "minimax-m3",
        .verified = true,
        .norm = .rms_gemma,
        .qk_norm = .head,
        .mlp = .gated_fused,
        .names = .{
            .q_norm = "self_attn.q_norm.weight",
            .k_norm = "self_attn.k_norm.weight",
            .gate = null,
            .up = null,
            .gate_up = "mlp.gate_up_proj.weight",
            .down = "mlp.down_proj.weight",
            .router = "block_sparse_moe.gate.weight",
            .router_correction_bias = "block_sparse_moe.e_score_correction_bias",
            .expert = "block_sparse_moe.experts.{e}.",
            .expert_gate = "w1.weight",
            .expert_up = "w3.weight",
            .expert_down = "w2.weight",
            .fused_gate_up = &.{},
            .fused_down = &.{},
            .shared_expert = "block_sparse_moe.shared_experts.",
            .shared_gate_up = "gate_up_proj.weight",
            .shared_down = "down_proj.weight",
        },
        .notes = "fixture: (1 + w) norms, per-head (1 + w) q/k norm, partial rotary, dense layers (mlp_layer_types) with a fused gate_up and the clamped swiglu, sigmoid MoE with correction bias, routed scaling and a fused-gate_up shared expert, minimax_m3_sparse layers run as dense attention (exact while the context fits index_topk_blocks blocks); the indexer weights pass through exports untouched. The image tower of the VL wrapper is never executed.",
        .extra = extraMiniMaxM3,
    },
    .{
        .model_type = "ernie4_5_moe",
        .llama_cpp = "ernie4_5-moe",
        .verified = true,
        .rope_style = .gptj,
        .names = .{
            .router_correction_bias = "mlp.moe_statics.e_score_correction_bias",
            .shared_expert = "mlp.shared_experts.",
        },
        .notes = "fixture: interleaved rotary, softmax routing with the moe_statics correction bias and renormalised top-k, shared experts, moe_layer_start_index / interval, use_bias. ERNIE 4.5 MoE (PT checkpoints).",
        .extra = extraErnieMoe,
    },
    .{
        .model_type = "hunyuan_v1_moe",
        .aliases = &.{"hunyuan"},
        .llama_cpp = "hunyuan-moe",
        .verified = true,
        .qk_norm = .head,
        .names = .{
            .q_norm = "self_attn.query_layernorm.weight",
            .k_norm = "self_attn.key_layernorm.weight",
            .router = "mlp.gate.wg.weight",
            .shared_expert = "mlp.shared_mlp.",
        },
        .notes = "fixture: per-head q/k RMSNorm after RoPE, NTK-alpha dynamic rope base, softmax top-k renormalised, shared MLP, per-layer (uniform) expert counts. Hunyuan-A13B.",
        .extra = extraHunyuanMoe,
    },
    .{
        .model_type = "granitemoe",
        .aliases = &.{"granitemoeshared"},
        .llama_cpp = "granitemoe",
        .chat = "granite",
        .verified = true,
        .names = .{
            .router = "block_sparse_moe.router.layer.weight",
            .expert = "block_sparse_moe.experts.{e}.",
            .expert_down = "output_linear.weight",
            .fused_gate_up = &.{"block_sparse_moe.input_linear.weight"},
            .fused_down = &.{"block_sparse_moe.output_linear.weight"},
            .shared_expert = "shared_mlp.",
            .shared_gate_up = "input_linear.weight",
            .shared_down = "output_linear.weight",
        },
        .notes = "fixture: Granite multipliers, fused [E, 2I, H] / [E, H, I] expert tensors (input_linear / output_linear), top-k softmax routing. GraniteMoeShared adds the fused shared_mlp (implemented, covered by the granitemoehybrid fixture).",
        .extra = extraGraniteMoe,
    },
    .{
        .model_type = "granitemoehybrid",
        .llama_cpp = "granitehybrid",
        .chat = "granite",
        .verified = true,
        .positional = .none,
        .mlp = .gated_fused,
        .ssm = .mamba2,
        .names = .{
            .prefixes = &.{ "model.", "" },
            .gate = null,
            .up = null,
            .gate_up = "shared_mlp.input_linear.weight",
            .down = "shared_mlp.output_linear.weight",
            .router = "block_sparse_moe.router.layer.weight",
            .expert = "block_sparse_moe.experts.{e}.",
            .expert_down = "output_linear.weight",
            .fused_gate_up = &.{"block_sparse_moe.input_linear.weight"},
            .fused_down = &.{"block_sparse_moe.output_linear.weight"},
            .shared_expert = "shared_mlp.",
            .shared_gate_up = "input_linear.weight",
            .shared_down = "output_linear.weight",
            .ssm = "mamba.",
        },
        .notes = "fixtures: Mamba2 and attention layers from layer_types (also the legacy mamba / attention names; the attention-only layout too), embedding / attention / residual / logits multipliers, fused input_linear / output_linear routed experts plus the fused shared_mlp, or a dense shared_mlp when num_local_experts is 0, optional RoPE (position_embedding_type). Granite 4.0 H (tiny, small).",
        .extra = extraGraniteHybrid,
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
        .notes = "fixtures: Kimi Delta Attention layers (per-channel decay from the low-rank forget gate, q/k/v short convolution, sigmoid-gated output norm) in the original checkpoint layout (linear_attn_config, split q/k/v convolutions, block_sparse_moe with w1/w3/w2 experts) and in the Hugging Face module layout (layer_types, fused conv1d, stacked experts); MLA full-attention layers without RoPE; sigmoid MoE with correction bias, top-2 group scores, routed_scaling_factor and shared experts. Kimi-Linear-48B-A3B.",
        .extra = extraKimiLinear,
    },
    .{
        .model_type = "kimi_k25",
        .llama_cpp = "deepseek2",
        .chat = "kimi",
        .verified = true,
        .names = deepseek_v3_names,
        .notes = "fixture: the Kimi K2.5 / K2.6 image-video wrapper (Kimi_K25ForConditionalGeneration) around a DeepSeek V3 text config (model_type kimi_k2 or deepseek_v3 under text_config): MLA, sigmoid routing with correction bias and group-limited top-k, shared experts, language_model prefix. The vision tower and projector pass through exports untouched..",
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

test "parseConfig handles the MiniMax, HunYuan, ERNIE and Granite MoE keys" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    // Remote-code MiniMax-M1 keys: attn_type_list and layernorm_* scales.
    const m1 = try parseConfig(a,
        \\{"model_type":"minimax_m1","hidden_size":64,"num_attention_heads":4,"num_key_value_heads":2,"head_dim":16,"num_hidden_layers":4,"vocab_size":100,"rotary_dim":8,"attn_type_list":[0,0,0,1],"layernorm_full_attention_alpha":3.5,"layernorm_linear_attention_alpha":3.5,"layernorm_mlp_alpha":3.5,"layernorm_mlp_beta":1,"num_local_experts":8,"num_experts_per_tok":2}
    );
    try std.testing.expectEqual(LinearKind.lightning, m1.linear_kind);
    try std.testing.expectEqual(ResidualLayout.minimax, m1.residual_layout);
    try std.testing.expect(m1.linear_layers[0] and m1.linear_layers[2] and !m1.linear_layers[3] and m1.has_linear);
    try std.testing.expectEqual(@as(usize, 8), m1.rotary_dim);
    try std.testing.expectEqual(@as(f32, 3.5), m1.minimax_scales[1][0]);
    try std.testing.expectEqual(@as(f32, 1.0), m1.minimax_scales[2][1]);
    try std.testing.expect(m1.norm_topk_prob and m1.moe_layers[0]);
    // MiniMax M3: dense/sparse MLP schedule from moe_layer_freq, sparse attention accepted.
    const m3 = try parseConfig(a,
        \\{"model_type":"minimax_m3_vl","text_config":{"model_type":"minimax_m3_vl_text","hidden_size":64,"intermediate_size":16,"dense_intermediate_size":48,"num_attention_heads":4,"num_key_value_heads":2,"head_dim":16,"num_hidden_layers":3,"vocab_size":100,"num_local_experts":8,"num_experts_per_tok":2,"moe_layer_freq":[0,1,1],"sparse_attention_config":{"sparse_attention_freq":[0,1,1],"sparse_block_size":64,"sparse_topk_blocks":8},"layer_types":["full_attention","minimax_m3_sparse","minimax_m3_sparse"]}}
    );
    try std.testing.expectEqual(NormKind.rms_gemma, m3.norm);
    try std.testing.expect(!m3.moe_layers[0] and m3.moe_layers[1] and m3.moe_layers[2]);
    try std.testing.expectEqual(@as(usize, 48), m3.intermediate_size);
    try std.testing.expectEqual(@as(usize, 16), m3.moe_intermediate_size);
    try std.testing.expectEqual(RouterScoring.sigmoid, m3.moe.scoring);
    try std.testing.expect(m3.moe.swiglu != null);
    // HunYuan: per-layer lists, NTK-alpha base, q/k norm after RoPE.
    const hy = try parseConfig(a,
        \\{"model_type":"hunyuan_v1_moe","hidden_size":64,"num_attention_heads":4,"num_key_value_heads":2,"head_dim":16,"num_hidden_layers":2,"vocab_size":100,"intermediate_size":32,"num_experts":[8,8],"moe_topk":[3,3],"rope_theta":10000.0,"rope_scaling":{"type":"dynamic","alpha":1000.0}}
    );
    try std.testing.expectEqual(@as(usize, 8), hy.num_experts);
    try std.testing.expectEqual(@as(usize, 3), hy.num_experts_per_tok);
    try std.testing.expect(hy.qk_norm_after_rope and hy.moe_layers[1]);
    try std.testing.expectApproxEqRel(@as(f32, 10000.0 * std.math.pow(f32, 1000.0, 16.0 / 14.0)), hy.rope_theta, 1e-5);
    // ERNIE: moe_layer_start_index / interval schedule, interleaved rotary.
    const er = try parseConfig(a,
        \\{"model_type":"ernie4_5_moe","hidden_size":64,"num_attention_heads":4,"num_key_value_heads":2,"num_hidden_layers":5,"vocab_size":100,"moe_num_experts":8,"moe_k":2,"moe_layer_start_index":1,"moe_layer_end_index":3,"moe_layer_interval":2,"moe_intermediate_size":16}
    );
    try std.testing.expectEqual(RopeStyle.gptj, er.rope_style);
    try std.testing.expectEqualSlices(bool, &.{ false, true, false, true, false }, er.moe_layers);
    try std.testing.expectEqual(@as(usize, 2), er.num_experts_per_tok);
    // Granite hybrid: attention-only dense configs run, Mamba layers are refused.
    const gh = try parseConfig(a,
        \\{"model_type":"granitemoehybrid","hidden_size":64,"num_attention_heads":4,"num_hidden_layers":2,"vocab_size":100,"num_local_experts":0,"shared_intermediate_size":128,"layer_types":["attention","attention"],"position_embedding_type":null,"embedding_multiplier":12.0,"logits_scaling":8.0}
    );
    try std.testing.expectEqual(Positional.none, gh.positional);
    try std.testing.expect(!gh.rope_layers[0] and gh.num_experts == 0 and !gh.moe_layers[0]);
    try std.testing.expectEqual(MlpKind.gated_fused, gh.mlp);
    try std.testing.expectEqual(@as(f32, 12.0), gh.embed_scale);
    try std.testing.expectEqualStrings("granitemoe", lookup("granitemoeshared").?.model_type);
}

test "parseConfig: Mamba families" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    // Nemotron-H: one block per layer from the pattern, relu² MLPs, non-gated
    // experts, no positional encoding; `Infinity` (json.dump's spelling) is accepted.
    const nh = try parseConfig(a,
        \\{"model_type":"nemotron_h","hidden_size":64,"num_attention_heads":4,"num_key_value_heads":2,"head_dim":16,"hybrid_override_pattern":"M*-E","mamba_num_heads":8,"mamba_head_dim":16,"ssm_state_size":16,"n_groups":2,"conv_kernel":4,"n_routed_experts":4,"num_experts_per_tok":2,"moe_intermediate_size":32,"vocab_size":100,"time_step_limit":[0.001, Infinity],"note":"Infinity stays a string"}
    );
    try std.testing.expectEqual(@as(usize, 4), nh.num_layers);
    try std.testing.expectEqualSlices(bool, &.{ true, false, false, false }, nh.ssm_layers);
    try std.testing.expectEqualSlices(bool, &.{ false, true, false, false }, nh.attn_layers);
    try std.testing.expectEqualSlices(bool, &.{ false, false, true, true }, nh.mlp_layers);
    try std.testing.expectEqualSlices(bool, &.{ false, false, false, true }, nh.moe_layers);
    try std.testing.expect(nh.has_ssm and nh.ssm.kind == .mamba2);
    try std.testing.expectEqual(@as(usize, 128), nh.ssm.inter);
    try std.testing.expectEqual(@as(usize, 2), nh.ssm.norm_groups);
    try std.testing.expectEqual(@as(f32, 0.001), nh.ssm.dt_min);
    try std.testing.expect(nh.ssm.dt_max == std.math.inf(f32));
    try std.testing.expectEqual(tensor.Activation.relu2, nh.activation);
    try std.testing.expectEqual(tensor.Activation.silu, nh.ssm.act);
    try std.testing.expectEqual(Positional.none, nh.positional);
    try std.testing.expect(nh.moe.dense_experts and nh.moe.scoring == .sigmoid);
    // Jamba: attention and expert layers from period / offset, dt_rank "auto".
    const jm = try parseConfig(a,
        \\{"model_type":"jamba","hidden_size":64,"num_attention_heads":4,"num_hidden_layers":4,"attn_layer_period":2,"attn_layer_offset":1,"expert_layer_period":2,"expert_layer_offset":0,"num_experts":4,"num_experts_per_tok":2,"mamba_d_state":16,"mamba_d_conv":4,"mamba_expand":2,"mamba_dt_rank":"auto","vocab_size":100}
    );
    try std.testing.expectEqualSlices(bool, &.{ true, false, true, false }, jm.ssm_layers);
    try std.testing.expectEqualSlices(bool, &.{ false, true, false, true }, jm.attn_layers);
    try std.testing.expectEqualSlices(bool, &.{ true, false, true, false }, jm.moe_layers);
    try std.testing.expect(jm.ssm.kind == .mamba1);
    try std.testing.expectEqual(@as(usize, 128), jm.ssm.inter);
    try std.testing.expectEqual(@as(usize, 4), jm.ssm.dt_rank);
    // Falcon-H1: every layer runs both blocks; the multipliers are read.
    const fh = try parseConfig(a,
        \\{"model_type":"falcon_h1","hidden_size":64,"num_attention_heads":4,"num_hidden_layers":2,"mamba_d_ssm":128,"mamba_n_heads":8,"mamba_d_head":"auto","mamba_n_groups":2,"mamba_d_state":16,"mamba_d_conv":4,"mamba_rms_norm":true,"key_multiplier":0.5,"ssm_multipliers":[1,2,3,4,5],"mlp_multipliers":[1.5,0.5],"lm_head_multiplier":0.25,"vocab_size":100}
    );
    try std.testing.expect(fh.parallel_ssm and fh.ssm_layers[1] and fh.attn_layers[1] and fh.mlp_layers[1]);
    try std.testing.expectEqual(@as(usize, 16), fh.ssm.head_dim);
    try std.testing.expectEqual(@as(f32, 0.5), fh.mult.key);
    try std.testing.expectEqual(@as(f32, 5), fh.mult.ssm_proj[4]);
    try std.testing.expectEqual(@as(f32, 0.5), fh.mult.mlp_down);
    try std.testing.expectEqual(@as(f32, 0.25), fh.logit_scale);
    // Pure Mamba2 has no attention heads at all.
    const m2 = try parseConfig(a,
        \\{"model_type":"mamba2","hidden_size":64,"num_hidden_layers":2,"num_heads":8,"head_dim":16,"state_size":16,"n_groups":2,"expand":2,"conv_kernel":4,"vocab_size":100}
    );
    try std.testing.expect(!m2.attn_layers[0] and !m2.mlp_layers[0] and m2.ssm_layers[0]);
    try std.testing.expectEqual(@as(usize, 1), m2.num_kv_heads);
    try std.testing.expect(!m2.rope_layers[0]);
    // Granite: RoPE only on request.
    const gh = try parseConfig(a,
        \\{"model_type":"granitemoehybrid","hidden_size":64,"num_attention_heads":4,"num_hidden_layers":2,"layer_types":["mamba","attention"],"position_embedding_type":"rope","num_local_experts":0,"mamba_n_heads":8,"mamba_d_state":16,"vocab_size":100}
    );
    try std.testing.expect(gh.ssm_layers[0] and gh.attn_layers[1] and gh.rope_layers[1] and !gh.moe_layers[0]);
    try std.testing.expectEqual(@as(usize, 16), gh.ssm.head_dim);
}

test "sanitizeJson replaces the non-JSON float literals outside strings" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    try std.testing.expectEqualStrings("{\"a\": [0, null, null], \"b\": \"Infinity\\\"NaN\", \"c\": null}", try sanitizeJson(a, "{\"a\": [0, Infinity, -Infinity], \"b\": \"Infinity\\\"NaN\", \"c\": NaN}"));
    const plain = "{\"x\": 1}";
    try std.testing.expect((try sanitizeJson(a, plain)).ptr == plain.ptr);
}
