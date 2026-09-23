//! Architecture registry: one descriptor per Hugging Face `model_type`
//! capturing everything the loader and the forward pass need to know about a
//! family (tensor-name templates, norm and residual layout, attention and MLP
//! layouts, positional encoding, MoE routing) plus the generalised `Config`
//! and its parser. Adding a family is a table entry here and, only when the
//! family needs a genuinely new computation, a small code path in model.zig
//! or moe.zig.
//!
//! Every entry documents what was verified against a NumPy reference fixture
//! (`tools/make_fixture.py`, `src/model_test.zig`). Every entry has such a
//! fixture; `verified = false` marks a family implemented from the Hugging
//! Face reference implementation but not yet checked against one.

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
    /// Non-parametric RMSNorm (nanochat).
    rms_none,
};

/// Which pairs of coordinates a rotary embedding rotates.
pub const RopeStyle = enum {
    /// `(x[i], x[i + d/2])` (Hugging Face `rotate_half`).
    neox,
    /// `(x[2i], x[2i + 1])` (GPT-J, GLM, Llama 4, DeepSeek, Helium).
    gptj,
};

/// `sinusoidal`: the fairseq / XGLM table `[sin(p·f) | cos(p·f)]` computed
/// from the model size, indexed at `position + position_offset`.
pub const Positional = enum { rope, learned, alibi, none, sinusoidal };

/// Row layout of a fused query/key/value projection `[rows][hidden]`.
pub const QkvLayout = enum {
    separate,
    /// `[q (all heads) | k (all kv heads) | v (all kv heads)]`.
    concat,
    /// `[head][q | k | v]` (GPT-NeoX, BLOOM, Falcon multi-head).
    heads_interleaved,
    /// `[kv group][q heads of the group | k | v]` (InternLM2, Falcon new decoder).
    grouped,
    /// `[block][q | v | k]` over `Config.qkv_mp` tensor-parallel blocks, each
    /// holding `heads / qkv_mp` heads (CodeGen).
    mp_blocks,
};

/// Sigmoid or softplus gate on the attention output, computed from the
/// layer input by a separate projection (AFMoE, Laguna): `[heads * head_dim]`
/// rows gate every coordinate, `[heads]` rows gate per head.
pub const AttnGate = enum { none, sigmoid, softplus };

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

pub const RouterScoring = enum {
    softmax,
    sigmoid,
    /// `sqrt(softplus(x))` (DeepSeek V4).
    sqrtsoftplus,
};

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
    /// Epsilon of `q_a_layernorm` and `kv_a_layernorm`. DeepSeek's own code,
    /// and transformers after it, build both with the RMSNorm default 1e-6
    /// whatever `rms_norm_eps` says (Kimi K2 / K2.5, Kimi-Linear, GLM-4.7-Flash
    /// and GLM-5.3 set 1e-5); GLM-5.3-Flash passes `rms_norm_eps`.
    latent_norm_eps: f32 = 1e-6,
};

/// Position-dependent query scaling `1 + attn_scale * log(floor((pos + offset) / floor_scale) + 1)`:
/// Llama 4 (offset 1, layers without RoPE only), Mistral 4 and Ministral 3
/// (`llama_4_scaling_beta`, offset 0, every layer).
pub const AttnTemperature = struct { floor_scale: f32, attn_scale: f32, offset: f32 = 1.0, all_layers: bool = false };

/// Rotary table of the local (sliding) layers when it differs from the global
/// one (Gemma 3 family, Granite SWA, per-layer-type `rope_parameters`): base,
/// rotated coordinates and the frequency denominator.
pub const LocalRope = struct { theta: f32, rotary_dim: usize, freq_dim: usize };

/// gpt-oss gated activation: `(clamp(up) + 1) * clamp(gate) * sigmoid(alpha * gate)`.
pub const Swiglu = struct { alpha: f32, limit: f32 };

/// Kimi K3 SiTU gated activation: `beta · tanh(gate / beta) · sigmoid(gate) · up'`
/// with `up' = linear_beta · tanh(up / linear_beta)` when `linear_beta` is set.
pub const Situ = struct { beta: f32, linear_beta: ?f32 };
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
    /// Multiplies the value projection (MiMo V2 `attention_value_scale`).
    value: f32 = 1,
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
    /// SiTU in every gated MLP (dense layers, routed and shared experts; Kimi K3).
    situ: ?Situ = null,
    /// Experts (and the shared expert) are plain `down(act(up(x)))` MLPs
    /// without a gate projection (Nemotron-H).
    dense_experts: bool = false,
    /// Clamp on the expert pre-activations before the gated activation: the
    /// gate from above, the up projection on both sides (DeepSeek V4).
    swiglu_limit: ?f32 = null,
    /// Divisor of the router logits before scoring (DeepSeek V4.1 `gate_temp`).
    gate_temp: f32 = 1.0,
    /// Routing weights are renormalised with a `+1e-20` floor on their sum (DeepSeek V4).
    norm_eps_floor: bool = false,
    /// Group scores are the sum of the two best selection scores even without
    /// a correction bias (Kimi Linear's router).
    group_score_top2: bool = false,
    /// `tanh` softcap on the router logits before scoring (Laguna).
    router_softcap: ?f32 = null,
};

/// Compressed-KV branch of a DeepSeek V4 / V4.1 attention layer.
pub const CompressBranch = enum {
    /// Sliding window only.
    none,
    /// V4 compressed sparse attention: two overlapping series (Ca/Cb) pooled
    /// over `2 * ratio` slots with a position bias, then a Lightning Indexer.
    csa,
    /// V4 heavily compressed attention: one series pooled over `ratio` slots.
    hca,
    /// V4.1 shared compressed attention (CSA2): the group's KV source pools
    /// `ratio` tokens with a gate (no position bias, no overlap); a ratio of
    /// 1 is a plain per-token latent.
    shared,
};

/// Engram conditional memory (DeepSeek V4.1): n-gram hash tables gated into
/// the residual streams at `layer_ids`.
pub const Engram = struct {
    layer_ids: []const usize,
    /// Rows of every layer's table (`engram_num_embeddings`).
    num_embeddings: []const usize,
    max_ngram: usize,
    n_heads: usize,
    head_dim: usize,
    pad_id: u32,
    /// Size of the tokenizer-normalised vocabulary the hashes run on; the
    /// value derived from the tokenizer must match it.
    compressed_vocab_size: usize,
    /// Bucket sizes are the primes above this (per (n-gram size, head)).
    vocab_size: usize,
};

/// How a hyper-connection family mixes its `hc_mult` residual streams around
/// every block (hyper.zig).
pub const HyperKind = enum {
    /// Manifold-constrained hyper-connections (DeepSeek V4, GLM-5.3-Flash):
    /// one projection of the RMS-normalised flattened streams gives the
    /// collapse weights, the expansion weights and a Sinkhorn-projected
    /// stream mixing matrix; the site's own collapse weights are used.
    mhc,
    /// mHC in DeepSeek V4.1's single pass: the collapse weights of the
    /// previous site are used, the site's own travel to the next one.
    mhc_single_pass,
    /// Qwen4-Exp gated residual: per-stream (1 + w) norms, a low-rank
    /// sigmoid input mixer averaged over the streams, and a sigmoid
    /// injection weight per stream on the block output (no stream mixing).
    gated,
};

/// The final collapse of the streams before the last norm.
pub const HyperHead = enum {
    /// `hc_head`: sigmoid weights from a projection of the normalised streams (DeepSeek V4).
    weighted,
    /// The collapse weights of the last site (DeepSeek V4.1).
    previous_pre,
    /// The unweighted mean of the streams (GLM-5.3-Flash).
    mean,
    /// The gated input mixer without injection weights (Qwen4-Exp `hyper_connection_mixer`).
    gated_mixer,
};

pub const Hyper = struct {
    kind: HyperKind,
    head: HyperHead,
    sinkhorn_iters: usize = 20,
    eps: f32 = 1e-6,
    /// Rank of the gated input mixer (`hc_lowrank`).
    lowrank: usize = 0,
};

/// A sparse-attention indexer run as its dense equivalent: the reference
/// scores blocks of `block` consecutive keys and keeps the best `max_blocks`
/// complete blocks plus the incomplete tail, so every reachable key is
/// selected while `(pos + 1) / block <= max_blocks` for every query
/// position; longer contexts are refused rather than approximated.
pub const IndexBound = struct {
    block: usize,
    max_blocks: usize,

    /// Whether dense attention is still exactly the reference for a query at
    /// (zero-based) position `pos`: every complete key block it can reach is
    /// one the indexer would have selected.
    pub fn fits(self: IndexBound, pos: usize) bool {
        return (pos + 1) / self.block <= self.max_blocks;
    }
};

/// Qwen4-Exp per-layer n-gram embeddings (PLE, qwen4_exp.zig): hashed
/// 2..`ngram_size`-grams of the token ids, one embedding head per
/// (n-gram size, head) pair in a prime-sized bucket range, gated into every
/// residual stream and followed by a dilated depthwise convolution.
pub const NgramPle = struct {
    /// Zero-based layer indices (sorted).
    layer_ids: []const usize,
    embed_dim: usize,
    conv_kernel: usize,
    ngram_size: usize,
    heads_per_ngram: usize,
    /// Bucket sizes are consecutive primes from this value on.
    vocab_base: u64,
    divisible_by: usize,
    seed: u64,
    /// `vocab_size` of the config (the hash multipliers derive from it).
    vocab_size: u64,
    eos_id: u32,
};

/// DeepSeek V4 / V4.1 family layout: manifold-constrained hyper-connections
/// (`hc_mult` residual streams), low-rank query and grouped output
/// projections, shared-KV sliding-window attention with sinks and a
/// compressed-KV branch, hash-routed or n-gram-memory layers.
pub const DsV4 = struct {
    /// DeepSeek V4.1: single-pass hyper-connections, CSA2 KV sharing, QAT
    /// fake quantisation, engram layers.
    v41: bool,
    q_lora_rank: usize,
    o_groups: usize,
    o_lora_rank: usize,
    /// Per layer: tokens pooled into one compressed entry (0 = no compressed branch).
    compress_ratio: []const usize,
    branch: []const CompressBranch,
    /// Per layer: the layer whose compressed cache this layer attends to (the
    /// layer itself for V4; the group's KV source for V4.1).
    kv_source: []const ?usize,
    /// Lightning Indexer top-k: the dense equivalent is exact while every
    /// reachable compressed entry fits (`(pos + 1) / ratio <= index_topk`).
    index_topk: usize,
    index_n_heads: usize,
    index_head_dim: usize,
    /// V4.1 two-level top-k: at most `candidate_topk_blocks` blocks of
    /// `candidate_block_size` entries are reachable once the candidate
    /// source layer runs (null: disabled).
    candidate_source: ?usize,
    candidate_topk_blocks: usize,
    candidate_block_size: usize,
    /// RoPE of the compressed branches (queries and keys of CSA/HCA layers).
    compress_rope_theta: f32,
    compress_rope_scaling: RopeScaling,
    /// Per layer: expert selection comes from the `tid2eid` table of the
    /// input token instead of the router scores (V4 `hash_moe`).
    hash_moe_layers: []const bool,
    hc_mult: usize,
    hc_sinkhorn_iters: usize,
    hc_eps: f32,
    /// V4.1 quantisation-aware training semantics: the window KV is rounded
    /// to block-scaled FP8 and the compressed latents to block-scaled FP4
    /// even when the weights are not quantised.
    fake_quant: bool,
    engram: ?Engram,
};

/// Tensor-name templates. `{p}` is the model prefix, `{i}` the layer index and
/// `{e}` an expert index; layer-level names are relative to `layer`. Biases
/// are always optional and named by replacing a trailing `.weight` with
/// `.bias`; norm biases likewise.
pub const Names = struct {
    prefixes: []const []const u8 = &.{ "model.", "language_model.model.", "model.language_model.", "thinker.model.", "language_model.", "" },
    embed: []const u8 = "{p}embed_tokens.weight",
    /// Learned absolute position table `[positions][hidden]`.
    pos_embed: ?[]const u8 = null,
    /// LayerNorm applied to the embeddings (BLOOM).
    embed_norm: ?[]const u8 = null,
    /// Final norm (null for a family without one: Qwen4-Exp's stream mixer already normalises).
    final_norm: ?[]const u8 = "{p}norm.weight",
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
    /// `post_ff_norm` exists only on some layers (A.X-K1 keeps it in the MoE block).
    post_ff_norm_optional: bool = false,
    /// Separate MLP input norm in parallel-residual layouts (Falcon 40B `ln_mlp`).
    mlp_norm: ?[]const u8 = null,
    q_norm: ?[]const u8 = null,
    k_norm: ?[]const u8 = null,
    /// Norm on the attention output before the output projection and on the
    /// activated MLP intermediate before the down projection (BitNet).
    attn_sub_norm: ?[]const u8 = null,
    ffn_sub_norm: ?[]const u8 = null,
    /// xIELU activation parameters (Apertus), stored as `log(expm1(alpha))`.
    xielu_alpha_p: ?[]const u8 = null,
    xielu_alpha_n: ?[]const u8 = null,
    q: ?[]const u8 = "self_attn.q_proj.weight",
    k: ?[]const u8 = "self_attn.k_proj.weight",
    v: ?[]const u8 = "self_attn.v_proj.weight",
    qkv: ?[]const u8 = null,
    o: []const u8 = "self_attn.o_proj.weight",
    sinks: ?[]const u8 = null,
    /// Alternative name of the sink tensor (MiMo V2 checkpoints:
    /// `attention_sink_bias`, renamed to `sinks` by transformers).
    sinks_alt: ?[]const u8 = null,
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
    /// Full-rank KDA output gate (Kimi K3 `use_full_rank_gate`), used instead
    /// of `lin_g_a`/`lin_g_b` when `Config.linear_full_rank_gate` is set.
    lin_g: ?[]const u8 = null,
    /// Gate on the attention output before `o`: the sigmoid gate of Kimi K3's
    /// MLA layers (`mla_use_output_gate`, read when `Config.mla_output_gate`
    /// is set) or the sigmoid / softplus gate of AFMoE and Laguna (read when
    /// `Config.attn_gate` is not `.none`).
    attn_gate: ?[]const u8 = null,
    /// Attention Residual (Kimi K3): per layer the RMSNorm and `[1][hidden]`
    /// score projection of the attention-side and MLP-side aggregations, and
    /// at model level those of the output aggregation. Lists hold alternative
    /// spellings (first present wins).
    attn_res_norm: []const []const u8 = &.{},
    attn_res_proj: []const []const u8 = &.{},
    mlp_res_norm: []const []const u8 = &.{},
    mlp_res_proj: []const []const u8 = &.{},
    output_res_norm: []const []const u8 = &.{},
    output_res_proj: []const []const u8 = &.{},
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
    /// Latent MoE (Kimi K3 `routed_expert_hidden_size`): the routed experts
    /// read `latent_down(x)` (`[latent][hidden]`) and their weighted sum,
    /// optionally RMS-normalised by `latent_norm`, is written back through
    /// `latent_up` (`[hidden][latent]`). Names are relative to the layer.
    latent_down: ?[]const u8 = null,
    latent_up: ?[]const u8 = null,
    latent_norm: ?[]const u8 = null,
    /// Alternative MoE names (router, experts, shared expert) of checkpoints
    /// that predate the family's Hugging Face module layout; picked when the
    /// primary router tensor is absent and this one's is present.
    moe_alt: ?*const Names = null,
    /// Gated short convolution of LFM2 conv layers; its output projection is
    /// `conv_out` (loaded into the layer's `o` slot).
    conv_in: ?[]const u8 = null,
    conv_kernel: ?[]const u8 = null,
    conv_out: ?[]const u8 = null,
    /// Per-layer input embeddings (Gemma 3n / 4): the packed
    /// `[vocab][layers * dim]` table, its projection from the token embedding,
    /// the projection norm and the per-layer gate / output projection / norm.
    ple_embed: ?[]const u8 = null,
    ple_proj: ?[]const u8 = null,
    ple_proj_norm: ?[]const u8 = null,
    ple_gate: ?[]const u8 = null,
    ple_out: ?[]const u8 = null,
    ple_norm: ?[]const u8 = null,
    /// Per-layer output scalar (Gemma 4 `layer_scalar`, optional in checkpoints).
    layer_scale: ?[]const u8 = null,
    /// AltUp (Gemma 3n): `{e}` indexes the `altup_num_inputs - 1` projections.
    altup_proj: ?[]const u8 = null,
    altup_unembed: ?[]const u8 = null,
    altup_router: ?[]const u8 = null,
    altup_router_norm: ?[]const u8 = null,
    altup_predict: ?[]const u8 = null,
    altup_correct: ?[]const u8 = null,
    altup_scale: ?[]const u8 = null,
    /// Learned augmented residual (Gemma 3n).
    laurel_l: ?[]const u8 = null,
    laurel_r: ?[]const u8 = null,
    laurel_norm: ?[]const u8 = null,
    /// Hyper-connection sites around the attention and MLP blocks, relative
    /// to `layer` (hyper.zig names the tensors under them per `HyperKind`).
    hc_attn: []const u8 = "attn_hc",
    hc_ffn: []const u8 = "ffn_hc",
    /// The released mHC checkpoints (DeepSeek V4 / V4.1, GLM-5.3-Flash) flatten
    /// a site's three tensors into `hc_attn_fn` / `hc_attn_base` / `hc_attn_scale`
    /// instead of the module spelling `attn_hc.fn`; whichever the store has is
    /// used. Null on families whose sites are not mHC.
    hc_attn_flat: ?[]const u8 = "hc_attn_",
    hc_ffn_flat: ?[]const u8 = "hc_ffn_",
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
    /// The norm epsilon when config.json gives none: the family's transformers
    /// config-class default (OLMoE and others ship without the key). Null: 1e-6
    /// for RMSNorm families, 1e-5 otherwise.
    default_norm_eps: ?f32 = null,
    /// The RoPE base when config.json gives none (neither `rope_theta` nor
    /// `rope_parameters`): the family's transformers config-class default.
    /// Null: 10000.
    default_rope_theta: ?f32 = null,
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
    rope_scaling: RopeScaling,
    /// Rotated coordinates per head (`<= head_dim`).
    rotary_dim: usize,
    /// Denominator of the frequency exponents: `rotary_dim`, or the full head
    /// size when only a proportion of it rotates (Gemma 4 "proportional" RoPE).
    rope_freq_dim: usize,
    /// Separate rotary table of the sliding layers (Gemma 3 family); null when
    /// every layer uses the global table.
    rope_local: ?LocalRope,
    rope_style: RopeStyle,
    /// Per layer: whether rotary embeddings are applied.
    rope_layers: []bool,
    /// Per layer: head size, KV heads and attention scale. Equal to `head_dim`,
    /// `num_kv_heads` and `attention_scale` except on Gemma 4, whose global
    /// layers use a larger head.
    layer_head_dim: []usize,
    layer_kv_heads: []usize,
    /// Per layer: query heads. Equal to `num_heads` except on Laguna, whose
    /// `num_attention_heads_per_layer` gives the sliding layers more heads.
    layer_heads: []usize,
    layer_attn_scale: []f32,
    /// Per layer: the layer whose keys/values this layer attends over (itself
    /// unless the layer is KV-shared, Gemma 3n / 4).
    kv_source: []usize,
    /// Weightless per-head RMS norm on the values (Gemma 3n / 4).
    v_norm: bool,
    /// A layer without `v_proj` reuses its key projection as values (Gemma 4 `attention_k_eq_v`).
    k_eq_v: bool,
    /// Per-layer input embeddings (Gemma 3n / 4): width per layer (0 = none) and vocabulary.
    ple_dim: usize,
    ple_vocab: usize,
    /// AltUp (Gemma 3n): number of residual streams (0 = none), the active one,
    /// and whether the corrected active stream is scaled before the per-layer gate.
    altup_inputs: usize,
    altup_active: usize,
    altup_correct_scale: bool,
    laurel_rank: usize,
    /// Per layer: gate activation sparsity (Gemma 3n `activation_sparsity_pattern`); 0 = dense.
    activation_sparsity: []f32,
    /// The layers' feed-forward widths differ (Gemma 3n per-layer sizes,
    /// Gemma 4 double-wide MLPs); `intermediate_size` is then the largest.
    intermediate_varies: bool,
    /// Per layer: true for gated short-convolution layers (LFM2).
    conv_layers: []bool,
    has_conv: bool,
    conv_kernel: usize,
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
    /// KDA output gate is one full-rank projection (`g_proj`) instead of the
    /// low-rank `g_a`/`g_b` pair (Kimi K3).
    linear_full_rank_gate: bool = false,
    /// KDA "safe" forget gate (Kimi K3 `gate_lower_bound`): the per-channel
    /// log-decay is `lower_bound · sigmoid(exp(A_log) · (f + dt_bias))`
    /// instead of `-exp(A_log) · softplus(f + dt_bias)`.
    linear_gate_lower_bound: ?f32 = null,
    /// Full-attention (MLA) output is multiplied by `sigmoid(g_proj(h))` before
    /// the output projection (Kimi K3 `mla_use_output_gate`).
    mla_output_gate: bool = false,
    /// Attention Residual block size (Kimi K3 `attn_res_block_size`; 0 = a
    /// plain accumulated residual stream). See `model.zig` (`layerBlockAttnRes`).
    attn_res_block: usize = 0,
    /// Latent MoE width (Kimi K3 `routed_expert_hidden_size`; 0 = the routed
    /// experts read and write `hidden_size` directly) and whether the summed
    /// latent output is RMS-normalised before the up projection.
    moe_latent: usize = 0,
    moe_latent_norm: bool = false,
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
    /// Gated DeltaNet output norm gated by a sigmoid instead of a silu
    /// (Qwen4-Exp `output_gate_type = "sigmoid"`).
    linear_gate_sigmoid: bool,
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
    /// Rotate by `-angle` (NanoChat's `rotate_half` is `cat(x2, -x1)`): the
    /// sine tables are negated.
    rope_reverse: bool,
    residual_layout: ResidualLayout,
    /// MiniMax-01 residual scales `[full attention, linear attention, mlp]`
    /// as `[α (residual), β (sublayer output)]`.
    minimax_scales: [3][2]f32,
    /// Tensor-parallel blocks of a `mp_blocks` fused qkv projection (CodeGen).
    qkv_mp: usize,
    /// Gate on the attention output from a separate projection (AFMoE, Laguna).
    attn_gate: AttnGate,
    clip_qkv: ?f32,
    /// Scale on every sublayer output before the residual add (Granite, MiniCPM).
    residual_multiplier: f32,
    /// Multiplier on the final logits.
    logit_scale: f32,
    attn_temperature: ?AttnTemperature,
    sinks: bool,
    /// Sinks exist only on sliding-window layers (MiMo V2): `sinks` then
    /// requires them there and tolerates their absence on full layers.
    sinks_sliding_only: bool,
    /// Values are narrower than the keys outside MLA (MiMo V2 `v_head_dim`):
    /// `layerVDim` is then `v_head_dim` rather than the layer's head size.
    narrow_values: bool,
    /// Layout of the optional fused `names.qkv` tensor of a family whose
    /// primary layout is separate q/k/v (MiMo V2 Pro checkpoints): a layer
    /// that has the fused tensor and no `q_proj` is read from it.
    qkv_alt: ?QkvLayout,
    /// Chunks of a `.grouped` fused qkv tensor, each `[q heads | k heads | v
    /// heads]` of its share of the heads; 0 = one chunk per kv head.
    qkv_chunks: usize,
    mla: ?Mla,
    /// Mixture-of-experts settings (num_experts == 0 for dense models).
    num_experts: usize,
    num_experts_per_tok: usize,
    norm_topk_prob: bool,
    moe_intermediate_size: usize,
    /// Per layer: true if the layer's MLP is a routed mixture of experts.
    moe_layers: []bool,
    moe: MoeConfig,
    /// Residual streams per token (1 for conventional residuals; `hc_mult`
    /// for hyper-connection families).
    hc_mult: usize,
    /// How the streams are mixed around every block (null when `hc_mult == 1`).
    hyper: ?Hyper,
    /// Sparse-attention indexer of the full-attention layers, run as its
    /// dense equivalent within this bound (null: no indexer).
    index_bound: ?IndexBound,
    /// Qwen4-Exp per-layer n-gram embeddings (null for other families).
    ngram_ple: ?NgramPle,
    /// DeepSeek V4 / V4.1 layout (null for other families).
    dsv4: ?DsV4,

    pub fn isGemma(self: *const Config) bool {
        return self.norm == .rms_gemma;
    }

    /// Width of one KV cache row: the largest `kv_heads * head_dim` of any layer.
    pub fn kvDim(self: *const Config) usize {
        var d: usize = self.num_kv_heads * self.head_dim;
        for (self.layer_kv_heads, self.layer_head_dim) |kvh, hd| d = @max(d, kvh * hd);
        return d;
    }

    /// Largest query width `heads * head_dim` of any layer.
    pub fn maxQDim(self: *const Config) usize {
        var d: usize = self.num_heads * self.head_dim;
        for (self.layer_heads, self.layer_head_dim) |nh, hd| d = @max(d, nh * hd);
        return d;
    }

    /// Value width of layer `li`.
    pub fn layerVDim(self: *const Config, li: usize) usize {
        return if (self.mla != null or self.narrow_values) self.v_head_dim else self.layer_head_dim[li];
    }

    /// True when layer `li` reads another layer's keys and values.
    pub fn kvShared(self: *const Config, li: usize) bool {
        return self.kv_source[li] != li;
    }

    /// Any layer keeps a recurrent state (linear attention, Mamba or a short convolution).
    pub fn hasRecurrent(self: *const Config) bool {
        return self.has_linear or self.has_ssm or self.has_conv;
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
        // xIELU is parameterised per layer (Apertus); this entry only keeps the
        // parser quiet, the learned activation comes from the layer's tensors.
        .{ "xielu", tensor.Activation.silu },
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
        .{ "kimi_k2", "Kimi K2 (use the kimi_k25 wrapper config or a deepseek_v3 config; the standalone kimi_k2 model_type is untested)" },
        // Families surveyed from transformers' causal-LM mapping whose forward
        // pass needs a computation ditch does not implement.
        .{ "dbrx", "DBRX (experts stored as stacked [E*I][H] w1/v1/w2 blocks)" },
        .{ "phimoe", "Phi-3.5-MoE (sparsemixer routing, not a plain top-k)" },
        .{ "jetmoe", "JetMoE (mixture-of-attention-heads: the attention projections are themselves routed experts)" },
        .{ "zaya", "Zaya (per-channel residual scaling and a two-layer MLP router)" },
        .{ "longcat_flash", "LongCat-Flash (two attention and MLP blocks per layer with zero-computation experts)" },
        .{ "modernbert-decoder", "ModernBERT-decoder (a prediction head of dense + activation + norm sits between the final norm and the decoder)" },
        .{ "blt", "Byte Latent Transformer (byte patching with encoder, decoder and patcher towers)" },
        .{ "hrm_text", "HRM (the hierarchical recurrent reasoning loop repeats the stack)" },
        .{ "diffllama", "DiffLlama (differential attention: two attention maps combined with a learned lambda)" },
        .{ "doge", "Doge (dynamic mask attention)" },
        .{ "olmo_hybrid", "OLMo hybrid (Mamba-style recurrent layers)" },
        .{ "inkling_text", "Inkling (Mamba-style recurrent layers)" },
        .{ "axk2", "A.X-K2 (hyper-connections, gated norms and a lightning indexer)" },
        .{ "hy_v4", "Hunyuan V4 (hyper-connections and a lightning indexer)" },
        .{ "step3p7", "Step 3.7 (per-layer head counts and per-layer SwiGLU clamps)" },
        .{ "muse_glimmer_text", "Muse Glimmer (centred RMSNorm and a scaled weightless q/k norm)" },
        .{ "cohere_compass_text", "Cohere Compass (parallel residual with per-layer rope switching and pooling)" },
        .{ "cosmos3_edge_text", "Cosmos 3 Edge (three-section mrope with frequency recomposition)" },
        .{ "aria_text", "Aria (grouped-GEMM experts in a [E][H][2I] layout with a separate shared-expert activation)" },
        .{ "ernie4_5_vl_moe_text", "ERNIE 4.5 VL MoE (separate text and vision expert sets per layer)" },
        .{ "mllama_text_model", "Llama 3.2 Vision (cross-attention layers interleaved with the self-attention ones)" },
        .{ "mllama", "Llama 3.2 Vision (cross-attention layers interleaved with the self-attention ones)" },
        .{ "minicpm3", "MiniCPM3 (MLA with a partial rotary over the query LoRA)" },
        .{ "gpt_neox_japanese", "GPT-NeoX Japanese (per-layer bias sharing)" },
        .{ "cpmant", "CPM-Ant (relative position buckets)" },
        .{ "ctrl", "CTRL (sinusoidal positions with control codes; encoder-style blocks)" },
        .{ "openai-gpt", "OpenAI GPT-1 (learned positions with tied attention/feed-forward dropout blocks)" },
        .{ "cohere2_moe", "Command A MoE (parallel residual with an averaged shared expert)" },
        .{ "granitemoe_swa", "GraniteMoE SWA (sinks plus the Granite MoE layout; not yet verified)" },
        .{ "lfm2_moe", "LFM2-MoE (short-convolution hybrid with sigmoid-routed experts, not yet verified against a reference forward pass)" },
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
    var quant = try dequant.parseQuantConfig(getObj(top, "quantization_config") orelse getObj(obj, "quantization_config"));
    if (getStr(obj, "expert_dtype") orelse getStr(top, "expert_dtype")) |dt| {
        if (!dequant.expertDtypeSupported(dt)) {
            std.log.err("unsupported model: '{s}' expert dtype cannot be dequantised (bf16/f16/f32, fp8, fp4, mxfp4 and pack-quantized int4 are supported)", .{dt});
            return error.UnsupportedArchitecture;
        }
        // DeepSeek V4 spells its FP4 experts at the top level of config.json.
        if (std.ascii.eqlIgnoreCase(dt, "fp4")) {
            if (quant.method != .fp8) {
                std.log.err("unsupported model: expert_dtype fp4 without an fp8 quantization_config (found: {s})", .{quant.label});
                return error.UnsupportedArchitecture;
            }
            quant.fp4_experts = true;
            quant.label = "fp8 with fp4 experts";
        }
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
    // Omni wrappers (Qwen2.5-Omni, Qwen3-Omni) nest the language model under
    // the thinker; the talker and vocoder tensors are passed through.
    if (getObj(obj, "thinker_config")) |tc| {
        obj = tc;
        if (getStr(obj, "model_type")) |m| model_type = m;
    }
    // Multimodal wrappers keep the text config nested.
    var arch_opt: ?*const Arch = null;
    if (getObj(obj, "text_config")) |tc| {
        obj = tc;
        if (getStr(obj, "model_type")) |m| model_type = m;
        arch_opt = lookup(model_type);
        // A wrapper with its own entry wins over the family of its text
        // config: Kimi K3 nests a `kimi_linear` config but adds AttnRes,
        // latent MoE and SiTU on top of it.
        if (lookup(top_type)) |ta| {
            if (arch_opt == null or ta != arch_opt.?) {
                arch_opt = ta;
                model_type = top_type;
            }
        }
    } else arch_opt = lookup(model_type);
    const arch = arch_opt orelse {
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
    var heads = getIntAny(obj, &.{ "num_attention_heads", "n_head", "n_heads", "attention_heads", "num_heads" }, 0);
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
    var situ: ?Situ = null;
    if (std.mem.eql(u8, act_name, "situ")) {
        // Kimi K3: SiTU in the gated MLPs; the KDA convolution keeps its silu.
        situ = .{ .beta = getF32(obj, "activation_situ_beta", 1.0), .linear_beta = if (getNum(obj, "activation_situ_linear_beta")) |b| @as(f32, @floatCast(b)) else null };
        act = .silu;
    } else if (act_name.len > 0) {
        if (parseActivation(act_name)) |a| act = a else std.log.warn("unknown activation '{s}'; using {s}", .{ act_name, @tagName(act) });
    }

    const max_pos = getIntAny(obj, &.{ "max_position_embeddings", "n_positions", "max_seq_len", "seq_length" }, 4096);
    // Newer configs keep the base and the scaling together in `rope_parameters`,
    // either flat or as one dictionary per layer type (the Gemma 3 family reads
    // that form again in its own hook, for its two head sizes).
    var rope_params: ?std.json.ObjectMap = null;
    var rp_local: ?std.json.ObjectMap = null;
    if (getObj(obj, "rope_parameters")) |rp| {
        if (getObj(rp, "full_attention")) |full| {
            rope_params = full;
            rp_local = getObj(rp, "sliding_attention");
        } else if (getObj(rp, "sliding_attention")) |sl| {
            rp_local = sl;
        } else rope_params = rp;
    }
    const rs_obj: ?std.json.ObjectMap = getObj(obj, "rope_scaling") orelse rope_params;
    var rope_theta = getF32Any(obj, &.{ "rope_theta", "rotary_emb_base", "rope_base" }, arch.default_rope_theta orelse 10000.0);
    if (getNum(obj, "rope_theta") == null) if (rope_params) |rp| {
        rope_theta = getF32(rp, "rope_theta", rope_theta);
    };
    // Granite SWA keeps one base per layer type in `layer_rope_theta`.
    var local_theta: ?f32 = if (rp_local) |l| getF32(l, "rope_theta", rope_theta) else null;
    if (getObj(obj, "layer_rope_theta")) |lt| {
        if (getNum(obj, "rope_theta") == null) rope_theta = getF32(lt, "full_attention", rope_theta);
        local_theta = getF32(lt, "sliding_attention", rope_theta);
    }
    var rotary_dim = head_dim;
    if (getNum(obj, "partial_rotary_factor")) |f| rotary_dim = @intFromFloat(@as(f64, @floatFromInt(head_dim)) * f);
    if (getNum(obj, "rotary_pct")) |f| rotary_dim = @intFromFloat(@as(f64, @floatFromInt(head_dim)) * f);
    if (getNum(obj, "partial_rotary_factor") == null and mla == null) if (rope_params) |rp| {
        if (getNum(rp, "partial_rotary_factor")) |f| rotary_dim = @intFromFloat(@as(f64, @floatFromInt(head_dim)) * f);
    };
    // MiniMax checkpoints express partial rotary as an absolute `rotary_dim`.
    if (getNum(obj, "rotary_dim") != null) rotary_dim = @min(getInt(obj, "rotary_dim", head_dim), head_dim);
    if (mla) |m| rotary_dim = m.qk_rope_head_dim;
    rotary_dim -= rotary_dim % 2;

    const rope_scaling: RopeScaling = if (rs_obj) |rs| try parseRopeScaling(arena, obj, rs, rotary_dim, max_pos) else .none;
    // The sliding layers' own table (the Gemma 3 family sets its own in its
    // hook); its scaling is always the unscaled one of the released configs.
    // A per-layer-type `rope_parameters` table can give the local layers their
    // own `partial_rotary_factor` as well as their own base (Laguna rotates
    // half the head on its full-attention layers and all of it on the sliding
    // ones), so the local rotary width is read from the local entry.
    var local_rotary_dim = rotary_dim;
    if (rp_local) |l| if (getNum(l, "partial_rotary_factor")) |f| {
        local_rotary_dim = @intFromFloat(@as(f64, @floatFromInt(head_dim)) * f);
        local_rotary_dim -= local_rotary_dim % 2;
    };
    const rope_local: ?LocalRope = if (local_theta) |t| .{ .theta = t, .rotary_dim = local_rotary_dim, .freq_dim = local_rotary_dim } else null;

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
    const conv_layers = try arena.alloc(bool, layers);
    @memset(conv_layers, false);
    var has_linear = false;
    // Mamba families: `linear_attention` (or the legacy `mamba`) marks a
    // state-space layer; single-block families also name `mlp` / `moe` layers.
    const ssm_layers = try arena.alloc(bool, layers);
    @memset(ssm_layers, false);
    const mlp_only = try arena.alloc(bool, layers);
    @memset(mlp_only, false);
    const moe_only = try arena.alloc(bool, layers);
    @memset(moe_only, false);
    var has_conv = false;
    var has_block_types = false;
    if (block_types) |lt| {
        if (lt == .array) {
            has_block_types = true;
            for (lt.array.items, 0..) |v, i| {
                if (i < layers and v == .string) {
                    const t = v.string;
                    if (std.mem.eql(u8, t, "sliding_attention") or std.mem.eql(u8, t, "chunked_attention")) {
                        sliding_layers[i] = true;
                    } else if (std.mem.eql(u8, t, "compressed_sparse_attention") or std.mem.eql(u8, t, "heavily_compressed_attention") or std.mem.eql(u8, t, "shared_compressed_attention")) {
                        // DeepSeek V4 / V4.1: a sliding window plus a compressed
                        // branch (read again by the family hook).
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
                    } else if (std.mem.eql(u8, t, "conv")) {
                        conv_layers[i] = true;
                        has_conv = true;
                    } else if (std.mem.eql(u8, t, "full_attention") or std.mem.eql(u8, t, "attention") or std.mem.eql(u8, t, "hybrid") or std.mem.eql(u8, t, "indexed_attention")) {
                        // `indexed_attention` (Qwen4-Exp, GLM-5.3-Flash,
                        // DeepSeek V3.2): full attention behind a sparse
                        // indexer whose dense equivalent runs within
                        // `Config.index_bound`.
                    } else if (std.mem.eql(u8, t, "mlp") or std.mem.eql(u8, t, "moe")) {
                        if (!arch.single_mixer) {
                            std.log.err("unsupported layer type '{s}' in a {s} model", .{ t, model_type });
                            return error.UnsupportedArchitecture;
                        }
                        if (t[1] == 'l') mlp_only[i] = true else moe_only[i] = true;
                    } else if (std.mem.eql(u8, t, "deepseek_sparse_attention")) {
                        // Zhipu's sparse indexer selects a top-k over the keys;
                        // for the short calibration contexts ditch scores, the
                        // top-k covers the whole context, so dense attention is
                        // exact.
                        if (i == 0) std.log.warn("deepseek_sparse_attention runs as dense attention (exact for short contexts)", .{});
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
                        std.log.err("unsupported layer type '{s}' (only full/sliding/linear attention, conv, Mamba, mlp and moe blocks are implemented)", .{t});
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
        const first_dense = getIntAny(obj, &.{ "first_k_dense_replace", "num_dense_layers" }, 0);
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
        // `mlp_layer_types` ("dense" / "sparse" per layer) overrides the schedule.
        if (obj.get("mlp_layer_types")) |ml| {
            if (ml == .array) for (ml.array.items, 0..) |v, i| {
                if (i < layers and v == .string) moe_layers[i] = !std.mem.eql(u8, v.string, "dense");
            };
        }
    }
    // `intermediate_size` may be a per-layer list (Gemma 3n); the weights carry
    // the exact sizes and the largest one sizes the workspace.
    var intermediate_size = getIntAny(obj, &.{ "intermediate_size", "n_inner", "ffn_dim", "ffn_hidden_size" }, 0);
    var intermediate_varies = false;
    if (obj.get("intermediate_size")) |isz| {
        if (isz == .array) for (isz.array.items) |v| {
            if (v == .integer and v.integer > 0) {
                const n: usize = @intCast(v.integer);
                if (intermediate_size != 0 and n != intermediate_size) intermediate_varies = true;
                intermediate_size = @max(intermediate_size, n);
            }
        };
    }

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
    // Per-layer geometry: a hook may set individual layers (Gemma 4); the rest
    // follow the model-level values once every hook has run.
    const layer_head_dim = try arena.alloc(usize, layers);
    @memset(layer_head_dim, 0);
    const layer_kv_heads = try arena.alloc(usize, layers);
    @memset(layer_kv_heads, 0);
    const layer_heads = try arena.alloc(usize, layers);
    @memset(layer_heads, 0);
    const layer_attn_scale = try arena.alloc(f32, layers);
    const kv_source = try arena.alloc(usize, layers);
    for (kv_source, 0..) |*s, i| s.* = i;
    const activation_sparsity = try arena.alloc(f32, layers);
    @memset(activation_sparsity, 0);

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
        .rms_norm_eps = getF32Any(obj, &.{ "rms_norm_eps", "layer_norm_eps", "layer_norm_epsilon", "layernorm_epsilon", "norm_eps", "norm_epsilon" }, arch.default_norm_eps orelse if (arch.norm == .rms or arch.norm == .rms_gemma) 1e-6 else 1e-5),
        .rope_theta = rope_theta,
        .rope_scaling = rope_scaling,
        .rotary_dim = rotary_dim,
        .rope_freq_dim = rotary_dim,
        .rope_local = rope_local,
        .rope_style = arch.rope_style,
        .rope_layers = rope_layers,
        .layer_head_dim = layer_head_dim,
        .layer_kv_heads = layer_kv_heads,
        .layer_heads = layer_heads,
        .layer_attn_scale = layer_attn_scale,
        .kv_source = kv_source,
        .v_norm = false,
        .k_eq_v = false,
        .ple_dim = 0,
        .ple_vocab = 0,
        .altup_inputs = 0,
        .altup_active = 0,
        .altup_correct_scale = true,
        .laurel_rank = 0,
        .activation_sparsity = activation_sparsity,
        .intermediate_varies = intermediate_varies,
        .conv_layers = conv_layers,
        .has_conv = has_conv,
        .conv_kernel = getInt(obj, "conv_L_cache", 0),
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
        .linear_gate_sigmoid = false,
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
        .rope_reverse = false,
        .residual_layout = .pre,
        .minimax_scales = .{ .{ 1, 1 }, .{ 1, 1 }, .{ 1, 1 } },
        .qkv_mp = 1,
        .attn_gate = .none,
        .clip_qkv = null,
        .residual_multiplier = 1.0,
        .logit_scale = 1.0,
        .attn_temperature = null,
        .sinks = false,
        .sinks_sliding_only = false,
        .narrow_values = false,
        .qkv_alt = null,
        .qkv_chunks = 0,
        .mla = mla,
        .num_experts = num_experts,
        .num_experts_per_tok = getInt(obj, "num_experts_per_tok", 2),
        .norm_topk_prob = getBool(obj, "norm_topk_prob", false),
        .moe_intermediate_size = getInt(obj, "moe_intermediate_size", if (intermediate_size > 0) intermediate_size else 4 * hidden),
        .moe_layers = moe_layers,
        .moe = .{ .situ = situ },
        .hc_mult = 1,
        .hyper = null,
        .index_bound = null,
        .ngram_ple = null,
        .dsv4 = null,
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
            if (rs_obj) |rs| {
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
    for (0..layers) |i| c.attn_layers[i] = c.attn_layers[i] and !c.linear_layers[i] and !c.conv_layers[i] and (!c.ssm_layers[i] or c.parallel_ssm);
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
    @memset(c.layer_attn_scale, c.attention_scale);
    for (c.layer_head_dim, c.layer_kv_heads, c.layer_heads) |*hd_l, *kv_l, *nh_l| {
        if (hd_l.* == 0) hd_l.* = c.head_dim;
        if (kv_l.* == 0) kv_l.* = c.num_kv_heads;
        if (nh_l.* == 0) nh_l.* = c.num_heads;
    }
    // Only the Gemma tables derive frequencies from a wider head than they rotate.
    if (c.rope_local == null) c.rope_freq_dim = c.rotary_dim;
    if (c.has_conv and c.conv_kernel < 2) {
        std.log.err("conv layers need conv_L_cache >= 2", .{});
        return error.InvalidConfig;
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

/// Parses a `rope_scaling` / `rope_parameters` dictionary `rs` (`obj` is the
/// model config it belongs to, for the fallback keys some families keep at
/// the top level).
fn parseRopeScaling(arena: Allocator, obj: std.json.ObjectMap, rs: std.json.ObjectMap, rotary_dim: usize, max_pos: usize) !RopeScaling {
    var result: RopeScaling = .none;
    const t = getStr(rs, "rope_type") orelse getStr(rs, "type") orelse "";
    if (std.mem.eql(u8, t, "llama3")) {
        result = .{ .llama3 = .{
            .factor = getF32(rs, "factor", 8),
            .low_freq_factor = getF32(rs, "low_freq_factor", 1),
            .high_freq_factor = getF32(rs, "high_freq_factor", 4),
            .original_max_position = getF32(rs, "original_max_position_embeddings", 8192),
        } };
    } else if (std.mem.eql(u8, t, "linear")) {
        result = .{ .linear = getF32(rs, "factor", 1) };
    } else if (std.mem.eql(u8, t, "yarn")) {
        const factor = getF32(rs, "factor", 1);
        const mscale = getF32(rs, "mscale", 1);
        const mscale_all_dim = getF32(rs, "mscale_all_dim", 0);
        const attention_factor: f32 = if (getNum(rs, "attention_factor")) |af| @floatCast(af) else if (mscale != 0 and mscale_all_dim != 0)
            yarnMscale(factor, mscale) / yarnMscale(factor, mscale_all_dim)
        else
            yarnMscale(factor, 1);
        result = .{ .yarn = .{
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
        result = .{ .longrope = .{ .factors = factors, .attention_factor = attention_factor } };
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
                result = .{ .factors = factors };
            }
        }
    } else if (std.mem.eql(u8, t, "dynamic")) {
        std.log.warn("dynamic NTK rope scaling is treated as unscaled RoPE (exact below the original context length)", .{});
    } else if (std.mem.eql(u8, t, "proportional")) {
        // Gemma 4: a proportion of the head rotates with frequencies derived
        // from the full head (the family hook sets the dims); `factor`
        // divides the frequencies like linear scaling.
        const factor = getF32(rs, "factor", 1);
        if (factor != 1) result = .{ .linear = factor };
    } else if (std.mem.eql(u8, t, "mrope") or std.mem.eql(u8, t, "default") or t.len == 0) {
        // mrope over text-only positions equals plain RoPE.
    } else {
        std.log.warn("rope scaling type '{s}' is not supported; using unscaled RoPE", .{t});
    }
    return result;
}

fn evenDim(x: f64) usize {
    var d: usize = @intFromFloat(x);
    d -= d % 2;
    return d;
}

/// Rotary tables of the Gemma 3 family: the global layers' base and scaling
/// (`rope_theta` / `rope_scaling`, or `rope_parameters.full_attention`) and the
/// sliding layers' own base (`rope_local_base_freq` or
/// `rope_parameters.sliding_attention`). `hd_local` / `hd_global` are the head
/// sizes of the two layer kinds.
fn gemmaRope(c: *Config, arena: Allocator, obj: std.json.ObjectMap, hd_local: usize, hd_global: usize) !void {
    var local_theta = getF32(obj, "rope_local_base_freq", 10000.0);
    var local_dim = hd_local;
    var global_dim = hd_global;
    var freq_dim = hd_global;
    if (getNum(obj, "partial_rotary_factor")) |f| {
        local_dim = evenDim(@as(f64, @floatFromInt(hd_local)) * f);
        global_dim = evenDim(@as(f64, @floatFromInt(hd_global)) * f);
        freq_dim = global_dim;
    }
    if (getObj(obj, "rope_parameters")) |rp| {
        if (getObj(rp, "full_attention")) |full| {
            c.rope_theta = getF32(full, "rope_theta", c.rope_theta);
            c.rope_scaling = try parseRopeScaling(arena, obj, full, global_dim, c.max_position_embeddings);
            const t = getStr(full, "rope_type") orelse "default";
            if (getNum(full, "partial_rotary_factor")) |f| {
                if (std.mem.eql(u8, t, "proportional")) {
                    // `int(f * head_dim // 2)` angles; the frequencies follow the full head.
                    global_dim = 2 * @as(usize, @intFromFloat(@floor(@as(f64, @floatFromInt(hd_global)) * f / 2.0)));
                    freq_dim = hd_global;
                } else {
                    global_dim = evenDim(@as(f64, @floatFromInt(hd_global)) * f);
                    freq_dim = global_dim;
                }
            } else if (std.mem.eql(u8, t, "proportional")) {
                freq_dim = hd_global;
            }
        }
        if (getObj(rp, "sliding_attention")) |sl| {
            local_theta = getF32(sl, "rope_theta", local_theta);
            if (getNum(sl, "partial_rotary_factor")) |f| local_dim = evenDim(@as(f64, @floatFromInt(hd_local)) * f);
        }
    }
    c.rotary_dim = global_dim;
    c.rope_freq_dim = freq_dim;
    c.rope_local = .{ .theta = local_theta, .rotary_dim = local_dim, .freq_dim = local_dim };
}

/// The last `num_kv_shared_layers` layers read the keys and values of the last
/// non-shared layer of their kind (sliding or global) instead of computing their own.
fn kvSharing(c: *Config, obj: std.json.ObjectMap) !void {
    const n = getInt(obj, "num_kv_shared_layers", 0);
    if (n == 0 or n >= c.num_layers) return;
    const first = c.num_layers - n;
    for (first..c.num_layers) |i| {
        var j = first;
        var found = false;
        while (j > 0) {
            j -= 1;
            if (c.sliding_layers[j] == c.sliding_layers[i]) {
                c.kv_source[i] = j;
                found = true;
                break;
            }
        }
        if (!found) {
            std.log.err("layer {d} shares keys/values but no earlier layer of its kind exists", .{i});
            return error.InvalidConfig;
        }
    }
}

fn yarnMscale(scale: f32, mscale: f32) f32 {
    if (scale <= 1) return 1.0;
    return 0.1 * mscale * @log(scale) + 1.0;
}

// ---------------------------------------------------------------------------
// Family-specific config hooks
// ---------------------------------------------------------------------------

fn extraGemma(c: *Config, arena: Allocator, obj: std.json.ObjectMap) !void {
    if (getStr(obj, "hidden_activation") == null and getStr(obj, "hidden_act") == null) c.activation = .gelu_tanh;
    if (getNum(obj, "sliding_window_pattern") == null and obj.get("layer_types") == null and c.sliding_window != null) {
        // Gemma 3 defaults to a pattern of 6 (5 local, 1 global); Gemma 2 alternates.
        if (std.mem.startsWith(u8, c.model_type, "gemma3")) {
            for (c.sliding_layers, 0..) |*s, i| s.* = ((i + 1) % 6 != 0);
        }
    }
    if (std.mem.startsWith(u8, c.model_type, "gemma3")) try gemmaRope(c, arena, obj, c.head_dim, c.head_dim);
}

fn perLayerEmbeddings(c: *Config, obj: std.json.ObjectMap, default_dim: usize) void {
    c.ple_dim = getInt(obj, "hidden_size_per_layer_input", default_dim);
    c.ple_vocab = getInt(obj, "vocab_size_per_layer_input", c.vocab_size);
    if (c.ple_vocab == 0) c.ple_vocab = c.vocab_size;
}

fn extraGemma3n(c: *Config, arena: Allocator, obj: std.json.ObjectMap) !void {
    if (getStr(obj, "hidden_activation") == null and getStr(obj, "hidden_act") == null) c.activation = .gelu_tanh;
    if (obj.get("layer_types") == null) {
        // Every fifth layer is global.
        for (c.sliding_layers, 0..) |*s, i| s.* = ((i + 1) % 5 != 0);
    }
    if (c.sliding_window == null) c.sliding_window = 512;
    c.attention_scale = 1.0;
    c.v_norm = true;
    try gemmaRope(c, arena, obj, c.head_dim, c.head_dim);
    try kvSharing(c, obj);
    perLayerEmbeddings(c, obj, 256);
    c.altup_inputs = getInt(obj, "altup_num_inputs", 4);
    c.altup_active = getInt(obj, "altup_active_idx", 0);
    c.altup_correct_scale = getBool(obj, "altup_correct_scale", true);
    c.laurel_rank = getInt(obj, "laurel_rank", 64);
    if (c.altup_inputs < 1 or c.altup_active >= c.altup_inputs or c.ple_dim == 0 or c.laurel_rank == 0) return error.InvalidConfig;
    // Activation sparsity: one value per layer, one for all, or the default
    // (the first 10 layers at 0.95 on models deeper than 10 layers).
    if (obj.get("activation_sparsity_pattern")) |asp| {
        switch (asp) {
            .array => |a| for (a.items, 0..) |v, i| {
                if (i < c.num_layers) c.activation_sparsity[i] = switch (v) {
                    .float => |f| @floatCast(f),
                    .integer => |n| @floatFromInt(n),
                    else => 0,
                };
            },
            .float => |f| @memset(c.activation_sparsity, @floatCast(f)),
            .integer => |n| @memset(c.activation_sparsity, @floatFromInt(n)),
            else => {},
        }
    } else if (c.num_layers > 10) {
        for (c.activation_sparsity[0..10]) |*s| s.* = 0.95;
    }
    for (c.activation_sparsity) |s| if (s < 0 or s >= 1) return error.InvalidConfig;
}

fn extraGemma4(c: *Config, arena: Allocator, obj: std.json.ObjectMap) !void {
    if (getStr(obj, "hidden_activation") == null and getStr(obj, "hidden_act") == null) c.activation = .gelu_tanh;
    if (getStr(obj, "use_bidirectional_attention")) |m| {
        if (std.mem.eql(u8, m, "all")) {
            std.log.err("unsupported model: Gemma 4 with bidirectional attention on every token", .{});
            return error.UnsupportedArchitecture;
        }
    }
    if (getBool(obj, "enable_moe_block", false)) {
        std.log.err("unsupported model: Gemma 4 MoE block (a routed expert block in parallel with the dense MLP, as in gemma-4-26B-A4B) is not implemented", .{});
        return error.UnsupportedArchitecture;
    }
    if (obj.get("layer_types") == null) {
        // 5 local, 1 global; the last layer is always global.
        for (c.sliding_layers, 0..) |*s, i| s.* = ((i + 1) % 6 != 0);
    }
    if (c.num_layers > 0) c.sliding_layers[c.num_layers - 1] = false;
    if (c.sliding_window == null) c.sliding_window = 512;
    c.attention_scale = 1.0;
    c.v_norm = true;
    c.k_eq_v = getBool(obj, "attention_k_eq_v", false);
    // Global layers use their own head size and KV heads: `global_head_dim`
    // (512 unless given) and `num_global_key_value_heads`, or an explicit
    // `per_layer_config` table indexed by layer.
    var global_hd = getInt(obj, "global_head_dim", 512);
    var global_kv = c.num_kv_heads;
    if (getNum(obj, "num_global_key_value_heads") != null and c.k_eq_v) global_kv = getInt(obj, "num_global_key_value_heads", global_kv);
    if (getObj(obj, "per_layer_config")) |plc| {
        var it = plc.iterator();
        var found = false;
        while (it.next()) |kv| {
            const idx = std.fmt.parseInt(usize, kv.key_ptr.*, 10) catch continue;
            if (idx >= c.num_layers or c.sliding_layers[idx] or kv.value_ptr.* != .object or found) continue;
            const lc = kv.value_ptr.object;
            global_hd = getInt(lc, "head_dim", global_hd);
            global_kv = getInt(lc, "num_key_value_heads", global_kv);
            found = true;
        }
    }
    for (0..c.num_layers) |i| {
        if (!c.sliding_layers[i]) {
            c.layer_head_dim[i] = global_hd;
            c.layer_kv_heads[i] = global_kv;
        }
    }
    try gemmaRope(c, arena, obj, c.head_dim, global_hd);
    try kvSharing(c, obj);
    perLayerEmbeddings(c, obj, 256);
    // KV-shared layers may carry a double-wide MLP; the weights say which.
    if (getBool(obj, "use_double_wide_mlp", false) and getInt(obj, "num_kv_shared_layers", 0) > 0) c.intermediate_varies = true;
}

fn extraLfm2(c: *Config, _: Allocator, obj: std.json.ObjectMap) !void {
    if (obj.get("layer_types") == null) {
        // `full_attn_idxs` lists the attention layers; every other layer is a conv layer.
        if (obj.get("full_attn_idxs")) |fa| {
            @memset(c.conv_layers, true);
            if (fa == .array) for (fa.array.items) |v| {
                if (v == .integer and v.integer >= 0 and v.integer < c.num_layers) c.conv_layers[@intCast(v.integer)] = false;
            };
        }
        c.has_conv = false;
        for (c.conv_layers) |l| c.has_conv = c.has_conv or l;
    }
    c.conv_kernel = getInt(obj, "conv_L_cache", 3);
    if (getNum(obj, "rope_theta") == null and getObj(obj, "rope_parameters") == null) c.rope_theta = 1000000.0;
    c.tie_word_embeddings = getBool(obj, "tie_word_embeddings", getBool(obj, "tie_embedding", true));
    // Feed-forward width as Lfm2MLP derives it from `block_ff_dim`.
    var ff = getInt(obj, "block_ff_dim", c.intermediate_size);
    if (getBool(obj, "block_auto_adjust_ff_dim", true)) {
        ff = @intFromFloat(2.0 * @as(f64, @floatFromInt(ff)) / 3.0);
        const mult = getF32(obj, "block_ffn_dim_multiplier", 1.0);
        ff = @intFromFloat(@as(f64, mult) * @as(f64, @floatFromInt(ff)));
        const mo = getInt(obj, "block_multiple_of", 256);
        if (mo > 0) ff = mo * ((ff + mo - 1) / mo);
    }
    c.intermediate_size = ff;
}

fn extraMistral4(c: *Config, _: Allocator, obj: std.json.ObjectMap) !void {
    c.rope_style = if (getBool(obj, "rope_interleave", true)) .gptj else .neox;
    c.moe.scoring = .softmax;
    c.moe.topk_method = .group_limited;
    c.moe.group_score_top2 = true;
    c.moe.n_group = @max(1, getInt(obj, "n_group", 1));
    c.moe.topk_group = @max(1, getInt(obj, "topk_group", 1));
    c.moe.routed_scaling_factor = getF32(obj, "routed_scaling_factor", 1.0);
    c.norm_topk_prob = getBool(obj, "norm_topk_prob", true);
    c.num_experts_per_tok = getInt(obj, "num_experts_per_tok", 4);
    if (c.num_experts % c.moe.n_group != 0) return error.InvalidConfig;
    // Queries are scaled by `1 + beta * log(1 + floor(pos / original_max_position_embeddings))`.
    const rp = getObj(obj, "rope_parameters") orelse getObj(obj, "rope_scaling");
    if (rp) |r| {
        if (getNum(r, "llama_4_scaling_beta")) |beta| {
            c.attn_temperature = .{
                .floor_scale = getF32(r, "original_max_position_embeddings", 8192),
                .attn_scale = @floatCast(beta),
                .offset = 0,
                .all_layers = true,
            };
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
    if (getNum(obj, "partial_rotary_factor") == null) {
        c.rotary_dim = c.head_dim / 2;
        c.rope_freq_dim = c.rotary_dim;
    }
    c.attention_bias = getBool(obj, "attention_bias", true);
}

fn extraChatGlm(c: *Config, _: Allocator, obj: std.json.ObjectMap) !void {
    // Rotary embeddings cover half of `kv_channels`, interleaved; theta is scaled by rope_ratio.
    c.rotary_dim = c.head_dim / 2;
    c.rope_freq_dim = c.rotary_dim;
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
    c.moe.scoring = parseScoring(getStr(obj, "scoring_func") orelse "softmax");
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

/// Arcee, Jais2, Cosmos3 Edge: a llama layout whose MLP is the two-projection
/// `down(act(up(x)))` form (`relu²` by default, biases from `mlp_bias`).
fn extraDenseMlp(c: *Config, _: Allocator, obj: std.json.ObjectMap) !void {
    if (getStr(obj, "hidden_act") == null) c.activation = .relu2;
}

fn extraErnie(c: *Config, _: Allocator, obj: std.json.ObjectMap) !void {
    c.attention_bias = getBool(obj, "use_bias", false);
}

fn extraHunyuanDense(c: *Config, _: Allocator, obj: std.json.ObjectMap) !void {
    // Per-head q/k RMSNorm after RoPE, NTK-alpha "dynamic" scaling folded into the base.
    c.qk_norm_after_rope = true;
    c.attention_bias = getBool(obj, "attention_bias", false);
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

fn extraNanoChat(c: *Config, _: Allocator, obj: std.json.ObjectMap) !void {
    // Weightless RMSNorm everywhere, including the q/k norms, which run after RoPE.
    c.qk_norm = .l2;
    c.qk_norm_after_rope = true;
    // `rotate_half` returns `cat(x2, -x1)` (karpathy's `apply_rotary_emb`
    // too): the rotation runs the other way from Llama's.
    c.rope_reverse = true;
    // nanochat always softcaps the logits at 15, and transformers'
    // `NanoChatConfig` defaults `final_logit_softcapping` to it; some
    // conversions spell the key `logits_soft_cap`, which transformers ignores.
    if (c.final_logit_softcapping == null and obj.get("final_logit_softcapping") == null) c.final_logit_softcapping = 15.0;
}

fn extraPersimmon(c: *Config, _: Allocator, obj: std.json.ObjectMap) !void {
    if (getBool(obj, "qk_layernorm", true)) c.qk_norm = .head;
    c.num_kv_heads = c.num_heads;
}

fn extraGptJ(c: *Config, _: Allocator, obj: std.json.ObjectMap) !void {
    c.num_kv_heads = c.num_heads;
    if (getNum(obj, "rotary_dim") == null) c.rotary_dim = c.head_dim;
}

fn extraCodeGen(c: *Config, _: Allocator, obj: std.json.ObjectMap) !void {
    try extraGptJ(c, undefined, obj);
    // The fused projection is `mp_num` tensor-parallel blocks of [q | v | k].
    c.qkv_mp = 4;
    if (c.num_heads % c.qkv_mp != 0) return error.UnsupportedArchitecture;
}

fn extraGptNeo(c: *Config, arena: Allocator, obj: std.json.ObjectMap) !void {
    // GPT-Neo does not scale the attention logits, and alternates global and
    // `window_size` local layers (`attention_types` expands to `attention_layers`).
    c.attention_scale = 1.0;
    c.num_kv_heads = c.num_heads;
    c.sliding_window = getInt(obj, "window_size", 256);
    @memset(c.sliding_layers, false);
    var types: ?[]const []const u8 = null;
    if (obj.get("attention_layers")) |al| {
        if (al == .array) {
            const list = try arena.alloc([]const u8, al.array.items.len);
            for (al.array.items, 0..) |v, i| list[i] = if (v == .string) v.string else "global";
            types = list;
        }
    }
    if (types == null) {
        if (obj.get("attention_types")) |at| {
            // `[[["global", "local"], n], ...]`: each inner list repeated n times.
            var list = std.ArrayList([]const u8).empty;
            if (at == .array) for (at.array.items) |item| {
                if (item != .array or item.array.items.len != 2) continue;
                const names_v = item.array.items[0];
                const rep = item.array.items[1];
                if (names_v != .array or rep != .integer) continue;
                var r: i64 = 0;
                while (r < rep.integer) : (r += 1) {
                    for (names_v.array.items) |nv| if (nv == .string) try list.append(arena, nv.string);
                }
            };
            if (list.items.len > 0) types = list.items;
        }
    }
    const list = types orelse blk: {
        const d = try arena.alloc([]const u8, c.num_layers);
        for (d, 0..) |*t, i| t.* = if (i % 2 == 0) "global" else "local";
        break :blk d;
    };
    for (c.sliding_layers, 0..) |*s, i| s.* = i < list.len and std.mem.eql(u8, list[i], "local");
}

fn extraXglm(c: *Config, _: Allocator, obj: std.json.ObjectMap) !void {
    c.num_kv_heads = c.num_heads;
    c.position_offset = 2;
    if (getBool(obj, "scale_embedding", true)) c.embed_scale = @sqrt(@as(f32, @floatFromInt(c.hidden_size)));
}

fn extraBioGpt(c: *Config, _: Allocator, obj: std.json.ObjectMap) !void {
    c.num_kv_heads = c.num_heads;
    c.position_offset = 2;
    if (getBool(obj, "scale_embedding", true)) c.embed_scale = @sqrt(@as(f32, @floatFromInt(c.hidden_size)));
}

fn extraMinistral3(c: *Config, _: Allocator, obj: std.json.ObjectMap) !void {
    // Ministral 3 scales the queries of every layer by
    // `1 + beta * log(1 + floor(pos / max_position_embeddings))`.
    var beta: ?f32 = null;
    if (getObj(obj, "rope_parameters")) |rp| {
        if (getNum(rp, "llama_4_scaling_beta")) |b| beta = @floatCast(b);
    }
    if (getNum(obj, "llama_4_scaling_beta")) |b| beta = @floatCast(b);
    if (beta) |b| c.attn_temperature = .{
        .floor_scale = @floatFromInt(c.max_position_embeddings),
        .attn_scale = b,
        .offset = 0,
        .all_layers = true,
    };
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

/// Reads a `mlp_layer_types` list (`"dense"` / `"sparse"`) into `moe_layers`;
/// without the key, layers from `first_dense` on are sparse.
fn mlpLayerTypes(c: *Config, obj: std.json.ObjectMap, first_dense: usize) !void {
    if (c.num_experts == 0) return;
    if (obj.get("mlp_layer_types")) |ml| {
        if (ml != .array) return error.InvalidConfig;
        @memset(c.moe_layers, false);
        for (ml.array.items, 0..) |v, i| {
            if (i >= c.num_layers) break;
            if (v != .string) return error.InvalidConfig;
            if (std.mem.eql(u8, v.string, "sparse")) c.moe_layers[i] = true else if (!std.mem.eql(u8, v.string, "dense")) {
                std.log.err("unsupported mlp_layer_types entry '{s}' (dense or sparse)", .{v.string});
                return error.UnsupportedArchitecture;
            }
        }
        return;
    }
    for (c.moe_layers, 0..) |*m, i| m.* = i >= first_dense;
}

/// GLM-4.7-Flash (`glm4_moe_lite`): DeepSeek V3 MLA with interleaved RoPE and
/// the GLM-4.5 router (sigmoid scores, correction bias, top-2 group scores,
/// renormalised with a floor, `routed_scaling_factor`), an `mlp_layer_types`
/// schedule that defaults to one dense layer.
fn extraGlm4MoeLite(c: *Config, _: Allocator, obj: std.json.ObjectMap) !void {
    c.rope_style = if (getBool(obj, "rope_interleave", true)) .gptj else .neox;
    if (getNum(obj, "rms_norm_eps") == null) c.rms_norm_eps = 1e-5;
    c.norm_topk_prob = getBool(obj, "norm_topk_prob", true);
    c.num_experts_per_tok = getInt(obj, "num_experts_per_tok", 4);
    c.moe.scoring = .sigmoid;
    c.moe.topk_method = .group_limited;
    c.moe.n_group = @max(1, getInt(obj, "n_group", 1));
    c.moe.topk_group = @max(1, getInt(obj, "topk_group", 1));
    c.moe.routed_scaling_factor = getF32(obj, "routed_scaling_factor", 1.8);
    c.moe.group_score_top2 = true;
    c.moe.norm_eps_floor = true;
    if (c.num_experts % c.moe.n_group != 0) return error.InvalidConfig;
    if (c.mla == null) {
        std.log.err("glm4_moe_lite: the MLA keys (kv_lora_rank, qk_rope_head_dim, ...) are required", .{});
        return error.InvalidConfig;
    }
    try mlpLayerTypes(c, obj, 1);
}

/// Multi-head latent attention without rotary channels (`qk_rope_head_dim`
/// 0, NoPE): builds `Config.mla` when the generic parser skipped it because
/// the key was absent, and sets the head sizes and scale for it.
fn mlaNoPe(c: *Config, obj: std.json.ObjectMap, family: []const u8) !void {
    if (c.mla == null) {
        if (getNum(obj, "kv_lora_rank") == null) {
            std.log.err("{s}: the MLA keys (q_lora_rank, kv_lora_rank, qk_nope_head_dim, v_head_dim) are required", .{family});
            return error.InvalidConfig;
        }
        c.mla = .{
            .q_lora_rank = if (getNum(obj, "q_lora_rank")) |_| getInt(obj, "q_lora_rank", 0) else null,
            .kv_lora_rank = getInt(obj, "kv_lora_rank", 0),
            .qk_nope_head_dim = getInt(obj, "qk_nope_head_dim", 0),
            .qk_rope_head_dim = getInt(obj, "qk_rope_head_dim", 0),
            .v_head_dim = getInt(obj, "v_head_dim", 0),
        };
    }
    const m = c.mla.?;
    if (m.qk_rope_head_dim != 0) {
        std.log.err("{s}: the attention layers are NoPE (qk_rope_head_dim must be 0, config has {d})", .{ family, m.qk_rope_head_dim });
        return error.UnsupportedArchitecture;
    }
    if (m.q_lora_rank == null or m.q_lora_rank.? == 0) {
        std.log.err("{s}: q_lora_rank is required (the sparse indexer reads the low-rank query)", .{family});
        return error.InvalidConfig;
    }
    if (m.kv_lora_rank == 0 or m.qk_nope_head_dim == 0 or m.v_head_dim == 0 or m.v_head_dim > m.qk_nope_head_dim) return error.InvalidConfig;
    c.head_dim = m.qk_nope_head_dim;
    c.v_head_dim = m.v_head_dim;
    c.num_kv_heads = c.num_heads;
    c.rotary_dim = 0;
    c.rope_scaling = .none;
    c.attention_scale = 1.0 / @sqrt(@as(f32, @floatFromInt(m.qk_nope_head_dim)));
    @memset(c.rope_layers, false);
    for (c.layer_head_dim) |*hd| hd.* = c.head_dim;
    for (c.layer_kv_heads) |*kv| kv.* = c.num_kv_heads;
    for (c.layer_attn_scale) |*sc| sc.* = c.attention_scale;
}

/// GLM-5.3-Flash (`glm5_next`, `Glm5NextTextConfig`): Kimi Delta Attention
/// layers (with the safe lower-bound forget gate) 3:1 with NoPE MLA layers
/// whose DSA indexer pools `index_kpool` keys (run as its dense equivalent
/// within `index_topk`), manifold-constrained hyper-connections collapsed by
/// an unweighted mean, DeepSeek V3 routing and clamped SwiGLU everywhere.
fn extraGlm5Next(c: *Config, _: Allocator, obj: std.json.ObjectMap) !void {
    const lac: ?std.json.ObjectMap = getObj(obj, "linear_attn_config");
    var heads = getInt(obj, "linear_num_heads", 64);
    var head_dim = getInt(obj, "linear_head_dim", 128);
    var kernel = getInt(obj, "linear_conv_kernel_dim", 4);
    // `linear_lower_bound` (default -5.0; an explicit null selects the
    // softplus gate), or `linear_attn_config.gate_lower_bound` with
    // `safe_gate` restoring the default when it is null.
    var lower: ?f32 = -5.0;
    if (obj.get("linear_lower_bound")) |v| lower = switch (v) {
        .float => |f| @as(?f32, @floatCast(f)),
        .integer => |i| @as(?f32, @floatFromInt(i)),
        else => null,
    };
    if (lac) |l| {
        heads = getInt(l, "num_heads", heads);
        head_dim = getInt(l, "head_dim", head_dim);
        kernel = getInt(l, "short_conv_kernel_size", kernel);
        if (l.get("gate_lower_bound")) |v| lower = switch (v) {
            .float => |f| @as(?f32, @floatCast(f)),
            .integer => |i| @as(?f32, @floatFromInt(i)),
            else => null,
        };
        if (getBool(l, "safe_gate", true) and lower == null) lower = -5.0;
    }
    c.linear_k_heads = heads;
    c.linear_v_heads = heads;
    c.linear_k_dim = head_dim;
    c.linear_v_dim = head_dim;
    c.linear_conv_kernel = kernel;
    c.linear_gate_lower_bound = lower;
    if (obj.get("layer_types") == null) {
        for (c.linear_layers, 0..) |*is_lin, i| is_lin.* = (i % 4 != 3);
    }
    c.has_linear = false;
    for (c.linear_layers) |l| c.has_linear = c.has_linear or l;
    try mlaNoPe(c, obj, "glm5_next");
    if (getNum(obj, "rms_norm_eps") == null) c.rms_norm_eps = 1e-5;
    // Mixture of experts: sigmoid scores, correction bias, top-2 group
    // scores, renormalised with a floor, scaled; clamped SwiGLU in the
    // experts, the shared experts and the dense layers.
    c.num_experts_per_tok = getInt(obj, "num_experts_per_tok", 8);
    c.norm_topk_prob = getBool(obj, "norm_topk_prob", true);
    c.moe.scoring = .sigmoid;
    c.moe.topk_method = .group_limited;
    c.moe.n_group = @max(1, getInt(obj, "n_group", 1));
    c.moe.topk_group = @max(1, getInt(obj, "topk_group", 1));
    c.moe.routed_scaling_factor = getF32(obj, "routed_scaling_factor", 2.5);
    c.moe.group_score_top2 = true;
    c.moe.norm_eps_floor = true;
    if (c.num_experts % c.moe.n_group != 0) return error.InvalidConfig;
    const limit = getF32(obj, "swiglu_limit", 10.0);
    c.moe.swiglu_limit = if (limit > 0) limit else null;
    try mlpLayerTypes(c, obj, @min(3, c.num_layers));
    // Hyper-connections.
    c.hc_mult = @max(1, getInt(obj, "hc_mult", 4));
    c.hyper = .{
        .kind = .mhc,
        .head = .mean,
        .sinkhorn_iters = getInt(obj, "hc_sinkhorn_iters", 20),
        .eps = getF32(obj, "hc_eps", 1e-6),
    };
    // DSA indexer over k-pools: exact as dense while every complete pool is selected.
    const kpool = @max(1, getInt(obj, "index_kpool", 16));
    const topk = getInt(obj, "index_topk", 2048);
    if (topk % kpool != 0) return error.InvalidConfig;
    if (!getBool(obj, "index_kpool_always_select_tail", true)) {
        std.log.err("unsupported model: glm5_next without index_kpool_always_select_tail (the incomplete tail would be dropped)", .{});
        return error.UnsupportedArchitecture;
    }
    c.index_bound = .{ .block = kpool, .max_blocks = topk / kpool };
    // Glm5NextTextAttention builds its latent norms with `rms_norm_eps`.
    if (c.mla != null) c.mla.?.latent_norm_eps = c.rms_norm_eps;
    std.log.warn("glm5_next: the DSA indexer runs as dense attention, exact for prompts up to {d} tokens (index_topk); longer prompts are refused", .{topk});
}

/// Qwen3.8-Flash-Next (`qwen4_exp`, `Qwen4ExpTextConfig`): Gated DeltaNet
/// layers 3:1 with gated full attention behind a QSA indexer (run as its
/// dense equivalent within `indexer_budget`), gated hyper-connections
/// (`hc_count` streams, no final norm), softmax MoE with a gated shared
/// expert on every layer, per-layer n-gram embeddings on `ple_layer_ids`.
fn extraQwen4Exp(c: *Config, arena: Allocator, obj: std.json.ObjectMap) !void {
    c.qk_norm = .head;
    c.gated_attention = true;
    // The attention gate is always a sigmoid; `output_gate_type` (default
    // `hidden_act`) picks the Gated DeltaNet output gate.
    c.gate_swish = false;
    const gate = getStr(obj, "output_gate_type") orelse getStr(obj, "hidden_act") orelse "silu";
    if (std.mem.eql(u8, gate, "sigmoid")) {
        c.linear_gate_sigmoid = true;
    } else if (!std.mem.eql(u8, gate, "silu") and !std.mem.eql(u8, gate, "swish")) {
        std.log.err("unsupported output_gate_type '{s}' (sigmoid or silu)", .{gate});
        return error.UnsupportedArchitecture;
    }
    if (obj.get("layer_types") == null and getNum(obj, "full_attention_interval") == null) {
        for (c.linear_layers, 0..) |*l, i| l.* = ((i + 1) % 4 != 0);
        c.has_linear = false;
        for (c.linear_layers) |l| c.has_linear = c.has_linear or l;
    }
    // Every layer is a mixture of experts with a sigmoid-gated shared expert.
    if (getNum(obj, "num_experts") == null) c.num_experts = 512;
    c.num_experts_per_tok = getInt(obj, "num_experts_per_tok", 10);
    c.norm_topk_prob = getBool(obj, "norm_topk_prob", true);
    if (c.num_experts > 0) @memset(c.moe_layers, true);
    if (getNum(obj, "moe_intermediate_size") == null) c.moe_intermediate_size = 512;
    // Gated hyper-connections.
    c.hc_mult = @max(1, getInt(obj, "hc_count", 4));
    if (c.hc_mult < 2) {
        std.log.err("qwen4_exp needs hc_count > 1 (config has {d})", .{c.hc_mult});
        return error.InvalidConfig;
    }
    c.hyper = .{ .kind = .gated, .head = .gated_mixer, .lowrank = getInt(obj, "hc_lowrank", 320) };
    if (c.hyper.?.lowrank == 0) return error.InvalidConfig;
    // QSA indexer: `indexer_budget` tokens from complete blocks of
    // `indexer_compress_ratio` keys plus the incomplete tail.
    if (getNum(obj, "indexer_budget") != null or getNum(obj, "indexer_compress_ratio") != null or getNum(obj, "indexer_n_heads") != null) {
        const budget = getInt(obj, "indexer_budget", 0);
        const ratio = getInt(obj, "indexer_compress_ratio", 0);
        if (budget == 0 or ratio == 0 or budget % ratio != 0 or getInt(obj, "indexer_kv_heads", 1) != 1) {
            std.log.err("qwen4_exp: the QSA config needs indexer_budget (a multiple of indexer_compress_ratio) and indexer_kv_heads = 1", .{});
            return error.InvalidConfig;
        }
        c.index_bound = .{ .block = ratio, .max_blocks = budget / ratio };
        std.log.warn("qwen4_exp: the QSA indexer runs as dense attention, exact for prompts up to {d} tokens (indexer_budget); longer prompts are refused", .{budget});
    }
    // Per-layer n-gram embeddings.
    if (try intList(arena, obj, "ple_layer_ids")) |ids_1| {
        if (ids_1.len > 0) {
            var ids = std.ArrayList(usize).empty;
            for (ids_1) |id1| {
                if (id1 < 1 or id1 > c.num_layers) {
                    std.log.err("qwen4_exp: ple_layer_ids entry {d} is outside 1..{d}", .{ id1, c.num_layers });
                    return error.InvalidConfig;
                }
                const id0 = id1 - 1;
                var seen = false;
                for (ids.items) |x| seen = seen or x == id0;
                if (!seen) try ids.append(arena, id0);
            }
            std.mem.sort(usize, ids.items, {}, std.sort.asc(usize));
            for (ids.items) |id0| if (!c.linear_layers[id0]) {
                std.log.err("qwen4_exp: PLE layer {d} is not a linear_attention layer", .{id0 + 1});
                return error.UnsupportedArchitecture;
            };
            const ngram = getInt(obj, "ngram_size", 3);
            const heads = getInt(obj, "heads_per_ngram", 8);
            const embed_dim = getInt(obj, "ple_embed_dim", c.hidden_size);
            const n_cols = (ngram -| 1) * heads;
            if (ngram < 2 or heads == 0 or embed_dim == 0 or embed_dim % n_cols != 0) return error.InvalidConfig;
            const eos: u32 = blk: {
                const v = obj.get("eos_token_id") orelse break :blk 0xFFFFFFFF;
                break :blk switch (v) {
                    .integer => |i| if (i >= 0) @as(u32, @intCast(i)) else 0xFFFFFFFF,
                    .array => |a| if (a.items.len > 0 and a.items[0] == .integer and a.items[0].integer >= 0) @as(u32, @intCast(a.items[0].integer)) else 0xFFFFFFFF,
                    else => 0xFFFFFFFF,
                };
            };
            if (eos == 0xFFFFFFFF) {
                std.log.err("qwen4_exp: eos_token_id must be set in the text config when PLE layers are enabled (it pads the n-gram history)", .{});
                return error.InvalidConfig;
            }
            c.ngram_ple = .{
                .layer_ids = ids.items,
                .embed_dim = embed_dim,
                .conv_kernel = @max(1, getInt(obj, "ple_conv_kernel_size", 4)),
                .ngram_size = ngram,
                .heads_per_ngram = heads,
                .vocab_base = getInt(obj, "ngram_vocab_size_base", 20_000_000),
                .divisible_by = @max(1, getInt(obj, "make_ngram_vocab_size_divisible_by", 128)),
                .seed = getInt(obj, "seed", 1234),
                .vocab_size = c.vocab_size,
                .eos_id = eos,
            };
        }
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
    // `.minimax` takes the residual *after* each norm: the remote code's
    // `postnorm: true`, which every released checkpoint sets, and the only
    // layout transformers' native `minimax` has (it ignores the key). The
    // remote code's default, `postnorm: false`, keeps the residual from
    // before the norm; no release uses it and it is not implemented.
    if (!std.mem.eql(u8, c.model_type, "minimax") and !getBool(obj, "postnorm", false)) {
        std.log.err("unsupported model: MiniMax with postnorm: false (the residual taken before the norm); every release sets postnorm: true", .{});
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
///
/// Kimi K3 reuses this config with `attn_res_block_size` (Attention
/// Residual), `routed_expert_hidden_size` / `latent_moe_use_norm` (latent
/// MoE), `mla_use_output_gate`, `use_full_rank_gate` / `gate_lower_bound` in
/// `linear_attn_config` and the `situ` activation (parsed generically).
fn extraKimiLinear(c: *Config, _: Allocator, obj: std.json.ObjectMap) !void {
    const lac: ?std.json.ObjectMap = getObj(obj, "linear_attn_config");
    var heads = getInt(obj, "linear_num_heads", 32);
    var head_dim = getInt(obj, "linear_head_dim", 128);
    var kernel = getInt(obj, "linear_conv_kernel_dim", 4);
    if (lac) |l| {
        heads = getInt(l, "num_heads", heads);
        head_dim = getInt(l, "head_dim", head_dim);
        kernel = getInt(l, "short_conv_kernel_size", kernel);
        c.linear_full_rank_gate = getBool(l, "use_full_rank_gate", false);
        if (getNum(l, "gate_lower_bound")) |lb| c.linear_gate_lower_bound = @floatCast(lb);
    }
    c.mla_output_gate = getBool(obj, "mla_use_output_gate", false);
    c.attn_res_block = getInt(obj, "attn_res_block_size", 0);
    c.moe_latent = getInt(obj, "routed_expert_hidden_size", 0);
    c.moe_latent_norm = getBool(obj, "latent_moe_use_norm", false);
    if (getBool(obj, "mla_use_nope", true) == false) {
        std.log.err("kimi_linear: full-attention layers with RoPE (mla_use_nope = false) are not implemented", .{});
        return error.UnsupportedArchitecture;
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

/// Xiaomi MiMo V2 (`MiMoV2FlashConfig` in transformers; the hub checkpoints of
/// MiMo-V2-Flash, V2.5 and V2.6 carry the remote-code spellings of the same
/// settings: `hybrid_layer_pattern`, `swa_*`, `moe_layer_freq`, ...). Hybrid
/// attention: full layers with `num_key_value_heads` KV heads, sliding layers
/// with `swa_num_key_value_heads` (twice as many) plus attention sinks; values
/// are `v_head_dim` wide and scaled by `attention_value_scale`; partial rotary
/// with one base per layer type; the first layer is dense, the rest
/// DeepSeek-V3-style sigmoid MoE (correction bias, group-limited top-k) without
/// shared experts. MTP (`model.mtp.*`), vision and audio tensors are never read.
fn extraMiMoV2(c: *Config, arena: Allocator, obj: std.json.ObjectMap) !void {
    const n = c.num_layers;
    const hf_spelling = std.mem.eql(u8, c.model_type, "mimo_v2_flash") or obj.get("layer_types") != null or obj.get("rope_parameters") != null or obj.get("mlp_layer_types") != null;
    if (getNum(obj, "rms_norm_eps") == null) c.rms_norm_eps = getF32(obj, "layernorm_epsilon", 1e-5);
    // `store_dtype` names the expert storage format. MiMo V2.6 carries it in
    // `quantization_config` (parsed by dequant.zig, which decodes the MXFP4
    // experts on read); a copy at the config level says the same thing.
    if (getStr(obj, "store_dtype")) |sd| {
        if (!dequant.expertDtypeSupported(sd)) {
            std.log.err("unsupported model: MiMo experts stored as '{s}' (store_dtype) cannot be dequantised; convert the experts to bf16 first", .{sd});
            return error.UnsupportedArchitecture;
        }
    }
    // The router runs in f32 on the bf16 gate weights whatever
    // `moe_router_dtype` says (MiMo V2.6: bfloat16, the earlier ones float32);
    // a quantised router would need a reader of its own.
    if (getStr(obj, "moe_router_dtype")) |rd| {
        if (!dequant.storeFloatDtype(rd)) {
            std.log.err("unsupported model: MiMo MoE router in '{s}' (moe_router_dtype)", .{rd});
            return error.UnsupportedArchitecture;
        }
    }
    // Attention layout per layer: `layer_types` (parsed above), the remote-code
    // `hybrid_layer_pattern` (1 = sliding) or the default (first and every
    // sixth layer full, the rest sliding).
    if (obj.get("layer_types") == null) {
        if (obj.get("hybrid_layer_pattern")) |hp| {
            if (hp != .array or hp.array.items.len < n) return error.InvalidConfig;
            for (c.sliding_layers, 0..) |*s, i| s.* = hp.array.items[i] == .integer and hp.array.items[i].integer == 1;
        } else {
            for (c.sliding_layers, 0..) |*s, i| s.* = !(i == 0 or (i + 1) % 6 == 0);
        }
    }
    if (c.sliding_window == null) c.sliding_window = getInt(obj, "sliding_window_size", 128);
    var any_sliding = false;
    for (c.sliding_layers) |s| any_sliding = any_sliding or s;
    c.sinks = any_sliding and getBool(obj, "add_swa_attention_sink_bias", true);
    c.sinks_sliding_only = true;
    // Heads: the sliding layers double the kv heads; every other dimension is shared.
    const kv_full = getInt(obj, "num_key_value_heads", c.num_heads);
    const kv_swa = getInt(obj, "swa_num_key_value_heads", 2 * kv_full);
    c.v_head_dim = getInt(obj, "v_head_dim", c.head_dim);
    c.narrow_values = true;
    if (getInt(obj, "swa_num_attention_heads", c.num_heads) != c.num_heads or getInt(obj, "swa_head_dim", c.head_dim) != c.head_dim or getInt(obj, "swa_v_head_dim", c.v_head_dim) != c.v_head_dim) {
        std.log.err("unsupported model: MiMo sliding layers with their own head count or head size (swa_num_attention_heads / swa_head_dim / swa_v_head_dim)", .{});
        return error.UnsupportedArchitecture;
    }
    if (c.v_head_dim > c.head_dim) {
        // Values share the keys' cache stride, so they may be narrower (every
        // release: 192 / 128) but not wider.
        std.log.err("unsupported model: MiMo v_head_dim {d} is wider than head_dim {d}", .{ c.v_head_dim, c.head_dim });
        return error.UnsupportedArchitecture;
    }
    if (c.v_head_dim == 0 or kv_full == 0 or kv_swa == 0) return error.InvalidConfig;
    if (c.num_heads % kv_full != 0 or c.num_heads % kv_swa != 0) return error.InvalidConfig;
    for (c.layer_kv_heads, 0..) |*k, i| k.* = if (c.sliding_layers[i]) kv_swa else kv_full;
    c.num_kv_heads = kv_full;
    // Values are scaled before attention. transformers defaults the scale to
    // 0.707 and reads an explicit null as 1; the remote-code modules default to 1.
    c.mult.value = if (obj.get("attention_value_scale")) |v| (if (v == .null) 1.0 else getF32(obj, "attention_value_scale", 1.0)) else if (hf_spelling) 0.707 else 1.0;
    // Rotary: one base per layer type, partial rotary factor 0.334 (int(head_dim * f) dims).
    var factor: f64 = if (getNum(obj, "partial_rotary_factor")) |f| f else if (hf_spelling) 0.334 else 1.0;
    if (hf_spelling and getNum(obj, "rope_theta") == null and getObj(obj, "rope_parameters") == null) c.rope_theta = 5_000_000.0;
    var local_theta = getF32(obj, "swa_rope_theta", if (hf_spelling) 10_000.0 else c.rope_theta);
    if (getObj(obj, "rope_parameters")) |rp| {
        const full: ?std.json.ObjectMap = getObj(rp, "full_attention") orelse (if (getNum(rp, "rope_theta") != null) rp else null);
        const swa: ?std.json.ObjectMap = getObj(rp, "sliding_attention") orelse (if (getNum(rp, "rope_theta") != null) rp else null);
        if (full) |f| {
            c.rope_theta = getF32(f, "rope_theta", 5_000_000.0);
            factor = getNum(f, "partial_rotary_factor") orelse 0.334;
            c.rope_scaling = try parseRopeScaling(arena, obj, f, c.rotary_dim, c.max_position_embeddings);
        }
        if (swa) |s| {
            local_theta = getF32(s, "rope_theta", 10_000.0);
            const sf = getNum(s, "partial_rotary_factor") orelse 0.334;
            const t = getStr(s, "rope_type") orelse getStr(s, "type") orelse "default";
            if (sf != factor or !(std.mem.eql(u8, t, "default") or std.mem.eql(u8, t, "mrope"))) {
                std.log.err("unsupported model: MiMo sliding layers with their own rotary factor or scaling ({s})", .{t});
                return error.UnsupportedArchitecture;
            }
        }
    }
    c.rotary_dim = evenDim(@as(f64, @floatFromInt(c.head_dim)) * factor);
    if (c.rotary_dim == 0 or c.rotary_dim > c.head_dim) return error.InvalidConfig;
    c.rope_freq_dim = c.rotary_dim;
    // The sliding layers rotate the same coordinates with their own base.
    c.rope_local = .{ .theta = local_theta, .rotary_dim = c.rotary_dim, .freq_dim = c.rotary_dim };
    // Mixture of experts: sigmoid scores plus a correction bias choose the
    // experts (group-limited top-k), the raw scores weight them.
    c.moe.scoring = parseScoring(getStr(obj, "scoring_func") orelse "sigmoid");
    const method = getStr(obj, "topk_method") orelse "noaux_tc";
    c.moe.topk_method = if (std.mem.eql(u8, method, "greedy")) .greedy else .group_limited;
    c.moe.n_group = @max(1, getInt(obj, "n_group", 1));
    c.moe.topk_group = @max(1, getInt(obj, "topk_group", 1));
    c.moe.routed_scaling_factor = getF32(obj, "routed_scaling_factor", 1.0);
    c.norm_topk_prob = getBool(obj, "norm_topk_prob", true);
    c.num_experts_per_tok = getInt(obj, "num_experts_per_tok", 8);
    if (c.num_experts > 0) {
        if (c.moe.topk_method == .group_limited and c.num_experts % c.moe.n_group != 0) return error.InvalidConfig;
        for (c.moe_layers, 0..) |*m, i| m.* = i > 0;
        if (obj.get("mlp_layer_types")) |ml| {
            if (ml != .array or ml.array.items.len < n) return error.InvalidConfig;
            for (ml.array.items[0..n], 0..) |v, i| {
                if (v != .string) return error.InvalidConfig;
                c.moe_layers[i] = if (std.mem.eql(u8, v.string, "sparse")) true else if (std.mem.eql(u8, v.string, "dense")) false else return error.InvalidConfig;
            }
        } else if (obj.get("moe_layer_freq")) |mf| {
            if (mf == .array) {
                if (mf.array.items.len < n) return error.InvalidConfig;
                for (mf.array.items[0..n], 0..) |v, i| c.moe_layers[i] = !(v == .integer and v.integer == 0);
            }
        }
    }
    // MiMo-V2.5 / V2.6 Pro checkpoints fuse q/k/v into one `qkv_proj`
    // pre-sharded over `num_key_value_heads` chunks, each `[Q | K | V]`.
    c.qkv_alt = .grouped;
    c.qkv_chunks = kv_full;
}

/// OLMoE / FlexOLMo: softmax over every expert, top-k, optional renormalisation.
fn extraOlmoe(c: *Config, _: Allocator, obj: std.json.ObjectMap) !void {
    c.qk_norm = .full;
    c.norm_topk_prob = getBool(obj, "norm_topk_prob", false);
    if (c.num_experts > 0) @memset(c.moe_layers, true);
}

/// The DeepSeek-V3 router shared by dots.llm1, EXAONE-MoE, Solar Open and
/// A.X-K1: sigmoid scores, a correction bias that only steers the choice,
/// group-limited top-k on the two best scores per group, renormalisation and
/// a routed scaling factor, plus always-on shared experts.
fn dsRouter(c: *Config, obj: std.json.ObjectMap) !void {
    c.moe.scoring = .sigmoid;
    c.moe.topk_method = .group_limited;
    c.moe.n_group = @max(1, getInt(obj, "n_group", 1));
    c.moe.topk_group = @max(1, getInt(obj, "topk_group", 1));
    c.moe.routed_scaling_factor = getF32(obj, "routed_scaling_factor", 1.0);
    c.moe.norm_eps_floor = true;
    c.norm_topk_prob = getBool(obj, "norm_topk_prob", true);
    if (c.num_experts > 0 and c.num_experts % c.moe.n_group != 0) return error.InvalidConfig;
}

fn extraDots1(c: *Config, _: Allocator, obj: std.json.ObjectMap) !void {
    c.qk_norm = .head;
    try dsRouter(c, obj);
}

fn extraExaoneMoe(c: *Config, arena: Allocator, obj: std.json.ObjectMap) !void {
    c.qk_norm = .head;
    try dsRouter(c, obj);
    if (obj.get("layer_types") == null and c.sliding_window != null) {
        const pattern = getInt(obj, "sliding_window_pattern", 4);
        if (pattern > 0) for (c.sliding_layers, 0..) |*s, i| {
            s.* = ((i + 1) % pattern != 0);
        };
    }
    _ = arena;
}

fn extraSolarOpen(c: *Config, _: Allocator, obj: std.json.ObjectMap) !void {
    try dsRouter(c, obj);
    if (c.num_experts > 0) @memset(c.moe_layers, true);
}

fn extraAxk1(c: *Config, _: Allocator, obj: std.json.ObjectMap) !void {
    c.rope_style = if (getBool(obj, "rope_interleave", true)) .gptj else .neox;
    try dsRouter(c, obj);
}

/// AFMoE: a sigmoid router whose selection adds a per-expert bias, weights
/// renormalised and scaled by `route_scale`, always-on shared experts, a
/// sigmoid gate on the attention output and 1-in-`global_attn_every_n_layers`
/// global attention.
fn extraAfmoe(c: *Config, _: Allocator, obj: std.json.ObjectMap) !void {
    c.qk_norm = .head;
    c.attn_gate = .sigmoid;
    c.moe.scoring = .sigmoid;
    c.moe.routed_scaling_factor = getF32(obj, "route_scale", 1.0);
    c.moe.norm_eps_floor = true;
    c.norm_topk_prob = true;
    if (obj.get("layer_types") == null) {
        const every = @max(1, getInt(obj, "global_attn_every_n_layers", 4));
        for (c.sliding_layers, 0..) |*s, i| s.* = ((i + 1) % every != 0);
    }
    // Only the local layers are roped; the full-attention ones are NoPE
    // (`AfmoeAttention` applies the rotary under `if self.is_local_attention`).
    for (c.rope_layers, 0..) |*r, i| r.* = c.sliding_layers[i];
    // μP: the embeddings are scaled by sqrt(hidden_size) (arcee-ai/Trinity-Nano-Preview sets it).
    if (getBool(obj, "mup_enabled", false)) c.embed_scale = @sqrt(@as(f32, @floatFromInt(c.hidden_size)));
}

/// Mellum: a plain softmax top-k router (renormalised) over fused experts.
fn extraMellum(c: *Config, _: Allocator, obj: std.json.ObjectMap) !void {
    c.qk_norm = .head;
    c.norm_topk_prob = getBool(obj, "norm_topk_prob", true);
}

/// Laguna: sigmoid routing with a `tanh` softcap on the router logits and a
/// correction bias, renormalised weights scaled after the shared expert is
/// added, and a softplus gate on the attention output.
fn extraLaguna(c: *Config, _: Allocator, obj: std.json.ObjectMap) !void {
    // The released checkpoints give full and sliding layers different query
    // head counts (48 / 64 on Laguna XS.2); the kv heads are shared.
    if (obj.get("num_attention_heads_per_layer")) |v| if (v == .array) {
        if (v.array.items.len != c.num_layers) return error.InvalidConfig;
        for (v.array.items, c.layer_heads) |item, *nh| {
            if (item != .integer or item.integer <= 0) return error.InvalidConfig;
            nh.* = @intCast(item.integer);
            if (c.num_kv_heads == 0 or nh.* % c.num_kv_heads != 0) return error.InvalidConfig;
        }
    };
    c.qk_norm = .head;
    c.attn_gate = .softplus;
    c.moe.scoring = .sigmoid;
    c.norm_topk_prob = true;
    c.moe.routed_scaling_factor = getF32(obj, "moe_routed_scaling_factor", 1.0);
    const cap = getF32(obj, "moe_router_logit_softcapping", 0);
    if (cap > 0) c.moe.router_softcap = cap;
    if (getBool(obj, "moe_apply_router_weight_on_input", false)) c.moe.scale_input = true;
}

/// HunYuan V3: sigmoid routing with a correction bias, renormalised and
/// scaled, shared experts and a per-head q/k norm before RoPE.
fn extraHyV3(c: *Config, _: Allocator, obj: std.json.ObjectMap) !void {
    c.qk_norm = .head;
    c.moe.scoring = .sigmoid;
    c.moe.norm_eps_floor = true;
    c.norm_topk_prob = true;
    c.moe.routed_scaling_factor = getF32Any(obj, &.{ "router_scaling_factor", "routed_scaling_factor" }, 1.0);
}

/// Granite 4 SWA: the Granite multipliers plus per-head attention sinks and
/// sliding layers with their own rope base (`layer_rope_theta`).
fn extraGraniteSwa(c: *Config, arena: Allocator, obj: std.json.ObjectMap) !void {
    try extraGranite(c, arena, obj);
    c.sinks = true;
}

fn extraGraniteMoeSwa(c: *Config, arena: Allocator, obj: std.json.ObjectMap) !void {
    try extraGraniteMoe(c, arena, obj);
    c.sinks = true;
}

fn extraDeepseekV32(c: *Config, arena: Allocator, obj: std.json.ObjectMap) !void {
    try extraDeepseek(c, arena, obj);
    // The lightning indexer keeps the best `index_topk` keys per query, so
    // dense attention is exactly the reference up to that many tokens.
    const topk = getInt(obj, "index_topk", 2048);
    if (topk > 0) c.index_bound = .{ .block = 1, .max_blocks = topk };
}

fn parseScoring(name: []const u8) RouterScoring {
    if (std.mem.eql(u8, name, "sigmoid")) return .sigmoid;
    if (std.mem.eql(u8, name, "sqrtsoftplus")) return .sqrtsoftplus;
    return .softmax;
}

/// Integer list `key` of `obj` (null when absent or not an array).
fn intList(arena: Allocator, obj: std.json.ObjectMap, key: []const u8) !?[]usize {
    const v = obj.get(key) orelse return null;
    if (v != .array) return null;
    const out = try arena.alloc(usize, v.array.items.len);
    for (v.array.items, 0..) |item, i| out[i] = switch (item) {
        .integer => |n| if (n >= 0) @intCast(n) else 0,
        .float => |f| if (f >= 0) @intFromFloat(f) else 0,
        else => 0,
    };
    return out;
}

/// The parts of the DeepSeek V4 / V4.1 configs both families share: hyper-
/// connections, the low-rank query and grouped output projections, the
/// compressed-branch RoPE (`compress_rope_theta`, yarn only there, with the
/// reference's `attention_factor = 1`), sinks, the clamped expert SwiGLU and
/// the MoE routing.
fn dsv4Common(c: *Config, arena: Allocator, obj: std.json.ObjectMap, d: *DsV4) !void {
    c.rope_style = .gptj;
    c.sinks = true;
    c.norm = .rms;
    c.num_kv_heads = getInt(obj, "num_key_value_heads", 1);
    if (c.num_kv_heads != 1) {
        std.log.err("deepseek_v4: shared-KV attention needs num_key_value_heads = 1 (config has {d})", .{c.num_kv_heads});
        return error.UnsupportedArchitecture;
    }
    // Every layer carries the sliding window; the compressed branch is extra.
    @memset(c.sliding_layers, true);
    if (c.sliding_window == null) c.sliding_window = 128;
    if (getNum(obj, "qk_rope_head_dim")) |_| c.rotary_dim = getInt(obj, "qk_rope_head_dim", c.rotary_dim);
    c.rotary_dim -= c.rotary_dim % 2;
    if (c.rotary_dim == 0 or c.rotary_dim > c.head_dim) return error.InvalidConfig;
    c.v_head_dim = c.head_dim;
    c.attention_scale = 1.0 / @sqrt(@as(f32, @floatFromInt(c.head_dim)));
    c.hc_mult = @max(1, getInt(obj, "hc_mult", 4));
    d.hc_mult = c.hc_mult;
    d.hc_sinkhorn_iters = getInt(obj, "hc_sinkhorn_iters", 20);
    d.hc_eps = getF32(obj, "hc_eps", 1e-6);
    c.hyper = .{
        .kind = if (d.v41) .mhc_single_pass else .mhc,
        .head = if (d.v41) .previous_pre else .weighted,
        .sinkhorn_iters = d.hc_sinkhorn_iters,
        .eps = d.hc_eps,
    };
    d.q_lora_rank = getInt(obj, "q_lora_rank", 0);
    d.o_groups = @max(1, getInt(obj, "o_groups", 8));
    d.o_lora_rank = getInt(obj, "o_lora_rank", 1024);
    if (d.q_lora_rank == 0 or d.o_lora_rank == 0 or (c.num_heads * c.head_dim) % d.o_groups != 0) return error.InvalidConfig;
    d.index_topk = getInt(obj, "index_topk", 512);
    d.index_head_dim = getInt(obj, "index_head_dim", 128);
    // The compressed branches rotate with their own base and (yarn) scaling;
    // the sliding-window rope is plain. The reference never multiplies the
    // compress cos/sin by yarn's mscale unless the config says so.
    d.compress_rope_theta = getF32(obj, "compress_rope_theta", 160000.0);
    var compress_dict: ?std.json.ObjectMap = null;
    if (getObj(obj, "rope_parameters")) |rp| {
        if (getObj(rp, "main")) |m| c.rope_theta = getF32(m, "rope_theta", c.rope_theta);
        if (getObj(rp, "compress")) |cp| {
            compress_dict = cp;
            d.compress_rope_theta = getF32(cp, "rope_theta", d.compress_rope_theta);
        }
    }
    if (compress_dict == null) compress_dict = getObj(obj, "rope_scaling");
    d.compress_rope_scaling = .none;
    if (compress_dict) |cd| {
        d.compress_rope_scaling = try parseRopeScaling(arena, obj, cd, c.rotary_dim, c.max_position_embeddings);
        if (d.compress_rope_scaling == .yarn and getNum(cd, "attention_factor") == null) d.compress_rope_scaling.yarn.attention_factor = 1.0;
    }
    c.rope_scaling = .none;
    // MoE: sqrtsoftplus scores, plain top-k on the corrected scores, renormalised.
    c.moe.scoring = parseScoring(getStr(obj, "scoring_func") orelse "sqrtsoftplus");
    c.moe.topk_method = .greedy;
    c.moe.routed_scaling_factor = getF32(obj, "routed_scaling_factor", 1.5);
    c.moe.norm_eps_floor = true;
    const limit = getF32(obj, "swiglu_limit", 10.0);
    c.moe.swiglu_limit = if (limit > 0) limit else null;
    c.num_experts_per_tok = getInt(obj, "num_experts_per_tok", 6);
    if (getNum(obj, "intermediate_size") == null) c.intermediate_size = c.moe_intermediate_size;
    if (c.num_experts > 0) @memset(c.moe_layers, true);
}

fn extraDeepseekV4(c: *Config, arena: Allocator, obj: std.json.ObjectMap) !void {
    var d: DsV4 = undefined;
    d.v41 = false;
    d.fake_quant = false;
    d.engram = null;
    d.candidate_source = null;
    d.candidate_topk_blocks = 0;
    d.candidate_block_size = 1;
    try dsv4Common(c, arena, obj, &d);
    d.index_n_heads = getInt(obj, "index_n_heads", 64);
    c.norm_topk_prob = true;
    const n = c.num_layers;
    // Per-layer-type compression rates (`compress_rates`, or the legacy scalars).
    var rate_csa: usize = 4;
    var rate_hca: usize = 128;
    if (getObj(obj, "compress_rates")) |cr| {
        rate_csa = getInt(cr, "compressed_sparse_attention", rate_csa);
        rate_hca = getInt(cr, "heavily_compressed_attention", rate_hca);
    }
    rate_csa = getInt(obj, "compress_rate_csa", rate_csa);
    rate_hca = getInt(obj, "compress_rate_hca", rate_hca);
    const branch = try arena.alloc(CompressBranch, n);
    const ratio = try arena.alloc(usize, n);
    const source = try arena.alloc(?usize, n);
    if (obj.get("layer_types")) |lt| {
        if (lt != .array or lt.array.items.len < n) return error.InvalidConfig;
        for (lt.array.items[0..n], 0..) |v, i| {
            if (v != .string) return error.InvalidConfig;
            branch[i] = if (std.mem.eql(u8, v.string, "sliding_attention")) .none else if (std.mem.eql(u8, v.string, "compressed_sparse_attention")) .csa else if (std.mem.eql(u8, v.string, "heavily_compressed_attention")) .hca else return error.InvalidConfig;
        }
    } else if (try intList(arena, obj, "compress_ratios")) |legacy| {
        // Legacy per-layer ints keyed by the default rates: 0 / 4 / 128.
        if (legacy.len < n) return error.InvalidConfig;
        for (legacy[0..n], 0..) |r, i| branch[i] = switch (r) {
            0 => .none,
            4 => .csa,
            128 => .hca,
            else => return error.InvalidConfig,
        };
    } else {
        // V4-Pro default: two HCA layers, then CSA on odd and HCA on even indices.
        for (branch, 0..) |*b, i| b.* = if (i < 2) .hca else if ((i - 2) % 2 == 1) .csa else .hca;
    }
    for (branch, 0..) |b, i| {
        ratio[i] = switch (b) {
            .none => 0,
            .csa => rate_csa,
            .hca => rate_hca,
            .shared => unreachable,
        };
        source[i] = if (b == .none) null else i;
        if (b != .none and ratio[i] == 0) return error.InvalidConfig;
    }
    d.branch = branch;
    d.compress_ratio = ratio;
    d.kv_source = source;
    // Hash-routed MoE layers: `mlp_layer_types`, or the first `num_hash_layers` (default 3).
    const hash = try arena.alloc(bool, n);
    if (obj.get("mlp_layer_types")) |mt| {
        if (mt != .array or mt.array.items.len < n) return error.InvalidConfig;
        for (mt.array.items[0..n], 0..) |v, i| {
            if (v != .string) return error.InvalidConfig;
            hash[i] = if (std.mem.eql(u8, v.string, "hash_moe")) true else if (std.mem.eql(u8, v.string, "moe")) false else return error.InvalidConfig;
        }
    } else {
        const n_hash = getInt(obj, "num_hash_layers", 3);
        for (hash, 0..) |*h, i| h.* = i < n_hash;
    }
    d.hash_moe_layers = hash;
    c.dsv4 = d;
}

fn extraDeepseekV41(c: *Config, arena: Allocator, obj: std.json.ObjectMap) !void {
    var d: DsV4 = undefined;
    d.v41 = true;
    d.fake_quant = true;
    if (getNum(obj, "rms_norm_eps") == null) c.rms_norm_eps = 1e-20;
    try dsv4Common(c, arena, obj, &d);
    d.index_n_heads = getInt(obj, "index_n_heads", 32);
    c.norm_topk_prob = getBool(obj, "norm_topk_prob", true);
    c.moe.gate_temp = getF32(obj, "gate_temp", 1.0);
    if (c.moe.gate_temp == 0) return error.InvalidConfig;
    const n = c.num_layers;
    const n_nextn = getInt(obj, "num_nextn_predict_layers", 3);
    // Per-layer pooling ratio: 0 sliding only, 1 full-resolution shared KV,
    // r > 1 pooled. Trailing entries (the draft layers) are ignored.
    const ratio = try arena.alloc(usize, n);
    if (try intList(arena, obj, "compress_ratios")) |cr| {
        if (cr.len < n or cr.len > n + n_nextn) return error.InvalidConfig;
        @memcpy(ratio, cr[0..n]);
    } else if (n == 40) {
        for (ratio, 0..) |*r, i| r.* = if (i < 2) 0 else if (i < 20) 2 else 1;
    } else {
        const n_slide = @min(2, n);
        const n_enc = (n - n_slide + 1) / 2;
        for (ratio, 0..) |*r, i| r.* = if (i < n_slide) 0 else if (i < n_slide + n_enc) 2 else 1;
    }
    // KV sources: explicit, or the first layer of every run of equal ratios.
    var sources: []usize = &.{};
    if (try intList(arena, obj, "kv_source_layer_ids")) |ks| {
        sources = ks;
    } else {
        var list = std.ArrayList(usize).empty;
        for (ratio, 0..) |r, i| {
            if (r > 0 and (i == 0 or ratio[i - 1] != r)) try list.append(arena, i);
        }
        sources = list.items;
    }
    const branch = try arena.alloc(CompressBranch, n);
    const source = try arena.alloc(?usize, n);
    for (0..n) |i| {
        branch[i] = if (ratio[i] > 0) .shared else .none;
        source[i] = null;
        if (ratio[i] == 0) continue;
        for (sources) |sidx| {
            if (sidx <= i and (source[i] == null or sidx > source[i].?)) source[i] = sidx;
        }
        const src = source[i] orelse {
            std.log.err("deepseek_v41: layer {d} has a compressed branch but no kv_source_layer_ids entry at or before it", .{i});
            return error.InvalidConfig;
        };
        if (src >= n or ratio[src] == 0) return error.InvalidConfig;
        if (ratio[src] != ratio[i]) {
            std.log.err("deepseek_v41: layer {d} (ratio {d}) reads the compressed cache of layer {d} (ratio {d})", .{ i, ratio[i], src, ratio[src] });
            return error.InvalidConfig;
        }
    }
    d.branch = branch;
    d.compress_ratio = ratio;
    d.kv_source = source;
    d.candidate_topk_blocks = getInt(obj, "candidate_topk_blocks", 2048);
    d.candidate_block_size = @max(1, getInt(obj, "candidate_block_size", 8));
    d.candidate_source = null;
    if (obj.get("candidate_source_layer_id")) |v| {
        if (v == .integer and v.integer >= 0) d.candidate_source = @intCast(v.integer);
    } else if (sources.len > 0) d.candidate_source = sources[sources.len - 1];
    if (d.candidate_source) |cs| if (cs >= n) return error.InvalidConfig;
    const hash = try arena.alloc(bool, n);
    @memset(hash, false);
    d.hash_moe_layers = hash;
    // Engram conditional memory.
    d.engram = null;
    if (try intList(arena, obj, "engram_layer_ids")) |ids| {
        if (ids.len > 0) {
            const counts = (try intList(arena, obj, "engram_num_embeddings")) orelse return error.InvalidConfig;
            if (counts.len != ids.len) return error.InvalidConfig;
            for (ids) |li| if (li >= n) return error.InvalidConfig;
            const pad = getInt(obj, "engram_pad_id", 2);
            d.engram = .{
                .layer_ids = ids,
                .num_embeddings = counts,
                .max_ngram = @max(2, getInt(obj, "engram_max_ngram_size", 4)),
                .n_heads = @max(1, getInt(obj, "engram_n_heads", 8)),
                .head_dim = getInt(obj, "engram_head_dim", 256),
                .pad_id = @intCast(pad),
                .compressed_vocab_size = getInt(obj, "engram_compressed_vocab_size", 99092),
                .vocab_size = getInt(obj, "engram_vocab_size", 16000000),
            };
        }
    }
    c.dsv4 = d;
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
    .latent_down = "block_sparse_moe.routed_expert_down_proj.weight",
    .latent_up = "block_sparse_moe.routed_expert_up_proj.weight",
    .latent_norm = "block_sparse_moe.routed_expert_norm.weight",
};

/// Kimi Linear and Kimi K3 (`KimiLinearForCausalLM`): the K3-only tensors
/// (full-rank KDA gate, MLA output gate, Attention Residual scorers, latent
/// MoE projections) are read only when the config enables them.
const kimi_linear_names = Names{
    .q_a = "self_attn.q_a_proj.weight",
    .q_a_norm = "self_attn.q_a_layernorm.weight",
    .q_b = "self_attn.q_b_proj.weight",
    .kv_a = "self_attn.kv_a_proj_with_mqa.weight",
    .kv_a_norm = "self_attn.kv_a_layernorm.weight",
    .kv_b = "self_attn.kv_b_proj.weight",
    .attn_gate = "self_attn.g_proj.weight",
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
    .lin_g = "self_attn.g_proj.weight",
    .lin_norm = "self_attn.o_norm.weight",
    .lin_out = "self_attn.o_proj.weight",
    .attn_res_norm = &.{ "self_attention_res_norm.weight", "self_attention_res.norm_weight" },
    .attn_res_proj = &.{ "self_attention_res_proj.weight", "self_attention_res.proj_weight" },
    .mlp_res_norm = &.{ "mlp_res_norm.weight", "mlp_res.norm_weight" },
    .mlp_res_proj = &.{ "mlp_res_proj.weight", "mlp_res.proj_weight" },
    .output_res_norm = &.{ "{p}output_attn_res_norm.weight", "{p}output_attn_res.norm_weight" },
    .output_res_proj = &.{ "{p}output_attn_res_proj.weight", "{p}output_attn_res.proj_weight" },
    .router_correction_bias = "mlp.gate.e_score_correction_bias",
    .shared_expert = "mlp.shared_experts.",
    .latent_down = "mlp.routed_expert_down_proj.weight",
    .latent_up = "mlp.routed_expert_up_proj.weight",
    .latent_norm = "mlp.routed_expert_norm.weight",
    .moe_alt = &kimi_linear_checkpoint_moe,
};

/// Hunyuan V3 as the transformers module names it (the released checkpoints
/// use `mlp.router.gate`, `mlp.expert_bias` and `mlp.shared_mlp`).
const hy_v3_module_moe = Names{
    .router = "mlp.gate.weight",
    .router_correction_bias = "mlp.e_score_correction_bias",
    .shared_expert = "mlp.shared_experts.",
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
        .aliases = &.{ "mistral3_text", "smollm", "cwm", "emu3_text_model", "emu3" },
        .llama_cpp = "llama",
        .chat = "llama3",
        .verified = true,
        .notes = "fixture: GQA, llama3 rope scaling, untied lm_head, byte-level BPE. Yi, SOLAR, TinyLlama, SmolLM 1/2 and Mistral 3 text configs are plain llama.",
    },
    .{
        .model_type = "mistral",
        .aliases = &.{"ministral"},
        .llama_cpp = "llama",
        .chat = "mistral",
        .verified = true,
        .notes = "fixture: a sliding window on every layer (shorter than the prompt, so the local mask bites) and an explicit head_dim that is not hidden_size / num_attention_heads. Otherwise the llama layout.",
    },
    .{
        .model_type = "qwen2",
        .aliases = &.{ "qwen2_5_vl", "qwen2_5_vl_text", "qwen2_vl", "qwen2_vl_text", "qwen2_5_omni", "qwen2_5_omni_thinker", "qwen2_5_omni_text" },
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
        .verified = true,
        .notes = "fixture: (1 + w) norms, pre/post feedforward norms, alternating local (sliding) and global layers, query_pre_attn_scalar, sqrt(H) embedding scale, tanh softcapping on the attention logits and on the output logits.",
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
        .model_type = "gemma3n",
        .aliases = &.{"gemma3n_text"},
        .llama_cpp = "gemma3n",
        .chat = "gemma",
        .verified = true,
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
            .ple_embed = "{p}embed_tokens_per_layer.weight",
            .ple_proj = "{p}per_layer_model_projection.weight",
            .ple_proj_norm = "{p}per_layer_projection_norm.weight",
            .ple_gate = "per_layer_input_gate.weight",
            .ple_out = "per_layer_projection.weight",
            .ple_norm = "post_per_layer_input_norm.weight",
            .altup_proj = "{p}altup_projections.{e}.weight",
            .altup_unembed = "{p}altup_unembed_projections.{e}.weight",
            .altup_router = "altup.modality_router.weight",
            .altup_router_norm = "altup.router_norm.weight",
            .altup_predict = "altup.prediction_coefs.weight",
            .altup_correct = "altup.correction_coefs.weight",
            .altup_scale = "altup.correct_output_scale",
            .laurel_l = "laurel.linear_left.weight",
            .laurel_r = "laurel.linear_right.weight",
            .laurel_norm = "laurel.post_laurel_norm.weight",
        },
        .notes = "fixture (text): AltUp residual streams (predict / correct, magnitude-matched embed and unembed projections), Laurel blocks, per-layer input embeddings, KV-shared layers, weightless value norm, unit attention scale, gaussian-top-k gate sparsity, sliding layers with a local rope base, final logit softcapping. The gemma3n image/audio wrapper runs its text config; the towers pass through exports untouched.",
        .extra = extraGemma3n,
    },
    .{
        .model_type = "gemma4",
        // `gemma4_unified` is the non-E-series wrapper (gemma-4-12B / -31B /
        // -26B-A4B); the same text config under another name.
        .aliases = &.{ "gemma4_text", "gemma4_unified", "gemma4_unified_text" },
        .llama_cpp = "gemma4",
        .chat = "gemma",
        .verified = true,
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
            .ple_embed = "{p}embed_tokens_per_layer.weight",
            .ple_proj = "{p}per_layer_model_projection.weight",
            .ple_proj_norm = "{p}per_layer_projection_norm.weight",
            .ple_gate = "per_layer_input_gate.weight",
            .ple_out = "per_layer_projection.weight",
            .ple_norm = "post_per_layer_input_norm.weight",
            .layer_scale = "layer_scalar",
        },
        .notes = "fixture (text): global layers with their own head size and KV heads (global_head_dim / per_layer_config), proportional rope on global layers and a local base on sliding ones, keys reused as values (attention_k_eq_v), KV-shared layers, weightless value norm, unit attention scale, per-layer input embeddings, layer_scalar, double-wide MLPs on shared layers. The gemma4 image/audio wrapper runs its text config; the towers pass through exports untouched. The MoE block of gemma-4-26B-A4B (enable_moe_block) is not implemented.",
        .extra = extraGemma4,
    },
    .{
        .model_type = "seed_oss",
        .chat = "seed",
        .llama_cpp = "seed_oss",
        .verified = true,
        .attention_bias = true,
        .notes = "fixture: llama layout with q/k/v biases and an unbiased o_proj (attention_out_bias), explicit head_dim. Seed-OSS 36B.",
    },
    .{
        .model_type = "lfm2",
        .default_norm_eps = 1e-05,
        .llama_cpp = "lfm2",
        .chat = "chatml",
        .verified = true,
        .qk_norm = .head,
        .tie_word_embeddings = true,
        .names = .{
            .final_norm = "{p}embedding_norm.weight",
            .input_norm = &.{"operator_norm.weight"},
            .pre_ff_norm = "ffn_norm.weight",
            .q_norm = "self_attn.q_layernorm.weight",
            .k_norm = "self_attn.k_layernorm.weight",
            .o = "self_attn.out_proj.weight",
            .gate = "feed_forward.w1.weight",
            .up = "feed_forward.w3.weight",
            .down = "feed_forward.w2.weight",
            .conv_in = "conv.in_proj.weight",
            .conv_kernel = "conv.conv.weight",
            .conv_out = "conv.out_proj.weight",
        },
        .notes = "fixture: gated short-convolution layers (in_proj B/C/x split, depthwise causal conv with conv_L_cache taps, out_proj) mixed with attention layers carrying per-head q/k norms, block_ff_dim sizing, embedding_norm as the final norm. LFM2 / LFM2.5 dense (lfm2_moe is unsupported).",
        .extra = extraLfm2,
    },
    .{
        .model_type = "mistral4",
        .aliases = &.{"mistral4_text"},
        .llama_cpp = "mistral4",
        .chat = "mistral",
        .verified = true,
        .names = .{
            .q_a = "self_attn.q_a_proj.weight",
            .q_a_norm = "self_attn.q_a_layernorm.weight",
            .q_b = "self_attn.q_b_proj.weight",
            .kv_a = "self_attn.kv_a_proj_with_mqa.weight",
            .kv_a_norm = "self_attn.kv_a_layernorm.weight",
            .kv_b = "self_attn.kv_b_proj.weight",
            .shared_expert = "mlp.shared_experts.",
        },
        .notes = "fixture (text): MLA with interleaved rotary, yarn (mscale_all_dim) from rope_parameters, llama_4_scaling_beta query scaling on every layer, softmax group-limited top-k (two best experts per group) with renormalisation, fused [E][2I][H] experts, shared experts, first_k_dense_replace. Mistral Small 4 text config.",
        .extra = extraMistral4,
    },
    .{
        .model_type = "qwen2_moe",
        .llama_cpp = "qwen2moe",
        .chat = "chatml",
        .attention_bias = true,
        .verified = true,
        .names = .{ .shared_expert = "mlp.shared_expert.", .shared_expert_gate = "mlp.shared_expert_gate.weight" },
        .notes = "fixture: softmax top-k routing over experts of moe_intermediate_size plus a shared expert of shared_expert_intermediate_size behind a sigmoid shared_expert_gate, both widths different from the dense intermediate_size a mlp_only_layers layer keeps; decoder_sparse_step picks the routed layers.",
    },
    .{
        .model_type = "qwen3_moe",
        .aliases = &.{ "qwen3_vl_moe", "qwen3_vl_moe_text", "qwen3_omni_moe", "qwen3_omni_moe_thinker", "qwen3_omni_moe_text" },
        .llama_cpp = "qwen3moe",
        .chat = "chatml",
        .verified = true,
        .qk_norm = .head,
        .names = .{ .q_norm = "self_attn.q_norm.weight", .k_norm = "self_attn.k_norm.weight" },
        .notes = "fixtures: softmax top-k with renormalisation, dense layers via mlp_only_layers, separate / fused / transposed-fused expert tensors.",
    },
    .{
        .model_type = "mixtral",
        .default_rope_theta = 1000000.0,
        .default_norm_eps = 1e-05,
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
        .verified = true,
        .notes = "fixture: softmax top-k routing with renormalisation over the separate per-expert tensors released Mixtral checkpoints store (block_sparse_moe.experts.{e}.w1 / w2 / w3).",
        .extra = extraMixtral,
    },
    .{
        .model_type = "phi3",
        .default_norm_eps = 1e-05,
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
        .default_norm_eps = 1e-05,
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
        .default_rope_theta = 500000.0,
        .aliases = &.{"cohere2"},
        .llama_cpp = "command-r",
        .chat = "cohere",
        .verified = true,
        .norm = .layer,
        // `CohereRotaryEmbedding` repeat-interleaves the frequencies and
        // `rotate_half` pairs adjacent coordinates: GPT-J pairs, not NeoX halves.
        .rope_style = .gptj,
        .parallel_residual = true,
        .tie_word_embeddings = true,
        .names = .{ .pre_ff_norm = null, .q_norm = "self_attn.q_norm.weight", .k_norm = "self_attn.k_norm.weight" },
        .notes = "fixture: LayerNorm without bias, parallel residual, interleaved rotary, logit_scale, tied embeddings, per-head q/k LayerNorm (use_qk_norm). cohere2 (Command R7B: sliding layers with RoPE, global layers without) is verified on the first four layers of the real checkpoint.",
        .extra = extraCohere,
    },
    .{
        .model_type = "glm4",
        .default_norm_eps = 1.5625e-07,
        .aliases = &.{ "glm", "glm4v", "glm4v_text" },
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
        .aliases = &.{ "deepseek_ocr2", "deepseek_ocr2_text", "youtu" },
        .llama_cpp = "deepseek2",
        .chat = "deepseek_v2",
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
        .default_rope_theta = 500000.0,
        .default_norm_eps = 1e-05,
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
        .default_rope_theta = 150000.0,
        .default_norm_eps = 1e-05,
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
        .default_norm_eps = 1e-05,
        .llama_cpp = "exaone4",
        .chat = "exaone4",
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
        .default_rope_theta = 2000000.0,
        .llama_cpp = "smollm3",
        .chat = "smollm3",
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
        .norm = .rms_gemma,
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
        .chat = "qwen3_5",
        .verified = true,
        .norm = .rms_gemma,
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
        .chat = "qwen3_5",
        .verified = true,
        .norm = .rms_gemma,
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
        .model_type = "qwen4_exp",
        .aliases = &.{"qwen4_exp_text"},
        .llama_cpp = null,
        .chat = "chatml",
        .verified = true,
        .norm = .rms_gemma,
        .qk_norm = .head,
        .names = .{
            .input_norm = &.{},
            .pre_ff_norm = null,
            .final_norm = null,
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
            .hc_attn = "attn_hyper_connection",
            .hc_ffn = "mlp_hyper_connection",
            .hc_attn_flat = null,
            .hc_ffn_flat = null,
        },
        .notes = "fixture: Qwen3.8-Flash-Next text config under the multimodal wrapper: gated hyper-connections (hc_count streams, (1 + w) group norms, low-rank sigmoid input mixer, sigmoid injection weights, no final norm), Gated DeltaNet layers with a sigmoid or silu output gate, sigmoid-gated full attention with (1 + w) head norms and partial rotary behind a QSA indexer (run as dense: exact while every complete block fits indexer_budget, longer prompts refused), per-layer n-gram embeddings (PLE: splitmix64 hash multipliers, prime bucket ranges, sharded tables read row by row, gated values, dilated depthwise convolution), fused softmax MoE with a gated shared expert on every layer. Vision and indexer tensors pass through exports untouched.",
        .extra = extraQwen4Exp,
    },
    .{
        .model_type = "glm4_moe",
        .default_norm_eps = 1e-05,
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
        .default_norm_eps = 1e-05,
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
        .model_type = "glm4_moe_lite",
        .default_norm_eps = 1e-05,
        .aliases = &.{"glm_moe_lite"},
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
        .notes = "fixture: GLM-4.7-Flash: DeepSeek V3 MLA with interleaved partial rotary, sigmoid MoE with correction bias, top-2 group scores over n_group groups, floored renormalisation and routed_scaling_factor, shared experts, stacked expert tensors, mlp_layer_types with a dense first layer.",
        .extra = extraGlm4MoeLite,
    },
    .{
        .model_type = "mamba2",
        .default_norm_eps = 1e-05,
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
        .default_norm_eps = 1e-05,
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
        .default_norm_eps = 1e-05,
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
        .chat = "jamba",
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
        .chat = "minimax_m2",
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
        .default_norm_eps = 1e-05,
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
        .notes = "fixture: lightning attention layers (`hidden_act` on the fused qkv, silu on the releases; per-head decay recurrence, RMSNorm, sigmoid output gate) alternating with softmax attention with partial rotary, the renormalised residual layout with α/β scales, softmax top-k MoE. MiniMax-Text-01 / M1 (`layer_types` or `attn_type_list`). The recurrence runs sequentially.",
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
            // The released MiniMax-M3 checkpoints keep the dense and shared
            // MLPs split (`gate_proj` / `up_proj`); the fused `gate_up_proj`
            // spelling is used when the checkpoint has it.
            .gate = "mlp.gate_proj.weight",
            .up = "mlp.up_proj.weight",
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
            .shared_gate = "gate_proj.weight",
            .shared_up = "up_proj.weight",
            .shared_gate_up = "gate_up_proj.weight",
            .shared_down = "down_proj.weight",
        },
        .notes = "fixture: (1 + w) norms, per-head (1 + w) q/k norm, partial rotary, dense layers (mlp_layer_types) with a fused gate_up and the clamped swiglu, sigmoid MoE with correction bias, routed scaling and a fused-gate_up shared expert, minimax_m3_sparse layers run as dense attention (exact while the context fits index_topk_blocks blocks); the indexer weights pass through exports untouched. The image tower of the VL wrapper is never executed.",
        .extra = extraMiniMaxM3,
    },
    .{
        .model_type = "ernie4_5_moe",
        .default_norm_eps = 1e-05,
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
        .default_norm_eps = 1e-05,
        .chat = "hunyuan_moe",
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
        .default_norm_eps = 1e-05,
        .llama_cpp = "kimi-linear",
        .verified = true,
        .linear = .kda,
        .names = kimi_linear_names,
        .notes = "fixtures: Kimi Delta Attention layers (per-channel decay from the low-rank forget gate, q/k/v short convolution, sigmoid-gated output norm) in the original checkpoint layout (linear_attn_config, split q/k/v convolutions, block_sparse_moe with w1/w3/w2 experts) and in the Hugging Face module layout (layer_types, fused conv1d, stacked experts); MLA full-attention layers without RoPE; sigmoid MoE with correction bias, top-2 group scores, routed_scaling_factor and shared experts. Kimi-Linear-48B-A3B.",
        .extra = extraKimiLinear,
    },
    .{
        .model_type = "glm5_next",
        .default_norm_eps = 1e-05,
        .aliases = &.{"glm5_next_text"},
        .llama_cpp = null,
        .chat = "glm4",
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
            // The released GLM-5.3-Flash checkpoints put the forget gate's
            // tensors directly under `self_attn` and keep one convolution per
            // projection, where transformers has a `forget_gate` submodule and
            // one fused `conv1d`; both spellings are accepted.
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
        },
        .notes = "fixture: GLM-5.3-Flash text config under the multimodal wrapper: manifold-constrained hyper-connections collapsed by an unweighted mean, Kimi Delta Attention layers with the safe lower-bound forget gate, NoPE MLA layers behind a k-pool DSA indexer with full and shared indexer types (run as dense: exact while every complete pool fits index_topk, longer prompts refused), sigmoid MoE with correction bias, top-2 group scores, floored renormalisation, routed_scaling_factor and shared experts, clamped SwiGLU in experts, shared experts and dense layers, mlp_layer_types. Vision and indexer tensors pass through exports untouched.",
        .extra = extraGlm5Next,
    },
    .{
        .model_type = "kimi_k3",
        .aliases = &.{"kimi_k3_text"},
        .llama_cpp = null,
        .chat = "kimi_k3",
        .verified = true,
        .linear = .kda,
        .names = kimi_linear_names,
        .notes = "fixtures: the Kimi K3 image-video wrapper (KimiK3ForConditionalGeneration) around a kimi_linear text config with Attention Residual (attn_res_block_size: softmax-weighted retrieval over the banked block prefixes and the running one, at the attention input, the MLP input and the output norm), KDA layers with the full-rank output gate and the safe forget gate (gate_lower_bound), MLA layers with the sigmoid output gate, latent MoE (routed_expert_down_proj / routed_expert_up_proj with routed_expert_norm), SiTU activation, two shared experts, sigmoid routing with correction bias, language_model prefix; bf16 and compressed-tensors mxfp4-pack-quantized experts. The vision tower and projector pass through exports untouched.",
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
    .{
        .model_type = "mimo_v2_flash",
        .default_norm_eps = 1e-05,
        .aliases = &.{"mimo_v2"},
        .llama_cpp = "mimo2",
        .chat = "mimo",
        .verified = true,
        .names = .{
            .sinks = "self_attn.sinks",
            .sinks_alt = "self_attn.attention_sink_bias",
            .qkv = "self_attn.qkv_proj.weight",
            .router_correction_bias = "mlp.gate.e_score_correction_bias",
        },
        .notes = "fixtures: hybrid full / sliding-window attention (window 128 in the released configs) with attention sinks and doubled kv heads on the sliding layers, v_head_dim < head_dim with attention_value_scale, partial rotary with one base per layer type (rope_parameters, or rope_theta / swa_rope_theta), a dense first layer (mlp_layer_types / moe_layer_freq) then sigmoid MoE with correction bias and group-limited top-k, no shared experts; both the transformers spelling (layer_types, stacked experts, sinks) and the hub checkpoint spelling of MiMo-V2-Flash / V2.5 / V2.6 (model_type mimo_v2: hybrid_layer_pattern, swa_*, attention_sink_bias, per-expert tensors, the Pro layout's fused qkv_proj chunked per kv head). MTP (model.mtp.*), vision and audio encoder tensors of the V2.5 / V2.6 omni checkpoints pass through exports untouched. The V2.6 checkpoints' MXFP4 experts (quant_method fp8 with store_dtype mxfp4: U8 weight/weight_scale next to the fp8 dense weights) and their bf16 MoE router (moe_router_dtype) are read as they are.",
        .extra = extraMiMoV2,
    },
    .{
        .model_type = "deepseek_v4",
        .llama_cpp = null,
        .chat = "deepseek",
        .verified = true,
        .rope_style = .gptj,
        .names = dsv4_names,
        .notes = "fixture: hyper-connections (hc_mult streams, Sinkhorn-mixed), low-rank q with unweighted head norm, shared-KV sliding attention with sinks and inverse-roped output, grouped output projection, CSA (overlapping pooled windows; Lightning Indexer as dense: exact while every reachable entry fits index_topk) and HCA branches with their own rope, sqrtsoftplus MoE with correction bias, hash-routed (tid2eid) layers, clamped SwiGLU, shared expert, MTP tensors passed through. The released checkpoints (DeepSeek's own tensor names, FP8 with ue8m0 block scales, FP4 e2m1 experts) are renamed and dequantised on load; DeepSeek-V4-Flash's first four layers (sliding, CSA and HCA attention, hash and learned routing) match transformers in float32 on the real weights.",
        .extra = extraDeepseekV4,
    },
    .{
        .model_type = "deepseek_v41",
        .aliases = &.{"deepseek_v41_text"},
        .llama_cpp = null,
        .chat = "deepseek",
        .verified = true,
        .rope_style = .gptj,
        .names = dsv4_names,
        .notes = "fixture: single-pass hyper-connections, CSA2 shared compressed KV (kv_source groups, ratio 1 and pooled branches, indexer as dense), FP8/FP4 fake quantisation of the window KV and latents, engram n-gram hash layers (lazy table rows, tokenizer-derived compressed ids), gate_temp routing, nested text_config with vision tensors passed through. The released checkpoints (DeepSeek's own tensor names, FP8 with ue8m0 block scales, FP4 e2m1 experts) are renamed and dequantised on load.",
        .extra = extraDeepseekV41,
    },
    // ---- llama-layout dense families -------------------------------------
    .{
        .model_type = "arcee",
        .default_norm_eps = 1e-05,
        .llama_cpp = "arcee",
        .chat = "llama3",
        .verified = true,
        .mlp = .dense,
        .activation = .relu2,
        .names = .{ .gate = null },
        .notes = "fixture: llama attention with the two-projection relu² MLP (no gate), mlp_bias. AFM / Arcee.",
        .extra = extraDenseMlp,
    },
    .{
        .model_type = "apertus",
        .default_rope_theta = 12000000.0,
        .default_norm_eps = 1e-05,
        .llama_cpp = null,
        .chat = "apertus",
        .verified = true,
        .qk_norm = .head,
        .mlp = .dense,
        .names = .{
            .input_norm = &.{"attention_layernorm.weight"},
            .pre_ff_norm = "feedforward_layernorm.weight",
            .q_norm = "self_attn.q_norm.weight",
            .k_norm = "self_attn.k_norm.weight",
            .gate = null,
            .xielu_alpha_p = "mlp.act_fn.alpha_p",
            .xielu_alpha_n = "mlp.act_fn.alpha_n",
        },
        .notes = "fixture: attention/feedforward norm names, per-head q/k RMSNorm, the two-projection xIELU MLP (learned alpha_p / alpha_n). Apertus (Swiss AI).",
    },
    .{
        .model_type = "bitnet",
        .default_rope_theta = 500000.0,
        .default_norm_eps = 1e-05,
        .llama_cpp = "bitnet-25",
        .chat = "chatml",
        .verified = true,
        .activation = .relu2,
        .names = .{
            .attn_sub_norm = "self_attn.attn_sub_norm.weight",
            .ffn_sub_norm = "mlp.ffn_sub_norm.weight",
        },
        .notes = "fixture: the sub-layer RMSNorms on the attention output and the gated MLP intermediate, relu² activation. No *released* BitNet checkpoint can be run: b1.58 ternarises its weights and quantises its activations inside every linear at run time (quantization_config.quant_method = bitnet), so neither the packed release nor the bf16 master weights are the model the reference runs; both are refused with that reason. The entry covers the layout for a checkpoint that ships plain weights.",
    },
    .{
        .model_type = "helium",
        .default_rope_theta = 100000.0,
        .default_norm_eps = 1e-08,
        .llama_cpp = null,
        .chat = "chatml",
        .verified = true,
        .rope_style = .gptj,
        .notes = "fixture: llama layout with mlp/attention biases and interleaved rotary pairing. Helium's reference builds its table as `cat(f, f)` and then takes `[:d/2].repeat_interleave(2)` of it, which is the plain `(x[2i], x[2i+1])` pairing against frequency `i` — the same thing GPT-J does, written differently. Helium 1 (Kyutai).",
    },
    .{
        .model_type = "hunyuan_v1_dense",
        .default_norm_eps = 1e-05,
        .chat = "hunyuan",
        .aliases = &.{ "hunyuan_vl_text", "hunyuan_vl" },
        .llama_cpp = "hunyuan-dense",
        .verified = true,
        .qk_norm = .head,
        .names = .{
            .q_norm = "self_attn.query_layernorm.weight",
            .k_norm = "self_attn.key_layernorm.weight",
        },
        .notes = "fixture: per-head q/k RMSNorm after RoPE, NTK-alpha dynamic rope base. Hunyuan dense (and the HunYuan-VL text config).",
        .extra = extraHunyuanDense,
    },
    .{
        .model_type = "jais2",
        .llama_cpp = null,
        .chat = "chatml",
        .verified = true,
        .norm = .layer,
        .mlp = .dense,
        .activation = .relu2,
        .attention_bias = true,
        .names = .{ .gate = null },
        .notes = "fixture: LayerNorm with biases, biased projections, the two-projection relu² MLP. Jais 2.",
        .extra = extraDenseMlp,
    },
    .{
        .model_type = "nanochat",
        .default_norm_eps = 1e-06,
        .llama_cpp = null,
        .chat = "nanochat",
        .verified = true,
        .norm = .rms_none,
        .mlp = .dense,
        .activation = .relu2,
        .names = .{
            .embed_norm = "norm.weight",
            .gate = null,
            .up = "mlp.fc1.weight",
            .down = "mlp.fc2.weight",
        },
        .notes = "fixture: non-parametric RMSNorm everywhere (including an extra norm on the embeddings), weightless per-head q/k norm after RoPE, fc1/fc2 relu² MLP, final logit softcapping. nanochat.",
        .extra = extraNanoChat,
    },
    .{
        .model_type = "persimmon",
        .llama_cpp = "persimmon",
        .verified = true,
        .norm = .layer,
        .qkv = .heads_interleaved,
        .mlp = .dense,
        .activation = .relu2,
        .attention_bias = true,
        .names = .{
            .final_norm = "{p}final_layernorm.weight",
            .q = null,
            .k = null,
            .v = null,
            .qkv = "self_attn.query_key_value.weight",
            .o = "self_attn.dense.weight",
            .gate = null,
            .up = "mlp.dense_h_to_4h.weight",
            .down = "mlp.dense_4h_to_h.weight",
            .q_norm = "self_attn.q_layernorm.weight",
            .k_norm = "self_attn.k_layernorm.weight",
        },
        .notes = "fixture: head-interleaved query_key_value with bias, per-head q/k LayerNorm, partial rotary, dense_h_to_4h/dense_4h_to_h relu² MLP. Persimmon 8B (and Fuyu's text tower).",
        .extra = extraPersimmon,
    },
    .{
        .model_type = "gptj",
        .llama_cpp = "gptj",
        .verified = true,
        .norm = .layer,
        .parallel_residual = true,
        .rope_style = .gptj,
        .mlp = .dense,
        .activation = .gelu_tanh,
        .names = .{
            .prefixes = &.{ "transformer.", "" },
            .embed = "{p}wte.weight",
            .final_norm = "{p}ln_f.weight",
            .layer = "{p}h.{i}.",
            .input_norm = &.{"ln_1.weight"},
            .pre_ff_norm = null,
            .q = "attn.q_proj.weight",
            .k = "attn.k_proj.weight",
            .v = "attn.v_proj.weight",
            .o = "attn.out_proj.weight",
            .gate = null,
            .up = "mlp.fc_in.weight",
            .down = "mlp.fc_out.weight",
        },
        .notes = "fixture: parallel residual with one LayerNorm, interleaved rotary over an absolute rotary_dim, fc_in/fc_out MLP with biases, lm_head bias. GPT-J 6B.",
        .extra = extraGptJ,
    },
    .{
        .model_type = "codegen",
        .llama_cpp = null,
        .verified = true,
        .norm = .layer,
        .parallel_residual = true,
        .rope_style = .gptj,
        .qkv = .mp_blocks,
        .mlp = .dense,
        .activation = .gelu_tanh,
        .names = .{
            .prefixes = &.{ "transformer.", "" },
            .embed = "{p}wte.weight",
            .final_norm = "{p}ln_f.weight",
            .layer = "{p}h.{i}.",
            .input_norm = &.{"ln_1.weight"},
            .pre_ff_norm = null,
            .q = null,
            .k = null,
            .v = null,
            .qkv = "attn.qkv_proj.weight",
            .o = "attn.out_proj.weight",
            .gate = null,
            .up = "mlp.fc_in.weight",
            .down = "mlp.fc_out.weight",
        },
        .notes = "fixture: the GPT-J layout with a fused qkv_proj laid out as four tensor-parallel [q | v | k] blocks. CodeGen / CodeGen 2.",
        .extra = extraCodeGen,
    },
    .{
        .model_type = "gpt_neo",
        .llama_cpp = "gptneo",
        .verified = true,
        .norm = .layer,
        .positional = .learned,
        .mlp = .dense,
        .activation = .gelu_tanh,
        .tie_word_embeddings = true,
        .names = .{
            .prefixes = &.{ "transformer.", "" },
            .embed = "{p}wte.weight",
            .pos_embed = "{p}wpe.weight",
            .final_norm = "{p}ln_f.weight",
            .layer = "{p}h.{i}.",
            .input_norm = &.{"ln_1.weight"},
            .pre_ff_norm = "ln_2.weight",
            .q = "attn.attention.q_proj.weight",
            .k = "attn.attention.k_proj.weight",
            .v = "attn.attention.v_proj.weight",
            .o = "attn.attention.out_proj.weight",
            .gate = null,
            .up = "mlp.c_fc.weight",
            .down = "mlp.c_proj.weight",
        },
        .notes = "fixture: learned positions, unscaled attention logits, alternating global and window_size local layers (attention_types), c_fc/c_proj MLP. GPT-Neo 1.3B/2.7B.",
        .extra = extraGptNeo,
    },
    .{
        .model_type = "xglm",
        .llama_cpp = null,
        .verified = true,
        .norm = .layer,
        .positional = .sinusoidal,
        .mlp = .dense,
        .activation = .gelu,
        .attention_bias = true,
        .tie_word_embeddings = true,
        .names = .{
            .final_norm = "{p}layer_norm.weight",
            .input_norm = &.{"self_attn_layer_norm.weight"},
            .pre_ff_norm = "final_layer_norm.weight",
            .o = "self_attn.out_proj.weight",
            .gate = null,
            .up = "fc1.weight",
            .down = "fc2.weight",
        },
        .notes = "fixture: fairseq sinusoidal positions with offset 2, sqrt(hidden) embedding scale, LayerNorm biases, fc1/fc2 MLP. XGLM.",
        .extra = extraXglm,
    },
    .{
        .model_type = "biogpt",
        .default_norm_eps = 1e-12,
        .llama_cpp = null,
        .verified = true,
        .norm = .layer,
        .positional = .learned,
        .mlp = .dense,
        .activation = .gelu,
        .attention_bias = true,
        .names = .{
            .prefixes = &.{ "biogpt.", "model.", "" },
            .pos_embed = "{p}embed_positions.weight",
            .final_norm = "{p}layer_norm.weight",
            .lm_head = &.{ "output_projection.weight", "lm_head.weight" },
            .input_norm = &.{"self_attn_layer_norm.weight"},
            .pre_ff_norm = "final_layer_norm.weight",
            .o = "self_attn.out_proj.weight",
            .gate = null,
            .up = "fc1.weight",
            .down = "fc2.weight",
        },
        .notes = "fixture: learned positions with offset 2, sqrt(hidden) embedding scale, output_projection head. BioGPT.",
        .extra = extraBioGpt,
    },
    .{
        .model_type = "ernie4_5",
        .default_rope_theta = 500000.0,
        .default_norm_eps = 1e-05,
        .chat = "ernie",
        .aliases = &.{"paddleocr_vl_text"},
        .llama_cpp = "ernie4_5",
        .verified = true,
        .rope_style = .gptj,
        .notes = "fixture: interleaved rotary, use_bias projections. ERNIE 4.5 dense (and the PaddleOCR-VL text config).",
        .extra = extraErnie,
    },
    .{
        .model_type = "ministral3",
        .default_rope_theta = 1000000.0,
        .default_norm_eps = 1e-05,
        .llama_cpp = "llama",
        .chat = "mistral_v7",
        .verified = true,
        .notes = "fixture: llama layout with Ministral 3's query scaling (1 + beta*log(1 + floor(pos / max_position_embeddings)) on every layer) and an optional sliding window.",
        .extra = extraMinistral3,
    },
    .{
        .model_type = "granite_swa",
        .default_norm_eps = 1e-05,
        .llama_cpp = "granite",
        .chat = "granite",
        .verified = true,
        .names = .{ .sinks = "self_attn.sinks" },
        .notes = "fixture: the Granite multipliers plus per-head attention sinks (an extra softmax logit) and sliding layers with their own rope base (layer_rope_theta). Granite 4 SWA dense.",
        .extra = extraGraniteSwa,
    },
    // ---- mixture-of-experts families --------------------------------------
    .{
        .model_type = "olmoe",
        .default_norm_eps = 1e-05,
        .llama_cpp = "olmoe",
        .chat = "olmo",
        .verified = true,
        .qk_norm = .full,
        .names = .{ .q_norm = "self_attn.q_norm.weight", .k_norm = "self_attn.k_norm.weight" },
        .notes = "fixture: q/k RMSNorm over the full projection, clip_qkv, softmax top-k routing (optionally renormalised) over separate or fused experts. OLMoE.",
        .extra = extraOlmoe,
    },
    .{
        .model_type = "flex_olmo",
        .default_rope_theta = 500000.0,
        .llama_cpp = "olmoe",
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
        .notes = "fixture: the OLMo 2 post-norm layout with OLMoE's softmax top-k routing. FlexOlmo.",
        .extra = extraOlmoe,
    },
    .{
        .model_type = "dots1",
        .llama_cpp = "dots1",
        .chat = "dots",
        .verified = true,
        .names = .{
            .q_norm = "self_attn.q_norm.weight",
            .k_norm = "self_attn.k_norm.weight",
            .router_correction_bias = "mlp.gate.e_score_correction_bias",
            .shared_expert = "mlp.shared_experts.",
        },
        .notes = "fixture: per-head q/k norm, DeepSeek-V3 routing (sigmoid scores, correction bias, group-limited top-k, renormalisation, routed scaling), shared experts, first_k_dense_replace. dots.llm1.",
        .extra = extraDots1,
    },
    .{
        .model_type = "exaone_moe",
        .default_norm_eps = 1e-05,
        .llama_cpp = "exaone4",
        .chat = "k_exaone",
        .verified = true,
        .names = .{
            .q_norm = "self_attn.q_norm.weight",
            .k_norm = "self_attn.k_norm.weight",
            .router_correction_bias = "mlp.gate.e_score_correction_bias",
            .shared_expert = "mlp.shared_experts.",
        },
        .notes = "fixture: per-head q/k norm, sliding_window_pattern local layers, DeepSeek-V3 routing with shared experts and a dense/sparse mlp_layer_types schedule. EXAONE 4 MoE.",
        .extra = extraExaoneMoe,
    },
    .{
        .model_type = "solar_open",
        .default_rope_theta = 1000000.0,
        .default_norm_eps = 1e-05,
        .llama_cpp = null,
        .chat = "solar_open",
        .verified = true,
        .names = .{
            .router_correction_bias = "mlp.gate.e_score_correction_bias",
            .shared_expert = "mlp.shared_experts.",
        },
        .notes = "fixture: partial rotary, DeepSeek-V3 routing with shared experts on every layer. Solar Open (Upstage).",
        .extra = extraSolarOpen,
    },
    .{
        .model_type = "afmoe",
        .default_norm_eps = 1e-05,
        .llama_cpp = null,
        .chat = "chatml",
        .verified = true,
        .names = .{
            .post_attn_norm = "post_attention_layernorm.weight",
            .pre_ff_norm = "pre_mlp_layernorm.weight",
            .post_ff_norm = "post_mlp_layernorm.weight",
            .q_norm = "self_attn.q_norm.weight",
            .k_norm = "self_attn.k_norm.weight",
            .attn_gate = "self_attn.gate_proj.weight",
            .router = "mlp.router.gate.weight",
            .router_correction_bias = "mlp.expert_bias",
            .shared_expert = "mlp.shared_experts.",
        },
        .notes = "fixture: norms on both sublayer inputs and outputs, a sigmoid gate on the attention output, sliding layers every n, sigmoid routing with a selection bias, renormalisation and route_scale, shared experts and dense first layers. AFM (Arcee) MoE.",
        .extra = extraAfmoe,
    },
    .{
        .model_type = "mellum",
        .default_rope_theta = 500000.0,
        .llama_cpp = null,
        .chat = "chatml",
        .verified = true,
        .names = .{ .q_norm = "self_attn.q_norm.weight", .k_norm = "self_attn.k_norm.weight" },
        .notes = "fixture: per-head q/k norm, per-layer-type rope parameters, softmax top-k routing renormalised over fused experts. Mellum (JetBrains).",
        .extra = extraMellum,
    },
    .{
        .model_type = "laguna",
        .default_rope_theta = 500000.0,
        .llama_cpp = null,
        .chat = "laguna",
        .verified = true,
        .names = .{
            .q_norm = "self_attn.q_norm.weight",
            .k_norm = "self_attn.k_norm.weight",
            .attn_gate = "self_attn.g_proj.weight",
            // The released spelling; transformers renames both on load.
            .router_correction_bias = "mlp.experts.e_score_correction_bias",
            .shared_expert = "mlp.shared_expert.",
        },
        .notes = "fixture: a softplus gate on the attention output (per head or per coordinate), sigmoid routing with a tanh softcap on the router logits and a correction bias, renormalised weights scaled by moe_routed_scaling_factor, shared experts, and per-layer query head counts (`num_attention_heads_per_layer`).",
        .extra = extraLaguna,
    },
    .{
        .model_type = "hy_v3",
        .default_rope_theta = 11158840.0,
        .default_norm_eps = 1e-05,
        .llama_cpp = "hunyuan-moe",
        .verified = true,
        .names = .{
            .q_norm = "self_attn.q_norm.weight",
            .k_norm = "self_attn.k_norm.weight",
            .router = "mlp.router.gate.weight",
            .router_correction_bias = "mlp.expert_bias",
            .shared_expert = "mlp.shared_mlp.",
            .moe_alt = &hy_v3_module_moe,
        },
        .notes = "fixture: per-head q/k norm, sigmoid routing with a correction bias, renormalisation and a router scaling factor, a shared MLP and a dense/sparse mlp_layer_types schedule. Hunyuan V3 (released checkpoints and the transformers module layout).",
        .extra = extraHyV3,
    },
    .{
        .model_type = "deepseek_v32",
        .llama_cpp = "deepseek2",
        .chat = "deepseek",
        .names = deepseek_v3_names,
        .verified = true,
        .notes = "fixture: DeepSeek V3.2-Exp, the V3 layout (MLA, sigmoid group-limited routing, shared experts) whose `indexed_attention` layers select the top `index_topk` keys with a lightning indexer; ditch runs them as dense attention, which is exactly the reference for prompts up to index_topk tokens and refused beyond it. The indexer's own tensors are never read and pass through exports untouched.",
        .extra = extraDeepseekV32,
    },
};

/// DeepSeek V4 / V4.1 tensor names shared with the generic loader; the
/// family-specific tensors (hyper-connections, grouped output projection,
/// compressors, engram) are named in deepseek_v4.zig.
const dsv4_names = Names{
    .q_a = "self_attn.q_a_proj.weight",
    .q_a_norm = "self_attn.q_a_norm.weight",
    .q_b = "self_attn.q_b_proj.weight",
    .kv_a = "self_attn.kv_proj.weight",
    .kv_a_norm = "self_attn.kv_norm.weight",
    .o = "self_attn.o_b_proj.weight",
    .sinks = "self_attn.sinks",
    .router_correction_bias = "mlp.gate.e_score_correction_bias",
    .shared_expert = "mlp.shared_experts.",
};

test "registry lookup and aliases" {
    try std.testing.expect(lookup("llama") != null);
    try std.testing.expectEqualStrings("gemma3", lookup("gemma3_text").?.model_type);
    try std.testing.expectEqualStrings("qwen2", lookup("qwen2_5_vl").?.model_type);
    try std.testing.expectEqualStrings("gemma4", lookup("gemma4_text").?.model_type);
    try std.testing.expectEqualStrings("gemma3n", lookup("gemma3n_text").?.model_type);
    try std.testing.expectEqualStrings("mistral4", lookup("mistral4_text").?.model_type);
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
    // The latent norms keep the RMSNorm default whatever rms_norm_eps says.
    try std.testing.expectEqual(@as(f32, 1e-6), ds.mla.?.latent_norm_eps);
    const eps5 = try parseConfig(a,
        \\{"model_type":"deepseek_v3","hidden_size":64,"num_attention_heads":4,"num_hidden_layers":2,"vocab_size":100,"q_lora_rank":32,"kv_lora_rank":16,"qk_nope_head_dim":8,"qk_rope_head_dim":4,"v_head_dim":8,"rms_norm_eps":1e-5}
    );
    try std.testing.expectEqual(@as(f32, 1e-5), eps5.rms_norm_eps);
    try std.testing.expectEqual(@as(f32, 1e-6), eps5.mla.?.latent_norm_eps);
    try std.testing.expectEqual(@as(usize, 8), ds.v_head_dim);
    try std.testing.expectEqual(@as(usize, 4), ds.rotary_dim);
    try std.testing.expect(!ds.moe_layers[0] and ds.moe_layers[1] and ds.moe_layers[2]);
    try std.testing.expectEqual(RouterScoring.sigmoid, ds.moe.scoring);
    try std.testing.expectEqual(TopkMethod.group_limited, ds.moe.topk_method);
    try std.testing.expect(ds.rope_scaling == .yarn);
    const ms = 0.1 * @log(@as(f32, 40)) + 1.0;
    try std.testing.expectApproxEqRel(ms * ms / @sqrt(@as(f32, 12)), ds.attention_scale, 1e-5);
    // DeepSeek V4 with the legacy keys: per-layer compress ratios, scalar rates,
    // num_hash_layers, top-level yarn (compress branch only, no mscale).
    const v4 = try parseConfig(a,
        \\{"model_type":"deepseek_v4","hidden_size":64,"num_attention_heads":4,"num_hidden_layers":4,"head_dim":16,"vocab_size":100,"q_lora_rank":16,"qk_rope_head_dim":4,"n_routed_experts":8,"num_experts_per_tok":2,"moe_intermediate_size":16,"compress_ratios":[0,4,128,4],"compress_rate_csa":2,"num_hash_layers":1,"rope_scaling":{"rope_type":"yarn","factor":16,"original_max_position_embeddings":65536}}
    );
    const d4 = v4.dsv4.?;
    try std.testing.expect(!d4.v41 and v4.hc_mult == 4 and v4.sinks and v4.rope_style == .gptj);
    try std.testing.expectEqualSlices(usize, &.{ 0, 2, 128, 2 }, d4.compress_ratio);
    try std.testing.expectEqual(CompressBranch.hca, d4.branch[2]);
    try std.testing.expect(d4.hash_moe_layers[0] and !d4.hash_moe_layers[1]);
    try std.testing.expectEqual(@as(usize, 4), v4.rotary_dim);
    try std.testing.expect(v4.rope_scaling == .none and d4.compress_rope_scaling == .yarn);
    try std.testing.expectEqual(@as(f32, 1.0), d4.compress_rope_scaling.yarn.attention_factor);
    try std.testing.expectEqual(RouterScoring.sqrtsoftplus, v4.moe.scoring);
    try std.testing.expect(v4.moe.swiglu_limit.? == 10.0 and v4.moe_layers[0] and v4.sliding_layers[3]);
    // DeepSeek V4.1 text config: kv-source groups, candidate source, engram, defaults.
    const v41 = try parseConfig(a,
        \\{"model_type":"deepseek_v41","text_config":{"model_type":"deepseek_v41_text","hidden_size":64,"num_attention_heads":4,"num_hidden_layers":6,"head_dim":16,"vocab_size":100,"q_lora_rank":16,"qk_rope_head_dim":4,"n_routed_experts":8,"num_experts_per_tok":2,"moe_intermediate_size":16,"compress_ratios":[0,2,2,1,1,1,0,0,0],"kv_source_layer_ids":[1,3],"index_source_layer_ids":[1,3],"engram_layer_ids":[1],"engram_num_embeddings":[1000],"gate_temp":2.0}}
    );
    const d41 = v41.dsv4.?;
    try std.testing.expect(d41.v41 and d41.fake_quant and v41.rms_norm_eps == 1e-20);
    try std.testing.expectEqualSlices(usize, &.{ 0, 2, 2, 1, 1, 1 }, d41.compress_ratio);
    try std.testing.expectEqual(@as(?usize, 1), d41.kv_source[2]);
    try std.testing.expectEqual(@as(?usize, 3), d41.kv_source[5]);
    try std.testing.expectEqual(@as(?usize, null), d41.kv_source[0]);
    try std.testing.expectEqual(@as(?usize, 3), d41.candidate_source);
    try std.testing.expectEqual(@as(f32, 2.0), v41.moe.gate_temp);
    try std.testing.expectEqual(@as(usize, 4), d41.engram.?.max_ngram);
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
    // Kimi K3: the wrapper entry wins over the nested kimi_linear text config
    // and carries AttnRes, latent MoE, SiTU and the K3 attention gates.
    const k3 = try parseConfig(a,
        \\{"model_type":"kimi_k3","text_config":{"model_type":"kimi_linear","hidden_size":64,"num_attention_heads":4,"num_hidden_layers":5,"vocab_size":100,"q_lora_rank":32,"kv_lora_rank":16,"qk_nope_head_dim":8,"qk_rope_head_dim":4,"v_head_dim":8,"num_experts":8,"num_experts_per_token":2,"num_shared_experts":2,"moe_intermediate_size":16,"routed_expert_hidden_size":24,"latent_moe_use_norm":true,"attn_res_block_size":2,"hidden_act":"situ","activation_situ_beta":4.0,"activation_situ_linear_beta":25.0,"mla_use_nope":true,"mla_use_output_gate":true,"first_k_dense_replace":1,"linear_attn_config":{"kda_layers":[1,2,4],"full_attn_layers":[3,5],"head_dim":16,"num_heads":2,"short_conv_kernel_size":4,"use_full_rank_gate":true,"gate_lower_bound":-5.0}}}
    );
    try std.testing.expectEqualStrings("kimi_k3", k3.arch.model_type);
    try std.testing.expectEqualStrings("kimi_k3", k3.model_type);
    try std.testing.expectEqual(@as(usize, 2), k3.attn_res_block);
    try std.testing.expectEqual(@as(usize, 24), k3.moe_latent);
    try std.testing.expect(k3.moe_latent_norm and k3.mla_output_gate and k3.linear_full_rank_gate);
    try std.testing.expectEqual(@as(f32, -5.0), k3.linear_gate_lower_bound.?);
    try std.testing.expectEqual(tensor.Activation.silu, k3.activation);
    try std.testing.expectEqual(@as(f32, 4.0), k3.moe.situ.?.beta);
    try std.testing.expectEqual(@as(f32, 25.0), k3.moe.situ.?.linear_beta.?);
    try std.testing.expect(k3.linear_layers[0] and !k3.linear_layers[2] and k3.linear_layers[3] and !k3.linear_layers[4]);
    try std.testing.expect(!k3.moe_layers[0] and k3.moe_layers[4]);
}

test "parseConfig handles both MiMo V2 spellings" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    // transformers spelling with the defaults of MiMoV2FlashConfig.
    const hf = try parseConfig(a,
        \\{"model_type":"mimo_v2_flash","hidden_size":64,"num_attention_heads":8,"num_key_value_heads":2,"num_hidden_layers":12,"vocab_size":100,"head_dim":24,"v_head_dim":16,"n_routed_experts":8,"num_experts_per_tok":2,"moe_intermediate_size":16,"sliding_window":128}
    );
    try std.testing.expectEqualStrings("mimo_v2_flash", hf.arch.model_type);
    try std.testing.expect(!hf.sliding_layers[0] and hf.sliding_layers[1] and !hf.sliding_layers[5] and hf.sliding_layers[6] and !hf.sliding_layers[11]);
    try std.testing.expectEqual(@as(usize, 2), hf.layer_kv_heads[0]);
    try std.testing.expectEqual(@as(usize, 4), hf.layer_kv_heads[1]);
    try std.testing.expectEqual(@as(usize, 4 * 24), hf.kvDim());
    try std.testing.expectEqual(@as(usize, 16), hf.layerVDim(0));
    try std.testing.expectEqual(@as(usize, 8), hf.rotary_dim); // int(24 * 0.334)
    try std.testing.expectEqual(@as(f32, 5_000_000.0), hf.rope_theta);
    try std.testing.expectEqual(@as(f32, 10_000.0), hf.rope_local.?.theta);
    try std.testing.expectEqual(@as(usize, 8), hf.rope_local.?.rotary_dim);
    try std.testing.expect(hf.sinks and hf.sinks_sliding_only and hf.narrow_values);
    try std.testing.expectEqual(@as(f32, 0.707), hf.mult.value);
    try std.testing.expectEqual(@as(f32, 1e-5), hf.rms_norm_eps);
    try std.testing.expect(!hf.moe_layers[0] and hf.moe_layers[1] and hf.moe_layers[11]);
    try std.testing.expectEqual(RouterScoring.sigmoid, hf.moe.scoring);
    try std.testing.expect(hf.norm_topk_prob and hf.moe.topk_method == .group_limited);
    try std.testing.expectEqual(QkvLayout.grouped, hf.qkv_alt.?);
    try std.testing.expectEqual(@as(usize, 2), hf.qkv_chunks);
    // Hub checkpoint spelling (MiMo-V2-Flash / V2.5 / V2.6): explicit lists and swa_* keys.
    const hub = try parseConfig(a,
        \\{"model_type":"mimo_v2","hidden_size":64,"num_attention_heads":8,"num_key_value_heads":2,"swa_num_key_value_heads":4,"swa_num_attention_heads":8,"swa_head_dim":24,"swa_v_head_dim":16,"num_hidden_layers":4,"vocab_size":100,"head_dim":24,"v_head_dim":16,"hybrid_layer_pattern":[0,1,1,0],"sliding_window_size":64,"add_swa_attention_sink_bias":true,"rope_theta":5000000.0,"swa_rope_theta":10000.0,"partial_rotary_factor":0.334,"attention_value_scale":null,"n_routed_experts":8,"num_experts_per_tok":2,"moe_intermediate_size":16,"moe_layer_freq":[0,1,1,1],"layernorm_epsilon":1e-5,"topk_method":"noaux_tc","routed_scaling_factor":null}
    );
    try std.testing.expectEqualStrings("mimo_v2_flash", hub.arch.model_type);
    try std.testing.expectEqualSlices(bool, &.{ false, true, true, false }, hub.sliding_layers);
    try std.testing.expectEqualSlices(usize, &.{ 2, 4, 4, 2 }, hub.layer_kv_heads);
    try std.testing.expectEqual(@as(?usize, 64), hub.sliding_window);
    try std.testing.expectEqual(@as(f32, 1.0), hub.mult.value);
    try std.testing.expectEqual(@as(f32, 1.0), hub.moe.routed_scaling_factor);
    try std.testing.expectEqual(@as(usize, 8), hub.rotary_dim);
    try std.testing.expectEqualSlices(bool, &.{ false, true, true, true }, hub.moe_layers);
}

test "parseConfig handles the swept families' keys" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    // transformers v5 `rope_parameters`, flat and per layer type.
    const mel = try parseConfig(a,
        \\{"model_type":"mellum","hidden_size":32,"num_attention_heads":4,"num_key_value_heads":2,"num_hidden_layers":2,"head_dim":8,"vocab_size":100,"num_experts":4,"num_experts_per_tok":2,"moe_intermediate_size":12,"mlp_layer_types":["dense","sparse"],"rope_parameters":{"full_attention":{"rope_type":"default","rope_theta":50000.0},"sliding_attention":{"rope_type":"default","rope_theta":10000.0}}}
    );
    try std.testing.expectEqual(@as(f32, 50000.0), mel.rope_theta);
    try std.testing.expectEqual(@as(f32, 10000.0), mel.rope_local.?.theta);
    try std.testing.expect(!mel.moe_layers[0] and mel.moe_layers[1]);
    // GPT-Neo expands `attention_types` into per-layer local/global attention.
    const neo = try parseConfig(a,
        \\{"model_type":"gpt_neo","hidden_size":32,"num_heads":4,"num_layers":4,"vocab_size":100,"window_size":8,"attention_types":[[["global","local"],2]]}
    );
    try std.testing.expectEqual(@as(f32, 1.0), neo.attention_scale);
    try std.testing.expectEqualSlices(bool, &.{ false, true, false, true }, neo.sliding_layers);
    try std.testing.expectEqual(@as(?usize, 8), neo.sliding_window);
    // AFMoE: sigmoid routing scaled by route_scale, a gated attention output,
    // dense first layers and 1-in-n global attention.
    const af = try parseConfig(a,
        \\{"model_type":"afmoe","hidden_size":32,"num_attention_heads":4,"num_hidden_layers":4,"head_dim":8,"vocab_size":100,"num_experts":4,"num_experts_per_tok":2,"moe_intermediate_size":12,"num_dense_layers":1,"route_scale":1.5,"sliding_window":4,"global_attn_every_n_layers":2}
    );
    try std.testing.expectEqual(AttnGate.sigmoid, af.attn_gate);
    try std.testing.expectEqual(@as(f32, 1.5), af.moe.routed_scaling_factor);
    try std.testing.expect(!af.moe_layers[0] and af.moe_layers[1]);
    try std.testing.expectEqualSlices(bool, &.{ true, false, true, false }, af.sliding_layers);
    // Laguna: a softplus gate and a tanh softcap on the router logits.
    const lag = try parseConfig(a,
        \\{"model_type":"laguna","hidden_size":32,"num_attention_heads":4,"num_key_value_heads":2,"num_hidden_layers":2,"head_dim":8,"vocab_size":100,"num_experts":4,"num_experts_per_tok":2,"moe_intermediate_size":12,"moe_router_logit_softcapping":5.0,"moe_routed_scaling_factor":2.0}
    );
    try std.testing.expectEqual(AttnGate.softplus, lag.attn_gate);
    try std.testing.expectEqual(@as(?f32, 5.0), lag.moe.router_softcap);
    try std.testing.expectEqualSlices(usize, &.{ 4, 4 }, lag.layer_heads);
    // A config that leaves out the norm epsilon or the RoPE base gets the
    // family's transformers default (allenai/OLMoE-1B-7B-0924 has no
    // rms_norm_eps: 1e-5, not the generic RMSNorm 1e-6).
    const oe = try parseConfig(a,
        \\{"model_type":"olmoe","hidden_size":32,"num_attention_heads":4,"num_hidden_layers":2,"vocab_size":100,"num_experts":4,"num_experts_per_tok":2,"intermediate_size":16}
    );
    try std.testing.expectEqual(@as(f32, 1e-5), oe.rms_norm_eps);
    const mx = try parseConfig(a,
        \\{"model_type":"mixtral","hidden_size":32,"num_attention_heads":4,"num_hidden_layers":2,"vocab_size":100,"num_local_experts":4,"num_experts_per_tok":2,"intermediate_size":16}
    );
    try std.testing.expectEqual(@as(f32, 1e6), mx.rope_theta);
    try std.testing.expectEqual(@as(f32, 1e-5), mx.rms_norm_eps);
    // nanochat: logits softcapped at 15 even when the config does not say so,
    // and the rotation runs backwards.
    const nc = try parseConfig(a,
        \\{"model_type":"nanochat","hidden_size":32,"num_attention_heads":4,"num_hidden_layers":2,"vocab_size":100,"logits_soft_cap":15.0}
    );
    try std.testing.expectEqual(@as(?f32, 15.0), nc.final_logit_softcapping);
    try std.testing.expect(nc.rope_reverse and !lag.rope_reverse);
    // Released Laguna checkpoints vary the query heads per layer.
    const lag2 = try parseConfig(a,
        \\{"model_type":"laguna","hidden_size":32,"num_attention_heads":4,"num_attention_heads_per_layer":[4,6],"num_key_value_heads":2,"num_hidden_layers":2,"head_dim":8,"vocab_size":100,"num_experts":4,"num_experts_per_tok":2,"moe_intermediate_size":12}
    );
    try std.testing.expectEqualSlices(usize, &.{ 4, 6 }, lag2.layer_heads);
    try std.testing.expectEqual(@as(usize, 6 * 8), lag2.maxQDim());
    try std.testing.expectError(error.InvalidConfig, parseConfig(a,
        \\{"model_type":"laguna","hidden_size":32,"num_attention_heads":4,"num_attention_heads_per_layer":[4,5],"num_key_value_heads":2,"num_hidden_layers":2,"head_dim":8,"vocab_size":100,"num_experts":4,"num_experts_per_tok":2,"moe_intermediate_size":12}
    ));
    // Gemma 2 alternates local and global layers, scales the queries by
    // `query_pre_attn_scalar` and softcaps both the attention and the output
    // logits.
    const g2 = try parseConfig(a,
        \\{"model_type":"gemma2","hidden_size":32,"intermediate_size":32,"num_attention_heads":4,"num_key_value_heads":2,"num_hidden_layers":4,"head_dim":8,"vocab_size":100,"sliding_window":4,"query_pre_attn_scalar":16,"attn_logit_softcapping":1.0,"final_logit_softcapping":20.0}
    );
    try std.testing.expectEqualSlices(bool, &.{ true, false, true, false }, g2.sliding_layers);
    try std.testing.expectEqual(@as(f32, 0.25), g2.attention_scale);
    try std.testing.expectEqual(@as(?f32, 1.0), g2.attn_logit_softcapping);
    try std.testing.expectEqual(@as(?f32, 20.0), g2.final_logit_softcapping);
    // Qwen2-MoE: `decoder_sparse_step` picks the routed layers and
    // `mlp_only_layers` takes some of them back out.
    const q2m = try parseConfig(a,
        \\{"model_type":"qwen2_moe","hidden_size":32,"intermediate_size":32,"moe_intermediate_size":12,"shared_expert_intermediate_size":16,"num_attention_heads":4,"num_key_value_heads":2,"num_hidden_layers":4,"vocab_size":100,"num_experts":4,"num_experts_per_tok":2,"decoder_sparse_step":2}
    );
    try std.testing.expectEqualSlices(bool, &.{ false, true, false, true }, q2m.moe_layers);
    try std.testing.expectEqual(@as(usize, 12), q2m.moe_intermediate_size);
    try std.testing.expectEqual(@as(usize, 32), q2m.intermediate_size);
    const q2m_only = try parseConfig(a,
        \\{"model_type":"qwen2_moe","hidden_size":32,"intermediate_size":32,"moe_intermediate_size":12,"num_attention_heads":4,"num_key_value_heads":2,"num_hidden_layers":4,"vocab_size":100,"num_experts":4,"num_experts_per_tok":2,"decoder_sparse_step":2,"mlp_only_layers":[3]}
    );
    try std.testing.expectEqualSlices(bool, &.{ false, true, false, false }, q2m_only.moe_layers);
    // Ministral 3 scales every layer's queries; Persimmon and CodeGen pick
    // their fused-qkv layouts.
    const min3 = try parseConfig(a,
        \\{"model_type":"ministral3","hidden_size":32,"num_attention_heads":4,"num_key_value_heads":2,"num_hidden_layers":2,"head_dim":8,"vocab_size":100,"max_position_embeddings":64,"rope_parameters":{"rope_type":"default","rope_theta":10000.0,"llama_4_scaling_beta":0.5}}
    );
    try std.testing.expect(min3.attn_temperature.?.all_layers and min3.attn_temperature.?.offset == 0);
    try std.testing.expectEqual(@as(f32, 64), min3.attn_temperature.?.floor_scale);
    const cg = try parseConfig(a,
        \\{"model_type":"codegen","n_embd":32,"n_head":4,"n_layer":2,"n_positions":128,"rotary_dim":4,"vocab_size":100}
    );
    try std.testing.expectEqual(QkvLayout.mp_blocks, cg.qkv_layout);
    try std.testing.expectEqual(@as(usize, 4), cg.qkv_mp);
    try std.testing.expectEqual(@as(usize, 4), cg.rotary_dim);
    // Aliases and guards.
    try std.testing.expectEqualStrings("llama", lookup("cwm").?.model_type);
    try std.testing.expectEqualStrings("qwen3_moe", lookup("qwen3_vl_moe_text").?.model_type);
    try std.testing.expectEqualStrings("deepseek_v2", lookup("youtu").?.model_type);
    // The families the sweep left out are named guards, not unknown types.
    try std.testing.expect(lookup("dbrx") == null);
    try std.testing.expect(lookup("phimoe") == null);
}

test "parseConfig handles the MiniMax, HunYuan, ERNIE and Granite MoE keys" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    // Remote-code MiniMax-M1 keys: attn_type_list and layernorm_* scales.
    const m1 = try parseConfig(a,
        \\{"model_type":"minimax_m1","hidden_size":64,"num_attention_heads":4,"num_key_value_heads":2,"head_dim":16,"num_hidden_layers":4,"vocab_size":100,"rotary_dim":8,"attn_type_list":[0,0,0,1],"layernorm_full_attention_alpha":3.5,"layernorm_linear_attention_alpha":3.5,"layernorm_mlp_alpha":3.5,"layernorm_mlp_beta":1,"num_local_experts":8,"num_experts_per_tok":2,"postnorm":true}
    );
    try std.testing.expectEqual(LinearKind.lightning, m1.linear_kind);
    try std.testing.expectEqual(ResidualLayout.minimax, m1.residual_layout);
    try std.testing.expect(m1.linear_layers[0] and m1.linear_layers[2] and !m1.linear_layers[3] and m1.has_linear);
    try std.testing.expectEqual(@as(usize, 8), m1.rotary_dim);
    try std.testing.expectEqual(@as(f32, 3.5), m1.minimax_scales[1][0]);
    try std.testing.expectEqual(@as(f32, 1.0), m1.minimax_scales[2][1]);
    try std.testing.expect(m1.norm_topk_prob and m1.moe_layers[0]);
    // Released checkpoints set `postnorm: true`, which is `.minimax` (it was
    // refused before); the native `minimax` type is post-norm whatever the key says.
    const mhf = try parseConfig(a,
        \\{"model_type":"minimax","hidden_size":64,"num_attention_heads":4,"num_key_value_heads":2,"head_dim":16,"num_hidden_layers":2,"vocab_size":100,"num_local_experts":8,"num_experts_per_tok":2,"postnorm":true}
    );
    try std.testing.expectEqual(ResidualLayout.minimax, mhf.residual_layout);
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

test "parseConfig picks the Gemma 4, Gemma 3n, LFM2 and Mistral 4 knobs" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const g4 = try parseConfig(a,
        \\{"model_type":"gemma4","text_config":{"model_type":"gemma4_text","hidden_size":64,"num_attention_heads":4,"num_key_value_heads":2,"head_dim":16,"num_hidden_layers":12,"vocab_size":100,"global_head_dim":32,"num_global_key_value_heads":1,"attention_k_eq_v":true,"num_kv_shared_layers":2,"hidden_size_per_layer_input":8,"sliding_window":512,"rope_parameters":{"full_attention":{"rope_type":"proportional","partial_rotary_factor":0.25,"rope_theta":1000000.0},"sliding_attention":{"rope_type":"default","rope_theta":10000.0}}}}
    );
    // Default pattern: 5 sliding, 1 global; the last layer is global.
    try std.testing.expect(g4.sliding_layers[0] and !g4.sliding_layers[5] and g4.sliding_layers[6] and !g4.sliding_layers[11]);
    try std.testing.expectEqual(@as(usize, 32), g4.layer_head_dim[5]);
    try std.testing.expectEqual(@as(usize, 1), g4.layer_kv_heads[5]);
    try std.testing.expectEqual(@as(usize, 16), g4.layer_head_dim[0]);
    try std.testing.expectEqual(@as(usize, 2), g4.layer_kv_heads[0]);
    try std.testing.expectEqual(@as(usize, 32), g4.kvDim());
    try std.testing.expectEqual(@as(f32, 1.0), g4.layer_attn_scale[3]);
    // Proportional rope: 8 rotated coordinates with frequencies over the 32-wide head.
    try std.testing.expectEqual(@as(usize, 8), g4.rotary_dim);
    try std.testing.expectEqual(@as(usize, 32), g4.rope_freq_dim);
    try std.testing.expectEqual(@as(f32, 1000000.0), g4.rope_theta);
    try std.testing.expectEqual(@as(f32, 10000.0), g4.rope_local.?.theta);
    try std.testing.expectEqual(@as(usize, 16), g4.rope_local.?.rotary_dim);
    // Layers 10 (sliding) and 11 (global) read layers 9 and 5.
    try std.testing.expectEqual(@as(usize, 9), g4.kv_source[10]);
    try std.testing.expectEqual(@as(usize, 5), g4.kv_source[11]);
    try std.testing.expectEqual(@as(usize, 8), g4.kv_source[8]);
    try std.testing.expect(g4.k_eq_v and g4.v_norm and g4.ple_dim == 8);

    const g3n = try parseConfig(a,
        \\{"model_type":"gemma3n_text","hidden_size":64,"intermediate_size":[128,256],"num_attention_heads":4,"num_key_value_heads":2,"head_dim":16,"num_hidden_layers":2,"vocab_size":100,"altup_num_inputs":4,"laurel_rank":8,"activation_sparsity_pattern":[0.95,0.0]}
    );
    try std.testing.expectEqual(@as(usize, 256), g3n.intermediate_size);
    try std.testing.expect(g3n.intermediate_varies);
    try std.testing.expectEqual(@as(usize, 4), g3n.altup_inputs);
    try std.testing.expectEqual(@as(f32, 0.95), g3n.activation_sparsity[0]);
    try std.testing.expectEqual(@as(usize, 256), g3n.ple_dim);

    const lfm = try parseConfig(a,
        \\{"model_type":"lfm2","hidden_size":64,"num_attention_heads":4,"num_key_value_heads":2,"num_hidden_layers":4,"vocab_size":100,"norm_eps":1e-5,"conv_L_cache":3,"block_ff_dim":12288,"block_multiple_of":256,"block_ffn_dim_multiplier":1.0,"block_auto_adjust_ff_dim":true,"full_attn_idxs":[2]}
    );
    try std.testing.expect(lfm.conv_layers[0] and lfm.conv_layers[1] and !lfm.conv_layers[2] and lfm.conv_layers[3]);
    try std.testing.expect(lfm.has_conv);
    try std.testing.expectEqual(@as(usize, 8192), lfm.intermediate_size);
    try std.testing.expectEqual(@as(f32, 1000000.0), lfm.rope_theta);
    try std.testing.expectEqual(QkNorm.head, lfm.qk_norm);

    const m4 = try parseConfig(a,
        \\{"model_type":"mistral4","hidden_size":64,"num_attention_heads":4,"num_hidden_layers":3,"vocab_size":100,"q_lora_rank":32,"kv_lora_rank":16,"qk_nope_head_dim":8,"qk_rope_head_dim":4,"v_head_dim":8,"n_routed_experts":8,"n_shared_experts":1,"num_experts_per_tok":2,"n_group":2,"topk_group":1,"first_k_dense_replace":1,"rope_parameters":{"rope_type":"yarn","rope_theta":10000.0,"factor":128.0,"original_max_position_embeddings":8192,"beta_fast":32.0,"beta_slow":1.0,"mscale":1.0,"mscale_all_dim":1.0,"llama_4_scaling_beta":0.1}}
    );
    try std.testing.expect(m4.rope_scaling == .yarn);
    try std.testing.expectEqual(RopeStyle.gptj, m4.rope_style);
    try std.testing.expect(m4.moe.group_score_top2 and m4.moe.topk_method == .group_limited);
    try std.testing.expect(!m4.moe_layers[0] and m4.moe_layers[1]);
    const t = m4.attn_temperature.?;
    try std.testing.expect(t.all_layers and t.offset == 0 and t.floor_scale == 8192 and t.attn_scale == 0.1);
    const ms4 = 0.1 * @log(@as(f32, 128)) + 1.0;
    try std.testing.expectApproxEqRel(ms4 * ms4 / @sqrt(@as(f32, 12)), m4.attention_scale, 1e-5);
}

test "parseConfig picks the Qwen4-Exp, GLM-5.3-Flash and GLM-4.7-Flash knobs" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    // Qwen4-Exp: the text config under the wrapper, `full_attention` spelling
    // of the indexed layers, sigmoid linear gate, PLE on two linear layers.
    const q4 = try parseConfig(a,
        \\{"model_type":"qwen4_exp","text_config":{"model_type":"qwen4_exp_text","hidden_size":64,"num_attention_heads":4,"num_key_value_heads":2,"head_dim":16,"num_hidden_layers":4,"vocab_size":100,"eos_token_id":[7,8],"layer_types":["linear_attention","linear_attention","linear_attention","full_attention"],"linear_num_key_heads":2,"linear_key_head_dim":8,"linear_num_value_heads":4,"linear_value_head_dim":8,"linear_conv_kernel_dim":4,"num_experts":8,"num_experts_per_tok":2,"moe_intermediate_size":16,"shared_expert_intermediate_size":16,"hc_count":3,"hc_lowrank":12,"output_gate_type":"sigmoid","indexer_n_heads":2,"indexer_kv_heads":1,"indexer_head_dim":16,"indexer_budget":8,"indexer_compress_ratio":2,"ple_layer_ids":[3,1],"ngram_size":3,"heads_per_ngram":2,"ngram_vocab_size_base":50,"rope_parameters":{"rope_type":"default","rope_theta":10000.0,"partial_rotary_factor":0.25,"mrope_section":[11,11,10]}}}
    );
    try std.testing.expectEqualStrings("qwen4_exp", q4.arch.model_type);
    try std.testing.expect(q4.linear_layers[0] and q4.linear_layers[2] and !q4.linear_layers[3] and q4.has_linear);
    try std.testing.expect(q4.gated_attention and !q4.gate_swish and q4.linear_gate_sigmoid);
    try std.testing.expectEqual(NormKind.rms_gemma, q4.norm);
    try std.testing.expectEqual(@as(usize, 3), q4.hc_mult);
    try std.testing.expectEqual(HyperKind.gated, q4.hyper.?.kind);
    try std.testing.expectEqual(HyperHead.gated_mixer, q4.hyper.?.head);
    try std.testing.expectEqual(@as(usize, 12), q4.hyper.?.lowrank);
    try std.testing.expectEqual(@as(usize, 4), q4.rotary_dim);
    try std.testing.expect(q4.moe_layers[0] and q4.moe_layers[3] and q4.norm_topk_prob);
    try std.testing.expectEqual(RouterScoring.softmax, q4.moe.scoring);
    try std.testing.expectEqual(@as(usize, 2), q4.index_bound.?.block);
    try std.testing.expectEqual(@as(usize, 4), q4.index_bound.?.max_blocks);
    const ple = q4.ngram_ple.?;
    try std.testing.expectEqualSlices(usize, &.{ 0, 2 }, ple.layer_ids);
    try std.testing.expectEqual(@as(u32, 7), ple.eos_id);
    try std.testing.expectEqual(@as(usize, 64), ple.embed_dim);
    try std.testing.expectEqual(@as(u64, 50), ple.vocab_base);
    try std.testing.expectEqual(@as(u64, 100), ple.vocab_size);
    // Defaults: 3:1 layers, no indexer, no PLE.
    const q4d = try parseConfig(a,
        \\{"model_type":"qwen4_exp_text","hidden_size":64,"num_attention_heads":4,"num_key_value_heads":2,"head_dim":16,"num_hidden_layers":8,"vocab_size":100,"linear_num_key_heads":2,"linear_key_head_dim":8,"linear_num_value_heads":4,"linear_value_head_dim":8,"linear_conv_kernel_dim":4,"num_experts":8,"moe_intermediate_size":16}
    );
    try std.testing.expect(q4d.linear_layers[0] and !q4d.linear_layers[3] and q4d.linear_layers[4] and !q4d.linear_layers[7]);
    try std.testing.expect(q4d.index_bound == null and q4d.ngram_ple == null and !q4d.linear_gate_sigmoid);
    try std.testing.expectEqual(@as(usize, 10), q4d.num_experts_per_tok);
    try std.testing.expectEqual(@as(usize, 4), q4d.hc_mult);

    // GLM-5.3-Flash: the text config under the wrapper, default layer and
    // MLP schedules, NoPE MLA, the lower-bound forget gate, k-pool indexer.
    const g5 = try parseConfig(a,
        \\{"model_type":"glm5_next","text_config":{"model_type":"glm5_next_text","hidden_size":64,"num_attention_heads":4,"num_key_value_heads":4,"num_hidden_layers":8,"vocab_size":100,"q_lora_rank":16,"kv_lora_rank":16,"qk_nope_head_dim":16,"qk_rope_head_dim":0,"v_head_dim":8,"n_routed_experts":8,"n_shared_experts":1,"num_experts_per_tok":2,"moe_intermediate_size":16,"linear_num_heads":2,"linear_head_dim":16,"linear_conv_kernel_dim":4,"index_kpool":4,"index_topk":64,"swiglu_limit":10.0,"hc_mult":4}}
    );
    try std.testing.expectEqualStrings("glm5_next", g5.arch.model_type);
    try std.testing.expectEqual(LinearKind.kda, g5.linear_kind);
    try std.testing.expect(g5.linear_layers[0] and g5.linear_layers[2] and !g5.linear_layers[3] and !g5.linear_layers[7]);
    try std.testing.expect(!g5.moe_layers[2] and g5.moe_layers[3]);
    try std.testing.expectEqual(@as(f32, -5.0), g5.linear_gate_lower_bound.?);
    try std.testing.expectEqual(@as(usize, 16), g5.head_dim);
    try std.testing.expectEqual(@as(usize, 8), g5.v_head_dim);
    try std.testing.expectEqual(@as(usize, 0), g5.rotary_dim);
    try std.testing.expect(!g5.rope_layers[3] and g5.mla != null and g5.mla.?.qk_rope_head_dim == 0);
    try std.testing.expectEqual(@as(f32, 0.25), g5.attention_scale);
    try std.testing.expectEqual(@as(f32, 1e-5), g5.rms_norm_eps);
    try std.testing.expectEqual(HyperKind.mhc, g5.hyper.?.kind);
    try std.testing.expectEqual(HyperHead.mean, g5.hyper.?.head);
    try std.testing.expectEqual(@as(usize, 4), g5.index_bound.?.block);
    try std.testing.expectEqual(@as(usize, 16), g5.index_bound.?.max_blocks);
    try std.testing.expect(g5.moe.swiglu_limit.? == 10.0 and g5.moe.group_score_top2 and g5.moe.norm_eps_floor);
    try std.testing.expectEqual(RouterScoring.sigmoid, g5.moe.scoring);
    try std.testing.expectEqual(@as(f32, 2.5), g5.moe.routed_scaling_factor);
    // Original checkpoint spellings: linear_attn_config with a null lower bound and safe_gate off, explicit layer types.
    const g5b = try parseConfig(a,
        \\{"model_type":"glm5_next_text","hidden_size":64,"num_attention_heads":4,"num_hidden_layers":3,"vocab_size":100,"q_lora_rank":16,"kv_lora_rank":16,"qk_nope_head_dim":16,"qk_rope_head_dim":0,"v_head_dim":8,"n_routed_experts":8,"num_experts_per_tok":2,"moe_intermediate_size":16,"linear_attn_config":{"num_heads":2,"head_dim":16,"short_conv_kernel_size":3,"gate_lower_bound":null,"safe_gate":false},"layer_types":["linear_attention","indexed_attention","linear_attention"],"mlp_layer_types":["dense","sparse","sparse"],"indexer_types":["shared","full","shared"]}
    );
    try std.testing.expect(g5b.linear_gate_lower_bound == null);
    try std.testing.expectEqual(@as(usize, 3), g5b.linear_conv_kernel);
    try std.testing.expect(g5b.linear_layers[0] and !g5b.linear_layers[1] and g5b.linear_layers[2]);
    try std.testing.expect(!g5b.moe_layers[0] and g5b.moe_layers[1]);

    // GLM-4.7-Flash: DeepSeek V3 MLA, GLM-4.5 routing, a dense first layer by default.
    const gl = try parseConfig(a,
        \\{"model_type":"glm4_moe_lite","hidden_size":64,"num_attention_heads":4,"num_key_value_heads":4,"num_hidden_layers":3,"vocab_size":100,"q_lora_rank":16,"kv_lora_rank":16,"qk_nope_head_dim":8,"qk_rope_head_dim":4,"v_head_dim":8,"n_routed_experts":8,"n_shared_experts":1,"num_experts_per_tok":2,"moe_intermediate_size":16,"n_group":2,"topk_group":1,"routed_scaling_factor":1.8}
    );
    try std.testing.expectEqualStrings("glm4_moe_lite", gl.arch.model_type);
    try std.testing.expectEqualStrings("glm4_moe_lite", lookup("glm_moe_lite").?.model_type);
    try std.testing.expect(!gl.moe_layers[0] and gl.moe_layers[1] and gl.moe_layers[2]);
    try std.testing.expectEqual(RopeStyle.gptj, gl.rope_style);
    try std.testing.expectEqual(@as(usize, 4), gl.rotary_dim);
    try std.testing.expectEqual(@as(usize, 12), gl.head_dim);
    try std.testing.expect(gl.moe.group_score_top2 and gl.moe.norm_eps_floor and gl.norm_topk_prob);
    try std.testing.expectEqual(@as(usize, 2), gl.moe.n_group);
    try std.testing.expectEqual(@as(f32, 1e-5), gl.rms_norm_eps);
    try std.testing.expectEqual(@as(f32, 1.8), gl.moe.routed_scaling_factor);
}
