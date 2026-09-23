//! Architecture descriptors: what the loader and the forward pass need to
//! know about a family (tensor-name templates, norm and residual layout,
//! attention and MLP layouts, positional encoding, MoE routing), the
//! generalised `Config` and its generic config.json parser. The families
//! themselves are Lua model definitions (src/models/*.lua, read by
//! models.zig; docs/models.md): adding one is a definition file and, only
//! when the family needs a genuinely new computation, a new layout value
//! here and a small code path in model.zig or moe.zig.
//!
//! Every built-in definition documents what was verified against a NumPy
//! reference fixture (`tools/make_fixture.py`, `src/model_test.zig`). Every
//! one has such a fixture; `verified = false` marks a family implemented from
//! the Hugging Face reference implementation but not yet checked against one.

const std = @import("std");
const tensor = @import("tensor.zig");
const dequant = @import("dequant.zig");
const models = @import("models.zig");
const build_options = @import("build_options");

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
    /// Family-specific config keys: a Zig hook (`hooks`, named by a
    /// definition's `hook` field) ...
    extra: ?*const fn (*Config, Allocator, std.json.ObjectMap) anyerror!void = null,
    /// ... and a Lua `config` function run after it (a reference into the
    /// definitions' Lua state, see models.zig).
    script: ?c_int = null,
    /// The family a definition was derived from with `base`: its hook and
    /// config function see this as `model_type`, so that one shared by
    /// several families (Gemma 2 and 3) takes the base family's branch.
    inherits: ?[]const u8 = null,
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
    /// Proportional RoPE (Gemma 4's global layers): only the first
    /// `rope_angles` frequency pairs of the global table turn, the others are
    /// zero; the table spans the whole head, so pair `i` is coordinates `i`
    /// and `i + head_dim / 2`, as transformers' `rotate_half` over the head
    /// pairs them. 0 = every pair turns.
    rope_angles: usize = 0,
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
    /// Factor on the ALiBi bias: Falcon adds it before scaling the scores by
    /// 1/sqrt(head_dim), so its bias is divided by sqrt(head_dim); BLOOM and MPT add it after.
    alibi_scale: f32 = 1.0,
    /// The LM head normalises each row to unit length at inference (Baichuan 2's `NormHead`).
    lm_head_l2norm: bool = false,
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
    /// Full-attention `q_proj` carries q rows then gate rows; the gate's
    /// sigmoid multiplies the attention output. (`output_gate_type` never
    /// names this gate: in the Qwen configs it is the Gated DeltaNet's.)
    gated_attention: bool,
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

/// Config parsing reports why it refuses a checkpoint as an error message;
/// the Lua/Zig equivalence test parses configs that are meant to be refused
/// and sets this to keep those messages at debug level.
pub var quiet_errors = false;

pub fn logErr(comptime format: []const u8, args: anytype) void {
    if (quiet_errors) std.log.debug(format, args) else std.log.err(format, args);
}

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

/// Finds the definition of a `model_type` (the Lua model definitions, see models.zig).
pub fn lookup(model_type: []const u8) ?*const Arch {
    return models.lookup(model_type);
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
            logErr("unsupported model: {s}", .{e[1]});
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
            logErr("unsupported model: '{s}' expert dtype cannot be dequantised (bf16/f16/f32, fp8, fp4, mxfp4 and pack-quantized int4 are supported)", .{dt});
            return error.UnsupportedArchitecture;
        }
        // DeepSeek V4 spells its FP4 experts at the top level of config.json.
        if (std.ascii.eqlIgnoreCase(dt, "fp4")) {
            if (quant.method != .fp8) {
                logErr("unsupported model: expert_dtype fp4 without an fp8 quantization_config (found: {s})", .{quant.label});
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

/// The family of a remote-code config that names no `model_type` (MiniCPM4
/// writes only `architectures` and `auto_map`): `MiniCPMForCausalLM` ->
/// `minicpm`, when the lowercased class prefix is a registered model type.
/// Without it such a config would be read as Llama and lose its family's
/// scalings.
fn typeFromArchitectures(arena: Allocator, obj: std.json.ObjectMap) ?[]const u8 {
    const archs = obj.get("architectures") orelse return null;
    if (archs != .array or archs.array.items.len == 0 or archs.array.items[0] != .string) return null;
    const name = archs.array.items[0].string;
    for ([_][]const u8{ "ForCausalLM", "LMHeadModel", "ForConditionalGeneration" }) |suffix| {
        if (!std.mem.endsWith(u8, name, suffix)) continue;
        const lower = std.ascii.allocLowerString(arena, name[0 .. name.len - suffix.len]) catch return null;
        if (lookup(lower)) |a| return a.model_type;
    }
    return null;
}

pub fn parseConfig(arena: Allocator, json_text: []const u8) !Config {
    return parseConfigAs(arena, json_text, null);
}

/// `parseConfig` with the family given instead of looked up (`ditch
/// add-model` compares a draft with the definition it would replace).
pub fn parseConfigAs(arena: Allocator, json_text: []const u8, family: ?*const Arch) !Config {
    var parsed = try std.json.parseFromSlice(std.json.Value, arena, try sanitizeJson(arena, json_text), .{});
    defer parsed.deinit();
    if (parsed.value != .object) return error.InvalidConfig;
    var obj = parsed.value.object;
    const top_type = getStr(obj, "model_type") orelse typeFromArchitectures(arena, obj) orelse "llama";
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
    if (family) |f| arch_opt = f;
    const arch = arch_opt orelse {
        try rejectKnownHybrid(model_type);
        try rejectKnownHybrid(top_type);
        _ = try rejectUnsupportedMath(parsed.value.object, obj);
        // Families whose name shares the unknown one's stem are likely relatives.
        var similar: std.Io.Writer.Allocating = .init(arena);
        const stem = std.mem.trimEnd(u8, model_type[0 .. std.mem.indexOfAny(u8, model_type, "_-") orelse model_type.len], "0123456789.");
        var n_similar: usize = 0;
        for (models.families()) |a| {
            if (stem.len < 3 or !std.mem.startsWith(u8, a.model_type, stem) or n_similar == 6) continue;
            try similar.writer.writeAll(if (n_similar == 0) " (related families: " else ", ");
            try similar.writer.writeAll(a.model_type);
            n_similar += 1;
        }
        if (n_similar > 0) try similar.writer.writeAll(")");
        logErr("unknown model_type: {s}: there is no built-in or user model definition for it{s}", .{ model_type, similar.written() });
        return error.UnknownModelType;
    };
    const quant = try rejectUnsupportedMath(parsed.value.object, obj);
    // Nested attention config (MPT).
    const attn_cfg: std.json.ObjectMap = getObj(obj, "attn_config") orelse obj;

    const hidden = getIntAny(obj, &.{ "hidden_size", "n_embd", "n_embed", "d_model" }, 0); // n_embed: the BLOOM releases
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
                            logErr("unsupported layer type 'mamba' in a {s} model", .{model_type});
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
                            logErr("unsupported layer type '{s}' in a {s} model", .{ t, model_type });
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
                        logErr("unsupported layer type '{s}' (only full/sliding/linear attention, conv, Mamba, mlp and moe blocks are implemented)", .{t});
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
    const own_type = c.model_type;
    if (arch.inherits) |t| c.model_type = t;
    if (arch.extra) |f| try f(&c, arena, obj);
    if (arch.script) |ref| try models.runConfig(ref, &c, arena, obj);
    c.model_type = own_type;
    // Hooks may mark linear / Mamba layers after the fact (Kimi Linear's
    // kda_layers, the Jamba periods): those layers hold no attention block
    // unless the family runs both side by side.
    for (0..layers) |i| c.attn_layers[i] = c.attn_layers[i] and !c.linear_layers[i] and !c.conv_layers[i] and (!c.ssm_layers[i] or c.parallel_ssm);
    for (c.ssm_layers) |s| c.has_ssm = c.has_ssm or s;
    if (c.has_ssm) {
        if (c.ssm.kind != arch.ssm) return error.InvalidConfig;
        const d = &c.ssm;
        if (d.inter == 0 or d.state == 0 or d.conv_kernel == 0 or (d.kind == .mamba2 and (d.heads == 0 or d.head_dim == 0 or d.groups == 0 or d.heads % d.groups != 0 or d.heads * d.head_dim != d.inter)) or (d.kind == .mamba1 and d.dt_rank == 0)) {
            logErr("inconsistent Mamba block dimensions in config.json", .{});
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
        logErr("conv layers need conv_L_cache >= 2", .{});
        return error.InvalidConfig;
    }
    if (c.has_linear and c.linear_kind != .lightning and (c.linear_k_heads == 0 or c.linear_k_dim == 0 or c.linear_v_heads == 0 or c.linear_v_dim == 0 or c.linear_conv_kernel == 0)) {
        logErr("linear_attention layers need linear_num_key_heads/key_head_dim/num_value_heads/value_head_dim/conv_kernel_dim", .{});
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
pub fn parseRopeScaling(arena: Allocator, obj: std.json.ObjectMap, rs: std.json.ObjectMap, rotary_dim: usize, max_pos: usize) !RopeScaling {
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

pub fn yarnMscale(scale: f32, mscale: f32) f32 {
    if (scale <= 1) return 1.0;
    return 0.1 * mscale * @log(scale) + 1.0;
}

/// The family hooks by name: Zig building blocks a model definition
/// (models.zig) names in its `hook` field, for family-specific config.json
/// keys whose reading Lua cannot express. A hook runs after the generic
/// parser and before the definition's `config` function. Every built-in
/// family reads its keys in Lua, so the table is empty.
pub const Hook = *const fn (*Config, Allocator, std.json.ObjectMap) anyerror!void;

pub const hooks = [_]struct { name: []const u8, f: Hook }{};

/// The hook registered under `name`.
pub fn hookByName(name: []const u8) ?Hook {
    for (hooks) |h| if (std.mem.eql(u8, h.name, name)) return h.f;
    return null;
}

/// The name of a hook, for writing a definition back out.
pub fn hookName(f: Hook) ?[]const u8 {
    for (hooks) |h| if (h.f == f) return h.name;
    return null;
}

test "registry lookup and aliases" {
    try std.testing.expect(lookup("llama") != null);
    try std.testing.expectEqualStrings("gemma3", lookup("gemma3_text").?.model_type);
    try std.testing.expectEqualStrings("qwen2", lookup("qwen2_5_vl").?.model_type);
    try std.testing.expectEqualStrings("gemma4", lookup("gemma4_text").?.model_type);
    try std.testing.expectEqualStrings("gemma3n", lookup("gemma3n_text").?.model_type);
    try std.testing.expectEqualStrings("mistral4", lookup("mistral4_text").?.model_type);
    try std.testing.expect(lookup("mamba") == null);
    // Every built-in definition has a unique model_type.
    const builtins = models.builtins();
    for (builtins, 0..) |a, i| {
        for (builtins[i + 1 ..]) |b| try std.testing.expect(!std.mem.eql(u8, a.model_type, b.model_type));
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
    {
        // The refusal says why; the test checks the error, not the message.
        quiet_errors = true;
        defer quiet_errors = false;
        try std.testing.expectError(error.InvalidConfig, parseConfig(a,
            \\{"model_type":"laguna","hidden_size":32,"num_attention_heads":4,"num_attention_heads_per_layer":[4,5],"num_key_value_heads":2,"num_hidden_layers":2,"head_dim":8,"vocab_size":100,"num_experts":4,"num_experts_per_tok":2,"moe_intermediate_size":12}
        ));
    }
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
    // Proportional rope: 4 turning pairs of the 32-wide head's 16, frequencies over the head.
    try std.testing.expectEqual(@as(usize, 32), g4.rotary_dim);
    try std.testing.expectEqual(@as(usize, 32), g4.rope_freq_dim);
    try std.testing.expectEqual(@as(usize, 4), g4.rope_angles);
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
    try std.testing.expect(q4.gated_attention and q4.linear_gate_sigmoid);
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

test "parseConfig: EXAONE MoE global layers have no RoPE" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    // K-EXAONE's release: explicit layer_types, and the string pattern.
    const lt = try parseConfig(a,
        \\{"model_type":"exaone_moe","hidden_size":64,"num_attention_heads":4,"num_key_value_heads":2,"num_hidden_layers":4,"vocab_size":100,"num_experts":8,"num_experts_per_tok":2,"moe_intermediate_size":16,"num_shared_experts":1,"first_k_dense_replace":1,"sliding_window":128,"sliding_window_pattern":"LLLG","layer_types":["sliding_attention","sliding_attention","sliding_attention","full_attention"]}
    );
    try std.testing.expect(lt.sliding_layers[0] and lt.sliding_layers[2] and !lt.sliding_layers[3]);
    try std.testing.expect(lt.rope_layers[0] and lt.rope_layers[2] and !lt.rope_layers[3]);
    const pat = try parseConfig(a,
        \\{"model_type":"exaone_moe","hidden_size":64,"num_attention_heads":4,"num_key_value_heads":2,"num_hidden_layers":4,"vocab_size":100,"num_experts":8,"num_experts_per_tok":2,"moe_intermediate_size":16,"sliding_window":128,"sliding_window_pattern":"LLLG"}
    );
    try std.testing.expect(pat.rope_layers[1] and !pat.sliding_layers[3] and !pat.rope_layers[3]);
}

test "parseConfig: BLOOM's n_embed spelling of the hidden size" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const c = try parseConfig(arena.allocator(),
        \\{"model_type":"bloom","n_embed":64,"n_layer":2,"num_attention_heads":4,"vocab_size":100,"layer_norm_epsilon":1e-5}
    );
    try std.testing.expectEqual(@as(usize, 64), c.hidden_size);
    try std.testing.expectEqual(@as(usize, 2), c.num_layers);
}

test "parseConfig: a config without model_type takes its family from architectures" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const c = try parseConfig(arena.allocator(),
        \\{"architectures":["MiniCPMForCausalLM"],"hidden_size":64,"num_attention_heads":4,"num_key_value_heads":2,"num_hidden_layers":2,"vocab_size":100,"scale_emb":12,"scale_depth":1.4,"dim_model_base":16}
    );
    try std.testing.expectEqualStrings("minicpm", c.arch.model_type);
    try std.testing.expectEqual(@as(f32, 12), c.embed_scale);
}

test "parseConfig: Baichuan 2 normalises its LM head, Baichuan 1 does not" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const b2 = try parseConfig(arena.allocator(),
        \\{"model_type":"baichuan","hidden_size":64,"num_attention_heads":4,"num_hidden_layers":2,"vocab_size":125696,"max_position_embeddings":4096}
    );
    try std.testing.expect(b2.lm_head_l2norm);
    const b1 = try parseConfig(arena.allocator(),
        \\{"model_type":"baichuan","hidden_size":64,"num_attention_heads":4,"num_hidden_layers":2,"vocab_size":64000,"max_position_embeddings":4096}
    );
    try std.testing.expect(!b1.lm_head_l2norm);
}

const CorpusEntry = struct { name: []const u8, text: []const u8 };

/// The config.json of every fixture, then the variants of
/// tests/config_variants/<fixture>.json: each is a list of patches applied
/// to the fixture's config (a key set to "$delete" is removed), covering the
/// keys the families read beyond what their fixture sets.
fn configCorpus(arena: Allocator) ![]CorpusEntry {
    const io = std.testing.io;
    var out: std.ArrayList(CorpusEntry) = .empty;
    var dir = try std.Io.Dir.cwd().openDir(io, "tests/fixtures", .{ .iterate = true });
    defer dir.close(io);
    var names: std.ArrayList([]const u8) = .empty;
    var it = dir.iterate();
    while (try it.next(io)) |entry| if (entry.kind == .directory) try names.append(arena, try arena.dupe(u8, entry.name));
    std.mem.sort([]const u8, names.items, {}, struct {
        fn lt(_: void, x: []const u8, y: []const u8) bool {
            return std.mem.lessThan(u8, x, y);
        }
    }.lt);
    for (names.items) |name| {
        const text = dir.readFileAlloc(io, try std.fmt.allocPrint(arena, "{s}/config.json", .{name}), arena, .limited(1 << 20)) catch continue;
        try out.append(arena, .{ .name = name, .text = text });
        const vpath = try std.fmt.allocPrint(arena, "tests/config_variants/{s}.json", .{name});
        const vtext = std.Io.Dir.cwd().readFileAlloc(io, vpath, arena, .limited(1 << 20)) catch continue;
        const base = try std.json.parseFromSliceLeaky(std.json.Value, arena, try sanitizeJson(arena, text), .{});
        const patches = try std.json.parseFromSliceLeaky(std.json.Value, arena, vtext, .{});
        if (patches != .array) return error.InvalidConfig;
        for (patches.array.items, 0..) |patch, k| {
            if (patch != .object) return error.InvalidConfig;
            var obj = try base.object.clone(arena);
            var pit = patch.object.iterator();
            while (pit.next()) |kv| {
                if (kv.value_ptr.* == .string and std.mem.eql(u8, kv.value_ptr.string, "$delete")) {
                    _ = obj.orderedRemove(kv.key_ptr.*);
                } else try obj.put(arena, kv.key_ptr.*, kv.value_ptr.*);
            }
            const patched = try std.json.Stringify.valueAlloc(arena, std.json.Value{ .object = obj }, .{});
            try out.append(arena, .{ .name = try std.fmt.allocPrint(arena, "{s} variant {d}", .{ name, k + 1 }), .text = patched });
        }
    }
    return out.items;
}

test "the model definitions parse every fixture and variant as recorded" {
    // tests/config_variants/snapshot.txt holds a hash of every built-in
    // family and of the configuration every fixture's config.json and
    // every variant of it parses to. It was recorded when the registry
    // moved from Zig to Lua, from the Zig registry, which the Lua
    // definitions reproduced exactly. After an intended change, rewrite
    // it with `zig build test -Dupdate-snapshot`.
    const gpa = std.testing.allocator;
    var arena_state = std.heap.ArenaAllocator.init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const io = std.testing.io;
    var now: std.Io.Writer.Allocating = .init(arena);
    var dumps: std.StringHashMapUnmanaged([]const u8) = .{};
    for (models.builtins()) |f| {
        const d = try models.dumpFamily(arena, f);
        const key = try std.fmt.allocPrint(arena, "family {s}", .{f.model_type});
        try dumps.put(arena, key, d);
        try now.writer.print("{s}\t{x}\n", .{ key, std.hash.Wyhash.hash(0, d) });
    }
    quiet_errors = true;
    defer quiet_errors = false;
    for (try configCorpus(arena)) |entry| {
        const d = try dumpParsedConfig(arena, entry.text);
        const key = try std.fmt.allocPrint(arena, "config {s}", .{entry.name});
        try dumps.put(arena, key, d);
        try now.writer.print("{s}\t{x}\n", .{ key, std.hash.Wyhash.hash(0, d) });
    }
    const path = "tests/config_variants/snapshot.txt";
    if (build_options.update_snapshot) {
        try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = path, .data = now.written() });
        return;
    }
    const recorded = try std.Io.Dir.cwd().readFileAlloc(io, path, arena, .limited(16 << 20));
    var want = std.mem.splitScalar(u8, recorded, '\n');
    var got = std.mem.splitScalar(u8, now.written(), '\n');
    var bad: usize = 0;
    while (true) {
        const w = want.next();
        const g = got.next();
        if (w == null and g == null) break;
        const wl = w orelse "";
        const gl = g orelse "";
        if (std.mem.eql(u8, wl, gl)) continue;
        bad += 1;
        if (bad <= 3) {
            const key = gl[0 .. std.mem.indexOfScalar(u8, gl, '\t') orelse gl.len];
            std.debug.print("recorded: {s}\nnow:      {s}\n{s}\n", .{ wl, gl, dumps.get(key) orelse "" });
        }
    }
    if (bad > 0) {
        std.debug.print("{d} line(s) differ from {s}; if the change is intended, run zig build test -Dupdate-snapshot\n", .{ bad, path });
        return error.TestUnexpectedResult;
    }
}

fn dumpParsedConfig(arena: Allocator, text: []const u8) ![]const u8 {
    const cfg = parseConfig(arena, text) catch |err| return @errorName(err);
    var out: std.Io.Writer.Allocating = .init(arena);
    try models.dump(Config, &out.writer, cfg, "");
    return out.written();
}
