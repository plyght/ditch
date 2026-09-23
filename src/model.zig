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
const compute = @import("compute.zig");
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
pub const dsv4 = @import("deepseek_v4.zig");
pub const hyper = @import("hyper.zig");
pub const qwen4 = @import("qwen4_exp.zig");
const dequant = @import("dequant.zig");

const Allocator = std.mem.Allocator;
const Weight = tensor.Weight;
const Delta = tensor.Delta;
const WeightRef = stream.WeightRef;

pub const Config = arch.Config;
const QkvLayout = arch.QkvLayout;
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
pub const Slot = enum {
    q,
    k,
    v,
    qkv,
    o,
    gate,
    up,
    gate_up,
    down,
    router,
    q_a,
    q_b,
    kv_a,
    kv_b,
    attn_gate,
    lin_qkvz,
    lin_qkv,
    lin_z,
    lin_b,
    lin_a,
    lin_ba,
    lin_conv,
    lin_q,
    lin_k,
    lin_v,
    lin_conv_q,
    lin_conv_k,
    lin_conv_v,
    lin_f_a,
    lin_f_b,
    lin_g_a,
    lin_g_b,
    lin_g,
    light_qkv,
    light_gate,
    // Mamba blocks
    ssm_in,
    ssm_x,
    ssm_dt,
    ssm_out,
    // DeepSeek V4 / V4.1 (deepseek_v4.zig)
    d_q_a,
    d_q_b,
    d_kv,
    d_o_a,
    d_comp_kv,
    d_comp_gate,
    d_engram_wkv,
    // Hyper-connection sites (hyper.zig) and Qwen4-Exp per-layer embeddings
    hc_attn_fn,
    hc_attn_down,
    hc_attn_up,
    hc_attn_inject,
    hc_ffn_fn,
    hc_ffn_down,
    hc_ffn_up,
    hc_ffn_inject,
    ple_key,
    ple_value,
    ple_conv,
    // LFM2 short convolution, Gemma 3n / 4 per-layer inputs, AltUp and Laurel
    conv_in,
    conv_kernel,
    ple_gate,
    ple_out,
    laurel_l,
    laurel_r,
    altup_router,
    altup_predict,
    altup_correct,
};

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

    pub const max = 24;

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
/// low-rank output gate `g_a`/`g_b`; MiniMax lightning attention carries
/// `light_qkv`, `light_gate` and `norm`. The output projection lives in
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
    /// Full-rank KDA output gate `[heads * head_dim][hidden]` (Kimi K3), replacing `g_a`/`g_b`.
    g: ?Weight = null,
    /// Depthwise causal convolution `[conv_dim][1][kernel]` (read as
    /// `[conv_dim][kernel]`), or one tensor per q/k/v projection (original
    /// Kimi Linear checkpoints).
    conv: ?Weight = null,
    conv_q: ?Weight = null,
    conv_k: ?Weight = null,
    conv_v: ?Weight = null,
    /// Gated DeltaNet: `[v_heads]` each; KDA: `dt_bias[heads * head_dim]`, `a_log[heads]`.
    dt_bias: []const f32 = &.{},
    a_log: []const f32 = &.{},
    /// Gated norm weight `[v_dim]` (DeltaNet, KDA) or `[heads * head_dim]` (lightning).
    norm: []const f32 = &.{},
    /// Lightning attention `[heads][q | k | v]` projection and sigmoid output gate.
    light_qkv: ?Weight = null,
    light_gate: ?Weight = null,
};

/// Mamba block weights (Mamba2 or Mamba1, see `arch.SsmKind`). When the
/// block replaces attention its output projection lives in `Layer.o` (with
/// `o_delta`), so abliteration and export treat it like any attention
/// output; in the parallel layout (Falcon-H1) it is `out` with
/// `Layer.ssm_out_delta`, next to the attention's o_proj.
pub const SsmWeights = struct {
    /// Mamba2: `[inter + conv_dim + heads][hidden]` (gate, conv channels, dt).
    /// Mamba1: `[2 inter][hidden]` (conv channels, gate).
    in_proj: Weight,
    in_bias: ?[]const f32,
    /// Depthwise causal convolution `[conv_dim][kernel]`.
    conv: []const f32,
    conv_bias: ?[]const f32,
    /// Mamba2: `[heads]`; Mamba1: the `dt_proj` bias `[inter]`.
    dt_bias: []const f32,
    /// `-exp(A_log)`: Mamba2 `[heads]`, Mamba1 `[inter][state]`.
    a: []const f32,
    /// Skip connection: Mamba2 `[heads]`, Mamba1 `[inter]`.
    d: []const f32,
    /// Gated RMSNorm weight `[inter]` (null: the gate alone scales the scan output).
    norm: ?[]const f32,
    /// Mamba1: `[dt_rank + 2 state][inter]`, `[inter][dt_rank]` and the RMS
    /// norms of the dt / B / C parts (Jamba).
    x_proj: ?Weight = null,
    dt_proj: ?Weight = null,
    dt_norm: ?[]const f32 = null,
    b_norm: ?[]const f32 = null,
    c_norm: ?[]const f32 = null,
    /// Output projection of the parallel layout (null: `Layer.o`).
    out: ?Weight = null,
    out_bias: ?[]const f32 = null,
};

/// Gated short convolution of an LFM2 conv layer (`in_proj` produces the B, C
/// and x thirds; the depthwise causal kernel is `[hidden][taps]`); its output
/// projection lives in `Layer.o`.
pub const ConvWeights = struct {
    in: Weight,
    in_bias: ?[]const f32,
    kernel: Weight,
    kernel_bias: ?[]const f32,
};

/// Per-layer input embedding block (Gemma 3n / 4): `gate` `[ple_dim][hidden]`,
/// `out` `[hidden][ple_dim]` and the norm on its output.
pub const PleWeights = struct {
    gate: Weight,
    out: Weight,
    norm: Norm,
};

/// AltUp and Laurel parameters of a Gemma 3n layer.
pub const AltUpWeights = struct {
    /// `[altup_inputs][hidden]` modality router and its norm.
    router: Weight,
    router_norm: Norm,
    /// `[altup_inputs²][altup_inputs]` prediction and `[altup_inputs][altup_inputs]` correction coefficients.
    predict: Weight,
    correct: Weight,
    /// `[hidden]` scale of the corrected active stream (null when `altup_correct_scale` is off).
    scale: ?[]const f32,
    /// Laurel: `[rank][hidden]`, `[hidden][rank]` and the norm on the result.
    laurel_l: Weight,
    laurel_r: Weight,
    laurel_norm: Norm,
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
    /// BitNet sub-layer norms: on the attention output before `o` and on the
    /// activated MLP intermediate before `down`.
    attn_sub_norm: ?Norm = null,
    ffn_sub_norm: ?Norm = null,
    /// xIELU activation parameters `(alpha_p, alpha_n)` after softplus (Apertus).
    xielu: ?[2]f32 = null,
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
    /// Gate on the attention output, before `o`: `[heads * v_head_dim][hidden]`
    /// (Kimi K3 MLA layers, AFMoE) or `[heads][hidden]` (Laguna, per head).
    attn_gate: ?Weight = null,
    /// Attention Residual scorers of this layer (Kimi K3; null otherwise).
    attn_res: ?AttnResLayer = null,
    mla: ?MlaWeights,
    /// Gated DeltaNet weights (null for full-attention layers).
    linear: ?LinearWeights = null,
    /// Mamba block weights (null for layers without one).
    ssm: ?SsmWeights = null,
    /// Short-convolution weights (LFM2 conv layers).
    conv: ?ConvWeights = null,
    /// Per-layer input embedding block (Gemma 3n / 4).
    ple: ?PleWeights = null,
    /// AltUp / Laurel parameters (Gemma 3n).
    altup: ?AltUpWeights = null,
    /// Scalar on the layer output (Gemma 4 `layer_scalar`).
    layer_scale: ?f32 = null,
    /// DeepSeek V4 / V4.1 weights (hyper-connections, low-rank and grouped
    /// projections, compressor, engram); null for other families.
    dsv4: ?dsv4.LayerWeights = null,
    /// Hyper-connection sites around this layer's blocks (`hc_mult > 1`).
    hyper: ?hyper.LayerWeights = null,
    /// Qwen4-Exp per-layer n-gram embedding block (null on other layers).
    ngram: ?qwen4.NgramWeights = null,
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
    /// Delta on `ssm.out` (parallel layout only; otherwise the Mamba output projection is `o`).
    ssm_out_delta: ?Delta = null,

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
            .attn_gate => self.attn_gate = w,
            .lin_g => self.linear.?.g = w,
            .lin_qkvz => self.linear.?.qkvz = w,
            .lin_qkv => self.linear.?.qkv = w,
            .lin_z => self.linear.?.z = w,
            .lin_b => self.linear.?.b = w,
            .lin_a => self.linear.?.a = w,
            .lin_ba => self.linear.?.ba = w,
            .lin_conv => self.linear.?.conv = w,
            .ssm_in => self.ssm.?.in_proj = w,
            .ssm_x => self.ssm.?.x_proj = w,
            .ssm_dt => self.ssm.?.dt_proj = w,
            .ssm_out => self.ssm.?.out = w,
            .conv_in => self.conv.?.in = w,
            .conv_kernel => self.conv.?.kernel = w,
            .ple_gate => self.ple.?.gate = w,
            .ple_out => self.ple.?.out = w,
            .laurel_l => self.altup.?.laurel_l = w,
            .laurel_r => self.altup.?.laurel_r = w,
            .altup_router => self.altup.?.router = w,
            .altup_predict => self.altup.?.predict = w,
            .altup_correct => self.altup.?.correct = w,
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
            .light_qkv => self.linear.?.light_qkv = w,
            .light_gate => self.linear.?.light_gate = w,
            .d_q_a => self.dsv4.?.q_a = w,
            .d_q_b => self.dsv4.?.q_b = w,
            .d_kv => self.dsv4.?.kv = w,
            .d_o_a => self.dsv4.?.o_a = w,
            .d_comp_kv => self.dsv4.?.compressor.?.kv = w,
            .d_comp_gate => self.dsv4.?.compressor.?.gate = w,
            .d_engram_wkv => self.dsv4.?.engram.?.wkv = w,
            .hc_attn_fn => self.hyper.?.attn.mhc.fn_w = w,
            .hc_attn_down => self.hyper.?.attn.gated.down = w,
            .hc_attn_up => self.hyper.?.attn.gated.up = w,
            .hc_attn_inject => self.hyper.?.attn.gated.inject = w,
            .hc_ffn_fn => self.hyper.?.ffn.mhc.fn_w = w,
            .hc_ffn_down => self.hyper.?.ffn.gated.down = w,
            .hc_ffn_up => self.hyper.?.ffn.gated.up = w,
            .hc_ffn_inject => self.hyper.?.ffn.gated.inject = w,
            .ple_key => self.ngram.?.key = w,
            .ple_value => self.ngram.?.value = w,
            .ple_conv => self.ngram.?.conv = w,
        }
    }
};

/// One Attention Residual aggregation point (Kimi K3): the score of a
/// candidate row `r` is `rmsnorm(r; norm) · proj`, i.e. `(r · cw) / rms(r)`
/// with `cw = norm ⊙ proj`; the rows (the banked block prefixes and the
/// running prefix) are mixed by the softmax of their scores.
pub const AttnResScorer = struct {
    norm: []const f32,
    /// The `[1][hidden]` projection as a vector.
    proj: []const f32,
};

/// The two aggregation points of a layer: before the attention (its result
/// feeds `input_norm`) and before the MLP (feeding `pre_ff_norm`).
pub const AttnResLayer = struct {
    attn: AttnResScorer,
    mlp: AttnResScorer,
};

/// Cosine / sine table of one rotary embedding: `[len][half]` with `dim = 2 * half` rotated coordinates.
pub const RopeTable = struct {
    cos: []f32,
    sin: []f32,
    half: usize,
    dim: usize,
    len: usize,
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
/// Loads a matrix of exactly `rows` x `cols`, registering it in `layer.refs`
/// under `slot`; an error names the tensor and the shapes when it does not
/// match, so a mis-declared config fails at load instead of silently running.
pub fn loadMatChecked(model: *Model, layer: *Layer, slot: Slot, name: []const u8, rows: usize, cols: usize) !Weight {
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

/// Loads a vector of exactly `len` entries (norms, biases and scales).
pub fn loadVecChecked(model: *Model, name: []const u8, len: usize) ![]const f32 {
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
    /// Sinusoidal position table `[positions][hidden]` (XGLM), computed on load.
    sin_pos: []f32,
    /// LayerNorm on the embeddings (BLOOM).
    embed_norm: ?Norm,
    /// Per-layer input embeddings (Gemma 3n / 4): the packed
    /// `[ple_vocab][layers * ple_dim]` table, the `[layers * ple_dim][hidden]`
    /// projection of the token embedding and the `[ple_dim]` norm on it.
    ple_embed_ref: ?WeightRef,
    ple_proj_ref: ?WeightRef,
    ple_proj_norm: ?Norm,
    /// AltUp (Gemma 3n): `[hidden][hidden]` projections that expand the
    /// embedding into the extra residual streams and fold them back.
    altup_proj_refs: []WeightRef,
    altup_unembed_refs: []WeightRef,
    largest_layer_bytes: u64,
    /// Largest layer without its routed experts (attention, norms, router, shared expert).
    largest_trunk_layer_bytes: u64,
    /// Largest routed expert (0 for dense models).
    largest_expert_bytes: u64,
    /// Bytes of all routed experts of all layers.
    total_expert_bytes: u64,
    largest_tensor_bytes: u64,
    spill_always: bool,
    /// Final norm before the LM head (null for a family without one).
    final_norm: ?Norm,
    /// Attention Residual: the output aggregation feeding `final_norm` (Kimi K3).
    output_res: ?AttnResScorer = null,
    layers: []Layer,
    eos_ids: []u32,
    pad_id: u32,
    /// Directory the model was loaded from.
    source_dir: []const u8,
    /// Raw JSON texts kept for export.
    config_json: []const u8,
    /// `config_json` as an export writes it: without `quantization_config`
    /// when the checkpoint was dequantised on load (exports are bf16).
    export_config_json: []const u8,
    /// Tensors dequantised on load (see dequant.zig); 0 for plain checkpoints.
    dequantised: usize,
    tokenizer_json: []const u8,
    generation_config_json: ?[]const u8,
    tokenizer_config_json: ?[]const u8,
    chat_template: ?[]const u8,
    dtype: tensor.DType,
    /// Rotary tables of the global layers and (Gemma 3 family) the sliding
    /// layers; `rope_local` aliases `rope` when there is no separate table.
    rope: RopeTable,
    rope_local: RopeTable,
    /// ALiBi slope per head (empty unless `config.positional == .alibi`).
    alibi_slopes: []f32,
    /// RoPE tables of the compressed branches (DeepSeek V4 / V4.1; empty otherwise).
    rope_cos_compress: []f32,
    rope_sin_compress: []f32,
    /// DeepSeek V4 final stream collapse and V4.1 engram hash state.
    /// Final stream collapse of a hyper-connection family (hyper.zig).
    hyper_head: hyper.Head,
    engram: ?dsv4.EngramState,
    /// Qwen4-Exp n-gram hash state (null for other families).
    ngram: ?qwen4.NgramState,

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
    pub fn loadWithOptions(gpa: Allocator, io: Io, pool: *const tensor.Pool, dir_path: []const u8, options: LoadOptions) !*Model {
        var opts = options;
        // `DITCH_NO_MMAP=1` reads the weights instead of memory-mapping them.
        // The two paths produce bit-identical results (see the streamed
        // versus mapped test in stream_test.zig), so this only trades memory
        // for a syscall; it exists for environments whose mmap cannot serve
        // the mapping, such as qemu-user, which rejects the
        // `MAP_SHARED_VALIDATE` that Zig's `MemoryMap` asks for.
        if (tensor.mmapDisabled()) opts.store = .streamed;
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
        self.dequantised = 0;
        if (gguf_path) |p| {
            try gguf_model.attach(self, p, opts.store == .mapped, .{ .ignore_embedded = opts.gguf_ignore_embedded });
            self.export_config_json = self.config_json;
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

    /// Loads a Mamba block (Mamba2 or Mamba1) of layer `li`: the input
    /// projection, the causal convolution and the small time-step / skip
    /// vectors as f32, the gated norm and the output projection, which goes
    /// to `layer.o` unless the layer also has attention (`separate_out`).
    fn loadSsm(self: *Model, layer: *Layer, arena: Allocator, li: usize, lp: []const u8, separate_out: bool) !void {
        const c = &self.config;
        const d = &c.ssm;
        const hidden = c.hidden_size;
        const sp = try cat(arena, lp, c.arch.names.ssm orelse return error.InvalidConfig);
        const conv_dim = ssmConvDim(c);
        const in_name = try cat(arena, sp, "in_proj.weight");
        var s = SsmWeights{
            .in_proj = try self.loadMat(in_name),
            .in_bias = self.loadVecOpt(try biasName(arena, in_name)),
            .conv = try self.loadVec(try cat(arena, sp, "conv1d.weight")),
            .conv_bias = self.loadVecOpt(try cat(arena, sp, "conv1d.bias")),
            .dt_bias = &.{},
            .a = &.{},
            .d = try self.loadVec(try cat(arena, sp, "D")),
            .norm = null,
        };
        layer.refs.add(.ssm_in, try self.ref(in_name), false);
        const want_in: usize = switch (d.kind) {
            .mamba2 => d.inter + conv_dim + d.heads,
            .mamba1 => 2 * d.inter,
            .none => unreachable,
        };
        if (s.in_proj.rows != want_in or s.in_proj.cols != hidden) {
            std.log.err("layer {d}: Mamba in_proj is [{d}][{d}], expected [{d}][{d}]", .{ li, s.in_proj.rows, s.in_proj.cols, want_in, hidden });
            return error.InvalidConfig;
        }
        if (s.conv.len != conv_dim * d.conv_kernel or (s.conv_bias != null and s.conv_bias.?.len != conv_dim)) {
            std.log.err("layer {d}: Mamba conv1d has {d} weights, expected [{d}][{d}]", .{ li, s.conv.len, conv_dim, d.conv_kernel });
            return error.InvalidConfig;
        }
        const a_log = try self.loadVec(try cat(arena, sp, "A_log"));
        switch (d.kind) {
            .mamba2 => {
                s.dt_bias = try self.loadVec(try cat(arena, sp, "dt_bias"));
                if (d.rms_norm) s.norm = try self.loadVec(try cat(arena, sp, "norm.weight"));
                if (s.dt_bias.len != d.heads or a_log.len != d.heads or s.d.len != d.heads or (s.norm != null and s.norm.?.len != d.inter)) {
                    std.log.err("layer {d}: Mamba2 per-head vectors do not match {d} heads", .{ li, d.heads });
                    return error.InvalidConfig;
                }
            },
            .mamba1 => {
                const x_name = try cat(arena, sp, "x_proj.weight");
                const dt_name = try cat(arena, sp, "dt_proj.weight");
                s.x_proj = try self.loadMat(x_name);
                s.dt_proj = try self.loadMat(dt_name);
                layer.refs.add(.ssm_x, try self.ref(x_name), false);
                layer.refs.add(.ssm_dt, try self.ref(dt_name), false);
                s.dt_bias = try self.loadVec(try biasName(arena, dt_name));
                s.dt_norm = self.loadVecOpt(try cat(arena, sp, "dt_layernorm.weight"));
                s.b_norm = self.loadVecOpt(try cat(arena, sp, "b_layernorm.weight"));
                s.c_norm = self.loadVecOpt(try cat(arena, sp, "c_layernorm.weight"));
                if (s.x_proj.?.rows != d.dt_rank + 2 * d.state or s.x_proj.?.cols != d.inter or s.dt_proj.?.rows != d.inter or s.dt_proj.?.cols != d.dt_rank or s.dt_bias.len != d.inter or a_log.len != d.inter * d.state or s.d.len != d.inter) {
                    std.log.err("layer {d}: Mamba1 projections do not match inter {d}, state {d}, dt_rank {d}", .{ li, d.inter, d.state, d.dt_rank });
                    return error.InvalidConfig;
                }
            },
            .none => unreachable,
        }
        const a = try arena.alloc(f32, a_log.len);
        for (a, a_log) |*dst, v| dst.* = -@exp(v);
        s.a = a;
        const o_name = try cat(arena, sp, "out_proj.weight");
        const ow = try self.loadMat(o_name);
        if (ow.rows != hidden or ow.cols != d.inter) {
            std.log.err("layer {d}: Mamba out_proj is [{d}][{d}], expected [{d}][{d}]", .{ li, ow.rows, ow.cols, hidden, d.inter });
            return error.InvalidConfig;
        }
        if (separate_out) {
            s.out = ow;
            s.out_bias = self.loadVecOpt(try biasName(arena, o_name));
            layer.refs.add(.ssm_out, try self.ref(o_name), false);
        } else {
            layer.o = ow;
            layer.o_bias = self.loadVecOpt(try biasName(arena, o_name));
            layer.refs.add(.o, try self.ref(o_name), false);
        }
        layer.ssm = s;
    }

    /// Loads an LFM2 short-convolution layer: `in_proj` `[3 hidden][hidden]`,
    /// the depthwise kernel `[hidden][taps]` and `out_proj` into `layer.o`.
    fn loadConv(self: *Model, layer: *Layer, arena: Allocator, li: usize, lp: []const u8) !void {
        const c = &self.config;
        const names = &c.arch.names;
        const hidden = c.hidden_size;
        const in_name = try cat(arena, lp, names.conv_in orelse return error.InvalidConfig);
        const kernel_name = try cat(arena, lp, names.conv_kernel orelse return error.InvalidConfig);
        const out_name = try cat(arena, lp, names.conv_out orelse return error.InvalidConfig);
        const cw = ConvWeights{
            .in = try self.loadMat(in_name),
            .in_bias = self.loadVecOpt(try biasName(arena, in_name)),
            .kernel = try self.loadMat(kernel_name),
            .kernel_bias = self.loadVecOpt(try biasName(arena, kernel_name)),
        };
        layer.refs.add(.conv_in, try self.ref(in_name), false);
        layer.refs.add(.conv_kernel, try self.ref(kernel_name), false);
        if (cw.in.rows != 3 * hidden or cw.in.cols != hidden) {
            std.log.err("layer {d}: conv in_proj is [{d}][{d}], expected [{d}][{d}]", .{ li, cw.in.rows, cw.in.cols, 3 * hidden, hidden });
            return error.InvalidConfig;
        }
        if (cw.kernel.rows != hidden or cw.kernel.cols != c.conv_kernel) {
            std.log.err("layer {d}: conv kernel is [{d}][{d}], expected [{d}][{d}] (conv_L_cache)", .{ li, cw.kernel.rows, cw.kernel.cols, hidden, c.conv_kernel });
            return error.InvalidConfig;
        }
        if (cw.in_bias) |b| if (b.len != 3 * hidden) return error.InvalidConfig;
        if (cw.kernel_bias) |b| if (b.len != hidden) return error.InvalidConfig;
        layer.o = try self.loadMat(out_name);
        layer.o_bias = self.loadVecOpt(try biasName(arena, out_name));
        layer.refs.add(.o, try self.ref(out_name), false);
        if (layer.o.rows != hidden or layer.o.cols != hidden) {
            std.log.err("layer {d}: conv out_proj is [{d}][{d}], expected [{d}][{d}]", .{ li, layer.o.rows, layer.o.cols, hidden, hidden });
            return error.InvalidConfig;
        }
        layer.conv = cw;
    }

    /// Loads the per-layer input embedding block of a layer (Gemma 3n / 4).
    fn loadPle(self: *Model, layer: *Layer, arena: Allocator, li: usize, lp: []const u8) !void {
        const c = &self.config;
        const names = &c.arch.names;
        const gate_name = try cat(arena, lp, names.ple_gate orelse return error.InvalidConfig);
        const out_name = try cat(arena, lp, names.ple_out orelse return error.InvalidConfig);
        const ple = PleWeights{
            .gate = try self.loadMat(gate_name),
            .out = try self.loadMat(out_name),
            .norm = try self.loadNorm(try cat(arena, lp, names.ple_norm orelse return error.InvalidConfig)),
        };
        layer.refs.add(.ple_gate, try self.ref(gate_name), false);
        layer.refs.add(.ple_out, try self.ref(out_name), false);
        if (ple.gate.rows != c.ple_dim or ple.gate.cols != c.hidden_size or ple.out.rows != c.hidden_size or ple.out.cols != c.ple_dim) {
            std.log.err("layer {d}: per-layer input projections do not match hidden_size_per_layer_input {d}", .{ li, c.ple_dim });
            return error.InvalidConfig;
        }
        layer.ple = ple;
    }

    /// Loads the AltUp and Laurel parameters of a layer (Gemma 3n).
    fn loadAltUp(self: *Model, layer: *Layer, arena: Allocator, li: usize, lp: []const u8) !void {
        const c = &self.config;
        const names = &c.arch.names;
        const a = c.altup_inputs;
        const router_name = try cat(arena, lp, names.altup_router orelse return error.InvalidConfig);
        const predict_name = try cat(arena, lp, names.altup_predict orelse return error.InvalidConfig);
        const correct_name = try cat(arena, lp, names.altup_correct orelse return error.InvalidConfig);
        const ll_name = try cat(arena, lp, names.laurel_l orelse return error.InvalidConfig);
        const lr_name = try cat(arena, lp, names.laurel_r orelse return error.InvalidConfig);
        const alt = AltUpWeights{
            .router = try self.loadMat(router_name),
            .router_norm = try self.loadNorm(try cat(arena, lp, names.altup_router_norm orelse return error.InvalidConfig)),
            .predict = try self.loadMat(predict_name),
            .correct = try self.loadMat(correct_name),
            .scale = if (c.altup_correct_scale) try self.loadVec(try cat(arena, lp, names.altup_scale orelse return error.InvalidConfig)) else null,
            .laurel_l = try self.loadMat(ll_name),
            .laurel_r = try self.loadMat(lr_name),
            .laurel_norm = try self.loadNorm(try cat(arena, lp, names.laurel_norm orelse return error.InvalidConfig)),
        };
        layer.refs.add(.altup_router, try self.ref(router_name), false);
        layer.refs.add(.altup_predict, try self.ref(predict_name), false);
        layer.refs.add(.altup_correct, try self.ref(correct_name), false);
        layer.refs.add(.laurel_l, try self.ref(ll_name), false);
        layer.refs.add(.laurel_r, try self.ref(lr_name), false);
        const hidden = c.hidden_size;
        if (alt.router.rows != a or alt.router.cols != hidden or alt.predict.rows != a * a or alt.predict.cols != a or alt.correct.rows != a or alt.correct.cols != a) {
            std.log.err("layer {d}: AltUp coefficient shapes do not match altup_num_inputs {d}", .{ li, a });
            return error.InvalidConfig;
        }
        if (alt.laurel_l.rows != c.laurel_rank or alt.laurel_l.cols != hidden or alt.laurel_r.rows != hidden or alt.laurel_r.cols != c.laurel_rank) {
            std.log.err("layer {d}: Laurel projections do not match laurel_rank {d}", .{ li, c.laurel_rank });
            return error.InvalidConfig;
        }
        if (alt.scale) |s| if (s.len != hidden) return error.InvalidConfig;
        layer.altup = alt;
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
        const proj_names = .{ names.lin_q, names.lin_k, names.lin_v, names.lin_b, names.lin_g_a, names.lin_g_b, names.lin_g };
        const proj_slots = .{ Slot.lin_q, Slot.lin_k, Slot.lin_v, Slot.lin_b, Slot.lin_g_a, Slot.lin_g_b, Slot.lin_g };
        const proj_rows = .{ dim, dim, dim, heads, hd, dim, dim };
        const proj_cols = .{ hidden, hidden, hidden, hidden, hidden, hd, hidden };
        inline for (proj_names, proj_slots, proj_rows, proj_cols) |t, slot, rows, cols| {
            // The output gate is the full-rank `g` or the low-rank `g_a`/`g_b` pair.
            const wanted = switch (slot) {
                .lin_g => c.linear_full_rank_gate,
                .lin_g_a, .lin_g_b => !c.linear_full_rank_gate,
                else => true,
            };
            if (wanted) {
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
                    .lin_g => lin.g = w,
                    else => unreachable,
                }
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
        // Kimi K3 checkpoints store A_log with `head_dim` entries of which the
        // first `num_heads` are the per-head decays (what every loader reads).
        if (lin.a_log.len > heads) lin.a_log = lin.a_log[0..heads];
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

    /// Loads a MiniMax lightning-attention layer: the fused `[heads][q|k|v]`
    /// projection, the sigmoid output gate, the RMSNorm over `heads * head_dim`
    /// and the output projection into `layer.o`.
    fn loadLightning(self: *Model, layer: *Layer, arena: Allocator, li: usize, lp: []const u8) !void {
        const c = &self.config;
        const names = &c.arch.names;
        const hidden = c.hidden_size;
        const qd = c.num_heads * c.head_dim;
        var lin = LinearWeights{};
        const qkv_name = try cat(arena, lp, names.light_qkv orelse return error.InvalidConfig);
        lin.light_qkv = try self.loadMat(qkv_name);
        layer.refs.add(.light_qkv, try self.ref(qkv_name), false);
        if (lin.light_qkv.?.rows != 3 * qd or lin.light_qkv.?.cols != hidden) {
            std.log.err("layer {d}: lightning qkv projection is [{d}][{d}], expected [{d}][{d}]", .{ li, lin.light_qkv.?.rows, lin.light_qkv.?.cols, 3 * qd, hidden });
            return error.InvalidConfig;
        }
        const gate_name = try cat(arena, lp, names.light_gate orelse return error.InvalidConfig);
        lin.light_gate = try self.loadMat(gate_name);
        layer.refs.add(.light_gate, try self.ref(gate_name), false);
        if (lin.light_gate.?.rows != qd or lin.light_gate.?.cols != hidden) {
            std.log.err("layer {d}: lightning output gate has wrong shape", .{li});
            return error.InvalidConfig;
        }
        lin.norm = try self.loadVec(try cat(arena, lp, names.light_norm orelse return error.InvalidConfig));
        if (lin.norm.len != qd) {
            std.log.err("layer {d}: lightning norm has {d} entries, expected {d}", .{ li, lin.norm.len, qd });
            return error.InvalidConfig;
        }
        const o_name = try cat(arena, lp, names.light_out orelse return error.InvalidConfig);
        layer.o = try self.loadMat(o_name);
        layer.o_bias = self.loadVecOpt(try biasName(arena, o_name));
        layer.refs.add(.o, try self.ref(o_name), false);
        if (layer.o.rows != hidden or layer.o.cols != qd) {
            std.log.err("layer {d}: lightning output projection is [{d}][{d}], expected [{d}][{d}]", .{ li, layer.o.rows, layer.o.cols, hidden, qd });
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
        self.sin_pos = &.{};
        if (c.positional == .sinusoidal) {
            // fairseq / XGLM: `[sin(p * f_i) | cos(p * f_i)]`, `f_i = 10000^(-i / (half - 1))`,
            // indexed at `position + position_offset`.
            const hidden = c.hidden_size;
            const half = hidden / 2;
            const rows = @min(c.max_position_embeddings, 8192) + c.position_offset;
            self.sin_pos = try arena.alloc(f32, rows * hidden);
            @memset(self.sin_pos, 0);
            for (0..rows) |p| {
                const row = self.sin_pos[p * hidden ..][0..hidden];
                for (0..half) |i| {
                    const freq = @exp(-@log(10000.0) * @as(f64, @floatFromInt(i)) / @as(f64, @floatFromInt(@max(half - 1, 1))));
                    const ang = @as(f64, @floatFromInt(p)) * freq;
                    row[i] = @floatCast(@sin(ang));
                    row[half + i] = @floatCast(@cos(ang));
                }
            }
        }
        const nonparametric = c.norm == .none or c.norm == .rms_none;
        self.embed_norm = if (names.embed_norm) |t| (if (nonparametric) Norm{ .w = &.{} } else try self.loadNormOpt(try self.name(t))) else null;
        // Qwen4-Exp has no final norm: its stream mixer already normalises.
        self.final_norm = if (names.final_norm) |t|
            (if (nonparametric) Norm{ .w = &.{} } else try self.loadNorm(try self.name(t)))
        else
            null;
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

        // Per-layer input embeddings and AltUp projections (Gemma 3n / 4).
        self.ple_embed_ref = null;
        self.ple_proj_ref = null;
        self.ple_proj_norm = null;
        if (c.ple_dim > 0) {
            const emb = try self.ref(try self.name(names.ple_embed orelse return error.InvalidConfig));
            const proj = try self.ref(try self.name(names.ple_proj orelse return error.InvalidConfig));
            if (emb.cols != c.num_layers * c.ple_dim or proj.rows != c.num_layers * c.ple_dim or proj.cols != c.hidden_size) {
                std.log.err("per-layer embedding tensors do not match {d} layers x hidden_size_per_layer_input {d}", .{ c.num_layers, c.ple_dim });
                return error.InvalidConfig;
            }
            self.ple_embed_ref = emb;
            self.ple_proj_ref = proj;
            self.ple_proj_norm = try self.loadNorm(try self.name(names.ple_proj_norm orelse return error.InvalidConfig));
            if (self.ple_proj_norm.?.w.len != c.ple_dim) return error.InvalidConfig;
        }
        self.altup_proj_refs = &.{};
        self.altup_unembed_refs = &.{};
        if (c.altup_inputs > 1) {
            const n_extra = c.altup_inputs - 1;
            self.altup_proj_refs = try arena.alloc(WeightRef, n_extra);
            self.altup_unembed_refs = try arena.alloc(WeightRef, n_extra);
            for (0..n_extra) |e| {
                self.altup_proj_refs[e] = try self.ref(try resolveName(arena, names.altup_proj orelse return error.InvalidConfig, self.prefix, 0, e));
                self.altup_unembed_refs[e] = try self.ref(try resolveName(arena, names.altup_unembed orelse return error.InvalidConfig, self.prefix, 0, e));
                for ([_]WeightRef{ self.altup_proj_refs[e], self.altup_unembed_refs[e] }) |r| {
                    if (r.rows != c.hidden_size or r.cols != c.hidden_size) return error.InvalidConfig;
                }
            }
        }

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
                .post_ff_norm = try self.normSlot(lp, names.post_ff_norm, !names.post_ff_norm_optional),
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
                .sinks = if (names.sinks) |t| (self.loadVecOpt(try cat(arena, lp, t)) orelse if (names.sinks_alt) |alt| self.loadVecOpt(try cat(arena, lp, alt)) else null) else null,
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
            if (names.attn_sub_norm) |t| layer.attn_sub_norm = try self.loadNormOpt(try cat(arena, lp, t));
            if (names.ffn_sub_norm) |t| layer.ffn_sub_norm = try self.loadNormOpt(try cat(arena, lp, t));
            if (names.xielu_alpha_p) |ap| {
                // Stored pre-softplus: `alpha_p = softplus(p)`, `alpha_n = beta + softplus(n)` (beta 0.5).
                const p = try self.loadVec(try cat(arena, lp, ap));
                const n = try self.loadVec(try cat(arena, lp, names.xielu_alpha_n orelse return error.InvalidConfig));
                if (p.len != 1 or n.len != 1) return error.InvalidConfig;
                layer.xielu = .{ softplus(p[0]), 0.5 + softplus(n[0]) };
            }
            if (nonparametric) {
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
            if (c.sinks and layer.sinks == null and (!c.sinks_sliding_only or c.sliding_layers[i])) {
                std.log.err("missing attention sinks in layer {d}", .{i});
                return error.MissingWeights;
            }
            if (layer.sinks) |sk| if (sk.len != c.layer_heads[i]) {
                std.log.err("layer {d}: attention sinks have {d} entries, expected one per head ({d})", .{ i, sk.len, c.layer_heads[i] });
                return error.InvalidConfig;
            };

            // Attention projections.
            layer.o = .{ .data = &.{}, .dtype = self.dtype, .rows = 0, .cols = 0 };
            const hd = c.layer_head_dim[i];
            const nkv = c.layer_kv_heads[i];
            const kv_shared = c.kvShared(i);
            if (c.hyper != null) try hyper.loadLayer(self, layer, arena, lp);
            try qwen4.loadLayer(self, layer, arena, i, lp);
            if (c.dsv4 != null) {
                try dsv4.loadLayer(self, layer, arena, i, lp);
            } else if (c.linear_layers[i]) {
                switch (c.linear_kind) {
                    .gated_deltanet => try self.loadLinear(layer, arena, i, lp),
                    .kda => try self.loadKda(layer, arena, i, lp),
                    .lightning => try self.loadLightning(layer, arena, i, lp),
                }
            } else if (c.conv_layers[i]) {
                try self.loadConv(layer, arena, i, lp);
            } else if (!c.attn_layers[i]) {
                // A Mamba, MLP or MoE block (loaded below).
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
                if (c.mla_output_gate) {
                    const g_name = try cat(arena, lp, names.attn_gate orelse return error.InvalidConfig);
                    layer.attn_gate = try self.loadMat(g_name);
                    layer.refs.add(.attn_gate, try self.ref(g_name), false);
                    if (layer.attn_gate.?.rows != c.num_heads * m.v_head_dim or layer.attn_gate.?.cols != c.hidden_size) {
                        std.log.err("layer {d}: attention output gate {s} is [{d}][{d}], expected [{d}][{d}]", .{ i, g_name, layer.attn_gate.?.rows, layer.attn_gate.?.cols, c.num_heads * m.v_head_dim, c.hidden_size });
                        return error.InvalidConfig;
                    }
                }
            } else if (c.qkv_layout != .separate or (c.qkv_alt != null and names.qkv != null and names.q != null and self.find(try cat(arena, lp, names.q.?)) == null and self.find(try cat(arena, lp, names.qkv.?)) != null)) {
                const qkv_name = try cat(arena, lp, names.qkv orelse return error.InvalidConfig);
                layer.qkv = try self.loadMatT(qkv_name, c.arch.conv1d);
                layer.qkv_bias = self.loadVecOpt(try biasName(arena, qkv_name));
                layer.refs.add(.qkv, try self.ref(qkv_name), c.arch.conv1d);
                const want = c.layer_heads[i] * hd + nkv * (hd + c.layerVDim(i));
                if (layer.qkv.?.rows != want or layer.qkv.?.cols != c.hidden_size) {
                    std.log.err("layer {d}: fused qkv tensor is [{d}][{d}], expected [{d}][{d}]", .{ i, layer.qkv.?.rows, layer.qkv.?.cols, want, c.hidden_size });
                    return error.InvalidConfig;
                }
                if (c.qkv_chunks != 0 and (c.layer_heads[i] % c.qkv_chunks != 0 or nkv % c.qkv_chunks != 0)) {
                    std.log.err("layer {d}: fused qkv tensor is chunked {d} ways, which does not divide {d} heads / {d} kv heads", .{ i, c.qkv_chunks, c.layer_heads[i], nkv });
                    return error.InvalidConfig;
                }
            } else {
                const q_name = try cat(arena, lp, names.q orelse return error.InvalidConfig);
                layer.q = try self.loadMat(q_name);
                layer.q_bias = self.loadVecOpt(try biasName(arena, q_name));
                layer.refs.add(.q, try self.ref(q_name), false);
                const nh = c.layer_heads[i];
                const qwant = if (c.gated_attention) 2 * nh * hd else nh * hd;
                if (layer.q.?.rows != qwant or layer.q.?.cols != c.hidden_size) {
                    std.log.err("layer {d}: q projection is [{d}][{d}], expected [{d}][{d}] (heads {d}, head_dim {d})", .{ i, layer.q.?.rows, layer.q.?.cols, qwant, c.hidden_size, nh, hd });
                    return error.InvalidConfig;
                }
                // A KV-shared layer (Gemma 3n / 4) has no key/value projections.
                if (!kv_shared) {
                    const k_name = try cat(arena, lp, names.k orelse return error.InvalidConfig);
                    const v_name = try cat(arena, lp, names.v orelse return error.InvalidConfig);
                    layer.k = try self.loadMat(k_name);
                    layer.k_bias = self.loadVecOpt(try biasName(arena, k_name));
                    layer.refs.add(.k, try self.ref(k_name), false);
                    if (c.k_eq_v and self.store.lookup(v_name) == null) {
                        // Keys double as values (Gemma 4 global layers).
                        layer.v = null;
                    } else {
                        layer.v = try self.loadMat(v_name);
                        layer.v_bias = self.loadVecOpt(try biasName(arena, v_name));
                        layer.refs.add(.v, try self.ref(v_name), false);
                        if (layer.v.?.rows != nkv * c.layerVDim(i) or layer.v.?.cols != c.hidden_size) {
                            std.log.err("layer {d}: v projection is [{d}][{d}], expected [{d}][{d}]", .{ i, layer.v.?.rows, layer.v.?.cols, nkv * c.layerVDim(i), c.hidden_size });
                            return error.InvalidConfig;
                        }
                    }
                    if (layer.k.?.rows != nkv * hd or layer.k.?.cols != c.hidden_size) {
                        std.log.err("layer {d}: k projection is [{d}][{d}], expected [{d}][{d}] (kv heads {d}, head_dim {d})", .{ i, layer.k.?.rows, layer.k.?.cols, nkv * hd, c.hidden_size, nkv, hd });
                        return error.InvalidConfig;
                    }
                }
            }
            // Attention Residual scorers.
            if (c.attn_res_block > 0) {
                layer.attn_res = .{
                    .attn = .{
                        .norm = try self.loadVec(try self.requireFirst(arena, lp, names.attn_res_norm)),
                        .proj = try self.loadVec(try self.requireFirst(arena, lp, names.attn_res_proj)),
                    },
                    .mlp = .{
                        .norm = try self.loadVec(try self.requireFirst(arena, lp, names.mlp_res_norm)),
                        .proj = try self.loadVec(try self.requireFirst(arena, lp, names.mlp_res_proj)),
                    },
                };
                const ar = layer.attn_res.?;
                if (ar.attn.norm.len != c.hidden_size or ar.attn.proj.len != c.hidden_size or ar.mlp.norm.len != c.hidden_size or ar.mlp.proj.len != c.hidden_size) {
                    std.log.err("layer {d}: attention residual scorers must have hidden_size ({d}) entries", .{ i, c.hidden_size });
                    return error.InvalidConfig;
                }
            }

            if (kv_shared) {
                if (layer.k != null or layer.v != null or layer.qkv != null or layer.mla != null) {
                    std.log.err("layer {d}: KV sharing needs separate q/k/v projections", .{i});
                    return error.UnsupportedArchitecture;
                }
                layer.k_norm = null;
            }
            // AFMoE / Laguna gate the attention output from the layer input.
            if (c.attn_gate != .none and c.attn_layers[i]) {
                const g_name = try cat(arena, lp, names.attn_gate orelse return error.InvalidConfig);
                layer.attn_gate = try self.loadMat(g_name);
                layer.refs.add(.attn_gate, try self.ref(g_name), false);
                const g = layer.attn_gate.?;
                const nh = c.layer_heads[i];
                if (g.cols != c.hidden_size or (g.rows != nh * c.layerVDim(i) and g.rows != nh)) {
                    std.log.err("layer {d}: attention gate is [{d}][{d}], expected [{d} or {d}][{d}]", .{ i, g.rows, g.cols, nh * c.layerVDim(i), nh, c.hidden_size });
                    return error.InvalidConfig;
                }
            }
            if (c.attn_layers[i] and c.dsv4 == null) {
                const o_name = try cat(arena, lp, names.o);
                layer.o = try self.loadMatT(o_name, c.arch.conv1d);
                layer.o_bias = self.loadVecOpt(try biasName(arena, o_name));
                layer.refs.add(.o, try self.ref(o_name), c.arch.conv1d);
                const owant = c.layer_heads[i] * c.layerVDim(i);
                if (layer.o.rows != c.hidden_size or layer.o.cols != owant) {
                    std.log.err("layer {d}: output projection is [{d}][{d}], expected [{d}][{d}]", .{ i, layer.o.rows, layer.o.cols, c.hidden_size, owant });
                    return error.InvalidConfig;
                }
            }
            if (c.ssm_layers[i]) try self.loadSsm(layer, arena, i, lp, c.attn_layers[i]);
            if (c.ple_dim > 0) try self.loadPle(layer, arena, i, lp);
            if (c.altup_inputs > 0) try self.loadAltUp(layer, arena, i, lp);
            if (names.layer_scale) |t| {
                if (self.loadVecOpt(try cat(arena, lp, t))) |s| {
                    if (s.len != 1) return error.InvalidConfig;
                    layer.layer_scale = s[0];
                }
            }

            // MLP.
            if (!c.mlp_layers[i]) {
                // Single-block layer without an MLP (Mamba2, Nemotron-H).
            } else if (c.moe_layers[i]) {
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
                        // A family whose dense MLP is fused in one checkpoint
                        // can ship it split in another (MiniMax M3's released
                        // weights keep `gate_proj` / `up_proj`).
                        if (self.store.lookup(gu_name) == null and names.gate != null and names.up != null) {
                            const gate_name = try cat(arena, lp, names.gate.?);
                            const up_name = try cat(arena, lp, names.up.?);
                            layer.gate = try self.loadMat(gate_name);
                            layer.up = try self.loadMat(up_name);
                            layer.gate_bias = self.loadVecOpt(try biasName(arena, gate_name));
                            layer.up_bias = self.loadVecOpt(try biasName(arena, up_name));
                            layer.refs.add(.gate, try self.ref(gate_name), false);
                            layer.refs.add(.up, try self.ref(up_name), false);
                        } else {
                            layer.gate_up = try self.loadMat(gu_name);
                            layer.up_bias = self.loadVecOpt(try biasName(arena, gu_name));
                            layer.refs.add(.gate_up, try self.ref(gu_name), false);
                        }
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
                if (i == 0 and inter != c.intermediate_size and !c.intermediate_varies) {
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
        self.output_res = null;
        if (c.attn_res_block > 0) {
            if ((c.num_layers + c.attn_res_block - 1) / c.attn_res_block > max_attn_res_rows) {
                std.log.err("attn_res_block_size {d} banks more than {d} block prefixes", .{ c.attn_res_block, max_attn_res_rows });
                return error.UnsupportedArchitecture;
            }
            if (c.parallel_residual or c.residual_layout != .pre) {
                std.log.err("attention residuals are only implemented for the sequential pre-norm layout", .{});
                return error.UnsupportedArchitecture;
            }
            self.output_res = .{
                .norm = try self.loadVec(try self.requireFirst(arena, "", try self.namesOf(names.output_res_norm))),
                .proj = try self.loadVec(try self.requireFirst(arena, "", try self.namesOf(names.output_res_proj))),
            };
            if (self.output_res.?.norm.len != c.hidden_size or self.output_res.?.proj.len != c.hidden_size) {
                std.log.err("output attention residual scorer must have hidden_size ({d}) entries", .{c.hidden_size});
                return error.InvalidConfig;
            }
        }
        self.engram = null;
        self.ngram = null;
        if (c.dsv4 != null) try dsv4.loadModel(self, arena);
        if (c.ngram_ple) |ng| self.ngram = try qwen4.buildState(arena, ng);
        self.hyper_head = if (c.hyper != null) try hyper.loadModel(self, arena) else .none;
    }

    /// Resolves each model-level template of `templates` with the detected prefix.
    fn namesOf(self: *Model, templates: []const []const u8) ![]const []const u8 {
        const arena = self.arena.allocator();
        const out = try arena.alloc([]const u8, templates.len);
        for (templates, 0..) |t, i| out[i] = try self.name(t);
        return out;
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

    /// Makes the latent up projection of a latent-MoE layer resident; release with `DownLease.release`.
    pub fn acquireLatentUp(self: *const Model, layer: usize) !moe.DownLease {
        return self.layers[layer].moe.?.acquireLatentUp(self);
    }

    pub fn setLatentDelta(self: *Model, layer: usize, delta: Delta) void {
        const lat = &self.layers[layer].moe.?.latent.?;
        if (lat.up_delta) |d| {
            self.gpa.free(d.a);
            self.gpa.free(d.b);
        }
        lat.up_delta = delta;
    }

    pub fn getLatentDelta(self: *const Model, layer: usize) ?Delta {
        const m = &(self.layers[layer].moe orelse return null);
        const lat = m.latent orelse return null;
        return lat.up_delta;
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
        // tokenizer.json, else a tiktoken vocabulary (tiktoken.model / tokenizer.model).
        const loaded = try Tokenizer.loadDir(gpa, io, arena, dir, dir_path, self.tokenizer_config_json);
        self.tokenizer_json = loaded.json;
        self.tokenizer = loaded.tokenizer;
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
                // A Hugging Face hub snapshot directory is symbolic links into
                // `blobs/`, so a link that resolves to a file counts too.
                if (entry.kind != .file and entry.kind != .sym_link) continue;
                if (!std.mem.endsWith(u8, entry.name, ".safetensors")) continue;
                if (entry.kind == .sym_link) {
                    const st = dir.statFile(io, entry.name, .{}) catch continue;
                    if (st.kind != .file) continue;
                }
                try names.append(arena, try arena.dupe(u8, entry.name));
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

        // Quantised checkpoints: every recognised group of storage tensors
        // becomes one virtual bf16 tensor; anything left over is skipped.
        // DeepSeek V4 / V4.1 as released: DeepSeek's own tensor names.
        if (self.config.dsv4 != null) {
            const renamed = try @import("deepseek_v4.zig").renameNative(self.files);
            if (renamed > 0) std.log.info("read {d} tensors under DeepSeek's own names", .{renamed});
        }
        const reg = try dequant.register(gpa, io, self.files, self.config.quant);
        self.dequantised = reg.count;
        self.export_config_json = self.config_json;
        if (reg.count > 0) {
            std.log.info("dequantising {d} tensors on load ({d} fp8, {d} mxfp4, {d} pack-quantized); exports are bf16", .{ reg.count, reg.fp8, reg.mxfp4, reg.int_packed });
            self.export_config_json = try dequant.stripQuantizationConfig(arena, self.config_json);
        }
        for (self.files) |f| {
            var it = f.raw.iterator();
            while (it.next()) |kv| std.log.warn("skipping tensor {s} with unsupported dtype {s}", .{ kv.key_ptr.*, kv.value_ptr.dtype });
        }
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
        if (self.config.norm == .none or self.config.norm == .rms_none) return Norm{ .w = &.{} };
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
        // At least 512 rows, so that a model with a tiny declared context
        // still rotates the positions ditch scores exactly.
        const len = @min(@max(c.max_position_embeddings, 512), 8192);
        self.rope = try self.fillRope(len, c.rope_theta, c.rotary_dim, c.rope_freq_dim, c.rope_scaling);
        self.rope_local = if (c.rope_local) |l| try self.fillRope(len, l.theta, l.rotary_dim, l.freq_dim, .none) else self.rope;
        self.rope_cos_compress = &.{};
        self.rope_sin_compress = &.{};
        if (c.dsv4) |d| {
            const t = try self.fillRope(len, d.compress_rope_theta, c.rotary_dim, c.rotary_dim, d.compress_rope_scaling);
            self.rope_cos_compress = t.cos;
            self.rope_sin_compress = t.sin;
        }
    }

    /// Builds a `[len][dim / 2]` table: `dim` coordinates rotate with
    /// frequencies `theta^(-2i / freq_dim)` (`freq_dim == dim` except for
    /// proportional RoPE, which derives them from the whole head).
    fn fillRope(self: *Model, len: usize, theta: f32, dim: usize, freq_dim: usize, scaling: RopeScaling) !RopeTable {
        const arena = self.arena.allocator();
        const half = dim / 2;
        const cos = try arena.alloc(f32, len * half);
        const sin = try arena.alloc(f32, len * half);
        const table = RopeTable{ .cos = cos, .sin = sin, .half = half, .dim = 2 * half, .len = len };
        if (half == 0) return table;
        const inv_freq = try self.gpa.alloc(f64, half);
        defer self.gpa.free(inv_freq);
        for (inv_freq, 0..) |*f, i| {
            const exponent: f64 = @as(f64, @floatFromInt(2 * i)) / @as(f64, @floatFromInt(freq_dim));
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
        while (pos < len) : (pos += 1) {
            for (inv_freq, 0..) |f, i| {
                const angle = @as(f64, @floatFromInt(pos)) * f;
                cos[pos * half + i] = @floatCast(@cos(angle) * attention_factor);
                sin[pos * half + i] = @floatCast(@sin(angle) * attention_factor);
            }
        }
        if (self.config.rope_reverse) {
            for (sin) |*v| v.* = -v.*;
        }
        return table;
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
        if (names.ssm) |sp| {
            if (std.mem.startsWith(u8, ls.suffix, sp) and std.mem.eql(u8, ls.suffix[sp.len..], "out_proj.weight")) {
                const s = layer.ssm orelse return null;
                return wholeEdit(if (s.out != null) layer.ssm_out_delta else layer.o_delta, false);
            }
        }
        if (names.conv_out) |co| if (std.mem.eql(u8, ls.suffix, co)) return wholeEdit(layer.o_delta, false);
        if (names.light_out) |lo| if (std.mem.eql(u8, ls.suffix, lo)) return wholeEdit(layer.o_delta, false);
        if (layer.moe) |*m| {
            const target = m.exportTarget(ls.suffix) orelse return null;
            return switch (target) {
                .expert => |e| wholeEdit(m.getDownDelta(e), false),
                .fused_down => if (m.anyExpertDelta()) .{ .fused_down = ls.layer } else null,
                .latent_up => wholeEdit(m.latent.?.up_delta, false),
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
            if (l.ssm_out_delta) |d| {
                self.gpa.free(d.a);
                self.gpa.free(d.b);
                l.ssm_out_delta = null;
            }
            if (l.moe) |*m| m.resetDeltas(self.gpa);
        }
    }

    /// Whether layer `layer` has the component at all: single-block layers
    /// (Mamba2, Nemotron-H) hold either an attention / Mamba block or an MLP.
    pub fn hasComponent(self: *const Model, layer: usize, comp: Component) bool {
        const l = &self.layers[layer];
        return switch (comp) {
            .attn_o_proj => l.refs.get(.o) != null,
            .mlp_down_proj => l.moe != null or l.down != null,
        };
    }

    /// Whether layer `layer` has a Mamba output projection separate from its
    /// attention output (parallel layout); it is edited together with
    /// `.attn_o_proj`.
    pub fn hasSsmOut(self: *const Model, layer: usize) bool {
        return self.layers[layer].refs.get(.ssm_out) != null;
    }

    /// Makes the separate Mamba output projection resident; release with `self.store.release`.
    pub fn acquireSsmOut(self: *const Model, layer: usize) !stream.Lease {
        const store: *stream.WeightStore = @constCast(&self.store);
        const r = self.layers[layer].refs.get(.ssm_out) orelse return error.NotDenseLayer;
        return store.acquire(r);
    }

    pub fn setSsmOutDelta(self: *Model, layer: usize, delta: Delta) void {
        const l = &self.layers[layer];
        if (l.ssm_out_delta) |d| {
            self.gpa.free(d.a);
            self.gpa.free(d.b);
        }
        l.ssm_out_delta = delta;
    }

    pub fn getSsmOutDelta(self: *const Model, layer: usize) ?Delta {
        return self.layers[layer].ssm_out_delta;
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
    /// Compressed-KV entries and engram look-back (DeepSeek V4 / V4.1).
    compress: ?dsv4.CompressCache = null,
    /// Qwen4-Exp n-gram look-back and PLE convolution state.
    ngram: ?qwen4.NgramCache = null,

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
        const kvd = c.kvDim();
        var use_scratch = model.spill_always;
        if (model.budget) |b| {
            const need = bytesFor(c.num_layers, batch, max_len, kvd) + LinearCache.bytesFor(c, batch) + qwen4.NgramCache.bytesFor(c, batch) + model.residentWeightNeed();
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
        if (c.hasRecurrent()) cache.linear = try LinearCache.init(gpa, c, batch);
        if (c.dsv4 != null) cache.compress = try dsv4.CompressCache.init(gpa, c, batch, max_len);
        if (c.ngram_ple != null) cache.ngram = try qwen4.NgramCache.init(gpa, c, batch);
        return cache;
    }

    pub fn deinit(self: *KvCache) void {
        self.gpa.free(self.k);
        self.gpa.free(self.v);
        if (self.scratch) |*s| s.deinit();
        if (self.layer_written.len > 0) self.gpa.free(self.layer_written);
        if (self.linear) |*l| l.deinit();
        if (self.compress) |*cc| cc.deinit();
        if (self.ngram) |*nc| nc.deinit();
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

/// Recurrent state of the sequence-mixing layers that carry state between
/// tokens: the linear-attention layers (Gated DeltaNet and KDA: a
/// `[v_heads][k_dim][v_dim]` f32 state plus the causal-convolution history;
/// lightning attention: a `[heads][head_dim][head_dim]` state) and LFM2
/// short-convolution layers (the `[hidden][taps - 1]` history). One
/// slot per (recurrent layer, batch slot) with the next expected position.
/// Slots reset when a sequence restarts at position 0, so caches stay
/// reusable across batch operations like the KV cache; anything else
/// discontinuous is an error rather than a silent wrong result.
pub const LinearCache = struct {
    gpa: Allocator,
    batch: usize,
    /// Per model layer: index among the recurrent layers, or null.
    rec_index: []?u32,
    /// Per recurrent layer: offset of its first slot in `buf` and slot length.
    offsets: []usize,
    lens: []usize,
    state_len: usize,
    conv_len: usize,
    buf: []f32,
    next_pos: []usize,

    fn slotLenOf(c: *const Config, li: usize) usize {
        if (c.linear_layers[li]) return linearStateLen(c) + linearConvLen(c);
        if (c.ssm_layers[li]) return ssmStateLen(c) + ssmConvLen(c);
        if (c.conv_layers[li]) return c.hidden_size * (c.conv_kernel - 1);
        return 0;
    }

    pub fn bytesFor(c: *const Config, batch: usize) u64 {
        if (!c.hasRecurrent()) return 0;
        var floats: u64 = 0;
        var n_rec: u64 = 0;
        for (0..c.num_layers) |li| {
            const len = slotLenOf(c, li);
            if (len == 0) continue;
            floats += @as(u64, len) * batch;
            n_rec += 1;
        }
        return floats * 4 + n_rec * @as(u64, batch) * 8;
    }

    pub fn init(gpa: Allocator, c: *const Config, batch: usize) !LinearCache {
        var n_rec: usize = 0;
        const rec_index = try gpa.alloc(?u32, c.num_layers);
        errdefer gpa.free(rec_index);
        for (0..c.num_layers) |li| {
            if (slotLenOf(c, li) > 0) {
                rec_index[li] = @intCast(n_rec);
                n_rec += 1;
            } else rec_index[li] = null;
        }
        const offsets = try gpa.alloc(usize, n_rec);
        errdefer gpa.free(offsets);
        const lens = try gpa.alloc(usize, n_rec);
        errdefer gpa.free(lens);
        var total: usize = 0;
        for (0..c.num_layers) |li| {
            const idx = rec_index[li] orelse continue;
            offsets[idx] = total;
            lens[idx] = slotLenOf(c, li);
            total += lens[idx] * batch;
        }
        const buf = try gpa.alloc(f32, total);
        errdefer gpa.free(buf);
        @memset(buf, 0);
        const next_pos = try gpa.alloc(usize, n_rec * batch);
        errdefer gpa.free(next_pos);
        @memset(next_pos, 0);
        return .{
            .gpa = gpa,
            .batch = batch,
            .rec_index = rec_index,
            .offsets = offsets,
            .lens = lens,
            .state_len = if (c.has_linear) linearStateLen(c) else if (c.has_ssm) ssmStateLen(c) else 0,
            .conv_len = if (c.has_linear) linearConvLen(c) else if (c.has_ssm) ssmConvLen(c) else 0,
            .buf = buf,
            .next_pos = next_pos,
        };
    }

    pub fn deinit(self: *LinearCache) void {
        self.gpa.free(self.buf);
        self.gpa.free(self.next_pos);
        self.gpa.free(self.offsets);
        self.gpa.free(self.lens);
        self.gpa.free(self.rec_index);
    }

    fn slot(self: *LinearCache, li: usize, b: usize) []f32 {
        const idx = self.rec_index[li].?;
        return self.buf[self.offsets[idx] + b * self.lens[idx] ..][0..self.lens[idx]];
    }

    /// Delta-rule state of a linear-attention layer.
    pub fn state(self: *LinearCache, li: usize, b: usize) []f32 {
        return self.slot(li, b)[0..self.state_len];
    }

    /// Convolution history of a linear-attention layer.
    pub fn conv(self: *LinearCache, li: usize, b: usize) []f32 {
        return self.slot(li, b)[self.state_len..][0..self.conv_len];
    }

    /// Convolution history `[hidden][taps - 1]` of a short-convolution layer.
    pub fn shortConv(self: *LinearCache, li: usize, b: usize) []f32 {
        return self.slot(li, b);
    }

    pub fn nextPos(self: *LinearCache, li: usize, b: usize) *usize {
        const idx = self.rec_index[li].?;
        return &self.next_pos[idx * self.batch + b];
    }
};

fn linearConvDim(c: *const Config) usize {
    return 2 * c.linear_k_heads * c.linear_k_dim + c.linear_v_heads * c.linear_v_dim;
}

/// Channels of a Mamba block's causal convolution: x, B and C for Mamba2
/// (`inter + 2 groups * state`), the inner projection for Mamba1.
fn ssmConvDim(c: *const Config) usize {
    const d = &c.ssm;
    return switch (d.kind) {
        .mamba2 => d.inter + 2 * d.groups * d.state,
        .mamba1 => d.inter,
        .none => 0,
    };
}

/// Floats of recurrent state per (Mamba layer, sequence): `[heads][head_dim][state]`
/// for Mamba2, `[inter][state]` for Mamba1.
fn ssmStateLen(c: *const Config) usize {
    const d = &c.ssm;
    return switch (d.kind) {
        .mamba2 => d.heads * d.head_dim * d.state,
        .mamba1 => d.inter * d.state,
        .none => 0,
    };
}

/// Floats of convolution history per (Mamba layer, sequence).
fn ssmConvLen(c: *const Config) usize {
    return if (c.ssm.kind == .none) 0 else ssmConvDim(c) * (c.ssm.conv_kernel - 1);
}

/// Floats of recurrent state per (linear layer, sequence): `[v_heads][k_dim][v_dim]`
/// for Gated DeltaNet, `[heads][head_dim][head_dim]` for lightning attention.
fn linearStateLen(c: *const Config) usize {
    return switch (c.linear_kind) {
        .gated_deltanet, .kda => c.linear_v_heads * c.linear_k_dim * c.linear_v_dim,
        .lightning => c.num_heads * c.head_dim * c.head_dim,
    };
}

/// Floats of convolution history per (linear layer, sequence).
fn linearConvLen(c: *const Config) usize {
    return switch (c.linear_kind) {
        .gated_deltanet, .kda => linearConvDim(c) * (c.linear_conv_kernel - 1),
        .lightning => 0,
    };
}

/// Decay rate of lightning-attention head `h` in layer `li`: `2^(-8 (h + 1) / heads)`
/// scaled by a factor that falls linearly from 1 in the first layer to ~0 in the last.
fn lightningSlope(c: *const Config, li: usize, h: usize) f32 {
    const heads: f64 = @floatFromInt(c.num_heads);
    const base = std.math.pow(f64, 2.0, -8.0 / heads);
    const rate = std.math.pow(f64, base, @as(f64, @floatFromInt(h + 1)));
    const factor = 1.0 - @as(f64, @floatFromInt(li)) / (@as(f64, @floatFromInt(c.num_layers)) - 1.0 + 1e-5) + 1e-5;
    return @floatCast(rate * factor);
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
    /// Layer whose keys/values are read (the source layer of a KV-shared layer).
    layer: usize,
    rows: []const Row,
    q: []f32, // [n][heads*hd] (already roped)
    out: []f32, // [n][heads*hd]; per head the first vd entries are written
    /// Head geometry of the layer.
    hd: usize,
    vd: usize,
    heads: usize,
    groups: usize,
    scale: f32,
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
    const hd = ctx.hd;
    const vd = ctx.vd;
    const groups = ctx.groups;
    const slot = start / ctx.per;
    const scores_buf = ctx.scores[slot * ctx.max_keys ..][0..ctx.max_keys];
    const stride = ctx.cache.kv_dim;
    // The vector path needs whole vectors along the head dimension.
    const vectorised = hd % tensor.VL == 0 and vd % tensor.VL == 0;
    var i = start;
    while (i < end) : (i += 1) {
        const task = (i % ctx.per) * ctx.chunks + slot;
        if (task >= ctx.n_tasks) continue;
        const r = task / ctx.heads;
        const h = task % ctx.heads;
        const row = ctx.rows[r];
        const kvh = h / groups;
        const q = ctx.q[r * ctx.heads * hd + h * hd ..][0..hd];
        const out = ctx.out[r * ctx.heads * hd + h * hd ..][0..vd];
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
            compute.attentionScores(scores, q, kbase, stride, ctx.scale);
        } else {
            for (scores, 0..) |*s, p| s.* = tensor.dot(q, kbase[p * stride ..][0..hd]) * ctx.scale;
        }
        if (c.attn_logit_softcapping) |cap| {
            for (scores) |*s| s.* = cap * std.math.tanh(s.* / cap);
        }
        if (slope != 0) {
            for (scores, 0..) |*s, p| s.* += slope * @as(f32, @floatFromInt(lo + p));
        }
        if (ctx.sinks) |sk| softmaxWithSink(scores, sk[h]) else compute.softmaxInPlace(scores);
        if (vectorised) {
            compute.attentionValues(out, scores, vbase, stride);
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
    /// Extra AltUp residual streams `[rows][(altup_inputs - 1) * hidden]` (Gemma 3n).
    alt: []f32,
    /// Per-layer inputs `[rows][layers * ple_dim]` (Gemma 3n / 4).
    ple: []f32,
    logits: []f32,
    max_rows: usize,
    max_logit_rows: usize,

    /// Bytes `init` allocates per row (everything but the logits).
    pub fn bytesPerRow(c: *const Config) u64 {
        const hidden: u64 = c.hidden_size;
        const qd: u64 = c.maxQDim();
        const kvd: u64 = c.kvDim();
        const inter: u64 = c.intermediate_size;
        return (4 * hidden + streamWidth(c) + 2 * qd + 2 * kvd + fusedQkvRows(c) + 2 * inter + fusedGateUpCols(c) + linearCols(c) + altCols(c) + pleCols(c)) * 4;
    }

    /// Width of one residual row: `hidden` times the number of residual streams.
    pub fn streamWidth(c: *const Config) usize {
        return c.hc_mult * c.hidden_size;
    }

    fn altCols(c: *const Config) usize {
        return if (c.altup_inputs > 1) (c.altup_inputs - 1) * c.hidden_size else 0;
    }

    fn pleCols(c: *const Config) usize {
        return c.num_layers * c.ple_dim;
    }

    fn fusedQkvRows(c: *const Config) usize {
        if ((c.qkv_layout == .separate and c.qkv_alt == null) or c.mla != null) return 0;
        return c.maxQDim() + 2 * c.kvDim();
    }

    fn fusedGateUpCols(c: *const Config) usize {
        return if (c.mlp == .gated_fused) 2 * c.intermediate_size else 0;
    }

    fn linearCols(c: *const Config) usize {
        if (!c.has_linear) return 0;
        return switch (c.linear_kind) {
            .gated_deltanet, .kda => 2 * c.linear_k_heads * c.linear_k_dim + 3 * c.linear_v_heads * c.linear_v_dim + 2 * c.linear_v_heads,
            .lightning => 5 * c.num_heads * c.head_dim,
        };
    }

    pub fn init(gpa: Allocator, c: *const Config, max_rows: usize, max_logit_rows: usize) !Workspace {
        const hidden = c.hidden_size;
        const qd = c.maxQDim();
        const kvd = c.kvDim();
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
            .alt = &.{},
            .ple = &.{},
            .logits = &.{},
            .max_rows = max_rows,
            .max_logit_rows = max_logit_rows,
        };
        errdefer self.deinit();
        self.x = try gpa.alloc(f32, max_rows * streamWidth(c));
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
        self.alt = try gpa.alloc(f32, max_rows * altCols(c));
        self.ple = try gpa.alloc(f32, max_rows * pleCols(c));
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
        self.gpa.free(self.alt);
        self.gpa.free(self.ple);
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
        .rms => compute.rmsnorm(out, x, nm.w, c.rms_norm_eps, false),
        .rms_gemma => compute.rmsnorm(out, x, nm.w, c.rms_norm_eps, true),
        .layer => compute.layernorm(out, x, nm.w, nm.b, c.rms_norm_eps, false),
        .layer_1p => compute.layernorm(out, x, nm.w, nm.b, c.rms_norm_eps, true),
        .none => compute.layernorm(out, x, &.{}, null, c.rms_norm_eps, false),
        .rms_none => compute.rmsnorm(out, x, &.{}, c.rms_norm_eps, false),
    }
}

/// `out[i] = norm(x[i])` for `n` rows, or a copy when there is no norm.
pub fn normRows(c: *const Config, out: []f32, x: []const f32, n: usize, hidden: usize, nm: ?Norm) void {
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
    // The kernels take their statistics before writing, element by element,
    // so they run in place. (A fixed stack buffer here once capped the vector
    // at 1024 floats; OLMo 2 / OLMoE normalise the whole q projection,
    // 2048–4096 of them.)
    const out = x;
    if (w.len == 0) {
        var ss: f32 = 0;
        for (x) |v| ss += v * v;
        const inv = 1.0 / @sqrt(ss / @as(f32, @floatFromInt(x.len)) + c.rms_norm_eps);
        for (x) |*v| v.* *= inv;
        return;
    }
    switch (c.norm) {
        .rms, .none, .rms_none => compute.rmsnorm(out, x, w, c.rms_norm_eps, false),
        .rms_gemma => compute.rmsnorm(out, x, w, c.rms_norm_eps, true),
        .layer, .layer_1p => compute.layernorm(out, x, w, b, c.rms_norm_eps, false),
    }
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

/// Rotates the first `table.dim` coordinates of a head vector at position `pos`.
fn ropeHead(c: *const Config, x: []f32, table: *const RopeTable, pos: usize) void {
    const d = table.dim;
    if (d == 0) return;
    const p = @min(pos, table.len - 1);
    const cos_row = table.cos[p * table.half ..][0..table.half];
    const sin_row = table.sin[p * table.half ..][0..table.half];
    switch (c.rope_style) {
        .neox => compute.applyRope(x[0..d], cos_row, sin_row),
        .gptj => compute.applyRopeInterleaved(x[0..d], cos_row, sin_row),
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
fn scatterQkv(c: *const Config, layout: QkvLayout, hd: usize, vd: usize, nkv: usize, fused: []const f32, q: []f32, k: []f32, v: []f32) void {
    const nh = c.num_heads;
    const qd = nh * hd;
    const kd = nkv * hd;
    switch (layout) {
        .separate => unreachable,
        .concat => {
            @memcpy(q, fused[0..qd]);
            @memcpy(k, fused[qd..][0..kd]);
            for (0..nkv) |g| @memcpy(v[g * hd ..][0..vd], fused[qd + kd + g * vd ..][0..vd]);
        },
        .heads_interleaved => {
            std.debug.assert(nkv == nh);
            for (0..nh) |h| {
                @memcpy(q[h * hd ..][0..hd], fused[h * (2 * hd + vd) ..][0..hd]);
                @memcpy(k[h * hd ..][0..hd], fused[h * (2 * hd + vd) + hd ..][0..hd]);
                @memcpy(v[h * hd ..][0..vd], fused[h * (2 * hd + vd) + 2 * hd ..][0..vd]);
            }
        },
        .grouped => {
            // `chunks` blocks of `[q heads of the chunk | its k heads | its v heads]`;
            // one chunk per kv head unless the family says otherwise (MiMo V2 Pro).
            const chunks = if (c.qkv_chunks != 0) c.qkv_chunks else nkv;
            const qpc = nh / chunks;
            const kpc = nkv / chunks;
            const chunk_len = qpc * hd + kpc * (hd + vd);
            for (0..chunks) |g| {
                const base = g * chunk_len;
                for (0..qpc) |j| @memcpy(q[(g * qpc + j) * hd ..][0..hd], fused[base + j * hd ..][0..hd]);
                for (0..kpc) |j| @memcpy(k[(g * kpc + j) * hd ..][0..hd], fused[base + (qpc + j) * hd ..][0..hd]);
                for (0..kpc) |j| @memcpy(v[(g * kpc + j) * hd ..][0..vd], fused[base + (qpc + kpc) * hd + j * vd ..][0..vd]);
            }
        },
        .mp_blocks => {
            // CodeGen: `qkv_mp` blocks of `[q | v | k]`, each over `heads / qkv_mp` heads.
            std.debug.assert(nkv == nh);
            const mp = c.qkv_mp;
            const local = qd / mp;
            for (0..mp) |b| {
                const base = b * 3 * local;
                @memcpy(q[b * local ..][0..local], fused[base..][0..local]);
                @memcpy(v[b * local ..][0..local], fused[base + local ..][0..local]);
                @memcpy(k[b * local ..][0..local], fused[base + 2 * local ..][0..local]);
            }
        },
    }
}

/// Re-lays a `[n][kv heads * v_head_dim]` value projection out as
/// `[n][kv heads][head_dim]` (the KV cache stride), in place, when the
/// value heads are narrower than the key heads (MiMo V2).
fn spreadValues(v: []f32, n: usize, nkv: usize, hd: usize, vd: usize) void {
    if (vd == hd) return;
    var i: usize = n * nkv;
    while (i > 0) {
        i -= 1;
        std.mem.copyBackwards(f32, v[i * hd ..][0..vd], v[i * vd ..][0..vd]);
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
    const eps = m.latent_norm_eps;
    if (mw.q_a) |qa| {
        const qlr = m.q_lora_rank.?;
        const qa_out = try gpa.alloc(f32, n * qlr);
        defer gpa.free(qa_out);
        try compute.matmulT(model.pool, gpa, qa_out, h, n, qa, null);
        const tmp = try gpa.alloc(f32, qlr);
        defer gpa.free(tmp);
        for (0..n) |t| {
            const row = qa_out[t * qlr ..][0..qlr];
            compute.rmsnorm(tmp, row, mw.q_a_norm.?, eps, false);
            @memcpy(row, tmp);
        }
        try compute.matmulT(model.pool, gpa, ws.q, qa_out, n, mw.q_b, null);
    } else {
        try compute.matmulT(model.pool, gpa, ws.q, h, n, mw.q_b, null);
    }
    const kvr = m.kv_lora_rank + rd;
    const kva = try gpa.alloc(f32, n * kvr);
    defer gpa.free(kva);
    try compute.matmulT(model.pool, gpa, kva, h, n, mw.kv_a, null);
    const ckv = try gpa.alloc(f32, n * m.kv_lora_rank);
    defer gpa.free(ckv);
    for (0..n) |t| compute.rmsnorm(ckv[t * m.kv_lora_rank ..][0..m.kv_lora_rank], kva[t * kvr ..][0..m.kv_lora_rank], mw.kv_a_norm, eps, false);
    const kvb_rows = nh * (nope + vd);
    const kvb = try gpa.alloc(f32, n * kvb_rows);
    defer gpa.free(kvb);
    try compute.matmulT(model.pool, gpa, kvb, ckv, n, mw.kv_b, null);
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
        try compute.matmulT(model.pool, gpa, p, h, n, w, null);
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
        try compute.matmulT(model.pool, gpa, mixed, h, n, lin.qkv.?, null);
        try compute.matmulT(model.pool, gpa, zbuf, h, n, lin.z.?, null);
    }
    if (lin.ba) |w| {
        const p = try gpa.alloc(f32, n * 2 * vh);
        defer gpa.free(p);
        try compute.matmulT(model.pool, gpa, p, h, n, w, null);
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
        try compute.matmulT(model.pool, gpa, bbuf, h, n, lin.b.?, null);
        try compute.matmulT(model.pool, gpa, abuf, h, n, lin.a.?, null);
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
            if (c.linear_gate_sigmoid) {
                for (o, 0..) |*v, jj| v.* = v.* * inv * lin.norm[jj] / (1.0 + @exp(-z[jj]));
            } else {
                for (o, 0..) |*v, jj| v.* = v.* * inv * lin.norm[jj] * tensor.silu(z[jj]);
            }
        }
        next.* = pos + 1;
    }
    try compute.matmulT(model.pool, gpa, ws.o, core, n, layer.o, if (layer.o_delta) |*d| d else null);
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
            try compute.matmulT(model.pool, gpa, tmp, h, n, w, null);
            for (0..n) |t| @memcpy(mixed[t * conv_dim + s * dim ..][0..dim], tmp[t * dim ..][0..dim]);
        }
    }
    // Forget gate, per channel: g = -exp(A_log[head]) * softplus(f_b(f_a(h)) + dt_bias),
    // or the Kimi K3 safe gate g = lower_bound * sigmoid(exp(A_log[head]) * (f + dt_bias)).
    const low = try gpa.alloc(f32, n * hd);
    defer gpa.free(low);
    const decay = try gpa.alloc(f32, n * dim);
    defer gpa.free(decay);
    try compute.matmulT(model.pool, gpa, low, h, n, lin.f_a.?, null);
    try compute.matmulT(model.pool, gpa, decay, low, n, lin.f_b.?, null);
    for (0..n) |t| {
        const g = decay[t * dim ..][0..dim];
        for (0..nh) |hh| {
            const rate = @exp(lin.a_log[hh]);
            if (c.linear_gate_lower_bound) |lb| {
                for (0..hd) |i| g[hh * hd + i] = lb / (1.0 + @exp(-rate * (g[hh * hd + i] + lin.dt_bias[hh * hd + i])));
            } else {
                for (0..hd) |i| g[hh * hd + i] = -rate * softplus(g[hh * hd + i] + lin.dt_bias[hh * hd + i]);
            }
        }
    }
    const bbuf = try gpa.alloc(f32, n * nh);
    defer gpa.free(bbuf);
    try compute.matmulT(model.pool, gpa, bbuf, h, n, lin.b.?, null);
    // Output gate: g(h) (full rank) or g_b(g_a(h)).
    const gate = try gpa.alloc(f32, n * dim);
    defer gpa.free(gate);
    if (lin.g) |g| {
        try compute.matmulT(model.pool, gpa, gate, h, n, g, null);
    } else {
        try compute.matmulT(model.pool, gpa, low, h, n, lin.g_a.?, null);
        try compute.matmulT(model.pool, gpa, gate, low, n, lin.g_b.?, null);
    }

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
    try compute.matmulT(model.pool, gpa, ws.o, core, n, layer.o, if (layer.o_delta) |*d| d else null);
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

/// MiniMax lightning-attention sublayer: `act(h Wqkvᵀ)` split per head into
/// q, k, v; per head the decayed recurrence `S = exp(-s_h) S + kᵀ v`,
/// `o = q S` (the sequential form of the reference's blocked prefill, which
/// it equals exactly); RMSNorm over every head, the sigmoid gate `σ(h Wgᵀ)`
/// and the output projection (with its delta). Reads `h`, writes `ws.o`.
fn lightningForward(model: *const Model, layer: *const Layer, li: usize, ws: *Workspace, cache: *KvCache, h: []const f32, rows: []const Row) !void {
    const c = &model.config;
    const gpa = model.gpa;
    const lin = layer.linear.?;
    const lcache = &(cache.linear orelse return error.MissingLinearCache);
    const n = rows.len;
    const hidden = c.hidden_size;
    const nh = c.num_heads;
    const hd = c.head_dim;
    const qd = nh * hd;
    // The reference's `MiniMaxRMSNorm(head_dim * heads)` keeps its default epsilon.
    const norm_eps: f32 = 1e-6;

    const proj = try gpa.alloc(f32, n * 3 * qd);
    defer gpa.free(proj);
    try compute.matmulT(model.pool, gpa, proj, h, n, lin.light_qkv.?, null);
    // `ACT2FN[config.hidden_act]`: silu on the releases, but not fixed.
    for (proj) |*v| v.* = c.activation.apply(v.*);
    const gate = try gpa.alloc(f32, n * qd);
    defer gpa.free(gate);
    try compute.matmulT(model.pool, gpa, gate, h, n, lin.light_gate.?, null);
    const core = try gpa.alloc(f32, n * qd);
    defer gpa.free(core);
    const decay = try gpa.alloc(f32, nh);
    defer gpa.free(decay);
    for (decay, 0..) |*d, hh| d.* = @exp(-lightningSlope(c, li, hh));

    for (0..n) |t| {
        const b = rows[t].b;
        const pos = rows[t].pos;
        const S = lcache.state(li, b);
        const next = lcache.nextPos(li, b);
        if (pos == 0) {
            @memset(S, 0);
            next.* = 0;
        }
        if (pos != next.*) return error.NonContiguousRows;
        const p = proj[t * 3 * qd ..][0 .. 3 * qd];
        for (0..nh) |hh| {
            const q = p[hh * 3 * hd ..][0..hd];
            const k = p[hh * 3 * hd + hd ..][0..hd];
            const v = p[hh * 3 * hd + 2 * hd ..][0..hd];
            const St = S[hh * hd * hd ..][0 .. hd * hd];
            const r = decay[hh];
            for (0..hd) |i| {
                const row = St[i * hd ..][0..hd];
                for (row, 0..) |*s, j| s.* = r * s.* + k[i] * v[j];
            }
            const out = core[t * qd + hh * hd ..][0..hd];
            @memset(out, 0);
            for (0..hd) |i| tensor.axpy(out, q[i], St[i * hd ..][0..hd]);
        }
        next.* = pos + 1;
    }
    const tmp = try gpa.alloc(f32, qd);
    defer gpa.free(tmp);
    for (0..n) |t| {
        const row = core[t * qd ..][0..qd];
        compute.rmsnorm(tmp, row, lin.norm, norm_eps, false);
        const g = gate[t * qd ..][0..qd];
        for (row, 0..) |*v, j| v.* = tmp[j] / (1.0 + @exp(-g[j]));
    }
    try compute.matmulT(model.pool, gpa, ws.o, core, n, layer.o, if (layer.o_delta) |*d| d else null);
    if (layer.o_bias) |bias| addBias(ws.o, n, hidden, bias);
}

pub fn elemAt(dtype: tensor.DType, data: []const u8, idx: usize) f32 {
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

/// Mamba sublayer (Mamba2 or Mamba1): the input projection over every row
/// at once, then the causal convolution and the selective scan per batch
/// slot in row order against `cache.linear`, the gate / gated norm and the
/// output projection (with its delta). Reads `h`, writes `out[n][hidden]`.
/// The scan runs sequentially over tokens; heads (Mamba2) or channels
/// (Mamba1) are independent and spread over the thread pool.
fn ssmForward(model: *const Model, layer: *const Layer, li: usize, cache: *KvCache, h: []const f32, rows: []const Row, out: []f32) !void {
    const c = &model.config;
    const d = &c.ssm;
    const gpa = model.gpa;
    const s = &layer.ssm.?;
    const rc = &(cache.linear orelse return error.MissingLinearCache);
    const n = rows.len;
    const hidden = c.hidden_size;
    const inter = d.inter;

    var hin: []const f32 = h[0 .. n * hidden];
    var scaled: ?[]f32 = null;
    defer if (scaled) |b| gpa.free(b);
    if (c.mult.ssm_in != 1.0) {
        const b = try gpa.alloc(f32, n * hidden);
        @memcpy(b, hin);
        tensor.scale(b, c.mult.ssm_in);
        scaled = b;
        hin = b;
    }
    const proj_cols = s.in_proj.rows;
    const proj = try gpa.alloc(f32, n * proj_cols);
    defer gpa.free(proj);
    try compute.matmulT(model.pool, gpa, proj, hin, n, s.in_proj, null);
    if (s.in_bias) |b| addBias(proj, n, proj_cols, b);

    // Sequence bookkeeping: a slot restarts at position 0 and otherwise
    // continues where the previous call left it; within one call a slot's
    // rows must be consecutive positions.
    const seen = try gpa.alloc(bool, rc.batch);
    defer gpa.free(seen);
    @memset(seen, false);
    for (rows) |row| {
        const next = rc.nextPos(li, row.b);
        if (row.pos == 0) {
            if (seen[row.b]) return error.NonContiguousRows;
            @memset(rc.state(li, row.b), 0);
            @memset(rc.conv(li, row.b), 0);
            next.* = 0;
        }
        if (row.pos != next.*) return error.NonContiguousRows;
        next.* = row.pos + 1;
        seen[row.b] = true;
    }

    const core = try gpa.alloc(f32, n * inter);
    defer gpa.free(core);
    switch (d.kind) {
        .mamba2 => try mamba2Scan(model, s, li, rc, proj, rows, core),
        .mamba1 => try mamba1Scan(model, s, li, rc, proj, rows, core),
        .none => unreachable,
    }
    const separate = s.out != null;
    const ow = s.out orelse layer.o;
    const delta: ?*const Delta = if (separate) (if (layer.ssm_out_delta) |*dl| dl else null) else (if (layer.o_delta) |*dl| dl else null);
    try compute.matmulT(model.pool, gpa, out, core, n, ow, delta);
    const bias = if (separate) s.out_bias else layer.o_bias;
    if (bias) |b| addBias(out, n, hidden, b);
}

/// One step of a depthwise causal convolution over `conv_dim` channels:
/// `y[j] = act(bias[j] + Σ_i w[j][K-1-i] · x_{t-i})`, with `hist` holding the
/// previous `K - 1` inputs of every channel (oldest first), which it advances.
fn causalConvStep(w: []const f32, bias: ?[]const f32, kc: usize, conv_dim: usize, hist: []f32, x: []const f32, y: []f32, act: tensor.Activation) void {
    for (0..conv_dim) |j| {
        var acc: f32 = if (bias) |b| b[j] else 0;
        acc += w[j * kc + kc - 1] * x[j];
        if (kc > 1) {
            const hj = hist[j * (kc - 1) ..][0 .. kc - 1];
            for (1..kc) |i| acc += w[j * kc + kc - 1 - i] * hj[kc - 1 - i];
            @memmove(hj[0 .. kc - 2], hj[1 .. kc - 1]);
            hj[kc - 2] = x[j];
        }
        y[j] = act.apply(acc);
    }
}

/// `y = norm(y ⊙ silu(z)) ⊙ w`, or `norm(y) ⊙ w ⊙ silu(z)` when the norm
/// comes first, the RMS taken over `norm_groups` equal groups of channels;
/// without a norm weight only the gate applies.
fn gatedNorm(d: *const arch.SsmDims, w: ?[]const f32, eps: f32, y: []f32, z: []const f32) void {
    const nw = w orelse {
        for (y, 0..) |*v, j| v.* *= tensor.silu(z[j]);
        return;
    };
    if (!d.norm_before_gate) {
        for (y, 0..) |*v, j| v.* *= tensor.silu(z[j]);
    }
    const gs = y.len / d.norm_groups;
    for (0..d.norm_groups) |g| {
        const yg = y[g * gs ..][0..gs];
        var ss: f32 = 0;
        for (yg) |v| ss += v * v;
        const inv = 1.0 / @sqrt(ss / @as(f32, @floatFromInt(gs)) + eps);
        for (yg, 0..) |*v, j| v.* = v.* * inv * nw[g * gs + j];
    }
    if (d.norm_before_gate) {
        for (y, 0..) |*v, j| v.* *= tensor.silu(z[j]);
    }
}

const Mamba2Ctx = struct {
    rows: []const Row,
    /// Convolved, activated x / B / C channels `[n][conv_dim]`.
    xbc: []const f32,
    /// Discretised time steps `[n][heads]`.
    dt: []const f32,
    s: *const SsmWeights,
    rc: *LinearCache,
    li: usize,
    /// Scan output `[n][inter]`.
    core: []f32,
    inter: usize,
    hd: usize,
    state: usize,
    groups: usize,
    heads: usize,
    conv_dim: usize,
};

/// The Mamba2 recurrence for heads `[start, end)`, every row in order:
/// `S_h = exp(dt A_h) S_h + dt x_h Bᵀ`, `y_h = S_h C + D_h x_h`, with the
/// `[head_dim][state]` state of each head kept per batch slot.
fn mamba2Worker(ctx: *const Mamba2Ctx, start: usize, end: usize) void {
    const N = ctx.state;
    const hd = ctx.hd;
    const inter = ctx.inter;
    const gn = ctx.groups * N;
    const per_group = ctx.heads / ctx.groups;
    for (start..end) |h| {
        const g = h / per_group;
        const a_h = ctx.s.a[h];
        const d_h = ctx.s.d[h];
        for (ctx.rows, 0..) |row, t| {
            const S = ctx.rc.state(ctx.li, row.b)[h * hd * N ..][0 .. hd * N];
            const x = ctx.xbc[t * ctx.conv_dim + h * hd ..][0..hd];
            const B = ctx.xbc[t * ctx.conv_dim + inter + g * N ..][0..N];
            const C = ctx.xbc[t * ctx.conv_dim + inter + gn + g * N ..][0..N];
            const dt = ctx.dt[t * ctx.heads + h];
            const da = @exp(dt * a_h);
            const y = ctx.core[t * inter + h * hd ..][0..hd];
            for (0..hd) |p| {
                const srow = S[p * N ..][0..N];
                tensor.scale(srow, da);
                tensor.axpy(srow, dt * x[p], B);
                y[p] = tensor.dot(srow, C) + d_h * x[p];
            }
        }
    }
}

/// Mamba2 (SSD) block body on the projected rows `proj[n][inter + conv_dim + heads]`
/// (gate, conv channels, dt): writes the gated, normalised scan output to `core[n][inter]`.
fn mamba2Scan(model: *const Model, s: *const SsmWeights, li: usize, rc: *LinearCache, proj: []f32, rows: []const Row, core: []f32) !void {
    const c = &model.config;
    const d = &c.ssm;
    const gpa = model.gpa;
    const n = rows.len;
    const inter = d.inter;
    const heads = d.heads;
    const N = d.state;
    const gn = d.groups * N;
    const kc = d.conv_kernel;
    const conv_dim = inter + 2 * gn;
    const proj_cols = inter + conv_dim + heads;

    // Falcon-H1 muP: per-section multipliers on the projection (gate, x, B, C, dt).
    const mp = c.mult.ssm_proj;
    const bounds = [_]usize{ 0, inter, 2 * inter, 2 * inter + gn, 2 * inter + 2 * gn, proj_cols };
    for (0..5) |k| {
        if (mp[k] == 1.0) continue;
        for (0..n) |t| tensor.scale(proj[t * proj_cols + bounds[k] .. t * proj_cols + bounds[k + 1]], mp[k]);
    }
    // Causal convolution over the x / B / C channels, in row order per sequence.
    const xbc = try gpa.alloc(f32, n * conv_dim);
    defer gpa.free(xbc);
    for (rows, 0..) |row, t| {
        causalConvStep(s.conv, s.conv_bias, kc, conv_dim, rc.conv(li, row.b), proj[t * proj_cols + inter ..][0..conv_dim], xbc[t * conv_dim ..][0..conv_dim], d.act);
    }
    // Time steps: softplus(dt + dt_bias), clamped to the family's limits.
    const dt = try gpa.alloc(f32, n * heads);
    defer gpa.free(dt);
    for (0..n) |t| {
        for (0..heads) |hh| dt[t * heads + hh] = std.math.clamp(softplus(proj[t * proj_cols + inter + conv_dim + hh] + s.dt_bias[hh]), d.dt_min, d.dt_max);
    }
    const ctx = Mamba2Ctx{
        .rows = rows,
        .xbc = xbc,
        .dt = dt,
        .s = s,
        .rc = rc,
        .li = li,
        .core = core,
        .inter = inter,
        .hd = d.head_dim,
        .state = N,
        .groups = d.groups,
        .heads = heads,
        .conv_dim = conv_dim,
    };
    model.pool.parallelFor(heads, &ctx, mamba2Worker);
    for (0..n) |t| gatedNorm(d, s.norm, c.rms_norm_eps, core[t * inter ..][0..inter], proj[t * proj_cols ..][0..inter]);
}

const Mamba1Ctx = struct {
    rows: []const Row,
    /// Convolved, activated inner channels `[n][inter]`.
    xa: []const f32,
    /// `softplus(dt_proj(dt) + bias)` `[n][inter]`.
    dt: []const f32,
    /// Normalised B then C per row `[n][2 state]`.
    bc: []const f32,
    /// The projection rows (`[n][2 inter]`); the gate is the second half.
    proj: []const f32,
    s: *const SsmWeights,
    rc: *LinearCache,
    li: usize,
    core: []f32,
    inter: usize,
    state: usize,
};

/// The Mamba1 recurrence for channels `[start, end)`, every row in order:
/// `S_c = exp(dt_c A_c) ⊙ S_c + dt_c x_c B`, `y_c = S_c · C + D_c x_c`,
/// gated by `silu(z_c)`, with the `[state]` vector of each channel kept per batch slot.
fn mamba1Worker(ctx: *const Mamba1Ctx, start: usize, end: usize) void {
    const N = ctx.state;
    const inter = ctx.inter;
    for (start..end) |ch| {
        const a = ctx.s.a[ch * N ..][0..N];
        const d_c = ctx.s.d[ch];
        for (ctx.rows, 0..) |row, t| {
            const S = ctx.rc.state(ctx.li, row.b)[ch * N ..][0..N];
            const dt = ctx.dt[t * inter + ch];
            const x = ctx.xa[t * inter + ch];
            const B = ctx.bc[t * 2 * N ..][0..N];
            const C = ctx.bc[t * 2 * N + N ..][0..N];
            const dtx = dt * x;
            var y: f32 = 0;
            for (0..N) |i| {
                S[i] = S[i] * @exp(dt * a[i]) + dtx * B[i];
                y += S[i] * C[i];
            }
            const z = ctx.proj[t * 2 * inter + inter + ch];
            ctx.core[t * inter + ch] = (y + d_c * x) * tensor.silu(z);
        }
    }
}

/// Mamba1 block body on the projected rows `proj[n][2 inter]` (conv channels,
/// gate): writes the gated scan output to `core[n][inter]`.
fn mamba1Scan(model: *const Model, s: *const SsmWeights, li: usize, rc: *LinearCache, proj: []f32, rows: []const Row, core: []f32) !void {
    const c = &model.config;
    const d = &c.ssm;
    const gpa = model.gpa;
    const n = rows.len;
    const inter = d.inter;
    const N = d.state;
    const R = d.dt_rank;
    const kc = d.conv_kernel;
    const proj_cols = 2 * inter;
    const eps = c.rms_norm_eps;

    const xa = try gpa.alloc(f32, n * inter);
    defer gpa.free(xa);
    for (rows, 0..) |row, t| {
        causalConvStep(s.conv, s.conv_bias, kc, inter, rc.conv(li, row.b), proj[t * proj_cols ..][0..inter], xa[t * inter ..][0..inter], d.act);
    }
    // x_proj → dt | B | C, each RMS-normalised when the family has the norms (Jamba).
    const xcols = R + 2 * N;
    const xp = try gpa.alloc(f32, n * xcols);
    defer gpa.free(xp);
    try compute.matmulT(model.pool, gpa, xp, xa, n, s.x_proj.?, null);
    const dtr = try gpa.alloc(f32, n * R);
    defer gpa.free(dtr);
    const bc = try gpa.alloc(f32, n * 2 * N);
    defer gpa.free(bc);
    for (0..n) |t| {
        const row = xp[t * xcols ..][0..xcols];
        const dt_part = row[0..R];
        const b_part = row[R..][0..N];
        const c_part = row[R + N ..][0..N];
        if (s.dt_norm) |w| compute.rmsnorm(dt_part, dt_part, w, eps, false);
        if (s.b_norm) |w| compute.rmsnorm(b_part, b_part, w, eps, false);
        if (s.c_norm) |w| compute.rmsnorm(c_part, c_part, w, eps, false);
        @memcpy(dtr[t * R ..][0..R], dt_part);
        @memcpy(bc[t * 2 * N ..][0..N], b_part);
        @memcpy(bc[t * 2 * N + N ..][0..N], c_part);
    }
    // dt = softplus(dt_proj(dt) + bias).
    const dt = try gpa.alloc(f32, n * inter);
    defer gpa.free(dt);
    try compute.matmulT(model.pool, gpa, dt, dtr, n, s.dt_proj.?, null);
    addBias(dt, n, inter, s.dt_bias);
    for (dt) |*v| v.* = softplus(v.*);
    const ctx = Mamba1Ctx{
        .rows = rows,
        .xa = xa,
        .dt = dt,
        .bc = bc,
        .proj = proj,
        .s = s,
        .rc = rc,
        .li = li,
        .core = core,
        .inter = inter,
        .state = N,
    };
    model.pool.parallelFor(inter, &ctx, mamba1Worker);
}

/// Gated short-convolution sublayer (LFM2 conv layers): `in_proj` splits into
/// B, C and x; `B * x` runs through a depthwise causal convolution whose
/// history lives in `cache.linear`; the result is gated by C and projected by
/// `out_proj` (with its delta). Reads `h`, writes `ws.o`.
fn convForward(model: *const Model, layer: *const Layer, li: usize, ws: *Workspace, cache: *KvCache, h: []const f32, rows: []const Row) !void {
    const c = &model.config;
    const gpa = model.gpa;
    const cw = layer.conv.?;
    const lcache = &(cache.linear orelse return error.MissingLinearCache);
    const n = rows.len;
    const hidden = c.hidden_size;
    const kc = c.conv_kernel;
    const bcx = try gpa.alloc(f32, n * 3 * hidden);
    defer gpa.free(bcx);
    try compute.matmulT(model.pool, gpa, bcx, h, n, cw.in, null);
    if (cw.in_bias) |b| addBias(bcx, n, 3 * hidden, b);
    const y = try gpa.alloc(f32, n * hidden);
    defer gpa.free(y);
    const kw = cw.kernel.data;
    const kdt = cw.kernel.dtype;
    for (0..n) |t| {
        const b = rows[t].b;
        const pos = rows[t].pos;
        const hist = lcache.shortConv(li, b);
        const next = lcache.nextPos(li, b);
        if (pos == 0) {
            @memset(hist, 0);
            next.* = 0;
        }
        if (pos != next.*) return error.NonContiguousRows;
        const row = bcx[t * 3 * hidden ..][0 .. 3 * hidden];
        const bg = row[0..hidden];
        const cg = row[hidden .. 2 * hidden];
        const xg = row[2 * hidden ..][0..hidden];
        const out = y[t * hidden ..][0..hidden];
        for (0..hidden) |ch| {
            const bx = bg[ch] * xg[ch];
            const hrow = hist[ch * (kc - 1) ..][0 .. kc - 1];
            var acc: f32 = elemAt(kdt, kw, ch * kc + kc - 1) * bx;
            for (0..kc - 1) |i| acc += elemAt(kdt, kw, ch * kc + i) * hrow[i];
            if (cw.kernel_bias) |kb| acc += kb[ch];
            if (kc > 2) @memmove(hrow[0 .. kc - 2], hrow[1 .. kc - 1]);
            hrow[kc - 2] = bx;
            out[ch] = cg[ch] * acc;
        }
        next.* = pos + 1;
    }
    try compute.matmulT(model.pool, gpa, ws.o, y, n, layer.o, if (layer.o_delta) |*d| d else null);
    if (layer.o_bias) |bias| addBias(ws.o, n, hidden, bias);
}

/// Attention sublayer: projections, q/k norms, RoPE, KV cache update,
/// attention and the output projection (with its delta). Reads `h`, writes `ws.o`.
fn attention(model: *const Model, layer: *const Layer, li: usize, ws: *Workspace, cache: *KvCache, h: []const f32, rows: []const Row) !void {
    const c = &model.config;
    const gpa = model.gpa;
    const n = rows.len;
    const hd = c.layer_head_dim[li];
    const qd = c.layer_heads[li] * hd;
    const kvd = c.layer_kv_heads[li] * hd;

    if (layer.mla != null) {
        try mlaProject(model, layer, ws, h, n);
        if (layer.attn_gate) |g| {
            // Kimi K3: sigmoid gate on the `[n][heads * v_head_dim]` attention output.
            const gate = try gpa.alloc(f32, n * g.rows);
            defer gpa.free(gate);
            try compute.matmulT(model.pool, gpa, gate, h, n, g, null);
            try attentionTail(model, layer, li, ws, cache, rows, gate, true);
            return;
        }
    } else if (layer.qkv) |w| {
        const qkv_rows = w.rows;
        const layout = if (c.qkv_layout != .separate) c.qkv_layout else c.qkv_alt.?;
        try compute.matmulT(model.pool, gpa, ws.qkv, h, n, w, null);
        if (layer.qkv_bias) |b| addBias(ws.qkv, n, qkv_rows, b);
        var i: usize = 0;
        while (i < n) : (i += 1) scatterQkv(c, layout, hd, c.layerVDim(li), c.layer_kv_heads[li], ws.qkv[i * qkv_rows ..][0..qkv_rows], ws.q[i * qd ..][0..qd], ws.k[i * kvd ..][0..kvd], ws.v[i * kvd ..][0..kvd]);
    } else if (c.gated_attention) {
        const tmp = try gpa.alloc(f32, n * 2 * qd);
        defer gpa.free(tmp);
        const gate = try gpa.alloc(f32, n * qd);
        defer gpa.free(gate);
        try compute.matmulT(model.pool, gpa, tmp, h, n, layer.q.?, null);
        if (layer.q_bias) |b| {
            if (b.len == 2 * qd) addBias(tmp, n, 2 * qd, b);
        }
        for (0..n) |i| {
            const qr = tmp[i * 2 * qd ..][0 .. 2 * qd];
            const qo = ws.q[i * qd ..][0..qd];
            const go = gate[i * qd ..][0..qd];
            for (0..c.layer_heads[li]) |hh| {
                @memcpy(qo[hh * hd ..][0..hd], qr[hh * 2 * hd ..][0..hd]);
                @memcpy(go[hh * hd ..][0..hd], qr[hh * 2 * hd + hd ..][0..hd]);
            }
        }
        if (layer.q_bias) |b| {
            if (b.len == qd) addBias(ws.q, n, qd, b);
        }
        try compute.matmulT(model.pool, gpa, ws.k, h, n, layer.k.?, null);
        try compute.matmulT(model.pool, gpa, ws.v, h, n, layer.v.?, null);
        if (layer.k_bias) |b| addBias(ws.k, n, kvd, b);
        if (layer.v_bias) |b| addBias(ws.v, n, kvd, b);
        try attentionTail(model, layer, li, ws, cache, rows, gate, false);
        return;
    } else {
        try compute.matmulT(model.pool, gpa, ws.q, h, n, layer.q.?, null);
        if (layer.q_bias) |b| addBias(ws.q, n, qd, b);
        // A KV-shared layer reads its source layer's cache and projects nothing else.
        if (!c.kvShared(li)) {
            try compute.matmulT(model.pool, gpa, ws.k, h, n, layer.k.?, null);
            const vd = c.layerVDim(li);
            if (layer.v) |vw| {
                try compute.matmulT(model.pool, gpa, ws.v, h, n, vw, null);
            } else {
                // Keys double as values (Gemma 4 `attention_k_eq_v`), before the norms.
                @memcpy(ws.v[0 .. n * kvd], ws.k[0 .. n * kvd]);
            }
            if (layer.k_bias) |b| addBias(ws.k, n, kvd, b);
            if (layer.v_bias) |b| addBias(ws.v, n, c.layer_kv_heads[li] * vd, b);
            // Narrow values (MiMo V2) are spread to the cache's head stride.
            if (layer.v != null) spreadValues(ws.v, n, c.layer_kv_heads[li], hd, vd);
        }
    }
    try attentionTail(model, layer, li, ws, cache, rows, null, false);
}

/// Norms, RoPE, KV cache update, attention and the output projection shared
/// by the projection layouts. `gate` (gated full attention) multiplies the
/// attention output before the output projection: laid out `[n][heads][head_dim]`
/// like the query, or, with `gate_compact`, `[n][heads * v_head_dim]` like
/// the compacted attention output (Kimi K3 MLA).
fn attentionTail(model: *const Model, layer: *const Layer, li: usize, ws: *Workspace, cache: *KvCache, rows: []const Row, gate: ?[]const f32, gate_compact: bool) !void {
    const c = &model.config;
    const gpa = model.gpa;
    const n = rows.len;
    try checkIndexBound(c, li, rows);
    const hidden = c.hidden_size;
    const hd = c.layer_head_dim[li];
    const nkv = c.layer_kv_heads[li];
    const nh = c.layer_heads[li];
    const qd = nh * hd;
    const kvd = nkv * hd;
    const shared = c.kvShared(li);
    if (c.clip_qkv) |clip| {
        clampAll(ws.q[0 .. n * qd], clip);
        if (!shared) {
            clampAll(ws.k[0 .. n * kvd], clip);
            clampAll(ws.v[0 .. n * kvd], clip);
        }
    }
    if (c.mult.key != 1.0) tensor.scale(ws.k[0 .. n * kvd], c.mult.key);
    if (c.mult.value != 1.0 and !shared) tensor.scale(ws.v[0 .. n * kvd], c.mult.value);

    const use_rope = c.rope_layers[li];
    const sliding = c.sliding_layers[li];
    const table: *const RopeTable = if (sliding) &model.rope_local else &model.rope;
    const rope_off: usize = if (c.mla) |m| m.qk_nope_head_dim else 0;
    const do_qk_norm = !(c.qk_norm_rope_only and !use_rope);
    var i: usize = 0;
    while (i < n) : (i += 1) {
        const pos = rows[i].pos;
        const qrow = ws.q[i * qd ..][0..qd];
        const krow = ws.k[i * kvd ..][0..kvd];
        if (do_qk_norm and c.qk_norm == .full) {
            if (layer.q_norm) |nm| normVecInPlace(c, qrow, nm.w, nm.b);
            if (!shared) if (layer.k_norm) |nm| normVecInPlace(c, krow, nm.w, nm.b);
        }
        var temp: ?f32 = null;
        if (c.attn_temperature) |t| {
            if (!use_rope or t.all_layers) {
                const p: f32 = @floatFromInt(pos);
                temp = @log(@floor((p + t.offset) / t.floor_scale) + 1.0) * t.attn_scale + 1.0;
            }
        }
        const norm_before = do_qk_norm and !c.qk_norm_after_rope;
        const norm_after = do_qk_norm and c.qk_norm_after_rope;
        var hh: usize = 0;
        while (hh < nh) : (hh += 1) {
            const q = qrow[hh * hd ..][0..hd];
            if (norm_before) if (layer.q_norm) |nm| qkNormHead(c, q, nm, hh);
            if (use_rope) ropeHead(c, q[rope_off..], table, pos);
            if (norm_after) if (layer.q_norm) |nm| qkNormHead(c, q, nm, hh);
            if (temp) |t| tensor.scale(q, t);
        }
        if (shared) continue;
        const vrow = ws.v[i * kvd ..][0..kvd];
        hh = 0;
        while (hh < nkv) : (hh += 1) {
            const k = krow[hh * hd ..][0..hd];
            if (norm_before) if (layer.k_norm) |nm| qkNormHead(c, k, nm, hh);
            if (use_rope) ropeHead(c, k[rope_off..], table, pos);
            if (norm_after) if (layer.k_norm) |nm| qkNormHead(c, k, nm, hh);
            if (c.v_norm) normVecInPlace(c, vrow[hh * hd ..][0..hd], &.{}, null);
        }
        @memcpy(cache.kSlot(li, rows[i].b, pos)[0..kvd], krow);
        @memcpy(cache.vSlot(li, rows[i].b, pos)[0..kvd], vrow);
        cache.noteWrite(pos);
    }
    const n_tasks = n * nh;
    const chunks = @max(1, @min(model.pool.threads, n_tasks));
    const per = (n_tasks + chunks - 1) / chunks;
    var max_keys: usize = 1;
    for (rows) |row| max_keys = @max(max_keys, row.pos + 1);
    const scores = try model.pool.allocScratch(gpa, chunks * max_keys);
    defer model.pool.freeScratch(gpa, scores);
    const vd = c.layerVDim(li);
    const actx = AttnCtx{
        .model = model,
        .cache = cache,
        .layer = c.kv_source[li],
        .rows = rows,
        .q = ws.q,
        .out = ws.attn,
        .hd = hd,
        .vd = vd,
        .heads = nh,
        .groups = nh / nkv,
        .scale = c.layer_attn_scale[li],
        .sliding = sliding,
        .sinks = layer.sinks,
        .n_tasks = n_tasks,
        .chunks = chunks,
        .per = per,
        .scores = scores,
        .max_keys = max_keys,
    };
    model.pool.parallelFor(chunks * per, &actx, attentionWorker);
    if (gate) |g| if (!gate_compact) {
        for (0..n) |r| {
            const a = ws.attn[r * qd ..][0..qd];
            const gg = g[r * qd ..][0..qd];
            for (a, 0..) |*v, j| v.* *= 1.0 / (1.0 + @exp(-gg[j]));
        }
    };
    if (vd != hd) {
        // Compact `[n][heads][head_dim]` (v_head_dim valid per head) to `[n][heads * v_head_dim]`.
        var dst: usize = 0;
        for (0..n * nh) |hi| {
            std.mem.copyForwards(f32, ws.attn[dst..][0..vd], ws.attn[hi * hd ..][0..vd]);
            dst += vd;
        }
    }
    const od = nh * vd;
    // AFMoE / Laguna gate the attention output from the layer input (the Kimi
    // K3 MLA gate is applied in `mlaProject`, before this point).
    if (c.attn_gate != .none) if (layer.attn_gate) |gw| {
        // Per coordinate (`[heads * head_dim]` rows) or per head.
        const per_head = gw.rows == nh;
        const g = try gpa.alloc(f32, n * gw.rows);
        defer gpa.free(g);
        try compute.matmulT(model.pool, gpa, g, ws.h[0 .. n * hidden], n, gw, null);
        for (0..n) |r| {
            const a = ws.attn[r * od ..][0..od];
            const gr = g[r * gw.rows ..][0..gw.rows];
            for (a, 0..) |*v, j| {
                const gv = if (per_head) gr[j / vd] else gr[j];
                v.* *= switch (c.attn_gate) {
                    .sigmoid => 1.0 / (1.0 + @exp(-gv)),
                    .softplus => softplus(gv),
                    .none => unreachable,
                };
            }
        }
    };
    if (layer.attn_sub_norm) |nm| {
        const tmp = try gpa.alloc(f32, n * od);
        defer gpa.free(tmp);
        normRowsInPlace(c, ws.attn[0 .. n * od], n, od, nm, tmp);
    }
    if (gate) |g| if (gate_compact) {
        for (ws.attn[0 .. n * od], 0..) |*v, j| v.* /= 1.0 + @exp(-g[j]);
    };
    try compute.matmulT(model.pool, gpa, ws.o, ws.attn, n, layer.o, if (layer.o_delta) |*d| d else null);
    if (layer.o_bias) |b| addBias(ws.o, n, hidden, b);
}

/// Inverse of the standard normal CDF (Acklam's rational approximation,
/// relative error below 1.2e-9), for the gaussian top-k cutoff.
fn normalPpf(p: f64) f64 {
    const a = [_]f64{ -3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02, 1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00 };
    const b = [_]f64{ -5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02, 6.680131188771972e+01, -1.328068155288572e+01 };
    const cc = [_]f64{ -7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00, -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00 };
    const d = [_]f64{ 7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00, 3.754408661907416e+00 };
    const plow = 0.02425;
    if (p < plow) {
        const q = @sqrt(-2.0 * @log(p));
        return (((((cc[0] * q + cc[1]) * q + cc[2]) * q + cc[3]) * q + cc[4]) * q + cc[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0);
    }
    if (p > 1.0 - plow) {
        const q = @sqrt(-2.0 * @log(1.0 - p));
        return -(((((cc[0] * q + cc[1]) * q + cc[2]) * q + cc[3]) * q + cc[4]) * q + cc[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0);
    }
    const q = p - 0.5;
    const r = q * q;
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0);
}

/// Gemma 3n activation sparsity: keeps the gate values above
/// `mean + std * ppf(sparsity)` of their row (population std), relu-shifted.
fn gaussianTopk(gate: []f32, n: usize, width: usize, sparsity: f32) void {
    const mult: f32 = @floatCast(normalPpf(sparsity));
    for (0..n) |r| {
        const row = gate[r * width ..][0..width];
        var mean: f32 = 0;
        for (row) |v| mean += v;
        mean /= @floatFromInt(width);
        var variance: f32 = 0;
        for (row) |v| variance += (v - mean) * (v - mean);
        variance /= @floatFromInt(width);
        const cutoff = mean + @sqrt(variance) * mult;
        for (row) |*v| v.* = @max(v.* - cutoff, 0);
    }
}

/// A sparse-attention indexer (Qwen4-Exp QSA, GLM-5.3-Flash DSA) scores blocks
/// of consecutive keys and keeps the best `max_blocks` complete ones plus the
/// incomplete tail. Dense attention is therefore exactly the reference while
/// every complete block is selected; beyond that ditch refuses rather than
/// silently approximating.
fn checkIndexBound(c: *const Config, li: usize, rows: []const Row) !void {
    const b = c.index_bound orelse return;
    if (c.linear_layers[li] or c.conv_layers[li]) return;
    var worst: usize = 0;
    for (rows) |row| worst = @max(worst, row.pos);
    if (b.fits(worst)) return;
    std.log.err("layer {d}: {d} complete key blocks are reachable but the indexer keeps {d}; the dense equivalent is only exact for contexts up to {d} tokens", .{ li, (worst + 1) / b.block, b.max_blocks, b.block * b.max_blocks });
    return error.ContextExceedsSparseIndexer;
}

/// MLP sublayer (dense or mixture of experts): reads `h_in`, writes `ws.m`.
pub fn mlpBlock(model: *const Model, layer: *const Layer, li: usize, ws: *Workspace, h_in: []const f32, n: usize, tokens: ?[]const u32) !void {
    const c = &model.config;
    const gpa = model.gpa;
    const hidden = c.hidden_size;
    if (layer.moe) |*m| return moe.forward(model, m, li, ws.m, h_in, n, tokens);
    const down = layer.down.?;
    const inter = down.cols;
    var din: []f32 = ws.gate;
    // A `gated_fused` family whose checkpoint ships the pair split (MiniMax M3)
    // is loaded, and run, as a plain gated MLP.
    switch (if (c.mlp == .gated_fused and layer.gate_up == null) arch.MlpKind.gated else c.mlp) {
        .gated => {
            try compute.matmulT(model.pool, gpa, ws.gate, h_in, n, layer.gate.?, null);
            try compute.matmulT(model.pool, gpa, ws.up, h_in, n, layer.up.?, null);
            if (layer.gate_bias) |b| addBias(ws.gate, n, inter, b);
            if (layer.up_bias) |b| addBias(ws.up, n, inter, b);
            if (c.mult.mlp_gate != 1.0) tensor.scale(ws.gate[0 .. n * inter], c.mult.mlp_gate);
            if (c.activation_sparsity[li] > 0) gaussianTopk(ws.gate, n, inter, c.activation_sparsity[li]);
            if (c.moe.swiglu_limit) |limit| {
                // The gate is clamped from above, the up projection on both sides.
                for (ws.gate[0 .. n * inter], 0..) |*g, j| {
                    g.* = @min(g.*, limit);
                    ws.up[j] = std.math.clamp(ws.up[j], -limit, limit);
                }
            }
            if (c.moe.swiglu) |sw| swigluOai(sw, ws.gate, ws.gate, ws.up, n, inter, inter) else if (c.moe.situ) |st| moe.situGlu(st, ws.gate, ws.gate, ws.up, n, inter, inter) else compute.gatedActivation(model.pool, c.activation, ws.gate, ws.gate, ws.up, n, inter, inter, inter);
        },
        .gated_fused => {
            const gu = layer.gate_up.?;
            try compute.matmulT(model.pool, gpa, ws.gate_up, h_in, n, gu, null);
            if (layer.up_bias) |b| addBias(ws.gate_up, n, 2 * inter, b);
            if (c.moe.swiglu) |sw| swigluOai(sw, ws.gate, ws.gate_up, ws.gate_up[inter..], n, inter, 2 * inter) else if (c.moe.situ) |st| moe.situGlu(st, ws.gate, ws.gate_up, ws.gate_up[inter..], n, inter, 2 * inter) else compute.gatedActivation(model.pool, c.activation, ws.gate, ws.gate_up, ws.gate_up[inter..], n, inter, 2 * inter, inter);
        },
        .dense => {
            try compute.matmulT(model.pool, gpa, ws.up, h_in, n, layer.up.?, null);
            if (layer.up_bias) |b| addBias(ws.up, n, inter, b);
            if (layer.xielu) |a| {
                // xIELU (Apertus): `alpha_p x² + beta x` for x > 0, `alpha_n (expm1(min(x, eps)) - x) + beta x` otherwise.
                for (ws.up[0 .. n * inter]) |*v| {
                    const x = v.*;
                    v.* = if (x > 0) a[0] * x * x + 0.5 * x else a[1] * (std.math.expm1(@min(x, -1e-6)) - x) + 0.5 * x;
                }
            } else {
                compute.gatedActivation(model.pool, c.activation, ws.up, ws.up, null, n, inter, inter, inter);
            }
            din = ws.up;
        },
    }
    if (layer.ffn_sub_norm) |nm| {
        const tmp = try gpa.alloc(f32, n * inter);
        defer gpa.free(tmp);
        normRowsInPlace(c, din[0 .. n * inter], n, inter, nm, tmp);
    }
    try compute.matmulT(model.pool, gpa, ws.m, din, n, down, if (layer.down_delta) |*d| d else null);
    if (layer.down_bias) |b| addBias(ws.m, n, hidden, b);
    if (c.mult.mlp_down != 1.0) tensor.scale(ws.m[0 .. n * hidden], c.mult.mlp_down);
}

/// Per-layer input block (Gemma 3n / 4): `norm(out(act(gate(src)) * ple[li]))`
/// for `n` rows, written to `ws.m`. `ple` holds `[n][layers * ple_dim]`.
fn pleBlock(model: *const Model, p: *const PleWeights, li: usize, ws: *Workspace, src: []const f32, ple: []const f32, n: usize) !void {
    const c = &model.config;
    const gpa = model.gpa;
    const hidden = c.hidden_size;
    const pd = c.ple_dim;
    const stride = c.num_layers * pd;
    const g = try gpa.alloc(f32, n * pd);
    defer gpa.free(g);
    try compute.matmulT(model.pool, gpa, g, src, n, p.gate, null);
    for (0..n) |r| {
        const row = g[r * pd ..][0..pd];
        const inp = ple[r * stride + li * pd ..][0..pd];
        for (row, 0..) |*v, j| v.* = c.activation.apply(v.*) * inp[j];
    }
    try compute.matmulT(model.pool, gpa, ws.m, g, n, p.out, null);
    normRowsInPlace(c, ws.m, n, hidden, p.norm, ws.h2);
}

/// RMS magnitude `sqrt(mean(x²))` of a row.
fn rowMagnitude(x: []const f32) f32 {
    var ss: f32 = 0;
    for (x) |v| ss += v * v;
    return @sqrt(ss / @as(f32, @floatFromInt(x.len)));
}

/// Projects an AltUp stream and rescales it to `target` (`x * target / max(rms(x), sqrt(1e-5))`).
fn altupProjectRows(model: *const Model, out: []f32, src: []const f32, n: usize, w: Weight, targets: []const f32) !void {
    const hidden = model.config.hidden_size;
    try compute.matmulT(model.pool, model.gpa, out, src, n, w, null);
    for (0..n) |r| {
        const row = out[r * hidden ..][0..hidden];
        const mag = @sqrt(@max(rowMagnitude(row) * rowMagnitude(row), 1e-5));
        tensor.scale(row, targets[r] / mag);
    }
}

/// Expands the embeddings `xs` (`[n][hidden]`) into the extra AltUp streams
/// `alt` (`[n][(altup_inputs - 1) * hidden]`), each magnitude-matched to the embedding.
fn altupExpand(model: *const Model, ws: *Workspace, leases: []const stream.Lease, xs: []const f32, alt: []f32, n: usize) !void {
    const c = &model.config;
    const gpa = model.gpa;
    const hidden = c.hidden_size;
    const alt_w = (c.altup_inputs - 1) * hidden;
    const targets = try gpa.alloc(f32, n);
    defer gpa.free(targets);
    for (0..n) |r| targets[r] = rowMagnitude(xs[r * hidden ..][0..hidden]);
    for (leases, 0..) |lease, e| {
        const tmp = ws.h2[0 .. n * hidden];
        try altupProjectRows(model, tmp, xs, n, lease.weight, targets);
        for (0..n) |r| @memcpy(alt[r * alt_w + e * hidden ..][0..hidden], tmp[r * hidden ..][0..hidden]);
    }
}

/// Folds the AltUp streams of one row back into `out`: the mean of the active
/// stream and the unembedded extra streams, each magnitude-matched to it.
fn altupFold(model: *const Model, leases: []const stream.Lease, x: []const f32, alt: []const f32, out: []f32, tmp: []f32) !void {
    const c = &model.config;
    const hidden = c.hidden_size;
    const target = [_]f32{rowMagnitude(x)};
    @memcpy(out[0..hidden], x[0..hidden]);
    for (leases, 0..) |lease, e| {
        try altupProjectRows(model, tmp[0..hidden], alt[e * hidden ..][0..hidden], 1, lease.weight, &target);
        tensor.axpy(out[0..hidden], 1.0, tmp[0..hidden]);
    }
    tensor.scale(out[0..hidden], 1.0 / @as(f32, @floatFromInt(c.altup_inputs)));
}

/// Fills `ple` (`[n][layers * ple_dim]`) with the per-layer inputs of `tokens`:
/// the projected, normalised embedding plus the token's own per-layer
/// embedding, scaled by `1/sqrt(2)`.
fn perLayerInputs(model: *const Model, proj: Weight, tokens: []const u32, xs: []const f32, ple: []f32, n: usize) !void {
    const c = &model.config;
    const gpa = model.gpa;
    const pd = c.ple_dim;
    const width = c.num_layers * pd;
    const store: *stream.WeightStore = @constCast(&model.store);
    const emb_ref = model.ple_embed_ref.?;
    try compute.matmulT(model.pool, gpa, ple, xs, n, proj, null);
    const emb = try gpa.alloc(f32, width);
    defer gpa.free(emb);
    const tmp = try gpa.alloc(f32, pd);
    defer gpa.free(tmp);
    const proj_scale = 1.0 / @sqrt(@as(f32, @floatFromInt(c.hidden_size)));
    const emb_scale = @sqrt(@as(f32, @floatFromInt(pd)));
    const mix_scale = 1.0 / @sqrt(2.0);
    for (0..n) |r| {
        try store.readRow(emb_ref, @min(tokens[r], emb_ref.rows - 1), emb);
        const row = ple[r * width ..][0..width];
        for (0..c.num_layers) |l| {
            const seg = row[l * pd ..][0..pd];
            tensor.scale(seg, proj_scale);
            applyNorm(c, tmp, seg, model.ple_proj_norm.?);
            for (seg, 0..) |*v, j| v.* = (tmp[j] + emb[l * pd + j] * emb_scale) * mix_scale;
        }
    }
}

/// The clamped gpt-oss / MiniMax M3 swiglu on `n` rows of `gate` and `up`
/// (row stride `in_stride`): `out = (clamp(up) + 1) · min(gate, limit) · σ(α gate)`,
/// written densely (`[n][inter]`).
fn swigluOai(sw: arch.Swiglu, out: []f32, gate: []const f32, up: []const f32, n: usize, inter: usize, in_stride: usize) void {
    for (0..n) |i| {
        const g = gate[i * in_stride ..][0..inter];
        const u = up[i * in_stride ..][0..inter];
        const o = out[i * inter ..][0..inter];
        for (o, 0..) |*v, j| {
            const gc = @min(g[j], sw.limit);
            const uc = std.math.clamp(u[j], -sw.limit, sw.limit);
            v.* = (uc + 1.0) * gc / (1.0 + @exp(-sw.alpha * gc));
        }
    }
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
///
/// Gemma 3n carries `altup_inputs - 1` extra residual streams and Gemma 3n / 4
/// per-layer inputs alongside the residual; they are computed with the
/// embeddings and chunked the same way.
pub fn forward(model: *const Model, ws: *Workspace, cache: *KvCache, tokens: []const u32, rows: []const Row, opts: ForwardOptions) !void {
    const c = &model.config;
    const gpa = model.gpa;
    const n = tokens.len;
    std.debug.assert(n == rows.len);
    const hidden = c.hidden_size;
    const hc = c.hc_mult;
    // Hyper-connection families keep `hc` residual streams per row.
    const sw = hc * hidden;
    const chunk_rows = ws.max_rows;
    const single = n <= chunk_rows;
    const alt_w = if (c.altup_inputs > 1) (c.altup_inputs - 1) * hidden else 0;
    const ple_w = c.num_layers * c.ple_dim;
    const act_opts = stream.Activations.InitOptions{
        .budget = model.budget,
        .scratch_dir = model.scratch_dir,
        .reserve = model.residentWeightNeed(),
        .evictable = model.expertCacheEvictable(),
        .force_scratch = model.spill_always,
    };
    var act: ?stream.Activations = null;
    defer if (act) |*a| a.deinit();
    var act_alt: ?stream.Activations = null;
    defer if (act_alt) |*a| a.deinit();
    var act_ple: ?stream.Activations = null;
    defer if (act_ple) |*a| a.deinit();
    if (!single) {
        act = try stream.Activations.init(gpa, model.io, n, sw, act_opts);
        if (alt_w > 0) act_alt = try stream.Activations.init(gpa, model.io, n, alt_w, act_opts);
        if (ple_w > 0) act_ple = try stream.Activations.init(gpa, model.io, n, ple_w, act_opts);
    }
    // Model-level projections of the embedding stage and the final fold.
    const store: *stream.WeightStore = @constCast(&model.store);
    // The final stream collapse of a gated hyper-connection family reads two
    // model-level matrices; acquire them once for the whole call.
    var head_lease = try hyper.acquireHead(model);
    defer head_lease.release(model);
    var ple_proj: ?stream.Lease = null;
    defer if (ple_proj) |l| store.release(l);
    if (ple_w > 0) ple_proj = try store.acquire(model.ple_proj_ref.?);
    const alt_leases = try gpa.alloc(stream.Lease, model.altup_proj_refs.len);
    defer gpa.free(alt_leases);
    var n_alt_leases: usize = 0;
    defer for (alt_leases[0..n_alt_leases]) |l| store.release(l);
    for (model.altup_proj_refs) |r| {
        alt_leases[n_alt_leases] = try store.acquire(r);
        n_alt_leases += 1;
    }

    // Embeddings.
    var start: usize = 0;
    while (start < n) : (start += chunk_rows) {
        const cn = @min(chunk_rows, n - start);
        const xs = if (single) ws.x[0 .. n * sw] else act.?.chunkUninit(start, cn, ws.x);
        for (tokens[start..][0..cn], 0..) |t, i| {
            const x = xs[i * sw ..][0..hidden];
            try model.embedRow(t, x);
            if (c.embed_scale != 1.0) tensor.scale(x, c.embed_scale);
            if (model.pos_embed_ref != null) {
                const pe = ws.h[0..hidden];
                try model.posEmbedRow(rows[start + i].pos, pe);
                tensor.axpy(x, 1.0, pe);
            }
            if (model.sin_pos.len > 0) {
                const p = @min(rows[start + i].pos + c.position_offset, model.sin_pos.len / hidden - 1);
                tensor.axpy(x, 1.0, model.sin_pos[p * hidden ..][0..hidden]);
            }
            if (model.embed_norm) |nm| {
                applyNorm(c, ws.h[0..hidden], x, nm);
                @memcpy(x, ws.h[0..hidden]);
            }
            // Every stream starts as a copy of the embedding.
            for (1..hc) |j| @memcpy(xs[i * sw + j * hidden ..][0..hidden], x);
        }
        if (ple_w > 0) {
            const ps = if (single) ws.ple[0 .. n * ple_w] else act_ple.?.chunkUninit(start, cn, ws.ple);
            try perLayerInputs(model, ple_proj.?.weight, tokens[start..][0..cn], xs, ps, cn);
            if (!single) try act_ple.?.commit(start, cn, ps);
        }
        if (alt_w > 0) {
            const as = if (single) ws.alt[0 .. n * alt_w] else act_alt.?.chunkUninit(start, cn, ws.alt);
            try altupExpand(model, ws, alt_leases[0..n_alt_leases], xs, as, cn);
            if (!single) try act_alt.?.commit(start, cn, as);
        }
        if (!single) try act.?.commit(start, cn, xs);
        if (hc == 1) captureResiduals(opts, 0, hidden, start, cn, xs);
    }

    // Attention Residual (Kimi K3): per token, the bank of block prefixes the
    // aggregation points retrieve from (`nb` rows, written by the first layer
    // of every block), kept in RAM for the whole call; `mix` receives one
    // chunk's aggregated block input, the residual ditch captures.
    const nb: usize = if (c.attn_res_block > 0) (c.num_layers + c.attn_res_block - 1) / c.attn_res_block else 0;
    var bank: []f32 = &.{};
    defer if (bank.len > 0) gpa.free(bank);
    var mix: []f32 = &.{};
    defer if (mix.len > 0) gpa.free(mix);
    if (nb > 0) {
        bank = try gpa.alloc(f32, n * nb * hidden);
        mix = try gpa.alloc(f32, chunk_rows * hidden);
    }
    // V4.1 single-pass hyper-connections: the collapse weights travel from
    // one site to the next; the first site reads stream 0 only.
    var pre_mix: ?[]f32 = null;
    defer if (pre_mix) |p| gpa.free(p);
    if (c.dsv4) |d| if (d.v41) {
        const p = try gpa.alloc(f32, n * hc);
        @memset(p, 0);
        for (0..n) |t| p[t * hc] = 1.0;
        pre_mix = p;
    };
    // The residual of a hyper-connection layer is its collapsed block input.
    var capture_buf: ?[]f32 = null;
    defer if (capture_buf) |b| gpa.free(b);
    if (hc > 1 and opts.residuals != null) capture_buf = try gpa.alloc(f32, chunk_rows * hidden);

    for (0..c.num_layers) |li| {
        if (model.budget) |b| try b.checkTime();
        var lease = try model.acquireLayer(li);
        defer lease.release();
        const layer = &lease.layer;
        // A KV-shared layer reads (and never writes) its source layer's cache.
        const kv_layer = c.kv_source[li];
        try cache.beginLayer(kv_layer);
        start = 0;
        while (start < n) : (start += chunk_rows) {
            const cn = @min(chunk_rows, n - start);
            const xs = if (single) ws.x[0 .. n * sw] else try act.?.chunk(start, cn, ws.x);
            if (hc > 1) {
                const pm: ?[]f32 = if (pre_mix) |p| p[start * hc ..][0 .. cn * hc] else null;
                try hyper.layerBlock(model, layer, li, ws, cache, xs, rows[start..][0..cn], tokens[start..][0..cn], pm, capture_buf);
                if (capture_buf) |cb| captureResiduals(opts, li, hidden, start, cn, cb);
            } else if (nb > 0) {
                try layerBlockAttnRes(model, layer, li, ws, cache, xs, rows[start..][0..cn], bank[start * nb * hidden ..][0 .. cn * nb * hidden], mix[0 .. cn * hidden]);
                captureResiduals(opts, li, hidden, start, cn, mix);
            } else {
                const alts: []f32 = if (alt_w == 0) &.{} else if (single) ws.alt[0 .. n * alt_w] else try act_alt.?.chunk(start, cn, ws.alt);
                const ples: []const f32 = if (ple_w == 0) &.{} else if (single) ws.ple[0 .. n * ple_w] else try act_ple.?.chunk(start, cn, ws.ple);
                try layerBlock(model, layer, li, ws, cache, xs, alts, ples, rows[start..][0..cn]);
                if (!single and alt_w > 0) try act_alt.?.commit(start, cn, alts);
            }
            if (!single) try act.?.commit(start, cn, xs);
            if (hc == 1 and nb == 0) captureResiduals(opts, li + 1, hidden, start, cn, xs);
        }
        if (kv_layer == li) try cache.endLayer(li);
    }

    // Hyper-connections: the final collapse is the last residual entry.
    if (hc > 1) {
        if (opts.residuals) |res| {
            const row = try gpa.alloc(f32, sw);
            defer gpa.free(row);
            for (opts.capture_rows, 0..) |r, ci| {
                if (single) @memcpy(row, ws.x[r * sw ..][0..sw]) else try act.?.readRow(r, row);
                const pm: ?[]const f32 = if (pre_mix) |p| p[r * hc ..][0..hc] else null;
                try hyper.finalCollapse(model, &head_lease, row, pm, res[(c.num_layers * opts.capture_rows.len + ci) * hidden ..][0..hidden]);
            }
        }
    }

    if (nb > 0) {
        // Output aggregation: the final norm reads the mixture of every banked
        // block prefix and the running one, which becomes the last residual entry.
        const scorer = model.output_res.?;
        start = 0;
        while (start < n) : (start += chunk_rows) {
            const cn = @min(chunk_rows, n - start);
            const xs = if (single) ws.x[0 .. n * hidden] else try act.?.chunk(start, cn, ws.x);
            for (0..cn) |i| {
                const t = start + i;
                attnResMix(c, scorer, bank[t * nb * hidden ..][0 .. nb * hidden], nb, xs[i * hidden ..][0..hidden], mix[i * hidden ..][0..hidden]);
            }
            @memcpy(xs[0 .. cn * hidden], mix[0 .. cn * hidden]);
            if (!single) try act.?.commit(start, cn, xs);
            captureResiduals(opts, c.num_layers, hidden, start, cn, xs);
        }
    }

    // Final norm + logits for requested rows.
    if (opts.logit_rows.len > 0) {
        std.debug.assert(opts.logit_rows.len <= ws.max_logit_rows);
        // `ws.h` holds `max_rows` rows, which may be fewer than the logit rows under chunking.
        const h = try gpa.alloc(f32, opts.logit_rows.len * hidden);
        defer gpa.free(h);
        const unembed = try gpa.alloc(stream.Lease, model.altup_unembed_refs.len);
        defer gpa.free(unembed);
        var n_unembed: usize = 0;
        defer for (unembed[0..n_unembed]) |l| store.release(l);
        for (model.altup_unembed_refs) |r| {
            unembed[n_unembed] = try store.acquire(r);
            n_unembed += 1;
        }
        const alt_row = try gpa.alloc(f32, alt_w);
        defer gpa.free(alt_row);
        const tmp = try gpa.alloc(f32, sw);
        defer gpa.free(tmp);
        for (opts.logit_rows, 0..) |r, i| {
            const dst = h[i * hidden ..][0..hidden];
            var src: []const f32 = undefined;
            if (single) {
                src = ws.x[r * sw ..][0..sw];
            } else {
                try act.?.readRow(r, tmp);
                src = tmp;
            }
            if (alt_w > 0) {
                // Gemma 3n: mean of the active stream and the unembedded extra streams.
                const alts: []const f32 = if (single) ws.alt[r * alt_w ..][0..alt_w] else blk: {
                    try act_alt.?.readRow(r, alt_row);
                    break :blk alt_row;
                };
                try altupFold(model, unembed[0..n_unembed], src[0..hidden], alts, ws.h2[0..hidden], ws.h[0..hidden]);
                src = ws.h2[0..hidden];
            }
            if (hc > 1) {
                const pm: ?[]const f32 = if (pre_mix) |p| p[r * hc ..][0..hc] else null;
                try hyper.finalCollapse(model, &head_lease, src, pm, ws.h2[0..hidden]);
                src = ws.h2[0..hidden];
            }
            if (model.final_norm) |nm| applyNorm(c, dst, src, nm) else @memcpy(dst, src[0..hidden]);
        }
        const lm = try model.acquireLmHead();
        defer store.release(lm);
        try compute.matmulT(model.pool, gpa, ws.logits, h, opts.logit_rows.len, lm.weight, null);
        const logits = ws.logits[0 .. opts.logit_rows.len * c.vocab_size];
        if (model.lm_head_bias) |b| addBias(logits, opts.logit_rows.len, c.vocab_size, b);
        if (c.logit_scale != 1.0) tensor.scale(logits, c.logit_scale);
        if (c.final_logit_softcapping) |cap| compute.softcap(logits, cap);
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
/// rows `x` (`rows.len` tokens). `layer` must hold resident weights. `alt`
/// carries the extra AltUp streams and `ple` the per-layer inputs of the rows
/// (both empty when the family has none).
///
/// Sequential layout: `x += attn(norm(x)); x += mlp(norm(x))`, with optional
/// norms on the sublayer outputs (Gemma, OLMo 2, GLM-4) and no input norm for
/// the post-norm families. Parallel layout (GPT-NeoX, Falcon, Phi, Cohere):
/// `x += attn(h) + mlp(h)` with `h = norm(x)` (or a second norm for the MLP).
///
/// The "attention" half may be a linear-attention, short-convolution or
/// Mamba block instead, or (Falcon-H1) a Mamba block and attention side by
/// side on the same input whose outputs are summed; single-block layers
/// (Mamba2, Nemotron-H) run only one of the two halves behind the layer
/// norm. Gemma 4 then adds the per-layer input block and its output scalar;
/// Gemma 3n takes the AltUp path.
fn layerBlock(model: *const Model, layer: *const Layer, li: usize, ws: *Workspace, cache: *KvCache, x: []f32, alt: []f32, ple: []const f32, rows: []const Row) !void {
    if (layer.altup != null) return altupLayer(model, layer, li, ws, cache, x, alt, ple, rows);
    const c = &model.config;
    const gpa = model.gpa;
    const n = rows.len;
    const hidden = c.hidden_size;
    const h = ws.h[0 .. n * hidden];
    const rm = c.residual_multiplier;
    const has_attn = c.attn_layers[li];
    const has_ssm = c.ssm_layers[li];
    const has_mixer = has_attn or has_ssm or c.linear_layers[li] or c.conv_layers[li];
    const has_mlp = c.mlp_layers[li];
    const attn_out = ws.o[0 .. n * hidden];

    if (has_mixer) {
        normRows(c, h, x, n, hidden, layer.input_norm);
        if (has_attn and has_ssm) {
            // Parallel Mamba + attention with the muP multipliers.
            var attn_in: []const f32 = h;
            var scaled: ?[]f32 = null;
            defer if (scaled) |b| gpa.free(b);
            if (c.mult.attn_in != 1.0) {
                const b = try gpa.alloc(f32, n * hidden);
                @memcpy(b, h);
                tensor.scale(b, c.mult.attn_in);
                scaled = b;
                attn_in = b;
            }
            try attention(model, layer, li, ws, cache, attn_in, rows);
            if (c.mult.attn_out != 1.0) tensor.scale(attn_out, c.mult.attn_out);
            const ssm_out = try gpa.alloc(f32, n * hidden);
            defer gpa.free(ssm_out);
            try ssmForward(model, layer, li, cache, h, rows, ssm_out);
            tensor.axpy(attn_out, c.mult.ssm_out, ssm_out);
        } else if (has_ssm) {
            try ssmForward(model, layer, li, cache, h, rows, attn_out);
        } else {
            try mixer(model, layer, li, ws, cache, h, rows);
        }
        if (layer.post_attn_norm) |nm| normRowsInPlace(c, attn_out, n, hidden, nm, ws.h2);
    }

    if (c.residual_layout == .minimax) {
        // MiniMax-01: the normalised input is the residual of each sublayer.
        const sa = c.minimax_scales[if (c.linear_layers[li]) 1 else 0];
        for (x[0 .. n * hidden], h, attn_out) |*xv, hv, av| xv.* = sa[0] * hv + sa[1] * av;
        normRows(c, h, x, n, hidden, layer.pre_ff_norm);
        try mlpBlock(model, layer, li, ws, h, n, null);
        const sm = c.minimax_scales[2];
        for (x[0 .. n * hidden], h, ws.m[0 .. n * hidden]) |*xv, hv, mv| xv.* = sm[0] * hv + sm[1] * mv;
        return;
    }
    if (c.parallel_residual) {
        var mlp_in: []const f32 = h;
        if (layer.mlp_norm) |nm| {
            normRows(c, ws.h2, x, n, hidden, nm);
            mlp_in = ws.h2[0 .. n * hidden];
        }
        try mlpBlock(model, layer, li, ws, mlp_in, n, null);
        const m = ws.m[0 .. n * hidden];
        if (layer.post_ff_norm) |nm| normRowsInPlace(c, m, n, hidden, nm, ws.h2);
        tensor.axpy(x[0 .. n * hidden], rm, attn_out);
        tensor.axpy(x[0 .. n * hidden], rm, m);
    } else {
        if (has_mixer) tensor.axpy(x[0 .. n * hidden], rm, attn_out);
        if (has_mlp) {
            // A single-block MLP layer is normalised by the layer's only norm.
            normRows(c, h, x, n, hidden, if (has_mixer) layer.pre_ff_norm else layer.input_norm);
            try mlpBlock(model, layer, li, ws, h, n, null);
            const m = ws.m[0 .. n * hidden];
            if (layer.post_ff_norm) |nm| normRowsInPlace(c, m, n, hidden, nm, ws.h2);
            tensor.axpy(x[0 .. n * hidden], rm, m);
        }
    }
    if (layer.ple) |*p| {
        try pleBlock(model, p, li, ws, x, ple, n);
        tensor.axpy(x[0 .. n * hidden], 1.0, ws.m[0 .. n * hidden]);
    }
    if (layer.layer_scale) |s| tensor.scale(x[0 .. n * hidden], s);
}

/// The sequence-mixing sublayer of a layer: attention, one of the linear
/// attention recurrences or a short convolution. Reads `h`, writes `ws.o`.
pub fn mixer(model: *const Model, layer: *const Layer, li: usize, ws: *Workspace, cache: *KvCache, h: []const f32, rows: []const Row) !void {
    const c = &model.config;
    if (c.linear_layers[li]) {
        return switch (c.linear_kind) {
            .gated_deltanet => linearForward(model, layer, li, ws, cache, h, rows),
            .kda => kdaForward(model, layer, li, ws, cache, h, rows),
            .lightning => lightningForward(model, layer, li, ws, cache, h, rows),
        };
    }
    if (c.conv_layers[li]) return convForward(model, layer, li, ws, cache, h, rows);
    return attention(model, layer, li, ws, cache, h, rows);
}

/// AltUp modality router: `tanh(router(norm(x) / hidden))` for `n` rows into `out` (`[n][altup_inputs]`).
fn altupModalities(model: *const Model, au: *const AltUpWeights, ws: *Workspace, out: []f32, x: []const f32, n: usize) !void {
    const c = &model.config;
    const hidden = c.hidden_size;
    const tmp = ws.h2[0 .. n * hidden];
    normRows(c, tmp, x, n, hidden, au.router_norm);
    tensor.scale(tmp, 1.0 / @as(f32, @floatFromInt(hidden)));
    try compute.matmulT(model.pool, model.gpa, out, tmp, n, au.router, null);
    for (out[0 .. n * c.altup_inputs]) |*v| v.* = std.math.tanh(v.*);
}

/// A Gemma 3n layer over its `altup_inputs` residual streams (`x` is stream
/// 0, `alt` the others): predict every stream from the active one, run
/// attention + Laurel and the MLP on the active prediction, correct every
/// stream by the innovation, then add the gated per-layer input to the
/// non-active streams.
fn altupLayer(model: *const Model, layer: *const Layer, li: usize, ws: *Workspace, cache: *KvCache, x: []f32, alt: []f32, ple: []const f32, rows: []const Row) !void {
    const c = &model.config;
    const gpa = model.gpa;
    const au = layer.altup.?;
    const n = rows.len;
    const hidden = c.hidden_size;
    const na = c.altup_inputs;
    const active = c.altup_active;
    const alt_w = (na - 1) * hidden;
    const inv_sqrt2: f32 = 1.0 / @sqrt(2.0);

    // Streams of row r: stream 0 is x[r], stream a >= 1 is alt[r][a - 1].
    const streamOf = struct {
        fn get(xx: []f32, aa: []f32, hid: usize, aw: usize, r: usize, a: usize) []f32 {
            return if (a == 0) xx[r * hid ..][0..hid] else aa[r * aw + (a - 1) * hid ..][0..hid];
        }
    };

    // Predict: pred[a] = s[a] + sum_i coef[a][i] * s[i], coefficients from the active stream.
    const act_in = try gpa.alloc(f32, n * hidden);
    defer gpa.free(act_in);
    for (0..n) |r| @memcpy(act_in[r * hidden ..][0..hidden], streamOf.get(x, alt, hidden, alt_w, r, active));
    const modal = try gpa.alloc(f32, n * na);
    defer gpa.free(modal);
    try altupModalities(model, &au, ws, modal, act_in, n);
    const coefs = try gpa.alloc(f32, n * na * na);
    defer gpa.free(coefs);
    try compute.matmulT(model.pool, gpa, coefs, modal, n, au.predict, null);
    const pred = try gpa.alloc(f32, n * na * hidden);
    defer gpa.free(pred);
    for (0..n) |r| {
        for (0..na) |a| {
            const dst = pred[(r * na + a) * hidden ..][0..hidden];
            @memcpy(dst, streamOf.get(x, alt, hidden, alt_w, r, a));
            for (0..na) |i| tensor.axpy(dst, coefs[r * na * na + a * na + i], streamOf.get(x, alt, hidden, alt_w, r, i));
        }
    }

    // Attention and Laurel on the normalised active prediction.
    const h = ws.h[0 .. n * hidden];
    for (0..n) |r| @memcpy(act_in[r * hidden ..][0..hidden], pred[(r * na + active) * hidden ..][0..hidden]);
    normRows(c, h, act_in, n, hidden, layer.input_norm);
    const laurel = try gpa.alloc(f32, n * hidden);
    defer gpa.free(laurel);
    {
        const lr = try gpa.alloc(f32, n * c.laurel_rank);
        defer gpa.free(lr);
        try compute.matmulT(model.pool, gpa, lr, h, n, au.laurel_l, null);
        try compute.matmulT(model.pool, gpa, laurel, lr, n, au.laurel_r, null);
        normRowsInPlace(c, laurel, n, hidden, au.laurel_norm, ws.h2);
        tensor.axpy(laurel, 1.0, h);
    }
    try mixer(model, layer, li, ws, cache, h, rows);
    const attn_out = ws.o[0 .. n * hidden];
    if (layer.post_attn_norm) |nm| normRowsInPlace(c, attn_out, n, hidden, nm, ws.h2);
    // attn_laurel = (active + attn + laurel) / sqrt(2), kept in `act_in`.
    tensor.axpy(act_in, 1.0, attn_out);
    tensor.axpy(act_in, 1.0, laurel);
    tensor.scale(act_in, inv_sqrt2);

    // MLP; `act_in` becomes the activated output.
    normRows(c, h, act_in, n, hidden, layer.pre_ff_norm);
    try mlpBlock(model, layer, li, ws, h, n, null);
    const m = ws.m[0 .. n * hidden];
    if (layer.post_ff_norm) |nm| normRowsInPlace(c, m, n, hidden, nm, ws.h2);
    tensor.axpy(act_in, 1.0, m);

    // Correct: corrected[a] = pred[a] + (activated - pred[active]) * (coef[a] + 1).
    try altupModalities(model, &au, ws, modal, act_in, n);
    const cc = try gpa.alloc(f32, n * na);
    defer gpa.free(cc);
    try compute.matmulT(model.pool, gpa, cc, modal, n, au.correct, null);
    for (0..n) |r| {
        const innovation = ws.h2[r * hidden ..][0..hidden];
        @memcpy(innovation, act_in[r * hidden ..][0..hidden]);
        tensor.axpy(innovation, -1.0, pred[(r * na + active) * hidden ..][0..hidden]);
        for (0..na) |a| {
            const dst = streamOf.get(x, alt, hidden, alt_w, r, a);
            @memcpy(dst, pred[(r * na + a) * hidden ..][0..hidden]);
            tensor.axpy(dst, cc[r * na + a] + 1.0, innovation);
        }
    }

    // Per-layer input from the (scaled) corrected active stream, added to the other streams.
    for (0..n) |r| {
        const src = act_in[r * hidden ..][0..hidden];
        @memcpy(src, streamOf.get(x, alt, hidden, alt_w, r, active));
        if (au.scale) |s| for (src, 0..) |*v, j| {
            v.* *= s[j];
        };
    }
    try pleBlock(model, &layer.ple.?, li, ws, act_in, ple, n);
    for (0..n) |r| {
        for (1..na) |a| tensor.axpy(streamOf.get(x, alt, hidden, alt_w, r, a), 1.0, ws.m[r * hidden ..][0..hidden]);
    }
}

/// Largest Attention Residual bank (`ceil(num_layers / attn_res_block)` rows) the mixer handles.
pub const max_attn_res_rows = 255;

/// Attention Residual aggregation of one token: mixes the `nvb` banked rows
/// and the running prefix `p` by the softmax of their scores under `s`
/// (`AttnResScorer`), writing the pre-norm mixture to `out`.
fn attnResMix(c: *const Config, s: AttnResScorer, bank: []const f32, nvb: usize, p: []const f32, out: []f32) void {
    const hidden = c.hidden_size;
    if (nvb == 0) {
        @memcpy(out, p);
        return;
    }
    std.debug.assert(nvb <= max_attn_res_rows);
    var scores: [max_attn_res_rows + 1]f32 = undefined;
    for (0..nvb) |j| scores[j] = attnResScore(c, s, bank[j * hidden ..][0..hidden]);
    scores[nvb] = attnResScore(c, s, p);
    compute.softmaxInPlace(scores[0 .. nvb + 1]);
    @memset(out, 0);
    for (0..nvb) |j| tensor.axpy(out, scores[j], bank[j * hidden ..][0..hidden]);
    tensor.axpy(out, scores[nvb], p);
}

/// `rmsnorm(row; s.norm) · s.proj`, the retrieval score of one candidate row.
fn attnResScore(c: *const Config, s: AttnResScorer, row: []const f32) f32 {
    var ss: f32 = 0;
    var d: f32 = 0;
    for (row, 0..) |v, i| {
        ss += v * v;
        d += v * s.norm[i] * s.proj[i];
    }
    return d / @sqrt(ss / @as(f32, @floatFromInt(row.len)) + c.rms_norm_eps);
}

/// One transformer layer of an Attention Residual model (Kimi K3), applied
/// to the running block prefix `x` of `rows.len` tokens with their banked
/// block prefixes `bank` (`[token][nb][hidden]`).
///
/// Instead of one accumulated residual stream, each sublayer reads a
/// softmax-weighted mixture (`attnResMix`) of the prefixes banked at the
/// start of every earlier block and the running prefix of the current block:
///
///     mix = aggregate(bank[0..nvb], x)      // what the attention reads: this layer's residual
///     if li % B == 0: bank[nvb] = x; x = 0  // a block boundary banks the prefix and restarts it
///     x += attn(input_norm(mix))
///     x += mlp(pre_ff_norm(aggregate(bank, x)))
///
/// `mix` receives the attention-side mixture, which is what ditch treats as
/// the layer's residual for direction extraction (entry `li`; the output
/// aggregation is entry `num_layers`). The scorers (`*_res_norm`,
/// `*_res_proj`) are never edited.
fn layerBlockAttnRes(model: *const Model, layer: *const Layer, li: usize, ws: *Workspace, cache: *KvCache, x: []f32, rows: []const Row, bank: []f32, mix: []f32) !void {
    const c = &model.config;
    const n = rows.len;
    const hidden = c.hidden_size;
    const B = c.attn_res_block;
    const nb = bank.len / (n * hidden);
    const write = li % B == 0;
    var nvb = (li + B - 1) / B;
    const ar = layer.attn_res.?;
    const h = ws.h[0 .. n * hidden];
    const h2 = ws.h2[0 .. n * hidden];

    for (0..n) |i| attnResMix(c, ar.attn, bank[i * nb * hidden ..][0 .. nvb * hidden], nvb, x[i * hidden ..][0..hidden], mix[i * hidden ..][0..hidden]);
    normRows(c, h, mix, n, hidden, layer.input_norm);
    if (write) {
        for (0..n) |i| @memcpy(bank[(i * nb + nvb) * hidden ..][0..hidden], x[i * hidden ..][0..hidden]);
        nvb += 1;
    }
    if (c.linear_layers[li]) {
        switch (c.linear_kind) {
            .gated_deltanet => try linearForward(model, layer, li, ws, cache, h, rows),
            .kda => try kdaForward(model, layer, li, ws, cache, h, rows),
            .lightning => try lightningForward(model, layer, li, ws, cache, h, rows),
        }
    } else {
        try attention(model, layer, li, ws, cache, h, rows);
    }
    const attn_out = ws.o[0 .. n * hidden];
    if (write) @memcpy(x[0 .. n * hidden], attn_out) else tensor.axpy(x[0 .. n * hidden], 1.0, attn_out);

    for (0..n) |i| attnResMix(c, ar.mlp, bank[i * nb * hidden ..][0 .. nvb * hidden], nvb, x[i * hidden ..][0..hidden], h2[i * hidden ..][0..hidden]);
    normRows(c, h, h2, n, hidden, layer.pre_ff_norm);
    try mlpBlock(model, layer, li, ws, h, n, null);
    tensor.axpy(x[0 .. n * hidden], 1.0, ws.m[0 .. n * hidden]);
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
