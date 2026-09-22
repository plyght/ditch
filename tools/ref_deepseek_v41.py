"""DeepSeek V4.1's own inference code as a CPU reference for a (truncated) released checkpoint.

    python3 tools/probe_reference.py MODEL probe.json --dtype float32 --raw --factory tools/ref_deepseek_v41.py

transformers has no DeepSeek V4.1, so the reference is the one the release
ships: `inference/model.py` and `inference/engram.py` from
deepseek-ai/DeepSeek-V4.1-Flash, fetched here and imported unmodified. Only
what cannot run on a CPU is replaced:

* `kernel.py` (tilelang, CUDA) by torch functions with the same arithmetic:
  `act_quant` / `fp4_act_quant` in their in-place (quantise-dequantise) form,
  which is how the model calls them on the window KV, the compressed latents
  and the indexer (power-of-two scales by `ceil(log2(amax / max))`, the
  E4M3-rounded scales of the latents, the amax floors, round-to-nearest-even
  onto the e4m3 and e2m1 grids); `sparse_attn` (gathered keys, sink, the
  `-1e30` floor for rows with no key); `hc_split_sinkhorn`.
* `linear()`: the release quantises every linear's activations to FP8 before
  its FP8 / FP4 GEMM. ditch computes linears in float on the dequantised
  weights, as for every FP8 checkpoint (see the second pass), so the
  reference does too: weights are dequantised here (e4m3 / e2m1 codes times
  the ue8m0 block scales) and `linear` is `F.linear`. The fake quantisation
  the model applies explicitly (window KV, latents, indexer) stays.
* The routed experts, the shared expert and the engram table are read lazily
  (tools/lazy_checkpoint.py): only the experts a token is routed to and the
  table rows a prompt hashes to are fetched, so a cut whose experts are holes
  of a sparse file is filled with exactly those.

The residual reported per layer is what ditch reports: the streams after the
layer's engram, collapsed by the pre-mix the layer's attention reads.
"""
import json
import os
import sys
import types

import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lazy_checkpoint import LazyCheckpoint  # noqa: E402
from ref_deepseek_v4 import dequant  # noqa: E402

REPO = "deepseek-ai/DeepSeek-V4.1-Flash"

# ---------------------------------------------------------------------------
# kernel.py on the CPU
# ---------------------------------------------------------------------------

E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def pow2_ceil_log2(t):
    """`fast_pow2(fast_log2_ceil(t))`: 2^ceil(log2(t)) from the float's bits."""
    bits = t.float().contiguous().view(torch.int32)
    e = ((bits >> 23) & 0xFF) - 127 + ((bits & 0x7FFFFF) != 0).int()
    return torch.exp2(e.float())


def to_e2m1(v):
    """Round onto the e2m1 grid, to nearest, ties to the even code (cvt.rn)."""
    a = v.abs().clamp(max=6.0)
    d = (a.unsqueeze(-1) - E2M1).abs()
    best = d.min(-1, keepdim=True).values
    cand = (d == best)
    # ties: prefer the even code (codes are the grid indices)
    even = cand & (torch.arange(8) % 2 == 0)
    code = torch.where(even.any(-1), even.float().argmax(-1), cand.float().argmax(-1))
    return torch.copysign(E2M1[code], v)


def act_quant(x, block_size=128, scale_fmt=None, scale_dtype=torch.float32, inplace=False):
    assert inplace, "linear() is replaced; only the in-place fake quantisation is called"
    xb = x.float().unflatten(-1, (-1, block_size))
    amax = xb.abs().amax(-1, keepdim=True).clamp(min=1e-4)
    t = amax * torch.tensor(1 / 448.0, dtype=torch.float32)
    s = pow2_ceil_log2(t) if scale_fmt is not None else t
    y = (xb / s).clamp(-448.0, 448.0).to(torch.float8_e4m3fn).float() * s
    x.copy_(y.flatten(-2))
    return x


def fp4_act_quant(x, block_size=32, inplace=False, scale_dtype=torch.float8_e8m0fnu):
    assert inplace
    xb = x.float().unflatten(-1, (-1, block_size))
    amax = xb.abs().amax(-1, keepdim=True)
    if scale_dtype == torch.float8_e4m3fn:
        amax = amax.clamp(min=6 * 2.0**-9)
        s = (amax / 6.0).to(torch.float8_e4m3fn).float()
    else:
        amax = amax.clamp(min=6 * 2.0**-126)
        s = pow2_ceil_log2(amax * torch.tensor(1 / 6.0, dtype=torch.float32))
    y = to_e2m1((xb / s).clamp(-6.0, 6.0)) * s
    x.copy_(y.flatten(-2))
    return x


def sparse_attn(q, kv, attn_sink, topk_idxs, softmax_scale):
    b, m, h, d = q.shape
    idx = topk_idxs.long()
    valid = idx >= 0
    g = kv[torch.arange(b)[:, None, None], idx.clamp(min=0)]  # [b, m, k, d]
    g = torch.where(valid.unsqueeze(-1), g, 0.0)
    s = torch.einsum("bmhd,bmkd->bmhk", q.float(), g.float()) * softmax_scale
    s = s.masked_fill(~valid.unsqueeze(2), float("-inf"))
    mx = s.amax(-1, keepdim=True).clamp(min=-1e30)
    p = torch.exp(s - mx)
    denom = p.sum(-1, keepdim=True) + torch.exp(attn_sink.float().view(1, 1, h, 1) - mx)
    return (torch.einsum("bmhk,bmkd->bmhd", p, g.float()) / denom).to(q.dtype)


def hc_split_sinkhorn(mixes, hc_scale, hc_base, hc_mult=4, sinkhorn_iters=20, eps=1e-6):
    hc = hc_mult
    pre = torch.sigmoid(mixes[..., :hc] * hc_scale[0] + hc_base[:hc]) + eps
    post = 2 * torch.sigmoid(mixes[..., hc:2 * hc] * hc_scale[1] + hc_base[hc:2 * hc])
    comb = (mixes[..., 2 * hc:] * hc_scale[2] + hc_base[2 * hc:]).unflatten(-1, (hc, hc))
    comb = torch.softmax(comb, dim=-1) + eps
    comb = comb / (comb.sum(-2, keepdim=True) + eps)
    for _ in range(sinkhorn_iters - 1):
        comb = comb / (comb.sum(-1, keepdim=True) + eps)
        comb = comb / (comb.sum(-2, keepdim=True) + eps)
    return pre, post, comb


def _unused(*a, **k):
    raise RuntimeError("quantised GEMMs are not used by the CPU reference")


def import_reference():
    src = os.path.dirname(hf_hub_download(REPO, "inference/model.py"))
    hf_hub_download(REPO, "inference/engram.py")
    kernel = types.ModuleType("kernel")
    kernel.act_quant, kernel.fp4_act_quant, kernel.sparse_attn = act_quant, fp4_act_quant, sparse_attn
    kernel.hc_split_sinkhorn, kernel.fp8_gemm, kernel.fp4_gemm = hc_split_sinkhorn, _unused, _unused
    vision = types.ModuleType("vision")
    vision.ViT = vision.Aligner = None
    image_processor = types.ModuleType("image_processor")
    image_processor.IMAGE, image_processor.IMAGE_END, image_processor.IMAGE_NEW_LINE, image_processor.IMAGE_START = 0, 1, 2, 3
    sys.modules.update({"kernel": kernel, "vision": vision, "image_processor": image_processor})
    sys.path.insert(0, src)
    import model as ref  # noqa: E402  (DeepSeek's inference/model.py)
    return ref


# ---------------------------------------------------------------------------
# config.json (transformers spelling) -> ModelArgs (inference/config.json spelling)
# ---------------------------------------------------------------------------

def model_args(ref, tc):
    rs = tc.get("rope_scaling") or {}
    return ref.ModelArgs(
        max_batch_size=1, max_seq_len=2048, dtype="bf16", expert_dtype=None,
        vocab_size=tc["vocab_size"], dim=tc["hidden_size"], moe_inter_dim=tc["moe_intermediate_size"],
        n_layers=tc["num_hidden_layers"], n_mtp_layers=0, n_heads=tc["num_attention_heads"],
        n_routed_experts=tc["n_routed_experts"], n_shared_experts=tc["n_shared_experts"],
        n_activated_experts=tc["num_experts_per_tok"], score_func=tc["scoring_func"],
        gate_temp=tc.get("gate_temp", 1.0), norm_topk_prob=tc.get("norm_topk_prob", True),
        route_scale=tc["routed_scaling_factor"], swiglu_limit=tc["swiglu_limit"], q_lora_rank=tc["q_lora_rank"],
        head_dim=tc["head_dim"], rope_head_dim=tc["qk_rope_head_dim"], norm_eps=tc["rms_norm_eps"],
        o_groups=tc["o_groups"], o_lora_rank=tc["o_lora_rank"], window_size=tc["sliding_window"],
        compress_ratios=tuple(tc["compress_ratios"][: tc["num_hidden_layers"]]),
        kv_source_layers=tuple(tc["kv_source_layer_ids"]), index_source_layers=tuple(tc["index_source_layer_ids"]),
        compress_rope_theta=tc["compress_rope_theta"], original_seq_len=rs.get("original_max_position_embeddings", 0),
        rope_theta=tc["rope_theta"], rope_factor=rs.get("factor", 40), beta_fast=rs.get("beta_fast", 32),
        beta_slow=rs.get("beta_slow", 1), index_n_heads=tc["index_n_heads"], index_head_dim=tc["index_head_dim"],
        index_topk=tc["index_topk"], candidate_source_layer=tc.get("candidate_source_layer_id", -1),
        candidate_topk_blocks=tc.get("candidate_topk_blocks", 0), candidate_block_size=tc.get("candidate_block_size", 0),
        hc_mult=tc["hc_mult"], hc_sinkhorn_iters=tc["hc_sinkhorn_iters"], hc_eps=tc["hc_eps"],
        engram_layer_ids=tuple(tc.get("engram_layer_ids", ())), engram_num_embeddings=tuple(tc.get("engram_num_embeddings", ())),
        engram_max_ngram_size=tc.get("engram_max_ngram_size", 1), engram_vocab_size=tc.get("engram_vocab_size", 0),
        engram_n_heads=tc.get("engram_n_heads", 0), engram_head_dim=tc.get("engram_head_dim", 0),
        engram_pad_id=tc.get("engram_pad_token_id", 2), engram_compressed_vocab_size=tc.get("engram_compressed_vocab_size", 0),
        vision_n_layers=0, dspark_block_size=0, dspark_target_layer_ids=(),
    )


# ---------------------------------------------------------------------------
# Lazy experts and engram table
# ---------------------------------------------------------------------------

def make_lazy(ref, store):
    class LazyExpert(torch.nn.Module):
        """`Expert.forward` with w1 / w3 / w2 read (and dequantised) on use."""

        def __init__(self, dim, inter_dim, dtype=None, swiglu_limit=0.0):
            super().__init__()
            self.swiglu_limit = swiglu_limit
            self.prefix = None

        def w(self, n):
            name = f"{self.prefix}.{n}.weight"
            t = store.tensor(name)
            sname = f"{self.prefix}.{n}.scale"
            if sname in store.keys():
                t = dequant(t, store.tensor(sname))
            return t.float()

        def forward(self, x, weights=None):
            dtype = x.dtype
            gate = F.linear(x, self.w("w1")).float()
            up = F.linear(x, self.w("w3")).float()
            if self.swiglu_limit > 0:
                up = torch.clamp(up, min=-self.swiglu_limit, max=self.swiglu_limit)
                gate = torch.clamp(gate, max=self.swiglu_limit)
            x = F.silu(gate) * up
            if weights is not None:
                x = weights * x
            return F.linear(x.to(dtype), self.w("w2"))

    class LazyEngramEmbedding(torch.nn.Module):
        """`ParallelEngramEmbedding.forward` (world size 1), reading only the hashed rows."""

        def __init__(self, num_embeddings, dim):
            super().__init__()
            self.num_embeddings, self.dim, self.block_size = num_embeddings, dim, ref.fp8_block_size
            self.name = None

        def forward(self, indices):
            flat = indices.reshape(-1)
            uniq, inv = torch.unique(flat, return_inverse=True)
            w = store.rows(self.name + ".weight", uniq)
            s = store.rows(self.name + ".scale", uniq)
            values = w.float().unflatten(-1, (-1, self.block_size)) * s.float().unsqueeze(-1)
            values = values.flatten(-2).to(torch.bfloat16).float()
            return values[inv].reshape(*indices.shape, self.dim)

    return LazyExpert, LazyEngramEmbedding


class Reference(torch.nn.Module):
    """The interface tools/probe_reference.py needs: logits, per-layer residuals, greedy decoding."""

    def __init__(self, ref, model, store):
        super().__init__()
        self.ref, self.model, self.store = ref, model, store

    def run(self, ids):
        m = self.model
        hashes = m.engram_hash(ids, 0) if m.engram_hash is not None else None
        h = m.embed(ids).float()
        h = h.unsqueeze(2).repeat(1, 1, m.hc_mult, 1)
        pre_mix = self.ref.make_identity_pre_mix(h, m.hc_mult)
        residuals = []
        for layer in m.layers:
            if layer.engram is not None:
                h = layer.engram(h, hashes[:, :, layer.engram.layer_hash_index, :])
            residuals.append(layer.hc_pre(h, pre_mix))
            h, pre_mix = layer(h, 0, pre_mix, None)
        final = m.layers[-1].hc_pre(h, pre_mix)
        logits = m.head(m.norm(final), full_logits=True)
        self.store.flush()
        return logits, residuals + [final]

    def forward(self, input_ids, output_hidden_states=False):
        with torch.inference_mode():
            logits, hidden = self.run(input_ids)
        return types.SimpleNamespace(logits=logits, hidden_states=hidden if output_hidden_states else None)

    def generate(self, input_ids, max_new_tokens=8, do_sample=False):
        ids = input_ids
        for _ in range(max_new_tokens):
            with torch.inference_mode():
                logits, _ = self.run(ids)
            ids = torch.cat([ids, logits[:, -1].argmax(-1, keepdim=True)], dim=1)
        return ids


def load(model_dir, dtype=torch.float32):
    from transformers import AutoTokenizer

    ref = import_reference()
    store = LazyCheckpoint(model_dir)
    cfg = json.load(open(os.path.join(model_dir, "config.json")))
    tc = cfg.get("text_config", cfg)
    args = model_args(ref, tc)
    LazyExpert, LazyEngramEmbedding = make_lazy(ref, store)
    ref.Expert, ref.ParallelEngramEmbedding = LazyExpert, LazyEngramEmbedding
    tok = AutoTokenizer.from_pretrained(model_dir)
    torch.set_default_dtype(torch.float32)
    model = ref.Transformer(args, tok)
    ref.linear = lambda x, weight, bias=None: F.linear(x.float(), weight.float())
    # Everything but the lazy modules, under DeepSeek's own names (the model's
    # parameter names), one parameter at a time so the float32 copies replace
    # the placeholders instead of sitting next to them.
    with torch.no_grad():
        for name, p in model.named_parameters():
            t = store.tensor(name)
            base = name[: -len(".weight")] if name.endswith(".weight") else None
            if base and base + ".scale" in store.keys() and t.dtype in (torch.int8, torch.float8_e4m3fn):
                t = dequant(t, store.tensor(base + ".scale"))
            if tuple(t.shape) != tuple(p.shape):
                raise SystemExit(f"reference: {name} is {tuple(t.shape)} in the checkpoint, {tuple(p.shape)} in the model")
            p.data = t.float()
    for i, layer in enumerate(model.layers):
        for e, ex in enumerate(layer.ffn.experts):
            ex.prefix = f"layers.{i}.ffn.experts.{e}"
        layer.ffn.shared_experts.prefix = f"layers.{i}.ffn.shared_experts"
        if layer.engram is not None:
            layer.engram.embed.name = f"layers.{i}.engram.embed"
    model.eval()
    return Reference(ref, model, store)
