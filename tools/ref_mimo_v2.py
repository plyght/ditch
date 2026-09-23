"""MiMo V2.5 / V2.6's own modeling code as a CPU reference for a (truncated) released checkpoint.

    python3 tools/probe_reference.py MODEL probe.json --dtype float32 --raw --trust-remote-code --factory tools/ref_mimo_v2.py

transformers has no `mimo_v2`, so the reference is the release's
`modeling_mimo_v2.py` (MiMoV2ForCausalLM), run unmodified, text only (the
vision and audio towers are not built). It has no dequantisation of its own
(the release is served by engines that dequantise), so the weights are
dequantised here: the fp8 trunk with its `weight_scale_inv` blocks
(`weight_block_size`, rounded to bf16 as transformers' fp8 integration does)
and the routed experts' MXFP4 store (`store_dtype: mxfp4`: e2m1 nibble pairs,
low nibble first, with one E8M0 scale per 32 columns), read on use when the
cut left them lazy. The attention projections are `num_key_value_heads`
tensor-parallel row shards, each fp8-blocked on its own, and the fused
`qkv_proj` is regrouped from them as SGLang's loader does (`fused_qkv`). The trunk is kept in its stored (or dequantised bf16)
values and every Linear / Embedding computes in float32.

The residual reported per layer is what enters the layer (what its
`input_layernorm` reads), then what the final norm reads.
"""
import json
import os
import sys
import types

import torch
import torch.nn.functional as F
from torch import nn
from transformers.dynamic_module_utils import get_class_from_dynamic_module

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ref_lazy_moe  # noqa: E402
from lazy_checkpoint import LazyCheckpoint  # noqa: E402
from ref_kimi_k3 import LazyLinear, Reference  # noqa: E402
from ref_lazy_moe import dequant_expert  # noqa: E402


def fused_qkv(store, name, config, layer):
    """A fused `qkv_proj` as the release's code splits it, `[all q | all k | all v]`.

    The checkpoint holds `num_key_value_heads` tensor-parallel shards, each
    `[q | k | v]` of its heads and fp8-blocked on its own (its own partial last
    block, its own rows of `weight_scale_inv`); SGLang's loader dequantises
    shard by shard (`_resolve_deferred_qkv_scale_inv`) and regroups the parts
    (`_deinterleave_qkv_shards`), and so does this."""
    tp = config.num_key_value_heads
    swa = config.hybrid_layer_pattern[layer] == 1
    nh = config.swa_num_attention_heads if swa else config.num_attention_heads
    nkv = config.swa_num_key_value_heads if swa else config.num_key_value_heads
    hd = config.swa_head_dim if swa else config.head_dim
    vd = getattr(config, "swa_v_head_dim" if swa else "v_head_dim", hd)
    qs, ks, vs = nh // tp * hd, max(1, nkv // tp) * hd, max(1, nkv // tp) * vd
    if name + "_scale_inv" in store.keys():
        w = dequant_expert(store, name[: -len(".weight")], {}).to(torch.bfloat16)  # shard by shard
    else:
        w = store.tensor(name)
    parts = [[], [], []]
    for sh in w.chunk(tp, 0):
        assert sh.shape[0] == qs + ks + vs, (name, sh.shape, qs, ks, vs)
        for i, p in enumerate(sh.split([qs, ks, vs], 0)):
            parts[i].append(p)
    return torch.cat([torch.cat(p, 0) for p in parts], 0)


def load(model_dir, dtype=torch.float32):
    store = LazyCheckpoint(model_dir)
    cfg = json.load(open(os.path.join(model_dir, "config.json")))
    q = cfg.get("quantization_config") or {}
    ref_lazy_moe.FP8_BLOCK = tuple(q["weight_block_size"]) if q.get("weight_block_size") else None
    ref_lazy_moe.ATTN_ROW_SHARDS = cfg.get("num_key_value_heads", 1)
    cls = get_class_from_dynamic_module("modeling_mimo_v2.MiMoV2ForCausalLM", model_dir)
    cfg_cls = get_class_from_dynamic_module("configuration_mimo_v2.MiMoV2Config", model_dir)
    mod = sys.modules[cls.__module__]
    for name in ("create_causal_mask", "create_sliding_window_causal_mask"):
        orig = getattr(mod, name)

        def shim(*a, input_embeds=None, cache_position=None, _orig=orig, **kw):  # transformers 4 spelling
            if input_embeds is not None:
                kw["inputs_embeds"] = input_embeds
            return _orig(*a, **kw)

        setattr(mod, name, shim)
    c = {k: v for k, v in cfg.items() if k != "quantization_config"}
    c["vision_config"] = None
    c["audio_config"] = None
    config = cfg_cls(**c)
    config._attn_implementation = "eager"
    with torch.device("meta"):
        model = cls(config)
    model.eval()
    # Buffers computed at construction (the rotary frequencies) are rebuilt off the meta device.
    rope = mod.MiMoV2RotaryEmbedding
    model.model.rotary_emb = rope(config=config, is_swa=False)
    model.model.swa_rotary_emb = rope(config=config, is_swa=True)

    keys = set(store.keys())
    with torch.no_grad():
        missing = []
        for mname, m in model.named_modules():
            for pname, p in list(m._parameters.items()):
                if p is None:
                    continue
                name = f"{mname}.{pname}" if mname else pname
                if ".mlp.experts." in name:
                    continue
                if name.endswith("self_attn.qkv_proj.weight"):
                    t = fused_qkv(store, name, config, int(name.split(".layers.")[1].split(".")[0]))
                elif pname == "weight" and name + "_scale_inv" in keys:
                    t = dequant_expert(store, mname, {}).to(torch.bfloat16)
                elif name in keys:
                    t = store.tensor(name)
                else:
                    missing.append(name)
                    continue
                if tuple(t.shape) != tuple(p.shape):
                    raise SystemExit(f"reference: {name} is {tuple(t.shape)} in the checkpoint, {tuple(p.shape)} in the model")
                m._parameters[pname] = nn.Parameter(t, requires_grad=False)
        if missing:
            raise SystemExit(f"reference: {len(missing)} parameters not in the checkpoint, e.g. {missing[:6]}")
    # Routed experts: lazy linears under the original expert forward.
    for i, layer in enumerate(model.model.layers):
        experts = getattr(layer.mlp, "experts", None)
        if experts is None:
            continue
        for e, ex in enumerate(experts):
            base = f"model.layers.{i}.mlp.experts.{e}."
            ex.gate_proj, ex.up_proj, ex.down_proj = (LazyLinear(store, base + n) for n in ("gate_proj", "up_proj", "down_proj"))
    left = [n for n, t in list(model.named_parameters()) + list(model.named_buffers()) if t.device.type == "meta"]
    if left:
        raise SystemExit(f"reference: left on the meta device: {left[:6]}")

    def f32_linear(self, x):
        # In blocks of output rows, so the LM head is never copied to float32 all at once.
        x = x.float()
        w = self.weight
        step = max(1, (64 << 20) // (w.shape[1] * 4))
        out = torch.cat([F.linear(x, w[i:i + step].float()) for i in range(0, w.shape[0], step)], dim=-1)
        return out if self.bias is None else out + self.bias.float()

    linears = set()
    for m in model.modules():
        if isinstance(m, nn.Linear):
            linears.add(id(m.weight))
            m.forward = types.MethodType(f32_linear, m)
        elif isinstance(m, nn.Embedding):
            linears.add(id(m.weight))
            m.forward = types.MethodType(lambda self, ids: F.embedding(ids, self.weight).float(), m)
    with torch.no_grad():
        for p in model.parameters():
            if id(p) not in linears and p.is_floating_point():
                p.data = p.data.float()
    return Reference(model, store)
