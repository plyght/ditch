"""A transformers DeepSeek V4 reference for a (truncated) checkpoint as released.

    python3 tools/probe_reference.py MODEL probe.json --dtype float32 --factory tools/ref_deepseek_v4.py

The released V4 checkpoints are in DeepSeek's own naming with FP8 (ue8m0 block
scales) dense weights and FP4 (e2m1, one ue8m0 scale per 32 columns) routed
experts. transformers reads them through its `conversion_mapping` renames and
its fine-grained FP8 integration; this does the same on the CPU without
materialising every expert: the names go through transformers' own
`WeightRenaming` list, every FP8 / FP4 tensor is dequantised here in torch
(transformers' `_dequantize_one` arithmetic: the e2m1 table, `2^(byte - 127)`
scales, block-wise), and `DeepseekV4Experts` is replaced by a module with the
same forward that fetches and dequantises only the experts a token is routed
to. Everything else is transformers' `modeling_deepseek_v4.py`, unmodified.
"""
import json
import os

import torch
import torch.nn.functional as F
from safetensors import safe_open
from transformers import conversion_mapping as cm
from transformers.models.deepseek_v4 import modeling_deepseek_v4 as mod
from transformers.models.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config

FP4 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0])


def e8m0(t):
    return torch.exp2(t.view(torch.uint8).to(torch.float32) - 127.0)


def dequant(w, s):
    """FP8 (`float8_e4m3fn`) or FP4 (int8 nibble pairs, low nibble first) with ue8m0 block scales."""
    if w.dtype == torch.int8:
        u8 = w.view(torch.uint8)
        q = torch.stack([FP4[(u8 & 0xF).long()], FP4[(u8 >> 4).long()]], dim=-1).reshape(*w.shape[:-1], 2 * w.shape[-1])
    else:
        q = w.to(torch.float32)
    sc = e8m0(s)
    br, bc = q.shape[-2] // sc.shape[-2], q.shape[-1] // sc.shape[-1]
    return q * sc.repeat_interleave(br, dim=-2).repeat_interleave(bc, dim=-1)


class Store:
    def __init__(self, path):
        self.f = safe_open(path, "pt")
        self.keys = set(self.f.keys())

    def get(self, name, dtype):
        base = name[: -len(".weight")] if name.endswith(".weight") else name
        t = self.f.get_tensor(name)
        if base + ".scale" in self.keys and t.dtype in (torch.int8, torch.float8_e4m3fn):
            t = dequant(t, self.f.get_tensor(base + ".scale"))
        return t.to(dtype) if t.is_floating_point() else t


class LazyExperts(torch.nn.Module):
    """`DeepseekV4Experts.forward`, reading expert `e`'s w1 / w3 / w2 on demand."""

    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_local_experts
        self.act_fn = mod.ACT2FN[config.hidden_act]
        self.limit = config.swiglu_limit
        self.store = None
        self.prefix = None
        self.dtype = torch.float32
        # Empty stand-ins (plain attributes, not parameters) for the weight
        # init of `DeepseekV4PreTrainedModel._init_weights`.
        self.gate_up_proj = torch.empty(0)
        self.down_proj = torch.empty(0)

    def weights(self, e):
        p = f"{self.prefix}.experts.{e}."
        w1 = self.store.get(p + "w1.weight", self.dtype)
        w3 = self.store.get(p + "w3.weight", self.dtype)
        w2 = self.store.get(p + "w2.weight", self.dtype)
        return torch.cat([w1, w3], dim=0), w2  # transformers' MergeModulelist + Concatenate(dim=1)

    _apply_gate = mod.DeepseekV4Experts._apply_gate

    def forward(self, hidden_states, top_k_index, top_k_weights):
        final = torch.zeros_like(hidden_states)
        with torch.no_grad():
            mask = F.one_hot(top_k_index, num_classes=self.num_experts).permute(2, 1, 0)
            hit = torch.greater(mask.sum(dim=(-1, -2)), 0).nonzero()
        for expert_idx in hit:
            expert_idx = int(expert_idx[0])
            top_k_pos, token_idx = torch.where(mask[expert_idx])
            gate_up, down = self.weights(expert_idx)
            current = self._apply_gate(F.linear(hidden_states[token_idx], gate_up))
            current = F.linear(current, down) * top_k_weights[token_idx, top_k_pos, None]
            final.index_add_(0, token_idx, current.to(final.dtype))
        return final


def load(model_dir, dtype=torch.float32):
    cfg = json.load(open(os.path.join(model_dir, "config.json")))
    cfg.pop("quantization_config", None)
    cfg.pop("expert_dtype", None)
    config = DeepseekV4Config(**{k: v for k, v in cfg.items() if k not in ("architectures", "model_type", "torch_dtype", "transformers_version")})
    config._experts_implementation = "eager"
    orig = mod.DeepseekV4Experts
    mod.DeepseekV4Experts = LazyExperts
    try:
        torch.set_default_dtype(dtype)
        model = mod.DeepseekV4ForCausalLM(config)
    finally:
        mod.DeepseekV4Experts = orig
        torch.set_default_dtype(torch.float32)
    model.eval()

    store = Store(os.path.join(model_dir, "model.safetensors"))
    renames = [r for r in cm.get_checkpoint_conversion_mapping("deepseek_v4") if type(r).__name__ == "WeightRenaming"]
    want = model.state_dict()
    state, used = {}, set()
    for k in sorted(store.keys):
        if k.endswith(".scale") and k[: -len(".scale")] + ".weight" in store.keys:
            continue
        if ".experts." in k and ".shared_experts." not in k:
            continue
        name = k
        for r in renames:
            name = r.rename_source_key(name)[0]
        for cand in (name, "model." + name):
            if cand in want:
                state[cand] = store.get(k, dtype if want[cand].is_floating_point() else want[cand].dtype)
                used.add(k)
                break
    missing = [k for k in want if k not in state]
    if missing:
        raise SystemExit(f"reference: {len(missing)} parameters not in the checkpoint, e.g. {missing[:8]}")
    unused = [k for k in store.keys if k not in used and not k.endswith(".scale") and ".experts." not in k]
    if unused:
        print(f"reference: {len(unused)} checkpoint tensors unused, e.g. {unused[:8]}")
    model.load_state_dict(state, strict=True)
    for i, layer in enumerate(model.model.layers):
        layer.mlp.experts.store = store
        layer.mlp.experts.prefix = f"layers.{i}.ffn"
        layer.mlp.experts.dtype = dtype
    return model
