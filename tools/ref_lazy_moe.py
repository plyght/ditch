"""A transformers reference for a mixture-of-experts cut whose routed experts are lazy.

    python3 tools/probe_reference.py MODEL probe.json --dtype float32 --factory tools/ref_lazy_moe.py

For a checkpoint written by `truncate_checkpoint.py --lazy` (routed experts in
`model-lazy.safetensors`, holes until read): everything else is loaded by
transformers' own `from_pretrained`, with its own renames, converters and
dequantisation, from a view of the directory that leaves the lazy file out.
The family's `*Experts` module (stacked `gate_up_proj` / `down_proj`) is
built on the meta device and its two stacks are replaced by objects whose
`[e]` reads expert `e` on demand (tools/lazy_checkpoint.py fetches it from the
Hub the first time), so transformers' own experts `forward` runs unchanged
and only the experts a token is routed to are ever read.

Per-expert tensors are found by name: `...layers.N.<moe>.experts.E.<proj>`
with <proj> one of w1/w3/w2 (Mixtral spelling) or gate_proj/up_proj/down_proj,
and dequantised here when stored quantised: FP8 with `weight_scale_inv`
blocks, compressed-tensors pack-quantized INT (`weight_packed` I32 +
`weight_scale` [+ `weight_zero_point`]) and mxfp4-pack-quantized
(`weight_packed` U8 + `weight_scale` U8), using compressed-tensors' own
unpacking where it provides it.
"""
import importlib
import json
import os
import re
import sys
import tempfile

import torch
from transformers import AutoConfig, AutoModelForCausalLM

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lazy_checkpoint import LazyCheckpoint  # noqa: E402

E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0])
PROJ = {"gate": ("w1", "gate_proj"), "up": ("w3", "up_proj"), "down": ("w2", "down_proj")}


def dequant_expert(store, module, cfg_q):
    """The float32 `[out, in]` weight of `<module>` (a Linear's prefix), however it is stored."""
    k = store.keys()
    if module + ".weight" in k:
        w = store.tensor(module + ".weight")
        if module + ".weight_scale_inv" in k:  # FP8 blocks
            s = store.tensor(module + ".weight_scale_inv").float()
            br, bc = -(-w.shape[0] // s.shape[0]), -(-w.shape[1] // s.shape[1])
            return w.float() * s.repeat_interleave(br, 0)[: w.shape[0]].repeat_interleave(bc, 1)[:, : w.shape[1]]
        return w.float()
    packed = store.tensor(module + ".weight_packed")
    scale = store.tensor(module + ".weight_scale")
    if packed.dtype == torch.uint8:  # mxfp4-pack-quantized: e2m1 nibble pairs, E8M0 scales
        q = torch.stack([E2M1[(packed & 0xF).long()], E2M1[(packed >> 4).long()]], -1).reshape(packed.shape[0], -1)
        return q * torch.exp2(scale.float() - 127).repeat_interleave(32, 1)
    from compressed_tensors.compressors.quantized_compressors.pack_quantized import unpack_from_int32
    shape = store.tensor(module + ".weight_shape").tolist()
    bits = cfg_q["num_bits"]
    q = unpack_from_int32(packed, bits, torch.Size(shape)).float()
    group = shape[1] // scale.shape[1]
    s = scale.float().repeat_interleave(group, 1)
    if module + ".weight_zero_point" in k:
        zp = unpack_from_int32(store.tensor(module + ".weight_zero_point"), bits, torch.Size([shape[0], scale.shape[1]]), packed_dim=0)
        q = q - zp.float().repeat_interleave(group, 1)
    return q * s


class LazyStack:
    """Stands in for a stacked expert parameter: `stack[e]` is expert e's matrix."""

    def __init__(self, loader, n):
        self.loader, self.n = loader, n

    def __getitem__(self, e):
        return self.loader(int(e))

    def __len__(self):
        return self.n


def load(model_dir, dtype=torch.float32):
    store = LazyCheckpoint(model_dir)
    cfg = json.load(open(os.path.join(model_dir, "config.json")))
    qcfg = (cfg.get("quantization_config") or {})
    groups = qcfg.get("config_groups") or {}
    wq = next(iter(groups.values()), {}).get("weights", {}) if groups else {}
    lazy_names = [k for k in store.keys() if k in store.lazy["holes"]]
    pat = re.compile(r"^(.*layers\.(\d+)\..*experts)\.(\d+)\.(w1|w2|w3|gate_proj|up_proj|down_proj)\.")
    prefixes = {}
    for k in lazy_names:
        m = pat.match(k)
        if m:
            prefixes[int(m.group(2))] = m.group(1)

    # A view of the directory without the lazy file: transformers loads the trunk.
    view = tempfile.mkdtemp(prefix="ref_trunk_")
    for fn in os.listdir(model_dir):
        if fn in ("model-lazy.safetensors", "lazy.json", "model.safetensors.index.json") or fn.startswith("."):
            continue
        os.symlink(os.path.abspath(os.path.join(model_dir, fn)), os.path.join(view, fn))

    config = AutoConfig.from_pretrained(view, trust_remote_code=False)
    mt = config.model_type if hasattr(config, "num_hidden_layers") else config.text_config.model_type
    mod = importlib.import_module(f"transformers.models.{mt}.modeling_{mt}")
    experts_cls = [getattr(mod, n) for n in dir(mod) if n.endswith("Experts") and isinstance(getattr(mod, n), type)]
    patched = {}
    for cls in experts_cls:
        class Meta(cls):
            def __init__(self, *a, **kw):
                with torch.device("meta"):
                    super().__init__(*a, **kw)
                # Empty placeholders: from_pretrained would otherwise allocate the stacks.
                self.lazy_n = self.gate_up_proj.shape[0]
                self.gate_up_proj = torch.nn.Parameter(torch.empty(0), requires_grad=False)
                self.down_proj = torch.nn.Parameter(torch.empty(0), requires_grad=False)
        Meta.__name__ = cls.__name__
        patched[cls.__name__] = (cls, Meta)
        setattr(mod, cls.__name__, Meta)
    try:
        model = AutoModelForCausalLM.from_pretrained(view, dtype=dtype, experts_implementation="eager")
    finally:
        for name, (cls, _) in patched.items():
            setattr(mod, name, cls)
    model.eval()

    keys = set(store.keys())

    def module_of(base, part):
        for p in PROJ[part]:
            if base + p + ".weight" in keys or base + p + ".weight_packed" in keys:
                return base + p
        raise KeyError(f"no {part} projection under {base}")

    def expert_loader(prefix, part):
        def load_e(e):
            base = f"{prefix}.{e}."
            if part == "gate_up":  # transformers' MergeModulelist + Concatenate: gate rows, then up rows
                g = dequant_expert(store, module_of(base, "gate"), wq)
                u = dequant_expert(store, module_of(base, "up"), wq)
                return torch.cat([g, u], 0).to(dtype)
            return dequant_expert(store, module_of(base, "down"), wq).to(dtype)
        return load_e

    n_swapped = 0
    for name, m in model.named_modules():
        if type(m).__name__ in patched and "gate_up_proj" in m._parameters:
            li = int(re.search(r"layers\.(\d+)\.", name).group(1))
            prefix = prefixes[li]
            n = m.lazy_n
            del m._parameters["gate_up_proj"], m._parameters["down_proj"]
            m.gate_up_proj = LazyStack(expert_loader(prefix, "gate_up"), n)
            m.down_proj = LazyStack(expert_loader(prefix, "down"), n)
            n_swapped += 1
    if n_swapped == 0:
        raise SystemExit("reference: no experts module found to make lazy")
    orig_forward = model.forward

    def forward(*a, **kw):
        out = orig_forward(*a, **kw)
        store.flush()
        return out

    model.forward = forward
    return model
