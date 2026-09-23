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

# transformers binds flash-linear-attention's Triton kernels at import time
# whenever fla is importable, and they cannot run on a CPU: hide it, so the
# linear-attention families take transformers' own torch paths.
if "fla" not in sys.modules:
    sys.modules["fla"] = None

from transformers import AutoConfig, AutoModelForCausalLM  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lazy_checkpoint import LazyCheckpoint  # noqa: E402

# FP8 block size (`weight_block_size`), set from the checkpoint's config by load().
FP8_BLOCK = None
# MiMo V2 (SGLang's loader): the fp8 attention projections are
# `num_key_value_heads` row shards, each block-quantised on its own.
ATTN_ROW_SHARDS = 1

E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0])
PROJ = {"gate": ("w1", "gate_proj"), "up": ("w3", "up_proj"), "down": ("w2", "down_proj")}


def dequant_expert(store, module, cfg_q):
    """The float32 `[out, in]` weight of `<module>` (a Linear's prefix), however it is stored."""
    k = store.keys()
    if module + ".weight" in k and store.header[module + ".weight"]["dtype"] == "U8" and module + ".weight_scale" in k:
        # MiMo V2.6's MXFP4 store (`store_dtype: mxfp4`): the same nibbles and
        # E8M0 scales, named `weight` / `weight_scale`.
        packed, scale = store.tensor(module + ".weight"), store.tensor(module + ".weight_scale")
        q = torch.stack([E2M1[(packed & 0xF).long()], E2M1[(packed >> 4).long()]], -1).reshape(packed.shape[0], -1)
        return q * torch.exp2(scale.float() - 127).repeat_interleave(32, 1)
    if module + ".weight" in k:
        w = store.tensor(module + ".weight")
        if module + ".weight_scale_inv" in k:  # FP8 blocks, dequantised to bf16 as the integrations do
            s = store.tensor(module + ".weight_scale_inv").float()
            # The configured block, the last one partial (GLM-5.3's kv_a_proj
            # is 576 rows in 5 blocks of 128); only without one is it derived.
            br, bc = FP8_BLOCK or (-(-w.shape[0] // s.shape[0]), -(-w.shape[1] // s.shape[1]))
            # MiMo V2's attention projections are blocked per tensor-parallel
            # shard (`ATTN_ROW_SHARDS` of them), each from its own first row.
            k = ATTN_ROW_SHARDS if re.search(r"self_attn\.(qkv|q|k|v)_proj$", module) else 1
            parts = [(w, s)]
            if k > 1 and w.shape[0] % k == 0 and s.shape[0] == k * -(-(w.shape[0] // k) // br):
                parts = list(zip(w.chunk(k, 0), s.chunk(k, 0)))
            deq = torch.cat([pw.float() * ps.repeat_interleave(br, 0)[: pw.shape[0]].repeat_interleave(bc, 1)[:, : pw.shape[1]]
                             for pw, ps in parts], 0)
            return deq.to(torch.bfloat16).float()
        return w.float()
    packed = store.tensor(module + ".weight_packed")
    scale = store.tensor(module + ".weight_scale")
    if packed.dtype == torch.uint8:  # mxfp4-pack-quantized: e2m1 nibble pairs, E8M0 scales
        q = torch.stack([E2M1[(packed & 0xF).long()], E2M1[(packed >> 4).long()]], -1).reshape(packed.shape[0], -1)
        return q * torch.exp2(scale.float() - 127).repeat_interleave(32, 1)
    from compressed_tensors.compressors.pack_quantized.helpers import unpack_from_int32
    shape = store.tensor(module + ".weight_shape").tolist()
    bits = cfg_q["num_bits"]
    from compressed_tensors.quantization.lifecycle.forward_helpers import _dequantize
    q = unpack_from_int32(packed, bits, torch.Size(shape))
    group = shape[1] // scale.shape[1]
    zp = None
    if module + ".weight_zero_point" in k:
        zp = unpack_from_int32(store.tensor(module + ".weight_zero_point"), bits, torch.Size([shape[0], scale.shape[1]]), packed_dim=0)
        zp = zp.repeat_interleave(group, 1)
    # compressed-tensors' own arithmetic: in the scale's dtype (bf16 here).
    return _dequantize(q, scale.repeat_interleave(group, 1), zp).float()


def dequant_fp8_trunk(model, store, cfg_q=None):
    """Every Linear whose checkpoint weight is quantised gets its dequantised
    weight in bf16 (what ditch and the Hugging Face integrations produce):
    FP8 with a block `weight_scale_inv` in place of the raw codes the load
    copied, compressed-tensors `weight_packed` (INT or MXFP4) in place of the
    placeholder the load left for a weight it did not find."""
    keys = set(store.keys())
    n = 0
    with torch.no_grad():
        for name, m in model.named_modules():
            if not isinstance(m, torch.nn.Linear) or m.weight.device.type == "meta":
                continue
            for cand in (name, name.split(".", 1)[-1], "model." + name, name.replace("model.language_model.", "language_model.model.")):
                if (cand + ".weight_scale_inv" in keys or cand + ".weight_packed" in keys) and cand + ".weight_packed" not in store.lazy["holes"]:
                    m.weight.data = dequant_expert(store, cand, cfg_q or {}).to(torch.bfloat16)
                    n += 1
                    break
    return n


def f32_arithmetic(model, store):
    """Float32 arithmetic on the stored values: Linear and Embedding cast their
    (bf16) weights to float32 as they compute, in blocks of output rows;
    every other parameter and buffer becomes float32. Tensors the checkpoint
    stores as F32 (routers' correction biases, ...) are restored exactly: the
    bf16 load rounded them."""
    import types
    import torch.nn.functional as F
    # Checkpoint names go through transformers' own renames for the family
    # (GLM-5.3-Flash's `hc_attn_base` is the model's `attn_hc.base`).
    from transformers import conversion_mapping as cm
    renames = []
    for c in (model.config, getattr(model.config, "text_config", None)):
        if c is not None:
            renames += [r for r in (cm.get_checkpoint_conversion_mapping(c.model_type) or []) if type(r).__name__ == "WeightRenaming"]
    f32_names = {}
    for k in store.keys():
        if store.header[k]["dtype"] == "F32" and k not in store.lazy["holes"]:
            name = k
            for r in renames:
                name = r.rename_source_key(name)[0]
            for cand in (name, name.split(".", 1)[1] if "." in name else name):
                f32_names[cand] = k
    sd_names = dict(model.named_parameters())
    sd_names.update(dict(model.named_buffers()))
    with torch.no_grad():
        for name, t in sd_names.items():
            key = f32_names.get(name) or next((f32_names[n] for n in (name.split(".", 1)[-1], "model." + name) if n in f32_names), None)
            if key is not None and tuple(store.header[key]["shape"]) == tuple(t.shape):
                t.data = store.tensor(key).float()

    def linear(self, x):
        x = x.float()
        w = self.weight
        step = max(1, (64 << 20) // (w.shape[1] * 4))
        out = torch.cat([F.linear(x, w[i:i + step].float()) for i in range(0, w.shape[0], step)], dim=-1)
        return out if self.bias is None else out + self.bias.float()

    keep = set()
    for m in model.modules():
        if isinstance(m, torch.nn.Linear) and m.weight.device.type != "meta":
            keep.add(id(m.weight))
            m.forward = types.MethodType(linear, m)
        elif isinstance(m, torch.nn.Embedding):
            keep.add(id(m.weight))
            m.forward = types.MethodType(lambda self, ids: F.embedding(ids, self.weight).float(), m)
    with torch.no_grad():
        for m in model.modules():
            # A rotary table rounded by the bf16 load is rebuilt from its own init function.
            inv = getattr(m, "inv_freq", None)
            if torch.is_tensor(inv) and inv.dtype != torch.float32:
                fn = getattr(m, "rope_init_fn", None) or getattr(m, "compute_default_rope_parameters", None)
                if fn is None:
                    raise SystemExit(f"reference: cannot rebuild the bf16 inv_freq of {type(m).__name__}")
                new, _ = fn(m.config, "cpu") if getattr(m, "rope_type", None) is None else fn(m.config, "cpu")
                m.inv_freq = new.float()
                if hasattr(m, "original_inv_freq"):
                    m.original_inv_freq = new.float()
        for t in list(model.parameters()) + list(model.buffers()):
            if id(t) not in keep and t.is_floating_point() and t.device.type != "meta" and t.dtype != torch.float32:
                t.data = t.data.float()


class LazyRows(torch.nn.Module):
    """An `nn.Embedding` whose table is the row-concatenation of checkpoint
    tensors `names` (Qwen4-Exp's `ngram_embedding.shard_{k}`, which
    transformers concatenates at load: ~100 GB), reading only the rows looked up."""

    def __init__(self, store, names):
        super().__init__()
        self.store, self.names = store, names
        self.starts = [0]
        for n in names:
            self.starts.append(self.starts[-1] + store.header[n]["shape"][0])
        self.dim = store.header[names[0]]["shape"][1]
        self.weight = torch.empty(0)  # its device is all the caller asks about

    def forward(self, ids):
        flat = ids.reshape(-1).tolist()
        out = torch.empty(len(flat), self.dim)
        for i, r in enumerate(flat):
            k = next(k for k in range(len(self.names)) if r < self.starts[k + 1])
            out[i] = self.store.rows(self.names[k], [r - self.starts[k]])[0].float()
        return out.view(*ids.shape, self.dim)


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
    qcfg = cfg.get("quantization_config") or (cfg.get("text_config") or {}).get("quantization_config") or {}
    groups = qcfg.get("config_groups") or {}
    wq = next(iter(groups.values()), {}).get("weights", {}) if groups else {}
    global FP8_BLOCK
    FP8_BLOCK = tuple(qcfg["weight_block_size"]) if qcfg.get("weight_block_size") else None
    global ATTN_ROW_SHARDS
    tcfg = cfg.get("text_config") or cfg
    ATTN_ROW_SHARDS = tcfg.get("num_key_value_heads", 1) if tcfg.get("model_type") in ("mimo_v2", "mimo_v2_flash") else 1
    lazy_names = [k for k in store.keys() if k in store.lazy["holes"]]
    pat = re.compile(r"^(.*layers\.(\d+)\..*experts)\.(\d+)\.(w1|w2|w3|gate_proj|up_proj|down_proj)\.")
    stacked_pat = re.compile(r"^(.*layers\.(\d+)\..*experts)\.(gate_up_proj|down_proj)(?:_blocks)?$")
    prefixes, stacked = {}, {}
    for k in lazy_names:
        m = pat.match(k)
        if m:
            prefixes[int(m.group(2))] = m.group(1)
        m = stacked_pat.match(k)
        if m:
            stacked.setdefault(int(m.group(2)), {})[m.group(3)] = k

    ngram_pat = re.compile(r"^.*layers\.(\d+)\..*\.ngram_embedding\.shard_(\d+)\.weight$")
    ngram = {}
    for k in lazy_names:
        m = ngram_pat.match(k)
        if m:
            ngram.setdefault(int(m.group(1)), {})[int(m.group(2))] = k

    # A view of the directory without the lazy file: transformers loads the trunk.
    view = tempfile.mkdtemp(prefix="ref_trunk_")
    for fn in os.listdir(model_dir):
        if fn in ("model-lazy.safetensors", "lazy.json", "model.safetensors.index.json") or fn.startswith("."):
            continue
        if fn == "config.json":
            # Quantised weights (routed experts and trunk alike) are
            # dequantised here: keep transformers' quantizer out.
            c = dict(cfg)
            c.pop("quantization_config", None)
            if isinstance(c.get("text_config"), dict):
                c["text_config"] = {k: v for k, v in c["text_config"].items() if k != "quantization_config"}
            json.dump(c, open(os.path.join(view, fn), "w"))
            continue
        os.symlink(os.path.abspath(os.path.join(model_dir, fn)), os.path.join(view, fn))

    config = AutoConfig.from_pretrained(view, trust_remote_code=False)
    # The experts class lives in the text model's module, which for a wrapper
    # (Kimi K2.5 around DeepSeek V3) is not the wrapper's own.
    # (Kimi K2.5's text_config keeps model_type kimi_k2 but is a DeepseekV3Config),
    # so the modules come from the config classes.
    cfgs = [config] + ([config.text_config] if getattr(config, "text_config", None) is not None else [])
    mods = []
    for c in cfgs:
        name = type(c).__module__.replace(".configuration_", ".modeling_")
        try:
            mods.append(importlib.import_module(name))
        except ImportError:
            pass
    experts_cls = [(mod, getattr(mod, n)) for mod in mods for n in dir(mod)
                   if n.endswith("Experts") and isinstance(getattr(mod, n), type)]
    patched = {}
    for mod, cls in experts_cls:
        class Meta(cls):
            def __init__(self, *a, **kw):
                with torch.device("meta"):
                    super().__init__(*a, **kw)
                # Empty placeholders: from_pretrained would otherwise allocate the stacks.
                self.lazy_n = self.gate_up_proj.shape[0]
                self.lazy_shapes = (tuple(self.gate_up_proj.shape[1:]), tuple(self.down_proj.shape[1:]))
                self.gate_up_proj = torch.nn.Parameter(torch.empty(0), requires_grad=False)
                self.down_proj = torch.nn.Parameter(torch.empty(0), requires_grad=False)
        Meta.__name__ = cls.__name__
        patched[cls.__name__] = (mod, cls, Meta)
        setattr(mod, cls.__name__, Meta)
    # Lazy n-gram tables: a one-row placeholder at load, swapped for LazyRows below.
    for mod in mods:
        for n in dir(mod):
            c = getattr(mod, n)
            if not (ngram and isinstance(c, type) and n.endswith("NGramEmbedding")):
                continue

            class MetaNgram(c):
                def __init__(self, *a, **kw):
                    # Only the table is left out (its buffers are computed as usual).
                    emb = torch.nn.Embedding
                    torch.nn.Embedding = lambda rows, dim, *a2, **kw2: emb(1, dim)
                    try:
                        super().__init__(*a, **kw)
                    finally:
                        torch.nn.Embedding = emb
            MetaNgram.__name__ = n
            patched[n] = (mod, c, MetaNgram)
            setattr(mod, n, MetaNgram)
    # Text-only probes: an image-text wrapper's vision tower and projector are
    # built empty (their tensors load as unexpected and are dropped), which
    # keeps a float32 reference of a large model inside the RAM.
    class NoVision(torch.nn.Module):
        def __init__(self, *a, **kw):
            super().__init__()

        @classmethod
        def _from_config(cls, *a, **kw):
            return cls()

    for mod in mods:
        for n in dir(mod):
            c = getattr(mod, n)
            if isinstance(c, type) and issubclass(c, torch.nn.Module) and (n.endswith("VisionModel") or n.endswith("MultimodalProjection") or n.endswith("MultiModalProjector")):
                patched[n] = (mod, c, NoVision)
                setattr(mod, n, NoVision)
    # The trunk is held in its stored dtype (bf16; FP8 dequantised to bf16 as
    # transformers' FP8 integration does) and computed in float32 (below):
    # float32 arithmetic on the same values, without a float32 copy of a
    # 6144-wide model.
    # (FP8 weights are dequantised here after the load: transformers' CPU
    # `dequantize=True` path left GLM-5.3's codes unscaled.)
    kw = dict(dtype=torch.bfloat16, experts_implementation="eager", trust_remote_code=False)
    try:
        try:
            model = AutoModelForCausalLM.from_pretrained(view, **kw)
        except ValueError:
            # Not registered with AutoModelForCausalLM (image-text wrappers): the class the config names.
            from transformers import AutoModelForImageTextToText
            model = AutoModelForImageTextToText.from_pretrained(view, **kw)
    finally:
        for name, (mod, cls, _) in patched.items():
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

    dequant_fp8_trunk(model, store, wq)
    f32_arithmetic(model, store)
    def slab_loader(name, want):
        # Expert e of a stacked `[E, ...]` checkpoint tensor is its slab e.
        def load_e(e):
            if name.endswith("_blocks"):
                # gpt-oss MXFP4: `[E, rows, cols/32, 16]` e2m1 nibble pairs (low
                # nibble first) with `_scales` `[E, rows, cols/32]` E8M0 exponents.
                b = store.rows(name, [e])[0]
                sc = store.rows(name[: -len("_blocks")] + "_scales", [e])[0]
                q = torch.stack([E2M1[(b & 0xF).long()], E2M1[(b >> 4).long()]], -1).reshape(*b.shape[:-1], 32)
                t = (q * torch.exp2(sc.float() - 127).unsqueeze(-1)).reshape(b.shape[0], -1)
                # Stored `[out, in]`; GptOssExperts computes `x @ W`, `[in, out]`
                # (said explicitly: gpt-oss's down projection is square).
                t = t.transpose(0, 1).contiguous()
            else:
                t = store.rows(name, [e])[0].float()
            if tuple(t.shape) != want:
                if tuple(t.shape[::-1]) == want:
                    t = t.transpose(0, 1).contiguous()
                else:
                    raise SystemExit(f"reference: {name}[{e}] is {tuple(t.shape)}, the model wants {want}")
            return t.to(dtype)
        return load_e

    n_swapped = 0
    for name, m in model.named_modules():
        if type(m).__name__ in patched and "gate_up_proj" in m._parameters and hasattr(m, "lazy_n"):
            li = int(re.search(r"layers\.(\d+)\.", name).group(1))
            n = m.lazy_n
            del m._parameters["gate_up_proj"], m._parameters["down_proj"]
            if li in stacked:
                m.gate_up_proj = LazyStack(slab_loader(stacked[li]["gate_up_proj"], m.lazy_shapes[0]), n)
                m.down_proj = LazyStack(slab_loader(stacked[li]["down_proj"], m.lazy_shapes[1]), n)
            else:
                prefix = prefixes[li]
                m.gate_up_proj = LazyStack(expert_loader(prefix, "gate_up"), n)
                m.down_proj = LazyStack(expert_loader(prefix, "down"), n)
            n_swapped += 1
    for name, m in model.named_modules():
        if type(m).__name__.endswith("NGramEmbedding") and ngram:
            li = int(re.search(r"layers\.(\d+)\.", name).group(1))
            parts = ngram[li]
            m.ngram_embedding = LazyRows(store, [parts[i] for i in range(len(parts))])
            n_swapped += 1
    if n_swapped == 0 and store.lazy["holes"]:
        raise SystemExit("reference: the cut has lazy tensors but no experts module was made lazy")
    orig_forward = model.forward

    def forward(*a, **kw):
        out = orig_forward(*a, **kw)
        store.flush()
        return out

    model.forward = forward
    return model
