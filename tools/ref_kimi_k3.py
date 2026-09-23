"""Kimi K3's own modeling code as a CPU reference for a (truncated) released checkpoint.

    python3 tools/probe_reference.py MODEL probe.json --dtype float32 --raw --trust-remote-code --factory tools/ref_kimi_k3.py

MODEL is a truncated checkpoint directory, or the Hub id `moonshotai/Kimi-K3`
for the whole release at full depth: each decoder layer is then read from the
Hub before it runs and dropped after (tools/ref_stream.py), and the experts
the gate picks are read in parallel as it picks them.

transformers has no Kimi K3, so the reference is the release's
`modeling_kimi_linear.py` (the text model inside KimiK3ForConditionalGeneration),
run unmodified except for what needs a GPU: its flash-linear-attention (fla)
entry points are replaced by fla's own torch references of the same maths —
`chunk_kda` / `fused_recurrent_kda` by the lower-bound gate
(`naive_kda_lowerbound_gate`, or `naive_kda_gate`), fla's L2 norm (eps 1e-6),
the sigmoid beta and `naive_recurrent_kda`; `ShortConvolution` by the causal
depthwise `conv1d` + SiLU it is; `FusedRMSNormGated` by `rmsnorm(x) * w *
sigmoid(g)`.

The routed experts (MXFP4 packed, lazy in the cut) are read on use. The trunk
is kept in its stored dtype (bf16) and every Linear / Embedding computes in
float32 on the fly, which is float32 arithmetic on the same values without a
float32 copy of a 7168-wide model; tensors stored as F32 stay F32.

The residual reported per layer is what ditch reports with Attention
Residual: the attention input mixture (what `input_layernorm` reads), then
what the final norm reads.
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
from lazy_checkpoint import LazyCheckpoint  # noqa: E402
# ref_lazy_moe hides fla unless it is already imported; the release's modeling
# file imports it at load (its entry points are replaced below), so import it first.
import fla  # noqa: E402,F401
from ref_lazy_moe import dequant_expert  # noqa: E402

PREFIX = "language_model."


def cpu_kda(q, k, v, g, beta, A_log=None, dt_bias=None, initial_state=None, output_final_state=False,
            use_qk_l2norm_in_kernel=False, use_gate_in_kernel=False, use_beta_sigmoid_in_kernel=False,
            safe_gate=False, lower_bound=None, transpose_state_layout=False, cu_seqlens=None, scale=None, **_):
    from fla.ops.kda.gate import naive_kda_gate, naive_kda_lowerbound_gate
    from fla.ops.kda.naive import naive_recurrent_kda
    assert cu_seqlens is None and initial_state is None
    q, k, v = q.float(), k.float(), v.float()
    if use_qk_l2norm_in_kernel:
        q = q / torch.sqrt((q * q).sum(-1, keepdim=True) + 1e-6)
        k = k / torch.sqrt((k * k).sum(-1, keepdim=True) + 1e-6)
    if use_gate_in_kernel:
        g = naive_kda_lowerbound_gate(g, A_log, dt_bias, lower_bound) if lower_bound is not None else naive_kda_gate(g, A_log, dt_bias)
    if use_beta_sigmoid_in_kernel:
        beta = beta.float().sigmoid()
    return naive_recurrent_kda(q, k, v, g.float(), beta.float(), scale=scale, output_final_state=output_final_state)


class CpuShortConvolution(nn.Conv1d):
    """fla's ShortConvolution: causal depthwise conv1d, then SiLU."""

    def __init__(self, hidden_size, kernel_size, bias=False, activation="silu", backend=None, device=None, dtype=None, **kw):
        super().__init__(hidden_size, hidden_size, kernel_size, groups=hidden_size, bias=bias, padding=kernel_size - 1, device=device, dtype=dtype)
        self.activation = activation

    def forward(self, x, residual=None, mask=None, cache=None, output_final_state=False, cu_seqlens=None, **kw):
        T = x.shape[1]
        y = F.conv1d(x.float().transpose(1, 2), self.weight.float(), None if self.bias is None else self.bias.float(),
                     padding=self.padding, groups=self.groups)[..., :T].transpose(1, 2)
        if self.activation in ("silu", "swish"):
            y = F.silu(y)
        return y, None


class CpuRMSNormGated(nn.Module):
    """fla's FusedRMSNormGated with a sigmoid (or swish) gate."""

    def __init__(self, hidden_size, elementwise_affine=True, eps=1e-5, activation="swish", **kw):
        super().__init__()
        self.eps, self.activation = eps, activation
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = None

    def forward(self, x, g, **kw):
        x, g = x.float(), g.float()
        y = x * torch.rsqrt((x * x).mean(-1, keepdim=True) + self.eps) * self.weight.float()
        return y * torch.sigmoid(g) if self.activation == "sigmoid" else y * g * torch.sigmoid(g)


class LazyLinear(nn.Module):
    def __init__(self, store, module):
        super().__init__()
        self.store, self.module = store, module

    def forward(self, x):
        return F.linear(x.float(), dequant_expert(self.store, self.module, {}))


def shim_transformers():
    """Names the release's code imports from where transformers 4 kept them."""
    import transformers.utils.generic as generic
    from transformers.utils.output_capturing import OutputRecorder
    if not hasattr(generic, "OutputRecorder"):
        generic.OutputRecorder = OutputRecorder


def a_log_heads(t, p):
    """The release stores A_log with head_dim (128) entries for num_heads (96)
    heads, which its own code cannot load; the first num_heads are taken, as
    ditch does (an assumption: no released loader was available to confirm it)."""
    return t[: p.shape[0]].clone() if t.shape[0] > p.shape[0] else t


def load(model_dir, dtype=torch.float32):
    shim_transformers()
    # A Hub id streams the whole release a layer at a time (tools/ref_stream.py).
    streamed = not os.path.isdir(model_dir)
    if streamed:
        import ref_stream
        from huggingface_hub import hf_hub_download
        store = ref_stream.Store(ref_stream.Source(model_dir, os.environ.get("REF_STREAM_REVISION")))
        cfg = json.load(open(hf_hub_download(model_dir, "config.json")))
    else:
        store = LazyCheckpoint(model_dir)
        cfg = json.load(open(os.path.join(model_dir, "config.json")))
    cls = get_class_from_dynamic_module("modeling_kimi_linear.KimiLinearForCausalLM", model_dir)
    cfg_cls = get_class_from_dynamic_module("configuration_kimi_k3.KimiLinearConfig", model_dir)
    mod = sys.modules[cls.__module__]
    ccm = mod.create_causal_mask

    def create_causal_mask(*a, input_embeds=None, cache_position=None, **kw):  # transformers 4 spelling
        if input_embeds is not None:
            kw["inputs_embeds"] = input_embeds
        return ccm(*a, **kw)

    mod.create_causal_mask = create_causal_mask
    mod.chunk_kda = cpu_kda
    mod.fused_recurrent_kda = cpu_kda
    mod.ShortConvolution = CpuShortConvolution
    mod.FusedRMSNormGated = CpuRMSNormGated
    orig_expert = mod.KimiBlockSparseMLP

    class MetaExpert(orig_expert):
        def __init__(self, *a, **kw):
            with torch.device("meta"):
                super().__init__(*a, **kw)

    mod.KimiBlockSparseMLP = MetaExpert
    tc = dict(cfg["text_config"])
    tc.pop("quantization_config", None)
    config = cfg_cls(**tc)
    config._attn_implementation = "eager"
    # Built on the meta device; every parameter is then the checkpoint's tensor as stored.
    try:
        with torch.device("meta"):
            model = cls(config)
    finally:
        mod.KimiBlockSparseMLP = orig_expert
    model.eval()
    for m in model.modules():  # the code asks for flash attention; there is none on a CPU
        if hasattr(m, "config") and hasattr(m.config, "_attn_implementation"):
            m.config._attn_implementation = "eager"

    with torch.no_grad():
        missing = []
        for mname, m in model.named_modules():
            for pname, p in list(m._parameters.items()):
                if p is None:
                    continue
                name = f"{mname}.{pname}" if mname else pname
                if ".experts." in name and ".shared_experts." not in name:
                    continue
                if streamed and name.startswith("model.layers."):
                    continue  # read when the layer runs
                key = PREFIX + name
                if key not in store.keys():
                    missing.append(name)
                    continue
                t = store.tensor(key)
                if pname == "A_log":
                    t = a_log_heads(t, p)
                if tuple(t.shape) != tuple(p.shape):
                    raise SystemExit(f"reference: {name} is {tuple(t.shape)} in the checkpoint, {tuple(p.shape)} in the model")
                m._parameters[pname] = nn.Parameter(t, requires_grad=False)
        if missing:
            raise SystemExit(f"reference: {len(missing)} parameters not in the checkpoint, e.g. {missing[:6]}")
        meta_buffers = [n for n, b in model.named_buffers() if b.device.type == "meta"]
        if meta_buffers:
            raise SystemExit(f"reference: buffers left on the meta device: {meta_buffers[:6]}")
    # Routed experts: lazy linears under the original expert forward.
    for i, layer in enumerate(model.model.layers):
        moe = getattr(layer, "block_sparse_moe", None)
        if moe is None:
            continue
        for e, ex in enumerate(moe.experts):
            base = f"{PREFIX}model.layers.{i}.block_sparse_moe.experts.{e}."
            ex.w1, ex.w2, ex.w3 = (LazyLinear(store, base + n) for n in ("w1", "w2", "w3"))
        if streamed:
            def routed(gate, inp, out, i=i):
                # Every expert the gate picked is read now, in parallel.
                for e in torch.unique(out[0]).tolist():
                    base = f"{PREFIX}model.layers.{i}.block_sparse_moe.experts.{e}."
                    store.prefetch(base, [base + n + sfx for n in ("w1", "w2", "w3") for sfx in
                                          (".weight", ".weight_packed", ".weight_scale", ".weight_shape", ".weight_zero_point")])
            moe.gate.register_forward_hook(routed)
    # Float32 arithmetic on the stored values.
    def f32_linear(self, x):
        # In blocks of output rows, so a large weight (the LM head) is never
        # copied to float32 all at once.
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
            if id(p) not in linears and p.is_floating_point() and p.device.type != "meta":
                p.data = p.data.float()
    if streamed:
        layers = list(model.model.layers)
        linear_names = [{n + ".weight" for n, m in layer.named_modules() if isinstance(m, nn.Linear)} for layer in layers]

        def fetch(i):
            # As stored, Linear weights in their stored dtype (they compute in
            # float32 as above), everything else float32.
            params = dict(layers[i].named_parameters())
            got = store.tensors([f"{PREFIX}model.layers.{i}.{n}" for n in params])
            out = {}
            for n, p in params.items():
                t = got[f"{PREFIX}model.layers.{i}.{n}"]
                if n.endswith("A_log"):
                    t = a_log_heads(t, p)
                out[n] = t if n in linear_names[i] or not t.is_floating_point() else t.float()
            return out

        stream = ref_stream.LayerStreamer(layers, fetch)
        for i, layer in enumerate(layers):
            layer.register_forward_pre_hook(lambda m, a, i=i: stream.materialise(i))
            layer.register_forward_hook(lambda m, a, o, i=i: stream.release(i))
    return Reference(model, store)


class Reference(nn.Module):
    def __init__(self, model, store):
        super().__init__()
        self.model, self.store = model, store
        self.captured = []
        for layer in model.model.layers:
            layer.input_layernorm.register_forward_pre_hook(lambda m, a: self.captured.append(a[0]))

    def run(self, ids):
        self.captured = []
        out = self.model(input_ids=ids, use_cache=False)
        self.store.flush()
        return out.logits.float(), list(self.captured)

    def forward(self, input_ids, output_hidden_states=False):
        with torch.no_grad():
            logits, hidden = self.run(input_ids)
        # The final entry is supplied by probe_reference.py's final-norm hook.
        return types.SimpleNamespace(logits=logits, hidden_states=(hidden + [hidden[-1]]) if output_hidden_states else None)

    def generate(self, input_ids, max_new_tokens=8, do_sample=False):
        ids = input_ids
        for _ in range(max_new_tokens):
            with torch.no_grad():
                logits, _ = self.run(ids)
            ids = torch.cat([ids, logits[:, -1].argmax(-1, keepdim=True)], dim=1)
        return ids
