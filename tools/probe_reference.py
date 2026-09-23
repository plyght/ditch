#!/usr/bin/env python3
"""Compare `ditch probe --json` with Hugging Face transformers on the CPU.

    ditch probe MODEL --prompt "..." --prompt "..." --json > probe.json
    python3 tools/probe_reference.py MODEL probe.json [--dtype bfloat16]

A base model with no chat template is probed with `ditch probe --raw` and
compared with `--raw` here: the prompt is the text, verbatim, with no
special tokens.

For every prompt the script renders the chat template with the same system
prompt, tokenises it with `tokenizers`, runs the model once and compares:
the rendered text, the token ids, the first-token logits (max absolute
difference relative to the logit range, and the argmax) and the greedy
continuation. Exit status 1 when the ids differ or the relative logit
error exceeds --tolerance.

Run `ditch probe` with `--residuals` as well and the last token's residual
is compared layer by layer, which names the layer a forward pass first
diverges at instead of only showing that the logits are wrong. Note that
transformers' last `hidden_states` entry is taken *after* the final norm,
while ditch reports the residual the final norm reads, so the reference for
the last entry is captured with a pre-forward hook on the final norm.

`--factory FILE` builds the reference with `load(model_dir, dtype)` from a
Python file instead of `AutoModelForCausalLM.from_pretrained`: for released
checkpoints transformers cannot hold on this machine as they are (DeepSeek
V4's FP4 experts, see tools/ref_deepseek_v4.py), or whose only reference is
the repository's own inference code. `--trust-remote-code` passes that flag to
`from_pretrained` for checkpoints that ship their own modeling file.

A checkpoint whose chat template starts with the BOS token renders it into
the text, while ditch adds the id at tokenisation time; the texts are
compared with that leading token removed, and the token ids decide.
"""
import argparse
import json
import sys

import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


LAYER_PARTS = (".layers.", ".h.", ".blocks.", ".block.", ".layer.")


def find_final_norm(model):
    """The module whose input is the residual the final norm reads.

    Named `norm` on most families, but also `ln_f` (GPT-2, GPT-Neo),
    `final_layernorm` (Phi, Nemotron-H) or `embedding_norm` (LFM2), and the
    decoder hangs off `model`, `transformer`, `gpt_neox` or `language_model`
    depending on the family. So: the last normalisation module that is not
    inside the layer stack.
    """
    found = None
    for name, mod in model.named_modules():
        if any(part in "." + name + "." for part in LAYER_PARTS):
            continue
        if "hc_head" in name or "hyper_connection_mixer" in name:  # the hyper-connection head's own norm reads the streams
            continue
        if "attn_res" in name:  # Attention Residual's scoring norm (Kimi K3), never called as a module
            continue
        if "norm" in type(mod).__name__.lower() or "norm" in name.rsplit(".", 1)[-1]:
            found = mod
    return found


def install_remote_code_shims():
    """Names that checkpoints' own modeling code (written for transformers 4)
    still imports but transformers 5 removed. Only reached with --trust-remote-code."""
    import transformers.utils.import_utils as iu
    import transformers.utils as tu
    from transformers.cache_utils import DynamicCache
    for mod in (iu, tu):
        if not hasattr(mod, "is_torch_fx_available"):
            mod.is_torch_fx_available = lambda: False
    if not hasattr(DynamicCache, "from_legacy_cache"):
        def from_legacy_cache(cls, past_key_values=None):
            cache = cls()
            for layer, (k, v) in enumerate(past_key_values or ()):
                cache.update(k, v, layer)
            return cache
        DynamicCache.from_legacy_cache = classmethod(from_legacy_cache)
    # transformers 4 declared ties as a list of the tied names (tied to the
    # input embedding); 5 wants {tied name: source name}.
    from transformers import PreTrainedModel
    expand = PreTrainedModel.get_expanded_tied_weights_keys
    if not getattr(expand, "_ditch_shim", False):
        def expanded(self, *a, **kw):
            tied = getattr(self, "_tied_weights_keys", None)
            if isinstance(tied, list):
                src = next((n for n, _ in self.named_parameters() if n.endswith("embed_tokens.weight")), None)
                self._tied_weights_keys = {k: src for k in tied} if src and self.config.tie_word_embeddings else {}
            return expand(self, *a, **kw)
        expanded._ditch_shim = True
        PreTrainedModel.get_expanded_tied_weights_keys = expanded
    if not hasattr(DynamicCache, "to_legacy_cache"):
        DynamicCache.to_legacy_cache = lambda self: tuple((l.keys, l.values) for l in self.layers)


PER_LAYER = False


def compare_residuals(entry, out, pre_norm, tolerance, collapsed=None):
    """Compares ditch's per-layer residuals with transformers' hidden states.

    Entry L is the vector layer L reads, so entry 0 is the embedding output
    and entry `num_layers` is what the final norm reads. transformers reports
    the *normalised* last hidden state, so the pre-norm hook supplies it.

    With hyper-connections (DeepSeek V4, GLM-5.3-Flash) the hidden state is
    `hc_mult` streams, and ditch reports what each layer's block reads: the
    streams collapsed by the layer's attention-site weights. `collapsed` holds
    those, captured from the reference's own hyper-connection modules.
    """
    got = [np.asarray(r, dtype=np.float32) for r in entry["residuals"]]
    if collapsed:
        hidden = [c[0, -1].float().numpy() for c in collapsed] + [None]
    else:
        # Gemma 3n stacks its AltUp streams first, `[streams, batch, seq, hidden]`;
        # stream 0 is the residual the next layer reads.
        hidden = [(h[0, 0, -1] if h.dim() == 4 else h[0, -1]).float().numpy() for h in out.hidden_states]
    if len(got) != len(hidden):
        print(f"  residuals: ditch has {len(got)} entries, transformers {len(hidden)}")
        return False
    # A stacked (Gemma 3n) last entry is recorded before the streams are
    # combined and normalised, so it is already the pre-norm residual.
    if pre_norm is not None and out.hidden_states[-1].dim() != 4:
        hidden[-1] = pre_norm[0, -1].float().numpy()
    worst, worst_layer = 0.0, 0
    first_bad = None
    for i, (a, b) in enumerate(zip(hidden, got)):
        scale = max(float(np.abs(a).max()), 1e-6)
        # Relative to the layer's own magnitude, so a bf16 reference (which a
        # model too big for a float32 one needs) is judged on the same scale.
        rel = float(np.abs(a - b).max()) / scale
        if PER_LAYER:
            print(f"    layer {i}: {rel:.2e}")
        if rel > worst:
            worst, worst_layer = rel, i
        if first_bad is None and rel > tolerance:
            first_bad = (i, rel, float(np.abs(a - b).max()), scale)
    if first_bad is not None:
        i, rel, d, scale = first_bad
        print(f"  residuals diverge first at layer {i}: max |difference| {d:.5f} = {rel:.2e} of |reference| {scale:.4f}")
        print(f"    layer {i} is the {'embedding output' if i == 0 else f'output of layer {i - 1}'};"
              f" the layers before it agree")
        return False
    print(f"  residuals: all {len(got)} layers agree (worst {worst:.2e} relative, at layer {worst_layer})")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("probe_json")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--tolerance", type=float, default=1e-2)
    ap.add_argument("--system-prompt", default="You are a helpful assistant.")
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--trust-remote-code", action="store_true",
                    help="let transformers run the checkpoint's own modeling code")
    ap.add_argument("--raw", action="store_true",
                    help="the probe was run with --raw: no chat template and no BOS")
    ap.add_argument("--factory", help="Python file whose load(model_dir, dtype) returns the reference model")
    ap.add_argument("--per-layer", action="store_true", help="print every layer's relative residual difference")
    ap.add_argument("--residual-tolerance", type=float, default=1e-3,
                    help="max |difference| of a per-layer residual, relative to that layer's own"
                         " magnitude, before it counts as a mismatch")
    args = ap.parse_args()
    global PER_LAYER
    PER_LAYER = args.per_layer

    probe = json.load(open(args.probe_json))
    trc = args.trust_remote_code
    if trc:
        install_remote_code_shims()
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=trc)
    if args.factory:
        import importlib.util
        spec = importlib.util.spec_from_file_location("reference_factory", args.factory)
        factory = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(factory)
        model = factory.load(args.model, getattr(torch, args.dtype))
    else:
        # transformers' default sdpa attention silently skips an attention
        # logit softcap (Gemma 2); eager attention applies it, as the model does.
        try:
            cfg = AutoConfig.from_pretrained(args.model, trust_remote_code=trc)
            softcap = getattr(getattr(cfg, "text_config", cfg), "attn_logit_softcapping", None)
        except Exception:
            softcap = None
        extra = {"attn_implementation": "eager"} if softcap else {}
        try:
            model = AutoModelForCausalLM.from_pretrained(args.model, dtype=getattr(torch, args.dtype), trust_remote_code=trc, **extra)
        except ValueError:
            # Not registered with AutoModelForCausalLM (Mistral 4): use the class the config names.
            import transformers
            cls = getattr(transformers, AutoConfig.from_pretrained(args.model, trust_remote_code=trc).architectures[0])
            model = cls.from_pretrained(args.model, dtype=getattr(torch, args.dtype))
    model.eval()
    ok = True
    captured = {}
    final_norm = find_final_norm(model)
    if final_norm is not None:
        final_norm.register_forward_pre_hook(lambda mod, inp: captured.__setitem__("pre_norm", inp[0]))
    # Hyper-connection sites: `layers.N.attn_hc` returns (post, comb, collapsed);
    # Qwen4-Exp's `layers.N.attn_hyper_connection` returns (collapsed, streams, weights).
    hc_out = {".attn_hc": 2, ".attn_hyper_connection": 0}
    hc_sites = [(int(n.split(".layers.")[-1].split(".")[0]), m, i) for n, m in model.named_modules()
                for suffix, i in hc_out.items() if n.endswith(suffix) and ".layers." in n and "mtp" not in n]
    for li, m, i in hc_sites:
        m.register_forward_hook(lambda mod, inp, outp, li=li, i=i: captured.setdefault("hc", {}).__setitem__(li, outp[i]))
    # Qwen4-Exp has no final norm: its hyper-connection mixer's collapse is what the LM head reads.
    mixers = [m for n, m in model.named_modules() if n.endswith("hyper_connection_mixer") and "mtp" not in n]
    if mixers:
        mixers[-1].register_forward_hook(lambda mod, inp, outp: captured.__setitem__("pre_norm", outp))
    for entry in probe["prompts"]:
        messages = [{"role": "system", "content": args.system_prompt}, {"role": "user", "content": entry["user"]}]
        if args.raw:
            text = entry["user"]
            ids = tok(text, add_special_tokens=False)["input_ids"]
        else:
            try:
                text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            except Exception as exc:  # models without a system role (Gemma)
                print(f"chat template with system role failed ({exc}); retrying without it")
                text = tok.apply_chat_template(messages[1:], tokenize=False, add_generation_prompt=True)
            # As apply_chat_template(tokenize=True) does: the template writes every
            # special token it wants, and the tokenizer adds none (no BOS the
            # template leaves out, Nemotron Nano 2).
            ids = tok(text, add_special_tokens=False)["input_ids"]
        print(f"\n== {entry['user'][:60]!r}")
        bos = tok.bos_token or ""
        if bos and text.startswith(bos) and not entry["text"].startswith(bos):
            text = text[len(bos) :]  # ditch adds the BOS id, not the literal
        if text != entry["text"]:
            print("  rendered text differs:")
            print("    transformers:", repr(text))
            print("    ditch:       ", repr(entry["text"]))
            ok = False
        if ids != entry["ids"]:
            print(f"  token ids differ: transformers {ids}\n                    ditch        {entry['ids']}")
            ok = False
        else:
            print(f"  token ids match ({len(ids)} tokens)")
        want_residuals = "residuals" in entry
        captured.pop("hc", None)
        with torch.no_grad():
            input_ids = torch.tensor([entry["ids"]])
            out = model(input_ids=input_ids, output_hidden_states=want_residuals)
            ref = out.logits[0, -1].float().numpy()
            # The Mamba families record each block's *output* and no embedding
            # entry: put the embedding first so entry L is what layer L reads (the
            # last block's output is then the pre-norm last entry; the normed one goes).
            if want_residuals and getattr(model.config, "model_type", "") in ("mamba", "mamba2", "falcon_mamba"):
                out.hidden_states = (model.get_input_embeddings()(input_ids),) + tuple(out.hidden_states)[:-1]
        if want_residuals:
            hc = captured.get("hc")
            collapsed = [hc[i] for i in sorted(hc)] if hc else None
            ok &= compare_residuals(entry, out, captured.get("pre_norm"), args.residual_tolerance, collapsed)
        got = np.asarray(entry["logits"], dtype=np.float32)
        if got.shape != ref.shape:
            n = min(got.shape[0], ref.shape[0])
            print(f"  logit vector lengths differ: ditch {got.shape[0]}, transformers {ref.shape[0]} (comparing {n})")
            got, ref = got[:n], ref[:n]
        scale = float(ref.max() - ref.min())
        rel = float(np.abs(got - ref).max()) / scale
        print(f"  first-token logits: max |diff| = {np.abs(got - ref).max():.4f}, relative to range {scale:.2f}: {rel:.2e}; argmax ditch {int(got.argmax())} vs transformers {int(ref.argmax())}")
        top_ref = np.argsort(-ref)[:5].tolist()
        top_got = np.argsort(-got)[:5].tolist()
        print(f"  top-5 transformers {top_ref} {[tok.decode([t]) for t in top_ref]}")
        print(f"  top-5 ditch        {top_got} {[tok.decode([t]) for t in top_got]}")
        if rel > args.tolerance or int(got.argmax()) != int(ref.argmax()):
            ok = False
        with torch.no_grad():
            try:
                gen = model.generate(input_ids, max_new_tokens=args.max_new_tokens, do_sample=False)
            except Exception:
                if not trc:
                    raise
                # Old remote code whose generation hooks transformers 5 no longer
                # drives: greedy by hand, recomputing the whole prefix each step.
                gen = input_ids
                for _ in range(args.max_new_tokens):
                    nxt = model(input_ids=gen, use_cache=False).logits[:, -1].argmax(-1, keepdim=True)
                    gen = torch.cat([gen, nxt], dim=-1)
        ref_text = tok.decode(gen[0, input_ids.shape[1]:].tolist(), skip_special_tokens=False)
        print(f"  greedy transformers: {ref_text!r}")
        resp = entry["response"]
        if isinstance(resp, list):  # bytes that are not valid UTF-8 on their own
            resp = bytes(resp).decode("utf-8", errors="replace")
        print(f"  greedy ditch:        {resp!r}")
    print("\nRESULT:", "OK" if ok else "MISMATCH")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
