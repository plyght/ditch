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

A checkpoint whose chat template starts with the BOS token renders it into
the text, while ditch adds the id at tokenisation time; the texts are
compared with that leading token removed, and the token ids decide.
"""
import argparse
import json
import sys

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


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
        if "norm" in type(mod).__name__.lower() or "norm" in name.rsplit(".", 1)[-1]:
            found = mod
    return found


def compare_residuals(entry, out, pre_norm, tolerance):
    """Compares ditch's per-layer residuals with transformers' hidden states.

    Entry L is the vector layer L reads, so entry 0 is the embedding output
    and entry `num_layers` is what the final norm reads. transformers reports
    the *normalised* last hidden state, so the pre-norm hook supplies it.
    """
    got = [np.asarray(r, dtype=np.float32) for r in entry["residuals"]]
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
    ap.add_argument("--raw", action="store_true",
                    help="the probe was run with --raw: no chat template and no BOS")
    ap.add_argument("--residual-tolerance", type=float, default=1e-3,
                    help="max |difference| of a per-layer residual, relative to that layer's own"
                         " magnitude, before it counts as a mismatch")
    args = ap.parse_args()

    probe = json.load(open(args.probe_json))
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=getattr(torch, args.dtype))
    model.eval()
    ok = True
    captured = {}
    final_norm = find_final_norm(model)
    if final_norm is not None:
        final_norm.register_forward_pre_hook(lambda mod, inp: captured.__setitem__("pre_norm", inp[0]))
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
            ids = tok(text, add_special_tokens=True if tok.bos_token and not text.startswith(tok.bos_token or "\0") else False)["input_ids"]
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
        with torch.no_grad():
            input_ids = torch.tensor([entry["ids"]])
            out = model(input_ids=input_ids, output_hidden_states=want_residuals)
            ref = out.logits[0, -1].float().numpy()
        if want_residuals:
            ok &= compare_residuals(entry, out, captured.get("pre_norm"), args.residual_tolerance)
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
            gen = model.generate(input_ids, max_new_tokens=args.max_new_tokens, do_sample=False)
        ref_text = tok.decode(gen[0, input_ids.shape[1]:], skip_special_tokens=False)
        print(f"  greedy transformers: {ref_text!r}")
        print(f"  greedy ditch:        {entry['response']!r}")
    print("\nRESULT:", "OK" if ok else "MISMATCH")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
