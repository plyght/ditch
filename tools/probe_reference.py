#!/usr/bin/env python3
"""Compare `ditch probe --json` with Hugging Face transformers on the CPU.

    ditch probe MODEL --prompt "..." --prompt "..." --json > probe.json
    python3 tools/probe_reference.py MODEL probe.json [--dtype bfloat16]

For every prompt the script renders the chat template with the same system
prompt, tokenises it with `tokenizers`, runs the model once and compares:
the rendered text, the token ids, the first-token logits (max absolute
difference relative to the logit range, and the argmax) and the greedy
continuation. Exit status 1 when the ids differ or the relative logit
error exceeds --tolerance.
"""
import argparse
import json
import sys

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("probe_json")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--tolerance", type=float, default=1e-2)
    ap.add_argument("--system-prompt", default="You are a helpful assistant.")
    ap.add_argument("--max-new-tokens", type=int, default=32)
    args = ap.parse_args()

    probe = json.load(open(args.probe_json))
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=getattr(torch, args.dtype))
    model.eval()
    ok = True
    for entry in probe["prompts"]:
        messages = [{"role": "system", "content": args.system_prompt}, {"role": "user", "content": entry["user"]}]
        try:
            text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception as exc:  # models without a system role (Gemma)
            print(f"chat template with system role failed ({exc}); retrying without it")
            text = tok.apply_chat_template(messages[1:], tokenize=False, add_generation_prompt=True)
        ids = tok(text, add_special_tokens=True if tok.bos_token and not text.startswith(tok.bos_token or "\0") else False)["input_ids"]
        print(f"\n== {entry['user'][:60]!r}")
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
        with torch.no_grad():
            input_ids = torch.tensor([entry["ids"]])
            out = model(input_ids=input_ids)
            ref = out.logits[0, -1].float().numpy()
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
