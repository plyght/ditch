"""The reference half of `ditch verify`.

    python3 verify_reference.py --check-env
    python3 verify_reference.py MODEL PROBE_JSON OUT_JSON [--raw] [--max-new-tokens N] [--per-layer]

`--check-env` prints what is importable as JSON (`{"ok": bool, "missing":
[...], "versions": {...}}`) and exits 0 either way, so ditch can tell the user
what to install. Otherwise MODEL (a directory, or a Hub id for a full-depth
run) is compared with `ditch probe --residuals --json` output PROBE_JSON by
tools/probe_reference.py, whose numbers go to OUT_JSON.

The reference is the official implementation: transformers' own model class
for every family it has (tools/ref_stream.py, which loads a checkpoint of any
size a layer at a time through transformers' own loader), else the release's
own modeling code (DeepSeek V4 / V4.1, Kimi K3, MiMo V2, through their
tools/ref_*.py factories; any other `auto_map` checkpoint through
`trust_remote_code`).
"""
import importlib
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
NEEDED = ("torch", "transformers", "accelerate", "safetensors", "huggingface_hub", "numpy", "requests")
PIP = "pip install torch --index-url https://download.pytorch.org/whl/cpu && pip install transformers accelerate safetensors huggingface_hub numpy requests"

# model_type -> (factory, trust_remote_code), for families transformers does not have.
OWN_CODE = {
    "deepseek_v41": ("ref_deepseek_v41.py", False),
    "deepseek_v41_text": ("ref_deepseek_v41.py", False),
    "deepseek_v4": ("ref_deepseek_v4.py", False),
    "kimi_k3": ("ref_kimi_k3.py", True),
    "mimo_v2": ("ref_mimo_v2.py", True),
}


def check_env():
    missing, versions = [], {}
    for name in NEEDED:
        try:
            versions[name] = getattr(importlib.import_module(name), "__version__", "?")
        except Exception:
            missing.append(name)
    print(json.dumps({"ok": not missing, "missing": missing, "versions": versions, "python": sys.version.split()[0], "pip": PIP}))


def load_config(model):
    if os.path.isdir(model):
        return json.load(open(os.path.join(model, "config.json")))
    from huggingface_hub import hf_hub_download
    return json.load(open(hf_hub_download(model, "config.json")))


def choose(cfg):
    """(factory file or None, trust_remote_code) for a checkpoint's config."""
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING
    kinds = [cfg.get("model_type"), (cfg.get("text_config") or {}).get("model_type")]
    arch = " ".join(cfg.get("architectures") or [])
    if "KimiK3" in arch:
        return OWN_CODE["kimi_k3"]
    for k in kinds:
        if k in OWN_CODE and k not in CONFIG_MAPPING:
            return OWN_CODE[k]
    if cfg.get("model_type") in CONFIG_MAPPING:
        return "ref_stream.py", False
    return None, bool(cfg.get("auto_map"))


def main():
    if "--check-env" in sys.argv:
        check_env()
        return
    args = [a for a in sys.argv[1:]]
    model, probe_json, out_json = args[:3]
    rest = args[3:]
    factory, trust = choose(load_config(model))
    cmd = [sys.executable, os.path.join(HERE, "probe_reference.py"), model, probe_json, "--dtype", "float32", "--json-out", out_json]
    if factory:
        cmd += ["--factory", os.path.join(HERE, factory)]
    if trust:
        cmd.append("--trust-remote-code")
    cmd += rest
    print("reference:", factory or "from_pretrained", "(trust_remote_code)" if trust else "", flush=True)
    sys.exit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
