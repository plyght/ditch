#!/usr/bin/env python3
"""Check ditch's rendering of each release's own chat template against transformers.

    zig build
    python3 tools/chat_template_check.py [--only SUBSTR] [--limit N] [--revision REV]
        [--ditch zig-out/bin/ditch] [--cache DIR] [--report FILE] [--reference-only]

For every release in RELEASES (at least one per `model_type` of the registry
in src/arch.zig, plus extra releases where a family has had several template
generations) the script downloads the tokenizer and template files only — no
weights — into `~/.cache/ditch-template-check/<owner>__<name>`, writes a
CASES.json there, renders every case with transformers and with

    ditch render-template <DIR> <CASES.json>

and compares the two per case: the rendered text, and the token ids when both
sides have them. One line per release is printed as it runs, then a markdown
table (release | family | cases | text matches | ids matches | notes). Every
mismatching case gets a unified diff of the two texts, one repr'd line per
rendered line so whitespace is visible, in the report file (default
`<cache>/report.md`).

The reference is `AutoTokenizer.from_pretrained(dir)` (retried with
`trust_remote_code=True`) and `apply_chat_template(..., tokenize=False)` for
the text, `tokenize=True` for the ids. When no tokenizer loads (a slow-only
SentencePiece or tiktoken tokenizer whose package is missing, say), the
template is rendered with transformers' own `render_jinja_template` and the
special tokens transformers would pass, and the ids are marked unavailable.
A template that raises is recorded as that error; ditch then reports an
error or its fallback rendering, and both are shown.

`--reference-only` runs only the transformers half. Reference outputs are
cached as `reference.json` next to the downloaded files, keyed by the files,
the cases and the transformers version (and the date, for templates that
write today's date), so later runs do not recompute them.
"""
import argparse
import difflib
import hashlib
import json
import os
import re
import subprocess
import sys
import time
import warnings

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
warnings.filterwarnings("ignore")

import transformers  # noqa: E402
from huggingface_hub import HfApi, hf_hub_download  # noqa: E402
from huggingface_hub.utils import (  # noqa: E402
    EntryNotFoundError,
    GatedRepoError,
    HfHubHTTPError,
    RepositoryNotFoundError,
    RevisionNotFoundError,
)
from transformers import AutoTokenizer  # noqa: E402
from transformers.utils.chat_template_utils import render_jinja_template  # noqa: E402

transformers.logging.set_verbosity_error()

DEFAULT_DITCH = "/home/user/ditch/zig-out/bin/ditch"
DEFAULT_CACHE = os.path.expanduser("~/.cache/ditch-template-check")

# (repo id, registry model_type, note). The model_type is the registry entry
# (src/arch.zig) the release resolves to, aliases folded into their entry.
# Gated repos (meta-llama/*, google/gemma-*, CohereLabs/*, ...) are replaced by
# ungated re-uploads of the same files.
RELEASES = [
    # llama: Llama 2 / 3 / 3.1 / 3.2 / 3.3, and the releases that reuse its layout
    ("unsloth/llama-2-7b-chat", "llama", "Llama 2 [INST]"),
    ("unsloth/llama-3-8b-Instruct", "llama", "Llama 3"),
    ("unsloth/Meta-Llama-3.1-8B-Instruct", "llama", "Llama 3.1 dated system block"),
    ("unsloth/Llama-3.2-1B-Instruct", "llama", "Llama 3.2"),
    ("unsloth/Llama-3.3-70B-Instruct", "llama", "Llama 3.3"),
    ("HuggingFaceTB/SmolLM2-1.7B-Instruct", "llama", "SmolLM2 ChatML"),
    ("tiiuae/Falcon3-1B-Instruct", "llama", "Falcon 3"),
    ("nvidia/Llama-3.1-Nemotron-Nano-8B-v1", "llama", "Nemotron Nano, detailed thinking"),
    ("deepseek-ai/DeepSeek-R1-Distill-Llama-8B", "llama", "R1 distill"),
    ("mistralai/Mistral-Small-3.1-24B-Instruct-2503", "llama", "mistral3 (mistral3_text) V7-tekken"),
    # mistral: v0.1 / v0.3 / Nemo / V7-tekken
    ("mistralai/Mistral-7B-Instruct-v0.1", "mistral", "v0.1"),
    ("mistralai/Mistral-7B-Instruct-v0.3", "mistral", "v0.3"),
    ("mistralai/Mistral-Nemo-Instruct-2407", "mistral", "Nemo tekken"),
    ("mistralai/Mistral-Small-24B-Instruct-2501", "mistral", "V7-tekken"),
    ("mistralai/Ministral-8B-Instruct-2410", "mistral", "ministral"),
    ("mistralai/Ministral-3-3B-Instruct-2512", "ministral3", ""),
    ("mistralai/Mistral-Small-4-119B-2603", "mistral4", ""),
    ("mistralai/Mixtral-8x7B-Instruct-v0.1", "mixtral", ""),
    # qwen
    ("Qwen/Qwen2-0.5B-Instruct", "qwen2", "Qwen2"),
    ("Qwen/Qwen2.5-0.5B-Instruct", "qwen2", "Qwen2.5"),
    ("Qwen/Qwen2.5-VL-3B-Instruct", "qwen2", "qwen2_5_vl, processor template"),
    ("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", "qwen2", "R1 distill"),
    ("Qwen/Qwen3-0.6B", "qwen3", "Qwen3 hybrid thinking"),
    ("Qwen/Qwen3-4B-Instruct-2507", "qwen3", "Qwen3 2507 instruct"),
    ("Qwen/Qwen3-4B-Thinking-2507", "qwen3", "Qwen3 2507 thinking"),
    ("Qwen/Qwen3-VL-2B-Instruct", "qwen3", "qwen3_vl"),
    ("Qwen/Qwen1.5-MoE-A2.7B-Chat", "qwen2_moe", ""),
    ("Qwen/Qwen3-30B-A3B", "qwen3_moe", ""),
    ("Qwen/Qwen3-30B-A3B-Instruct-2507", "qwen3_moe", "2507"),
    ("Qwen/Qwen3-Next-80B-A3B-Instruct", "qwen3_next", ""),
    ("Qwen/Qwen3-Next-80B-A3B-Thinking", "qwen3_next", "thinking"),
    ("Qwen/Qwen3.5-0.8B", "qwen3_5", "Qwen3.5"),
    ("Qwen/Qwen3.8-27B", "qwen3_5", "Qwen3.8"),
    ("Qwen/Qwen3.5-35B-A3B", "qwen3_5_moe", ""),
    ("Qwen/Qwen3.5-397B-A17B", "qwen3_5_moe", ""),
    ("Qwen/Qwen3.8-Flash-Next", "qwen4_exp", ""),
    # gemma
    ("unsloth/gemma-2-2b-it", "gemma2", ""),
    ("unsloth/gemma-3-1b-it", "gemma3", "text only"),
    ("unsloth/gemma-3-4b-it", "gemma3", "multimodal"),
    ("unsloth/gemma-3n-E2B-it", "gemma3n", ""),
    ("google/gemma-4-E2B-it", "gemma4", ""),
    ("google/gemma-4-12B-it", "gemma4", "gemma4_unified"),
    # phi
    ("microsoft/phi-2", "phi", "base"),
    ("microsoft/Phi-3-mini-4k-instruct", "phi3", "Phi-3"),
    ("microsoft/Phi-3.5-mini-instruct", "phi3", "Phi-3.5"),
    ("microsoft/phi-4", "phi3", "Phi-4 im_sep"),
    ("microsoft/Phi-4-mini-instruct", "phi3", "Phi-4-mini"),
    ("microsoft/Phi-4-mini-reasoning", "phi3", "Phi-4-mini reasoning"),
    # older base models (most have no template)
    ("EleutherAI/pythia-160m", "gpt_neox", ""),
    ("openai-community/gpt2", "gpt2", ""),
    ("tiiuae/falcon-7b-instruct", "falcon", ""),
    ("stabilityai/stablelm-2-1_6b-chat", "stablelm", ""),
    ("stabilityai/stablelm-zephyr-3b", "stablelm", "Zephyr"),
    ("internlm/internlm2_5-1_8b-chat", "internlm2", ""),
    ("allenai/OLMo-2-0425-1B-Instruct", "olmo2", "OLMo 2"),
    ("allenai/Olmo-3-7B-Instruct", "olmo2", "olmo3"),
    ("allenai/Olmo-3-7B-Think", "olmo2", "olmo3 think"),
    ("allenai/OLMo-1B-hf", "olmo", ""),
    ("allenai/OLMo-7B-0724-Instruct-hf", "olmo", "instruct"),
    # cohere
    ("adamo1139/aya-expanse-8b-ungated", "cohere", "Aya Expanse (Command R format)"),
    ("Cossale/aya-expanse-8b-formal", "cohere", "Aya Expanse fine-tune"),
    ("estrogen/c4ai-command-r7b-12-2024", "cohere", "Command R7B (cohere2)"),
    # glm
    ("THUDM/glm-4-9b-chat-hf", "glm4", "glm (GLM-4 HF port)"),
    ("zai-org/GLM-4-9B-0414", "glm4", "GLM-4 0414"),
    ("THUDM/glm-4-9b-chat", "chatglm", "remote code"),
    ("THUDM/chatglm3-6b", "chatglm", "ChatGLM3"),
    ("zai-org/GLM-4.5-Air", "glm4_moe", "GLM-4.5"),
    ("zai-org/GLM-4.6", "glm4_moe", "GLM-4.6"),
    ("zai-org/GLM-4.7", "glm4_moe", "GLM-4.7"),
    ("zai-org/GLM-4.7-Flash", "glm4_moe_lite", ""),
    ("zai-org/GLM-5.2", "glm_moe_dsa", "GLM-5.2"),
    ("zai-org/GLM-5.3", "glm_moe_dsa", "GLM-5.3"),
    ("zai-org/GLM-5.3-Flash", "glm5_next", ""),
    # granite
    ("ibm-granite/granite-3.1-2b-instruct", "granite", "Granite 3.1"),
    ("ibm-granite/granite-3.3-2b-instruct", "granite", "Granite 3.3"),
    ("ibm-granite/granite-4.1-3b", "granite", "Granite 4.1"),
    ("ibm-granite/granite-3.0-1b-a400m-instruct", "granitemoe", ""),
    ("ibm-granite/granite-4.0-h-350m", "granitemoehybrid", "Granite 4 H"),
    ("ibm-granite/granite-4.0-micro", "granitemoehybrid", "Granite 4 micro"),
    ("ibm-granite/granite-swash-2b", "granite_swa", ""),
    # deepseek
    ("deepseek-ai/DeepSeek-V2-Lite-Chat", "deepseek_v2", ""),
    ("deepseek-ai/DeepSeek-V3", "deepseek_v3", "V3"),
    ("deepseek-ai/DeepSeek-R1-0528", "deepseek_v3", "R1 0528"),
    ("deepseek-ai/DeepSeek-V3.1", "deepseek_v3", "V3.1 thinking kwarg"),
    ("moonshotai/Kimi-K2-Instruct-0905", "deepseek_v3", "Kimi K2 (model_type kimi_k2)"),
    ("deepseek-ai/DeepSeek-V3.2-Exp", "deepseek_v32", ""),
    ("deepseek-ai/DeepSeek-V3.2", "deepseek_v32", "V3.2"),
    ("deepseek-ai/DeepSeek-V4-Flash", "deepseek_v4", ""),
    ("deepseek-ai/DeepSeek-V4.1-Flash", "deepseek_v41", ""),
    ("unsloth/Llama-4-Scout-17B-16E-Instruct", "llama4", ""),
    ("openai/gpt-oss-20b", "gpt_oss", "harmony"),
    ("openbmb/MiniCPM-2B-sft-bf16", "minicpm", ""),
    ("openbmb/MiniCPM4-0.5B", "minicpm", "MiniCPM4"),
    ("LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct", "exaone", "EXAONE 3.5"),
    ("LGAI-EXAONE/EXAONE-4.0-1.2B", "exaone4", "EXAONE 4"),
    ("LGAI-EXAONE/EXAONE-4.0.1-32B", "exaone4", "EXAONE 4.0.1"),
    ("LGAI-EXAONE/K-EXAONE-236B-A23B", "exaone_moe", ""),
    ("nvidia/Nemotron-Mini-4B-Instruct", "nemotron", ""),
    ("nvidia/Minitron-4B-Base", "nemotron", "base"),
    ("HuggingFaceTB/SmolLM3-3B", "smollm3", ""),
    ("bigscience/bloomz-560m", "bloom", ""),
    ("facebook/opt-125m", "opt", ""),
    ("vinai/PhoGPT-4B-Chat", "mpt", "PhoGPT; mosaicml/mpt-* are gone"),
    ("bigcode/starcoder2-3b", "starcoder2", "base"),
    ("bigcode/starcoder2-15b-instruct-v0.1", "starcoder2", "instruct"),
    ("bigcode/tiny_starcoder_py", "gpt_bigcode", "base"),
    ("HuggingFaceH4/starchat-beta", "gpt_bigcode", "StarChat"),
    ("baichuan-inc/Baichuan2-7B-Chat", "baichuan", "remote code"),
    ("AntonV/mamba2-130m-hf", "mamba2", ""),
    ("nvidia/NVIDIA-Nemotron-Nano-9B-v2", "nemotron_h", "Nemotron Nano 2"),
    ("nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16", "nemotron_h", "Nemotron 3 Nano"),
    ("tiiuae/Falcon-H1-0.5B-Instruct", "falcon_h1", ""),
    ("ai21labs/Jamba-tiny-dev", "jamba", ""),
    ("ai21labs/AI21-Jamba-Reasoning-3B", "jamba", "reasoning"),
    ("MiniMaxAI/MiniMax-M2", "minimax_m2", ""),
    ("MiniMaxAI/MiniMax-M1-40k", "minimax", "M1"),
    ("MiniMaxAI/MiniMax-Text-01-hf", "minimax", "Text-01"),
    ("MiniMaxAI/MiniMax-M3", "minimax_m3_vl_text", ""),
    ("baidu/ERNIE-4.5-21B-A3B-PT", "ernie4_5_moe", ""),
    ("baidu/ERNIE-4.5-0.3B-PT", "ernie4_5", ""),
    ("tencent/Hunyuan-A13B-Instruct", "hunyuan_v1_moe", ""),
    ("tencent/Hunyuan-0.5B-Instruct", "hunyuan_v1_dense", ""),
    ("tencent/Hunyuan-7B-Instruct", "hunyuan_v1_dense", "7B"),
    ("tencent/Hy3-preview", "hy_v3", ""),
    ("moonshotai/Kimi-Linear-48B-A3B-Instruct", "kimi_linear", ""),
    ("moonshotai/Kimi-K2.5", "kimi_k25", ""),
    ("moonshotai/Kimi-K3", "kimi_k3", ""),
    ("XiaomiMiMo/MiMo-V2-Flash", "mimo_v2_flash", ""),
    ("XiaomiMiMo/MiMo-V2.5", "mimo_v2_flash", "mimo_v2"),
    ("arcee-ai/AFM-4.5B", "arcee", ""),
    ("swiss-ai/Apertus-8B-Instruct-2509", "apertus", ""),
    ("microsoft/bitnet-b1.58-2B-4T", "bitnet", ""),
    ("kyutai/helium-1-preview-2b", "helium", ""),
    ("ByteDance-Seed/Seed-OSS-36B-Instruct", "seed_oss", ""),
    ("LiquidAI/LFM2-350M", "lfm2", ""),
    ("HeshamSA/jais-2-8b-chat-mxfp4-msa", "jais2", "quantised re-upload; inception42/Jais-2-8B-Chat is gated"),
    ("nanochat-students/nanochat-d20", "nanochat", ""),
    ("pankajmathur/nanochat-d34-sft-hf", "nanochat", "HF port"),
    ("adept/persimmon-8b-chat", "persimmon", ""),
    ("EleutherAI/gpt-j-6b", "gptj", ""),
    ("Salesforce/codegen-350M-mono", "codegen", ""),
    ("EleutherAI/gpt-neo-125m", "gpt_neo", ""),
    ("facebook/xglm-564M", "xglm", ""),
    ("microsoft/biogpt", "biogpt", ""),
    ("allenai/OLMoE-1B-7B-0924-Instruct", "olmoe", ""),
    ("allenai/OLMoE-1B-7B-0125-Instruct", "olmoe", "0125"),
    ("allenai/FlexOlmo-7x7B-1T-RT", "flex_olmo", ""),
    ("rednote-hilab/dots.llm1.inst", "dots1", ""),
    ("upstage/Solar-Open-100B", "solar_open", ""),
    ("arcee-ai/Trinity-Nano-Preview", "afmoe", ""),
    ("JetBrains/Mellum2-12B-A2.5B-Instruct", "mellum", ""),
    ("poolside/Laguna-XS.2", "laguna", ""),
    ("poolside/Laguna-XS-2.1", "laguna", "2.1"),
]

# Top-level files fetched from each repo. The spec list plus what
# AutoTokenizer needs for older layouts (config.json names the tokenizer
# class of some releases; vocab/merges for slow-only tokenizers) and remote
# tokenizer code. Never weights.
WANTED = {
    "tokenizer_config.json", "chat_template.jinja", "chat_template.json",
    "special_tokens_map.json", "tokenizer.json", "tokenizer.model",
    "config.json", "vocab.json", "merges.txt", "vocab.txt", "added_tokens.json",
}


def wanted(name):
    if "/" in name:
        return False
    if name in WANTED or "tiktoken" in name:
        return True
    return name.endswith(".py") and name.startswith(("tokenization", "configuration"))


SYSTEM = "You are a helpful assistant."
QUESTION = "What is the capital of France?"
HARD_SYSTEM = "  You are a careful assistant. Answer in \"plain\" text; keep {braces} as they are.\n"
HARD_USER = (" Explain this snippet, s'il vous plaît — naïve café 東京 😀:\n\n"
             "```python\ndef f(x):\n    return {'a': x}  # \"quoted\"\n```\n  ")


def build_cases(template):
    """The CASES.json list for a release, given its template text (or None)."""
    sys_user = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": QUESTION}]
    cases = [
        {"name": "sys+user", "messages": sys_user, "add_generation_prompt": True, "kwargs": {}},
        {"name": "user", "messages": [{"role": "user", "content": QUESTION}], "add_generation_prompt": True, "kwargs": {}},
        {"name": "multi-turn", "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": "Hi! Who are you?"},
            {"role": "assistant", "content": "I am a helpful assistant. How can I help?"},
            {"role": "user", "content": QUESTION},
        ], "add_generation_prompt": True, "kwargs": {}},
        {"name": "sys+user no-agp", "messages": sys_user, "add_generation_prompt": False, "kwargs": {}},
        {"name": "whitespace/unicode", "messages": [
            {"role": "system", "content": HARD_SYSTEM}, {"role": "user", "content": HARD_USER},
        ], "add_generation_prompt": True, "kwargs": {}},
    ]
    t = template or ""
    variants = []
    if "enable_thinking" in t:
        variants += [{"enable_thinking": True}, {"enable_thinking": False}]
    if "reasoning_effort" in t:
        # The two extreme levels the template names (they differ between
        # releases: "none"/"high", "low"/"xhigh"), else low and high.
        levels = [v for v in ("none", "minimal", "low", "medium", "high", "xhigh", "max")
                  if re.search(rf"['\"]{v}['\"]", t)]
        if len(levels) < 2:
            levels = ["low", "high"]
        variants += [{"reasoning_effort": levels[0]}, {"reasoning_effort": levels[-1]}]
    if re.search(r"\bthinking\s+is\s+(?:defined|sameas|true|false)|\bif\s+(?:not\s+)?thinking\b", t):
        variants += [{"thinking": True}, {"thinking": False}]
    if "thinking_mode" in t:
        variants += [{"thinking_mode": "on"}, {"thinking_mode": "off"}]
    for kw in variants:
        k, v = next(iter(kw.items()))
        cases.append({"name": f"sys+user {k}={v}", "messages": sys_user,
                      "add_generation_prompt": True, "kwargs": kw})
    return cases


def release_dir(cache, repo, revision):
    name = repo.replace("/", "__")
    if revision:
        name += "@" + revision.replace("/", "_")
    return os.path.join(cache, name)


def http_status(e):
    resp = getattr(e, "response", None)
    return getattr(resp, "status_code", None)


def download(repo, dest, revision, refresh):
    """Fetches the tokenizer/template files. Returns (files, error)."""
    os.makedirs(dest, exist_ok=True)
    manifest = os.path.join(dest, "files.json")
    if not refresh and os.path.exists(manifest):
        with open(manifest) as f:
            m = json.load(f)
        if m.get("revision") == revision:
            return m["files"], m.get("error")
    try:
        names = [n for n in HfApi().list_repo_files(repo, revision=revision) if wanted(n)]
    except GatedRepoError:
        return [], "gated (401)"
    except RevisionNotFoundError:
        return [], "revision not found (404)"
    except RepositoryNotFoundError:
        return [], "not found or private (401/404)"
    except HfHubHTTPError as e:
        return [], f"HTTP {http_status(e)}"
    got, error = [], None
    for n in sorted(names):
        path = os.path.join(dest, n)
        if os.path.exists(path) and not refresh:
            got.append(n)
            continue
        try:
            hf_hub_download(repo, n, revision=revision, local_dir=dest)
            got.append(n)
        except GatedRepoError:
            error = "gated (401)"
            break
        except (EntryNotFoundError, RepositoryNotFoundError):
            continue
        except HfHubHTTPError as e:
            error = f"HTTP {http_status(e)} on {n}"
    if error and not got:
        return got, error
    with open(manifest, "w") as f:
        json.dump({"revision": revision, "files": got, "error": error}, f)
    return got, error


def read_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def find_template(d):
    """(template text, source) the way transformers picks it, or (None, None)."""
    p = os.path.join(d, "chat_template.jinja")
    if os.path.exists(p):
        with open(p) as f:
            return f.read(), "chat_template.jinja"
    tc = read_json(os.path.join(d, "tokenizer_config.json")) or {}
    ct = tc.get("chat_template")
    if isinstance(ct, list):
        named = {e.get("name"): e.get("template") for e in ct if isinstance(e, dict)}
        if named.get("default"):
            return named["default"], "tokenizer_config.json[default]"
    elif isinstance(ct, str) and ct:
        return ct, "tokenizer_config.json"
    cj = read_json(os.path.join(d, "chat_template.json")) or {}
    if isinstance(cj.get("chat_template"), str):
        return cj["chat_template"], "chat_template.json"
    return None, None


def special_tokens(d):
    """The special-token template variables transformers passes, from the files."""
    def text(v):
        if isinstance(v, str):
            return v
        if isinstance(v, dict) and isinstance(v.get("content"), str):
            return v["content"]
        return None
    out = {}
    stm = read_json(os.path.join(d, "special_tokens_map.json")) or {}
    tc = read_json(os.path.join(d, "tokenizer_config.json")) or {}
    for src in (stm, tc):
        for k, v in src.items():
            if k.endswith("_token") and text(v) is not None:
                out[k] = text(v)
    return out


def ids_of(x):
    if hasattr(x, "keys") and "input_ids" in x:
        x = x["input_ids"]
    if hasattr(x, "tolist"):
        x = x.tolist()
    if x and isinstance(x[0], list):
        x = x[0]
    return [int(i) for i in x]


def load_tokenizer(d):
    errors = []
    for trust in (False, True):
        try:
            return AutoTokenizer.from_pretrained(d, trust_remote_code=trust), None
        except Exception as e:  # noqa: BLE001 - any loader failure falls back
            errors.append(f"{type(e).__name__}: {str(e).splitlines()[0][:160] if str(e) else ''}")
    return None, errors[-1]


def reference(d, cases, template):
    """transformers' rendering of every case: {"loader", "note", "results"}."""
    tok, load_error = load_tokenizer(d)
    note = ""
    if tok is not None and not getattr(tok, "chat_template", None) and template:
        note = "template passed explicitly"
    if tok is None:
        note = f"no tokenizer ({load_error}); ids unavailable"
    results = []
    for c in cases:
        if not template:
            results.append({"error": "no chat template"})
            continue
        msgs, agp, kw = c["messages"], c["add_generation_prompt"], c["kwargs"]
        try:
            if tok is not None:
                extra = {"chat_template": template} if note else {}
                text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=agp, **extra, **kw)
                ids = ids_of(tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=agp, **extra, **kw))
                results.append({"text": text, "ids": ids})
            else:
                rendered, _ = render_jinja_template(
                    conversations=[msgs], chat_template=template,
                    add_generation_prompt=agp, **{**special_tokens(d), **kw})
                results.append({"text": rendered[0]})
        except Exception as e:  # noqa: BLE001 - a raising template is a result
            results.append({"error": f"{type(e).__name__}: {e}"})
    return {"loader": "tokenizer" if tok is not None else "jinja", "template": bool(template),
            "note": note, "results": results}


def reference_key(d, files, cases, template):
    h = hashlib.sha256()
    for n in sorted(files):
        p = os.path.join(d, n)
        if os.path.exists(p):
            h.update(n.encode())
            h.update(str(os.path.getsize(p)).encode())
            h.update(str(int(os.path.getmtime(p))).encode())
    h.update(json.dumps(cases, sort_keys=True).encode())
    h.update(transformers.__version__.encode())
    h.update(b"v2")  # bump when the reference format changes
    if template and re.search(r"strftime_now|date_string|today", template):
        h.update(time.strftime("%Y-%m-%d").encode())
    return h.hexdigest()


def cached_reference(d, files, cases, template):
    key = reference_key(d, files, cases, template)
    path = os.path.join(d, "reference.json")
    old = read_json(path)
    if old and old.get("key") == key:
        return old, True
    ref = reference(d, cases, template)
    ref["key"] = key
    with open(path, "w") as f:
        json.dump(ref, f, ensure_ascii=False)
    return ref, False


def run_ditch(ditch, d, cases_path):
    """ditch's output object, or {"fatal": message}."""
    if not os.path.exists(ditch):
        return {"fatal": f"{ditch} not found (zig build)"}
    try:
        p = subprocess.run([ditch, "render-template", d, cases_path], capture_output=True, text=True, timeout=300)
    except subprocess.TimeoutExpired:
        return {"fatal": "ditch timed out"}
    if p.returncode != 0:
        tail = (p.stderr.strip().splitlines() or ["(no stderr)"])[-1]
        return {"fatal": f"ditch exit {p.returncode}: {tail[:200]}"}
    try:
        return json.loads(p.stdout)
    except ValueError:
        return {"fatal": "ditch printed no JSON"}


def text_diff(a, b):
    la = [repr(x) for x in a.splitlines(keepends=True)]
    lb = [repr(x) for x in b.splitlines(keepends=True)]
    return "\n".join(difflib.unified_diff(la, lb, "transformers", "ditch", lineterm=""))


def compare(repo, family, cases, ref, out, report):
    """Per-release counts and notes; mismatches go to the report list."""
    n = len(cases)
    text_ok = ids_ok = ids_n = 0
    notes = []
    ref_err = sum(1 for r in ref["results"] if "error" in r)
    if ref_err and ref.get("template"):
        notes.append(f"transformers errors {ref_err}")
    if "fatal" in out:
        notes.append(out["fatal"])
        return {"text": f"-/{n}", "ids": "-", "notes": notes}
    notes.append(f"ditch: {out.get('renderer')}")
    if out.get("warnings"):
        notes.append("; ".join(out["warnings"])[:160])
    results = out.get("results", [])
    if not ref.get("template"):
        # Nothing to compare with: ditch's fallback is recorded, not scored.
        rendered = sum(1 for o in results if "text" in o)
        notes.append(f"ditch rendered {rendered}/{n} without a template")
        return {"text": "n/a", "ids": "n/a", "notes": notes}
    if len(results) != n:
        notes.append(f"ditch returned {len(results)} results for {n} cases")
    both_err = 0
    for i, c in enumerate(cases):
        r = ref["results"][i]
        o = results[i] if i < len(results) else {"error": "missing"}
        if "error" in r or "error" in o:
            if "error" in r and "error" in o:
                text_ok += 1
                both_err += 1
            else:
                report.append((repo, c["name"], "transformers: " + r.get("error", "(rendered)")
                               + "\nditch: " + o.get("error", "(rendered)"),
                               text_diff(r.get("text", ""), o.get("text", ""))))
            continue
        same = r["text"] == o.get("text")
        text_ok += same
        if "ids" in r and "ids" in o:
            ids_n += 1
            ids_ok += r["ids"] == o["ids"]
        if not same or ("ids" in r and "ids" in o and r["ids"] != o["ids"]):
            head = "text differs" if not same else "ids differ, text equal"
            if same:
                a, b = r["ids"], o["ids"]
                k = next((j for j in range(min(len(a), len(b))) if a[j] != b[j]), min(len(a), len(b)))
                head += f" (first difference at {k}: {a[k:k + 8]} vs {b[k:k + 8]})"
            report.append((repo, c["name"], head, text_diff(r["text"], o.get("text", ""))))
    if both_err:
        notes.append(f"both sides error on {both_err} (counted as matches)")
    return {"text": f"{text_ok}/{n}", "ids": f"{ids_ok}/{ids_n}" if ids_n else "n/a", "notes": notes}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ditch", default=DEFAULT_DITCH)
    ap.add_argument("--cache", default=DEFAULT_CACHE)
    ap.add_argument("--report", default=None, help="diff report (default <cache>/report.md)")
    ap.add_argument("--only", action="append", default=[], help="keep releases whose repo id or model_type contains this (repeatable)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--revision", default=None)
    ap.add_argument("--reference-only", action="store_true")
    ap.add_argument("--refresh", action="store_true", help="download the files again")
    args = ap.parse_args()

    releases = [r for r in RELEASES if not args.only or any(s.lower() in (r[0] + " " + r[1]).lower() for s in args.only)]
    if args.limit:
        releases = releases[: args.limit]
    os.makedirs(args.cache, exist_ok=True)
    report_path = args.report or os.path.join(args.cache, "report.md")

    rows, report = [], []
    for repo, family, note in releases:
        d = release_dir(args.cache, repo, args.revision)
        files, err = download(repo, d, args.revision, args.refresh)
        if not files:
            row = {"text": "-", "ids": "-", "n": 0, "notes": [f"download failed: {err or 'no tokenizer files'}"]}
            rows.append((repo, family, row))
            print(f"{repo:55} {family:18} {row['notes'][0]}", flush=True)
            continue
        template, source = find_template(d)
        cases = build_cases(template)
        cases_path = os.path.join(d, "CASES.json")
        with open(cases_path, "w") as f:
            json.dump(cases, f, ensure_ascii=False, indent=1)
        ref, hit = cached_reference(d, files, cases, template)
        extra = [note] if note else []
        extra.append(f"template: {source}" if template else "no template")
        if ref.get("note"):
            extra.append(ref["note"])
        if args.reference_only:
            errs = [r["error"] for r in ref["results"] if "error" in r and template]
            if errs:
                extra.append(f"transformers errors {len(errs)}: {errs[0][:120]}")
            row = {"text": "-", "ids": "-", "notes": extra}
        else:
            row = compare(repo, family, cases, ref, run_ditch(args.ditch, d, cases_path), report)
            row["notes"] = extra + row["notes"]
        row["n"] = len(cases)
        rows.append((repo, family, row))
        print(f"{repo:55} {family:18} cases {len(cases)} text {row['text']:>5} ids {row['ids']:>5}"
              f"{' (cached)' if hit else ''}  {'; '.join(row['notes'])}", flush=True)

    print()
    print("| release | family model_type | cases | text matches | ids matches | notes |")
    print("| --- | --- | ---: | ---: | ---: | --- |")
    for repo, family, row in rows:
        print(f"| {repo} | `{family}` | {row['n']} | {row['text']} | {row['ids']} | {'; '.join(row['notes']).replace('|', '/')} |")

    with open(report_path, "w") as f:
        f.write(f"# Chat template check ({time.strftime('%Y-%m-%d')}, transformers {transformers.__version__})\n\n")
        if args.reference_only:
            f.write("Reference only: ditch was not run.\n")
        elif not report:
            f.write("No mismatching case.\n")
        for repo, name, head, diff in report:
            f.write(f"## {repo} — {name}\n\n{head}\n\n```diff\n{diff}\n```\n\n")
    print(f"\nreport: {report_path}  cache: {args.cache}")


if __name__ == "__main__":
    main()
