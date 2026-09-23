#!/usr/bin/env python3
"""Writes the chat-template fixtures for src/jinja.zig's tests.

Usage: make_template_fixtures.py [<cache_dir>] [<out_dir>]
    cache_dir: where tools/chat_template_check.py downloaded the releases
               (default ~/.cache/ditch-template-check)
    out_dir:   default tests/fixtures/chat_templates

For each release in FIXTURES below, the release's own template (picked as
transformers picks it) is copied to `<name>.jinja`, and `expected.json`
records transformers' rendering of a few conversations with it: the special
tokens transformers would pass, `add_generation_prompt`, and the kwargs of
each case, with `strftime_now` pinned to NOW (UTC) so the dated templates
are reproducible. A case the template refuses is recorded as an error.

Needs transformers (its `_compile_jinja_template`, the same environment
`apply_chat_template` renders with).
"""
import datetime
import json
import os
import sys

from transformers.utils.chat_template_utils import _compile_jinja_template

NOW = 1790000000  # 2026-09-21 13:33:20 UTC

# (fixture name, release directory in the cache)
FIXTURES = [
    ("qwen2_5", "Qwen__Qwen2.5-0.5B-Instruct"),
    ("qwen3", "Qwen__Qwen3-0.6B"),
    ("qwen3_5", "Qwen__Qwen3.5-0.8B"),
    ("llama3_1", "unsloth__Meta-Llama-3.1-8B-Instruct"),
    ("llama3_2", "unsloth__Llama-3.2-1B-Instruct"),
    ("llama4", "unsloth__Llama-4-Scout-17B-16E-Instruct"),
    ("gemma2", "unsloth__gemma-2-2b-it"),
    ("gemma3", "unsloth__gemma-3-1b-it"),
    ("gemma4", "google__gemma-4-E2B-it"),
    ("mistral_v0_1", "mistralai__Mistral-7B-Instruct-v0.1"),
    ("mistral_v0_3", "mistralai__Mistral-7B-Instruct-v0.3"),
    ("ministral3", "mistralai__Ministral-3-3B-Instruct-2512"),
    ("gpt_oss", "openai__gpt-oss-20b"),
    ("deepseek_v3_1", "deepseek-ai__DeepSeek-V3.1"),
    ("glm4_7", "zai-org__GLM-4.7-Flash"),
    ("phi4_mini", "microsoft__Phi-4-mini-instruct"),
    ("granite3_3", "ibm-granite__granite-3.3-2b-instruct"),
    ("smollm3", "HuggingFaceTB__SmolLM3-3B"),
    ("command_r7b", "estrogen__c4ai-command-r7b-12-2024"),
    ("jamba", "ai21labs__Jamba-tiny-dev"),
    ("minimax_text01", "MiniMaxAI__MiniMax-Text-01-hf"),
    ("kimi_k2", "moonshotai__Kimi-K2-Instruct-0905"),
    ("exaone4", "LGAI-EXAONE__EXAONE-4.0-1.2B"),
    ("hunyuan", "tencent__Hunyuan-7B-Instruct"),
    ("seed_oss", "ByteDance-Seed__Seed-OSS-36B-Instruct"),
    ("falcon_h1", "tiiuae__Falcon-H1-0.5B-Instruct"),
    ("apertus", "swiss-ai__Apertus-8B-Instruct-2509"),
    ("laguna", "poolside__Laguna-XS.2"),
    ("nemotron_nano2", "nvidia__NVIDIA-Nemotron-Nano-9B-v2"),
    ("olmo3_think", "allenai__Olmo-3-7B-Think"),
]

S = "You are a helpful assistant."
U = "What is the capital of France?"
CASES = [
    ([{"role": "system", "content": S}, {"role": "user", "content": U}], True, {}),
    ([{"role": "user", "content": U}], True, {}),
    ([{"role": "system", "content": S}, {"role": "user", "content": "Hi"},
      {"role": "assistant", "content": "Hello! How can I help?"}, {"role": "user", "content": U}], True, {}),
    ([{"role": "system", "content": "  Be terse.\n"},
      {"role": "user", "content": " Hi é 日本 {x} 'q' \"d\"\n```py\nprint(1)\n```\n "}], False, {}),
    ([{"role": "system", "content": S}, {"role": "user", "content": U}], True, {"enable_thinking": False}),
]


def pick_template(d):
    """chat_template.jinja, else tokenizer_config.json's (default of a list), else chat_template.json."""
    if os.path.exists(os.path.join(d, "chat_template.jinja")):
        return open(os.path.join(d, "chat_template.jinja"), encoding="utf-8").read()
    tc = json.load(open(os.path.join(d, "tokenizer_config.json"), encoding="utf-8"))
    ct = tc.get("chat_template")
    if isinstance(ct, str):
        return ct
    if isinstance(ct, list):
        named = {x["name"]: x["template"] for x in ct}
        return named.get("default", ct[0]["template"])
    return json.load(open(os.path.join(d, "chat_template.json"), encoding="utf-8"))["chat_template"]


def special_tokens(d):
    """The `*_token` entries of tokenizer_config.json, as transformers passes them."""
    p = os.path.join(d, "tokenizer_config.json")
    tc = json.load(open(p, encoding="utf-8")) if os.path.exists(p) else {}
    out = {}
    for k, v in tc.items():
        if k.endswith("_token"):
            if isinstance(v, dict) and "content" in v:
                v = v["content"]
            if isinstance(v, str):
                out[k] = v
    return out


def main():
    cache = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/.cache/ditch-template-check")
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "tests/fixtures/chat_templates"
    os.makedirs(out_dir, exist_ok=True)
    expected = []
    for name, release in FIXTURES:
        d = os.path.join(cache, release)
        src = pick_template(d)
        with open(os.path.join(out_dir, name + ".jinja"), "w", encoding="utf-8") as f:
            f.write(src)
        tmpl = _compile_jinja_template(src)
        tmpl.globals["strftime_now"] = lambda fmt: datetime.datetime.fromtimestamp(NOW, datetime.timezone.utc).strftime(fmt)
        tokens = special_tokens(d)
        for messages, agp, kwargs in CASES:
            if kwargs and "enable_thinking" not in src:
                continue
            case = {"template": name, "tokens": tokens, "messages": messages, "add_generation_prompt": agp, "kwargs": kwargs}
            try:
                case["want"] = tmpl.render(messages=messages, tools=None, documents=None, add_generation_prompt=agp, **tokens, **kwargs)
            except Exception as e:  # the template's own raise_exception, or a content shape it rejects
                case["error"] = str(e)
            expected.append(case)
        print(name, release)
    with open(os.path.join(out_dir, "expected.json"), "w", encoding="utf-8") as f:
        json.dump({"now": NOW, "cases": expected}, f, ensure_ascii=False, indent=1)
        f.write("\n")
    print(f"{len(expected)} cases in {out_dir}/expected.json")


if __name__ == "__main__":
    main()
