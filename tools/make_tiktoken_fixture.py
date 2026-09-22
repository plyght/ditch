#!/usr/bin/env python3
"""Generates the tiktoken tokenizer fixtures in tests/fixtures/tiktoken_kimi and
tests/fixtures/tiktoken_llama3: a small byte-level BPE vocabulary in tiktoken's
`base64-token rank` line format, the matching tokenizer_config.json, and
reference encodings computed by the `tiktoken` package for a set of strings
(unicode, whitespace runs, digits, contractions, special tokens).

    pip install tiktoken numpy
    python3 tools/make_tiktoken_fixture.py

The vocabulary is trained by plain byte-pair merging over a small corpus that is
pre-tokenised with the Kimi pattern, so merges follow tiktoken's rank order.
One token (`qzxw`) is added that no sequence of merges produces: tiktoken
encodes a whole pre-token that is in the vocabulary as that single token before
merging (the "fast path"), and the fixture checks that ditch does the same.

The Kimi fixture mirrors moonshotai's `tokenization_kimi.py` (the pattern and
the special-token block of 256 ids after the base vocabulary, named through
`added_tokens_decoder`); the Llama 3 fixture uses Meta's `tokenizer.model`
convention (Llama 3 pattern, the fixed special-token list of the reference
tokenizer, no `added_tokens_decoder`).
"""
import base64
import json
import os
from collections import Counter

import regex
import tiktoken

KIMI_PAT = "|".join([
    r"""[\p{Han}]+""",
    r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?""",
    r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?""",
    r"""\p{N}{1,3}""",
    r""" ?[^\s\p{L}\p{N}]+[\r\n]*""",
    r"""\s*[\r\n]+""",
    r"""\s+(?!\S)""",
    r"""\s+""",
])
LLAMA3_PAT = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"

CORPUS = """
Hello world, hello there! It's a small world after all. The quick brown fox jumps over the lazy dog.
Kimi is a model made by Moonshot AI. Moonshot AI trained Kimi K2 on many tokens. The model answers questions.
你好世界。月之暗面训练了一个模型。这是一个测试。中文文本和英文文本混合在一起。模型模型模型。
naïve café résumé ÉCOLE École déjà vu. Straße Größe. niño mañana. Ωμέγα αβγ. Привет мир. こんにちは世界。한국어 텍스트.
The numbers 1 2 3 12 123 1234 12345 and 2024 and 1999. Prices: $12.50, 3.14, 100%.
    indented code block
        with more indentation
	tabs	and	spaces   and   runs   of   spaces
Line one.
Line two.

Line four after a blank line.
"What?" she said. 'Yes,' he said — and then... nothing. «Guillemets» and “curly quotes”.
I'm sure it's fine; we've done this; they'll come; you're right; he'd know; I'd say; we're here.
HTTPServer MixedCaseWords camelCase snake_case CONSTANT_NAME iPhone macOS.
emoji 😀 👍 🎉 done. symbols: + - * / = < > @ # & | ~ ^ % ! ? , . ; : ( ) [ ] { }
user assistant system tool response think message role content end begin
""" * 3


def pretokenize(pat, text):
    return [m.group(0) for m in regex.finditer(pat, text, flags=regex.V1)]


def train(pat, n_merges):
    """Byte-pair merging in rank order: returns the list of (bytes, rank)."""
    words = Counter(tuple(bytes([b]) for b in w.encode("utf-8")) for w in pretokenize(pat, CORPUS))
    ranks = {bytes([b]): b for b in range(256)}
    for _ in range(n_merges):
        pairs = Counter()
        for w, c in words.items():
            for a, b in zip(w, w[1:]):
                pairs[(a, b)] += c
        if not pairs:
            break
        (a, b), _ = max(pairs.items(), key=lambda kv: (kv[1], [-x for x in (kv[0][0] + kv[0][1])]))
        merged = a + b
        if merged in ranks:
            break
        ranks[merged] = len(ranks)
        new_words = Counter()
        for w, c in words.items():
            out = []
            i = 0
            while i < len(w):
                if i + 1 < len(w) and w[i] == a and w[i + 1] == b:
                    out.append(merged)
                    i += 2
                else:
                    out.append(w[i])
                    i += 1
            new_words[tuple(out)] += c
        words = new_words
    return ranks


def write_model(path, ranks):
    with open(path, "w") as f:
        for token, rank in sorted(ranks.items(), key=lambda kv: kv[1]):
            f.write(f"{base64.b64encode(token).decode()} {rank}\n")


TEXTS = [
    "Hello world",
    "  leading and   multiple   spaces  ",
    "trailing space ",
    "Kimi是Moonshot的模型，你好世界！",
    "it's 12345 6789.00 and I'M FINE, you'RE not; we'll see",
    "tabs\tand\nnewlines\r\n\n end",
    "naïve café résumé ÉCOLE Straße",
    "emoji 😀👍🏽 done",
    "<|im_user|>user<|im_middle|>Hi<|im_end|><|im_assistant|>assistant<|im_middle|>",
    "qzxw and qzxwq and qzx",
    "MixedCaseWORDSlikeHTTPServer camelCase",
    "  \n  ",
    "\n\n",
    "数字123和４５６以及٣٤٥",
    "אבג ABC אבגA Aאבג",
    "‘quotes’ «guillemets» — dash… (parens) [brackets]",
    "a b　c d",
    "日本語テキストとカタカナ、ひらがな。",
    "한국어 텍스트 입니다",
    "x = 1; y += 2 /* comment */ // trailing",
    "hello<|reserved_token_5|>world",
    "unterminated <|im_end",
    "",
]


def main():
    ranks = train(KIMI_PAT, 400)
    ranks[b"qzxw"] = len(ranks)  # unreachable by merges: exercises the whole-piece fast path
    n_base = len(ranks)

    # --- Kimi K2 layout ---------------------------------------------------
    out = "tests/fixtures/tiktoken_kimi"
    os.makedirs(out, exist_ok=True)
    write_model(f"{out}/tiktoken.model", ranks)
    named = {0: "[BOS]", 1: "[EOS]", 2: "<|im_end|>", 3: "<|im_user|>", 4: "<|im_assistant|>", 6: "<|start_header_id|>",
             7: "<|end_header_id|>", 9: "[EOT]", 10: "<|im_system|>", 11: "<|tool_calls_section_begin|>", 17: "<|im_middle|>",
             254: "[UNK]", 255: "[PAD]"}
    special = {named.get(i, f"<|reserved_token_{n_base + i}|>"): n_base + i for i in range(256)}
    decoder = {str(n_base + i): {"content": name, "lstrip": False, "normalized": False, "rstrip": False, "single_word": False,
                                 "special": name != "<|tool_calls_section_begin|>"} for i, name in named.items()}
    cfg = {"added_tokens_decoder": decoder, "bos_token": "[BOS]", "eos_token": "[EOS]", "pad_token": "[PAD]", "unk_token": "[UNK]",
           "clean_up_tokenization_spaces": False, "tokenizer_class": "TikTokenTokenizer",
           "auto_map": {"AutoTokenizer": ["tokenization_kimi.TikTokenTokenizer", None]},
           "chat_template": "{%- for message in messages -%}{%- if message['role'] == 'system' -%}{{ '<|im_system|>system<|im_middle|>' }}"
                            "{%- elif message['role'] == 'user' -%}{{ '<|im_user|>user<|im_middle|>' }}{%- elif message['role'] == 'assistant' -%}"
                            "{{ '<|im_assistant|>assistant<|im_middle|>' }}{%- endif -%}{{ message['content'] + '<|im_end|>' }}{%- endfor -%}"
                            "{%- if add_generation_prompt -%}{{ '<|im_assistant|>assistant<|im_middle|>' }}{%- endif -%}"}
    json.dump(cfg, open(f"{out}/tokenizer_config.json", "w"), indent=1, ensure_ascii=False)
    enc = tiktoken.Encoding(name="kimi-fixture", pat_str=KIMI_PAT, mergeable_ranks=ranks, special_tokens=special)
    cases = [{"text": t, "ids": enc.encode(t, allowed_special="all")} for t in TEXTS]
    json.dump({"n_base": n_base, "n_vocab": enc.n_vocab, "cases": cases}, open(f"{out}/reference.json", "w"), indent=1, ensure_ascii=False)

    # --- Llama 3 layout ---------------------------------------------------
    out = "tests/fixtures/tiktoken_llama3"
    os.makedirs(out, exist_ok=True)
    write_model(f"{out}/tokenizer.model", ranks)
    names = ["<|begin_of_text|>", "<|end_of_text|>", "<|reserved_special_token_0|>", "<|reserved_special_token_1|>",
             "<|reserved_special_token_2|>", "<|reserved_special_token_3|>", "<|start_header_id|>", "<|end_header_id|>",
             "<|reserved_special_token_4|>", "<|eot_id|>"] + [f"<|reserved_special_token_{i}|>" for i in range(5, 256 - 5)]
    special = {name: n_base + i for i, name in enumerate(names)}
    cfg = {"bos_token": "<|begin_of_text|>", "eos_token": "<|end_of_text|>", "tokenizer_class": "PreTrainedTokenizerFast",
           "chat_template": "{{ bos_token }}{% for message in messages %}{{ '<|start_header_id|>' + message['role'] + '<|end_header_id|>\\n\\n' + message['content'] | trim + '<|eot_id|>' }}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\\n\\n' }}{% endif %}"}
    json.dump(cfg, open(f"{out}/tokenizer_config.json", "w"), indent=1, ensure_ascii=False)
    enc = tiktoken.Encoding(name="llama3-fixture", pat_str=LLAMA3_PAT, mergeable_ranks=ranks, special_tokens=special)
    texts = [t.replace("<|im_user|>", "<|begin_of_text|>").replace("<|im_middle|>", "<|start_header_id|>").replace("<|im_end|>", "<|eot_id|>")
             .replace("<|im_assistant|>", "<|end_header_id|>").replace("<|reserved_token_5|>", "<|reserved_special_token_5|>") for t in TEXTS]
    cases = [{"text": t, "ids": enc.encode(t, allowed_special="all")} for t in texts]
    json.dump({"n_base": n_base, "n_vocab": enc.n_vocab, "cases": cases}, open(f"{out}/reference.json", "w"), indent=1, ensure_ascii=False)
    print(f"base vocabulary: {n_base} tokens; {len(TEXTS)} reference strings per fixture")


if __name__ == "__main__":
    main()
