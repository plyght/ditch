#!/usr/bin/env python3
"""Generates tiny synthetic Hugging Face-format models plus a NumPy reference
forward pass, used to validate ditch's inference against known-good numbers.

Usage: make_fixture.py <family> [<out_dir>] [--gguf]
    family: llama | qwen2 | qwen3 | gemma3 | qwen3_moe | qwen3_moe_fused | qwen3_moe_fused_t
            or any family of the registry-driven generator (`SPECS` below:
            phi3, phi, gpt_neox, gpt2, falcon, ... , gpt_oss, deepseek_v3)
            | qwen3_moe_big
    --gguf: additionally write <out_dir>_gguf/model.gguf, the same model as a
            llama.cpp GGUF file (f16 attention and embedding matrices, Q8_0
            feed-forward matrices, f32 norms, the llama q/k permutation, the
            ggml vocabulary) with a reference.json computed on the rounded weights.

The qwen3_moe variants share identical weights: `qwen3_moe` stores one tensor
per expert, `qwen3_moe_fused` the fused [E, 2I, H] / [E, H, I] layout and
`qwen3_moe_fused_t` the transposed fused [E, H, 2I] / [E, I, H] layout.
`qwen3_moe_big` is a larger routed model (4 MoE layers, 16 experts, top-2,
hidden size 64) used to exercise the expert cache with real evictions.

Only NumPy is required. Weights are random but deterministic.
"""
import json
import os
import struct
import sys
import tempfile

import numpy as np

ARGS = [a for a in sys.argv[1:] if not a.startswith("--")]
WRITE_GGUF = "--gguf" in sys.argv
FAMILY = ARGS[0] if len(ARGS) > 0 else "llama"
# The GGUF variant needs block-aligned feed-forward matrices (a multiple of 32
# columns), so it is a slightly different model; its Hugging Face twin goes to
# a scratch directory unless an output directory is given explicitly.
OUT = ARGS[1] if len(ARGS) > 1 else (tempfile.mkdtemp(prefix="ditch-fixture-") if WRITE_GGUF else f"tests/fixtures/{FAMILY}")
GOUT = f"tests/fixtures/{FAMILY}_gguf" if len(ARGS) < 2 else ARGS[1].rstrip("/") + "_gguf"
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(1234)

# ---------------------------------------------------------------------------
# Registry-driven fixtures. Every family below is described by a spec (its
# Hugging Face config.json, tensor names and layouts) from which the weights
# are generated in the exact on-disk layout of the family, and a NumPy
# reference forward pass written from the Hugging Face modeling code computes
# the expected logits and residual stream. The legacy families further down
# (llama, qwen2, qwen3, gemma3, qwen3_moe*) keep their original generator so
# the committed fixtures stay byte-identical.
# ---------------------------------------------------------------------------

import re


def bf16_round(a):
    a = np.ascontiguousarray(a, dtype=np.float32)
    u = a.view(np.uint32)
    rounded = u + 0x7FFF + ((u >> 16) & 1)
    return ((rounded >> 16).astype(np.uint16).astype(np.uint32) << 16).view(np.float32)


def bf16_bits(a):
    a = np.ascontiguousarray(a, dtype=np.float32)
    u = a.view(np.uint32)
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)


B2U_MAP = None


def b2u():
    global B2U_MAP
    if B2U_MAP is None:
        bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
        cs = bs[:]
        n = 0
        for b in range(256):
            if b not in bs:
                bs.append(b)
                cs.append(256 + n)
                n += 1
        B2U_MAP = dict(zip(bs, [chr(c) for c in cs]))
    return B2U_MAP


REGEX_GPT2 = r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"
REGEX_QWEN2 = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
REGEX_LLAMA3 = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
REGEX_O200K = (r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|"
               r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|"
               r"\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+")
REGEX_DS3 = r"[!\"#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~][A-Za-z]+|[^\r\n\p{L}\p{P}\p{S}]?[\p{L}\p{M}]+| ?[\p{P}\p{S}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
REGEX_DS2_LETTERS = r"\s?[A-Za-zµÀ-ÖØ-öø-ƿǄ-ʓʕ-ʯͰ-ͳͶͷͻ-ͽΆΈ-ΊΌΎ-ΡΣ-ϵϷ-ҁҊ-ԯԱ-ՖႠ-ჅᎠ-Ᏽᏸ-ᏽᲐ-ᲺᲽ-Ჿᴀ-ᴫᵫ-ᵷᵹ-ᶚḀ-ἕἘ-Ἕἠ-ὅὈ-Ὅὐ-ὗὙὛὝὟ-ώᾀ-ᾴᾶ-ᾼιῂ-ῄῆ-ῌῐ-ΐῖ-Ίῠ-Ῥῲ-ῴῶ-ῼℂℇℊ-ℓℕℙ-ℝℤΩℨK-ℭℯ-ℴℹℼ-ℿⅅ-ⅉⅎↃↄⰀ-ⱻⱾ-ⳤⳫ-ⳮⳲⳳꙀ-ꙭꚀ-ꚛꜢ-ꝯꝱ-ꞇꞋ-ꞎꭰ-ꮿﬀ-ﬆﬓ-ﬗＡ-Ｚａ-ｚ𐐀-𐑏𐒰-𐓓𐓘-𐓻𐲀-𐲲𐳀-𐳲𑢠-𑣟𞤀-𞥃]+"
REGEX_DS2_PUNCT = r"\s?[!-/:-~！-／：-～‘-‟　-。]+"

# Pre-tokenizer configurations (the `pre_tokenizer` of tokenizer.json) and the
# ASCII-only Python equivalent used to tokenise the reference prompts.
PRETOK = {
    "gpt2": ([{"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": True, "use_regex": True}], ["gpt2"]),
    "qwen2": ([{"type": "Split", "pattern": {"Regex": REGEX_QWEN2}, "behavior": "Isolated", "invert": False},
               {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": False, "use_regex": False}], ["qwen2"]),
    "llama3": ([{"type": "Split", "pattern": {"Regex": REGEX_LLAMA3}, "behavior": "Isolated", "invert": False},
                {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": False, "use_regex": False}], ["llama3"]),
    "o200k": ([{"type": "Split", "pattern": {"Regex": REGEX_O200K}, "behavior": "Isolated", "invert": False},
               {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": True, "use_regex": False}], ["llama3"]),
    "deepseek3": ([{"type": "Split", "pattern": {"Regex": r"\p{N}{1,3}"}, "behavior": "Isolated", "invert": False},
                   {"type": "Split", "pattern": {"Regex": "[一-龥぀-ゟ゠-ヿ]+"}, "behavior": "Isolated", "invert": False},
                   {"type": "Split", "pattern": {"Regex": REGEX_DS3}, "behavior": "Isolated", "invert": False},
                   {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": True, "use_regex": False}], ["digits3", "gpt2"]),
    "deepseek2": ([{"type": "Split", "pattern": {"Regex": "[\r\n]"}, "behavior": "Isolated", "invert": False},
                   {"type": "Split", "pattern": {"Regex": REGEX_DS2_LETTERS}, "behavior": "Isolated", "invert": False},
                   {"type": "Split", "pattern": {"Regex": REGEX_DS2_PUNCT}, "behavior": "Isolated", "invert": False},
                   {"type": "Split", "pattern": {"Regex": r"\s+$"}, "behavior": "Isolated", "invert": False},
                   {"type": "Split", "pattern": {"Regex": "[一-龥ࠀ-一가-퟿]+"}, "behavior": "Isolated", "invert": False},
                   {"type": "Digits", "individual_digits": True},
                   {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": True, "use_regex": False}], ["ds2"]),
    "falcon": ([{"type": "Punctuation", "behavior": "Contiguous"},
                {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": True, "use_regex": True}], ["punct", "gpt2"]),
    "starcoder": ([{"type": "Digits", "individual_digits": True},
                   {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": True, "use_regex": True}], ["digits1", "gpt2"]),
}

PY_RE = {
    "gpt2": r"'s|'t|'re|'ve|'m|'ll|'d| ?[A-Za-z]+| ?[0-9]+| ?[^\sA-Za-z0-9]+|\s+(?!\S)|\s+",
    "qwen2": r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\nA-Za-z0-9]?[A-Za-z]+|[0-9]| ?[^\sA-Za-z0-9]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
    "llama3": r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\nA-Za-z0-9]?[A-Za-z]+|[0-9]{1,3}| ?[^\sA-Za-z0-9]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
}


def py_step(kind, piece):
    """One ASCII pre-tokenisation step: returns the pieces `piece` splits into."""
    if kind in PY_RE:
        return re.findall(PY_RE[kind], piece)
    if kind == "digits3":
        return [m for m in re.findall(r"[0-9]{1,3}|[^0-9]+", piece)]
    if kind == "digits1":
        return [m for m in re.findall(r"[0-9]|[^0-9]+", piece)]
    if kind == "punct":
        return [m for m in re.findall(r"[!-/:-@\[-`{-~]+|[^!-/:-@\[-`{-~]+", piece)]
    if kind == "ds2":
        out = []
        for m in re.findall(r"\s?[A-Za-z]+|[^A-Za-z]+", piece):
            if re.fullmatch(r"\s?[A-Za-z]+", m):
                out.append(m)
            else:
                out.extend(re.findall(r"[0-9]|[^0-9]+", m))
        return out
    raise KeyError(kind)


def make_tokenizer(kind, out_dir):
    """Writes tokenizer.json / tokenizer_config.json; returns (vocab, encode, bos_name, eos_name)."""
    merge_pairs_bl = [("Ġ", "t"), ("h", "e"), ("Ġt", "he"), ("i", "n"), ("a", "n"), ("o", "r"), ("Ġ", "a"), ("Ġa", "n"), ("e", "r"), ("Ġ", "y"), ("Ġy", "o"), ("Ġyo", "u"), ("l", "l"), ("h", "el"), ("hel", "lo")]
    if kind == "spm":
        specials = ["<unk>", "<s>", "</s>", "<|user|>", "<|assistant|>", "<|end|>"]
        vocab = {}
        for s in specials:
            vocab[s] = len(vocab)
        for b in range(256):
            vocab[f"<0x{b:02X}>"] = len(vocab)
        for ch in ["▁"] + [chr(c) for c in range(32, 127)]:
            if ch not in vocab:
                vocab[ch] = len(vocab)
        merge_pairs = [("▁", "t"), ("h", "e"), ("▁t", "he"), ("i", "n"), ("a", "n"), ("o", "r"), ("▁", "a"), ("▁a", "n"), ("e", "r"), ("▁", "y"), ("▁y", "o"), ("▁yo", "u"), ("l", "l"), ("h", "el"), ("▁", "hel"), ("▁hel", "lo")]
        merges = []
        for a, b in merge_pairs:
            merges.append(f"{a} {b}")
            vocab[a + b] = len(vocab)
        added = [{"id": vocab[s], "content": s, "single_word": False, "lstrip": False, "rstrip": False, "normalized": False, "special": True} for s in specials]
        tok = {"version": "1.0", "added_tokens": added,
               "normalizer": {"type": "Sequence", "normalizers": [{"type": "Prepend", "prepend": "▁"}, {"type": "Replace", "pattern": {"String": " "}, "content": "▁"}]},
               "pre_tokenizer": None,
               "post_processor": {"type": "TemplateProcessing", "single": [{"SpecialToken": {"id": "<s>", "type_id": 0}}, {"Sequence": {"id": "A", "type_id": 0}}],
                                  "pair": [], "special_tokens": {"<s>": {"id": "<s>", "ids": [vocab["<s>"]], "tokens": ["<s>"]}}},
               "decoder": {"type": "Sequence", "decoders": [{"type": "Replace", "pattern": {"String": "▁"}, "content": " "}, {"type": "ByteFallback"}, {"type": "Fuse"}, {"type": "Strip", "content": " ", "start": 1, "stop": 0}]},
               "model": {"type": "BPE", "dropout": None, "unk_token": "<unk>", "continuing_subword_prefix": None, "end_of_word_suffix": None,
                         "fuse_unk": True, "byte_fallback": True, "ignore_merges": False, "vocab": vocab, "merges": merges}}
        ranks = {m: i for i, m in enumerate(merges)}

        def encode(text):
            syms = list("▁" + text.replace(" ", "▁"))
            bpe_merge(syms, ranks)
            ids = []
            for s in syms:
                if s in vocab:
                    ids.append(vocab[s])
                else:
                    ids.extend(vocab[f"<0x{b:02X}>"] for b in s.encode())
            return ids
        bos, eos = "<s>", "</s>"
    else:
        specials = ["<|endoftext|>", "<|user|>", "<|assistant|>", "<|end|>"]
        vocab = {}
        for b in range(256):
            vocab[b2u()[b]] = len(vocab)
        merges = []
        for a, b in merge_pairs_bl:
            merges.append(f"{a} {b}")
            vocab[a + b] = len(vocab)
        added = []
        for s in specials:
            added.append({"id": len(vocab), "content": s, "single_word": False, "lstrip": False, "rstrip": False, "normalized": False, "special": True})
            vocab[s] = len(vocab)
        pre, py_kinds = PRETOK[kind]
        tok = {"version": "1.0", "added_tokens": added, "normalizer": None,
               "pre_tokenizer": {"type": "Sequence", "pretokenizers": pre},
               "post_processor": None,
               "decoder": {"type": "ByteLevel", "add_prefix_space": True, "trim_offsets": True, "use_regex": True},
               "model": {"type": "BPE", "dropout": None, "unk_token": None, "continuing_subword_prefix": None, "end_of_word_suffix": None,
                         "fuse_unk": False, "byte_fallback": False, "ignore_merges": False, "vocab": vocab, "merges": merges}}
        ranks = {m: i for i, m in enumerate(merges)}

        def encode(text):
            pieces = [text]
            for k in py_kinds:
                nxt = []
                for p in pieces:
                    nxt.extend(py_step(k, p))
                pieces = nxt
            ids = []
            for piece in pieces:
                syms = [b2u()[b] for b in piece.encode()]
                bpe_merge(syms, ranks)
                ids.extend(vocab[s] for s in syms)
            return ids
        bos, eos = None, "<|endoftext|>"
    json.dump(tok, open(f"{out_dir}/tokenizer.json", "w"), ensure_ascii=False)
    cfg = {"eos_token": eos, "add_bos_token": bos is not None, "tokenizer_class": "PreTrainedTokenizerFast",
           "chat_template": "{% for message in messages %}{{'<|' + message['role'] + '|>\n' + message['content'] + '<|end|>\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|assistant|>\n' }}{% endif %}"}
    if bos:
        cfg["bos_token"] = bos
    json.dump(cfg, open(f"{out_dir}/tokenizer_config.json", "w"))
    return vocab, encode, bos, eos


def bpe_merge(syms, ranks):
    while len(syms) > 1:
        best, bi = None, None
        for i in range(len(syms) - 1):
            r = ranks.get(f"{syms[i]} {syms[i + 1]}")
            if r is not None and (best is None or r < best):
                best, bi = r, i
        if best is None:
            break
        syms[bi:bi + 2] = [syms[bi] + syms[bi + 1]]


# --- family specs ---------------------------------------------------------

def base(**kw):
    """Defaults shared by the specs (llama layout)."""
    d = dict(
        H=32, I=32, L=2, NH=4, NKV=2, HD=8, VD=None,
        tok="gpt2", prefix="model.", layer="layers.{i}.",
        embed="embed_tokens.weight", pos_embed=None, embed_norm=None, final_norm="norm.weight", lm_head="lm_head.weight",
        norm="rms", eps=1e-6, ln_bias=True,
        in_norm="input_layernorm.weight", post_attn_norm=None, pre_ff_norm="post_attention_layernorm.weight", post_ff_norm=None, mlp_norm=None,
        q="self_attn.q_proj.weight", k="self_attn.k_proj.weight", v="self_attn.v_proj.weight", qkv=None, qkv_layout="concat", o="self_attn.o_proj.weight",
        attn_bias=False, o_bias=None, conv1d=False,
        mlp="gated", gate="mlp.gate_proj.weight", up="mlp.up_proj.weight", gate_up=None, down="mlp.down_proj.weight", mlp_bias=False, act="silu",
        parallel=False, pos="rope", rope_style="neox", rotary_dim=None, theta=10000.0, scaling=None, rope_layers=None, attn_scale=None,
        qk_norm=None, q_norm="self_attn.q_norm.weight", k_norm="self_attn.k_norm.weight", clip=None,
        residual_mult=1.0, logit_scale=1.0, embed_scale=1.0, lm_bias=False, sinks=None, temp=None, pos_offset=0,
        sliding=None, sliding_layers=None, mla=None, moe=None, linear=None, linear_layers=None, full_interval=0,
        gated_q=False, gate_swish=False,
        # Gemma 3n / 4 and LFM2 features: per-layer head size / KV heads, KV
        # sharing, keys reused as values, weightless value norm, a local rope
        # table (theta, rotary dim) and a global (rotary dim, freq dim) pair,
        # per-layer inputs, AltUp / Laurel, gate sparsity, conv layers,
        # per-layer output scalars, per-layer FFN widths, final softcapping.
        layer_hd=None, layer_nkv=None, kv_shared=0, k_eq_v=False, v_norm=False, local_rope=None, global_rotary=None,
        ple_dim=0, altup=None, sparsity=None, conv_layers=None, conv_K=3, layer_scale=False, layer_inter=None, final_softcap=None,
        config={}, extra_config={},
    )
    d.update(kw)
    if d["VD"] is None:
        d["VD"] = d["HD"]
    if d["rotary_dim"] is None:
        d["rotary_dim"] = d["HD"]
    return d


def llama_config(s, model_type, **extra):
    c = {"model_type": model_type, "hidden_size": s["H"], "intermediate_size": s["I"], "num_hidden_layers": s["L"],
         "num_attention_heads": s["NH"], "num_key_value_heads": s["NKV"], "rms_norm_eps": s["eps"], "rope_theta": s["theta"],
         "max_position_embeddings": 128, "hidden_act": s["act"], "tie_word_embeddings": s["lm_head"] is None, "torch_dtype": "bfloat16"}
    c.update(extra)
    return c


SPECS = {}


def spec(name, **kw):
    SPECS[name] = base(**kw)


spec("phi3", tok="spm", lm_head=None, qkv="self_attn.qkv_proj.weight", mlp="gated_fused", gate_up="mlp.gate_up_proj.weight",
     scaling={"type": "longrope", "short_factor": [1.0, 1.5, 2.0, 3.0], "long_factor": [1.0, 4.0, 8.0, 16.0]},
     extra_config={"original_max_position_embeddings": 64, "max_position_embeddings": 128},
     config={"model_type": "phi3", "hidden_size": 32, "intermediate_size": 32, "num_hidden_layers": 2, "num_attention_heads": 4, "num_key_value_heads": 2,
             "rms_norm_eps": 1e-6, "rope_theta": 10000.0, "hidden_act": "silu", "tie_word_embeddings": True})
spec("phi", NKV=4, norm="ln", eps=1e-5, parallel=True, pre_ff_norm=None, final_norm="final_layernorm.weight", o="self_attn.dense.weight",
     attn_bias=True, mlp="dense", up="mlp.fc1.weight", down="mlp.fc2.weight", mlp_bias=True, act="gelu_new", rotary_dim=4, lm_bias=True, lm_head="lm_head.weight",
     config={"model_type": "phi", "hidden_size": 32, "intermediate_size": 32, "num_hidden_layers": 2, "num_attention_heads": 4, "num_key_value_heads": 4,
             "layer_norm_eps": 1e-5, "partial_rotary_factor": 0.5, "rope_theta": 10000.0, "hidden_act": "gelu_new", "qk_layernorm": False,
             "max_position_embeddings": 128, "tie_word_embeddings": False})
spec("gpt_neox", NKV=4, prefix="gpt_neox.", embed="embed_in.weight", final_norm="final_layer_norm.weight", lm_head="embed_out.weight", norm="ln", eps=1e-5,
     parallel=True, qkv="attention.query_key_value.weight", qkv_layout="heads", o="attention.dense.weight", attn_bias=True,
     mlp="dense", up="mlp.dense_h_to_4h.weight", down="mlp.dense_4h_to_h.weight", mlp_bias=True, act="gelu", rotary_dim=2, theta=20000.0,
     config={"model_type": "gpt_neox", "hidden_size": 32, "intermediate_size": 32, "num_hidden_layers": 2, "num_attention_heads": 4,
             "rotary_pct": 0.25, "rotary_emb_base": 20000, "layer_norm_eps": 1e-5, "use_parallel_residual": True, "hidden_act": "gelu",
             "max_position_embeddings": 128, "tie_word_embeddings": False, "vocab_size": 0})
spec("gpt2", NKV=4, I=64, prefix="transformer.", layer="h.{i}.", embed="wte.weight", pos_embed="wpe.weight", final_norm="ln_f.weight", lm_head=None,
     norm="ln", eps=1e-5, in_norm="ln_1.weight", pre_ff_norm="ln_2.weight", qkv="attn.c_attn.weight", o="attn.c_proj.weight", attn_bias=True, conv1d=True,
     mlp="dense", up="mlp.c_fc.weight", down="mlp.c_proj.weight", mlp_bias=True, act="gelu_new", pos="learned",
     config={"model_type": "gpt2", "n_embd": 32, "n_layer": 2, "n_head": 4, "n_positions": 64, "n_inner": 64, "activation_function": "gelu_new",
             "layer_norm_epsilon": 1e-5, "vocab_size": 0, "tie_word_embeddings": True})
spec("falcon", NKV=1, I=64, prefix="transformer.", layer="h.{i}.", embed="word_embeddings.weight", final_norm="ln_f.weight", lm_head=None,
     norm="ln", eps=1e-5, parallel=True, pre_ff_norm=None, qkv="self_attention.query_key_value.weight", o="self_attention.dense.weight",
     mlp="dense", up="mlp.dense_h_to_4h.weight", down="mlp.dense_4h_to_h.weight", act="gelu", tok="falcon",
     config={"model_type": "falcon", "hidden_size": 32, "num_hidden_layers": 2, "num_attention_heads": 4, "layer_norm_epsilon": 1e-5,
             "ffn_hidden_size": 64, "multi_query": True, "parallel_attn": True, "new_decoder_architecture": False, "bias": False, "alibi": False, "rope_theta": 10000.0,
             "max_position_embeddings": 128, "tie_word_embeddings": True, "vocab_size": 0})
spec("stablelm", norm="ln", eps=1e-5, attn_bias=True, o_bias=False, rotary_dim=2, lm_head=None,
     config={"model_type": "stablelm", "hidden_size": 32, "intermediate_size": 32, "num_hidden_layers": 2, "num_attention_heads": 4, "num_key_value_heads": 2,
             "layer_norm_eps": 1e-5, "partial_rotary_factor": 0.25, "rope_theta": 10000.0, "use_qkv_bias": True, "use_parallel_residual": False,
             "qk_layernorm": False, "hidden_act": "silu", "max_position_embeddings": 128, "tie_word_embeddings": True})
spec("internlm2", tok="spm", embed="tok_embeddings.weight", lm_head="output.weight", in_norm="attention_norm.weight", pre_ff_norm="ffn_norm.weight",
     qkv="attention.wqkv.weight", qkv_layout="grouped", o="attention.wo.weight", gate="feed_forward.w1.weight", up="feed_forward.w3.weight", down="feed_forward.w2.weight",
     config={"model_type": "internlm2", "hidden_size": 32, "intermediate_size": 32, "num_hidden_layers": 2, "num_attention_heads": 4, "num_key_value_heads": 2,
             "rms_norm_eps": 1e-6, "rope_theta": 10000.0, "rope_scaling": {"type": "dynamic", "factor": 2.0}, "hidden_act": "silu", "bias": False,
             "max_position_embeddings": 128, "tie_word_embeddings": False})
spec("olmo2", tok="llama3", in_norm=None, post_attn_norm="post_attention_layernorm.weight", pre_ff_norm=None, post_ff_norm="post_feedforward_layernorm.weight",
     qk_norm="full", lm_head=None,
     config={"model_type": "olmo2", "hidden_size": 32, "intermediate_size": 32, "num_hidden_layers": 2, "num_attention_heads": 4, "num_key_value_heads": 2,
             "rms_norm_eps": 1e-6, "rope_theta": 10000.0, "hidden_act": "silu", "max_position_embeddings": 128, "tie_word_embeddings": True})
spec("olmo", norm="none", eps=1e-5, in_norm="", pre_ff_norm="", final_norm="", clip=0.6, lm_head=None,
     config={"model_type": "olmo", "hidden_size": 32, "intermediate_size": 32, "num_hidden_layers": 2, "num_attention_heads": 4, "num_key_value_heads": 2,
             "clip_qkv": 0.6, "rope_theta": 10000.0, "hidden_act": "silu", "max_position_embeddings": 128, "tie_word_embeddings": True})
spec("cohere", norm="ln", eps=1e-5, ln_bias=False, parallel=True, pre_ff_norm=None, qk_norm="heads", logit_scale=0.5, lm_head=None,
     config={"model_type": "cohere", "hidden_size": 32, "intermediate_size": 32, "num_hidden_layers": 2, "num_attention_heads": 4, "num_key_value_heads": 2,
             "layer_norm_eps": 1e-5, "logit_scale": 0.5, "use_qk_norm": True, "rope_theta": 10000.0, "hidden_act": "silu",
             "max_position_embeddings": 128, "tie_word_embeddings": True})
spec("glm4", tok="llama3", post_attn_norm="post_self_attn_layernorm.weight", pre_ff_norm="post_attention_layernorm.weight", post_ff_norm="post_mlp_layernorm.weight",
     attn_bias=True, o_bias=False, mlp="gated_fused", gate_up="mlp.gate_up_proj.weight", rope_style="gptj", rotary_dim=4, lm_head=None,
     config={"model_type": "glm4", "hidden_size": 32, "intermediate_size": 32, "num_hidden_layers": 2, "num_attention_heads": 4, "num_key_value_heads": 2,
             "head_dim": 8, "rms_norm_eps": 1e-6, "rope_theta": 10000.0, "partial_rotary_factor": 0.5, "attention_bias": True, "hidden_act": "silu",
             "max_position_embeddings": 128, "tie_word_embeddings": True})
spec("chatglm", tok="spm", prefix="transformer.", layer="encoder.layers.{i}.", embed="embedding.word_embeddings.weight", final_norm="encoder.final_layernorm.weight",
     lm_head="transformer.output_layer.weight", qkv="self_attention.query_key_value.weight", attn_bias=True, o_bias=False, o="self_attention.dense.weight",
     mlp="gated_fused", gate_up="mlp.dense_h_to_4h.weight", down="mlp.dense_4h_to_h.weight", rope_style="gptj", rotary_dim=4, theta=20000.0,
     config={"model_type": "chatglm", "hidden_size": 32, "ffn_hidden_size": 32, "num_layers": 2, "num_attention_heads": 4, "multi_query_attention": True,
             "multi_query_group_num": 2, "kv_channels": 8, "layernorm_epsilon": 1e-6, "rmsnorm": True, "add_qkv_bias": True, "add_bias_linear": False,
             "rope_ratio": 2.0, "seq_length": 128, "apply_residual_connection_post_layernorm": False, "post_layer_norm": True, "tie_word_embeddings": False,
             "padded_vocab_size": 0})
spec("granite", tok="starcoder", embed_scale=2.0, attn_scale=0.25, residual_mult=0.5, logit_scale=0.25, lm_head=None,
     config={"model_type": "granite", "hidden_size": 32, "intermediate_size": 32, "num_hidden_layers": 2, "num_attention_heads": 4, "num_key_value_heads": 2,
             "rms_norm_eps": 1e-6, "rope_theta": 10000.0, "embedding_multiplier": 2.0, "attention_multiplier": 0.25, "residual_multiplier": 0.5,
             "logits_scaling": 4.0, "hidden_act": "silu", "max_position_embeddings": 128, "tie_word_embeddings": True})
spec("deepseek_v2", tok="deepseek2", NKV=4, HD=12, VD=8, L=3, rope_style="gptj", rotary_dim=4, lm_head=None,
     mla={"q_lora_rank": None, "kv_lora_rank": 16, "nope": 8, "rope": 4, "v": 8},
     scaling={"type": "yarn", "factor": 4.0, "beta_fast": 32, "beta_slow": 1, "mscale": 0.707, "mscale_all_dim": 0.707, "original_max_position_embeddings": 32},
     moe={"E": 4, "K": 2, "MI": 12, "shared": 1, "scoring": "softmax", "group_limited": True, "n_group": 2, "topk_group": 1, "rsf": 1.5, "norm": False,
          "layers": [1, 2], "corr_bias": False, "layout": "separate", "prefix": "mlp.", "router": "gate.weight", "shared_name": "shared_experts."},
     config={"model_type": "deepseek_v2", "hidden_size": 32, "intermediate_size": 32, "moe_intermediate_size": 12, "num_hidden_layers": 3,
             "num_attention_heads": 4, "num_key_value_heads": 4, "q_lora_rank": None, "kv_lora_rank": 16, "qk_nope_head_dim": 8, "qk_rope_head_dim": 4,
             "v_head_dim": 8, "n_routed_experts": 4, "n_shared_experts": 1, "num_experts_per_tok": 2, "first_k_dense_replace": 1, "moe_layer_freq": 1,
             "scoring_func": "softmax", "topk_method": "group_limited_greedy", "n_group": 2, "topk_group": 1, "routed_scaling_factor": 1.5,
             "norm_topk_prob": False, "rms_norm_eps": 1e-6, "rope_theta": 10000.0, "max_position_embeddings": 128, "hidden_act": "silu",
             "tie_word_embeddings": True})
spec("deepseek_v3", tok="deepseek3", NKV=4, HD=12, VD=8, L=3, rope_style="gptj", rotary_dim=4, lm_head="lm_head.weight",
     mla={"q_lora_rank": 12, "kv_lora_rank": 16, "nope": 8, "rope": 4, "v": 8},
     scaling={"type": "yarn", "factor": 40.0, "beta_fast": 32, "beta_slow": 1, "mscale": 1.0, "mscale_all_dim": 1.0, "original_max_position_embeddings": 32},
     moe={"E": 4, "K": 2, "MI": 12, "shared": 1, "scoring": "sigmoid", "group_limited": True, "n_group": 2, "topk_group": 1, "rsf": 2.5, "norm": True,
          "layers": [1, 2], "corr_bias": True, "layout": "separate", "prefix": "mlp.", "router": "gate.weight", "shared_name": "shared_experts."},
     config={"model_type": "deepseek_v3", "hidden_size": 32, "intermediate_size": 32, "moe_intermediate_size": 12, "num_hidden_layers": 3,
             "num_attention_heads": 4, "num_key_value_heads": 4, "q_lora_rank": 12, "kv_lora_rank": 16, "qk_nope_head_dim": 8, "qk_rope_head_dim": 4,
             "v_head_dim": 8, "n_routed_experts": 4, "n_shared_experts": 1, "num_experts_per_tok": 2, "first_k_dense_replace": 1, "moe_layer_freq": 1,
             "scoring_func": "sigmoid", "topk_method": "noaux_tc", "n_group": 2, "topk_group": 1, "routed_scaling_factor": 2.5,
             "norm_topk_prob": True, "rms_norm_eps": 1e-6, "rope_theta": 10000.0, "max_position_embeddings": 128, "hidden_act": "silu",
             "tie_word_embeddings": False, "rope_interleave": True})
spec("llama4", tok="llama3", L=3, prefix="language_model.model.", lm_head="language_model.lm_head.weight", rope_style="gptj", rope_layers=[1, 0, 1],
     qk_norm="l2", temp={"floor_scale": 2.0, "attn_scale": 0.1}, gate="feed_forward.gate_proj.weight", up="feed_forward.up_proj.weight", down="feed_forward.down_proj.weight",
     moe={"E": 4, "K": 1, "MI": 12, "shared": 1, "scoring": "sigmoid", "group_limited": False, "rsf": 1.0, "norm": False, "scale_input": True,
          "layers": [1], "corr_bias": False, "layout": "fused_t", "prefix": "feed_forward.", "router": "router.weight", "shared_name": "shared_expert."},
     config={"model_type": "llama4", "architectures": ["Llama4ForConditionalGeneration"],
             "text_config": {"model_type": "llama4_text", "hidden_size": 32, "intermediate_size": 12, "intermediate_size_mlp": 32, "num_hidden_layers": 3,
                             "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 8, "num_local_experts": 4, "num_experts_per_tok": 1,
                             "interleave_moe_layer_step": 2, "moe_layers": [1], "no_rope_layers": [1, 0, 1], "attn_temperature_tuning": True,
                             "floor_scale": 2.0, "attn_scale": 0.1, "use_qk_norm": True, "rms_norm_eps": 1e-6, "rope_theta": 10000.0,
                             "hidden_act": "silu", "max_position_embeddings": 128, "tie_word_embeddings": False},
             "vision_config": {"model_type": "llama4_vision_model"}})
spec("gpt_oss", tok="o200k", L=3, attn_bias=True, o_bias=True, sinks="self_attn.sinks", sliding=4, sliding_layers=[1, 0, 1], lm_head=None,
     scaling={"rope_type": "yarn", "factor": 8.0, "beta_fast": 32.0, "beta_slow": 1.0, "original_max_position_embeddings": 32, "truncate": False},
     moe={"E": 4, "K": 2, "MI": 12, "shared": 0, "scoring": "softmax", "group_limited": False, "rsf": 1.0, "norm": True, "router_bias": True,
          "layers": [0, 1, 2], "corr_bias": False, "layout": "fused_t_interleaved", "prefix": "mlp.", "router": "router.weight", "bias": True,
          "swiglu": (1.702, 7.0)},
     config={"model_type": "gpt_oss", "hidden_size": 32, "intermediate_size": 12, "num_hidden_layers": 3, "num_attention_heads": 4, "num_key_value_heads": 2,
             "head_dim": 8, "num_local_experts": 4, "num_experts_per_tok": 2, "sliding_window": 4, "layer_types": ["sliding_attention", "full_attention", "sliding_attention"],
             "rope_theta": 10000.0, "rope_scaling": {"rope_type": "yarn", "factor": 8.0, "beta_fast": 32.0, "beta_slow": 1.0, "original_max_position_embeddings": 32, "truncate": False},
             "max_position_embeddings": 128, "rms_norm_eps": 1e-5, "swiglu_limit": 7.0, "attention_bias": True, "hidden_act": "silu", "tie_word_embeddings": True})
spec("minicpm", tok="spm", embed_scale=12.0, residual_mult=1.4 / np.sqrt(2), logit_scale=0.5, lm_head=None,
     config={"model_type": "minicpm", "hidden_size": 32, "intermediate_size": 32, "num_hidden_layers": 2, "num_attention_heads": 4, "num_key_value_heads": 2,
             "scale_emb": 12, "scale_depth": 1.4, "dim_model_base": 16, "rms_norm_eps": 1e-6, "rope_theta": 10000.0, "hidden_act": "silu",
             "max_position_embeddings": 128, "tie_word_embeddings": True})
spec("exaone", prefix="transformer.", layer="h.{i}.", embed="wte.weight", final_norm="ln_f.weight", lm_head=None, in_norm="ln_1.weight", pre_ff_norm="ln_2.weight",
     q="attn.attention.q_proj.weight", k="attn.attention.k_proj.weight", v="attn.attention.v_proj.weight", o="attn.attention.out_proj.weight",
     gate="mlp.c_fc_0.weight", up="mlp.c_fc_1.weight", down="mlp.c_proj.weight",
     config={"model_type": "exaone", "hidden_size": 32, "intermediate_size": 32, "num_layers": 2, "num_attention_heads": 4, "num_key_value_heads": 2,
             "head_dim": 8, "layer_norm_epsilon": 1e-6, "rope_theta": 10000.0, "activation_function": "silu", "max_position_embeddings": 128, "tie_word_embeddings": True})
spec("exaone4", L=4, in_norm=None, post_attn_norm="post_attention_layernorm.weight", pre_ff_norm=None, post_ff_norm="post_feedforward_layernorm.weight",
     qk_norm="head", sliding=4, sliding_layers=[1, 1, 1, 0], rope_layers=[1, 1, 1, 0], lm_head=None,
     config={"model_type": "exaone4", "hidden_size": 32, "intermediate_size": 32, "num_hidden_layers": 4, "num_attention_heads": 4, "num_key_value_heads": 2,
             "head_dim": 8, "rms_norm_eps": 1e-6, "rope_theta": 10000.0, "sliding_window": 4, "sliding_window_pattern": "LLLG", "hidden_act": "silu",
             "max_position_embeddings": 128, "tie_word_embeddings": True})
spec("nemotron", tok="spm", norm="ln1p", eps=1e-5, mlp="dense", up="mlp.up_proj.weight", act="relu2", rotary_dim=4, lm_head=None,
     config={"model_type": "nemotron", "hidden_size": 32, "intermediate_size": 32, "num_hidden_layers": 2, "num_attention_heads": 4, "num_key_value_heads": 2,
             "norm_eps": 1e-5, "partial_rotary_factor": 0.5, "rope_theta": 10000.0, "hidden_act": "relu2", "attention_bias": False, "mlp_bias": False,
             "max_position_embeddings": 128, "tie_word_embeddings": True})
spec("smollm3", tok="llama3", L=3, rope_layers=[1, 1, 0], lm_head=None,
     config={"model_type": "smollm3", "hidden_size": 32, "intermediate_size": 32, "num_hidden_layers": 3, "num_attention_heads": 4, "num_key_value_heads": 2,
             "rms_norm_eps": 1e-6, "rope_theta": 10000.0, "no_rope_layers": [1, 1, 0], "no_rope_layer_interval": 3, "use_sliding_window": False,
             "hidden_act": "silu", "max_position_embeddings": 128, "tie_word_embeddings": True})
spec("bloom", NKV=4, I=128, prefix="transformer.", layer="h.{i}.", embed="word_embeddings.weight", embed_norm="word_embeddings_layernorm.weight", final_norm="ln_f.weight",
     lm_head=None, norm="ln", eps=1e-5, qkv="self_attention.query_key_value.weight", qkv_layout="heads", attn_bias=True, o="self_attention.dense.weight",
     mlp="dense", up="mlp.dense_h_to_4h.weight", down="mlp.dense_4h_to_h.weight", mlp_bias=True, act="gelu_tanh", pos="alibi",
     config={"model_type": "bloom", "hidden_size": 32, "n_layer": 2, "n_head": 4, "layer_norm_epsilon": 1e-5, "apply_residual_connection_post_layernorm": False,
             "vocab_size": 0, "tie_word_embeddings": True})
spec("opt", NKV=4, prefix="model.decoder.", pos_embed="embed_positions.weight", final_norm="final_layer_norm.weight", lm_head=None, norm="ln", eps=1e-5,
     in_norm="self_attn_layer_norm.weight", pre_ff_norm="final_layer_norm.weight", attn_bias=True, o="self_attn.out_proj.weight",
     mlp="dense", up="fc1.weight", down="fc2.weight", mlp_bias=True, act="relu", pos="learned", pos_offset=2,
     config={"model_type": "opt", "hidden_size": 32, "ffn_dim": 32, "num_hidden_layers": 2, "num_attention_heads": 4, "max_position_embeddings": 64,
             "word_embed_proj_dim": 32, "do_layer_norm_before": True, "activation_function": "relu", "enable_bias": True, "layer_norm_elementwise_affine": True,
             "vocab_size": 0, "tie_word_embeddings": True})
spec("mpt", NKV=4, I=64, prefix="transformer.", layer="blocks.{i}.", embed="wte.weight", final_norm="norm_f.weight", lm_head=None, norm="ln", eps=1e-5, ln_bias=False,
     in_norm="norm_1.weight", pre_ff_norm="norm_2.weight", qkv="attn.Wqkv.weight", o="attn.out_proj.weight", mlp="dense", up="ffn.up_proj.weight", down="ffn.down_proj.weight",
     act="gelu", pos="alibi",
     config={"model_type": "mpt", "d_model": 32, "n_heads": 4, "n_layers": 2, "expansion_ratio": 2, "max_seq_len": 128, "no_bias": True,
             "attn_config": {"alibi": True, "alibi_bias_max": 8, "clip_qkv": None, "qk_ln": False, "kv_n_heads": 4}, "layer_norm_epsilon": 1e-5,
             "vocab_size": 0, "tie_word_embeddings": True})
spec("starcoder2", tok="starcoder", norm="ln", eps=1e-5, attn_bias=True, o_bias=True, mlp="dense", up="mlp.c_fc.weight", down="mlp.c_proj.weight", mlp_bias=True,
     act="gelu_tanh", sliding=4, sliding_layers=[1, 1], lm_head=None,
     config={"model_type": "starcoder2", "hidden_size": 32, "intermediate_size": 32, "num_hidden_layers": 2, "num_attention_heads": 4, "num_key_value_heads": 2,
             "norm_epsilon": 1e-5, "norm_type": "layer_norm", "use_bias": True, "rope_theta": 10000.0, "sliding_window": 4, "hidden_act": "gelu_pytorch_tanh",
             "max_position_embeddings": 128, "tie_word_embeddings": True})
spec("gpt_bigcode", tok="starcoder", NKV=1, I=64, prefix="transformer.", layer="h.{i}.", embed="wte.weight", pos_embed="wpe.weight", final_norm="ln_f.weight", lm_head=None,
     norm="ln", eps=1e-5, in_norm="ln_1.weight", pre_ff_norm="ln_2.weight", qkv="attn.c_attn.weight", o="attn.c_proj.weight", attn_bias=True,
     mlp="dense", up="mlp.c_fc.weight", down="mlp.c_proj.weight", mlp_bias=True, act="gelu_tanh", pos="learned",
     config={"model_type": "gpt_bigcode", "n_embd": 32, "n_layer": 2, "n_head": 4, "n_positions": 64, "n_inner": 64, "multi_query": True,
             "layer_norm_epsilon": 1e-5, "activation_function": "gelu_pytorch_tanh", "vocab_size": 0, "tie_word_embeddings": True})
spec("baichuan", tok="spm", NKV=4, qkv="self_attn.W_pack.weight", lm_head=None,
     config={"model_type": "baichuan", "hidden_size": 32, "intermediate_size": 32, "num_hidden_layers": 2, "num_attention_heads": 4, "rms_norm_eps": 1e-6,
             "max_position_embeddings": 128, "hidden_act": "silu", "tie_word_embeddings": True})
spec("mistral", tok="llama3", sliding=4, sliding_layers=[1, 1], lm_head=None,
     config={"model_type": "mistral", "hidden_size": 32, "intermediate_size": 32, "num_hidden_layers": 2, "num_attention_heads": 4, "num_key_value_heads": 2,
             "head_dim": 8, "rms_norm_eps": 1e-6, "rope_theta": 10000.0, "sliding_window": 4, "hidden_act": "silu", "max_position_embeddings": 128,
             "tie_word_embeddings": True})
spec("mixtral", tok="llama3", L=2, lm_head=None,
     moe={"E": 4, "K": 2, "MI": 12, "shared": 0, "scoring": "softmax", "group_limited": False, "rsf": 1.0, "norm": True,
          "layers": [0, 1], "corr_bias": False, "layout": "separate", "prefix": "block_sparse_moe.", "router": "gate.weight",
          "expert_names": ("w1.weight", "w3.weight", "w2.weight")},
     config={"model_type": "mixtral", "hidden_size": 32, "intermediate_size": 12, "num_hidden_layers": 2, "num_attention_heads": 4, "num_key_value_heads": 2,
             "num_local_experts": 4, "num_experts_per_tok": 2, "rms_norm_eps": 1e-6, "rope_theta": 10000.0, "hidden_act": "silu",
             "max_position_embeddings": 128, "tie_word_embeddings": True})
spec("qwen2_moe", tok="qwen2", L=2, attn_bias=True, o_bias=False, lm_head=None,
     moe={"E": 4, "K": 2, "MI": 12, "shared": 1, "shared_inter": 16, "shared_gate": True, "scoring": "softmax", "group_limited": False, "rsf": 1.0, "norm": False,
          "layers": [0, 1], "corr_bias": False, "layout": "separate", "prefix": "mlp.", "router": "gate.weight", "shared_name": "shared_expert."},
     config={"model_type": "qwen2_moe", "hidden_size": 32, "intermediate_size": 32, "moe_intermediate_size": 12, "shared_expert_intermediate_size": 16,
             "num_hidden_layers": 2, "num_attention_heads": 4, "num_key_value_heads": 2, "num_experts": 4, "num_experts_per_tok": 2, "norm_topk_prob": False,
             "decoder_sparse_step": 1, "mlp_only_layers": [], "rms_norm_eps": 1e-6, "rope_theta": 10000.0, "hidden_act": "silu",
             "max_position_embeddings": 128, "tie_word_embeddings": True})
spec("qwen3_next", tok="qwen2", H=32, I=32, L=4, NH=4, NKV=2, HD=8, rotary_dim=2, qk_norm="head", gated_q=True,
     linear={"KH": 2, "KD": 4, "VH": 4, "VD": 4, "KC": 2, "fused": True}, full_interval=4,
     moe={"E": 4, "K": 2, "MI": 12, "shared": 1, "shared_inter": 16, "shared_gate": True, "scoring": "softmax", "group_limited": False, "rsf": 1.0, "norm": True,
          "layers": [0, 1, 2, 3], "corr_bias": False, "layout": "separate", "prefix": "mlp.", "router": "gate.weight", "shared_name": "shared_expert."},
     config={"model_type": "qwen3_next", "hidden_size": 32, "intermediate_size": 32, "moe_intermediate_size": 12, "shared_expert_intermediate_size": 16,
             "num_hidden_layers": 4, "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 8,
             "num_experts": 4, "num_experts_per_tok": 2, "norm_topk_prob": True, "decoder_sparse_step": 1, "mlp_only_layers": [],
             "linear_num_key_heads": 2, "linear_key_head_dim": 4, "linear_num_value_heads": 4, "linear_value_head_dim": 4,
             "linear_conv_kernel_dim": 2, "full_attention_interval": 4, "partial_rotary_factor": 0.25,
             "rms_norm_eps": 1e-6, "rope_theta": 10000.0, "hidden_act": "silu", "max_position_embeddings": 128, "tie_word_embeddings": True})
spec("qwen3_5", tok="qwen2", H=32, I=32, L=2, NH=4, NKV=2, HD=8, rotary_dim=2,
     qk_norm="head", gated_q=True,
     linear={"KH": 2, "KD": 4, "VH": 4, "VD": 4, "KC": 2, "fused": False}, linear_layers=[1, 0],
     config={"model_type": "qwen3_5",
             "text_config": {"model_type": "qwen3_5_text", "hidden_size": 32, "intermediate_size": 32, "num_hidden_layers": 2,
                             "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 8,
                             "rms_norm_eps": 1e-6, "rope_theta": 10000.0, "hidden_act": "silu", "max_position_embeddings": 128,
                             "tie_word_embeddings": False, "partial_rotary_factor": 0.25,
                             "layer_types": ["linear_attention", "full_attention"],
                             "linear_num_key_heads": 2, "linear_key_head_dim": 4, "linear_num_value_heads": 4,
                             "linear_value_head_dim": 4, "linear_conv_kernel_dim": 2},
             "vision_config": {"model_type": "qwen3_5"}})
spec("qwen3_5_moe", tok="qwen2", prefix="model.language_model.", H=32, I=32, L=4, NH=4, NKV=2, HD=8, rotary_dim=2, qk_norm="head", gated_q=True, gate_swish=True,
     linear={"KH": 2, "KD": 4, "VH": 4, "VD": 4, "KC": 2, "fused": False}, linear_layers=[1, 1, 1, 0],
     moe={"E": 4, "K": 2, "MI": 12, "shared": 1, "shared_inter": 16, "shared_gate": True, "scoring": "softmax", "group_limited": False, "rsf": 1.0, "norm": True,
          "layers": [0, 1, 2, 3], "corr_bias": False, "layout": "fused", "prefix": "mlp.", "router": "gate.weight", "shared_name": "shared_expert."},
     config={"model_type": "qwen3_5_moe",
             "text_config": {"model_type": "qwen3_5_moe_text", "hidden_size": 32, "num_hidden_layers": 4, "num_attention_heads": 4,
                             "num_key_value_heads": 2, "head_dim": 8, "num_experts": 4, "num_experts_per_tok": 2,
                             "moe_intermediate_size": 12, "shared_expert_intermediate_size": 16, "rms_norm_eps": 1e-6,
                             "rope_theta": 10000.0, "hidden_act": "silu", "max_position_embeddings": 128, "tie_word_embeddings": False,
                             "partial_rotary_factor": 0.25, "output_gate_type": "swish",
                             "layer_types": ["linear_attention", "linear_attention", "linear_attention", "full_attention"],
                             "linear_num_key_heads": 2, "linear_key_head_dim": 4, "linear_num_value_heads": 4,
                             "linear_value_head_dim": 4, "linear_conv_kernel_dim": 2},
             "vision_config": {"model_type": "qwen3_5_moe"}})
spec("glm4_moe", tok="llama3", H=32, I=32, L=3, NH=4, NKV=2, HD=8, rotary_dim=4, qk_norm="head", attn_bias=True, o_bias=False,
     moe={"E": 4, "K": 2, "MI": 12, "shared": 1, "shared_inter": 16, "scoring": "sigmoid", "group_limited": True, "n_group": 1, "topk_group": 1,
          "rsf": 2.5, "norm": True, "layers": [1, 2], "corr_bias": True, "layout": "separate", "prefix": "mlp.", "router": "gate.weight",
          "shared_name": "shared_experts."},
     config={"model_type": "glm4_moe", "hidden_size": 32, "intermediate_size": 32, "moe_intermediate_size": 12, "num_hidden_layers": 3,
             "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 8, "n_routed_experts": 4, "n_shared_experts": 1,
             "num_experts_per_tok": 2, "first_k_dense_replace": 1, "n_group": 1, "topk_group": 1, "routed_scaling_factor": 2.5,
             "norm_topk_prob": True, "use_qk_norm": True, "partial_rotary_factor": 0.5, "attention_bias": True,
             "rms_norm_eps": 1e-6, "rope_theta": 10000.0, "hidden_act": "silu", "max_position_embeddings": 128, "tie_word_embeddings": True})
spec("glm_moe_dsa", tok="llama3", H=32, I=32, L=3, NH=4, NKV=4, HD=12, VD=8, rope_style="gptj", rotary_dim=4,
     mla={"q_lora_rank": 12, "kv_lora_rank": 16, "nope": 8, "rope": 4, "v": 8},
     moe={"E": 4, "K": 2, "MI": 12, "shared": 1, "shared_inter": 16, "scoring": "sigmoid", "group_limited": True, "n_group": 1, "topk_group": 1,
          "rsf": 2.5, "norm": True, "layers": [1, 2], "corr_bias": True, "layout": "separate", "prefix": "mlp.", "router": "gate.weight",
          "shared_name": "shared_experts."},
     config={"model_type": "glm_moe_dsa", "hidden_size": 32, "intermediate_size": 32, "moe_intermediate_size": 12, "num_hidden_layers": 3,
             "num_attention_heads": 4, "q_lora_rank": 12, "kv_lora_rank": 16, "qk_nope_head_dim": 8, "qk_rope_head_dim": 4,
             "v_head_dim": 8, "n_routed_experts": 4, "n_shared_experts": 1, "num_experts_per_tok": 2, "first_k_dense_replace": 1,
             "n_group": 1, "topk_group": 1, "routed_scaling_factor": 2.5, "norm_topk_prob": True,
             "rms_norm_eps": 1e-6, "rope_theta": 10000.0, "rope_interleave": True, "index_topk": 2048, "hidden_act": "silu",
             "max_position_embeddings": 128, "tie_word_embeddings": True})

spec("seed_oss", tok="llama3", HD=16, attn_bias=True, o_bias=False, lm_head="lm_head.weight",
     config={"model_type": "seed_oss", "hidden_size": 32, "intermediate_size": 32, "num_hidden_layers": 2, "num_attention_heads": 4,
             "num_key_value_heads": 2, "head_dim": 16, "rms_norm_eps": 1e-6, "rope_parameters": {"rope_type": "default", "rope_theta": 10000.0},
             "attention_bias": True, "attention_out_bias": False, "mlp_bias": False, "hidden_act": "silu", "max_position_embeddings": 128,
             "tie_word_embeddings": False})
spec("lfm2", tok="qwen2", L=3, eps=1e-5, theta=1000000.0, qk_norm="head", q_norm="self_attn.q_layernorm.weight", k_norm="self_attn.k_layernorm.weight",
     o="self_attn.out_proj.weight", final_norm="embedding_norm.weight", in_norm="operator_norm.weight", pre_ff_norm="ffn_norm.weight",
     gate="feed_forward.w1.weight", up="feed_forward.w3.weight", down="feed_forward.w2.weight", lm_head=None, conv_layers=[1, 0, 1], conv_K=3,
     config={"model_type": "lfm2", "vocab_size": 0, "hidden_size": 32, "num_hidden_layers": 3, "num_attention_heads": 4, "num_key_value_heads": 2,
             "norm_eps": 1e-5, "rope_theta": 1000000.0, "conv_bias": False, "conv_L_cache": 3, "block_ff_dim": 48, "block_multiple_of": 16,
             "block_ffn_dim_multiplier": 1.0, "block_auto_adjust_ff_dim": True, "full_attn_idxs": [1], "layer_types": ["conv", "full_attention", "conv"],
             "max_position_embeddings": 128, "tie_word_embeddings": True})
spec("mistral4", tok="llama3", NKV=4, HD=12, VD=8, L=3, prefix="model.language_model.", rope_style="gptj", rotary_dim=4, lm_head="lm_head.weight",
     mla={"q_lora_rank": 12, "kv_lora_rank": 16, "nope": 8, "rope": 4, "v": 8},
     scaling={"type": "yarn", "factor": 4.0, "beta_fast": 32, "beta_slow": 1, "mscale": 1.0, "mscale_all_dim": 1.0, "original_max_position_embeddings": 32},
     temp={"floor_scale": 32.0, "attn_scale": 0.1, "offset": 0.0, "all": True},
     moe={"E": 4, "K": 2, "MI": 12, "shared": 1, "scoring": "softmax", "group_limited": True, "group_top2": True, "n_group": 2, "topk_group": 1,
          "rsf": 1.5, "norm": True, "layers": [1, 2], "corr_bias": False, "layout": "fused_n", "prefix": "mlp.", "router": "gate.weight",
          "shared_name": "shared_experts."},
     config={"model_type": "mistral3", "architectures": ["Mistral3ForConditionalGeneration"],
             "text_config": {"model_type": "mistral4", "hidden_size": 32, "intermediate_size": 32, "moe_intermediate_size": 12, "num_hidden_layers": 3,
                             "num_attention_heads": 4, "num_key_value_heads": 4, "q_lora_rank": 12, "kv_lora_rank": 16, "qk_nope_head_dim": 8,
                             "qk_rope_head_dim": 4, "v_head_dim": 8, "n_routed_experts": 4, "n_shared_experts": 1, "num_experts_per_tok": 2,
                             "first_k_dense_replace": 1, "n_group": 2, "topk_group": 1, "routed_scaling_factor": 1.5, "norm_topk_prob": True,
                             "rms_norm_eps": 1e-6, "rope_interleave": True, "hidden_act": "silu", "max_position_embeddings": 128, "tie_word_embeddings": False,
                             "rope_parameters": {"rope_type": "yarn", "rope_theta": 10000.0, "factor": 4.0, "original_max_position_embeddings": 32,
                                                 "beta_fast": 32.0, "beta_slow": 1.0, "mscale": 1.0, "mscale_all_dim": 1.0, "llama_4_scaling_beta": 0.1,
                                                 "partial_rotary_factor": 4 / 12}},
             "vision_config": {"model_type": "pixtral"}})
spec("gemma4", tok="spm", L=5, NH=4, NKV=2, HD=8, prefix="model.language_model.", lm_head=None, act="gelu_tanh", attn_scale=1.0,
     post_attn_norm="post_attention_layernorm.weight", pre_ff_norm="pre_feedforward_layernorm.weight", post_ff_norm="post_feedforward_layernorm.weight",
     embed_scale=np.sqrt(32.0), qk_norm="head", v_norm=True, k_eq_v=True, sliding=4, sliding_layers=[1, 0, 1, 1, 0], layer_hd=[8, 16, 8, 8, 16],
     layer_nkv=[2, 1, 2, 2, 1], kv_shared=2, theta=1000000.0, local_rope=(10000.0, 8), global_rotary=(4, 16), ple_dim=4, layer_scale=True,
     layer_inter=[32, 32, 32, 64, 64],
     config={"model_type": "gemma4", "architectures": ["Gemma4ForConditionalGeneration"],
             "text_config": {"model_type": "gemma4_text", "hidden_size": 32, "intermediate_size": 32, "num_hidden_layers": 5, "num_attention_heads": 4,
                             "num_key_value_heads": 2, "head_dim": 8, "global_head_dim": 16, "num_global_key_value_heads": 1, "attention_k_eq_v": True,
                             "num_kv_shared_layers": 2, "use_double_wide_mlp": True, "hidden_size_per_layer_input": 4, "vocab_size_per_layer_input": 0,
                             "sliding_window": 4, "layer_types": ["sliding_attention", "full_attention", "sliding_attention", "sliding_attention", "full_attention"],
                             "rope_parameters": {"full_attention": {"rope_type": "proportional", "partial_rotary_factor": 0.25, "rope_theta": 1000000.0},
                                                 "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0}},
                             "rms_norm_eps": 1e-6, "hidden_activation": "gelu_pytorch_tanh", "final_logit_softcapping": None, "enable_moe_block": False,
                             "max_position_embeddings": 128, "tie_word_embeddings": True},
             "vision_config": {"model_type": "gemma4_vision"}})
spec("gemma3n", tok="spm", L=4, NH=4, NKV=2, HD=8, prefix="model.language_model.", lm_head=None, act="gelu_tanh", attn_scale=1.0,
     post_attn_norm="post_attention_layernorm.weight", pre_ff_norm="pre_feedforward_layernorm.weight", post_ff_norm="post_feedforward_layernorm.weight",
     embed_scale=np.sqrt(32.0), qk_norm="head", v_norm=True, sliding=4, sliding_layers=[1, 1, 0, 1], kv_shared=1, theta=1000000.0, local_rope=(10000.0, 8),
     ple_dim=4, altup={"A": 3, "active": 0, "correct_scale": True, "rank": 6}, sparsity=[0.95, 0.5, 0.0, 0.0], layer_inter=[32, 32, 48, 32],
     final_softcap=30.0,
     config={"model_type": "gemma3n", "architectures": ["Gemma3nForConditionalGeneration"],
             "text_config": {"model_type": "gemma3n_text", "hidden_size": 32, "intermediate_size": [32, 32, 48, 32], "num_hidden_layers": 4,
                             "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 8, "sliding_window": 4,
                             "layer_types": ["sliding_attention", "sliding_attention", "full_attention", "sliding_attention"],
                             "rope_theta": 1000000.0, "rope_local_base_freq": 10000.0, "rope_scaling": None, "rms_norm_eps": 1e-6,
                             "hidden_activation": "gelu_pytorch_tanh", "final_logit_softcapping": 30.0, "hidden_size_per_layer_input": 4,
                             "vocab_size_per_layer_input": 0, "altup_num_inputs": 3, "altup_active_idx": 0, "altup_coef_clip": 120.0,
                             "altup_correct_scale": True, "num_kv_shared_layers": 1, "laurel_rank": 6, "activation_sparsity_pattern": [0.95, 0.5, 0.0, 0.0],
                             "max_position_embeddings": 128, "tie_word_embeddings": True},
             "vision_config": {"model_type": "gemma3n_vision"}, "audio_config": {"model_type": "gemma3n_audio"}})


# --- generation ------------------------------------------------------------

def generate_generic(family, out_dir):
    s = SPECS[family]
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(4321)
    vocab, encode, bos, eos = make_tokenizer(s["tok"], out_dir)
    V = len(vocab)
    H, I, L, NH, NKV, HD, VD = s["H"], s["I"], s["L"], s["NH"], s["NKV"], s["HD"], s["VD"]
    weights = {}

    def mat(name, rows, cols, scale=0.2, register=True):
        w = bf16_round(rng.normal(0, scale, size=(rows, cols)))
        if register:
            weights[name] = w.T.copy() if s["conv1d"] and name.endswith(".weight") and "wte" not in name and "wpe" not in name and "ln" not in name else w
        return w

    def vec(name, n, scale=0.2):
        w = bf16_round(rng.normal(0, scale, size=(n,)))
        weights[name] = w
        return w

    def normw(name, n):
        """Norm weight (stored as w - 1 for the (1 + w) families) and, for LayerNorm families, its bias."""
        if name is None or name == "":
            return None
        w = bf16_round(1.0 + rng.normal(0, 0.1, size=(n,)))
        stored = w - 1.0 if s["norm"] in ("rms1p", "ln1p") else w
        weights[name] = bf16_round(stored)
        b = None
        if s["norm"] in ("ln", "ln1p") and s["ln_bias"]:
            b = vec(name[:-len(".weight")] + ".bias", n, 0.1)
        return (w, b)

    def bias_for(name, n, on):
        return vec(name[:-len(".weight")] + ".bias", n, 0.1) if on else None

    P = s["prefix"]
    embed = mat(P + s["embed"], V, H, 1.0)
    pos_embed = mat(P + s["pos_embed"], 64 + s["pos_offset"], H, 0.5) if s["pos_embed"] else None
    embed_norm = normw(P + s["embed_norm"], H) if s["embed_norm"] else None
    final_norm = normw(P + s["final_norm"], H) if s["norm"] != "none" else None
    lm_head = mat(s["lm_head"], V, H, 1.0) if s["lm_head"] else embed
    lm_bias = vec(s["lm_head"][:-len(".weight")] + ".bias", V, 0.1) if s["lm_bias"] else None
    layers = []
    lin_layers = s["linear_layers"]
    if lin_layers is None and s["full_interval"]:
        lin_layers = [((i + 1) % s["full_interval"] != 0) for i in range(L)]
    if lin_layers is None:
        lin_layers = [False] * L
    conv_layers = [bool(v) for v in (s["conv_layers"] or [0] * L)]
    layer_hd = s["layer_hd"] or [HD] * L
    layer_nkv = s["layer_nkv"] or [NKV] * L
    sliding_of = [bool(v) for v in (s["sliding_layers"] or [0] * L)]
    # KV sharing: the last `kv_shared` layers read the last earlier layer of their kind.
    kv_source = list(range(L))
    for i in range(L - s["kv_shared"], L):
        kv_source[i] = max(j for j in range(L - s["kv_shared"]) if sliding_of[j] == sliding_of[i])
    ple_dim = s["ple_dim"]
    if ple_dim:
        ple_embed = mat(P + "embed_tokens_per_layer.weight", V, L * ple_dim, 1.0)
        ple_proj = mat(P + "per_layer_model_projection.weight", L * ple_dim, H)
        ple_norm = normw(P + "per_layer_projection_norm.weight", ple_dim)[0]
    altup = s["altup"]
    if altup:
        altup_proj = [mat(P + f"altup_projections.{e}.weight", H, H) for e in range(altup["A"] - 1)]
        altup_unembed = [mat(P + f"altup_unembed_projections.{e}.weight", H, H) for e in range(altup["A"] - 1)]
    for i in range(L):
        lp = P + s["layer"].format(i=i)
        d = {}
        hd_l, nkv_l = layer_hd[i], layer_nkv[i]
        vd_l = VD if s["mla"] else hd_l
        d["in_norm"] = normw(lp + s["in_norm"], H) if s["in_norm"] else None
        d["post_attn_norm"] = normw(lp + s["post_attn_norm"], H) if s["post_attn_norm"] else None
        d["pre_ff_norm"] = normw(lp + s["pre_ff_norm"], H) if s["pre_ff_norm"] else None
        d["post_ff_norm"] = normw(lp + s["post_ff_norm"], H) if s["post_ff_norm"] else None
        d["mlp_norm"] = normw(lp + s["mlp_norm"], H) if s["mlp_norm"] else None
        qd, kvd = NH * hd_l, nkv_l * hd_l
        if conv_layers[i]:
            d["conv_in"] = mat(lp + "conv.in_proj.weight", 3 * H, H)
            d["conv_w"] = bf16_round(rng.normal(0, 0.3, size=(H, s["conv_K"])))
            weights[lp + "conv.conv.weight"] = d["conv_w"].reshape(H, 1, s["conv_K"])
            d["o"] = mat(lp + "conv.out_proj.weight", H, H)
            d["ob"] = None
        elif lin_layers[i]:
            ln, ap = s["linear"], "linear_attn."
            kd_tot, vd_tot = ln["KH"] * ln["KD"], ln["VH"] * ln["VD"]
            if ln["fused"]:
                d["qkvz"] = mat(lp + ap + "in_proj_qkvz.weight", 2 * kd_tot + 2 * vd_tot, H)
                d["ba"] = mat(lp + ap + "in_proj_ba.weight", 2 * ln["VH"], H)
            else:
                d["qkv"] = mat(lp + ap + "in_proj_qkv.weight", 2 * kd_tot + vd_tot, H)
                d["z"] = mat(lp + ap + "in_proj_z.weight", vd_tot, H)
                d["b"] = mat(lp + ap + "in_proj_b.weight", ln["VH"], H)
                d["a"] = mat(lp + ap + "in_proj_a.weight", ln["VH"], H)
            d["conv"] = bf16_round(rng.normal(0, 0.2, size=(2 * kd_tot + vd_tot, ln["KC"])))
            weights[lp + ap + "conv1d.weight"] = d["conv"]
            d["dt"] = vec(lp + ap + "dt_bias", ln["VH"], 0.1)
            d["alog"] = vec(lp + ap + "A_log", ln["VH"], 0.1)
            d["lnorm"] = normw(lp + ap + "norm.weight", ln["VD"])[0]
            d["o"] = mat(lp + ap + "out_proj.weight", H, vd_tot)
            d["ob"] = None
        elif s["mla"]:
            m = s["mla"]
            if m["q_lora_rank"]:
                d["q_a"] = mat(lp + "self_attn.q_a_proj.weight", m["q_lora_rank"], H)
                d["q_a_norm"] = normw(lp + "self_attn.q_a_layernorm.weight", m["q_lora_rank"])[0]
                d["q_b"] = mat(lp + "self_attn.q_b_proj.weight", NH * (m["nope"] + m["rope"]), m["q_lora_rank"])
            else:
                d["q_b"] = mat(lp + "self_attn.q_proj.weight", NH * (m["nope"] + m["rope"]), H)
            d["kv_a"] = mat(lp + "self_attn.kv_a_proj_with_mqa.weight", m["kv_lora_rank"] + m["rope"], H)
            d["kv_a_norm"] = normw(lp + "self_attn.kv_a_layernorm.weight", m["kv_lora_rank"])[0]
            d["kv_b"] = mat(lp + "self_attn.kv_b_proj.weight", NH * (m["nope"] + m["v"]), m["kv_lora_rank"])
        elif s["qkv"]:
            w = mat(lp + s["qkv"], qd + 2 * kvd, H)
            b = bias_for(lp + s["qkv"], qd + 2 * kvd, s["attn_bias"])
            d["qkv"], d["qkv_b"] = w, b
        else:
            d["q"] = mat(lp + s["q"], (2 * qd if s["gated_q"] else qd), H)
            d["qb"] = bias_for(lp + s["q"], qd, s["attn_bias"])
            if kv_source[i] == i:
                d["k"] = mat(lp + s["k"], kvd, H)
                d["kb"] = bias_for(lp + s["k"], kvd, s["attn_bias"])
                if not (s["k_eq_v"] and not sliding_of[i]):
                    d["v"] = mat(lp + s["v"], kvd, H)
                    d["vb"] = bias_for(lp + s["v"], kvd, s["attn_bias"])
                else:
                    d["vb"] = None
            else:
                d["kb"], d["vb"] = None, None
        if not lin_layers[i] and not conv_layers[i]:
            d["o"] = mat(lp + s["o"], H, NH * vd_l)
            d["ob"] = bias_for(lp + s["o"], H, s["attn_bias"] if s["o_bias"] is None else s["o_bias"])
        if not lin_layers[i] and not conv_layers[i] and s["qk_norm"] in ("head", "heads", "full"):
            qn = {"head": hd_l, "heads": NH * hd_l, "full": NH * hd_l}[s["qk_norm"]]
            kn = {"head": hd_l, "heads": nkv_l * hd_l, "full": nkv_l * hd_l}[s["qk_norm"]]
            d["qn"] = normw(lp + s["q_norm"], qn)
            if kv_source[i] == i:
                d["kn"] = normw(lp + s["k_norm"], kn)
            if s["qk_norm"] == "heads":  # Cohere stores [heads, head_dim]
                weights[lp + s["q_norm"]] = weights[lp + s["q_norm"]].reshape(NH, hd_l)
                weights[lp + s["k_norm"]] = weights[lp + s["k_norm"]].reshape(nkv_l, hd_l)
        elif conv_layers[i] and s["qk_norm"] == "head":
            pass
        if not lin_layers[i] and s["sinks"]:
            d["sinks"] = vec(lp + s["sinks"], NH, 1.0)
        if ple_dim:
            d["ple_gate"] = mat(lp + "per_layer_input_gate.weight", ple_dim, H)
            d["ple_out"] = mat(lp + "per_layer_projection.weight", H, ple_dim)
            d["ple_norm"] = normw(lp + "post_per_layer_input_norm.weight", H)[0]
        if altup:
            A = altup["A"]
            d["altup_scale"] = vec(lp + "altup.correct_output_scale", H, 0.5) if altup["correct_scale"] else None
            d["altup_correct"] = mat(lp + "altup.correction_coefs.weight", A, A, 0.5)
            d["altup_predict"] = mat(lp + "altup.prediction_coefs.weight", A * A, A, 0.5)
            d["altup_router"] = mat(lp + "altup.modality_router.weight", A, H, 2.0)
            d["altup_router_norm"] = normw(lp + "altup.router_norm.weight", H)[0]
            d["laurel_l"] = mat(lp + "laurel.linear_left.weight", altup["rank"], H)
            d["laurel_r"] = mat(lp + "laurel.linear_right.weight", H, altup["rank"])
            d["laurel_norm"] = normw(lp + "laurel.post_laurel_norm.weight", H)[0]
        if s["layer_scale"]:
            d["layer_scale"] = bf16_round(1.0 + rng.normal(0, 0.1, size=(1,)))
            weights[lp + "layer_scalar"] = d["layer_scale"]
        moe = s["moe"]
        if moe and i in moe["layers"]:
            mp = lp + moe["prefix"]
            E, K, MI = moe["E"], moe["K"], moe["MI"]
            d["router"] = mat(mp + moe["router"], E, H)
            d["router_b"] = bias_for(mp + moe["router"], E, moe.get("router_bias", False))
            d["corr_b"] = vec(mp + "gate.e_score_correction_bias", E, 0.5) if moe["corr_bias"] else None
            experts = []
            for e in range(E):
                ex = {"gate": bf16_round(rng.normal(0, 0.2, size=(MI, H))), "up": bf16_round(rng.normal(0, 0.2, size=(MI, H))),
                      "down": bf16_round(rng.normal(0, 0.2, size=(H, MI))), "gb": None, "ub": None, "db": None}
                if moe.get("bias"):
                    ex["gb"], ex["ub"], ex["db"] = (bf16_round(rng.normal(0, 0.1, size=(n,))) for n in (MI, MI, H))
                experts.append(ex)
            if moe["layout"] == "separate":
                names = moe.get("expert_names", ("gate_proj.weight", "up_proj.weight", "down_proj.weight"))
                for e, ex in enumerate(experts):
                    weights[f"{mp}experts.{e}.{names[0]}"] = ex["gate"]
                    weights[f"{mp}experts.{e}.{names[1]}"] = ex["up"]
                    weights[f"{mp}experts.{e}.{names[2]}"] = ex["down"]
            elif moe["layout"] == "fused_n":
                # [E, 2I, H] / [E, H, I] as nn.Linear would store them (Mistral 4, Gemma 4).
                weights[mp + "experts.gate_up_proj"] = np.stack([np.concatenate([ex["gate"], ex["up"]], 0) for ex in experts])
                weights[mp + "experts.down_proj"] = np.stack([ex["down"] for ex in experts])
            else:
                gu = np.zeros((E, H, 2 * MI), np.float32)
                dn = np.zeros((E, MI, H), np.float32)
                for e, ex in enumerate(experts):
                    if moe["layout"] == "fused_t_interleaved":
                        gu[e, :, 0::2] = ex["gate"].T
                        gu[e, :, 1::2] = ex["up"].T
                    else:
                        gu[e] = np.concatenate([ex["gate"], ex["up"]], 0).T
                    dn[e] = ex["down"].T
                weights[mp + "experts.gate_up_proj"] = gu
                weights[mp + "experts.down_proj"] = dn
                if moe.get("bias"):
                    gub = np.zeros((E, 2 * MI), np.float32)
                    dnb = np.zeros((E, H), np.float32)
                    for e, ex in enumerate(experts):
                        gub[e, 0::2], gub[e, 1::2], dnb[e] = ex["gb"], ex["ub"], ex["db"]
                    weights[mp + "experts.gate_up_proj_bias"] = gub
                    weights[mp + "experts.down_proj_bias"] = dnb
            d["experts"] = experts
            if moe["shared"]:
                sp = mp + moe["shared_name"]
                si = moe.get("shared_inter", moe["shared"] * MI)
                d["shared"] = {"gate": mat(sp + "gate_proj.weight", si, H), "up": mat(sp + "up_proj.weight", si, H), "down": mat(sp + "down_proj.weight", H, si),
                               "gate_vec": mat(lp + "mlp.shared_expert_gate.weight", 1, H)[0] if moe.get("shared_gate") else None}
        else:
            inter = s["layer_inter"][i] if s["layer_inter"] else I
            if s["mlp"] == "gated":
                d["gate"] = mat(lp + s["gate"], inter, H)
                d["up"] = mat(lp + s["up"], inter, H)
                d["gb"] = bias_for(lp + s["gate"], inter, s["mlp_bias"])
                d["ub"] = bias_for(lp + s["up"], inter, s["mlp_bias"])
            elif s["mlp"] == "gated_fused":
                d["gate_up"] = mat(lp + s["gate_up"], 2 * inter, H)
                d["gub"] = bias_for(lp + s["gate_up"], 2 * inter, s["mlp_bias"])
            else:
                d["up"] = mat(lp + s["up"], inter, H)
                d["ub"] = bias_for(lp + s["up"], inter, s["mlp_bias"])
            d["down"] = mat(lp + s["down"], H, inter)
            d["db"] = bias_for(lp + s["down"], H, s["mlp_bias"])
        layers.append(d)

    # config.json (with the vocabulary size filled in), generation config, weights.
    config = json.loads(json.dumps(s["config"]))
    if s.get("extra_config"):
        config.update(s["extra_config"])
    tc = config.get("text_config", config)
    for key in ("vocab_size", "padded_vocab_size"):
        if key in tc:
            tc[key] = V
    if "vocab_size" not in tc and "padded_vocab_size" not in tc:
        tc["vocab_size"] = V
    if s["scaling"] and "rope_scaling" not in tc and "rope_parameters" not in tc:
        tc["rope_scaling"] = s["scaling"]
    json.dump(config, open(f"{out_dir}/config.json", "w"), indent=1)
    json.dump({"eos_token_id": vocab[eos], "bos_token_id": vocab[bos] if bos else None, "do_sample": False}, open(f"{out_dir}/generation_config.json", "w"))
    header, blobs, offset = {}, [], 0
    for n in sorted(weights):
        u = bf16_bits(weights[n])
        header[n] = {"dtype": "BF16", "shape": list(u.shape), "data_offsets": [offset, offset + u.nbytes]}
        blobs.append(u.tobytes())
        offset += u.nbytes
    hb = json.dumps(header).encode()
    hb += b" " * ((8 - len(hb) % 8) % 8)
    with open(f"{out_dir}/model.safetensors", "wb") as f:
        f.write(struct.pack("<Q", len(hb)))
        f.write(hb)
        for b in blobs:
            f.write(b)

    # --- reference forward pass (float32, written from the Hugging Face modeling code) ---
    eps = s["eps"]

    def norm(x, nw):
        if s["norm"] == "none":
            mu = x.mean(-1, keepdims=True)
            var = ((x - mu) ** 2).mean(-1, keepdims=True)
            return (x - mu) / np.sqrt(var + eps)
        w, b = nw
        if s["norm"] in ("rms", "rms1p"):
            return x / np.sqrt(np.mean(x * x, -1, keepdims=True) + eps) * w
        mu = x.mean(-1, keepdims=True)
        var = ((x - mu) ** 2).mean(-1, keepdims=True)
        y = (x - mu) / np.sqrt(var + eps) * w
        return y + b if b is not None else y

    def head_norm(x, w, b):
        """q/k norm over the last axis of x (weightless RMS when w is None)."""
        if w is None:
            return x / np.sqrt(np.mean(x * x, -1, keepdims=True) + eps)
        if s["norm"] in ("rms", "rms1p"):
            return x / np.sqrt(np.mean(x * x, -1, keepdims=True) + eps) * w
        mu = x.mean(-1, keepdims=True)
        var = ((x - mu) ** 2).mean(-1, keepdims=True)
        y = (x - mu) / np.sqrt(var + eps) * w
        return y + b if b is not None else y

    def act_fn(x):
        a = s["act"]
        if a == "silu":
            return x / (1 + np.exp(-x))
        if a in ("gelu_new", "gelu_tanh"):
            return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))
        if a == "gelu":
            from math import erf
            return 0.5 * x * (1 + np.vectorize(erf)(x / np.sqrt(2)))
        if a == "relu":
            return np.maximum(x, 0)
        if a == "relu2":
            return np.maximum(x, 0) ** 2
        raise KeyError(a)

    rd = s["rotary_dim"]
    # Global table: `rd` coordinates rotate with frequencies over `fd` (proportional rope).
    rd, fd = s["global_rotary"] if s["global_rotary"] else (rd, rd)

    def inv_freq_and_factor(theta, rd=rd, fd=fd, sc=None):
        inv = 1.0 / (theta ** (np.arange(0, rd, 2, dtype=np.float64) / fd))
        sc = s["scaling"] if sc is None else (sc or None)
        factor = 1.0
        if sc and sc.get("type", sc.get("rope_type")) == "yarn":
            f = sc["factor"]
            base_ = theta
            om = sc["original_max_position_embeddings"]
            bf, bs_ = sc.get("beta_fast", 32), sc.get("beta_slow", 1)

            def corr_dim(rot):
                return (rd * np.log(om / (rot * 2 * np.pi))) / (2 * np.log(base_))
            low, high = corr_dim(bf), corr_dim(bs_)
            if sc.get("truncate", True):
                low, high = np.floor(low), np.ceil(high)
            low, high = max(low, 0), min(high, rd - 1)
            if low == high:
                high += 0.001
            ramp = np.clip((np.arange(rd // 2) - low) / (high - low), 0, 1)
            extrap = 1 - ramp
            inv = (inv / f) * (1 - extrap) + inv * extrap

            def mscale(scale, m):
                return 1.0 if scale <= 1 else 0.1 * m * np.log(scale) + 1.0
            if "attention_factor" in sc:
                factor = sc["attention_factor"]
            elif sc.get("mscale") and sc.get("mscale_all_dim"):
                factor = mscale(f, sc["mscale"]) / mscale(f, sc["mscale_all_dim"])
            else:
                factor = mscale(f, 1.0)
        elif sc and sc.get("type") == "longrope":
            inv = inv / np.array(sc["short_factor"], dtype=np.float64)
            om = s["extra_config"]["original_max_position_embeddings"]
            f = s["extra_config"]["max_position_embeddings"] / om
            factor = 1.0 if f <= 1 else np.sqrt(1 + np.log(f) / np.log(om))
        return inv, factor

    def rope_tables(T):
        inv, factor = inv_freq_and_factor(s["theta"])
        ang = np.outer(np.arange(T, dtype=np.float64), inv)
        return (np.cos(ang) * factor).astype(np.float32), (np.sin(ang) * factor).astype(np.float32)

    def rope_tables_local(T):
        theta, lrd = s["local_rope"]
        inv, factor = inv_freq_and_factor(theta, lrd, lrd, {})
        ang = np.outer(np.arange(T, dtype=np.float64), inv)
        return (np.cos(ang) * factor).astype(np.float32), (np.sin(ang) * factor).astype(np.float32)

    def apply_rope(x, cos, sin, off):  # x [T, nh, HD]
        rd = 2 * cos.shape[1]
        rot = x[..., off:off + rd]
        c, sn = cos[:, None, :], sin[:, None, :]
        if s["rope_style"] == "neox":
            x1, x2 = rot[..., :rd // 2], rot[..., rd // 2:]
            out = np.concatenate([x1 * c - x2 * sn, x2 * c + x1 * sn], -1)
        else:
            x1, x2 = rot[..., 0::2], rot[..., 1::2]
            out = np.empty_like(rot)
            out[..., 0::2] = x1 * c - x2 * sn
            out[..., 1::2] = x2 * c + x1 * sn
        y = x.copy()
        y[..., off:off + rd] = out
        return y

    def alibi_slopes(n):
        pow2 = 1
        while pow2 < n:
            pow2 *= 2
        full = [2.0 ** (-8.0 * (i + 1) / pow2) for i in range(pow2)]
        if pow2 == n:
            return full
        return (full[1::2] + full[0::2])[:n]

    attn_scale = s["attn_scale"] if s["attn_scale"] is not None else 1.0 / np.sqrt(HD)
    if s["mla"]:
        sc = s["scaling"]
        if sc and sc.get("mscale_all_dim"):
            ms = 0.1 * sc["mscale_all_dim"] * np.log(sc["factor"]) + 1.0
            attn_scale = attn_scale * ms * ms
    rope_layers = s["rope_layers"] or ([1] * L if s["pos"] == "rope" else [0] * L)
    sliding_layers = s["sliding_layers"] or [0] * L
    shared_kv = {}

    def head_rms(x):
        return x / np.sqrt(np.mean(x * x, -1, keepdims=True) + eps)

    def attention(d, li, h, cos, sin):
        T = h.shape[0]
        HD, NKV, VD = layer_hd[li], layer_nkv[li], (s["VD"] if s["mla"] else layer_hd[li])
        if s["mla"]:
            m = s["mla"]
            nope, rp, vd = m["nope"], m["rope"], m["v"]
            if "q_a" in d:
                qa = h @ d["q_a"].T
                qa = qa / np.sqrt(np.mean(qa * qa, -1, keepdims=True) + eps) * d["q_a_norm"]
                q = qa @ d["q_b"].T
            else:
                q = h @ d["q_b"].T
            q = q.reshape(T, NH, nope + rp)
            kva = h @ d["kv_a"].T
            ckv, k_pe = kva[:, :m["kv_lora_rank"]], kva[:, m["kv_lora_rank"]:]
            ckv = ckv / np.sqrt(np.mean(ckv * ckv, -1, keepdims=True) + eps) * d["kv_a_norm"]
            kvb = (ckv @ d["kv_b"].T).reshape(T, NH, nope + vd)
            k_nope, v = kvb[..., :nope], kvb[..., nope:]
            q = apply_rope(q, cos, sin, nope)
            k_pe = apply_rope(k_pe.reshape(T, 1, rp), cos, sin, 0)
            k = np.concatenate([k_nope, np.repeat(k_pe, NH, axis=1)], -1)
            v = v
            groups = 1
        else:
            qd, kvd = NH * HD, NKV * HD
            if "qkv" in d:
                f = h @ d["qkv"].T
                if d["qkv_b"] is not None:
                    f = f + d["qkv_b"]
                if s["qkv_layout"] == "concat":
                    q, k, v = f[:, :qd], f[:, qd:qd + kvd], f[:, qd + kvd:]
                elif s["qkv_layout"] == "heads":
                    f = f.reshape(T, NH, 3, HD)
                    q, k, v = f[:, :, 0].reshape(T, qd), f[:, :, 1].reshape(T, kvd), f[:, :, 2].reshape(T, kvd)
                else:  # grouped
                    g = NH // NKV
                    f = f.reshape(T, NKV, g + 2, HD)
                    q = f[:, :, :g].reshape(T, qd)
                    k = f[:, :, g].reshape(T, kvd)
                    v = f[:, :, g + 1].reshape(T, kvd)
            else:
                q = h @ d["q"].T
                if d["qb"] is not None:
                    q = q + d["qb"]
                if kv_source[li] == li:
                    k = h @ d["k"].T
                    v = h @ d["v"].T if "v" in d else k.copy()
                    if d["kb"] is not None:
                        k = k + d["kb"]
                    if d["vb"] is not None:
                        v = v + d["vb"]
                else:
                    k = v = None
                gate = None
                if s["gated_q"]:
                    q = q.reshape(T, NH, 2 * HD)
                    gate = q[:, :, HD:].reshape(T, qd)
                    q = q[:, :, :HD].reshape(T, qd)
            if s["clip"] is not None:
                q, k, v = (np.clip(t, -s["clip"], s["clip"]) for t in (q, k, v))
            qn = s["qk_norm"]
            use_rope = bool(rope_layers[li])
            own_kv = kv_source[li] == li
            if qn == "full":
                q = head_norm(q, *d["qn"])
                if own_kv:
                    k = head_norm(k, *d["kn"])
            q = q.reshape(T, NH, HD)
            if own_kv:
                k, v = k.reshape(T, NKV, HD), v.reshape(T, NKV, HD)
            if qn == "head":
                q = head_norm(q, *d["qn"])
                if own_kv:
                    k = head_norm(k, *d["kn"])
            elif qn == "heads":
                q = head_norm(q, d["qn"][0].reshape(NH, HD), None)
                k = head_norm(k, d["kn"][0].reshape(NKV, HD), None)
            elif qn == "l2" and use_rope:
                q, k = head_norm(q, None, None), head_norm(k, None, None)
            if use_rope:
                q = apply_rope(q, cos, sin, 0)
                if own_kv:
                    k = apply_rope(k, cos, sin, 0)
            if s["temp"] and (not use_rope or s["temp"].get("all")):
                p = np.arange(T, dtype=np.float32)
                scl = np.log(np.floor((p + s["temp"].get("offset", 1.0)) / s["temp"]["floor_scale"]) + 1.0) * s["temp"]["attn_scale"] + 1.0
                q = q * scl[:, None, None]
            if own_kv:
                if s["v_norm"]:
                    v = head_rms(v)
                shared_kv[li] = (k, v)
            else:
                k, v = shared_kv[kv_source[li]]
            groups = NH // NKV
        slopes = alibi_slopes(NH) if s["pos"] == "alibi" else None
        out = np.zeros((T, NH, VD), np.float32)
        for hh in range(NH):
            kv = hh // groups
            sc_ = (q[:, hh, :] @ k[:, kv, :].T) * attn_scale
            if slopes:
                sc_ = sc_ + slopes[hh] * np.arange(T)[None, :]
            mask = np.triu(np.ones((T, T), dtype=bool), k=1)
            if sliding_layers[li]:
                mask |= np.tril(np.ones((T, T), dtype=bool), k=-s["sliding"])
            sc_ = np.where(mask, -np.inf, sc_)
            if "sinks" in d:
                sc_ = np.concatenate([sc_, np.full((T, 1), d["sinks"][hh])], 1)
            sc_ = sc_ - sc_.max(-1, keepdims=True)
            pr = np.exp(sc_)
            pr = pr / pr.sum(-1, keepdims=True)
            if "sinks" in d:
                pr = pr[:, :T]
            out[:, hh, :] = pr @ v[:, kv, :]
        o = out.reshape(T, NH * VD) @ d["o"].T
        if d["ob"] is not None:
            o = o + d["ob"]
        if s["gated_q"] and not s["mla"] and not s["qkv"]:
            gs = 1 / (1 + np.exp(-gate))
            if s["gate_swish"]:
                gs = gate * gs
            o = (out.reshape(T, NH * VD) * gs) @ d["o"].T
            if d["ob"] is not None:
                o = o + d["ob"]
        return o

    def conv_block(d, h):
        # LFM2 short convolution: B * x through a depthwise causal conv, gated by C.
        T = h.shape[0]
        bcx = h @ d["conv_in"].T
        B, C, x = bcx[:, :H], bcx[:, H:2 * H], bcx[:, 2 * H:]
        bx = B * x
        K = d["conv_w"].shape[1]
        y = np.zeros((T, H), np.float32)
        for t in range(T):
            for i in range(K):
                if t - (K - 1 - i) >= 0:
                    y[t] += d["conv_w"][:, i] * bx[t - (K - 1 - i)]
        return (C * y) @ d["o"].T

    def normal_ppf(p):
        from math import erf
        lo, hi = -10.0, 10.0
        for _ in range(200):
            mid = (lo + hi) / 2
            if 0.5 * (1 + erf(mid / np.sqrt(2))) < p:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2

    def gaussian_topk(g, sparsity):
        mean = g.mean(-1, keepdims=True)
        std = np.sqrt(((g - mean) ** 2).mean(-1, keepdims=True))
        return np.maximum(g - (mean + std * normal_ppf(sparsity)), 0)

    def linear_attn(d, h):
        ln = s["linear"]
        KH, KD, VH, VD, KC = ln["KH"], ln["KD"], ln["VH"], ln["VD"], ln["KC"]
        kd_tot, vd_tot = KH * KD, VH * VD
        T = h.shape[0]
        if "qkvz" in d:
            p = h @ d["qkvz"].T
            sub = vd_tot // KH
            p = p.reshape(T, KH, 2 * KD + 2 * sub)
            q = p[:, :, :KD].reshape(T, kd_tot)
            k = p[:, :, KD:2 * KD].reshape(T, kd_tot)
            v = p[:, :, 2 * KD:2 * KD + sub].reshape(T, vd_tot)
            z = p[:, :, 2 * KD + sub:].reshape(T, vd_tot)
            mixed = np.concatenate([q, k, v], -1)
            pba = h @ d["ba"].T
            subb = VH // KH
            pba = pba.reshape(T, KH, 2 * subb)
            b = pba[:, :, :subb].reshape(T, VH)
            a = pba[:, :, subb:].reshape(T, VH)
        else:
            mixed = h @ d["qkv"].T
            z = h @ d["z"].T
            b, a = h @ d["b"].T, h @ d["a"].T
        conv_dim = 2 * kd_tot + vd_tot
        y = np.zeros((T, conv_dim), np.float32)
        for t in range(T):
            acc = np.zeros(conv_dim, np.float32)
            for i in range(KC):
                if t - i >= 0:
                    acc += d["conv"][:, KC - 1 - i] * mixed[t - i]
            y[t] = acc / (1 + np.exp(-acc))
        q = y[:, :kd_tot].reshape(T, KH, KD)
        k = y[:, kd_tot:2 * kd_tot].reshape(T, KH, KD)
        v = y[:, 2 * kd_tot:].reshape(T, VH, VD)
        beta = 1 / (1 + np.exp(-b))
        g = -np.exp(d["alog"]) * np.log1p(np.exp(a + d["dt"]))
        if VH > KH:
            rep = VH // KH
            q = np.repeat(q, rep, axis=1)
            k = np.repeat(k, rep, axis=1)
        q = q / np.sqrt(np.sum(q * q, -1, keepdims=True) + 1e-6) / np.sqrt(KD)
        k = k / np.sqrt(np.sum(k * k, -1, keepdims=True) + 1e-6)
        S = np.zeros((VH, KD, VD), np.float32)
        core = np.zeros((T, VH, VD), np.float32)
        for t in range(T):
            S = S * np.exp(g[t])[:, None, None]
            for vhi in range(VH):
                mem = S[vhi].T @ k[t, vhi]
                delta = (v[t, vhi] - mem) * beta[t, vhi]
                S[vhi] = S[vhi] + np.outer(k[t, vhi], delta)
                core[t, vhi] = S[vhi].T @ q[t, vhi]
        o = core.reshape(T, VH, VD)
        o = o / np.sqrt(np.mean(o * o, -1, keepdims=True) + eps) * d["lnorm"]
        o = o * (z.reshape(T, VH, VD) / (1 + np.exp(-z.reshape(T, VH, VD))))
        return o.reshape(T, vd_tot) @ d["o"].T

    def expert_out(ex, x, biases=True):
        g, u = x @ ex["gate"].T, x @ ex["up"].T
        if biases and ex.get("gb") is not None:
            g, u = g + ex["gb"], u + ex["ub"]
        moe = s["moe"]
        if moe and moe.get("swiglu"):
            alpha, limit = moe["swiglu"]
            g = np.minimum(g, limit)
            u = np.clip(u, -limit, limit)
            hmid = (u + 1.0) * (g / (1 + np.exp(-alpha * g)))
        else:
            hmid = act_fn(g) * u
        y = hmid @ ex["down"].T
        if biases and ex.get("db") is not None:
            y = y + ex["db"]
        return y

    def mlp(d, h, li=0):
        if "experts" in d:
            moe = s["moe"]
            E, K = moe["E"], moe["K"]
            logits = h @ d["router"].T
            if d["router_b"] is not None:
                logits = logits + d["router_b"]
            if moe["scoring"] == "softmax":
                sc_ = np.exp(logits - logits.max(-1, keepdims=True))
                sc_ = sc_ / sc_.sum(-1, keepdims=True)
            else:
                sc_ = 1 / (1 + np.exp(-logits))
            out = np.zeros_like(h)
            for t in range(h.shape[0]):
                choice = sc_[t] + (d["corr_b"] if d["corr_b"] is not None else 0)
                if moe["group_limited"]:
                    ng = moe["n_group"]
                    per = E // ng
                    gs = []
                    for g in range(ng):
                        grp = np.sort(choice[g * per:(g + 1) * per])[::-1]
                        gs.append(grp[:2].sum() if (d["corr_b"] is not None or moe.get("group_top2")) else grp[0])
                    keep = np.argsort(-np.array(gs), kind="stable")[:moe["topk_group"]]
                    masked = np.full(E, -np.inf)
                    for g in keep:
                        masked[g * per:(g + 1) * per] = choice[g * per:(g + 1) * per]
                    choice = masked
                idx = np.argsort(-choice, kind="stable")[:K]
                w = sc_[t][idx]
                if moe["norm"]:
                    w = w / (w.sum() + 1e-20)
                w = w * moe["rsf"]
                for e, we in zip(idx, w):
                    if moe.get("scale_input"):
                        out[t] += expert_out(d["experts"][e], h[t] * we)
                    else:
                        out[t] += we * expert_out(d["experts"][e], h[t])
            if "shared" in d:
                sh = d["shared"]
                so = expert_out(sh, h, biases=False)
                if sh["gate_vec"] is not None:
                    so = so * (1 / (1 + np.exp(-(h @ sh["gate_vec"]))))[:, None]
                out = out + so
            return out
        if s["mlp"] == "gated":
            g, u = h @ d["gate"].T, h @ d["up"].T
            if d["gb"] is not None:
                g, u = g + d["gb"], u + d["ub"]
            if s["sparsity"] and s["sparsity"][li] > 0:
                g = gaussian_topk(g, s["sparsity"][li])
            m = act_fn(g) * u
        elif s["mlp"] == "gated_fused":
            gu = h @ d["gate_up"].T
            if d["gub"] is not None:
                gu = gu + d["gub"]
            inter = gu.shape[1] // 2
            m = act_fn(gu[:, :inter]) * gu[:, inter:]
        else:
            u = h @ d["up"].T
            if d["ub"] is not None:
                u = u + d["ub"]
            m = act_fn(u)
        y = m @ d["down"].T
        if d["db"] is not None:
            y = y + d["db"]
        return y

    def rms_magnitude(x):
        return np.sqrt(np.mean(x * x, -1, keepdims=True))

    def per_layer_inputs(tokens, x):
        # Gemma 3n / 4: normalised projection of the embedding plus the token's per-layer embedding.
        T = len(tokens)
        proj = (x @ ple_proj.T) * np.float32(H ** -0.5)
        proj = proj.reshape(T, L, ple_dim)
        proj = proj / np.sqrt(np.mean(proj * proj, -1, keepdims=True) + eps) * ple_norm
        emb = ple_embed[tokens].reshape(T, L, ple_dim) * np.float32(np.sqrt(ple_dim))
        return (proj + emb) * np.float32(2 ** -0.5)

    def ple_block(d, first, ple_li):
        g = act_fn(first @ d["ple_gate"].T) * ple_li
        return norm(g @ d["ple_out"].T, (d["ple_norm"], None))

    def modalities(d, x):
        r = norm(x, (d["altup_router_norm"], None)) * np.float32(1.0 / H)
        return np.tanh(r @ d["altup_router"].T)

    def layer_std(d, li, x, h_fn, ple_li):
        # Standard layer; returns the new residual.
        rm = np.float32(s["residual_mult"])
        h = norm(x, d["in_norm"]) if (d["in_norm"] is not None or s["norm"] == "none") and s["in_norm"] is not None else x
        a = h_fn(h)
        if d["post_attn_norm"] is not None:
            a = norm(a, d["post_attn_norm"])
        if s["parallel"]:
            m_in = norm(x, d["mlp_norm"]) if d["mlp_norm"] is not None else h
            m = mlp(d, m_in, li)
            if d["post_ff_norm"] is not None:
                m = norm(m, d["post_ff_norm"])
            x = x + rm * (a + m)
        else:
            x = x + rm * a
            h2 = norm(x, d["pre_ff_norm"]) if (d["pre_ff_norm"] is not None or s["norm"] == "none") and s["pre_ff_norm"] is not None else x
            m = mlp(d, h2, li)
            if d["post_ff_norm"] is not None:
                m = norm(m, d["post_ff_norm"])
            x = x + rm * m
        if "ple_gate" in d:
            x = x + ple_block(d, x, ple_li)
        if "layer_scale" in d:
            x = x * d["layer_scale"]
        return x

    def layer_altup(d, li, streams, h_fn, ple_li):
        # Gemma 3n: predict every stream, run the block on the active one, correct all streams.
        A, active = altup["A"], altup["active"]
        T = streams.shape[1]
        mod = modalities(d, streams[active])
        coefs = (mod @ d["altup_predict"].T).reshape(T, A, A)  # [t][a][i]
        pred = streams + np.einsum("tai,itd->atd", coefs, streams)
        act = pred[active]
        h = norm(act, d["in_norm"])
        laurel = h + norm((h @ d["laurel_l"].T) @ d["laurel_r"].T, (d["laurel_norm"], None))
        a = norm(h_fn(h), d["post_attn_norm"])
        attn_laurel = (act + a + laurel) / np.float32(np.sqrt(2))
        m = norm(mlp(d, norm(attn_laurel, d["pre_ff_norm"]), li), d["post_ff_norm"])
        activated = attn_laurel + m
        cc = modalities(d, activated) @ d["altup_correct"].T + 1.0  # [t][a]
        innovation = activated - pred[active]
        corrected = pred + cc.T[:, :, None] * innovation[None]
        first = corrected[active].copy()
        if d["altup_scale"] is not None:
            first = first * d["altup_scale"]
        corrected[1:] += ple_block(d, first, ple_li)
        return corrected

    def forward(tokens):
        T = len(tokens)
        x = embed[tokens] * np.float32(s["embed_scale"])
        if pos_embed is not None:
            x = x + pos_embed[s["pos_offset"] + np.arange(T)]
        if embed_norm is not None:
            x = norm(x, embed_norm)
        hidden = [x.copy()]
        cos, sin = rope_tables(T)
        cos_l, sin_l = rope_tables_local(T) if s["local_rope"] else (cos, sin)
        ple = per_layer_inputs(tokens, x) if ple_dim else None
        shared_kv.clear()
        streams = None
        if altup:
            target = rms_magnitude(x)
            extra = []
            for pw in altup_proj:
                y = x @ pw.T
                extra.append(y * target / np.sqrt(np.maximum(np.mean(y * y, -1, keepdims=True), 1e-5)))
            streams = np.stack([x] + extra)
        for li, d in enumerate(layers):
            c_, s_ = (cos_l, sin_l) if sliding_layers[li] else (cos, sin)

            def h_fn(h, d=d, li=li, c_=c_, s_=s_):
                if conv_layers[li]:
                    return conv_block(d, h)
                return linear_attn(d, h) if lin_layers[li] else attention(d, li, h, c_, s_)
            ple_li = ple[:, li] if ple is not None else None
            if altup:
                streams = layer_altup(d, li, streams, h_fn, ple_li)
                x = streams[0]
            else:
                x = layer_std(d, li, x, h_fn, ple_li)
            hidden.append(x.copy())
        if altup:
            target = rms_magnitude(streams[0])
            merged = [streams[0]]
            for e, uw in enumerate(altup_unembed):
                y = streams[e + 1] @ uw.T
                merged.append(y * target / np.sqrt(np.maximum(np.mean(y * y, -1, keepdims=True), 1e-5)))
            x = np.mean(np.stack(merged), 0)
        hfin = norm(x, final_norm) if s["norm"] != "none" else norm(x, None)
        logits = hfin @ lm_head.T
        if lm_bias is not None:
            logits = logits + lm_bias
        logits = logits * np.float32(s["logit_scale"])
        if s["final_softcap"]:
            logits = s["final_softcap"] * np.tanh(logits / s["final_softcap"])
        return logits, hidden

    cases = []
    for t in ["the ant or you", "hello 42"]:
        ids = encode(t)
        if bos:
            ids = [vocab[bos]] + ids
        logits, hidden = forward(ids)
        cases.append({"text": t, "ids": ids, "last_logits": [round(float(v), 4) for v in logits[-1]], "argmax": int(np.argmax(logits[-1])),
                      "last_hidden": [[round(float(v), 4) for v in hh[-1]] for hh in hidden]})
    json.dump({"family": family, "cases": cases}, open(f"{out_dir}/reference.json", "w"), separators=(",", ":"))
    print(f"wrote fixture to {out_dir}: vocab={V}, layers={L}, hidden={H}")


if FAMILY in SPECS:
    generate_generic(FAMILY, OUT)
    sys.exit(0)


# ---------------------------------------------------------------------------
# Tokenizer: byte-level BPE (Qwen/Llama-3 style) or SentencePiece style.
# ---------------------------------------------------------------------------

def bytes_to_unicode():
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))

B2U = bytes_to_unicode()

MOE = FAMILY.startswith("qwen3_moe")
BIG = FAMILY == "qwen3_moe_big"
FUSED = FAMILY in ("qwen3_moe_fused", "qwen3_moe_fused_t")
FUSED_T = FAMILY == "qwen3_moe_fused_t"
byte_level = FAMILY.startswith("qwen") or FAMILY == "llama"
if byte_level:
    specials = ["<|im_start|>", "<|im_end|>", "<|endoftext|>"] if FAMILY.startswith("qwen") else [
        "<|begin_of_text|>", "<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>", "<|end_of_text|>"]
    vocab = {}
    for b in range(256):
        vocab[B2U[b]] = len(vocab)
    # A few merges so that BPE has something to do.
    merge_pairs = [("Ġ", "t"), ("h", "e"), ("Ġt", "he"), ("i", "n"), ("a", "n"), ("o", "r"), ("Ġ", "a"), ("Ġa", "n"), ("e", "r"), ("Ġ", "y"), ("Ġy", "o"), ("Ġyo", "u")]
    merges = []
    for a, b in merge_pairs:
        merges.append(f"{a} {b}")
        vocab[a + b] = len(vocab)
    added = []
    for s in specials:
        added.append({"id": len(vocab), "content": s, "single_word": False, "lstrip": False, "rstrip": False, "normalized": False, "special": True})
        vocab[s] = len(vocab)
    regex = (r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
             if FAMILY == "llama" else
             r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+")
    tokenizer = {
        "version": "1.0",
        "added_tokens": added,
        "normalizer": None,
        "pre_tokenizer": {"type": "Sequence", "pretokenizers": [
            {"type": "Split", "pattern": {"Regex": regex}, "behavior": "Isolated", "invert": False},
            {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": False, "use_regex": False}]},
        "post_processor": None if FAMILY.startswith("qwen") else {
            "type": "TemplateProcessing",
            "single": [{"SpecialToken": {"id": "<|begin_of_text|>", "type_id": 0}}, {"Sequence": {"id": "A", "type_id": 0}}],
            "pair": [], "special_tokens": {"<|begin_of_text|>": {"id": "<|begin_of_text|>", "ids": [vocab["<|begin_of_text|>"]], "tokens": ["<|begin_of_text|>"]}}},
        "decoder": {"type": "ByteLevel", "add_prefix_space": True, "trim_offsets": True, "use_regex": True},
        "model": {"type": "BPE", "dropout": None, "unk_token": None, "continuing_subword_prefix": None,
                  "end_of_word_suffix": None, "fuse_unk": False, "byte_fallback": False,
                  "ignore_merges": FAMILY == "llama", "vocab": vocab, "merges": merges},
    }
    if FAMILY.startswith("qwen"):
        bos, eos = None, "<|im_end|>"
        chat_template = ("{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}"
                         "{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}")
    else:
        bos, eos = "<|begin_of_text|>", "<|eot_id|>"
        chat_template = ("{{ bos_token }}{% for message in messages %}{{ '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n' + message['content'] | trim + '<|eot_id|>' }}"
                         "{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% endif %}")
else:
    # SentencePiece-style (Gemma).
    specials = ["<pad>", "<eos>", "<bos>", "<unk>", "<start_of_turn>", "<end_of_turn>"]
    vocab = {}
    for s in specials:
        vocab[s] = len(vocab)
    for b in range(256):
        vocab[f"<0x{b:02X}>"] = len(vocab)
    chars = ["▁"] + [chr(c) for c in range(32, 127)]
    for ch in chars:
        if ch not in vocab:
            vocab[ch] = len(vocab)
    merge_pairs = [("▁", "t"), ("h", "e"), ("▁t", "he"), ("i", "n"), ("a", "n"), ("o", "r"), ("▁", "a"), ("▁a", "n"), ("e", "r"), ("▁", "y"), ("▁y", "o"), ("▁yo", "u")]
    merges = []
    for a, b in merge_pairs:
        merges.append(f"{a} {b}")
        vocab[a + b] = len(vocab)
    added = [{"id": vocab[s], "content": s, "single_word": False, "lstrip": False, "rstrip": False, "normalized": False, "special": True} for s in specials]
    tokenizer = {
        "version": "1.0",
        "added_tokens": added,
        "normalizer": {"type": "Replace", "pattern": {"String": " "}, "content": "▁"},
        "pre_tokenizer": None,
        "post_processor": {"type": "TemplateProcessing",
                           "single": [{"SpecialToken": {"id": "<bos>", "type_id": 0}}, {"Sequence": {"id": "A", "type_id": 0}}],
                           "pair": [], "special_tokens": {"<bos>": {"id": "<bos>", "ids": [vocab["<bos>"]], "tokens": ["<bos>"]}}},
        "decoder": {"type": "Sequence", "decoders": [
            {"type": "Replace", "pattern": {"String": "▁"}, "content": " "},
            {"type": "ByteFallback"}, {"type": "Fuse"}]},
        "model": {"type": "BPE", "dropout": None, "unk_token": "<unk>", "continuing_subword_prefix": None,
                  "end_of_word_suffix": None, "fuse_unk": True, "byte_fallback": True, "ignore_merges": False,
                  "vocab": vocab, "merges": merges},
    }
    bos, eos = "<bos>", "<eos>"
    chat_template = ("{{ bos_token }}{% for message in messages %}{{ '<start_of_turn>' + message['role'] + '\n' + message['content'] | trim + '<end_of_turn>\n' }}"
                     "{% endfor %}{% if add_generation_prompt %}{{'<start_of_turn>model\n'}}{% endif %}")

json.dump(tokenizer, open(f"{OUT}/tokenizer.json", "w"), ensure_ascii=False)
tok_cfg = {"chat_template": chat_template, "eos_token": eos, "add_bos_token": bos is not None, "tokenizer_class": "PreTrainedTokenizerFast"}
if bos:
    tok_cfg["bos_token"] = bos
json.dump(tok_cfg, open(f"{OUT}/tokenizer_config.json", "w"))
vocab_size = len(vocab)

# ---------------------------------------------------------------------------
# Model config and weights
# ---------------------------------------------------------------------------

H, I, L, NH, NKV = 32, (64 if WRITE_GGUF else 48), 3, 4, 2
HD = H // NH
# MoE: E routed experts of size MI, top-K routing; layer 1 stays dense (mlp_only_layers).
E, K, MI = 4, 2, 12
MLP_ONLY = [1]
if BIG:
    # Every layer routed: 16 experts of 9 KB each, 64 experts in total, so a
    # cache of a few experts sees evictions on every forward pass.
    H, I, L, E, K, MI = 64, 96, 4, 16, 2, 24
    HD = H // NH
    MLP_ONLY = []
if FAMILY == "gemma3":
    HD = 16  # gemma uses an explicit head_dim
CFG_FAMILY = "qwen3_moe" if MOE else FAMILY
config = {
    "model_type": {"llama": "llama", "qwen2": "qwen2", "qwen3": "qwen3", "gemma3": "gemma3_text", "qwen3_moe": "qwen3_moe"}[CFG_FAMILY],
    "architectures": [{"llama": "LlamaForCausalLM", "qwen2": "Qwen2ForCausalLM", "qwen3": "Qwen3ForCausalLM", "gemma3": "Gemma3ForCausalLM", "qwen3_moe": "Qwen3MoeForCausalLM"}[CFG_FAMILY]],
    "hidden_size": H, "intermediate_size": I, "num_hidden_layers": L, "num_attention_heads": NH,
    "num_key_value_heads": NKV, "head_dim": HD, "vocab_size": vocab_size, "rms_norm_eps": 1e-6,
    "rope_theta": 10000.0, "max_position_embeddings": 512, "tie_word_embeddings": FAMILY != "llama",
    "torch_dtype": "bfloat16",
}
if FAMILY == "llama":
    config["rope_scaling"] = {"rope_type": "llama3", "factor": 8.0, "low_freq_factor": 1.0, "high_freq_factor": 4.0, "original_max_position_embeddings": 64}
    config["hidden_act"] = "silu"
if FAMILY == "qwen2":
    config["attention_bias"] = True
    config["hidden_act"] = "silu"
if FAMILY == "qwen3":
    config["hidden_act"] = "silu"
if MOE:
    config.update({"hidden_act": "silu", "num_experts": E, "num_experts_per_tok": K, "norm_topk_prob": True,
                   "moe_intermediate_size": MI, "decoder_sparse_step": 1, "mlp_only_layers": MLP_ONLY})
if FAMILY == "gemma3":
    config.update({"hidden_activation": "gelu_pytorch_tanh", "query_pre_attn_scalar": HD, "sliding_window": 8,
                   "sliding_window_pattern": 2, "rope_local_base_freq": 10000.0, "rope_theta": 1000000.0,
                   "rope_scaling": {"rope_type": "linear", "factor": 8.0}, "final_logit_softcapping": None,
                   "attn_logit_softcapping": None})
json.dump(config, open(f"{OUT}/config.json", "w"), indent=2)
json.dump({"eos_token_id": vocab[eos], "bos_token_id": vocab[bos] if bos else None, "do_sample": False}, open(f"{OUT}/generation_config.json", "w"))


def bf16(a):
    """Rounds an f32 array to bf16 precision (round-to-nearest-even) and returns the u16 view."""
    a = np.ascontiguousarray(a, dtype=np.float32)
    u = a.view(np.uint32)
    lsb = (u >> 16) & 1
    rounded = u + 0x7FFF + lsb
    return (rounded >> 16).astype(np.uint16)


def from_bf16(u16):
    return (u16.astype(np.uint32) << 16).view(np.float32)


weights = {}


def W(name, shape, scale=0.2):
    w = rng.normal(0, scale, size=shape).astype(np.float32)
    u = bf16(w)
    weights[name] = u
    return from_bf16(u).reshape(shape)


def rand_bf16(shape, scale=0.2):
    """Random bf16-rounded weights that are not registered for writing (returns (u16, f32))."""
    w = rng.normal(0, scale, size=shape).astype(np.float32)
    u = bf16(w)
    return u, from_bf16(u).reshape(shape)


def ones(name, shape, scale=0.1):
    w = (1.0 + rng.normal(0, scale, size=shape)).astype(np.float32)
    if FAMILY == "gemma3":
        w = w - 1.0  # gemma stores (w - 1)
    u = bf16(w)
    weights[name] = u
    return from_bf16(u).reshape(shape)


embed = W("model.embed_tokens.weight", (vocab_size, H), 1.0)
final_norm = ones("model.norm.weight", (H,))
layers = []
for i in range(L):
    p = f"model.layers.{i}."
    d = {
        "in_norm": ones(p + "input_layernorm.weight", (H,)),
        "post_attn_norm": ones(p + "post_attention_layernorm.weight", (H,)),
        "q": W(p + "self_attn.q_proj.weight", (NH * HD, H)),
        "k": W(p + "self_attn.k_proj.weight", (NKV * HD, H)),
        "v": W(p + "self_attn.v_proj.weight", (NKV * HD, H)),
        "o": W(p + "self_attn.o_proj.weight", (H, NH * HD)),
    }
    if MOE and i not in MLP_ONLY:
        d["router"] = W(p + "mlp.gate.weight", (E, H))
        experts = []
        for e in range(E):
            gu, gf = rand_bf16((MI, H))
            uu, uf = rand_bf16((MI, H))
            du, df = rand_bf16((H, MI))
            experts.append({"gate": gf, "up": uf, "down": df})
            if not FUSED:
                weights[p + f"mlp.experts.{e}.gate_proj.weight"] = gu
                weights[p + f"mlp.experts.{e}.up_proj.weight"] = uu
                weights[p + f"mlp.experts.{e}.down_proj.weight"] = du
            else:
                gate_up = np.concatenate([gu, uu], axis=0)  # [2I, H]
                if FUSED_T:
                    gate_up, du = gate_up.T, du.T  # [H, 2I], [I, H]
                weights.setdefault(p + "mlp.experts.gate_up_proj", []).append(np.ascontiguousarray(gate_up))
                weights.setdefault(p + "mlp.experts.down_proj", []).append(np.ascontiguousarray(du))
        if FUSED:
            weights[p + "mlp.experts.gate_up_proj"] = np.stack(weights[p + "mlp.experts.gate_up_proj"])
            weights[p + "mlp.experts.down_proj"] = np.stack(weights[p + "mlp.experts.down_proj"])
        d["experts"] = experts
    else:
        d["gate"] = W(p + "mlp.gate_proj.weight", (I, H))
        d["up"] = W(p + "mlp.up_proj.weight", (I, H))
        d["down"] = W(p + "mlp.down_proj.weight", (H, I))
    if FAMILY == "qwen2":
        d["qb"] = W(p + "self_attn.q_proj.bias", (NH * HD,))
        d["kb"] = W(p + "self_attn.k_proj.bias", (NKV * HD,))
        d["vb"] = W(p + "self_attn.v_proj.bias", (NKV * HD,))
    if FAMILY.startswith("qwen3"):
        d["qn"] = ones(p + "self_attn.q_norm.weight", (HD,))
        d["kn"] = ones(p + "self_attn.k_norm.weight", (HD,))
    if FAMILY == "gemma3":
        d["pre_ff"] = ones(p + "pre_feedforward_layernorm.weight", (H,))
        d["post_ff"] = ones(p + "post_feedforward_layernorm.weight", (H,))
    layers.append(d)
lm_head = embed if config["tie_word_embeddings"] else W("lm_head.weight", (vocab_size, H), 1.0)

# Write safetensors (bf16) as two shards to exercise the index handling.
names = sorted(weights)
shards = [names[: len(names) // 2], names[len(names) // 2:]]
index = {"metadata": {"total_size": 0}, "weight_map": {}}
for si, shard in enumerate(shards):
    fname = f"model-{si + 1:05d}-of-{len(shards):05d}.safetensors"
    header = {}
    offset = 0
    blobs = []
    for n in shard:
        u = weights[n]
        nbytes = u.nbytes
        shape = list(u.shape)
        header[n] = {"dtype": "BF16", "shape": shape, "data_offsets": [offset, offset + nbytes]}
        blobs.append(u.tobytes())
        offset += nbytes
        index["weight_map"][n] = fname
        index["metadata"]["total_size"] += nbytes
    hb = json.dumps(header).encode()
    hb += b" " * ((8 - len(hb) % 8) % 8)
    with open(f"{OUT}/{fname}", "wb") as f:
        f.write(struct.pack("<Q", len(hb)))
        f.write(hb)
        for b in blobs:
            f.write(b)
json.dump(index, open(f"{OUT}/model.safetensors.index.json", "w"), indent=2)

# ---------------------------------------------------------------------------
# Reference forward pass (float32)
# ---------------------------------------------------------------------------

gemma = FAMILY == "gemma3"


def rmsnorm(x, w, eps=1e-6):
    v = x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)
    return v * ((1.0 + w) if gemma else w)


def rope_tables(theta, scaling, T):
    inv = 1.0 / (theta ** (np.arange(0, HD, 2, dtype=np.float64) / HD))
    if scaling and scaling.get("rope_type") == "linear":
        inv = inv / scaling["factor"]
    if scaling and scaling.get("rope_type") == "llama3":
        f, lo, hi, om = scaling["factor"], scaling["low_freq_factor"], scaling["high_freq_factor"], scaling["original_max_position_embeddings"]
        low_wl, high_wl = om / lo, om / hi
        out = []
        for x in inv:
            wl = 2 * np.pi / x
            if wl < high_wl:
                out.append(x)
            elif wl > low_wl:
                out.append(x / f)
            else:
                s = (om / wl - lo) / (hi - lo)
                out.append((1 - s) * x / f + s * x)
        inv = np.array(out)
    pos = np.arange(T, dtype=np.float64)
    ang = np.outer(pos, inv)
    return np.cos(ang).astype(np.float32), np.sin(ang).astype(np.float32)


def rope(x, cos, sin):  # x: [T, nh, HD]
    half = HD // 2
    x1, x2 = x[..., :half], x[..., half:]
    c = cos[:, None, :]
    s = sin[:, None, :]
    return np.concatenate([x1 * c - x2 * s, x2 * c + x1 * s], axis=-1)


def act(x):
    if gemma:
        return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))
    return x / (1 + np.exp(-x))


def forward(tokens):
    T = len(tokens)
    x = embed[tokens].astype(np.float32)
    if gemma:
        x = x * np.float32(np.sqrt(H))
    hidden = [x.copy()]
    cos_g, sin_g = rope_tables(config["rope_theta"], config.get("rope_scaling"), T)
    cos_l, sin_l = rope_tables(config.get("rope_local_base_freq", 10000.0), None, T)
    scale = 1.0 / np.sqrt(config.get("query_pre_attn_scalar", HD))
    for li, d in enumerate(layers):
        sliding = gemma and ((li + 1) % config["sliding_window_pattern"] != 0)
        cos, sin = (cos_l, sin_l) if sliding else (cos_g, sin_g)
        h = rmsnorm(x, d["in_norm"])
        q = h @ d["q"].T
        k = h @ d["k"].T
        v = h @ d["v"].T
        if "qb" in d:
            q, k, v = q + d["qb"], k + d["kb"], v + d["vb"]
        q = q.reshape(T, NH, HD)
        k = k.reshape(T, NKV, HD)
        v = v.reshape(T, NKV, HD)
        if "qn" in d:
            q = q / np.sqrt(np.mean(q * q, axis=-1, keepdims=True) + 1e-6) * d["qn"]
            k = k / np.sqrt(np.mean(k * k, axis=-1, keepdims=True) + 1e-6) * d["kn"]
        q = rope(q, cos, sin)
        k = rope(k, cos, sin)
        groups = NH // NKV
        out = np.zeros((T, NH, HD), dtype=np.float32)
        for hh in range(NH):
            kv = hh // groups
            s = (q[:, hh, :] @ k[:, kv, :].T) * scale
            mask = np.triu(np.ones((T, T), dtype=bool), k=1)
            if sliding:
                w = config["sliding_window"]
                mask |= np.tril(np.ones((T, T), dtype=bool), k=-w)
            s = np.where(mask, -np.inf, s)
            s = s - s.max(axis=-1, keepdims=True)
            p = np.exp(s)
            p = p / p.sum(axis=-1, keepdims=True)
            out[:, hh, :] = p @ v[:, kv, :]
        o = out.reshape(T, NH * HD) @ d["o"].T
        if gemma:
            o = rmsnorm(o, d["post_attn_norm"])
        x = x + o
        h = rmsnorm(x, d["pre_ff"] if gemma else d["post_attn_norm"])
        if "router" in d:
            # Softmax over all experts, top-K, renormalise (norm_topk_prob), weighted expert sum.
            logits = h @ d["router"].T
            probs = np.exp(logits - logits.max(axis=-1, keepdims=True))
            probs = probs / probs.sum(axis=-1, keepdims=True)
            m = np.zeros((T, H), dtype=np.float32)
            for t in range(T):
                idx = np.argsort(-probs[t], kind="stable")[:K]
                w = probs[t][idx]
                w = w / w.sum()
                for e, we in zip(idx, w):
                    ex = d["experts"][e]
                    m[t] += we * ((act(h[t] @ ex["gate"].T) * (h[t] @ ex["up"].T)) @ ex["down"].T)
        else:
            m = (act(h @ d["gate"].T) * (h @ d["up"].T)) @ d["down"].T
        if gemma:
            m = rmsnorm(m, d["post_ff"])
        x = x + m
        hidden.append(x.copy())
    logits = rmsnorm(x, final_norm) @ lm_head.T
    return logits, hidden


# Reference prompts: tokenised with a tiny reimplementation of the BPE so the
# Zig side can check both tokenizer and model. Token ids are given explicitly.
def encode_bytelevel(text):
    # Only used for ASCII fixture text; mirrors the regex roughly by splitting on spaces (keeping the space).
    import re
    pieces = re.findall(r" ?[A-Za-z]+|\d| ?[^\sA-Za-z\d]+|\s+", text)
    ids = []
    ranks = {m: i for i, m in enumerate(merges)}
    for piece in pieces:
        syms = [B2U[b] for b in piece.encode()]
        if config["model_type"] == "llama" and "".join(syms) in vocab:
            ids.append(vocab["".join(syms)])
            continue
        while len(syms) > 1:
            best, bi = None, None
            for i in range(len(syms) - 1):
                r = ranks.get(f"{syms[i]} {syms[i + 1]}")
                if r is not None and (best is None or r < best):
                    best, bi = r, i
            if best is None:
                break
            syms[bi:bi + 2] = [syms[bi] + syms[bi + 1]]
        ids.extend(vocab[s] for s in syms)
    return ids


def encode_spm(text):
    text = text.replace(" ", "▁")
    syms = list(text)
    ranks = {m: i for i, m in enumerate(merges)}
    while len(syms) > 1:
        best, bi = None, None
        for i in range(len(syms) - 1):
            r = ranks.get(f"{syms[i]} {syms[i + 1]}")
            if r is not None and (best is None or r < best):
                best, bi = r, i
        if best is None:
            break
        syms[bi:bi + 2] = [syms[bi] + syms[bi + 1]]
    ids = []
    for s in syms:
        if s in vocab:
            ids.append(vocab[s])
        else:
            ids.extend(vocab[f"<0x{b:02X}>"] for b in s.encode())
    return ids


texts = ["the ant or you", "an era in the", "hello"]


def references():
    cases = []
    for t in texts:
        ids = encode_bytelevel(t) if byte_level else encode_spm(t)
        if bos:
            ids = [vocab[bos]] + ids
        logits, hidden = forward(ids)
        cases.append({
            "text": t,
            "ids": ids,
            "last_logits": logits[-1].tolist(),
            "argmax": int(np.argmax(logits[-1])),
            # residual stream at the last position for every layer entry
            "last_hidden": [h[-1].tolist() for h in hidden],
        })
    return cases


json.dump({"family": FAMILY, "cases": references()}, open(f"{OUT}/reference.json", "w"))
print(f"wrote fixture to {OUT}: vocab={vocab_size}, layers={L}, hidden={H}")

# ---------------------------------------------------------------------------
# GGUF variant (--gguf): the same model as llama.cpp would store it.
# ---------------------------------------------------------------------------

if WRITE_GGUF:
    if MOE or FAMILY == "gemma3":
        sys.exit("--gguf is implemented for the llama and qwen families only")
    os.makedirs(GOUT, exist_ok=True)

    def f16(a):
        """Rounds to f16 and returns (raw u16, f32 values)."""
        h = np.ascontiguousarray(a, dtype=np.float32).astype(np.float16)
        return h.view(np.uint16), h.astype(np.float32)

    def q8_0(a):
        """ggml Q8_0: blocks of 32 with an f16 scale d = amax / 127 and int8 quants.
        Returns (block bytes, dequantised f32 values)."""
        x = np.ascontiguousarray(a, dtype=np.float32)
        rows, cols = x.shape
        assert cols % 32 == 0
        blocks = x.reshape(rows, cols // 32, 32)
        amax = np.abs(blocks).max(axis=-1)
        d = (amax / 127.0).astype(np.float32)
        d16 = d.astype(np.float16)
        inv = np.where(d != 0, 1.0 / np.where(d != 0, d, 1.0), 0.0).astype(np.float32)
        v = blocks * inv[..., None]
        q = (np.sign(v) * np.floor(np.abs(v) + 0.5)).astype(np.int8)  # roundf: half away from zero
        deq = (q.astype(np.float32) * d16.astype(np.float32)[..., None]).reshape(rows, cols)
        out = bytearray()
        for r in range(rows):
            for b in range(cols // 32):
                out += d16[r, b].tobytes() + q[r, b].tobytes()
        return bytes(out), deq

    def permute(w, n_head):
        """llama.cpp's q/k permutation: per head, rows [2][hd/2] -> [hd/2][2]."""
        hd = w.shape[0] // n_head
        return w.reshape(n_head, 2, hd // 2, *w.shape[1:]).swapaxes(1, 2).reshape(w.shape)

    F32, F16, Q8_0 = 0, 1, 8
    gtensors = []  # (name, hf_shape, ggml_type, bytes)

    def add(name, arr_or_bytes, gtype, shape):
        data = arr_or_bytes if isinstance(arr_or_bytes, (bytes, bytearray)) else np.ascontiguousarray(arr_or_bytes).tobytes()
        gtensors.append((name, list(shape), gtype, data))

    # Embeddings (f16, tied when the config says so) and norms (f32).
    eu, ef = f16(embed)
    embed = ef
    add("token_embd.weight", eu, F16, embed.shape)
    add("output_norm.weight", final_norm.astype(np.float32), F32, final_norm.shape)
    if not config["tie_word_embeddings"]:
        lu, lf = f16(lm_head)
        lm_head = lf
        add("output.weight", lu, F16, lm_head.shape)
    else:
        lm_head = embed
    if FAMILY == "llama":
        # llama.cpp stores the llama3 rope scaling as per-frequency factors.
        rs = config["rope_scaling"]
        inv = 1.0 / (config["rope_theta"] ** (np.arange(0, HD, 2, dtype=np.float32) / HD))
        low_wl, high_wl = rs["original_max_position_embeddings"] / rs["low_freq_factor"], rs["original_max_position_embeddings"] / rs["high_freq_factor"]
        factors = []
        for fr in inv:
            wl = 2 * np.pi / fr
            if wl < high_wl:
                factors.append(1.0)
            elif wl > low_wl:
                factors.append(rs["factor"])
            else:
                sm = (rs["original_max_position_embeddings"] / wl - rs["low_freq_factor"]) / (rs["high_freq_factor"] - rs["low_freq_factor"])
                factors.append(1.0 / ((1 - sm) / rs["factor"] + sm))
        add("rope_freqs.weight", np.array(factors, dtype=np.float32), F32, [len(factors)])
    for i, d in enumerate(layers):
        b = f"blk.{i}."
        add(b + "attn_norm.weight", d["in_norm"].astype(np.float32), F32, (H,))
        add(b + "ffn_norm.weight", d["post_attn_norm"].astype(np.float32), F32, (H,))
        for key, gname, nh in (("q", "attn_q", NH), ("k", "attn_k", NKV), ("v", "attn_v", None), ("o", "attn_output", None)):
            u, f = f16(d[key])
            d[key] = f
            stored = permute(u, nh) if (FAMILY == "llama" and nh) else u
            add(b + gname + ".weight", stored, F16, d[key].shape)
        if "qb" in d:
            for key, gname, nh in (("qb", "attn_q", NH), ("kb", "attn_k", NKV), ("vb", "attn_v", None)):
                v = d[key].astype(np.float32)
                add(b + gname + ".bias", permute(v, nh) if (FAMILY == "llama" and nh) else v, F32, v.shape)
        if "qn" in d:
            add(b + "attn_q_norm.weight", d["qn"].astype(np.float32), F32, (HD,))
            add(b + "attn_k_norm.weight", d["kn"].astype(np.float32), F32, (HD,))
        for key, gname in (("gate", "ffn_gate"), ("up", "ffn_up"), ("down", "ffn_down")):
            blocks, deq = q8_0(d[key])
            d[key] = deq
            add(b + gname + ".weight", blocks, Q8_0, deq.shape)

    # Vocabulary as llama.cpp stores it (gpt2 model, byte-level tokens).
    arch = "llama" if FAMILY == "llama" else FAMILY
    id_to_tok = sorted(vocab.items(), key=lambda kv: kv[1])
    assert [i for _, i in id_to_tok] == list(range(len(id_to_tok)))
    tokens = [t for t, _ in id_to_tok]
    special = set(specials)
    types = [3 if t in special else 1 for t in tokens]
    kv = [
        ("general.architecture", "str", arch),
        ("general.type", "str", "model"),
        ("general.name", "str", f"{FAMILY} fixture"),
        ("general.quantization_version", "u32", 2),
        ("general.file_type", "u32", 7),  # MOSTLY_Q8_0
        (f"{arch}.context_length", "u32", config["max_position_embeddings"]),
        (f"{arch}.embedding_length", "u32", H),
        (f"{arch}.block_count", "u32", L),
        (f"{arch}.feed_forward_length", "u32", I),
        (f"{arch}.attention.head_count", "u32", NH),
        (f"{arch}.attention.head_count_kv", "u32", NKV),
        (f"{arch}.attention.layer_norm_rms_epsilon", "f32", config["rms_norm_eps"]),
        (f"{arch}.attention.key_length", "u32", HD),
        (f"{arch}.attention.value_length", "u32", HD),
        (f"{arch}.rope.dimension_count", "u32", HD),
        (f"{arch}.rope.freq_base", "f32", config["rope_theta"]),
        (f"{arch}.vocab_size", "u32", len(tokens)),
        ("tokenizer.ggml.model", "str", "gpt2"),
        ("tokenizer.ggml.pre", "str", "llama-bpe" if FAMILY == "llama" else "qwen2"),
        ("tokenizer.ggml.tokens", "[str]", tokens),
        ("tokenizer.ggml.token_type", "[i32]", types),
        ("tokenizer.ggml.merges", "[str]", merges),
        ("tokenizer.ggml.eos_token_id", "u32", vocab[eos]),
        ("tokenizer.ggml.padding_token_id", "u32", vocab[eos]),
        ("tokenizer.ggml.add_bos_token", "bool", bos is not None),
        ("tokenizer.chat_template", "str", chat_template),
    ]
    if bos:
        kv.append(("tokenizer.ggml.bos_token_id", "u32", vocab[bos]))

    def gstr(s):
        b = s.encode("utf-8")
        return struct.pack("<Q", len(b)) + b

    TYPES = {"u32": (4, lambda v: struct.pack("<I", v)), "i32": (5, lambda v: struct.pack("<i", v)),
             "f32": (6, lambda v: struct.pack("<f", v)), "bool": (7, lambda v: struct.pack("<B", 1 if v else 0)),
             "str": (8, gstr)}

    def gvalue(t, v):
        if t.startswith("["):
            et, enc = TYPES[t[1:-1]]
            return struct.pack("<I", 9) + struct.pack("<I", et) + struct.pack("<Q", len(v)) + b"".join(enc(x) for x in v)
        et, enc = TYPES[t]
        return struct.pack("<I", et) + enc(v)

    ALIGN = 32
    header = b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", len(gtensors)) + struct.pack("<Q", len(kv))
    for key, t, v in kv:
        header += gstr(key) + gvalue(t, v)
    offset = 0
    offsets = []
    for name, shape, gtype, data in gtensors:
        header += gstr(name) + struct.pack("<I", len(shape))
        for dim in reversed(shape):
            header += struct.pack("<Q", dim)
        header += struct.pack("<I", gtype) + struct.pack("<Q", offset)
        offsets.append(offset)
        offset += (len(data) + ALIGN - 1) // ALIGN * ALIGN
    header += b"\0" * ((ALIGN - len(header) % ALIGN) % ALIGN)
    with open(f"{GOUT}/model.gguf", "wb") as f:
        f.write(header)
        for name, shape, gtype, data in gtensors:
            f.write(data)
            f.write(b"\0" * ((ALIGN - len(data) % ALIGN) % ALIGN))
    json.dump({"family": FAMILY + "_gguf", "cases": references()}, open(f"{GOUT}/reference.json", "w"))
    print(f"wrote GGUF fixture to {GOUT}: {len(gtensors)} tensors, {len(header) + offset} bytes")
