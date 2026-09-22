#!/usr/bin/env python3
"""Generates tiny synthetic Hugging Face-format models plus a NumPy reference
forward pass, used to validate ditch's inference against known-good numbers.

Usage: make_fixture.py <family> [<out_dir>] [--gguf]
    family: llama | qwen2 | qwen3 | gemma3 | qwen3_moe | qwen3_moe_fused | qwen3_moe_fused_t
            or any family of the registry-driven generator (`SPECS` below:
            phi3, phi, gpt_neox, gpt2, falcon, ... , gpt_oss, deepseek_v3, kimi_linear,
            the Mamba families mamba2, nemotron_h, falcon_h1, jamba, granitemoehybrid
            and their variants, and the quantised variants qwen2_fp8, qwen2_int4,
            gpt_oss_mxfp4)
            | deepseek_v4 | deepseek_v41 | qwen4_exp | glm5_next (hyper-connection
              families, own generators)
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


# ---------------------------------------------------------------------------
# Quantised checkpoint variants. A spec with `quant` stores its linear weights
# (or, for MXFP4, its fused expert tensors) in one of the quantised safetensors
# layouts ditch dequantises on load; the reference forward pass runs on the
# dequantised (bf16-rounded) values, exactly what the loader must produce.
#   fp8:   DeepSeek V3 / Kimi K2 style `weight` F8_E4M3 + `weight_scale_inv` F32
#          with block scales (`weight_block_size` in quantization_config).
#   int:   compressed-tensors pack-quantized: `weight_packed` I32 (num_bits-wide
#          fields packed densely, little end first), `weight_scale` BF16 per
#          group, `weight_zero_point` (asymmetric, packed along the rows) and
#          `weight_shape` I64.
#   mxfp4_store: MiMo V2.6's `store_dtype = mxfp4` experts: the mxfp4_packed
#          bytes under the plain `weight` / `weight_scale` names.
#   mxfp4: gpt-oss `*_blocks` U8 (E2M1 nibble pairs, low nibble first) and
#          `*_scales` U8 (E8M0, bias 127) per 32 elements of the natural
#          [E, out, in] layout; the bf16 tensor is its [E, in, out] transpose.
# ---------------------------------------------------------------------------

E2M1_VALUES = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6], np.float32)


def fp8_e4m3_table():
    """Value of every F8_E4M3 code (OCP fn variant: no infinities, 0x7f/0xff are NaN)."""
    codes = np.arange(256, dtype=np.uint32)
    sign, exp, man = codes >> 7, (codes >> 3) & 0xF, (codes & 7).astype(np.float64)
    val = np.where(exp == 0, man / 8 * 2.0 ** -6, (1 + man / 8) * 2.0 ** (exp.astype(np.float64) - 7))
    val = np.where((exp == 15) & (man == 7), np.nan, val)
    return np.where(sign == 1, -val, val).astype(np.float32)


def nearest_code(x, table):
    """Index of the table entry nearest to every element of `x` (NaN entries never match)."""
    t = np.where(np.isnan(table), np.inf, table)
    return np.abs(np.asarray(x, np.float32)[..., None] - t).argmin(-1)


def quant_fp8_block(name, w, block):
    """DeepSeek-style FP8: one f32 scale per `block` (rows, cols) tile, `w = code * scale`."""
    table = fp8_e4m3_table()
    br, bc = block
    out_, in_ = w.shape
    nb0, nb1 = -(-out_ // br), -(-in_ // bc)
    scale = np.zeros((nb0, nb1), np.float32)
    codes = np.zeros(w.shape, np.uint8)
    deq = np.zeros(w.shape, np.float32)
    for i in range(nb0):
        for j in range(nb1):
            blk = w[i * br:(i + 1) * br, j * bc:(j + 1) * bc]
            s = np.float32(max(float(np.abs(blk).max()), 1e-12) / 448.0)
            scale[i, j] = s
            c = nearest_code(blk / s, table)
            codes[i * br:(i + 1) * br, j * bc:(j + 1) * bc] = c
            deq[i * br:(i + 1) * br, j * bc:(j + 1) * bc] = table[c] * s
    return bf16_round(deq), [(name, "F8_E4M3", codes), (name + "_scale_inv", "F32", scale)]


def pack_bits(vals, bits):
    """compressed-tensors `pack_to_int32` along the last axis: element i occupies bits
    [i * bits, (i + 1) * bits) of the little-endian word stream of its row."""
    vals = np.asarray(vals, np.int64)
    rows, n = vals.shape
    words = -(-n * bits // 32)
    out = np.zeros((rows, words), np.uint64)
    for i in range(n):
        start = i * bits
        w, off = start // 32, start % 32
        out[:, w] |= (vals[:, i].astype(np.uint64) << np.uint64(off)) & np.uint64(0xFFFFFFFF)
        if off + bits > 32:
            out[:, w + 1] |= vals[:, i].astype(np.uint64) >> np.uint64(32 - off)
    return out.astype(np.uint32).view(np.int32)


def quant_int_packed(name, w, bits, group, symmetric, g_idx=None):
    """compressed-tensors pack-quantized INT weights with per-group scales.

    `g_idx` (compressed-tensors `actorder`) names each column's scale group
    instead of the contiguous `column // group`; it is written out as
    `weight_g_idx` and the columns of a group are then scattered."""
    out_, in_ = w.shape
    assert in_ % group == 0
    groups = in_ // group
    actorder = g_idx is not None
    if g_idx is None:
        g_idx = np.arange(in_) // group
    qmin, qmax = -(1 << (bits - 1)), (1 << (bits - 1)) - 1
    scale = np.zeros((out_, groups), np.float32)
    zp = np.zeros((out_, groups), np.int64)
    q = np.zeros((out_, in_), np.int64)
    deq = np.zeros((out_, in_), np.float32)
    for g in range(groups):
        cols = np.nonzero(g_idx == g)[0]
        wg = w[:, cols]
        if symmetric:
            scale[:, g] = bf16_round(np.maximum(np.abs(wg).max(-1), 1e-8) / qmax)
        else:
            mn, mx = np.minimum(wg.min(-1), 0), np.maximum(wg.max(-1), 0)
            scale[:, g] = bf16_round(np.maximum(mx - mn, 1e-8) / (qmax - qmin))
            zp[:, g] = np.clip(np.round(qmin - mn / scale[:, g]), qmin, qmax).astype(np.int64)
        qg = np.clip(np.round(wg / scale[:, g, None]) + zp[:, g, None], qmin, qmax).astype(np.int64)
        q[:, cols] = qg
        deq[:, cols] = bf16_round((qg - zp[:, g, None]).astype(np.float32) * scale[:, g, None])
    offset = 1 << (bits - 1)
    tensors = [(name[:-len(".weight")] + ".weight_packed", "I32", pack_bits(q + offset, bits)),
               (name[:-len(".weight")] + ".weight_scale", "BF16", scale),
               (name[:-len(".weight")] + ".weight_shape", "I64", np.array([out_, in_], np.int64))]
    if actorder:
        tensors.append((name[:-len(".weight")] + ".weight_g_idx", "I32", g_idx.astype(np.int32)))
    if not symmetric:
        # Zero points are packed along the rows (`packed_dim=0`): [ceil(out * bits / 32), groups].
        tensors.append((name[:-len(".weight")] + ".weight_zero_point", "I32", pack_bits((zp + offset).T, bits).T.copy()))
    return deq, tensors


def quant_mxfp4(name, w):
    """gpt-oss MXFP4 of the natural `[E, out, in]` tensor `w` (in a multiple of 32)."""
    assert w.shape[-1] % 32 == 0
    blocks = w.reshape(*w.shape[:-1], w.shape[-1] // 32, 32)
    amax = np.abs(blocks).max(-1)
    e = np.where(amax > 0, np.ceil(np.log2(np.maximum(amax, 1e-30) / 6.0)), 0).astype(np.int64)
    e = np.clip(e, -127, 127)
    codes = nearest_code(blocks / (2.0 ** e)[..., None], E2M1_VALUES).astype(np.uint8)
    packed = (codes[..., 0::2] | (codes[..., 1::2] << 4)).astype(np.uint8)
    deq = (E2M1_VALUES[codes] * (2.0 ** e)[..., None].astype(np.float32)).reshape(w.shape)
    return bf16_round(deq), [(name + "_blocks", "U8", packed), (name + "_scales", "U8", (e + 127).astype(np.uint8))]


def quant_mxfp4_packed(name, w):
    """compressed-tensors `mxfp4-pack-quantized` (Kimi K3 experts) of the natural
    `[out, in]` tensor `w`: E2M1 nibble pairs (low nibble first) in `weight_packed`
    and E8M0 group scales (`floor(log2(amax / 6))`, as the compressor stores them)
    in `weight_scale`."""
    assert w.shape[-1] % 32 == 0
    blocks = w.reshape(*w.shape[:-1], w.shape[-1] // 32, 32)
    amax = np.abs(blocks).max(-1)
    e = np.where(amax > 0, np.floor(np.log2(np.maximum(amax, 1e-30) / 6.0)), 0).astype(np.int64)
    e = np.clip(e, -127, 127)
    codes = nearest_code(np.clip(blocks / (2.0 ** e)[..., None], -6.0, 6.0), E2M1_VALUES).astype(np.uint8)
    packed = (codes[..., 0::2] | (codes[..., 1::2] << 4)).astype(np.uint8).reshape(*w.shape[:-1], w.shape[-1] // 2)
    deq = (E2M1_VALUES[codes] * (2.0 ** e)[..., None].astype(np.float32)).reshape(w.shape)
    module = name[:-len(".weight")]
    return bf16_round(deq), [(module + ".weight_packed", "U8", packed), (module + ".weight_scale", "U8", (e + 127).astype(np.uint8))]


def quant_mxfp4_store(name, w):
    """MiMo V2.6's `store_dtype = mxfp4` experts: the same bytes as
    `mxfp4-pack-quantized` (E2M1 nibble pairs of the natural `[out, in]`
    layout, low nibble first, one E8M0 group scale per 32 columns) under the
    plain `weight` / `weight_scale` names, next to the fp8 dense weights."""
    deq, tensors = quant_mxfp4_packed(name, w)
    module = name[:-len(".weight")]
    renamed = [(module + ".weight", dtype, arr) if n.endswith(".weight_packed") else (n, dtype, arr)
               for n, dtype, arr in tensors]
    return deq, renamed


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
        # `tanh` softcap on the attention logits, applied to the scaled scores
        # before the causal / sliding mask (Gemma 2).
        attn_softcap=None,
        qk_norm=None, q_norm="self_attn.q_norm.weight", k_norm="self_attn.k_norm.weight", clip=None,
        residual_mult=1.0, logit_scale=1.0, embed_scale=1.0, lm_bias=False, sinks=None, temp=None, pos_offset=0,
        sliding=None, sliding_layers=None, mla=None, moe=None, linear=None, linear_layers=None, full_interval=0,
        gated_q=False, gate_swish=False,
        # Mamba blocks: `ssm` describes the block, `ssm_layers` / `attn_layers` /
        # `mlp_layers` which layers hold which block (None: attention everywhere
        # except Mamba / linear layers, an MLP everywhere). `parallel_ssm` runs
        # the Mamba block and attention side by side (Falcon-H1); `mult` holds
        # its muP multipliers.
        ssm=None, ssm_layers=None, attn_layers=None, mlp_layers=None, parallel_ssm=False, mult=None,
        # Gemma 3n / 4 and LFM2 features: per-layer head size / KV heads, KV
        # sharing, keys reused as values, weightless value norm, a local rope
        # table (theta, rotary dim) and a global (rotary dim, freq dim) pair,
        # per-layer inputs, AltUp / Laurel, gate sparsity, conv layers,
        # per-layer output scalars, per-layer FFN widths, final softcapping.
        layer_hd=None, layer_nkv=None, kv_shared=0, k_eq_v=False, v_norm=False, local_rope=None, global_rotary=None,
        ple_dim=0, altup=None, sparsity=None, conv_layers=None, conv_K=3, layer_scale=False, layer_inter=None, final_softcap=None,
        # "gdn" (Gated DeltaNet) or "lightning" (MiniMax) linear-attention layers.
        linear_kind="gdn",
        # "pre" or "minimax" (h = norm(x); x = alpha * h + beta * f(h)); scales per sublayer.
        residual_layout="pre", mm_scales=None,
        # HunYuan applies the per-head q/k norm after the rotary embedding.
        qk_norm_after_rope=False,
        # Sub-layer norms on the attention output and the MLP intermediate (BitNet).
        attn_sub_norm=None, ffn_sub_norm=None,
        # Gate on the attention output from a separate projection (AFMoE, Laguna):
        # (tensor name, "sigmoid" | "softplus", per-head?).
        attn_gate=None,
        # xIELU activation (Apertus): (alpha_p, alpha_n) after softplus; the
        # tensors store the pre-softplus values.
        xielu=None,
        # Tensor-parallel blocks of a fused [q | v | k] projection (CodeGen).
        qkv_mp=0,
        # (alpha, limit) of the clamped swiglu in dense MLPs (MiniMax M3).
        dense_swiglu=None,
        # Kimi K3: (beta, linear_beta) of the SiTU activation in every gated MLP
        # (act stays "silu" for the KDA convolution), the Attention Residual
        # block size (None: a plain residual stream) and the MLA output gate.
        situ=None, attn_res=None, mla_gate=False,
        # {layer index: [(name relative to the layer, shape), ...]} of tensors the
        # reference never reads (they must survive exports untouched).
        extra_layer_tensors={},
        # [(absolute name, shape), ...] of further unread tensors (MTP layers,
        # the vision and audio towers of multimodal wrappers).
        extra_tensors=[],
        # MiMo V2: values narrower than the keys outside MLA, sinks only on the
        # sliding layers, and a fused qkv ("chunked" layout) of `qkv_chunks`
        # chunks of [q heads | k heads | v heads].
        narrow_v=False, sinks_sliding_only=False, qkv_chunks=0,
        quant=None,
        config={}, extra_config={},
    )
    d.update(kw)
    m = dict(ssm_in=1.0, ssm_out=1.0, attn_in=1.0, attn_out=1.0, key=1.0, value=1.0, mlp_gate=1.0, mlp_down=1.0, ssm_proj=[1.0] * 5)
    m.update(d["mult"] or {})
    d["mult"] = m
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
# DeepSeek V3.2-Exp: the V3 layout with every layer an `indexed_attention`
# layer whose lightning indexer keeps the best `index_topk` keys. While the
# prompt fits `index_topk` the indexer selects every key, so the reference is
# the plain dense V3 forward pass; the indexer's own tensors are never read and
# must survive exports untouched. `index_topk` is 16 here, which covers the
# fixture prompts and the tokens generated from them; a longer prompt is
# refused rather than silently approximated.
DSV32_INDEXER = [("self_attn.indexer.wq_b.weight", (2 * 4, 12)), ("self_attn.indexer.wk.weight", (4, 32)),
                 ("self_attn.indexer.k_norm.weight", (4,)), ("self_attn.indexer.k_norm.bias", (4,)),
                 ("self_attn.indexer.weights_proj.weight", (2, 32))]
spec("deepseek_v32", tok="deepseek3", NKV=4, HD=12, VD=8, L=3, rope_style="gptj", rotary_dim=4, lm_head="lm_head.weight",
     mla={"q_lora_rank": 12, "kv_lora_rank": 16, "nope": 8, "rope": 4, "v": 8},
     scaling={"type": "yarn", "factor": 40.0, "beta_fast": 32, "beta_slow": 1, "mscale": 1.0, "mscale_all_dim": 1.0, "original_max_position_embeddings": 32},
     moe={"E": 4, "K": 2, "MI": 12, "shared": 1, "scoring": "sigmoid", "group_limited": True, "n_group": 2, "topk_group": 1, "rsf": 2.5, "norm": True,
          "layers": [1, 2], "corr_bias": True, "layout": "separate", "prefix": "mlp.", "router": "gate.weight", "shared_name": "shared_experts."},
     extra_layer_tensors={i: DSV32_INDEXER for i in range(3)},
     config={"model_type": "deepseek_v32", "hidden_size": 32, "intermediate_size": 32, "moe_intermediate_size": 12, "num_hidden_layers": 3,
             "num_attention_heads": 4, "num_key_value_heads": 4, "q_lora_rank": 12, "kv_lora_rank": 16, "qk_nope_head_dim": 8, "qk_rope_head_dim": 4,
             "v_head_dim": 8, "n_routed_experts": 4, "n_shared_experts": 1, "num_experts_per_tok": 2, "first_k_dense_replace": 1,
             "mlp_layer_types": ["dense", "sparse", "sparse"], "layer_types": ["indexed_attention"] * 3,
             "index_topk": 16, "index_n_heads": 2, "index_head_dim": 4,
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
# Quantised variants (see "Quantised checkpoint variants" above): the qwen2
# layout with DeepSeek-style FP8 block scales and with compressed-tensors
# pack-quantized INT4 (asymmetric, so zero points are exercised too), and
# gpt-oss with MXFP4 experts (hidden size 64 and expert width 64, so rows span two blocks).
QWEN2_CONFIG = {"model_type": "qwen2", "hidden_size": 32, "intermediate_size": 32, "num_hidden_layers": 2, "num_attention_heads": 4, "num_key_value_heads": 2,
                "rms_norm_eps": 1e-6, "rope_theta": 10000.0, "hidden_act": "silu", "max_position_embeddings": 128, "tie_word_embeddings": True,
                "torch_dtype": "bfloat16"}
spec("qwen2_fp8", tok="qwen2", attn_bias=True, o_bias=False, lm_head=None, quant={"kind": "fp8", "block": [16, 8]},
     config=dict(QWEN2_CONFIG, quantization_config={"activation_scheme": "dynamic", "fmt": "e4m3", "quant_method": "fp8", "weight_block_size": [16, 8]}))
spec("qwen2_int4", tok="qwen2", attn_bias=True, o_bias=False, lm_head=None, quant={"kind": "int", "bits": 4, "group": 16, "symmetric": False},
     config=dict(QWEN2_CONFIG, quantization_config={
         "config_groups": {"group_0": {"input_activations": None, "output_activations": None, "targets": ["Linear"],
                                       "weights": {"actorder": None, "block_structure": None, "dynamic": False, "group_size": 16, "num_bits": 4,
                                                   "observer": "minmax", "strategy": "group", "symmetric": False, "type": "int"}}},
         "format": "pack-quantized", "global_compression_ratio": None, "ignore": ["lm_head"], "kv_cache_scheme": None,
         "quant_method": "compressed-tensors", "quantization_status": "compressed"}))
# `actorder`: the same INT4 layout with a `weight_g_idx` that scatters each
# group's columns, so a loader that assumes contiguous groups decodes every
# column with the wrong scale (and does so silently).
spec("qwen2_int4_actorder", tok="qwen2", attn_bias=True, o_bias=False, lm_head=None,
     quant={"kind": "int", "bits": 4, "group": 16, "symmetric": False, "actorder": True},
     config=dict(QWEN2_CONFIG, quantization_config={
         "config_groups": {"group_0": {"input_activations": None, "output_activations": None, "targets": ["Linear"],
                                       "weights": {"actorder": "group", "block_structure": None, "dynamic": False, "group_size": 16, "num_bits": 4,
                                                   "observer": "minmax", "strategy": "group", "symmetric": False, "type": "int"}}},
         "format": "pack-quantized", "global_compression_ratio": None, "ignore": ["lm_head"], "kv_cache_scheme": None,
         "quant_method": "compressed-tensors", "quantization_status": "compressed"}))
spec("gpt_oss_mxfp4", tok="o200k", H=64, L=3, attn_bias=True, o_bias=True, sinks="self_attn.sinks", sliding=4, sliding_layers=[1, 0, 1], lm_head=None,
     quant={"kind": "mxfp4"},
     scaling={"rope_type": "yarn", "factor": 8.0, "beta_fast": 32.0, "beta_slow": 1.0, "original_max_position_embeddings": 32, "truncate": False},
     moe={"E": 4, "K": 2, "MI": 64, "shared": 0, "scoring": "softmax", "group_limited": False, "rsf": 1.0, "norm": True, "router_bias": True,
          "layers": [0, 1, 2], "corr_bias": False, "layout": "fused_t_interleaved", "prefix": "mlp.", "router": "router.weight", "bias": True,
          "swiglu": (1.702, 7.0)},
     config={"model_type": "gpt_oss", "hidden_size": 64, "intermediate_size": 64, "num_hidden_layers": 3, "num_attention_heads": 4, "num_key_value_heads": 2,
             "head_dim": 8, "num_local_experts": 4, "num_experts_per_tok": 2, "sliding_window": 4, "layer_types": ["sliding_attention", "full_attention", "sliding_attention"],
             "rope_theta": 10000.0, "rope_scaling": {"rope_type": "yarn", "factor": 8.0, "beta_fast": 32.0, "beta_slow": 1.0, "original_max_position_embeddings": 32, "truncate": False},
             "max_position_embeddings": 128, "rms_norm_eps": 1e-5, "swiglu_limit": 7.0, "attention_bias": True, "hidden_act": "silu", "tie_word_embeddings": True,
             "torch_dtype": "bfloat16",
             "quantization_config": {"modules_to_not_convert": ["model.layers.*.self_attn", "model.layers.*.mlp.router", "model.embed_tokens", "lm_head"],
                                     "quant_method": "mxfp4"}})
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
# Mistral: the llama layout with a sliding window on every layer and an
# explicit head_dim. The window (4) is shorter than the fixture prompts, so the
# local mask bites, and head_dim (12) is not hidden_size / num_attention_heads,
# so the explicit key is what sizes the projections.
spec("mistral", tok="llama3", HD=12, sliding=4, sliding_layers=[1, 1], lm_head=None,
     config={"model_type": "mistral", "hidden_size": 32, "intermediate_size": 32, "num_hidden_layers": 2, "num_attention_heads": 4, "num_key_value_heads": 2,
             "head_dim": 12, "rms_norm_eps": 1e-6, "rope_theta": 10000.0, "sliding_window": 4, "hidden_act": "silu", "max_position_embeddings": 128,
             "tie_word_embeddings": True})
# Mixtral: softmax top-k routing with renormalisation over separate expert
# tensors named the way released Mixtral checkpoints store them
# (block_sparse_moe.experts.{e}.w1 / w2 / w3).
spec("mixtral", tok="llama3", L=2, lm_head=None,
     moe={"E": 4, "K": 2, "MI": 12, "shared": 0, "scoring": "softmax", "group_limited": False, "rsf": 1.0, "norm": True,
          "layers": [0, 1], "corr_bias": False, "layout": "separate", "prefix": "block_sparse_moe.", "router": "gate.weight",
          "expert_names": ("w1.weight", "w3.weight", "w2.weight")},
     config={"model_type": "mixtral", "hidden_size": 32, "intermediate_size": 12, "num_hidden_layers": 2, "num_attention_heads": 4, "num_key_value_heads": 2,
             "num_local_experts": 4, "num_experts_per_tok": 2, "rms_norm_eps": 1e-6, "rope_theta": 10000.0, "hidden_act": "silu",
             "max_position_embeddings": 128, "tie_word_embeddings": True})
# Qwen2-MoE: softmax top-k routing over experts of `moe_intermediate_size`
# (12) plus a shared expert of `shared_expert_intermediate_size` (16) behind a
# sigmoid gate, all three widths different from the dense `intermediate_size`
# (32) that layer 0 keeps through `mlp_only_layers`.
spec("qwen2_moe", tok="qwen2", L=3, attn_bias=True, o_bias=False, lm_head=None,
     moe={"E": 4, "K": 2, "MI": 12, "shared": 1, "shared_inter": 16, "shared_gate": True, "scoring": "softmax", "group_limited": False, "rsf": 1.0, "norm": False,
          "layers": [1, 2], "corr_bias": False, "layout": "separate", "prefix": "mlp.", "router": "gate.weight", "shared_name": "shared_expert."},
     config={"model_type": "qwen2_moe", "hidden_size": 32, "intermediate_size": 32, "moe_intermediate_size": 12, "shared_expert_intermediate_size": 16,
             "num_hidden_layers": 3, "num_attention_heads": 4, "num_key_value_heads": 2, "num_experts": 4, "num_experts_per_tok": 2, "norm_topk_prob": False,
             "decoder_sparse_step": 1, "mlp_only_layers": [0], "rms_norm_eps": 1e-6, "rope_theta": 10000.0, "hidden_act": "silu",
             "max_position_embeddings": 128, "tie_word_embeddings": True})
spec("qwen3_next", tok="qwen2", H=32, I=32, L=4, NH=4, NKV=2, HD=8, rotary_dim=2, qk_norm="head", gated_q=True, norm="rms1p",
     linear={"KH": 2, "KD": 4, "VH": 4, "VD": 4, "KC": 2, "fused": True}, full_interval=4,
     moe={"E": 4, "K": 2, "MI": 12, "shared": 1, "shared_inter": 16, "shared_gate": True, "scoring": "softmax", "group_limited": False, "rsf": 1.0, "norm": True,
          "layers": [0, 1, 2, 3], "corr_bias": False, "layout": "separate", "prefix": "mlp.", "router": "gate.weight", "shared_name": "shared_expert."},
     config={"model_type": "qwen3_next", "hidden_size": 32, "intermediate_size": 32, "moe_intermediate_size": 12, "shared_expert_intermediate_size": 16,
             "num_hidden_layers": 4, "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 8,
             "num_experts": 4, "num_experts_per_tok": 2, "norm_topk_prob": True, "decoder_sparse_step": 1, "mlp_only_layers": [],
             "linear_num_key_heads": 2, "linear_key_head_dim": 4, "linear_num_value_heads": 4, "linear_value_head_dim": 4,
             "linear_conv_kernel_dim": 2, "full_attention_interval": 4, "partial_rotary_factor": 0.25,
             "rms_norm_eps": 1e-6, "rope_theta": 10000.0, "hidden_act": "silu", "max_position_embeddings": 128, "tie_word_embeddings": True})
# Kimi Linear (KDA + MLA hybrid, DeepSeek-V3-style MoE). `kimi_linear` is the
# original checkpoint layout (linear_attn_config, q/k/v convolutions on
# self_attn, block_sparse_moe with w1/w3/w2 experts); `kimi_linear_hf` the
# Hugging Face module layout (layer_types, forget_gate, fused conv1d, stacked
# experts). Full-attention layers carry no positional encoding.
_KIMI_MLA = {"q_lora_rank": None, "kv_lora_rank": 16, "nope": 8, "rope": 4, "v": 8}
_KIMI_KDA = {"kind": "kda", "KH": 4, "KD": 4, "VH": 4, "VD": 4, "KC": 3}
spec("kimi_linear", tok="deepseek3", NKV=4, HD=12, VD=8, L=4, rotary_dim=4, rope_layers=[0, 0, 0, 0], lm_head="lm_head.weight", eps=1e-5,
     mla=_KIMI_MLA, linear=dict(_KIMI_KDA, layout="checkpoint"), linear_layers=[1, 1, 1, 0],
     moe={"E": 4, "K": 2, "MI": 12, "shared": 1, "scoring": "sigmoid", "group_limited": True, "n_group": 2, "topk_group": 1, "rsf": 2.446, "norm": True,
          "layers": [1, 2, 3], "corr_bias": True, "layout": "separate", "prefix": "block_sparse_moe.", "router": "gate.weight",
          "expert_names": ("w1.weight", "w3.weight", "w2.weight"), "shared_name": "shared_experts."},
     config={"model_type": "kimi_linear", "architectures": ["KimiLinearForCausalLM"], "hidden_size": 32, "intermediate_size": 32, "moe_intermediate_size": 12,
             "num_hidden_layers": 4, "num_attention_heads": 4, "num_key_value_heads": 4, "q_lora_rank": None, "kv_lora_rank": 16,
             "qk_nope_head_dim": 8, "qk_rope_head_dim": 4, "v_head_dim": 8, "num_experts": 4, "num_shared_experts": 1, "num_experts_per_token": 2,
             "num_expert_group": 2, "topk_group": 1, "moe_renormalize": True, "routed_scaling_factor": 2.446, "first_k_dense_replace": 1,
             "linear_attn_config": {"kda_layers": [1, 2, 3], "full_attn_layers": [4], "head_dim": 4, "num_heads": 4, "short_conv_kernel_size": 3},
             "rms_norm_eps": 1e-5, "hidden_act": "silu", "model_max_length": 128, "tie_word_embeddings": False})
spec("kimi_linear_hf", tok="deepseek3", NKV=4, HD=12, VD=8, L=4, rotary_dim=4, rope_layers=[0, 0, 0, 0], lm_head="lm_head.weight", eps=1e-5,
     mla=_KIMI_MLA, linear=dict(_KIMI_KDA, layout="hf"), linear_layers=[1, 0, 1, 1],
     moe={"E": 4, "K": 2, "MI": 12, "shared": 1, "scoring": "sigmoid", "group_limited": True, "n_group": 1, "topk_group": 1, "rsf": 2.446, "norm": True,
          "layers": [1, 2, 3], "corr_bias": True, "layout": "fused_eih", "prefix": "mlp.", "router": "gate.weight", "shared_name": "shared_experts."},
     config={"model_type": "kimi_linear", "architectures": ["KimiLinearForCausalLM"], "hidden_size": 32, "intermediate_size": 32, "moe_intermediate_size": 12,
             "num_hidden_layers": 4, "num_attention_heads": 4, "num_key_value_heads": 4, "head_dim": 4, "q_lora_rank": None, "kv_lora_rank": 16,
             "qk_nope_head_dim": 8, "qk_rope_head_dim": 4, "v_head_dim": 8, "num_local_experts": 4, "n_shared_experts": 1, "num_experts_per_tok": 2,
             "n_group": 1, "topk_group": 1, "norm_topk_prob": True, "routed_scaling_factor": 2.446,
             "layer_types": ["linear_attention", "full_attention", "linear_attention", "linear_attention"],
             "mlp_layer_types": ["dense", "sparse", "sparse", "sparse"],
             "linear_head_dim": 4, "linear_num_heads": 4, "linear_conv_kernel_dim": 3,
             "rms_norm_eps": 1e-5, "hidden_act": "silu", "max_position_embeddings": 128, "tie_word_embeddings": False})
# Kimi K3: the image-video wrapper (KimiK3ForConditionalGeneration) around a
# kimi_linear text config with Attention Residual (attn_res_block_size 2 over 5
# layers: prefixes banked at layers 0, 2 and 4), KDA layers with the full-rank
# output gate and the safe forget gate (gate_lower_bound), MLA layers with
# q_lora_rank and the sigmoid output gate, latent MoE (routed_expert_hidden_size
# 32 under hidden_size 48, with routed_expert_norm; w2 spans two MXFP4 blocks),
# SiTU, two shared experts
# and the original checkpoint names (block_sparse_moe, w1/w3/w2, q/k/v_conv1d,
# language_model prefix). `kimi_k3_mxfp4` stores the routed experts in the
# compressed-tensors mxfp4-pack-quantized format of the released checkpoint.
_K3_TEXT = {"model_type": "kimi_linear", "architectures": ["KimiLinearForCausalLM"], "hidden_size": 48, "intermediate_size": 32, "moe_intermediate_size": 64,
            "routed_expert_hidden_size": 32, "latent_moe_use_norm": True, "attn_res_block_size": 2, "num_hidden_layers": 5, "num_attention_heads": 4,
            "num_key_value_heads": 4, "q_lora_rank": 12, "kv_lora_rank": 16, "qk_nope_head_dim": 8, "qk_rope_head_dim": 4, "v_head_dim": 8,
            "mla_use_nope": True, "mla_use_output_gate": True, "num_experts": 4, "num_shared_experts": 2, "num_experts_per_token": 2,
            "num_expert_group": 1, "topk_group": 1, "topk_method": "noaux_tc", "use_grouped_topk": True, "moe_router_activation_func": "sigmoid",
            "moe_renormalize": True, "routed_scaling_factor": 1.0, "first_k_dense_replace": 1, "moe_layer_freq": 1,
            "hidden_act": "situ", "activation_situ_beta": 4.0, "activation_situ_linear_beta": 25.0,
            "linear_attn_config": {"kda_layers": [1, 2, 4], "full_attn_layers": [3, 5], "head_dim": 4, "num_heads": 4, "short_conv_kernel_size": 3,
                                   "use_full_rank_gate": True, "gate_lower_bound": -5.0},
            "rms_norm_eps": 1e-5, "rope_theta": 10000.0, "max_position_embeddings": 128, "tie_word_embeddings": False}
_K3_VISION = {"model_type": "kimi_k3_vision", "vt_hidden_size": 16, "vt_num_hidden_layers": 1, "vt_num_attention_heads": 2, "patch_size": 14, "text_hidden_size": 48}
_K3_QUANT = {"config_groups": {"group_0": {"format": "mxfp4-pack-quantized", "input_activations": None, "output_activations": None, "targets": ["Linear"],
                                           "weights": {"actorder": None, "block_structure": None, "dynamic": False, "group_size": 32, "num_bits": 4,
                                                       "observer": "minmax", "observer_kwargs": {}, "scale_dtype": "torch.uint8", "strategy": "group",
                                                       "symmetric": True, "type": "float", "zp_dtype": None}}},
             "format": "mxfp4-pack-quantized", "global_compression_ratio": None,
             "ignore": ["re:.*self_attn.*", "re:.*shared_experts.*", "re:.*mlp\\.(gate|up|gate_up|down)_proj.*", "re:.*lm_head.*", "re:.*vision_tower.*", "re:.*mm_projector.*"],
             "kv_cache_scheme": None, "quant_method": "compressed-tensors", "quantization_status": "compressed"}
_K3 = dict(tok="deepseek3", H=48, I=32, NKV=4, HD=12, VD=8, L=5, rotary_dim=4, rope_layers=[0] * 5, prefix="language_model.model.", lm_head="language_model.lm_head.weight",
           eps=1e-5, situ=(4.0, 25.0), attn_res=2, mla_gate=True,
           mla={"q_lora_rank": 12, "kv_lora_rank": 16, "nope": 8, "rope": 4, "v": 8},
           linear=dict(_KIMI_KDA, layout="checkpoint", full_rank_gate=True, lower_bound=-5.0), linear_layers=[1, 1, 0, 1, 0],
           moe={"E": 4, "K": 2, "MI": 64, "shared": 2, "latent": 32, "latent_norm": True, "scoring": "sigmoid", "group_limited": True, "n_group": 1, "topk_group": 1,
                "rsf": 1.0, "norm": True, "layers": [1, 2, 3, 4], "corr_bias": True, "layout": "separate", "prefix": "block_sparse_moe.", "router": "gate.weight",
                "expert_names": ("w1.weight", "w3.weight", "w2.weight"), "shared_name": "shared_experts."})
spec("kimi_k3", config={"model_type": "kimi_k3", "architectures": ["KimiK3ForConditionalGeneration"], "tie_word_embeddings": False, "media_placeholder_token_id": 200,
                        "vision_config": _K3_VISION, "text_config": _K3_TEXT}, **_K3)
spec("kimi_k3_mxfp4", quant={"kind": "mxfp4_packed"},
     config={"model_type": "kimi_k3", "architectures": ["KimiK3ForConditionalGeneration"], "tie_word_embeddings": False, "media_placeholder_token_id": 200,
             "vision_config": _K3_VISION, "text_config": dict(_K3_TEXT, quantization_config=_K3_QUANT)}, **_K3)
# Kimi K2.5 / K2.6: the image-video wrapper around a DeepSeek V3 text config
# (model_type kimi_k2 under text_config, language_model prefix).
spec("kimi_k25", tok="deepseek3", NKV=4, HD=12, VD=8, L=3, prefix="language_model.model.", lm_head="language_model.lm_head.weight",
     rope_style="gptj", rotary_dim=4,
     mla={"q_lora_rank": 12, "kv_lora_rank": 16, "nope": 8, "rope": 4, "v": 8},
     scaling={"type": "yarn", "factor": 32.0, "beta_fast": 1.0, "beta_slow": 1.0, "mscale": 1.0, "mscale_all_dim": 1.0, "original_max_position_embeddings": 32},
     moe={"E": 4, "K": 2, "MI": 12, "shared": 1, "scoring": "sigmoid", "group_limited": True, "n_group": 1, "topk_group": 1, "rsf": 2.827, "norm": True,
          "layers": [1, 2], "corr_bias": True, "layout": "separate", "prefix": "mlp.", "router": "gate.weight", "shared_name": "shared_experts."},
     config={"model_type": "kimi_k25", "architectures": ["Kimi_K25ForConditionalGeneration"], "tie_word_embeddings": False,
             "image_token_id": 200, "video_token_id": 201,
             "vision_config": {"model_type": "kimi_k25_vision", "hidden_size": 16, "num_hidden_layers": 1, "num_attention_heads": 2, "patch_size": 14},
             "text_config": {"model_type": "kimi_k2", "hidden_size": 32, "intermediate_size": 32, "moe_intermediate_size": 12, "num_hidden_layers": 3,
                             "num_attention_heads": 4, "num_key_value_heads": 4, "q_lora_rank": 12, "kv_lora_rank": 16, "qk_nope_head_dim": 8, "qk_rope_head_dim": 4,
                             "v_head_dim": 8, "n_routed_experts": 4, "n_shared_experts": 1, "num_experts_per_tok": 2, "first_k_dense_replace": 1, "moe_layer_freq": 1,
                             "scoring_func": "sigmoid", "topk_method": "noaux_tc", "n_group": 1, "topk_group": 1, "routed_scaling_factor": 2.827,
                             "norm_topk_prob": True, "rms_norm_eps": 1e-6, "rope_theta": 10000.0, "max_position_embeddings": 128, "hidden_act": "silu",
                             "tie_word_embeddings": False, "rope_interleave": True}})
spec("qwen3_5", tok="qwen2", H=32, I=32, L=2, NH=4, NKV=2, HD=8, rotary_dim=2,
     qk_norm="head", gated_q=True, norm="rms1p",
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
spec("qwen3_5_moe", tok="qwen2", prefix="model.language_model.", H=32, I=32, L=4, NH=4, NKV=2, HD=8, rotary_dim=2, qk_norm="head", gated_q=True, gate_swish=True, norm="rms1p",
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
# GLM-4.7-Flash: DeepSeek V3 MLA with interleaved partial rotary and the
# GLM-4.5 router (sigmoid scores, correction bias, top-2 group scores over
# n_group groups, floored renormalisation, routed_scaling_factor), stacked
# expert tensors and an mlp_layer_types schedule with a dense first layer.
spec("glm4_moe_lite", tok="llama3", H=32, I=32, L=3, NH=4, NKV=4, HD=12, VD=8, rope_style="gptj", rotary_dim=4, eps=1e-5,
     mla={"q_lora_rank": 12, "kv_lora_rank": 16, "nope": 8, "rope": 4, "v": 8},
     moe={"E": 8, "K": 2, "MI": 12, "shared": 1, "shared_inter": 16, "scoring": "sigmoid", "group_limited": True, "n_group": 2, "topk_group": 1,
          "rsf": 1.8, "norm": True, "layers": [1, 2], "corr_bias": True, "layout": "fused_eih", "prefix": "mlp.", "router": "gate.weight",
          "shared_name": "shared_experts."},
     config={"model_type": "glm4_moe_lite", "architectures": ["Glm4MoeLiteForCausalLM"], "hidden_size": 32, "intermediate_size": 32,
             "moe_intermediate_size": 12, "num_hidden_layers": 3, "num_attention_heads": 4, "num_key_value_heads": 4,
             "q_lora_rank": 12, "kv_lora_rank": 16, "qk_nope_head_dim": 8, "qk_rope_head_dim": 4, "v_head_dim": 8,
             "n_routed_experts": 8, "n_shared_experts": 1, "num_experts_per_tok": 2, "n_group": 2, "topk_group": 1,
             "routed_scaling_factor": 1.8, "norm_topk_prob": True, "mlp_layer_types": ["dense", "sparse", "sparse"],
             "rms_norm_eps": 1e-5, "rope_parameters": {"rope_type": "default", "rope_theta": 10000.0}, "rope_interleave": True,
             "hidden_act": "silu", "max_position_embeddings": 128, "tie_word_embeddings": False})
MIXTRAL_EXPERTS = ("w1.weight", "w3.weight", "w2.weight")
spec("minimax_m2", tok="o200k", L=2, rotary_dim=4, qk_norm="full", lm_head="lm_head.weight", theta=5000000.0,
     moe={"E": 4, "K": 2, "MI": 12, "shared": 0, "scoring": "sigmoid", "group_limited": False, "rsf": 1.0, "norm": True,
          "layers": [0, 1], "corr_bias": True, "corr_bias_name": "block_sparse_moe.e_score_correction_bias", "layout": "separate",
          "prefix": "block_sparse_moe.", "router": "gate.weight", "expert_names": MIXTRAL_EXPERTS},
     config={"model_type": "minimax_m2", "hidden_size": 32, "intermediate_size": 12, "num_hidden_layers": 2, "num_attention_heads": 4,
             "num_key_value_heads": 2, "head_dim": 8, "num_local_experts": 4, "num_experts_per_tok": 2, "rotary_dim": 4,
             "rms_norm_eps": 1e-6, "rope_theta": 5000000.0, "hidden_act": "silu", "max_position_embeddings": 128, "tie_word_embeddings": False})
spec("minimax", tok="llama3", L=4, rotary_dim=4, lm_head="lm_head.weight", theta=1000000.0, eps=1e-5,
     linear_kind="lightning", linear_layers=[1, 0, 1, 0], residual_layout="minimax",
     mm_scales={"full": (2.0, 1.0), "linear": (1.5, 1.0), "mlp": (3.0, 0.5)},
     moe={"E": 4, "K": 2, "MI": 12, "shared": 0, "scoring": "softmax", "group_limited": False, "rsf": 1.0, "norm": True,
          "layers": [0, 1, 2, 3], "corr_bias": False, "layout": "separate", "prefix": "block_sparse_moe.", "router": "gate.weight",
          "expert_names": MIXTRAL_EXPERTS},
     config={"model_type": "minimax", "hidden_size": 32, "intermediate_size": 12, "num_hidden_layers": 4, "num_attention_heads": 4,
             "num_key_value_heads": 2, "head_dim": 8, "num_local_experts": 4, "num_experts_per_tok": 2, "rotary_dim": 4, "block_size": 256,
             "layer_types": ["linear_attention", "full_attention", "linear_attention", "full_attention"],
             "full_attn_alpha_factor": 2.0, "full_attn_beta_factor": 1.0, "linear_attn_alpha_factor": 1.5, "linear_attn_beta_factor": 1.0,
             "mlp_alpha_factor": 3.0, "mlp_beta_factor": 0.5,
             "rms_norm_eps": 1e-5, "rope_theta": 1000000.0, "hidden_act": "silu", "max_position_embeddings": 128, "tie_word_embeddings": False})
spec("minimax_m3", tok="o200k", L=3, I=16, rotary_dim=4, prefix="language_model.model.", lm_head="language_model.lm_head.weight", theta=5000000.0,
     norm="rms1p", qk_norm="head", mlp="gated_fused", gate_up="mlp.gate_up_proj.weight", down="mlp.down_proj.weight", dense_swiglu=(1.702, 7.0),
     moe={"E": 4, "K": 2, "MI": 12, "shared": 1, "shared_inter": 16, "shared_fused": ("gate_up_proj.weight", "down_proj.weight"),
          "scoring": "sigmoid", "group_limited": False, "rsf": 2.0, "norm": True, "layers": [1, 2], "corr_bias": True,
          "corr_bias_name": "block_sparse_moe.e_score_correction_bias", "layout": "separate", "prefix": "block_sparse_moe.",
          "router": "gate.weight", "expert_names": MIXTRAL_EXPERTS, "shared_name": "shared_experts.", "swiglu": (1.702, 7.0)},
     extra_layer_tensors={i: [("self_attn.index_q_proj.weight", (2 * 4, 32)), ("self_attn.index_k_proj.weight", (4, 32)),
                              ("self_attn.index_q_norm.weight", (4,)), ("self_attn.index_k_norm.weight", (4,))] for i in (1, 2)},
     config={"model_type": "minimax_m3_vl", "architectures": ["MiniMaxM3SparseForConditionalGeneration"],
             "text_config": {"model_type": "minimax_m3_vl_text", "hidden_size": 32, "intermediate_size": 12, "dense_intermediate_size": 16,
                             "shared_intermediate_size": 16, "num_hidden_layers": 3, "num_attention_heads": 4, "num_key_value_heads": 2,
                             "head_dim": 8, "num_local_experts": 4, "num_experts_per_tok": 2, "routed_scaling_factor": 2.0, "rotary_dim": 4,
                             "swiglu_alpha": 1.702, "swiglu_limit": 7.0, "mlp_layer_types": ["dense", "sparse", "sparse"],
                             "layer_types": ["full_attention", "minimax_m3_sparse", "minimax_m3_sparse"],
                             "index_n_heads": 2, "index_head_dim": 4, "index_block_size": 4, "index_topk_blocks": 16, "index_local_blocks": 1,
                             "rms_norm_eps": 1e-6, "rope_theta": 5000000.0, "hidden_act": "silu", "max_position_embeddings": 128,
                             "tie_word_embeddings": False},
             "vision_config": {"model_type": "minimax_m3_vl_vision"}})
# The released MiniMax-M3 checkpoints keep the dense and shared MLPs split
# (`gate_proj` / `up_proj`) where the reference implementation fuses them;
# same architecture, the other spelling.
spec("minimax_m3_split", tok="o200k", L=3, I=16, rotary_dim=4, prefix="language_model.model.", lm_head="language_model.lm_head.weight", theta=5000000.0,
     norm="rms1p", qk_norm="head", mlp="gated", gate="mlp.gate_proj.weight", up="mlp.up_proj.weight", down="mlp.down_proj.weight", dense_swiglu=(1.702, 7.0),
     moe={"E": 4, "K": 2, "MI": 12, "shared": 1, "shared_inter": 16, "shared_gate": False,
          "scoring": "sigmoid", "group_limited": False, "rsf": 2.0, "norm": True, "layers": [1, 2], "corr_bias": True,
          "corr_bias_name": "block_sparse_moe.e_score_correction_bias", "layout": "separate", "prefix": "block_sparse_moe.",
          "router": "gate.weight", "expert_names": MIXTRAL_EXPERTS, "shared_name": "shared_experts.", "swiglu": (1.702, 7.0)},
     extra_layer_tensors={i: [("self_attn.index_q_proj.weight", (2 * 4, 32)), ("self_attn.index_k_proj.weight", (4, 32)),
                              ("self_attn.index_q_norm.weight", (4,)), ("self_attn.index_k_norm.weight", (4,))] for i in (1, 2)},
     config={"model_type": "minimax_m3_vl", "architectures": ["MiniMaxM3SparseForConditionalGeneration"],
             "text_config": {"model_type": "minimax_m3_vl_text", "hidden_size": 32, "intermediate_size": 12, "dense_intermediate_size": 16,
                             "shared_intermediate_size": 16, "num_hidden_layers": 3, "num_attention_heads": 4, "num_key_value_heads": 2,
                             "head_dim": 8, "num_local_experts": 4, "num_experts_per_tok": 2, "routed_scaling_factor": 2.0, "rotary_dim": 4,
                             "swiglu_alpha": 1.702, "swiglu_limit": 7.0, "mlp_layer_types": ["dense", "sparse", "sparse"],
                             "layer_types": ["full_attention", "minimax_m3_sparse", "minimax_m3_sparse"],
                             "index_n_heads": 2, "index_head_dim": 4, "index_block_size": 4, "index_topk_blocks": 16, "index_local_blocks": 1,
                             "rms_norm_eps": 1e-6, "rope_theta": 5000000.0, "hidden_act": "silu", "max_position_embeddings": 128,
                             "tie_word_embeddings": False},
             "vision_config": {"model_type": "minimax_m3_vl_vision"}})
spec("ernie4_5_moe", tok="spm", L=3, rope_style="gptj", lm_head="lm_head.weight", theta=500000.0, eps=1e-5,
     moe={"E": 4, "K": 2, "MI": 12, "shared": 2, "shared_inter": 24, "scoring": "softmax", "group_limited": False, "rsf": 1.0, "norm": True,
          "layers": [1, 2], "corr_bias": True, "corr_bias_name": "mlp.moe_statics.e_score_correction_bias", "corr_bias_shape": (1, 4),
          "layout": "separate", "prefix": "mlp.", "router": "gate.weight", "shared_name": "shared_experts."},
     config={"model_type": "ernie4_5_moe", "hidden_size": 32, "intermediate_size": 32, "moe_intermediate_size": 12, "num_hidden_layers": 3,
             "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 8, "moe_num_experts": 4, "moe_k": 2, "moe_num_shared_experts": 2,
             "moe_layer_start_index": 1, "moe_layer_end_index": -1, "moe_layer_interval": 1, "moe_norm_min": 1e-12, "use_bias": False,
             "rms_norm_eps": 1e-5, "rope_theta": 500000.0, "hidden_act": "silu", "max_position_embeddings": 128, "tie_word_embeddings": False})
spec("hunyuan_v1_moe", tok="gpt2", L=2, lm_head="lm_head.weight", eps=1e-5, qk_norm="head", qk_norm_after_rope=True,
     q_norm="self_attn.query_layernorm.weight", k_norm="self_attn.key_layernorm.weight",
     theta=10000.0 * 1000.0 ** (8 / 6),  # NTK-alpha dynamic scaling: base * alpha^(head_dim / (head_dim - 2))
     moe={"E": 4, "K": 2, "MI": 12, "shared": 1, "shared_inter": 12, "scoring": "softmax", "group_limited": False, "rsf": 1.0, "norm": True,
          "layers": [0, 1], "corr_bias": False, "layout": "separate", "prefix": "mlp.", "router": "gate.wg.weight", "shared_name": "shared_mlp."},
     config={"model_type": "hunyuan_v1_moe", "hidden_size": 32, "intermediate_size": 12, "num_hidden_layers": 2, "num_attention_heads": 4,
             "num_key_value_heads": 2, "head_dim": 8, "num_experts": [4, 4], "moe_topk": [2, 2], "num_shared_expert": 1, "use_mixed_mlp_moe": True,
             "use_qk_norm": True, "attention_bias": False, "rope_scaling": {"type": "dynamic", "alpha": 1000.0},
             "rms_norm_eps": 1e-5, "rope_theta": 10000.0, "hidden_act": "silu", "max_position_embeddings": 128, "tie_word_embeddings": False})
GRANITE_MOE = {"E": 4, "K": 2, "MI": 12, "shared": 0, "scoring": "topk_softmax", "group_limited": False, "rsf": 1.0, "norm": True,
               "corr_bias": False, "layout": "fused_rows", "fused_names": ("input_linear.weight", "output_linear.weight"),
               "prefix": "block_sparse_moe.", "router": "router.layer.weight"}
GRANITE_CONFIG = {"hidden_size": 32, "intermediate_size": 12, "num_attention_heads": 4, "num_key_value_heads": 2, "num_local_experts": 4,
                  "num_experts_per_tok": 2, "embedding_multiplier": 2.0, "attention_multiplier": 0.25, "residual_multiplier": 0.5,
                  "logits_scaling": 4.0, "rms_norm_eps": 1e-6, "rope_theta": 10000.0, "hidden_act": "silu", "max_position_embeddings": 128,
                  "tie_word_embeddings": True}
spec("granitemoe", tok="starcoder", L=2, embed_scale=2.0, attn_scale=0.25, residual_mult=0.5, logit_scale=0.25, lm_head=None,
     moe=dict(GRANITE_MOE, layers=[0, 1]),
     config=dict(GRANITE_CONFIG, model_type="granitemoe", num_hidden_layers=2))
spec("granitemoehybrid_attn", tok="starcoder", L=3, embed_scale=2.0, attn_scale=0.25, residual_mult=0.5, logit_scale=0.25, lm_head=None,
     moe=dict(GRANITE_MOE, layers=[0, 1, 2], shared=1, shared_inter=16, shared_name="shared_mlp.", shared_at_layer=True,
              shared_fused=("input_linear.weight", "output_linear.weight")),
     config=dict(GRANITE_CONFIG, model_type="granitemoehybrid", num_hidden_layers=3, shared_intermediate_size=16, position_embedding_type="rope",
                 layer_types=["attention", "attention", "attention"], mamba_n_heads=4, mamba_d_state=8, mamba_d_conv=4, mamba_expand=2))

# Mamba families. `Infinity` in time_step_limit is what json.dump writes for
# float("inf"), as transformers does; ditch's config reader must accept it.
INF = float("inf")
MAMBA2_SSM = {"kind": "mamba2", "prefix": "mixer.", "heads": 4, "hd": 16, "N": 8, "G": 2, "K": 4, "norm_groups": 1, "rms_norm": True,
              "norm_before_gate": False, "dt_min": 0.0, "dt_max": INF, "conv_bias": True, "proj_bias": False, "dt_bias_mean": 0.0}
spec("mamba2", L=3, NH=1, NKV=1, HD=1, prefix="backbone.", embed="embeddings.weight", final_norm="norm_f.weight", in_norm="norm.weight", pre_ff_norm=None,
     eps=1e-5, pos="none", ssm=MAMBA2_SSM, ssm_layers=[1, 1, 1], attn_layers=[0, 0, 0], mlp_layers=[0, 0, 0],
     config={"model_type": "mamba2", "hidden_size": 32, "num_hidden_layers": 3, "num_heads": 4, "head_dim": 16, "state_size": 8, "n_groups": 2,
             "expand": 2, "conv_kernel": 4, "use_bias": False, "use_conv_bias": True, "hidden_act": "silu", "layer_norm_epsilon": 1e-5,
             "time_step_limit": [0.0, INF], "chunk_size": 256, "tie_word_embeddings": False})
spec("nemotron_h", tok="llama3", L=4, prefix="backbone.", embed="embeddings.weight", final_norm="norm_f.weight", in_norm="norm.weight", pre_ff_norm=None,
     eps=1e-5, pos="none", q="mixer.q_proj.weight", k="mixer.k_proj.weight", v="mixer.v_proj.weight", o="mixer.o_proj.weight",
     mlp="dense", up="mixer.up_proj.weight", down="mixer.down_proj.weight", act="relu2",
     ssm=dict(MAMBA2_SSM, norm_groups=2, dt_min=0.1, dt_bias_mean=-3.0), ssm_layers=[1, 0, 0, 0], attn_layers=[0, 1, 0, 0], mlp_layers=[0, 0, 1, 1],
     moe={"E": 4, "K": 2, "MI": 12, "shared": 1, "shared_inter": 16, "scoring": "sigmoid", "group_limited": True, "n_group": 1, "topk_group": 1,
          "rsf": 1.5, "norm": True, "layers": [3], "corr_bias": True, "layout": "separate", "prefix": "mixer.", "router": "gate.weight",
          "shared_name": "shared_experts.", "dense": True},
     config={"model_type": "nemotron_h", "hidden_size": 32, "num_hidden_layers": 4, "hybrid_override_pattern": "M*-E", "num_attention_heads": 4,
             "num_key_value_heads": 2, "head_dim": 8, "intermediate_size": 32, "mlp_hidden_act": "relu2", "mlp_bias": False, "attention_bias": False,
             "mamba_num_heads": 4, "mamba_head_dim": 16, "ssm_state_size": 8, "n_groups": 2, "conv_kernel": 4, "expand": 2, "use_bias": False,
             "use_conv_bias": True, "mamba_hidden_act": "silu", "layer_norm_epsilon": 1e-5, "time_step_min": 0.1, "chunk_size": 128,
             "n_routed_experts": 4, "num_experts_per_tok": 2, "moe_intermediate_size": 12, "moe_shared_expert_intermediate_size": 16,
             "routed_scaling_factor": 1.5, "n_group": 1, "topk_group": 1, "norm_topk_prob": True, "moe_latent_size": None,
             "max_position_embeddings": 128, "tie_word_embeddings": False})
FALCON_CFG = {"model_type": "falcon_h1", "hidden_size": 32, "intermediate_size": 32, "num_hidden_layers": 2, "num_attention_heads": 4,
              "num_key_value_heads": 2, "head_dim": 8, "hidden_act": "silu", "rms_norm_eps": 1e-5, "rope_theta": 10000.0, "max_position_embeddings": 128,
              "mamba_d_ssm": 64, "mamba_n_heads": 4, "mamba_d_head": "auto", "mamba_n_groups": 2, "mamba_d_state": 8, "mamba_d_conv": 4,
              "mamba_expand": 2, "mamba_chunk_size": 256, "mamba_conv_bias": True, "mamba_proj_bias": False, "mamba_norm_before_gate": True,
              "mamba_rms_norm": True, "projectors_bias": False, "lm_head_multiplier": 0.5, "embedding_multiplier": 2.0,
              "mlp_multipliers": [1.5, 0.5], "key_multiplier": 0.8, "attention_out_multiplier": 0.7, "attention_in_multiplier": 1.2,
              "ssm_multipliers": [0.9, 1.1, 0.8, 1.2, 1.3], "ssm_in_multiplier": 1.1, "ssm_out_multiplier": 0.6, "attention_bias": False,
              "mlp_bias": False, "tie_word_embeddings": False}
FALCON_MULT = {"ssm_in": 1.1, "ssm_out": 0.6, "attn_in": 1.2, "attn_out": 0.7, "key": 0.8, "mlp_gate": 1.5, "mlp_down": 0.5, "ssm_proj": [0.9, 1.1, 0.8, 1.2, 1.3]}
spec("falcon_h1", tok="qwen2", L=2, eps=1e-5, final_norm="final_layernorm.weight", pre_ff_norm="pre_ff_layernorm.weight",
     gate="feed_forward.gate_proj.weight", up="feed_forward.up_proj.weight", down="feed_forward.down_proj.weight",
     embed_scale=2.0, logit_scale=0.5, mult=FALCON_MULT, parallel_ssm=True,
     ssm=dict(MAMBA2_SSM, prefix="mamba.", norm_groups=2, norm_before_gate=True), ssm_layers=[1, 1], attn_layers=[1, 1],
     config=FALCON_CFG)
spec("falcon_h1_nonorm", tok="qwen2", L=2, eps=1e-5, final_norm="final_layernorm.weight", pre_ff_norm="pre_ff_layernorm.weight",
     gate="feed_forward.gate_proj.weight", up="feed_forward.up_proj.weight", down="feed_forward.down_proj.weight",
     embed_scale=2.0, logit_scale=0.5, mult=FALCON_MULT, parallel_ssm=True,
     ssm=dict(MAMBA2_SSM, prefix="mamba.", norm_groups=2, rms_norm=False), ssm_layers=[1, 1], attn_layers=[1, 1],
     config=dict(FALCON_CFG, mamba_rms_norm=False, mamba_norm_before_gate=False, time_step_limit=[0.0, INF]))
spec("jamba", tok="llama3", L=4, pos="none", final_norm="final_layernorm.weight", pre_ff_norm="pre_ff_layernorm.weight",
     gate="feed_forward.gate_proj.weight", up="feed_forward.up_proj.weight", down="feed_forward.down_proj.weight",
     ssm={"kind": "mamba1", "prefix": "mamba.", "inter": 64, "N": 8, "K": 4, "R": 4, "conv_bias": True, "proj_bias": False},
     ssm_layers=[1, 0, 1, 0], attn_layers=[0, 1, 0, 1],
     moe={"E": 4, "K": 2, "MI": 32, "shared": 0, "scoring": "softmax", "group_limited": False, "rsf": 1.0, "norm": False,
          "layers": [0, 2], "corr_bias": False, "layout": "separate", "prefix": "feed_forward.", "router": "router.weight"},
     config={"model_type": "jamba", "hidden_size": 32, "intermediate_size": 32, "num_hidden_layers": 4, "num_attention_heads": 4,
             "num_key_value_heads": 2, "hidden_act": "silu", "rms_norm_eps": 1e-6, "max_position_embeddings": 128, "num_experts": 4,
             "num_experts_per_tok": 2, "expert_layer_period": 2, "expert_layer_offset": 0, "attn_layer_period": 2, "attn_layer_offset": 1,
             "mamba_d_state": 8, "mamba_d_conv": 4, "mamba_expand": 2, "mamba_dt_rank": 4, "mamba_conv_bias": True, "mamba_proj_bias": False,
             "use_mamba_kernels": False, "tie_word_embeddings": False})
GRANITE_SSM = dict(MAMBA2_SSM, prefix="mamba.", G=1)
spec("granitemoehybrid", tok="starcoder", L=3, pos="none", embed_scale=2.0, attn_scale=0.25, residual_mult=0.5, logit_scale=0.25, lm_head=None,
     mlp="gated_fused", gate_up="shared_mlp.input_linear.weight", down="shared_mlp.output_linear.weight",
     ssm=GRANITE_SSM, ssm_layers=[1, 0, 1], attn_layers=[0, 1, 0],
     moe={"E": 4, "K": 2, "MI": 12, "shared": 1, "shared_inter": 16, "shared_fused": ("input_linear.weight", "output_linear.weight"), "shared_at_layer": True,
          "scoring": "softmax", "group_limited": False, "rsf": 1.0, "norm": True, "layers": [0, 1, 2], "corr_bias": False, "layout": "fused_rows",
          "prefix": "block_sparse_moe.", "router": "router.layer.weight", "fused_names": ("input_linear.weight", "output_linear.weight"),
          "shared_name": "shared_mlp."},
     config={"model_type": "granitemoehybrid", "hidden_size": 32, "intermediate_size": 12, "num_hidden_layers": 3, "num_attention_heads": 4,
             "num_key_value_heads": 2, "hidden_act": "silu", "rms_norm_eps": 1e-6, "max_position_embeddings": 128, "embedding_multiplier": 2.0,
             "logits_scaling": 4.0, "residual_multiplier": 0.5, "attention_multiplier": 0.25, "num_local_experts": 4, "num_experts_per_tok": 2,
             "shared_intermediate_size": 16, "position_embedding_type": None, "layers_block_type": ["mamba", "attention", "mamba"],
             "mamba_n_heads": 4, "mamba_n_groups": 1, "mamba_d_state": 8, "mamba_d_head": "auto", "mamba_d_conv": 4, "mamba_expand": 2,
             "mamba_chunk_size": 256, "mamba_conv_bias": True, "mamba_proj_bias": False, "time_step_limit": [0.0, INF],
             "attention_bias": False, "tie_word_embeddings": True})
spec("granitemoehybrid_dense", tok="starcoder", L=2, embed_scale=2.0, attn_scale=0.25, residual_mult=0.5, logit_scale=0.25, lm_head=None,
     mlp="gated_fused", gate_up="shared_mlp.input_linear.weight", down="shared_mlp.output_linear.weight",
     ssm=GRANITE_SSM, ssm_layers=[1, 0], attn_layers=[0, 1],
     config={"model_type": "granitemoehybrid", "hidden_size": 32, "intermediate_size": 12, "shared_intermediate_size": 32, "num_hidden_layers": 2,
             "num_attention_heads": 4, "num_key_value_heads": 2, "hidden_act": "silu", "rms_norm_eps": 1e-6, "rope_theta": 10000.0,
             "max_position_embeddings": 128, "embedding_multiplier": 2.0, "logits_scaling": 4.0, "residual_multiplier": 0.5,
             "attention_multiplier": 0.25, "num_local_experts": 0, "num_experts_per_tok": 0, "position_embedding_type": "rope",
             "layer_types": ["linear_attention", "full_attention"], "mamba_n_heads": 4, "mamba_n_groups": 1, "mamba_d_state": 8,
             "mamba_d_head": 16, "mamba_d_conv": 4, "mamba_expand": 2, "mamba_conv_bias": True, "mamba_proj_bias": False,
             "attention_bias": False, "tie_word_embeddings": True})
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
          "rsf": 1.5, "norm": True, "layers": [1, 2], "corr_bias": False, "layout": "fused_eih", "prefix": "mlp.", "router": "gate.weight",
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
# Gemma 2: the Gemma layout with (1 + w) norms, four norms per layer,
# alternating local (sliding) and global layers, `query_pre_attn_scalar`
# instead of 1/sqrt(head_dim), and tanh softcapping on both the attention
# logits and the output logits. The window (4) is shorter than the prompts, so
# the local mask bites, and the caps are small enough that the tanh is well
# inside its non-linear range.
spec("gemma2", tok="spm", L=3, NH=4, NKV=2, HD=8, lm_head=None, act="gelu_tanh", norm="rms1p",
     post_attn_norm="post_attention_layernorm.weight", pre_ff_norm="pre_feedforward_layernorm.weight",
     post_ff_norm="post_feedforward_layernorm.weight", embed_scale=np.sqrt(32.0),
     attn_scale=1.0 / np.sqrt(16.0), attn_softcap=1.0, final_softcap=20.0, sliding=4, sliding_layers=[1, 0, 1],
     config={"model_type": "gemma2", "hidden_size": 32, "intermediate_size": 32, "num_hidden_layers": 3,
             "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 8, "sliding_window": 4,
             "query_pre_attn_scalar": 16, "attn_logit_softcapping": 1.0, "final_logit_softcapping": 20.0,
             "rms_norm_eps": 1e-6, "rope_theta": 10000.0, "hidden_activation": "gelu_pytorch_tanh",
             "max_position_embeddings": 128, "tie_word_embeddings": True})
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


# Xiaomi MiMo V2: hybrid attention (full layers with NKV kv heads, sliding
# layers with twice as many plus attention sinks), values narrower than the
# keys and scaled by attention_value_scale, partial rotary with one base per
# layer type, a dense first layer then DeepSeek-V3-style sigmoid MoE without
# shared experts, and MTP tensors the reference never reads. `mimo_v2_flash`
# is the transformers spelling (layer_types, rope_parameters, stacked experts,
# `sinks`); `mimo_v2` the hub checkpoint spelling of MiMo-V2-Flash / V2.5 /
# V2.6 (hybrid_layer_pattern, swa_*, moe_layer_freq, `attention_sink_bias`,
# per-expert tensors) in the Pro layout (one qkv_proj per layer, chunked per
# kv head of the full layers) with the omni wrapper's vision and audio tensors.
MIMO_MTP = [("model.mtp.layers.0.enorm.weight", (32,)), ("model.mtp.layers.0.hnorm.weight", (32,)), ("model.mtp.layers.0.eh_proj.weight", (32, 64)),
            ("model.mtp.layers.0.input_layernorm.weight", (32,)), ("model.mtp.layers.0.pre_mlp_layernorm.weight", (32,)),
            ("model.mtp.layers.0.self_attn.o_proj.weight", (32, 32)), ("model.mtp.layers.0.mlp.gate_proj.weight", (32, 32)),
            ("model.mtp.layers.0.mlp.up_proj.weight", (32, 32)), ("model.mtp.layers.0.mlp.down_proj.weight", (32, 32)),
            ("model.mtp.layers.0.final_layernorm.weight", (32,))]
spec("mimo_v2_flash", tok="gpt2", L=4, NH=4, NKV=1, HD=12, VD=8, narrow_v=True, rotary_dim=4, eps=1e-5, theta=50000.0, local_rope=(10000.0, 4),
     lm_head="lm_head.weight", sinks="self_attn.sinks", sinks_sliding_only=True, sliding=4, sliding_layers=[0, 1, 1, 0], layer_nkv=[1, 2, 2, 1],
     mult={"value": 0.707},
     moe={"E": 4, "K": 2, "MI": 12, "shared": 0, "scoring": "sigmoid", "group_limited": True, "n_group": 2, "topk_group": 1, "rsf": 1.5, "norm": True,
          "layers": [1, 2, 3], "corr_bias": True, "layout": "fused_eih", "prefix": "mlp.", "router": "gate.weight"},
     extra_tensors=MIMO_MTP + [("model.mtp.layers.0.self_attn.q_proj.weight", (48, 32)), ("model.mtp.layers.0.self_attn.k_proj.weight", (24, 32)),
                               ("model.mtp.layers.0.self_attn.v_proj.weight", (16, 32)), ("model.mtp.layers.0.self_attn.sinks", (4,))],
     config={"model_type": "mimo_v2_flash", "architectures": ["MiMoV2FlashForCausalLM"], "hidden_size": 32, "intermediate_size": 32, "moe_intermediate_size": 12,
             "num_hidden_layers": 4, "num_attention_heads": 4, "num_key_value_heads": 1, "head_dim": 12, "v_head_dim": 8, "n_routed_experts": 4,
             "num_experts_per_tok": 2, "n_group": 2, "topk_group": 1, "norm_topk_prob": True, "routed_scaling_factor": 1.5, "sliding_window": 4,
             "layer_types": ["full_attention", "sliding_attention", "sliding_attention", "full_attention"],
             "mlp_layer_types": ["dense", "sparse", "sparse", "sparse"],
             "rope_parameters": {"full_attention": {"rope_type": "default", "rope_theta": 50000.0, "partial_rotary_factor": 0.334},
                                 "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0, "partial_rotary_factor": 0.334}},
             "attention_value_scale": 0.707, "rms_norm_eps": 1e-5, "hidden_act": "silu", "max_position_embeddings": 128, "tie_word_embeddings": False})
_MIMO_V2 = dict(tok="gpt2", L=4, NH=4, NKV=2, HD=12, VD=8, narrow_v=True, rotary_dim=4, eps=1e-5, theta=50000.0, local_rope=(10000.0, 4),
     lm_head="lm_head.weight", sinks="self_attn.attention_sink_bias", sinks_sliding_only=True, sliding=4, sliding_layers=[0, 1, 1, 0], layer_nkv=[2, 4, 4, 2],
     mult={"value": 0.707}, qkv="self_attn.qkv_proj.weight", qkv_layout="chunked", qkv_chunks=2,
     moe={"E": 4, "K": 2, "MI": 12, "shared": 0, "scoring": "sigmoid", "group_limited": True, "n_group": 2, "topk_group": 1, "rsf": 1.0, "norm": True,
          "layers": [1, 2, 3], "corr_bias": True, "layout": "separate", "prefix": "mlp.", "router": "gate.weight"},
     extra_tensors=MIMO_MTP + [("model.mtp.layers.0.self_attn.qkv_proj.weight", (48 + 4 * 12 + 4 * 8, 32)), ("model.mtp.layers.0.self_attn.attention_sink_bias", (4,)),
                               ("visual.patch_embed.proj.weight", (16, 3, 2, 4, 4)), ("visual.blocks.0.attn.qkv.weight", (48, 16)), ("visual.merger.mlp.0.weight", (32, 64)),
                               ("audio_encoder.conv1.weight", (16, 8, 3)), ("audio_encoder.projection.mlp.0.weight", (32, 16)), ("speech_embeddings.0.weight", (10, 32))])
_MIMO_V2_CONFIG = {"model_type": "mimo_v2", "architectures": ["MiMoV2ForCausalLM"], "hidden_size": 32, "intermediate_size": 32, "moe_intermediate_size": 12,
             "num_hidden_layers": 4, "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 12, "v_head_dim": 8,
             "swa_num_attention_heads": 4, "swa_num_key_value_heads": 4, "swa_head_dim": 12, "swa_v_head_dim": 8,
             "hybrid_layer_pattern": [0, 1, 1, 0], "sliding_window_size": 4, "sliding_window": 4, "add_swa_attention_sink_bias": True,
             "rope_theta": 50000.0, "swa_rope_theta": 10000.0, "partial_rotary_factor": 0.334, "attention_value_scale": 0.707,
             "n_routed_experts": 4, "num_experts_per_tok": 2, "n_group": 2, "topk_group": 1, "norm_topk_prob": True, "routed_scaling_factor": None,
             "scoring_func": "sigmoid", "topk_method": "noaux_tc", "moe_layer_freq": [0, 1, 1, 1], "first_k_dense_replace": 1, "moe_router_dtype": "float32",
             "layernorm_epsilon": 1e-5, "hidden_act": "silu", "max_position_embeddings": 128, "tie_word_embeddings": False,
             "vision_config": {"model_type": "mimovl", "depth": 1, "hidden_size": 16, "num_heads": 2},
             "audio_config": {"audio_channels": 1, "input_local_layers": 1, "group_size": 4}}
spec("mimo_v2", config=_MIMO_V2_CONFIG, **_MIMO_V2)
# MiMo-V2.6 as released: the routed experts are MXFP4 (`store_dtype`, U8
# `weight` / `weight_scale` per expert projection) while the rest of the
# checkpoint stays bf16 (`ignored_layers`), and the MoE router is bf16
# (`moe_router_dtype`). The expert input dimension must be a multiple of the
# 32-element group, so the routed experts are wider here than in `mimo_v2`.
_MIMO_V2_MXFP4_QUANT = {"activation_scheme": "dynamic", "fmt": "e4m3", "ignored_layers": ["model.layers.*.self_attn", "model.layers.*.mlp.gate", "model.embed_tokens", "lm_head"],
                        "mxfp4_block_size": 32, "quant_method": "fp8", "store_dtype": "mxfp4", "weight_block_size": [128, 128]}
spec("mimo_v2_mxfp4", quant={"kind": "mxfp4_store"},
     config=dict(_MIMO_V2_CONFIG, moe_intermediate_size=32, moe_router_dtype="bfloat16", quantization_config=_MIMO_V2_MXFP4_QUANT),
     **dict(_MIMO_V2, moe=dict(_MIMO_V2["moe"], MI=32)))

# --- families swept from transformers' causal-LM mapping --------------------
# Dense llama-layout variants: each differs in its MLP shape, norm placement,
# rotary pairing or positional encoding.
spec("arcee", tok="llama3", mlp="dense", up="mlp.up_proj.weight", act="relu2", mlp_bias=True, lm_head=None,
     config={"model_type": "arcee", "hidden_size": 32, "intermediate_size": 32, "num_hidden_layers": 2, "num_attention_heads": 4,
             "num_key_value_heads": 2, "rms_norm_eps": 1e-6, "rope_theta": 10000.0, "hidden_act": "relu2", "mlp_bias": True,
             "max_position_embeddings": 128, "tie_word_embeddings": True})
spec("jais2", tok="gpt2", norm="ln", eps=1e-5, mlp="dense", up="mlp.up_proj.weight", act="relu2", mlp_bias=True,
     attn_bias=True, o_bias=True, lm_head="lm_head.weight",
     config={"model_type": "jais2", "hidden_size": 32, "intermediate_size": 32, "num_hidden_layers": 2, "num_attention_heads": 4,
             "num_key_value_heads": 2, "head_dim": 8, "layer_norm_eps": 1e-5, "rope_theta": 10000.0, "hidden_act": "relu2",
             "attention_bias": True, "mlp_bias": True, "max_position_embeddings": 128, "tie_word_embeddings": False})
spec("ernie4_5", tok="spm", rope_style="gptj", attn_bias=True, o_bias=True, mlp_bias=True, lm_head="lm_head.weight", eps=1e-5,
     config={"model_type": "ernie4_5", "hidden_size": 32, "intermediate_size": 32, "num_hidden_layers": 2, "num_attention_heads": 4,
             "num_key_value_heads": 2, "head_dim": 8, "rms_norm_eps": 1e-5, "rope_theta": 10000.0, "hidden_act": "silu",
             "use_bias": True, "max_position_embeddings": 128, "tie_word_embeddings": False})
spec("helium", tok="llama3", NKV=4, rope_style="gptj", attn_bias=True, o_bias=False, mlp_bias=True,
     lm_head="lm_head.weight", eps=1e-8,
     config={"model_type": "helium", "hidden_size": 32, "intermediate_size": 32, "num_hidden_layers": 2, "num_attention_heads": 4,
             "num_key_value_heads": 4, "head_dim": 8, "rms_norm_eps": 1e-8, "rope_theta": 10000.0, "hidden_act": "silu",
             "attention_bias": True, "mlp_bias": True, "max_position_embeddings": 128, "tie_word_embeddings": False})
spec("hunyuan_v1_dense", tok="gpt2", qk_norm="head", qk_norm_after_rope=True, eps=1e-5, lm_head="lm_head.weight",
     q_norm="self_attn.query_layernorm.weight", k_norm="self_attn.key_layernorm.weight",
     theta=10000.0 * 1000.0 ** (8 / 6),
     config={"model_type": "hunyuan_v1_dense", "hidden_size": 32, "intermediate_size": 32, "num_hidden_layers": 2,
             "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 8, "rms_norm_eps": 1e-5, "rope_theta": 10000.0,
             "rope_scaling": {"type": "dynamic", "alpha": 1000.0}, "hidden_act": "silu", "attention_bias": False,
             "max_position_embeddings": 128, "tie_word_embeddings": False})
spec("bitnet", tok="llama3", act="relu2", attn_sub_norm="self_attn.attn_sub_norm.weight", ffn_sub_norm="mlp.ffn_sub_norm.weight",
     lm_head="lm_head.weight", eps=1e-5,
     config={"model_type": "bitnet", "hidden_size": 32, "intermediate_size": 32, "num_hidden_layers": 2, "num_attention_heads": 4,
             "num_key_value_heads": 2, "rms_norm_eps": 1e-5, "rope_theta": 10000.0, "hidden_act": "relu2",
             "max_position_embeddings": 128, "tie_word_embeddings": False})
spec("apertus", tok="llama3", qk_norm="head", mlp="dense", up="mlp.up_proj.weight", xielu=(0.8, 0.8), eps=1e-5,
     in_norm="attention_layernorm.weight", pre_ff_norm="feedforward_layernorm.weight", lm_head="lm_head.weight",
     config={"model_type": "apertus", "hidden_size": 32, "intermediate_size": 32, "num_hidden_layers": 2, "num_attention_heads": 4,
             "num_key_value_heads": 2, "rms_norm_eps": 1e-5, "rope_theta": 10000.0, "hidden_act": "xielu",
             "max_position_embeddings": 128, "tie_word_embeddings": False})
spec("nanochat", tok="gpt2", norm="rms_none", qk_norm="weightless", qk_norm_after_rope=True, embed_norm="norm.weight",
     mlp="dense", up="mlp.fc1.weight", down="mlp.fc2.weight", act="relu2", final_softcap=15.0, lm_head="lm_head.weight",
     config={"model_type": "nanochat", "hidden_size": 32, "intermediate_size": 32, "num_hidden_layers": 2, "num_attention_heads": 4,
             "num_key_value_heads": 2, "rms_norm_eps": 1e-6, "rope_theta": 10000.0, "hidden_act": "relu2",
             "final_logit_softcapping": 15.0, "max_position_embeddings": 128, "tie_word_embeddings": False})
spec("persimmon", tok="spm", NKV=4, norm="ln", eps=1e-5, qkv="self_attn.query_key_value.weight", qkv_layout="heads",
     o="self_attn.dense.weight", attn_bias=True, o_bias=True, final_norm="final_layernorm.weight",
     mlp="dense", up="mlp.dense_h_to_4h.weight", down="mlp.dense_4h_to_h.weight", mlp_bias=True, act="relu2",
     qk_norm="head", q_norm="self_attn.q_layernorm.weight", k_norm="self_attn.k_layernorm.weight", rotary_dim=4,
     lm_head="lm_head.weight",
     config={"model_type": "persimmon", "hidden_size": 32, "intermediate_size": 32, "num_hidden_layers": 2,
             "num_attention_heads": 4, "layer_norm_eps": 1e-5, "partial_rotary_factor": 0.5, "rope_theta": 10000.0,
             "hidden_act": "relu2", "qk_layernorm": True, "max_position_embeddings": 128, "tie_word_embeddings": False})
spec("gptj", tok="gpt2", NKV=4, I=64, prefix="transformer.", layer="h.{i}.", embed="wte.weight", final_norm="ln_f.weight",
     norm="ln", eps=1e-5, parallel=True, pre_ff_norm=None, in_norm="ln_1.weight",
     q="attn.q_proj.weight", k="attn.k_proj.weight", v="attn.v_proj.weight", o="attn.out_proj.weight",
     mlp="dense", up="mlp.fc_in.weight", down="mlp.fc_out.weight", mlp_bias=True, act="gelu_new",
     rope_style="gptj", rotary_dim=4, lm_bias=True, lm_head="lm_head.weight",
     config={"model_type": "gptj", "n_embd": 32, "n_layer": 2, "n_head": 4, "n_positions": 128, "n_inner": 64, "rotary_dim": 4,
             "activation_function": "gelu_new", "layer_norm_epsilon": 1e-5, "vocab_size": 0, "tie_word_embeddings": False})
spec("codegen", tok="gpt2", NKV=4, I=64, prefix="transformer.", layer="h.{i}.", embed="wte.weight", final_norm="ln_f.weight",
     norm="ln", eps=1e-5, parallel=True, pre_ff_norm=None, in_norm="ln_1.weight",
     qkv="attn.qkv_proj.weight", qkv_layout="mp", qkv_mp=4, o="attn.out_proj.weight",
     mlp="dense", up="mlp.fc_in.weight", down="mlp.fc_out.weight", mlp_bias=True, act="gelu_new",
     rope_style="gptj", rotary_dim=4, lm_bias=True, lm_head="lm_head.weight",
     config={"model_type": "codegen", "n_embd": 32, "n_layer": 2, "n_head": 4, "n_positions": 128, "n_ctx": 128, "n_inner": 64,
             "rotary_dim": 4, "activation_function": "gelu_new", "layer_norm_epsilon": 1e-5, "vocab_size": 0,
             "tie_word_embeddings": False})
spec("gpt_neo", tok="gpt2", NKV=4, I=64, prefix="transformer.", layer="h.{i}.", embed="wte.weight", pos_embed="wpe.weight",
     final_norm="ln_f.weight", norm="ln", eps=1e-5, in_norm="ln_1.weight", pre_ff_norm="ln_2.weight", pos="learned",
     q="attn.attention.q_proj.weight", k="attn.attention.k_proj.weight", v="attn.attention.v_proj.weight",
     o="attn.attention.out_proj.weight", attn_bias=False, o_bias=True, attn_scale=1.0,
     mlp="dense", up="mlp.c_fc.weight", down="mlp.c_proj.weight", mlp_bias=True, act="gelu_new",
     sliding=4, sliding_layers=[0, 1], lm_head=None,
     config={"model_type": "gpt_neo", "hidden_size": 32, "num_layers": 2, "num_heads": 4, "intermediate_size": 64,
             "max_position_embeddings": 64, "window_size": 4, "attention_types": [[["global", "local"], 1]],
             "activation_function": "gelu_new", "layer_norm_epsilon": 1e-5, "vocab_size": 0, "tie_word_embeddings": True})
spec("xglm", tok="spm", NKV=4, pos="sinusoidal", pos_offset=2, embed_scale=float(np.sqrt(32.0)), norm="ln", eps=1e-5,
     in_norm="self_attn_layer_norm.weight", pre_ff_norm="final_layer_norm.weight", final_norm="layer_norm.weight",
     o="self_attn.out_proj.weight", attn_bias=True, o_bias=True,
     mlp="dense", up="fc1.weight", down="fc2.weight", mlp_bias=True, act="gelu", lm_head=None,
     config={"model_type": "xglm", "d_model": 32, "ffn_dim": 32, "num_layers": 2, "attention_heads": 4,
             "max_position_embeddings": 128, "activation_function": "gelu", "scale_embedding": True, "vocab_size": 0,
             "tie_word_embeddings": True})
spec("biogpt", tok="spm", NKV=4, prefix="biogpt.", pos_embed="embed_positions.weight", pos="learned", pos_offset=2,
     embed_scale=float(np.sqrt(32.0)), norm="ln", eps=1e-5, in_norm="self_attn_layer_norm.weight",
     pre_ff_norm="final_layer_norm.weight", final_norm="layer_norm.weight", o="self_attn.out_proj.weight",
     attn_bias=True, o_bias=True, mlp="dense", up="fc1.weight", down="fc2.weight", mlp_bias=True, act="gelu",
     lm_head="output_projection.weight",
     config={"model_type": "biogpt", "hidden_size": 32, "intermediate_size": 32, "num_hidden_layers": 2,
             "num_attention_heads": 4, "max_position_embeddings": 64, "hidden_act": "gelu", "scale_embedding": True,
             "vocab_size": 0, "tie_word_embeddings": False})
spec("ministral3", tok="llama3", temp={"floor_scale": 4.0, "attn_scale": 0.5, "offset": 0.0, "all": True},
     lm_head="lm_head.weight",
     config={"model_type": "ministral3", "hidden_size": 32, "intermediate_size": 32, "num_hidden_layers": 2,
             "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 8, "rms_norm_eps": 1e-6, "hidden_act": "silu",
             "rope_parameters": {"rope_type": "default", "rope_theta": 10000.0, "llama_4_scaling_beta": 0.5},
             "max_position_embeddings": 4, "tie_word_embeddings": False})
spec("granite_swa", tok="starcoder", L=3, embed_scale=2.0, attn_scale=0.25, residual_mult=0.5, logit_scale=0.25,
     sinks="self_attn.sinks", sliding=4, sliding_layers=[1, 0, 1], local_rope=(50000.0, 8), lm_head=None,
     config={"model_type": "granite_swa", "hidden_size": 32, "intermediate_size": 32, "num_hidden_layers": 3,
             "num_attention_heads": 4, "num_key_value_heads": 2, "rms_norm_eps": 1e-6, "rope_theta": 10000.0,
             "layer_rope_theta": {"full_attention": 10000.0, "sliding_attention": 50000.0},
             "layer_types": ["sliding_attention", "full_attention", "sliding_attention"], "sliding_window": 4,
             "embedding_multiplier": 2.0, "attention_multiplier": 0.25, "residual_multiplier": 0.5, "logits_scaling": 4.0,
             "hidden_act": "silu", "max_position_embeddings": 128, "tie_word_embeddings": True})

# Mixture-of-experts families. `DS_ROUTER` is the DeepSeek-V3 router shared by
# dots.llm1, EXAONE-MoE and Solar Open: sigmoid scores, a correction bias that
# only steers the choice, group-limited top-k, renormalisation and a routed
# scaling factor, plus always-on shared experts.
DS_ROUTER = {"E": 4, "K": 2, "MI": 12, "shared": 1, "shared_inter": 16, "scoring": "sigmoid", "group_limited": True,
             "n_group": 2, "topk_group": 1, "rsf": 2.5, "norm": True, "corr_bias": True, "layout": "separate",
             "prefix": "mlp.", "router": "gate.weight", "shared_name": "shared_experts."}
spec("olmoe", tok="llama3", L=2, qk_norm="full", clip=0.8, lm_head="lm_head.weight", eps=1e-5,
     moe={"E": 4, "K": 2, "MI": 12, "shared": 0, "scoring": "softmax", "group_limited": False, "rsf": 1.0, "norm": True,
          "layers": [0, 1], "corr_bias": False, "layout": "separate", "prefix": "mlp.", "router": "gate.weight"},
     config={"model_type": "olmoe", "hidden_size": 32, "intermediate_size": 12, "num_hidden_layers": 2, "num_attention_heads": 4,
             "num_key_value_heads": 2, "num_experts": 4, "num_experts_per_tok": 2, "norm_topk_prob": True, "clip_qkv": 0.8,
             "rms_norm_eps": 1e-5, "rope_theta": 10000.0, "hidden_act": "silu", "max_position_embeddings": 128,
             "tie_word_embeddings": False})
spec("flex_olmo", tok="llama3", L=2, qk_norm="full", in_norm=None, post_attn_norm="post_attention_layernorm.weight",
     pre_ff_norm=None, post_ff_norm="post_feedforward_layernorm.weight", lm_head="lm_head.weight",
     moe={"E": 4, "K": 2, "MI": 12, "shared": 0, "scoring": "softmax", "group_limited": False, "rsf": 1.0, "norm": False,
          "layers": [0, 1], "corr_bias": False, "layout": "fused", "prefix": "mlp.", "router": "gate.weight"},
     config={"model_type": "flex_olmo", "hidden_size": 32, "intermediate_size": 12, "num_hidden_layers": 2,
             "num_attention_heads": 4, "num_key_value_heads": 2, "num_experts": 4, "num_experts_per_tok": 2,
             "norm_topk_prob": False, "rms_norm_eps": 1e-6, "rope_theta": 10000.0, "hidden_act": "silu",
             "max_position_embeddings": 128, "tie_word_embeddings": False})
spec("dots1", tok="qwen2", L=3, qk_norm="head", sliding=4, sliding_layers=[1, 1, 0], lm_head="lm_head.weight",
     moe=dict(DS_ROUTER, layers=[1, 2]),
     config={"model_type": "dots1", "hidden_size": 32, "intermediate_size": 32, "moe_intermediate_size": 12,
             "num_hidden_layers": 3, "num_attention_heads": 4, "num_key_value_heads": 2, "n_routed_experts": 4,
             "n_shared_experts": 1, "num_experts_per_tok": 2, "first_k_dense_replace": 1, "n_group": 2, "topk_group": 1,
             "routed_scaling_factor": 2.5, "norm_topk_prob": True, "sliding_window": 4,
             "layer_types": ["sliding_attention", "sliding_attention", "full_attention"],
             "rms_norm_eps": 1e-6, "rope_theta": 10000.0, "hidden_act": "silu", "max_position_embeddings": 128,
             "tie_word_embeddings": False})
spec("exaone_moe", tok="llama3", L=4, qk_norm="head", sliding=4, sliding_layers=[1, 1, 1, 0], lm_head="lm_head.weight",
     moe=dict(DS_ROUTER, layers=[1, 2, 3]),
     config={"model_type": "exaone_moe", "hidden_size": 32, "intermediate_size": 32, "moe_intermediate_size": 12,
             "num_hidden_layers": 4, "num_attention_heads": 4, "num_key_value_heads": 2, "num_experts": 4,
             "num_shared_experts": 1, "num_experts_per_tok": 2, "n_group": 2, "topk_group": 1, "routed_scaling_factor": 2.5,
             "norm_topk_prob": True, "sliding_window": 4, "sliding_window_pattern": 4,
             "mlp_layer_types": ["dense", "sparse", "sparse", "sparse"],
             "rms_norm_eps": 1e-5, "rope_theta": 10000.0, "hidden_act": "silu", "max_position_embeddings": 128,
             "tie_word_embeddings": False})
spec("solar_open", tok="llama3", L=2, rotary_dim=4, lm_head="lm_head.weight",
     moe=dict(DS_ROUTER, layers=[0, 1], layout="fused"),
     config={"model_type": "solar_open", "hidden_size": 32, "moe_intermediate_size": 12, "num_hidden_layers": 2,
             "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 8, "n_routed_experts": 4, "n_shared_experts": 1,
             "num_experts_per_tok": 2, "n_group": 2, "topk_group": 1, "routed_scaling_factor": 2.5, "norm_topk_prob": True,
             "rms_norm_eps": 1e-5, "rope_theta": 10000.0, "rope_parameters": {"rope_type": "default", "rope_theta": 10000.0,
                                                                              "partial_rotary_factor": 0.5},
             "hidden_act": "silu", "max_position_embeddings": 128, "tie_word_embeddings": False})
spec("afmoe", tok="qwen2", L=3, qk_norm="head", attn_gate=("self_attn.gate_proj.weight", "sigmoid", False),
     post_attn_norm="post_attention_layernorm.weight", pre_ff_norm="pre_mlp_layernorm.weight",
     post_ff_norm="post_mlp_layernorm.weight", sliding=4, sliding_layers=[1, 0, 1], lm_head="lm_head.weight",
     moe={"E": 4, "K": 2, "MI": 12, "shared": 1, "shared_inter": 24, "scoring": "sigmoid", "group_limited": False,
          "rsf": 1.5, "norm": True, "layers": [1, 2], "corr_bias": True, "corr_bias_name": "mlp.expert_bias",
          "layout": "separate", "prefix": "mlp.", "router": "router.gate.weight", "shared_name": "shared_experts."},
     config={"model_type": "afmoe", "hidden_size": 32, "intermediate_size": 32, "moe_intermediate_size": 12,
             "num_hidden_layers": 3, "num_dense_layers": 1, "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 8,
             "num_experts": 4, "num_experts_per_tok": 2, "num_shared_experts": 2, "route_scale": 1.5,
             "global_attn_every_n_layers": 2, "sliding_window": 4,
             "layer_types": ["sliding_attention", "full_attention", "sliding_attention"],
             "rms_norm_eps": 1e-5, "rope_theta": 10000.0, "hidden_act": "silu", "max_position_embeddings": 128,
             "tie_word_embeddings": False})
spec("mellum", tok="gpt2", L=2, qk_norm="head", lm_head="lm_head.weight",
     moe={"E": 4, "K": 2, "MI": 12, "shared": 0, "scoring": "softmax", "group_limited": False, "rsf": 1.0, "norm": True,
          "layers": [0, 1], "corr_bias": False, "layout": "fused", "prefix": "mlp.", "router": "gate.weight"},
     config={"model_type": "mellum", "hidden_size": 32, "intermediate_size": 32, "moe_intermediate_size": 12,
             "num_hidden_layers": 2, "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 8, "num_experts": 4,
             "num_experts_per_tok": 2, "norm_topk_prob": True, "mlp_layer_types": ["sparse", "sparse"],
             "layer_types": ["full_attention", "full_attention"],
             "rms_norm_eps": 1e-6, "rope_parameters": {"full_attention": {"rope_type": "default", "rope_theta": 10000.0}},
             "hidden_act": "silu", "max_position_embeddings": 128, "tie_word_embeddings": False})
spec("laguna", tok="llama3", L=3, qk_norm="head", attn_gate=("self_attn.g_proj.weight", "softplus", True),
     sliding=4, sliding_layers=[1, 0, 1], lm_head="lm_head.weight",
     moe={"E": 4, "K": 2, "MI": 12, "shared": 1, "shared_inter": 16, "scoring": "sigmoid", "group_limited": False,
          "rsf": 2.0, "norm": True, "softcap": 5.0, "layers": [1, 2], "corr_bias": True, "layout": "separate",
          "prefix": "mlp.", "router": "gate.weight", "shared_name": "shared_experts."},
     config={"model_type": "laguna", "hidden_size": 32, "intermediate_size": 32, "moe_intermediate_size": 12,
             "shared_expert_intermediate_size": 16, "num_hidden_layers": 3, "num_attention_heads": 4,
             "num_key_value_heads": 2, "head_dim": 8, "num_experts": 4, "num_experts_per_tok": 2, "gating": "per-head",
             "moe_routed_scaling_factor": 2.0, "moe_router_logit_softcapping": 5.0, "sliding_window": 4,
             "layer_types": ["sliding_attention", "full_attention", "sliding_attention"],
             "mlp_layer_types": ["dense", "sparse", "sparse"],
             "rms_norm_eps": 1e-6, "rope_theta": 10000.0, "hidden_act": "silu", "max_position_embeddings": 128,
             "tie_word_embeddings": False})
spec("hy_v3", tok="gpt2", L=3, qk_norm="head", lm_head="lm_head.weight", eps=1e-5,
     moe={"E": 4, "K": 2, "MI": 12, "shared": 1, "shared_inter": 12, "scoring": "sigmoid", "group_limited": False,
          "rsf": 2.826, "norm": True, "layers": [1, 2], "corr_bias": True, "corr_bias_name": "mlp.expert_bias",
          "layout": "separate", "prefix": "mlp.", "router": "router.gate.weight", "shared_name": "shared_mlp."},
     config={"model_type": "hy_v3", "hidden_size": 32, "intermediate_size": 32, "moe_intermediate_size": 12,
             "num_hidden_layers": 3, "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 8, "num_experts": 4,
             "num_experts_per_tok": 2, "num_shared_experts": 1, "router_scaling_factor": 2.826,
             "mlp_layer_types": ["dense", "sparse", "sparse"],
             "rms_norm_eps": 1e-5, "rope_theta": 10000.0, "hidden_act": "silu", "max_position_embeddings": 128,
             "tie_word_embeddings": False})

# --- generation ------------------------------------------------------------

def generate_generic(family, out_dir):
    s = SPECS[family]
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(4321)
    vocab, encode, bos, eos = make_tokenizer(s["tok"], out_dir)
    V = len(vocab)
    H, I, L, NH, NKV, HD, VD = s["H"], s["I"], s["L"], s["NH"], s["NKV"], s["HD"], s["VD"]
    weights = {}
    # Quantised storage of a weight: name -> [(tensor name, safetensors dtype, array)];
    # `weights[name]` then holds the dequantised values the reference computes with.
    qtensors = {}

    def quantize(name, w):
        """Re-encodes `w` in the spec's quantised format (in place) and records its storage tensors."""
        q = s["quant"]
        if q["kind"] == "fp8":
            deq, tensors = quant_fp8_block(name, w, q["block"])
        elif q["kind"] == "int":
            gi = None
            if q.get("actorder"):
                # A fixed permutation of the columns over the groups, so every
                # group's columns are scattered (what `actorder` produces).
                perm = np.random.default_rng(7).permutation(w.shape[1])
                gi = np.empty(w.shape[1], np.int64)
                gi[perm] = np.arange(w.shape[1]) // q["group"]
            deq, tensors = quant_int_packed(name, w, q["bits"], q["group"], q["symmetric"], gi)
        elif q["kind"] == "mxfp4":
            deq, tensors = quant_mxfp4(name, w)
        elif q["kind"] == "mxfp4_packed":
            deq, tensors = quant_mxfp4_packed(name, w)
        elif q["kind"] == "mxfp4_store":
            deq, tensors = quant_mxfp4_store(name, w)
        else:
            raise ValueError(q["kind"])
        w[...] = deq
        qtensors[name] = tensors

    def mat(name, rows, cols, scale=0.2, register=True):
        w = bf16_round(rng.normal(0, scale, size=(rows, cols)))
        if register:
            weights[name] = w.T.copy() if s["conv1d"] and name.endswith(".weight") and "wte" not in name and "wpe" not in name and "ln" not in name else w
            if s["quant"] and s["quant"]["kind"] in ("fp8", "int") and name.endswith("_proj.weight") and not s["conv1d"]:
                quantize(name, w)
        return w

    def vec(name, n, scale=0.2):
        w = bf16_round(rng.normal(0, scale, size=(n,)))
        weights[name] = w
        return w

    def normw(name, n, plain=False):
        """Norm weight (stored as w - 1 for the (1 + w) families) and, for LayerNorm families, its bias.

        `plain` is for the gated output norms of the linear-attention blocks,
        which stay `x * w` even on the (1 + w) families (transformers'
        `RMSNormGated` against its `RMSNorm`).
        """
        if name is None or name == "" or s["norm"] == "rms_none":
            return None
        w = bf16_round(1.0 + rng.normal(0, 0.1, size=(n,)))
        stored = w - 1.0 if (not plain and s["norm"] in ("rms1p", "ln1p")) else w
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

    def res_scorer(prefix):
        """Attention Residual scorer: an RMSNorm weight and a [1, H] projection."""
        return {"norm": normw(prefix + "norm.weight", H)[0], "proj": mat(prefix + "proj.weight", 1, H, 0.5)[0]}

    output_res = res_scorer(P + "output_attn_res_") if s["attn_res"] else None
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
    ssm_layers = [bool(x) for x in (s["ssm_layers"] or [0] * L)]
    attn_layers = [bool(x) for x in s["attn_layers"]] if s["attn_layers"] is not None else [not (lin_layers[i] or ssm_layers[i] or conv_layers[i]) for i in range(L)]
    mlp_layers = [bool(x) for x in (s["mlp_layers"] or [1] * L)]

    def raw(name, arr):
        """Registers an explicit (bf16-rounded) array."""
        w = bf16_round(np.asarray(arr, dtype=np.float32))
        weights[name] = w
        return w

    def ssm_weights(lp):
        """Weights of one Mamba block under `lp`."""
        ss = s["ssm"]
        sp = lp + ss["prefix"]
        d = {}
        if ss["kind"] == "mamba2":
            inter, gn = ss["heads"] * ss["hd"], ss["G"] * ss["N"]
            conv_dim = inter + 2 * gn
            in_rows = inter + conv_dim + ss["heads"]
            d["dt_bias"] = raw(sp + "dt_bias", ss["dt_bias_mean"] + rng.normal(0, 1.0, size=(ss["heads"],)))
            d["A_log"] = raw(sp + "A_log", np.log(1.0 + rng.uniform(0, 3, size=(ss["heads"],))))
            d["D"] = raw(sp + "D", 1.0 + rng.normal(0, 0.2, size=(ss["heads"],)))
            d["norm"] = normw(sp + "norm.weight", inter)[0] if ss["rms_norm"] else None
        else:
            inter, conv_dim = ss["inter"], ss["inter"]
            in_rows = 2 * inter
            d["x_proj"] = mat(sp + "x_proj.weight", ss["R"] + 2 * ss["N"], inter)
            d["dt_proj"] = mat(sp + "dt_proj.weight", inter, ss["R"], 0.5)
            d["dt_bias"] = raw(sp + "dt_proj.bias", rng.normal(-1.0, 1.0, size=(inter,)))
            d["A_log"] = raw(sp + "A_log", np.log(np.arange(1, ss["N"] + 1, dtype=np.float32))[None, :] * (1.0 + rng.normal(0, 0.1, size=(inter, 1))))
            d["D"] = raw(sp + "D", 1.0 + rng.normal(0, 0.2, size=(inter,)))
            d["dt_norm"] = normw(sp + "dt_layernorm.weight", ss["R"])[0]
            d["b_norm"] = normw(sp + "b_layernorm.weight", ss["N"])[0]
            d["c_norm"] = normw(sp + "c_layernorm.weight", ss["N"])[0]
        d["in_proj"] = mat(sp + "in_proj.weight", in_rows, H)
        d["in_b"] = bias_for(sp + "in_proj.weight", in_rows, ss["proj_bias"])
        conv = bf16_round(rng.normal(0, 0.3, size=(conv_dim, 1, ss["K"])))
        weights[sp + "conv1d.weight"] = conv  # the Hugging Face [channels, 1, kernel] layout
        d["conv"] = conv[:, 0, :]
        d["conv_b"] = vec(sp + "conv1d.bias", conv_dim, 0.1) if ss["conv_bias"] else None
        d["out"] = mat(sp + "out_proj.weight", H, inter)
        d["out_b"] = bias_for(sp + "out_proj.weight", H, ss["proj_bias"])
        return d

    for i in range(L):
        lp = P + s["layer"].format(i=i)
        d = {}
        hd_l, nkv_l = layer_hd[i], layer_nkv[i]
        vd_l = VD if (s["mla"] or s["narrow_v"]) else hd_l
        d["in_norm"] = normw(lp + s["in_norm"], H) if s["in_norm"] else None
        d["post_attn_norm"] = normw(lp + s["post_attn_norm"], H) if s["post_attn_norm"] else None
        d["pre_ff_norm"] = normw(lp + s["pre_ff_norm"], H) if s["pre_ff_norm"] else None
        d["post_ff_norm"] = normw(lp + s["post_ff_norm"], H) if s["post_ff_norm"] else None
        d["mlp_norm"] = normw(lp + s["mlp_norm"], H) if s["mlp_norm"] else None
        qd, kvd = NH * hd_l, nkv_l * hd_l
        for name, shape in s["extra_layer_tensors"].get(i, []):
            weights[lp + name] = bf16_round(rng.normal(0, 0.2, size=shape))
        if conv_layers[i]:
            d["conv_in"] = mat(lp + "conv.in_proj.weight", 3 * H, H)
            d["conv_w"] = bf16_round(rng.normal(0, 0.3, size=(H, s["conv_K"])))
            weights[lp + "conv.conv.weight"] = d["conv_w"].reshape(H, 1, s["conv_K"])
            d["o"] = mat(lp + "conv.out_proj.weight", H, H)
            d["ob"] = None
        elif lin_layers[i] and s["linear_kind"] == "lightning":
            d["light_qkv"] = mat(lp + "self_attn.qkv_proj.weight", 3 * qd, H)
            d["light_gate"] = mat(lp + "self_attn.output_gate.weight", qd, H)
            d["light_norm"] = normw(lp + "self_attn.norm.weight", qd)[0]
            d["o"] = mat(lp + "self_attn.out_proj.weight", H, qd)
            d["ob"] = None
        elif lin_layers[i] and s["linear"].get("kind") == "kda":
            ln, ap = s["linear"], "self_attn."
            NHl, D, KC = ln["VH"], ln["VD"], ln["KC"]
            dim = NHl * D
            hf = ln["layout"] == "hf"
            fp = ap + ("forget_gate." if hf else "")
            d["q"] = mat(lp + ap + "q_proj.weight", dim, H)
            d["k"] = mat(lp + ap + "k_proj.weight", dim, H)
            d["v"] = mat(lp + ap + "v_proj.weight", dim, H)
            # nn.Conv1d weights are [channels, 1, kernel]: one fused tensor over q | k | v, or one per projection.
            parts = [bf16_round(rng.normal(0, 0.2, size=(dim, KC))) for _ in range(3)]
            if hf:
                weights[lp + ap + "conv1d.weight"] = np.concatenate(parts, 0).reshape(3 * dim, 1, KC)
            else:
                for nm, w in zip(("q", "k", "v"), parts):
                    weights[lp + ap + f"{nm}_conv1d.weight"] = w.reshape(dim, 1, KC)
            d["conv"] = np.concatenate(parts, 0)
            d["f_a"] = mat(lp + fp + "f_a_proj.weight", D, H)
            d["f_b"] = mat(lp + fp + "f_b_proj.weight", dim, D)
            d["dt"] = vec(lp + fp + "dt_bias", dim, 0.5)
            d["alog"] = vec(lp + fp + "A_log", NHl, 0.5)
            weights[lp + fp + "A_log"] = d["alog"].reshape(1, 1, NHl, 1)
            d["b"] = mat(lp + ap + "b_proj.weight", NHl, H)
            if ln.get("full_rank_gate"):
                d["g"] = mat(lp + ap + "g_proj.weight", dim, H)
            else:
                d["g_a"] = mat(lp + ap + "g_a_proj.weight", D, H)
                d["g_b"] = mat(lp + ap + "g_b_proj.weight", dim, D)
            d["lnorm"] = normw(lp + ap + "o_norm.weight", D, plain=True)[0]
            d["o"] = mat(lp + ap + "o_proj.weight", H, dim)
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
            d["lnorm"] = normw(lp + ap + "norm.weight", ln["VD"], plain=True)[0]
            d["o"] = mat(lp + ap + "out_proj.weight", H, vd_tot)
            d["ob"] = None
        elif not attn_layers[i]:
            pass  # a Mamba, MLP or MoE block without attention
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
            if s["mla_gate"]:
                d["attn_gate"] = mat(lp + "self_attn.g_proj.weight", NH * m["v"], H)
        elif s["qkv"] and s["qkv_layout"] == "chunked":
            # MiMo V2 Pro: `qkv_chunks` chunks of [q heads | k heads | v heads] (the
            # checkpoint's tensor-parallel shards); the reference reads the parts.
            d["q"] = mat("q", qd, H, register=False)
            d["k"] = mat("k", kvd, H, register=False)
            d["v"] = mat("v", nkv_l * vd_l, H, register=False)
            d["qb"] = d["kb"] = d["vb"] = None
            nc = s["qkv_chunks"]
            qpc, kpc = NH // nc, nkv_l // nc
            weights[lp + s["qkv"]] = np.concatenate(
                [np.concatenate([d["q"][c * qpc * hd_l:(c + 1) * qpc * hd_l], d["k"][c * kpc * hd_l:(c + 1) * kpc * hd_l],
                                 d["v"][c * kpc * vd_l:(c + 1) * kpc * vd_l]], 0) for c in range(nc)], 0)
        elif s["qkv"]:
            w = mat(lp + s["qkv"], qd + 2 * kvd, H)
            b = bias_for(lp + s["qkv"], qd + 2 * kvd, s["attn_bias"])
            d["qkv"], d["qkv_b"] = w, b
        else:
            d["q"] = mat(lp + s["q"], (2 * qd if s["gated_q"] else qd), H)
            own_kv = kv_source[i] == i
            has_v = own_kv and not (s["k_eq_v"] and not sliding_of[i])
            if own_kv:
                d["k"] = mat(lp + s["k"], kvd, H)
            if has_v:
                d["v"] = mat(lp + s["v"], nkv_l * vd_l, H)
            d["qb"] = bias_for(lp + s["q"], qd, s["attn_bias"])
            d["kb"] = bias_for(lp + s["k"], kvd, s["attn_bias"]) if own_kv else None
            d["vb"] = bias_for(lp + s["v"], nkv_l * vd_l, s["attn_bias"]) if has_v else None
        if attn_layers[i]:
            d["o"] = mat(lp + s["o"], H, NH * vd_l)
            d["ob"] = bias_for(lp + s["o"], H, s["attn_bias"] if s["o_bias"] is None else s["o_bias"])
        if ssm_layers[i]:
            d["ssm"] = ssm_weights(lp)
        if attn_layers[i] and s["qk_norm"] in ("head", "heads", "full"):
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
        if attn_layers[i] and s["sinks"] and (sliding_of[i] or not s["sinks_sliding_only"]):
            d["sinks"] = vec(lp + s["sinks"], NH, 1.0)
        if s["attn_sub_norm"]:
            d["asn"] = normw(lp + s["attn_sub_norm"], NH * VD)
        if s["ffn_sub_norm"]:
            d["fsn"] = normw(lp + s["ffn_sub_norm"], I)
        if s["attn_gate"]:
            gname, _, per_head = s["attn_gate"]
            d["ag"] = mat(lp + gname, NH if per_head else NH * VD, H)
        if s["xielu"]:
            # Stored pre-softplus: alpha_p = softplus(p), alpha_n = 0.5 + softplus(n).
            ap, an = s["xielu"]
            stored = []
            for nm, val in (("alpha_p", np.log(np.expm1(ap))), ("alpha_n", np.log(np.expm1(an - 0.5)))):
                w = bf16_round(np.array([val], np.float32))
                weights[lp + f"mlp.act_fn.{nm}"] = w
                stored.append(float(w[0]))
            d["xielu"] = (np.log1p(np.exp(stored[0])), 0.5 + np.log1p(np.exp(stored[1])))
        if s["attn_res"]:
            d["attn_res"] = res_scorer(lp + "self_attention_res_")
            d["mlp_res"] = res_scorer(lp + "mlp_res_")
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
        if moe and i in moe["layers"] and mlp_layers[i]:
            mp = lp + moe["prefix"]
            E, K, MI = moe["E"], moe["K"], moe["MI"]
            # Latent MoE: the routed experts read and write `EW` (the latent width) instead of H.
            EW = moe.get("latent") or H
            if moe.get("latent"):
                d["latent_down"] = mat(mp + "routed_expert_down_proj.weight", EW, H)
                d["latent_up"] = mat(mp + "routed_expert_up_proj.weight", H, EW)
                d["latent_norm"] = normw(mp + "routed_expert_norm.weight", EW)[0] if moe.get("latent_norm") else None
            d["router"] = mat(mp + moe["router"], E, H)
            d["router_b"] = bias_for(mp + moe["router"], E, moe.get("router_bias", False))
            d["corr_b"] = None
            if moe["corr_bias"]:
                cb_name = lp + moe["corr_bias_name"] if "corr_bias_name" in moe else mp + "gate.e_score_correction_bias"
                d["corr_b"] = vec(cb_name, E, 0.5)
                if "corr_bias_shape" in moe:
                    weights[cb_name] = weights[cb_name].reshape(moe["corr_bias_shape"])
            experts = []
            for e in range(E):
                ex = {"gate": bf16_round(rng.normal(0, 0.2, size=(MI, EW))), "up": bf16_round(rng.normal(0, 0.2, size=(MI, EW))),
                      "down": bf16_round(rng.normal(0, 0.2, size=(EW, MI))), "gb": None, "ub": None, "db": None}
                if moe.get("dense"):
                    ex["gate"] = None  # `down(act(up(x)))`, no gate projection
                if moe.get("bias"):
                    ex["gb"], ex["ub"], ex["db"] = (bf16_round(rng.normal(0, 0.1, size=(n,))) for n in (MI, MI, EW))
                experts.append(ex)
            if moe["layout"] == "separate":
                names = moe.get("expert_names", ("gate_proj.weight", "up_proj.weight", "down_proj.weight"))
                for e, ex in enumerate(experts):
                    for key, nm in zip(("gate", "up", "down"), names):
                        if ex[key] is None:
                            continue
                        weights[f"{mp}experts.{e}.{nm}"] = ex[key]
                        if s["quant"] and s["quant"]["kind"] in ("mxfp4_packed", "mxfp4_store"):
                            # Only the routed experts are packed (the compressor ignores everything else).
                            quantize(f"{mp}experts.{e}.{nm}", ex[key])
            elif moe["layout"] == "fused_rows":
                # [E, 2I, H] / [E, H, I] (GraniteMoe input_linear / output_linear).
                gu = np.stack([np.concatenate([ex["gate"], ex["up"]], 0) for ex in experts])
                dn = np.stack([ex["down"] for ex in experts])
                fn = moe.get("fused_names", ("experts.gate_up_proj", "experts.down_proj"))
                weights[mp + fn[0]] = gu
                weights[mp + fn[1]] = dn
            elif moe["layout"] == "fused_eih":
                # The Hugging Face `Experts` module layout: gate rows then up rows per expert, down as [hidden, I].
                gu = np.zeros((E, 2 * MI, H), np.float32)
                dn = np.zeros((E, H, MI), np.float32)
                for e, ex in enumerate(experts):
                    gu[e] = np.concatenate([ex["gate"], ex["up"]], 0)
                    dn[e] = ex["down"]
                weights[mp + "experts.gate_up_proj"] = gu
                weights[mp + "experts.down_proj"] = dn
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
                if s["quant"] and s["quant"]["kind"] == "mxfp4":
                    # Stored as blocks of the natural [E, out, in] layout; the
                    # experts compute with the dequantised values.
                    for key, arr in ((mp + "experts.gate_up_proj", gu), (mp + "experts.down_proj", dn)):
                        nat = np.ascontiguousarray(arr.transpose(0, 2, 1))
                        quantize(key, nat)
                        arr[...] = nat.transpose(0, 2, 1)
                    for e, ex in enumerate(experts):
                        if moe["layout"] == "fused_t_interleaved":
                            ex["gate"], ex["up"] = gu[e, :, 0::2].T.copy(), gu[e, :, 1::2].T.copy()
                        else:
                            ex["gate"], ex["up"] = gu[e, :, :MI].T.copy(), gu[e, :, MI:].T.copy()
                        ex["down"] = dn[e].T.copy()
                if moe.get("bias"):
                    gub = np.zeros((E, 2 * MI), np.float32)
                    dnb = np.zeros((E, H), np.float32)
                    for e, ex in enumerate(experts):
                        gub[e, 0::2], gub[e, 1::2], dnb[e] = ex["gb"], ex["ub"], ex["db"]
                    weights[mp + "experts.gate_up_proj_bias"] = gub
                    weights[mp + "experts.down_proj_bias"] = dnb
            d["experts"] = experts
            if moe["shared"]:
                sp = (lp if moe.get("shared_at_layer") else mp) + moe["shared_name"]
                si = moe.get("shared_inter", moe["shared"] * MI)
                if moe.get("shared_fused"):
                    # One [2I, H] gate/up tensor (GraniteMoeShared input_linear, MiniMax M3 gate_up_proj).
                    gu_name, dn_name = moe["shared_fused"]
                    gu = mat(sp + gu_name, 2 * si, H, register=False)
                    weights[sp + gu_name] = gu
                    d["shared"] = {"gate": gu[:si], "up": gu[si:], "down": mat(sp + dn_name, H, si), "gate_vec": None}
                elif moe.get("dense"):
                    d["shared"] = {"gate": None, "up": mat(sp + "up_proj.weight", si, H), "down": mat(sp + "down_proj.weight", H, si), "gate_vec": None}
                else:
                    d["shared"] = {"gate": mat(sp + "gate_proj.weight", si, H), "up": mat(sp + "up_proj.weight", si, H), "down": mat(sp + "down_proj.weight", H, si),
                                   "gate_vec": mat(lp + "mlp.shared_expert_gate.weight", 1, H)[0] if moe.get("shared_gate") else None}
        elif not mlp_layers[i]:
            pass  # single-block layer without an MLP
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
    # Tensors the reference never runs but that must carry through exports.
    for name, shape in s["extra_tensors"]:
        weights[name] = bf16_round(rng.normal(0, 0.2, size=shape))

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
    stored = []
    for n in sorted(weights):
        if n in qtensors:
            stored.extend(qtensors[n])
        else:
            stored.append((n, "BF16", bf16_bits(weights[n])))
    for n, dtype, u in sorted(stored, key=lambda t: t[0]):
        if dtype == "BF16" and u.dtype != np.uint16:
            u = bf16_bits(u)
        u = np.ascontiguousarray(u)
        header[n] = {"dtype": dtype, "shape": list(u.shape), "data_offsets": [offset, offset + u.nbytes]}
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
        if s["norm"] == "rms_none":
            return x / np.sqrt(np.mean(x * x, -1, keepdims=True) + eps)
        if s["norm"] == "none":
            mu = x.mean(-1, keepdims=True)
            var = ((x - mu) ** 2).mean(-1, keepdims=True)
            return (x - mu) / np.sqrt(var + eps)
        w, b = nw
        if s["norm"] in ("rms", "rms1p", "rms_none"):
            return x / np.sqrt(np.mean(x * x, -1, keepdims=True) + eps) * w
        mu = x.mean(-1, keepdims=True)
        var = ((x - mu) ** 2).mean(-1, keepdims=True)
        y = (x - mu) / np.sqrt(var + eps) * w
        return y + b if b is not None else y

    def head_norm(x, w, b):
        """q/k norm over the last axis of x (weightless RMS when w is None)."""
        if w is None:
            return x / np.sqrt(np.mean(x * x, -1, keepdims=True) + eps)
        if s["norm"] in ("rms", "rms1p", "rms_none"):
            return x / np.sqrt(np.mean(x * x, -1, keepdims=True) + eps) * w
        mu = x.mean(-1, keepdims=True)
        var = ((x - mu) ** 2).mean(-1, keepdims=True)
        y = (x - mu) / np.sqrt(var + eps) * w
        return y + b if b is not None else y

    def glu(g, u):
        """The gated MLP product `act(gate) * up`, or Kimi K3's SiTU."""
        if s["situ"]:
            beta, linear_beta = s["situ"]
            a = beta * np.tanh(g / beta) * (1 / (1 + np.exp(-g)))
            if linear_beta is not None:
                u = linear_beta * np.tanh(u / linear_beta)
            return a * u
        return act_fn(g) * u

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

    def rope_tables(T, theta=None, sc=None):
        inv, factor = inv_freq_and_factor(s["theta"] if theta is None else theta, rd, fd, sc)
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
        HD, NKV, VD = layer_hd[li], layer_nkv[li], (s["VD"] if (s["mla"] or s["narrow_v"]) else layer_hd[li])
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
            k_pe = k_pe.reshape(T, 1, rp)
            if rope_layers[li]:
                q = apply_rope(q, cos, sin, nope)
                k_pe = apply_rope(k_pe, cos, sin, 0)
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
                elif s["qkv_layout"] == "mp":
                    # CodeGen: `qkv_mp` tensor-parallel blocks of [q | v | k].
                    mp = s["qkv_mp"]
                    local = qd // mp
                    blocks = f.reshape(T, mp, 3 * local)
                    q = blocks[:, :, :local].reshape(T, qd)
                    v = blocks[:, :, local:2 * local].reshape(T, kvd)
                    k = blocks[:, :, 2 * local:].reshape(T, kvd)
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
                    k = k * np.float32(s["mult"]["key"])
                    v = v * np.float32(s["mult"]["value"])
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
                k, v = k.reshape(T, NKV, HD), v.reshape(T, NKV, VD)
            if qn == "weightless" and not s["qk_norm_after_rope"]:
                q = head_norm(q, None, None)
                if own_kv:
                    k = head_norm(k, None, None)
            elif qn == "head" and not s["qk_norm_after_rope"]:
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
            if qn == "weightless" and s["qk_norm_after_rope"]:
                q = head_norm(q, None, None)
                if own_kv:
                    k = head_norm(k, None, None)
            elif qn == "head" and s["qk_norm_after_rope"]:
                q = head_norm(q, *d["qn"])
                if own_kv:
                    k = head_norm(k, *d["kn"])
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
            if s["attn_softcap"]:
                cap = np.float32(s["attn_softcap"])
                sc_ = cap * np.tanh(sc_ / cap)
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
        if "attn_gate" in d:
            # Kimi K3 MLA: sigmoid output gate on the [T, NH * VD] attention output.
            out = out * (1 / (1 + np.exp(-(h @ d["attn_gate"].T)))).reshape(T, NH, VD)
        flat = out.reshape(T, NH * VD)
        if s["attn_gate"]:
            # AFMoE / Laguna: a sigmoid or softplus gate from the layer input,
            # per coordinate or per head.
            _, kind, per_head = s["attn_gate"]
            g = h @ d["ag"].T
            if per_head:
                g = np.repeat(g, VD, axis=-1)
            flat = flat * (1 / (1 + np.exp(-g)) if kind == "sigmoid" else np.log1p(np.exp(g)))
        if s["attn_sub_norm"]:
            flat = norm(flat, d["asn"])
        o = flat @ d["o"].T
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
    def lightning_attn(d, li, h):
        """MiniMax lightning attention (modeling_minimax.MiniMaxLightningAttention),
        as the per-token recurrence the blocked prefill is equivalent to."""
        T = h.shape[0]
        p = h @ d["light_qkv"].T
        p = (p / (1 + np.exp(-p))).reshape(T, NH, 3 * HD)
        q, k, v = p[:, :, :HD], p[:, :, HD:2 * HD], p[:, :, 2 * HD:]
        base = 1 / (2 ** (8 / NH))
        factor = 1 - li / (L - 1 + 1e-5) + 1e-5
        rate = (base ** (np.arange(NH) + 1) * factor).astype(np.float32)
        ratio = np.exp(-rate)
        S = np.zeros((NH, HD, HD), np.float32)
        out = np.zeros((T, NH, HD), np.float32)
        for t in range(T):
            S = ratio[:, None, None] * S + np.einsum("hi,hj->hij", k[t], v[t])
            out[t] = np.einsum("hi,hij->hj", q[t], S)
        o = out.reshape(T, NH * HD)
        o = o / np.sqrt(np.mean(o * o, -1, keepdims=True) + 1e-6) * d["light_norm"]
        o = o * (1 / (1 + np.exp(-(h @ d["light_gate"].T))))
        return o @ d["o"].T

    def kda_attn(d, h):
        """Kimi Delta Attention (KimiLinearDeltaAttention, fused_recurrent_kda): the
        decay is per channel of the [k_dim, v_dim] state, from a low-rank forget
        gate; the output norm is gated by a sigmoid of a low-rank gate."""
        ln = s["linear"]
        NHl, D, KC = ln["VH"], ln["VD"], ln["KC"]
        dim = NHl * D
        T = h.shape[0]
        mixed = np.concatenate([h @ d["q"].T, h @ d["k"].T, h @ d["v"].T], -1)
        y = np.zeros((T, 3 * dim), np.float32)
        for t in range(T):
            acc = np.zeros(3 * dim, np.float32)
            for i in range(KC):
                if t - i >= 0:
                    acc += d["conv"][:, KC - 1 - i] * mixed[t - i]
            y[t] = act_fn(acc)
        q = y[:, :dim].reshape(T, NHl, D)
        k = y[:, dim:2 * dim].reshape(T, NHl, D)
        v = y[:, 2 * dim:].reshape(T, NHl, D)
        g = ((h @ d["f_a"].T) @ d["f_b"].T + d["dt"]).reshape(T, NHl, D)
        if ln.get("lower_bound") is not None:
            # Kimi K3 safe gate: lower_bound * sigmoid(exp(A_log) * (f + dt_bias)).
            g = ln["lower_bound"] * (1 / (1 + np.exp(-np.exp(d["alog"])[None, :, None] * g)))
        else:
            g = -np.exp(d["alog"])[None, :, None] * np.where(g > 20.0, g, np.log1p(np.exp(np.minimum(g, 20.0))))
        beta = 1 / (1 + np.exp(-(h @ d["b"].T)))
        q = q / np.sqrt(np.sum(q * q, -1, keepdims=True) + 1e-6) / np.sqrt(D)
        k = k / np.sqrt(np.sum(k * k, -1, keepdims=True) + 1e-6)
        S = np.zeros((NHl, D, D), np.float32)
        core = np.zeros((T, NHl, D), np.float32)
        for t in range(T):
            for hh in range(NHl):
                S[hh] = S[hh] * np.exp(g[t, hh])[:, None]
                mem = S[hh].T @ k[t, hh]
                delta = (v[t, hh] - mem) * beta[t, hh]
                S[hh] = S[hh] + np.outer(k[t, hh], delta)
                core[t, hh] = S[hh].T @ q[t, hh]
        gate = (h @ d["g"].T if "g" in d else (h @ d["g_a"].T) @ d["g_b"].T).reshape(T, NHl, D)
        o = core / np.sqrt(np.mean(core * core, -1, keepdims=True) + eps) * d["lnorm"]
        o = o * (1 / (1 + np.exp(-gate)))
        return o.reshape(T, dim) @ d["o"].T

    def linear_attn(d, h):
        ln = s["linear"]
        if ln.get("kind") == "kda":
            return kda_attn(d, h)
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

    def causal_conv(w, b, x, K):
        """Depthwise causal convolution (+ bias, activation) over x [T, channels]."""
        T = x.shape[0]
        y = np.zeros_like(x)
        for t in range(T):
            acc = np.zeros(x.shape[1], np.float32) if b is None else b.copy()
            for i in range(K):
                if t - i >= 0:
                    acc += w[:, K - 1 - i] * x[t - i]
            y[t] = acc / (1 + np.exp(-acc))
        return y

    def softplus(x):
        return np.where(x > 20, x, np.log1p(np.exp(np.minimum(x, 20))))

    def silu(x):
        return x / (1 + np.exp(-x))

    def mamba2_block(d, h):
        """Mamba2 (SSD) block as in transformers' Mamba2Mixer / FalconH1Mixer torch path."""
        ss, m = s["ssm"], s["mult"]
        heads, hd, N, G, K = ss["heads"], ss["hd"], ss["N"], ss["G"], ss["K"]
        inter, gn = heads * hd, G * N
        conv_dim = inter + 2 * gn
        T = h.shape[0]
        proj = (h * np.float32(m["ssm_in"])) @ d["in_proj"].T
        if d["in_b"] is not None:
            proj = proj + d["in_b"]
        mup = np.concatenate([np.full(inter, m["ssm_proj"][0]), np.full(inter, m["ssm_proj"][1]), np.full(gn, m["ssm_proj"][2]),
                              np.full(gn, m["ssm_proj"][3]), np.full(heads, m["ssm_proj"][4])]).astype(np.float32)
        proj = proj * mup
        gate, xbc, dt = proj[:, :inter], proj[:, inter:inter + conv_dim], proj[:, inter + conv_dim:]
        y = causal_conv(d["conv"], d["conv_b"], xbc, K)
        x = y[:, :inter].reshape(T, heads, hd)
        B = y[:, inter:inter + gn].reshape(T, G, N)
        C = y[:, inter + gn:].reshape(T, G, N)
        dt = np.clip(softplus(dt + d["dt_bias"]), ss["dt_min"], ss["dt_max"])
        A = -np.exp(d["A_log"])
        S = np.zeros((heads, hd, N), np.float32)
        out = np.zeros((T, heads, hd), np.float32)
        for t in range(T):
            for hh in range(heads):
                g = hh // (heads // G)
                S[hh] = S[hh] * np.exp(dt[t, hh] * A[hh]) + dt[t, hh] * np.outer(x[t, hh], B[t, g])
                out[t, hh] = S[hh] @ C[t, g] + d["D"][hh] * x[t, hh]
        y = out.reshape(T, inter)
        if ss["rms_norm"]:
            if not ss["norm_before_gate"]:
                y = y * silu(gate)
            yg = y.reshape(T, ss["norm_groups"], inter // ss["norm_groups"])
            yg = yg / np.sqrt(np.mean(yg * yg, -1, keepdims=True) + eps)
            y = yg.reshape(T, inter) * d["norm"]
            if ss["norm_before_gate"]:
                y = y * silu(gate)
        else:
            y = y * silu(gate)
        o = y @ d["out"].T
        if d["out_b"] is not None:
            o = o + d["out_b"]
        return o

    def mamba1_block(d, h):
        """Mamba1 block as in transformers' JambaMambaMixer torch path."""
        ss = s["ssm"]
        inter, N, K, R = ss["inter"], ss["N"], ss["K"], ss["R"]
        T = h.shape[0]
        proj = h @ d["in_proj"].T
        if d["in_b"] is not None:
            proj = proj + d["in_b"]
        xr, z = proj[:, :inter], proj[:, inter:]
        xa = causal_conv(d["conv"], d["conv_b"], xr, K)
        xdbc = xa @ d["x_proj"].T
        dt_r, B, C = xdbc[:, :R], xdbc[:, R:R + N], xdbc[:, R + N:]

        def rms(v, w):
            return v / np.sqrt(np.mean(v * v, -1, keepdims=True) + eps) * w
        dt_r, B, C = rms(dt_r, d["dt_norm"]), rms(B, d["b_norm"]), rms(C, d["c_norm"])
        dt = softplus(dt_r @ d["dt_proj"].T + d["dt_bias"])  # [T, inter]
        A = -np.exp(d["A_log"])  # [inter, N]
        S = np.zeros((inter, N), np.float32)
        y = np.zeros((T, inter), np.float32)
        for t in range(T):
            S = np.exp(dt[t][:, None] * A) * S + (dt[t] * xa[t])[:, None] * B[t][None, :]
            y[t] = S @ C[t] + d["D"] * xa[t]
        y = y * silu(z)
        o = y @ d["out"].T
        if d["out_b"] is not None:
            o = o + d["out_b"]
        return o

    def ssm_block(d, h):
        return mamba2_block(d["ssm"], h) if s["ssm"]["kind"] == "mamba2" else mamba1_block(d["ssm"], h)

    def expert_out(ex, x, biases=True):
        if ex["gate"] is None:
            u = x @ ex["up"].T
            if biases and ex.get("ub") is not None:
                u = u + ex["ub"]
            y = act_fn(u) @ ex["down"].T
            if biases and ex.get("db") is not None:
                y = y + ex["db"]
            return y
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
            hmid = glu(g, u)
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
            if moe.get("softcap"):
                cap = np.float32(moe["softcap"])
                logits = cap * np.tanh(logits / cap)
            if moe["scoring"] in ("softmax", "topk_softmax"):
                sc_ = np.exp(logits - logits.max(-1, keepdims=True))
                sc_ = sc_ / sc_.sum(-1, keepdims=True)
            else:
                sc_ = 1 / (1 + np.exp(-logits))
            # Latent MoE: the routed experts read `down(h)` and their sum is
            # normalised and written back through `up`; the shared experts read h.
            xin = h @ d["latent_down"].T if "latent_down" in d else h
            out = np.zeros_like(xin)
            for t in range(h.shape[0]):
                if moe["scoring"] == "topk_softmax":
                    # GraniteMoe: top-k of the logits, softmax over the selected ones.
                    idx = np.argsort(-logits[t], kind="stable")[:K]
                    w = np.exp(logits[t][idx] - logits[t][idx].max())
                    w = w / w.sum()
                    for e, we in zip(idx, w):
                        out[t] += we * expert_out(d["experts"][e], h[t])
                    continue
                choice = sc_[t] + (np.squeeze(d["corr_b"]) if d["corr_b"] is not None else 0)
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
                        out[t] += expert_out(d["experts"][e], xin[t] * we)
                    else:
                        out[t] += we * expert_out(d["experts"][e], xin[t])
            if "latent_down" in d:
                if d["latent_norm"] is not None:
                    out = out / np.sqrt(np.mean(out * out, -1, keepdims=True) + eps) * d["latent_norm"]
                out = out @ d["latent_up"].T
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
            if s["dense_swiglu"]:
                alpha, limit = s["dense_swiglu"]
                gc = np.minimum(g * np.float32(s["mult"]["mlp_gate"]), limit)
                uc = np.clip(u, -limit, limit)
                m = (uc + 1.0) * (gc / (1 + np.exp(-alpha * gc)))
            else:
                m = glu(g * np.float32(s["mult"]["mlp_gate"]), u)
        elif s["mlp"] == "gated_fused":
            gu = h @ d["gate_up"].T
            if d["gub"] is not None:
                gu = gu + d["gub"]
            inter = gu.shape[1] // 2
            if s["dense_swiglu"]:
                alpha, limit = s["dense_swiglu"]
                g = np.minimum(gu[:, :inter], limit)
                u = np.clip(gu[:, inter:], -limit, limit)
                m = (u + 1.0) * (g / (1 + np.exp(-alpha * g)))
            else:
                m = glu(gu[:, :inter], gu[:, inter:])
        else:
            u = h @ d["up"].T
            if d["ub"] is not None:
                u = u + d["ub"]
            if s["xielu"]:
                # xIELU (Apertus): alpha_p x² + beta x for x > 0, else
                # alpha_n (expm1(min(x, eps)) - x) + beta x, with beta = 0.5.
                ap, an = d["xielu"]
                m = np.where(u > 0, ap * u * u + 0.5 * u, an * (np.expm1(np.minimum(u, -1e-6)) - u) + 0.5 * u)
            else:
                m = act_fn(u)
        if s["ffn_sub_norm"]:
            m = norm(m, d["fsn"])
        y = m @ d["down"].T
        if d["db"] is not None:
            y = y + d["db"]
        return y * np.float32(s["mult"]["mlp_down"])

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
        # Standard layer; returns the new residual. Single-block layers (Mamba2,
        # Nemotron-H) hold either the mixer or the MLP behind the layer's one norm.
        rm = np.float32(s["residual_mult"])
        has_mixer = attn_layers[li] or ssm_layers[li] or lin_layers[li] or conv_layers[li]
        h, a = x, None
        if has_mixer:
            h = norm(x, d["in_norm"]) if (d["in_norm"] is not None or s["norm"] in ("none", "rms_none")) and s["in_norm"] is not None else x
            a = h_fn(h)
            if d["post_attn_norm"] is not None:
                a = norm(a, d["post_attn_norm"])
        if s["residual_layout"] == "minimax":
            # h = norm(x); x = alpha * h + beta * f(h) for both sublayers.
            sa = s["mm_scales"]["linear" if lin_layers[li] else "full"]
            x = np.float32(sa[0]) * h + np.float32(sa[1]) * a
            h2 = norm(x, d["pre_ff_norm"])
            sm = s["mm_scales"]["mlp"]
            return np.float32(sm[0]) * h2 + np.float32(sm[1]) * mlp(d, h2, li)
        if s["parallel"]:
            m_in = norm(x, d["mlp_norm"]) if d["mlp_norm"] is not None else h
            m = mlp(d, m_in, li)
            if d["post_ff_norm"] is not None:
                m = norm(m, d["post_ff_norm"])
            x = x + rm * (a + m)
        else:
            if has_mixer:
                x = x + rm * a
            if mlp_layers[li]:
                nw, tmpl = (d["pre_ff_norm"], s["pre_ff_norm"]) if has_mixer else (d["in_norm"], s["in_norm"])
                h2 = norm(x, nw) if (nw is not None or s["norm"] in ("none", "rms_none")) and tmpl is not None else x
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

    def sinusoidal(T):
        """fairseq / XGLM table `[sin(p f) | cos(p f)]`, indexed at `p + offset`."""
        half = H // 2
        step = np.log(10000) / (half - 1)
        freqs = np.exp(np.arange(half, dtype=np.float64) * -step)
        ang = np.outer(np.arange(T + s["pos_offset"], dtype=np.float64), freqs)
        return np.concatenate([np.sin(ang), np.cos(ang)], 1).astype(np.float32)

    def forward(tokens):
        T = len(tokens)
        x = embed[tokens] * np.float32(s["embed_scale"])
        if pos_embed is not None:
            x = x + pos_embed[s["pos_offset"] + np.arange(T)]
        if s["pos"] == "sinusoidal":
            x = x + sinusoidal(T)[s["pos_offset"]:s["pos_offset"] + T]
        if s["embed_norm"]:
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
        if s["attn_res"]:
            return forward_attn_res(x, cos, sin)
        for li, d in enumerate(layers):
            c_, s_ = (cos_l, sin_l) if sliding_layers[li] else (cos, sin)

            def h_fn(h, d=d, li=li, c_=c_, s_=s_):
                if conv_layers[li]:
                    return conv_block(d, h)
                if lin_layers[li] and s["linear_kind"] == "lightning":
                    return lightning_attn(d, li, h)
                if lin_layers[li]:
                    return linear_attn(d, h)
                if ssm_layers[li] and attn_layers[li]:
                    mult = s["mult"]
                    return ssm_block(d, h) * np.float32(mult["ssm_out"]) + attention(d, li, h * np.float32(mult["attn_in"]), c_, s_) * np.float32(mult["attn_out"])
                if ssm_layers[li]:
                    return ssm_block(d, h)
                return attention(d, li, h, c_, s_)
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
        hfin = norm(x, final_norm) if s["norm"] not in ("none", "rms_none") else norm(x, None)
        logits = hfin @ lm_head.T
        if lm_bias is not None:
            logits = logits + lm_bias
        logits = logits * np.float32(s["logit_scale"])
        if s["final_softcap"]:
            logits = np.float32(s["final_softcap"]) * np.tanh(logits / np.float32(s["final_softcap"]))
        return logits, hidden

    def attn_res_mix(bank, p, scorer):
        """Kimi K3 Attention Residual aggregation (aggregate_stream): the banked
        block prefixes and the running prefix `p` are mixed by the softmax of
        their scores `rmsnorm(row; norm) · proj`, per token."""
        rows = bank + [p]
        scores = np.stack([np.sum(r / np.sqrt(np.mean(r * r, -1, keepdims=True) + eps) * scorer["norm"] * scorer["proj"], -1) for r in rows])  # [R, T]
        pr = np.exp(scores - scores.max(0, keepdims=True))
        pr = pr / pr.sum(0, keepdims=True)
        return sum(pr[j][:, None] * r for j, r in enumerate(rows))

    def forward_attn_res(x, cos, sin):
        """Kimi K3 decoder (KimiK3DecoderLayer with attn_res): every block boundary
        banks the running prefix and restarts it; each sublayer reads the mixture
        of the bank and the running prefix. The recorded hidden state of layer li
        is that attention-side mixture (ditch's residual definition), and the last
        entry the output mixture the final norm reads."""
        B = s["attn_res"]
        bank = []
        hidden = []
        for li, d in enumerate(layers):
            mix = attn_res_mix(bank, x, d["attn_res"])
            hidden.append(mix.copy())
            h = norm(mix, d["in_norm"])
            if li % B == 0:
                bank.append(x.copy())
                x = None
            a = linear_attn(d, h) if lin_layers[li] else attention(d, li, h, cos, sin)
            x = a if x is None else x + a
            h2 = norm(attn_res_mix(bank, x, d["mlp_res"]), d["pre_ff_norm"])
            x = x + mlp(d, h2)
        x = attn_res_mix(bank, x, output_res)
        hidden.append(x.copy())
        logits = norm(x, final_norm) @ lm_head.T
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
# DeepSeek V4 / V4.1 fixtures: hyper-connections (hc_mult residual streams),
# shared-KV sliding attention with sinks, compressed-KV branches, sqrtsoftplus
# MoE with a clamped SwiGLU and, per family, hash-routed layers plus MTP
# tensors (V4) or CSA2 KV sharing, FP8/FP4 fake quantisation and an engram
# n-gram memory layer (V4.1). The reference forward pass follows
# transformers' modeling_deepseek_v4.py / modeling_deepseek_v41.py.
# ---------------------------------------------------------------------------

import unicodedata

RUST_WHITESPACE = set([chr(c) for c in range(0x09, 0x0E)] + [" ", "\x85", "\xa0", " "] +
                      [chr(c) for c in range(0x2000, 0x200B)] + [" ", " ", " ", " ", "　"])


def engram_normalize(text):
    """The engram token normaliser: NFKC, NFD, strip nonspacing marks, lowercase,
    collapse ASCII whitespace runs, keep a lone space, else strip Unicode whitespace."""
    s = unicodedata.normalize("NFD", unicodedata.normalize("NFKC", text))
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    s = "".join(c.lower() for c in s)
    s = re.sub(r"[ \t\r\n]+", " ", s)
    if s == " ":
        return " "
    i, j = 0, len(s)
    while i < j and s[i] in RUST_WHITESPACE:
        i += 1
    while j > i and s[j - 1] in RUST_WHITESPACE:
        j -= 1
    return s[i:j]


def compressed_token_map(id_to_token, added_ids):
    """`build_compressed_token_map` over a byte-level vocabulary."""
    u2b = {v: k for k, v in b2u().items()}
    keys, lookup = {}, []
    for tid in range(len(id_to_token)):
        tok = id_to_token[tid]
        if tid in added_ids:
            text = tok
        else:
            raw = bytes(u2b[ch] for ch in tok)
            try:
                text = raw.decode("utf-8")
            except UnicodeDecodeError:
                text = None
        if text is None or "�" in text:
            key = tok
        else:
            key = engram_normalize(text) or text
        lookup.append(keys.setdefault(key, len(keys)))
    return lookup, len(keys)


def is_prime(n):
    if n < 2:
        return False
    if n % 2 == 0:
        return n == 2
    i = 3
    while i * i <= n:
        if n % i == 0:
            return False
        i += 2
    return True


def engram_layout(layer_ids, max_ngram, n_heads, vocab_size):
    primes, seen = [], set()
    for _ in layer_ids:
        per = []
        for _ in range(max_ngram - 1):
            sizes, cur = [], vocab_size - 1
            for _ in range(n_heads):
                cur += 1
                while not is_prime(cur) or cur in seen:
                    cur += 1
                seen.add(cur)
                sizes.append(cur)
            per.append(sizes)
        primes.append(per)
    offsets, totals = [], []
    for layer in primes:
        row, total = [], 0
        for p in (p for per in layer for p in per):
            row.append(total)
            total += p
        offsets.append(row)
        totals.append(total)
    return primes, offsets, totals


def hash_multipliers(layer_ids, max_ngram, compressed_vocab_size):
    bound = max(1, (np.iinfo(np.int64).max // compressed_vocab_size) // 2)
    rows = []
    for lid in layer_ids:
        g = np.random.default_rng(10007 * lid)
        rows.append(g.integers(low=0, high=bound, size=(max_ngram,), dtype=np.int64) * 2 + 1)
    return np.stack(rows)


def pow2_ceil(t):
    bits = np.ascontiguousarray(t, dtype=np.float32).view(np.uint32)
    e = ((bits >> 23) & 0xFF).astype(np.int32)
    m = bits & 0x7FFFFF
    return np.ldexp(np.float32(1.0), e - 127 + (m != 0)).astype(np.float32)


def round_e4m3(v):
    """Round-to-nearest-even onto float8 e4m3fn (inputs are clamped to +-448)."""
    v = np.clip(np.asarray(v, dtype=np.float32), -448.0, 448.0)
    a = np.abs(v)
    safe = np.where(a > 0, a, np.float32(1.0))
    step = np.where(a < 2.0 ** -6, np.float32(2.0 ** -9), np.ldexp(np.float32(1.0), np.floor(np.log2(safe)).astype(np.int32) - 3)).astype(np.float32)
    q = (np.rint(a / step) * step).astype(np.float32)
    return np.where(v < 0, -q, q).astype(np.float32)


def fake_quant_fp8(x, block=32):
    n = x.shape[-1]
    if n % block:
        return x
    blocks = x.astype(np.float32).reshape(*x.shape[:-1], n // block, block)
    amax = np.maximum(np.abs(blocks).max(-1), np.float32(1e-4))
    scale = pow2_ceil(amax * np.float32(1.0 / 448.0))
    q = np.clip(blocks / scale[..., None], -448.0, 448.0).astype(np.float32)
    return (round_e4m3(q) * scale[..., None]).reshape(x.shape).astype(np.float32)


FP4_TABLE = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0], dtype=np.float32)


def e2m1_codes(q):
    mag = np.abs(q).astype(np.float32)
    boundaries = np.array([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], dtype=np.float32)
    ties_up = np.array([False, True, False, True, False, True, False])
    thresholds = np.where(ties_up, boundaries, np.nextafter(boundaries, np.float32(np.inf)))
    codes = (mag[..., None] >= thresholds).sum(-1).astype(np.uint8)
    return codes | (np.signbit(q).astype(np.uint8) << 3)


def fake_quant_fp4(x, block, e4m3_scales=False):
    n = x.shape[-1]
    if n % block:
        return x
    blocks = x.astype(np.float32).reshape(*x.shape[:-1], n // block, block)
    amax = np.abs(blocks).max(-1)
    if e4m3_scales:
        scale = round_e4m3(np.maximum(amax, np.float32(6.0 * 2.0 ** -9)) / np.float32(6.0))
    else:
        scale = pow2_ceil(np.maximum(amax, np.float32(6.0 * 2.0 ** -126)) * np.float32(1.0 / 6.0))
    q = np.clip(blocks / scale[..., None], -6.0, 6.0).astype(np.float32)
    return (FP4_TABLE[e2m1_codes(q)] * scale[..., None]).reshape(x.shape).astype(np.float32)


def generate_dsv4(family, out_dir, hub_names=False):
    """`hub_names` writes the hyper-connection tensors the way the released
    DeepSeek V4 / GLM-5.3-Flash checkpoints spell them (`layers.N.hc_attn_fn`,
    `hc_head_fn`) instead of transformers' module spelling (`attn_hc.fn`,
    `hc_head.hc_fn`). Everything else, including the reference outputs, is
    identical, so the fixture pins that both spellings load the same model."""
    v41 = family == "deepseek_v41"
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(2026 if v41 else 2025)
    vocab, encode, bos, eos = make_tokenizer("deepseek3", out_dir)
    V = len(vocab)
    tok_json = json.load(open(f"{out_dir}/tokenizer.json"))
    id_to_token = [None] * V
    for t, i in vocab.items():
        id_to_token[i] = t
    added_ids = set(a["id"] for a in tok_json["added_tokens"])
    H, HC, E, K, MI = 32, 2, 4, 2, 12
    SW, QLR, OG, OLR = 4, 12, 2, 8
    if v41:
        L, NH, HD, RD = 4, 2, 32, 8
        ratios = [0, 2, 1, 1]
        kv_sources, index_sources, cand_source = [1, 2], [1, 2], 2
        cand_blocks, cand_block = 16, 2
        index_topk, index_n_heads, index_head_dim = 32, 2, 8
        eps, gate_temp, limit, rsf = 1e-20, 1.5, 1.0, 1.5
        engram_ids, eg_vocab, eg_ngram, eg_heads, eg_hd = [1], 16, 4, 2, 8
        hash_layers = [False] * L
    else:
        L, NH, HD, RD = 4, 4, 16, 4
        layer_types = ["sliding_attention", "compressed_sparse_attention", "heavily_compressed_attention", "compressed_sparse_attention"]
        rates = {"compressed_sparse_attention": 2, "heavily_compressed_attention": 3}
        ratios = [0 if t == "sliding_attention" else rates[t] for t in layer_types]
        kv_sources = [i for i in range(L) if ratios[i]]
        index_topk, index_n_heads, index_head_dim = 64, 2, 8
        eps, gate_temp, limit, rsf = 1e-6, 1.0, 1.0, 1.5
        hash_layers = [True, True, False, False]
    theta, ctheta = 10000.0, 20000.0
    yarn = {"rope_type": "yarn", "factor": 4.0, "beta_fast": 32, "beta_slow": 1, "original_max_position_embeddings": 32}
    hc_iters, hc_eps = 20, 1e-6
    MIX = (2 + HC) * HC
    weights = {}

    def mat(name, rows, cols, scale=0.2):
        w = bf16_round(rng.normal(0, scale, size=(rows, cols)))
        weights[name] = w
        return w

    def vec(name, n, scale=0.2, mean=0.0):
        w = bf16_round(mean + rng.normal(0, scale, size=(n,)))
        weights[name] = w
        return w

    P = "model."
    embed = mat(P + "embed_tokens.weight", V, H, 1.0)
    final_norm = vec(P + "norm.weight", H, 0.1, 1.0)
    lm_head = mat("lm_head.weight", V, H, 1.0)
    if not v41:
        hp = P + ("hc_head_" if hub_names else "hc_head.hc_")
        hc_head = {"fn": mat(hp + "fn", HC, HC * H, 0.5), "base": vec(hp + "base", HC, 0.3),
                   "scale": vec(hp + "scale", 1, 0.2, 1.0)[0]}
    # Engram hash state (V4.1).
    if v41:
        token_map, cvs = compressed_token_map(id_to_token, added_ids)
        eg_primes, eg_offsets, eg_totals = engram_layout(engram_ids, eg_ngram, eg_heads, eg_vocab)
        eg_mult = hash_multipliers(engram_ids, eg_ngram, cvs)
        eg_pad = vocab[eos]
        n_cols = (eg_ngram - 1) * eg_heads
        eg_tables = {li: mat(P + f"engram_tables.{li}.weight", eg_totals[k], eg_hd, 0.5) for k, li in enumerate(engram_ids)}
    layers = []
    for i in range(L):
        lp = P + f"layers.{i}."
        d = {"in_norm": vec(lp + "input_layernorm.weight", H, 0.1, 1.0), "post_norm": vec(lp + "post_attention_layernorm.weight", H, 0.1, 1.0)}
        for site in ("attn_hc", "ffn_hc"):
            sp = lp + ("hc_" + site[: -len("_hc")] + "_" if hub_names else site + ".")
            d[site] = {"fn": mat(sp + "fn", MIX, HC * H, 0.5), "base": vec(sp + "base", MIX, 0.3), "scale": vec(sp + "scale", 3, 0.2, 1.0)}
        d["q_a"] = mat(lp + "self_attn.q_a_proj.weight", QLR, H)
        d["q_a_norm"] = vec(lp + "self_attn.q_a_norm.weight", QLR, 0.1, 1.0)
        d["q_b"] = mat(lp + "self_attn.q_b_proj.weight", NH * HD, QLR, 0.3)
        d["kv"] = mat(lp + "self_attn.kv_proj.weight", HD, H)
        d["kv_norm"] = vec(lp + "self_attn.kv_norm.weight", HD, 0.1, 1.0)
        d["o_a"] = mat(lp + "self_attn.o_a_proj.weight", OG * OLR, NH * HD // OG)
        d["o_b"] = mat(lp + "self_attn.o_b_proj.weight", H, OG * OLR)
        d["sinks"] = vec(lp + "self_attn.sinks", NH, 1.0)
        if ratios[i] and i in kv_sources:
            cp = lp + "self_attn.compressor."
            width = 2 * HD if (not v41 and layer_types[i] == "compressed_sparse_attention") else HD
            d["c_kv"] = mat(cp + "kv_proj.weight", width, H)
            d["c_gate"] = mat(cp + "gate_proj.weight", width, H) if (not v41 or ratios[i] > 1) else None
            d["c_pb"] = mat(cp + "position_bias", ratios[i], width, 0.5) if not v41 else None
            d["c_norm"] = vec(cp + "kv_norm.weight", HD, 0.1, 1.0)
            if not v41 and layer_types[i] == "compressed_sparse_attention":
                ip = cp + "indexer."
                mat(ip + "kv_proj.weight", 2 * index_head_dim, H)
                mat(ip + "gate_proj.weight", 2 * index_head_dim, H)
                mat(ip + "position_bias", ratios[i], 2 * index_head_dim, 0.5)
                vec(ip + "kv_norm.weight", index_head_dim, 0.1, 1.0)
                mat(ip + "q_b_proj.weight", index_n_heads * index_head_dim, QLR)
                mat(ip + "scorer.weights_proj.weight", index_n_heads, H)
        if v41 and i in index_sources:
            ip = lp + "self_attn.indexer."
            mat(ip + "q_b_proj.weight", index_n_heads * index_head_dim, QLR)
            mat(ip + "weights_proj.weight", index_n_heads, H)
            if i in kv_sources:
                mat(ip + "k_proj.weight", index_head_dim, HD)
                vec(ip + "k_norm.weight", index_head_dim, 0.1, 1.0)
        if v41 and i in engram_ids:
            ep = lp + "engram."
            d["eg_wkv"] = mat(ep + "wkv.weight", H * (HC + 1), n_cols * eg_hd, 0.3)
            d["eg_q"] = mat(ep + "q_weight", HC, H, 0.3)
            d["eg_k"] = mat(ep + "k_weight", HC, H, 0.3)
        d["router"] = mat(lp + "mlp.gate.weight", E, H)
        d["corr_b"] = vec(lp + "mlp.gate.e_score_correction_bias", E, 0.5)
        if v41:
            vec(lp + "mlp.gate.e_score_correction_bias_vl", E, 0.5)
        if hash_layers[i]:
            table = np.stack([rng.permutation(E)[:K] for _ in range(V)]).astype(np.int64)
            weights[lp + "mlp.gate.tid2eid"] = table
            d["tid2eid"] = table
        experts = []
        gu = np.zeros((E, 2 * MI, H), np.float32)
        dn = np.zeros((E, H, MI), np.float32)
        for e in range(E):
            ex = {"gate": bf16_round(rng.normal(0, 0.2, size=(MI, H))), "up": bf16_round(rng.normal(0, 0.2, size=(MI, H))),
                  "down": bf16_round(rng.normal(0, 0.2, size=(H, MI)))}
            gu[e] = np.concatenate([ex["gate"], ex["up"]], 0)
            dn[e] = ex["down"]
            experts.append(ex)
        weights[lp + "mlp.experts.gate_up_proj"] = gu
        weights[lp + "mlp.experts.down_proj"] = dn
        d["experts"] = experts
        sp = lp + "mlp.shared_experts."
        d["shared"] = {"gate": mat(sp + "gate_proj.weight", MI, H), "up": mat(sp + "up_proj.weight", MI, H), "down": mat(sp + "down_proj.weight", H, MI)}
        layers.append(d)
    # Tensors ditch never runs but must carry through exports.
    mat(P + "mtp.0.eh_proj.weight", H, 2 * H)
    vec(P + "mtp.0.norm.weight", H, 0.1, 1.0)
    mat(P + f"layers.{L}.self_attn.q_a_proj.weight", QLR, H)
    if v41:
        mat("vision.blocks.0.attn.qkv.weight", 3 * H, H)
        mat("aligner.weight", H, H)
        vec("image_start", H, 0.5)

    text_cfg = {"hidden_size": H, "num_hidden_layers": L, "num_attention_heads": NH, "num_key_value_heads": 1, "head_dim": HD,
                "q_lora_rank": QLR, "qk_rope_head_dim": RD, "o_groups": OG, "o_lora_rank": OLR, "sliding_window": SW,
                "n_routed_experts": E, "n_shared_experts": 1, "num_experts_per_tok": K, "moe_intermediate_size": MI,
                "scoring_func": "sqrtsoftplus", "norm_topk_prob": True, "routed_scaling_factor": rsf, "swiglu_limit": limit,
                "hc_mult": HC, "hc_sinkhorn_iters": hc_iters, "hc_eps": hc_eps, "rms_norm_eps": eps, "rope_theta": theta,
                "compress_rope_theta": ctheta, "rope_scaling": dict(yarn), "max_position_embeddings": 128,
                "index_n_heads": index_n_heads, "index_head_dim": index_head_dim, "index_topk": index_topk,
                "hidden_act": "silu", "tie_word_embeddings": False, "vocab_size": V, "torch_dtype": "bfloat16"}
    if v41:
        text_cfg.update({"model_type": "deepseek_v41_text", "compress_ratios": ratios + [0, 0, 0], "kv_source_layer_ids": kv_sources,
                         "index_source_layer_ids": index_sources, "candidate_source_layer_id": cand_source,
                         "candidate_topk_blocks": cand_blocks, "candidate_block_size": cand_block, "gate_temp": gate_temp,
                         "topk_method": "noaux_tc", "num_nextn_predict_layers": 3, "engram_layer_ids": engram_ids,
                         "engram_vocab_size": eg_vocab, "engram_num_embeddings": eg_totals, "engram_max_ngram_size": eg_ngram,
                         "engram_n_heads": eg_heads, "engram_head_dim": eg_hd, "engram_pad_id": eg_pad,
                         "engram_compressed_vocab_size": cvs})
        config = {"model_type": "deepseek_v41", "architectures": ["DeepseekV41ForCausalLM"], "text_config": text_cfg,
                  "vision_config": {"model_type": "deepseek_v41_vision", "hidden_size": H, "num_hidden_layers": 1},
                  "image_token_id": vocab[eos], "tie_word_embeddings": False}
    else:
        text_cfg.update({"model_type": "deepseek_v4", "architectures": ["DeepseekV4ForCausalLM"], "layer_types": layer_types,
                         "compress_rates": rates, "mlp_layer_types": ["hash_moe" if h else "moe" for h in hash_layers],
                         "num_nextn_predict_layers": 1})
        config = text_cfg
    json.dump(config, open(f"{out_dir}/config.json", "w"), indent=1)
    json.dump({"eos_token_id": vocab[eos], "bos_token_id": None, "do_sample": False}, open(f"{out_dir}/generation_config.json", "w"))
    header, blobs, offset = {}, [], 0
    for n in sorted(weights):
        w = weights[n]
        if w.dtype == np.int64:
            u, dtype = w, "I64"
        else:
            u, dtype = bf16_bits(w), "BF16"
        header[n] = {"dtype": dtype, "shape": list(u.shape), "data_offsets": [offset, offset + u.nbytes]}
        blobs.append(u.tobytes())
        offset += u.nbytes
    hb = json.dumps(header).encode()
    hb += b" " * ((8 - len(hb) % 8) % 8)
    with open(f"{out_dir}/model.safetensors", "wb") as f:
        f.write(struct.pack("<Q", len(hb)))
        f.write(hb)
        for b in blobs:
            f.write(b)

    # --- reference forward pass (float32) ---
    f32 = np.float32

    def rms(x, w):
        return (x / np.sqrt(np.mean(x * x, -1, keepdims=True) + f32(eps)) * w).astype(f32)

    def sigmoid(x):
        return (1 / (1 + np.exp(-x))).astype(f32)

    def rope_tables(base, scaling, T):
        inv = 1.0 / (base ** (np.arange(0, RD, 2, dtype=np.float64) / RD))
        factor = 1.0
        if scaling:
            f = scaling["factor"]
            om = scaling["original_max_position_embeddings"]
            bf, bs_ = scaling.get("beta_fast", 32), scaling.get("beta_slow", 1)

            def corr_dim(rot):
                return (RD * np.log(om / (rot * 2 * np.pi))) / (2 * np.log(base))
            low, high = np.floor(corr_dim(bf)), np.ceil(corr_dim(bs_))
            low, high = max(low, 0), min(high, RD - 1)
            if low == high:
                high += 0.001
            ramp = np.clip((np.arange(RD // 2) - low) / (high - low), 0, 1)
            inv = (inv / f) * ramp + inv * (1 - ramp)
            factor = scaling.get("attention_factor", 1.0)
        ang = np.outer(np.arange(T, dtype=np.float64), inv)
        return (np.cos(ang) * factor).astype(f32), (np.sin(ang) * factor).astype(f32)

    def rope_tail(x, cos, sin, sign=1.0):
        """Interleaved rope on the trailing RD channels; x [..., D], cos/sin [T, RD/2] broadcast over leading dims."""
        nope, rot = x[..., :-RD], x[..., -RD:]
        ev, od = rot[..., 0::2], rot[..., 1::2]
        s = sin * f32(sign)
        out = np.empty_like(rot)
        out[..., 0::2] = ev * cos - od * s
        out[..., 1::2] = od * cos + ev * s
        return np.concatenate([nope, out], -1).astype(f32)

    def hc_site(x, site):  # x [T, HC, H]
        T = x.shape[0]
        flat = x.reshape(T, HC * H)
        flat = flat / np.sqrt(np.mean(flat * flat, -1, keepdims=True) + f32(eps))
        p = flat @ site["fn"].T
        base, scale = site["base"], site["scale"]
        pre = sigmoid(p[:, :HC] * scale[0] + base[:HC]) + f32(hc_eps)
        post = 2 * sigmoid(p[:, HC:2 * HC] * scale[1] + base[HC:2 * HC])
        logits = (p[:, 2 * HC:] * scale[2] + base[2 * HC:]).reshape(T, HC, HC)
        ex = np.exp(logits - logits.max(-1, keepdims=True))
        comb = ex / ex.sum(-1, keepdims=True) + f32(hc_eps)
        comb = comb / (comb.sum(-2, keepdims=True) + f32(hc_eps))
        for _ in range(hc_iters - 1):
            comb = comb / (comb.sum(-1, keepdims=True) + f32(hc_eps))
            comb = comb / (comb.sum(-2, keepdims=True) + f32(hc_eps))
        return pre.astype(f32), post.astype(f32), comb.astype(f32)

    def collapse(x, pre):
        return (pre[:, :, None] * x).sum(1).astype(f32)

    def expand(y, x, post, comb):
        return (post[:, :, None] * y[:, None, :] + np.einsum("tjk,tjd->tkd", comb, x)).astype(f32)

    def compressor(d, li, h, ccos, csin):
        """Compressed entries [G, HD] of the layer owning the compressor."""
        T, ratio = h.shape[0], ratios[li]
        G = T // ratio
        if G == 0:
            return np.zeros((0, HD), f32)
        kvp = (h @ d["c_kv"].T).astype(f32)
        gp = (h @ d["c_gate"].T).astype(f32) if d["c_gate"] is not None else None
        if v41:
            gk = kvp[:G * ratio].reshape(G, ratio, HD)
            if gp is None:
                latent = gk[:, 0]
            else:
                gg = gp[:G * ratio].reshape(G, ratio, HD)
                w = np.exp(gg - gg.max(1, keepdims=True))
                w = w / w.sum(1, keepdims=True)
                latent = (gk * w).sum(1)
        else:
            width = kvp.shape[1]
            ck = kvp[:G * ratio].reshape(G, ratio, width)
            cg = gp[:G * ratio].reshape(G, ratio, width) + d["c_pb"][None]
            if layer_types[li] == "compressed_sparse_attention":
                nk = np.zeros((G, 2 * ratio, HD), f32)
                ng = np.full((G, 2 * ratio, HD), -np.inf, f32)
                nk[:, ratio:] = ck[..., HD:]
                ng[:, ratio:] = cg[..., HD:]
                if G > 1:
                    nk[1:, :ratio] = ck[:-1, :, :HD]
                    ng[1:, :ratio] = cg[:-1, :, :HD]
            else:
                nk, ng = ck, cg
            w = np.exp(ng - ng.max(1, keepdims=True))
            w = w / w.sum(1, keepdims=True)
            latent = (nk * w).sum(1)
        latent = rms(latent.astype(f32), d["c_norm"])
        pos = np.arange(G) * ratio
        latent = rope_tail(latent, ccos[pos], csin[pos])
        if v41:
            latent = fake_quant_fp4(latent, 16, e4m3_scales=True)
        return latent.astype(f32)

    def attention(d, li, h, entries, mcos, msin, ccos, csin):
        T = h.shape[0]
        use_c = ratios[li] > 0
        cos, sin = (ccos, csin) if use_c else (mcos, msin)
        q_res = rms(h @ d["q_a"].T, d["q_a_norm"])
        q = (q_res @ d["q_b"].T).reshape(T, NH, HD)
        if not v41:
            q = q / np.sqrt(np.mean(q * q, -1, keepdims=True) + f32(eps))
        q = rope_tail(q, cos[:T, None, :], sin[:T, None, :])
        kv = rms(h @ d["kv"].T, d["kv_norm"])
        kv = rope_tail(kv, cos[:T], sin[:T])
        if v41:
            kv = fake_quant_fp8(kv, 32)
        scale = f32(1.0 / np.sqrt(HD))
        idx = np.arange(T)
        wmask = (idx[None, :] > idx[:, None]) | (idx[:, None] - idx[None, :] >= SW)
        G = entries.shape[0] if use_c else 0
        if use_c:
            reach = (idx + 1) // ratios[li]
            assert reach.max() <= index_topk
            cmask = np.arange(G)[None, :] >= reach[:, None]
        out = np.zeros((T, NH, HD), f32)
        for hh in range(NH):
            s = (q[:, hh, :] @ kv.T) * scale
            s = np.where(wmask, -np.inf, s)
            if use_c:
                sc = (q[:, hh, :] @ entries.T) * scale
                sc = np.where(cmask, -np.inf, sc)
                s = np.concatenate([s, sc], 1)
            s = np.concatenate([s, np.full((T, 1), d["sinks"][hh], f32)], 1)
            s = s - s.max(-1, keepdims=True)
            p = np.exp(s)
            p = p / p.sum(-1, keepdims=True)
            o = p[:, :T] @ kv
            if use_c:
                o = o + p[:, T:T + G] @ entries
            out[:, hh, :] = o
        out = rope_tail(out, cos[:T, None, :], sin[:T, None, :], -1.0)
        per = NH * HD // OG
        grouped = out.reshape(T, OG, per)
        ya = np.stack([grouped[:, g] @ d["o_a"][g * OLR:(g + 1) * OLR].T for g in range(OG)], 1).reshape(T, OG * OLR)
        return (ya @ d["o_b"].T).astype(f32)

    def expert_out(ex, x):
        g = np.minimum(x @ ex["gate"].T, f32(limit))
        u = np.clip(x @ ex["up"].T, -limit, limit)
        return ((g / (1 + np.exp(-g)) * u) @ ex["down"].T).astype(f32)

    def moe(d, h, tokens):
        T = h.shape[0]
        logits = (h @ d["router"].T) / f32(gate_temp)
        scores = np.sqrt(np.log1p(np.exp(logits))).astype(f32)
        out = np.zeros_like(h)
        for t in range(T):
            if "tid2eid" in d:
                idx = d["tid2eid"][tokens[t]]
            else:
                idx = np.argsort(-(scores[t] + d["corr_b"]), kind="stable")[:K]
            w = scores[t][idx]
            w = w / (w.sum() + f32(1e-20)) * f32(rsf)
            for e, we in zip(idx, w):
                out[t] += we * expert_out(d["experts"][e], h[t])
        return out + expert_out(d["shared"], h)

    def engram(d, li, x, tokens):
        T = x.shape[0]
        k = engram_ids.index(li)
        cids = [token_map[t] for t in tokens]
        ctx = eg_ngram - 1
        hashes = np.zeros((T, n_cols), np.int64)
        for t in range(T):
            grams, blocked = [], False
            for shift in range(eg_ngram):
                src = cids[t - shift] if t - shift >= 0 else -1
                blocked = blocked or src < 0
                grams.append(eg_pad_cid if blocked else src)
            prod = np.array(grams, np.int64) * eg_mult[k]
            rolling = prod[0]
            for i in range(1, eg_ngram):
                rolling = np.bitwise_xor(rolling, prod[i])
                for hh in range(eg_heads):
                    col = (i - 1) * eg_heads + hh
                    hashes[t, col] = rolling % np.int64(eg_primes[k][i - 1][hh]) + eg_offsets[k][col]
        rows = eg_tables[li][hashes]  # [T, n_cols, eg_hd]
        kv = rows.reshape(T, -1) @ d["eg_wkv"].T
        key, value = kv[:, :HC * H].reshape(T, HC, H), kv[:, HC * H:]
        weight = d["eg_q"] * d["eg_k"]
        rstd = (1 / np.sqrt(np.mean(x * x, -1) + f32(eps))) * (1 / np.sqrt(np.mean(key * key, -1) + f32(eps)))
        dot = (x * weight * key).sum(-1) * rstd * f32(H ** -0.5)
        gate = sigmoid(np.copysign(np.sqrt(np.maximum(np.abs(dot), f32(1e-6))), dot))
        return (x + gate[..., None] * value[:, None, :]).astype(f32)

    if v41:
        eg_pad_cid = token_map[eg_pad]

    def forward(tokens):
        T = len(tokens)
        x = np.repeat(embed[tokens][:, None, :], HC, axis=1).astype(f32)
        mcos, msin = rope_tables(theta, None, T)
        ccos, csin = rope_tables(ctheta, yarn, T)
        hidden = []
        pre_mix = np.zeros((T, HC), f32)
        pre_mix[:, 0] = 1.0
        entries = {}
        for li, d in enumerate(layers):
            if v41 and li in engram_ids:
                x = engram(d, li, x, tokens)
            pre, post, comb = hc_site(x, d["attn_hc"])
            collapsed = collapse(x, pre_mix if v41 else pre)
            hidden.append(collapsed.copy())
            h = rms(collapsed, d["in_norm"])
            ent = None
            if ratios[li]:
                src = max(s for s in kv_sources if s <= li) if v41 else li
                if src == li:
                    entries[li] = compressor(d, li, h, ccos, csin)
                ent = entries[src]
            a = attention(d, li, h, ent, mcos, msin, ccos, csin)
            x = expand(a, x, post, comb)
            if v41:
                pre_mix = pre
            pre, post, comb = hc_site(x, d["ffn_hc"])
            collapsed = collapse(x, pre_mix if v41 else pre)
            m = moe(d, rms(collapsed, d["post_norm"]), tokens)
            x = expand(m, x, post, comb)
            if v41:
                pre_mix = pre
        if v41:
            final = collapse(x, pre_mix)
        else:
            flat = x.reshape(T, HC * H)
            flat = flat / np.sqrt(np.mean(flat * flat, -1, keepdims=True) + f32(eps))
            p = flat @ hc_head["fn"].T
            pre = sigmoid(p * hc_head["scale"] + hc_head["base"]) + f32(hc_eps)
            final = collapse(x, pre)
        hidden.append(final.copy())
        logits = rms(final, final_norm) @ lm_head.T
        return logits, hidden

    cases = []
    for t in ["the ant or you", "hello 42 the ant or you an era in the"]:
        ids = encode(t)
        logits, hidden = forward(ids)
        cases.append({"text": t, "ids": ids, "last_logits": [round(float(v), 4) for v in logits[-1]], "argmax": int(np.argmax(logits[-1])),
                      "last_hidden": [[round(float(v), 4) for v in hh[-1]] for hh in hidden]})
    json.dump({"family": family, "cases": cases}, open(f"{out_dir}/reference.json", "w"), separators=(",", ":"))
    print(f"wrote fixture to {out_dir}: vocab={V}, layers={L}, hidden={H}, streams={HC}")


# ---------------------------------------------------------------------------
# Qwen3.8-Flash-Next (qwen4_exp) and GLM-5.3-Flash (glm5_next) fixtures: the
# two hybrid families whose residual is several parallel streams. Qwen4-Exp
# mixes them with gated residuals (per-stream (1 + w) norms, a low-rank
# sigmoid input mixer, sigmoid injection weights) and adds per-layer n-gram
# embeddings; GLM-5.3-Flash uses DeepSeek V4's manifold-constrained
# hyper-connections collapsed by an unweighted mean. The reference forward
# passes follow transformers' modular_qwen4_exp.py / modular_glm5_next.py.
# ---------------------------------------------------------------------------


def _is_prime(n):
    if n < 2:
        return False
    if n % 2 == 0:
        return n == 2
    i = 3
    while i * i <= n:
        if n % i == 0:
            return False
        i += 2
    return True


_MASK64 = (1 << 64) - 1
_SPLITMIX_GAMMA = 0x9E3779B97F4A7C15
_SPLITMIX_M1 = 0xBF58476D1CE4E5B9
_SPLITMIX_M2 = 0x94D049BB133111EB


def _splitmix64(value):
    value = (value + _SPLITMIX_GAMMA) & _MASK64
    value = ((value ^ (value >> 30)) * _SPLITMIX_M1) & _MASK64
    value = ((value ^ (value >> 27)) * _SPLITMIX_M2) & _MASK64
    return (value ^ (value >> 31)) & _MASK64


def ple_layout(n_layers, ngram_size, heads_per_ngram, vocab_base, vocab_size, seed, divisible_by):
    """Bucket primes, their per-layer offsets and the hash multipliers of every PLE layer."""
    n_cols = (ngram_size - 1) * heads_per_ngram
    primes, current = [], vocab_base - 1
    for _ in range(n_layers * n_cols):
        current += 1
        while not _is_prime(current):
            current += 1
        primes.append(current)
    primes = [primes[l * n_cols:(l + 1) * n_cols] for l in range(n_layers)]
    offsets, totals = [], []
    for row in primes:
        off, total = [], 0
        for p in row:
            off.append(total)
            total += p
        offsets.append(off)
        totals.append(-(-total // divisible_by) * divisible_by)
    half_bound = max(1, ((2 ** 63 - 1) // max(vocab_size, 1)) // 2)
    mult = []
    for l in range(n_layers):
        base_seed = (seed + 10007 * l) & _MASK64
        mult.append([2 * (_splitmix64((base_seed + _SPLITMIX_GAMMA * (i + 1)) & _MASK64) % half_bound) + 1
                     for i in range(ngram_size)])
    return primes, offsets, totals, mult


def shift_right_ignore_eos(tokens, shift, eos):
    """The reference's `_shift_right_ignore_eos`: a shift that would cross the
    previous end-of-sequence token reads the EOS id instead."""
    prev_eos, prev = -1, []
    for t in tokens:
        prev.append(prev_eos)
        if t == eos:
            prev_eos = len(prev) - 1
    out = []
    for t in range(len(tokens)):
        src = t - shift
        out.append(tokens[src] if (t - (prev[t] + 1)) >= shift and src >= 0 else eos)
    return out


def generate_hyper(family, out_dir):
    qwen = family == "qwen4_exp"
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(2027 if qwen else 2028)
    vocab, encode, bos, eos = make_tokenizer("qwen2" if qwen else "llama3", out_dir)
    V = len(vocab)
    eos_id = vocab[eos]
    f32 = np.float32
    H, L = 32, 4
    E, K, MI = 4, 2, 12
    eps = 1e-6 if qwen else 1e-5
    # Layer schedule: linear attention interleaved with indexed (full) attention.
    layer_types = (["linear_attention", "indexed_attention"] * 2) if qwen else (["linear_attention"] * 3 + ["indexed_attention"])
    lin_layers = [t == "linear_attention" for t in layer_types]
    weights = {}

    def mat(name, rows, cols, scale=0.2):
        w = bf16_round(rng.normal(0, scale, size=(rows, cols)))
        weights[name] = w
        return w

    def vec(name, n, scale=0.2, mean=0.0):
        w = bf16_round(mean + rng.normal(0, scale, size=(n,)))
        weights[name] = w
        return w

    def normw(name, n, one_plus):
        """A norm weight, stored as `w - 1` for the (1 + w) families."""
        w = bf16_round(1.0 + rng.normal(0, 0.1, size=(n,)))
        weights[name] = bf16_round(w - 1.0 if one_plus else w)
        return w

    if qwen:
        HC, LOWRANK = 3, 12
        NH, NKV, HD, RD = 4, 2, 8, 2
        KH, KD, VH, VD, KC = 2, 8, 4, 8, 2
        SI = 16  # shared expert width
        theta = 10000.0
        # Per-layer embeddings on two of the linear-attention layers.
        ple_ids0 = [0, 2]
        NGRAM, HPN, PLE_DIM, PLE_KC, VOCAB_BASE, DIVBY, SEED = 3, 2, 32, 2, 50, 8, 1234
        ple_primes, ple_offsets, ple_totals, ple_mult = ple_layout(
            len(ple_ids0), NGRAM, HPN, VOCAB_BASE, V, SEED, DIVBY)
        n_cols = (NGRAM - 1) * HPN
        ple_head = PLE_DIM // n_cols
    else:
        HC = 4
        NH, NKV = 4, 4
        QLR, KVLR, NOPE, VHD = 12, 16, 8, 8
        HD = NOPE
        NHl, DK, KC = 2, 8, 3
        LOWER = -5.0
        LIMIT = 10.0
        KPOOL, INDEX_TOPK, IDX_HEADS, IDX_HD = 2, 64, 2, 8
        RSF, NGROUP, TOPKG = 2.5, 2, 1
        # Only the indexed layer runs the indexer; its type is "full".
        indexer_types = ["shared"] * 3 + ["full"]
    SW = HC * H
    MIX = (2 + HC) * HC

    P = "model."
    embed = mat(P + "embed_tokens.weight", V, H, 1.0)
    lm_head = mat("lm_head.weight", V, H, 1.0)
    final_norm = None if qwen else vec(P + "norm.weight", H, 0.1, 1.0)
    # The final stream collapse: Qwen4-Exp reuses the gated mixer, GLM-5.3
    # takes the unweighted mean (no weights).
    mixer = None
    if qwen:
        mp = P + "hyper_connection_mixer."
        mixer = {"norm": normw(mp + "hc_norm.weight", SW, True),
                 "down": mat(mp + "input_mix_weight_down.weight", LOWRANK, SW),
                 "up": mat(mp + "input_mix_weight_up.weight", SW, LOWRANK)}

    layers = []
    for i in range(L):
        lp = P + f"layers.{i}."
        d = {}
        for site, name in (("attn", "attn_hyper_connection" if qwen else "attn_hc"),
                           ("ffn", "mlp_hyper_connection" if qwen else "ffn_hc")):
            sp = lp + name
            if qwen:
                d[site] = {"norm": normw(sp + ".hc_norm.weight", SW, True),
                           "down": mat(sp + ".input_mix_weight_down.weight", LOWRANK, SW),
                           "up": mat(sp + ".input_mix_weight_up.weight", SW, LOWRANK),
                           "inject": mat(sp + ".block_inject_weight.weight", HC, SW)}
            else:
                d[site] = {"fn": mat(sp + ".fn", MIX, SW, 0.5), "base": vec(sp + ".base", MIX, 0.3),
                           "scale": vec(sp + ".scale", 3, 0.2, 1.0)}
        if not qwen:
            d["in_norm"] = vec(lp + "input_layernorm.weight", H, 0.1, 1.0)
            d["post_norm"] = vec(lp + "post_attention_layernorm.weight", H, 0.1, 1.0)
        if lin_layers[i]:
            if qwen:
                ap = lp + "linear_attn."
                d["qkv"] = mat(ap + "in_proj_qkv.weight", 2 * KH * KD + VH * VD, H)
                d["z"] = mat(ap + "in_proj_z.weight", VH * VD, H)
                d["b"] = mat(ap + "in_proj_b.weight", VH, H)
                d["a"] = mat(ap + "in_proj_a.weight", VH, H)
                d["conv"] = bf16_round(rng.normal(0, 0.2, size=(2 * KH * KD + VH * VD, KC)))
                weights[ap + "conv1d.weight"] = d["conv"]
                d["dt"] = vec(ap + "dt_bias", VH, 0.1)
                d["alog"] = vec(ap + "A_log", VH, 0.1)
                d["lnorm"] = normw(ap + "norm.weight", VD, False)
                d["o"] = mat(ap + "out_proj.weight", H, VH * VD)
            else:
                ap = lp + "self_attn."
                dim = NHl * DK
                d["q"] = mat(ap + "q_proj.weight", dim, H)
                d["k"] = mat(ap + "k_proj.weight", dim, H)
                d["v"] = mat(ap + "v_proj.weight", dim, H)
                conv = bf16_round(rng.normal(0, 0.2, size=(3 * dim, KC)))
                weights[ap + "conv1d.weight"] = conv.reshape(3 * dim, 1, KC)
                d["conv"] = conv
                fp = ap + "forget_gate."
                d["f_a"] = mat(fp + "f_a_proj.weight", DK, H)
                d["f_b"] = mat(fp + "f_b_proj.weight", dim, DK)
                d["dt"] = vec(fp + "dt_bias", dim, 0.5)
                d["alog"] = vec(fp + "A_log", NHl, 0.5)
                d["b"] = mat(ap + "b_proj.weight", NHl, H)
                d["g_a"] = mat(ap + "g_a_proj.weight", DK, H)
                d["g_b"] = mat(ap + "g_b_proj.weight", dim, DK)
                d["lnorm"] = normw(ap + "o_norm.weight", DK, False)
                d["o"] = mat(ap + "o_proj.weight", H, dim)
        elif qwen:
            ap = lp + "self_attn."
            d["q"] = mat(ap + "q_proj.weight", 2 * NH * HD, H)
            d["k"] = mat(ap + "k_proj.weight", NKV * HD, H)
            d["v"] = mat(ap + "v_proj.weight", NKV * HD, H)
            d["o"] = mat(ap + "o_proj.weight", H, NH * HD)
            d["qn"] = normw(ap + "q_norm.weight", HD, True)
            d["kn"] = normw(ap + "k_norm.weight", HD, True)
            # QSA indexer tensors: never read (the dense equivalent is exact
            # within indexer_budget) but they must survive exports.
            ip = ap + "indexer."
            mat(ip + "index_qk_proj.weight", (2 + 1) * 8, H)
            normw(ip + "q_layernorm.weight", 8, True)
            normw(ip + "k_layernorm.weight", 8, True)
        else:
            ap = lp + "self_attn."
            d["q_a"] = mat(ap + "q_a_proj.weight", QLR, H)
            d["q_a_norm"] = vec(ap + "q_a_layernorm.weight", QLR, 0.1, 1.0)
            d["q_b"] = mat(ap + "q_b_proj.weight", NH * NOPE, QLR, 0.3)
            d["kv_a"] = mat(ap + "kv_a_proj_with_mqa.weight", KVLR, H)
            d["kv_a_norm"] = vec(ap + "kv_a_layernorm.weight", KVLR, 0.1, 1.0)
            d["kv_b"] = mat(ap + "kv_b_proj.weight", NH * (NOPE + VHD), KVLR)
            d["o"] = mat(ap + "o_proj.weight", H, NH * VHD)
            # DSA indexer tensors: pass-through, like the QSA ones above.
            ip = ap + "indexer."
            mat(ip + "wq_b.weight", IDX_HEADS * IDX_HD, QLR)
            mat(ip + "wk.weight", IDX_HD, H)
            vec(ip + "k_norm.weight", IDX_HD, 0.1, 1.0)
            vec(ip + "k_norm.bias", IDX_HD, 0.1)
            mat(ip + "weights_proj.weight", IDX_HEADS, H)
            mat(ip + "index_kpool_compress_ape", KPOOL, IDX_HD, 0.3)
            mat(ip + "index_kpool_compress_gate", IDX_HD, H, 0.3)
        # Mixture of experts. Qwen4-Exp routes every layer with a gated shared
        # expert; GLM-5.3 keeps the first layer dense.
        sparse = True if qwen else (i > 0)
        if sparse:
            d["router"] = mat(lp + "mlp.gate.weight", E, H)
            d["corr_b"] = None if qwen else vec(lp + "mlp.gate.e_score_correction_bias", E, 0.5)
            gu = np.zeros((E, 2 * MI, H), np.float32)
            dn = np.zeros((E, H, MI), np.float32)
            experts = []
            for e in range(E):
                ex = {"gate": bf16_round(rng.normal(0, 0.2, size=(MI, H))),
                      "up": bf16_round(rng.normal(0, 0.2, size=(MI, H))),
                      "down": bf16_round(rng.normal(0, 0.2, size=(H, MI)))}
                gu[e] = np.concatenate([ex["gate"], ex["up"]], 0)
                dn[e] = ex["down"]
                experts.append(ex)
            weights[lp + "mlp.experts.gate_up_proj"] = gu
            weights[lp + "mlp.experts.down_proj"] = dn
            d["experts"] = experts
            if qwen:
                sp = lp + "mlp.shared_expert."
                d["shared"] = {"gate": mat(sp + "gate_proj.weight", SI, H), "up": mat(sp + "up_proj.weight", SI, H),
                               "down": mat(sp + "down_proj.weight", H, SI)}
                d["shared_gate"] = mat(lp + "mlp.shared_expert_gate.weight", 1, H)[0]
            else:
                sp = lp + "mlp.shared_experts."
                d["shared"] = {"gate": mat(sp + "gate_proj.weight", MI, H), "up": mat(sp + "up_proj.weight", MI, H),
                               "down": mat(sp + "down_proj.weight", H, MI)}
        else:
            d["gate"] = mat(lp + "mlp.gate_proj.weight", 32, H)
            d["up"] = mat(lp + "mlp.up_proj.weight", 32, H)
            d["down"] = mat(lp + "mlp.down_proj.weight", H, 32)
        if qwen and i in ple_ids0:
            k = ple_ids0.index(i)
            pp = lp + "ple."
            d["ple"] = {
                "key": mat(pp + "key_proj.weight", SW, PLE_DIM),
                "value": mat(pp + "value_proj.weight", H, PLE_DIM),
                "norm_key": normw(pp + "norm_key.weight", SW, True),
                "norm_query": normw(pp + "norm_query.weight", SW, True),
                "norm_conv": normw(pp + "norm_conv.weight", SW, True),
                "conv": bf16_round(rng.normal(0, 0.2, size=(SW, PLE_KC))),
                "table": bf16_round(rng.normal(0, 0.5, size=(ple_totals[k], ple_head))),
                "index": k,
            }
            weights[pp + "conv1d.weight"] = d["ple"]["conv"].reshape(SW, 1, PLE_KC)
            # The released checkpoints shard the table; two shards exercise that path.
            tbl = d["ple"]["table"]
            cut = ple_totals[k] // 2
            weights[pp + "ple_embedding.ngram_embedding.shard_0.weight"] = tbl[:cut]
            weights[pp + "ple_embedding.ngram_embedding.shard_1.weight"] = tbl[cut:]
        layers.append(d)
    # A tensor ditch never runs but that must survive exports.
    mat("visual.blocks.0.attn.qkv.weight", 3 * H, H)

    text_cfg = {
        "hidden_size": H, "num_hidden_layers": L, "num_attention_heads": NH, "vocab_size": V,
        "hidden_act": "silu", "max_position_embeddings": 128, "tie_word_embeddings": False,
        "layer_types": layer_types, "rms_norm_eps": eps, "torch_dtype": "bfloat16",
    }
    if qwen:
        text_cfg.update({
            "model_type": "qwen4_exp_text", "num_key_value_heads": NKV, "head_dim": HD,
            "linear_num_key_heads": KH, "linear_key_head_dim": KD, "linear_num_value_heads": VH,
            "linear_value_head_dim": VD, "linear_conv_kernel_dim": KC,
            "num_experts": E, "num_experts_per_tok": K, "moe_intermediate_size": MI,
            "shared_expert_intermediate_size": SI, "norm_topk_prob": True,
            "hc_count": HC, "hc_lowrank": LOWRANK, "output_gate_type": "sigmoid",
            "indexer_n_heads": 2, "indexer_kv_heads": 1, "indexer_head_dim": 8,
            "indexer_budget": 64, "indexer_compress_ratio": 2,
            "ple_layer_ids": [i + 1 for i in ple_ids0], "ple_embed_dim": PLE_DIM,
            "ple_conv_kernel_size": PLE_KC, "ngram_size": NGRAM, "heads_per_ngram": HPN,
            "ngram_vocab_size_base": VOCAB_BASE, "make_ngram_vocab_size_divisible_by": DIVBY,
            "seed": SEED, "split_ngram_parts": 2, "eos_token_id": eos_id,
            "rope_parameters": {"rope_type": "default", "rope_theta": theta, "partial_rotary_factor": 0.25},
        })
        config = {"model_type": "qwen4_exp", "architectures": ["Qwen4ExpForConditionalGeneration"],
                  "text_config": text_cfg, "vision_config": {"model_type": "qwen4_exp_vision", "hidden_size": H, "depth": 1},
                  "tie_word_embeddings": False}
    else:
        text_cfg.update({
            "model_type": "glm5_next_text", "num_key_value_heads": NKV,
            "q_lora_rank": QLR, "kv_lora_rank": KVLR, "qk_nope_head_dim": NOPE, "qk_rope_head_dim": 0,
            "v_head_dim": VHD, "n_routed_experts": E, "n_shared_experts": 1, "num_experts_per_tok": K,
            "moe_intermediate_size": MI, "intermediate_size": 32, "n_group": NGROUP, "topk_group": TOPKG,
            "routed_scaling_factor": RSF, "norm_topk_prob": True, "swiglu_limit": LIMIT,
            "mlp_layer_types": ["dense"] + ["sparse"] * (L - 1), "indexer_types": indexer_types,
            "linear_num_heads": NHl, "linear_head_dim": DK, "linear_conv_kernel_dim": KC,
            "linear_lower_bound": LOWER, "hc_mult": HC, "hc_eps": 1e-6, "hc_sinkhorn_iters": 20,
            "index_kpool": KPOOL, "index_topk": INDEX_TOPK, "index_head_dim": IDX_HD,
            "index_n_heads": IDX_HEADS, "index_kpool_always_select_tail": True,
        })
        config = {"model_type": "glm5_next", "architectures": ["Glm5NextForConditionalGeneration"],
                  "text_config": text_cfg, "vision_config": {"model_type": "glm5_next_vision", "hidden_size": H, "depth": 1},
                  "tie_word_embeddings": False}
    json.dump(config, open(f"{out_dir}/config.json", "w"), indent=1)
    json.dump({"eos_token_id": eos_id, "bos_token_id": vocab[bos] if bos else None, "do_sample": False},
              open(f"{out_dir}/generation_config.json", "w"))
    header, blobs, offset = {}, [], 0
    for n in sorted(weights):
        u = bf16_bits(weights[n])
        u = np.ascontiguousarray(u)
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

    # --- reference forward pass (float32) ---

    def rms(x, w=None):
        """RMSNorm by the effective weight (`normw` already resolved any (1 + w) storage)."""
        y = x / np.sqrt(np.mean(x * x, -1, keepdims=True) + f32(eps))
        return y.astype(f32) if w is None else (y * w).astype(f32)

    def group_rms(x, w):
        """(1 + w) RMSNorm over `HC` groups of `H` channels (the weight is already 1 + w)."""
        y = x.reshape(*x.shape[:-1], HC, H)
        y = y / np.sqrt(np.mean(y * y, -1, keepdims=True) + f32(eps))
        return (y.reshape(*x.shape) * w).astype(f32)

    def sigmoid(x):
        return (1 / (1 + np.exp(-x))).astype(f32)

    def silu(x):
        return (x / (1 + np.exp(-x))).astype(f32)

    def l2norm(x):
        return x / np.sqrt(np.sum(x * x, -1, keepdims=True) + 1e-6)

    def softplus(x):
        return np.where(x > 20.0, x, np.log1p(np.exp(np.minimum(x, 20.0))))

    def causal_conv(mixed, kernel, act=True, dilation=1):
        """Depthwise causal convolution; tap `j` back reaches `j * dilation` tokens."""
        T, C = mixed.shape
        KCl = kernel.shape[1]
        y = np.zeros((T, C), f32)
        for t in range(T):
            acc = np.zeros(C, f32)
            for j in range(KCl):
                src = t - j * dilation
                if src >= 0:
                    acc += kernel[:, KCl - 1 - j] * mixed[src]
            y[t] = silu(acc) if act else acc
        return y

    # --- hyper-connection sites ---
    def gated_site(x, site, inject=True):
        """Qwen4-Exp gated residual: returns (block input, injection weights)."""
        T = x.shape[0]
        flat = x.reshape(T, SW)
        normed = group_rms(flat, site["norm"])
        w = sigmoid(silu(normed @ site["down"].T / f32(HC)) @ site["up"].T)
        mixed = (w.reshape(T, HC, H) * normed.reshape(T, HC, H)).mean(1).astype(f32)
        if not inject:
            return mixed, None
        inj = 2 * sigmoid(normed @ site["inject"].T / f32(HC))
        return mixed, inj.astype(f32)

    def mhc_site(x, site):
        """GLM-5.3 mHC site: the collapse, expansion and Sinkhorn mixing weights."""
        T = x.shape[0]
        flat = x.reshape(T, SW)
        flat = flat / np.sqrt(np.mean(flat * flat, -1, keepdims=True) + f32(eps))
        p = flat @ site["fn"].T
        base, scale = site["base"], site["scale"]
        hc_eps = f32(1e-6)
        pre = sigmoid(p[:, :HC] * scale[0] + base[:HC]) + hc_eps
        post = 2 * sigmoid(p[:, HC:2 * HC] * scale[1] + base[HC:2 * HC])
        logits = (p[:, 2 * HC:] * scale[2] + base[2 * HC:]).reshape(T, HC, HC)
        ex = np.exp(logits - logits.max(-1, keepdims=True))
        comb = ex / ex.sum(-1, keepdims=True) + hc_eps
        comb = comb / (comb.sum(-2, keepdims=True) + hc_eps)
        for _ in range(19):
            comb = comb / (comb.sum(-1, keepdims=True) + hc_eps)
            comb = comb / (comb.sum(-2, keepdims=True) + hc_eps)
        return pre.astype(f32), post.astype(f32), comb.astype(f32)

    # --- sublayers ---
    def gdn(d, h):
        """Qwen4-Exp Gated DeltaNet (split projections, sigmoid output gate)."""
        T = h.shape[0]
        kd_tot, vd_tot = KH * KD, VH * VD
        mixed = (h @ d["qkv"].T).astype(f32)
        z = (h @ d["z"].T).astype(f32)
        b, a = (h @ d["b"].T).astype(f32), (h @ d["a"].T).astype(f32)
        y = causal_conv(mixed, d["conv"])
        q = y[:, :kd_tot].reshape(T, KH, KD)
        k = y[:, kd_tot:2 * kd_tot].reshape(T, KH, KD)
        v = y[:, 2 * kd_tot:].reshape(T, VH, VD)
        beta = sigmoid(b)
        g = -np.exp(d["alog"]) * softplus(a + d["dt"])
        rep = VH // KH
        q, k = np.repeat(q, rep, axis=1), np.repeat(k, rep, axis=1)
        q = l2norm(q) / np.sqrt(KD)
        k = l2norm(k)
        S = np.zeros((VH, KD, VD), f32)
        core = np.zeros((T, VH, VD), f32)
        for t in range(T):
            S = S * np.exp(g[t])[:, None, None]
            for vh in range(VH):
                mem = S[vh].T @ k[t, vh]
                delta = (v[t, vh] - mem) * beta[t, vh]
                S[vh] = S[vh] + np.outer(k[t, vh], delta)
                core[t, vh] = S[vh].T @ q[t, vh]
        o = core / np.sqrt(np.mean(core * core, -1, keepdims=True) + f32(eps)) * d["lnorm"]
        o = o * sigmoid(z.reshape(T, VH, VD))
        return (o.reshape(T, vd_tot) @ d["o"].T).astype(f32)

    def kda(d, h):
        """GLM-5.3 Kimi Delta Attention with the safe lower-bound forget gate."""
        T = h.shape[0]
        dim = NHl * DK
        mixed = np.concatenate([h @ d["q"].T, h @ d["k"].T, h @ d["v"].T], -1).astype(f32)
        y = causal_conv(mixed, d["conv"])
        q = y[:, :dim].reshape(T, NHl, DK)
        k = y[:, dim:2 * dim].reshape(T, NHl, DK)
        v = y[:, 2 * dim:].reshape(T, NHl, DK)
        raw = ((h @ d["f_a"].T) @ d["f_b"].T + d["dt"]).reshape(T, NHl, DK)
        rate = np.exp(d["alog"])[None, :, None]
        g = (f32(LOWER) * sigmoid(rate * raw)).astype(f32)
        beta = sigmoid(h @ d["b"].T)
        q = l2norm(q) / np.sqrt(DK)
        k = l2norm(k)
        S = np.zeros((NHl, DK, DK), f32)
        core = np.zeros((T, NHl, DK), f32)
        for t in range(T):
            for hh in range(NHl):
                S[hh] = S[hh] * np.exp(g[t, hh])[:, None]
                mem = S[hh].T @ k[t, hh]
                delta = (v[t, hh] - mem) * beta[t, hh]
                S[hh] = S[hh] + np.outer(k[t, hh], delta)
                core[t, hh] = S[hh].T @ q[t, hh]
        gate = ((h @ d["g_a"].T) @ d["g_b"].T).reshape(T, NHl, DK)
        o = core / np.sqrt(np.mean(core * core, -1, keepdims=True) + f32(eps)) * d["lnorm"]
        o = o * sigmoid(gate)
        return (o.reshape(T, dim) @ d["o"].T).astype(f32)

    def qwen_attention(d, h, cos, sin):
        """Sigmoid-gated attention with (1 + w) per-head q/k norms and partial rotary."""
        T = h.shape[0]
        qg = (h @ d["q"].T).reshape(T, NH, 2 * HD)
        q, gate = qg[:, :, :HD], qg[:, :, HD:].reshape(T, NH * HD)
        k = (h @ d["k"].T).reshape(T, NKV, HD)
        v = (h @ d["v"].T).reshape(T, NKV, HD)
        # `normw` already returns the effective weight (the checkpoint stores w - 1).
        q = rms(q, d["qn"])
        k = rms(k, d["kn"])

        def rope(x):
            rot, rest = x[..., :RD], x[..., RD:]
            x1, x2 = rot[..., :RD // 2], rot[..., RD // 2:]
            c, s = cos[:T, None, :], sin[:T, None, :]
            return np.concatenate([x1 * c - x2 * s, x2 * c + x1 * s, rest], -1).astype(f32)
        q, k = rope(q), rope(k)
        groups = NH // NKV
        out = np.zeros((T, NH, HD), f32)
        mask = np.triu(np.ones((T, T), dtype=bool), k=1)
        for hh in range(NH):
            sc = (q[:, hh, :] @ k[:, hh // groups, :].T) / np.sqrt(HD)
            sc = np.where(mask, -np.inf, sc)
            sc = sc - sc.max(-1, keepdims=True)
            pr = np.exp(sc)
            out[:, hh, :] = (pr / pr.sum(-1, keepdims=True)) @ v[:, hh // groups, :]
        o = out.reshape(T, NH * HD) * sigmoid(gate)
        return (o @ d["o"].T).astype(f32)

    def mla_attention(d, h):
        """GLM-5.3 NoPE multi-head latent attention (the indexer's dense equivalent)."""
        T = h.shape[0]
        q = (rms(h @ d["q_a"].T, d["q_a_norm"]) @ d["q_b"].T).reshape(T, NH, NOPE)
        ckv = rms(h @ d["kv_a"].T, d["kv_a_norm"])
        kv = (ckv @ d["kv_b"].T).reshape(T, NH, NOPE + VHD)
        k, v = kv[..., :NOPE], kv[..., NOPE:]
        out = np.zeros((T, NH, VHD), f32)
        mask = np.triu(np.ones((T, T), dtype=bool), k=1)
        for hh in range(NH):
            sc = (q[:, hh, :] @ k[:, hh, :].T) / np.sqrt(NOPE)
            sc = np.where(mask, -np.inf, sc)
            sc = sc - sc.max(-1, keepdims=True)
            pr = np.exp(sc)
            out[:, hh, :] = (pr / pr.sum(-1, keepdims=True)) @ v[:, hh, :]
        return (out.reshape(T, NH * VHD) @ d["o"].T).astype(f32)

    def expert_out(ex, x, clamp):
        g, u = x @ ex["gate"].T, x @ ex["up"].T
        if clamp:
            g = np.minimum(g, f32(LIMIT))
            u = np.clip(u, -LIMIT, LIMIT)
        return ((silu(g) * u) @ ex["down"].T).astype(f32)

    def mlp(d, h):
        if "experts" not in d:
            g = np.minimum(h @ d["gate"].T, f32(LIMIT))
            u = np.clip(h @ d["up"].T, -LIMIT, LIMIT)
            return ((silu(g) * u) @ d["down"].T).astype(f32)
        T = h.shape[0]
        logits = h @ d["router"].T
        if qwen:
            sc = np.exp(logits - logits.max(-1, keepdims=True))
            sc = (sc / sc.sum(-1, keepdims=True)).astype(f32)
        else:
            sc = sigmoid(logits)
        out = np.zeros_like(h)
        for t in range(T):
            if qwen:
                idx = np.argsort(-sc[t], kind="stable")[:K]
                w = sc[t][idx]
                w = w / w.sum()
            else:
                choice = sc[t] + d["corr_b"]
                per = E // NGROUP
                gs = [np.sort(choice[g * per:(g + 1) * per])[::-1][:2].sum() for g in range(NGROUP)]
                keep = np.argsort(-np.array(gs), kind="stable")[:TOPKG]
                masked = np.full(E, -np.inf)
                for g in keep:
                    masked[g * per:(g + 1) * per] = choice[g * per:(g + 1) * per]
                idx = np.argsort(-masked, kind="stable")[:K]
                w = sc[t][idx]
                w = w / (w.sum() + f32(1e-20)) * f32(RSF)
            for e, we in zip(idx, w):
                out[t] += we * expert_out(d["experts"][e], h[t], not qwen)
        shared = expert_out(d["shared"], h, not qwen)
        if qwen:
            shared = shared * sigmoid(h @ d["shared_gate"])[:, None]
        return (out + shared).astype(f32)

    def ple_block(d, streams, tokens):
        """Qwen4-Exp per-layer n-gram embedding, added to every stream."""
        p = d["ple"]
        kidx = p["index"]
        T = len(tokens)
        ctx = NGRAM - 1
        hist = [eos_id] * ctx + list(tokens)
        shifted = [shift_right_ignore_eos(hist, s, eos_id) for s in range(NGRAM)]
        ids = np.zeros((len(hist), n_cols), np.int64)
        for ngram in range(2, NGRAM + 1):
            start = (ngram - 2) * HPN
            mixed = [shifted[0][t] * ple_mult[kidx][0] for t in range(len(hist))]
            for pos in range(1, ngram):
                mixed = [mixed[t] ^ (shifted[pos][t] * ple_mult[kidx][pos]) for t in range(len(hist))]
            for t in range(len(hist)):
                for hh in range(HPN):
                    col = start + hh
                    ids[t, col] = mixed[t] % ple_primes[kidx][col] + ple_offsets[kidx][col]
        emb = p["table"][ids[-T:]].reshape(T, PLE_DIM).astype(f32)
        key = group_rms(emb @ p["key"].T, p["norm_key"]).reshape(T, HC, H)
        value = (emb @ p["value"].T).astype(f32)
        query = group_rms(streams.reshape(T, SW), p["norm_query"]).reshape(T, HC, H)
        gate = (key * query).sum(-1) / np.sqrt(H)
        gate = np.copysign(np.sqrt(np.maximum(np.abs(gate), f32(1e-6))), gate)
        gated = (sigmoid(gate)[..., None] * value[:, None, :]).reshape(T, SW).astype(f32)
        normed = group_rms(gated, p["norm_conv"])
        return (gated + causal_conv(normed, p["conv"], dilation=NGRAM)).reshape(T, HC, H)

    def rope_tables(T):
        inv = 1.0 / (theta ** (np.arange(0, RD, 2, dtype=np.float64) / RD))
        ang = np.outer(np.arange(T, dtype=np.float64), inv)
        return np.cos(ang).astype(f32), np.sin(ang).astype(f32)

    def forward(tokens):
        T = len(tokens)
        x = np.repeat(embed[tokens][:, None, :], HC, axis=1).astype(f32)
        cos, sin = rope_tables(T) if qwen else (None, None)
        hidden = []
        for li, d in enumerate(layers):
            if qwen and "ple" in d:
                x = (x + ple_block(d, x, tokens)).astype(f32)
            if qwen:
                h, inj = gated_site(x, d["attn"])
                hidden.append(h.copy())
                a = gdn(d, h) if lin_layers[li] else qwen_attention(d, h, cos, sin)
                x = (x + inj[:, :, None] * a[:, None, :]).astype(f32)
                h, inj = gated_site(x, d["ffn"])
                m = mlp(d, h)
                x = (x + inj[:, :, None] * m[:, None, :]).astype(f32)
            else:
                pre, post, comb = mhc_site(x, d["attn"])
                collapsed = (pre[:, :, None] * x).sum(1).astype(f32)
                hidden.append(collapsed.copy())
                h = rms(collapsed, d["in_norm"])
                a = kda(d, h) if lin_layers[li] else mla_attention(d, h)
                x = (post[:, :, None] * a[:, None, :] + np.einsum("tjk,tjd->tkd", comb, x)).astype(f32)
                pre, post, comb = mhc_site(x, d["ffn"])
                collapsed = (pre[:, :, None] * x).sum(1).astype(f32)
                m = mlp(d, rms(collapsed, d["post_norm"]))
                x = (post[:, :, None] * m[:, None, :] + np.einsum("tjk,tjd->tkd", comb, x)).astype(f32)
        # The last residual entry is the collapse itself, before the final norm.
        if qwen:
            final, _ = gated_site(x, mixer, inject=False)
            hidden.append(final.copy())
            normed = final
        else:
            final = x.mean(1).astype(f32)
            hidden.append(final.copy())
            normed = rms(final, final_norm)
        return (normed @ lm_head.T).astype(f32), hidden

    cases = []
    for t in ["the ant or you", "hello 42 the ant or you an era in the"]:
        ids = encode(t)
        if bos:
            ids = [vocab[bos]] + ids
        logits, hid = forward(ids)
        cases.append({"text": t, "ids": ids, "last_logits": [round(float(v), 4) for v in logits[-1]],
                      "argmax": int(np.argmax(logits[-1])),
                      "last_hidden": [[round(float(v), 4) for v in hh[-1]] for hh in hid]})
    json.dump({"family": family, "cases": cases}, open(f"{out_dir}/reference.json", "w"), separators=(",", ":"))
    print(f"wrote fixture to {out_dir}: vocab={V}, layers={L}, hidden={H}, streams={HC}")


if FAMILY in ("qwen4_exp", "glm5_next"):
    generate_hyper(FAMILY, OUT)
    sys.exit(0)


if FAMILY == "deepseek_v4_hubnames":
    generate_dsv4("deepseek_v4", OUT, hub_names=True)
    sys.exit(0)

if FAMILY in ("deepseek_v4", "deepseek_v41"):
    generate_dsv4(FAMILY, OUT)
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
