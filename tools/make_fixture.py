#!/usr/bin/env python3
"""Generates tiny synthetic Hugging Face-format models plus a NumPy reference
forward pass, used to validate ditch's inference against known-good numbers.

Usage: make_fixture.py <family> <out_dir>
    family: llama | qwen2 | qwen3 | gemma3 | qwen3_moe | qwen3_moe_fused | qwen3_moe_fused_t

The qwen3_moe variants share identical weights: `qwen3_moe` stores one tensor
per expert, `qwen3_moe_fused` the fused [E, 2I, H] / [E, H, I] layout and
`qwen3_moe_fused_t` the transposed fused [E, H, 2I] / [E, I, H] layout.

Only NumPy is required. Weights are random but deterministic.
"""
import json
import os
import struct
import sys

import numpy as np

FAMILY = sys.argv[1] if len(sys.argv) > 1 else "llama"
OUT = sys.argv[2] if len(sys.argv) > 2 else f"tests/fixtures/{FAMILY}"
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(1234)

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

H, I, L, NH, NKV = 32, 48, 3, 4, 2
HD = H // NH
# MoE: E routed experts of size MI, top-K routing; layer 1 stays dense (mlp_only_layers).
E, K, MI = 4, 2, 12
MLP_ONLY = [1]
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
json.dump({"family": FAMILY, "cases": cases}, open(f"{OUT}/reference.json", "w"))
print(f"wrote fixture to {OUT}: vocab={vocab_size}, layers={L}, hidden={H}")
