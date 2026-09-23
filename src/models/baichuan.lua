return {
  model_type = "baichuan",
  llama_cpp = "baichuan",
  verified = true,
  notes = "fixture: fused W_pack (7B, RoPE). The 13B ALiBi variant (detected by model_max_length) is unverified. Needs a tokenizer.json (the SentencePiece-only checkpoints must be converted).",
  qkv = "concat",
  names = {
    q = false,
    k = false,
    v = false,
    qkv = "self_attn.W_pack.weight",
  },
  hook = "baichuan",
}
