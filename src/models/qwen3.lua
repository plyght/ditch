return {
  model_type = "qwen3",
  aliases = { "qwen3_vl", "qwen3_vl_text" },
  llama_cpp = "qwen3",
  chat = "chatml",
  verified = true,
  notes = "fixture: per-head q/k RMSNorm. Qwen3-VL text config uses the same path.",
  qk_norm = "head",
  names = {
    q_norm = "self_attn.q_norm.weight",
    k_norm = "self_attn.k_norm.weight",
  },
}
