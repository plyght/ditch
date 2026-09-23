return {
  model_type = "olmoe",
  llama_cpp = "olmoe",
  chat = "olmo",
  verified = true,
  notes = "fixture: q/k RMSNorm over the full projection, clip_qkv, softmax top-k routing (optionally renormalised) over separate or fused experts. OLMoE.",
  default_norm_eps = 1e-5,
  qk_norm = "full",
  names = {
    q_norm = "self_attn.q_norm.weight",
    k_norm = "self_attn.k_norm.weight",
  },
  hook = "olmoe",
}
