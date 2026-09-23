return {
  model_type = "mellum",
  llama_cpp = nil,
  chat = "chatml",
  verified = true,
  notes = "fixture: per-head q/k norm, per-layer-type rope parameters, softmax top-k routing renormalised over fused experts. Mellum (JetBrains).",
  default_rope_theta = 500000.0,
  names = {
    q_norm = "self_attn.q_norm.weight",
    k_norm = "self_attn.k_norm.weight",
  },
  hook = "mellum",
}
