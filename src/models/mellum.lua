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
  -- Mellum: a plain softmax top-k router (renormalised) over fused experts.
  config = function(cfg, c)
    c.qk_norm = "head"
    c.norm_topk_prob = flag(cfg.norm_topk_prob, true)
  end,
}
