return {
  model_type = "laguna",
  llama_cpp = nil,
  chat = "laguna",
  verified = true,
  notes = "fixture: a softplus gate on the attention output (per head or per coordinate), sigmoid routing with a tanh softcap on the router logits and a correction bias, renormalised weights scaled by moe_routed_scaling_factor, shared experts, and per-layer query head counts (`num_attention_heads_per_layer`).",
  default_rope_theta = 500000.0,
  names = {
    q_norm = "self_attn.q_norm.weight",
    k_norm = "self_attn.k_norm.weight",
    attn_gate = "self_attn.g_proj.weight",
    router_correction_bias = "mlp.experts.e_score_correction_bias",
    shared_expert = "mlp.shared_expert.",
  },
  hook = "laguna",
}
