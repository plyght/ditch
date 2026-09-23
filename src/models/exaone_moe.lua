return {
  model_type = "exaone_moe",
  llama_cpp = "exaone4",
  chat = "k_exaone",
  verified = true,
  notes = "fixture: per-head q/k norm, sliding_window_pattern local layers, DeepSeek-V3 routing with shared experts and a dense/sparse mlp_layer_types schedule. EXAONE 4 MoE.",
  default_norm_eps = 1e-5,
  names = {
    q_norm = "self_attn.q_norm.weight",
    k_norm = "self_attn.k_norm.weight",
    router_correction_bias = "mlp.gate.e_score_correction_bias",
    shared_expert = "mlp.shared_experts.",
  },
  hook = "exaone_moe",
}
