return {
  model_type = "solar_open",
  llama_cpp = nil,
  chat = "solar_open",
  verified = true,
  notes = "fixture: partial rotary, DeepSeek-V3 routing with shared experts on every layer. Solar Open (Upstage).",
  default_norm_eps = 1e-5,
  default_rope_theta = 1000000.0,
  names = {
    router_correction_bias = "mlp.gate.e_score_correction_bias",
    shared_expert = "mlp.shared_experts.",
  },
  hook = "solar_open",
}
