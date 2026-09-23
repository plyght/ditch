return {
  model_type = "glm4_moe",
  aliases = { "glm4v_moe_text" },
  llama_cpp = nil,
  chat = "glm4",
  verified = true,
  notes = "fixture: dense attention with per-head q/k norms (use_qk_norm), partial rotary, sigmoid MoE with correction bias and shared experts. The glm4v_moe image/video wrapper runs its text config; vision weights pass through exports untouched.",
  default_norm_eps = 1e-5,
  attention_bias = true,
  names = {
    prefixes = { "model.", "model.language_model.", "language_model.model.", "" },
    q_norm = "self_attn.q_norm.weight",
    k_norm = "self_attn.k_norm.weight",
    router_correction_bias = "mlp.gate.e_score_correction_bias",
    shared_expert = "mlp.shared_experts.",
  },
  hook = "glm4_moe",
}
