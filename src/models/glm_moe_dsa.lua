return {
  model_type = "glm_moe_dsa",
  llama_cpp = nil,
  chat = "glm4",
  verified = true,
  notes = "fixture: MLA with a sparse indexer (run as dense attention: exact for short contexts), sigmoid MoE with correction bias and shared experts.",
  default_norm_eps = 1e-5,
  names = {
    q_a = "self_attn.q_a_proj.weight",
    q_a_norm = "self_attn.q_a_layernorm.weight",
    q_b = "self_attn.q_b_proj.weight",
    kv_a = "self_attn.kv_a_proj_with_mqa.weight",
    kv_a_norm = "self_attn.kv_a_layernorm.weight",
    kv_b = "self_attn.kv_b_proj.weight",
    router_correction_bias = "mlp.gate.e_score_correction_bias",
    shared_expert = "mlp.shared_experts.",
  },
  hook = "glm_moe_dsa",
}
