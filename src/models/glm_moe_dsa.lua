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
  config = function(cfg, c)
    c.rope_style = flag(cfg.rope_interleave, true) and "gptj" or "neox"
    c.moe.scoring = "sigmoid"
    c.moe.topk_method = "group_limited"
    c.moe.n_group = math.max(1, int(cfg.n_group, 1))
    c.moe.topk_group = math.max(1, int(cfg.topk_group, 1))
    c.moe.routed_scaling_factor = num(cfg.routed_scaling_factor, 1.0)
    if type(cfg.index_topk) == "number" then
      warn("sparse indexer runs as dense attention (exact for short contexts)")
    end
  end,
}
