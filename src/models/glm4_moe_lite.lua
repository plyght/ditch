local moe = require("moe")

return {
  model_type = "glm4_moe_lite",
  aliases = { "glm_moe_lite" },
  llama_cpp = nil,
  chat = "glm4",
  verified = true,
  notes = "fixture: GLM-4.7-Flash: DeepSeek V3 MLA with interleaved partial rotary, sigmoid MoE with correction bias, top-2 group scores over n_group groups, floored renormalisation and routed_scaling_factor, shared experts, stacked expert tensors, mlp_layer_types with a dense first layer.",
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
  -- GLM-4.7-Flash (`glm4_moe_lite`): DeepSeek V3 MLA with interleaved RoPE and
  -- the GLM-4.5 router (sigmoid scores, correction bias, top-2 group scores,
  -- renormalised with a floor, `routed_scaling_factor`), an `mlp_layer_types`
  -- schedule that defaults to one dense layer.
  config = function(cfg, c)
    c.rope_style = flag(cfg.rope_interleave, true) and "gptj" or "neox"
    if type(cfg.rms_norm_eps) ~= "number" then c.rms_norm_eps = 1e-5 end
    c.norm_topk_prob = flag(cfg.norm_topk_prob, true)
    c.num_experts_per_tok = int(cfg.num_experts_per_tok, 4)
    c.moe.scoring = "sigmoid"
    c.moe.topk_method = "group_limited"
    c.moe.n_group = math.max(1, int(cfg.n_group, 1))
    c.moe.topk_group = math.max(1, int(cfg.topk_group, 1))
    c.moe.routed_scaling_factor = num(cfg.routed_scaling_factor, 1.8)
    c.moe.group_score_top2 = true
    c.moe.norm_eps_floor = true
    if c.num_experts % c.moe.n_group ~= 0 then invalid("the routed experts do not split into n_group groups") end
    if c.mla == nil then
      invalid("glm4_moe_lite: the MLA keys (kv_lora_rank, qk_rope_head_dim, ...) are required")
    end
    moe.mlp_layer_types(cfg, c, 1)
  end,
}
