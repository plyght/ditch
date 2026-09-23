return {
  model_type = "hy_v3",
  llama_cpp = "hunyuan-moe",
  verified = true,
  notes = "fixture: per-head q/k norm, sigmoid routing with a correction bias, renormalisation and a router scaling factor, a shared MLP and a dense/sparse mlp_layer_types schedule. Hunyuan V3 (released checkpoints and the transformers module layout).",
  default_norm_eps = 1e-5,
  default_rope_theta = 11158840.0,
  names = {
    q_norm = "self_attn.q_norm.weight",
    k_norm = "self_attn.k_norm.weight",
    router = "mlp.router.gate.weight",
    router_correction_bias = "mlp.expert_bias",
    shared_expert = "mlp.shared_mlp.",
    moe_alt = {
      router_correction_bias = "mlp.e_score_correction_bias",
      shared_expert = "mlp.shared_experts.",
    },
  },
  -- HunYuan V3: sigmoid routing with a correction bias, renormalised and
  -- scaled, shared experts and a per-head q/k norm before RoPE.
  config = function(cfg, c)
    c.qk_norm = "head"
    c.moe.scoring = "sigmoid"
    c.moe.norm_eps_floor = true
    c.norm_topk_prob = true
    local scale = cfg.router_scaling_factor
    if type(scale) ~= "number" then scale = cfg.routed_scaling_factor end
    c.moe.routed_scaling_factor = num(scale, 1.0)
  end,
}
