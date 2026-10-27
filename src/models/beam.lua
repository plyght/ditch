return {
  model_type = "beam",
  verified = true,
  notes = "fixture: per-head q/k RMSNorm, a sigmoid gate on the attention output with a per-head bias, sandwich norms around both sublayers with depth-scaled residual adds, sliding layers with RoPE and full layers without it that scale the queries by (1 + (pos + 1) / 2048) ^ 0.1, dense then sparse MLPs (mlp_layer_types) with sigmoid group-limited routing (top-2 group scores, e_score_correction_bias, 1e-20 renormalisation floor, routed_scaling_factor), final logit softcapping. Beam 1.0 (reflection).",
  default_norm_eps = 1e-6,
  qk_norm = "head",
  names = {
    q_norm = "self_attn.q_norm.weight",
    k_norm = "self_attn.k_norm.weight",
    post_attn_norm = "post_attention_layernorm.weight",
    pre_ff_norm = "pre_feedforward_layernorm.weight",
    post_ff_norm = "post_feedforward_layernorm.weight",
    attn_gate = "self_attn.gate_proj.weight",
    attn_gate_bias = "self_attn.gate_bias",
    router_correction_bias = "mlp.gate.e_score_correction_bias",
  },
  config = function(cfg, c)
    c.attn_gate = "sigmoid"
    c.residual_layout = "depth_scaled"
    c.moe.scoring = "sigmoid"
    c.moe.topk_method = "group_limited"
    c.moe.n_group = math.max(1, int(cfg.n_group, 1))
    c.moe.topk_group = math.max(1, int(cfg.topk_group, 1))
    c.moe.group_score_top2 = true
    c.moe.norm_eps_floor = true
    c.moe.routed_scaling_factor = f32(num(cfg.routed_scaling_factor, 1.0))
    c.norm_topk_prob = flag(cfg.norm_topk_prob, true)
    if c.num_experts % c.moe.n_group ~= 0 then invalid("the routed experts do not split into n_group groups") end
    -- A layer type whose rope_parameters entry is null runs without RoPE and
    -- with BeamModel's scalable-softmax query scale instead.
    local rp = obj(cfg.rope_parameters)
    local types = cfg.layer_types
    if rp ~= nil and is_array(types) then
      local nope = false
      each_layer(c.rope_layers, function(i)
        local t = str(types[i + 1])
        local p = t and rp[t]
        local has = p ~= nil and p ~= null
        nope = nope or not has
        return has
      end)
      if nope then c.attn_temperature = { floor_scale = 2048.0, attn_scale = 0.1, offset = 1.0, power = true } end
    end
  end,
}
