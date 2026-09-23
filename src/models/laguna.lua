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
  -- Laguna: sigmoid routing with a `tanh` softcap on the router logits and a
  -- correction bias, renormalised weights scaled after the shared expert is
  -- added, and a softplus gate on the attention output.
  config = function(cfg, c)
    -- The released checkpoints give full and sliding layers different query
    -- head counts (48 / 64 on Laguna XS.2); the kv heads are shared.
    local per_layer = cfg.num_attention_heads_per_layer
    if is_array(per_layer) then
      if #per_layer ~= c.num_layers then invalid("num_attention_heads_per_layer needs one entry per layer") end
      for i, n in ipairs(per_layer) do
        if math.type(n) ~= "integer" or n <= 0 then invalid("num_attention_heads_per_layer entries must be positive integers") end
        c.layer_heads[i] = n
        if c.num_kv_heads == 0 or n % c.num_kv_heads ~= 0 then
          invalid("num_attention_heads_per_layer entries must be multiples of num_key_value_heads")
        end
      end
    end
    c.qk_norm = "head"
    c.attn_gate = "softplus"
    c.moe.scoring = "sigmoid"
    c.norm_topk_prob = true
    c.moe.routed_scaling_factor = num(cfg.moe_routed_scaling_factor, 1.0)
    local cap = f32(num(cfg.moe_router_logit_softcapping, 0))
    if cap > 0 then c.moe.router_softcap = cap end
    if flag(cfg.moe_apply_router_weight_on_input, false) then c.moe.scale_input = true end
  end,
}
