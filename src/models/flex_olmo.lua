return {
  model_type = "flex_olmo",
  llama_cpp = "olmoe",
  chat = "olmo",
  verified = true,
  notes = "fixture: the OLMo 2 post-norm layout with OLMoE's softmax top-k routing. FlexOlmo.",
  default_rope_theta = 500000.0,
  qk_norm = "full",
  names = {
    input_norm = {},
    post_attn_norm = "post_attention_layernorm.weight",
    pre_ff_norm = false,
    post_ff_norm = "post_feedforward_layernorm.weight",
    q_norm = "self_attn.q_norm.weight",
    k_norm = "self_attn.k_norm.weight",
  },
  -- OLMoE / FlexOLMo: softmax over every expert, top-k, optional renormalisation.
  config = function(cfg, c)
    c.qk_norm = "full"
    c.norm_topk_prob = flag(cfg.norm_topk_prob, false)
    if c.num_experts > 0 then each_layer(c.moe_layers, function() return true end) end
  end,
}
