return {
  model_type = "ernie4_5_moe",
  llama_cpp = "ernie4_5-moe",
  verified = true,
  notes = "fixture: interleaved rotary, softmax routing with the moe_statics correction bias and renormalised top-k, shared experts, moe_layer_start_index / interval, use_bias. ERNIE 4.5 MoE (PT checkpoints).",
  default_norm_eps = 1e-5,
  rope_style = "gptj",
  names = {
    router_correction_bias = "mlp.moe_statics.e_score_correction_bias",
    shared_expert = "mlp.shared_experts.",
  },
  config = function(cfg, c)
    -- Softmax routing; the correction bias steers the choice only; the
    -- selected probabilities are renormalised (clamped at moe_norm_min).
    c.norm_topk_prob = true
    c.attention_bias = flag(cfg.use_bias, false)
    if type(cfg.rope_theta) ~= "number" then c.rope_theta = 500000.0 end
    c.num_experts = int(cfg.moe_num_experts, c.num_experts)
    c.num_experts_per_tok = int(cfg.moe_k, int(cfg.num_experts_per_tok, 6))
    each_layer(c.moe_layers, function() return false end)
    if c.num_experts > 0 then
      local start = int(cfg.moe_layer_start_index, 1)
      local last = c.num_layers - 1
      local e = cfg.moe_layer_end_index
      if type(e) == "number" and e >= 0 then last = math.floor(e) end
      local interval = math.max(int(cfg.moe_layer_interval, 1), 1)
      each_layer(c.moe_layers, function(i) return (i + 1) % interval == 0 and i >= start and i <= last end)
    end
  end,
}
