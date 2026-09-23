return {
  model_type = "olmoe",
  llama_cpp = "olmoe",
  chat = "olmo",
  verified = true,
  notes = "fixture: q/k RMSNorm over the full projection, clip_qkv, softmax top-k routing (optionally renormalised) over separate or fused experts. OLMoE.",
  default_norm_eps = 1e-5,
  qk_norm = "full",
  names = {
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
