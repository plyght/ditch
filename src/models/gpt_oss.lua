return {
  model_type = "gpt_oss",
  llama_cpp = "gpt-oss",
  chat = "harmony",
  verified = true,
  notes = "fixture: attention sinks, alternating sliding layers, yarn, router bias with top-k softmax, interleaved fused experts with biases and the clamped swiglu. BF16 and MXFP4 checkpoints (experts dequantised on load).",
  default_norm_eps = 1e-5,
  default_rope_theta = 150000.0,
  attention_bias = true,
  names = {
    sinks = "self_attn.sinks",
    router = "mlp.router.weight",
    fused_gate_up = { "mlp.experts.gate_up_proj" },
    fused_down = { "mlp.experts.down_proj" },
  },
  config = function(cfg, c)
    c.sinks = true
    c.attention_bias = flag(cfg.attention_bias, true)
    c.moe.gate_up_interleaved = true
    c.moe.expert_bias = true
    c.moe.swiglu = { alpha = num(cfg.swiglu_alpha, 1.702), limit = num(cfg.swiglu_limit, 7.0) }
    c.norm_topk_prob = true
    c.num_experts_per_tok = int(cfg.num_experts_per_tok, 4)
    if c.num_experts > 0 then each_layer(c.moe_layers, function() return true end) end
  end,
}
