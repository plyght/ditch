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
  hook = "gpt_oss",
}
