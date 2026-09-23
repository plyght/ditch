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
  hook = "ernie_moe",
}
