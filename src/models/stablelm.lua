return {
  model_type = "stablelm",
  llama_cpp = "stablelm",
  chat = "zephyr",
  verified = true,
  notes = "fixture: LayerNorm with biases, partial rotary, qkv biases (use_qkv_bias). Parallel residual and qk_layernorm (StableLM 2 12B) are implemented but unverified.",
  norm = "layer",
  names = {
    q_norm = "self_attn.q_layernorm.weight",
    k_norm = "self_attn.k_layernorm.weight",
  },
  hook = "stablelm",
}
