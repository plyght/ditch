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
  config = function(cfg, c)
    c.attention_bias = flag(cfg.use_qkv_bias, false)
    if flag(cfg.qk_layernorm, false) then c.qk_norm = "head" end
  end,
}
