return {
  model_type = "glm4",
  aliases = { "glm", "glm4v", "glm4v_text" },
  llama_cpp = "glm4",
  chat = "glm4",
  verified = true,
  notes = "fixture: post_self_attn / post_mlp norms, fused gate_up_proj, interleaved half rotary, q/k/v biases. GLM-4 (0414) and the `glm` model_type (GLM-4-9B HF port).",
  default_norm_eps = 1.5625e-7,
  rope_style = "gptj",
  mlp = "gated_fused",
  attention_bias = true,
  names = {
    post_attn_norm = "post_self_attn_layernorm.weight",
    post_ff_norm = "post_mlp_layernorm.weight",
    gate = false,
    up = false,
    gate_up = "mlp.gate_up_proj.weight",
  },
  config = function(cfg, c)
    if type(cfg.partial_rotary_factor) ~= "number" then
      c.rotary_dim = c.head_dim // 2
      c.rope_freq_dim = c.rotary_dim
    end
    c.attention_bias = flag(cfg.attention_bias, true)
  end,
}
