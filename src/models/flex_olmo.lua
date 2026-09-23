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
  hook = "olmoe",
}
