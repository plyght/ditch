return {
  model_type = "exaone4",
  llama_cpp = "exaone4",
  chat = "exaone4",
  verified = true,
  notes = "fixture: post-norms, per-head q/k norm, hybrid sliding layers with RoPE and global layers without.",
  default_norm_eps = 1e-5,
  qk_norm = "head",
  names = {
    input_norm = {},
    post_attn_norm = "post_attention_layernorm.weight",
    pre_ff_norm = false,
    post_ff_norm = "post_feedforward_layernorm.weight",
    q_norm = "self_attn.q_norm.weight",
    k_norm = "self_attn.k_norm.weight",
  },
  hook = "exaone4",
}
