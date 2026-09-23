return {
  model_type = "olmo2",
  aliases = { "olmo3" },
  llama_cpp = "olmo2",
  chat = "olmo",
  verified = true,
  notes = "fixture: post-norms on the sublayer outputs (no input norm), q/k RMSNorm over the full projection.",
  default_norm_eps = 1e-5,
  qk_norm = "full",
  names = {
    input_norm = {},
    post_attn_norm = "post_attention_layernorm.weight",
    pre_ff_norm = false,
    post_ff_norm = "post_feedforward_layernorm.weight",
    q_norm = "self_attn.q_norm.weight",
    k_norm = "self_attn.k_norm.weight",
  },
  config = function(cfg, c)
    c.qk_norm = "full"
  end,
}
