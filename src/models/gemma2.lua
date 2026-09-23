return {
  model_type = "gemma2",
  llama_cpp = "gemma2",
  chat = "gemma",
  verified = true,
  notes = "fixture: (1 + w) norms, pre/post feedforward norms, alternating local (sliding) and global layers, query_pre_attn_scalar, sqrt(H) embedding scale, tanh softcapping on the attention logits and on the output logits.",
  norm = "rms_gemma",
  activation = "gelu_tanh",
  tie_word_embeddings = true,
  embed_scale_sqrt = true,
  names = {
    post_attn_norm = "post_attention_layernorm.weight",
    pre_ff_norm = "pre_feedforward_layernorm.weight",
    post_ff_norm = "post_feedforward_layernorm.weight",
  },
  config = require("gemma").config_gemma,
}
