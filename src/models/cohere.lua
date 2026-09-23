return {
  model_type = "cohere",
  aliases = { "cohere2" },
  llama_cpp = "command-r",
  chat = "cohere",
  verified = true,
  notes = "fixture: LayerNorm without bias, parallel residual, interleaved rotary, logit_scale, tied embeddings, per-head q/k LayerNorm (use_qk_norm). cohere2 (Command R7B: sliding layers with RoPE, global layers without) is verified on the first four layers of the real checkpoint.",
  norm = "layer",
  default_rope_theta = 500000.0,
  parallel_residual = true,
  rope_style = "gptj",
  tie_word_embeddings = true,
  names = {
    pre_ff_norm = false,
    q_norm = "self_attn.q_norm.weight",
    k_norm = "self_attn.k_norm.weight",
  },
  hook = "cohere",
}
