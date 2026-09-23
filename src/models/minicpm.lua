return {
  model_type = "minicpm",
  llama_cpp = "minicpm",
  chat = "chatml",
  verified = true,
  notes = "fixture: scale_emb, scale_depth residual scaling, dim_model_base logit scaling. MiniCPM 1/2 (MiniCPM3 with MLA is unsupported).",
  tie_word_embeddings = true,
  hook = "minicpm",
}
