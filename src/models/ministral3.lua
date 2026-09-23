return {
  model_type = "ministral3",
  llama_cpp = "llama",
  chat = "mistral_v7",
  verified = true,
  notes = "fixture: llama layout with Ministral 3's query scaling (1 + beta*log(1 + floor(pos / max_position_embeddings)) on every layer) and an optional sliding window.",
  default_norm_eps = 1e-5,
  default_rope_theta = 1000000.0,
  hook = "ministral3",
}
