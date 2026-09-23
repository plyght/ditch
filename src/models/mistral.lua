return {
  model_type = "mistral",
  aliases = { "ministral" },
  llama_cpp = "llama",
  chat = "mistral",
  verified = true,
  notes = "fixture: a sliding window on every layer (shorter than the prompt, so the local mask bites) and an explicit head_dim that is not hidden_size / num_attention_heads. Otherwise the llama layout.",
}
