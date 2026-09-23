return {
  model_type = "helium",
  llama_cpp = nil,
  chat = "chatml",
  verified = true,
  notes = "fixture: llama layout with mlp/attention biases and interleaved rotary pairing. Helium's reference builds its table as `cat(f, f)` and then takes `[:d/2].repeat_interleave(2)` of it, which is the plain `(x[2i], x[2i+1])` pairing against frequency `i` — the same thing GPT-J does, written differently. Helium 1 (Kyutai).",
  default_norm_eps = 1e-8,
  default_rope_theta = 100000.0,
  rope_style = "gptj",
}
