return {
  model_type = "arcee",
  llama_cpp = "arcee",
  chat = "llama3",
  verified = true,
  notes = "fixture: llama attention with the two-projection relu² MLP (no gate), mlp_bias. AFM / Arcee.",
  default_norm_eps = 1e-5,
  mlp = "dense",
  activation = "relu2",
  names = {
    gate = false,
  },
  hook = "dense_mlp",
}
