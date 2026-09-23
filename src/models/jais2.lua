return {
  model_type = "jais2",
  llama_cpp = nil,
  chat = "chatml",
  verified = true,
  notes = "fixture: LayerNorm with biases, biased projections, the two-projection relu² MLP. Jais 2.",
  norm = "layer",
  mlp = "dense",
  activation = "relu2",
  attention_bias = true,
  names = {
    gate = false,
  },
  hook = "dense_mlp",
}
