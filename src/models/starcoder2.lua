return {
  model_type = "starcoder2",
  llama_cpp = "starcoder2",
  verified = true,
  notes = "fixture: LayerNorm with biases, biased projections, c_fc/c_proj dense MLP, sliding window.",
  norm = "layer",
  mlp = "dense",
  activation = "gelu_tanh",
  attention_bias = true,
  names = {
    gate = false,
    up = "mlp.c_fc.weight",
    down = "mlp.c_proj.weight",
  },
  hook = "starcoder2",
}
