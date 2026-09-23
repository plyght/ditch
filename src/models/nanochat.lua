return {
  model_type = "nanochat",
  llama_cpp = nil,
  chat = "nanochat",
  verified = true,
  notes = "fixture: non-parametric RMSNorm everywhere (including an extra norm on the embeddings), weightless per-head q/k norm after RoPE, fc1/fc2 relu² MLP, final logit softcapping. nanochat.",
  norm = "rms_none",
  default_norm_eps = 1e-6,
  mlp = "dense",
  activation = "relu2",
  names = {
    embed_norm = "norm.weight",
    gate = false,
    up = "mlp.fc1.weight",
    down = "mlp.fc2.weight",
  },
  hook = "nanochat",
}
