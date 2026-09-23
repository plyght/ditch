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
  config = function(cfg, c)
    -- Arcee, Jais2, Cosmos3 Edge: a llama layout whose MLP is the two-projection
    -- `down(act(up(x)))` form (`relu²` by default, biases from `mlp_bias`).
    if str(cfg.hidden_act) == nil then c.activation = "relu2" end
  end,
}
