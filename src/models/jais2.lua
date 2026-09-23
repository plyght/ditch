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
  config = function(cfg, c)
    -- Arcee, Jais2, Cosmos3 Edge: a llama layout whose MLP is the two-projection
    -- `down(act(up(x)))` form (`relu²` by default, biases from `mlp_bias`).
    if str(cfg.hidden_act) == nil then c.activation = "relu2" end
  end,
}
