return {
  model_type = "nemotron",
  llama_cpp = "nemotron",
  verified = true,
  notes = "fixture: LayerNorm1p with bias, relu² dense MLP, partial rotary. Nemotron-H is the `nemotron_h` entry.",
  norm = "layer_1p",
  mlp = "dense",
  activation = "relu2",
  names = {
    gate = false,
  },
  config = function(cfg, c)
    if str(cfg.hidden_act) == nil then c.activation = "relu2" end
    c.attention_bias = flag(cfg.attention_bias, false)
  end,
}
