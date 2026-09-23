local granite = require("granite")

return {
  model_type = "granite_swa",
  llama_cpp = "granite",
  chat = "granite",
  verified = true,
  notes = "fixture: the Granite multipliers plus per-head attention sinks (an extra softmax logit) and sliding layers with their own rope base (layer_rope_theta). Granite 4 SWA dense.",
  default_norm_eps = 1e-5,
  names = {
    sinks = "self_attn.sinks",
  },
  -- Granite 4 SWA: the Granite multipliers plus per-head attention sinks and
  -- sliding layers with their own rope base (`layer_rope_theta`).
  config = function(cfg, c)
    granite.multipliers(cfg, c)
    c.sinks = true
  end,
}
