local granite = require("granite")

return {
  model_type = "granite",
  llama_cpp = "granite",
  chat = "granite",
  verified = true,
  notes = "fixture: embedding, attention and residual multipliers, logits scaling. Granite 3.x dense (GraniteMoE and Granite 4.0 H `granitemoehybrid` have their own entries).",
  config = function(cfg, c)
    granite.multipliers(cfg, c)
  end,
}
