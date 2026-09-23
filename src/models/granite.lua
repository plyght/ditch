return {
  model_type = "granite",
  llama_cpp = "granite",
  chat = "granite",
  verified = true,
  notes = "fixture: embedding, attention and residual multipliers, logits scaling. Granite 3.x dense (GraniteMoE and Granite 4.0 H `granitemoehybrid` have their own entries).",
  config = function(cfg, c)
    c.embed_scale = num(cfg.embedding_multiplier, 1.0)
    c.residual_multiplier = num(cfg.residual_multiplier, 1.0)
    if type(cfg.attention_multiplier) == "number" then c.attention_scale = cfg.attention_multiplier end
    local ls = f32(num(cfg.logits_scaling, 1.0))
    if ls ~= 0 then c.logit_scale = f32(1.0 / ls) end
  end,
}
