-- The Granite muP multipliers, shared by the Granite families.
local M = {}

-- Embedding, residual, attention and logit multipliers.
function M.multipliers(cfg, c)
  c.embed_scale = num(cfg.embedding_multiplier, 1.0)
  c.residual_multiplier = num(cfg.residual_multiplier, 1.0)
  if type(cfg.attention_multiplier) == "number" then c.attention_scale = cfg.attention_multiplier end
  local ls = f32(num(cfg.logits_scaling, 1.0))
  if ls ~= 0 then c.logit_scale = f32(1.0 / ls) end
end

return M
