-- GPT-J's reading of config.json, shared by CodeGen (the GPT-J layout with a
-- fused projection).
local M = {}

function M.config(cfg, c)
  c.num_kv_heads = c.num_heads
  if type(cfg.rotary_dim) ~= "number" then c.rotary_dim = c.head_dim end
end

return M
