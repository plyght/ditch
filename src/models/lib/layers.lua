-- Per-layer settings read from config.json lists.
local M = {}

-- Reads a JSON array of 0/1 or bools into a per-layer table of `n` entries
-- (missing entries keep `default`); nil when `v` is missing or not an array.
-- An empty JSON object cannot be told from an empty array and reads as one.
function M.flags(v, n, default)
  if type(v) ~= "table" or v == null or (#v == 0 and next(v) ~= nil) then return nil end
  local out = {}
  for i = 1, n do out[i] = default end
  for i = 1, math.min(#v, n) do out[i] = flag(v[i], default) end
  return out
end

return M
