-- EXAONE 4's layer kinds, shared by the EXAONE families.
local M = {}

-- Sliding (local) layers use RoPE; global layers have no positional encoding.
function M.layer_kinds(cfg, c)
  if c.sliding_window ~= nil and cfg.layer_types == nil then
    local pat = str(cfg.sliding_window_pattern)
    if pat then
      -- e.g. "LLLG": L = local (sliding), G = global.
      if #pat > 0 then
        each_layer(c.sliding_layers, function(i)
          local k = i % #pat + 1
          return pat:sub(k, k) == "L"
        end)
      end
    else
      local pattern = int(cfg.sliding_window_pattern, 4)
      each_layer(c.sliding_layers, function(i) return (i + 1) % pattern ~= 0 end)
    end
  end
  if c.sliding_window ~= nil then
    each_layer(c.rope_layers, function(i) return c.sliding_layers[i + 1] end)
  end
end

return M
