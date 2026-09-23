-- The Qwen2 / Qwen3 families (dense and MoE) and their multimodal text configs.
local M = {}

-- The sliding window as transformers applies it: `sliding_window` counts
-- only with `use_sliding_window` (false by default, and in every released
-- config), and then only the layers from `max_window_layers` on slide
-- (unless `layer_types` names them). Without this, a released config's
-- `sliding_window` (32768 or 131072) would make every layer slide.
function M.sliding(cfg, c)
  if not flag(cfg.use_sliding_window, false) then
    c.sliding_window = nil
    each_layer(c.sliding_layers, function() return false end)
  elseif cfg.layer_types == nil and c.sliding_window then
    local first = int(cfg.max_window_layers, 28)
    each_layer(c.sliding_layers, function(i) return i >= first end)
  end
end

return M
