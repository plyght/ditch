return {
  model_type = "smollm3",
  llama_cpp = "smollm3",
  chat = "smollm3",
  verified = true,
  notes = "fixture: llama layout with no_rope_layers.",
  default_rope_theta = 2000000.0,
  tie_word_embeddings = true,
  config = function(cfg, c)
    local flags = require("layers").flags(cfg.no_rope_layers, c.num_layers, true)
    if flags then
      each_layer(c.rope_layers, function(i) return flags[i + 1] end)
    elseif type(cfg.no_rope_layer_interval) == "number" then
      local interval = int(cfg.no_rope_layer_interval, 4)
      if interval > 0 then
        each_layer(c.rope_layers, function(i) return (i + 1) % interval ~= 0 end)
      end
    end
  end,
}
