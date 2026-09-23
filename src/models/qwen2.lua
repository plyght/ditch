return {
  model_type = "qwen2",
  aliases = { "qwen2_5_vl", "qwen2_5_vl_text", "qwen2_vl", "qwen2_vl_text", "qwen2_5_omni", "qwen2_5_omni_thinker", "qwen2_5_omni_text" },
  llama_cpp = "qwen2",
  chat = "chatml",
  verified = true,
  notes = "fixture: q/k/v biases, tied embeddings. Qwen2-VL / Qwen2.5-VL text configs (nested text_config, mrope over text positions) use the same path.",
  attention_bias = true,
  config = function(cfg, c)
    -- `c.model_type` is the config's own spelling (qwen2_5_vl_text, ...).
    if string.sub(c.model_type, 1, 5) == "qwen2" then c.attention_bias = flag(cfg.attention_bias, true) end
    require("qwen").sliding(cfg, c)
  end,
}
