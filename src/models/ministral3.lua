return {
  model_type = "ministral3",
  llama_cpp = "llama",
  chat = "mistral_v7",
  verified = true,
  notes = "fixture: llama layout with Ministral 3's query scaling (1 + beta*log(1 + floor(pos / max_position_embeddings)) on every layer) and an optional sliding window.",
  default_norm_eps = 1e-5,
  default_rope_theta = 1000000.0,
  config = function(cfg, c)
    -- Ministral 3 scales the queries of every layer by
    -- `1 + beta * log(1 + floor(pos / max_position_embeddings))`.
    local beta = nil
    local rp = obj(cfg.rope_parameters)
    if rp then beta = num(rp.llama_4_scaling_beta, beta) end
    beta = num(cfg.llama_4_scaling_beta, beta)
    if beta then
      c.attn_temperature = {
        floor_scale = c.max_position_embeddings,
        attn_scale = beta,
        offset = 0,
        all_layers = true,
      }
    end
  end,
}
