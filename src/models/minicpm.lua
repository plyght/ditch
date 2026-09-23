return {
  model_type = "minicpm",
  llama_cpp = "minicpm",
  chat = "chatml",
  verified = true,
  notes = "fixture: scale_emb, scale_depth residual scaling, dim_model_base logit scaling. MiniCPM 1/2 (MiniCPM3 with MLA is unsupported).",
  tie_word_embeddings = true,
  config = function(cfg, c)
    c.embed_scale = num(cfg.scale_emb, 1.0)
    c.residual_multiplier = f32(f32(num(cfg.scale_depth, 1.0)) / f32(math.sqrt(f32(c.num_layers))))
    local base = f32(num(cfg.dim_model_base, f32(c.hidden_size)))
    c.logit_scale = f32(base / f32(c.hidden_size))
    if type(cfg.kv_lora_rank) == "number" then
      -- MiniCPM3 (MLA) is not covered.
      unsupported("MiniCPM3 (multi-head latent attention, kv_lora_rank) is not supported")
    end
  end,
}
