return {
  model_type = "cohere",
  aliases = { "cohere2" },
  llama_cpp = "command-r",
  chat = "cohere",
  verified = true,
  notes = "fixture: LayerNorm without bias, parallel residual, interleaved rotary, logit_scale, tied embeddings, per-head q/k LayerNorm (use_qk_norm). cohere2 (Command R7B: sliding layers with RoPE, global layers without) is verified on the first four layers of the real checkpoint.",
  norm = "layer",
  default_rope_theta = 500000.0,
  parallel_residual = true,
  rope_style = "gptj",
  tie_word_embeddings = true,
  names = {
    pre_ff_norm = false,
    q_norm = "self_attn.q_norm.weight",
    k_norm = "self_attn.k_norm.weight",
  },
  config = function(cfg, c)
    c.logit_scale = num(cfg.logit_scale, 1.0)
    if flag(cfg.use_qk_norm, false) then c.qk_norm = "heads" end
    if c.model_type == "cohere2" then
      -- Command R7B: local layers use RoPE, global layers none.
      if type(cfg.sliding_window_pattern) ~= "number" then
        each_layer(c.sliding_layers, function(i) return (i + 1) % 4 ~= 0 end)
      end
      each_layer(c.rope_layers, function(i) return c.sliding_layers[i + 1] end)
    end
  end,
}
