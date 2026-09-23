return {
  model_type = "exaone4",
  llama_cpp = "exaone4",
  chat = "exaone4",
  verified = true,
  notes = "fixture: post-norms, per-head q/k norm, hybrid sliding layers with RoPE and global layers without.",
  default_norm_eps = 1e-5,
  qk_norm = "head",
  names = {
    input_norm = {},
    post_attn_norm = "post_attention_layernorm.weight",
    pre_ff_norm = false,
    post_ff_norm = "post_feedforward_layernorm.weight",
    q_norm = "self_attn.q_norm.weight",
    k_norm = "self_attn.k_norm.weight",
  },
  config = function(cfg, c)
    -- Sliding (local) layers use RoPE; global layers have no positional encoding.
    if c.sliding_window and cfg.layer_types == nil then
      local pat = str(cfg.sliding_window_pattern)
      if pat then
        -- e.g. "LLLG": L = local (sliding), G = global.
        if #pat > 0 then
          each_layer(c.sliding_layers, function(i) return pat:byte(i % #pat + 1) == string.byte("L") end)
        end
      else
        local pattern = int(cfg.sliding_window_pattern, 4)
        each_layer(c.sliding_layers, function(i) return (i + 1) % pattern ~= 0 end)
      end
    end
    if c.sliding_window then
      each_layer(c.rope_layers, function(i) return c.sliding_layers[i + 1] end)
    end
  end,
}
