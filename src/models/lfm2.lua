return {
  model_type = "lfm2",
  llama_cpp = "lfm2",
  chat = "chatml",
  verified = true,
  notes = "fixture: gated short-convolution layers (in_proj B/C/x split, depthwise causal conv with conv_L_cache taps, out_proj) mixed with attention layers carrying per-head q/k norms, block_ff_dim sizing, embedding_norm as the final norm. LFM2 / LFM2.5 dense (lfm2_moe is unsupported).",
  default_norm_eps = 1e-5,
  tie_word_embeddings = true,
  qk_norm = "head",
  names = {
    final_norm = "{p}embedding_norm.weight",
    input_norm = { "operator_norm.weight" },
    pre_ff_norm = "ffn_norm.weight",
    q_norm = "self_attn.q_layernorm.weight",
    k_norm = "self_attn.k_layernorm.weight",
    o = "self_attn.out_proj.weight",
    gate = "feed_forward.w1.weight",
    up = "feed_forward.w3.weight",
    down = "feed_forward.w2.weight",
    conv_in = "conv.in_proj.weight",
    conv_kernel = "conv.conv.weight",
    conv_out = "conv.out_proj.weight",
  },
  config = function(cfg, c)
    if cfg.layer_types == nil then
      -- `full_attn_idxs` lists the attention layers; every other layer is a conv layer.
      local fa = cfg.full_attn_idxs
      if fa ~= nil then
        each_layer(c.conv_layers, function() return true end)
        for k = 1, len(fa) do
          local v = fa[k]
          if math.type(v) == "integer" and v >= 0 and v < c.num_layers then c.conv_layers[v + 1] = false end
        end
      end
      c.has_conv = false
      for _, l in ipairs(c.conv_layers) do c.has_conv = c.has_conv or l end
    end
    c.conv_kernel = int(cfg.conv_L_cache, 3)
    if type(cfg.rope_theta) ~= "number" and obj(cfg.rope_parameters) == nil then c.rope_theta = 1000000.0 end
    c.tie_word_embeddings = flag(cfg.tie_word_embeddings, flag(cfg.tie_embedding, true))
    -- Feed-forward width as Lfm2MLP derives it from `block_ff_dim`.
    local ff = int(cfg.block_ff_dim, c.intermediate_size)
    if flag(cfg.block_auto_adjust_ff_dim, true) then
      ff = math.tointeger(math.floor(2.0 * ff / 3.0))
      local mult = f32(num(cfg.block_ffn_dim_multiplier, 1.0))
      ff = math.tointeger(math.floor(mult * ff))
      local mo = int(cfg.block_multiple_of, 256)
      if mo > 0 then ff = mo * ((ff + mo - 1) // mo) end
    end
    c.intermediate_size = ff
  end,
}
