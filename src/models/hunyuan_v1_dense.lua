return {
  model_type = "hunyuan_v1_dense",
  aliases = { "hunyuan_vl_text", "hunyuan_vl" },
  llama_cpp = "hunyuan-dense",
  chat = "hunyuan",
  verified = true,
  notes = "fixture: per-head q/k RMSNorm after RoPE, NTK-alpha dynamic rope base. Hunyuan dense (and the HunYuan-VL text config).",
  default_norm_eps = 1e-5,
  qk_norm = "head",
  names = {
    q_norm = "self_attn.query_layernorm.weight",
    k_norm = "self_attn.key_layernorm.weight",
  },
  config = function(cfg, c)
    -- Per-head q/k RMSNorm after RoPE, NTK-alpha "dynamic" scaling folded into the base.
    c.qk_norm_after_rope = true
    c.attention_bias = flag(cfg.attention_bias, false)
    local rs = obj(cfg.rope_scaling) or obj(cfg.rope_parameters)
    if rs then
      local t = str(rs.rope_type) or str(rs.type) or ""
      if t == "dynamic" and type(rs.alpha) == "number" then
        local d = c.head_dim + 0.0
        c.rope_theta = f32(c.rope_theta * pow(rs.alpha, d / (d - 2.0)))
      end
    end
  end,
}
