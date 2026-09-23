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
  hook = "hunyuan_dense",
}
