return {
  model_type = "hunyuan_v1_moe",
  aliases = { "hunyuan" },
  llama_cpp = "hunyuan-moe",
  chat = "hunyuan_moe",
  verified = true,
  notes = "fixture: per-head q/k RMSNorm after RoPE, NTK-alpha dynamic rope base, softmax top-k renormalised, shared MLP, per-layer (uniform) expert counts. Hunyuan-A13B.",
  default_norm_eps = 1e-5,
  qk_norm = "head",
  names = {
    q_norm = "self_attn.query_layernorm.weight",
    k_norm = "self_attn.key_layernorm.weight",
    router = "mlp.gate.wg.weight",
    shared_expert = "mlp.shared_mlp.",
  },
  hook = "hunyuan_moe",
}
