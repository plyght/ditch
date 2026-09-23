return {
  model_type = "qwen3_moe",
  aliases = { "qwen3_vl_moe", "qwen3_vl_moe_text", "qwen3_omni_moe", "qwen3_omni_moe_thinker", "qwen3_omni_moe_text" },
  llama_cpp = "qwen3moe",
  chat = "chatml",
  verified = true,
  notes = "fixtures: softmax top-k with renormalisation, dense layers via mlp_only_layers, separate / fused / transposed-fused expert tensors.",
  qk_norm = "head",
  names = {
    q_norm = "self_attn.q_norm.weight",
    k_norm = "self_attn.k_norm.weight",
  },
}
