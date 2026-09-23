local moe = require("moe")

return {
  model_type = "deepseek_v3",
  llama_cpp = "deepseek2",
  chat = "deepseek",
  verified = true,
  notes = "fixture: MLA, sigmoid routing with e_score_correction_bias, group-limited (noaux_tc) top-k, routed_scaling_factor, shared experts. BF16/F16 and FP8 block-quantised checkpoints (dequantised on load).",
  names = {
    q_a = "self_attn.q_a_proj.weight",
    q_a_norm = "self_attn.q_a_layernorm.weight",
    q_b = "self_attn.q_b_proj.weight",
    kv_a = "self_attn.kv_a_proj_with_mqa.weight",
    kv_a_norm = "self_attn.kv_a_layernorm.weight",
    kv_b = "self_attn.kv_b_proj.weight",
    router_correction_bias = "mlp.gate.e_score_correction_bias",
    shared_expert = "mlp.shared_experts.",
  },
  config = function(cfg, c)
    moe.deepseek(cfg, c)
  end,
}
