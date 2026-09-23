local moe = require("moe")

return {
  model_type = "kimi_k25",
  llama_cpp = "deepseek2",
  chat = "kimi",
  verified = true,
  notes = "fixture: the Kimi K2.5 / K2.6 image-video wrapper (Kimi_K25ForConditionalGeneration) around a DeepSeek V3 text config (model_type kimi_k2 or deepseek_v3 under text_config): MLA, sigmoid routing with correction bias and group-limited top-k, shared experts, language_model prefix. The vision tower and projector pass through exports untouched..",
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
