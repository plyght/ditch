local moe = require("moe")

return {
  model_type = "deepseek_v2",
  aliases = { "deepseek_ocr2", "deepseek_ocr2_text", "youtu" },
  llama_cpp = "deepseek2",
  chat = "deepseek_v2",
  verified = true,
  notes = "fixture: MLA with q_lora_rank (q_a/q_b) and without, softmax routing with group-limited top-k, shared experts, first_k_dense_replace, yarn with mscale. BF16/F16 and FP8 block-quantised checkpoints.",
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
