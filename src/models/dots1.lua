local moe = require("moe")

return {
  model_type = "dots1",
  llama_cpp = "dots1",
  chat = "dots",
  verified = true,
  notes = "fixture: per-head q/k norm, DeepSeek-V3 routing (sigmoid scores, correction bias, group-limited top-k, renormalisation, routed scaling), shared experts, first_k_dense_replace. dots.llm1.",
  names = {
    q_norm = "self_attn.q_norm.weight",
    k_norm = "self_attn.k_norm.weight",
    router_correction_bias = "mlp.gate.e_score_correction_bias",
    shared_expert = "mlp.shared_experts.",
  },
  config = function(cfg, c)
    c.qk_norm = "head"
    moe.ds_router(cfg, c)
  end,
}
