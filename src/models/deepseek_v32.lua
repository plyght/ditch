return {
  model_type = "deepseek_v32",
  llama_cpp = "deepseek2",
  chat = "deepseek",
  verified = true,
  notes = "fixture: DeepSeek V3.2-Exp, the V3 layout (MLA, sigmoid group-limited routing, shared experts) whose `indexed_attention` layers select the top `index_topk` keys with a lightning indexer; ditch runs them as dense attention, which is exactly the reference for prompts up to index_topk tokens and refused beyond it. The indexer's own tensors are never read and pass through exports untouched.",
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
  hook = "deepseek_v32",
}
