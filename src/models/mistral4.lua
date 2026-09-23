return {
  model_type = "mistral4",
  aliases = { "mistral4_text" },
  llama_cpp = "mistral4",
  chat = "mistral",
  verified = true,
  notes = "fixture (text): MLA with interleaved rotary, yarn (mscale_all_dim) from rope_parameters, llama_4_scaling_beta query scaling on every layer, softmax group-limited top-k (two best experts per group) with renormalisation, fused [E][2I][H] experts, shared experts, first_k_dense_replace. Mistral Small 4 text config.",
  names = {
    q_a = "self_attn.q_a_proj.weight",
    q_a_norm = "self_attn.q_a_layernorm.weight",
    q_b = "self_attn.q_b_proj.weight",
    kv_a = "self_attn.kv_a_proj_with_mqa.weight",
    kv_a_norm = "self_attn.kv_a_layernorm.weight",
    kv_b = "self_attn.kv_b_proj.weight",
    shared_expert = "mlp.shared_experts.",
  },
  hook = "mistral4",
}
