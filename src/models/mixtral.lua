return {
  model_type = "mixtral",
  llama_cpp = "llama",
  chat = "mistral",
  verified = true,
  notes = "fixture: softmax top-k routing with renormalisation over the separate per-expert tensors released Mixtral checkpoints store (block_sparse_moe.experts.{e}.w1 / w2 / w3).",
  default_norm_eps = 1e-5,
  default_rope_theta = 1000000.0,
  names = {
    router = "block_sparse_moe.gate.weight",
    expert = "block_sparse_moe.experts.{e}.",
    expert_gate = "w1.weight",
    expert_up = "w3.weight",
    expert_down = "w2.weight",
    fused_gate_up = {},
    fused_down = {},
  },
  hook = "mixtral",
}
