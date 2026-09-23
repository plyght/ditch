return {
  model_type = "qwen2_moe",
  llama_cpp = "qwen2moe",
  chat = "chatml",
  verified = true,
  notes = "fixture: softmax top-k routing over experts of moe_intermediate_size plus a shared expert of shared_expert_intermediate_size behind a sigmoid shared_expert_gate, both widths different from the dense intermediate_size a mlp_only_layers layer keeps; decoder_sparse_step picks the routed layers.",
  attention_bias = true,
  names = {
    shared_expert = "mlp.shared_expert.",
    shared_expert_gate = "mlp.shared_expert_gate.weight",
  },
  config = require("qwen").sliding,
}
