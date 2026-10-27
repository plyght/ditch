return {
  model_type = "memt_qwen3_5_moe",
  chat = "qwen3_5",
  verified = true,
  notes = "fixture: MemT-1.0-Nano (MemTensor): a Qwen3.5-MoE backbone (config under backbone_configs.text_config, weights under model.language_model.) with a persistent memory module on every full-attention layer (self_attn.memt.*, in memt_block.safetensors). The memory is read only when a caller passes a memory state (the release's commit_memory / session API); without one the model is exactly its backbone, which is what ditch runs. The memory tensors pass through exports untouched; its read is added before o_proj, so an o_proj edit also applies to what the memory contributes.",
  -- The Qwen3.5-MoE layout (src/models/qwen3_5_moe.lua; definitions load
  -- in name order, so `base` cannot name it from here).
  norm = "rms_gemma",
  qk_norm = "head",
  names = {
    q_norm = "self_attn.q_norm.weight",
    k_norm = "self_attn.k_norm.weight",
    lin_qkv = "linear_attn.in_proj_qkv.weight",
    lin_z = "linear_attn.in_proj_z.weight",
    lin_b = "linear_attn.in_proj_b.weight",
    lin_a = "linear_attn.in_proj_a.weight",
    lin_conv = "linear_attn.conv1d.weight",
    lin_dt_bias = { "linear_attn.dt_bias" },
    lin_a_log = { "linear_attn.A_log" },
    lin_norm = "linear_attn.norm.weight",
    lin_out = "linear_attn.out_proj.weight",
    shared_expert = "mlp.shared_expert.",
    shared_expert_gate = "mlp.shared_expert_gate.weight",
  },
  config = require("qwen_hybrid").config,
}
