return {
  model_type = "memt_deepseek_v4",
  base = "deepseek_v4",
  verified = true,
  notes = "fixture: MemT-1.0-Flash (MemTensor): a DeepSeek-V4-Flash backbone (flat deepseek_v4 config, bf16 experts) with a persistent memory module on the attention of the layers in memt.layers (self_attn.memt.*, in memt-*.safetensors). The memory is read only when a caller passes a memory state (the release's commit_memory / session API); without one the model is exactly its backbone, which is what ditch runs. The memory tensors pass through exports untouched; its read enters through o_a_proj / o_b_proj, so an o_b_proj edit also applies to what the memory contributes.",
}
