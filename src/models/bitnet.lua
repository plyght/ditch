return {
  model_type = "bitnet",
  llama_cpp = "bitnet-25",
  chat = "chatml",
  verified = true,
  notes = "fixture: the sub-layer RMSNorms on the attention output and the gated MLP intermediate, relu² activation. No *released* BitNet checkpoint can be run: b1.58 ternarises its weights and quantises its activations inside every linear at run time (quantization_config.quant_method = bitnet), so neither the packed release nor the bf16 master weights are the model the reference runs; both are refused with that reason. The entry covers the layout for a checkpoint that ships plain weights.",
  default_norm_eps = 1e-5,
  default_rope_theta = 500000.0,
  activation = "relu2",
  names = {
    attn_sub_norm = "self_attn.attn_sub_norm.weight",
    ffn_sub_norm = "mlp.ffn_sub_norm.weight",
  },
}
