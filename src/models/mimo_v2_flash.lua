return {
  model_type = "mimo_v2_flash",
  aliases = { "mimo_v2" },
  llama_cpp = "mimo2",
  chat = "mimo",
  verified = true,
  notes = "fixtures: hybrid full / sliding-window attention (window 128 in the released configs) with attention sinks and doubled kv heads on the sliding layers, v_head_dim < head_dim with attention_value_scale, partial rotary with one base per layer type (rope_parameters, or rope_theta / swa_rope_theta), a dense first layer (mlp_layer_types / moe_layer_freq) then sigmoid MoE with correction bias and group-limited top-k, no shared experts; both the transformers spelling (layer_types, stacked experts, sinks) and the hub checkpoint spelling of MiMo-V2-Flash / V2.5 / V2.6 (model_type mimo_v2: hybrid_layer_pattern, swa_*, attention_sink_bias, per-expert tensors, the Pro layout's fused qkv_proj chunked per kv head). MTP (model.mtp.*), vision and audio encoder tensors of the V2.5 / V2.6 omni checkpoints pass through exports untouched. The V2.6 checkpoints' MXFP4 experts (quant_method fp8 with store_dtype mxfp4: U8 weight/weight_scale next to the fp8 dense weights) and their bf16 MoE router (moe_router_dtype) are read as they are.",
  default_norm_eps = 1e-5,
  names = {
    qkv = "self_attn.qkv_proj.weight",
    sinks = "self_attn.sinks",
    sinks_alt = "self_attn.attention_sink_bias",
    router_correction_bias = "mlp.gate.e_score_correction_bias",
  },
  hook = "mimo_v2",
}
