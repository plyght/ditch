return {
  model_type = "spark2_5",
  verified = true,
  notes = "fixture: a fused [q | k | v] projection, a per-head sigmoid (or SiLU) gate on the attention output from the layer input (headwise_attn_output_gate), sliding and full layers with their own RoPE base and rotary fraction (rope_parameters per layer type), a gated GELU MLP, embeddings under model.embedding and tied to the output. Spark-X 2.5 / 2.6.",
  qkv = "concat",
  activation = "gelu",
  tie_word_embeddings = true,
  names = {
    embed = "{p}embedding.weight",
    qkv = "self_attn.q_k_v_proj.weight",
    o = "self_attn.out_proj.weight",
    attn_gate = "self_attn.g_proj.weight",
  },
  config = function(cfg, c)
    if flag(cfg.headwise_attn_output_gate, false) then
      local mode = str(cfg.gate_attn_act_mode) or "sigmoid"
      if mode ~= "sigmoid" and mode ~= "silu" then unsupported("gate_attn_act_mode " .. mode) end
      c.attn_gate = mode
    end
  end,
}
