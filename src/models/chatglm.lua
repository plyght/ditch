return {
  model_type = "chatglm",
  llama_cpp = "chatglm",
  chat = "glm4",
  verified = true,
  notes = "fixture: ChatGLM3 / GLM-4 (remote-code layout): concatenated query_key_value with bias, fused dense_h_to_4h, interleaved half rotary with rope_ratio, output_layer.",
  rope_style = "gptj",
  qkv = "concat",
  mlp = "gated_fused",
  attention_bias = true,
  names = {
    prefixes = { "transformer." },
    embed = "{p}embedding.word_embeddings.weight",
    final_norm = "{p}encoder.final_layernorm.weight",
    lm_head = { "transformer.output_layer.weight" },
    layer = "{p}encoder.layers.{i}.",
    q = false,
    k = false,
    v = false,
    qkv = "self_attention.query_key_value.weight",
    o = "self_attention.dense.weight",
    gate = false,
    up = false,
    gate_up = "mlp.dense_h_to_4h.weight",
    down = "mlp.dense_4h_to_h.weight",
  },
  config = function(cfg, c)
    -- Rotary embeddings cover half of `kv_channels`, interleaved; theta is scaled by rope_ratio.
    c.rotary_dim = c.head_dim // 2
    c.rope_freq_dim = c.rotary_dim
    c.rope_theta = f32(10000.0 * f32(num(cfg.rope_ratio, 1.0)))
    c.attention_bias = flag(cfg.add_qkv_bias, true)
    if not flag(cfg.rmsnorm, true) then c.norm = "layer" end
    if flag(cfg.apply_residual_connection_post_layernorm, false) then
      unsupported("chatglm: apply_residual_connection_post_layernorm (the residual taken after the input norm) is not supported")
    end
    if flag(cfg.post_layer_norm, true) == false then
      unsupported("chatglm: post_layer_norm = false (no final layer norm) is not supported")
    end
    if flag(cfg.multi_query_attention, false) then
      c.num_kv_heads = int(cfg.multi_query_group_num, c.num_heads)
    else
      c.num_kv_heads = c.num_heads
    end
    c.tie_word_embeddings = flag(cfg.tie_word_embeddings, false)
  end,
}
