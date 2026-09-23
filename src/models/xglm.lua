return {
  model_type = "xglm",
  llama_cpp = nil,
  verified = true,
  notes = "fixture: fairseq sinusoidal positions with offset 2, sqrt(hidden) embedding scale, LayerNorm biases, fc1/fc2 MLP. XGLM.",
  norm = "layer",
  positional = "sinusoidal",
  mlp = "dense",
  activation = "gelu",
  attention_bias = true,
  tie_word_embeddings = true,
  names = {
    final_norm = "{p}layer_norm.weight",
    input_norm = { "self_attn_layer_norm.weight" },
    pre_ff_norm = "final_layer_norm.weight",
    o = "self_attn.out_proj.weight",
    gate = false,
    up = "fc1.weight",
    down = "fc2.weight",
  },
  config = function(cfg, c)
    c.num_kv_heads = c.num_heads
    c.position_offset = 2
    if flag(cfg.scale_embedding, true) then c.embed_scale = f32(math.sqrt(f32(c.hidden_size))) end
  end,
}
