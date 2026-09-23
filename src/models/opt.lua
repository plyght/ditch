return {
  model_type = "opt",
  llama_cpp = nil,
  verified = true,
  notes = "fixture: learned positions with offset 2, ReLU, LayerNorm biases. Pre-norm variants only (OPT-350m's projection layers are unsupported).",
  norm = "layer",
  positional = "learned",
  mlp = "dense",
  activation = "relu",
  attention_bias = true,
  tie_word_embeddings = true,
  names = {
    prefixes = { "model.decoder.", "decoder." },
    pos_embed = "{p}embed_positions.weight",
    final_norm = "{p}final_layer_norm.weight",
    input_norm = { "self_attn_layer_norm.weight" },
    pre_ff_norm = "final_layer_norm.weight",
    o = "self_attn.out_proj.weight",
    gate = false,
    up = "fc1.weight",
    down = "fc2.weight",
  },
  config = function(cfg, c)
    c.position_offset = 2
    if not flag(cfg.do_layer_norm_before, true) then
      unsupported("opt: do_layer_norm_before = false (post-norm layers, OPT-350m) is not supported")
    end
    if int(cfg.word_embed_proj_dim, c.hidden_size) ~= c.hidden_size then
      unsupported("opt: word_embed_proj_dim different from hidden_size (projected embeddings, OPT-350m) is not supported")
    end
    c.attention_bias = flag(cfg.enable_bias, true)
  end,
}
