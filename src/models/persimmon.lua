return {
  model_type = "persimmon",
  llama_cpp = "persimmon",
  verified = true,
  notes = "fixture: head-interleaved query_key_value with bias, per-head q/k LayerNorm, partial rotary, dense_h_to_4h/dense_4h_to_h relu² MLP. Persimmon 8B (and Fuyu's text tower).",
  norm = "layer",
  qkv = "heads_interleaved",
  mlp = "dense",
  activation = "relu2",
  attention_bias = true,
  names = {
    final_norm = "{p}final_layernorm.weight",
    q_norm = "self_attn.q_layernorm.weight",
    k_norm = "self_attn.k_layernorm.weight",
    q = false,
    k = false,
    v = false,
    qkv = "self_attn.query_key_value.weight",
    o = "self_attn.dense.weight",
    gate = false,
    up = "mlp.dense_h_to_4h.weight",
    down = "mlp.dense_4h_to_h.weight",
  },
  config = function(cfg, c)
    if flag(cfg.qk_layernorm, true) then c.qk_norm = "head" end
    c.num_kv_heads = c.num_heads
  end,
}
