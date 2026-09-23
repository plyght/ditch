return {
  model_type = "falcon",
  aliases = { "RefinedWebModel" },
  llama_cpp = "falcon",
  verified = true,
  notes = "fixture: multi-query fused qkv (7B layout), parallel attention with one LayerNorm. The 40B/180B grouped layout with ln_attn/ln_mlp and the ALiBi variant are implemented but unverified. Falcon-H1 is the `falcon_h1` entry.",
  norm = "layer",
  parallel_residual = true,
  qkv = "concat",
  mlp = "dense",
  activation = "gelu",
  tie_word_embeddings = true,
  names = {
    prefixes = { "transformer.", "" },
    embed = "{p}word_embeddings.weight",
    final_norm = "{p}ln_f.weight",
    layer = "{p}h.{i}.",
    input_norm = { "input_layernorm.weight", "ln_attn.weight" },
    mlp_norm = "ln_mlp.weight",
    q = false,
    k = false,
    v = false,
    qkv = "self_attention.query_key_value.weight",
    o = "self_attention.dense.weight",
    gate = false,
    up = "mlp.dense_h_to_4h.weight",
    down = "mlp.dense_4h_to_h.weight",
  },
  config = function(cfg, c)
    -- Parallel attention unless the config says otherwise.
    if cfg.parallel_attn == nil then c.parallel_residual = true end
    if flag(cfg.new_decoder_architecture, false) then
      -- 40B / 180B: grouped [kv group][q heads | k | v] rows.
      c.qkv_layout = "grouped"
      c.num_kv_heads = int(cfg.num_kv_heads, c.num_heads)
    elseif flag(cfg.multi_query, true) then
      c.qkv_layout = "concat"
      c.num_kv_heads = 1
    else
      c.qkv_layout = "heads_interleaved"
      c.num_kv_heads = c.num_heads
    end
    if flag(cfg.alibi, false) then
      -- Falcon adds the bias before scaling the scores by 1/sqrt(head_dim).
      c.positional = "alibi"
      c.alibi_scale = f32(1 / f32(math.sqrt(c.head_dim)))
    end
  end,
}
