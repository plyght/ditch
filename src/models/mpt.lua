return {
  model_type = "mpt",
  llama_cpp = "mpt",
  verified = true,
  notes = "fixture: ALiBi (alibi_bias_max), concatenated Wqkv, LayerNorm without bias, expansion_ratio.",
  norm = "layer",
  positional = "alibi",
  qkv = "concat",
  mlp = "dense",
  activation = "gelu",
  tie_word_embeddings = true,
  names = {
    prefixes = { "transformer.", "" },
    embed = "{p}wte.weight",
    final_norm = "{p}norm_f.weight",
    layer = "{p}blocks.{i}.",
    input_norm = { "norm_1.weight" },
    pre_ff_norm = "norm_2.weight",
    q = false,
    k = false,
    v = false,
    qkv = "attn.Wqkv.weight",
    o = "attn.out_proj.weight",
    gate = false,
    up = "ffn.up_proj.weight",
    down = "ffn.down_proj.weight",
  },
  config = function(cfg, c)
    local ratio = f32(num(cfg.expansion_ratio, 4.0))
    if type(cfg.intermediate_size) ~= "number" then
      c.intermediate_size = math.tointeger(math.floor(f32(f32(c.hidden_size) * ratio)))
    end
    local ac = obj(cfg.attn_config)
    if ac then
      if flag(ac.qk_ln, false) then c.qk_norm = "full" end
      if flag(ac.alibi, false) then c.positional = "alibi" else c.positional = "none" end
    end
    local fc = obj(cfg.ffn_config)
    if fc then
      local t = str(fc.ffn_type)
      if t and t ~= "mptmlp" then
        unsupported("mpt: ffn_type '" .. t .. "' is not supported (only mptmlp)")
      end
    end
  end,
}
