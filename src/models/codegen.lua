return {
  model_type = "codegen",
  llama_cpp = nil,
  verified = true,
  notes = "fixture: the GPT-J layout with a fused qkv_proj laid out as four tensor-parallel [q | v | k] blocks. CodeGen / CodeGen 2.",
  norm = "layer",
  parallel_residual = true,
  rope_style = "gptj",
  qkv = "mp_blocks",
  mlp = "dense",
  activation = "gelu_tanh",
  names = {
    prefixes = { "transformer.", "" },
    embed = "{p}wte.weight",
    final_norm = "{p}ln_f.weight",
    layer = "{p}h.{i}.",
    input_norm = { "ln_1.weight" },
    pre_ff_norm = false,
    q = false,
    k = false,
    v = false,
    qkv = "attn.qkv_proj.weight",
    o = "attn.out_proj.weight",
    gate = false,
    up = "mlp.fc_in.weight",
    down = "mlp.fc_out.weight",
  },
  config = function(cfg, c)
    require("gptj").config(cfg, c)
    -- The fused projection is `mp_num` tensor-parallel blocks of [q | v | k].
    c.qkv_mp = 4
    if c.num_heads % c.qkv_mp ~= 0 then
      unsupported("codegen: num_attention_heads must be a multiple of the 4 tensor-parallel blocks of qkv_proj")
    end
  end,
}
