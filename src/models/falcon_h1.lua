local mamba = require("mamba")

return {
  model_type = "falcon_h1",
  llama_cpp = "falcon-h1",
  chat = "chatml",
  verified = true,
  notes = "fixtures: Mamba2 and attention in parallel on one input norm with the muP multipliers (ssm/attention in and out, key, mlp, per-section in_proj, embedding, lm_head), grouped gated RMSNorm with either gate order, and the norm-free variant. Both out projections are abliterated.",
  default_norm_eps = 1e-5,
  ssm = "mamba2",
  parallel_ssm = true,
  names = {
    prefixes = { "model.", "" },
    final_norm = "{p}final_layernorm.weight",
    pre_ff_norm = "pre_ff_layernorm.weight",
    ssm = "mamba.",
    gate = "feed_forward.gate_proj.weight",
    up = "feed_forward.up_proj.weight",
    down = "feed_forward.down_proj.weight",
  },
  config = function(cfg, c)
    mamba.dims(cfg, c)
    local d = c.ssm
    d.norm_groups = d.groups
    d.rms_norm = flag(cfg.mamba_rms_norm, false)
    d.norm_before_gate = flag(cfg.mamba_norm_before_gate, true)
    local m = c.mult
    m.ssm_in = num(cfg.ssm_in_multiplier, 1)
    m.ssm_out = num(cfg.ssm_out_multiplier, 1)
    m.attn_in = num(cfg.attention_in_multiplier, 1)
    m.attn_out = num(cfg.attention_out_multiplier, 1)
    m.key = num(cfg.key_multiplier, 1)
    local mlp = { 1, 1 }
    mamba.float_list(cfg.mlp_multipliers, mlp)
    m.mlp_gate = mlp[1]
    m.mlp_down = mlp[2]
    mamba.float_list(cfg.ssm_multipliers, m.ssm_proj)
    c.embed_scale = num(cfg.embedding_multiplier, 1)
    c.logit_scale = num(cfg.lm_head_multiplier, 1)
    each_layer(c.mlp_layers, function() return true end)
  end,
}
