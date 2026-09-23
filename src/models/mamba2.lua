local mamba = require("mamba")

return {
  model_type = "mamba2",
  llama_cpp = "mamba2",
  verified = true,
  notes = "fixture: pure Mamba2 (SSD) blocks: in_proj split into gate / conv channels / dt, biased causal conv1d, grouped B/C, per-head decay, D skip, gated RMSNorm, out_proj; no attention, no MLP. Mamba-Codestral, state-spaces/mamba2-*-hf.",
  default_norm_eps = 1e-5,
  positional = "none",
  ssm = "mamba2",
  single_mixer = true,
  names = {
    prefixes = { "backbone.", "" },
    embed = "{p}embeddings.weight",
    final_norm = "{p}norm_f.weight",
    input_norm = { "norm.weight" },
    pre_ff_norm = false,
    ssm = "mixer.",
  },
  config = function(cfg, c)
    local d = c.ssm
    d.kind = "mamba2"
    d.heads = int(cfg.num_heads, 128)
    d.head_dim = int(cfg.head_dim, 64)
    d.inter = d.heads * d.head_dim
    d.state = int(cfg.state_size, 128)
    d.groups = int(cfg.n_groups, 8)
    d.conv_kernel = int(cfg.conv_kernel, 4)
    d.act = c.activation
    mamba.dt_limit(cfg, d)
    local want = math.floor(f32(f32(num(cfg.expand, 2)) * f32(c.hidden_size)))
    if want ~= d.inter then
      invalid(string.format("mamba2: expand * hidden_size (%d) must equal num_heads * head_dim (%d)", want, d.inter))
    end
    -- No attention anywhere: placeholder attention dimensions keep the KV
    -- cache (one float per position) out of the way.
    c.num_heads = 1
    c.num_kv_heads = 1
    c.head_dim = 1
    c.v_head_dim = 1
    c.rotary_dim = 0
    c.positional = "none"
    each_layer(c.ssm_layers, function() return true end)
    each_layer(c.attn_layers, function() return false end)
    each_layer(c.mlp_layers, function() return false end)
    each_layer(c.moe_layers, function() return false end)
  end,
}
