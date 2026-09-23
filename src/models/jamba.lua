local mamba = require("mamba")

return {
  model_type = "jamba",
  llama_cpp = "jamba",
  chat = "jamba",
  verified = true,
  notes = "fixture: Mamba1 layers (in_proj, conv1d, x_proj with RMS-normalised dt/B/C, dt_proj, per-channel A_log, D, silu(z) gate) at attn_layer_period / offset, attention without positional encoding, softmax MoE at expert_layer_period / offset (separate expert tensors) and dense MLPs.",
  positional = "none",
  ssm = "mamba1",
  names = {
    prefixes = { "model.", "" },
    final_norm = "{p}final_layernorm.weight",
    pre_ff_norm = "pre_ff_layernorm.weight",
    ssm = "mamba.",
    gate = "feed_forward.gate_proj.weight",
    up = "feed_forward.up_proj.weight",
    down = "feed_forward.down_proj.weight",
    router = "feed_forward.router.weight",
    expert = "feed_forward.experts.{e}.",
    fused_gate_up = { "feed_forward.experts.gate_up_proj" },
    fused_down = { "feed_forward.experts.down_proj" },
  },
  config = function(cfg, c)
    local d = c.ssm
    d.kind = "mamba1"
    d.inter = math.floor(f32(f32(num(cfg.mamba_expand, 2)) * f32(c.hidden_size)))
    d.state = int(cfg.mamba_d_state, 16)
    d.conv_kernel = int(cfg.mamba_d_conv, 4)
    -- `mamba_dt_rank` may be the string "auto": ceil(hidden / 16).
    if type(cfg.mamba_dt_rank) == "number" then
      d.dt_rank = int(cfg.mamba_dt_rank, 0)
    else
      d.dt_rank = (c.hidden_size + 15) // 16
    end
    d.act = c.activation
    -- Attention layers carry no positional encoding.
    c.positional = "none"
    if cfg.layer_types == nil and cfg.layers_block_type == nil then
      local period = math.max(1, int(cfg.attn_layer_period, 8))
      local offset = int(cfg.attn_layer_offset, 4)
      for i = 0, c.num_layers - 1 do
        c.attn_layers[i + 1] = i % period == offset
        c.ssm_layers[i + 1] = not c.attn_layers[i + 1]
      end
    end
    -- Every layer has an MLP; every `expert_layer_period`-th one is a mixture.
    if c.num_experts <= 1 then c.num_experts = 0 end
    local eperiod = math.max(1, int(cfg.expert_layer_period, 2))
    local eoffset = int(cfg.expert_layer_offset, 1)
    each_layer(c.moe_layers, function(i) return c.num_experts > 0 and i % eperiod == eoffset end)
    each_layer(c.mlp_layers, function() return true end)
  end,
}
