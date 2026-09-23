local mamba = require("mamba")

return {
  model_type = "nemotron_h",
  llama_cpp = "nemotron_h",
  verified = true,
  notes = "fixture: one block per layer from hybrid_override_pattern / layers_block_type (M: Mamba2 with grouped gated norm and dt floor, *: attention without positional encoding, -: relu² MLP, E: non-gated experts with sigmoid group-limited routing, correction bias and a shared expert). Nemotron-H, Nemotron 3 Nano (llama.cpp: nemotron_h_moe).",
  default_norm_eps = 1e-5,
  positional = "none",
  mlp = "dense",
  activation = "relu2",
  ssm = "mamba2",
  single_mixer = true,
  names = {
    prefixes = { "backbone.", "" },
    embed = "{p}embeddings.weight",
    final_norm = "{p}norm_f.weight",
    input_norm = { "norm.weight" },
    pre_ff_norm = false,
    q = "mixer.q_proj.weight",
    k = "mixer.k_proj.weight",
    v = "mixer.v_proj.weight",
    o = "mixer.o_proj.weight",
    ssm = "mixer.",
    gate = false,
    up = "mixer.up_proj.weight",
    down = "mixer.down_proj.weight",
    router = "mixer.gate.weight",
    router_correction_bias = "mixer.gate.e_score_correction_bias",
    expert = "mixer.experts.{e}.",
    fused_gate_up = {},
    fused_down = {},
    shared_expert = "mixer.shared_experts.",
  },
  config = function(cfg, c)
    local d = c.ssm
    d.kind = "mamba2"
    d.heads = int(cfg.mamba_num_heads, 128)
    d.head_dim = int(cfg.mamba_head_dim, 64)
    d.inter = d.heads * d.head_dim
    d.state = int(cfg.ssm_state_size, 128)
    d.groups = int(cfg.n_groups, 8)
    d.conv_kernel = int(cfg.conv_kernel, 4)
    -- The gated RMSNorm normalises each B/C group of channels separately,
    -- and the time step is floored at `time_step_min` (the chunked scan's
    -- `dt_limit`; ditch applies it on every token).
    d.norm_groups = d.groups
    d.dt_min = num(cfg.time_step_min, 0.001)
    d.act = "silu"
    local a = str(cfg.mamba_hidden_act)
    if a then
      d.act = mamba.activation(a) or unsupported("nemotron_h: unknown mamba_hidden_act '" .. a .. "'")
    end
    -- MLP and expert activation: `mlp_hidden_act` (relu²), not `hidden_act`.
    c.activation = "relu2"
    a = str(cfg.mlp_hidden_act)
    if a then
      c.activation = mamba.activation(a) or unsupported("nemotron_h: unknown mlp_hidden_act '" .. a .. "'")
    end
    -- Attention layers carry no positional encoding.
    c.positional = "none"
    if cfg.layer_types == nil and cfg.layers_block_type == nil then
      local pat = str(cfg.hybrid_override_pattern)
      if not pat then invalid("nemotron_h: config.json needs layers_block_type or hybrid_override_pattern") end
      if #pat ~= c.num_layers then
        invalid(string.format("nemotron_h: hybrid_override_pattern has %d entries for %d layers", #pat, c.num_layers))
      end
      for i = 1, #pat do
        local ch = pat:sub(i, i)
        c.ssm_layers[i] = ch == "M"
        c.attn_layers[i] = ch == "*"
        c.mlp_layers[i] = ch == "-" or ch == "E"
        c.moe_layers[i] = ch == "E"
        if ch ~= "M" and ch ~= "*" and ch ~= "-" and ch ~= "E" then
          unsupported("nemotron_h: unknown block '" .. ch .. "' in hybrid_override_pattern")
        end
      end
    end
    local any_moe = false
    for _, m in ipairs(c.moe_layers) do any_moe = any_moe or m end
    if any_moe and c.num_experts == 0 then invalid("nemotron_h: moe layers without n_routed_experts") end
    if type(cfg.moe_latent_size) == "number" then
      unsupported("nemotron_h latent expert projections (moe_latent_size) are not implemented")
    end
    c.moe.scoring = "sigmoid"
    c.moe.topk_method = "group_limited"
    c.moe.n_group = math.max(1, int(cfg.n_group, 1))
    c.moe.topk_group = math.max(1, int(cfg.topk_group, 1))
    c.moe.routed_scaling_factor = num(cfg.routed_scaling_factor, 1.0)
    c.moe.dense_experts = true
    if c.moe.topk_method == "group_limited" and c.num_experts > 0 and c.num_experts % c.moe.n_group ~= 0 then
      invalid(string.format("nemotron_h: n_routed_experts (%d) is not a multiple of n_group (%d)", c.num_experts, c.moe.n_group))
    end
  end,
}
