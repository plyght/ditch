local keys = require("hybrid_keys")

return {
  model_type = "raon1_mini",
  verified = true,
  notes = "fixture: Raon-1-mini (KRAFTON): Raon mHC residual streams (sigmoid collapse and expansion selectors and a Helmert chart of a doubly-stochastic stream mixing matrix blended with the identity, embeddings copied into every stream, the mean of the streams into the final norm), Raon Delta Attention layers (Kimi Delta Attention with the safe lower-bound forget gate, full-rank output gate, separate squeezed q/k/v convolutions and the rda_w_proj write gate on the values), NoPE MLA layers with a sigmoid output gate, SiTU-GLU MLPs, sigmoid MoE with correction bias, renormalisation, routed_scaling_factor and a shared expert under moe.; bf16 or the released MXFP8 routed experts (E4M3 with an E8M0 scale per 32 columns).",
  default_norm_eps = 1e-5,
  linear = "kda",
  names = {
    lin_b = "self_attn.b_proj.weight",
    lin_q = "self_attn.q_proj.weight",
    lin_k = "self_attn.k_proj.weight",
    lin_v = "self_attn.v_proj.weight",
    lin_f_a = { "self_attn.f_a_proj.weight" },
    lin_f_b = { "self_attn.f_b_proj.weight" },
    lin_g = "self_attn.g_proj.weight",
    lin_w_gate = "self_attn.rda_w_proj.weight",
    lin_conv_split = { "self_attn.q_conv1d.weight", "self_attn.k_conv1d.weight", "self_attn.v_conv1d.weight" },
    lin_dt_bias = { "self_attn.dt_bias" },
    lin_a_log = { "self_attn.A_log" },
    lin_norm = "self_attn.o_norm.weight",
    lin_out = "self_attn.o_proj.weight",
    attn_gate = "self_attn.g_proj.weight",
    q_a = "self_attn.q_a_proj.weight",
    q_a_norm = "self_attn.q_a_layernorm.weight",
    q_b = "self_attn.q_b_proj.weight",
    kv_a = "self_attn.kv_a_proj_with_mqa.weight",
    kv_a_norm = "self_attn.kv_a_layernorm.weight",
    kv_b = "self_attn.kv_b_proj.weight",
    router = "moe.gate.weight",
    router_correction_bias = "moe.gate.e_score_correction_bias",
    expert = "moe.experts.{e}.",
    shared_expert = "moe.shared_experts.",
    hc_attn = "self_attn_raon_mhc",
    hc_ffn = "mlp_raon_mhc",
    hc_attn_flat = false,
    hc_ffn_flat = false,
  },
  -- `rda_config` holds the delta-attention settings and the 1-indexed
  -- `rda_layers` / `mla_layers`; the full-attention layers are MLA whose
  -- 64 "rope" channels are never rotated (NoPE, `mla_use_nope`).
  config = function(cfg, c)
    local rda = keys.is_object(cfg.rda_config) and cfg.rda_config or {}
    local heads = int(rda.num_heads, c.num_heads)
    local head_dim = int(rda.head_dim, int(cfg.head_dim, 128))
    c.linear_k_heads = heads
    c.linear_v_heads = heads
    c.linear_k_dim = head_dim
    c.linear_v_dim = head_dim
    c.linear_conv_kernel = int(rda.conv_kernel_size, 4)
    -- The output gate is always the full-rank g_proj in this family.
    c.linear_full_rank_gate = true
    if rda.gate_lower_bound == nil then
      c.linear_gate_lower_bound = -5.0
    elseif type(rda.gate_lower_bound) == "number" then
      c.linear_gate_lower_bound = rda.gate_lower_bound
    end
    if int(cfg.rda_decay_rank, head_dim) ~= head_dim then
      unsupported("raon1_mini: rda_decay_rank must equal the RDA head_dim")
    end
    local rl = rda.rda_layers
    if keys.is_array(rl) then
      each_layer(c.linear_layers, function() return false end)
      for _, v in ipairs(rl) do
        if math.type(v) ~= "integer" or v < 1 then invalid("rda_layers entries must be 1-based layer numbers") end
        if v <= c.num_layers then c.linear_layers[v] = true end
      end
    else
      invalid("raon1_mini: rda_config.rda_layers is required")
    end
    keys.update_has_linear(c)
    if c.mla == nil then invalid("raon1_mini: the MLA keys (kv_lora_rank, qk_rope_head_dim, ...) are required") end
    if flag(cfg.mla_use_nope, true) == false then
      unsupported("raon1_mini: MLA layers with RoPE (mla_use_nope = false) are not implemented")
    end
    c.mla.latent_norm_eps = f32(num(cfg.mla_latent_norm_eps, 1e-6))
    c.mla_output_gate = flag(cfg.mla_use_output_gate, true)
    each_layer(c.rope_layers, function() return false end)
    -- Router: sigmoid scores, a correction bias that only steers the choice,
    -- renormalisation over sum + 1e-20, routed_scaling_factor.
    if (str(cfg.moe_router_activation_func) or "sigmoid") ~= "sigmoid" then
      unsupported("raon1_mini: only the sigmoid router is implemented")
    end
    c.num_experts_per_tok = keys.int_any(cfg, { "num_experts_per_tok", "num_experts_per_token" }, 6)
    c.norm_topk_prob = flag(cfg.moe_renormalize, true)
    c.moe.scoring = "sigmoid"
    c.moe.topk_method = "group_limited"
    c.moe.n_group = 1
    c.moe.topk_group = 1
    c.moe.routed_scaling_factor = num(cfg.routed_scaling_factor, 1.0)
    c.moe.norm_eps = 1e-20
    -- Raon mHC streams.
    local streams = int(cfg.raon_mhc_streams, 4)
    if streams < 2 or streams > 16 then unsupported("raon1_mini: raon_mhc_streams must be 2 to 16") end
    c.hc_mult = streams
    c.hyper = {
      kind = "raon",
      head = "mean",
      sinkhorn_iters = 0,
      eps = num(cfg.raon_mhc_selector_eps, 1e-5),
      lowrank = 0,
      post_factor = num(cfg.raon_mhc_post_factor, 2.0),
    }
  end,
}
