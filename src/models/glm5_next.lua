local moe = require("moe")

return {
  model_type = "glm5_next",
  aliases = { "glm5_next_text" },
  llama_cpp = nil,
  chat = "glm4",
  verified = true,
  notes = "fixture: GLM-5.3-Flash text config under the multimodal wrapper: manifold-constrained hyper-connections collapsed by an unweighted mean, Kimi Delta Attention layers with the safe lower-bound forget gate, NoPE MLA layers behind a k-pool DSA indexer with full and shared indexer types (run as dense: exact while every complete pool fits index_topk, longer prompts refused), sigmoid MoE with correction bias, top-2 group scores, floored renormalisation, routed_scaling_factor and shared experts, clamped SwiGLU in experts, shared experts and dense layers, mlp_layer_types. Vision and indexer tensors pass through exports untouched.",
  default_norm_eps = 1e-5,
  linear = "kda",
  names = {
    lin_b = "self_attn.b_proj.weight",
    lin_q = "self_attn.q_proj.weight",
    lin_k = "self_attn.k_proj.weight",
    lin_v = "self_attn.v_proj.weight",
    lin_f_a = { "self_attn.forget_gate.f_a_proj.weight", "self_attn.f_a_proj.weight" },
    lin_f_b = { "self_attn.forget_gate.f_b_proj.weight", "self_attn.f_b_proj.weight" },
    lin_g_a = "self_attn.g_a_proj.weight",
    lin_g_b = "self_attn.g_b_proj.weight",
    lin_conv = "self_attn.conv1d.weight",
    lin_conv_split = { "self_attn.q_conv1d.weight", "self_attn.k_conv1d.weight", "self_attn.v_conv1d.weight" },
    lin_dt_bias = { "self_attn.forget_gate.dt_bias", "self_attn.dt_bias" },
    lin_a_log = { "self_attn.forget_gate.A_log", "self_attn.A_log" },
    lin_norm = "self_attn.o_norm.weight",
    lin_out = "self_attn.o_proj.weight",
    q_a = "self_attn.q_a_proj.weight",
    q_a_norm = "self_attn.q_a_layernorm.weight",
    q_b = "self_attn.q_b_proj.weight",
    kv_a = "self_attn.kv_a_proj_with_mqa.weight",
    kv_a_norm = "self_attn.kv_a_layernorm.weight",
    kv_b = "self_attn.kv_b_proj.weight",
    router_correction_bias = "mlp.gate.e_score_correction_bias",
    shared_expert = "mlp.shared_experts.",
  },
  -- GLM-5.3-Flash (`glm5_next`, `Glm5NextTextConfig`): Kimi Delta Attention
  -- layers (with the safe lower-bound forget gate) 3:1 with NoPE MLA layers
  -- whose DSA indexer pools `index_kpool` keys (run as its dense equivalent
  -- within `index_topk`), manifold-constrained hyper-connections collapsed by
  -- an unweighted mean, DeepSeek V3 routing and clamped SwiGLU everywhere.
  config = function(cfg, c)
    local lac = obj(cfg.linear_attn_config)
    local heads = int(cfg.linear_num_heads, 64)
    local head_dim = int(cfg.linear_head_dim, 128)
    local kernel = int(cfg.linear_conv_kernel_dim, 4)
    -- `linear_lower_bound` (default -5.0; an explicit null selects the
    -- softplus gate), or `linear_attn_config.gate_lower_bound` with
    -- `safe_gate` restoring the default when it is null.
    local lower = -5.0
    if cfg.linear_lower_bound ~= nil then
      lower = nil
      if type(cfg.linear_lower_bound) == "number" then lower = f32(cfg.linear_lower_bound) end
    end
    if lac then
      heads = int(lac.num_heads, heads)
      head_dim = int(lac.head_dim, head_dim)
      kernel = int(lac.short_conv_kernel_size, kernel)
      if lac.gate_lower_bound ~= nil then
        lower = nil
        if type(lac.gate_lower_bound) == "number" then lower = f32(lac.gate_lower_bound) end
      end
      if flag(lac.safe_gate, true) and lower == nil then lower = -5.0 end
    end
    c.linear_k_heads = heads
    c.linear_v_heads = heads
    c.linear_k_dim = head_dim
    c.linear_v_dim = head_dim
    c.linear_conv_kernel = kernel
    c.linear_gate_lower_bound = lower
    if cfg.layer_types == nil then
      each_layer(c.linear_layers, function(i) return i % 4 ~= 3 end)
    end
    c.has_linear = false
    for _, l in ipairs(c.linear_layers) do c.has_linear = c.has_linear or l end
    moe.mla_nope(cfg, c, "glm5_next")
    if type(cfg.rms_norm_eps) ~= "number" then c.rms_norm_eps = 1e-5 end
    -- Mixture of experts: sigmoid scores, correction bias, top-2 group
    -- scores, renormalised with a floor, scaled; clamped SwiGLU in the
    -- experts, the shared experts and the dense layers.
    c.num_experts_per_tok = int(cfg.num_experts_per_tok, 8)
    c.norm_topk_prob = flag(cfg.norm_topk_prob, true)
    c.moe.scoring = "sigmoid"
    c.moe.topk_method = "group_limited"
    c.moe.n_group = math.max(1, int(cfg.n_group, 1))
    c.moe.topk_group = math.max(1, int(cfg.topk_group, 1))
    c.moe.routed_scaling_factor = num(cfg.routed_scaling_factor, 2.5)
    c.moe.group_score_top2 = true
    c.moe.norm_eps_floor = true
    if c.num_experts % c.moe.n_group ~= 0 then invalid("the routed experts do not split into n_group groups") end
    local limit = f32(num(cfg.swiglu_limit, 10.0))
    c.moe.swiglu_limit = nil
    if limit > 0 then c.moe.swiglu_limit = limit end
    moe.mlp_layer_types(cfg, c, math.min(3, c.num_layers))
    -- Hyper-connections.
    c.hc_mult = math.max(1, int(cfg.hc_mult, 4))
    c.hyper = {
      kind = "mhc",
      head = "mean",
      sinkhorn_iters = int(cfg.hc_sinkhorn_iters, 20),
      eps = num(cfg.hc_eps, 1e-6),
      lowrank = 0,
    }
    -- DSA indexer over k-pools: exact as dense while every complete pool is selected.
    local kpool = math.max(1, int(cfg.index_kpool, 16))
    local topk = int(cfg.index_topk, 2048)
    if topk % kpool ~= 0 then invalid("glm5_next: index_topk must be a multiple of index_kpool") end
    if not flag(cfg.index_kpool_always_select_tail, true) then
      unsupported("glm5_next without index_kpool_always_select_tail (the incomplete tail would be dropped)")
    end
    c.index_bound = { block = kpool, max_blocks = topk // kpool }
    -- Glm5NextTextAttention builds its latent norms with `rms_norm_eps`.
    if c.mla ~= nil then c.mla.latent_norm_eps = c.rms_norm_eps end
    warn(string.format("glm5_next: the DSA indexer runs as dense attention, exact for prompts up to %d tokens (index_topk); longer prompts are refused", topk))
  end,
}
