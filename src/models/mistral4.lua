return {
  model_type = "mistral4",
  aliases = { "mistral4_text" },
  llama_cpp = "mistral4",
  chat = "mistral",
  verified = true,
  notes = "fixture (text): MLA with interleaved rotary, yarn (mscale_all_dim) from rope_parameters, llama_4_scaling_beta query scaling on every layer, softmax group-limited top-k (two best experts per group) with renormalisation, fused [E][2I][H] experts, shared experts, first_k_dense_replace. Mistral Small 4 text config.",
  names = {
    q_a = "self_attn.q_a_proj.weight",
    q_a_norm = "self_attn.q_a_layernorm.weight",
    q_b = "self_attn.q_b_proj.weight",
    kv_a = "self_attn.kv_a_proj_with_mqa.weight",
    kv_a_norm = "self_attn.kv_a_layernorm.weight",
    kv_b = "self_attn.kv_b_proj.weight",
    shared_expert = "mlp.shared_experts.",
  },
  config = function(cfg, c)
    c.rope_style = flag(cfg.rope_interleave, true) and "gptj" or "neox"
    c.moe.scoring = "softmax"
    c.moe.topk_method = "group_limited"
    c.moe.group_score_top2 = true
    c.moe.n_group = math.max(1, int(cfg.n_group, 1))
    c.moe.topk_group = math.max(1, int(cfg.topk_group, 1))
    c.moe.routed_scaling_factor = num(cfg.routed_scaling_factor, 1.0)
    c.norm_topk_prob = flag(cfg.norm_topk_prob, true)
    c.num_experts_per_tok = int(cfg.num_experts_per_tok, 4)
    if c.num_experts % c.moe.n_group ~= 0 then
      invalid(string.format("mistral4: %d experts do not split into n_group = %d groups", c.num_experts, c.moe.n_group))
    end
    -- Queries are scaled by `1 + beta * log(1 + floor(pos / original_max_position_embeddings))`.
    local r = obj(cfg.rope_parameters) or obj(cfg.rope_scaling)
    if r then
      local beta = num(r.llama_4_scaling_beta, nil)
      if beta then
        c.attn_temperature = {
          floor_scale = num(r.original_max_position_embeddings, 8192),
          attn_scale = beta,
          offset = 0,
          all_layers = true,
        }
      end
    end
  end,
}
