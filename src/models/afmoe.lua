return {
  model_type = "afmoe",
  llama_cpp = nil,
  chat = "chatml",
  verified = true,
  notes = "fixture: norms on both sublayer inputs and outputs, a sigmoid gate on the attention output, sliding layers every n, sigmoid routing with a selection bias, renormalisation and route_scale, shared experts and dense first layers. AFM (Arcee) MoE.",
  default_norm_eps = 1e-5,
  names = {
    post_attn_norm = "post_attention_layernorm.weight",
    pre_ff_norm = "pre_mlp_layernorm.weight",
    post_ff_norm = "post_mlp_layernorm.weight",
    q_norm = "self_attn.q_norm.weight",
    k_norm = "self_attn.k_norm.weight",
    attn_gate = "self_attn.gate_proj.weight",
    router = "mlp.router.gate.weight",
    router_correction_bias = "mlp.expert_bias",
    shared_expert = "mlp.shared_experts.",
  },
  -- AFMoE: a sigmoid router whose selection adds a per-expert bias, weights
  -- renormalised and scaled by `route_scale`, always-on shared experts, a
  -- sigmoid gate on the attention output and 1-in-`global_attn_every_n_layers`
  -- global attention.
  config = function(cfg, c)
    c.qk_norm = "head"
    c.attn_gate = "sigmoid"
    c.moe.scoring = "sigmoid"
    c.moe.routed_scaling_factor = num(cfg.route_scale, 1.0)
    c.moe.norm_eps_floor = true
    c.norm_topk_prob = true
    if cfg.layer_types == nil then
      local every = math.max(1, int(cfg.global_attn_every_n_layers, 4))
      each_layer(c.sliding_layers, function(i) return (i + 1) % every ~= 0 end)
    end
    -- Only the local layers are roped; the full-attention ones are NoPE
    -- (`AfmoeAttention` applies the rotary under `if self.is_local_attention`).
    each_layer(c.rope_layers, function(i) return c.sliding_layers[i + 1] end)
    -- muP: the embeddings are scaled by sqrt(hidden_size) (arcee-ai/Trinity-Nano-Preview sets it).
    if flag(cfg.mup_enabled, false) then c.embed_scale = f32(math.sqrt(f32(c.hidden_size))) end
  end,
}
