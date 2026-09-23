return {
  model_type = "minimax_m2",
  llama_cpp = "minimax-m2",
  chat = "minimax_m2",
  verified = true,
  notes = "fixture: q/k RMSNorm over the whole projection, partial rotary (rotary_dim), sigmoid routing with e_score_correction_bias and renormalised top-k, Mixtral-style expert tensors.",
  qk_norm = "full",
  names = {
    q_norm = "self_attn.q_norm.weight",
    k_norm = "self_attn.k_norm.weight",
    router = "block_sparse_moe.gate.weight",
    router_correction_bias = "block_sparse_moe.e_score_correction_bias",
    expert = "block_sparse_moe.experts.{e}.",
    expert_gate = "w1.weight",
    expert_up = "w3.weight",
    expert_down = "w2.weight",
    fused_gate_up = {},
    fused_down = {},
  },
  config = function(cfg, c)
    -- Sigmoid routing, top-k on the biased scores, renormalised routing weights.
    c.moe.scoring = "sigmoid"
    c.norm_topk_prob = true
    c.num_experts_per_tok = int(cfg.num_experts_per_tok, 8)
    if type(cfg.rope_theta) ~= "number" then c.rope_theta = 5000000.0 end
    if c.num_experts > 0 then each_layer(c.moe_layers, function() return true end) end
  end,
}
