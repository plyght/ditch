return {
  model_type = "lfm2_moe",
  base = "lfm2",
  llama_cpp = "lfm2moe",
  verified = true,
  notes = "fixture: the LFM2 short-convolution / attention hybrid with num_dense_layers dense feed-forward layers of the plain intermediate_size, then sigmoid-routed experts (feed_forward.experts.{e}.w1 / w3 / w2) chosen with an expert bias, renormalised with a 1e-6 epsilon and scaled by routed_scaling_factor. LFM2 / LFM2.5 MoE (LFM2-8B-A1B, LFM2.5-24B-A2B).",
  names = {
    router = "feed_forward.gate.weight",
    router_correction_bias = "feed_forward.expert_bias",
    expert = "feed_forward.experts.{e}.",
    expert_gate = "w1.weight",
    expert_up = "w3.weight",
    expert_down = "w2.weight",
    fused_gate_up = {},
    fused_down = {},
  },
  -- The LFM2 settings, except that the dense width is intermediate_size as
  -- given (Lfm2MoeMLP has no block_ff_dim sizing), and the router.
  config = function(cfg, c)
    c.conv_kernel = int(cfg.conv_L_cache, 3)
    if type(cfg.rope_theta) ~= "number" and obj(cfg.rope_parameters) == nil then c.rope_theta = 1000000.0 end
    c.tie_word_embeddings = flag(cfg.tie_word_embeddings, flag(cfg.tie_embedding, true))
    c.moe.scoring = "sigmoid"
    c.moe.routed_scaling_factor = f32(num(cfg.routed_scaling_factor, 1.0))
    c.moe.norm_eps = 1e-6
    c.norm_topk_prob = flag(cfg.norm_topk_prob, true)
  end,
}
