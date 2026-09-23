local dsv4 = require("dsv4")

return {
  model_type = "deepseek_v4",
  llama_cpp = nil,
  chat = "deepseek",
  verified = true,
  notes = "fixture: hyper-connections (hc_mult streams, Sinkhorn-mixed), low-rank q with unweighted head norm, shared-KV sliding attention with sinks and inverse-roped output, grouped output projection, CSA (overlapping pooled windows; Lightning Indexer as dense: exact while every reachable entry fits index_topk) and HCA branches with their own rope, sqrtsoftplus MoE with correction bias, hash-routed (tid2eid) layers, clamped SwiGLU, shared expert, MTP tensors passed through. The released checkpoints (DeepSeek's own tensor names, FP8 with ue8m0 block scales, FP4 e2m1 experts) are renamed and dequantised on load; DeepSeek-V4-Flash's first four layers (sliding, CSA and HCA attention, hash and learned routing) match transformers in float32 on the real weights.",
  rope_style = "gptj",
  names = {
    o = "self_attn.o_b_proj.weight",
    sinks = "self_attn.sinks",
    q_a = "self_attn.q_a_proj.weight",
    q_a_norm = "self_attn.q_a_norm.weight",
    q_b = "self_attn.q_b_proj.weight",
    kv_a = "self_attn.kv_proj.weight",
    kv_a_norm = "self_attn.kv_norm.weight",
    router_correction_bias = "mlp.gate.e_score_correction_bias",
    shared_expert = "mlp.shared_experts.",
  },
  config = function(cfg, c)
    local d = {
      v41 = false,
      fake_quant = false,
      engram = nil,
      candidate_source = nil,
      candidate_topk_blocks = 0,
      candidate_block_size = 1,
    }
    dsv4.common(cfg, c, d)
    d.index_n_heads = int(cfg.index_n_heads, 64)
    c.norm_topk_prob = true
    local n = c.num_layers
    -- Per-layer-type compression rates (`compress_rates`, or the legacy scalars).
    local rate_csa, rate_hca = 4, 128
    local cr = dsv4.object(cfg.compress_rates)
    if cr then
      rate_csa = int(cr.compressed_sparse_attention, rate_csa)
      rate_hca = int(cr.heavily_compressed_attention, rate_hca)
    end
    rate_csa = int(cfg.compress_rate_csa, rate_csa)
    rate_hca = int(cfg.compress_rate_hca, rate_hca)
    local branch, ratio, source = {}, {}, {}
    local legacy = dsv4.int_list(cfg.compress_ratios)
    if cfg.layer_types ~= nil then
      local lt = cfg.layer_types
      if not dsv4.is_array(lt) or #lt < n then invalid("deepseek_v4: layer_types needs one entry per layer") end
      for i = 1, n do
        local v = lt[i]
        if type(v) ~= "string" then invalid("deepseek_v4: layer_types entries must be strings") end
        if v == "sliding_attention" then
          branch[i] = "none"
        elseif v == "compressed_sparse_attention" then
          branch[i] = "csa"
        elseif v == "heavily_compressed_attention" then
          branch[i] = "hca"
        else
          invalid("deepseek_v4: unknown layer type '" .. v .. "'")
        end
      end
    elseif legacy then
      -- Legacy per-layer ints keyed by the default rates: 0 / 4 / 128.
      if #legacy < n then invalid("deepseek_v4: compress_ratios needs one entry per layer") end
      for i = 1, n do
        local r = legacy[i]
        if r == 0 then
          branch[i] = "none"
        elseif r == 4 then
          branch[i] = "csa"
        elseif r == 128 then
          branch[i] = "hca"
        else
          invalid(string.format("deepseek_v4: unknown compress ratio %d", r))
        end
      end
    else
      -- V4-Pro default: two HCA layers, then CSA on odd and HCA on even indices.
      for i = 0, n - 1 do
        branch[i + 1] = (i < 2 and "hca") or ((i - 2) % 2 == 1 and "csa") or "hca"
      end
    end
    for i = 1, n do
      local b = branch[i]
      ratio[i] = (b == "none" and 0) or (b == "csa" and rate_csa) or rate_hca
      source[i] = b ~= "none" and i - 1 or false
      if b ~= "none" and ratio[i] == 0 then invalid(string.format("deepseek_v4: layer %d has a compression rate of 0", i - 1)) end
    end
    d.branch = branch
    d.compress_ratio = ratio
    d.kv_source = source
    -- Hash-routed MoE layers: `mlp_layer_types`, or the first `num_hash_layers` (default 3).
    local hash = {}
    if cfg.mlp_layer_types ~= nil then
      local mt = cfg.mlp_layer_types
      if not dsv4.is_array(mt) or #mt < n then invalid("deepseek_v4: mlp_layer_types needs one entry per layer") end
      for i = 1, n do
        local v = mt[i]
        if type(v) ~= "string" then invalid("deepseek_v4: mlp_layer_types entries must be strings") end
        if v == "hash_moe" then
          hash[i] = true
        elseif v == "moe" then
          hash[i] = false
        else
          invalid("deepseek_v4: unknown mlp layer type '" .. v .. "'")
        end
      end
    else
      local n_hash = int(cfg.num_hash_layers, 3)
      for i = 0, n - 1 do hash[i + 1] = i < n_hash end
    end
    d.hash_moe_layers = hash
    c.dsv4 = d
  end,
}
