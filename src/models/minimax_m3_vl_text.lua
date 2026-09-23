return {
  model_type = "minimax_m3_vl_text",
  aliases = { "minimax_m3_vl", "minimax_m3" },
  llama_cpp = "minimax-m3",
  verified = true,
  notes = "fixture: (1 + w) norms, per-head (1 + w) q/k norm, partial rotary, dense layers (mlp_layer_types) with a fused gate_up and the clamped swiglu, sigmoid MoE with correction bias, routed scaling and a fused-gate_up shared expert, minimax_m3_sparse layers run as dense attention (exact while the context fits index_topk_blocks blocks); the indexer weights pass through exports untouched. The image tower of the VL wrapper is never executed.",
  norm = "rms_gemma",
  mlp = "gated_fused",
  qk_norm = "head",
  names = {
    q_norm = "self_attn.q_norm.weight",
    k_norm = "self_attn.k_norm.weight",
    gate_up = "mlp.gate_up_proj.weight",
    router = "block_sparse_moe.gate.weight",
    router_correction_bias = "block_sparse_moe.e_score_correction_bias",
    expert = "block_sparse_moe.experts.{e}.",
    expert_gate = "w1.weight",
    expert_up = "w3.weight",
    expert_down = "w2.weight",
    fused_gate_up = {},
    fused_down = {},
    shared_expert = "block_sparse_moe.shared_experts.",
    shared_gate = "gate_proj.weight",
    shared_up = "up_proj.weight",
    shared_down = "down_proj.weight",
    shared_gate_up = "gate_up_proj.weight",
  },
  config = function(cfg, c)
    local keys = require("hybrid_keys")
    -- MiniMax M3 (transformers `minimax_m3_vl_text`): Gemma-style (1 + w)
    -- norms everywhere, per-head q/k norm before a partial rotary, sigmoid
    -- routing with a correction bias and a routed scaling factor, a shared
    -- expert and the clamped gpt-oss swiglu in every MLP. Attention on
    -- `minimax_m3_sparse` layers is MiniMax Sparse Attention: every query
    -- scores the key blocks of `index_block_size` tokens with a small indexer
    -- and attends to the best `index_topk_blocks` plus `index_local_blocks`
    -- (the top-k always includes every block once the context is shorter
    -- than `index_topk_blocks * index_block_size` tokens, 2048 with the
    -- released config), so ditch runs those layers as dense causal attention
    -- and never reads the indexer weights; longer prompts diverge from the
    -- reference and are rejected.
    c.moe.scoring = "sigmoid"
    c.norm_topk_prob = true
    c.moe.routed_scaling_factor = num(cfg.routed_scaling_factor, 2.0)
    c.moe.swiglu = { alpha = num(cfg.swiglu_alpha, 1.702), limit = num(cfg.swiglu_limit, 7.0) }
    c.num_experts_per_tok = int(cfg.num_experts_per_tok, 4)
    if type(cfg.rope_theta) ~= "number" then c.rope_theta = 5000000.0 end
    if type(cfg.rotary_dim) ~= "number" and type(cfg.partial_rotary_factor) ~= "number" then
      c.rotary_dim = math.min(64, c.head_dim)
    end
    -- Experts use `intermediate_size`; dense layers `dense_intermediate_size`.
    c.moe_intermediate_size = c.intermediate_size
    c.intermediate_size = int(cfg.dense_intermediate_size, c.intermediate_size)
    if c.num_experts > 0 then
      each_layer(c.moe_layers, function() return true end)
      local ml, mf = cfg.mlp_layer_types, cfg.moe_layer_freq
      if ml ~= nil then
        if keys.is_array(ml) then
          for i = 1, math.min(#ml, c.num_layers) do
            if type(ml[i]) == "string" then c.moe_layers[i] = ml[i] ~= "dense" end
          end
        end
      elseif mf ~= nil then
        if keys.is_array(mf) then
          for i = 1, math.min(#mf, c.num_layers) do
            c.moe_layers[i] = not (math.type(mf[i]) == "integer" and mf[i] == 0)
          end
        end
      end
    end
    local block = int(cfg.index_block_size, 128)
    local topk = int(cfg.index_topk_blocks, 16)
    local sc = nil
    if keys.is_object(cfg.sparse_attention_config) then sc = cfg.sparse_attention_config end
    if sc then
      block = int(sc.sparse_block_size, block)
      topk = int(sc.sparse_topk_blocks, topk)
    end
    local any_sparse = false
    local lt = cfg.layer_types
    if lt ~= nil then
      if keys.is_array(lt) then
        for i = 1, #lt do
          if lt[i] == "minimax_m3_sparse" then any_sparse = true end
        end
      end
    elseif sc then
      local f = sc.sparse_attention_freq
      if keys.is_array(f) then
        for i = 1, #f do
          if math.type(f[i]) == "integer" and f[i] ~= 0 then any_sparse = true end
        end
      end
    end
    if any_sparse then
      -- Dense attention is exact only while every key block is selected.
      warn("minimax_m3: sparse attention runs as dense attention, exact for prompts up to " .. (block * topk) .. " tokens (index_block_size * index_topk_blocks)")
    end
  end,
}
