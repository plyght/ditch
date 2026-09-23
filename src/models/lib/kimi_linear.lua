-- The config function of Kimi Linear and Kimi K3 (src/models/kimi_linear.lua,
-- src/models/kimi_k3.lua).
local keys = require("hybrid_keys")
local M = {}

-- Kimi Linear (`KimiLinearConfig`): the original checkpoints keep the KDA
-- settings in a `linear_attn_config` sub-dict with 1-indexed layer lists and
-- use the `attribute_map` spellings (`num_experts_per_token`,
-- `moe_renormalize`, `num_expert_group`, `model_max_length`); the Hugging
-- Face module spells them out (`linear_num_heads`, `layer_types`,
-- `mlp_layer_types`). Full-attention layers are MLA without RoPE.
--
-- Kimi K3 reuses this config with `attn_res_block_size` (Attention
-- Residual), `routed_expert_hidden_size` / `latent_moe_use_norm` (latent
-- MoE), `mla_use_output_gate`, `use_full_rank_gate` / `gate_lower_bound` in
-- `linear_attn_config` and the `situ` activation (parsed generically).
function M.config(cfg, c)
  local lac = nil
  if keys.is_object(cfg.linear_attn_config) then lac = cfg.linear_attn_config end
  local heads = int(cfg.linear_num_heads, 32)
  local head_dim = int(cfg.linear_head_dim, 128)
  local kernel = int(cfg.linear_conv_kernel_dim, 4)
  if lac then
    heads = int(lac.num_heads, heads)
    head_dim = int(lac.head_dim, head_dim)
    kernel = int(lac.short_conv_kernel_size, kernel)
    c.linear_full_rank_gate = flag(lac.use_full_rank_gate, false)
    if type(lac.gate_lower_bound) == "number" then c.linear_gate_lower_bound = lac.gate_lower_bound end
  end
  c.mla_output_gate = flag(cfg.mla_use_output_gate, false)
  c.attn_res_block = int(cfg.attn_res_block_size, 0)
  c.moe_latent = int(cfg.routed_expert_hidden_size, 0)
  c.moe_latent_norm = flag(cfg.latent_moe_use_norm, false)
  if flag(cfg.mla_use_nope, true) == false then
    unsupported("kimi_linear: full-attention layers with RoPE (mla_use_nope = false) are not implemented")
  end
  c.linear_k_heads = heads
  c.linear_v_heads = heads
  c.linear_k_dim = head_dim
  c.linear_v_dim = head_dim
  c.linear_conv_kernel = kernel
  if cfg.layer_types == nil then
    local from_lists = false
    if lac then
      if lac.full_attn_layers ~= nil and lac.kda_layers ~= nil then
        from_lists = true
        each_layer(c.linear_layers, function() return false end)
        local ka = lac.kda_layers
        if keys.is_array(ka) then
          for k = 1, #ka do
            local v = ka[k]
            if math.type(v) == "integer" and v >= 1 and v <= c.num_layers then c.linear_layers[v] = true end
          end
        end
      end
    end
    if not from_lists then
      each_layer(c.linear_layers, function(i) return not (i > 0 and i % 4 == 0) end)
    end
  end
  keys.update_has_linear(c)
  if c.mla == nil then
    invalid("kimi_linear: full-attention layers need the MLA keys (kv_lora_rank, qk_rope_head_dim, ...)")
  end
  -- MLA layers carry no positional encoding (positions come from the KDA layers).
  each_layer(c.rope_layers, function() return false end)
  if type(cfg.rms_norm_eps) ~= "number" then c.rms_norm_eps = 1e-5 end
  if type(cfg.model_max_length) == "number" then
    c.max_position_embeddings = int(cfg.model_max_length, c.max_position_embeddings)
  end
  -- Mixture of experts: sigmoid scores, correction bias, top-2 group scores.
  c.num_experts_per_tok = keys.int_any(cfg, { "num_experts_per_tok", "num_experts_per_token" }, 8)
  c.norm_topk_prob = flag(cfg.norm_topk_prob, flag(cfg.moe_renormalize, true))
  c.moe.scoring = "sigmoid"
  c.moe.topk_method = "group_limited"
  c.moe.n_group = math.max(1, keys.int_any(cfg, { "n_group", "num_expert_group" }, 1))
  c.moe.topk_group = math.max(1, int(cfg.topk_group, 1))
  c.moe.routed_scaling_factor = num(cfg.routed_scaling_factor, 2.446)
  c.moe.group_score_top2 = true
  if c.num_experts % c.moe.n_group ~= 0 then
    invalid("kimi_linear: num_experts must be a multiple of n_group")
  end
  if c.num_experts > 0 then
    local ml = cfg.mlp_layer_types
    if ml ~= nil then
      if keys.is_array(ml) then
        for i = 1, math.min(#ml, c.num_layers) do
          if type(ml[i]) == "string" then c.moe_layers[i] = ml[i] == "sparse" end
        end
      end
    else
      local first_dense = int(cfg.first_k_dense_replace, 1)
      each_layer(c.moe_layers, function(i) return i >= first_dense end)
    end
  end
end

return M
