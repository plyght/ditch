-- Readings of config.json shared by the mixture-of-experts families
-- (DeepSeek-style routers, GLM schedules, latent attention without RoPE).
local M = {}

-- The router scoring function named by `scoring_func` (softmax otherwise).
function M.scoring(name)
  if name == "sigmoid" then return "sigmoid" end
  if name == "sqrtsoftplus" then return "sqrtsoftplus" end
  return "softmax"
end

-- DeepSeek V2 / V3 (and Kimi K2.5): interleaved rotary, the router named by
-- `scoring_func` and `topk_method`, group-limited top-k.
function M.deepseek(cfg, c)
  -- The original checkpoints pair rotary coordinates as (2i, 2i+1) (`rope_interleave`).
  c.rope_style = flag(cfg.rope_interleave, true) and "gptj" or "neox"
  c.moe.scoring = M.scoring(str(cfg.scoring_func) or "softmax")
  local method = str(cfg.topk_method) or "greedy"
  if method == "group_limited_greedy" or method == "noaux_tc" then
    c.moe.topk_method = "group_limited"
  else
    c.moe.topk_method = "greedy"
  end
  c.moe.n_group = math.max(1, int(cfg.n_group, 1))
  c.moe.topk_group = math.max(1, int(cfg.topk_group, 1))
  c.moe.routed_scaling_factor = num(cfg.routed_scaling_factor, 1.0)
  if c.moe.topk_method == "group_limited" and c.num_experts % c.moe.n_group ~= 0 then
    invalid("the routed experts do not split into n_group groups")
  end
end

-- The DeepSeek-V3 router shared by dots.llm1, EXAONE-MoE, Solar Open and
-- A.X-K1: sigmoid scores, a correction bias that only steers the choice,
-- group-limited top-k on the two best scores per group, renormalisation and
-- a routed scaling factor, plus always-on shared experts.
function M.ds_router(cfg, c)
  c.moe.scoring = "sigmoid"
  c.moe.topk_method = "group_limited"
  c.moe.n_group = math.max(1, int(cfg.n_group, 1))
  c.moe.topk_group = math.max(1, int(cfg.topk_group, 1))
  c.moe.routed_scaling_factor = num(cfg.routed_scaling_factor, 1.0)
  c.moe.norm_eps_floor = true
  c.norm_topk_prob = flag(cfg.norm_topk_prob, true)
  if c.num_experts > 0 and c.num_experts % c.moe.n_group ~= 0 then
    invalid("the routed experts do not split into n_group groups")
  end
end

-- Reads a `mlp_layer_types` list (`"dense"` / `"sparse"`) into `moe_layers`;
-- without the key, layers from `first_dense` on are sparse.
function M.mlp_layer_types(cfg, c, first_dense)
  if c.num_experts == 0 then return end
  local ml = cfg.mlp_layer_types
  if ml ~= nil then
    if not is_array(ml) then invalid("mlp_layer_types must be a list") end
    each_layer(c.moe_layers, function() return false end)
    for i, v in ipairs(ml) do
      if i > c.num_layers then break end
      if type(v) ~= "string" then invalid("mlp_layer_types entries must be strings") end
      if v == "sparse" then
        c.moe_layers[i] = true
      elseif v ~= "dense" then
        unsupported("unsupported mlp_layer_types entry '" .. v .. "' (dense or sparse)")
      end
    end
    return
  end
  each_layer(c.moe_layers, function(i) return i >= first_dense end)
end

-- Multi-head latent attention without rotary channels (`qk_rope_head_dim`
-- 0, NoPE): builds `c.mla` when the generic parser skipped it because the
-- key was absent, and sets the head sizes and scale for it.
function M.mla_nope(cfg, c, family)
  if c.mla == nil then
    if type(cfg.kv_lora_rank) ~= "number" then
      invalid(family .. ": the MLA keys (q_lora_rank, kv_lora_rank, qk_nope_head_dim, v_head_dim) are required")
    end
    local q_lora_rank = nil
    if type(cfg.q_lora_rank) == "number" then q_lora_rank = int(cfg.q_lora_rank, 0) end
    c.mla = {
      q_lora_rank = q_lora_rank,
      kv_lora_rank = int(cfg.kv_lora_rank, 0),
      qk_nope_head_dim = int(cfg.qk_nope_head_dim, 0),
      qk_rope_head_dim = int(cfg.qk_rope_head_dim, 0),
      v_head_dim = int(cfg.v_head_dim, 0),
      latent_norm_eps = 1e-6,
    }
  end
  local m = c.mla
  if m.qk_rope_head_dim ~= 0 then
    unsupported(string.format("%s: the attention layers are NoPE (qk_rope_head_dim must be 0, config has %d)", family, m.qk_rope_head_dim))
  end
  if m.q_lora_rank == nil or m.q_lora_rank == 0 then
    invalid(family .. ": q_lora_rank is required (the sparse indexer reads the low-rank query)")
  end
  if m.kv_lora_rank == 0 or m.qk_nope_head_dim == 0 or m.v_head_dim == 0 or m.v_head_dim > m.qk_nope_head_dim then
    invalid(family .. ": inconsistent MLA head sizes")
  end
  c.head_dim = m.qk_nope_head_dim
  c.v_head_dim = m.v_head_dim
  c.num_kv_heads = c.num_heads
  c.rotary_dim = 0
  c.rope_scaling = { type = "none" }
  c.attention_scale = f32(1.0 / f32(math.sqrt(f32(m.qk_nope_head_dim))))
  each_layer(c.rope_layers, function() return false end)
  each_layer(c.layer_head_dim, function() return c.head_dim end)
  each_layer(c.layer_kv_heads, function() return c.num_kv_heads end)
  each_layer(c.layer_attn_scale, function() return c.attention_scale end)
end

-- First element of an integer array config value (or the scalar), requiring
-- every element to agree: ditch has one value per model, not per layer.
function M.uniform_int(cfg, key, default)
  local v = cfg[key]
  if v == nil then return default end
  if type(v) == "number" then return int(v, default) end
  if not is_array(v) then return default end
  if #v == 0 then return default end
  local first = nil
  for _, item in ipairs(v) do
    if math.type(item) ~= "integer" then invalid("'" .. key .. "' must hold integers") end
    if first == nil then first = item end
    if item ~= first then
      unsupported("per-layer values of '" .. key .. "' differ; only uniform values are supported")
    end
  end
  if first < 0 then invalid("'" .. key .. "' must not be negative") end
  return first
end

return M
