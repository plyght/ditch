-- Config keys shared by DeepSeek V4 and V4.1: `local dsv4 = require("dsv4")`.
local M = {}

-- Whether `v` is a JSON array. Lua cannot tell an empty object from an
-- empty array (both are an empty table): it counts as an empty array.
function M.is_array(v)
  return type(v) == "table" and v ~= null and (#v > 0 or next(v) == nil)
end

-- A JSON object (not an array), or nil.
function M.object(v)
  if obj(v) and #v == 0 then return v end
  return nil
end

function M.scoring(name)
  if name == "sigmoid" then return "sigmoid" end
  if name == "sqrtsoftplus" then return "sqrtsoftplus" end
  return "softmax"
end

-- Integer list `v` (nil when not an array): negative and non-numeric
-- entries read as 0, floats are truncated.
function M.int_list(v)
  if not M.is_array(v) then return nil end
  local out = {}
  for i, item in ipairs(v) do
    if math.type(item) == "integer" then
      out[i] = item >= 0 and item or 0
    elseif math.type(item) == "float" then
      out[i] = item >= 0 and math.floor(item) or 0
    else
      out[i] = 0
    end
  end
  return out
end

-- The parts of the DeepSeek V4 / V4.1 configs both families share: hyper-
-- connections, the low-rank query and grouped output projections, the
-- compressed-branch RoPE (`compress_rope_theta`, yarn only there, with the
-- reference's `attention_factor = 1`), sinks, the clamped expert SwiGLU and
-- the MoE routing. `d` is the new `c.dsv4` table.
function M.common(cfg, c, d)
  c.rope_style = "gptj"
  c.sinks = true
  c.norm = "rms"
  c.num_kv_heads = int(cfg.num_key_value_heads, 1)
  if c.num_kv_heads ~= 1 then
    unsupported(string.format("deepseek_v4: shared-KV attention needs num_key_value_heads = 1 (config has %d)", c.num_kv_heads))
  end
  -- Every layer carries the sliding window; the compressed branch is extra.
  each_layer(c.sliding_layers, function() return true end)
  if c.sliding_window == nil then c.sliding_window = 128 end
  if type(cfg.qk_rope_head_dim) == "number" then c.rotary_dim = int(cfg.qk_rope_head_dim, c.rotary_dim) end
  c.rotary_dim = c.rotary_dim - c.rotary_dim % 2
  if c.rotary_dim == 0 or c.rotary_dim > c.head_dim then
    invalid(string.format("deepseek_v4: qk_rope_head_dim (%d) must be between 2 and head_dim (%d)", c.rotary_dim, c.head_dim))
  end
  c.v_head_dim = c.head_dim
  c.attention_scale = f32(1.0 / f32(math.sqrt(f32(c.head_dim))))
  c.hc_mult = math.max(1, int(cfg.hc_mult, 4))
  d.hc_mult = c.hc_mult
  d.hc_sinkhorn_iters = int(cfg.hc_sinkhorn_iters, 20)
  d.hc_eps = num(cfg.hc_eps, 1e-6)
  c.hyper = {
    kind = d.v41 and "mhc_single_pass" or "mhc",
    head = d.v41 and "previous_pre" or "weighted",
    sinkhorn_iters = d.hc_sinkhorn_iters,
    eps = d.hc_eps,
    lowrank = 0,
  }
  d.q_lora_rank = int(cfg.q_lora_rank, 0)
  d.o_groups = math.max(1, int(cfg.o_groups, 8))
  d.o_lora_rank = int(cfg.o_lora_rank, 1024)
  if d.q_lora_rank == 0 or d.o_lora_rank == 0 or (c.num_heads * c.head_dim) % d.o_groups ~= 0 then
    invalid("deepseek_v4: q_lora_rank and o_lora_rank must be set, and o_groups must divide the attention width")
  end
  d.index_topk = int(cfg.index_topk, 512)
  d.index_head_dim = int(cfg.index_head_dim, 128)
  -- The compressed branches rotate with their own base and (yarn) scaling;
  -- the sliding-window rope is plain. The reference never multiplies the
  -- compress cos/sin by yarn's mscale unless the config says so.
  d.compress_rope_theta = f32(num(cfg.compress_rope_theta, 160000.0))
  local compress_dict = nil
  local rp = M.object(cfg.rope_parameters)
  if rp then
    local m = M.object(rp.main)
    if m then c.rope_theta = num(m.rope_theta, c.rope_theta) end
    local cp = M.object(rp.compress)
    if cp then
      compress_dict = cp
      d.compress_rope_theta = num(cp.rope_theta, d.compress_rope_theta)
    end
  end
  if compress_dict == nil then compress_dict = M.object(cfg.rope_scaling) end
  d.compress_rope_scaling = { type = "none" }
  if compress_dict then
    d.compress_rope_scaling = rope_scaling(compress_dict, cfg, c.rotary_dim, c.max_position_embeddings)
    if d.compress_rope_scaling.type == "yarn" and type(compress_dict.attention_factor) ~= "number" then
      d.compress_rope_scaling.attention_factor = 1.0
    end
  end
  c.rope_scaling = { type = "none" }
  -- MoE: sqrtsoftplus scores, plain top-k on the corrected scores, renormalised.
  c.moe.scoring = M.scoring(str(cfg.scoring_func) or "sqrtsoftplus")
  c.moe.topk_method = "greedy"
  c.moe.routed_scaling_factor = num(cfg.routed_scaling_factor, 1.5)
  c.moe.norm_eps_floor = true
  local limit = f32(num(cfg.swiglu_limit, 10.0))
  c.moe.swiglu_limit = limit > 0 and limit or nil
  c.num_experts_per_tok = int(cfg.num_experts_per_tok, 6)
  if type(cfg.intermediate_size) ~= "number" then c.intermediate_size = c.moe_intermediate_size end
  if c.num_experts > 0 then each_layer(c.moe_layers, function() return true end) end
end

return M
