-- Config-key helpers shared by the MiniMax, MiMo V2, Kimi Linear and
-- Qwen4-Exp definitions (the Lua side of arch.zig's getIntAny, getF32Any,
-- intList, evenDim, parseScoring and dequant.zig's dtype tables).
local M = {}

-- Whether `v` is a JSON array. A JSON object is a table with string keys
-- only, so a non-empty one has no array part; an empty `{}` cannot be told
-- from `[]` and counts as an array here.
function M.is_array(v)
  return type(v) == "table" and v ~= null and (#v > 0 or next(v) == nil)
end

-- Whether `v` is a JSON object (an empty table counts as one: see is_array).
function M.is_object(v)
  return type(v) == "table" and v ~= null and #v == 0
end

-- The integer of the first of `keys` that holds a number, or `d`.
function M.int_any(cfg, keys, d)
  for _, k in ipairs(keys) do
    if type(cfg[k]) == "number" then return int(cfg[k], d) end
  end
  return d
end

-- The number of the first of `keys` that holds a number, or `d`.
function M.num_any(cfg, keys, d)
  for _, k in ipairs(keys) do
    if type(cfg[k]) == "number" then return num(cfg[k], d) end
  end
  return d
end

-- Integer list `v` (nil when it is not an array): negative numbers and
-- anything that is not a number read as 0, floats are truncated.
function M.int_list(v)
  if not M.is_array(v) then return nil end
  local out = {}
  for i = 1, #v do
    local x = v[i]
    if type(x) == "number" and x >= 0 then
      out[i] = math.tointeger(math.floor(x))
    else
      out[i] = 0
    end
  end
  return out
end

-- `x` truncated to an integer, rounded down to an even number.
function M.even_dim(x)
  local d = math.tointeger(math.floor(x))
  return d - d % 2
end

-- The router scoring a `scoring_func` names (softmax for anything unknown).
function M.scoring(name)
  if name == "sigmoid" then return "sigmoid" end
  if name == "sqrtsoftplus" then return "sqrtsoftplus" end
  return "softmax"
end

local function one_of(s, known)
  local l = string.lower(s)
  for _, k in ipairs(known) do
    if l == k then return true end
  end
  return false
end

-- Whether a float dtype name is one the weights can be stored in.
function M.store_float_dtype(dt)
  return one_of(dt, { "bfloat16", "bf16", "float16", "fp16", "float32", "fp32", "fp8", "float8", "fp8_e4m3", "float8_e4m3fn", "e4m3" })
end

-- Whether an `expert_dtype` config value names a storage format ditch reads.
function M.expert_dtype_supported(dt)
  return one_of(dt, { "bfloat16", "float32", "float16", "bf16", "fp32", "fp16", "fp8", "float8", "fp8_e4m3", "float8_e4m3fn", "e4m3", "fp4", "mxfp4", "int4", "pack-quantized" })
end

-- Recomputes `c.has_linear` from `c.linear_layers`.
function M.update_has_linear(c)
  c.has_linear = false
  for _, l in ipairs(c.linear_layers) do c.has_linear = c.has_linear or l end
end

return M
