-- Config keys shared by the Mamba families (Mamba2, Nemotron-H, Falcon-H1,
-- Jamba, Granite 4.0 H): `local mamba = require("mamba")`.
local M = {}

-- A JSON number as ditch's f32 scalars hold it, or nil.
function M.json_f32(v)
  if type(v) == "number" then return f32(v) end
  return nil
end

-- Whether `v` is a JSON array. Lua cannot tell an empty object from an
-- empty array (both are an empty table): it counts as an empty array.
function M.is_array(v)
  return type(v) == "table" and v ~= null and (#v > 0 or next(v) == nil)
end

-- Fills `out` (a list) from a JSON array of numbers (entries beyond the
-- array keep their value).
function M.float_list(v, out)
  if not M.is_array(v) then return end
  for i = 1, #v do
    if i > #out then break end
    local f = M.json_f32(v[i])
    if f then out[i] = f end
  end
end

-- `time_step_limit`: a `[min, max]` clamp of the discretised time step. A
-- missing or null entry (JSON has no infinity) leaves that bound open.
function M.dt_limit(cfg, d)
  local v = cfg.time_step_limit
  if not M.is_array(v) or #v ~= 2 then return end
  local lo = M.json_f32(v[1])
  if lo then d.dt_min = lo end
  local hi = M.json_f32(v[2])
  if hi then d.dt_max = hi end
end

-- ditch's activation for a config.json spelling (nil: unknown).
local activations = {
  silu = "silu", swish = "silu", swiglu = "silu", gelu = "gelu",
  gelu_new = "gelu_tanh", gelu_pytorch_tanh = "gelu_tanh",
  gelu_tanh = "gelu_tanh", gelu_fast = "gelu_tanh",
  relu = "relu", relu2 = "relu2", relu_squared = "relu2", quick_gelu = "quick_gelu",
  -- xIELU is parameterised per layer (Apertus); this entry only keeps the
  -- parser quiet, the learned activation comes from the layer's tensors.
  xielu = "silu",
}

function M.activation(name)
  return activations[name]
end

-- Mamba2 dimensions from the `mamba_*` keys of the Falcon-H1 / Granite configs.
function M.dims(cfg, c)
  local d = c.ssm
  d.kind = "mamba2"
  local expand = f32(num(cfg.mamba_expand, 2))
  if type(cfg.mamba_d_ssm) == "number" then
    d.inter = int(cfg.mamba_d_ssm, 0)
  else
    d.inter = math.floor(f32(expand * f32(c.hidden_size)))
  end
  d.heads = int(cfg.mamba_n_heads, 128)
  -- `mamba_d_head` may be the string "auto".
  if type(cfg.mamba_d_head) == "number" then
    d.head_dim = int(cfg.mamba_d_head, 0)
  elseif d.heads > 0 then
    d.head_dim = d.inter // d.heads
  else
    d.head_dim = 0
  end
  d.state = int(cfg.mamba_d_state, 256)
  d.groups = int(cfg.mamba_n_groups, 1)
  d.conv_kernel = int(cfg.mamba_d_conv, 4)
  d.act = c.activation
  M.dt_limit(cfg, d)
end

return M
