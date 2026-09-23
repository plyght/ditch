-- The Gemma family's config keys: the rotary tables of Gemma 3 and later,
-- KV-shared layers and per-layer input embeddings (Gemma 3n / 4), and the
-- config functions of Gemma 2 / 3, Gemma 3n and Gemma 4.
local M = {}

-- `x` truncated to an integer and rounded down to an even number.
function M.even_dim(x)
  local d = math.tointeger(math.floor(x))
  return d - d % 2
end

-- Rotary tables of the Gemma 3 family: the global layers' base and scaling
-- (`rope_theta` / `rope_scaling`, or `rope_parameters.full_attention`) and the
-- sliding layers' own base (`rope_local_base_freq` or
-- `rope_parameters.sliding_attention`). `hd_local` / `hd_global` are the head
-- sizes of the two layer kinds.
function M.rope(cfg, c, hd_local, hd_global)
  local local_theta = num(cfg.rope_local_base_freq, 10000.0)
  local local_dim = hd_local
  local global_dim = hd_global
  local freq_dim = hd_global
  local f = num(cfg.partial_rotary_factor, nil)
  if f then
    local_dim = M.even_dim(hd_local * f)
    global_dim = M.even_dim(hd_global * f)
    freq_dim = global_dim
  end
  local rp = obj(cfg.rope_parameters)
  if rp then
    local full = obj(rp.full_attention)
    if full then
      c.rope_theta = num(full.rope_theta, c.rope_theta)
      c.rope_scaling = rope_scaling(full, cfg, global_dim, c.max_position_embeddings)
      local t = str(full.rope_type) or "default"
      local ff = num(full.partial_rotary_factor, nil)
      if ff then
        if t == "proportional" then
          -- `int(f * head_dim // 2)` turning angles with frequencies
          -- over the full head, the rest of its pairs still: the
          -- table covers the whole head (see `Config.rope_angles`).
          c.rope_angles = math.max(1, math.tointeger(math.floor(hd_global * ff / 2.0)))
          global_dim = hd_global
          freq_dim = hd_global
        else
          global_dim = M.even_dim(hd_global * ff)
          freq_dim = global_dim
        end
      elseif t == "proportional" then
        freq_dim = hd_global
      end
    end
    local sl = obj(rp.sliding_attention)
    if sl then
      local_theta = num(sl.rope_theta, local_theta)
      local sf = num(sl.partial_rotary_factor, nil)
      if sf then local_dim = M.even_dim(hd_local * sf) end
    end
  end
  c.rotary_dim = global_dim
  c.rope_freq_dim = freq_dim
  c.rope_local = { theta = local_theta, rotary_dim = local_dim, freq_dim = local_dim }
end

-- The last `num_kv_shared_layers` layers read the keys and values of the last
-- non-shared layer of their kind (sliding or global) instead of computing their own.
function M.kv_sharing(cfg, c)
  local n = int(cfg.num_kv_shared_layers, 0)
  if n == 0 or n >= c.num_layers then return end
  local first = c.num_layers - n
  for i = first, c.num_layers - 1 do
    local found = false
    for j = first - 1, 0, -1 do
      if c.sliding_layers[j + 1] == c.sliding_layers[i + 1] then
        c.kv_source[i + 1] = j
        found = true
        break
      end
    end
    if not found then
      invalid(string.format("layer %d shares keys/values but no earlier layer of its kind exists", i))
    end
  end
end

-- Per-layer input embeddings: their width (`default_dim` unless given) and
-- vocabulary (the model's unless given and non-zero).
function M.per_layer_embeddings(cfg, c, default_dim)
  c.ple_dim = int(cfg.hidden_size_per_layer_input, default_dim)
  c.ple_vocab = int(cfg.vocab_size_per_layer_input, c.vocab_size)
  if c.ple_vocab == 0 then c.ple_vocab = c.vocab_size end
end

-- Gemma's GeLU unless the config names an activation.
local function default_activation(cfg, c)
  if str(cfg.hidden_activation) == nil and str(cfg.hidden_act) == nil then c.activation = "gelu_tanh" end
end

local function is_gemma3(c)
  return c.model_type:sub(1, #"gemma3") == "gemma3"
end

-- Gemma 2 and Gemma 3.
function M.config_gemma(cfg, c)
  default_activation(cfg, c)
  if type(cfg.sliding_window_pattern) ~= "number" and cfg.layer_types == nil and c.sliding_window then
    -- Gemma 3 defaults to a pattern of 6 (5 local, 1 global); Gemma 2
    -- alternates, its even layers local.
    if is_gemma3(c) then
      each_layer(c.sliding_layers, function(i) return (i + 1) % 6 ~= 0 end)
    else
      each_layer(c.sliding_layers, function(i) return i % 2 == 0 end)
    end
  end
  if is_gemma3(c) then M.rope(cfg, c, c.head_dim, c.head_dim) end
end

-- Gemma 3n.
function M.config_gemma3n(cfg, c)
  default_activation(cfg, c)
  if cfg.layer_types == nil then
    -- Every fifth layer is global.
    each_layer(c.sliding_layers, function(i) return (i + 1) % 5 ~= 0 end)
  end
  if c.sliding_window == nil then c.sliding_window = 512 end
  c.attention_scale = 1.0
  c.v_norm = true
  M.rope(cfg, c, c.head_dim, c.head_dim)
  M.kv_sharing(cfg, c)
  M.per_layer_embeddings(cfg, c, 256)
  c.altup_inputs = int(cfg.altup_num_inputs, 4)
  c.altup_active = int(cfg.altup_active_idx, 0)
  c.altup_correct_scale = flag(cfg.altup_correct_scale, true)
  c.laurel_rank = int(cfg.laurel_rank, 64)
  if c.altup_inputs < 1 or c.altup_active >= c.altup_inputs or c.ple_dim == 0 or c.laurel_rank == 0 then
    invalid("gemma3n: altup_num_inputs must be at least 1 and above altup_active_idx, and hidden_size_per_layer_input and laurel_rank non-zero")
  end
  -- Activation sparsity: one value per layer, one for all, or the default
  -- (the first 10 layers at 0.95 on models deeper than 10 layers).
  local asp = cfg.activation_sparsity_pattern
  if asp ~= nil then
    if type(asp) == "number" then
      each_layer(c.activation_sparsity, function() return f32(asp) end)
    else
      for i = 1, math.min(len(asp), c.num_layers) do
        local v = asp[i]
        c.activation_sparsity[i] = type(v) == "number" and f32(v) or 0
      end
    end
  elseif c.num_layers > 10 then
    for i = 1, 10 do c.activation_sparsity[i] = f32(0.95) end
  end
  for _, s in ipairs(c.activation_sparsity) do
    if s < 0 or s >= 1 then invalid("gemma3n: activation_sparsity_pattern values must lie in [0, 1)") end
  end
end

-- A key of `per_layer_config` read as a layer index the way Zig's
-- `std.fmt.parseInt(usize, key, 10)` reads it: an optional sign, decimal
-- digits with underscores between them, and "-" only before zero.
-- nil when it is not one; math.huge when it exceeds every layer index.
local function layer_index(key)
  local sign, body = key:match("^([+-]?)(.*)$")
  if body == "" or body:sub(1, 1) == "_" or body:sub(-1) == "_" or body:find("[^0-9_]") then return nil end
  local digits = body:gsub("_", ""):gsub("^0+", "")
  if digits == "" then return 0 end
  if sign == "-" then return nil end
  if #digits > 15 then return math.huge end
  return math.tointeger(tonumber(digits))
end

-- Gemma 4.
function M.config_gemma4(cfg, c)
  default_activation(cfg, c)
  if str(cfg.use_bidirectional_attention) == "all" then
    unsupported("Gemma 4 with bidirectional attention on every token")
  end
  if flag(cfg.enable_moe_block, false) then
    unsupported("Gemma 4 MoE block (a routed expert block in parallel with the dense MLP, as in gemma-4-26B-A4B) is not implemented")
  end
  if cfg.layer_types == nil then
    -- 5 local, 1 global; the last layer is always global.
    each_layer(c.sliding_layers, function(i) return (i + 1) % 6 ~= 0 end)
  end
  if c.num_layers > 0 then c.sliding_layers[c.num_layers] = false end
  if c.sliding_window == nil then c.sliding_window = 512 end
  c.attention_scale = 1.0
  c.v_norm = true
  c.k_eq_v = flag(cfg.attention_k_eq_v, false)
  -- Global layers use their own head size and KV heads: `global_head_dim`
  -- (512 unless given) and `num_global_key_value_heads`, or an explicit
  -- `per_layer_config` table indexed by layer.
  local global_hd = int(cfg.global_head_dim, 512)
  local global_kv = c.num_kv_heads
  if type(cfg.num_global_key_value_heads) == "number" and c.k_eq_v then
    global_kv = int(cfg.num_global_key_value_heads, global_kv)
  end
  local plc = obj(cfg.per_layer_config)
  if plc then
    -- The first entry, in config.json's order, of a global layer.
    for _, k in ipairs(keys(plc)) do
      local idx = layer_index(k)
      local lc = obj(plc[k])
      if idx and idx < c.num_layers and not c.sliding_layers[idx + 1] and lc then
        global_hd = int(lc.head_dim, global_hd)
        global_kv = int(lc.num_key_value_heads, global_kv)
        break
      end
    end
  end
  for i = 1, c.num_layers do
    if not c.sliding_layers[i] then
      c.layer_head_dim[i] = global_hd
      c.layer_kv_heads[i] = global_kv
    end
  end
  M.rope(cfg, c, c.head_dim, global_hd)
  M.kv_sharing(cfg, c)
  M.per_layer_embeddings(cfg, c, 256)
  -- KV-shared layers may carry a double-wide MLP; the weights say which.
  if flag(cfg.use_double_wide_mlp, false) and int(cfg.num_kv_shared_layers, 0) > 0 then c.intermediate_varies = true end
end

return M
