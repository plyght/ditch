return {
  model_type = "mimo_v2_flash",
  aliases = { "mimo_v2" },
  llama_cpp = "mimo2",
  chat = "mimo",
  verified = true,
  notes = "fixtures: hybrid full / sliding-window attention (window 128 in the released configs) with attention sinks and doubled kv heads on the sliding layers, v_head_dim < head_dim with attention_value_scale, partial rotary with one base per layer type (rope_parameters, or rope_theta / swa_rope_theta), a dense first layer (mlp_layer_types / moe_layer_freq) then sigmoid MoE with correction bias and group-limited top-k, no shared experts; both the transformers spelling (layer_types, stacked experts, sinks) and the hub checkpoint spelling of MiMo-V2-Flash / V2.5 / V2.6 (model_type mimo_v2: hybrid_layer_pattern, swa_*, attention_sink_bias, per-expert tensors, the Pro layout's fused qkv_proj chunked per kv head). MTP (model.mtp.*), vision and audio encoder tensors of the V2.5 / V2.6 omni checkpoints pass through exports untouched. The V2.6 checkpoints' MXFP4 experts (quant_method fp8 with store_dtype mxfp4: U8 weight/weight_scale next to the fp8 dense weights) and their bf16 MoE router (moe_router_dtype) are read as they are.",
  default_norm_eps = 1e-5,
  names = {
    qkv = "self_attn.qkv_proj.weight",
    sinks = "self_attn.sinks",
    sinks_alt = "self_attn.attention_sink_bias",
    router_correction_bias = "mlp.gate.e_score_correction_bias",
  },
  -- Xiaomi MiMo V2 (`MiMoV2FlashConfig` in transformers; the hub checkpoints of
  -- MiMo-V2-Flash, V2.5 and V2.6 carry the remote-code spellings of the same
  -- settings: `hybrid_layer_pattern`, `swa_*`, `moe_layer_freq`, ...). Hybrid
  -- attention: full layers with `num_key_value_heads` KV heads, sliding layers
  -- with `swa_num_key_value_heads` (twice as many) plus attention sinks; values
  -- are `v_head_dim` wide and scaled by `attention_value_scale`; partial rotary
  -- with one base per layer type; the first layer is dense, the rest
  -- DeepSeek-V3-style sigmoid MoE (correction bias, group-limited top-k) without
  -- shared experts. MTP (`model.mtp.*`), vision and audio tensors are never read.
  config = function(cfg, c)
    local keys = require("hybrid_keys")
    local n = c.num_layers
    local hf_spelling = c.model_type == "mimo_v2_flash" or cfg.layer_types ~= nil or cfg.rope_parameters ~= nil or cfg.mlp_layer_types ~= nil
    if type(cfg.rms_norm_eps) ~= "number" then c.rms_norm_eps = num(cfg.layernorm_epsilon, 1e-5) end
    -- `store_dtype` names the expert storage format. MiMo V2.6 carries it in
    -- `quantization_config` (parsed by dequant.zig, which decodes the MXFP4
    -- experts on read); a copy at the config level says the same thing.
    local sd = str(cfg.store_dtype)
    if sd and not keys.expert_dtype_supported(sd) then
      unsupported("MiMo experts stored as '" .. sd .. "' (store_dtype) cannot be dequantised; convert the experts to bf16 first")
    end
    -- The router runs in f32 on the bf16 gate weights whatever
    -- `moe_router_dtype` says (MiMo V2.6: bfloat16, the earlier ones float32);
    -- a quantised router would need a reader of its own.
    local rd = str(cfg.moe_router_dtype)
    if rd and not keys.store_float_dtype(rd) then
      unsupported("MiMo MoE router in '" .. rd .. "' (moe_router_dtype)")
    end
    -- Attention layout per layer: `layer_types` (parsed generically), the
    -- remote-code `hybrid_layer_pattern` (1 = sliding) or the default (first
    -- and every sixth layer full, the rest sliding).
    if cfg.layer_types == nil then
      local hp = cfg.hybrid_layer_pattern
      if hp ~= nil then
        if not keys.is_array(hp) or #hp < n then
          invalid("MiMo hybrid_layer_pattern needs one entry per layer")
        end
        each_layer(c.sliding_layers, function(i) return math.type(hp[i + 1]) == "integer" and hp[i + 1] == 1 end)
      else
        each_layer(c.sliding_layers, function(i) return not (i == 0 or (i + 1) % 6 == 0) end)
      end
    end
    if c.sliding_window == nil then c.sliding_window = int(cfg.sliding_window_size, 128) end
    local any_sliding = false
    for _, s in ipairs(c.sliding_layers) do any_sliding = any_sliding or s end
    c.sinks = any_sliding and flag(cfg.add_swa_attention_sink_bias, true)
    c.sinks_sliding_only = true
    -- Heads: the sliding layers double the kv heads; every other dimension is shared.
    local kv_full = int(cfg.num_key_value_heads, c.num_heads)
    local kv_swa = int(cfg.swa_num_key_value_heads, 2 * kv_full)
    c.v_head_dim = int(cfg.v_head_dim, c.head_dim)
    c.narrow_values = true
    if int(cfg.swa_num_attention_heads, c.num_heads) ~= c.num_heads or int(cfg.swa_head_dim, c.head_dim) ~= c.head_dim or int(cfg.swa_v_head_dim, c.v_head_dim) ~= c.v_head_dim then
      unsupported("MiMo sliding layers with their own head count or head size (swa_num_attention_heads / swa_head_dim / swa_v_head_dim)")
    end
    if c.v_head_dim > c.head_dim then
      -- Values share the keys' cache stride, so they may be narrower (every
      -- release: 192 / 128) but not wider.
      unsupported("MiMo v_head_dim " .. c.v_head_dim .. " is wider than head_dim " .. c.head_dim)
    end
    if c.v_head_dim == 0 or kv_full == 0 or kv_swa == 0 then
      invalid("MiMo v_head_dim, num_key_value_heads and swa_num_key_value_heads must be positive")
    end
    if c.num_heads % kv_full ~= 0 or c.num_heads % kv_swa ~= 0 then
      invalid("MiMo num_attention_heads must be a multiple of num_key_value_heads and swa_num_key_value_heads")
    end
    each_layer(c.layer_kv_heads, function(i)
      if c.sliding_layers[i + 1] then return kv_swa end
      return kv_full
    end)
    c.num_kv_heads = kv_full
    -- Values are scaled before attention. transformers defaults the scale to
    -- 0.707 and reads an explicit null as 1; the remote-code modules default to 1.
    if cfg.attention_value_scale ~= nil then
      if cfg.attention_value_scale == null then
        c.mult.value = 1.0
      else
        c.mult.value = num(cfg.attention_value_scale, 1.0)
      end
    elseif hf_spelling then
      c.mult.value = 0.707
    else
      c.mult.value = 1.0
    end
    -- Rotary: one base per layer type, partial rotary factor 0.334 (int(head_dim * f) dims).
    local factor
    if type(cfg.partial_rotary_factor) == "number" then
      factor = cfg.partial_rotary_factor
    elseif hf_spelling then
      factor = 0.334
    else
      factor = 1.0
    end
    local rp = nil
    if keys.is_object(cfg.rope_parameters) then rp = cfg.rope_parameters end
    if hf_spelling and type(cfg.rope_theta) ~= "number" and rp == nil then c.rope_theta = 5000000.0 end
    local default_local = c.rope_theta
    if hf_spelling then default_local = 10000.0 end
    local local_theta = f32(num(cfg.swa_rope_theta, default_local))
    if rp then
      local full, swa = nil, nil
      if keys.is_object(rp.full_attention) then
        full = rp.full_attention
      elseif type(rp.rope_theta) == "number" then
        full = rp
      end
      if keys.is_object(rp.sliding_attention) then
        swa = rp.sliding_attention
      elseif type(rp.rope_theta) == "number" then
        swa = rp
      end
      if full then
        c.rope_theta = num(full.rope_theta, 5000000.0)
        factor = num(full.partial_rotary_factor, 0.334)
        c.rope_scaling = rope_scaling(full, cfg, c.rotary_dim, c.max_position_embeddings)
      end
      if swa then
        -- A flat `rope_parameters` (MiMo V2.5 / V2.6: `{rope_theta: 1e7, ...}`)
        -- is the full layers' base; the sliding layers keep `swa_rope_theta`,
        -- which the remote code writes over it for them.
        if not keys.is_object(rp.sliding_attention) and type(cfg.swa_rope_theta) == "number" then
          local_theta = num(cfg.swa_rope_theta, 10000.0)
        else
          local_theta = num(swa.rope_theta, 10000.0)
        end
        local sf = num(swa.partial_rotary_factor, 0.334)
        local t = str(swa.rope_type) or str(swa.type) or "default"
        if sf ~= factor or not (t == "default" or t == "mrope") then
          unsupported("MiMo sliding layers with their own rotary factor or scaling (" .. t .. ")")
        end
      end
    end
    c.rotary_dim = keys.even_dim(c.head_dim * factor)
    if c.rotary_dim == 0 or c.rotary_dim > c.head_dim then
      invalid("MiMo rotary dimension " .. c.rotary_dim .. " is outside 1.." .. c.head_dim)
    end
    c.rope_freq_dim = c.rotary_dim
    -- The sliding layers rotate the same coordinates with their own base.
    c.rope_local = { theta = local_theta, rotary_dim = c.rotary_dim, freq_dim = c.rotary_dim }
    -- Mixture of experts: sigmoid scores plus a correction bias choose the
    -- experts (group-limited top-k), the raw scores weight them.
    c.moe.scoring = keys.scoring(str(cfg.scoring_func) or "sigmoid")
    local method = str(cfg.topk_method) or "noaux_tc"
    if method == "greedy" then c.moe.topk_method = "greedy" else c.moe.topk_method = "group_limited" end
    c.moe.n_group = math.max(1, int(cfg.n_group, 1))
    c.moe.topk_group = math.max(1, int(cfg.topk_group, 1))
    c.moe.routed_scaling_factor = num(cfg.routed_scaling_factor, 1.0)
    c.norm_topk_prob = flag(cfg.norm_topk_prob, true)
    c.num_experts_per_tok = int(cfg.num_experts_per_tok, 8)
    if c.num_experts > 0 then
      if c.moe.topk_method == "group_limited" and c.num_experts % c.moe.n_group ~= 0 then
        invalid("MiMo num_experts must be a multiple of n_group")
      end
      each_layer(c.moe_layers, function(i) return i > 0 end)
      local ml, mf = cfg.mlp_layer_types, cfg.moe_layer_freq
      if ml ~= nil then
        if not keys.is_array(ml) or #ml < n then
          invalid("MiMo mlp_layer_types needs one entry per layer")
        end
        for i = 1, n do
          local v = ml[i]
          if v == "sparse" then
            c.moe_layers[i] = true
          elseif v == "dense" then
            c.moe_layers[i] = false
          else
            invalid("MiMo mlp_layer_types entries are \"sparse\" or \"dense\"")
          end
        end
      elseif mf ~= nil then
        if keys.is_array(mf) then
          if #mf < n then invalid("MiMo moe_layer_freq needs one entry per layer") end
          for i = 1, n do c.moe_layers[i] = not (math.type(mf[i]) == "integer" and mf[i] == 0) end
        end
      end
    end
    -- MiMo-V2.5 / V2.6 Pro checkpoints fuse q/k/v into one `qkv_proj`
    -- pre-sharded over `num_key_value_heads` chunks, each `[Q | K | V]`.
    c.qkv_alt = "grouped"
    c.qkv_chunks = kv_full
    -- The fp8 attention projections (fused or not) are block-quantised
    -- shard by shard, one shard per full-layer kv head (SGLang's loader).
    c.quant.attn_row_shards = kv_full
  end,
}
