return {
  model_type = "gpt_neo",
  llama_cpp = "gptneo",
  verified = true,
  notes = "fixture: learned positions, unscaled attention logits, alternating global and window_size local layers (attention_types), c_fc/c_proj MLP. GPT-Neo 1.3B/2.7B.",
  norm = "layer",
  positional = "learned",
  mlp = "dense",
  activation = "gelu_tanh",
  tie_word_embeddings = true,
  names = {
    prefixes = { "transformer.", "" },
    embed = "{p}wte.weight",
    pos_embed = "{p}wpe.weight",
    final_norm = "{p}ln_f.weight",
    layer = "{p}h.{i}.",
    input_norm = { "ln_1.weight" },
    pre_ff_norm = "ln_2.weight",
    q = "attn.attention.q_proj.weight",
    k = "attn.attention.k_proj.weight",
    v = "attn.attention.v_proj.weight",
    o = "attn.attention.out_proj.weight",
    gate = false,
    up = "mlp.c_fc.weight",
    down = "mlp.c_proj.weight",
  },
  config = function(cfg, c)
    -- GPT-Neo does not scale the attention logits, and alternates global and
    -- `window_size` local layers (`attention_types` expands to `attention_layers`).
    c.attention_scale = 1.0
    c.num_kv_heads = c.num_heads
    c.sliding_window = int(cfg.window_size, 256)
    each_layer(c.sliding_layers, function() return false end)
    local types = nil
    local al = cfg.attention_layers
    if is_array(al) then
      types = {}
      for i = 1, #al do types[i] = str(al[i]) or "global" end
    end
    if types == nil and cfg.attention_types ~= nil then
      -- `[[["global", "local"], n], ...]`: each inner list repeated n times.
      local list = {}
      local at = cfg.attention_types
      for k = 1, len(at) do
        local item = at[k]
        if is_array(item) and #item == 2 then
          local names_v, rep = item[1], item[2]
          if is_array(names_v) and math.type(rep) == "integer" then
            for _ = 1, rep do
              for _, nv in ipairs(names_v) do
                if type(nv) == "string" then list[#list + 1] = nv end
              end
            end
          end
        end
      end
      if #list > 0 then types = list end
    end
    local list = types
    if list == nil then
      list = {}
      for i = 0, c.num_layers - 1 do list[i + 1] = i % 2 == 0 and "global" or "local" end
    end
    each_layer(c.sliding_layers, function(i) return i < #list and list[i + 1] == "local" end)
  end,
}
