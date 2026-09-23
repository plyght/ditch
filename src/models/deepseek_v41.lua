local dsv4 = require("dsv4")

return {
  model_type = "deepseek_v41",
  aliases = { "deepseek_v41_text" },
  llama_cpp = nil,
  chat = "deepseek",
  verified = true,
  notes = "fixture: single-pass hyper-connections, CSA2 shared compressed KV (kv_source groups, ratio 1 and pooled branches, indexer as dense), FP8/FP4 fake quantisation of the window KV and latents, engram n-gram hash layers (lazy table rows, tokenizer-derived compressed ids), gate_temp routing, nested text_config with vision tensors passed through. The released checkpoints (DeepSeek's own tensor names, FP8 with ue8m0 block scales, FP4 e2m1 experts) are renamed and dequantised on load.",
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
    local d = { v41 = true, fake_quant = true }
    if type(cfg.rms_norm_eps) ~= "number" then c.rms_norm_eps = 1e-20 end
    dsv4.common(cfg, c, d)
    d.index_n_heads = int(cfg.index_n_heads, 32)
    c.norm_topk_prob = flag(cfg.norm_topk_prob, true)
    c.moe.gate_temp = num(cfg.gate_temp, 1.0)
    if f32(c.moe.gate_temp) == 0 then invalid("deepseek_v41: gate_temp must not be 0") end
    local n = c.num_layers
    local n_nextn = int(cfg.num_nextn_predict_layers, 3)
    -- Per-layer pooling ratio: 0 sliding only, 1 full-resolution shared KV,
    -- r > 1 pooled. Trailing entries (the draft layers) are ignored.
    local ratio = {}
    local cr = dsv4.int_list(cfg.compress_ratios)
    if cr then
      if #cr < n or #cr > n + n_nextn then
        invalid("deepseek_v41: compress_ratios needs one entry per layer (and at most one per draft layer)")
      end
      for i = 1, n do ratio[i] = cr[i] end
    elseif n == 40 then
      for i = 0, n - 1 do ratio[i + 1] = (i < 2 and 0) or (i < 20 and 2) or 1 end
    else
      local n_slide = math.min(2, n)
      local n_enc = (n - n_slide + 1) // 2
      for i = 0, n - 1 do ratio[i + 1] = (i < n_slide and 0) or (i < n_slide + n_enc and 2) or 1 end
    end
    -- KV sources: explicit, or the first layer of every run of equal ratios.
    local sources = dsv4.int_list(cfg.kv_source_layer_ids)
    if not sources then
      sources = {}
      for i = 0, n - 1 do
        local r = ratio[i + 1]
        if r > 0 and (i == 0 or ratio[i] ~= r) then sources[#sources + 1] = i end
      end
    end
    local branch, source = {}, {}
    for i = 0, n - 1 do
      branch[i + 1] = ratio[i + 1] > 0 and "shared" or "none"
      source[i + 1] = false
      if ratio[i + 1] ~= 0 then
        local src = nil
        for _, sidx in ipairs(sources) do
          if sidx <= i and (src == nil or sidx > src) then src = sidx end
        end
        if src == nil then
          invalid(string.format("deepseek_v41: layer %d has a compressed branch but no kv_source_layer_ids entry at or before it", i))
        end
        source[i + 1] = src
        if src >= n or ratio[src + 1] == 0 then
          invalid(string.format("deepseek_v41: layer %d reads the compressed cache of layer %d, which has none", i, src))
        end
        if ratio[src + 1] ~= ratio[i + 1] then
          invalid(string.format("deepseek_v41: layer %d (ratio %d) reads the compressed cache of layer %d (ratio %d)", i, ratio[i + 1], src, ratio[src + 1]))
        end
      end
    end
    d.branch = branch
    d.compress_ratio = ratio
    d.kv_source = source
    d.candidate_topk_blocks = int(cfg.candidate_topk_blocks, 2048)
    d.candidate_block_size = math.max(1, int(cfg.candidate_block_size, 8))
    d.candidate_source = nil
    local v = cfg.candidate_source_layer_id
    if v ~= nil then
      if math.type(v) == "integer" and v >= 0 then d.candidate_source = v end
    elseif #sources > 0 then
      d.candidate_source = sources[#sources]
    end
    if d.candidate_source and d.candidate_source >= n then
      invalid(string.format("deepseek_v41: candidate_source_layer_id %d is not a layer", d.candidate_source))
    end
    local hash = {}
    for i = 1, n do hash[i] = false end
    d.hash_moe_layers = hash
    -- Engram conditional memory.
    d.engram = nil
    local ids = dsv4.int_list(cfg.engram_layer_ids)
    if ids and #ids > 0 then
      local counts = dsv4.int_list(cfg.engram_num_embeddings)
      if not counts then invalid("deepseek_v41: engram_layer_ids needs engram_num_embeddings") end
      if #counts ~= #ids then invalid("deepseek_v41: engram_num_embeddings needs one entry per engram layer") end
      for _, li in ipairs(ids) do
        if li >= n then invalid(string.format("deepseek_v41: engram layer %d is not a layer", li)) end
      end
      local pad = int(cfg.engram_pad_id, 2)
      d.engram = {
        layer_ids = ids,
        num_embeddings = counts,
        max_ngram = math.max(2, int(cfg.engram_max_ngram_size, 4)),
        n_heads = math.max(1, int(cfg.engram_n_heads, 8)),
        head_dim = int(cfg.engram_head_dim, 256),
        pad_id = pad,
        compressed_vocab_size = int(cfg.engram_compressed_vocab_size, 99092),
        vocab_size = int(cfg.engram_vocab_size, 16000000),
      }
    end
    c.dsv4 = d
  end,
}
