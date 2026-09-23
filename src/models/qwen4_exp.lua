return {
  model_type = "qwen4_exp",
  aliases = { "qwen4_exp_text" },
  llama_cpp = nil,
  chat = "chatml",
  verified = true,
  notes = "fixture: Qwen3.8-Flash-Next text config under the multimodal wrapper: gated hyper-connections (hc_count streams, (1 + w) group norms, low-rank sigmoid input mixer, sigmoid injection weights, no final norm), Gated DeltaNet layers with a sigmoid or silu output gate, sigmoid-gated full attention with (1 + w) head norms and partial rotary behind a QSA indexer (run as dense: exact while every complete block fits indexer_budget, longer prompts refused), per-layer n-gram embeddings (PLE: splitmix64 hash multipliers, prime bucket ranges, sharded tables read row by row, gated values, dilated depthwise convolution), fused softmax MoE with a gated shared expert on every layer. Vision and indexer tensors pass through exports untouched.",
  norm = "rms_gemma",
  qk_norm = "head",
  names = {
    final_norm = false,
    input_norm = {},
    pre_ff_norm = false,
    q_norm = "self_attn.q_norm.weight",
    k_norm = "self_attn.k_norm.weight",
    lin_qkv = "linear_attn.in_proj_qkv.weight",
    lin_z = "linear_attn.in_proj_z.weight",
    lin_b = "linear_attn.in_proj_b.weight",
    lin_a = "linear_attn.in_proj_a.weight",
    lin_conv = "linear_attn.conv1d.weight",
    lin_dt_bias = { "linear_attn.dt_bias" },
    lin_a_log = { "linear_attn.A_log" },
    lin_norm = "linear_attn.norm.weight",
    lin_out = "linear_attn.out_proj.weight",
    shared_expert = "mlp.shared_expert.",
    shared_expert_gate = "mlp.shared_expert_gate.weight",
    hc_attn = "attn_hyper_connection",
    hc_ffn = "mlp_hyper_connection",
    hc_attn_flat = false,
    hc_ffn_flat = false,
  },
  -- Qwen3.8-Flash-Next (`qwen4_exp`, `Qwen4ExpTextConfig`): Gated DeltaNet
  -- layers 3:1 with gated full attention behind a QSA indexer (run as its
  -- dense equivalent within `indexer_budget`), gated hyper-connections
  -- (`hc_count` streams, no final norm), softmax MoE with a gated shared
  -- expert on every layer, per-layer n-gram embeddings on `ple_layer_ids`.
  config = function(cfg, c)
    local keys = require("hybrid_keys")
    c.qk_norm = "head"
    c.gated_attention = true
    -- The attention gate is always a sigmoid; `output_gate_type` (default
    -- `hidden_act`) picks the Gated DeltaNet output gate.
    local gate = str(cfg.output_gate_type) or str(cfg.hidden_act) or "silu"
    if gate == "sigmoid" then
      c.linear_gate_sigmoid = true
    elseif gate ~= "silu" and gate ~= "swish" then
      unsupported("unsupported output_gate_type '" .. gate .. "' (sigmoid or silu)")
    end
    if cfg.layer_types == nil and type(cfg.full_attention_interval) ~= "number" then
      each_layer(c.linear_layers, function(i) return (i + 1) % 4 ~= 0 end)
      keys.update_has_linear(c)
    end
    -- Every layer is a mixture of experts with a sigmoid-gated shared expert.
    if type(cfg.num_experts) ~= "number" then c.num_experts = 512 end
    c.num_experts_per_tok = int(cfg.num_experts_per_tok, 10)
    c.norm_topk_prob = flag(cfg.norm_topk_prob, true)
    if c.num_experts > 0 then each_layer(c.moe_layers, function() return true end) end
    if type(cfg.moe_intermediate_size) ~= "number" then c.moe_intermediate_size = 512 end
    -- Gated hyper-connections.
    c.hc_mult = math.max(1, int(cfg.hc_count, 4))
    if c.hc_mult < 2 then
      invalid("qwen4_exp needs hc_count > 1 (config has " .. c.hc_mult .. ")")
    end
    c.hyper = { kind = "gated", head = "gated_mixer", sinkhorn_iters = 20, eps = 1e-6, lowrank = int(cfg.hc_lowrank, 320) }
    if c.hyper.lowrank == 0 then invalid("qwen4_exp: hc_lowrank must be positive") end
    -- QSA indexer: `indexer_budget` tokens from complete blocks of
    -- `indexer_compress_ratio` keys plus the incomplete tail.
    if type(cfg.indexer_budget) == "number" or type(cfg.indexer_compress_ratio) == "number" or type(cfg.indexer_n_heads) == "number" then
      local budget = int(cfg.indexer_budget, 0)
      local ratio = int(cfg.indexer_compress_ratio, 0)
      if budget == 0 or ratio == 0 or budget % ratio ~= 0 or int(cfg.indexer_kv_heads, 1) ~= 1 then
        invalid("qwen4_exp: the QSA config needs indexer_budget (a multiple of indexer_compress_ratio) and indexer_kv_heads = 1")
      end
      c.index_bound = { block = ratio, max_blocks = budget // ratio }
      warn("qwen4_exp: the QSA indexer runs as dense attention, exact for prompts up to " .. budget .. " tokens (indexer_budget); longer prompts are refused")
    end
    -- Per-layer n-gram embeddings.
    local ids_1 = keys.int_list(cfg.ple_layer_ids)
    if ids_1 and #ids_1 > 0 then
      local ids = {}
      for _, id1 in ipairs(ids_1) do
        if id1 < 1 or id1 > c.num_layers then
          invalid("qwen4_exp: ple_layer_ids entry " .. id1 .. " is outside 1.." .. c.num_layers)
        end
        local id0 = id1 - 1
        local seen = false
        for _, x in ipairs(ids) do seen = seen or x == id0 end
        if not seen then ids[#ids + 1] = id0 end
      end
      table.sort(ids)
      for _, id0 in ipairs(ids) do
        if not c.linear_layers[id0 + 1] then
          unsupported("qwen4_exp: PLE layer " .. (id0 + 1) .. " is not a linear_attention layer")
        end
      end
      local ngram = int(cfg.ngram_size, 3)
      local heads = int(cfg.heads_per_ngram, 8)
      local embed_dim = int(cfg.ple_embed_dim, c.hidden_size)
      local n_cols = math.max(ngram - 1, 0) * heads
      if ngram < 2 or heads == 0 or embed_dim == 0 or embed_dim % n_cols ~= 0 then
        invalid("qwen4_exp: the PLE config needs ngram_size >= 2, heads_per_ngram > 0 and ple_embed_dim a multiple of (ngram_size - 1) * heads_per_ngram")
      end
      local none = 0xFFFFFFFF
      local eos = none
      local v = cfg.eos_token_id
      if math.type(v) == "integer" then
        if v >= 0 then eos = v end
      elseif keys.is_array(v) and #v > 0 and math.type(v[1]) == "integer" and v[1] >= 0 then
        eos = v[1]
      end
      if eos == none then
        invalid("qwen4_exp: eos_token_id must be set in the text config when PLE layers are enabled (it pads the n-gram history)")
      end
      c.ngram_ple = {
        layer_ids = ids,
        embed_dim = embed_dim,
        conv_kernel = math.max(1, int(cfg.ple_conv_kernel_size, 4)),
        ngram_size = ngram,
        heads_per_ngram = heads,
        vocab_base = int(cfg.ngram_vocab_size_base, 20000000),
        divisible_by = math.max(1, int(cfg.make_ngram_vocab_size_divisible_by, 128)),
        seed = int(cfg.seed, 1234),
        vocab_size = c.vocab_size,
        eos_id = eos,
      }
    end
  end,
}
