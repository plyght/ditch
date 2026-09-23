return {
  model_type = "baichuan",
  llama_cpp = "baichuan",
  verified = true,
  notes = "fixture: fused W_pack (7B, RoPE). The 13B ALiBi variant (detected by model_max_length) is unverified. Needs a tokenizer.json (the SentencePiece-only checkpoints must be converted).",
  qkv = "concat",
  names = {
    q = false,
    k = false,
    v = false,
    qkv = "self_attn.W_pack.weight",
  },
  config = function(cfg, c)
    -- Baichuan 2 (vocabulary 125696; Baichuan 1 has 64000) normalises its LM
    -- head rows at inference (`NormHead`); llama.cpp tells the two apart the same way.
    c.lm_head_l2norm = c.vocab_size == 125696
    -- Baichuan 13B checkpoints carry `model_max_length` instead of
    -- `max_position_embeddings` and use ALiBi; 7B uses RoPE.
    if cfg.max_position_embeddings == nil and type(cfg.model_max_length) == "number" then
      c.positional = "alibi"
      c.max_position_embeddings = int(cfg.model_max_length, 4096)
    end
  end,
}
