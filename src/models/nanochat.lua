return {
  model_type = "nanochat",
  llama_cpp = nil,
  chat = "nanochat",
  verified = true,
  notes = "fixture: non-parametric RMSNorm everywhere (including an extra norm on the embeddings), weightless per-head q/k norm after RoPE, fc1/fc2 relu² MLP, final logit softcapping. nanochat.",
  norm = "rms_none",
  default_norm_eps = 1e-6,
  mlp = "dense",
  activation = "relu2",
  names = {
    embed_norm = "norm.weight",
    gate = false,
    up = "mlp.fc1.weight",
    down = "mlp.fc2.weight",
  },
  config = function(cfg, c)
    -- Weightless RMSNorm everywhere, including the q/k norms, which run after RoPE.
    c.qk_norm = "l2"
    c.qk_norm_after_rope = true
    -- `rotate_half` returns `cat(x2, -x1)` (karpathy's `apply_rotary_emb`
    -- too): the rotation runs the other way from Llama's.
    c.rope_reverse = true
    -- nanochat always softcaps the logits at 15, and transformers'
    -- `NanoChatConfig` defaults `final_logit_softcapping` to it; some
    -- conversions spell the key `logits_soft_cap`, which transformers ignores.
    if c.final_logit_softcapping == nil and cfg.final_logit_softcapping == nil then
      c.final_logit_softcapping = 15.0
    end
  end,
}
