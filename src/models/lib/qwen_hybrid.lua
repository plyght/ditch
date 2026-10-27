-- The Qwen hybrid families (Qwen3-Next, Qwen3.5, Qwen3.5 MoE): per-head q/k
-- norms and gated full attention beside the linear-attention layers.
local M = {}

function M.config(cfg, c)
  c.qk_norm = "head"
  c.gated_attention = true
  -- Qwen3.5 MoE renormalises the top-k weights unless config.json says
  -- otherwise. `c.model_type` is the config's own spelling (qwen3_5_moe_text,
  -- or a wrapper's such as memt_qwen3_5_moe).
  if c.num_experts > 0 and (cfg.norm_topk_prob == nil or cfg.norm_topk_prob == null) and string.find(c.model_type, "qwen3_5_moe", 1, true) then
    c.norm_topk_prob = true
  end
end

return M
