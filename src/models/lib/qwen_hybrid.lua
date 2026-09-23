-- The Qwen hybrid families (Qwen3-Next, Qwen3.5, Qwen3.5 MoE): per-head q/k
-- norms and gated full attention beside the linear-attention layers.
local M = {}

function M.config(cfg, c)
  c.qk_norm = "head"
  c.gated_attention = true
  -- `c.model_type` is the config's own spelling (qwen3_5_moe_text, ...).
  if c.num_experts > 0 and type(cfg.norm_topk_prob) ~= "number" and string.sub(c.model_type, 1, 11) == "qwen3_5_moe" then
    c.norm_topk_prob = true
  end
end

return M
