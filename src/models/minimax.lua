return {
  model_type = "minimax",
  aliases = { "minimax_text_01", "minimax_m1", "MiniMaxText01", "MiniMaxM1" },
  llama_cpp = "minimax-01",
  verified = true,
  notes = "fixture: lightning attention layers (`hidden_act` on the fused qkv, silu on the releases; per-head decay recurrence, RMSNorm, sigmoid output gate) alternating with softmax attention with partial rotary, the renormalised residual layout with α/β scales, softmax top-k MoE. MiniMax-Text-01 / M1 (`layer_types` or `attn_type_list`). The recurrence runs sequentially.",
  default_norm_eps = 1e-5,
  linear = "lightning",
  names = {
    light_qkv = "self_attn.qkv_proj.weight",
    light_gate = "self_attn.output_gate.weight",
    light_norm = "self_attn.norm.weight",
    light_out = "self_attn.out_proj.weight",
    router = "block_sparse_moe.gate.weight",
    expert = "block_sparse_moe.experts.{e}.",
    expert_gate = "w1.weight",
    expert_up = "w3.weight",
    expert_down = "w2.weight",
    fused_gate_up = {},
    fused_down = {},
  },
  config = function(cfg, c)
    local keys = require("hybrid_keys")
    -- MiniMax-01 / M1: lightning (linear) attention on every layer but each
    -- `full_attention` one, a renormalised residual stream with per-sublayer
    -- α/β scales, softmax top-k routing renormalised over the selected experts.
    c.residual_layout = "minimax"
    c.norm_topk_prob = true
    if type(cfg.rope_theta) ~= "number" then c.rope_theta = 1000000.0 end
    if cfg.layer_types == nil then
      -- Remote-code checkpoints carry `attn_type_list` (0 = lightning, 1 =
      -- softmax attention); the Hugging Face default alternates, starting linear.
      local al = cfg.attn_type_list
      if al ~= nil then
        if keys.is_array(al) then
          for i = 1, math.min(#al, c.num_layers) do
            c.linear_layers[i] = math.type(al[i]) == "integer" and al[i] == 0
          end
        end
      else
        each_layer(c.linear_layers, function(i) return (i + 1) % 2 ~= 0 end)
      end
      keys.update_has_linear(c)
    end
    local scale_keys = {
      { { "full_attn_alpha_factor", "layernorm_full_attention_alpha" }, { "full_attn_beta_factor", "layernorm_full_attention_beta" } },
      { { "linear_attn_alpha_factor", "layernorm_linear_attention_alpha" }, { "linear_attn_beta_factor", "layernorm_linear_attention_beta" } },
      { { "mlp_alpha_factor", "layernorm_mlp_alpha" }, { "mlp_beta_factor", "layernorm_mlp_beta" } },
    }
    for k = 1, 3 do
      c.minimax_scales[k][1] = keys.num_any(cfg, scale_keys[k][1], 1.0)
      c.minimax_scales[k][2] = keys.num_any(cfg, scale_keys[k][2], 1.0)
    end
    -- `minimax` takes the residual *after* each norm: the remote code's
    -- `postnorm: true`, which every released checkpoint sets, and the only
    -- layout transformers' native `minimax` has (it ignores the key). The
    -- remote code's default, `postnorm: false`, keeps the residual from
    -- before the norm; no release uses it and it is not implemented.
    if c.model_type ~= "minimax" and not flag(cfg.postnorm, false) then
      unsupported("MiniMax with postnorm: false (the residual taken before the norm); every release sets postnorm: true")
    end
    if c.num_experts > 0 then each_layer(c.moe_layers, function() return true end) end
  end,
}
