local moe = require("moe")

return {
  model_type = "hunyuan_v1_moe",
  aliases = { "hunyuan" },
  llama_cpp = "hunyuan-moe",
  chat = "hunyuan_moe",
  verified = true,
  notes = "fixture: per-head q/k RMSNorm after RoPE, NTK-alpha dynamic rope base, softmax top-k renormalised, shared MLP, per-layer (uniform) expert counts. Hunyuan-A13B.",
  default_norm_eps = 1e-5,
  qk_norm = "head",
  names = {
    q_norm = "self_attn.query_layernorm.weight",
    k_norm = "self_attn.key_layernorm.weight",
    router = "mlp.gate.wg.weight",
    shared_expert = "mlp.shared_mlp.",
  },
  config = function(cfg, c)
    -- Softmax top-k renormalised; per-head q/k RMSNorm applied after RoPE;
    -- NTK-alpha "dynamic" rope scaling folded into the base.
    c.norm_topk_prob = true
    c.qk_norm_after_rope = true
    c.attention_bias = flag(cfg.attention_bias, false)
    if flag(cfg.use_cla, false) then unsupported("HunYuan cross-layer attention (use_cla)") end
    c.num_experts = moe.uniform_int(cfg, "num_experts", c.num_experts)
    c.num_experts_per_tok = moe.uniform_int(cfg, "moe_topk", int(cfg.num_experts_per_tok, 1))
    c.moe_intermediate_size = moe.uniform_int(cfg, "moe_intermediate_size", c.intermediate_size)
    each_layer(c.moe_layers, function() return c.num_experts > 0 end)
    local rs = obj(cfg.rope_scaling) or obj(cfg.rope_parameters)
    if rs then
      local t = str(rs.rope_type) or str(rs.type) or ""
      if t == "dynamic" and type(rs.alpha) == "number" then
        local d = c.head_dim
        c.rope_theta = f32(c.rope_theta * pow(rs.alpha, d / (d - 2.0)))
      end
    end
  end,
}
