local mamba = require("mamba")

return {
  model_type = "granitemoehybrid",
  llama_cpp = "granitehybrid",
  chat = "granite",
  verified = true,
  notes = "fixtures: Mamba2 and attention layers from layer_types (also the legacy mamba / attention names; the attention-only layout too), embedding / attention / residual / logits multipliers, fused input_linear / output_linear routed experts plus the fused shared_mlp, or a dense shared_mlp when num_local_experts is 0, optional RoPE (position_embedding_type). Granite 4.0 H (tiny, small).",
  positional = "none",
  mlp = "gated_fused",
  ssm = "mamba2",
  names = {
    prefixes = { "model.", "" },
    ssm = "mamba.",
    gate = false,
    up = false,
    gate_up = "shared_mlp.input_linear.weight",
    down = "shared_mlp.output_linear.weight",
    router = "block_sparse_moe.router.layer.weight",
    expert = "block_sparse_moe.experts.{e}.",
    expert_down = "output_linear.weight",
    fused_gate_up = { "block_sparse_moe.input_linear.weight" },
    fused_down = { "block_sparse_moe.output_linear.weight" },
    shared_expert = "shared_mlp.",
    shared_down = "output_linear.weight",
    shared_gate_up = "input_linear.weight",
  },
  config = function(cfg, c)
    -- Granite's multipliers (the `granite` hook), then Granite MoE's routing.
    c.embed_scale = num(cfg.embedding_multiplier, 1.0)
    c.residual_multiplier = num(cfg.residual_multiplier, 1.0)
    if type(cfg.attention_multiplier) == "number" then c.attention_scale = cfg.attention_multiplier end
    local ls = f32(num(cfg.logits_scaling, 1.0))
    if ls ~= 0 then c.logit_scale = f32(1.0 / ls) end
    -- Top-k over the router logits, softmax over the selected ones (equal
    -- to a renormalised softmax over every expert).
    c.norm_topk_prob = true
    if c.num_experts > 0 then each_layer(c.moe_layers, function() return true end) end
    mamba.dims(cfg, c)
    -- Attention layers use RoPE only when the config asks for it.
    c.positional = "none"
    if str(cfg.position_embedding_type) == "rope" then
      c.positional = "rope"
      each_layer(c.rope_layers, function() return true end)
    end
    if cfg.layer_types == nil and cfg.layers_block_type == nil then
      each_layer(c.ssm_layers, function() return true end)
      each_layer(c.attn_layers, function() return false end)
    end
    -- Softmax over the top-k router logits equals a renormalised softmax top-k.
    c.norm_topk_prob = true
    each_layer(c.mlp_layers, function() return true end)
  end,
}
