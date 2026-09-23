local granite = require("granite")

return {
  model_type = "granitemoe",
  aliases = { "granitemoeshared" },
  llama_cpp = "granitemoe",
  chat = "granite",
  verified = true,
  notes = "fixture: Granite multipliers, fused [E, 2I, H] / [E, H, I] expert tensors (input_linear / output_linear), top-k softmax routing. GraniteMoeShared adds the fused shared_mlp (implemented, covered by the granitemoehybrid fixture).",
  names = {
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
    granite.multipliers(cfg, c)
    -- Top-k over the router logits, softmax over the selected ones (equal
    -- to a renormalised softmax over every expert).
    c.norm_topk_prob = true
    if c.num_experts > 0 then each_layer(c.moe_layers, function() return true end) end
  end,
}
