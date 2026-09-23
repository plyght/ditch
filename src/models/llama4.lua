return {
  model_type = "llama4",
  aliases = { "llama4_text" },
  llama_cpp = "llama4",
  chat = "llama4",
  verified = true,
  notes = "fixture (text): top-1 sigmoid routing scaling the expert input, shared expert, transposed fused experts, no_rope_layers with attention temperature tuning, L2 qk norm, interleaved rope, dense layers with intermediate_size_mlp. Chunked attention runs as full attention.",
  default_norm_eps = 1e-5,
  default_rope_theta = 500000.0,
  rope_style = "gptj",
  names = {
    gate = "feed_forward.gate_proj.weight",
    up = "feed_forward.up_proj.weight",
    down = "feed_forward.down_proj.weight",
    router = "feed_forward.router.weight",
    expert = "feed_forward.experts.{e}.",
    fused_gate_up = { "feed_forward.experts.gate_up_proj" },
    fused_down = { "feed_forward.experts.down_proj" },
    shared_expert = "feed_forward.shared_expert.",
  },
  config = function(cfg, c)
    local flags = require("layers").flags(cfg.no_rope_layers, c.num_layers, true)
    if flags then each_layer(c.rope_layers, function(i) return flags[i + 1] end) end
    if flag(cfg.use_qk_norm, true) then
      c.qk_norm = "l2"
      c.qk_norm_rope_only = true
    end
    if flag(cfg.attn_temperature_tuning, false) then
      c.attn_temperature = {
        floor_scale = num(cfg.floor_scale, 8192),
        attn_scale = num(cfg.attn_scale, 0.1),
        offset = 1.0,
        all_layers = false,
      }
    end
    c.num_experts_per_tok = int(cfg.num_experts_per_tok, 1)
    c.moe.scoring = "sigmoid"
    c.moe.scale_input = true
    -- Routed and shared experts use `intermediate_size`; dense layers `intermediate_size_mlp`.
    c.moe_intermediate_size = c.intermediate_size
    c.intermediate_size = int(cfg.intermediate_size_mlp, c.intermediate_size)
    if type(cfg.attention_chunk_size) == "number" then
      warn("llama4: chunked local attention is run as full attention (exact for prompts shorter than attention_chunk_size)")
      each_layer(c.sliding_layers, function() return false end)
    end
  end,
}
