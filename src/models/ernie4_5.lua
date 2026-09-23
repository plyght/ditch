return {
  model_type = "ernie4_5",
  aliases = { "paddleocr_vl_text" },
  llama_cpp = "ernie4_5",
  chat = "ernie",
  verified = true,
  notes = "fixture: interleaved rotary, use_bias projections. ERNIE 4.5 dense (and the PaddleOCR-VL text config).",
  default_norm_eps = 1e-5,
  default_rope_theta = 500000.0,
  rope_style = "gptj",
  hook = "ernie",
}
