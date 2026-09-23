return {
  model_type = "seed_oss",
  llama_cpp = "seed_oss",
  chat = "seed",
  verified = true,
  notes = "fixture: llama layout with q/k/v biases and an unbiased o_proj (attention_out_bias), explicit head_dim. Seed-OSS 36B.",
  attention_bias = true,
}
