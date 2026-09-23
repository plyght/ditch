return {
  model_type = "mamba2",
  llama_cpp = "mamba2",
  verified = true,
  notes = "fixture: pure Mamba2 (SSD) blocks: in_proj split into gate / conv channels / dt, biased causal conv1d, grouped B/C, per-head decay, D skip, gated RMSNorm, out_proj; no attention, no MLP. Mamba-Codestral, state-spaces/mamba2-*-hf.",
  default_norm_eps = 1e-5,
  positional = "none",
  ssm = "mamba2",
  single_mixer = true,
  names = {
    prefixes = { "backbone.", "" },
    embed = "{p}embeddings.weight",
    final_norm = "{p}norm_f.weight",
    input_norm = { "norm.weight" },
    pre_ff_norm = false,
    ssm = "mixer.",
  },
  hook = "mamba2",
}
