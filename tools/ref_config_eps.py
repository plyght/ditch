"""A transformers reference whose MLA latent norms use the config's epsilon.

    python3 tools/probe_reference.py MODEL probe.json --dtype float32 --factory tools/ref_config_eps.py

transformers builds `q_a_layernorm` and `kv_a_layernorm` of its DeepSeek-style
attention (glm4_moe_lite and others) with the RMSNorm default of 1e-6 and
ignores `rms_norm_eps`. vLLM and SGLang pass `rms_norm_eps` to both, as ditch
does. For a checkpoint whose `rms_norm_eps` is not 1e-6 (GLM-4.7-Flash: 1e-5)
that alone moves the first layer's output by ~1e-3; this factory loads the
model as usual and sets the two latent norms to the config's epsilon so the
comparison measures everything else.
"""
import torch
from transformers import AutoModelForCausalLM


def load(model_dir, dtype=torch.float32):
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=dtype)
    eps = model.config.rms_norm_eps
    for layer in model.model.layers:
        for name in ("q_a_layernorm", "kv_a_layernorm"):
            norm = getattr(layer.self_attn, name, None)
            if norm is not None:
                norm.variance_epsilon = eps
    return model
