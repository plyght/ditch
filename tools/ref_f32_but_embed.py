"""A float32 reference that keeps the input embedding in bf16 (a lookup of bf16
values is exact either way) and casts its output to float32: for cuts whose
float32 embedding would not fit in memory."""
import torch
from transformers import AutoModelForCausalLM
def load(d, dtype=torch.float32):
    m = AutoModelForCausalLM.from_pretrained(d, dtype=torch.bfloat16)
    emb = m.get_input_embeddings()
    for mod in m.modules():
        if mod is emb: continue
        for n, p in list(mod.named_parameters(recurse=False)):
            p.data = p.data.float()
        for n, b in list(mod.named_buffers(recurse=False)):
            if b.is_floating_point(): setattr(mod, n, b.float())
    emb.register_forward_hook(lambda mod, i, o: o.float())
    m.config.torch_dtype = torch.float32
    return m
