"""A plain transformers reference for a full (not lazy) directory.

    python3 tools/probe_reference.py MODEL probe.json --dtype float32 --factory tools/ref_plain.py

flash-linear-attention is hidden so the linear-attention families take their
torch paths on the CPU, and image-text wrappers (GLM-5.3-Flash) load through
their own auto class. For an abliterated export, which has every expert:
tools/ref_lazy_moe.py builds the routed experts empty for a lazy file.

With REF_F32_ARITH=1 the model is loaded in its stored bf16 and computed in
float32 (ref_lazy_moe's `f32_arithmetic`: Linear weights cast in row blocks as
they compute, everything else float32, F32-stored tensors restored exactly),
for an export whose float32 copy would not fit in memory.
"""
import sys
if "fla" not in sys.modules:
    sys.modules["fla"] = None
import os  # noqa: E402
import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoModelForImageTextToText  # noqa: E402
def load(d, dtype=torch.float32):
    arith = os.environ.get("REF_F32_ARITH") == "1"
    kw = dict(dtype=torch.bfloat16 if arith else dtype)
    try:
        model = AutoModelForCausalLM.from_pretrained(d, **kw)
    except ValueError:
        model = AutoModelForImageTextToText.from_pretrained(d, **kw)
    if arith:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from lazy_checkpoint import LazyCheckpoint
        from ref_lazy_moe import f32_arithmetic
        f32_arithmetic(model, LazyCheckpoint(d))
    return model
