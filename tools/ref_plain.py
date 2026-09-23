"""A plain transformers reference for a full (not lazy) directory.

    python3 tools/probe_reference.py MODEL probe.json --dtype float32 --factory tools/ref_plain.py

flash-linear-attention is hidden so the linear-attention families take their
torch paths on the CPU, and image-text wrappers (GLM-5.3-Flash) load through
their own auto class. For an abliterated export, which has every expert:
tools/ref_lazy_moe.py builds the routed experts empty for a lazy file.
"""
import sys
if "fla" not in sys.modules:
    sys.modules["fla"] = None
import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoModelForImageTextToText  # noqa: E402
def load(d, dtype=torch.float32):
    try:
        return AutoModelForCausalLM.from_pretrained(d, dtype=dtype)
    except ValueError:
        return AutoModelForImageTextToText.from_pretrained(d, dtype=dtype)
