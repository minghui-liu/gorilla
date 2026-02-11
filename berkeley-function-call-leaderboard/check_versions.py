#!/usr/bin/env python3
"""Check versions of vLLM, PyTorch, and CUDA."""

import vllm
import torch

print(f"vLLM version: {vllm.__version__}")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA version: {torch.version.cuda}")
