"""Vendored from https://github.com/mwfang/pestconformer (Pest-ConFormer, ESWA 2024).

Only change: unused timm/pdb imports removed, relative import in models_convvit.
"""
from .models_convvit import ConvViT, convvit_base_patch16

__all__ = ["ConvViT", "convvit_base_patch16"]
