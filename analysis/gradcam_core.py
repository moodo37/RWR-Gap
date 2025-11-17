
"""
Core Grad-CAM implementation used for both CXR and ISIC experiments.
"""

from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class GradCAM:
    """
    Generic Grad-CAM for CNN classifiers (ResNet, DenseNet, etc.).

    Usage:
        cam = GradCAM(model, target_layer="layer4")
        heatmap = cam.generate(x, class_idx=1)
    """

    def __init__(self, model: nn.Module, target_layer: str):
        self.model = model.eval()
        self.target_layer = target_layer
        self._features = None

        modules = dict(self.model.named_modules())
        if target_layer not in modules:
            raise ValueError(f"target_layer {target_layer} not found in model modules")

        layer = modules[target_layer]
        layer.register_forward_hook(self._forward_hook)

    def _forward_hook(self, module, inp, out):
        # Save feature map and keep gradient
        self._features = out
        if torch.is_grad_enabled():
            self._features.retain_grad()

    @torch.no_grad()
    def _upsample(self, cam: torch.Tensor, size_hw: Tuple[int, int]) -> torch.Tensor:
        cam = cam.unsqueeze(0).unsqueeze(0)
        cam = F.interpolate(cam, size=size_hw, mode="bilinear", align_corners=False)
        return cam.squeeze(0).squeeze(0)

    def generate(self, x: torch.Tensor, class_idx: int = None):
        """
        Generate a Grad-CAM heatmap for input x.

        Args:
            x: Input tensor (B, C, H, W), usually B=1.
            class_idx: Target class index. If None, use argmax.

        Returns:
            numpy array of shape (H, W) with values in [0, 1].
        """
        import numpy as np

        self.model.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(True):
            logits = self.model(x)
            if class_idx is None:
                class_idx = logits.argmax(dim=1).item()
            score = logits[:, class_idx]

            # Small score → scale up to avoid vanishing gradients
            if score.abs().max().item() < 0.05:
                score = score * 50.0

            score.backward()

        feats = self._features
        grads = self._features.grad
        if grads is None:
            raise RuntimeError("features.grad is None. Check that requires_grad is True.")

        weights = grads.mean(dim=(2, 3), keepdim=True)
        cam = (weights * feats).sum(dim=1)
        cam = F.relu(cam)[0]
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        cam = self._upsample(cam, (x.shape[2], x.shape[3]))
        return cam.detach().cpu().numpy()
