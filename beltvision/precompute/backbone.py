"""Frozen ResNet-18 patch-feature extractor for the PaDiM / PatchCore-lite banks.

PaDiM (arXiv:2011.08785) and PatchCore (arXiv:2106.08265) both build a "normal" model
from the intermediate feature maps of a frozen ImageNet backbone. This module extracts a
compact per-position feature grid from ResNet-18 layer2 + layer3 (concatenated on a small
common grid), which is enough for a small-sample belt proxy while keeping the committed
bank tiny.

Torch and torchvision are imported lazily: this file is only ever used in the precompute
lane, never in the slim runtime. Everything is CLAHE-first and deterministic (eval mode,
no grad, fixed transform), so the fit and the benchmark scorer use the identical features.
"""
from __future__ import annotations

import numpy as np

from ..methods.preprocess import apply_clahe_lab

# ImageNet normalization (RGB, [0, 1] domain).
_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)


def preprocess_for_backbone(bgr: np.ndarray, input_size: int = 256) -> np.ndarray:
    """CLAHE-first -> RGB -> resize -> ImageNet-normalize. Returns a (3, S, S) float32 array."""
    import cv2

    clahe = apply_clahe_lab(bgr)
    rgb = cv2.cvtColor(clahe, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (input_size, input_size), interpolation=cv2.INTER_AREA)
    x = resized.astype(np.float32) / 255.0
    x = (x - np.array(_MEAN, dtype=np.float32)) / np.array(_STD, dtype=np.float32)
    return np.ascontiguousarray(x.transpose(2, 0, 1))


class ResNetPatchFeatures:
    """A frozen ResNet-18 layer2+layer3 patch-feature extractor (deterministic, CPU)."""

    def __init__(self, input_size: int = 256, grid: int = 8) -> None:
        import torch
        from torchvision.models import ResNet18_Weights, resnet18

        self.input_size = int(input_size)
        self.grid = int(grid)
        self._torch = torch
        net = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        net.eval()
        for p in net.parameters():
            p.requires_grad_(False)
        self._net = net
        # feature dim = layer2 (128) + layer3 (256) channels
        self.feature_dim = 128 + 256

    def _forward_features(self, x):  # x: torch tensor (n, 3, S, S)
        torch = self._torch
        net = self._net
        with torch.no_grad():
            h = net.conv1(x)
            h = net.bn1(h)
            h = net.relu(h)
            h = net.maxpool(h)
            h = net.layer1(h)
            l2 = net.layer2(h)
            l3 = net.layer3(l2)
            pool = torch.nn.functional.adaptive_avg_pool2d
            g = self.grid
            f2 = pool(l2, (g, g))
            f3 = pool(l3, (g, g))
            feat = torch.cat([f2, f3], dim=1)  # (n, 384, g, g)
        return feat

    def extract(self, frames: list[np.ndarray], *, batch_size: int = 8) -> np.ndarray:
        """Return per-position features of shape ``(n_frames, grid*grid, feature_dim)``."""
        torch = self._torch
        arrs = [preprocess_for_backbone(f, self.input_size) for f in frames]
        out: list[np.ndarray] = []
        for i in range(0, len(arrs), batch_size):
            batch = np.stack(arrs[i : i + batch_size], axis=0)
            x = torch.from_numpy(batch)
            feat = self._forward_features(x).cpu().numpy()  # (b, D, g, g)
            b, d, gh, gw = feat.shape
            feat = feat.reshape(b, d, gh * gw).transpose(0, 2, 1)  # (b, g*g, D)
            out.append(feat.astype(np.float32))
        if not out:
            return np.zeros((0, self.grid * self.grid, self.feature_dim), dtype=np.float32)
        return np.concatenate(out, axis=0)
