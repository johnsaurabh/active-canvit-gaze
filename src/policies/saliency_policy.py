"""
Low-resolution saliency policy.

VALID active policy: computes saliency from a 64x64 downsampled scene view.
Does NOT receive the full-resolution image.

Saliency method: spectral residual (Hou & Zhang, 2007) — simple, transparent,
no pretrained networks. Captures high-contrast / unusual regions.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from .base import GazePolicy, PolicyMetadata, PolicyType, Viewpoint

LOWRES_SIZE = 64  # the downsampled image size the policy uses


def _box_filter(arr: np.ndarray, size: int) -> np.ndarray:
    """Box (mean) filter — pure numpy, no scipy dependency."""
    pad = size // 2
    padded = np.pad(arr, pad, mode='reflect')
    result = np.zeros_like(arr, dtype=np.float64)
    for i in range(size):
        for j in range(size):
            result += padded[i:i + arr.shape[0], j:j + arr.shape[1]]
    return (result / (size * size)).astype(arr.dtype)


def _spectral_residual_saliency(img_gray: np.ndarray) -> np.ndarray:
    """
    Compute spectral residual saliency map.

    img_gray: (H, W) float32 in [0, 1]
    Returns: (H, W) float32 saliency, not normalized
    """
    fft = np.fft.fft2(img_gray)
    log_amplitude = np.log(np.abs(fft) + 1e-8)
    phase = np.angle(fft)

    smoothed = _box_filter(log_amplitude, size=3)
    residual = log_amplitude - smoothed

    saliency = np.abs(np.fft.ifft2(np.exp(residual + 1j * phase))) ** 2
    saliency = _box_filter(saliency, size=5)
    return saliency.astype(np.float32)


def _image_to_saliency_map(lowres: torch.Tensor) -> np.ndarray:
    """
    lowres: [1, 3, H, W] float32 in [0, 1] (already downsampled to LOWRES_SIZE)
    Returns: (H, W) normalized saliency map in [0, 1]
    """
    # Convert to grayscale
    img = lowres.squeeze(0).cpu().numpy()  # [3, H, W]
    gray = 0.299 * img[0] + 0.587 * img[1] + 0.114 * img[2]  # [H, W]

    sal = _spectral_residual_saliency(gray)

    # Normalize to [0, 1]
    sal_min, sal_max = sal.min(), sal.max()
    if sal_max - sal_min < 1e-8:
        return np.ones_like(sal) / sal.size  # uniform if flat
    return (sal - sal_min) / (sal_max - sal_min)


def _saliency_to_viewpoint(
    saliency: np.ndarray, scale: float, priority: Optional[np.ndarray] = None
) -> Viewpoint:
    """
    Convert a saliency map to a CanViT viewpoint.

    saliency: (H, W) in [0, 1]
    priority: (H, W) — combined priority (saliency - IOR penalty). If None, use saliency.
    scale: local glimpse half-side-length

    Coordinate mapping:
        pixel (r, c) in (H, W) saliency map
        → scene coords x = (c / (W-1)) * 2 - 1  (horizontal)
                       y = (r / (H-1)) * 2 - 1  (vertical)
    """
    H, W = saliency.shape
    sal = priority if priority is not None else saliency

    flat_idx = int(np.argmax(sal))
    r = flat_idx // W
    c = flat_idx % W

    # Map pixel → scene coordinates
    x = (c / max(W - 1, 1)) * 2.0 - 1.0
    y = (r / max(H - 1, 1)) * 2.0 - 1.0

    return Viewpoint.clamp(x=x, y=y, s=scale)


class LowResSaliencyPolicy(GazePolicy):
    """
    Valid active-vision policy: low-resolution spectral-residual saliency.

    Receives a 64×64 downsampled scene preview. Does NOT see the full image.
    Selects the highest-saliency location for the next high-resolution glimpse.
    """

    def __init__(self, scale: float = 0.25, lowres_size: int = LOWRES_SIZE):
        self.scale = scale
        self.lowres_size = lowres_size

    @property
    def metadata(self) -> PolicyMetadata:
        return PolicyMetadata(
            name=f"saliency_lowres_s{self.scale}",
            policy_type=PolicyType.ACTIVE,
            information_available=[f"lowres_preview_{self.lowres_size}px"],
            is_learned=False,
            is_deterministic=True,
            notes=(
                "Spectral-residual saliency on 64px scene downsampling. "
                "No pretrained saliency network. VALID active policy."
            ),
        )

    def reset(self) -> None:
        pass

    def select_next(
        self,
        viewpoint_history: list[Viewpoint],
        model_state: object,
        lowres_preview: Optional[torch.Tensor],
        full_image: Optional[torch.Tensor],
    ) -> Viewpoint:
        if lowres_preview is None:
            raise ValueError(
                "LowResSaliencyPolicy requires lowres_preview. "
                "Caller must provide a downsampled scene image."
            )
        saliency = _image_to_saliency_map(lowres_preview)
        return _saliency_to_viewpoint(saliency, self.scale)
