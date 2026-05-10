"""
Saliency + Inhibition-of-Return policy.

VALID active policy. Combines low-resolution saliency with a spatial penalty
around previously visited locations.

This is an algorithmic approximation motivated by the behavioral principle of
inhibition-of-return (IOR) — not a neural implementation of IOR. The name
reflects the inspiration, not a mechanistic claim.

Priority map:
    priority(x, y) = saliency(x, y) - ior_penalty(x, y)

IOR penalty is a sum of Gaussians centered on prior fixations,
decaying over timesteps.

Parameters MUST NOT be tuned on the final evaluation set.
See docs/experimental_protocol.md for tuning rules.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch

from .base import GazePolicy, PolicyMetadata, PolicyType, Viewpoint
from .saliency_policy import (
    LOWRES_SIZE,
    _image_to_saliency_map,
    _saliency_to_viewpoint,
)


class SaliencyIORPolicy(GazePolicy):
    """
    Saliency with inhibition-of-return spatial penalty.

    ior_radius: spatial penalty radius in normalized [-1, 1] coords
    ior_strength: peak penalty magnitude (fraction of saliency range)
    ior_decay: penalty multiplier per elapsed timestep (1.0 = no decay)
    """

    def __init__(
        self,
        scale: float = 0.25,
        lowres_size: int = LOWRES_SIZE,
        ior_radius: float = 0.3,
        ior_strength: float = 0.8,
        ior_decay: float = 0.85,
    ):
        self.scale = scale
        self.lowres_size = lowres_size
        self.ior_radius = ior_radius
        self.ior_strength = ior_strength
        self.ior_decay = ior_decay

    @property
    def metadata(self) -> PolicyMetadata:
        return PolicyMetadata(
            name=f"saliency_ior_s{self.scale}_r{self.ior_radius}_str{self.ior_strength}",
            policy_type=PolicyType.ACTIVE,
            information_available=[
                f"lowres_preview_{self.lowres_size}px",
                "viewpoint_history",
            ],
            is_learned=False,
            is_deterministic=True,
            notes=(
                f"Spectral-residual saliency + IOR penalty "
                f"(radius={self.ior_radius}, strength={self.ior_strength}, "
                f"decay={self.ior_decay}). VALID active policy."
            ),
        )

    def reset(self) -> None:
        pass

    def _build_ior_penalty(
        self, saliency: np.ndarray, viewpoint_history: list[Viewpoint]
    ) -> np.ndarray:
        """
        Build an IOR penalty map from prior fixation history.

        Each prior fixation contributes a Gaussian penalty. Older fixations
        decay by self.ior_decay per timestep from the current step.
        """
        H, W = saliency.shape
        penalty = np.zeros((H, W), dtype=np.float32)
        n_prior = len(viewpoint_history)

        # Build grid of scene coordinates matching the saliency map
        ys = np.linspace(-1.0, 1.0, H)[:, None]  # (H, 1)
        xs = np.linspace(-1.0, 1.0, W)[None, :]  # (1, W)

        for t, vp in enumerate(viewpoint_history):
            age = n_prior - t  # 1 = most recent, n_prior = oldest
            decay_factor = self.ior_decay ** (age - 1)

            dist_sq = (xs - vp.x) ** 2 + (ys - vp.y) ** 2
            sigma = self.ior_radius / 2.0
            gaussian = np.exp(-dist_sq / (2 * sigma ** 2))
            penalty += self.ior_strength * decay_factor * gaussian

        return penalty

    def select_next(
        self,
        viewpoint_history: list[Viewpoint],
        model_state: object,
        lowres_preview: Optional[torch.Tensor],
        full_image: Optional[torch.Tensor],
    ) -> Viewpoint:
        if lowres_preview is None:
            raise ValueError(
                "SaliencyIORPolicy requires lowres_preview."
            )

        saliency = _image_to_saliency_map(lowres_preview)

        if not viewpoint_history:
            priority = saliency
        else:
            penalty = self._build_ior_penalty(saliency, viewpoint_history)
            priority = saliency - penalty
            # Do not clip — argmax still works on signed values

        return _saliency_to_viewpoint(saliency, self.scale, priority=priority)
