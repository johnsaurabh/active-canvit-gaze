"""
Negative control: inverse saliency.

Deliberately targets the LEAST salient regions.
If this performs similarly to sensible policies, the evaluation pipeline
is not sensitive enough to distinguish strategies.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from .base import GazePolicy, PolicyMetadata, PolicyType, Viewpoint
from .saliency_policy import (
    LOWRES_SIZE,
    _image_to_saliency_map,
    _saliency_to_viewpoint,
)


class InverseSaliencyPolicy(GazePolicy):
    """
    Negative control: targets the minimum saliency location.
    """

    def __init__(self, scale: float = 0.25, lowres_size: int = LOWRES_SIZE):
        self.scale = scale
        self.lowres_size = lowres_size

    @property
    def metadata(self) -> PolicyMetadata:
        return PolicyMetadata(
            name=f"inverse_saliency_s{self.scale}",
            policy_type=PolicyType.ACTIVE,
            information_available=[f"lowres_preview_{self.lowres_size}px"],
            is_learned=False,
            is_deterministic=True,
            notes="Negative control: targets lowest-saliency regions. Should perform poorly.",
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
            raise ValueError("InverseSaliencyPolicy requires lowres_preview.")
        saliency = _image_to_saliency_map(lowres_preview)
        # Invert: target minimum saliency
        return _saliency_to_viewpoint(1.0 - saliency, self.scale)
