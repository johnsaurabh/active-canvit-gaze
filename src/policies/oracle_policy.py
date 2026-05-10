"""
ORACLE policy: full-resolution saliency.

Uses the full 512px scene image to compute saliency.
This information is unavailable to a real active observer.

ORACLE policies are diagnostic upper bounds only.
They must NEVER be compared to valid policies without being labeled ORACLE.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from .base import GazePolicy, PolicyMetadata, PolicyType, Viewpoint
from .saliency_policy import _image_to_saliency_map, _saliency_to_viewpoint


class FullResSaliencyOracle(GazePolicy):
    """
    ORACLE: spectral-residual saliency on the full-resolution 512px image.

    For diagnostic use only. Answers: how much better could saliency be if
    the policy had full-resolution access?
    """

    def __init__(self, scale: float = 0.25):
        self.scale = scale

    @property
    def metadata(self) -> PolicyMetadata:
        return PolicyMetadata(
            name=f"ORACLE_saliency_fullres_s{self.scale}",
            policy_type=PolicyType.ORACLE,
            information_available=["full_resolution_image_512px"],
            is_learned=False,
            is_deterministic=True,
            notes=(
                "ORACLE: uses full 512px image. Diagnostic upper bound. "
                "Do not compare directly to valid active policies."
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
        if full_image is None:
            raise ValueError(
                "FullResSaliencyOracle requires full_image [1, 3, 512, 512]."
            )
        saliency = _image_to_saliency_map(full_image)
        return _saliency_to_viewpoint(saliency, self.scale)
