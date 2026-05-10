from __future__ import annotations

from typing import Optional

import torch

from .base import GazePolicy, PolicyMetadata, PolicyType, Viewpoint


class CenterPolicy(GazePolicy):
    """
    Baseline: always return the center of the scene at fixed scale.

    Intentionally weak — tests whether just increasing center resolution helps.
    Information used: none.
    """

    def __init__(self, scale: float = 0.25):
        self.scale = scale

    @property
    def metadata(self) -> PolicyMetadata:
        return PolicyMetadata(
            name=f"center_s{self.scale}",
            policy_type=PolicyType.ACTIVE,
            information_available=[],
            is_learned=False,
            is_deterministic=True,
            notes="Always fixates image center; tests pure center-resolution benefit",
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
        return Viewpoint(x=0.0, y=0.0, s=self.scale)
