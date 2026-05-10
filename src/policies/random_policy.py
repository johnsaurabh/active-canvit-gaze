from __future__ import annotations

import random
from typing import Optional

import torch

from .base import GazePolicy, PolicyMetadata, PolicyType, Viewpoint


class RandomPolicy(GazePolicy):
    """
    Baseline: uniformly sample a valid viewpoint center at fixed scale.

    Information used: none (purely random, seeded for reproducibility).
    """

    def __init__(self, scale: float = 0.25, seed: int = 0):
        if not (0.0 < scale <= 1.0):
            raise ValueError(f"scale must be in (0, 1], got {scale}")
        self.scale = scale
        self.seed = seed
        self._rng = random.Random(seed)

    @property
    def metadata(self) -> PolicyMetadata:
        return PolicyMetadata(
            name=f"random_s{self.scale}_seed{self.seed}",
            policy_type=PolicyType.ACTIVE,
            information_available=[],
            is_learned=False,
            is_deterministic=False,
            notes=f"Uniform random sampling, scale={self.scale}, seed={self.seed}",
        )

    def reset(self) -> None:
        self._rng = random.Random(self.seed)

    def select_next(
        self,
        viewpoint_history: list[Viewpoint],
        model_state: object,
        lowres_preview: Optional[torch.Tensor],
        full_image: Optional[torch.Tensor],
    ) -> Viewpoint:
        max_center = 1.0 - self.scale
        x = self._rng.uniform(-max_center, max_center)
        y = self._rng.uniform(-max_center, max_center)
        return Viewpoint(x=x, y=y, s=self.scale)
