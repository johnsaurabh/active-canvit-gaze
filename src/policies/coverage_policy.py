from __future__ import annotations

import math
from typing import Optional

import torch

from .base import GazePolicy, PolicyMetadata, PolicyType, Viewpoint


class CoveragePolicy(GazePolicy):
    """
    Baseline: maximize minimum distance from all prior fixation centers.

    Deterministic spatial exploration. Important baseline: tests whether
    any saliency advantage is merely due to avoiding spatial redundancy.

    Information used: viewpoint_history (prior (x, y) coordinates only).
    No image content.
    """

    def __init__(self, scale: float = 0.25, n_candidates: int = 200):
        self.scale = scale
        self.n_candidates = n_candidates  # candidate locations sampled per step

    @property
    def metadata(self) -> PolicyMetadata:
        return PolicyMetadata(
            name=f"coverage_s{self.scale}",
            policy_type=PolicyType.ACTIVE,
            information_available=["viewpoint_history"],
            is_learned=False,
            is_deterministic=True,
            notes=(
                "Maximizes minimum distance from prior fixations. "
                "Critical baseline for spatial-coverage confound."
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
        max_center = 1.0 - self.scale

        if not viewpoint_history:
            # No prior fixations: pick center
            return Viewpoint(x=0.0, y=0.0, s=self.scale)

        prior_centers = [(vp.x, vp.y) for vp in viewpoint_history]

        # Sample candidates uniformly from valid region
        # Deterministic: use a fixed grid for reproducibility
        n = int(math.sqrt(self.n_candidates))
        best_vp = None
        best_min_dist = -1.0

        for i in range(n):
            for j in range(n):
                x = -max_center + (2 * max_center * i / (n - 1)) if n > 1 else 0.0
                y = -max_center + (2 * max_center * j / (n - 1)) if n > 1 else 0.0
                min_dist = min(
                    math.sqrt((x - px) ** 2 + (y - py) ** 2)
                    for px, py in prior_centers
                )
                if min_dist > best_min_dist:
                    best_min_dist = min_dist
                    best_vp = Viewpoint(x=x, y=y, s=self.scale)

        return best_vp
