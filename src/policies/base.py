"""
Policy interface for CanViT active gaze selection.

Every policy is called once per timestep and returns the next viewpoint.
Policies must not access the full-resolution image unless explicitly labeled ORACLE.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import torch


class PolicyType(str, Enum):
    ACTIVE = "active"   # uses only information available to a real active observer
    ORACLE = "oracle"   # uses information unavailable to a real active observer


@dataclass(frozen=True)
class PolicyMetadata:
    name: str
    policy_type: PolicyType
    information_available: list[str]  # e.g. ["lowres_preview", "viewpoint_history"]
    is_learned: bool
    is_deterministic: bool
    notes: str = ""


@dataclass
class Viewpoint:
    """
    CanViT viewpoint: (x, y) ∈ [-1, +1]², scale s ∈ (0, 1].

    x, y: center of the glimpse in scene coordinates
    s:    half-side-length of the glimpse (fraction of scene)

    Example:
        Viewpoint(x=0.0, y=0.0, s=1.0)   # full scene
        Viewpoint(x=-0.5, y=-0.5, s=0.25) # top-left quadrant, zoomed in
    """
    x: float
    y: float
    s: float

    X_RANGE: tuple[float, float] = field(default=(-1.0, 1.0), init=False, repr=False)
    S_RANGE: tuple[float, float] = field(default=(0.05, 1.0), init=False, repr=False)

    def __post_init__(self):
        if not (-1.0 <= self.x <= 1.0):
            raise ValueError(f"x={self.x} out of range [-1, 1]")
        if not (-1.0 <= self.y <= 1.0):
            raise ValueError(f"y={self.y} out of range [-1, 1]")
        if not (0.0 < self.s <= 1.0):
            raise ValueError(f"s={self.s} out of range (0, 1]")
        # Ensure the glimpse window stays within scene bounds
        if abs(self.x) + self.s > 1.0 + 1e-6:
            raise ValueError(
                f"Glimpse extends outside scene: x={self.x}, s={self.s}, "
                f"|x|+s={abs(self.x) + self.s:.3f} > 1.0"
            )
        if abs(self.y) + self.s > 1.0 + 1e-6:
            raise ValueError(
                f"Glimpse extends outside scene: y={self.y}, s={self.s}, "
                f"|y|+s={abs(self.y) + self.s:.3f} > 1.0"
            )

    def to_canvit(self, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        """Convert to CanViT Viewpoint tensors: centers [1, 2] and scales [1] (1D)."""
        centers = torch.tensor([[self.x, self.y]], dtype=torch.float32, device=device)
        scales = torch.tensor([self.s], dtype=torch.float32, device=device)
        return centers, scales

    @staticmethod
    def full_scene() -> "Viewpoint":
        return Viewpoint(x=0.0, y=0.0, s=1.0)

    @staticmethod
    def clamp(x: float, y: float, s: float) -> "Viewpoint":
        """Create a valid viewpoint by clamping x, y so the window stays in-scene."""
        s = max(0.05, min(1.0, s))
        max_center = 1.0 - s
        x = max(-max_center, min(max_center, x))
        y = max(-max_center, min(max_center, y))
        return Viewpoint(x=x, y=y, s=s)


class GazePolicy(abc.ABC):
    """
    Abstract base for all gaze policies.

    Subclasses implement `select_next()`.
    Policies must NOT:
      - access `full_image` unless `metadata.policy_type == PolicyType.ORACLE`
      - access ground-truth labels
      - access segmentation masks
    """

    @property
    @abc.abstractmethod
    def metadata(self) -> PolicyMetadata:
        ...

    @abc.abstractmethod
    def reset(self) -> None:
        """Reset internal state for a new image."""
        ...

    @abc.abstractmethod
    def select_next(
        self,
        viewpoint_history: list[Viewpoint],
        model_state: object,
        lowres_preview: Optional[torch.Tensor],
        full_image: Optional[torch.Tensor],
    ) -> Viewpoint:
        """
        Select the next viewpoint.

        Args:
            viewpoint_history: List of viewpoints selected so far (index 0 = timestep 0).
            model_state: Current CanViT canvas state (after processing viewpoint_history[-1]).
            lowres_preview: Downsampled full-scene image [1, 3, H_lr, W_lr] (valid policies only).
            full_image: Full-resolution image [1, 3, 512, 512] (ORACLE policies only).

        Returns:
            Next Viewpoint. Must be valid (within scene bounds).
        """
        ...

    def __repr__(self) -> str:
        m = self.metadata
        return f"{self.__class__.__name__}(type={m.policy_type.value})"
