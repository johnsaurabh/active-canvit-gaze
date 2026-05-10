"""
Evaluation metrics for active gaze experiments.

Primary metric: AUGC (Area Under the Accuracy-vs-Glimpses Curve).
Computed per image to enable paired statistical testing.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np


def compute_augc(
    accuracies: Sequence[float],
    glimpse_counts: Sequence[int],
) -> float:
    """
    Compute Area Under the Accuracy-vs-Glimpses Curve (AUGC).

    Uses trapezoidal integration. NOT the same as ROC AUC.

    Args:
        accuracies: top-1 accuracy (0 or 1 per image, or mean accuracy) at each budget.
        glimpse_counts: number of glimpses corresponding to each accuracy value.
                        Must be monotonically increasing.

    Returns:
        AUGC in [0, max_budget] (not normalized). Normalize by dividing by
        (max_glimpses - min_glimpses) if you want a value in [0, 1].
    """
    if len(accuracies) != len(glimpse_counts):
        raise ValueError("accuracies and glimpse_counts must have the same length")
    if len(accuracies) < 2:
        raise ValueError("Need at least 2 points for AUGC")

    accs = list(accuracies)
    counts = list(glimpse_counts)

    augc = 0.0
    for i in range(len(counts) - 1):
        width = counts[i + 1] - counts[i]
        if width < 0:
            raise ValueError("glimpse_counts must be monotonically non-decreasing")
        augc += width * (accs[i] + accs[i + 1]) / 2.0

    return augc


def compute_accuracy_curve(
    per_image_correct_top1: np.ndarray,
    glimpse_counts: Sequence[int],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute mean accuracy and per-image AUGC across a dataset.

    Args:
        per_image_correct_top1: shape (N_images, N_budgets), dtype bool or int.
            per_image_correct_top1[i, t] = 1 if image i is correctly classified at budget t.
        glimpse_counts: budget values, length N_budgets.

    Returns:
        mean_accuracy: (N_budgets,) mean top-1 across images
        per_image_augc: (N_images,) AUGC per image (for paired statistics)
    """
    N, T = per_image_correct_top1.shape
    if T != len(glimpse_counts):
        raise ValueError("Second dimension must match len(glimpse_counts)")

    mean_accuracy = per_image_correct_top1.mean(axis=0)  # (T,)

    per_image_augc = np.array(
        [compute_augc(per_image_correct_top1[i].tolist(), list(glimpse_counts))
         for i in range(N)]
    )

    return mean_accuracy, per_image_augc


def compute_spatial_metrics(
    viewpoint_sequences: list[list[tuple[float, float]]],
) -> dict[str, float]:
    """
    Compute spatial behavior metrics across a set of fixation sequences.

    Args:
        viewpoint_sequences: list of sequences, each sequence is a list of (x, y) pairs
            in scene coordinates [-1, 1]. Index 0 = timestep 0 (full scene), index 1+ = local.

    Returns:
        dict with:
            mean_displacement: mean Euclidean distance between consecutive fixations
            revisit_rate: fraction of fixations within 0.3 scene units of a prior fixation
            mean_coverage: mean fraction of scene covered (approx, by grid)
            mean_center_distance: mean distance of local fixations from (0, 0)
    """
    if not viewpoint_sequences:
        return {}

    displacements = []
    revisit_counts = []
    revisit_totals = []
    center_distances = []

    revisit_radius = 0.3  # in scene units [-1, 1]

    for seq in viewpoint_sequences:
        if len(seq) < 2:
            continue
        # Only local fixations (skip index 0 = full scene)
        local = seq[1:]

        for i, (x, y) in enumerate(local):
            center_distances.append(math.sqrt(x ** 2 + y ** 2))

            if i > 0:
                px, py = local[i - 1]
                displacements.append(math.sqrt((x - px) ** 2 + (y - py) ** 2))

            # Check revisit against all prior local fixations
            prior = local[:i]
            revisit_totals.append(1)
            is_revisit = any(
                math.sqrt((x - px) ** 2 + (y - py) ** 2) < revisit_radius
                for px, py in prior
            )
            revisit_counts.append(1 if is_revisit else 0)

    return {
        "mean_displacement": float(np.mean(displacements)) if displacements else float("nan"),
        "revisit_rate": float(np.sum(revisit_counts) / max(np.sum(revisit_totals), 1)),
        "mean_center_distance": float(np.mean(center_distances)) if center_distances else float("nan"),
    }


def paired_bootstrap_ci(
    values_a: np.ndarray,
    values_b: np.ndarray,
    n_bootstrap: int = 10_000,
    ci: float = 0.95,
    rng_seed: int = 42,
) -> tuple[float, float, float]:
    """
    Paired bootstrap confidence interval for mean(A) - mean(B).

    Args:
        values_a: per-image metric for policy A, shape (N,)
        values_b: per-image metric for policy B, shape (N,)
        n_bootstrap: number of bootstrap samples
        ci: confidence level (e.g. 0.95 for 95% CI)
        rng_seed: for reproducibility

    Returns:
        (observed_diff, ci_lower, ci_upper)
    """
    if len(values_a) != len(values_b):
        raise ValueError("values_a and values_b must have the same length")

    rng = np.random.default_rng(rng_seed)
    diffs = values_a - values_b
    observed_diff = float(np.mean(diffs))

    N = len(diffs)
    bootstrap_means = np.array(
        [np.mean(rng.choice(diffs, size=N, replace=True)) for _ in range(n_bootstrap)]
    )

    alpha = (1.0 - ci) / 2.0
    ci_lower = float(np.percentile(bootstrap_means, 100 * alpha))
    ci_upper = float(np.percentile(bootstrap_means, 100 * (1 - alpha)))

    return observed_diff, ci_lower, ci_upper
