"""Unit tests for evaluation metrics."""

import pytest
import numpy as np
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent / "src"))

from evaluation.metrics import (
    compute_augc,
    compute_accuracy_curve,
    compute_spatial_metrics,
    paired_bootstrap_ci,
)


class TestAUGC:
    def test_perfect_accuracy(self):
        accs = [1.0, 1.0, 1.0, 1.0]
        counts = [0, 1, 2, 3]
        # Trapezoid: 3 intervals × 1.0 = 3.0
        assert compute_augc(accs, counts) == pytest.approx(3.0)

    def test_zero_accuracy(self):
        accs = [0.0, 0.0, 0.0]
        counts = [0, 1, 2]
        assert compute_augc(accs, counts) == pytest.approx(0.0)

    def test_rising_accuracy(self):
        accs = [0.0, 0.5, 1.0]
        counts = [0, 1, 2]
        # (0+0.5)/2*1 + (0.5+1.0)/2*1 = 0.25 + 0.75 = 1.0
        assert compute_augc(accs, counts) == pytest.approx(1.0)

    def test_requires_two_points(self):
        with pytest.raises(ValueError):
            compute_augc([1.0], [0])

    def test_requires_matching_lengths(self):
        with pytest.raises(ValueError):
            compute_augc([1.0, 1.0], [0])

    def test_uneven_spacing(self):
        accs = [0.5, 0.7, 0.9]
        counts = [0, 2, 8]
        # (0.5+0.7)/2*2 + (0.7+0.9)/2*6 = 1.2 + 4.8 = 6.0
        assert compute_augc(accs, counts) == pytest.approx(6.0)


class TestAccuracyCurve:
    def test_basic(self):
        per_image = np.array([[1, 1, 1], [0, 1, 1], [0, 0, 1]])
        counts = [0, 1, 2]
        mean_acc, per_image_augc = compute_accuracy_curve(per_image, counts)
        assert mean_acc.shape == (3,)
        assert per_image_augc.shape == (3,)
        assert mean_acc[0] == pytest.approx(1/3)
        assert mean_acc[2] == pytest.approx(1.0)

    def test_per_image_augc_order(self):
        """Image that's correct earlier should have higher AUGC."""
        per_image = np.array([[1, 1, 1], [0, 0, 1]])
        counts = [0, 1, 2]
        _, per_image_augc = compute_accuracy_curve(per_image, counts)
        assert per_image_augc[0] > per_image_augc[1]


class TestSpatialMetrics:
    def test_no_revisit_when_far_apart(self):
        # Two very distant fixations
        seq = [(0.0, 0.0), (0.7, 0.7), (-0.7, -0.7)]
        metrics = compute_spatial_metrics([seq])
        assert metrics["revisit_rate"] == pytest.approx(0.0)

    def test_revisit_when_same_location(self):
        seq = [(0.0, 0.0), (0.0, 0.0), (0.0, 0.0)]
        metrics = compute_spatial_metrics([seq])
        assert metrics["revisit_rate"] > 0.0

    def test_center_distance(self):
        seq = [(0.0, 0.0), (0.0, 0.0)]  # both at center
        metrics = compute_spatial_metrics([seq])
        assert metrics["mean_center_distance"] == pytest.approx(0.0, abs=1e-6)


class TestBootstrapCI:
    def test_no_difference(self):
        rng = np.random.default_rng(42)
        a = rng.normal(0.5, 0.1, 200)
        b = a.copy()  # identical → diff = 0
        diff, lo, hi = paired_bootstrap_ci(a, b, n_bootstrap=1000)
        assert diff == pytest.approx(0.0, abs=1e-10)
        assert lo <= 0.0 <= hi

    def test_clear_positive_difference(self):
        rng = np.random.default_rng(0)
        a = rng.normal(0.8, 0.05, 500)
        b = rng.normal(0.5, 0.05, 500)
        diff, lo, hi = paired_bootstrap_ci(a, b, n_bootstrap=2000)
        assert lo > 0.0  # 95% CI should be entirely positive

    def test_requires_equal_length(self):
        with pytest.raises(ValueError):
            paired_bootstrap_ci(np.array([1.0, 2.0]), np.array([1.0]))
