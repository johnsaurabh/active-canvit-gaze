"""
Unit tests for policy interface and all policies.
Run with: pytest tests/test_policies.py -v
"""

import math
import pytest
import torch
import numpy as np

import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent / "src"))

from policies.base import GazePolicy, PolicyType, Viewpoint
from policies.random_policy import RandomPolicy
from policies.center_policy import CenterPolicy
from policies.coverage_policy import CoveragePolicy
from policies.saliency_policy import LowResSaliencyPolicy, _image_to_saliency_map
from policies.saliency_ior_policy import SaliencyIORPolicy
from policies.oracle_policy import FullResSaliencyOracle
from policies.negative_control import InverseSaliencyPolicy


# ── Viewpoint ──────────────────────────────────────────────────────────────────

class TestViewpoint:
    def test_full_scene(self):
        vp = Viewpoint.full_scene()
        assert vp.x == 0.0 and vp.y == 0.0 and vp.s == 1.0

    def test_valid_local(self):
        vp = Viewpoint(x=-0.5, y=-0.5, s=0.25)
        assert vp.x == -0.5

    def test_rejects_out_of_range_x(self):
        with pytest.raises(ValueError):
            Viewpoint(x=1.5, y=0.0, s=0.25)

    def test_rejects_out_of_range_s(self):
        with pytest.raises(ValueError):
            Viewpoint(x=0.0, y=0.0, s=0.0)

    def test_rejects_window_outside_scene(self):
        # x=0.8, s=0.25 → |x|+s = 1.05 > 1.0
        with pytest.raises(ValueError):
            Viewpoint(x=0.8, y=0.0, s=0.25)

    def test_clamp_stays_valid(self):
        vp = Viewpoint.clamp(x=0.9, y=0.9, s=0.25)
        assert abs(vp.x) + vp.s <= 1.0 + 1e-6
        assert abs(vp.y) + vp.s <= 1.0 + 1e-6

    def test_to_canvit(self):
        vp = Viewpoint(x=-0.5, y=0.3, s=0.25)
        centers, scales = vp.to_canvit(torch.device("cpu"))
        assert centers.shape == (1, 2)
        assert scales.shape == (1, 1)
        assert float(centers[0, 0]) == pytest.approx(-0.5)
        assert float(scales[0, 0]) == pytest.approx(0.25)


# ── Random policy ──────────────────────────────────────────────────────────────

class TestRandomPolicy:
    def test_metadata_type(self):
        pol = RandomPolicy(scale=0.25, seed=0)
        assert pol.metadata.policy_type == PolicyType.ACTIVE

    def test_output_valid(self):
        pol = RandomPolicy(scale=0.25, seed=0)
        pol.reset()
        for _ in range(50):
            vp = pol.select_next([], None, None, None)
            assert abs(vp.x) + vp.s <= 1.0 + 1e-6
            assert abs(vp.y) + vp.s <= 1.0 + 1e-6

    def test_seeded_reproducibility(self):
        pol_a = RandomPolicy(scale=0.25, seed=7)
        pol_b = RandomPolicy(scale=0.25, seed=7)
        pol_a.reset(); pol_b.reset()
        for _ in range(10):
            vp_a = pol_a.select_next([], None, None, None)
            vp_b = pol_b.select_next([], None, None, None)
            assert vp_a.x == pytest.approx(vp_b.x)
            assert vp_a.y == pytest.approx(vp_b.y)

    def test_different_seeds_differ(self):
        pol_a = RandomPolicy(seed=0)
        pol_b = RandomPolicy(seed=99)
        pol_a.reset(); pol_b.reset()
        vp_a = pol_a.select_next([], None, None, None)
        vp_b = pol_b.select_next([], None, None, None)
        assert vp_a.x != pytest.approx(vp_b.x) or vp_a.y != pytest.approx(vp_b.y)

    def test_reset_restarts_sequence(self):
        pol = RandomPolicy(seed=0)
        pol.reset()
        vp1 = pol.select_next([], None, None, None)
        pol.reset()
        vp2 = pol.select_next([], None, None, None)
        assert vp1.x == pytest.approx(vp2.x)


# ── Center policy ──────────────────────────────────────────────────────────────

class TestCenterPolicy:
    def test_always_center(self):
        pol = CenterPolicy(scale=0.25)
        for _ in range(5):
            vp = pol.select_next([], None, None, None)
            assert vp.x == pytest.approx(0.0)
            assert vp.y == pytest.approx(0.0)

    def test_deterministic(self):
        pol = CenterPolicy(scale=0.25)
        vp1 = pol.select_next([], None, None, None)
        vp2 = pol.select_next([], None, None, None)
        assert vp1.x == vp2.x and vp1.y == vp2.y


# ── Coverage policy ────────────────────────────────────────────────────────────

class TestCoveragePolicy:
    def test_moves_away_from_prior(self):
        pol = CoveragePolicy(scale=0.25)
        prior = [Viewpoint(x=0.0, y=0.0, s=0.25)]
        vp = pol.select_next(prior, None, None, None)
        dist = math.sqrt(vp.x ** 2 + vp.y ** 2)
        # Should be far from center (the only prior fixation)
        assert dist > 0.3

    def test_valid_output(self):
        pol = CoveragePolicy(scale=0.25)
        history = [Viewpoint(x=0.0, y=0.0, s=0.25), Viewpoint(x=-0.5, y=-0.5, s=0.25)]
        vp = pol.select_next(history, None, None, None)
        assert abs(vp.x) + vp.s <= 1.0 + 1e-6


# ── Saliency policy ────────────────────────────────────────────────────────────

def _make_lowres(bright_x: float = 0.5, bright_y: float = 0.5) -> torch.Tensor:
    """Create a 64x64 image with a bright spot at (bright_x, bright_y) in [0,1] coords."""
    img = torch.zeros(1, 3, 64, 64)
    c = int(bright_x * 63)
    r = int(bright_y * 63)
    # Make a bright patch
    r0, r1 = max(0, r - 4), min(64, r + 5)
    c0, c1 = max(0, c - 4), min(64, c + 5)
    img[:, :, r0:r1, c0:c1] = 1.0
    return img


class TestSaliencyPolicy:
    def test_metadata_active(self):
        pol = LowResSaliencyPolicy()
        assert pol.metadata.policy_type == PolicyType.ACTIVE

    def test_requires_lowres(self):
        pol = LowResSaliencyPolicy()
        with pytest.raises(ValueError):
            pol.select_next([], None, lowres_preview=None, full_image=None)

    def test_output_valid(self):
        pol = LowResSaliencyPolicy(scale=0.25)
        lr = _make_lowres(0.5, 0.5)
        vp = pol.select_next([], None, lowres_preview=lr, full_image=None)
        assert abs(vp.x) + vp.s <= 1.0 + 1e-6

    def test_bright_patch_attracts_fixation(self):
        """A bright patch at top-left should draw fixation to top-left quadrant."""
        pol = LowResSaliencyPolicy(scale=0.25)
        lr = _make_lowres(bright_x=0.1, bright_y=0.1)
        vp = pol.select_next([], None, lowres_preview=lr, full_image=None)
        # Expect negative x (left) and negative y (top in scene coords)
        assert vp.x < 0.0
        assert vp.y < 0.0

    def test_deterministic(self):
        pol = LowResSaliencyPolicy()
        lr = _make_lowres()
        vp1 = pol.select_next([], None, lowres_preview=lr, full_image=None)
        vp2 = pol.select_next([], None, lowres_preview=lr, full_image=None)
        assert vp1.x == vp2.x and vp1.y == vp2.y


# ── IOR policy ─────────────────────────────────────────────────────────────────

class TestSaliencyIORPolicy:
    def test_ior_suppresses_visited(self):
        """
        With a bright patch in the left-center and a prior fixation there,
        IOR should push the next fixation away from that region.

        We use bright_x=0.35, bright_y=0.35 (not an extreme corner) so that
        Viewpoint.clamp does not map both the original and the IOR-shifted peak
        to the same boundary coordinate.
        """
        saliency_pol = LowResSaliencyPolicy(scale=0.25)
        # Use high ior_strength to ensure the penalty decisively overrides the peak
        ior_pol = SaliencyIORPolicy(scale=0.25, ior_strength=5.0, ior_radius=0.5)

        lr = _make_lowres(bright_x=0.35, bright_y=0.35)

        # Without IOR, should go to left-center quadrant
        vp_no_ior = saliency_pol.select_next([], None, lowres_preview=lr, full_image=None)

        # Place prior fixation at the saliency peak
        prior = [Viewpoint.clamp(vp_no_ior.x, vp_no_ior.y, 0.25)]
        vp_ior = ior_pol.select_next(prior, None, lowres_preview=lr, full_image=None)

        dist = math.sqrt((vp_ior.x - vp_no_ior.x) ** 2 + (vp_ior.y - vp_no_ior.y) ** 2)
        assert dist > 0.1, f"IOR should move fixation away; dist={dist:.3f}"

    def test_zero_penalty_matches_saliency(self):
        """ior_strength=0 should reproduce the saliency result exactly."""
        saliency_pol = LowResSaliencyPolicy(scale=0.25)
        ior_pol = SaliencyIORPolicy(scale=0.25, ior_strength=0.0)

        lr = _make_lowres(bright_x=0.7, bright_y=0.3)
        vp_sal = saliency_pol.select_next([], None, lowres_preview=lr, full_image=None)
        vp_ior = ior_pol.select_next([], None, lowres_preview=lr, full_image=None)

        assert vp_sal.x == pytest.approx(vp_ior.x, abs=0.1)
        assert vp_sal.y == pytest.approx(vp_ior.y, abs=0.1)

    def test_output_valid(self):
        pol = SaliencyIORPolicy(scale=0.25)
        lr = _make_lowres()
        vp = pol.select_next([], None, lowres_preview=lr, full_image=None)
        assert abs(vp.x) + vp.s <= 1.0 + 1e-6


# ── Oracle policy ──────────────────────────────────────────────────────────────

class TestOraclePolicy:
    def test_metadata_oracle(self):
        pol = FullResSaliencyOracle()
        assert pol.metadata.policy_type == PolicyType.ORACLE

    def test_requires_full_image(self):
        pol = FullResSaliencyOracle()
        with pytest.raises(ValueError):
            pol.select_next([], None, lowres_preview=None, full_image=None)

    def test_uses_full_image(self):
        pol = FullResSaliencyOracle(scale=0.25)
        full = _make_lowres()  # any 64px image works for the saliency computation
        full_512 = torch.nn.functional.interpolate(full, size=(512, 512), mode="bilinear")
        vp = pol.select_next([], None, lowres_preview=None, full_image=full_512)
        assert abs(vp.x) + vp.s <= 1.0 + 1e-6


# ── Negative control ───────────────────────────────────────────────────────────

class TestInverseSaliencyPolicy:
    def test_goes_opposite_to_saliency(self):
        """Bright patch at top-left → saliency goes there; inverse should go elsewhere."""
        sal_pol = LowResSaliencyPolicy(scale=0.25)
        inv_pol = InverseSaliencyPolicy(scale=0.25)

        lr = _make_lowres(bright_x=0.1, bright_y=0.1)
        vp_sal = sal_pol.select_next([], None, lowres_preview=lr, full_image=None)
        vp_inv = inv_pol.select_next([], None, lowres_preview=lr, full_image=None)

        dist = math.sqrt((vp_sal.x - vp_inv.x) ** 2 + (vp_sal.y - vp_inv.y) ** 2)
        assert dist > 0.2, f"Inverse should differ from saliency; dist={dist:.3f}"
