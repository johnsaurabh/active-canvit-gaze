# Experiment Log

**Project:** Biologically Motivated Active Gaze Policies for CanViT

---

## EXP-000

```
EXP-000
Purpose:    Phase 0 smoke test — sequential CanViT inference
Status:     PASS
Date:       2026-08-26
Device:     Colab GPU
Dataset:    Synthetic noise image (Wikimedia blocked)
N:          1
Policy:     N/A
Result:     5-timestep inference passed all checks. Predictions: wool/maze (expected on noise).
Notes:      scales tensor must be 1D [1], not 2D [1,1] — fixed in base.py and notebook.
```

---

## EXP-001

```
EXP-001
Purpose:    Phase 1 — CanViT reproduction on ImageNette (random baseline)
Status:     PASS
Date:       2026-08-26
Device:     Colab GPU
Dataset:    ImageNette val, 100-image subset (seed=42). NOT official ImageNet-1k.
N:          100
Policy:     Random (seed=0)
Glimpse scale: 0.25
Checkpoint: canvit/canvitb16-add-vpe-finetune-g128px-s512px-in1k-2026-04-06
Result:     Top-1 T=0=0.780, Top-1 T=8=0.830. Accuracy improves with glimpses.
Notes:      Pipeline verified end-to-end. Results are development-only (ImageNette, not ImageNet-1k).
            Proceed to Phase 2 — policy comparison.
```

---

## EXP-002

```
EXP-002
Purpose:    Phase 2 — 7-policy comparison on ImageNette dev subset
Status:     COMPLETE
Date:       2026-08-26
Device:     Colab GPU
Dataset:    ImageNette val, 100-image subset (seed=42). NOT official ImageNet-1k.
N:          100
Policies:   random, center, coverage, saliency_lowres, saliency_ior,
            inverse_saliency, ORACLE_saliency_fullres
Checkpoint: canvit/canvitb16-add-vpe-finetune-g128px-s512px-in1k-2026-04-06

AUGC ranking (valid policies):
  1. random          6.5450
  2. center          6.5100  (uncertain vs random)
  3. inverse_saliency 6.3550 (significantly worse, NO)
  4. saliency_lowres  6.3350 (uncertain)
  5. saliency_ior     6.3350 (significantly worse, NO)
  6. coverage         6.3100 (significantly worse, NO)
  ORACLE:            6.3800 (diagnostic only)

Key spatial finding:
  saliency_lowres, inverse_saliency, ORACLE all have mean_displacement=0.000
  and revisit_rate=0.875. They fixate the same pixel all 8 timesteps.
  Saliency without IOR is a static fixation policy — no temporal diversity.
  Random wins purely through spatial coverage, not intelligence.
  Coverage overshoots to extreme periphery (mean_center_dist=0.906), missing objects.

Interpretation:
  Spatial diversity matters more than saliency content on this task/dataset.
  Saliency_ior (displacement=0.907) is the only saliency policy that moves,
  but IOR parameters are too weak. This is a tuning problem, not a concept failure.

Notes:
  - N=100 CIs are wide; treat as directional only.
  - scipy incompatibility with numpy 2.x fixed (replaced with pure numpy box filter).
  - Next: tune IOR strength, scale to full ImageNette val (~3900 images).
```

---

## EXP-003

```
EXP-003
Purpose:    Phase 3 — Full ImageNette val policy comparison (scaled from EXP-002)
Status:     COMPLETE
Date:       2026-08-26
Device:     Colab GPU
Dataset:    ImageNette full val (~3900 images). NOT official ImageNet-1k.
N:          ~3900
Policies:   random, center, coverage, saliency_lowres, saliency_ior,
            saliency_ior_tuned (strength=2.0, radius=0.4), inverse_saliency,
            ORACLE_saliency_fullres
Checkpoint: canvit/canvitb16-add-vpe-finetune-g128px-s512px-in1k-2026-04-06
Metric:     AUGC over GLIMPSE_BUDGETS = [0,1,2,3,4,5,6,8]

AUGC results (ranked):
  1. random                6.7571  (baseline)
  2. center                6.7302  diff=-0.027  CI=[-0.062, +0.007]  uncertain
  3. saliency_ior          6.7096  diff=-0.048  CI=[-0.081, -0.014]  NO
  4. saliency_ior_tuned    6.7008  diff=-0.056  CI=[-0.090, -0.022]  NO
  5. coverage              6.6982  diff=-0.059  CI=[-0.094, -0.023]  NO
  6. saliency_lowres       6.6386  diff=-0.118  CI=[-0.156, -0.080]  NO
  7. inverse_saliency      6.6255  diff=-0.132  CI=[-0.173, -0.091]  NO
  ORACLE: 6.6632 (diagnostic only — also loses to random)

Hypothesis outcomes:
  H1 (saliency > random): FALSIFIED. saliency_lowres diff=-0.118, CI entirely negative.
  H2 (IOR reduces revisits): PARTIAL. IOR reduces revisit rate (0.405→0.138 tuned)
     but does not improve AUGC vs random. Revisit reduction ≠ accuracy improvement.
  H3 (benefit largest at small budgets): N/A. No policy benefited at any budget.
  H4 (coverage matches saliency): CONFIRMED in reverse. Coverage (6.698) >
     saliency_lowres (6.639). Spatial spread outperforms content targeting.
  H5 (optimal ≠ human scanpaths): Not yet tested.
  Secondary (ORACLE > lowres saliency): CONFIRMED. 6.663 > 6.639. But both
     lose to random, making the gap scientifically secondary.

Key finding:
  Random sampling outperforms all tested policies including ORACLE (full-res saliency).
  Spectral residual saliency does not identify classification-relevant regions for
  ImageNet-class objects. Spatial diversity beats content targeting on this task/dataset.

Mechanistic explanation:
  Saliency policies without IOR have zero displacement (same spot all 8 timesteps).
  With IOR, displacement increases (0.907→1.099) but AUGC does not improve —
  the policy explores more but still lands in uninformative regions.
  Coverage explores maximally (displacement 1.492) but overshoots to periphery
  (mean center distance 0.906), missing centered objects.

Notes:
  Results are on ImageNette (development dataset), not ImageNet-1k.
  Next: discuss with Prof. Krishna. Key questions:
  (1) Would task-driven or uncertainty-driven policies beat random?
  (2) Would human gaze data on recognition tasks improve over random?
  (3) Would results differ on a harder dataset with cluttered/small objects?
```

---

_Add entries here after each experiment run. Never overwrite previous entries._
