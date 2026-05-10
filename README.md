# Gaze-Selection Policies for CanViT

Do biologically motivated gaze policies allow an active vision model to acquire task-relevant information more efficiently than random sampling? This project tests that question using CanViT (Berreby et al., 2026), a policy-agnostic active vision model that classifies images from sequential glimpses.

---

## Background

Standard vision models process the full image at once. Active vision systems instead select a sequence of fixations, each revealing a local high-resolution crop. CanViT integrates these glimpses into a persistent canvas representation via cross-attention, without being tied to any particular gaze strategy. This makes it a natural platform for evaluating gaze policies independently of model architecture.

---

## Research questions

1. Can biologically motivated gaze-selection policies (saliency, inhibition of return) outperform random sampling on an active classification task?
2. Do policies that perform well exhibit spatial behaviour consistent with human goal-directed gaze?

---

## Policies

| Policy | Type | Information used |
|--------|------|-----------------|
| `random` | Valid | None (seeded uniform sampling) |
| `center` | Valid | None (always fixates centre) |
| `coverage` | Valid | Viewpoint history |
| `saliency_lowres` | Valid | 64px scene downsampling |
| `saliency_ior` | Valid | 64px preview + viewpoint history |
| `inverse_saliency` | Valid (negative control) | 64px preview |
| `ORACLE_saliency_fullres` | Oracle | Full 512px image |

Oracle policies are diagnostic upper bounds only and are not compared to valid policies.

---

## Results (ImageNette val, N = 3,925)

Random sampling outperformed all tested policies, including the full-resolution oracle. Five of six valid policies were significantly worse than random (paired bootstrap 95% CI, N = 10,000 samples).

| Policy | AUGC | vs random | Result |
|--------|------|-----------|--------|
| random | 6.757 | -- | baseline |
| center | 6.730 | -0.027 | uncertain |
| saliency_ior | 6.710 | -0.048 | NO |
| saliency_ior_tuned | 6.701 | -0.056 | NO |
| coverage | 6.698 | -0.059 | NO |
| saliency_lowres | 6.639 | -0.118 | NO |
| inverse_saliency | 6.625 | -0.132 | NO |
| ORACLE | 6.663 | N/A | diagnostic |

Saliency-based policies without inhibition of return fixated the same location at every timestep (zero mean displacement, revisit rate 0.875). Stronger IOR increased spatial diversity but did not improve accuracy, indicating that spectral residual saliency does not identify classification-relevant regions on this dataset.

Full write-up: [`docs/paper.pdf`](docs/paper.pdf)

---

## Model and dataset

- **Model:** CanViT-B finetuned on ImageNet-1k (`canvit/canvitb16-add-vpe-finetune-g128px-s512px-in1k-2026-04-06`)
- **Dataset:** ImageNette val (development). Full ImageNet-1k evaluation is future work.
- **Sequence:** T=0 full-scene glimpse, T=1-8 policy-selected local glimpses (s=0.25)
- **Metric:** AUGC — Area Under the Accuracy-vs-Glimpses Curve

---

## Repository structure

```
active-canvit-gaze/
├── src/
│   ├── policies/       # Policy implementations
│   ├── evaluation/     # AUGC, bootstrap CI, spatial metrics
│   └── data/           # Dataset loader
├── notebooks/          # Colab notebooks (one per phase)
├── docs/               # Hypotheses, protocol, findings, paper
├── tests/              # Unit tests
└── scripts/            # Utility scripts
```

---

## Running

**Unit tests (no GPU required):**
```bash
pip install numpy torch
pytest tests/ -v
```

**Experiments (Colab GPU):**
```
notebooks/phase0_environment_and_smoke_test.ipynb
notebooks/phase1_canvit_reproduction.ipynb
notebooks/phase2_policy_comparison.ipynb
notebooks/phase3_full_imagenette_comparison.ipynb
```

---

## References

- Berreby, F., Du, M., Durand, T., and Krishna, B. S. (2026). CanViT. arXiv:2603.22570.
- Hou, X. and Zhang, L. (2007). Saliency detection: A spectral residual approach. CVPR.
- Posner, M. I. and Cohen, Y. (1984). Components of visual orienting. Attention and Performance X.

---

## License

MIT
