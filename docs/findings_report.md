# Findings Report — Biologically Motivated Gaze Policies for CanViT

**Date:** 2026-08-26  
**Dataset:** ImageNette val (development only — NOT ImageNet-1k)  
**Model:** CanViT-B finetuned on ImageNet-1k (`canvit/canvitb16-add-vpe-finetune-g128px-s512px-in1k-2026-04-06`)  
**Status:** Development results. Not final. Not suitable for publication without further validation on ImageNet-1k.

---

## 1. What we did

We compared seven gaze-selection policies on CanViT — a policy-agnostic active vision model that integrates sequential glimpses into a persistent canvas representation. The question: can biologically motivated gaze-selection policies allow CanViT to acquire task-relevant visual information more efficiently than random sampling?

**Experimental setup:**
- T=0: full-scene glimpse (identical for all policies)
- T=1–8: policy-selected local glimpses at scale s=0.25 (128px crop from 512px scene)
- Metric: AUGC — Area Under the Accuracy-vs-Glimpses Curve over T ∈ {0,1,2,3,4,5,6,8}
- Statistics: paired bootstrap CI (10,000 samples) vs random baseline
- Dataset: ImageNette full val, ~3,900 images

---

## 2. Policies compared

| Policy | Type | Information used |
|--------|------|-----------------|
| `random` | Valid active | None (seeded uniform random) |
| `center` | Valid active | None (always fixates centre) |
| `coverage` | Valid active | Viewpoint history only |
| `saliency_lowres` | Valid active | 64px scene downsampling |
| `saliency_ior` | Valid active | 64px preview + viewpoint history |
| `saliency_ior_tuned` | Valid active | Same, stronger IOR (strength=2.0, radius=0.4) |
| `inverse_saliency` | Valid active (negative control) | 64px preview |
| `ORACLE_saliency_fullres` | ORACLE | Full 512px image |

Saliency method: spectral residual (Hou & Zhang, 2007) — log amplitude minus smoothed log amplitude in frequency domain. No pretrained networks. Computes a bottom-up visual novelty map.

---

## 3. Results

### 3.1 AUGC ranking

| Rank | Policy | AUGC | Diff vs random | 95% CI | Conclusion |
|------|--------|------|----------------|--------|------------|
| 1 | random | 6.7571 | — | — | baseline |
| 2 | center | 6.7302 | −0.027 | [−0.062, +0.007] | uncertain |
| 3 | saliency_ior | 6.7096 | −0.048 | [−0.081, −0.014] | **NO** |
| 4 | saliency_ior_tuned | 6.7008 | −0.056 | [−0.090, −0.022] | **NO** |
| 5 | coverage | 6.6982 | −0.059 | [−0.094, −0.023] | **NO** |
| 6 | saliency_lowres | 6.6386 | −0.118 | [−0.156, −0.080] | **NO** |
| 7 | inverse_saliency | 6.6255 | −0.132 | [−0.173, −0.091] | **NO** |
| — | ORACLE | 6.6632 | — | — | diagnostic only |

**Random outperforms all tested policies. The ORACLE (full 512px saliency) also loses to random.**

### 3.2 Accuracy-vs-Glimpses

All policies start at T=0 accuracy of ~0.780 (identical full-scene glimpse). Random reaches ~0.830 at T=8. No other policy exceeds this.

### 3.3 Spatial behaviour metrics

| Policy | Mean displacement | Revisit rate | Mean center distance |
|--------|------------------|--------------|---------------------|
| random | 0.697 | 0.375 | 0.419 |
| center | 0.000 | 0.875 | 0.000 |
| coverage | 1.492 | 0.000 | 0.906 |
| saliency_lowres | 0.000 | 0.875 | 0.641 |
| saliency_ior | 0.907 | 0.405 | 0.724 |
| saliency_ior_tuned | 1.099 | 0.138 | — |
| inverse_saliency | 0.000 | 0.875 | 0.804 |
| ORACLE | 0.000 | 0.875 | 0.741 |

---

## 4. Mechanistic explanation of why random wins

### 4.1 Saliency policies fixate the same spot repeatedly

`saliency_lowres`, `inverse_saliency`, and `ORACLE` all show **zero mean displacement** and revisit rate 0.875. They compute argmax of a static saliency map and return the same pixel at every timestep — equivalent to a broken center policy located off-center. The model receives 8 near-identical glimpses from the same location. No new information is added after T=1.

### 4.2 IOR helps mobility but not accuracy

`saliency_ior` reduces revisit rate (0.875 → 0.405) and increases displacement (0.000 → 0.907). The tuned version (strength=2.0) further reduces revisits (0.138) and increases displacement (1.099). Despite this, AUGC does not improve — the policy moves to more locations but those locations are not classification-relevant. Spectral residual saliency directs the policy to visually unusual textures, not to object-discriminative features.

### 4.3 Coverage explores maximally but goes to the wrong places

`coverage` achieves the highest displacement (1.492) and zero revisit rate — it never revisits a location. However, mean center distance is 0.906, meaning it consistently fixates at the extreme periphery of the image. ImageNette objects are typically centered or large, so peripheral fixations carry little discriminative information.

### 4.4 Random hits a useful middle ground

Random sampling achieves moderate displacement (0.697) and moderate revisit rate (0.375). Without trying, it distributes glimpses in a way that provides diverse, non-redundant scene coverage without going to uninformative extremes. On ImageNette's large, centered objects, this turns out to be optimal among the tested policies.

---

## 5. Hypothesis outcomes

| Hypothesis | Prediction | Outcome |
|-----------|------------|---------|
| H1 | Saliency > random | **FALSIFIED.** saliency_lowres AUGC diff = −0.118, CI entirely negative |
| H2 | IOR reduces revisits AND improves efficiency | **PARTIAL.** IOR reduces revisits (confirmed). Does not improve AUGC (falsified). |
| H3 | Policy benefit largest at small budgets | **N/A.** No policy showed benefit at any budget. |
| H4 | Coverage may match saliency | **CONFIRMED** (in reverse). Coverage > saliency_lowres. Spatial spread beats content targeting. |
| H5 | Optimal policies ≠ human scanpaths | Not yet tested. Requires human gaze data. |
| Secondary | ORACLE > lowres saliency | **CONFIRMED.** 6.663 > 6.639. Gap small; both lose to random. |

---

## 6. What this tells us

### What it rules out

Spectral residual bottom-up saliency — as a gaze policy for CanViT on ImageNet-class recognition — does not improve over random. This holds even at full resolution (ORACLE). The failure is not a resolution problem; it is a signal problem. Spectral residual finds visually unusual regions, not classification-relevant ones.

### What it does not rule out

1. **Task-driven saliency** — saliency conditioned on the model's current hypothesis (e.g. GradCAM, class activation maps). This targets discriminative features, not visual novelty.
2. **Uncertainty-driven policies** — fixating where the model's canvas is most uncertain. Uses the model's internal state to select the next glimpse.
3. **Harder datasets** — on cluttered scenes with small or occluded objects, where the object does not dominate the frame, WHERE you look would matter more.
4. **Learned policies** — end-to-end trained to maximise accuracy over the glimpse sequence, as the CanViT paper anticipates.
5. **Human gaze-derived policies** — using actual fixation sequences from humans performing recognition tasks.

### What the ORACLE result tells us specifically

The ORACLE uses full-resolution spectral residual saliency and still loses to random. This is the strongest result: **the bottleneck is the saliency signal itself, not the resolution at which it is computed.** Any improvement in resolution (64px → 512px) is irrelevant if the saliency method does not identify classification-relevant regions.

---

## 7. Open questions for discussion with Prof. Krishna

1. Would a task-driven policy (using model predictions or attention maps to select glimpses) beat random? This is the most principled next step.

2. Would human fixation sequences on recognition tasks, used as a gaze policy for CanViT, outperform random? And if yes, does this hold even though CanViT was not trained on human-generated fixation sequences?

3. Would results differ on a harder dataset — cluttered scenes, small objects, fine-grained recognition — where spatial location of the glimpse carries more information?

4. Does CanViT's canvas attention already implicitly compensate for bad policies? The model's robustness to glimpse location may be a designed property, not a bug. Understanding this would clarify the limits of inference-time policy improvement without retraining.

---

## 8. Technical notes

- **scipy incompatibility:** `scipy.ndimage.uniform_filter` fails on numpy 2.x (Python 3.13 Colab). Replaced with pure numpy box filter in `src/policies/saliency_policy.py`.
- **Colab session management:** Phase 3 notebook uses per-policy Drive checkpointing. Each policy saves its parquet immediately; sessions can be resumed without losing completed policies.
- **Reproducibility:** All results use fixed seeds. ImageNette subset (Phase 1/2) used `random.Random(42)`. Full val (Phase 3) uses all available val images in deterministic sorted order.

---

## 9. Files

| File | Description |
|------|-------------|
| `canvit_results/phase1_raw.parquet` | Phase 1: 100-image random baseline |
| `canvit_results/phase2_raw.parquet` | Phase 2: 7-policy, 100 images |
| `canvit_results/phase3/phase3_*.parquet` | Phase 3: per-policy, ~3900 images |
| `canvit_results/phase3/phase3_summary.csv` | Phase 3: AUGC + bootstrap CI summary |
| `canvit_results/phase3/phase3_policy_curves.png` | Accuracy-vs-glimpses plots |
