# Hypotheses

**Project:** Biologically Motivated Active Gaze Policies for CanViT  
**Registered:** 2026-08-26 — BEFORE any experimental results were seen.

**RULE: Do not modify these hypotheses after seeing results. If data contradict them, report that.**

---

## H1 — Saliency improves efficiency over random

A low-resolution saliency-guided policy will provide higher classification accuracy per high-resolution glimpse than random sampling, particularly at small glimpse budgets (1–3 glimpses).

**Rationale:** Saliency maps, even computed from a downsampled scene view, should capture high-contrast or semantically informative regions. These regions are more likely to contain discriminative object features than uniformly random crops.

**What would falsify this:** Saliency performing at or below random AUGC across all budgets, not explained by a specific confound like center bias.

---

## H2 — IOR reduces redundant sampling

Adding inhibition-of-return to a saliency policy will reduce revisit frequency and improve information-acquisition efficiency compared to saliency alone, particularly beyond the 2nd glimpse.

**Rationale:** Without IOR, a saliency map with a dominant peak will repeatedly draw attention to the same region. IOR forces exploration of novel regions, reducing redundant glimpses.

**What would falsify this:** IOR version performing worse than or equal to saliency-only at AUGC, OR revisit frequency not decreasing measurably.

---

## H3 — Policy benefit is largest at small glimpse budgets

The benefit of intelligent policies (saliency, saliency+IOR) over random sampling will be largest at small glimpse budgets (1–3 glimpses) and will decrease as the number of glimpses increases toward 6–8.

**Rationale:** With more glimpses, random sampling eventually covers the image adequately, erasing the advantage of prioritized selection. Early glimpse decisions carry the highest marginal value.

**What would falsify this:** Intelligent policies showing equal or greater advantage at large budgets as at small budgets.

---

## H4 — Spatial coverage may match saliency

A spatial coverage policy (maximizing distance from prior fixations) may perform competitively with the saliency policy at intermediate budgets, demonstrating that apparent saliency advantages cannot automatically be attributed to semantic attention.

**Rationale:** Coverage ensures no region is sampled twice and guarantees broad spatial information gathering. Much of the benefit of "saliency" may actually come from simply avoiding repeated sampling of the same region.

**This is an important alternative explanation.** If H4 is confirmed, H1/H2 must be reinterpreted: the advantage may come from spatial exploration, not semantic prioritization.

---

## H5 — Task-optimal policies will not look like human scanpaths

Policies that maximize CanViT classification accuracy will not necessarily produce fixation sequences that resemble human goal-directed gaze.

**Rationale:** Human visual search is conditioned on a specific target object and is subject to biological constraints (saccade latency, foveal resolution, working memory). CanViT operates on 128 px crops regardless of content. Optimizing for classification efficiency under these different constraints should produce different spatial behaviors.

**What would falsify this:** Finding that optimal-for-accuracy policies produce fixation statistics (e.g., center bias, saccade amplitude distribution) statistically indistinguishable from human scanpaths on matched tasks.

---

## Secondary prediction

The oracle policy (full-resolution saliency) will outperform the valid low-resolution saliency policy. The gap between them quantifies the information cost of the active-vision constraint.

---

## Notes on interpretation

All hypotheses concern comparative performance on the accuracy-efficiency curve (AUGC), not absolute accuracy at the maximum glimpse budget. At T=8, random may converge to similar accuracy. The question is whether intelligent policies get there faster.

Results must be evaluated with paired statistics (paired bootstrap CIs, McNemar where appropriate). A difference without a confidence interval is not a result.
