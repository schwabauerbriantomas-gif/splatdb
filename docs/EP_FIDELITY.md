# Epps–Pulley distributional fidelity (EP fidelity) — methodology + receipts

**Date:** 2026-11-05 · **Origin:** comparison against LeJEPA/SIGReg
(Balestriero & LeCun, arXiv:2511.08544) requested by the owner.
**Code:** `src/ebm/fidelity.rs` · **Receipts:** `docs/eval/ep_lejepa_x/`.

## What it measures

`fidelity.rs` implements the sketched Epps–Pulley test (the statistic behind
SIGReg) adapted to the unit sphere:

* L2-normalized embeddings, `M` random unit directions, standardized
  projections `u = √d · aᵀz`;
* per direction, quadrature EP vs N(0,1) with Gaussian window
  `w(t)=e^{−t²/2}` on `t∈[−5,5]`, mean over directions, no factor N.

Two entry points: `ep_statistic` (distance-to-isotropy of any batch) and
`fidelity_gap_ratio` (EP of real data vs EP of samples drawn from the splat
vMF mixture — the same fit as the one-class duel: k-means, α=5.0 const,
κ=cluster counts).

## Findings (H1 confirmed / H2 refuted, measured)

Experiment: 20 Newsgroups, all-MiniLM-L6-v2 384d, seed 42, CPU-only.

* **H1 — isotropy is NOT a universal optimum.** Distance-to-isotropy
  correlates POSITIVELY with one-class AUROC across the 20 classes:
  Spearman ρ=0.5023 (p=0.024), Pearson r=0.5469 (p=0.013); size confound
  excluded (r(EP, n_train)=0.136, p=0.57). Causal ladder: mixing train
  embeddings toward isotropic noise takes mean AUROC 0.956 → 0.940 → **0.500
  exact** while EP falls 0.087 → 0.0016. LeJEPA's isotropic target is
  optimal for linear-probe risk, and *destructive* for one-class detection:
  the non-Gaussian structure IS the detection signal.
* **H2 — splats compress enough to rank, not to reproduce the
  distribution.** EP(real) 0.049–0.121 vs EP(vMF mixture samples)
  0.0012–0.0017: median **gap ratio 0.016**. Mechanism: κ=counts is weak at
  d=384 (A₃₈₄(20)≈0.052), each splat samples nearly isotropically, and the
  mixture gaussianizes — losing the microstructure EP sees. The same splat
  model still scores AUROC 0.949–0.991 on the duel task: ranking does not
  need the structure EP measures. The gap is now a number, and the yardstick
  for a future refit (learned per-cluster κ/α instead of counts/const).

## Constraints

* **A5 stands (docs/A5_FINDINGS.md): EP (or any density signal) must never
  enter the retrieval ranking score** — energy-as-score collapsed
  recall@10 from 0.90 to ~0.00. This module is a diagnostic/fidelity
  metric only.

## Files

* `src/ebm/fidelity.rs` — implementation + 6 unit tests pinned to the
  experiment receipts (EP discrimination 1365×, vMF mean-resultant exact
  A₃(10)=0.9 and continued-fraction values, sampler R within tolerance,
  gap-ratio regime pinned at d=64/6 clusters/κ≈10 reproducing ratio ≪1).
* `docs/eval/ep_lejepa_x/stageA_arm.json` — per-class AUROC/EP + ladder.
* `docs/eval/ep_lejepa_x/stageB_arm.json` — per-class EP real/synth + ratios.
