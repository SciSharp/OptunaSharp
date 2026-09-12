# Plan 06 — VkD-CMA sampler

> **Status: DESIGN (not started).** Depends on: Core CMA infrastructure (shares patterns with
> `Cma`/`LmCma`); validated via [01](01-bbob-coco-benchmark-suite.md). Tier N. Value ★★ / effort med-high.
> Independent enrichment — blocks nothing, but completes the restricted-covariance family and
> reproduces a JANUS theory result.

## Goal

Port **VkD-CMA** (Akimoto & Hansen, GECCO 2016 — "Projection-Based Restricted Covariance Matrix
Adaptation for High Dimension"): a CMA-ES variant that restricts the covariance to a rank-`k` term
plus a diagonal, adapting both online, for `O(kd)` cost in high dimension.

## Motivation

- It completes the **restricted-covariance trio** alongside the in-flight LM-CMA and the existing full
  `Cma` — the three points on the "how much covariance structure to learn" axis.
- It is literally the subject of JANUS's **Proposition 1** and **Table 9**: a host-fixed rank-`k`
  covariance basis cannot express an arbitrary residual-aligned metric, and empirically LM-CMA/VkD-CMA
  underperform a full-rank host at `d ∈ {30,100}`. With VkD-CMA + LM-CMA + `Cma` + Plan 01, we can
  **reproduce Table 9** natively — a concrete, citable validation of the paper's structural claim.
- High-dimensional coverage: with LM-CMA and VkD-CMA, OptunaSharp has real options where full CMA's
  `O(d²)` covariance is impractical.

## Scope / deliverables

- A native `VkdCmaSampler` (beyond-core: no `cmaes`/Optuna/OptunaHub counterpart → convergence-validated
  like `LmCmaSampler`; reference the paper + the `cmaes` package's `VkD-CMA` for structure if useful).
- The rank-`k`+diagonal covariance representation, its update, and the `k`/`d`-adaptation heuristics.
- Per-trial ask/tell over the OptunaSharp protocol; MT19937 via `LazyRandomState`; the same
  independent-sampler fallback and dynamic-space handling as `LmCmaSampler`.

## Design notes

- Follow `LmCmaSampler`'s precedent for the protocol concessions (one candidate per trial from
  restored/held state, box-clamp to the transformed bounds) and document them the same way.
- Reuse the transformed-`[0,1]` box + `SearchSpaceTransform` convention shared by the CMA samplers.

## Validation

- Convergence gates: sphere to precision; ill-conditioned ellipsoid *descends* the valley (the
  restricted approximation is slower than full CMA — that is the documented trade-off, not a defect,
  exactly as `LmCmaSamplerTests` frames it).
- **Reproduce Table 9** (with Plan 01): geometric-mean final error of VkD-CMA vs full `Cma` at
  `d ∈ {30,100}`, confirming the full-rank host wins — the empirical face of Prop 1.

## Out of scope

- Matching any specific upstream trajectory bit-for-bit (no canonical reference exists for parity).
- The learned/meta variants; constrained handling.
