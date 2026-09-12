# Plan 05 — JANUS completion: all remaining phases (DE host + MOO)

> **Status: DESIGN (not started).** Depends on: [02](02-component-losses-channel.md) `component_losses`,
> [03](03-janus-infill-wrapper-infra.md) infill-wrapper infra, [04](04-sms-emoa-sampler.md) SMS-EMOA,
> **AGE-MOEA-2 (in flight, concurrent session)**, and a DE host it builds first. Validated via
> [01](01-bbob-coco-benchmark-suite.md). Tier N. Value ★★★ / effort high. The JANUS capstone.

## Goal

Finish JANUS beyond the Phase-1 CMA-ES host, in one pass: **Phase 2** — a native Differential
Evolution host + JANUS-DE (the paper's host-agnostic check, 86–96 % win); **Phase 3** — the
multi-objective JANUS (port of `janus_optuna/janus_moo_cmix.py`) attached to SMS-EMOA (low-objective)
and AGE-MOEA-2 (many-objective). Plus the two deferred Phase-1 items.

## Phase 2 — Differential Evolution host + JANUS-DE

- **2a. `DifferentialEvolutionSampler` (native).** OptunaSharp has **no DE sampler** — this is a real
  gap and a fundamental BBO algorithm. Standard DE/rand/1/bin (+ optionally best/1) with `F`, `CR`,
  per-trial ask/tell over the OptunaSharp protocol, `LazyRandomState` (MT19937). Beyond-core,
  convergence-validated. It stands alone as a useful sampler, independent of JANUS.
- **2b. JANUS-DE.** Wrap the DE host with the Plan-03 infill layer. DE exposes no covariance, so CMIX
  degrades to the paper's "analogous box and step-length constraints" and injection is GN-oriented
  (JANUS-GN-like); the residual metric still comes from the window (`JanusMetric`/`FitComponentJacobian`).

## Phase 3 — multi-objective JANUS (`JanusMooCmixSampler`)

- Port `janus_moo_cmix.py`: **subclass the existing (bit-exact) `NSGAIISampler`** and, past a warm-up,
  replace a child with probability `cmix_prob` by an inverse-curvature Gaussian sample from the
  window metric. The metric is `_fit_component_metric` (Plan 02 components) or the `diag_quadratic`
  scalar fallback — **already ported** as `JanusMetric` / `FitComponentJacobian` (mind the MOO file's
  slightly different `_standardize`: `where(sd<1e-12,1,sd)`, no `+1e-9`, and it symmetrizes — port
  faithfully as a MOO variant).
- Hosts: SMS-EMOA (Plan 04) for the 2–3-objective suites, AGE-MOEA-2 (in flight) for 5/8/10-objective.
- Center selection: crowding-ranked Pareto-front point (`_choose_center`), inverse-metric covariance
  `_inverse_metric_cov` (trace-normalized, `cmix_sigma`-scaled), sampled via `multivariate_normal`.

## Deferred Phase-1 items (fold in here)

- **`cmixMode:"reshape"`** — in-place host-covariance mixing `C ← (1−w)C + w·tr(C)/tr(T)·T`; needs a
  small internal `Cma.SetCovariance` (reset the cached eigendecomposition). Default is `portfolio`
  (candidates), which needs none.
- **`cmixCandidateCenter:"archive"`** — Pareto-dominance archive-center selection (`_archive_centers`).

## The parity payoff in MOO (ties back to the RNG-backend question)

Unlike the single-objective CMA host (torch/evotorch/float32 — not bit-reachable), the MOO reference is
**Optuna's NSGA-II (which OptunaSharp ports bit-exactly) + numpy `default_rng` (PCG64) for the CMIX
injection** — no torch. So JANUS-MOO can reach **much tighter parity**: the NSGA-II trajectory is
already bit-exact, and the CMIX injection becomes byte-comparable **if** we drive it through NumSharp's
**PCG64 `Generator`** and match numpy's `multivariate_normal` (Cholesky-based) sampling. This is the
one place the "add a PCG64 backend option" idea genuinely pays off. Treat the PCG64/`multivariate_normal`
match as a stretch goal within Phase 3; the metric itself is already live-byte-parity-testable
(`_fit_component_metric` / `_fit_diag_quadratic_metric` are pure numpy+optuna).

## Validation

- Phase 2: DE convergence gates + JANUS-DE beats/matches DE on BBOB (Plan 01), aggregated over seeds
  (early-budget, per the Phase-1 calibration lesson — don't gate a single ill-conditioned seed).
- Phase 3: hypervolume gates on ZDT/DTLZ/WFG; zero-significant-regression check vs the bare host
  (the paper's headline property); **live interop byte-parity** of the MOO metric functions against
  `janus_optuna.janus_moo_cmix` (torch-free), extending `JanusMetricParityTests`.

## Out of scope

- The learned MetaBBO/NN-BBO baselines the paper compares against (torch/RL, training-bound) — bridge-only.
- PSO/jDE21/SAHLPSO hosts (not JANUS hosts; see Plan 07 for the baseline swarm).
