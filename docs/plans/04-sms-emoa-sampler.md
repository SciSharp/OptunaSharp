# Plan 04 — SMS-EMOA sampler

> **Status: DONE.** Depends on: hypervolume infra (`src/OptunaSharp.Core/Internal/Hypervolume.cs`,
> present) + nondominated sorting (present in the NSGA infra). Tier N. Value ★★ / effort medium.
> A JANUS-SMS host dependency (Plan 05, Phase 3) and a respected MOO algorithm OptunaSharp lacks.

## Goal

Port **SMS-EMOA** (S-Metric Selection Evolutionary Multi-objective Optimization Algorithm — Beume,
Naujoks & Emmerich, 2007): a steady-state MOEA whose survival operator drops the individual with the
least **hypervolume contribution** within its (nondominated-sort) worst front. It is the low-objective
host JANUS-SMS wraps in the paper.

## Motivation

- Plan 05's MOO phase pairs JANUS with SMS-EMOA on the 2–3-objective suites (ZDT/DTLZ/WFG); without
  SMS-EMOA there is no JANUS-SMS.
- SMS-EMOA is a standard hypervolume-based MOEA the library is missing (OptunaSharp has NSGA-II/III but
  no S-metric selection). The required hypervolume machinery already exists.

## Scope / deliverables

- A native `SmsEmoaSampler` (beyond-core: SMS-EMOA has **no** Optuna/OptunaHub counterpart, so — like
  `LmCmaSampler` — it is validated by convergence/HV, not bit-matched to an upstream). Reference
  structure: the paper + pymoo's `SMSEMOA` for the survival details.
- Steady-state (μ+1) loop over the OptunaSharp per-trial protocol: generate one offspring
  (SBX + polynomial mutation, reusing the NSGA-II crossover/mutation operators already in
  `Samplers/Nsgaii/`), then survival = nondominated sort + least-HV-contribution removal on the worst
  front (using `Internal/Hypervolume.cs`).
- RNG via `LazyRandomState` (MT19937), consistent with the other samplers.

## Design notes

- Reuse: nondominated sorting and the SBX/PM operators already exist for NSGA-II — SMS-EMOA differs
  only in the survival rule (HV contribution vs crowding distance). Keep the operators shared.
- Reference-point handling for HV: follow the paper/pymoo (dynamic worst-point + offset); document the
  choice since it affects the survival ranking.

## Validation

- Convergence gates (the `LmCma` model): on ZDT1/DTLZ2 the achieved hypervolume approaches the known
  Pareto front's HV within tolerance; determinism (same seed → same front); state round-trip if
  serialized.
- Optional: compare final normalized HV against pymoo's SMS-EMOA on a few tasks (value parity, not
  bit — pymoo's RNG/operator details differ).

## Out of scope

- Constrained SMS-EMOA; parallel/asynchronous steady-state; the many-objective regime (that is
  AGE-MOEA-2's job, in flight, and the JANUS-AGE2 host).

## Outcome (done)

- **`Samplers/SmsEmoa/SmsEmoaSurvival.cs`** — the environmental survival, a pure objective-space
  operator (unit-testable): nondominated-sort via `MultiObjective.FastNonDominationRank`, keep whole
  fronts while they fit, then trim the single overflow front by repeatedly dropping the least
  *exclusive* hypervolume contributor (`HV(front) − HV(front∖{i})` via `Internal/Hypervolume.cs`).
  Follows pymoo's `LeastHypervolumeContributionSurvival`: population-normalized objectives, reference
  point `1 + eps` (default `eps = 10`), clamped to the reference so the HV module's "all points
  dominate the reference" contract holds. A 1-objective fast path drops worst-value points directly.
- **`Samplers/SmsEmoaSampler.cs`** — a steady-state (μ+1) `BaseSampler`: the first μ completed trials
  are the random initial population; every later trial proposes one offspring from the current μ
  survivors via the **shared NSGA-II child generation** (binary-tournament `SBXCrossover` +
  `PolynomialMutation` — nothing re-implemented), and each completed offspring is folded in and the
  worst survivor removed. The survivor set is reconstructed incrementally in memory (a documented
  beyond-core concession, like `JanusSampler`: no cross-process resume). `LazyRandomState` (MT19937)
  throughout.
- **Reuse, not reinvention.** Crossover/mutation/tournament come from `Samplers/Nsgaii/` (the plan's
  ask), nondominated sorting from `Internal/MultiObjective`, hypervolume from `Internal/Hypervolume`.
  Only the survival rule (S-metric contribution vs crowding distance) is new. The stable NSGA-II
  operators were reused rather than the concurrently-in-flight `Samplers/AgeMoea/` pymoo operators.

**Validation.** `SmsEmoaSurvivalTests` (4) pin the survival rule on synthetic fronts (least-contributor
removal, worst-front-first, duplicate removal, bounds). `SmsEmoaSamplerTests` (4) are convergence
gates on **ZDT1** (2-objective): achieved hypervolume ≥ 0.85 × the true-front HV (measured **0.757 /
0.876 = 86.4 %** at 600 trials, full 40-point spread); **matches or beats NSGA-II** on achieved HV
with the same SBX+PM operators (measured SMS beats NSGA on all 3 seeds — 0.585/0.668/0.692 vs
0.566/0.484/0.558); same-seed determinism (bit-exact); and a non-trivial Pareto spread. All 8 pass on
net8.0 and net10.0.
