# Plan 07 — MOEA/D · RVEA · PSO baselines

> **Status: DONE (implemented 2026-09-12).** Depends on: Core; validated via [01](01-bbob-coco-benchmark-suite.md)
> (PSO) and DTLZ/WFG (the MOEAs). Tier N. Value ★ / effort medium (each). Independent enrichment —
> blocks nothing; realized through the benchmark suite. Lowest priority in the order.
>
> All three native beyond-core samplers shipped in Core with 33 convergence/HV/determinism unit tests
> (both TFMs, full suite 806/806). See **Implementation** at the bottom.

## Goal

Port three standard baseline optimizers OptunaSharp lacks, purely to round out the benchmark field so
the VolleyEval leaderboards and the JANUS/MOO comparisons have the customary opponents:

- **MOEA/D** (Zhang & Li, 2007) — decomposition-based MOEA (Tchebycheff/PBI scalarizations over a
  weight-vector simplex with neighborhood mating).
- **RVEA** (Cheng et al., 2016) — reference-vector-guided EA for many-objective problems (angle-
  penalized distance selection, reference-vector adaptation).
- **PSO** (particle swarm) — the canonical single-objective swarm baseline (gbest/lbest with inertia).

## Motivation

- The JANUS paper and every MOO study benchmark against MOEA/D and RVEA; PSO is a ubiquitous scalar
  baseline. Having them native lets OptunaSharp's leaderboards be read the way the literature expects.
- They exercise regimes the current field under-covers: decomposition (MOEA/D), many-objective
  reference vectors (RVEA), and swarm dynamics (PSO) — extra spread for VolleyEval's regime matrix.

## Scope / deliverables

- Three native samplers (`MoeaDSampler`, `RveaSampler`, `PsoSampler`), each beyond-core (no Optuna
  counterpart → convergence/HV-validated like `LmCmaSampler`; reference pymoo for structure).
- Reuse the shared NSGA operators (SBX/PM) and nondominated / HV infra for the MOEAs; PSO needs only
  velocity/position state + the `[0,1]` transform box.
- MT19937 via `LazyRandomState`; per-trial ask/tell; independent-sampler fallback + dynamic-space
  handling per the sampler house pattern.

## Design notes

- MOEA/D: weight vectors via Das–Dennis simplex lattice; neighborhood `T`; Tchebycheff by default.
  Document the scalarization/replacement choices (they affect the front).
- RVEA: reference-vector generation + the APD selection with the `α`/generation-scaled penalty and the
  periodic reference-vector adaptation.
- PSO: gbest with inertia-weight schedule; box-reflect or clamp on bound violation (document which).

## Validation

- MOEA/D, RVEA: hypervolume gates on DTLZ/WFG at the target objective counts; determinism.
- PSO: sphere/Rastrigin convergence on BBOB (Plan 01); determinism.
- Optional value-parity vs pymoo on a few tasks (not bit — different RNG/operator details).

## Out of scope

- The adaptive/learned variants (jDE21, SAHLPSO, RLDEAFL) and any torch/RL MetaBBO baselines.
- Splitting into three separate plans — grouped here as one baseline-completeness workstream; they may
  be executed independently and in any order.

## Implementation (2026-09-12)

All three samplers are native, beyond-core (no Optuna counterpart → convergence/HV-validated like
`LmCmaSampler` / `SmsEmoaSampler`, pymoo 0.6.1.3 referenced for structure), MT19937 via
`LazyRandomState`, per-trial ask/tell, independent-sampler fallback, `[Experimental]` warning.

**Files added (Core):**

- `Internal/ReferenceDirections.cs` — shared Das–Dennis simplex lattice (`DasDennis`,
  `NumberOfUniformPoints`, `PartitionClosestToPoints`), the weight/reference vectors for MOEA/D + RVEA.
- `Samplers/ParticleSwarm/Pso.cs` + `Samplers/PsoSampler.cs` — **PSO**. Mirrors
  `DifferentialEvolutionSampler`+`De` exactly: the optimizer state serializes into `pso:*` trial system
  attrs (resumes across processes), generation + target-index ask/tell, candidates clamped to the [0,1]
  transform box. gbest swarm with Shi–Eberhart LDIW inertia (`w` 0.9→`wMin` 0.4 over
  `inertiaDecayGenerations`, since Optuna is trial-budget-driven). Realized velocity is recovered at
  `Tell` as `x_evaluated − x_previous` (velocity reset on bound contact).
- `Samplers/Moead/{MoeadDecomposition,MoeadOffspring}.cs` + `Samplers/MoeaDSampler.cs` — **MOEA/D**.
  In-memory reconstruction like `SmsEmoaSampler` (steady-state, does not resume across processes). N =
  the Das–Dennis lattice closest ≤ `populationSize`; subproblem `k = (#offspring) mod N` (stamped
  `moead:subproblem`); z\* updated per fold; up to `nReplacements` neighbors replaced (Zhang&Li nr,
  nearest-first). Tchebycheff default (+PBI θ=5, weighted-sum). Neighborhood mating with prob δ else
  global; reuses SBX + polynomial mutation.
- `Samplers/Rvea/RveaApdSurvival.cs` + `Samplers/RveaSampler.cs` — **RVEA**. Uses `BaseGASampler` like
  NSGA-III (parent cache resumes across processes); APD survival gets the generation via its own
  `SelectParent`. `nMaxGenerations` (default 100) sets the penalty/adaptation horizon; reference vectors
  re-scale to `nadir−ideal` every `ceil(nMaxGen·adaptFreq)` generations. Reuses the NSGA-II child
  generation; default ops SBX(η=30)+PM(η=20).

**Tests added (`test/OptunaSharp.Tests/Samplers/`, 33 tests, both TFMs):**
`ReferenceDirectionsTests` (lattice + decomposition math), `PsoSamplerTests` (sphere convergence 3e-9
optimizer / <1e-1 sampler, determinism, state round-trip, bounds, inertia schedule),
`MoeaDSamplerTests` (ZDT1 HV 94% of ideal, **beats** NSGA-II by 25%, determinism, spread, PBI/weighted-sum,
single-objective fallback), `RveaSamplerTests` (ZDT1 HV 90%, **matches** NSGA-II on 2-obj, 3-obj DTLZ2
converges to the unit-sphere front with a broad spread, determinism through adaptation).

**Measured (300–600 trials):** MOEA/D 0.825 HV (94% of ideal ZDT1), RVEA 0.790 (90%), both above NSGA-II
0.615 and SMS-EMOA 0.757. RVEA on 3-obj DTLZ2: 80–150 front points at avg `|‖f‖−1|` ≈ 0.06–0.09.

**Deferred (optional, per the plan's Validation §):** value-parity vs live pymoo through
`OptunaSharp.Interop.Pymoo` (the SMS-EMOA/LM-CMA precedent ships unit convergence gates only, since the
different RNG/operator details preclude bit-parity); DocFX algorithm pages; VolleyEval leaderboard entries
(realized through Plan 01).
