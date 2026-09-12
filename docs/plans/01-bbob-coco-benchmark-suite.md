# Plan 01 — BBOB/COCO benchmark suite (f1–f24)

> **Status: DESIGN (not started).** Depends on: nothing. Tier N (pure managed: BCL + NumSharp).
> Value ★★★ / effort low. First in the dependency order — the validation substrate for every sampler.

## Goal

Port the **noiseless BBOB (Black-Box Optimization Benchmarking) suite** — the 24 COCO functions
`f1…f24` in their standard 5 groups — as pure-managed OptunaSharp objectives, with COCO-compatible
instance generation (per-instance shift `x_opt`, rotations, and `f_opt`). This is the exact benchmark
JANUS validates on (a 16-function held-out subset), so it lets us **reproduce the paper's
JANUS-vs-CMA tables** and gives VolleyEval (`docs/BENCHMARK_VOLLEYEVAL_PLAN.md`) a respected standard
suite to score **every** sampler on a common footing.

## Motivation

- JANUS Phase 1 is currently validated only on ad-hoc sphere/ellipsoid objectives. BBOB turns "it
  converges" into "it reproduces the paper's per-function ratios across `d ∈ {30,100,500}`".
- VolleyEval's thesis is *there is no best sampler, only a best sampler per regime*. BBOB's five
  groups (separable, low-conditioning, high-conditioning unimodal, multimodal-adequate,
  multimodal-weak) are the canonical regime spread.
- It benchmarks CMA-ES, LM-CMA, TPE, GP, DE and JANUS on identical objectives — the only way the
  leaderboards mean anything.

## Scope / deliverables

- The 24 functions with their COCO definitions and the shared transform primitives:
  `T_osz` (oscillation), `T_asy(β)` (asymmetry), `Λ^α` (conditioning diagonal), rotations `R`,`Q`,
  `f_pen` (boundary penalty), and the `f_opt`/`x_opt` instance draws.
- COCO-compatible **instance generation** from an integer instance id (the seeded shift/rotation
  recipe), so a `(function, instance, dimension)` triple is reproducible and comparable to `cocoex`.
- A thin objective adapter so a function is drivable by an OptunaSharp `study.Optimize` over
  `SuggestFloat` in the standard `[-5, 5]^d` box.
- Placement: `test/OptunaSharp.Benchmark/Objectives/Bbob/` (objectives are benchmark artifacts, **not**
  Core — they are not Optuna algorithms). VolleyEval and the parity tests consume them from there.

## Design notes

- Everything is elementwise math + rotations (`LinAlg.Dot`), so it is naturally byte-comparable to the
  reference. Use `MathUtil.PairwiseSum` / sequential-vs-pairwise conventions where a reduction's order
  matters (mirror the Cma.cs discipline) if bit-exactness against `cocoex` is pursued.
- Rotation matrices come from COCO's `compute_rotation` (Gram–Schmidt on a seeded Gaussian); reproduce
  its seeding so instances line up with Python.

## Validation

- **Live interop parity** (`test/OptunaSharp.Tests.Interop`): for each `(function, instance, dim)` and
  a battery of probe points `x`, assert `f_cs(x)` matches `cocoex` (or `bbobbenchmarks.py`) to
  bit / ~1-ULP tolerance (pure math over OpenBLAS → expect byte-exact on the x64 pinned host, like
  `LinAlgParityTests`). Self-skip if `cocoex` is unavailable.
- Assert `f(x_opt) == f_opt` per function/instance.

## Out of scope

- The noisy BBOB suite; the bi-objective `bbob-biobj` suite; the COCO logging/post-processing (`cocopp`).
- Wiring into CI perf runs (BenchmarkDotNet) — that is a later VolleyEval concern.
