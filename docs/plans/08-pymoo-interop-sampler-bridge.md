# Plan 08 — `OptunaSharp.Interop.Pymoo`: the pymoo sampler bridge

> **Status: DONE (2026-09-10).** `src/OptunaSharp.Interop.Pymoo` ships `PymooSampler` + `PymooAlgorithm`
> (11 algorithms) + `PymooAlgorithmOptions`, wired into the sln + Core `InternalsVisibleTo` + the interop
> suite. Validated live: `PymooSamplerParityTests` = **17/17 green on both net8.0 and net10.0** (11
> behavioral-zoo + 3 reproducibility + 2 value-parity + 1 maximize). Depends on: Core,
> `OptunaSharp.Interop.pythonnet`, `NumSharp.Interop.pythonnet`. **Tier X** (satellite package — foreign
> runtime: embedded CPython + pymoo via pythonnet). Unlocked by the 2026-09-10 pymoo-on-numpy-2.4.2
> migration (pymoo now runs in the interop interpreter — see the [`age-moea-2-port`](../../) memory).
> Sibling of `OptunaSharp.Interop.BoTorch` / `.PyCma`; the live-runtime complement to the *native*
> baselines in [07](07-moead-rvea-pso-baselines.md). **See the Implementation outcome section at the end.**

## Goal

Ship `OptunaSharp.Interop.Pymoo` — a Tier-X satellite that exposes **pymoo's whole multi-objective
algorithm zoo** as first-class OptunaSharp samplers driving the **real pymoo** in-process, exactly as
`BoTorchSampler` drives real BoTorch. One `PymooSampler : BaseSampler` selects the algorithm by a
pythonnet-free enum:

- **`PymooAlgorithm`**: `Nsga2`, `Nsga3`, `RNsga2`, `RNsga3`, `UNsga3`, `AgeMoea`, `AgeMoea2`, `MoeaD`,
  `Rvea`, `SmsEmoa`, `Spea2`, `Ctaea`, `Dnsga2` (13 — everything in `pymoo.algorithms.moo`).

This gives OptunaSharp users the genuine article — pymoo's operators, reference-direction machinery,
decomposition and HV-based survival — while the Optuna side (study, storage, ask/tell, search space) is
native OptunaSharp. It does **not** require any Python glue package; users `pip install pymoo`.

## Motivation

- **Immediate breadth.** Optuna core has only NSGA-II/III; OptunaSharp adds native AGE-MOEA-2 (and, in
  flight, SMS-EMOA). The bridge delivers the remaining ~11 MOEAs *now*, faithfully, at wrapper cost —
  and doubles as the reference oracle for any we later port natively (the [07](07-moead-rvea-pso-baselines.md)
  MOEA/D · RVEA, a future SPEA2/MOEA-D native).
- **Reference-direction & decomposition regimes** (MOEA/D Tchebycheff/PBI, RVEA APD, NSGA-III/U-NSGA-III
  Das–Dennis) that OptunaSharp has no native analog for — extra spread for VolleyEval's regime matrix
  ([01](01-bbob-coco-benchmark-suite.md)) with the literature's canonical opponents.
- **Established, low-risk pattern.** `PyCmaSampler` already proves the population-based reconstruct-from-
  trials bridge (build the optimizer in a GIL scope, replay completed trials via `tell`, `ask` the next
  infill, dispose). pymoo exposes the identical **native `ask()` / `tell()`** interface, so the shape
  transfers directly.

## Scope / deliverables

- **Project** `src/OptunaSharp.Interop.Pymoo/` (`net8.0;net10.0`), csproj mirroring
  `OptunaSharp.Interop.BoTorch` (Tier-X package metadata, README, icon, strong-named); `ProjectReference`
  to Core + `OptunaSharp.Interop.pythonnet`, `PackageReference` `NumSharp.Interop.pythonnet`
  (`$(NumSharpVersion)`). Add `OptunaSharp.Interop.Pymoo` to Core's `InternalsVisibleTo`
  (`Assembly/Properties.cs`) — it reuses the internal `SearchSpaceTransform` + `IntersectionSearchSpace`
  + warning seam, exactly as BoTorch/PyCma do. Add to `SciSharp.OptunaSharp.sln`.
- **Runtime** `PymooRuntime` / `PymooInterop` (mirror `PyCmaInterop`): a cached `Python.pymoo.check`
  import-guard with a helpful "pip install pymoo" message; the `Python.pymoo.*` **pythonic facade**
  (algorithm constructors, `problem` wrapper, `ask`/`tell`, reference-direction factory) per the
  `pythonic` skill (facade rows for fixed calls; snippet runner only if a body must diff 1:1).
- **`PymooSampler : BaseSampler`** — the reconstruct-from-trials bridge:
  - Ctor: `PymooAlgorithm algorithm`, `int populationSize`, algorithm-specific options (a small typed
    options object per family: reference-direction spec for NSGA-III/RVEA/MOEA-D, `T`/decomposition for
    MOEA-D, `epsilon`/`weights` for R-NSGA), `int? seed`, `nStartupTrials`, `independentSampler`,
    `warnIndependentSampling` — the PyCma ctor shape.
  - `InferRelativeSearchSpace`: intersection space minus single-value distributions (the house pattern).
  - `SampleRelative`: inside one `PythonRuntime.Run` GIL scope — build the pymoo `Problem` (an
    `ElementwiseProblem`/`Problem` with `n_var`/`n_obj`/`xl`/`xu` from the transformed box), build the
    chosen `Algorithm(pop_size, …, seed)`, **replay** the completed trials generation-by-generation via
    `tell` (feeding evaluated `Population`s), `ask` the next infill `Population`, and dole out **one row
    per Optuna trial** using the `BaseGASampler` generation bookkeeping (like `AgeMoea2Sampler`).
  - `SampleIndependent` / startup: the independent sampler (default `RandomSampler(seed)`) for out-of-
    space and pre-startup params; direction flip to minimization; MO via `study.Directions`.
- **Search-space transform**: map Optuna `float`/`int`(+`log`) to a continuous `[xl,xu]` box (int rounds
  on the way out); categoricals as **ordinals** by default (like PyCma) with an opt-in one-hot mode
  (reuse `SearchSpaceTransform`). Document the mapping — it is the one lossy seam.
- **Data crossing**: the `X` (candidates) and `F` (objective) matrices cross via
  `NumSharp.Interop.pythonnet` zero-copy (`np.array(x, copy:false).ToNumpy()` in, `pop.get("X").ToNDArray()`
  out) — **never** a PyList/`tolist` element loop (per the `pythonic` skill §4).
- **README** + DocFX sampler note; an `examples/` MOEA/D-or-RVEA showcase (optional).

## Design notes

- **Statelessness via replay.** Like PyCma, the pymoo `Algorithm` lives *only* inside the GIL scope of a
  single `SampleRelative` — built, `tell`-replayed from storage, `ask`ed, disposed. Nothing pymoo-side
  persists between trials; the study storage is the source of truth. This keeps it correct under
  resumption, distributed workers, and `n_jobs`.
- **Population↔ask/tell doling.** pymoo `ask()` returns a whole `pop_size` `Population`; OptunaSharp wants
  one trial at a time. Reconstruct the current generation's infill population, index it by the trial's
  position within the generation (the `BaseGASampler:generation` attr), and return that row's params —
  the same doling `AgeMoea2Sampler`/`NSGAIISampler` do natively.
- **Seeding.** pymoo seeds `np.random` + stdlib `random` globally in `setup(seed=…)`. The bridge sets the
  seed on each rebuild; because the whole run is replayed deterministically inside one GIL scope, the
  global-RNG caveat that bit us in the AGE-MOEA-2 recon (numpy-array vs list drift — see the
  [`pycma-sampler-port-plan`](../../) memory) does not apply here (no incremental cross-call state).
- **Reference directions** (NSGA-III/RVEA/MOEA-D/U-NSGA-III/CTAEA) via `get_reference_directions` — expose
  a small spec (`das-dennis`/`energy`/`layer-energy`, `n_partitions`) on the options object; default to
  Das–Dennis sized to `n_obj`.
- **Constraints**: pymoo models `G ≤ 0`; wire OptunaSharp's constraint values (when present) into the
  problem's `out["G"]`. Phase 2 — the first cut is unconstrained MO (the AGE-MOEA-2 Layer-2 scope).

## Validation

- **Behavioral** (`test/OptunaSharp.Tests.Interop`, `[TestCategory("RequiresPython")]`, self-skip
  Inconclusive without pymoo; hard-fail under `OPTUNASHARP_PYTHONNET_REQUIRE_ENGINE=1`): each algorithm
  runs a 2–3-objective study, all trials complete, generation bookkeeping is correct, reproducible under
  a seed, and it drives a spread front (HV improves over generation 0) — the `AgeMoea2Sampler` bar.
- **Value-parity vs live pymoo** on a few seeded tasks: assert the bridge's front matches
  `pymoo.minimize(algorithm, …, seed)` in achieved objective values / final population (**not** bit-exact
  — the Optuna↔pymoo search-space transform + per-trial doling differ from a monolithic `minimize`;
  bit-exactness is Layer 1's job, [09](09-pymoo-live-parity-gate.md)). Where the transform is identity
  (a pure `[0,1]^d` float space) and pop_size divides the trial budget, tighten to near-exact.
- **Leak gate** returns to baseline (the `pythonic` DoD): no `PyObject` outlives its `using`, cached
  modules never disposed.

## Out of scope

- pymoo's single-objective / gradient / pattern-search algorithms (GA, DE, PSO, CMAES, NelderMead,
  PatternSearch) — PSO/DE baselines are the *native* [07](07-moead-rvea-pso-baselines.md) / JANUS DE-host
  workstream; add them to the bridge later only if a benchmark needs the genuine pymoo variant.
- Performance indicators (HV/IGD/…) and test problems (ZDT/DTLZ/WFG) — deliberately deferred (the
  benchmarking-facing buckets); revisit for VolleyEval if [01](01-bbob-coco-benchmark-suite.md) wants the
  pymoo reference fronts.
- Byte-exact parity with `pymoo.minimize` through the sampler (structurally impossible via Optuna ask/tell
  — that guarantee lives in native Layer-1 optimizers like `AgeMoea2Optimizer`).

## Implementation outcome (2026-09-10)

Shipped as designed, with these findings that refined the design:

- **The zoo is 11, not 13** — `MOEA/D` and `DNSGA-II` are excluded (typed out of the enum, documented),
  because their pymoo 0.6.x ask/tell contract is incompatible with a static, externally-evaluated bridge:
  - **MOEA/D** is a *steady-state* generator: the first `ask()` returns the whole population but every
    later `ask()` yields a single `Individual` (a generation advances only every `pop_size` steady tells)
    — it does not fit the uniform full-population batch replay the other 11 share. Its decomposition
    regime stays the target of a future *native* Layer-1 MOEA/D port ([07](07-moead-rvea-pso-baselines.md)).
  - **DNSGA-II** ("dynamic" NSGA-II) re-samples and **re-evaluates part of the population through the
    problem** on each `tell` (`self.initialization.sampling(self.problem, n)` + an evaluator call). The
    bridge's `Problem` is a bare box with no `_evaluate`, so DNSGA-II has nothing to call — it is the
    wrong tool for static MO. (Confirmed by the exact crash `'numpy.ndarray' object is not callable`.)
- **pymoo cannot own generation 0** through Optuna's define-by-run interface (the search space is unknown
  until ≥1 trial completes). So — exactly like the native `NSGAIISampler` / `AgeMoea2Sampler` — generation
  0 is the independent `RandomSampler`, and pymoo is then **rebuilt each `SampleRelative` with generation 0
  injected as its initial population** (`sampling=<gen0 X>`). This makes the tell-replay coherent
  (pymoo's population is exactly the trials whose objectives it is told) and every later generation is
  pymoo's genuine offspring. Verified: a rebuild-each-generation replay reproduces a single continuous
  pymoo run bit-for-bit.
- **`pop_size` is read from pymoo, never computed** (a two-phase build: build once to read the authoritative
  `pop_size`, gate on generation-0 completion, then rebuild with `sampling=`). Necessary because the
  reference-direction / reference-point families derive it non-obviously (e.g. `RNSGA3(pop_per_ref_point=4,
  2 ref points)` resolves to `pop_size=10`, not 8).
- **No independent-sampling warning** (unlike the PyCma ctor shape): with a random generation-0 startup,
  warning on every independent sample would storm the log during startup. The bridge matches the native
  GA-sampler idiom (`NSGAIISampler` / `AgeMoea2Sampler`) — it delegates to the independent sampler
  silently. (`warnIndependentSampling` was dropped from the ctor.)
- **Value-parity is tight, not loose.** The lossy transform seam is only the high-boundary clip of
  Optuna's half-open float interval (`min(x, nextafter(1,0))`); applying the same clip on the Python side
  makes the bridge reproduce a direct pymoo `ask`/`tell` run (same seed + injected generation 0) to
  < 1e-9 on every trial's params and objectives — effectively bit-exact for a pure unit-cube float space.
