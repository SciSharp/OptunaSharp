# pymoo Integration

[pymoo](https://pymoo.org) is the reference Python library for multi-objective evolutionary
optimization — the home of NSGA-III, RVEA, SMS-EMOA, C-TAEA and the rest of the literature's canonical
many-objective algorithms. **`OptunaSharp.Interop.Pymoo`** exposes that whole zoo as a first-class
OptunaSharp sampler: one `PymooSampler` drives the **real pymoo** in-process (via
[Python.NET](https://github.com/pythonnet/pythonnet)), while the Optuna side — study, storage,
ask/tell, search space, direction flipping, seeding — stays native OptunaSharp.

It is a **satellite package**: it ships separately from the `OptunaSharp` core because it takes a
foreign-runtime dependency (an embedded CPython + pymoo). Core stays pure-managed. It is the sibling of
`OptunaSharp.Interop.BoTorch` and `OptunaSharp.Interop.PyCma`, built on `OptunaSharp.Interop.pythonnet`
— the same *reconstruct-from-trials* bridge.

> **Beyond Optuna core.** Optuna itself ships only NSGA-II and NSGA-III. This bridge delivers the rest
> of pymoo's multi-objective optimizers *now*, faithfully, at wrapper cost — the live-runtime
> complement to OptunaSharp's [native](methods/index.md) `AgeMoea2Sampler` / `SmsEmoaSampler`.

## Install

Add the NuGet package:

```bash
dotnet add package OptunaSharp.Interop.Pymoo
```

and install pymoo into a CPython the host process can discover — **nothing else**; you do *not* need
any `optuna_integration`-style glue package:

```bash
pip install pymoo
```

The bridge is developed and tested against **pymoo 0.6.1.3**; pin that line (`pip install
"pymoo==0.6.1.*"`) if you want to match the validated behavior exactly. If several interpreters are
installed, point pythonnet at the right one with
`PYTHONNET_PYDLL=/path/to/libpython3.x.{dll,so,dylib}`. The `PymooSampler` constructor imports pymoo
up front and fails fast with a `pip install pymoo` message if it is missing.

## The algorithm zoo

Select the algorithm with the pythonnet-free `PymooAlgorithm` enum; tune the rest with
`PymooAlgorithmOptions`.

| `PymooAlgorithm` | pymoo class | Needs (via `PymooAlgorithmOptions`) |
|------------------|-------------|-------------------------------------|
| `Nsga2` | `NSGA2` | — (`populationSize`) |
| `Nsga3` | `NSGA3` | reference directions (population size derived) |
| `RNsga2` | `RNSGA2` | `ReferencePoints` |
| `RNsga3` | `RNSGA3` | `ReferencePoints` (population size derived) |
| `UNsga3` | `UNSGA3` | reference directions (population size derived) |
| `AgeMoea` | `AGEMOEA` | — (`populationSize`) |
| `AgeMoea2` | `AGEMOEA2` | — (`populationSize`) |
| `Rvea` | `RVEA` | reference directions (population size derived) |
| `SmsEmoa` | `SMSEMOA` | — (`populationSize`) |
| `Spea2` | `SPEA2` | — (`populationSize`) |
| `Ctaea` | `CTAEA` | reference directions (population size derived) |

Eleven algorithms — every member of `pymoo.algorithms.moo` whose ask/tell contract fits a static,
externally-evaluated bridge. (The four that don't are listed under
[Algorithms not exposed](#algorithms-not-exposed).) For the **reference-direction** algorithms
(`Nsga3`, `UNsga3`, `Rvea`, `Ctaea`) and the **reference-point** algorithms (`RNsga2`, `RNsga3`) the
population size is *derived* by pymoo from the directions / points, so the `populationSize` argument is
ignored there.

## Quick start

A two-objective study driven by SMS-EMOA (hypervolume-based survival):

```csharp
using OptunaSharp;
using OptunaSharp.Interop.Pymoo;

var study = Optuna.CreateStudy(
    directions: new[] { StudyDirection.Minimize, StudyDirection.Minimize },
    sampler: new PymooSampler(PymooAlgorithm.SmsEmoa, populationSize: 40, seed: 0));

study.Optimize(t =>
{
    double x = t.SuggestFloat("x", 0, 1);
    double y = t.SuggestFloat("y", 0, 1);
    return new[] { x, (1 - x) + y };   // a simple trade-off front
}, nTrials: 400);

foreach (FrozenTrial t in study.BestTrials)
    Console.WriteLine(string.Join(", ", t.Values!));
```

Everything the rest of OptunaSharp offers around a study applies unchanged: storage, naming objectives
(`study.SetMetricNames`), and the multi-objective [visualizations](visualization.md) (`ParetoFront`,
`HypervolumeHistory`). See [Multi-objective Optimization](multi-objective.md) for the study-side API.

## Reference-direction algorithms

NSGA-III, U-NSGA-III, RVEA and C-TAEA are steered by a set of **reference directions**, and pymoo
derives the population size from them. Generate them with Das–Dennis (the structured simplex lattice,
the default) or Riesz *s*-energy:

```csharp
var sampler = new PymooSampler(
    PymooAlgorithm.Nsga3,
    seed: 0,
    options: new PymooAlgorithmOptions
    {
        ReferenceDirectionsMethod = PymooReferenceDirections.DasDennis,
        NumberOfPartitions = 12,   // 13 directions for 2 objectives, 91 for 3, ...
    });
```

RVEA additionally honors its penalty-growth `RveaAlpha` and adaptation frequency
`RveaAdaptFrequency`. For Riesz energy, set `ReferenceDirectionsMethod =
PymooReferenceDirections.Energy` and `NumberOfReferenceDirections` (reproducible under the sampler's
seed).

## Reference-point algorithms

R-NSGA-II and R-NSGA-III focus the search around **aspiration points** you supply, one row per point in
the study's own objective space (the sampler applies the minimization sign flip for you):

```csharp
var sampler = new PymooSampler(
    PymooAlgorithm.RNsga2,
    populationSize: 40,
    seed: 0,
    options: new PymooAlgorithmOptions
    {
        ReferencePoints = new[] { new[] { 0.3, 0.7 }, new[] { 0.7, 0.3 } },
        Epsilon = 0.001,   // R-NSGA-II niche radius
    });
```

`ReferencePoints` is required for these two algorithms (there is no meaningful default) — omitting it
throws a clear `ValueError`. R-NSGA-II also reads `Weights` and `ExtremePointsAsReferencePoints`;
R-NSGA-III reads `PopulationPerReferencePoint` and `Mu`.

## Custom genetic operators

Every algorithm accepts simulated-binary crossover (SBX) and polynomial mutation (PM) — the standard
real-valued operators for the continuous unit cube the bridge optimizes in. Tune their distribution
indices and probabilities, or toggle duplicate elimination:

```csharp
var sampler = new PymooSampler(
    PymooAlgorithm.Nsga2,
    populationSize: 40,
    seed: 0,
    options: new PymooAlgorithmOptions
    {
        CrossoverEta = 30, CrossoverProbability = 0.9,   // SBX
        MutationEta = 20,  MutationProbability = 0.1,    // PM
        EliminateDuplicates = true,
    });
```

A `null` eta/probability keeps pymoo's own default (SBX `eta=15`/`prob=0.9`, PM `eta=20`/`prob=1/n`),
so the default path stays byte-identical to a stock pymoo run.

> pymoo's `n_offsprings` is deliberately **not** exposed: setting it below `pop_size` makes an `ask()`
> return fewer than a full population, which is incompatible with the bridge's uniform
> full-population generation replay.

## Constrained optimization

pymoo models inequality constraints as `G ≤ 0` (a value > 0 marks a violated constraint, ≤ 0 feasible)
— **the same sign as Optuna's** own `constraintsFunc`, so no flipping is needed. Supply the constraints
either by calling `Trial.SetConstraint(...)` in the objective or by passing a `constraintsFunc` to the
sampler:

```csharp
var sampler = new PymooSampler(
    PymooAlgorithm.Nsga2, populationSize: 40, seed: 0,
    constraintsFunc: t => new[] { Convert.ToDouble(t.Params["x"]) - 0.5 });   // feasible when x <= 0.5
```

pymoo then optimizes with feasibility-first (constrained-domination) survival, and **C-TAEA** — the
constrained two-archive EA — becomes meaningful. The constraint count is fixed by the first completed
trial, so the sampler is stateful under constraints; do not reuse one instance across studies.

## Configuration (`PymooAlgorithmOptions`)

All knobs live on one options object; each algorithm reads only the ones it needs.

- **Genetic operators (all algorithms):** `CrossoverEta` / `CrossoverProbability` (SBX),
  `MutationEta` / `MutationProbability` (PM), `EliminateDuplicates`.
- **RVEA:** `RveaAlpha`, `RveaAdaptFrequency`.
- **Reference directions (NSGA-III, U-NSGA-III, RVEA, C-TAEA):** `ReferenceDirectionsMethod`
  (Das–Dennis / Riesz energy), `NumberOfPartitions`, `NumberOfReferenceDirections`.
- **Reference points (R-NSGA-II, R-NSGA-III):** `ReferencePoints`; R-NSGA-II's `Epsilon` / `Weights` /
  `ExtremePointsAsReferencePoints`; R-NSGA-III's `PopulationPerReferencePoint` / `Mu`.

## How it works

pymoo exposes a native `ask()` / `tell()` interface, so `PymooSampler` uses the **reconstruct-from-
trials** bridge that `PyCmaSampler` proves. Inside each `SampleRelative` call (one GIL scope), it:

1. builds the chosen pymoo `Algorithm` fresh and seeds it;
2. **replays** the study's completed trials generation by generation — `ask()` the infill population,
   stamp its objective matrix `F` (and constraints `G`) from the trials' stored values, `tell()` it;
3. `ask()`s once more and hands out **one individual's parameters** for the current trial — the same
   one-row-per-trial doling as the native [`NSGAIISampler` / `AgeMoea2Sampler`](methods/index.md).

Nothing pymoo-side persists between trials; the study storage is the single source of truth, which keeps
the sampler correct under resumption and reproducible under a fixed seed (pymoo's `ask()` is
deterministic under its seed, so rebuilding-and-replaying reproduces the exact population it doled out).

Like every relative sampler under Optuna's define-by-run model, the first `population_size` trials —
**generation 0** — are sampled by an independent sampler (a `RandomSampler` by default), because the
search space is not known until a trial has completed. Once generation 0 is complete, pymoo takes over:
it is rebuilt with that generation **injected as its initial population** (pymoo's `sampling=`), so the
tell-replay is coherent and every later generation is pymoo's genuine offspring. The independent sampler
also covers parameters *outside* the relative search space (single-value and conditional / dynamic
parameters).

Parameters cross to pymoo's continuous unit cube via the bit-exact one-hot
[search-space transform](search-spaces.md): `float` / `int` (with `log` / `step`) scale to `[0, 1]`,
categoricals are one-hot encoded (pymoo's continuous operators move each column, the argmax on the way
back selects the choice). The `X` / `F` / `G` matrices cross zero-copy through NumSharp's
`NDArray` ⇄ numpy interop, never element by element.

## Scope of the parity guarantee

The bridge is validated two ways:

- **Behaviorally** — every algorithm completes a seeded study across 1-, 2- and 3-objective problems
  and Optuna's full parameter space (float / int with log / step, categoricals, dynamic spaces), drives
  a spreading Pareto front, and reproduces across runs under a fixed seed.
- **By value-parity vs a direct live pymoo `ask`/`tell` run** seeded with the same generation 0
  (including custom operators and constraints): the bridge reproduces that run to **`< 1e-9`** on every
  trial's parameters and objectives — effectively bit-exact for a unit-cube float space, the only seam
  being the high-boundary clip of Optuna's half-open float interval.

It is **not** bit-exact with a monolithic `pymoo.minimize` through the sampler, because generation 0
uses OptunaSharp's RNG rather than pymoo's own initial sampling, and the Optuna search-space transform
differs from a monolithic run. Bit-exactness against a monolithic reference is the job of the *native*
Layer-1 optimizers such as [`AgeMoea2Optimizer`](methods/age-moea-2.md).

## Algorithms not exposed

`pymoo.algorithms.moo` contains four more algorithms that the bridge does **not** expose, because their
ask/tell contract is incompatible with a static, externally-evaluated bridge (the objective is evaluated
on the OptunaSharp side; pymoo's `Problem` is a bare box with no `_evaluate`):

- **MOEA/D** (and its `ParallelMOEAD` variant) — a *steady-state* decomposition generator. Its first
  `ask()` returns the whole population, but every subsequent `ask()` yields a single `Individual`
  (a generation advances only every `pop_size` steady tells), which does not fit the uniform
  full-population batch replay every exposed algorithm shares. MOEA/D's decomposition regime
  (Tchebycheff / PBI) is instead the target of a future *native* `MoeaDSampler`.
- **DNSGA-II** and **KGB** — *dynamic* MOEAs built for time-varying objectives. On each `tell` they
  detect landscape changes and respond by re-sampling and **re-evaluating part of the population through
  the problem**. The bridge evaluates objectives on the OptunaSharp side, so these re-evaluation calls
  have nothing to run — the wrong tool for static multi-objective optimization.

pymoo's single-objective, gradient and pattern-search algorithms (GA, DE, PSO, CMA-ES, Nelder–Mead, …)
are out of scope by design; OptunaSharp's own [native beyond-core samplers](methods/index.md) (LM-CMA,
VkD-CMA, Differential Evolution, PSO, MOEA/D, RVEA, JANUS) cover that ground.

## Relationship to the native samplers

Two of the exposed algorithms have *native* OptunaSharp counterparts:
[`AgeMoea2Sampler`](methods/age-moea-2.md) and [`SmsEmoaSampler`](methods/sms-emoa.md). The native ones
are pure-managed (no Python), bit-exact against their reference, and belong in the `OptunaSharp` core;
the pymoo ones are the live-runtime complement — handy as an oracle and for the algorithms with no
native port yet. They are not guaranteed to agree trial-for-trial: for example, pymoo's `SMSEMOA` is
**generational** (a full offspring population each generation), whereas the native
[`SmsEmoaSampler`](methods/sms-emoa.md) is the classic **steady-state (μ+1)** variant.

## Relationship to Optuna

Optuna's Python ecosystem has no pymoo integration of its own — pymoo is normally used through its own
`minimize` API. This bridge is an OptunaSharp extension that makes pymoo's algorithms usable through the
ordinary [`Study`](study-and-trials.md) / [sampler](samplers.md) interface, exactly like any other
OptunaSharp sampler:

```csharp
var study = Optuna.CreateStudy(
    directions: new[] { StudyDirection.Minimize, StudyDirection.Minimize },
    sampler: new PymooSampler(PymooAlgorithm.Rvea, seed: 0,
        options: new PymooAlgorithmOptions { NumberOfPartitions = 12 }));
```

## See also

[Samplers](samplers.md) · [Multi-objective Optimization](multi-objective.md) ·
[Search Spaces](search-spaces.md) · [Visualization](visualization.md) ·
[Methods overview](methods/index.md) · the [API reference](../api/index.md).
