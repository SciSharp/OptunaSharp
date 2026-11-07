# OptunaSharp.Interop.Pymoo

**`PymooSampler`** — a native OptunaSharp `BaseSampler` that drives the **real
[pymoo](https://pymoo.org)** in-process via [Python.NET](https://github.com/pythonnet/pythonnet).
One sampler exposes pymoo's whole multi-objective evolutionary-algorithm zoo through a
pythonnet-free `PymooAlgorithm` enum, while the Optuna side — study, storage, ask/tell, search
space, direction flipping, seeding — stays native OptunaSharp.

It is a **Tier X satellite** of the [SciSharp STACK](https://github.com/SciSharp): it ships
separately from `OptunaSharp` (Core) because it takes a foreign-runtime dependency (an embedded
CPython + pymoo). Core stays pure managed. It is the sibling of `OptunaSharp.Interop.BoTorch` /
`OptunaSharp.Interop.PyCma`, built on `OptunaSharp.Interop.pythonnet`.

> **Beyond Optuna core.** Optuna ships only NSGA-II and NSGA-III. This bridge delivers the rest of
> the literature's canonical multi-objective optimizers *now*, faithfully, at wrapper cost — the
> live-runtime complement to OptunaSharp's *native* `AgeMoea2Sampler`.

## Install

```bash
dotnet add package OptunaSharp.Interop.Pymoo
```

and, into a CPython that the host process can discover (`pip install pymoo` — nothing else; you do
**not** need any Python glue package):

```bash
pip install pymoo
```

Pin a specific interpreter with `PYTHONNET_PYDLL=/path/to/libpython3.x.{dll,so,dylib}` if several
are installed.

## Algorithms (`PymooAlgorithm`)

| Value | pymoo class | Needs (via `PymooAlgorithmOptions`) |
|-------|-------------|-------------------------------------|
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

> **MOEA/D and DNSGA-II are intentionally absent** — their pymoo ask/tell contract is incompatible
> with a static, externally-evaluated bridge:
> - **MOEA/D** exposes a *steady-state* ask/tell generator (its first `ask()` returns the whole
>   population, then one `Individual` per step), incompatible with the uniform full-population batch
>   replay every algorithm above shares. Its decomposition regime (Tchebycheff / PBI) is the target of
>   a future *native* Layer-1 MOEA/D port.
> - **DNSGA-II** ("dynamic" NSGA-II) is built for *time-varying* objectives: on each `tell` it
>   re-samples and **re-evaluates part of the population through the problem**. This bridge evaluates
>   the objective on the OptunaSharp side (the pymoo `Problem` is a bare box), so DNSGA-II has nothing
>   to call — the wrong tool for static multi-objective optimization.

## Usage

```csharp
using OptunaSharp;
using OptunaSharp.Interop.Pymoo;

// A 2-objective study driven by SMS-EMOA (hypervolume survival).
var study = Optuna.CreateStudy(
    directions: new[] { StudyDirection.Minimize, StudyDirection.Minimize },
    sampler: new PymooSampler(PymooAlgorithm.SmsEmoa, populationSize: 40, seed: 0));

study.Optimize(t =>
{
    double x = t.SuggestFloat("x", 0, 1);
    double y = t.SuggestFloat("y", 0, 1);
    return new[] { x, (1 - x) + y };   // a simple trade-off front
}, nTrials: 400);

foreach (var t in study.BestTrials)
    Console.WriteLine(string.Join(", ", t.Values));
```

A reference-direction algorithm (population size follows the directions):

```csharp
var sampler = new PymooSampler(
    PymooAlgorithm.Nsga3,
    seed: 0,
    options: new PymooAlgorithmOptions { NumberOfPartitions = 12 });   // 13 directions for 2 objectives
```

A reference-point algorithm:

```csharp
var sampler = new PymooSampler(
    PymooAlgorithm.RNsga2,
    populationSize: 40,
    seed: 0,
    options: new PymooAlgorithmOptions
    {
        ReferencePoints = new[] { new[] { 0.3, 0.7 }, new[] { 0.7, 0.3 } },
    });
```

Custom genetic operators (SBX crossover + PM mutation), honored by every algorithm:

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

Constrained optimization (pymoo's `G ≤ 0`, matching Optuna's constraint sign) — either call
`Trial.SetConstraint(...)` in the objective, or pass a `constraintsFunc`:

```csharp
var sampler = new PymooSampler(
    PymooAlgorithm.Nsga2, populationSize: 40, seed: 0,
    constraintsFunc: t => new[] { Convert.ToDouble(t.Params["x"]) - 0.5 });   // feasible when x <= 0.5
```

## Configuration (`PymooAlgorithmOptions`)

- **Genetic operators (all algorithms):** `CrossoverEta` / `CrossoverProbability` (SBX),
  `MutationEta` / `MutationProbability` (PM), `EliminateDuplicates`. A `null` eta/prob keeps pymoo's
  own default.
- **RVEA:** `RveaAlpha`, `RveaAdaptFrequency`.
- **Reference directions (NSGA-III, U-NSGA-III, RVEA, C-TAEA):** `ReferenceDirectionsMethod`
  (Das–Dennis / Riesz energy), `NumberOfPartitions`, `NumberOfReferenceDirections`.
- **Reference points (R-NSGA-II, R-NSGA-III):** `ReferencePoints`, and R-NSGA-II's `Epsilon` /
  `Weights` / `ExtremePointsAsReferencePoints`, R-NSGA-III's `PopulationPerReferencePoint` / `Mu`.

> pymoo's `n_offsprings` is deliberately **not** exposed: setting it below `pop_size` makes an `ask()`
> return fewer than a full population, which is incompatible with the bridge's uniform full-population
> generation replay.

## How it works

pymoo has a native `ask()` / `tell()` interface, so `PymooSampler` uses the same
**reconstruct-from-trials** bridge as `PyCmaSampler`: inside each `SampleRelative`, it builds the
pymoo algorithm fresh, seeds it, **replays** the study's completed trials generation by generation
(`ask()` the infill population → stamp its objectives `F` (and constraints `G`) from the trials'
stored values → `tell()`), then `ask()`s once more and hands out one individual's parameters for the
current trial — the same one-row-per-trial doling as the native `NSGAIISampler` / `AgeMoea2Sampler`.
Nothing pymoo-side persists between trials; the study storage is the single source of truth, which
keeps the sampler correct under resumption and reproducible under a fixed seed.

Like every relative sampler under Optuna's define-by-run model (and the native `NSGAIISampler` /
`AgeMoea2Sampler`), the first `population_size` trials — **generation 0** — are sampled by the
independent sampler (a `RandomSampler` by default), because the search space is not known until a
trial has completed. Once generation 0 is complete, pymoo takes over: it is rebuilt with that
generation **injected as its initial population** (`sampling=`), so the tell-replay is coherent and
every later generation is pymoo's genuine offspring. The independent sampler also covers parameters
*outside* the relative search space (single-value distributions and conditional/dynamic parameters).
Parameters cross to pymoo's continuous unit cube via the bit-exact one-hot `SearchSpaceTransform`
(float/int with log/step scale to `[0,1]`; categoricals are one-hot encoded), and the `X`/`F`/`G`
matrices cross zero-copy through NumSharp's `NDArray` ⇄ numpy interop.

## Threads and macOS stacks

The sampler's pymoo calls run on the runtime's Python thread (`PythonRuntime`, from
`OptunaSharp.Interop.pythonnet`), so the sampler can be driven from any thread and at any `nJobs`.
Before that, driving it off the main thread crashed the process with an access violation in scipy's
`cdist`: 7 of 8 runs, measured on 2026-11-07.

`AgeMoea` and `AgeMoea2` run pymoo's numba-compiled survival kernels, and the first `tell` JIT-compiles
them.

- On macOS arm64 that compilation needs more stack than a thread .NET creates has by default.
- The Python thread has 16 MB, CPython's own size, so the sampler needs no remedy.
- Running such pymoo code yourself on a .NET thread (the thread pool, `Task.Run`, an ASP.NET request)
  still needs a thread with a `maxStackSize` of at least 16 MB, or `DOTNET_DefaultStackSize=1000000`
  (hexadecimal, 16 MB) set before the process starts. OptunaSharp's own `nJobs` threads reserve 16 MB.
- Windows x64 (measured down to 256 KB) and Linux x64 (default stacks) are not affected.

## Scope of the parity guarantee

Validated **behaviorally** (every algorithm completes a seeded study across 1-, 2- and 3-objective
problems and Optuna's full parameter space — float/int with log/step, categoricals, dynamic spaces —
drives a spreading front, and reproduces across runs) and by **value-parity vs a direct live pymoo
`ask`/`tell` run** seeded with the same generation 0 (including custom operators and constraints): the
bridge reproduces that run to `< 1e-9` on every trial's params and objectives — effectively bit-exact
for a unit-cube float space, the only seam being the high-boundary clip of Optuna's half-open float
interval. It is **not** bit-exact with a monolithic `pymoo.minimize` through the sampler, because
generation 0 uses OptunaSharp's RNG rather than pymoo's own initial sampling. Bit-exactness against a
monolithic run is the job of the *native* Layer-1 optimizers (e.g. `AgeMoea2Optimizer`).

---

Part of the SciSharp STACK. Licensed under the MIT License.
