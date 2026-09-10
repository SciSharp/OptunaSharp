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

## How it works

pymoo has a native `ask()` / `tell()` interface, so `PymooSampler` uses the same
**reconstruct-from-trials** bridge as `PyCmaSampler`: inside each `SampleRelative`, it builds the
pymoo algorithm fresh, seeds it, **replays** the study's completed trials generation by generation
(`ask()` the infill population → stamp its objectives `F` from the trials' stored values →
`tell()`), then `ask()`s once more and hands out one individual's parameters for the current trial —
the same one-row-per-trial doling as the native `NSGAIISampler` / `AgeMoea2Sampler`. Nothing
pymoo-side persists between trials; the study storage is the single source of truth, which keeps the
sampler correct under resumption and reproducible under a fixed seed.

pymoo owns **every** generation (including generation 0, its initial sampling); there is no
random-startup phase. The independent sampler is used only for parameters *outside* the relative
search space (single-value distributions and conditional/dynamic parameters). Parameters cross to
pymoo's continuous unit-cube via the bit-exact one-hot `SearchSpaceTransform` (float/int with
log/step scale to `[0,1]`; categoricals are one-hot encoded).

## Scope of the parity guarantee

Validated **behaviorally** (every algorithm completes a seeded study, drives a spreading front, and
reproduces across runs) and by **value-parity vs live `pymoo.minimize`** (the achieved front matches
a monolithic minimize run on the same seed on a plain continuous box). It is **not** bit-exact with
`pymoo.minimize` through the sampler — the Optuna ask/tell path and the search-space transform differ
from a monolithic run. Bit-exactness is the job of the *native* Layer-1 optimizers (e.g.
`AgeMoea2Optimizer`).

---

Part of the SciSharp STACK. Licensed under the MIT License.
