# pycma

`OptunaSharp.Interop.PyCma` provides `PyCmaSampler`, a C# sampler that calls the Python `cma` package's
`CMAEvolutionStrategy` and `BoundTransform`. The numerical CMA-ES implementation runs in Python;
OptunaSharp owns the study, trial history and parameter conversion.

This is the compatibility bridge for Python's `optuna_integration.PyCmaSampler`. Its constructor
emits the upstream deprecation warning introduced in 4.9.0 with removal scheduled for 6.0.0. The
managed `OptunaSharp.Samplers.CmaEsSampler` instead ports the separate `cmaes` implementation and is
the core alternative. The two backends should not be expected to return identical suggestions.

## Install and optimize

```shell
dotnet add package OptunaSharp.Interop.PyCma
python -m pip install cma
```

The package targets .NET 8 and .NET 10, brings the shared [Python.NET runtime](pythonnet.md)
transitively, and needs a discoverable CPython containing `cma`. It does not require the Python
`optuna_integration` package at application runtime.

```csharp
using System;
using OptunaSharp;
using OptunaSharp.Interop.PyCma;

var study = Optuna.CreateStudy(sampler: new PyCmaSampler(seed: 42));
study.Optimize(trial =>
{
    double x = trial.SuggestFloat("x", -5, 5);
    double y = trial.SuggestFloat("y", -5, 5);
    return x * x + y * y;
}, nTrials: 50, nJobs: 1);

Console.WriteLine(study.BestValue);
```

Use at least two varying parameters. A one-dimensional relative space falls back to the independent
sampler; a multi-objective study raises an error. Construction imports `cma` immediately.

## Configure the optimizer

| Argument | Behavior |
|---|---|
| `x0` | Initial values keyed by parameter name, in the objective's external representation |
| `sigma0` | Initial global step size in CMA coordinates |
| `cmaStds` | Per-parameter multipliers of the global step size; unspecified entries use 1 |
| `seed` | Seeds CMA-ES and the default independent sampler |
| `cmaOpts` | Additional Python `CMAEvolutionStrategy` options, such as `popsize` or `verbose` |
| `nStartupTrials` | Completed-trial threshold before CMA sampling; default 1 |
| `independentSampler` | Startup and out-of-intersection sampler; seeded `RandomSampler` by default |
| `warnIndependentSampling` | Controls the fallback warning; default `true` |

```csharp
using System.Collections.Generic;
using OptunaSharp.Interop.PyCma;

var sampler = new PyCmaSampler(
    x0: new Dictionary<string, object?> { ["x"] = 0.0, ["y"] = 0.0 },
    sigma0: 0.5,
    cmaStds: new Dictionary<string, double> { ["x"] = 1.0, ["y"] = 2.0 },
    cmaOpts: new Dictionary<string, object?> { ["popsize"] = 8, ["verbose"] = -2 },
    seed: 42);
```

Without `x0`, numeric distributions use their midpoint, with log distributions using a midpoint in
log space; categoricals start from a middle choice. Without `sigma0`, the sampler uses the smallest
transformed parameter range divided by six, with a small positive floor to avoid division by zero.
Provide `x0` for every parameter in the relative space when overriding it.

The sampler supplies `seed`, `BoundaryHandler` and `bounds` to `cmaOpts` itself; these values replace
conflicting dictionary entries. Use the typed `cmaStds` argument for per-parameter scaling, which
overrides `CMA_stds` when supplied. Additional option values should be Python.NET-convertible
primitives. `verbose` defaults to `-2` when absent. There is no generic Python callable option API.

## Parameter mapping

Unlike the native `CmaEsSampler`, this bridge keeps categoricals in the relative search space and
maps each choice to an ordinal coordinate. A category is recovered by rounding to its choice index;
the ordering of choices therefore participates in the search geometry.

Log distributions are optimized in log space. Stepped floats use an offset coordinate, while integer
and stepped values are rounded back onto their grids. Single-value distributions are skipped, and
conditional or changing parameters outside the intersection use the independent sampler. The
preserved upstream fallback warning mentions categorical distributions broadly, but the actual
relative-sampling implementation supports ordinal categoricals and has a dedicated parity test.

## Replay, persistence and concurrency

On each relative suggestion the sampler creates a fresh Python optimizer with its seed, reconstructs
completed generations from compatible stored trials, performs `ask`/`tell` for those generations, and
asks for the next candidate. Maximizing objective values are negated before `tell`, because CMA-ES
minimizes. Only full completed generations update the optimizer; an unfinished generation determines
which candidate is returned next.

The Python optimizer is disposed within the same GIL scope. No live CMA optimizer is retained between
trials, and `PyCmaSampler` does not implement `IDisposable`. Its managed seed, initial mean,
intersection cache and independent sampler are still stateful: use a separate instance per study.

The replay code accounts for compatible trials in other states when choosing a candidate index and
can advance extra `ask` calls when that index exceeds a population. This is not a promise of an
identical trajectory under asynchronous completion. Use fixed configuration, a fixed seed and
`nJobs: 1` for reference comparisons. Persisted study history is used for replay; arbitrary changes to
the configuration or independent sampler state on restart do not reproduce an uninterrupted run.

## Validation and scope

The [live parity tests](https://github.com/SciSharp/OptunaSharp/blob/main/test/OptunaSharp.Tests.Interop/PyCmaSamplerParityTests.cs)
compare against Python `optuna_integration.PyCmaSampler`: two- and three-dimensional spheres, mixed
float/integer spaces, categoricals, logarithmic and stepped parameters, maximization, startup counts,
explicit means/scales, population sizes and reproducibility. They also check the one-dimensional
fallback and multi-objective rejection. The fixture reports inconclusive when `cma` or the Python
reference integration is unavailable.

These seeded comparisons validate the tested configurations on the same backend. They do not claim
that pycma equals `cmaes`, that Python version upgrades preserve every trajectory, or that the sampler
provides native AOT support. There is no constraint callback or multi-objective extension on this
sampler's public API.

Source: [sampler and generation replay](https://github.com/SciSharp/OptunaSharp/blob/main/src/OptunaSharp.Interop.PyCma/PyCmaSampler.cs),
[Python crossing](https://github.com/SciSharp/OptunaSharp/blob/main/src/OptunaSharp.Interop.PyCma/PyCmaInterop.cs).
