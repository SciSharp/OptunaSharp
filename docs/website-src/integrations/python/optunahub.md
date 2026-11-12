# OptunaHub

The OptunaHub bridge loads a Python registry module and adapts its sampler, pruner or benchmark to
OptunaSharp. Your C# study remains responsible for trials, objective execution and storage. The loaded
Python package supplies the selected algorithm or benchmark evaluation.

## Install

```shell
dotnet add package OptunaSharp.Interop.pythonnet
python -m pip install optuna optunahub
```

Configure [the Python runtime](pythonnet.md), put `git` on `PATH` for the default registry download
path, and install the selected registry package's additional requirements. Loading a registry module
does not automatically make all of its optional libraries available.

## Load a sampler

The registry's random-search package is a small example of the complete bridge:

```csharp
using System;
using OptunaSharp;
using OptunaSharp.Interop.PythonNet;

using var module = OptunaHub.LoadModule("samplers/random_search");
using var sampler = module.CreateSampler("RandomSampler", new() { ["seed"] = 0 });
var study = Optuna.CreateStudy(sampler: sampler);

study.Optimize(trial =>
{
    double x = trial.SuggestFloat("x", -10, 10);
    return (x - 2) * (x - 2);
}, nTrials: 30, nJobs: 1);

Console.WriteLine(study.BestValue);
```

`CreateSampler(className, kwargs)` instantiates the named Python class. Constructor argument names
remain Python names, such as `seed` or `n_startup_trials`; the dictionary uses ordinary CLR scalar
values. Seed the Python sampler through these arguments. The bridge does not add an independent seed.

`LoadModule(package, repoOwner, repoName, gitRef, forceReload, baseUrl)` defaults to the
`optuna/optunahub-registry` repository at `main`. Supply a tag or commit in `gitRef` when a fixed
registry revision matters. `forceReload: true` asks OptunaHub to re-download instead of using its cache.
The repository and base URL options are passed to Python OptunaHub for other registries.

`LoadLocalModule(package, registryRoot)` reads a checkout whose root contains the `package/` tree:

```csharp
using var local = OptunaHub.LoadLocalModule(
    "samplers/random_search", @"C:\repos\optunahub-registry");
```

This local path requires no registry download; the package's imports must still be installed.

## Benchmark problems

A benchmark wrapper reads `Directions` and `SearchSpace` eagerly, preserving Python declaration order.
Its objective first calls C# `SuggestFloat`, `SuggestInt` or `SuggestCategorical` for each parameter,
then passes the resulting dictionary to Python `evaluate(params)`.

With OptunaHub 0.5.0 or newer installed, the objective does what `BaseProblem.__call__` does between those
two steps: it calls the problem's `evaluate_constraints(params)` and records each value on the trial with
`trial.SetConstraint(name, value)`, before `evaluate` runs. Constraint-aware samplers then see each trial's
feasibility through `FrozenTrial.Constraints`, with no `constraints_func`. A trial is feasible when every
value is zero or less. A benchmark package written for an older OptunaHub returns a list instead of a
dictionary: its values are named `"0"`, `"1"`, … and OptunaHub's own `FutureWarning` is raised, exactly as
in Python. With an OptunaHub older than 0.5.0, nothing is recorded, because that version's `__call__` never
applied constraints.

For example, after installing the requirements of `benchmarks/bbob`:

```csharp
using System;
using OptunaSharp;
using OptunaSharp.Interop.PythonNet;

using var module = OptunaHub.LoadModule("benchmarks/bbob");
using var problem = module.CreateProblem("Problem", new()
{
    ["function_id"] = 1,
    ["dimension"] = 2
});

var study = Optuna.CreateStudy(directions: problem.Directions);
study.Optimize(problem.SingleObjective, nTrials: 50, nJobs: 1);
Console.WriteLine(study.BestValue);
```

Use `problem.Objective` for multiple objectives and `study.BestTrials` for its Pareto front.
`SingleObjective` rejects a problem with more than one direction. Only float, integer and categorical
distributions are supported; a custom Python distribution is not automatically adapted. A problem's
search space is captured at creation, so a benchmark that changes its space during evaluation needs
a more specific adapter.

## Pruning

`CreatePruner` returns an OptunaSharp `BasePruner`. Continue reporting intermediate values and raising
`TrialPruned` in the C# objective:

```csharp
using System;
using OptunaSharp;
using OptunaSharp.Interop.PythonNet;

using var module = OptunaHub.LoadModule("pruners/median");
using var pruner = module.CreatePruner("MedianPruner", new()
{
    ["n_startup_trials"] = 3,
    ["n_warmup_steps"] = 1
});
var study = Optuna.CreateStudy(pruner: pruner);
study.Optimize(trial =>
{
    double x = trial.SuggestFloat("x", -10, 10);
    double target = (x - 2) * (x - 2);
    double loss = 0;
    for (int step = 0; step < 10; step++)
    {
        loss = target + (50 - target) * Math.Exp(-0.5 * (step + 1));
        trial.Report(loss, step);
        if (trial.ShouldPrune())
            throw new TrialPruned();
    }
    return loss;
}, nTrials: 30, nJobs: 1);
```

The pruner keeps its own shadow study, described below, so a stateful pruner sees what it wrote earlier:
successive halving and Hyperband record each rung a trial completes and compare it with the other
trials' records, and they prune as in Python. Pruning is single-objective.

## Compatibility and state

The sampler and the pruner each keep a **shadow study**: one in-memory Python study per OptunaSharp
study, kept in lockstep with it and handed to the Python object. Before every call the bridge mirrors
each OptunaSharp trial, in number order and in whatever state it is: its values, parameters,
distributions, user and system attributes and intermediate values, then the study's attributes. The
sampler forwards all five hooks, `before_trial`, `infer_relative_search_space`, `sample_relative`,
`sample_independent` and `after_trial`, with the trial objects Optuna itself passes. It clears the
shadow study's per-trial history cache as `Study.ask` does, so CMA-ES, TPE and the GA samplers read the
history as of the current trial. The shadow study carries the OptunaSharp study's name. Attributes the
Python object writes, such as a generation counter, a swarm, constraints or a pruning rung, are written
to the OptunaSharp trial or study as well. `ReseedRng` delegates to Python `reseed_rng` when present.

Measured against the same objects run natively: PSO, MO-CMA, the NSGA-II variants
(`nsgaii_with_initial_trials`, `speaii`), the implicit natural gradient, differential evolution,
restart CMA-ES, Lévy flight, Nelder-Mead, `tpe_union_multivariate`, random search and the Optuna team's
`AutoSampler`, and the successive-halving, Hyperband, median, percentile, Wilcoxon and threshold pruners,
give the same trials bit for bit
([OptunaHubNativeParityTests](https://github.com/SciSharp/OptunaSharp/blob/main/test/OptunaSharp.Tests.Interop/OptunaHubNativeParityTests.cs)).

Remaining limits:

- The Python object's internal state lives in Python, not in OptunaSharp's storage. Persisting the C#
  trials alone does not resume every registry sampler identically after a restart.
- Attribute values with no JSON shape, such as NumPy arrays or arbitrary objects, stay in the shadow
  study and are not written to the C# storage.
- Calls are serialized on the Python thread. Trials of parallel workers are mirrored with the state the
  storage holds when a call starts, as a native sampler reads them, but each Python call sees one shared
  history cache where native Optuna keeps one per thread.
- A study that the C# `HyperbandPruner` filters into brackets is mirrored whole.

The bridge is not a shared-database connection between Python Optuna and OptunaSharp: the shadow study
lives in the process and mirrors one OptunaSharp study.

## Lifetime and validation

Dispose modules, samplers, pruners and problems after their last use, as in the examples. Their
`Dispose` methods acquire the GIL. `module.HasAttr(name)` can check a class name; `module.WithRaw(body)`
provides advanced access inside a GIL scope. Its borrowed module must not be disposed or allowed to
escape the callback.

The existing [bridge tests](https://github.com/SciSharp/OptunaSharp/blob/main/test/OptunaSharp.Tests.Interop/OptunaHubBridgeParityTests.cs)
exercise a local mixed-distribution toy benchmark, the registry's `RandomSampler`, and a median pruner
that actually prunes trials. The
[native-parity tests](https://github.com/SciSharp/OptunaSharp/blob/main/test/OptunaSharp.Tests.Interop/OptunaHubNativeParityTests.cs)
run 12 registry samplers and 6 pruners natively and through the bridge, and compare every trial. The
registry holds more methods than they cover, so validate a method outside that list before relying on
it. The tests report inconclusive when Python, a required package or the pinned registry checkout is
missing.

Source: [module factories](https://github.com/SciSharp/OptunaSharp/blob/main/src/OptunaSharp.Interop.pythonnet/OptunaHubModule.cs),
[sampler](https://github.com/SciSharp/OptunaSharp/blob/main/src/OptunaSharp.Interop.pythonnet/OptunaHubSampler.cs),
[pruner](https://github.com/SciSharp/OptunaSharp/blob/main/src/OptunaSharp.Interop.pythonnet/OptunaHubPruner.cs),
[shadow study](https://github.com/SciSharp/OptunaSharp/blob/main/src/OptunaSharp.Interop.pythonnet/OptunaHubShadow.cs),
[problem](https://github.com/SciSharp/OptunaSharp/blob/main/src/OptunaSharp.Interop.pythonnet/OptunaHubProblem.cs).
