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

The pruner rebuilds a Python study for each decision, including finished trials' intermediate values
and the current trial's reports. Pruning is single-objective.

## Compatibility and state

The sampler maintains a **shadow study** for the current trial. On the next trial it rebuilds that
in-memory Python study from prior finished trials, opens a running trial, and delegates
`infer_relative_search_space`, `sample_relative` and `sample_independent`. `ReseedRng` delegates to
Python `reseed_rng` when present. The Python sampler instance survives across these rebuilt studies.

This is an intentionally limited adapter:

- Use one wrapper per study and `nJobs: 1`. Concurrent running trials are not mirrored.
- The sampler mirror copies finished states, objective values, parameters and distributions. It does
  not copy study/trial user attributes, system attributes, constraints or intermediate values.
- It does not forward Python sampler `before_trial` or `after_trial` hooks. Python mutations to the
  shadow study are not written back to C# storage.
- The pruner includes intermediate values, but likewise does not synchronize arbitrary metadata or
  Python storage writes.
- The Python sampler's internal state is not checkpointed by OptunaSharp storage. Persisting C# trials
  alone does not establish an identical restarted trajectory for every registry sampler.

Consequently, loading a module successfully is not evidence that every method in the registry is
compatible. Algorithms that depend on hooks, persistent storage attributes, concurrent trials or other
unmirrored state need additional adaptation and validation. The bridge is not a shared-database
connection between Python Optuna and OptunaSharp.

## Lifetime and validation

Dispose modules, samplers, pruners and problems after their last use, as in the examples. Their
`Dispose` methods acquire the GIL. `module.HasAttr(name)` can check a class name; `module.WithRaw(body)`
provides advanced access inside a GIL scope. Its borrowed module must not be disposed or allowed to
escape the callback.

The existing [bridge tests](https://github.com/SciSharp/OptunaSharp/blob/master/test/OptunaSharp.Tests.Interop/OptunaHubBridgeParityTests.cs)
exercise a local mixed-distribution toy benchmark, the registry's `RandomSampler`, and a median pruner
that actually prunes trials. These are behavioral integration checks, not a registry-wide parity
guarantee. They report inconclusive when Python, required packages or the pinned registry checkout
are missing.

Source: [module factories](https://github.com/SciSharp/OptunaSharp/blob/master/src/OptunaSharp.Interop.pythonnet/OptunaHubModule.cs),
[sampler](https://github.com/SciSharp/OptunaSharp/blob/master/src/OptunaSharp.Interop.pythonnet/OptunaHubSampler.cs),
[pruner](https://github.com/SciSharp/OptunaSharp/blob/master/src/OptunaSharp.Interop.pythonnet/OptunaHubPruner.cs),
[problem](https://github.com/SciSharp/OptunaSharp/blob/master/src/OptunaSharp.Interop.pythonnet/OptunaHubProblem.cs).
