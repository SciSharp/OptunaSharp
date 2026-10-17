# OptunaSharp.Interop.pythonnet

OptunaSharp's bridge to the **live Python Optuna ecosystem** via [Python.NET (pythonnet)](https://github.com/pythonnet/pythonnet) — most importantly **[OptunaHub](https://hub.optuna.org/)**, Optuna's community registry of black-box-optimization methods and benchmarks.

It is the .NET analog of `NumSharp.Interop.pythonnet` (which bridges `NDArray` ↔ NumPy): here, `OptunaHub.LoadModule("category/name")` mirrors `optunahub.load_module`, and the loaded contributor packages are wrapped so an OptunaSharp `Study` can drive them directly.

> **Core stays pure-managed.** This package takes a foreign-runtime dependency (an embedded CPython), so it ships separately — the `NumSharp.Interop.OpenBLAS` optional-package pattern.

## Prerequisites

A discoverable CPython with Optuna and OptunaHub:

```bash
pip install optuna optunahub
```

`git` on `PATH` (OptunaHub's default download path). Set `PYTHONNET_PYDLL` to your libpython if it can't be found automatically (e.g. `C:\...\python312.dll`, `/usr/lib/.../libpython3.12.so`).

## Benchmarks — optimize a real hub problem with any OptunaSharp sampler

```csharp
using OptunaSharp;
using OptunaSharp.Interop.PythonNet;

using var bbob = OptunaHub.LoadModule("benchmarks/bbob");
var problem = bbob.CreateProblem("Problem", new() { ["function_id"] = 1, ["dimension"] = 2 });

var study = Optuna.CreateStudy(directions: problem.Directions);   // Sphere is single-objective
study.Optimize(problem.SingleObjective, nTrials: 50);
Console.WriteLine(study.BestValue);
```

`problem.Directions` and `problem.SearchSpace` are exposed as OptunaSharp types; `problem.Objective`
(multi-objective) / `problem.SingleObjective` suggest each parameter on the C# trial, then call the
Python `evaluate`. With OptunaHub ≥ 0.5.0 they first record the problem's `evaluate_constraints(params)`
on the trial through `trial.SetConstraint`, as `BaseProblem.__call__` does, so constraint-aware samplers
see feasibility without a `constraints_func`.

## Samplers — drive an OptunaSharp study with a hub sampler

```csharp
using var nm = OptunaHub.LoadModule("samplers/nelder_mead");
using var sampler = nm.CreateSampler("NelderMeadSampler", new() { ["seed"] = 0 });

var study = Optuna.CreateStudy(sampler: sampler);
study.Optimize(t => {
    var x = t.SuggestFloat("x", -10, 10);
    return (x - 2) * (x - 2);
}, nTrials: 50);
```

The real Python sampler decides each suggestion; the bridge keeps a Python **shadow study** in sync
with the OptunaSharp study's finished trials and delegates
`infer_relative_search_space` / `sample_relative` / `sample_independent` to it. Designed for
single-worker `Study.Optimize` (`nJobs = 1`).

## Loading

- `OptunaHub.LoadModule(package, repoOwner, repoName, gitRef, forceReload, baseUrl)` — mirrors
  `optunahub.load_module`; `baseUrl`/`repoOwner`/`repoName` reach third-party registries.
- `OptunaHub.LoadLocalModule(package, registryRoot)` — mirrors `optunahub.load_local_module` against a
  local registry checkout (e.g. the pinned `refs/optunahub-registry` submodule; no network).

Part of the [SciSharp STACK](https://github.com/SciSharp).
