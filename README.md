<p align="center">
  <a href="https://github.com/SciSharp/OptunaSharp">
    <img src="docs/website-src/images/optunasharp-logo.png" alt="OptunaSharp" height="120">
  </a>
</p>

<h1 align="center">OptunaSharp</h1>

<p align="center">
  <strong>Optuna for .NET</strong>
</p>

<p align="center">
  <a href="https://github.com/SciSharp/OptunaSharp/actions/workflows/build-and-release.yml"><img alt="Build" src="https://github.com/SciSharp/OptunaSharp/actions/workflows/build-and-release.yml/badge.svg"></a>
  <a href="https://github.com/SciSharp/OptunaSharp/actions/workflows/docs.yml"><img alt="Docs" src="https://github.com/SciSharp/OptunaSharp/actions/workflows/docs.yml/badge.svg"></a>
  <a href="https://www.nuget.org/packages/OptunaSharp"><img alt="NuGet" src="https://img.shields.io/nuget/v/OptunaSharp.svg"></a>
  <a href="LICENSE"><img alt="License" src="https://img.shields.io/badge/license-MIT-blue.svg"></a>
</p>

<p align="center">
  <strong>OptunaSharp</strong> is a faithful and complete .NET port of <a href="https://github.com/optuna/optuna">Optuna</a>, an automatic hyperparameter optimization software framework, particularly designed for machine learning. 
</p>

<p align="center">
  OptunaSharp is <strong>thread-safe and truly parallel.</strong><br>
  <code>nJobs</code> runs trials on all your cores at once, with no GIL in the way, and workers on <br>many machines share one study, Python Optuna workers included.<br>
 OptunaSharp supports parallelism powered by:<br> single-threaded, reused dedicated threads and ThreadPool async-await.
</p>

<p align="center">
  OptunaSharp is <strong>cloud-ready</strong>, with supported integrations for:<br>
  <strong>Artifact storage</strong> —
  <a href="src/OptunaSharp.Artifacts.S3/README.md">Amazon S3</a> ·
  <a href="src/OptunaSharp.Artifacts.AzureBlob/README.md">Azure Blob Storage</a> ·
  <a href="src/OptunaSharp.Artifacts.Gcs/README.md">Google Cloud Storage</a> ·
  <a href="src/OptunaSharp.Artifacts.R2/README.md">Cloudflare R2</a><br>
  <strong>Study storage</strong> —
  <a href="src/OptunaSharp.Storage.EFCore/README.md">PostgreSQL</a> ·
  <a href="src/OptunaSharp.Storage.EFCore/README.md">MySQL</a> ·
  <a href="src/OptunaSharp.Storage.Redis/README.md">Redis</a> ·
  <a href="src/OptunaSharp.Storage.Grpc/README.md">gRPC storage proxy</a>
</p>

<p align="center">
  The compatibility target is <strong>Optuna 5.x</strong>. When OptunaSharp behavior and Optuna
  <br>behavior differ, Optuna is treated as the source of truth and OptunaSharp aligns to it.
</p>


---

## Scientific Lead and Contributions
OptunaSharp has reimplemented byte-for-byte many cutting-edge papers and state of the art algorithms and will continue to doing so.<br>
Found an <strong>incredible</strong> paper? You <strong>believe</strong> an optimizer/sampler/technique has achieved a breakthrough or has been referenced more than 50 times?
<h3>
<p align="center">
Open an issue (TODO LINK) or merge request (TODO LINK)!
</p>
</h3>


> AGE-MOEA-2, SMS-EMOA, MOEA/D, RVEA, LM-CMA, VkD-CMA, differential evolution, PSO, the natural evolution strategies (OpenAI-ES, SNES, xNES), TuRBO, REMBO, SAASBO and JANUS (with its DE and multi-objective variants).

And we would love to accept more!

---

## Features

The port of **Optuna 5.x is complete**: every Optuna module is ported (`study`, `trial`,
`distributions`, `samplers`, `pruners`, `storages`, `search_space`, `importance`, `terminator`,
`visualization`, `artifacts`, `integration`, `exceptions` and `logging`), with Optuna's names, defaults,
edge cases and error messages. [`docs/PORTING_PLAN.md`](docs/PORTING_PLAN.md) records how it was
ported, phase by phase.

- **Define-by-run API** — `Study`, `Trial` and `FrozenTrial` with every `suggest_*`, `Optimize`,
  ask-and-tell, enqueued trials, multi-objective directions, constraints, user attributes and
  callbacks; float, int and categorical distributions with search-space inference. All of it lives in
  the pure-managed Core package, built on NumSharp.
- **Samplers** — Optuna's Random, Grid, TPE, CMA-ES, GP, QMC, BruteForce, PartialFixed, NSGA-II and
  NSGA-III, plus samplers Optuna does not have: AGE-MOEA-2, SMS-EMOA, MOEA/D, RVEA, LM-CMA, VkD-CMA,
  differential evolution, PSO, the natural evolution strategies (OpenAI-ES, SNES, xNES), TuRBO, REMBO,
  SAASBO and JANUS (with its DE and multi-objective variants).
- **Pruners** — Median, Percentile, SuccessiveHalving, Hyperband, Threshold, Patient, Wilcoxon and Nop.
- **Analysis** — parameter importance (fANOVA, mean decrease impurity, PED-ANOVA and SHAP, all
  pure-managed and byte-exact), terminators, and the data behind all 12 `plot_*` functions, which the
  Plotly.NET package draws as interactive figures.
- **Storage** — in-memory and journal-file storage in Core; SQLite, PostgreSQL, MySQL, Redis journals
  and a gRPC storage proxy as companion packages. Python Optuna can work on the same study through
  the journal file, any of the databases, Redis or the gRPC proxy. On the SQLite, PostgreSQL and MySQL
  storages, heartbeats with `RetryFailedTrialCallback` retry the trials of a worker that died.
- **Artifacts** — a file-system store in Core; Amazon S3, Google Cloud Storage, Azure Blob Storage and
  Cloudflare R2 as companion packages.
- **Parallel and thread-safe** — `Optimize(nJobs: n)` runs n trials at once, with no GIL in the way.
  Measured on 32 logical CPUs, 64 trials of a ~50 ms CPU-bound objective finished 6.7× faster at
  `nJobs: 8` and 9.2× faster at `nJobs: 16` than at `nJobs: 1`; Python Optuna 5.0 gains about 1.1× at
  any `n_jobs`, because the GIL runs one Python thread at a time.
  - Trials run on OptunaSharp's own dedicated threads by default, or on the .NET ThreadPool.
  - Sampling is serial per sampler, as in Optuna, unless `SamplerConcurrency.PerWorker` gives each
    worker its own Random, TPE, CMA-ES, VkD-CMA or Grid sampler.
  - A `CancellationToken`, or `StopSignals` for Ctrl+C and SIGTERM, stops a run gracefully: no new
    trial starts, and the trials in flight finish and are stored.
  - Studies, trials, samplers, pruners and storages are safe to share between threads;
    [`docs/THREADING.md`](docs/THREADING.md) has the design, its guarantees and its limits.
- **Integrations** — a native scikit-learn-style `OptunaSearchCV`, `LightGBMTuner`, ML.NET,
  training-loop pruning callbacks, Microsoft.Extensions.DependencyInjection registrations, and the
  `optunasharp` command-line tool, a port of the `optuna` CLI.
- **Python bridges** — OptunaHub packages, BoTorch, pycma and pymoo, driven in-process through
  pythonnet.
- **Reproducible and tested** — a seeded run is **bit-identical to Python** wherever the path is
  deterministic, enforced by a live-Optuna parity harness (pythonnet) that asks real Optuna for each
  answer in-process and asserts that OptunaSharp matches. Everything builds and tests on **net8.0** and
  **net10.0** across Windows, Linux and macOS, behind strong-naming and package gates, and all 20
  packages release together from one tag.

## Install

```bash
dotnet add package OptunaSharp
```

The package targets **.NET 8** and **.NET 10**. Storages, cloud artifact stores, Plotly figures, the
CLI and the Python bridges are separate `OptunaSharp.*` packages (listed in
[the package README](src/OptunaSharp.Core/README.md#companion-packages)).

## Quick start

```csharp
using OptunaSharp;
using OptunaSharp.Samplers;

var study = Optuna.CreateStudy(sampler: new TPESampler(seed: 42));

study.Optimize(trial =>
{
    double x = trial.SuggestFloat("x", -10, 10);
    return (x - 2) * (x - 2);
}, nTrials: 20);

Console.WriteLine(study.BestValue);        // 0.0004973004749973951
Console.WriteLine(study.BestParams["x"]);  // 2.0223002348641756
```

Those two numbers are exactly what Python Optuna 5.0.0 prints for the same seeded study.

## What is Optuna?

Optuna is an automatic hyperparameter optimization framework with an imperative, *define-by-run*
API. The two core concepts:

- **Study** — an optimization session over an objective function.
- **Trial** — a single evaluation of the objective, during which hyperparameters are *suggested*.

```python
# Optuna (Python) — the behavior OptunaSharp targets
import optuna

def objective(trial):
    x = trial.suggest_float("x", -10, 10)
    return (x - 2) ** 2

study = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(objective, n_trials=20)
print(study.best_value, study.best_params)
```

The C# quick start above is the same study, written against the same define-by-run shape:
`snake_case` names become `PascalCase` (`suggest_float` → `SuggestFloat`, `n_trials` → `nTrials`),
while parameter order, defaults and error messages are kept as Optuna has them.

## Documentation

The documentation site, [scisharp.github.io/OptunaSharp](https://scisharp.github.io/OptunaSharp/):

- **Start here** —
  [Introduction](https://scisharp.github.io/OptunaSharp/docs/intro.html) ·
  [Getting Started](https://scisharp.github.io/OptunaSharp/docs/getting-started.html) ·
  [Key Features](https://scisharp.github.io/OptunaSharp/docs/key-features/index.html) ·
  [FAQ](https://scisharp.github.io/OptunaSharp/docs/faq.html)
- **Guides** —
  [Study and Trials](https://scisharp.github.io/OptunaSharp/docs/study-and-trials.html) ·
  [Distributions](https://scisharp.github.io/OptunaSharp/docs/distributions.html) ·
  [Search Spaces](https://scisharp.github.io/OptunaSharp/docs/search-spaces.html) ·
  [Samplers](https://scisharp.github.io/OptunaSharp/docs/samplers.html) ·
  [Pruners](https://scisharp.github.io/OptunaSharp/docs/pruners.html) ·
  [Multi-objective Optimization](https://scisharp.github.io/OptunaSharp/docs/multi-objective.html) ·
  [Storage](https://scisharp.github.io/OptunaSharp/docs/storage.html) ·
  [Artifacts](https://scisharp.github.io/OptunaSharp/docs/artifacts.html) ·
  [Terminators](https://scisharp.github.io/OptunaSharp/docs/terminators.html) ·
  [Parameter Importance](https://scisharp.github.io/OptunaSharp/docs/importance.html) ·
  [Visualization](https://scisharp.github.io/OptunaSharp/docs/visualization.html) ·
  [Logging](https://scisharp.github.io/OptunaSharp/docs/logging.html) ·
  [Exceptions](https://scisharp.github.io/OptunaSharp/docs/exceptions.html)
- **Recipes** —
  [Overview](https://scisharp.github.io/OptunaSharp/docs/recipes/index.html) ·
  [Saving/Resuming with RDB](https://scisharp.github.io/OptunaSharp/docs/recipes/rdb.html) ·
  [Multi-objective Optimization](https://scisharp.github.io/OptunaSharp/docs/recipes/multi-objective.html) ·
  [User Attributes](https://scisharp.github.io/OptunaSharp/docs/recipes/attributes.html) ·
  [Command-Line Interface](https://scisharp.github.io/OptunaSharp/docs/recipes/cli.html) ·
  [User-Defined Sampler](https://scisharp.github.io/OptunaSharp/docs/recipes/user-defined-sampler.html) ·
  [User-Defined Pruner](https://scisharp.github.io/OptunaSharp/docs/recipes/user-defined-pruner.html) ·
  [Callback for Study.Optimize](https://scisharp.github.io/OptunaSharp/docs/recipes/callback.html) ·
  [Specify Hyperparameters Manually](https://scisharp.github.io/OptunaSharp/docs/recipes/specify-params.html) ·
  [Ask-and-Tell Interface](https://scisharp.github.io/OptunaSharp/docs/recipes/ask-and-tell.html) ·
  [Re-use the Best Trial](https://scisharp.github.io/OptunaSharp/docs/recipes/reuse-best-trial.html) ·
  [Journal-File Storage](https://scisharp.github.io/OptunaSharp/docs/recipes/journal-storage.html) ·
  [Artifacts](https://scisharp.github.io/OptunaSharp/docs/recipes/artifacts.html) ·
  [Wilcoxon Pruner](https://scisharp.github.io/OptunaSharp/docs/recipes/wilcoxon-pruner.html) ·
  [Ablation Study with Brute Force](https://scisharp.github.io/OptunaSharp/docs/recipes/ablation-study.html)
- **Integrations** — [Overview](https://scisharp.github.io/OptunaSharp/integrations/index.html)
  - Model training —
    [scikit-learn-style search](https://scisharp.github.io/OptunaSharp/integrations/training/scikit-learn.html) ·
    [ML.NET](https://scisharp.github.io/OptunaSharp/integrations/training/ml-net.html) ·
    [LightGBM](https://scisharp.github.io/OptunaSharp/integrations/training/lightgbm.html) ·
    [Training-loop pruning](https://scisharp.github.io/OptunaSharp/integrations/training/pruning.html)
  - Storage —
    [SQLite](https://scisharp.github.io/OptunaSharp/integrations/storage/sqlite.html) ·
    [EF Core (PostgreSQL / MySQL)](https://scisharp.github.io/OptunaSharp/integrations/storage/ef-core.html) ·
    [Redis](https://scisharp.github.io/OptunaSharp/integrations/storage/redis/index.html) ·
    [gRPC](https://scisharp.github.io/OptunaSharp/integrations/storage/grpc/index.html)
  - Python bridges —
    [Python.NET runtime](https://scisharp.github.io/OptunaSharp/integrations/python/pythonnet.html) ·
    [OptunaHub](https://scisharp.github.io/OptunaSharp/integrations/python/optunahub.html) ·
    [BoTorch](https://scisharp.github.io/OptunaSharp/integrations/python/botorch.html) ·
    [PyCma](https://scisharp.github.io/OptunaSharp/integrations/python/pycma.html) ·
    [pymoo](https://scisharp.github.io/OptunaSharp/integrations/python/pymoo.html)
  - Cloud artifacts —
    [Amazon S3](https://scisharp.github.io/OptunaSharp/integrations/artifacts/s3.html) ·
    [Google Cloud Storage](https://scisharp.github.io/OptunaSharp/integrations/artifacts/gcs.html) ·
    [Azure Blob Storage](https://scisharp.github.io/OptunaSharp/integrations/artifacts/azure-blob.html) ·
    [Cloudflare R2](https://scisharp.github.io/OptunaSharp/integrations/artifacts/r2.html)
  - Tools —
    [Plotly.NET figures](https://scisharp.github.io/OptunaSharp/integrations/visualization/plotly.html) ·
    [Command-line tool](https://scisharp.github.io/OptunaSharp/integrations/cli.html)
- **Methods** — [Overview](https://scisharp.github.io/OptunaSharp/docs/methods/index.html): a page per
  algorithm, each with its paper — the samplers, pruners, importance evaluators and terminators, and
  the operators and kernels they are built from.
- **API reference** — [OptunaSharp](https://scisharp.github.io/OptunaSharp/api/index.html), generated
  from the XML documentation of the Core package.

## The NumSharp reference

`OptunaSharp.Core` depends on **NumSharp** and its native LAPACK/CBLAS backend
**NumSharp.Interop.OpenBLAS**, both consumed as **NuGet packages** from nuget.org (version pinned by
the `NumSharpVersion` property in `Directory.Build.props`). Build and CI restore them from nuget.org;
the OpenBLAS package ships the native LAPACK binaries (in `runtimes/<rid>/native/`), so `np.linalg`
computes through LAPACK and is bit-identical to NumPy — no extra setup.

The matching NumSharp source is vendored **reference-only** as a pinned git submodule at `refs/NumSharp`
(the same release tag), for browsing — it is not part of the build. Restore it, if you want it, with
`git submodule update --init refs/NumSharp`.

## Build & test

```bash
# Build everything
dotnet build SciSharp.OptunaSharp.sln -c Release

# Unit tests (both TFMs), excluding known-failing/heavy/slow categories as per-push CI does
# (drop `&TestCategory!=Slow` for the full run)
dotnet test test/OptunaSharp.Tests/OptunaSharp.Tests.csproj -c Release \
  --filter "TestCategory!=OpenBugs&TestCategory!=HighMemory&TestCategory!=RequiresPython&TestCategory!=Slow"

# Live-Optuna interop: needs a discoverable Python with the vendored Optuna installed, which is what
# the parity is measured against (`git submodule update --init refs/optuna`, then
# `pip install -e refs/optuna`)
dotnet test test/OptunaSharp.Tests.Interop/OptunaSharp.Tests.Interop.csproj -c Release

# Pack the OptunaSharp package
dotnet pack src/OptunaSharp.Core/OptunaSharp.Core.csproj -c Release -o packages
```

Test categories (`OpenBugs`, `Misaligned`, `RequiresPython`, `HighMemory`, `WindowsOnly`, `Slow`) are typed
attributes in `test/OptunaSharp.Tests/TestCategory.cs`, so CI filter strings stay stable as the
suite grows.

## Scripting with `dotnet run` (.NET 10 file-based apps)

Internal members are exposed to a `dotnet run` script that overrides its assembly name to a declared
friend (see `src/OptunaSharp.Core/Assembly/Properties.cs`). A script **under the repo root** inherits
`Directory.Build.props` and is signed automatically; one run from elsewhere must pass the signing
properties itself:

```csharp
#:project <path-to-your-clone>/src/OptunaSharp.Core/OptunaSharp.Core.csproj
#:property AssemblyName=OptunaSharp.DotNetRunScript
#:property PublishAot=false
#:property SignAssembly=true
#:property AssemblyOriginatorKeyFile=<path-to-your-clone>/Open.snk
```

## License

MIT — see [LICENSE](LICENSE). OptunaSharp is a derivative of Optuna
(© Preferred Networks, Inc.) and retains its copyright notice as the MIT License requires.

## Related SciSharp projects

- [NumSharp](https://github.com/SciSharp/NumSharp) — NumPy for .NET (OptunaSharp's numerical foundation)
- [The SciSharp STACK](https://github.com/SciSharp) — open-source machine learning for .NET
