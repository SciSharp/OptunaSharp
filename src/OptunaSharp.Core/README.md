# OptunaSharp

**Optuna for .NET**: a C#/F# port of [Optuna](https://github.com/optuna/optuna), the define-by-run
hyperparameter optimization framework, built on [NumSharp](https://github.com/SciSharp/NumSharp).

## Install

```bash
dotnet add package OptunaSharp
```

The package targets **.NET 8** and **.NET 10**. It needs dynamic code: NumSharp compiles its kernels at run time, so
OptunaSharp does not run under NativeAOT. A .NET 10 file-based app turns NativeAOT on by default, so declare
`#:property PublishAot=false` next to `#:package OptunaSharp@*`; without it the first trial fails with
`PlatformNotSupportedException: Dynamic code generation is not supported on this platform.`

## Quick start

```csharp
using OptunaSharp;

var study = Optuna.CreateStudy(direction: StudyDirection.Minimize);

study.Optimize(trial =>
{
    double x = trial.SuggestFloat("x", -10, 10);
    return (x - 2) * (x - 2);
}, nTrials: 100);

Console.WriteLine(study.BestValue);        // ~0
Console.WriteLine(study.BestParams["x"]);  // ~2
```

## What this package contains

- **Studies and trials**: `Optuna.CreateStudy` / `LoadStudy`, `Optimize`, ask-and-tell,
  multi-objective directions, constraints, user and system attributes, callbacks.
- **Distributions and search spaces**: float, int and categorical distributions, plus search-space
  inference for dynamic (define-by-run) spaces.
- **Samplers**: TPE, CMA-ES, GP, Random, Grid, QMC, BruteForce, PartialFixed, NSGA-II and NSGA-III.
  It also ships samplers Optuna does not have, among them AGE-MOEA-2, SMS-EMOA, MOEA/D, RVEA, LM-CMA,
  VkD-CMA, differential evolution, PSO, the natural evolution strategies (OpenAI-ES, SNES, xNES),
  TuRBO, REMBO, SAASBO and JANUS.
- **Pruners**: Median, Percentile, SuccessiveHalving, Hyperband, Threshold, Patient, Wilcoxon and Nop.
- **Storages**: in-memory and journal-file storage, heartbeats and `RetryFailedTrialCallback`.
- **Analysis**: parameter importance (fANOVA, mean decrease impurity, PED-ANOVA, SHAP), terminators,
  the data layer behind every `plot_*` function, and a scikit-learn-style `OptunaSearchCV`.
- **Artifacts**: a file-system artifact store.

## Companion packages

Anything that needs a foreign runtime or a heavy client library ships separately, so this package
depends only on NumSharp, its OpenBLAS backend and `Microsoft.Extensions.DependencyInjection.Abstractions`.

| Package | What it adds |
|---|---|
| `OptunaSharp.Storage.Sqlite` | SQLite `RDBStorage`, database-compatible with Python Optuna |
| `OptunaSharp.Storage.EFCore` | PostgreSQL and MySQL `RDBStorage` over Entity Framework Core |
| `OptunaSharp.Storage.Grpc` | the gRPC storage proxy, client and server |
| `OptunaSharp.Storage.StackExchange.Redis`, `.NRedisStack`, `.ServiceStack.Redis` | Redis journal backends, one per Redis client, over the shared `OptunaSharp.Storage.Redis` |
| `OptunaSharp.Artifacts.S3`, `.Gcs`, `.AzureBlob`, `.R2` | cloud artifact stores |
| `OptunaSharp.Visualization.Plotly` | interactive figures for every `plot_*` function |
| `OptunaSharp.Integration.LightGbm`, `.ML.NET` | `LightGBMTuner`, pruning callbacks and ML.NET-backed importance evaluators |
| `OptunaSharp.Interop.pythonnet`, `.BoTorch`, `.PyCma`, `.Pymoo` | bridges that drive OptunaHub, BoTorch, pycma and pymoo in-process |
| `OptunaSharp.Cli` | the `optunasharp` command-line tool (`dotnet tool install -g OptunaSharp.Cli`) |

## Compatibility target

OptunaSharp tracks **Optuna 5.x** as its behavioral source of truth: names, defaults, edge cases and
error messages follow Optuna. Where OptunaSharp and Optuna differ, Optuna is treated as correct and
OptunaSharp is aligned to it. The test suite checks OptunaSharp against live Python Optuna, including
seeded runs that must reproduce the values Optuna samples exactly.

## Links

- Documentation: <https://scisharp.github.io/OptunaSharp>
- Source and issues: <https://github.com/SciSharp/OptunaSharp>

## License

MIT. OptunaSharp is a derivative of Optuna (© Preferred Networks, Inc.) and retains its notice —
see `LICENSE`.
