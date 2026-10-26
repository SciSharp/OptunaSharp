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
  OptunaSharp is a .NET port of <a href="https://github.com/optuna/optuna">Optuna</a>, the
  define-by-run hyperparameter optimization framework — studies, trials, samplers, pruners,
  distributions and storages — for C# and F#, built on top of
  <a href="https://github.com/SciSharp/NumSharp">NumSharp</a> for numerical work.
</p>

<p align="center">
  The compatibility target is <strong>Optuna 5.x</strong>. When OptunaSharp behavior and Optuna
  behavior differ, Optuna is treated as the source of truth and OptunaSharp aligns to it.
</p>

---

## Status

OptunaSharp has ported the **core algorithm surface** of Optuna 5.x and much of its ecosystem. The
build/test/package/release pipeline has been green since day one, and the port grows against it
module-by-module — [`docs/PORTING_PLAN.md`](docs/PORTING_PLAN.md) is the governing roadmap.

In place today:

- **Core (pure-managed)** — Study / Trial / FrozenTrial (`suggest_*`, `optimize`, multi-objective
  directions), float/int/categorical distributions, search-space inference, in-memory + journal
  storage, exceptions and logging.
- **Samplers** — Random, Grid, TPE, CMA-ES, GP, QMC, BruteForce, PartialFixed, NSGA-II, NSGA-III
  (plus beyond-Optuna extensions: AGE-MOEA-2, SMS-EMOA, LM-CMA, JANUS).
- **Pruners** — Median, Percentile, SuccessiveHalving, Hyperband, Threshold, Patient, Wilcoxon, Nop.
- **Importance** — fANOVA, MeanDecreaseImpurity, PED-ANOVA, SHAP (pure-managed, byte-exact) — plus a
  scikit-learn-style `OptunaSearchCV`.
- **Terminator** and the **visualization** data layer (all 12 `plot_*`).
- **Satellite packages** (foreign-runtime, opt-in like `NumSharp.Interop.OpenBLAS`): SQLite / EF Core
  / gRPC / Redis storage, cloud artifact stores (S3 / GCS / Azure Blob / R2), Plotly visualization,
  LightGBM / ML.NET integration, an `optunasharp` CLI tool, and pythonnet interop bridges (OptunaHub,
  BoTorch, PyCma).

Reproducibility under a fixed seed is **bit-identical to Python** wherever the path is deterministic,
enforced by a live-Optuna parity harness (pythonnet) that asks real Optuna to compute answers
in-process and asserts OptunaSharp matches. The whole pipeline runs on **net8.0** and **net10.0**
across Windows / Linux / macOS, behind strong-naming and package gates, and a tag-driven NuGet
release of all 20 packages.

## Install

```bash
dotnet add package OptunaSharp
```

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

study = optuna.create_study()
study.optimize(objective, n_trials=100)
print(study.best_params)
```

The C# surface will mirror this define-by-run shape as the port lands.

## Repository layout

```
OptunaSharp/
├── src/
│   ├── OptunaSharp.Core/            # the OptunaSharp package — the pure-managed library (refs NumSharp)
│   └── OptunaSharp.*/               # optional satellite packages (Storage.*, Visualization.Plotly, Integration.*, Interop.*, Artifacts.*, Cli)
├── test/
│   ├── OptunaSharp.Tests/           # unit tests (MSTest v3 + AwesomeAssertions)
│   ├── OptunaSharp.Tests.Interop/   # live-Optuna parity harness (pythonnet)
│   └── OptunaSharp.Benchmark/       # BenchmarkDotNet harness (template; grows with the ops)
├── docs/
│   └── website-src/                 # DocFX site (docfx.json, conceptual docs, generated API ref)
├── .github/
│   ├── workflows/                   # build-and-release · package-parity · docs · documentation · examples · benchmark
│   ├── scripts/                     # pack_all.sh · verify_strong_name.cs · verify_packages.cs · check_documentation.cs
│   ├── ISSUE_TEMPLATE/ · PULL_REQUEST_TEMPLATE.md · FUNDING.yml · dependabot.yml
├── Directory.Build.props            # strong-naming + shared metadata
├── Open.snk                         # Microsoft's published OSS key (shared with the SciSharp STACK)
├── SECURITY.md · CITATION.cff · CONTRIBUTING.md · CODE_OF_CONDUCT.md
└── SciSharp.OptunaSharp.sln
```

## Documentation

The documentation site is built with **DocFX** from `docs/website-src/` and deployed to GitHub Pages
by `.github/workflows/docs.yml`. Build it locally:

```bash
dotnet tool install -g docfx
docfx docs/website-src/docfx.json --serve   # http://localhost:8080
```

It renders the conceptual docs (`docs/website-src/docs/`) plus an API reference generated from
`OptunaSharp.Core`.

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

# Unit tests (both TFMs), excluding known-failing/heavy categories as CI does
dotnet test test/OptunaSharp.Tests/OptunaSharp.Tests.csproj -c Release \
  --filter "TestCategory!=OpenBugs&TestCategory!=HighMemory&TestCategory!=RequiresPython"

# Live-Optuna interop (needs a discoverable Python with `pip install optuna`)
dotnet test test/OptunaSharp.Tests.Interop/OptunaSharp.Tests.Interop.csproj -c Release

# Pack the OptunaSharp package
dotnet pack src/OptunaSharp.Core/OptunaSharp.Core.csproj -c Release -o packages
```

Test categories (`OpenBugs`, `Misaligned`, `RequiresPython`, `HighMemory`, `WindowsOnly`) are typed
attributes in `test/OptunaSharp.Tests/TestCategory.cs`, so CI filter strings stay stable as the
suite grows.

## Roadmap — Optuna modules

Tracking Optuna's own module layout. The core surface is in place;
[`docs/PORTING_PLAN.md`](docs/PORTING_PLAN.md) is the authoritative, phase-by-phase roadmap.

| Optuna module | Role | OptunaSharp status |
|---------------|------|--------------------|
| `distributions` | Search-space distributions (float/int/categorical) | ✅ ported |
| `trial` | Trial + FrozenTrial, `suggest_*` | ✅ ported |
| `study` | Study, `optimize`, directions, storage glue | ✅ ported |
| `samplers` | Random, Grid, TPE, CMA-ES, GP, QMC, BruteForce, NSGA-II/III, … | ✅ ported |
| `pruners` | Median, Hyperband, successive halving, … | ✅ ported |
| `storages` | In-memory + journal (Core); RDB (SQLite/EF Core), gRPC, Redis (satellites) | ✅ ported |
| `search_space` | Search-space inference | ✅ ported |
| `importance` | Hyperparameter importance (fANOVA, MDI, PED-ANOVA, SHAP) | ✅ ported |
| `terminator` | Automated termination | ✅ ported |
| `visualization` | Optimization-history plots (data layer + Plotly) | ✅ ported |
| `exceptions` / `logging` | Framework exceptions + logging | ✅ ported |
| `integration` / `artifacts` | LightGBM/ML.NET/sklearn; S3/GCS/Azure/R2 artifacts; OptunaHub/BoTorch/PyCma bridges | ✅ ported |

## Scripting with `dotnet run` (.NET 10 file-based apps)

Internal members are exposed to a `dotnet run` script that overrides its assembly name to a declared
friend (see `src/OptunaSharp.Core/Assembly/Properties.cs`). A script **under the repo root** inherits
`Directory.Build.props` and is signed automatically; one run from elsewhere must pass the signing
properties itself:

```csharp
#:project src/OptunaSharp.Core/OptunaSharp.Core.csproj
#:property AssemblyName=OptunaSharp.DotNetRunScript
#:property SignAssembly=true
#:property AssemblyOriginatorKeyFile=K:/source/OptunaSharp/Open.snk
```

## License

MIT — see [LICENSE](LICENSE). OptunaSharp is a derivative of Optuna
(© Preferred Networks, Inc.) and retains its copyright notice as the MIT License requires.

## Related SciSharp projects

- [NumSharp](https://github.com/SciSharp/NumSharp) — NumPy for .NET (OptunaSharp's numerical foundation)
- [The SciSharp STACK](https://github.com/SciSharp) — open-source machine learning for .NET
