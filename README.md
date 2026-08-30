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

## ⚠️ Status: scaffolding

This repository currently contains **only the project stack** — solution, projects, strong-naming,
tests, pythonnet interop harness, packaging, and CI/CD — modeled on the
[NumSharp](https://github.com/SciSharp/NumSharp) stack. **No Optuna algorithms are ported yet.**
The scaffolding exists so the port can grow incrementally against a green build/test/package/release
pipeline from day one.

What already works:

- `dotnet build` / `dotnet test` / `dotnet pack` across **net8.0** and **net10.0**.
- A published-shaped, strong-named, **empty** `OptunaSharp` NuGet package.
- A unit-test project with smoke tests (assembly identity + the NumSharp dependency resolving).
- A pythonnet interop project that runs **real Optuna** in-process and self-skips without Python.
- GitHub Actions: cross-OS build/test, a strong-naming gate, a live-Optuna interop job, and a
  tag-driven NuGet release pipeline.

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
│   └── OptunaSharp.Core/            # the OptunaSharp package (empty scaffolding; refs NumSharp)
├── test/
│   ├── OptunaSharp.Tests/           # unit tests (MSTest v3 + AwesomeAssertions)
│   ├── OptunaSharp.Tests.Interop/   # live-Optuna parity harness (pythonnet)
│   └── OptunaSharp.Benchmark/       # BenchmarkDotNet harness (template; grows with the ops)
├── docs/
│   └── website-src/                 # DocFX site (docfx.json, conceptual docs, generated API ref)
├── .github/
│   ├── workflows/                   # build-and-release · docs · benchmark
│   ├── scripts/verify_strong_name.cs
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

## The NumSharp reference: "csproj for now, nuget in future"

`OptunaSharp.Core` depends on NumSharp. The reference is resolved conditionally:

- **Local dev** — when a sibling `NumSharp` checkout exists next to this repo
  (`K:\source\NumSharp` beside `K:\source\OptunaSharp`), it is referenced by **project**, so changes
  flow live.
- **Standalone / CI** — otherwise it falls back to the published **NuGet** package (`NumSharp`).

Both are mutually exclusive on `Exists(...)`, so exactly one applies and packaging always records a
`NumSharp` package dependency.

## Build & test

```bash
# Build everything
dotnet build SciSharp.OptunaSharp.sln -c Release

# Unit tests (both TFMs), excluding known-failing/heavy categories as CI does
dotnet test test/OptunaSharp.Tests/OptunaSharp.Tests.csproj -c Release \
  --filter "TestCategory!=OpenBugs&TestCategory!=HighMemory&TestCategory!=RequiresPython"

# Live-Optuna interop (needs a discoverable Python with `pip install optuna`)
dotnet test test/OptunaSharp.Tests.Interop/OptunaSharp.Tests.Interop.csproj -c Release

# Pack the (empty) OptunaSharp package
dotnet pack src/OptunaSharp.Core/OptunaSharp.Core.csproj -c Release -o packages
```

Test categories (`OpenBugs`, `Misaligned`, `RequiresPython`, `HighMemory`, `WindowsOnly`) are typed
attributes in `test/OptunaSharp.Tests/TestCategory.cs`, so CI filter strings stay stable as the
suite grows.

## Roadmap — Optuna modules to port

Tracking Optuna's own module layout. Nothing below is implemented yet; this is the porting map.

| Optuna module | Role | OptunaSharp status |
|---------------|------|--------------------|
| `distributions` | Search-space distributions (float/int/categorical) | not started |
| `trial` | Trial + FrozenTrial, `suggest_*` | not started |
| `study` | Study, `optimize`, directions, storage glue | not started |
| `samplers` | TPE, random, grid, CMA-ES, GP, … | not started |
| `pruners` | Median, Hyperband, successive halving, … | not started |
| `storages` | In-memory + RDB/journal persistence | not started |
| `search_space` | Search-space inference | not started |
| `importance` | Hyperparameter importance | not started |
| `terminator` | Automated termination | not started |
| `visualization` | Optimization-history plots | not started |
| `exceptions` / `logging` | Framework exceptions + logging | not started |

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
