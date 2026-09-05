# OptunaHub examples

Live [OptunaHub](https://hub.optuna.org/) integration — load and run **real hub packages** (samplers and
benchmarks the community publishes) from inside an OptunaSharp study, via
[`OptunaSharp.Interop.pythonnet`](../../src/OptunaSharp.Interop.pythonnet) (`OptunaHub.LoadModule`
mirrors `optunahub.load_module`). Unlike the single-file showcases in the parent
[`examples/`](../README.md), this is a small **group** (two programs + launchers), so it lives in its
own folder.

| File | What it shows |
|---|---|
| [`OptunaHubBenchmark.cs`](OptunaHubBenchmark.cs) | the paper's **Code 1** in .NET — `LoadModule("benchmarks/bbob")` (BBOB Sphere) optimized live by OptunaSharp's native TPE, with a colored convergence bar per trial |
| [`OptunaHubSamplerShootout.cs`](OptunaHubSamplerShootout.cs) | the **sampler bridge** — race real hub samplers (`nelder_mead`, `cmaes`, `random_search`) against OptunaSharp's native TPE on 2-D Rastrigin, with colored sparklines + a leaderboard |
| [`run-optunahub.sh`](run-optunahub.sh) / [`run-optunahub.bat`](run-optunahub.bat) | launchers: install the Python prerequisites, then run both in color |

## Prerequisites

A Python with `optuna` + `optunahub` (the benchmark adds `coco-experiment` for BBOB; the shootout adds
`cmaes` / `pandas` for those samplers — any hub sampler whose deps are missing is simply skipped). The
launcher installs them for you; otherwise:

```bash
pip install optuna optunahub coco-experiment cmaes pandas
```

## Run

```bash
examples/optunahub/run-optunahub.sh          # Linux/macOS — installs deps + runs both, in color
examples\optunahub\run-optunahub.bat         # Windows
# or one at a time (install the deps yourself first):
dotnet run examples/optunahub/OptunaHubBenchmark.cs
dotnet run examples/optunahub/OptunaHubSamplerShootout.cs
# pick the interpreter: PYTHON=/path/to/python   ·   disable colors: NO_COLOR=1
```

Each program probes first and prints friendly guidance if a prerequisite is missing (it never crashes),
loads the network hub first and falls back to the pinned `refs/optunahub-registry` submodule offline, and
honors `NO_COLOR` / `FORCE_COLOR`. The bridge itself is covered by
[`OptunaHubBridgeParityTests`](../../test/OptunaSharp.Tests.Interop/OptunaHubBridgeParityTests.cs).
