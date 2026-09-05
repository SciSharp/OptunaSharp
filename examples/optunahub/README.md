# OptunaHub examples

Live [OptunaHub](https://hub.optuna.org/) integration — load and run **real hub packages** (the samplers,
benchmarks and pruners the community publishes) from inside an OptunaSharp study, via
[`OptunaSharp.Interop.pythonnet`](../../src/OptunaSharp.Interop.pythonnet) (`OptunaHub.LoadModule` mirrors
`optunahub.load_module`). This folder covers all **three bridge directions** — samplers, benchmarks and
pruners — plus multi-objective, mixed search spaces, and offline loading.

| File | What it shows |
|---|---|
| [`OptunaHubCode1.cs`](OptunaHubCode1.cs) | the paper's **Code 1** — a hub **sampler** (`auto_sampler`) *and* a hub **benchmark** (`bbob`) together; nothing native but the study loop |
| [`OptunaHubBenchmark.cs`](OptunaHubBenchmark.cs) | the **benchmark bridge** alone — `bbob` Sphere optimized live by native TPE, colored convergence bar per trial |
| [`OptunaHubSamplerShootout.cs`](OptunaHubSamplerShootout.cs) | the **sampler bridge** — race hub samplers (`nelder_mead`, `cmaes`, `random_search`) vs native TPE on Rastrigin, colored sparklines + leaderboard |
| [`OptunaHubMultiObjective.cs`](OptunaHubMultiObjective.cs) | **multi-objective** — the bi-objective `bbob_biobj` + native NSGA-II, recovering the Pareto front |
| [`OptunaHubPruning.cs`](OptunaHubPruning.cs) | the **pruner bridge** — a hub `MedianPruner` cutting laggards short in a `report`/`should_prune` loop |
| [`OptunaHubLocalRegistry.cs`](OptunaHubLocalRegistry.cs) | **`LoadLocalModule`** (offline) + a mixed **Float/Int/Categorical** search space, and the registry knobs (`repoOwner`/`gitRef`/`baseUrl`) |
| [`run-optunahub.sh`](run-optunahub.sh) / [`run-optunahub.bat`](run-optunahub.bat) | launchers: install the Python prerequisites, then run the whole suite in color. On Windows you can **double-click the `.bat`** — the window stays open so you can read the output |
| [`requirements.txt`](requirements.txt) | the Python packages these examples need (the launchers `pip install -r` it) |

## Prerequisites

A Python with `optuna` + `optunahub`, plus `coco-experiment` (the BBOB benchmarks), `cmaes` and `pandas`
(the `cmaes` / `nelder_mead` hub samplers). Any hub sampler whose deps are missing is simply skipped, and
each program probes first and guides you if a prerequisite is absent — so nothing crashes. The launcher
installs everything; otherwise:

```bash
pip install -r examples/optunahub/requirements.txt
```

## Run

```bash
examples/optunahub/run-optunahub.sh          # Linux/macOS — installs deps + runs the whole suite, in color
examples\optunahub\run-optunahub.bat         # Windows — or just double-click it in Explorer (window stays open)
# or one at a time (install the deps yourself first):
dotnet run examples/optunahub/OptunaHubCode1.cs
dotnet run examples/optunahub/OptunaHubBenchmark.cs
dotnet run examples/optunahub/OptunaHubSamplerShootout.cs
dotnet run examples/optunahub/OptunaHubMultiObjective.cs
dotnet run examples/optunahub/OptunaHubPruning.cs
dotnet run examples/optunahub/OptunaHubLocalRegistry.cs
# pick the interpreter: PYTHON=/path/to/python   ·   disable colors: NO_COLOR=1
```

Each program loads the network hub first and falls back to the pinned `refs/optunahub-registry` submodule
offline, and honors `NO_COLOR` / `FORCE_COLOR`. The bridge itself (samplers, benchmarks **and** pruners) is
covered by [`OptunaHubBridgeParityTests`](../../test/OptunaSharp.Tests.Interop/OptunaHubBridgeParityTests.cs).
