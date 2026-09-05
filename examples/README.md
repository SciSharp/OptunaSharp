# OptunaSharp examples

Runnable OptunaSharp programs — [.NET 10 file-based apps](../.claude/CLAUDE.md) you launch with
`dotnet run examples/<Name>.cs`. Two kinds: **ports of battle-tested public gists** (the most-starred
public Optuna usage, harvested with the [`interop-test-porting`](../.claude/skills/interop-test-porting/SKILL.md)
skill), and **feature showcases** of canonical Optuna capabilities.

## Ported from real-world gists

| Example | Source gist | ★ | What it does |
|---|---|---|---|
| [`FFmpegX264Tuning.cs`](FFmpegX264Tuning.cs) | [`sile/8aa1ff78…`](https://gist.github.com/sile/8aa1ff7808dd55298f51dd70c8b83092) (Optuna maintainer) | 25 | **Real ffmpeg encode** of the rickroll — maximizes libx264's SSIM over a conditional x264 space + `SuccessiveHalvingPruner` |
| [`XGBoostHpo.cs`](XGBoostHpo.cs) | [`bexgboost/823df08d…`](https://gist.github.com/bexgboost/823df08d2e3760538e9b931d38439a68) | 16 | the classic 8-parameter XGBoost grid, minimizing RMSE |
| [`ParamRepeatPruner.cs`](ParamRepeatPruner.cs) | [`aisu-wata0/24045ab5…`](https://gist.github.com/aisu-wata0/24045ab5e8ad007b4a09f708adfe359f) | 6 | a userland pruner that skips trials whose params repeat an earlier trial's |

## Feature showcases

Each exercises a whole Optuna capability, with a deterministic surrogate objective so it runs anywhere.

| Example | First shows |
|---|---|
| [`MedianPrunedTraining.cs`](MedianPrunedTraining.cs) | `MedianPruner` + `report`/`should_prune` over epochs — the canonical prune-the-laggards loop (the sesame-finetune 113★ pattern) |
| [`MultiObjectivePareto.cs`](MultiObjectivePareto.cs) | multi-objective with `NSGAIISampler`, `directions=[…]`, and `study.BestTrials` — the Pareto front |
| [`HyperparameterImportance.cs`](HyperparameterImportance.cs) | `ImportanceFunctions.GetParamImportances` — fANOVA vs PedAnova on an objective rigged so `a ≫ b ≫ c` |
| [`AutomaticTermination.cs`](AutomaticTermination.cs) | the `Terminator` / `TerminatorCallback` — stop automatically when the search stops paying off |

## Redis storage

Persist a study's journal to a **Redis** server — so it survives restarts and can be shared by
distributed workers — the .NET port of Optuna's
[`JournalRedisBackend`](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.storages.journal.JournalRedisBackend.html).
Unlike the showcases above, these need a **reachable Redis**; each probes first and prints friendly
guidance (then exits cleanly) if none is found, so they never crash. See the
[Redis docs](../docs/website-src/redis/index.md).

| Example | First shows |
|---|---|
| [`RedisJournalStorage.cs`](RedisJournalStorage.cs) | `StackExchangeJournalRedisBackend` + `JournalStorage` with `loadIfExists` — a **resumable, shared** study (run it twice and the trial count grows) |
| [`RedisNativeClientConfig.cs`](RedisNativeClientConfig.cs) | the backends' **native client configuration** (URL / URL-plus-tweak / full `ConfigurationOptions` / bring-your-own `IConnectionMultiplexer`) and **interchangeable clients** — write with StackExchange.Redis, read the byte-identical journal back with NRedisStack |

```bash
docker run --rm -p 6379:6379 redis:7-alpine          # start a Redis (or use docker-compose, below)
dotnet run examples/RedisJournalStorage.cs           # run again to resume — the count grows
dotnet run examples/RedisNativeClientConfig.cs
# endpoint override: OPTUNASHARP_REDIS_URL=redis://host:6379
```

The repo's `test/OptunaSharp.Tests.Storage/docker-compose.yml` also starts one
(`docker compose -f test/OptunaSharp.Tests.Storage/docker-compose.yml up -d`). A third backend on the
**commercial** ServiceStack.Redis exists with the same surface but is not shown here (it isn't
referenced by the examples).

## OptunaHub (live Python bridge)

Load and run **real [OptunaHub](https://hub.optuna.org/) packages** — the samplers and benchmarks the
community publishes — from inside an OptunaSharp study, via
[`OptunaSharp.Interop.pythonnet`](../src/OptunaSharp.Interop.pythonnet) (the .NET analog of
`NumSharp.Interop.pythonnet`; `OptunaHub.LoadModule` mirrors `optunahub.load_module`). These need a
**Python** with `optuna` + `optunahub`; each probes first and prints friendly guidance if a prerequisite
is missing, so they never crash. The `source:` is the OptunaHub paper
([arXiv:2510.02798](https://arxiv.org/abs/2510.02798)).

| Example | First shows |
|---|---|
| [`OptunaHubBenchmark.cs`](OptunaHubBenchmark.cs) | the paper's **Code 1** in .NET — `LoadModule("benchmarks/bbob")` (BBOB Sphere) optimized live by OptunaSharp's native TPE, with a colored convergence bar per trial |
| [`OptunaHubSamplerShootout.cs`](OptunaHubSamplerShootout.cs) | the **sampler bridge** — race real hub samplers (`nelder_mead`, `cmaes`, `random_search`) against OptunaSharp's native TPE on 2-D Rastrigin, with colored convergence sparklines + a leaderboard |

A **launcher** installs the Python prerequisites and runs both, in color:

```bash
examples/run-optunahub.sh        # Linux/macOS — installs optuna optunahub coco-experiment cmaes pandas
examples\run-optunahub.bat       # Windows
# or run one directly (install the deps yourself first):
dotnet run examples/OptunaHubBenchmark.cs
dotnet run examples/OptunaHubSamplerShootout.cs
# choose the interpreter: PYTHON=/path/to/python   ·   disable colors: NO_COLOR=1
```

The bridge is covered by
[`OptunaHubBridgeParityTests`](../test/OptunaSharp.Tests.Interop/OptunaHubBridgeParityTests.cs).

## Run them

```bash
dotnet run examples/XGBoostHpo.cs            # pure managed — nothing extra
dotnet run examples/ParamRepeatPruner.cs     # pure managed
dotnet run examples/MedianPrunedTraining.cs  # pure managed
dotnet run examples/MultiObjectivePareto.cs  # pure managed
dotnet run examples/HyperparameterImportance.cs
dotnet run examples/AutomaticTermination.cs  # uses the OpenBLAS-backed GP terminator path
dotnet run examples/FFmpegX264Tuning.cs      # needs ffmpeg (+ yt-dlp to fetch the rickroll)
```

`FFmpegX264Tuning` shells out to **ffmpeg** and fetches the rickroll (`dQw4w9WgXcQ`) with **yt-dlp**; if
YouTube blocks the download (a `403` without cookies/PO-token) it falls back to an ffmpeg `testsrc2`
clip, so it always runs a real encode.

## Real vs. surrogate objectives

`FFmpegX264Tuning` (an actual encode), `ParamRepeatPruner` (a real repeat check), the two Redis
examples (real self-contained paraboloids — the point there is the storage, not the objective), and the
two OptunaHub examples (real BBOB evaluated in Python / a genuine Rastrigin — the point there is the
live hub bridge) run their **real** objectives. The rest use a **deterministic surrogate** for the objective body — OptunaSharp can't run
XGBoost/train a net in-process, and a sampler's trajectory is fixed by `(seed, search space, value
sequence)` alone, so the surrogate keeps the example self-contained while exercising the real feature.
The **real ML** runs (XGBoost, LightGBM, sklearn) happen in `OptunaSharp.Tests.PackageParity`, where
Python trains the real model and the packaged OptunaSharp replays the trajectory.

## Byte-for-byte parity

The gist search spaces are asserted bit-identical to live Optuna 5.0.0.dev —
[`GistExampleParityTests`](../test/OptunaSharp.Tests.Interop/GistExampleParityTests.cs) (FFmpeg +
XGBoost) and [`ParamRepeatPrunerParityTests`](../test/OptunaSharp.Tests.Interop/ParamRepeatPrunerParityTests.cs).
The features the showcases exercise have their own parity coverage in the interop suite (pruner
decisions, NSGA-II, importance, terminator). Parity is asserted on the RNG-exact `RandomSampler` path;
TPE over integer spaces isn't byte-comparable (NumPy simd-sort tie order — see the `testing` skill).

The Redis examples don't add their own parity test: the journal's **byte-compatibility** with Python
Optuna (and across the three client backends) is already proven by
[`JournalRedisInteropTests`](../test/OptunaSharp.Tests.Interop/JournalRedisInteropTests.cs) (Python ↔
Redis ↔ C#) and the `OptunaSharp.Tests.Storage` contract + cross-library suites, and the sampler
trajectory they persist is the same RNG-exact `RandomSampler` path asserted elsewhere.
