# OptunaSharp examples

Runnable OptunaSharp programs — [.NET 10 file-based apps](https://learn.microsoft.com/en-us/dotnet/core/sdk/file-based-apps) you launch with
`dotnet run examples/<Name>.cs`. Two kinds: **ports of battle-tested public gists** (the most-starred
public Optuna usage, harvested with the [`interop-test-porting`](../.claude/skills/interop-test-porting/SKILL.md)
skill), and **feature showcases** of canonical Optuna capabilities.

CI compiles every example here, including `optunahub/` and the projects under `projects/`, whenever
`examples/` or `src/` changes ([`examples.yml`](../.github/workflows/examples.yml)). It builds them but does
not run them, because several need ffmpeg, Redis or Python.

## Example projects

Full .NET projects, in [`projects/`](projects/README.md), that show OptunaSharp inside real application
hosts. Run them with `dotnet run --project examples/projects/<Name>`.

| Project | First shows |
|---|---|
| [`TuningService`](projects/TuningService/README.md) | **ASP.NET Core + dependency injection**: storage, sampler, pruner, study factory and chart renderer registered in the container; ask-and-tell over HTTP, background optimization jobs, Plotly charts, a dashboard, health checks, problem details, and Optuna's gRPC storage proxy |
| [`TuningWorker`](projects/TuningWorker/README.md) | **Distributed optimization** from a Generic Host worker: several processes, and the Python Optuna worker next to it, share one TuningService study over gRPC |
| [`MLNetModelTuning`](projects/MLNetModelTuning/README.md) | **Real in-process training**: `OptunaSearchCV` cross-validates ML.NET FastTree models, with a resumable SQLite study, a held-out score, fANOVA importance and Plotly charts |

## Real-world use cases (`0_*.cs`)

One file per real-world job, each with a different sampler / pruner / storage setup and an objective that
does real work: a simulation, an analytic engineering model, a real codec or pipeline, child processes, or
in-process training. The `0_` prefix keeps them together. Where a run was compared against live Optuna 5.0.0
the result is in the last column (**bit-exact** = every trial identical; tolerance = the documented erf/GP
classes).

| Example | Setup | The job |
|---|---|---|
| [`0_PidControllerTuning.cs`](0_PidControllerTuning.cs) | CMA-ES, `x0`/`sigma0`, `enqueue_trial` — **bit-exact** (150 trials) | tune a DC-motor PID loop on a closed-loop simulation with actuator limits and anti-windup |
| [`0_CloudRightSizing.cs`](0_CloudRightSizing.cs) | TPE multivariate, `Trial.set_constraint` | cheapest instance fleet meeting a p99 SLO (Erlang C), a memory floor and AZ-loss headroom |
| [`0_LogCompressionTradeoff.cs`](0_LogCompressionTradeoff.cs) | NSGA-II, 2 objectives, resumable `JournalStorage` file | archive size vs. throughput over real Brotli/GZip/ZLib/Deflate runs; re-run to resume |
| [`0_ChurnModelHyperband.cs`](0_ChurnModelHyperband.cs) | TPE + `HyperbandPruner` | a hand-written MLP churn model trained per epoch; held-out test score at the end |
| [`0_ReactionYieldBayesOpt.cs`](0_ReactionYieldBayesOpt.cs) | `GPSampler`, `create_trial` + `add_trial`, constraint | batch-reactor yield (Arrhenius kinetics) warm-started from a lab notebook, impurity spec |
| [`0_AnnealingSolverConfig.cs`](0_AnnealingSolverConfig.cs) | TPE + `WilcoxonPruner` | algorithm configuration of a simulated-annealing TSP solver over 24 instances |
| [`0_InventoryPolicyQmc.cs`](0_InventoryPolicyQmc.cs) | random vs. scrambled-Sobol `QMCSampler` vs. TPE — QMC/random **bit-exact** | (s, S) inventory policy on a noisy Monte Carlo, plus the winner's-curse re-check |
| [`0_DotnetRuntimeKnobs.cs`](0_DotnetRuntimeKnobs.cs) | ask-and-tell, `TrialState.FAIL` | real `DOTNET_*` runtime switches, timed on child processes of the same program |
| [`0_CrashTolerantSqliteWorkers.cs`](0_CrashTolerantSqliteWorkers.cs) | SQLite `RDBStorage`, heartbeats, `RetryHeartbeatStaleTrialCallback` | three worker processes, one killed mid-trial; its trial is failed and retried; Python reads the `.db` |
| [`0_CarSideImpactNsga3.cs`](0_CarSideImpactNsga3.cs) | NSGA-III, 3 objectives, 10 constraints — **bit-exact** (600 trials) | car side-impact design (RE41) under the side-crash safety limits |
| [`0_RecommenderParallelAsha.cs`](0_RecommenderParallelAsha.cs) | `n_jobs=4`, `SuccessiveHalvingPruner`, TPE constant liar | matrix-factorization recommender trained by SGD, tuned on four threads |
| [`0_CreditFeatureSelection.cs`](0_CreditFeatureSelection.cs) | TPE multivariate over 16 booleans, fANOVA, `trials_dataframe` | which credit-bureau attributes are worth buying for a logistic scorecard |
| [`0_PipelineGridSearch.cs`](0_PipelineGridSearch.cs) | `GridSampler` (stops the study itself) | channel capacity × consumers × batch size of a real `System.Threading.Channels` pipeline |
| [`0_DemandForecastArtifacts.cs`](0_DemandForecastArtifacts.cs) | TPE + `MedianPruner`, `FileSystemArtifactStore` | Holt-Winters demand forecast with rolling-origin CV; every model uploaded, the best downloaded |
| [`0_DenoiseFilterTwoStage.cs`](0_DenoiseFilterTwoStage.cs) | TPE, then `PartialFixedSampler` | two-stage tuning of a median + bilateral denoising pipeline (PSNR) |
| [`0_FastChargingProtocol.cs`](0_FastChargingProtocol.cs) | `GPSampler`, 2 objectives + constraint — tolerance (~1e-5) | EV fast-charging protocol: charge time vs. cycle life under a temperature limit |
| [`0_GpsKalmanStudyLifecycle.cs`](0_GpsKalmanStudyLifecycle.cs) | `copy_study`, `load_study`, summaries, `delete_study`, metric names | a GPS Kalman filter's study, from notebook prototype to the team's SQLite store |
| [`0_GpuFineTunePlanner.cs`](0_GpuFineTunePlanner.cs) | `catch=`, NaN → FAIL, `n_jobs=2`, `timeout`, `MaxTrialsCallback`, `FixedTrial` | fit a 7B LoRA fine-tune on one 24 GB GPU for the most tokens/s |
| [`0_AdBiddingReport.cs`](0_AdBiddingReport.cs) | TPE + `MedianPruner`, nine Plotly charts | a real-time-bidding strategy on a 14-day auction simulation, with the stakeholder report |
| [`0_MicrogridSizingCmaMargin.cs`](0_MicrogridSizingCmaMargin.cs) | CMA-ES `with_margin` (mixed integer) — tolerance (≤ 3e-14) | off-grid PV + battery + diesel sizing over 8,760 hourly steps (LCOE) |
| [`0_KnnExperimentTracking.cs`](0_KnnExperimentTracking.cs) | rich user attrs through journal → SQLite → Python | k-NN classifier search whose per-fold metadata must survive every storage hop |
| [`0_LoanDefaultLightGbmTunerCv.cs`](0_LoanDefaultLightGbmTunerCv.cs) | `LightGBMTunerCV` (ML.NET LightGBM) | Optuna's stepwise LightGBM tuner on a loan-default model, trained for real |
| [`0_TrussCrossoverShootout.cs`](0_TrussCrossoverShootout.cs) | NSGA-II × six crossovers — all **bit-exact** (2,400 trials) | four-bar truss (RE21): volume vs. displacement, crossover operators compared by hypervolume |
| [`0_CachePolicyBruteForce.cs`](0_CachePolicyBruteForce.cs) | `BruteForceSampler` on a conditional space — **bit-exact** order | every LRU/FIFO/LFU/TTL cache configuration replayed against a drifting Zipf trace |
| [`0_ServerlessPowerTuningMotpe.cs`](0_ServerlessPowerTuningMotpe.cs) | multi-objective TPE + constraint | Lambda-style memory / arch / batch / AOT: cost per million vs. p99 latency |
| [`0_RateLimiterHostedService.cs`](0_RateLimiterHostedService.cs) | Generic Host + DI (`AddOptunaStudyFactory` …), cancellation | a `BackgroundService` re-tunes an API gateway's token bucket every telemetry window |
| [`0_PressureVesselParetoReport.cs`](0_PressureVesselParetoReport.cs) | NSGA-II, mixed directions, constraints, Pareto + hypervolume charts | pressure-vessel cost vs. capacity under the ASME thickness rules (RE23) |
| [`0_SpamFilterTimeBudget.cs`](0_SpamFilterTimeBudget.cs) | TPE with only a `timeout`, `show_progress_bar` | a naive-Bayes spam filter tuned inside a fixed ten-second slot |
| [`0_DistributedJournalHyperband.cs`](0_DistributedJournalHyperband.cs) | 3 processes × `n_jobs=2` on one `JournalFileBackend`, `HyperbandPruner` | SGD ridge house-price model; Hyperband's brackets span all six workers through the shared file |
| [`0_ChurnScorecardTerminator.cs`](0_ChurnScorecardTerminator.cs) | `report_cross_validation_scores` + default `TerminatorCallback` | stop tuning a churn scorecard when the GP regret bound drops below the CV noise; compared with a full run |
| [`0_ContainerLoadingStopAtBound.cs`](0_ContainerLoadingStopAtBound.cs) | `enqueue_trial(skip_if_exists=True)`, `study.stop()` from a callback | a bin-packing heuristic for trailer loading, warm-started from planners' rules, stopped at the lower bound |

Notes from running them:

- Every one runs with `dotnet run -c Release examples/<name>.cs`. The SQLite, Plotly, LightGBM and hosting
  examples pull their extra package through `#:project` / `#:package`. The SQLite examples also read the
  database back from Python when a `python` with `optuna` is on `PATH`, and skip that step otherwise.
- A file-based app needs `#:property PublishAot=false`. The .NET 10 default (`PublishAot=true`) turns off
  dynamic code, and NumSharp's IL kernels then throw `PlatformNotSupportedException` on the first
  suggestion.
- `0_LoanDefaultLightGbmTunerCv` sets LightGBM's `seed` explicitly. Without one, the ML.NET adapter is not
  reproducible: native LightGBM falls back to fixed default seeds, the adapter draws a random one. That is
  an open bug, tracked by `LightGbmRealWorldBugTests`.
- The progress bar shows the best value as a full float repr (`0.9303817601391925`); Optuna prints it with
  `:.6g` (`0.930382`). Open bug, tracked by `QA/RealWorldExampleBugTests`.
- `0_GpuFineTunePlanner` fails most of its trials on purpose. Samplers learn nothing from a FAIL trial, so
  record a predictable failure with `SetConstraint` instead (as `0_CloudRightSizing` does).

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
| [`ConditionalParameters.cs`](ConditionalParameters.cs) | `ConditionalParams` activation rules (an OptunaSharp extension) — branch-only parameters that `NSGAIIICptSampler` (NSGA-III-CPT) evolves while active and leaves dormant while not, printed generation by generation against NSGA-III's plain intersection behavior, where they stay at the random-draw baseline |
| [`ConditionalModelSelection.cs`](ConditionalModelSelection.cs) | `ConditionalParams` read as code — "pick a model, tune only ITS knobs": the rule block (value-set + a `model → kernel → gamma` chain) sits next to a branching objective that mirrors it one-for-one |
| [`ConditionalNeuralArchitecture.cs`](ConditionalNeuralArchitecture.cs) | `ConditionalParams` with **numeric-predicate** activators (`n_layers >= i` gates each layer's width) and a **two-level chain** (`use_scheduler → scheduler → warmup_frac`), on a Pareto front where each depth carries exactly its own widths |
| [`ConditionalTrainingSchedule.cs`](ConditionalTrainingSchedule.cs) | `ConditionalParams` **combining rules** — `warmup_epochs` declared under two activators (a scheduler value **AND** `n_epochs >= 30`), the example to read for how OR / AND / predicate rules compose |
| [`ConditionalPipelineChain.cs`](ConditionalPipelineChain.cs) | `ConditionalParams` as an **optional chain over an array of enum stages** — each `Op` stage (Conv/Pool/Dense/Stop) gates the next, so the pipeline's length is discovered; the rules are built in a loop and `PreserveDormantValues` keeps a stage's width when it toggles off and on |

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

Load and run **real [OptunaHub](https://hub.optuna.org/) packages** — the samplers, benchmarks **and
pruners** the community publishes — from inside an OptunaSharp study, via
[`OptunaSharp.Interop.pythonnet`](../src/OptunaSharp.Interop.pythonnet) (the .NET analog of
`NumSharp.Interop.pythonnet`; `OptunaHub.LoadModule` mirrors `optunahub.load_module`). These live in
their own folder — **[`examples/optunahub/`](optunahub/README.md)** — a colorful suite covering all three
bridge directions plus multi-objective, mixed search spaces, and offline loading:

| Example | First shows |
|---|---|
| [`optunahub/OptunaHubCode1.cs`](optunahub/OptunaHubCode1.cs) | the paper's **Code 1** — a hub **sampler** (`auto_sampler`) *and* a hub **benchmark** (`bbob`) together |
| [`optunahub/OptunaHubBenchmark.cs`](optunahub/OptunaHubBenchmark.cs) | the **benchmark bridge** alone — `bbob` Sphere optimized live by native TPE, colored convergence bar |
| [`optunahub/OptunaHubSamplerShootout.cs`](optunahub/OptunaHubSamplerShootout.cs) | the **sampler bridge** — hub samplers (`nelder_mead`, `cmaes`, `random_search`) vs native TPE, sparklines + leaderboard |
| [`optunahub/OptunaHubMultiObjective.cs`](optunahub/OptunaHubMultiObjective.cs) | **multi-objective** — bi-objective `bbob_biobj` + native NSGA-II, the Pareto front |
| [`optunahub/OptunaHubPruning.cs`](optunahub/OptunaHubPruning.cs) | the **pruner bridge** — a hub `MedianPruner` cutting laggards short |
| [`optunahub/OptunaHubLocalRegistry.cs`](optunahub/OptunaHubLocalRegistry.cs) | **`LoadLocalModule`** (offline) + a mixed **Float/Int/Categorical** space, and the registry knobs |
| [`optunahub/OptunaHubDashboard.cs`](optunahub/OptunaHubDashboard.cs) | **optuna-dashboard** — a hub-sampled study persisted by **`OptunaSharp.Storage.Sqlite`** (Optuna's exact schema), opened live in the real Python dashboard |

They need a **Python** with `optuna optunahub optuna-dashboard coco-experiment cmaes pandas`; each probes
first and guides you if a prerequisite is missing (nothing crashes). A **launcher** installs the deps,
runs the whole suite in color, **then opens the persisted study in [optuna-dashboard](https://github.com/optuna/optuna-dashboard)**
— on Windows you can just double-click it:

```bash
examples/optunahub/run-optunahub.sh        # Linux/macOS — pip install -r …/requirements.txt, runs the suite, then the dashboard
examples\optunahub\run-optunahub.bat       # Windows (or double-click; the suite window stays open, the dashboard opens in its own)
# choose the interpreter: PYTHON=/path/to/python   ·   disable colors: NO_COLOR=1   ·   skip the browser: OPTUNASHARP_NO_BROWSER=1
```

The bridge (samplers, benchmarks **and** pruners) is covered by
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
