<p align="center">
  <img src="docs/website-src/images/optunasharp-logo.png" alt="OptunaSharp" height="120">
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
  OptunaSharp is <a href="https://scisharp.github.io/OptunaSharp/docs/key-features/parallelization.html"><strong>thread-safe and truly parallel.</strong></a><br>
  <code>nJobs</code> runs trials on all your cores at once, with no GIL in the way, and workers on <br>many machines share one study, <a href="https://scisharp.github.io/OptunaSharp/docs/storage.html#concurrency-workers-studies-and-languages-on-one-journal">Python Optuna workers included</a>.<br>
 OptunaSharp supports parallelism powered by:<br> single-threaded, reused dedicated threads or the .NET ThreadPool.
</p>

<p align="center">
  OptunaSharp is <strong>cloud-ready</strong>, with <a href="https://scisharp.github.io/OptunaSharp/integrations/index.html">supported integrations</a> for:<br>
  <a href="https://scisharp.github.io/OptunaSharp/integrations/artifacts/index.html"><strong>Artifact storage</strong></a> -
  Amazon S3 · Azure Blob Storage · Google Cloud Storage · Cloudflare R2<br>
  <a href="https://scisharp.github.io/OptunaSharp/integrations/storage/index.html"><strong>Study storage</strong></a> -
  PostgreSQL · MySQL · Redis · gRPC storage proxy
</p>

<p align="center">
  The compatibility target is <strong>Optuna 5.x</strong>. When OptunaSharp behavior and Optuna
  <br>behavior <a href="https://scisharp.github.io/OptunaSharp/docs/faq.html#notable-divergences-from-optuna">differ</a>, Optuna is treated as the source of truth and OptunaSharp aligns to it.
</p>

## Features

OptunaSharp is a complete 100% port of **Optuna 5.x** with full and backwards compatability with optuna's integrations.
- **Define-by-run API** - `Study`, `Trial` and `FrozenTrial` with every `suggest_*`, `Optimize`,
  [ask-and-tell][ask-and-tell], enqueued trials, multi-objective directions, constraints, user
  attributes and callbacks; float, int and categorical distributions with search-space inference. All
  of it lives in the [pure-managed Core package][lightweight], built on NumSharp.
- **Samplers** - Optuna's Random, Grid, TPE, CMA-ES, GP, QMC, BruteForce, PartialFixed, NSGA-II and
  NSGA-III, plus samplers Optuna does not have: AGE-MOEA-2, SMS-EMOA, MOEA/D, RVEA, LM-CMA, VkD-CMA,
  differential evolution, PSO, the natural evolution strategies (OpenAI-ES, SNES, xNES), TuRBO, REMBO,
  SAASBO and JANUS (with its DE and multi-objective variants).
- **Integrations** - a native [scikit-learn-style `OptunaSearchCV`][i-scikit-learn],
  [`LightGBMTuner`][i-lightgbm], [ML.NET][i-ml-net], [training-loop pruning callbacks][i-pruning],
  [Microsoft.Extensions.DependencyInjection registrations][api-extensions], and the
  [`optunasharp` command-line tool][i-cli], a port of the `optuna` CLI.
- **Python bridges** - [OptunaHub packages][i-optunahub], [BoTorch][i-botorch], [pycma][i-pycma] and
  [pymoo][i-pymoo], driven in-process through [pythonnet][i-pythonnet].
- **Pruners** - Median, Percentile, SuccessiveHalving, Hyperband, Threshold, Patient, Wilcoxon and Nop.
- **Analysis** - parameter importance (fANOVA, mean decrease impurity, PED-ANOVA and SHAP, all
  pure-managed and byte-exact), terminators, and the data behind all 12 `plot_*` functions, which the
  [Plotly.NET package][i-plotly] draws as interactive figures.
- **Storage** - in-memory and journal-file storage in Core; [SQLite][i-sqlite],
  [PostgreSQL / MySQL][i-ef-core], [Redis journals][i-redis] and a [gRPC storage proxy][i-grpc] as
  companion packages. Python Optuna can work on the same study through the journal file, any of the
  databases, Redis or the gRPC proxy. On the SQLite, PostgreSQL and MySQL storages,
  [heartbeats with `RetryFailedTrialCallback`][storage-resilience] retry the trials of a worker that
  died.
- **Artifacts** - a file-system store in Core; [Amazon S3][i-s3], [Google Cloud Storage][i-gcs],
  [Azure Blob Storage][i-azure-blob] and [Cloudflare R2][i-r2] as companion packages.
  
- **Parallel and thread-safe** - [`Optimize(nJobs: n)`][parallel-threads] runs n trials at once, with no
  GIL in the way.
  Measured on 10 cores (20 logical CPUs), 64 trials of a ~50 ms CPU-bound objective finished 8.0× faster
  at `nJobs: 8` and 10.3× faster at `nJobs: 16` than at `nJobs: 1`, while the times of Python Optuna and
  Rustuna stay flat because the GIL runs one Python thread at a time ([Performance](#performance)).
  - Trials run on OptunaSharp's own dedicated threads by default, or on the .NET ThreadPool.
  - Sampling is serial per sampler, as in Optuna, unless `SamplerConcurrency.PerWorker` gives each
    worker its own Random, TPE, CMA-ES, VkD-CMA or Grid sampler.
  - A `CancellationToken`, or `StopSignals` for Ctrl+C and SIGTERM, stops a run gracefully: no new
    trial starts, and the trials in flight finish and are stored.
  - Studies, trials, samplers, pruners and storages are safe to share between threads;
    [`docs/THREADING.md`](docs/THREADING.md) has the design, its guarantees and its limits.

- **Reproducible and tested** - [a seeded run][faq-reproducible] is **bit-identical to Python** wherever
  the path is deterministic, enforced by a [live-Optuna parity harness][parity] (pythonnet) that asks
  real Optuna for each answer in-process and asserts that OptunaSharp matches. Everything builds and
  tests on **net8.0** and **net10.0** across Windows, Linux and macOS, behind strong-naming and package
  gates, and all 20 packages release together from one tag.

## Scientific Lead and Contributions
OptunaSharp has reimplemented byte-for-byte [many cutting-edge papers and state of the art algorithms][methods] and will continue to doing so.<br>
Found an <strong>incredible</strong> paper? You <strong>believe</strong> a method/optimizer/sampler/technique has achieved a significant breakthrough or has been referenced by a bunch of papers?
<h3>
<p align="center">
Open an <a href="https://github.com/SciSharp/OptunaSharp/issues/new/choose">issue</a> or <a href="https://github.com/SciSharp/OptunaSharp/compare">merge request</a>!
</p>
</h3>

<br/>
OptunaSharp implements these methods based on papers and their original implementation (if there is one):

> [AGE-MOEA-2][m-age-moea-2], [SMS-EMOA][m-sms-emoa], [MOEA/D][m-moea-d], [RVEA][m-rvea], [LM-CMA][m-lm-cma],
> [VkD-CMA][m-vkd-cma], [differential evolution][m-de], [PSO][m-pso], the natural evolution strategies
> ([OpenAI-ES][m-openai-es], [SNES][m-snes], [xNES][m-xnes]), [TuRBO][m-turbo], [REMBO][m-rembo], SAASBO and
> [JANUS][m-janus] (with its [DE][m-janus-de] and [multi-objective][m-janus-moo] variants).

And we would love to accept more!

## What is Optuna?

Optuna is an automatic hyperparameter optimization framework with an imperative,
[*define-by-run*][define-by-run] API. The two [core concepts][intro-model]:

- **Study** - an optimization session over an objective function.
- **Trial** - a single evaluation of the objective, during which hyperparameters are *suggested*.

```python
# Optuna (Python) - the behavior OptunaSharp targets
import optuna

def objective(trial):
    x = trial.suggest_float("x", -10, 10)
    return (x - 2) ** 2

study = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(objective, n_trials=20)
print(study.best_value, study.best_params)
```

The C# quick start below is the same study, written against the same define-by-run shape:
`snake_case` names become `PascalCase` (`suggest_float` → `SuggestFloat`, `n_trials` → `nTrials`),
while parameter order, defaults and error messages are kept as Optuna has them.

## Documentation

The documentation site, [scisharp.github.io/OptunaSharp](https://scisharp.github.io/OptunaSharp/):

- **Start here** -
  [Introduction](https://scisharp.github.io/OptunaSharp/docs/intro.html) ·
  [Getting Started](https://scisharp.github.io/OptunaSharp/docs/getting-started.html) ·
  [Key Features](https://scisharp.github.io/OptunaSharp/docs/key-features/index.html) ·
  [FAQ](https://scisharp.github.io/OptunaSharp/docs/faq.html)
- **Guides** -
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
- **Recipes** - [Overview](https://scisharp.github.io/OptunaSharp/docs/recipes/index.html): short,
  runnable recipes, from saving and resuming a study with an RDB to an ablation study with brute force.
- **Integrations** - [Overview](https://scisharp.github.io/OptunaSharp/integrations/index.html): model
  training, storage, the Python bridges, cloud artifacts, Plotly.NET figures and the command-line tool
  (the Features list above links each one).
- **Methods** - [Overview](https://scisharp.github.io/OptunaSharp/docs/methods/index.html): a page per
  algorithm, each with its paper - the samplers, pruners, importance evaluators and terminators, and
  the operators and kernels they are built from.
- **API reference** - [OptunaSharp](https://scisharp.github.io/OptunaSharp/api/index.html), generated
  from the XML documentation of the Core package.


## Performance

Measured on 2026-11-22 on an Intel Xeon E5-2690 v2 server (Ivy Bridge: 10 cores, 20 logical CPUs, 3.0 GHz
with turbo to 3.6 GHz, AVX without AVX2) under Linux, otherwise idle: every block of runs started only once the
machine was under 5 % busy. OptunaSharp ran as .NET 10.0.15 Release builds, and Python 3.12.16 ran Optuna 5.0.0
and Rustuna 0.3.1 on the same NumPy. Each configuration ran five times, each time in a fresh process, with the
libraries taking turns on each workload in an order reversed every round. A time is the mean of the middle
three runs: the fastest and the slowest are dropped, so turbo's run-to-run wobble stays out. Studies that run
one trial at a time were pinned with `taskset` to two dedicated cores whose hyperthread siblings stayed idle:
one core for the main thread, one for the runtime's helpers (.NET's JIT and garbage collector, OpenBLAS).
Parallel runs had all 20 logical CPUs. Absolute times depend on the hardware, so compare within a row.

### Parallel trials

64 trials of a CPU-bound objective that takes about 50 ms on one core, with `RandomSampler`. Each
language's loop was calibrated to ~50 ms on its own, and each run keeps the median of three per row. Rustuna
has no `n_jobs`, so its workers are Python threads calling `optimize` on the same study.

| `nJobs` | OptunaSharp | Speedup | Python Optuna 5.0 | Rustuna 0.3.1 |
|---|---|---|---|---|
| 1 | 3.28 s | 1.0× | 3.20 s | 3.19 s |
| 2 | 1.62 s | 2.0× | 3.20 s | 3.20 s |
| 4 | 0.81 s | 4.0× | 3.18 s | 3.20 s |
| 8 | 0.41 s | 8.0× | 3.19 s | 3.20 s |
| 16 | 0.32 s | 10.3× | 3.18 s | 3.21 s |

Python Optuna's time stays flat because the GIL runs one Python thread at a time, and so does Rustuna's: it
samples without the GIL, but the objective is Python. Trials that wait rather than compute run all at once in
all three: 640 trials sleeping 100 ms took 1.09 s at `nJobs: 64`, and 1,280 took 1.10 s at `nJobs: 128`
(Python Optuna: 1.04 s and 1.09 s; Rustuna: 1.01 s and 1.01 s; ideal 1.0 s).

### Samplers

One trial at a time, the same seeded study in every library, on a cheap sphere objective so that the
sampler's own work dominates. OptunaSharp's times include the .NET JIT warming up. Rustuna 0.3.1 has no GP
or QMC sampler.

| Study | OptunaSharp | Python Optuna 5.0 | Rustuna 0.3.1 |
|---|---|---|---|
| TPE, 6 parameters, 500 trials | 1.73 s | 3.21 s | 0.15 s |
| TPE multivariate, 6 parameters, 500 trials | 1.75 s | 3.20 s | 0.15 s |
| TPE, 100 parameters, 150 trials | 3.2 s | 27.7 s | 0.76 s |
| GP, 6 parameters, 100 trials | 5.09 s | 12.24 s | not available |
| CMA-ES, 6 parameters, 3,000 trials | 1.07 s | 4.96 s | 0.93 s |
| CMA-ES, 200 parameters, 100 trials | 1.4 s | 2.3 s | 0.94 s |
| NSGA-II, 6 parameters, 3,000 trials | 0.63 s | 4.13 s | 0.09 s |
| QMC, 6 parameters, 3,000 trials | 0.24 s | 3.44 s | not available |
| Random, 6 parameters, 5,000 trials | 0.33 s | 2.62 s | 0.05 s |
| Random, 1,000 parameters, 200 trials | 0.8 s | 18.7 s | 0.22 s |
| Random, 4 parameters, 50,000 trials: each block of 10,000 | 0.12-0.50 s | 3.66-3.76 s | 0.08 s |

All three columns come from the same run, the libraries taking turns on each row. The Rustuna-derived speedups
of [plan 13](docs/plans/13-rustuna-derived-performance.md) and [plan 14](docs/plans/14-rustuna-second-pass.md)
behind these times keep every seeded result. On this machine the JIT is a large share of OptunaSharp's short
rows: a second 1,000-trial TPE study in the same process took 5.3 s against the first one's 6.7 s.

### Large studies

40 float parameters in [-10, 10], seed 0, each sampler's defaults (TPE multivariate for one objective, independent
for two), the workloads of [Rustuna's announcement](https://medium.com/optuna/announcing-rustuna-cc82a6815bf7).
[Rustuna](https://github.com/optuna/rustuna) is the Optuna team's Rust port. The three libraries took turns on each
workload in the same run; Python's 100,000-trial studies were not run, as they take hours. OptunaSharp's times
include the .NET JIT warming up.

| Study | OptunaSharp | Python Optuna 5.0 | Rustuna 0.3.1 |
|---|---|---|---|
| TPE, 1,000 trials | 6.7 s | 37.8 s | 2.63 s |
| TPE, two objectives, 1,000 trials | 18.1 s | 197 s | 10.4 s |
| CMA-ES, 10,000 trials | 4.0 s | 47.1 s | 2.78 s |
| CMA-ES, 100,000 trials | 24.9 s | not run | 22.6 s |
| NSGA-II, 10,000 trials | 2.0 s | 49.1 s | 1.13 s |
| NSGA-II, 100,000 trials | 11.7 s | not run | 23.1 s |

OptunaSharp's seeded trials are Optuna's: its CMA-ES and NSGA-II studies gave the same trials bit for bit (equal
hashes of every parameter and value), and its TPE trials agree to about 15 significant digits, since TPE's
truncated-normal sampling rounds differently in the last bits. Rustuna's trials differ from the first one (its own
random generator). Where OptunaSharp is still behind Rustuna, part of the gap is the protocol it keeps so Python
workers can share a study: CMA-ES pickles its state into the storage every generation, and TPE scores candidates
in NumPy's order. The [Performance and Compatibility](docs/website-src/docs/performance.md) guide lists the opt-in
switches that trade that compatibility for speed.

For memory, `new InMemoryStorage(compactFinishedTrials: true)` keeps finished trials compactly and gives the same
trials: the 100,000-trial CMA-ES study above keeps 257 MB instead of 1,287 MB once finished, and NSGA-II's 87 MB
instead of 668 MB, and both run faster (23.9 s and 9.8 s). Finished trials then hand out read-only views instead
of dictionaries; the guide lists what that changes.

### Executors

Trials run on [OptunaSharp's own dedicated threads by default, or on the .NET ThreadPool][parallel-executor]
(`TrialExecutorKind.DedicatedThreads` or `TrialExecutorKind.ThreadPool`). The blocking rows start twice and
once as many workers as the pool's minimum, which is the processor count:

| Workload | Dedicated threads | ThreadPool |
|---|---|---|
| 5,000 cheap Random trials, `nJobs: 8` | 13.9k trials/s | 14.2k trials/s |
| 2,000 TPE trials, `nJobs: 8` | 482 trials/s | 465 trials/s |
| 200 calls of `Optimize(nTrials: 8, nJobs: 8)` | 1,601 calls/s | 1,640 calls/s |
| 80 trials blocking 1.5 s, `nJobs: 40` (ideal 3.0 s) | 3.08 s | 6.07 s |
| Longest delay before other `Task.Run` work started, `nJobs: 40` / `20` | 6.7 ms / 3.9 ms | 1.53 s / 0.61 s |

The two match on throughput. Once `nJobs` reaches the pool's ready threads (20 on this machine), the
ThreadPool runs fewer trials at once and delays the rest of the application, so it suits an `nJobs` well
below the core count.

### Per-worker samplers

`nJobs: 8`, each run the best of three inside its process: one sampler shared by every worker, taking one call
at a time (`SamplerConcurrency.Shared`, Optuna's behaviour and the default), against
[one sampler per worker][parallel-sampling] (`SamplerConcurrency.PerWorker`):

| Study | Shared | PerWorker |
|---|---|---|
| TPE multivariate, 10 parameters, 600 trials, 5 ms objective | 0.80 s | 0.50 s |
| CMA-ES, 40 parameters, 2,000 trials, cheap objective | 0.92 s | 0.53 s |
| TPE, 3 parameters, 400 trials, 20 ms objective | 1.04 s | 1.04 s |

A sampler per worker pays where sampling dominates and changes nothing when the objective does. Slow
model-based samplers stay shared: a `GPSampler` study of 120 trials with a 20 ms objective took 7.73 s at
`nJobs: 1` and 11.80 s at `nJobs: 8`.

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

Those two numbers are exactly what Python Optuna 5.0.0 prints for [the same seeded study][first-study].

## The NumSharp reference

`OptunaSharp.Core` depends on **NumSharp** and its native LAPACK/CBLAS backend
**NumSharp.Interop.OpenBLAS**, both consumed as **NuGet packages** from nuget.org (version pinned by
the `NumSharpVersion` property in `Directory.Build.props`). Build and CI restore them from nuget.org;
the OpenBLAS package ships the native LAPACK binaries (in `runtimes/<rid>/native/`), so `np.linalg`
computes through LAPACK and is bit-identical to NumPy - no extra setup.

The matching NumSharp source is vendored **reference-only** as a pinned git submodule at `refs/NumSharp`
(the same release tag), for browsing - it is not part of the build. Restore it, if you want it, with
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

MIT - see [LICENSE](LICENSE). OptunaSharp is a derivative of Optuna
(© Preferred Networks, Inc.) and retains its copyright notice as the MIT License requires.

## Related SciSharp projects

- [NumSharp](https://github.com/SciSharp/NumSharp) - NumPy for .NET (OptunaSharp's numerical foundation)
- [The SciSharp STACK](https://github.com/SciSharp) - open-source machine learning for .NET

<!-- Links into the documentation site (https://scisharp.github.io/OptunaSharp), used by the Markdown above.
     The HTML blocks at the top cannot use these and spell their links out. -->

[storage-resilience]: https://scisharp.github.io/OptunaSharp/docs/storage.html#resilience-heartbeats-and-failed-trial-retry
[intro-model]: https://scisharp.github.io/OptunaSharp/docs/intro.html#the-study--trial-model
[parity]: https://scisharp.github.io/OptunaSharp/docs/intro.html#how-parity-is-verified
[first-study]: https://scisharp.github.io/OptunaSharp/docs/getting-started.html#your-first-study
[lightweight]: https://scisharp.github.io/OptunaSharp/docs/key-features/first.html
[define-by-run]: https://scisharp.github.io/OptunaSharp/docs/key-features/search-space.html#defining-spaces-with-control-flow
[parallel-threads]: https://scisharp.github.io/OptunaSharp/docs/key-features/parallelization.html#multi-thread-optimization
[parallel-executor]: https://scisharp.github.io/OptunaSharp/docs/key-features/parallelization.html#which-threads-run-the-trials
[parallel-sampling]: https://scisharp.github.io/OptunaSharp/docs/key-features/parallelization.html#sampling-in-parallel-and-openblas-threads
[faq-reproducible]: https://scisharp.github.io/OptunaSharp/docs/faq.html#how-can-i-obtain-reproducible-optimization-results
[ask-and-tell]: https://scisharp.github.io/OptunaSharp/docs/recipes/ask-and-tell.html
[methods]: https://scisharp.github.io/OptunaSharp/docs/methods/index.html
[m-age-moea-2]: https://scisharp.github.io/OptunaSharp/docs/methods/age-moea-2.html
[m-sms-emoa]: https://scisharp.github.io/OptunaSharp/docs/methods/sms-emoa.html
[m-moea-d]: https://scisharp.github.io/OptunaSharp/docs/methods/moea-d.html
[m-rvea]: https://scisharp.github.io/OptunaSharp/docs/methods/rvea.html
[m-lm-cma]: https://scisharp.github.io/OptunaSharp/docs/methods/lm-cma.html
[m-vkd-cma]: https://scisharp.github.io/OptunaSharp/docs/methods/vkd-cma.html
[m-de]: https://scisharp.github.io/OptunaSharp/docs/methods/differential-evolution.html
[m-pso]: https://scisharp.github.io/OptunaSharp/docs/methods/pso.html
[m-openai-es]: https://scisharp.github.io/OptunaSharp/docs/methods/openai-es.html
[m-snes]: https://scisharp.github.io/OptunaSharp/docs/methods/snes.html
[m-xnes]: https://scisharp.github.io/OptunaSharp/docs/methods/xnes.html
[m-turbo]: https://scisharp.github.io/OptunaSharp/docs/methods/turbo.html
[m-rembo]: https://scisharp.github.io/OptunaSharp/docs/methods/rembo.html
[m-janus]: https://scisharp.github.io/OptunaSharp/docs/methods/janus.html
[m-janus-de]: https://scisharp.github.io/OptunaSharp/docs/methods/janus-de.html
[m-janus-moo]: https://scisharp.github.io/OptunaSharp/docs/methods/janus-moo-cmix.html
[i-scikit-learn]: https://scisharp.github.io/OptunaSharp/integrations/training/scikit-learn.html
[i-ml-net]: https://scisharp.github.io/OptunaSharp/integrations/training/ml-net.html
[i-lightgbm]: https://scisharp.github.io/OptunaSharp/integrations/training/lightgbm.html
[i-pruning]: https://scisharp.github.io/OptunaSharp/integrations/training/pruning.html
[i-sqlite]: https://scisharp.github.io/OptunaSharp/integrations/storage/sqlite.html
[i-ef-core]: https://scisharp.github.io/OptunaSharp/integrations/storage/ef-core.html
[i-redis]: https://scisharp.github.io/OptunaSharp/integrations/storage/redis/index.html
[i-grpc]: https://scisharp.github.io/OptunaSharp/integrations/storage/grpc/index.html
[i-pythonnet]: https://scisharp.github.io/OptunaSharp/integrations/python/pythonnet.html
[i-optunahub]: https://scisharp.github.io/OptunaSharp/integrations/python/optunahub.html
[i-botorch]: https://scisharp.github.io/OptunaSharp/integrations/python/botorch.html
[i-pycma]: https://scisharp.github.io/OptunaSharp/integrations/python/pycma.html
[i-pymoo]: https://scisharp.github.io/OptunaSharp/integrations/python/pymoo.html
[i-s3]: https://scisharp.github.io/OptunaSharp/integrations/artifacts/s3.html
[i-gcs]: https://scisharp.github.io/OptunaSharp/integrations/artifacts/gcs.html
[i-azure-blob]: https://scisharp.github.io/OptunaSharp/integrations/artifacts/azure-blob.html
[i-r2]: https://scisharp.github.io/OptunaSharp/integrations/artifacts/r2.html
[i-plotly]: https://scisharp.github.io/OptunaSharp/integrations/visualization/plotly.html
[i-cli]: https://scisharp.github.io/OptunaSharp/integrations/cli.html
[api-extensions]: https://scisharp.github.io/OptunaSharp/api/OptunaSharp.Extensions.html
