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
  OptunaSharp is <a href="https://scisharp.github.io/OptunaSharp/docs/key-features/parallelization.html"><strong>thread-safe and truly parallel.</strong></a><br>
  <a href="https://scisharp.github.io/OptunaSharp/docs/key-features/parallelization.html#multi-thread-optimization"><code>nJobs</code></a> runs trials on all your cores at once, with no GIL in the way, and workers on <br><a href="https://scisharp.github.io/OptunaSharp/docs/key-features/parallelization.html#multi-node-optimization">many machines</a> share one study, <a href="https://scisharp.github.io/OptunaSharp/docs/storage.html#concurrency-workers-studies-and-languages-on-one-journal">Python Optuna workers included</a>.<br>
 OptunaSharp supports parallelism powered by:<br> single-threaded, <a href="https://scisharp.github.io/OptunaSharp/docs/key-features/parallelization.html#which-threads-run-the-trials">reused dedicated threads and ThreadPool</a> async-await.
</p>

<p align="center">
  OptunaSharp is <strong>cloud-ready</strong>, with <a href="https://scisharp.github.io/OptunaSharp/integrations/index.html">supported integrations</a> for:<br>
  <a href="https://scisharp.github.io/OptunaSharp/integrations/artifacts/index.html"><strong>Artifact storage</strong></a> -
  <a href="https://scisharp.github.io/OptunaSharp/integrations/artifacts/s3.html">Amazon S3</a> ·
  <a href="https://scisharp.github.io/OptunaSharp/integrations/artifacts/azure-blob.html">Azure Blob Storage</a> ·
  <a href="https://scisharp.github.io/OptunaSharp/integrations/artifacts/gcs.html">Google Cloud Storage</a> ·
  <a href="https://scisharp.github.io/OptunaSharp/integrations/artifacts/r2.html">Cloudflare R2</a><br>
  <a href="https://scisharp.github.io/OptunaSharp/integrations/storage/index.html"><strong>Study storage</strong></a> -
  <a href="https://scisharp.github.io/OptunaSharp/integrations/storage/ef-core.html">PostgreSQL</a> ·
  <a href="https://scisharp.github.io/OptunaSharp/integrations/storage/ef-core.html">MySQL</a> ·
  <a href="https://scisharp.github.io/OptunaSharp/integrations/storage/redis/index.html">Redis</a> ·
  <a href="https://scisharp.github.io/OptunaSharp/integrations/storage/grpc/index.html">gRPC storage proxy</a>
</p>

<p align="center">
  The compatibility target is <strong>Optuna 5.x</strong>. When OptunaSharp behavior and Optuna
  <br>behavior <a href="https://scisharp.github.io/OptunaSharp/docs/faq.html#notable-divergences-from-optuna">differ</a>, Optuna is treated as the source of truth and OptunaSharp aligns to it.
</p>


---

## Scientific Lead and Contributions
OptunaSharp has reimplemented byte-for-byte [many cutting-edge papers and state of the art algorithms][methods] and will continue to doing so.<br>
Found an <strong>incredible</strong> paper? You <strong>believe</strong> an optimizer/sampler/technique has achieved a breakthrough or has been referenced more than 50 times?
<h3>
<p align="center">
Open an <a href="https://github.com/SciSharp/OptunaSharp/issues/new/choose">issue</a> or <a href="https://github.com/SciSharp/OptunaSharp/compare">merge request</a>!
</p>
</h3>


> [AGE-MOEA-2][m-age-moea-2], [SMS-EMOA][m-sms-emoa], [MOEA/D][m-moea-d], [RVEA][m-rvea], [LM-CMA][m-lm-cma],
> [VkD-CMA][m-vkd-cma], [differential evolution][m-de], [PSO][m-pso], the natural evolution strategies
> ([OpenAI-ES][m-openai-es], [SNES][m-snes], [xNES][m-xnes]), [TuRBO][m-turbo], [REMBO][m-rembo], SAASBO and
> [JANUS][m-janus] (with its [DE][m-janus-de] and [multi-objective][m-janus-moo] variants).

And we would love to accept more!

---

## Features

The port of **Optuna 5.x is complete**: every Optuna module is ported ([`study`][study-and-trials],
[`trial`][study-and-trials], [`distributions`][distributions], [`samplers`][samplers],
[`pruners`][pruners], [`storages`][storage], [`search_space`][search-spaces], [`importance`][importance],
[`terminator`][terminators], [`visualization`][visualization], [`artifacts`][artifacts],
[`integration`][integrations], [`exceptions`][exceptions] and [`logging`][logging]), with Optuna's
names, defaults, edge cases and error messages. [`docs/PORTING_PLAN.md`](docs/PORTING_PLAN.md) records
how it was ported, phase by phase.

- **Define-by-run API** - [`Study`, `Trial` and `FrozenTrial`][study-and-trials] with every
  [`suggest_*`][suggest], `Optimize`, [ask-and-tell][ask-and-tell], [enqueued trials][enqueue],
  [multi-objective directions][multi-objective], [constraints][faq-constraints],
  [user attributes][attributes] and [callbacks][callback]; float, int and categorical
  [distributions][distributions] with [search-space inference][search-spaces]. All of it lives in the
  [pure-managed Core package][lightweight], built on NumSharp.
- **Samplers** - Optuna's [Random][s-qmc-random], [Grid][s-grid-bruteforce], [TPE][s-tpe],
  [CMA-ES][s-cma-es], [GP][s-gp], [QMC][s-qmc-random], [BruteForce][s-grid-bruteforce],
  [PartialFixed][s-partial-fixed], [NSGA-II][m-nsga-ii] and [NSGA-III][m-nsga-iii], plus samplers Optuna
  does not have: [AGE-MOEA-2][m-age-moea-2], [SMS-EMOA][m-sms-emoa], [MOEA/D][m-moea-d], [RVEA][m-rvea],
  [LM-CMA][m-lm-cma], [VkD-CMA][m-vkd-cma], [differential evolution][m-de], [PSO][m-pso], the natural
  evolution strategies ([OpenAI-ES][m-openai-es], [SNES][m-snes], [xNES][m-xnes]), [TuRBO][m-turbo],
  [REMBO][m-rembo], SAASBO and [JANUS][m-janus] (with its [DE][m-janus-de] and
  [multi-objective][m-janus-moo] variants).
- **Pruners** - [Median][m-median], [Percentile][m-percentile],
  [SuccessiveHalving][m-successive-halving], [Hyperband][m-hyperband], [Threshold][m-threshold],
  [Patient][m-patient], [Wilcoxon][wilcoxon] and [Nop][m-nop].
- **Analysis** - [parameter importance][importance] ([fANOVA][fanova], [mean decrease impurity][mdi],
  [PED-ANOVA][ped-anova] and [SHAP][shap], all pure-managed and byte-exact),
  [terminators][terminators], and the data behind all 12 [`plot_*` functions][visualization], which the
  [Plotly.NET package][i-plotly] draws as interactive figures.
- **Storage** - [in-memory and journal-file storage][storage] in Core; [SQLite][i-sqlite],
  [PostgreSQL][i-ef-core], [MySQL][i-ef-core], [Redis journals][i-redis] and a
  [gRPC storage proxy][i-grpc] as companion packages.
  [Python Optuna can work on the same study][storage-concurrency] through the journal file, any of the
  databases, Redis or the gRPC proxy. On the SQLite, PostgreSQL and MySQL storages,
  [heartbeats with `RetryFailedTrialCallback`][storage-resilience] retry the trials of a worker that
  died.
- **Artifacts** - [a file-system store][artifacts] in Core; [Amazon S3][i-s3],
  [Google Cloud Storage][i-gcs], [Azure Blob Storage][i-azure-blob] and [Cloudflare R2][i-r2] as
  companion packages.
- **Parallel and thread-safe** - [`Optimize(nJobs: n)`][parallel-threads] runs n trials at once, with no
  GIL in the way.
  Measured on 32 logical CPUs, 64 trials of a ~50 ms CPU-bound objective finished 6.7× faster at
  `nJobs: 8` and 9.2× faster at `nJobs: 16` than at `nJobs: 1`, while Python Optuna's time stays flat
  because the GIL runs one Python thread at a time ([Performance](#performance)).
  - Trials run on [OptunaSharp's own dedicated threads by default, or on the .NET ThreadPool][parallel-executor].
  - Sampling is serial per sampler, as in Optuna, unless
    [`SamplerConcurrency.PerWorker`][parallel-sampling] gives each worker its own Random, TPE, CMA-ES,
    VkD-CMA or Grid sampler.
  - A `CancellationToken`, or `StopSignals` for Ctrl+C and SIGTERM,
    [stops a run gracefully][parallel-stopping]: no new trial starts, and the trials in flight finish
    and are stored.
  - Studies, trials, samplers, pruners and storages are safe to share between threads;
    [`docs/THREADING.md`](docs/THREADING.md) has the design, its guarantees and its limits.
- **Integrations** - a native [scikit-learn-style `OptunaSearchCV`][i-scikit-learn],
  [`LightGBMTuner`][i-lightgbm], [ML.NET][i-ml-net], [training-loop pruning callbacks][i-pruning],
  [Microsoft.Extensions.DependencyInjection registrations][api-extensions], and the
  [`optunasharp` command-line tool][i-cli], a port of the `optuna` CLI.
- **Python bridges** - [OptunaHub packages][i-optunahub], [BoTorch][i-botorch], [pycma][i-pycma] and
  [pymoo][i-pymoo], driven in-process through [pythonnet][i-pythonnet].
- **Reproducible and tested** - [a seeded run][faq-reproducible] is **bit-identical to Python** wherever
  the path is deterministic, enforced by a [live-Optuna parity harness][parity] (pythonnet) that asks
  real Optuna for each answer in-process and asserts that OptunaSharp matches. Everything builds and
  tests on **net8.0** and **net10.0** across Windows, Linux and macOS, behind strong-naming and package
  gates, and all 20 packages release together from one tag.

## Install

```bash
dotnet add package OptunaSharp
```

The package targets **.NET 8** and **.NET 10**. [Storages][i-storage],
[cloud artifact stores][i-artifacts], [Plotly figures][i-plotly], [the CLI][i-cli] and
[the Python bridges][i-python] are separate `OptunaSharp.*` packages (listed in
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

## What is Optuna?

Optuna is an automatic hyperparameter optimization framework with an imperative,
[*define-by-run*][define-by-run] API. The two [core concepts][intro-model]:

- **[Study][study-and-trials]** - an optimization session over an objective function.
- **[Trial][study-and-trials]** - a single evaluation of the objective, during which hyperparameters
  are [*suggested*][suggest].

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

The C# quick start above is the same study, written against the same define-by-run shape:
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
- **Recipes** -
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
- **Integrations** - [Overview](https://scisharp.github.io/OptunaSharp/integrations/index.html)
  - Model training -
    [scikit-learn-style search](https://scisharp.github.io/OptunaSharp/integrations/training/scikit-learn.html) ·
    [ML.NET](https://scisharp.github.io/OptunaSharp/integrations/training/ml-net.html) ·
    [LightGBM](https://scisharp.github.io/OptunaSharp/integrations/training/lightgbm.html) ·
    [Training-loop pruning](https://scisharp.github.io/OptunaSharp/integrations/training/pruning.html)
  - Storage -
    [SQLite](https://scisharp.github.io/OptunaSharp/integrations/storage/sqlite.html) ·
    [EF Core (PostgreSQL / MySQL)](https://scisharp.github.io/OptunaSharp/integrations/storage/ef-core.html) ·
    [Redis](https://scisharp.github.io/OptunaSharp/integrations/storage/redis/index.html) ·
    [gRPC](https://scisharp.github.io/OptunaSharp/integrations/storage/grpc/index.html)
  - Python bridges -
    [Python.NET runtime](https://scisharp.github.io/OptunaSharp/integrations/python/pythonnet.html) ·
    [OptunaHub](https://scisharp.github.io/OptunaSharp/integrations/python/optunahub.html) ·
    [BoTorch](https://scisharp.github.io/OptunaSharp/integrations/python/botorch.html) ·
    [PyCma](https://scisharp.github.io/OptunaSharp/integrations/python/pycma.html) ·
    [pymoo](https://scisharp.github.io/OptunaSharp/integrations/python/pymoo.html)
  - Cloud artifacts -
    [Amazon S3](https://scisharp.github.io/OptunaSharp/integrations/artifacts/s3.html) ·
    [Google Cloud Storage](https://scisharp.github.io/OptunaSharp/integrations/artifacts/gcs.html) ·
    [Azure Blob Storage](https://scisharp.github.io/OptunaSharp/integrations/artifacts/azure-blob.html) ·
    [Cloudflare R2](https://scisharp.github.io/OptunaSharp/integrations/artifacts/r2.html)
  - Tools -
    [Plotly.NET figures](https://scisharp.github.io/OptunaSharp/integrations/visualization/plotly.html) ·
    [Command-line tool](https://scisharp.github.io/OptunaSharp/integrations/cli.html)
- **Methods** - [Overview](https://scisharp.github.io/OptunaSharp/docs/methods/index.html): a page per
  algorithm, each with its paper - the samplers, pruners, importance evaluators and terminators, and
  the operators and kernels they are built from.
- **API reference** - [OptunaSharp](https://scisharp.github.io/OptunaSharp/api/index.html), generated
  from the XML documentation of the Core package.

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

[study-and-trials]: https://scisharp.github.io/OptunaSharp/docs/study-and-trials.html
[distributions]: https://scisharp.github.io/OptunaSharp/docs/distributions.html
[search-spaces]: https://scisharp.github.io/OptunaSharp/docs/search-spaces.html
[samplers]: https://scisharp.github.io/OptunaSharp/docs/samplers.html
[pruners]: https://scisharp.github.io/OptunaSharp/docs/pruners.html
[multi-objective]: https://scisharp.github.io/OptunaSharp/docs/multi-objective.html
[storage]: https://scisharp.github.io/OptunaSharp/docs/storage.html
[storage-concurrency]: https://scisharp.github.io/OptunaSharp/docs/storage.html#concurrency-workers-studies-and-languages-on-one-journal
[storage-resilience]: https://scisharp.github.io/OptunaSharp/docs/storage.html#resilience-heartbeats-and-failed-trial-retry
[artifacts]: https://scisharp.github.io/OptunaSharp/docs/artifacts.html
[terminators]: https://scisharp.github.io/OptunaSharp/docs/terminators.html
[importance]: https://scisharp.github.io/OptunaSharp/docs/importance.html
[ped-anova]: https://scisharp.github.io/OptunaSharp/docs/importance.html#ped-anova-the-default-evaluator
[fanova]: https://scisharp.github.io/OptunaSharp/docs/importance.html#fanova
[mdi]: https://scisharp.github.io/OptunaSharp/docs/importance.html#mean-decrease-impurity-mdi
[shap]: https://scisharp.github.io/OptunaSharp/docs/importance.html#shapley-shap
[visualization]: https://scisharp.github.io/OptunaSharp/docs/visualization.html
[logging]: https://scisharp.github.io/OptunaSharp/docs/logging.html
[exceptions]: https://scisharp.github.io/OptunaSharp/docs/exceptions.html
[intro-model]: https://scisharp.github.io/OptunaSharp/docs/intro.html#the-study--trial-model
[parity]: https://scisharp.github.io/OptunaSharp/docs/intro.html#how-parity-is-verified
[first-study]: https://scisharp.github.io/OptunaSharp/docs/getting-started.html#your-first-study
[lightweight]: https://scisharp.github.io/OptunaSharp/docs/key-features/first.html
[define-by-run]: https://scisharp.github.io/OptunaSharp/docs/key-features/search-space.html#defining-spaces-with-control-flow
[suggest]: https://scisharp.github.io/OptunaSharp/docs/key-features/search-space.html#the-three-suggest-methods
[parallel-threads]: https://scisharp.github.io/OptunaSharp/docs/key-features/parallelization.html#multi-thread-optimization
[parallel-executor]: https://scisharp.github.io/OptunaSharp/docs/key-features/parallelization.html#which-threads-run-the-trials
[parallel-sampling]: https://scisharp.github.io/OptunaSharp/docs/key-features/parallelization.html#sampling-in-parallel-and-openblas-threads
[parallel-stopping]: https://scisharp.github.io/OptunaSharp/docs/key-features/parallelization.html#stopping-a-run-cancellation-ctrlc-and-sigterm
[faq-constraints]: https://scisharp.github.io/OptunaSharp/docs/faq.html#how-can-i-optimize-a-model-with-some-constraints
[faq-reproducible]: https://scisharp.github.io/OptunaSharp/docs/faq.html#how-can-i-obtain-reproducible-optimization-results
[ask-and-tell]: https://scisharp.github.io/OptunaSharp/docs/recipes/ask-and-tell.html
[enqueue]: https://scisharp.github.io/OptunaSharp/docs/recipes/specify-params.html
[attributes]: https://scisharp.github.io/OptunaSharp/docs/recipes/attributes.html
[callback]: https://scisharp.github.io/OptunaSharp/docs/recipes/callback.html
[wilcoxon]: https://scisharp.github.io/OptunaSharp/docs/recipes/wilcoxon-pruner.html
[s-tpe]: https://scisharp.github.io/OptunaSharp/docs/samplers.html#tpesampler
[s-cma-es]: https://scisharp.github.io/OptunaSharp/docs/samplers.html#cmaessampler
[s-gp]: https://scisharp.github.io/OptunaSharp/docs/samplers.html#gpsampler
[s-grid-bruteforce]: https://scisharp.github.io/OptunaSharp/docs/samplers.html#gridsampler-and-bruteforcesampler
[s-qmc-random]: https://scisharp.github.io/OptunaSharp/docs/samplers.html#qmcsampler-and-randomsampler
[s-partial-fixed]: https://scisharp.github.io/OptunaSharp/docs/samplers.html#partialfixedsampler
[methods]: https://scisharp.github.io/OptunaSharp/docs/methods/index.html
[m-nsga-ii]: https://scisharp.github.io/OptunaSharp/docs/methods/nsga-ii.html
[m-nsga-iii]: https://scisharp.github.io/OptunaSharp/docs/methods/nsga-iii.html
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
[m-median]: https://scisharp.github.io/OptunaSharp/docs/methods/median.html
[m-percentile]: https://scisharp.github.io/OptunaSharp/docs/methods/percentile.html
[m-successive-halving]: https://scisharp.github.io/OptunaSharp/docs/methods/successive-halving.html
[m-hyperband]: https://scisharp.github.io/OptunaSharp/docs/methods/hyperband.html
[m-threshold]: https://scisharp.github.io/OptunaSharp/docs/methods/threshold.html
[m-patient]: https://scisharp.github.io/OptunaSharp/docs/methods/patient.html
[m-nop]: https://scisharp.github.io/OptunaSharp/docs/methods/nop.html
[integrations]: https://scisharp.github.io/OptunaSharp/integrations/index.html
[i-scikit-learn]: https://scisharp.github.io/OptunaSharp/integrations/training/scikit-learn.html
[i-ml-net]: https://scisharp.github.io/OptunaSharp/integrations/training/ml-net.html
[i-lightgbm]: https://scisharp.github.io/OptunaSharp/integrations/training/lightgbm.html
[i-pruning]: https://scisharp.github.io/OptunaSharp/integrations/training/pruning.html
[i-storage]: https://scisharp.github.io/OptunaSharp/integrations/storage/index.html
[i-sqlite]: https://scisharp.github.io/OptunaSharp/integrations/storage/sqlite.html
[i-ef-core]: https://scisharp.github.io/OptunaSharp/integrations/storage/ef-core.html
[i-redis]: https://scisharp.github.io/OptunaSharp/integrations/storage/redis/index.html
[i-grpc]: https://scisharp.github.io/OptunaSharp/integrations/storage/grpc/index.html
[i-python]: https://scisharp.github.io/OptunaSharp/integrations/python/index.html
[i-pythonnet]: https://scisharp.github.io/OptunaSharp/integrations/python/pythonnet.html
[i-optunahub]: https://scisharp.github.io/OptunaSharp/integrations/python/optunahub.html
[i-botorch]: https://scisharp.github.io/OptunaSharp/integrations/python/botorch.html
[i-pycma]: https://scisharp.github.io/OptunaSharp/integrations/python/pycma.html
[i-pymoo]: https://scisharp.github.io/OptunaSharp/integrations/python/pymoo.html
[i-artifacts]: https://scisharp.github.io/OptunaSharp/integrations/artifacts/index.html
[i-s3]: https://scisharp.github.io/OptunaSharp/integrations/artifacts/s3.html
[i-gcs]: https://scisharp.github.io/OptunaSharp/integrations/artifacts/gcs.html
[i-azure-blob]: https://scisharp.github.io/OptunaSharp/integrations/artifacts/azure-blob.html
[i-r2]: https://scisharp.github.io/OptunaSharp/integrations/artifacts/r2.html
[i-plotly]: https://scisharp.github.io/OptunaSharp/integrations/visualization/plotly.html
[i-cli]: https://scisharp.github.io/OptunaSharp/integrations/cli.html
[api-extensions]: https://scisharp.github.io/OptunaSharp/api/OptunaSharp.Extensions.html
