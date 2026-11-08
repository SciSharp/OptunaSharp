# FAQ

A grab-bag of practical questions, ported one-to-one from Optuna's
[FAQ](https://optuna.readthedocs.io/en/stable/faq.html) and rewritten against the C# API. Where the
answer is genuinely different on .NET — no `pickle`, no GIL, storage-backed persistence instead of
object serialization — the difference is called out inline as a **Divergence**.

Throughout, remember C# has **no `**` operator**: Optuna's `x ** 2` becomes `x * x` (or
`Math.Pow(x, 2)` when the exponent isn't a small constant), and `(x - 2) ** 2` becomes
`(x - 2) * (x - 2)`.

## Can I use OptunaSharp with X? (where X is your favorite ML library)

OptunaSharp is **define-by-run**: it optimizes any function you can call from .NET, so it drops in
around whatever trainer you already use — you sample hyperparameters at the top of an objective, run
your own training/validation, and return the score. Nothing about the model library leaks into
OptunaSharp.

Ready-made hooks live in [Integrations](../integrations/index.md): a native scikit-learn-style
[`OptunaSearchCV`](../integrations/training/scikit-learn.md), an [ML.NET](../integrations/training/ml-net.md)
adapter, a [LightGBM](../integrations/training/lightgbm.md) tuner, and training-loop
[pruning callbacks](../integrations/training/pruning.md). If you want an *algorithm* from the Python
ecosystem rather than a trainer, the [Python bridges](../integrations/index.md) run
[BoTorch](../integrations/python/botorch.md), [PyCma](../integrations/python/pycma.md),
[pymoo](../integrations/python/pymoo.md), and [OptunaHub](../integrations/python/optunahub.md) modules
from C#. Runnable, end-to-end programs (XGBoost via ML.NET, conditional search spaces, multi-objective
Pareto, artifacts) are under the repository's [`examples/`](https://github.com/SciSharp/OptunaSharp/tree/main/examples)
directory.

## How to define objective functions that have own arguments?

There are two ways.

First, a small class can hold the extra arguments as fields and expose an objective method:

```csharp
using OptunaSharp;

// Holds the implementation-specific arguments as fields, so the objective can read them.
sealed class Objective
{
    private readonly double _minX;
    private readonly double _maxX;

    public Objective(double minX, double maxX)
    {
        _minX = minX;
        _maxX = maxX;
    }

    // Signature matches Func<Trial, double>: OptunaSharp calls this per trial.
    public double Invoke(Trial trial)
    {
        double x = trial.SuggestFloat("x", _minX, _maxX);
        return (x - 2) * (x - 2);
    }
}

// Execute an optimization by using an `Objective` instance.
var study = Optuna.CreateStudy();
study.Optimize(new Objective(-100, 100).Invoke, nTrials: 100);
```

Second, a **lambda** (closure) can capture the extra arguments — the direct analog of Python's
`lambda`/`functools.partial`:

```csharp
using OptunaSharp;

// Objective function that takes three arguments.
double Objective(Trial trial, double minX, double maxX)
{
    double x = trial.SuggestFloat("x", minX, maxX);
    return (x - 2) * (x - 2);
}

double minX = -100, maxX = 100;

// Wrap it in a lambda that closes over the extra arguments.
var study = Optuna.CreateStudy();
study.Optimize(trial => Objective(trial, minX, maxX), nTrials: 100);
```

A closure is also how you reuse a dataset loaded once, instead of reloading it in each trial: capture
the loaded data in the lambda.

## Can I use OptunaSharp without remote RDB servers?

Yes.

In the simplest form, OptunaSharp works with an in-memory storage (the default — nothing to
configure):

```csharp
var study = Optuna.CreateStudy();
study.Optimize(Objective, nTrials: 100);
```

If you want to save and resume studies, use **SQLite** as local storage. `RDBStorage` lives in the
**`OptunaSharp.Storage.Sqlite`** package:

```csharp
using OptunaSharp;
using OptunaSharp.Storages;   // OptunaSharp.Storage.Sqlite

using var storage = new RDBStorage("sqlite:///example.db");
var study = Optuna.CreateStudy(studyName: "foo_study", storage: storage);
study.Optimize(Objective, nTrials: 100);   // the study's state is persisted to the local SQLite file
```

See the [Storage guide](storage.md) and the [Saving/Resuming with RDB recipe](recipes/rdb.md).

## How can I save and resume studies?

**Divergence from Optuna.** Python can `pickle` or `joblib.dump` an in-memory `Study` object.
OptunaSharp does **not** serialize the `Study` object; it persists a study through a **storage
backend** and resumes it by name. This is the same mechanism Optuna recommends for cross-version
durability, so there is no separate object-serialization path to maintain.

Create the study on a durable storage:

```csharp
using OptunaSharp;
using OptunaSharp.Storages;   // OptunaSharp.Storage.Sqlite

using var storage = new RDBStorage("sqlite:///study.db");
var study = Optuna.CreateStudy(studyName: "foo", storage: storage);
study.Optimize(Objective, nTrials: 100);
```

Resume it later — same name, same storage, `loadIfExists: true`:

```csharp
using var storage = new RDBStorage("sqlite:///study.db");
var study = Optuna.CreateStudy(studyName: "foo", storage: storage, loadIfExists: true);

Console.WriteLine("Best trial until now:");
Console.WriteLine($" Value: {study.BestTrial.Value}");
Console.WriteLine(" Params:");
foreach (var (key, value) in study.BestTrial.Params)
    Console.WriteLine($"    {key}: {value}");
```

To move a study across OptunaSharp/Optuna versions whose schema changed, migrate the database with the
CLI's `optunasharp storage upgrade` (see the [command-line guide](../integrations/cli.md) and the
[Storage guide](storage.md)). For a file-based, server-less alternative that is also
cross-language-resumable, use [journal-file storage](recipes/journal-storage.md).

## How to suppress log messages of OptunaSharp?

By default OptunaSharp shows messages at the `Info` level. Change the level with
`OptunaLogging.SetVerbosity`:

```csharp
using OptunaSharp;
using OptunaSharp.Logging;

OptunaLogging.SetVerbosity(LogLevel.Warning);

var study = Optuna.CreateStudy();
study.Optimize(Objective, nTrials: 100);
// Per-trial 'Trial 0 finished with value: …' lines are now suppressed.
```

See the [Logging guide](logging.md) for handlers and custom sinks.

## How to save machine learning models trained in objective functions?

OptunaSharp stores hyperparameter values and their objective values, but it discards intermediate
objects such as trained models. To keep models or weights, use an **artifact store** — write the
model to a file with whatever serializer your framework provides, then upload the file and remember
its id on the trial:

```csharp
using OptunaSharp;
using OptunaSharp.Artifacts;

var artifactStore = new FileSystemArtifactStore("./artifacts");   // any directory (created if missing)

double Objective(Trial trial)
{
    double svcC = trial.SuggestFloat("svc_c", 1e-10, 1e10, log: true);
    var clf = TrainClassifier(svcC);          // your training code

    // Serialize the model to a file, then hand that file to the artifact store.
    const string modelPath = "model.bin";
    clf.Save(modelPath);                       // however your framework serializes
    string artifactId = ArtifactFunctions.UploadArtifact(artifactStore, modelPath, trial);
    trial.SetUserAttr("artifact_id", artifactId);

    return 1.0 - AccuracyScore(clf);
}

var study = Optuna.CreateStudy();
study.Optimize(Objective, nTrials: 100);
```

To retrieve models, list them with `GetAllArtifactMeta` and pull one down with `DownloadArtifact`:

```csharp
// List all models.
foreach (ArtifactMeta meta in ArtifactFunctions.GetAllArtifactMeta(study))
    Console.WriteLine($"{meta.ArtifactId}  {meta.Filename}  {meta.Mimetype}");

// Download the best model.
FrozenTrial best = study.BestTrial;
var bestArtifactId = (string)best.UserAttrs["artifact_id"]!;
ArtifactFunctions.DownloadArtifact(artifactStore, "best_model.bin", bestArtifactId);
```

For the full protocol and cloud backends (S3/GCS/Azure/R2), see the [Artifacts recipe](recipes/artifacts.md)
and [guide](artifacts.md).

## How can I obtain reproducible optimization results?

Give the sampler a fixed `seed`:

```csharp
using OptunaSharp;
using OptunaSharp.Samplers;

var sampler = new TPESampler(seed: 10);   // make the sampler behave deterministically
var study = Optuna.CreateStudy(sampler: sampler);
study.Optimize(Objective, nTrials: 100);
```

To make pruning by `HyperbandPruner` reproducible, also fix the study's `studyName` in addition to the
seed.

> **OptunaSharp guarantee.** Every sampler consumes NumPy's Mersenne-Twister (`MT19937`) through the
> same `LazyRandomState` port Optuna uses. Under a fixed seed, the suggested-parameter sequence is
> **bit-for-bit identical to Python Optuna** wherever the path is deterministic — this is exactly what
> the live-Optuna parity tests assert.

Two caveats carry over from Optuna:

1. Distributed or parallel optimization is inherently non-deterministic; run sequentially (`nJobs: 1`)
   if you need to reproduce a result.
2. If your objective itself is non-deterministic (it doesn't return the same value for the same
   params), the run can't be reproduced. Seed your ML library's own randomness where it offers a
   knob.

## How are exceptions from trials handled?

A trial whose objective throws an uncaught exception is recorded as a failure — `TrialState.Fail`.

By default every exception except `TrialPruned` propagates out of `Optimize`, aborting the study. To
keep going, pass the exception types to swallow via the `@catch` argument (a `Type[]`); trials that
throw those types are recorded as failures and the study continues:

```csharp
using OptunaSharp;

var study = Optuna.CreateStudy();
study.Optimize(Objective, nTrials: 100, @catch: new[] { typeof(ArithmeticException) });
```

`TrialPruned` is never propagated — it is the prune signal, not an error. Failed trials show up in the
log:

```text
[W 2018-12-07 16:38:36,889] Setting status of trial#0 as TrialState.FAIL because of the following error: ValueError('A sample error in objective.')
```

You can also find them by state — either as a table, or by filtering:

```csharp
TrialsTable df = study.TrialsDataframe();                                     // number, state, value, params, …
var failed = study.GetTrials(states: new[] { TrialState.Fail });              // just the failures
```

## How are NaNs returned by trials handled?

A trial that returns `double.NaN` is treated as a failure, but — unlike an uncaught exception — it
does **not** abort the study:

```csharp
double Objective(Trial trial) => double.NaN;   // recorded as TrialState.Fail; the study keeps running
```

```text
[W 2018-12-07 16:41:59,000] Setting status of trial#2 as TrialState.FAIL because the objective function returned nan.
```

## What happens when I dynamically alter a search space?

Because the search space is declared inside each suggestion call — `SuggestFloat`, `SuggestInt`, and
friends — you can, within one study, sample from different ranges in different trials. What the
sampler does with a changed range is defined per sampler.

> See the discussion on the TPE sampler: <https://github.com/optuna/optuna/issues/822>.

## How can I use two GPUs for evaluating two trials simultaneously?

If your training uses a GPU (CUDA) runtime — TorchSharp, ONNX Runtime, ML.NET's GPU trainers — the
`CUDA_VISIBLE_DEVICES` environment variable pins a process to a device, exactly as in Python. Run one
process per GPU, each joined to the **same shared storage** so they collaborate on one study:

```bash
# On one terminal — use the first GPU.
export CUDA_VISIBLE_DEVICES=0
dotnet run --project MyOptimization

# On another terminal — use the second GPU.
export CUDA_VISIBLE_DEVICES=1
dotnet run --project MyOptimization
```

The variable is read by the native CUDA runtime beneath your .NET code, not by OptunaSharp. See the
[CUDA C Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#env-vars).

## How can I test my objective functions?

When testing, you usually want fixed parameter values rather than sampled ones. `FixedTrial` replays a
supplied dictionary of parameters. **Type your objective against `BaseTrial`** — the shared base of
both the live `Trial` and `FixedTrial` — so the same objective can be optimized *and* unit-tested:

```csharp
using OptunaSharp;
using OptunaSharp.Collections;

// Note the parameter type: BaseTrial, so a FixedTrial can drive it too.
double Objective(BaseTrial trial)
{
    double x = trial.SuggestFloat("x", -1.0, 1.0);
    long   y = trial.SuggestInt("y", -5, 5);
    return x + y;
}

Objective(new FixedTrial(new OrderedDict<string, object?> { ["x"] = 1.0,  ["y"] = -1L }));  // 0.0
Objective(new FixedTrial(new OrderedDict<string, object?> { ["x"] = -1.0, ["y"] = -4L }));  // -5.0
```

A `Trial` *is a* `BaseTrial`, so optimizing is a one-line lambda:

```csharp
var study = Optuna.CreateStudy();
study.Optimize(trial => Objective(trial), nTrials: 100);
```

Unit tests then read naturally (MSTest/xUnit shown):

```csharp
[TestMethod]
public void TestObjective()
{
    Assert.AreEqual(1.0,  Objective(new FixedTrial(new OrderedDict<string, object?> { ["x"] = 1.0,  ["y"] = 0L })));
    Assert.AreEqual(-1.0, Objective(new FixedTrial(new OrderedDict<string, object?> { ["x"] = 0.0,  ["y"] = -1L })));
    Assert.AreEqual(0.0,  Objective(new FixedTrial(new OrderedDict<string, object?> { ["x"] = -1.0, ["y"] = 1L })));
}
```

## How do I avoid running out of memory (OOM) when optimizing studies?

If the memory footprint grows as trials accumulate, run the garbage collector periodically. Pass
`gcAfterTrial: true` to `Optimize`, or call `GC.Collect()` from a callback:

```csharp
using OptunaSharp;

var study = Optuna.CreateStudy();
study.Optimize(Objective, nTrials: 10, gcAfterTrial: true);

// `gcAfterTrial: true` is more or less identical to the following.
study.Optimize(Objective, nTrials: 10,
    callbacks: new Action<Study, FrozenTrial>[] { (s, t) => GC.Collect() });
```

There's a runtime cost to collecting, which may be non-negligible if your objective is otherwise very
fast — that's why `gcAfterTrial` defaults to `false`. Unlike collecting inside the objective,
`gcAfterTrial` (and a callback) also runs when an error or a `TrialPruned` is raised.

## How can I output a log only when the best value is updated?

Replace the built-in per-trial logging with your own callback. Store the last-seen best value as a
study user attribute and print only when it changes:

```csharp
using OptunaSharp;
using OptunaSharp.Logging;

// Turn off OptunaSharp's own per-trial notes.
OptunaLogging.SetVerbosity(LogLevel.Warning);

double Objective(Trial trial)
{
    double x = trial.SuggestFloat("x", 0, 1);
    return x * x;
}

void LoggingCallback(Study study, FrozenTrial frozenTrial)
{
    var previousBest = study.UserAttrs.GetValueOrDefault("previous_best_value");
    if (!Equals(previousBest, study.BestValue))
    {
        study.SetUserAttr("previous_best_value", study.BestValue);
        Console.WriteLine(
            $"Trial {frozenTrial.Number} finished with best value: {frozenTrial.Value} " +
            $"and parameters: {string.Join(", ", frozenTrial.Params)}.");
    }
}

var study = Optuna.CreateStudy();
study.Optimize(Objective, nTrials: 100,
    callbacks: new Action<Study, FrozenTrial>[] { LoggingCallback });
```

This callback can report wrong values under `nJobs != 1` (or other distributed setups): the reads and
writes to storage are prone to race conditions.

## How do I suggest variables that represent proportions (a Dirichlet distribution)?

To suggest `n` variables `p[0], …, p[n-1]` with `0 ≤ p[k] ≤ 1` and `p[0] + … + p[n-1] = 1` — useful,
for example, as interpolation weights — sample from a flat
[Dirichlet distribution](https://en.wikipedia.org/wiki/Dirichlet_distribution) with the following
trick:

```csharp
using System.Linq;
using OptunaSharp;
using OptunaSharp.Samplers;

double Objective(Trial trial)
{
    const int n = 5;
    var x = new double[n];
    for (int i = 0; i < n; i++)
        x[i] = -Math.Log(trial.SuggestFloat($"x_{i}", 0, 1));   // x[i] ~ Exp(1)

    double sum = x.Sum();
    for (int i = 0; i < n; i++)
        trial.SetUserAttr($"p_{i}", x[i] / sum);                // (p_0, …, p_{n-1}) ~ Dir(1, …, 1)

    return 0.0;
}

var study = Optuna.CreateStudy(sampler: new RandomSampler());
study.Optimize(Objective, nTrials: 1000);

// Read the sampled proportions back out of the trials' user attributes.
foreach (var t in study.Trials)
{
    var ps = Enumerable.Range(0, 5).Select(i => (double)t.UserAttrs[$"p_{i}"]!);
    // …inspect or plot ps (see the Visualization guide for scatter data).
}
```

Why it works: applying `x = -log(u)` to `u ~ Uniform(0, 1)` yields `x ~ Exp(1)`; for `n` independent
`Exp(1)` variables, normalizing by their sum (`p[i] = x[i] / Σ x[i]`) makes `p` follow the Dirichlet
distribution `Dir(1, …, 1)`. You can verify the transformation via the Jacobian. To visualize the
pairwise scatter of the `p_i`, feed the user attributes into the [Visualization](visualization.md)
layer.

## How can I optimize a model with some constraints?

Constraints are supported by `TPESampler`, `NSGAIISampler`, `NSGAIIISampler`, `GPSampler`, and the
[BoTorch bridge](../integrations/python/botorch.md). Report each constraint from inside the objective
with `trial.SetConstraint(name, value)`, where a value **≤ 0 is feasible**. This example is the Binh
and Korn multi-objective function with two constraints, solved with `NSGAIISampler`:

```csharp
using System.Linq;
using OptunaSharp;
using OptunaSharp.Samplers;

IReadOnlyList<double> Objective(Trial trial)
{
    // Binh and Korn function with constraints.
    double x = trial.SuggestFloat("x", -15, 30);
    double y = trial.SuggestFloat("y", -15, 30);

    // Constraints are feasible when <= 0. The feasible region is a circle centered at (5, 0)
    // intersected with the complement of a circle centered at (8, -3).
    double c0 = (x - 5) * (x - 5) + y * y - 25;
    double c1 = -((x - 8) * (x - 8)) - (y + 3) * (y + 3) + 7.7;
    trial.SetConstraint("c0", c0);
    trial.SetConstraint("c1", c1);

    double v0 = 4 * x * x + 4 * y * y;
    double v1 = (x - 5) * (x - 5) + (y - 5) * (y - 5);
    return new[] { v0, v1 };
}

var study = Optuna.CreateStudy(
    directions: new[] { StudyDirection.Minimize, StudyDirection.Minimize },
    sampler: new NSGAIISampler());
study.Optimize(Objective, nTrials: 32, timeout: 600);

Console.WriteLine($"Number of finished trials: {study.Trials.Count}");
Console.WriteLine("Pareto front:");
foreach (var trial in study.BestTrials.OrderBy(t => t.Values![0]).ThenBy(t => t.Values![1]))
{
    Console.WriteLine($"  Trial#{trial.Number}");
    Console.WriteLine($"    Values={string.Join(", ", trial.Values!)}, " +
                      $"Constraint={string.Join(", ", trial.Constraints.Values)}");
    Console.WriteLine($"    Params: {string.Join(", ", trial.Params)}");
}
```

There are two kinds of constrained optimization: **soft** constraints (allowed to be violated, but the
objective is penalized) and **hard** constraints (must be satisfied). Like Optuna, OptunaSharp adopts
the **soft** form and **does not** provide samplers for hard constraints.

## How can I parallelize optimization?

Three cases, mirroring Optuna:

### 1. Multi-threading on a single node

Set `nJobs` on `Optimize`:

```csharp
study.Optimize(Objective, nTrials: 100, nJobs: 4);
```

> **Divergence from Optuna (an improvement).** In Python, `n_jobs != 1` uses threads and is throttled
> by the GIL, so it only helps while waiting on I/O or C/NumPy work. .NET has **no GIL**, so a
> CPU-bound objective genuinely runs in parallel across threads. Two caveats hold: NumSharp's numeric
> kernels are single-threaded by house rule, and storage writes serialize — and, as under any parallel
> run, results are no longer reproducible (see
> [reproducible results](#how-can-i-obtain-reproducible-optimization-results)).

The workers are OptunaSharp's own dedicated threads, reused across trials and `Optimize` calls, so the
.NET thread pool stays free for your application. To run them on the thread pool instead, pass
`executorOptions: new TrialExecutorOptions { Executor = TrialExecutorKind.ThreadPool }`, or set it once
for a study, a DI container or the process. An `nJobs` at or above the pool's minimum thread count then
runs fewer trials at once and can make Redis and gRPC calls time out; the
[parallelization page](key-features/parallelization.md#which-threads-run-the-trials) has the levels, the
measurements and the remedies.

### 2. Multi-processing on a single node

Point multiple processes at the same [journal-file storage](recipes/journal-storage.md) or a
client/server RDB (PostgreSQL/MySQL via [`OptunaSharp.Storage.EFCore`](../integrations/storage/ef-core.md)).
A journal file can be shared by C# and Python processes at once, and by several studies; the
[journal concurrency section](storage.md#concurrency-workers-studies-and-languages-on-one-journal) lists
what is synchronized, and the limits.

### 3. Multi-processing across multiple nodes

Use a client/server RDB, or put a study behind the [gRPC storage proxy](../integrations/storage/grpc/index.md).
Without a shared, networked backend, cross-node multi-processing isn't possible.

## How can I solve the error that occurs when performing parallel optimization with SQLite?

SQLite is **not recommended** for parallel optimization: it doesn't support the row-locking used to
hand out enqueued trials, its concurrency is limited (you'll hit "database is locked" errors under
contention), and it doesn't work over NFS. For a file-based backend under these scenarios, use
[journal-file storage](recipes/journal-storage.md) instead:

```csharp
using OptunaSharp;
using OptunaSharp.Storages.Journal;

var storage = new JournalStorage(new JournalFileBackend("optuna_journal_storage.log"));
var study = Optuna.CreateStudy(storage: storage);
// …
```

The journal file is byte-compatible with Optuna's `JournalFileBackend`, so a shared log can be resumed
from either runtime. On Windows without symlink privileges, the default lock switches to the
`JournalFileOpenLock` protocol by itself; pass a lock explicitly to choose one — see the
[journal-file recipe](recipes/journal-storage.md).

## Can I monitor trials and fail them automatically when they are killed unexpectedly?

> The heartbeat mechanism is **experimental**; its API may change.

A process running a trial can be killed unexpectedly (for example by a cluster job scheduler),
leaving the trial stuck in `RUNNING`. With a **heartbeat**, a trial whose process stops recording
heartbeats is automatically moved from `Running` to `Fail`:

```csharp
using OptunaSharp;
using OptunaSharp.Storages;   // OptunaSharp.Storage.Sqlite

// Record heartbeats every 60 s; trials with no heartbeat for more than 120 s are failed automatically.
var storage = new RDBStorage("sqlite:///:memory:", heartbeatInterval: 60, gracePeriod: 120);
var study = Optuna.CreateStudy(storage: storage);
study.Optimize(Objective, nTrials: 100);
```

> The heartbeat is meant to be used with `Optimize`. If you drive the loop with `Ask`/`Tell`, fail the
> zombie trials yourself by calling `Tell`:
>
> ```csharp
> using System;
> using OptunaSharp;
>
> var study = Optuna.CreateStudy(storage: storage);
> // Tune this to your workload; here a trial running for over a day is treated as a zombie.
> int gracePeriod = 3600 * 24;
> foreach (var t in study.GetTrials(states: new[] { TrialState.Running }))
>     if (t.DatetimeStart is { } started && (DateTime.Now - started).TotalSeconds > gracePeriod)
>         study.Tell(t.Number, state: TrialState.Fail);
> ```

You can also run a callback to process heartbeat-stale trials. `RetryHeartbeatStaleTrialCallback`
re-enqueues them; it is invoked when a new trial starts:

```csharp
using OptunaSharp;
using OptunaSharp.Storages;   // RDBStorage + RetryHeartbeatStaleTrialCallback

var storage = new RDBStorage(
    "sqlite:///:memory:",
    heartbeatInterval: 60,
    gracePeriod: 120,
    heartbeatStaleTrialCallback: new RetryHeartbeatStaleTrialCallback(maxRetry: 3));

var study = Optuna.CreateStudy(storage: storage);
```

See the [Storage guide](storage.md#resilience-heartbeats-and-failed-trial-retry) for the full
heartbeat surface.

## How can I deal with permutation as a parameter?

There's no direct API for a permutation search space, but you can re-parametrize a permutation of `n`
items as an independent `n`-dimensional **integer** search space using a
[Lehmer code](https://en.wikipedia.org/wiki/Lehmer_code).

A Lehmer code of a sequence is a same-length integer sequence whose `i`-th entry counts how many
later entries are smaller than the `i`-th entry of the permutation. For example, the Lehmer code of
`(3, 1, 4, 2, 0)` is `(3, 1, 2, 1, 0)`. Beyond giving a unique integer encoding, it preserves
"closeness" between permutations (lexicographic order is preserved; the entry-sum equals the minimum
number of adjacent transpositions to reach the identity), which matters for the optimizer. Here it is
on a Euclidean TSP:

```csharp
using System;
using System.Linq;
using OptunaSharp;

// Euclidean coordinates of the cities.
double[][] cityCoordinates =
{
    new[] { 0.0, 0.0 }, new[] { 1.0, 0.0 }, new[] { 0.0, 1.0 },
    new[] { 1.0, 1.0 }, new[] { 2.0, 2.0 }, new[] { -1.0, -1.0 },
};
int n = cityCoordinates.Length;

// Decode a Lehmer code (a list of integers) into a permutation.
int[] Decode(int[] lehmerCode)
{
    var remaining = Enumerable.Range(0, n).ToList();
    var output = new int[lehmerCode.Length];
    for (int i = 0; i < lehmerCode.Length; i++)
    {
        int value = remaining[lehmerCode[i]];
        output[i] = value;
        remaining.Remove(value);
    }
    return output;
}

double Objective(Trial trial)
{
    // Suggest a permutation in its Lehmer-code representation.
    var lehmerCode = new int[n];
    for (int i = 0; i < n; i++)
        lehmerCode[i] = (int)trial.SuggestInt($"x{i}", 0, n - i - 1);
    int[] permutation = Decode(lehmerCode);

    // Total length of the closed tour; index (i-1) wraps around, matching Python's np.roll(perm, 1).
    double total = 0.0;
    for (int i = 0; i < n; i++)
    {
        double[] a = cityCoordinates[permutation[i]];
        double[] b = cityCoordinates[permutation[(i - 1 + n) % n]];
        total += Math.Sqrt((a[0] - b[0]) * (a[0] - b[0]) + (a[1] - b[1]) * (a[1] - b[1]));
    }
    return total;
}

var study = Optuna.CreateStudy();
study.Optimize(Objective, nTrials: 10);

var bestLehmer = study.BestParams.Values.Select(v => (int)Convert.ToInt64(v)).ToArray();
Console.WriteLine(string.Join(", ", Decode(bestLehmer)));
```

## How can I ignore duplicated samples?

OptunaSharp may suggest parameters it evaluated before. To reuse the earlier result instead of
recomputing, look through past trials inside the objective:

```csharp
using System.Linq;
using OptunaSharp;

double Objective(Trial trial)
{
    // Sample parameters.
    long x = trial.SuggestInt("x", -5, 5);
    long y = trial.SuggestInt("y", -5, 5);

    // Consider only completed trials here; you can add TrialState.Pruned / .Fail if you want.
    var considered = trial.Study.GetTrials(deepcopy: false, states: new[] { TrialState.Complete });
    foreach (var t in Enumerable.Reverse(considered))
        if (ParamsEqual(trial.Params, t.Params))
            return t.Value!.Value;   // reuse the existing value for the duplicated params

    // Not a duplicate: compute the objective (2D sphere function).
    return x * x + y * y;
}

// Compare two parameter dictionaries by name and value.
static bool ParamsEqual(IReadOnlyDictionary<string, object?> a, IReadOnlyDictionary<string, object?> b)
    => a.Count == b.Count && a.All(kv => b.TryGetValue(kv.Key, out var v) && Equals(kv.Value, v));

var study = Optuna.CreateStudy();
study.Optimize(Objective, nTrials: 100);
```

## How can I delete all the artifacts uploaded to a study?

After many experiments you may want to remove stored artifacts. **Strongly recommended:** give each
study its own directory or bucket, so deleting that directory/bucket removes everything at once.

> **Warning.** `AddTrial` and `CopyStudy` do **not** copy artifact files linked to a study or trial.
> Make sure **not** to delete artifacts still referenced by a source study or trial. `Remove` is an
> internal feature; OptunaSharp — like Optuna — does not officially support deleting artifacts this
> way and won't guarantee behavior around it.

If you must remove artifacts from code, pass in the storage you already hold and walk the trials:

```csharp
using OptunaSharp;
using OptunaSharp.Artifacts;
using OptunaSharp.Storages;   // BaseStorage

void RemoveArtifacts(Study study, IArtifactStore artifactStore, BaseStorage storage)
{
    // For each trial, remove its uploaded artifacts.
    foreach (var trial in study.Trials)
        foreach (var meta in ArtifactFunctions.GetAllArtifactMeta(trial, storage))
            artifactStore.Remove(meta.ArtifactId);

    // Remove artifacts attached to the study itself.
    foreach (var meta in ArtifactFunctions.GetAllArtifactMeta(study, storage))
        artifactStore.Remove(meta.ArtifactId);
}
```

## Can I specify parameter starting points before optimization?

Yes. Use `Study.EnqueueTrial` to have OptunaSharp evaluate configurations you provide, or
`Study.AddTrial` to register configurations you've already evaluated. See the
[Specify hyperparameters manually recipe](recipes/specify-params.md).

## How can I resolve case sensitivity issues with MySQL?

By default MySQL compares strings **case-insensitively**, but OptunaSharp treats parameter names
case-sensitively — so parameter names that differ only in case collide in MySQL
([EF Core storage](../integrations/storage/ef-core.md)) and only one gets registered:

```csharp
double Objective(Trial trial)
{
    long a = trial.SuggestInt("a", 0, 10);
    long A = trial.SuggestInt("A", 0, 10);
    return a + A;
}
```

Here OptunaSharp treats `a` and `A` as distinct, MySQL does not. Workarounds:

1. **Use a different backend.** PostgreSQL (also EF Core) and SQLite handle case sensitively.
2. **Rename the parameters** to avoid case-only differences (e.g. `a` and `b`).
3. **Change MySQL's collation** to be case-sensitive at the database, table, or column level — see the
   [MySQL documentation](https://dev.mysql.com/doc/refman/9.3/en/charset-syntax.html).

## Notable divergences from Optuna

| Topic | Optuna (Python) | OptunaSharp |
|-------|-----------------|-------------|
| Save/resume a study | `pickle`/`joblib` an in-memory `Study` | No object serialization — persist via a **storage backend** (RDB/journal) and resume by name |
| `n_jobs` multi-threading | Throttled by the GIL | No GIL — CPU-bound objectives genuinely parallelize (numeric kernels stay single-threaded) |
| Exponentiation | `x ** 2` | `x * x` or `Math.Pow(x, 2)` — C# has no `**` |
| Testing objectives | `objective(trial)` duck-types over `Trial`/`FixedTrial` | Type the objective against `BaseTrial` so both `Trial` and `FixedTrial` drive it |
| Journal snapshots | pickled | OptunaSharp's own binary format (log stays cross-language; snapshots don't) |

## See also

[Study and Trials](study-and-trials.md) · [Samplers](samplers.md) · [Storage](storage.md) ·
[Artifacts](artifacts.md) · [Logging](logging.md) · [Recipes](recipes/index.md) ·
[Integrations](../integrations/index.md) · the [API reference](../api/index.md).
