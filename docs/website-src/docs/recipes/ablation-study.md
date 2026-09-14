# Ablation study with brute force

An **ablation study** systematically evaluates every combination of techniques or hyperparameters to
see which components matter. Shell scripts with nested loops or SLURM array jobs get unwieldy fast —
especially when the search space is *conditional* ("Adam has `beta1`/`beta2`, SGD has `momentum`").
[`BruteForceSampler`](../methods/brute-force.md) solves this: it exhaustively enumerates the
define-by-run search space, conditional branches included, and stops the study automatically once
every combination has been tried.

## Basic ablation

Define the grid with `SuggestCategorical` and a **stepped** `SuggestFloat`, and let brute force try
every point. Note there is no `nTrials` — the sampler decides when the sweep is complete.

```csharp
using System.Collections.Generic;
using OptunaSharp;
using OptunaSharp.Samplers;

double Objective(Trial trial)
{
    string optimizer  = trial.SuggestCategorical("optimizer", new[] { "Adam", "SGD", "RMSprop" });
    double lr         = trial.SuggestFloat("lr", 0.001, 0.01, step: 0.001);   // finite step is required
    string lrSchedule = trial.SuggestCategorical("lr_schedule", new[] { "constant", "cosine" });

    // Replace with real training and evaluation; here a deterministic mock score.
    var mock = new Dictionary<string, double> { ["Adam"] = 0.9, ["SGD"] = 0.85, ["RMSprop"] = 0.88 };
    return mock[optimizer] + lr * 10 + (lrSchedule == "cosine" ? 0.01 : 0.0);
}

var study = Optuna.CreateStudy(direction: StudyDirection.Maximize, sampler: new BruteForceSampler());
study.Optimize(Objective);   // no nTrials: runs until every combination is evaluated

System.Console.WriteLine($"Total trials: {study.Trials.Count}");
```

> `BruteForceSampler` requires every continuous parameter to have a finite `step` — a plain
> `SuggestFloat` without `step` describes an infinite space and raises a `ValueError`.

## Conditional (hierarchical) search space

The real advantage over grid search is conditional structure. Different optimizers expose different
hyperparameters; brute force walks every valid path:

```csharp
double ObjectiveConditional(Trial trial)
{
    string optimizer = trial.SuggestCategorical("optimizer", new[] { "Adam", "SGD" });
    string config;
    if (optimizer == "Adam")
    {
        double beta1 = trial.SuggestCategorical("beta1", new[] { 0.9, 0.95 });
        double beta2 = trial.SuggestCategorical("beta2", new[] { 0.999, 0.9999 });
        config = $"Adam(beta1={beta1}, beta2={beta2})";
    }
    else
    {
        double momentum = trial.SuggestCategorical("momentum", new[] { 0.0, 0.9, 0.99 });
        bool   nesterov = trial.SuggestCategorical("nesterov", new[] { true, false });
        config = $"SGD(momentum={momentum}, nesterov={nesterov})";
    }
    double lr = trial.SuggestFloat("lr", 0.001, 0.01, step: 0.001);
    return MockScore(config) + lr;   // your training here
}

// A deterministic stand-in for real training (stable across runs, unlike GetHashCode).
static double MockScore(string config)
{
    int h = 0;
    foreach (char c in config) h = (h * 31 + c) % 100;
    return h / 100.0;
}

var study = Optuna.CreateStudy(direction: StudyDirection.Maximize, sampler: new BruteForceSampler());
study.Optimize(ObjectiveConditional);
```

This explores every path: 4 combinations for Adam (2 `beta1` × 2 `beta2`) and 6 for SGD
(3 `momentum` × 2 `nesterov`), each combined with 10 learning rates — `(4 + 6) × 10 = 100` trials.
Enumerating those conditional branches by hand in a shell script is exactly the boilerplate the
define-by-run API removes.

## Distributed ablation on a cluster

On an HPC cluster with a shared filesystem, [`JournalStorage`](journal-storage.md) lets many worker
processes collaborate on one study with no database server. Save the script and launch it from several
nodes (or a SLURM array); each worker picks up unevaluated combinations, and `loadIfExists: true` has
them all join the same study:

```csharp
using OptunaSharp;
using OptunaSharp.Samplers;
using OptunaSharp.Storages.Journal;

var storage = new JournalStorage(new JournalFileBackend("/shared/nfs/ablation_journal.log"));
var study = Optuna.CreateStudy(
    studyName: "my-ablation", storage: storage,
    direction: StudyDirection.Maximize, sampler: new BruteForceSampler(),
    loadIfExists: true);
study.Optimize(Objective);
```

Because `JournalStorage` replays its log on startup, crash recovery is automatic — relaunch a failed
worker and it skips completed trials and resumes.

## Retrying failed and stale trials

`BruteForceSampler` treats `Fail` trials as *visited* and won't re-sample them, and a worker that hangs
leaves its trial `Running` forever. `JournalStorage` has no heartbeat, so it can't detect stale trials.
For automatic retry, use SQLite [`RDBStorage`](rdb.md) with the heartbeat mechanism and
`RetryFailedTrialCallback`:

```csharp
using OptunaSharp;
using OptunaSharp.Samplers;
using OptunaSharp.Storages;   // RetryFailedTrialCallback (a deprecated alias of RetryHeartbeatStaleTrialCallback)

using var storage = new RDBStorage(
    "sqlite:///ablation.db",
    heartbeatInterval: 60,    // record a heartbeat every 60 seconds
    gracePeriod: 120,         // mark a trial Fail if no heartbeat for 120 seconds
    failedTrialCallback: new RetryFailedTrialCallback(maxRetry: 3).Invoke);

var study = Optuna.CreateStudy(
    studyName: "my-ablation", storage: storage,
    direction: StudyDirection.Maximize, sampler: new BruteForceSampler(),
    loadIfExists: true);
study.Optimize(Objective);
```

When a worker's heartbeat stops, the storage marks the stale trial `Fail` after `gracePeriod`, and
`RetryFailedTrialCallback` re-enqueues its parameters so another worker retries that combination.

## Relationship to Optuna

| Optuna (Python) | OptunaSharp |
|-----------------|-------------|
| `optuna.samplers.BruteForceSampler()` | `new BruteForceSampler()` |
| `study.optimize(objective)` (no `n_trials`) | `study.Optimize(Objective)` |
| `trial.suggest_float("lr", 0.001, 0.01, step=0.001)` | `trial.SuggestFloat("lr", 0.001, 0.01, step: 0.001)` |
| `JournalStorage(journal.JournalFileBackend("/shared/…"))` | `new JournalStorage(new JournalFileBackend("/shared/…"))` |
| `RDBStorage(url=…, heartbeat_interval=60, grace_period=120, failed_trial_callback=RetryFailedTrialCallback(max_retry=3))` | `new RDBStorage("sqlite:///ablation.db", heartbeatInterval: 60, gracePeriod: 120, failedTrialCallback: new RetryFailedTrialCallback(maxRetry: 3).Invoke)` |

## See also

[Brute-Force Search method page](../methods/brute-force.md) · [Journal-file storage](journal-storage.md) ·
[Storage](../storage.md) · [Saving/Resuming with RDB](rdb.md) · the [API reference](../../api/index.md).
