# Easy parallelization

Optimization parallelizes naturally: independent trials can run at the same time, coordinating through
a shared [storage](../storage.md). OptunaSharp supports three scales.

| Scale | How | Recommended storage |
|-------|-----|---------------------|
| Multi-thread | `nJobs` in `Optimize` | `InMemoryStorage`, `JournalStorage`, `RDBStorage` |
| Multi-process | Several processes, shared storage | `JournalStorage`, `RDBStorage` |
| Multi-node | Same study on many machines | `RDBStorage`; `GrpcStorageProxy` at very large scale |

## Multi-thread optimization

Run several trials in parallel within one process by setting `nJobs`:

```csharp
using OptunaSharp;
using OptunaSharp.Storages.Journal;

double Objective(Trial trial)
{
    double x = trial.SuggestFloat("x", -10, 10);
    return (x - 2) * (x - 2);
}

var storage = new JournalStorage(new JournalFileBackend("./journal.log"));
var study = Optuna.CreateStudy(storage: storage);
study.Optimize(Objective, nTrials: 20, nJobs: 4);   // up to 4 trials at once
```

`nJobs: -1` uses the processor count. Unlike CPython — where the GIL long limited thread parallelism —
.NET threads run in genuine parallel, so multi-threaded trial execution is effective today.
(OptunaSharp's numeric kernels stay single-threaded; the parallelism is at the *trial* level.)

### Which threads run the trials

By default the `nJobs` workers are OptunaSharp's own dedicated threads. They are created on demand,
reused by later trials and later `Optimize` calls, and end after 30 s without a trial. They never come
from the .NET thread pool, so the pool stays free for your application and for the storage clients that
complete their calls on it (StackExchange.Redis, gRPC). To run the trials on the thread pool instead,
give a `TrialExecutorOptions` at one of four levels. Each `Optimize` call uses the first level that is
set, in this order:

```csharp
using OptunaSharp;
using OptunaSharp.Extensions;

var pool = new TrialExecutorOptions { Executor = TrialExecutorKind.ThreadPool };

// 1. One call.
study.Optimize(Objective, nTrials: 20, nJobs: 4, executorOptions: pool);

// 2. One study, for all its later calls.
study.ExecutorOptions = pool;

// 3. Every study a DI container's study factory builds.
services.AddOptunaStudyFactory(o => o.DefaultExecutorOptions = pool);

// 4. The whole process.
TrialExecutorOptions.Default = pool;
```

A call reads its options once, when it starts, so changing them later affects only later calls.
`ThreadIdleTimeout` sets how long an idle dedicated thread waits for its next trial (30 s by default;
`Timeout.InfiniteTimeSpan` keeps it). `nJobs: 1` always runs on the calling thread. Python has no such
choice: `n_jobs` always runs on a `ThreadPoolExecutor` of Optuna's own threads, which is what the default
ports.

> **The thread pool has limits that OptunaSharp does not guard.** A trial blocks its thread for the
> whole trial. Once `nJobs` reaches the pool's minimum worker count (the processor count by default),
> the pool adds threads only about once a second. Fewer trials then run at once than `nJobs` asks for,
> StackExchange.Redis and gRPC calls stall or time out, and your application's own pool work waits.
> Measured on 32 cores, 192 trials of 1 s at `nJobs: 96`: the pool ran at most 33 at once and took
> 6.1 s, while dedicated threads ran 96 at once and took 2.0–2.3 s. If you hit this, go back to the
> default `DedicatedThreads`, which reuses OptunaSharp's own threads, or raise the pool's minimum with
> `ThreadPool.SetMinThreads`. The design is in
> [THREADING.md](https://github.com/SciSharp/OptunaSharp/blob/master/docs/THREADING.md).

### Stopping a run: cancellation, Ctrl+C and SIGTERM

`Optimize` takes a `CancellationToken` as `ct`:

- Once the token is cancelled, no new trial starts, and the trials in flight finish and are stored.
  `Optimize` then throws `OperationCanceledException` carrying the token.
- An already cancelled token runs no trial.
- Every trial sees the token as `trial.CancellationToken`, so a long objective can stop early. A trial
  whose objective throws for the token is stored as `FAIL`, whatever `catch` holds, as Python stores the
  trial its `KeyboardInterrupt` lands in. An objective that ignores the token runs to its end.

```csharp
using var signals = StopSignals.Listen();   // the first Ctrl+C or SIGTERM cancels signals.Token
try
{
    study.Optimize(trial =>
    {
        double lr = trial.SuggestFloat("lr", 1e-5, 1e-1, log: true);
        for (int epoch = 0; epoch < 100; epoch++)
        {
            trial.CancellationToken.ThrowIfCancellationRequested();   // optional: stop a long trial early
            Train(lr, epoch);
        }
        return Evaluate();
    }, nTrials: 500, nJobs: 4, ct: signals.Token);
}
catch (OperationCanceledException)
{
    Console.WriteLine($"Stopped after {study.Trials.Count} trials.");
}
```

Without a handler, .NET ends the process at Ctrl+C or SIGTERM, and the trials in flight stay `RUNNING`
in a persistent storage, where a journal never fails them.

- `StopSignals` is opt-in, because a signal handler is process-wide. The first SIGINT (Ctrl+C) or
  SIGTERM (`docker stop`, a Kubernetes pod's termination) cancels its token, and a second one ends the
  process at once.
- It costs no CPU until a signal arrives: the handlers are operating-system callbacks.
- A container's grace period, 10 s for Docker and 30 s for Kubernetes by default, must cover the trials
  in flight.
- In a .NET Generic Host, pass `IHostApplicationLifetime.ApplicationStopping` as `ct` instead: the host
  already handles both signals.
- On Windows, a process started with Ctrl+C disabled never receives it, with or without `StopSignals`.
  Windows lets a parent pass that setting on to its children, and some launchers do.

## Multi-process optimization

Point several processes at one shared storage and each pulls unevaluated work. `InMemoryStorage`
can't be shared, so use a `JournalStorage` (a shared file) or an `RDBStorage`. Have every worker
create the study with `loadIfExists: true` so they join instead of colliding on the name:

```csharp
using OptunaSharp;
using OptunaSharp.Storages.Journal;

var storage = new JournalStorage(new JournalFileBackend("./journal.log"));
var study = Optuna.CreateStudy(
    studyName: "journal_multiprocess", storage: storage, loadIfExists: true);
study.Optimize(Objective, nTrials: 3);
```

Launch the same program from several shells (or a process pool) and the trials accumulate in the one
journal. On a single host the file lock keeps writers safe; across machines over NFS the file lock can
misbehave, so prefer an RDB — or a [Redis journal backend](../../integrations/storage/redis/index.md) —
for multi-machine runs.

## Multi-node optimization

For many machines, use an `RDBStorage` on a real database server (PostgreSQL or MySQL via
[`OptunaSharp.Storage.EFCore`](../../integrations/storage/ef-core.md)). Every machine runs the same
script against the same database URL with `loadIfExists: true`; the first creates the study, the rest
attach.

At **very large** scale — thousands of worker processes — a single database server can become the
bottleneck. The [gRPC storage proxy](../../integrations/storage/grpc/index.md) puts a lightweight
network layer in front of an `RDBStorage`, so many workers share one backend efficiently. It even
interoperates with Python Optuna's proxy on the wire, in both directions.

## Relationship to Optuna

| Optuna (Python) | OptunaSharp |
|-----------------|-------------|
| `study.optimize(objective, n_trials=20, n_jobs=4)` | `study.Optimize(Objective, nTrials: 20, nJobs: 4)` |
| Ctrl+C raises `KeyboardInterrupt` out of `study.optimize` | `study.Optimize(…, ct: signals.Token)` throws `OperationCanceledException`, with `using var signals = StopSignals.Listen()` |
| `JournalStorage(JournalFileBackend("./journal.log"))` | `new JournalStorage(new JournalFileBackend("./journal.log"))` |
| `create_study(storage="mysql://…", load_if_exists=True)` | `Optuna.CreateStudy(storage: new RDBStorage("mysql://…"), loadIfExists: true)` (EF Core package) |
| `GrpcStorageProxy(host=…, port=…)` | `new GrpcStorageProxy(host: …, port: …)` |

## See also

[Storage](../storage.md) · [Journal-file storage](../recipes/journal-storage.md) ·
[gRPC storage proxy](../../integrations/storage/grpc/index.md) ·
[EF Core storage](../../integrations/storage/ef-core.md) · the [API reference](../../api/index.md).
