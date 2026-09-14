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
| `JournalStorage(JournalFileBackend("./journal.log"))` | `new JournalStorage(new JournalFileBackend("./journal.log"))` |
| `create_study(storage="mysql://…", load_if_exists=True)` | `Optuna.CreateStudy(storage: new RDBStorage("mysql://…"), loadIfExists: true)` (EF Core package) |
| `GrpcStorageProxy(host=…, port=…)` | `new GrpcStorageProxy(host: …, port: …)` |

## See also

[Storage](../storage.md) · [Journal-file storage](../recipes/journal-storage.md) ·
[gRPC storage proxy](../../integrations/storage/grpc/index.md) ·
[EF Core storage](../../integrations/storage/ef-core.md) · the [API reference](../../api/index.md).
