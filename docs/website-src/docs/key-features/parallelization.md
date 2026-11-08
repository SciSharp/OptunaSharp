# Easy parallelization

Trials are independent, so they can run at the same time and meet in a shared [storage](../storage.md). OptunaSharp
parallelizes the same three ways Optuna does:

1. **Multi-thread:** several trials at once in one process, with `nJobs` in `Optimize`.
2. **Multi-process:** several processes on one machine, sharing a storage.
3. **Multi-node:** many machines on one storage (a database or Redis), with a gRPC proxy in front for very large
   fleets.

## Choosing a strategy

| Your trials run in | Use | Storage |
|---|---|---|
| One process | [`nJobs`](#multi-thread-optimization) | `InMemoryStorage`, `JournalStorage`, `RDBStorage` |
| Several processes on one machine | [a shared storage](#multi-process-optimization) | `JournalStorage` (a file), `RDBStorage` |
| Several machines | [a database or Redis](#multi-node-optimization) | `RDBStorage` (PostgreSQL, MySQL), `JournalStorage` over Redis |
| Thousands of workers | [a gRPC proxy](#a-grpc-proxy-for-thousands-of-workers) | `GrpcStorageProxy` in front of an `RDBStorage` |

Start with threads. In .NET they run in parallel, where standard CPython runs one thread of Python code at a
time, and they need no shared file or server. Move to processes when the objective is not thread-safe (a library
with global state) or can crash its process, and to several machines when one is not enough.

## Multi-thread optimization

Set `nJobs` to run that many trials at once:

```csharp
using OptunaSharp;
using OptunaSharp.Storages.Journal;

double Objective(Trial trial)
{
    Console.WriteLine($"Running trial {trial.Number} on thread {Environment.CurrentManagedThreadId}");
    double x = trial.SuggestFloat("x", -10, 10);
    return (x - 2) * (x - 2);
}

var storage = new JournalStorage(new JournalFileBackend("./journal.log"));
var study = Optuna.CreateStudy(storage: storage);
study.Optimize(Objective, nTrials: 20, nJobs: 4);   // up to 4 trials at once
```

`nJobs: -1` runs `Environment.ProcessorCount` trials at once. `nJobs: 1`, the default, runs every trial on the
calling thread.

### What nJobs speeds up

That depends on what limits your trials:

| A trial spends its time | `nJobs` |
|---|---|
| Computing: a CPU-bound objective | Scales with the cores. 64 trials of 50 ms took 3.2 s at `nJobs: 1`, 0.48 s at 8 and 0.35 s at 16. Python Optuna stays at about 3.1 s, held by the GIL. |
| Waiting: a remote service, a subprocess, a file | Scales past the cores, because a waiting thread uses no CPU. 640 trials sleeping 100 ms took 1.26 s at `nJobs: 64` (Python: 1.06 s; ideal 1.0 s). Each trial in flight holds a thread for its whole life. |
| Sampling: a fast objective, a multivariate TPE over many parameters, a high-dimensional CMA-ES | Limited: the workers share the study's sampler, which proposes for one trial at a time. A [sampler per worker](#sampling-in-parallel-and-openblas-threads) lifts the limit for the samplers that support it. Slow model-based samplers gain nothing: a GP study took 7.53 s at `nJobs: 1` and 9.30 s at 8. |
| Storing: a fast objective on a remote or disk-backed storage | Limited by the storage. A file journal flushes every write to disk (80–150 appends per second with 16 workers on an SSD), and a Redis journal ran about 90–130 two-parameter trials per second. See [Storage › Concurrency](../storage.md#concurrency-workers-studies-and-languages-on-one-journal). |

The figures come from the [README's benchmarks](https://github.com/SciSharp/OptunaSharp#performance) (an i9-13900K
with 32 logical CPUs) and the [Storage page](../storage.md#concurrency-workers-studies-and-languages-on-one-journal).

### Your code under nJobs

- **The objective runs on several threads at once.** Anything it shares must be thread-safe. Each trial has a
  `Trial` of its own. Draw random numbers from `Random.Shared` or a generator per trial, never from one shared
  `System.Random` or NumSharp's `np.random`.
- **Callbacks** run on the thread that ran their trial, at the same time as other trials' callbacks.
- **A parallel run is not reproducible,** even with a seeded sampler. As in Optuna, each worker reseeds the
  sampler with fresh entropy before its trial, and trials finish in a different order every run. Use `nJobs: 1`
  when you need the same result twice.
- **`Ask`, `Tell` and every `Trial` method are thread-safe,** so you can also drive trials from threads or async
  code of your own ([below](#async-work-and-your-own-threads)).

[THREADING.md §10](https://github.com/SciSharp/OptunaSharp/blob/main/docs/THREADING.md#10-writing-thread-safe-code-against-optunasharp)
has the full rules, including those for samplers, pruners, storages and log handlers of your own.

### Which threads run the trials

By default the workers are OptunaSharp's own dedicated threads, as Optuna's are the threads of its own
`ThreadPoolExecutor`. They are created on demand, reused by later trials and later `Optimize` calls, and end after
30 s without a trial (`ThreadIdleTimeout`). The .NET thread pool stays free for your application and for the
storage clients that complete their calls on it (StackExchange.Redis, gRPC, the AWS SDK).

To run the trials on the thread pool instead, give a `TrialExecutorOptions`. Each `Optimize` call uses the first of
four levels that is set, and reads it once, when it starts:

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

> **The thread pool suits an `nJobs` well below the core count.** A trial blocks its pool thread for its whole
> life, and past the pool's minimum (the processor count by default) the pool adds about one thread a second. 128
> trials blocking 1.5 s at `nJobs: 64` took 3.2 s on dedicated threads and 6.2 s on the pool, and the
> application's other pool work waited up to 1.6 s for a thread. Redis and gRPC calls can stall or time out
> there too. OptunaSharp does not guard against this: keep the default, or raise the pool's minimum with
> `ThreadPool.SetMinThreads`.

### Sampling in parallel, and OpenBLAS threads

The workers share the study's sampler, as in Optuna, and it proposes for one trial at a time. When proposing,
rather than the objective, limits a run, let each worker sample with an instance of its own:

```csharp
var parallel = new TrialExecutorOptions { SamplerConcurrency = SamplerConcurrency.PerWorker };
study.Optimize(Objective, nTrials: 600, nJobs: 8, executorOptions: parallel);
```

- Each worker's instance is made from the study's sampler and reseeded before each trial, like a separate process
  on the same storage.
- Taking part: `RandomSampler`, `TPESampler`, `CmaEsSampler`, `VkdCmaSampler`, `GridSampler`, and
  `PartialFixedSampler` over one of them, the samplers whose instances never propose the same point twice this
  way. Every other sampler, and any sampler of your own, stays shared.
- At `nJobs: 8`, a multivariate TPE over 10 parameters ran 600 trials in 1.15 s instead of 1.39 s, and a
  40-dimensional CMA-ES 2,000 trials in 0.82 s instead of 2.17 s. A TPE study limited by its objective did not
  change.
- Functions you hand to such a sampler (TPE's `gamma`, `weights`, `constraintsFunc`) then run on several workers
  at once, so they must be thread-safe.

Shared or per worker, trials asked at the same time get different points from every sampler in the `OptunaSharp`
package. The exceptions are by design or Optuna's own:

- The genetic samplers, DE, JANUS-DE, PSO and LM-CMA can propose a point a finished trial already had (a child
  that copies its parent).
- CMA-ES and VkD-CMA give two trials of one generation the same point about once in 65,535 pairs, as Optuna does.
- The Python bridges hand out what the Python code they drive hands out.

[Storage › Concurrency](../storage.md#concurrency-workers-studies-and-languages-on-one-journal) has the
measurements for each sampler.

While several workers run, OptunaSharp runs OpenBLAS (the linear algebra behind CMA-ES, GP and NSGA-III) on one
thread, so the workers do not compete for the cores. `nJobs: 1` leaves OpenBLAS as it is, which keeps sequential
runs identical to Python's. To use another count for the duration of a call:

```csharp
var options = new TrialExecutorOptions { BlasThreads = 4 };   // null, the default, is the automatic policy
```

The count applies to the whole process while the call runs, and the count from before comes back when it ends.

## Stopping a run: cancellation, Ctrl+C and SIGTERM

Inside a run, `study.Stop()` from an objective or a callback, the `timeout` and `nTrials` end it. To end it from
outside, pass a `CancellationToken` as `ct`:

- Once the token is cancelled, no new trial starts. The trials in flight finish and are stored, then `Optimize`
  throws `OperationCanceledException` carrying the token. An already cancelled token runs no trial.
- Every trial sees the token as `trial.CancellationToken`, so a long objective can stop early. A trial whose
  objective throws for the token is stored as `FAIL`, whatever `catch` holds, as Python stores the trial its
  `KeyboardInterrupt` lands in. An objective that ignores the token runs to its end.

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

Without a handler, .NET ends the process at Ctrl+C or SIGTERM, and the trials in flight stay `RUNNING` in a
persistent storage. A journal never fails them.

- `StopSignals` is opt-in, because a signal handler is process-wide. The first SIGINT (Ctrl+C) or SIGTERM
  (`docker stop`, a Kubernetes pod's termination) cancels its token, and a second one ends the process at once.
  It costs no CPU until a signal arrives.
- A container's grace period, 10 s for Docker and 30 s for Kubernetes by default, must cover the trials in flight.
- In a .NET Generic Host, pass `IHostApplicationLifetime.ApplicationStopping` as `ct` instead: the host already
  handles both signals.
- On Windows, a process its parent started with Ctrl+C disabled never receives it, with or without `StopSignals`.
  Some launchers do that.

## Async work and your own threads

The objective is synchronous, as Optuna's is, and so are `Ask` and `Tell`. Work that is asynchronous fits in two
ways.

**Block on it inside the objective.** On the default executor each trial runs on a dedicated thread, so blocking
costs that thread, never the thread pool. Pass the trial's token, so a cancelled run cancels the work too:

```csharp
study.Optimize(trial =>
{
    double lr = trial.SuggestFloat("lr", 1e-5, 1e-1, log: true);
    return TrainRemotelyAsync(lr, trial.CancellationToken).GetAwaiter().GetResult();
}, nTrials: 100, nJobs: 16);
```

**Ask and tell from your own async code** when holding a thread per trial is too much, such as hundreds of trials
waiting at once:

```csharp
using var slots = new SemaphoreSlim(64);   // trials in flight
var running = new List<Task>();
for (int i = 0; i < 1000; i++)
{
    await slots.WaitAsync();
    Trial trial = study.Ask();
    double lr = trial.SuggestFloat("lr", 1e-5, 1e-1, log: true);
    running.Add(EvaluateAsync(trial, lr));
}
await Task.WhenAll(running);

async Task EvaluateAsync(Trial trial, double lr)
{
    try
    {
        study.Tell(trial, await TrainRemotelyAsync(lr, CancellationToken.None));
    }
    catch (Exception)
    {
        study.Tell(trial, state: TrialState.Fail);
    }
    finally
    {
        slots.Release();
    }
}
```

`Ask`, the suggestions and `Tell` stay quick synchronous calls, made on whatever thread your code runs on, the
thread pool included. With a storage whose client completes its calls on the thread pool (the gRPC proxy, and
Redis on .NET 8), many of them blocking pool threads at once can stall that client, so keep the number in flight
moderate or call them from threads of your own: the
[Storage page](../storage.md#concurrency-workers-studies-and-languages-on-one-journal) explains why. The
[ask-and-tell recipe](../recipes/ask-and-tell.md) has more patterns.

## Multi-process optimization

Point several processes at one storage. Every worker creates the study with `loadIfExists: true`, so the first
creates it and the others join, and each process's sampler learns from the trials all of them store.
`InMemoryStorage` lives inside one process, so use a `JournalStorage` (a file) or an `RDBStorage`:

```csharp
using OptunaSharp;
using OptunaSharp.Storages.Journal;

var storage = new JournalStorage(new JournalFileBackend("./journal.log"));
var study = Optuna.CreateStudy(
    studyName: "journal_multiprocess", storage: storage, loadIfExists: true);
study.Optimize(Objective, nTrials: 3);
```

.NET has no `multiprocessing.Pool`: start the program several times. Build it once and start the built
executable, so the processes do not rebuild over one another:

```bash
dotnet build -c Release
for i in 1 2 3 4; do ./bin/Release/net10.0/MyStudy & done; wait
```

```powershell
dotnet build -c Release
1..4 | ForEach-Object { Start-Process .\bin\Release\net10.0\MyStudy.exe -NoNewWindow -PassThru } | Wait-Process
```

The four processes above run 12 trials between them, numbered 0 to 11. Shell scripts can also take part through
the [`optunasharp` CLI](../cli.md) (`optunasharp ask`, `optunasharp tell`).

On one machine the journal file's lock keeps the writers safe. It is the lock Python Optuna uses, so C# and Python
workers can share a journal. Over NFS and other network file systems that lock can fail, so across machines use a
database or Redis.

## Multi-node optimization

Run the same program on every machine against one storage, with `loadIfExists: true`: the first machine creates
the study and the others attach. For PostgreSQL or MySQL, use the
[EF Core storage](../../integrations/storage/ef-core.md):

```csharp
using OptunaSharp;
using OptunaSharp.Storages.EFCore;

var study = Optuna.CreateStudy(
    studyName: "distributed_test",
    storage: new RDBStorage("mysql://username:password@127.0.0.1:3306/example"),
    loadIfExists: true);
study.Optimize(Objective, nTrials: 100);
```

A [journal on Redis](../../integrations/storage/redis/index.md) works the same way, without a database.

### A gRPC proxy for thousands of workers

With thousands of workers, one database server can become the bottleneck. A
[gRPC storage proxy](../../integrations/storage/grpc/index.md) puts a server in front of it, and the workers talk to
the proxy. It speaks Python Optuna's protocol, so Python workers can use an OptunaSharp proxy and C# workers a
Python one.

Start the server first:

```csharp
using OptunaSharp.Storages;
using OptunaSharp.Storages.EFCore;
using OptunaSharp.Storages.Grpc;

using var backend = new RDBStorage("mysql://username:password@127.0.0.1:3306/example");
GrpcProxyServer.RunGrpcProxyServer(
    StorageFunctions.GetStorage(backend), host: "0.0.0.0", port: 13000);   // every network interface
```

Then, on every machine:

```csharp
using OptunaSharp;
using OptunaSharp.Storages.Grpc;

using var storage = new GrpcStorageProxy("proxy-host", 13000);
var study = Optuna.CreateStudy(
    studyName: "grpc_proxy_multinode", storage: storage, loadIfExists: true);
study.Optimize(Objective, nTrials: 50);
```

The proxy uses cleartext HTTP/2 by default; the [server page](../../integrations/storage/grpc/server.md) shows how
to add TLS. Each client refreshes its trial cache by fetching every trial that is new or not finished yet, as
Optuna's client does, so a call costs more the more trials run at once.

## Relationship to Optuna

| Optuna (Python) | OptunaSharp |
|-----------------|-------------|
| `study.optimize(objective, n_trials=20, n_jobs=4)` | `study.Optimize(Objective, nTrials: 20, nJobs: 4)` |
| `n_jobs=-1`: `os.cpu_count()` threads | `nJobs: -1`: `Environment.ProcessorCount` threads |
| The threads of a `ThreadPoolExecutor`; standard CPython runs one at a time | Dedicated threads that run in parallel; the thread pool, a sampler per worker and the OpenBLAS thread count through `TrialExecutorOptions` |
| Ctrl+C raises `KeyboardInterrupt` out of `study.optimize` | `study.Optimize(…, ct: signals.Token)` throws `OperationCanceledException`, with `using var signals = StopSignals.Listen()` |
| A `multiprocessing.Pool` of workers | The program started several times |
| `JournalStorage(JournalFileBackend("./journal.log"))` | `new JournalStorage(new JournalFileBackend("./journal.log"))` |
| `create_study(storage="mysql://…", load_if_exists=True)` | `Optuna.CreateStudy(storage: new RDBStorage("mysql://…"), loadIfExists: true)` (EF Core package) |
| `run_grpc_proxy_server(storage, host=…, port=…)` | `GrpcProxyServer.RunGrpcProxyServer(storage, host: …, port: …)` |
| `GrpcStorageProxy(host=…, port=…)` | `new GrpcStorageProxy(host: …, port: …)` |

## See also

[Storage](../storage.md) · [Journal-file storage](../recipes/journal-storage.md) ·
[Ask-and-tell interface](../recipes/ask-and-tell.md) · [gRPC storage proxy](../../integrations/storage/grpc/index.md) ·
[EF Core storage](../../integrations/storage/ef-core.md) ·
[THREADING.md](https://github.com/SciSharp/OptunaSharp/blob/main/docs/THREADING.md) · the
[API reference](../../api/index.md).
