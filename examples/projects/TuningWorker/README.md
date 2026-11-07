# TuningWorker

A Generic Host worker that joins a [TuningService](../TuningService/README.md) study through Optuna's gRPC
storage proxy and optimizes it, then exits. Run several at once and they share one study: each process runs
its own sampler over every worker's trials, so they cooperate without knowing about each other.

```bash
dotnet run --project examples/projects/TuningService           # first, in its own terminal

dotnet run --project examples/projects/TuningWorker             # then any number of these
dotnet run --project examples/projects/TuningWorker -- --Worker:Trials=50 --Worker:Jobs=4
```

Each trial records its worker (`worker:<machine>:<pid>`) in the `worker` user attribute, so the service's
dashboard at <http://localhost:5080> shows who ran what. The study is `distributed` unless
`--Worker:StudyName` says otherwise. The service's own background jobs can work on it too
(`POST /api/studies/distributed/optimize`).

## Dependency injection

[`Program.cs`](Program.cs) is the client-side mirror of the service's registrations:

```csharp
builder.Services
    .AddOptunaLoggingBridge()                                   // OptunaSharp's logs to ILogger
    .AddOptunaGrpcStorageProxyClient(worker.Host, worker.Port)  // BaseStorage = the remote study storage
    .AddOptunaSampler(_ => new TPESampler())                    // unseeded, so workers explore different points
    .AddOptunaPruner(_ => new MedianPruner(nStartupTrials: 5, nWarmupSteps: 3))
    .AddOptunaStudyFactory(study =>
    {
        study.DefaultStudyName = worker.StudyName;              // CreateStudy() with no arguments
        study.LoadIfExists = true;                              // attaches to the shared study, or creates it
    });
```

[`OptimizationWorker`](OptimizationWorker.cs) waits for the proxy (`WaitServerReady`), attaches with
`CreateStudy()`, runs `Optimize(nTrials, nJobs)` on a dedicated thread, logs the study's best trial (often
another worker's), and stops the host.

## Settings

[`appsettings.json`](appsettings.json), section `Worker`; override any of them with `--Worker:Name=value`.

| Setting | Default | Meaning |
|---|---|---|
| `Host`, `Port` | `localhost`, `13000` | The TuningService's gRPC storage proxy. |
| `StudyName` | `distributed` | The study to join. |
| `Trials` | `30` | Trials this process runs. |
| `Jobs` | `2` | Of those, how many run at once (`nJobs`). |
| `EpochDelayMilliseconds` | `25` | Time per simulated epoch. |
| `ConnectTimeoutSeconds` | `15` | How long to wait for the proxy. |

## Stopping and exit codes

The Generic Host turns Ctrl+C into a graceful stop. A callback then calls `Study.Stop()`, and the trials in
flight finish and are stored before the process exits. The exit code is 0 after a finished batch and 1 when the
worker fails. For example, with no service running:

```
fail: TuningWorker.OptimizationWorker[0]
      No storage proxy answered at localhost:13001 within 2 s. Start the TuningService first.
```

## The Python worker

[`python_worker.py`](python_worker.py) is the same worker in plain Python Optuna 5. The service speaks
Optuna's own gRPC protocol, so nothing OptunaSharp-specific runs on the Python side:

```bash
pip install "optuna>=5" grpcio protobuf
python examples/projects/TuningWorker/python_worker.py --trials 20
```

Its objective is a line-for-line copy of [`SimulatedTraining`](../TuningCommon/SimulatedTraining.cs). Keep the
two identical, or the C# and Python workers optimize different problems under one study name.

## What was verified

A service job, two TuningWorker processes (`Jobs` 2 and 3) and `python_worker.py` ran 15 trials each on one
study at the same time. Result: 60 trials, numbered 0–59 without gaps, 39 complete and 21 pruned, none left
running, and every process exited 0. Each worker's closing line named the same best trial, found by one of the
other workers.
