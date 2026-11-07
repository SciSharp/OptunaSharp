# OptunaSharp example projects

The [single-file examples](../README.md) one level up show one feature each. These are full .NET projects
that show OptunaSharp inside real application hosts: an ASP.NET Core service built on dependency injection, a
Generic Host worker, and a console app that tunes a real ML.NET model.

| Project | Host | What it shows |
|---|---|---|
| [`TuningService`](TuningService/README.md) | ASP.NET Core | OptunaSharp registered in the DI container and injected everywhere. It serves Optuna's ask-and-tell interface over HTTP, background optimization jobs, Plotly charts, a dashboard, health checks, problem details, and Optuna's gRPC storage proxy. |
| [`TuningWorker`](TuningWorker/README.md) | Generic Host (Worker Service) | Distributed optimization: several processes join one TuningService study through the gRPC proxy, with DI on the client side. [`python_worker.py`](TuningWorker/python_worker.py) joins the same study from Python Optuna. |
| [`MLNetModelTuning`](MLNetModelTuning/README.md) | Console | Real in-process training: `OptunaSearchCV` tunes an ML.NET FastTree model with cross-validation, saved in SQLite so runs resume, then refit, held-out score, fANOVA importance and Plotly charts. |
| [`TuningCommon`](TuningCommon) | Class library | Shared by the service and the worker: the objective they evaluate, and a bridge that sends OptunaSharp's log records to `ILogger`. |

## How the tuning projects fit together

```
             browser · curl · any HTTP client
                         │  REST API, dashboard and charts (http://localhost:5080)
                         ▼
 ┌──────────────────── TuningService (ASP.NET Core) ────────────────────┐
 │  /api/studies · ask/report/tell · background jobs · charts · /health │
 │                                                                      │
 │   one BaseStorage singleton ──► SQLite tuning.db (Optuna's schema)   │
 │                ▲                                                     │
 │   gRPC storage proxy, Optuna's protocol (localhost:13000)            │
 └────────────────┼─────────────────────────────────────────────────────┘
                  │ gRPC
    ┌─────────────┼───────────────┬───────────────────────┐
    │             │               │                       │
 TuningWorker  TuningWorker   python_worker.py      any optuna client
 (C#, nJobs)   (C#, nJobs)    (Python Optuna 5)     (optuna.load_study)
```

Every worker runs its own sampler and reads the other workers' trials from the shared storage before each
suggestion, so the processes cooperate without knowing about each other. The service's dashboard shows the
`worker` user attribute of each trial: which process ran it.

## Run them

The projects need the .NET 10 SDK and nothing else. The Python worker also needs Python with
`pip install "optuna>=5" grpcio protobuf`.

```bash
# terminal 1: the service (open http://localhost:5080)
dotnet run --project examples/projects/TuningService

# terminals 2 and 3: two workers on one study
dotnet run --project examples/projects/TuningWorker
dotnet run --project examples/projects/TuningWorker -- --Worker:Trials=50 --Worker:Jobs=4

# terminal 4: a Python Optuna worker on the same study
python examples/projects/TuningWorker/python_worker.py --trials 20

# a background job on the service, with curl
curl -X POST http://localhost:5080/api/studies/distributed/optimize -H "Content-Type: application/json" -d "{\"nTrials\": 40, \"nJobs\": 4}"

# separately: tune an ML.NET model (about 20 s for 20 trials)
dotnet run --project examples/projects/MLNetModelTuning
```

[`TuningService/TuningService.http`](TuningService/TuningService.http) has every API request ready to send
from Visual Studio, Rider or VS Code.

## Dependency injection at a glance

| Registration | Package | Registers | Lifetime |
|---|---|---|---|
| `AddOptunaStorage(url)` | Core | the storage the URL names, e.g. `sqlite:///tuning.db` (`BaseStorage` + concrete type) | singleton: one connection for the whole app |
| `AddOptunaGrpcStorageProxyClient(host, port)` | Storage.Grpc | a client of a gRPC storage proxy (`BaseStorage` + `GrpcStorageProxy`) | singleton |
| `AddOptunaSampler(factory)` | Core | `BaseSampler` | transient: every study gets its own |
| `AddOptunaPruner(factory)` | Core | `BasePruner` | transient: every study gets its own |
| `AddOptunaStudyFactory(defaults)` | Core | `IOptunaStudyFactory`, which composes the three into studies | transient |
| `AddOptunaPlotlyVisualization(style)` | Visualization.Plotly | `IOptunaPlotlyVisualizer`, with an app-wide chart style | singleton |
| `AddGrpc()` + `MapOptunaStorageProxy()` | Storage.Grpc | Optuna's gRPC storage proxy, serving the container's `BaseStorage` | per call (gRPC) |
| `AddOptunaLoggingBridge()` | TuningCommon (here) | OptunaSharp's log records to `ILogger` | hosted service |

Samplers and pruners are transient because they keep per-study state (random streams, generation trackers,
caches). Value-configured objects such as importance evaluators are created with `new` where they are used,
which is the rule [`docs/DI_PLAN.md`](../../docs/DI_PLAN.md) gives for what belongs in the container.

## Outside this repository

Each project references OptunaSharp's source through `ProjectReference`, so it always builds against the
current code. In your own solution, replace each one with the package of the same name, for example:

```xml
<PackageReference Include="OptunaSharp" Version="x.y.z" />
<PackageReference Include="OptunaSharp.Storage.Sqlite" Version="x.y.z" />
<PackageReference Include="OptunaSharp.Storage.Grpc" Version="x.y.z" />
<PackageReference Include="OptunaSharp.Visualization.Plotly" Version="x.y.z" />
```

CI builds all four projects ([`examples.yml`](../../.github/workflows/examples.yml)) but does not run them.
