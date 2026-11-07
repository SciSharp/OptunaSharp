# TuningService

An ASP.NET Core app that hosts OptunaSharp studies. Every OptunaSharp piece (storage, sampler, pruner, study
factory, chart renderer) is registered once in the DI container, in [`Program.cs`](Program.cs), and injected
where it is needed. On top of that the app serves:

- Optuna's **ask-and-tell interface over HTTP**, so a program in any language can let the service pick its
  parameters, report progress, and be pruned by the service's pruner;
- **background optimization jobs**, a hosted service that runs the built-in objective with `nJobs` parallelism;
- **Plotly charts** of any study, as HTML pages or as figure JSON, and a self-refreshing **dashboard**;
- **Optuna's gRPC storage proxy**, so [TuningWorker](../TuningWorker/README.md) processes and Python Optuna
  workers can join the same studies;
- **health checks** and **RFC 7807 problem details** that carry Optuna's own error messages.

```bash
dotnet run --project examples/projects/TuningService
# open http://localhost:5080
```

## Endpoints

| Method and path | What it does |
|---|---|
| `GET /` | Dashboard: every study and job, with links to the charts. It reloads every 5 s. |
| `GET /api/studies` | Every study, with its best value. |
| `POST /api/studies` | Creates a study: `{"name": "demo", "direction": "Minimize"}` or `"directions": [...]`. |
| `GET /api/studies/{name}` | One study: trial counts by state, best trial, Pareto front. |
| `DELETE /api/studies/{name}` | Deletes a study. |
| `GET /api/studies/{name}/importance?evaluator=` | Parameter importance: `ped-anova` (Optuna's default), `fanova`, `mdi` or `shap`. |
| `GET /api/studies/{name}/trials?state=` | The trials, optionally of one state. |
| `POST /api/studies/{name}/trials` | **Ask**: samples a search space given in Optuna's distribution JSON. |
| `GET /api/studies/{name}/trials/{number}` | One trial. |
| `POST /api/studies/{name}/trials/{number}/report` | **Report** an intermediate value; answers `shouldPrune`. |
| `POST /api/studies/{name}/trials/{number}/tell` | **Tell** the result: `{"values": [0.42]}`, or `{"state": "Pruned"}`. |
| `POST /api/studies/{name}/optimize` | Queues a background job: `{"nTrials": 40, "nJobs": 4}`. Creates the study on first use. |
| `GET /api/jobs`, `GET /api/jobs/{id}`, `DELETE /api/jobs/{id}` | Lists, reads and cancels jobs. |
| `GET /studies/{name}/plots/{kind}` | A chart as an interactive page (plotly.js from Plotly's CDN). |
| `GET /api/studies/{name}/plots/{kind}` | The same chart as Plotly figure JSON. |
| `GET /health` | Lists the studies through the storage connection. |
| gRPC on port 13000 | Optuna's storage proxy (`optuna.storages.GrpcStorageProxy(host, port)`). |

Chart kinds: `history`, `intermediate`, `slice`, `parallel-coordinate`, `contour`, `importance`, `edf`, `rank`
and `timeline` for one objective; `pareto-front` and `timeline` for several. Every request is ready to send in
[`TuningService.http`](TuningService.http).

## Settings

[`appsettings.json`](appsettings.json), section `Optuna`, validated at startup. Override any setting with an
environment variable (`Optuna__Sampler=GP`) or an argument (`-- --Optuna:Sampler=GP`).

| Setting | Default | Meaning |
|---|---|---|
| `Storage` | `sqlite:///tuning.db` | The Optuna storage URL. A `postgresql://` or `mysql://` URL works once the project also references `OptunaSharp.Storage.EFCore`; no code changes. |
| `Sampler` | `TPE` | `TPE`, `Random`, `QMC`, `GP`, `CmaEs` or `NSGAII`, for every study. |
| `SamplerSeed` | `null` | See the first design note below before setting it. |
| `EpochDelayMilliseconds` | `25` | How long one simulated epoch of the built-in objective takes. |
| `MaxTrialsPerJob`, `MaxParallelTrials`, `MaxQueuedJobs` | `1000`, `16`, `32` | Limits on what a client may ask of a background job. |

Kestrel listens on `http://localhost:5080` (HTTP/1.1) and `http://localhost:13000` (cleartext HTTP/2 for gRPC);
both are in `appsettings.json` under `Kestrel:Endpoints`.

## Design notes

- **One sampler per study, one per request.** `AddOptunaSampler(factory)` and `AddOptunaPruner(factory)`
  default to transient. Samplers keep per-study state, so sharing one instance across studies mixes that
  state. The ask, report and tell endpoints take a `BaseSampler` and a `BasePruner` parameter, so every request
  gets fresh ones. Samplers rebuild their model from the stored trials on every call, so nothing is lost. The
  one trap is a fixed seed: every fresh sampler then draws the same random startup points, so `SamplerSeed`
  stays `null` unless you are reproducing one background job.
- **A scope per job.** [`OptimizationJobRunner`](Jobs/OptimizationJobRunner.cs) resolves
  `IOptunaStudyFactory` from a scope it creates for each job and disposes it only when the job's `Optimize`
  returns, so everything the study resolved lives as long as the study.
- **Stopping a job.** OptunaSharp has no `CancellationToken` overload, and `Study.Stop()` only works inside
  an objective or a callback. So a callback checks the job's token, linked to the host's stopping token, and
  calls `Stop()`: the trials in flight finish, then the job ends. Cancelling a running 500-trial job here
  ends it after its in-flight trials, with no trial left `RUNNING`.
- **No thread-pool threads held.** `Optimize` is synchronous, as in Optuna. A job runs it on a
  `LongRunning` thread, and with `nJobs > 1` the trials run on OptunaSharp's own worker threads, so the thread
  pool stays free for HTTP requests.
- **Optuna's errors become HTTP errors.** OptunaSharp raises Python's exception kinds with Optuna's messages.
  [`OptunaExceptionHandler`](Infrastructure/OptunaExceptionHandler.cs) maps `KeyError` to 404,
  `DuplicatedStudyError` and `UpdateFinishedTrialError` to 409, and `ValueError` to 400, the same mapping
  Optuna's gRPC server uses. For example, telling a trial twice answers
  `400 {"detail": "Cannot tell a COMPLETE trial."}`.
- **Logs.** [`OptunaLoggingBridge`](../TuningCommon/OptunaLoggingBridge.cs) replaces OptunaSharp's stderr
  handler, so `Trial 3 finished with value …` arrives in ASP.NET Core's logging under the category
  `optuna.study.study`. `Logging:LogLevel:optuna` filters all of it.
- **Study names are free text.** A name may contain `/`, which a URL escapes as `%2F`. ASP.NET Core leaves
  `%2F` encoded in route values, so [`StudyRoute`](Endpoints/StudyRoute.cs) restores it. The dashboard
  HTML-encodes every name, and shows only the storage URL's scheme, because a database URL can contain a
  password.
- **SQLite.** One `RDBStorage` serves the HTTP requests, the jobs and the gRPC calls through a single
  connection, the arrangement Optuna recommends: workers reach the database through the proxy instead of
  opening the file themselves. Python Optuna and optuna-dashboard can open `tuning.db` too. It is created in
  this folder, because `dotnet run` starts web apps in the project directory.

## Security

The service has no authentication, and the gRPC proxy speaks Optuna's protocol, which has none either, over
cleartext HTTP/2. Both listen on `localhost` only. Before you expose them, add authentication
(`.RequireAuthorization()` on the endpoints and on `MapOptunaStorageProxy()`) and TLS.

## What was verified

On Windows 11, .NET 10.0.1, with Python 3.12 and Optuna 5.0.0:

- every endpoint above, including the error cases, with curl;
- a background job of 30 trials at `nJobs: 4`: 21 complete and 9 pruned, in 8 s;
- one study shared at the same time by a service job, two TuningWorker processes and `python_worker.py`, 15
  trials each: 60 trials numbered 0–59 without gaps, none left running;
- all nine single-objective charts, as HTML and as figure JSON.

There is no parity test against Python: the objective is a surrogate, and the features it exercises
(sampling, pruning, storage, the gRPC protocol, the charts) have their own live-Optuna parity tests in
`test/OptunaSharp.Tests.Interop`.
