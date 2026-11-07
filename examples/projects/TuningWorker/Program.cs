// TuningWorker — a Generic Host worker that joins a TuningService study through Optuna's gRPC storage proxy and
// optimizes it. Start the TuningService, then run this in several terminals: every process works on the same
// study, and the service's dashboard (http://localhost:5080) shows each trial's worker. python_worker.py, next
// to this file, is the same worker in Python Optuna and can join too.
//
// The dependency-injection side is the client mirror of the service's:
//   storage    AddOptunaGrpcStorageProxyClient(host, port)   BaseStorage is the remote study storage
//   sampler    AddOptunaSampler(factory)                     unseeded TPE, so workers explore different points
//   pruner     AddOptunaPruner(factory)                      the same median pruner the service uses
//   studies    AddOptunaStudyFactory(defaults)               CreateStudy() attaches to the shared study
//
// source (distributed):  https://optuna.readthedocs.io/en/stable/tutorial/10_key_features/004_distributed.html
// source (gRPC client):  https://optuna.readthedocs.io/en/stable/reference/generated/optuna.storages.GrpcStorageProxy.html
// source (workers):      https://learn.microsoft.com/en-us/dotnet/core/extensions/workers
//
//   Run:  dotnet run --project examples/projects/TuningWorker
//         dotnet run --project examples/projects/TuningWorker -- --Worker:Trials=50 --Worker:Jobs=4

using OptunaSharp.Extensions;
using OptunaSharp.Pruners;
using OptunaSharp.Samplers;
using OptunaSharp.Storages.Grpc.Extensions;
using TuningCommon;
using TuningWorker;

HostApplicationBuilder builder = Host.CreateApplicationBuilder(args);

// ---- settings: the "Worker" section of appsettings.json, validated before the worker starts
builder.Services.AddOptions<WorkerOptions>()
    .Bind(builder.Configuration.GetSection(WorkerOptions.SectionName))
    .ValidateDataAnnotations()
    .ValidateOnStart();

// the proxy address and the study defaults are fixed at registration, so they are read now
WorkerOptions worker = builder.Configuration.GetSection(WorkerOptions.SectionName).Get<WorkerOptions>()
                       ?? new WorkerOptions();

// ---- OptunaSharp, registered once
builder.Services
    // first: hosted services start in registration order, and the worker's trials should log through it
    .AddOptunaLoggingBridge()
    // BaseStorage is a gRPC client of the service's storage proxy; it keeps Optuna's client-side trial cache,
    // and the container closes its channel at shutdown
    .AddOptunaGrpcStorageProxyClient(worker.Host, worker.Port)
    // unseeded on purpose: workers that share a seed would all draw the same random startup trials
    .AddOptunaSampler(_ => new TPESampler())
    .AddOptunaPruner(_ => new MedianPruner(nStartupTrials: 5, nWarmupSteps: 3))
    // "many workers, one study": every CreateStudy() without arguments attaches to the shared study, and the
    // first worker to arrive creates it
    .AddOptunaStudyFactory(study =>
    {
        study.DefaultStudyName = worker.StudyName;
        study.LoadIfExists = true;
    });

builder.Services.AddHostedService<OptimizationWorker>();

builder.Build().Run();
