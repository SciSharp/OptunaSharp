// TuningService — an ASP.NET Core app that hosts OptunaSharp studies and shows how OptunaSharp plugs into
// dependency injection.
//
// Every OptunaSharp piece is registered once, below, and injected wherever it is needed:
//   storage    AddOptunaStorage(url)            the Optuna storage URL from appsettings.json (SQLite by default)
//   sampler    AddOptunaSampler(factory)        transient: every study gets its own, of the configured kind
//   pruner     AddOptunaPruner(factory)         transient: every study gets its own
//   studies    AddOptunaStudyFactory()          IOptunaStudyFactory composes the three into studies
//   charts     AddOptunaPlotlyVisualization()   IOptunaPlotlyVisualizer, with the app's chart style
//   logging    AddOptunaLoggingBridge()         OptunaSharp's log records go to ILogger (TuningCommon)
// Around them the app adds what a service needs: Optuna's ask-and-tell interface over HTTP, background
// optimization jobs (a hosted service), Plotly charts, a dashboard, health checks, problem details, and
// Optuna's gRPC storage proxy, through which TuningWorker processes and Python Optuna join the same studies.
//
// source (ask-and-tell):  https://optuna.readthedocs.io/en/stable/tutorial/20_recipes/009_ask_and_tell.html
// source (distributed):   https://optuna.readthedocs.io/en/stable/tutorial/10_key_features/004_distributed.html
// source (gRPC proxy):    https://optuna.readthedocs.io/en/stable/reference/generated/optuna.storages.run_grpc_proxy_server.html
// source (hosted jobs):   https://learn.microsoft.com/en-us/aspnet/core/fundamentals/host/hosted-services
//
// The built-in objective is a deterministic surrogate of a training run (TuningCommon/SimulatedTraining.cs),
// so the service runs anywhere; the point is the hosting, not the model.
//
//   Run:  dotnet run --project examples/projects/TuningService
//         then open http://localhost:5080

using System.Text.Json.Serialization;
using Microsoft.Extensions.Options;
using OptunaSharp.Extensions;
using OptunaSharp.Pruners;
using OptunaSharp.Storages.Grpc;
using OptunaSharp.Visualization.Plotly;
using OptunaSharp.Visualization.Plotly.Extensions;
using Plotly.NET;
using TuningCommon;
using TuningService;
using TuningService.Endpoints;
using TuningService.Infrastructure;
using TuningService.Jobs;

WebApplicationBuilder builder = WebApplication.CreateBuilder(args);

// ---- settings: the "Optuna" section of appsettings.json, validated before the app accepts a request
builder.Services.AddOptions<TuningServiceOptions>()
    .Bind(builder.Configuration.GetSection(TuningServiceOptions.SectionName))
    .ValidateDataAnnotations()
    .ValidateOnStart();

// the storage URL decides which types get registered, so it is read here, at registration time, rather than
// through IOptions<T> later
string storageUrl = builder.Configuration.GetSection(TuningServiceOptions.SectionName)
                        .Get<TuningServiceOptions>()?.Storage
                    ?? new TuningServiceOptions().Storage;

// ---- OptunaSharp, registered once for the whole app
builder.Services
    // first: hosted services start in registration order, and the job runner's trials should log through it
    .AddOptunaLoggingBridge()
    // one storage for every study, request, job and gRPC call. Core reads the URL as Optuna does and loads
    // the package that serves its scheme (OptunaSharp.Storage.Sqlite for sqlite:///); the container owns the
    // connection and closes it at shutdown
    .AddOptunaStorage(storageUrl)
    // samplers and pruners keep per-study state, so these factories are transient: every study, and every
    // ask/report/tell request, gets fresh instances
    .AddOptunaSampler(services => SamplerSelection.Create(
        services.GetRequiredService<IOptions<TuningServiceOptions>>().Value))
    .AddOptunaPruner(_ => new MedianPruner(nStartupTrials: 5, nWarmupSteps: 3))
    // IOptunaStudyFactory.CreateStudy(name) composes the three, like optuna.create_study(storage=…, sampler=…, pruner=…)
    .AddOptunaStudyFactory()
    // IOptunaPlotlyVisualizer; every chart it draws gets this style
    .AddOptunaPlotlyVisualization(ChartStyle());

// ---- Optuna's gRPC storage proxy (Kestrel's "Grpc" endpoint in appsettings.json). MapOptunaStorageProxy
// below serves the container's BaseStorage, the one registered above. Unlimited message sizes, as on
// Optuna's own server, so that reading a large study is never cut off
builder.Services.AddGrpc(grpc =>
{
    grpc.MaxReceiveMessageSize = null;
    grpc.MaxSendMessageSize = null;
});

// ---- background optimization jobs: the endpoints queue, the hosted service runs
builder.Services.AddSingleton<OptimizationJobQueue>();
builder.Services.AddHostedService<OptimizationJobRunner>();

// ---- HTTP plumbing
builder.Services.AddProblemDetails();
builder.Services.AddExceptionHandler<OptunaExceptionHandler>();
builder.Services.AddHealthChecks().AddCheck<StorageHealthCheck>("optuna-storage");
builder.Services.ConfigureHttpJsonOptions(json =>
{
    // objective values may be inf or NaN, which Optuna stores; JSON has no literal for either
    json.SerializerOptions.NumberHandling = JsonNumberHandling.AllowNamedFloatingPointLiterals;
    // "Minimize" and "Pruned" in requests and responses, rather than enum numbers
    json.SerializerOptions.Converters.Add(new JsonStringEnumConverter());
});

WebApplication app = builder.Build();

// OptunaSharp's KeyError / ValueError / DuplicatedStudyError become 404 / 400 / 409 problem responses
app.UseExceptionHandler();

app.MapDashboard();
app.MapStudyEndpoints();
app.MapTrialEndpoints();
app.MapJobEndpoints();
app.MapPlotEndpoints();
app.MapHealthChecks("/health");
// before exposing the service beyond your machine, add .RequireAuthorization() here and on the API: the proxy
// speaks Optuna's protocol, which has no authentication, over cleartext HTTP/2
app.MapOptunaStorageProxy();

app.Run();

// the app's chart style, applied by IOptunaPlotlyVisualizer to every chart
static PlotlyChartCustomization ChartStyle()
{
    var config = new Config();
    config.SetValue("displaylogo", false);   // no Plotly logo in the chart's tool bar
    config.SetValue("responsive", true);     // the chart follows the size of the browser window
    return new PlotlyChartCustomization { Config = config };
}
