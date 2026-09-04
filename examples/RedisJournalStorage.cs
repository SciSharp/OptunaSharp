#:project ../src/OptunaSharp.Storage.StackExchange.Redis/OptunaSharp.Storage.StackExchange.Redis.csproj
#:property PublishAot=false
#:property SkipNDScopeWeave=true
#:property GeneratePackageOnBuild=false

// Redis journal storage — persist a study's operation log to a Redis server so it survives process
// restarts and can be shared by many workers at once. wrap a StackExchangeJournalRedisBackend in a
// JournalStorage, open the study with loadIfExists, and every run RESUMES the same study: run this
// twice and watch the trial count grow.
//
// source (feature):     https://optuna.readthedocs.io/en/stable/reference/generated/optuna.storages.journal.JournalRedisBackend.html
// source (distributed): https://optuna.readthedocs.io/en/stable/tutorial/10_key_features/004_distributed.html
//
// the objective is a real, self-contained computation (a 2-D paraboloid) — the point here is the
// STORAGE, not the objective, so no surrogate is needed. constructing the backend prints an
// ExperimentalWarning: JournalRedisBackend is @experimental_class("3.1.0") in Optuna, and OptunaSharp
// mirrors that. needs a reachable Redis; if none is found the example prints how to start one and exits.
//
//   Run:  docker run --rm -p 6379:6379 redis:7-alpine     # in another terminal
//         dotnet run examples/RedisJournalStorage.cs       # run again to resume
//
//   endpoint: $OPTUNASHARP_REDIS_URL (default redis://localhost:6379)

using System;
using System.Net.Sockets;
using OptunaSharp;
using OptunaSharp.Samplers;
using OptunaSharp.Storages;
using OptunaSharp.Storages.Journal;   // StackExchangeJournalRedisBackend, RedisUrl, RedisEndpoint
using StackExchange.Redis;            // ConnectionMultiplexer, ConfigurationOptions (reachability probe)

string url = Environment.GetEnvironmentVariable("OPTUNASHARP_REDIS_URL") is { Length: > 0 } u
    ? u : "redis://localhost:6379";

// a Redis example needs a Redis — probe first so we can print friendly guidance, not a stack trace.
if (!RedisReachable(url))
{
    PrintNoRedisHelp(url);
    return;
}

// the journal backend on StackExchange.Redis (the de-facto-standard, MIT client). the sibling
// NRedisStack / ServiceStack backends write byte-identical data — see examples/RedisNativeClientConfig.cs.
var backend = new StackExchangeJournalRedisBackend(url);   // redis://... — Optuna's own constructor
var storage = new JournalStorage(backend);                 // the JournalStorage seam wraps any journal backend

// loadIfExists: true → open the existing study if it is already in Redis, else create it. this is what
// makes the run resumable (and safe to launch from many workers against the same server + study name).
var study = Optuna.CreateStudy(
    storage: storage,
    sampler: new RandomSampler(seed: 0),           // seeded, so each run's new trials reproduce
    direction: StudyDirection.Minimize,            // lower is better
    studyName: "redis-resumable-demo",             // stable name + stable keys → the same study every run
    loadIfExists: true);

int before = study.Trials.Count;                   // trials already persisted in Redis from earlier runs
Console.WriteLine(before == 0
    ? "Fresh study — nothing in Redis yet."
    : $"Resumed — {before} trial(s) already persisted in Redis from earlier runs.");

study.Optimize(Objective, nTrials: 15);            // append 15 more trials to the shared journal

Console.WriteLine($"Now {study.Trials.Count} trial(s) total (+{study.Trials.Count - before} this run).");
Console.WriteLine($"Best value: {study.BestValue:F6}");
foreach (var kv in study.BestParams)
    Console.WriteLine($"    {kv.Key}: {kv.Value}");

// prove it is really in Redis, not just in memory: open a SECOND storage over the same server + study
// name (as another worker would) and read the trial count straight back.
var reader = Optuna.CreateStudy(
    storage: new JournalStorage(new StackExchangeJournalRedisBackend(url)),
    studyName: "redis-resumable-demo",
    loadIfExists: true);
Console.WriteLine($"A second client reads back {reader.Trials.Count} trial(s) from the same Redis.");
Console.WriteLine("Run this example again — the total grows, because the journal persists in Redis.");

// a real, deterministic objective: minimize a 2-D paraboloid centered at (2, -3).
static double Objective(Trial t)
{
    double x = t.SuggestFloat("x", -10, 10);
    double y = t.SuggestFloat("y", -10, 10);
    return Math.Pow(x - 2, 2.0) + Math.Pow(y + 3, 2.0);   // Math.Pow, not x*x, for Python x**2 parity
}

// --- reachability probe (public APIs only) --------------------------------------------------------
// parse the redis:// URL with the shared public parser, then try a short-timeout PING.
static bool RedisReachable(string url)
{
    try
    {
        RedisEndpoint ep = RedisUrl.Parse(url);
        var options = new ConfigurationOptions { ConnectTimeout = 1500, ConnectRetry = 0, AbortOnConnectFail = true };
        if (ep.UnixSocketPath is not null)
            options.EndPoints.Add(new UnixDomainSocketEndPoint(ep.UnixSocketPath));
        else { options.EndPoints.Add(ep.Host, ep.Port); options.Ssl = ep.Ssl; }
        if (ep.Password is not null) options.Password = ep.Password;
        using var mux = ConnectionMultiplexer.Connect(options);
        return mux.GetDatabase().Ping() >= TimeSpan.Zero;
    }
    catch { return false; }
}

static void PrintNoRedisHelp(string url)
{
    Console.WriteLine($"No Redis reachable at {url}.");
    Console.WriteLine("Start one, then re-run:");
    Console.WriteLine("    docker run --rm -p 6379:6379 redis:7-alpine");
    Console.WriteLine("  or the repo's compose file:");
    Console.WriteLine("    docker compose -f test/OptunaSharp.Tests.Storage/docker-compose.yml up -d");
    Console.WriteLine("  point elsewhere with:  OPTUNASHARP_REDIS_URL=redis://host:6379");
}
