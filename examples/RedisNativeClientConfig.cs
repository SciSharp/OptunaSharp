#:project ../src/OptunaSharp.Storage.StackExchange.Redis/OptunaSharp.Storage.StackExchange.Redis.csproj
#:project ../src/OptunaSharp.Storage.NRedisStack/OptunaSharp.Storage.NRedisStack.csproj
#:property PublishAot=false
#:property SkipNDScopeWeave=true
#:property GeneratePackageOnBuild=false

// Native Redis client configuration — the Redis backends are not limited to Optuna's URL string. each
// exposes its client's own configuration surface, so you drive StackExchange.Redis / NRedisStack exactly
// as you would on their own: a URL, a URL plus a tweak callback, a full ConfigurationOptions, or a
// client you already built and share across your app. and because all backends share one on-wire
// format, the clients are interchangeable — write with one, read with another, byte-for-byte.
//
// source (ported feature): https://optuna.readthedocs.io/en/stable/reference/generated/optuna.storages.journal.JournalRedisBackend.html
// source (client options):  https://seredis.dev/Configuration   (StackExchange.Redis ConfigurationOptions)
// source (NRedisStack):     https://github.com/redis/NRedisStack
//
// the objective is a self-contained computation (the point is the client configuration, not the
// objective). each backend constructor prints an ExperimentalWarning (JournalRedisBackend is
// @experimental_class in Optuna). needs a reachable Redis; if none is found the example exits cleanly.
//
//   Run:  docker run --rm -p 6379:6379 redis:7-alpine
//         dotnet run examples/RedisNativeClientConfig.cs
//
//   endpoint: $OPTUNASHARP_REDIS_URL (default redis://localhost:6379)

using System;
using System.Net.Sockets;
using OptunaSharp;
using OptunaSharp.Samplers;
using OptunaSharp.Storages;
using OptunaSharp.Storages.Journal;   // StackExchange/NRedisStack backends, RedisUrl, RedisEndpoint
using StackExchange.Redis;            // ConnectionMultiplexer, ConfigurationOptions

string url = Environment.GetEnvironmentVariable("OPTUNASHARP_REDIS_URL") is { Length: > 0 } u
    ? u : "redis://localhost:6379";

if (!RedisReachable(url))
{
    PrintNoRedisHelp(url);
    return;
}

// a unique key prefix so this demo never collides with other studies on the server.
string prefix = $"optunasharp-example:{Guid.NewGuid():N}";

// Four ways to construct a StackExchange.Redis backend, all also taking useCluster / prefix.
// The first two (shown here as comments) keep Optuna's URL; the two below are executed:
//
//   // 1. Optuna's URL constructor, verbatim:
//   new StackExchangeJournalRedisBackend(url, prefix: prefix);
//
//   // 2. keep the URL, tweak the derived ConfigurationOptions before connecting:
//   new StackExchangeJournalRedisBackend(url, o => { o.ClientName = "worker"; o.ConnectTimeout = 5000; }, prefix: prefix);

// 3. build a full ConfigurationOptions yourself — the connection entirely in the client's own terms.
ConfigurationOptions options = BuildOptions(url);
options.ClientName = "optunasharp-writer";         // shows up in Redis `CLIENT LIST` — a native knob

// 4. bring your own IConnectionMultiplexer: the one your app already pools / manages through DI. the
//    backend borrows it and NEVER disposes it — you own its lifetime.
using IConnectionMultiplexer mux = ConnectionMultiplexer.Connect(options);
var writeBackend = new StackExchangeJournalRedisBackend(mux, prefix: prefix);

// write a study through the shared StackExchange.Redis connection.
var writeStudy = Optuna.CreateStudy(
    storage: new JournalStorage(writeBackend),
    sampler: new RandomSampler(seed: 0),
    direction: StudyDirection.Minimize,
    studyName: "native-config-demo");
writeStudy.Optimize(Objective, nTrials: 20);
Console.WriteLine($"Wrote {writeStudy.Trials.Count} trials with StackExchange.Redis (best {writeStudy.BestValue:F6}).");

// Interchangeable clients: read the SAME study back with a DIFFERENT client library. NRedisStack
// (Redis's official client) replays the byte-identical journal StackExchange.Redis just wrote — same
// server, same prefix, same study name.
var readStudy = Optuna.CreateStudy(
    storage: new JournalStorage(new NRedisStackJournalRedisBackend(url, prefix: prefix)),
    studyName: "native-config-demo",
    loadIfExists: true);
Console.WriteLine($"Read {readStudy.Trials.Count} trials back with NRedisStack (best {readStudy.BestValue:F6}).");

bool interchangeable = readStudy.Trials.Count == writeStudy.Trials.Count
                       && readStudy.BestValue == writeStudy.BestValue;
Console.WriteLine(interchangeable
    ? "Match — the two client libraries are fully interchangeable over the same Redis."
    : "Mismatch — this should not happen.");

// (ServiceStack.Redis has a third backend with the same shape — incl. an IRedisClientsManager
//  constructor for full control — but it is a commercial package, so it is not referenced here.)

static double Objective(Trial t)
{
    double x = t.SuggestFloat("x", -10, 10);
    return Math.Pow(x - 2, 2.0);              // Math.Pow, not x*x, for Python x**2 parity
}

// --- native ConfigurationOptions from the redis:// URL (public APIs only) --------------------------
// maps the URL onto StackExchange.Redis' own options; AbortOnConnectFail=false mirrors redis-py's lazy
// connect (Optuna's redis.Redis.from_url does not dial until the first command).
static ConfigurationOptions BuildOptions(string url)
{
    RedisEndpoint ep = RedisUrl.Parse(url);
    var o = new ConfigurationOptions { AbortOnConnectFail = false };
    if (ep.UnixSocketPath is not null)
        o.EndPoints.Add(new UnixDomainSocketEndPoint(ep.UnixSocketPath));
    else { o.EndPoints.Add(ep.Host, ep.Port); o.Ssl = ep.Ssl; }
    if (ep.User is not null) o.User = ep.User;
    if (ep.Password is not null) o.Password = ep.Password;
    if (ep.Db is not null) o.DefaultDatabase = ep.Db;
    return o;
}

// --- reachability probe: the same options, tightened to fail fast ---------------------------------
static bool RedisReachable(string url)
{
    try
    {
        ConfigurationOptions o = BuildOptions(url);
        o.ConnectTimeout = 1500; o.ConnectRetry = 0; o.AbortOnConnectFail = true;
        using var mux = ConnectionMultiplexer.Connect(o);
        return mux.GetDatabase().Ping() >= TimeSpan.Zero;
    }
    catch { return false; }
}

static void PrintNoRedisHelp(string url)
{
    Console.WriteLine($"No Redis reachable at {url}.");
    Console.WriteLine("Start one, then re-run:");
    Console.WriteLine("    docker run --rm -p 6379:6379 redis:7-alpine");
    Console.WriteLine("  or:  docker compose -f test/OptunaSharp.Tests.Storage/docker-compose.yml up -d");
    Console.WriteLine("  point elsewhere with:  OPTUNASHARP_REDIS_URL=redis://host:6379");
}
