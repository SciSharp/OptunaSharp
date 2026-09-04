using System;
using OptunaSharp.Storages.Journal;    // RedisUrl.Parse, the three backends (friend access to ToConfiguration)
using StackExchange.Redis;

namespace OptunaSharp.Tests.Storage;

/// <summary>
///     Reachability gating + backend factories for the Redis server defined in <c>docker-compose.yml</c>
///     (this folder). Mirrors the Artifacts suite's <c>ArtifactEmulators</c> / the interop suite's
///     <c>PythonSession.EnsureOrInconclusive</c>: an <c>Integration</c> test calls
///     <see cref="EnsureReachable" /> first and reports <see cref="Assert.Inconclusive(string)" /> when
///     Redis is unreachable (so the suite is safe without Docker), unless <c>OPTUNASHARP_REDIS_REQUIRED</c>
///     is truthy (then absent = hard failure, for CI once wired).
///
///     Endpoint: <c>OPTUNASHARP_REDIS_URL</c> (default <c>redis://localhost:6379</c>).
/// </summary>
internal static class RedisSession
{
    internal static string Url =>
        Environment.GetEnvironmentVariable("OPTUNASHARP_REDIS_URL") is { Length: > 0 } u ? u : "redis://localhost:6379";

    /// <summary>
    ///     The three client backends under test, addressable by name for the per-backend contract and
    ///     the cross-library parity matrix. Each factory builds a backend bound to the test Redis under
    ///     <c>prefix</c>, in the requested standalone/cluster mode.
    /// </summary>
    internal static readonly (string Name, Func<string, bool, BaseJournalBackend> Create)[] Backends =
    {
        ("StackExchange", (prefix, cluster) => new StackExchangeJournalRedisBackend(Url, cluster, prefix)),
        ("NRedisStack",   (prefix, cluster) => new NRedisStackJournalRedisBackend(Url, cluster, prefix)),
        ("ServiceStack",  (prefix, cluster) => new ServiceStackJournalRedisBackend(Url, cluster, prefix)),
    };

    private static bool? _reachable;

    public static void EnsureReachable()
    {
        _reachable ??= Probe();
        if (_reachable.Value)
            return;

        string reason =
            $"Redis not reachable at {Url}. Start it with: " +
            "docker compose -f test/OptunaSharp.Tests.Storage/docker-compose.yml up -d";
        if (Required())
            Assert.Fail(reason + " (OPTUNASHARP_REDIS_REQUIRED is set, so this is a hard failure).");
        Assert.Inconclusive(reason);
    }

    /// <summary>A raw StackExchange multiplexer for inspecting/seeding keys directly (byte-compat).</summary>
    internal static ConnectionMultiplexer RawConnection()
        => ConnectionMultiplexer.Connect(StackExchangeJournalRedisBackend.ToConfiguration(RedisUrl.Parse(Url)));

    /// <summary>A fresh, unique key prefix so parallel/repeat runs never collide on a shared server.</summary>
    internal static string UniquePrefix() => $"optunasharp-test:{Guid.NewGuid():N}";

    private static bool Required()
    {
        string? v = Environment.GetEnvironmentVariable("OPTUNASHARP_REDIS_REQUIRED");
        return v is "1" or "true" or "TRUE" or "yes";
    }

    private static bool Probe()
    {
        try
        {
            var options = StackExchangeJournalRedisBackend.ToConfiguration(RedisUrl.Parse(Url));
            options.ConnectTimeout = 1500;
            options.ConnectRetry = 0;
            options.AbortOnConnectFail = true;
            using var mux = ConnectionMultiplexer.Connect(options);
            return mux.GetDatabase().Ping() >= TimeSpan.Zero;
        }
        catch
        {
            return false;
        }
    }
}
