using System;
using OptunaSharp.Storages.Journal;    // RedisUrl (friend access), JournalRedisBackend
using StackExchange.Redis;

namespace OptunaSharp.Tests.Storage;

/// <summary>
///     Reachability gating + connection factories for the Redis server defined in
///     <c>docker-compose.yml</c> (this folder). Mirrors the Artifacts suite's
///     <c>ArtifactEmulators</c> / the interop suite's <c>PythonSession.EnsureOrInconclusive</c>
///     pattern: an <c>Integration</c> test calls <see cref="EnsureReachable" /> first and reports
///     <see cref="Assert.Inconclusive(string)" /> when Redis is not reachable (so the suite is safe
///     to run without Docker), unless <c>OPTUNASHARP_REDIS_REQUIRED</c> is truthy — then an absent
///     server is a hard failure (for CI once wired).
///
///     Endpoint: <c>OPTUNASHARP_REDIS_URL</c> (default <c>redis://localhost:6379</c>).
/// </summary>
internal static class RedisSession
{
    public const int RedisPort = 6379;

    internal static string Url =>
        Environment.GetEnvironmentVariable("OPTUNASHARP_REDIS_URL") is { Length: > 0 } u ? u : "redis://localhost:6379";

    private static bool? _reachable;

    /// <summary>Skip (Inconclusive) — or fail, if required — when Redis is not reachable.</summary>
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

    /// <summary>A journal backend bound to the test Redis under a caller-chosen key prefix.</summary>
    internal static JournalRedisBackend NewBackend(string prefix, bool useCluster = false)
        => new(Url, useCluster: useCluster, prefix: prefix);

    /// <summary>A raw multiplexer for inspecting/seeding keys directly (byte-compat assertions).</summary>
    internal static ConnectionMultiplexer RawConnection()
        => ConnectionMultiplexer.Connect(RedisUrl.BuildConfiguration(Url));

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
            var options = RedisUrl.BuildConfiguration(Url);
            options.ConnectTimeout = 1500;
            options.ConnectRetry = 0;            // one shot — don't spend seconds retrying an absent server.
            options.AbortOnConnectFail = true;   // probe wants a hard yes/no, not lazy retries.
            using var mux = ConnectionMultiplexer.Connect(options);
            return mux.GetDatabase().Ping() >= TimeSpan.Zero;
        }
        catch
        {
            return false;
        }
    }
}
