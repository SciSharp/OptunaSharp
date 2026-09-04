using System;
using OptunaSharp.Storages.Journal;    // RedisUrl (friend access)
using StackExchange.Redis;

namespace OptunaSharp.Tests.Storage.Redis;

/// <summary>
///     Discovery + self-skip gate for the live-Redis parity tests — the analogue of the interop
///     suite's <c>PythonSession</c>. A machine without a reachable Redis reports <b>Inconclusive</b>
///     (never red), unless <see cref="RequireEnvVar" /> is truthy (CI sets it after standing up a
///     Redis, so a discovery break goes red instead of green-by-skipping).
///
///     The server URL comes from <see cref="UrlEnvVar" /> (default <c>redis://localhost:6379</c>).
///     Start one locally with e.g. <c>docker run --rm -p 6379:6379 redis:7-alpine</c>.
/// </summary>
internal static class RedisSession
{
    internal const string UrlEnvVar = "OPTUNASHARP_REDIS_URL";
    internal const string RequireEnvVar = "OPTUNASHARP_REDIS_REQUIRE_SERVER";

    internal static string Url =>
        Environment.GetEnvironmentVariable(UrlEnvVar) is { Length: > 0 } u ? u : "redis://localhost:6379";

    private static bool? _available;

    internal static bool IsAvailable()
    {
        _available ??= Probe();
        return _available.Value;
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

    /// <summary>Skip (Inconclusive) when no Redis is reachable — or fail hard when required.</summary>
    internal static void EnsureOrInconclusive()
    {
        if (IsAvailable())
            return;
        if (IsTruthy(Environment.GetEnvironmentVariable(RequireEnvVar)))
            Assert.Fail($"No Redis reachable at '{Url}', but {RequireEnvVar} is set (a discovery break must go red, not skip).");
        Assert.Inconclusive(
            $"No Redis reachable at '{Url}'; skipping live parity. Set {UrlEnvVar} or start one " +
            "(e.g. `docker run --rm -p 6379:6379 redis:7-alpine`).");
    }

    /// <summary>A fresh, unique key prefix so parallel/repeat runs never collide on a shared server.</summary>
    internal static string UniquePrefix() => $"optunasharp-test:{Guid.NewGuid():N}";

    private static bool IsTruthy(string? value) =>
        value is not null &&
        (value == "1" ||
         value.Equals("true", StringComparison.OrdinalIgnoreCase) ||
         value.Equals("yes", StringComparison.OrdinalIgnoreCase));
}
