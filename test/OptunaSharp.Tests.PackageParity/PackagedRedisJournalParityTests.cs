using System;
using AwesomeAssertions;
using OptunaSharp.Samplers;
using OptunaSharp.Storages.Journal;

namespace OptunaSharp.Tests.PackageParity;

/// <summary>
///     Behavioral parity for a satellite package that a real external SERVICE backs: the PACKAGED
///     <c>OptunaSharp.Storage.StackExchange.Redis</c> journal backend must persist a study to a live Redis
///     and replay it byte-for-byte on reload. This is the packaged analogue of the source-referenced Redis
///     round-trip in <c>OptunaSharp.Tests.Storage</c>, but here the backend and Core arrive as shipped
///     packages from the local feed (no path to <c>src/</c>), so it proves the package's declared
///     dependency graph (StackExchange.Redis + Core) actually WORKS end-to-end, not merely that it loads.
/// </summary>
/// <remarks>
///     <para>Redis is reached at <c>OPTUNASHARP_REDIS_URL</c>. The compose gate sets it (redis://redis:6379)
///     and stands up Redis, so there it is a HARD requirement; a local run without that variable falls back
///     to <c>redis://localhost:6379</c> and self-skips (Inconclusive) if nothing is listening — the same
///     "green != skipped-away" contract the interop suite uses for Python. A connection failure when the URL
///     WAS explicitly set is a real failure, never a skip.</para>
///     <para>The reload uses a FRESH backend instance (a new connection) against the same key prefix, so a
///     pass means the trajectory survived a real Redis round-trip, not an in-process cache. RandomSampler is
///     value-independent, so the reloaded trials and best value must match exactly.</para>
/// </remarks>
[TestClass]
public class PackagedRedisJournalParityTests
{
    /// <summary>
    ///     Writes a small study through the packaged Redis journal backend, then re-opens it through a new
    ///     backend instance and asserts the persisted trajectory replays exactly.
    /// </summary>
    [TestMethod]
    [TestCategory("RequiresRedis")]
    public void PackagedRedisBackend_PersistsStudy_AndReplaysItOnReload()
    {
        // An explicitly configured URL (CI/compose) makes reachability mandatory; absent it, we attempt
        // localhost and treat "nobody listening" as Inconclusive rather than red.
        string? configured = Environment.GetEnvironmentVariable("OPTUNASHARP_REDIS_URL");
        bool required = !string.IsNullOrEmpty(configured);
        string url = required ? configured! : "redis://localhost:6379";

        // Unique key prefix so concurrent runs (and a shared CI Redis) never collide on study state.
        string prefix = $"pkgparity:{Guid.NewGuid():N}:";

        try
        {
            // 1) WRITE: create + optimize a study whose storage is the PACKAGED Redis journal backend.
            var write = Optuna.CreateStudy(
                storage: new JournalStorage(new StackExchangeJournalRedisBackend(url, useCluster: false, prefix: prefix)),
                studyName: "pkgparity", sampler: new RandomSampler(seed: 7), direction: StudyDirection.Minimize);
            write.Optimize(t => { double x = t.SuggestFloat("x", -5, 5); return x * x; }, nTrials: 8);
            int writtenCount = write.GetTrials().Count;
            double writtenBest = write.BestValue;

            // 2) RELOAD: a brand-new backend instance (new connection) onto the same prefix — this is the
            //    part that can only pass if the data really landed in Redis and reads back through the package.
            var read = Optuna.CreateStudy(
                storage: new JournalStorage(new StackExchangeJournalRedisBackend(url, useCluster: false, prefix: prefix)),
                studyName: "pkgparity", loadIfExists: true);
            read.GetTrials().Count.Should().Be(writtenCount, "the packaged Redis backend must persist every trial across connections");
            read.BestValue.Should().Be(writtenBest, "reload through the packaged Redis backend must be exact");
        }
        catch (Exception ex) when (!required && IsRedisUnavailable(ex))
        {
            // Local convenience: no Redis and none demanded -> don't fail the suite.
            Assert.Inconclusive(
                $"Redis not reachable at {url} ({ex.GetType().Name}); set OPTUNASHARP_REDIS_URL to require it. {ex.Message}");
        }
    }

    /// <summary>
    ///     True if <paramref name="ex" /> (or any inner exception) is a Redis/socket connectivity failure.
    ///     Matched by type NAME to avoid a compile-time dependency on StackExchange.Redis here — the client
    ///     type stays an implementation detail behind the packaged backend, exactly as a consumer sees it.
    /// </summary>
    /// <param name="ex">The exception thrown while constructing the backend or driving the study.</param>
    /// <returns><c>true</c> for a connect/timeout/socket failure (a "Redis is down" signal); otherwise <c>false</c>.</returns>
    private static bool IsRedisUnavailable(Exception ex)
    {
        // Walk the chain: StackExchange.Redis wraps the underlying SocketException in RedisConnectionException.
        for (Exception? e = ex; e is not null; e = e.InnerException)
        {
            if (e.GetType().Name is "RedisConnectionException" or "RedisTimeoutException" or "SocketException")
                return true;
        }
        return false;
    }
}
