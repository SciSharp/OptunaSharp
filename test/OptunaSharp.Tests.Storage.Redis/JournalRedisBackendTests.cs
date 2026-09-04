using System.Collections.Generic;
using System.Linq;
using AwesomeAssertions;
using OptunaSharp;
using OptunaSharp.Collections;
using OptunaSharp.Samplers;
using OptunaSharp.Storages.Journal;
using StackExchange.Redis;

namespace OptunaSharp.Tests.Storage.Redis;

/// <summary>
///     Live round-trip parity for <see cref="JournalRedisBackend" /> against a real Redis server.
///     Self-skips (Inconclusive) when none is reachable — see <see cref="RedisSession" />. These are
///     the analogue of the journal-file interop gate: the on-wire encoding is asserted byte-compatible
///     with Python Optuna's (so a study is shareable across the language boundary), plus a full study
///     is written and replayed back through <see cref="JournalStorage" />.
/// </summary>
[TestClass]
[TestCategory("RequiresRedis")]
public class JournalRedisBackendTests
{
    private static OrderedDict<string, object?> Log(long op, string worker) =>
        new() { ["op"] = op, ["worker"] = worker };

    private static IDatabase RawDb(out ConnectionMultiplexer mux)
    {
        mux = ConnectionMultiplexer.Connect(RedisUrl.BuildConfiguration(RedisSession.Url));
        return mux.GetDatabase();
    }

    [TestMethod]
    public void AppendThenReadLogs_RoundTrips()
    {
        RedisSession.EnsureOrInconclusive();
        string prefix = RedisSession.UniquePrefix();
        var backend = new JournalRedisBackend(RedisSession.Url, prefix: prefix);

        backend.AppendLogs(new[] { Log(0, "a"), Log(1, "b"), Log(2, "c") });

        var all = backend.ReadLogs(0).ToList();
        all.Should().HaveCount(3);
        all[0]["op"].Should().Be(0L);
        all[0]["worker"].Should().Be("a");
        all[2]["worker"].Should().Be("c");

        // Reading from a middle offset returns only the tail.
        var tail = backend.ReadLogs(1).ToList();
        tail.Should().HaveCount(2);
        tail[0]["worker"].Should().Be("b");
    }

    [TestMethod]
    public void ReadLogs_WhenNoCounterKey_ReturnsNothing()
    {
        RedisSession.EnsureOrInconclusive();
        var backend = new JournalRedisBackend(RedisSession.Url, prefix: RedisSession.UniquePrefix());
        backend.ReadLogs(0).Should().BeEmpty();
    }

    [TestMethod]
    public void ClusterModeBranch_AppendThenReadLogs_RoundTrips()
    {
        // useCluster:true takes the non-atomic incr+set path instead of the Lua EVAL; on a standalone
        // server it must still round-trip (both are valid standalone commands).
        RedisSession.EnsureOrInconclusive();
        string prefix = RedisSession.UniquePrefix();
        var backend = new JournalRedisBackend(RedisSession.Url, useCluster: true, prefix: prefix);

        backend.AppendLogs(new[] { Log(0, "x"), Log(1, "y") });

        var all = backend.ReadLogs(0).ToList();
        all.Should().HaveCount(2);
        all[1]["worker"].Should().Be("y");
    }

    [TestMethod]
    public void StoredEncoding_IsByteCompatibleWithPython()
    {
        RedisSession.EnsureOrInconclusive();
        string prefix = RedisSession.UniquePrefix();
        var backend = new JournalRedisBackend(RedisSession.Url, prefix: prefix);
        backend.AppendLogs(new[] { Log(0, "w1") });

        var db = RawDb(out var mux);
        using (mux)
        {
            // The counter is initialized to -1 then incr'd to 0 for the first log.
            ((string?)db.StringGet($"{prefix}:log_number")).Should().Be("0");

            // Python's JournalRedisBackend stores json.dumps(log) with the DEFAULT (spaced)
            // separators — NOT the file backend's compact form. Assert the exact bytes.
            ((string?)db.StringGet($"{prefix}:log:0")).Should().Be("{\"op\": 0, \"worker\": \"w1\"}");
        }
    }

    [TestMethod]
    public void Snapshot_SaveLoad_RoundTrips()
    {
        RedisSession.EnsureOrInconclusive();
        string prefix = RedisSession.UniquePrefix();
        var backend = new JournalRedisBackend(RedisSession.Url, prefix: prefix);

        backend.LoadSnapshot().Should().BeNull();   // absent → null.

        var payload = new byte[] { 1, 2, 3, 0, 255, 42 };
        backend.SaveSnapshot(payload);
        backend.LoadSnapshot().Should().Equal(payload);
    }

    [TestMethod]
    public void Study_RoundTrips_ThroughJournalStorage()
    {
        RedisSession.EnsureOrInconclusive();
        string prefix = RedisSession.UniquePrefix();
        const string studyName = "redis-roundtrip";

        // Write a study to Redis.
        var writeStorage = new JournalStorage(new JournalRedisBackend(RedisSession.Url, prefix: prefix));
        var study = Optuna.CreateStudy(storage: writeStorage, studyName: studyName,
            sampler: new RandomSampler(seed: 7), direction: StudyDirection.Minimize);
        study.Optimize(t =>
        {
            double x = t.SuggestFloat("x", -10, 10);
            return (x - 2) * (x - 2);
        }, nTrials: 12);
        int writtenCount = study.GetTrials().Count;
        double bestValue = study.BestValue;

        // Replay it back from a fresh backend + storage on the same Redis keys.
        var readStorage = new JournalStorage(new JournalRedisBackend(RedisSession.Url, prefix: prefix));
        var reloaded = Optuna.CreateStudy(storage: readStorage, studyName: studyName, loadIfExists: true);
        reloaded.GetTrials().Count.Should().Be(writtenCount);
        reloaded.BestValue.Should().Be(bestValue);
    }
}
