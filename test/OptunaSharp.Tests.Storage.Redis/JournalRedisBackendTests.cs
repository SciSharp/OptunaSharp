using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using AwesomeAssertions;
using NumSharp;                          // ValueError
using OptunaSharp;
using OptunaSharp.Collections;
using OptunaSharp.Internal;              // OptunaWarnings / WarningCategory (Core friend grant)
using OptunaSharp.Samplers;
using OptunaSharp.Storages.Journal;
using StackExchange.Redis;

namespace OptunaSharp.Tests.Storage.Redis;

/// <summary>
///     Full integration coverage for <see cref="JournalRedisBackend" /> against a real Redis (the
///     <c>docker-compose.yml</c> in this folder). Category <c>Integration</c> so CI can include or
///     exclude the whole set; each test self-skips (Inconclusive) when Redis is unreachable — the
///     Artifacts suite's dockerized-emulator pattern.
///
///     The on-wire encoding is asserted byte-compatible with Python Optuna's JournalRedisBackend
///     (shared <c>{prefix}:log_number</c> counter, <c>{prefix}:log:{n}</c> entries stored as
///     <c>json.dumps(log)</c> with DEFAULT spaced separators, <c>{prefix}:snapshot</c> blob); the
///     Python↔C# study round-trip over this same server lives in the interop suite.
/// </summary>
[TestClass]
[TestCategory("Integration")]
public class JournalRedisBackendTests
{
    // ----------------------------------------------------------------- helpers

    private static OrderedDict<string, object?> Log(long op, string worker) =>
        new() { ["op"] = op, ["worker"] = worker };

    /// <summary>The append→read round-trip contract, shared by the standalone and cluster branches.</summary>
    private static void AssertAppendReadRoundTrip(JournalRedisBackend backend)
    {
        backend.AppendLogs(new[] { Log(0, "a"), Log(1, "b"), Log(2, "c") });

        var all = backend.ReadLogs(0).ToList();
        all.Should().HaveCount(3);
        all[0]["op"].Should().Be(0L);
        all[0]["worker"].Should().Be("a");
        all[1]["worker"].Should().Be("b");
        all[2]["op"].Should().Be(2L);
        all[2]["worker"].Should().Be("c");
    }

    // ----------------------------------------------------------------- append / read

    [TestMethod]
    public void AppendThenReadAll_RoundTrips()
    {
        RedisSession.EnsureReachable();
        AssertAppendReadRoundTrip(RedisSession.NewBackend(RedisSession.UniquePrefix()));
    }

    [TestMethod]
    public void ClusterModeBranch_RoundTrips()
    {
        // useCluster:true takes the non-atomic incr+set path instead of the Lua EVAL; on a standalone
        // server it must still round-trip (both are valid standalone commands).
        RedisSession.EnsureReachable();
        AssertAppendReadRoundTrip(RedisSession.NewBackend(RedisSession.UniquePrefix(), useCluster: true));
    }

    [TestMethod]
    public void ReadLogs_FromMiddleOffset_ReturnsTail()
    {
        RedisSession.EnsureReachable();
        var backend = RedisSession.NewBackend(RedisSession.UniquePrefix());
        backend.AppendLogs(new[] { Log(0, "a"), Log(1, "b"), Log(2, "c") });

        var tail = backend.ReadLogs(1).ToList();
        tail.Should().HaveCount(2);
        tail[0]["worker"].Should().Be("b");
        tail[1]["worker"].Should().Be("c");
    }

    [TestMethod]
    public void ReadLogs_FromBeyondMax_ReturnsEmpty()
    {
        RedisSession.EnsureReachable();
        var backend = RedisSession.NewBackend(RedisSession.UniquePrefix());
        backend.AppendLogs(new[] { Log(0, "a"), Log(1, "b") });   // max log number = 1

        backend.ReadLogs(5).Should().BeEmpty();
    }

    [TestMethod]
    public void ReadLogs_WhenNoCounterKey_ReturnsEmpty()
    {
        RedisSession.EnsureReachable();
        RedisSession.NewBackend(RedisSession.UniquePrefix()).ReadLogs(0).Should().BeEmpty();
    }

    [TestMethod]
    public void AppendLogs_MultipleCalls_CounterContinuesAcrossCalls()
    {
        RedisSession.EnsureReachable();
        var backend = RedisSession.NewBackend(RedisSession.UniquePrefix());
        backend.AppendLogs(new[] { Log(0, "a"), Log(1, "b") });
        backend.AppendLogs(new[] { Log(2, "c") });
        backend.AppendLogs(new[] { Log(3, "d"), Log(4, "e") });

        var all = backend.ReadLogs(0).ToList();
        all.Select(l => (string)l["worker"]!).Should().Equal("a", "b", "c", "d", "e");
    }

    [TestMethod]
    public void AppendLogs_EmptyList_InitializesCounterToMinusOne_AndReadsNothing()
    {
        RedisSession.EnsureReachable();
        string prefix = RedisSession.UniquePrefix();
        var backend = RedisSession.NewBackend(prefix);

        backend.AppendLogs(Array.Empty<OrderedDict<string, object?>>());

        using var mux = RedisSession.RawConnection();
        ((string?)mux.GetDatabase().StringGet($"{prefix}:log_number")).Should().Be("-1");
        backend.ReadLogs(0).Should().BeEmpty();
    }

    // ----------------------------------------------------------------- counter / key layout

    [TestMethod]
    public void Counter_IncrementsPerLog_ZeroBased()
    {
        RedisSession.EnsureReachable();
        string prefix = RedisSession.UniquePrefix();
        RedisSession.NewBackend(prefix).AppendLogs(new[] { Log(0, "a"), Log(1, "b"), Log(2, "c") });

        using var mux = RedisSession.RawConnection();
        // -1 init, then one incr per log -> last index is (count - 1).
        ((string?)mux.GetDatabase().StringGet($"{prefix}:log_number")).Should().Be("2");
    }

    [TestMethod]
    public void Keys_MatchOptunaLayout()
    {
        RedisSession.EnsureReachable();
        string prefix = RedisSession.UniquePrefix();
        RedisSession.NewBackend(prefix).AppendLogs(new[] { Log(0, "a") });

        var db = RedisSession.RawConnection().GetDatabase();
        db.KeyExists($"{prefix}:log_number").Should().BeTrue();
        db.KeyExists($"{prefix}:log:0").Should().BeTrue();
    }

    // ----------------------------------------------------------------- byte-compatibility

    [TestMethod]
    public void StoredLog_ExactBytes_MatchPythonJsonDumps()
    {
        RedisSession.EnsureReachable();
        string prefix = RedisSession.UniquePrefix();
        var log = new OrderedDict<string, object?>
        {
            ["op"] = 4L,
            ["worker"] = "w-1",
            ["x"] = 1.5,
            ["flag"] = true,
            ["nothing"] = null,
            ["items"] = new List<object?> { 1L, 2L, 3L },
            ["nested"] = new OrderedDict<string, object?> { ["a"] = 1L },
        };
        RedisSession.NewBackend(prefix).AppendLogs(new[] { log });

        // Python json.dumps(log) with the DEFAULT separators (", " / ": ") — the exact bytes Python
        // Optuna would store; NOT the file backend's compact form.
        using var mux = RedisSession.RawConnection();
        ((string?)mux.GetDatabase().StringGet($"{prefix}:log:0")).Should().Be(
            "{\"op\": 4, \"worker\": \"w-1\", \"x\": 1.5, \"flag\": true, \"nothing\": null, " +
            "\"items\": [1, 2, 3], \"nested\": {\"a\": 1}}");
    }

    [TestMethod]
    public void ClusterAndStandalone_ProduceIdenticalStoredBytes()
    {
        RedisSession.EnsureReachable();
        string standalone = RedisSession.UniquePrefix();
        string cluster = RedisSession.UniquePrefix();
        var log = Log(7, "same");
        RedisSession.NewBackend(standalone).AppendLogs(new[] { log });
        RedisSession.NewBackend(cluster, useCluster: true).AppendLogs(new[] { log });

        var db = RedisSession.RawConnection().GetDatabase();
        ((string?)db.StringGet($"{standalone}:log:0"))
            .Should().Be((string?)db.StringGet($"{cluster}:log:0"));
    }

    // ----------------------------------------------------------------- corrupt-log read rule

    [TestMethod]
    public void ReadLogs_ToleratesCorruptLastLog()
    {
        // A writer caught mid-append bumps the counter before the value is set; the LAST log may be
        // missing/garbled. Optuna tolerates that (yields the good prefix, no throw).
        RedisSession.EnsureReachable();
        string prefix = RedisSession.UniquePrefix();
        var db = RedisSession.RawConnection().GetDatabase();
        db.StringSet($"{prefix}:log_number", 1);
        db.StringSet($"{prefix}:log:0", "{\"op\": 0}");
        db.StringSet($"{prefix}:log:1", "{ this is not valid json");

        var logs = RedisSession.NewBackend(prefix).ReadLogs(0).ToList();
        logs.Should().HaveCount(1);
        logs[0]["op"].Should().Be(0L);
    }

    [TestMethod]
    public void ReadLogs_RaisesOnCorruptNonLastLog()
    {
        // A decode error on any log that is NOT the last is real corruption -> re-raised.
        RedisSession.EnsureReachable();
        string prefix = RedisSession.UniquePrefix();
        var db = RedisSession.RawConnection().GetDatabase();
        db.StringSet($"{prefix}:log_number", 2);
        db.StringSet($"{prefix}:log:0", "{\"op\": 0}");
        db.StringSet($"{prefix}:log:1", "{ not valid json");
        db.StringSet($"{prefix}:log:2", "{\"op\": 2}");

        var backend = RedisSession.NewBackend(prefix);
        var act = () => backend.ReadLogs(0).ToList();
        act.Should().Throw<ValueError>();
    }

    // ----------------------------------------------------------------- snapshot

    [TestMethod]
    public void Snapshot_SaveLoad_RoundTrips()
    {
        RedisSession.EnsureReachable();
        var backend = RedisSession.NewBackend(RedisSession.UniquePrefix());
        var payload = new byte[] { 1, 2, 3, 42 };
        backend.SaveSnapshot(payload);
        backend.LoadSnapshot().Should().Equal(payload);
    }

    [TestMethod]
    public void Snapshot_WhenAbsent_ReturnsNull()
    {
        RedisSession.EnsureReachable();
        RedisSession.NewBackend(RedisSession.UniquePrefix()).LoadSnapshot().Should().BeNull();
    }

    [TestMethod]
    public void Snapshot_Overwrite_LoadsLatest()
    {
        RedisSession.EnsureReachable();
        var backend = RedisSession.NewBackend(RedisSession.UniquePrefix());
        backend.SaveSnapshot(new byte[] { 1, 1, 1 });
        backend.SaveSnapshot(new byte[] { 9, 8, 7, 6 });
        backend.LoadSnapshot().Should().Equal(new byte[] { 9, 8, 7, 6 });
    }

    [TestMethod]
    public void Snapshot_BinaryBytes_PreservedExactly()
    {
        RedisSession.EnsureReachable();
        var backend = RedisSession.NewBackend(RedisSession.UniquePrefix());
        var payload = new byte[] { 0x00, 0xFF, 0x7F, 0x80, 0x0A, 0x0D, 0x00 };
        backend.SaveSnapshot(payload);
        backend.LoadSnapshot().Should().Equal(payload);
    }

    // ----------------------------------------------------------------- isolation / prefix

    [TestMethod]
    public void DifferentPrefixes_AreIsolated()
    {
        RedisSession.EnsureReachable();
        var a = RedisSession.NewBackend(RedisSession.UniquePrefix());
        var b = RedisSession.NewBackend(RedisSession.UniquePrefix());
        a.AppendLogs(new[] { Log(0, "only-a") });

        b.ReadLogs(0).Should().BeEmpty();
    }

    [TestMethod]
    public void EmptyPrefix_RoundTrips()
    {
        // Optuna's default prefix is "" -> literal keys ":log_number", ":log:{n}", ":snapshot".
        RedisSession.EnsureReachable();
        using var mux = RedisSession.RawConnection();
        var db = mux.GetDatabase();
        // Clear any leftovers from a prior run so the counter/keys start clean.
        db.KeyDelete(":log_number");
        db.KeyDelete(":snapshot");
        var server = mux.GetServer(mux.GetEndPoints()[0]);
        foreach (var k in server.Keys(pattern: ":log:*").ToArray())
            db.KeyDelete(k);

        var backend = RedisSession.NewBackend(prefix: "");
        backend.AppendLogs(new[] { Log(0, "root") });

        db.KeyExists(":log:0").Should().BeTrue();
        backend.ReadLogs(0).Single()["worker"].Should().Be("root");
    }

    // ----------------------------------------------------------------- deprecated alias

    [TestMethod]
    public void JournalRedisStorage_DeprecatedAlias_RoundTrips()
    {
        RedisSession.EnsureReachable();
        string prefix = RedisSession.UniquePrefix();
#pragma warning disable CS0618 // JournalRedisStorage is intentionally deprecated; test it still works.
        var backend = new JournalRedisStorage(RedisSession.Url, prefix: prefix);
#pragma warning restore CS0618
        backend.AppendLogs(new[] { Log(0, "legacy") });
        backend.ReadLogs(0).Single()["worker"].Should().Be("legacy");
    }

    // ----------------------------------------------------------------- concurrency (atomicity)

    [TestMethod]
    public void ConcurrentAppends_AreAtomic_NoLostUpdates()
    {
        // The standalone Lua EVAL makes incr+set atomic, so N writers appending to one prefix never
        // lose a counter slot: every log lands and is readable.
        RedisSession.EnsureReachable();
        string prefix = RedisSession.UniquePrefix();
        const int writers = 4;
        const int perWriter = 15;

        Parallel.For(0, writers, w =>
        {
            var backend = RedisSession.NewBackend(prefix);
            for (int i = 0; i < perWriter; i++)
                backend.AppendLogs(new[] { Log(i, $"w{w}") });
        });

        int total = writers * perWriter;
        RedisSession.NewBackend(prefix).ReadLogs(0).Should().HaveCount(total);
        using var mux = RedisSession.RawConnection();
        ((long)mux.GetDatabase().StringGet($"{prefix}:log_number")).Should().Be(total - 1);
    }

    // ----------------------------------------------------------------- study replay

    [TestMethod]
    public void Study_RoundTrips_ThroughJournalStorage()
    {
        RedisSession.EnsureReachable();
        string prefix = RedisSession.UniquePrefix();
        const string studyName = "redis-roundtrip";

        var writeStorage = new JournalStorage(RedisSession.NewBackend(prefix));
        var study = Optuna.CreateStudy(storage: writeStorage, studyName: studyName,
            sampler: new RandomSampler(seed: 7), direction: StudyDirection.Minimize);
        study.Optimize(t =>
        {
            double x = t.SuggestFloat("x", -10, 10);
            return (x - 2) * (x - 2);
        }, nTrials: 12);
        int writtenCount = study.GetTrials().Count;
        double bestValue = study.BestValue;

        // Replay from a fresh backend + storage on the same Redis keys.
        var readStorage = new JournalStorage(RedisSession.NewBackend(prefix));
        var reloaded = Optuna.CreateStudy(storage: readStorage, studyName: studyName, loadIfExists: true);
        reloaded.GetTrials().Count.Should().Be(writtenCount);
        reloaded.BestValue.Should().Be(bestValue);
    }

    // ----------------------------------------------------------------- read poll / backoff

    [TestMethod]
    public void ReadLogs_PollsUntilAPendingLogAppears()
    {
        // read_logs polls (with exponential backoff) for a log whose counter slot exists but whose
        // value has not been written yet — a concurrent writer caught between INCR and SET. Seed that
        // state and make the value appear mid-read; the reader must block then complete.
        RedisSession.EnsureReachable();
        string prefix = RedisSession.UniquePrefix();
        var db = RedisSession.RawConnection().GetDatabase();
        db.StringSet($"{prefix}:log_number", 1);            // max = 1 (slots 0 and 1)
        db.StringSet($"{prefix}:log:0", "{\"op\": 0}");
        // log:1 intentionally absent -> the reader will poll for it.

        var backend = RedisSession.NewBackend(prefix);
        var readTask = Task.Run(() => backend.ReadLogs(0).ToList());

        // Let the reader reach the poll loop on the missing log:1 (past at least one backoff), then supply it.
        Thread.Sleep(300);
        db.StringSet($"{prefix}:log:1", "{\"op\": 1}");

        readTask.Wait(TimeSpan.FromSeconds(15)).Should().BeTrue("ReadLogs must return once the pending log appears");
        var logs = readTask.Result;
        logs.Should().HaveCount(2);
        logs[1]["op"].Should().Be(1L);
    }

    // ----------------------------------------------------------------- experimental warning

    [TestMethod]
    public void Constructor_RaisesExperimentalWarning()
    {
        // @experimental_class("3.1.0"): constructing the backend must raise Optuna's ExperimentalWarning
        // with the verbatim message (asserted via Core's internal warning seam — hence the friend grant).
        RedisSession.EnsureReachable();
        (string Message, WarningCategory Category)? captured = null;
        void Handler(string m, WarningCategory c) => captured = (m, c);

        OptunaWarnings.WarningRaised += Handler;
        try
        {
            _ = RedisSession.NewBackend(RedisSession.UniquePrefix());
        }
        finally
        {
            OptunaWarnings.WarningRaised -= Handler;
        }

        captured.Should().NotBeNull();
        captured!.Value.Category.Should().Be(WarningCategory.ExperimentalWarning);
        captured.Value.Message.Should().Be(
            "JournalRedisBackend is experimental (supported from v3.1.0). The interface can change in the future.");
    }

    // ----------------------------------------------------------------- shared-multiplexer ctor

    [TestMethod]
    public void SharedMultiplexerConstructor_RoundTrips()
    {
        // The internal ctor binds to an already-built multiplexer (an app sharing one connection across
        // many backends) instead of connecting from the URL.
        RedisSession.EnsureReachable();
        using ConnectionMultiplexer mux = RedisSession.RawConnection();
        string prefix = RedisSession.UniquePrefix();

        var backend = new JournalRedisBackend(RedisSession.Url, mux, useCluster: false, prefix: prefix);
        backend.AppendLogs(new[] { Log(0, "shared") });
        backend.ReadLogs(0).Single()["worker"].Should().Be("shared");
    }
}
