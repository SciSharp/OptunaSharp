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

namespace OptunaSharp.Tests.Storage;

/// <summary>
///     The full behavioral contract every Redis journal backend must satisfy against a real Redis,
///     run once per client backend via the concrete subclasses (StackExchange / NRedisStack /
///     ServiceStack). The Artifacts suite's <c>ArtifactStoreContract</c> pattern: the assertions are
///     written once and parameterized by the abstract <see cref="Create" /> factory. Category
///     <c>Integration</c> (per method, so it is inherited); each self-skips when Redis is unreachable.
/// </summary>
public abstract class JournalBackendContractTests
{
    /// <summary>Builds the backend under test, bound to the shared test Redis under <paramref name="prefix" />.</summary>
    protected abstract BaseJournalBackend Create(string prefix, bool useCluster = false);

    private IBaseJournalSnapshot Snapshot(string prefix) => (IBaseJournalSnapshot)Create(prefix);

    private static OrderedDict<string, object?> Log(long op, string worker) =>
        new() { ["op"] = op, ["worker"] = worker };

    // ----------------------------------------------------------------- append / read

    [TestMethod]
    [TestCategory("Integration")]
    public void AppendThenReadAll_RoundTrips()
    {
        RedisSession.EnsureReachable();
        var backend = Create(RedisSession.UniquePrefix());
        backend.AppendLogs(new[] { Log(0, "a"), Log(1, "b"), Log(2, "c") });

        var all = backend.ReadLogs(0).ToList();
        all.Should().HaveCount(3);
        all[0]["op"].Should().Be(0L);
        all[0]["worker"].Should().Be("a");
        all[2]["worker"].Should().Be("c");
    }

    [TestMethod]
    [TestCategory("Integration")]
    public void ClusterModeBranch_RoundTrips()
    {
        RedisSession.EnsureReachable();
        var backend = Create(RedisSession.UniquePrefix(), useCluster: true);
        backend.AppendLogs(new[] { Log(0, "x"), Log(1, "y") });
        backend.ReadLogs(0).Select(l => (string)l["worker"]!).Should().Equal("x", "y");
    }

    [TestMethod]
    [TestCategory("Integration")]
    public void ReadLogs_FromMiddleOffset_ReturnsTail()
    {
        RedisSession.EnsureReachable();
        var backend = Create(RedisSession.UniquePrefix());
        backend.AppendLogs(new[] { Log(0, "a"), Log(1, "b"), Log(2, "c") });
        var tail = backend.ReadLogs(1).ToList();
        tail.Should().HaveCount(2);
        tail[0]["worker"].Should().Be("b");
    }

    [TestMethod]
    [TestCategory("Integration")]
    public void ReadLogs_FromBeyondMax_ReturnsEmpty()
    {
        RedisSession.EnsureReachable();
        var backend = Create(RedisSession.UniquePrefix());
        backend.AppendLogs(new[] { Log(0, "a"), Log(1, "b") });
        backend.ReadLogs(5).Should().BeEmpty();
    }

    [TestMethod]
    [TestCategory("Integration")]
    public void ReadLogs_WhenNoCounterKey_ReturnsEmpty()
    {
        RedisSession.EnsureReachable();
        Create(RedisSession.UniquePrefix()).ReadLogs(0).Should().BeEmpty();
    }

    [TestMethod]
    [TestCategory("Integration")]
    public void AppendLogs_MultipleCalls_CounterContinues()
    {
        RedisSession.EnsureReachable();
        var backend = Create(RedisSession.UniquePrefix());
        backend.AppendLogs(new[] { Log(0, "a"), Log(1, "b") });
        backend.AppendLogs(new[] { Log(2, "c") });
        backend.AppendLogs(new[] { Log(3, "d"), Log(4, "e") });
        backend.ReadLogs(0).Select(l => (string)l["worker"]!).Should().Equal("a", "b", "c", "d", "e");
    }

    [TestMethod]
    [TestCategory("Integration")]
    public void AppendLogs_EmptyList_InitializesCounterToMinusOne()
    {
        RedisSession.EnsureReachable();
        string prefix = RedisSession.UniquePrefix();
        var backend = Create(prefix);
        backend.AppendLogs(Array.Empty<OrderedDict<string, object?>>());

        using var mux = RedisSession.RawConnection();
        ((string?)mux.GetDatabase().StringGet(RedisJournalProtocol.LogNumberKey(prefix))).Should().Be("-1");
        backend.ReadLogs(0).Should().BeEmpty();
    }

    // ----------------------------------------------------------------- counter / keys

    [TestMethod]
    [TestCategory("Integration")]
    public void Counter_IncrementsPerLog_ZeroBased()
    {
        RedisSession.EnsureReachable();
        string prefix = RedisSession.UniquePrefix();
        Create(prefix).AppendLogs(new[] { Log(0, "a"), Log(1, "b"), Log(2, "c") });
        using var mux = RedisSession.RawConnection();
        ((string?)mux.GetDatabase().StringGet(RedisJournalProtocol.LogNumberKey(prefix))).Should().Be("2");
    }

    [TestMethod]
    [TestCategory("Integration")]
    public void Keys_MatchOptunaLayout()
    {
        RedisSession.EnsureReachable();
        string prefix = RedisSession.UniquePrefix();
        Create(prefix).AppendLogs(new[] { Log(0, "a") });
        var db = RedisSession.RawConnection().GetDatabase();
        db.KeyExists(RedisJournalProtocol.LogNumberKey(prefix)).Should().BeTrue();
        db.KeyExists(RedisJournalProtocol.LogKey(prefix, 0)).Should().BeTrue();
    }

    // ----------------------------------------------------------------- byte-compatibility

    [TestMethod]
    [TestCategory("Integration")]
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
        Create(prefix).AppendLogs(new[] { log });

        using var mux = RedisSession.RawConnection();
        ((string?)mux.GetDatabase().StringGet(RedisJournalProtocol.LogKey(prefix, 0))).Should().Be(
            "{\"op\": 4, \"worker\": \"w-1\", \"x\": 1.5, \"flag\": true, \"nothing\": null, " +
            "\"items\": [1, 2, 3], \"nested\": {\"a\": 1}}");
    }

    [TestMethod]
    [TestCategory("Integration")]
    public void ClusterAndStandalone_ProduceIdenticalStoredBytes()
    {
        RedisSession.EnsureReachable();
        string standalone = RedisSession.UniquePrefix();
        string cluster = RedisSession.UniquePrefix();
        var log = Log(7, "same");
        Create(standalone).AppendLogs(new[] { log });
        Create(cluster, useCluster: true).AppendLogs(new[] { log });

        var db = RedisSession.RawConnection().GetDatabase();
        ((string?)db.StringGet(RedisJournalProtocol.LogKey(standalone, 0)))
            .Should().Be((string?)db.StringGet(RedisJournalProtocol.LogKey(cluster, 0)));
    }

    // ----------------------------------------------------------------- corrupt-log read rule

    [TestMethod]
    [TestCategory("Integration")]
    public void ReadLogs_ToleratesCorruptLastLog()
    {
        RedisSession.EnsureReachable();
        string prefix = RedisSession.UniquePrefix();
        var db = RedisSession.RawConnection().GetDatabase();
        db.StringSet(RedisJournalProtocol.LogNumberKey(prefix), 1);
        db.StringSet(RedisJournalProtocol.LogKey(prefix, 0), "{\"op\": 0}");
        db.StringSet(RedisJournalProtocol.LogKey(prefix, 1), "{ this is not valid json");

        var logs = Create(prefix).ReadLogs(0).ToList();
        logs.Should().HaveCount(1);
        logs[0]["op"].Should().Be(0L);
    }

    [TestMethod]
    [TestCategory("Integration")]
    public void ReadLogs_RaisesOnCorruptNonLastLog()
    {
        RedisSession.EnsureReachable();
        string prefix = RedisSession.UniquePrefix();
        var db = RedisSession.RawConnection().GetDatabase();
        db.StringSet(RedisJournalProtocol.LogNumberKey(prefix), 2);
        db.StringSet(RedisJournalProtocol.LogKey(prefix, 0), "{\"op\": 0}");
        db.StringSet(RedisJournalProtocol.LogKey(prefix, 1), "{ not valid json");
        db.StringSet(RedisJournalProtocol.LogKey(prefix, 2), "{\"op\": 2}");

        var backend = Create(prefix);
        var act = () => backend.ReadLogs(0).ToList();
        act.Should().Throw<ValueError>();
    }

    [TestMethod]
    [TestCategory("Integration")]
    public void ReadLogs_PollsUntilAPendingLogAppears()
    {
        RedisSession.EnsureReachable();
        string prefix = RedisSession.UniquePrefix();
        var db = RedisSession.RawConnection().GetDatabase();
        db.StringSet(RedisJournalProtocol.LogNumberKey(prefix), 1);
        db.StringSet(RedisJournalProtocol.LogKey(prefix, 0), "{\"op\": 0}");
        // log:1 intentionally absent -> the reader polls for it.

        var backend = Create(prefix);
        var readTask = Task.Run(() => backend.ReadLogs(0).ToList());
        Thread.Sleep(300);
        db.StringSet(RedisJournalProtocol.LogKey(prefix, 1), "{\"op\": 1}");

        readTask.Wait(TimeSpan.FromSeconds(15)).Should().BeTrue("ReadLogs must return once the pending log appears");
        readTask.Result.Should().HaveCount(2);
    }

    // ----------------------------------------------------------------- snapshot

    [TestMethod]
    [TestCategory("Integration")]
    public void Snapshot_SaveLoad_RoundTrips()
    {
        RedisSession.EnsureReachable();
        var snap = Snapshot(RedisSession.UniquePrefix());
        var payload = new byte[] { 1, 2, 3, 42 };
        snap.SaveSnapshot(payload);
        snap.LoadSnapshot().Should().Equal(payload);
    }

    [TestMethod]
    [TestCategory("Integration")]
    public void Snapshot_WhenAbsent_ReturnsNull()
    {
        RedisSession.EnsureReachable();
        Snapshot(RedisSession.UniquePrefix()).LoadSnapshot().Should().BeNull();
    }

    [TestMethod]
    [TestCategory("Integration")]
    public void Snapshot_Overwrite_LoadsLatest()
    {
        RedisSession.EnsureReachable();
        var snap = Snapshot(RedisSession.UniquePrefix());
        snap.SaveSnapshot(new byte[] { 1, 1, 1 });
        snap.SaveSnapshot(new byte[] { 9, 8, 7, 6 });
        snap.LoadSnapshot().Should().Equal(new byte[] { 9, 8, 7, 6 });
    }

    [TestMethod]
    [TestCategory("Integration")]
    public void Snapshot_BinaryBytes_PreservedExactly()
    {
        RedisSession.EnsureReachable();
        var snap = Snapshot(RedisSession.UniquePrefix());
        var payload = new byte[] { 0x00, 0xFF, 0x7F, 0x80, 0x0A, 0x0D, 0x00 };
        snap.SaveSnapshot(payload);
        snap.LoadSnapshot().Should().Equal(payload);
    }

    // ----------------------------------------------------------------- isolation / prefix

    [TestMethod]
    [TestCategory("Integration")]
    public void DifferentPrefixes_AreIsolated()
    {
        RedisSession.EnsureReachable();
        var a = Create(RedisSession.UniquePrefix());
        var b = Create(RedisSession.UniquePrefix());
        a.AppendLogs(new[] { Log(0, "only-a") });
        b.ReadLogs(0).Should().BeEmpty();
    }

    // ----------------------------------------------------------------- concurrency (atomicity)

    [TestMethod]
    [TestCategory("Integration")]
    public void ConcurrentAppends_AreAtomic_NoLostUpdates()
    {
        RedisSession.EnsureReachable();
        string prefix = RedisSession.UniquePrefix();
        const int writers = 4;
        const int perWriter = 15;

        Parallel.For(0, writers, w =>
        {
            var backend = Create(prefix);
            for (int i = 0; i < perWriter; i++)
                backend.AppendLogs(new[] { Log(i, $"w{w}") });
        });

        int total = writers * perWriter;
        Create(prefix).ReadLogs(0).Should().HaveCount(total);
        using var mux = RedisSession.RawConnection();
        ((long)mux.GetDatabase().StringGet(RedisJournalProtocol.LogNumberKey(prefix))).Should().Be(total - 1);
    }

    // ----------------------------------------------------------------- experimental warning

    // [Misaligned] Optuna decorates JournalRedisBackend with @experimental_class("3.1.0"), which raises
    // an ExperimentalWarning on construction. OptunaSharp targets the Optuna 5.x surface and treats it
    // as stable, so no ExperimentalWarning is emitted — this guards that deliberate divergence.
    [TestMethod]
    [TestCategory("Integration")]
    [TestCategory("Misaligned")]
    public void Constructor_DoesNotRaiseExperimentalWarning()
    {
        RedisSession.EnsureReachable();
        (string Message, WarningCategory Category)? captured = null;
        void Handler(string m, WarningCategory c)
        {
            if (c == WarningCategory.ExperimentalWarning)
                captured = (m, c);
        }

        OptunaWarnings.WarningRaised += Handler;
        try
        {
            _ = Create(RedisSession.UniquePrefix());
        }
        finally
        {
            OptunaWarnings.WarningRaised -= Handler;
        }

        captured.Should().BeNull();
    }

    // ----------------------------------------------------------------- study replay

    [TestMethod]
    [TestCategory("Integration")]
    public void Study_RoundTrips_ThroughJournalStorage()
    {
        RedisSession.EnsureReachable();
        string prefix = RedisSession.UniquePrefix();
        const string studyName = "redis-roundtrip";

        var writeStorage = new JournalStorage(Create(prefix));
        var study = Optuna.CreateStudy(storage: writeStorage, studyName: studyName,
            sampler: new RandomSampler(seed: 7), direction: StudyDirection.Minimize);
        study.Optimize(t =>
        {
            double x = t.SuggestFloat("x", -10, 10);
            return (x - 2) * (x - 2);
        }, nTrials: 12);
        int writtenCount = study.GetTrials().Count;
        double bestValue = study.BestValue;

        var readStorage = new JournalStorage(Create(prefix));
        var reloaded = Optuna.CreateStudy(storage: readStorage, studyName: studyName, loadIfExists: true);
        reloaded.GetTrials().Count.Should().Be(writtenCount);
        reloaded.BestValue.Should().Be(bestValue);
    }
}
