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

    // ----------------------------------------------------------------- batched reads (plan 10, item 2.2)

    /// <summary>
    ///     Seeds a standalone journal of <paramref name="count" /> records straight through a raw connection, the way
    ///     Python's append script leaves them: the counter at the last number, one JSON record per key. Seeding raw keeps
    ///     a long journal from costing ServiceStack's request budget, which only its reads below then spend.
    /// </summary>
    /// <param name="prefix">The journal's key prefix.</param>
    /// <param name="count">How many records to write, numbered from 0.</param>
    private static void SeedRecords(string prefix, int count)
    {
        using var mux = RedisSession.RawConnection();
        var db = mux.GetDatabase();
        var batch = db.CreateBatch();
        var writes = Enumerable.Range(0, count)
            .Select(i => batch.StringSetAsync(RedisJournalProtocol.LogKey(prefix, i), $"{{\"op\": {i}}}"))
            .Append(batch.StringSetAsync(RedisJournalProtocol.LogNumberKey(prefix), count - 1))
            .ToArray();
        batch.Execute();
        Task.WaitAll(writes);
    }

    /// <summary>
    ///     A journal longer than two read batches (<see cref="RedisJournalProtocol.ReadLogsBatchLimit" /> records each)
    ///     reads back whole and in order, from the start and from the middle of a batch.
    /// </summary>
    [TestMethod]
    [TestCategory("Integration")]
    public void ReadLogs_AcrossSeveralBatches_ReturnsEveryRecordInOrder()
    {
        RedisSession.EnsureReachable();
        string prefix = RedisSession.UniquePrefix();
        int count = 2 * RedisJournalProtocol.ReadLogsBatchLimit + 500;
        SeedRecords(prefix, count);
        var backend = Create(prefix);

        backend.ReadLogs(0).Select(l => (long)l["op"]!).Should().Equal(Enumerable.Range(0, count).Select(i => (long)i));
        backend.ReadLogs(1500).Select(l => (long)l["op"]!).Should().Equal(Enumerable.Range(1500, count - 1500).Select(i => (long)i));
    }

    /// <summary>
    ///     A read ends at the counter value it saw first, as Python's does: records appended while it is under way, here
    ///     between its first and its second batch, are left for the next read.
    /// </summary>
    [TestMethod]
    [TestCategory("Integration")]
    public void ReadLogs_EndsAtTheCounterItSawFirst_EvenAcrossBatches()
    {
        RedisSession.EnsureReachable();
        string prefix = RedisSession.UniquePrefix();
        int count = RedisJournalProtocol.ReadLogsBatchLimit + 200;
        SeedRecords(prefix, count);
        var backend = Create(prefix);

        var read = new List<long>();
        using (var records = backend.ReadLogs(0).GetEnumerator())
        {
            for (int i = 0; i < RedisJournalProtocol.ReadLogsBatchLimit && records.MoveNext(); i++)
                read.Add((long)records.Current["op"]!);
            backend.AppendLogs(new[] { Log(9999, "late") });
            while (records.MoveNext())
                read.Add((long)records.Current["op"]!);
        }

        read.Should().Equal(Enumerable.Range(0, count).Select(i => (long)i));
        backend.ReadLogs(count).Select(l => (string)l["worker"]!).Should().Equal("late");
    }

    /// <summary>
    ///     A record missing in the middle of a batch (the counter covers it, a cluster-mode writer has not set it yet):
    ///     the reader returns the records before it, waits for it as Python's reader does, and goes on to the end.
    /// </summary>
    [TestMethod]
    [TestCategory("Integration")]
    public void ReadLogs_WaitsForARecordMissingInsideABatch_ThenFinishesTheRead()
    {
        RedisSession.EnsureReachable();
        string prefix = RedisSession.UniquePrefix();
        var db = RedisSession.RawConnection().GetDatabase();
        db.StringSet(RedisJournalProtocol.LogNumberKey(prefix), 3);
        db.StringSet(RedisJournalProtocol.LogKey(prefix, 0), "{\"op\": 0}");
        db.StringSet(RedisJournalProtocol.LogKey(prefix, 1), "{\"op\": 1}");
        db.StringSet(RedisJournalProtocol.LogKey(prefix, 3), "{\"op\": 3}");

        var backend = Create(prefix);
        var readTask = Task.Run(() => backend.ReadLogs(0).Select(l => (long)l["op"]!).ToList());
        Thread.Sleep(300);
        readTask.IsCompleted.Should().BeFalse("record 2 is covered by the counter, so the reader must wait for it");
        db.StringSet(RedisJournalProtocol.LogKey(prefix, 2), "{\"op\": 2}");

        readTask.Wait(TimeSpan.FromSeconds(15)).Should().BeTrue("ReadLogs must return once the missing record appears");
        readTask.Result.Should().Equal(0L, 1L, 2L, 3L);
    }

    /// <summary>
    ///     A standalone append seeds the counter inside its script, so a journal's first record is numbered 0 whether or
    ///     not the counter existed, also after the keys are deleted under a live backend.
    /// </summary>
    [TestMethod]
    [TestCategory("Integration")]
    public void Append_SeedsTheCounterInsideTheScript_EvenAfterTheJournalIsDeleted()
    {
        RedisSession.EnsureReachable();
        string prefix = RedisSession.UniquePrefix();
        var backend = Create(prefix);
        backend.AppendLogs(new[] { Log(0, "a") });
        var db = RedisSession.RawConnection().GetDatabase();
        db.KeyDelete(new RedisKey[] { RedisJournalProtocol.LogNumberKey(prefix), RedisJournalProtocol.LogKey(prefix, 0) });

        backend.AppendLogs(new[] { Log(1, "b") });

        ((string?)db.StringGet(RedisJournalProtocol.LogNumberKey(prefix))).Should().Be("0");
        backend.ReadLogs(0).Select(l => (string)l["worker"]!).Should().Equal("b");
    }

    // ----------------------------------------------------------------- snapshot keys (plan 10, item 2.2)

    /// <summary>OptunaSharp's snapshot goes to its own key; Python Optuna's key is never written.</summary>
    [TestMethod]
    [TestCategory("Integration")]
    public void SaveSnapshot_WritesTheOwnKey_AndNeverPythonsKey()
    {
        RedisSession.EnsureReachable();
        string prefix = RedisSession.UniquePrefix();
        Snapshot(prefix).SaveSnapshot(new byte[] { 0x7B, 1, 2 });

        var db = RedisSession.RawConnection().GetDatabase();
        ((byte[]?)db.StringGet(RedisJournalProtocol.OwnSnapshotKey(prefix))).Should().Equal(0x7B, 1, 2);
        db.KeyExists(RedisJournalProtocol.SnapshotKey(prefix)).Should().BeFalse();
    }

    /// <summary>
    ///     A snapshot an older OptunaSharp saved in Python's key (JSON, so it starts with <c>{</c>) is still found, once,
    ///     when the own key is empty.
    /// </summary>
    [TestMethod]
    [TestCategory("Integration")]
    public void LoadSnapshot_FindsAnOlderOwnSnapshotInPythonsKey()
    {
        RedisSession.EnsureReachable();
        string prefix = RedisSession.UniquePrefix();
        var legacy = System.Text.Encoding.UTF8.GetBytes("{\"format\":\"optunasharp-journal-snapshot-v1\"}");
        RedisSession.RawConnection().GetDatabase().StringSet(RedisJournalProtocol.SnapshotKey(prefix), legacy);

        Snapshot(prefix).LoadSnapshot().Should().Equal(legacy);
    }

    /// <summary>A Python pickle in Python's key (it starts with the byte 0x80) counts as no snapshot.</summary>
    [TestMethod]
    [TestCategory("Integration")]
    public void LoadSnapshot_IgnoresPythonsPickle()
    {
        RedisSession.EnsureReachable();
        string prefix = RedisSession.UniquePrefix();
        RedisSession.RawConnection().GetDatabase()
            .StringSet(RedisJournalProtocol.SnapshotKey(prefix), new byte[] { 0x80, 0x05, 0x95, 0x00 });

        Snapshot(prefix).LoadSnapshot().Should().BeNull();
    }

    /// <summary>When both keys hold a snapshot, the own key wins.</summary>
    [TestMethod]
    [TestCategory("Integration")]
    public void LoadSnapshot_PrefersTheOwnKey()
    {
        RedisSession.EnsureReachable();
        string prefix = RedisSession.UniquePrefix();
        var db = RedisSession.RawConnection().GetDatabase();
        db.StringSet(RedisJournalProtocol.SnapshotKey(prefix), "{\"old\":1}");
        db.StringSet(RedisJournalProtocol.OwnSnapshotKey(prefix), "{\"new\":2}");

        System.Text.Encoding.UTF8.GetString(Snapshot(prefix).LoadSnapshot()!).Should().Be("{\"new\":2}");
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

    /// <summary>
    ///     Every Redis backend states its snapshot interval per read mode: Optuna's 100 in both (the owner's choice of
    ///     2026-11-07, after the binary snapshot of plan 10, item 2.7; a standalone server used 1000 before). A journal
    ///     storage built without an interval takes the backend's; one built with an interval keeps its own.
    /// </summary>
    /// <param name="useCluster">The backend's read mode.</param>
    /// <param name="expected">The interval that mode should state.</param>
    [TestMethod]
    [TestCategory("Integration")]
    [DataRow(false, 100)]
    [DataRow(true, 100)]
    public void DefaultSnapshotInterval_FollowsTheReadMode_AndTheStorageTakesIt(bool useCluster, int expected)
    {
        RedisSession.EnsureReachable();
        var backend = Create(RedisSession.UniquePrefix(), useCluster);

        ((IJournalSnapshotInterval)backend).DefaultSnapshotInterval.Should().Be(expected);
        new JournalStorage(backend).SnapshotInterval.Should().Be(expected);
        new JournalStorage(backend, snapshotInterval: 7).SnapshotInterval.Should().Be(7);
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
