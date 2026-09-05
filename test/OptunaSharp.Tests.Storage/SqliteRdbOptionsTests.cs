using System;
using System.Collections.Generic;
using System.Data;
using System.IO;
using System.Linq;
using AwesomeAssertions;
using Microsoft.Data.Sqlite;
using NumSharp;
using OptunaSharp;
using OptunaSharp.Distributions;
using OptunaSharp.Samplers;
using OptunaSharp.Storages;

namespace OptunaSharp.Tests.Storage;

/// <summary>
///     Coverage for the "golden-standard" SQLite configuration surface (<see cref="SqliteRdbOptions" />
///     and the five <see cref="RDBStorage" /> constructors) — beyond Optuna's own API. Every
///     connection-string option and load-bearing <c>PRAGMA</c> is asserted to actually take effect on
///     the storage's own connection, plus the injected-connection ownership contract and the two
///     escape hatches.
/// </summary>
[TestClass]
public class SqliteRdbOptionsTests
{
    private readonly List<string> _tempFiles = new();

    [TestCleanup]
    public void Cleanup()
    {
        foreach (var f in _tempFiles)
        {
            foreach (var variant in new[] { f, f + "-wal", f + "-shm", f + "-journal" })
            {
                try
                {
                    if (File.Exists(variant))
                        File.Delete(variant);
                }
                catch (IOException)
                {
                }
            }
        }
    }

    private string TempDbPath()
    {
        string path = Path.Combine(Path.GetTempPath(), $"optunasharp-opt-{Guid.NewGuid():N}.db");
        _tempFiles.Add(path);
        return path;
    }

    private string TempFileUrl() => $"sqlite:///{TempDbPath().Replace('\\', '/')}";

    private const string MemoryUrl = "sqlite:///:memory:";

    private static string? Pragma(SqliteConnection conn, string name)
    {
        using var cmd = conn.CreateCommand();
        cmd.CommandText = "PRAGMA " + name;
        object? value = cmd.ExecuteScalar();
        return value is null or DBNull ? null : Convert.ToString(value, System.Globalization.CultureInfo.InvariantCulture);
    }

    private static void SmokeCheck(RDBStorage storage)
    {
        int sid = storage.CreateNewStudy(new[] { StudyDirection.Minimize }, Guid.NewGuid().ToString("N"));
        int tid = storage.CreateNewTrial(sid);
        storage.SetTrialParam(tid, "x", 1.5, new FloatDistribution(-10, 10));
        storage.SetTrialStateValues(tid, TrialState.Complete, new[] { 2.25 });
        storage.GetTrial(tid).Value.Should().Be(2.25);
    }

    // ------------------------------------------------------------------ the five constructors

    [TestMethod]
    public void Ctor_Url_Works()
    {
        using var s = new RDBStorage(MemoryUrl);
        SmokeCheck(s);
    }

    [TestMethod]
    public void Ctor_Url_WithConnectionStringAction_Works_AndActionRuns()
    {
        using var s = new RDBStorage(MemoryUrl, b => b.DefaultTimeout = 45);
        s.Connection.DefaultTimeout.Should().Be(45);
        SmokeCheck(s);
    }

    [TestMethod]
    public void Ctor_Url_WithConnectionStringAction_PropagatesExceptions()
    {
        Action act = () => new RDBStorage(MemoryUrl, _ => throw new InvalidOperationException("ran"));
        act.Should().Throw<InvalidOperationException>().WithMessage("ran");
    }

    [TestMethod]
    public void Ctor_Url_WithOptions_Works()
    {
        using var s = new RDBStorage(MemoryUrl, new SqliteRdbOptions { CacheSize = -4000 });
        Pragma(s.Connection, "cache_size").Should().Be("-4000");
        SmokeCheck(s);
    }

    [TestMethod]
    public void Ctor_ConnectionStringBuilder_Works_AndOptionsLayerOnTop()
    {
        var builder = new SqliteConnectionStringBuilder { DataSource = ":memory:" };
        using var s = new RDBStorage(builder, new SqliteRdbOptions { BusyTimeoutMs = 4321 });
        Pragma(s.Connection, "busy_timeout").Should().Be("4321");
        SmokeCheck(s);
    }

    [TestMethod]
    public void Ctor_InjectedConnection_Works()
    {
        using var conn = new SqliteConnection("Data Source=:memory:");
        conn.Open();
        using var s = new RDBStorage(conn, new SqliteRdbOptions { Synchronous = SqliteSynchronous.Normal }, ownsConnection: false);
        Pragma(conn, "synchronous").Should().Be("1");
        SmokeCheck(s);
    }

    [TestMethod]
    public void Ctor_InjectedConnection_OpensWhenClosed()
    {
        using var conn = new SqliteConnection("Data Source=:memory:"); // not opened
        conn.State.Should().Be(ConnectionState.Closed);
        using var s = new RDBStorage(conn);
        conn.State.Should().Be(ConnectionState.Open);
        SmokeCheck(s);
    }

    // ------------------------------------------------------------------ injected-connection ownership

    [TestMethod]
    public void InjectedConnection_NotOwned_SurvivesStorageDispose()
    {
        using var conn = new SqliteConnection("Data Source=:memory:");
        conn.Open();
        var s = new RDBStorage(conn, ownsConnection: false);
        int sid = s.CreateNewStudy(new[] { StudyDirection.Minimize }, "s");
        s.Dispose();

        conn.State.Should().Be(ConnectionState.Open, "the caller retains ownership");
        // The in-memory database is still alive on the same connection.
        Pragma(conn, "busy_timeout").Should().NotBeNull();
        using var reopened = new RDBStorage(conn, new SqliteRdbOptions { SkipTableCreation = true }, ownsConnection: false);
        reopened.GetStudyNameFromId(sid).Should().Be("s");
    }

    [TestMethod]
    public void InjectedConnection_Owned_IsDisposedWithStorage()
    {
        var conn = new SqliteConnection("Data Source=:memory:");
        conn.Open();
        var s = new RDBStorage(conn, ownsConnection: true);
        s.Dispose();
        conn.State.Should().Be(ConnectionState.Closed, "ownsConnection: true hands the lifetime to the storage");
    }

    // ------------------------------------------------------------------ per-connection PRAGMAs

    [TestMethod]
    public void BusyTimeout_DefaultsTo30s_OnOptunaCtor()
    {
        using var s = new RDBStorage(MemoryUrl);
        Pragma(s.Connection, "busy_timeout").Should().Be("30000");
    }

    [TestMethod]
    public void BusyTimeout_Null_LeavesSqliteDefault()
    {
        using var s = new RDBStorage(MemoryUrl, new SqliteRdbOptions { BusyTimeoutMs = null });
        Pragma(s.Connection, "busy_timeout").Should().Be("0");
    }

    [TestMethod]
    public void BusyTimeout_Custom_IsApplied()
    {
        using var s = new RDBStorage(MemoryUrl, new SqliteRdbOptions { BusyTimeoutMs = 12345 });
        Pragma(s.Connection, "busy_timeout").Should().Be("12345");
    }

    [TestMethod]
    [DataRow(SqliteSynchronous.Off, "0")]
    [DataRow(SqliteSynchronous.Normal, "1")]
    [DataRow(SqliteSynchronous.Full, "2")]
    [DataRow(SqliteSynchronous.Extra, "3")]
    public void Synchronous_IsApplied(SqliteSynchronous mode, string expected)
    {
        using var s = new RDBStorage(MemoryUrl, new SqliteRdbOptions { Synchronous = mode });
        Pragma(s.Connection, "synchronous").Should().Be(expected);
    }

    [TestMethod]
    [DataRow(SqliteTempStore.Default, "0")]
    [DataRow(SqliteTempStore.File, "1")]
    [DataRow(SqliteTempStore.Memory, "2")]
    public void TempStore_IsApplied(SqliteTempStore store, string expected)
    {
        using var s = new RDBStorage(MemoryUrl, new SqliteRdbOptions { TempStore = store });
        Pragma(s.Connection, "temp_store").Should().Be(expected);
    }

    [TestMethod]
    public void CacheSize_IsApplied()
    {
        using var s = new RDBStorage(MemoryUrl, new SqliteRdbOptions { CacheSize = -8000 });
        Pragma(s.Connection, "cache_size").Should().Be("-8000");
    }

    [TestMethod]
    public void SecureDelete_IsApplied()
    {
        using var s = new RDBStorage(MemoryUrl, new SqliteRdbOptions { SecureDelete = true });
        Pragma(s.Connection, "secure_delete").Should().Be("1");
    }

    [TestMethod]
    public void ForeignKeys_On_IsApplied_AndFullLifecycleWorks()
    {
        using var s = new RDBStorage(MemoryUrl, new SqliteRdbOptions { ForeignKeys = true });
        Pragma(s.Connection, "foreign_keys").Should().Be("1");

        // The whole create -> child rows -> delete cascade must satisfy DB-level FK enforcement.
        int sid = s.CreateNewStudy(new[] { StudyDirection.Minimize }, "s");
        int tid = s.CreateNewTrial(sid);
        s.SetTrialParam(tid, "x", 1.0, new FloatDistribution(-10, 10));
        s.SetTrialUserAttr(tid, "k", "v");
        s.SetTrialIntermediateValue(tid, 0, 0.5);
        s.SetTrialStateValues(tid, TrialState.Complete, new[] { 1.0 });
        s.DeleteStudy(sid);
        s.GetAllStudies().Should().BeEmpty();
    }

    [TestMethod]
    public void ForeignKeys_Off_ByDefault()
    {
        using var s = new RDBStorage(MemoryUrl);
        Pragma(s.Connection, "foreign_keys").Should().Be("0", "Optuna leaves foreign key enforcement off");
    }

    // ------------------------------------------------------------------ persistent (fresh-db) PRAGMAs

    [TestMethod]
    public void JournalMode_Wal_IsApplied_OnFileDatabase()
    {
        using var s = new RDBStorage(TempFileUrl(), new SqliteRdbOptions { JournalMode = SqliteJournalMode.Wal });
        Pragma(s.Connection, "journal_mode").Should().Be("wal");
        SmokeCheck(s);
    }

    [TestMethod]
    [DataRow(SqliteJournalMode.Delete, "delete")]
    [DataRow(SqliteJournalMode.Truncate, "truncate")]
    [DataRow(SqliteJournalMode.Persist, "persist")]
    [DataRow(SqliteJournalMode.Memory, "memory")]
    public void JournalMode_Variants_AreApplied_OnFileDatabase(SqliteJournalMode mode, string expected)
    {
        using var s = new RDBStorage(TempFileUrl(), new SqliteRdbOptions { JournalMode = mode });
        Pragma(s.Connection, "journal_mode").Should().Be(expected);
    }

    [TestMethod]
    public void PageSize_IsApplied_OnFreshFileDatabase()
    {
        using var s = new RDBStorage(TempFileUrl(), new SqliteRdbOptions { PageSize = 8192 });
        Pragma(s.Connection, "page_size").Should().Be("8192");
        SmokeCheck(s);
    }

    [TestMethod]
    public void AutoVacuum_IsApplied_OnFreshFileDatabase()
    {
        using var s = new RDBStorage(TempFileUrl(), new SqliteRdbOptions { AutoVacuum = SqliteAutoVacuum.Full });
        Pragma(s.Connection, "auto_vacuum").Should().Be("1");
    }

    [TestMethod]
    public void MmapSize_IsApplied_OnFileDatabase()
    {
        using var s = new RDBStorage(TempFileUrl(), new SqliteRdbOptions { MmapSize = 262144 });
        Pragma(s.Connection, "mmap_size").Should().Be("262144");
    }

    // ------------------------------------------------------------------ connection-string options

    [TestMethod]
    public void DefaultTimeoutSeconds_IsApplied()
    {
        using var s = new RDBStorage(MemoryUrl, new SqliteRdbOptions { DefaultTimeoutSeconds = 17 });
        s.Connection.DefaultTimeout.Should().Be(17);
    }

    [TestMethod]
    public void Cache_And_Pooling_And_RecursiveTriggers_FlowToConnectionString()
    {
        using var s = new RDBStorage(MemoryUrl, new SqliteRdbOptions
        {
            Cache = SqliteCacheMode.Shared,
            Pooling = false,
            RecursiveTriggers = true,
        });
        var csb = new SqliteConnectionStringBuilder(s.Connection.ConnectionString);
        csb.Cache.Should().Be(SqliteCacheMode.Shared);
        csb.Pooling.Should().BeFalse();
        csb.RecursiveTriggers.Should().BeTrue();
        SmokeCheck(s);
    }

    [TestMethod]
    public void Mode_ReadOnly_IsApplied_OnExistingDatabase()
    {
        string url = TempFileUrl();
        using (var writer = new RDBStorage(url))
            writer.CreateNewStudy(new[] { StudyDirection.Minimize }, "s");

        using var ro = new RDBStorage(url, new SqliteRdbOptions
        {
            Mode = SqliteOpenMode.ReadOnly,
            SkipTableCreation = true,
        });
        ro.GetAllStudies().Should().ContainSingle(); // reads still work
        Action write = () => ro.CreateNewStudy(new[] { StudyDirection.Minimize }, "s2");
        write.Should().Throw<SqliteException>("the database was opened read-only");
    }

    [TestMethod]
    public void MemoryUrl_DefaultsToMemoryMode()
    {
        using var s = new RDBStorage(MemoryUrl);
        new SqliteConnectionStringBuilder(s.Connection.ConnectionString).Mode.Should().Be(SqliteOpenMode.Memory);
    }

    // ------------------------------------------------------------------ escape hatches

    [TestMethod]
    public void AdditionalPragmas_AreApplied()
    {
        using var s = new RDBStorage(TempFileUrl(), new SqliteRdbOptions
        {
            JournalMode = SqliteJournalMode.Wal,
            AdditionalPragmas = new[] { "wal_autocheckpoint=250" },
        });
        Pragma(s.Connection, "wal_autocheckpoint").Should().Be("250");
    }

    [TestMethod]
    public void ConfigureConnection_EscapeHatch_Runs()
    {
        using var s = new RDBStorage(TempFileUrl(), new SqliteRdbOptions
        {
            ConfigureConnection = c =>
            {
                using var cmd = c.CreateCommand();
                cmd.CommandText = "PRAGMA application_id=987654";
                cmd.ExecuteNonQuery();
            },
        });
        Pragma(s.Connection, "application_id").Should().Be("987654");
    }

    // ------------------------------------------------------------------ Optuna knobs via options

    [TestMethod]
    public void Options_SkipTableCreation_AssumesExistingSchema()
    {
        string url = TempFileUrl();
        using (var creator = new RDBStorage(url))
            creator.CreateNewStudy(new[] { StudyDirection.Minimize }, "s");

        using var s = new RDBStorage(url, new SqliteRdbOptions { SkipTableCreation = true });
        s.GetAllStudies().Should().ContainSingle();
    }

    [TestMethod]
    public void Options_HeartbeatValidation_StillThrows()
    {
        Action badInterval = () => new RDBStorage(MemoryUrl, new SqliteRdbOptions { HeartbeatInterval = 0 });
        badInterval.Should().Throw<ValueError>();

        Action badGrace = () => new RDBStorage(MemoryUrl, new SqliteRdbOptions { HeartbeatInterval = 5, GracePeriod = -1 });
        badGrace.Should().Throw<ValueError>();
    }

    [TestMethod]
    public void Options_Heartbeat_IsExposed()
    {
        using var s = new RDBStorage(MemoryUrl, new SqliteRdbOptions { HeartbeatInterval = 9 });
        ((IBaseHeartbeat)s).GetHeartbeatInterval().Should().Be(9);
    }

    // ------------------------------------------------------------------ everything at once

    [TestMethod]
    public void HeavilyConfigured_FileStorage_OptimizesEndToEnd()
    {
        var options = new SqliteRdbOptions
        {
            JournalMode = SqliteJournalMode.Wal,
            Synchronous = SqliteSynchronous.Normal,
            ForeignKeys = true,
            BusyTimeoutMs = 15000,
            CacheSize = -16000,
            MmapSize = 268435456,
            TempStore = SqliteTempStore.Memory,
            PageSize = 4096,
            SecureDelete = false,
            AdditionalPragmas = new[] { "wal_autocheckpoint=1000" },
        };
        using var storage = new RDBStorage(TempFileUrl(), options);

        Pragma(storage.Connection, "journal_mode").Should().Be("wal");
        Pragma(storage.Connection, "synchronous").Should().Be("1");
        Pragma(storage.Connection, "foreign_keys").Should().Be("1");
        Pragma(storage.Connection, "busy_timeout").Should().Be("15000");
        Pragma(storage.Connection, "page_size").Should().Be("4096");

        var study = Optuna.CreateStudy(storage: storage, studyName: "quad", sampler: new RandomSampler(seed: 3));
        study.Optimize(t => { double x = t.SuggestFloat("x", -10, 10); return x * x; }, nTrials: 15);
        study.Trials.Should().HaveCount(15);
        study.BestValue.Should().BeLessThanOrEqualTo(study.Trials.Min(t => t.Value!.Value) + 1e-12);
    }
}
