using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using AwesomeAssertions;
using NumSharp;
using OptunaSharp;
using OptunaSharp.Distributions;
using OptunaSharp.Storages;

namespace OptunaSharp.Tests.Storage;

/// <summary>
///     Schema-migration tests for the SQLite <see cref="RDBStorage" /> (Phase 4.1 deferred sub-item:
///     the alembic upgrade chain). These are the OptunaSharp analogue of Optuna's own
///     <c>tests/storages_tests/rdb_tests/test_storage.py</c> upgrade tests: they replay each committed
///     per-version database asset (<c>refs/optuna/.../test_upgrade_assets/{version}.db</c>) through
///     <see cref="RDBStorage.Upgrade" /> and assert it lands at head and reads back correctly. No
///     Python is needed — the assets are checked into the vendored Optuna submodule, and
///     <c>Microsoft.Data.Sqlite</c> bundles the engine.
///
///     <para>The byte-for-byte equivalence of the migrated database with what <c>alembic upgrade</c>
///     produces is asserted separately, against live Python, in the interop suite.</para>
/// </summary>
[TestClass]
public class SqliteRdbUpgradeTests
{
    // Every migration revision, matching Optuna's own parametrization of the upgrade tests.
    private const string HeadVersion = "v3.2.0.a";

    private readonly List<string> _tempFiles = new();

    [TestCleanup]
    public void Cleanup()
    {
        foreach (var f in _tempFiles)
        {
            try
            {
                if (File.Exists(f))
                    File.Delete(f);
            }
            catch (IOException)
            {
            }
        }
    }

    // ------------------------------------------------------------------ version-manager surface

    [TestMethod]
    public void FreshDatabase_IsAtHead_AndKnowsEveryVersion()
    {
        using var storage = new RDBStorage("sqlite:///:memory:");
        storage.GetCurrentVersion().Should().Be(HeadVersion);
        storage.GetHeadVersion().Should().Be(HeadVersion);
        // Python `get_all_versions` walks revisions head -> base.
        storage.GetAllVersions().Should().Equal(
            "v3.2.0.a", "v3.0.0.d", "v3.0.0.c", "v3.0.0.b", "v3.0.0.a",
            "v2.6.0.a", "v2.4.0.a", "v1.3.0.a", "v1.2.0.a", "v0.9.0.a");
    }

    [TestMethod]
    public void Upgrade_OnUpToDateStorage_IsANoOp()
    {
        // Optuna `test_upgrade_identity`: upgrade() leaves an already-current storage untouched.
        using var storage = new RDBStorage("sqlite:///:memory:");
        string before = storage.GetCurrentVersion();
        storage.Upgrade();
        storage.GetCurrentVersion().Should().Be(before).And.Be(HeadVersion);
    }

    // ------------------------------------------------------------------ the migration chain

    [DataTestMethod]
    [DataRow("0.9.0.a")]
    [DataRow("1.2.0.a")]
    [DataRow("1.3.0.a")]
    [DataRow("2.4.0.a")]
    [DataRow("2.6.0.a")]
    [DataRow("3.0.0.a")]
    [DataRow("3.0.0.b")]
    [DataRow("3.0.0.c")]
    [DataRow("3.0.0.d")]
    [DataRow("3.2.0.a")]
    public void Upgrade_SingleObjective_LandsAtHeadAndReadsBack(string version)
    {
        string url = CopyAsset(version);

        using (var storage = new RDBStorage(url, skipCompatibilityCheck: true, skipTableCreation: true))
        {
            storage.GetCurrentVersion().Should().Be($"v{version}");
            storage.Upgrade();
            storage.GetCurrentVersion().Should().Be(HeadVersion);
        }

        // Reopening with the normal constructor exercises the compatibility check: it must pass now.
        using var reopened = new RDBStorage(url);

        // The empty study has no trials; running is out of scope (no sampler here) — just read it.
        Optuna.LoadStudy("single_empty", reopened).Trials.Should().BeEmpty();

        var single = Optuna.LoadStudy("single", reopened);
        single.Trials.Should().HaveCount(1);
        var trial = single.Trials[0];
        ((long)trial.UserAttrs["b"]!).Should().Be(1);
        trial.IntermediateValues[0].Should().Be(0.5);
        ((double)trial.Params["x"]!).Should().BeInRange(-5, 5);
        ((long)trial.Params["y"]!).Should().BeInRange(0, 10);
        trial.Params["z"].Should().BeOneOf(-5L, 0L, 5L);
        trial.Value.Should().NotBeNull();
        trial.Value!.Value.Should().BeInRange(0, 150);
        ((long)single.UserAttrs["d"]!).Should().Be(3);
    }

    [DataTestMethod]
    [DataRow("2.4.0.a")]
    [DataRow("2.6.0.a")]
    [DataRow("3.0.0.a")]
    [DataRow("3.0.0.b")]
    [DataRow("3.0.0.c")]
    [DataRow("3.0.0.d")]
    [DataRow("3.2.0.a")]
    public void Upgrade_MultiObjective_ReadsBack(string version)
    {
        string url = CopyAsset(version);
        using (var storage = new RDBStorage(url, skipCompatibilityCheck: true, skipTableCreation: true))
            storage.Upgrade();

        using var reopened = new RDBStorage(url);

        Optuna.LoadStudy("multi_empty", reopened).Trials.Should().BeEmpty();

        var multi = Optuna.LoadStudy("multi", reopened);
        multi.Directions.Should().Equal(StudyDirection.Minimize, StudyDirection.Minimize);
        multi.Trials.Should().HaveCount(1);
        var trial = multi.Trials[0];
        var values = trial.Values;
        values.Should().NotBeNull();
        values!.Should().HaveCount(2);
        values[0].Should().BeInRange(-5, 5);
        values[1].Should().BeInRange(0, 150);
        ((long)trial.UserAttrs["b"]!).Should().Be(1);
    }

    [DataTestMethod]
    [DataRow("2.4.0.a")]
    [DataRow("2.6.0.a")]
    [DataRow("3.0.0.a")]
    [DataRow("3.0.0.b")]
    [DataRow("3.0.0.c")]
    [DataRow("3.0.0.d")]
    [DataRow("3.2.0.a")]
    public void Upgrade_Distributions_AreConvertedToUnifiedForms(string version)
    {
        string url = CopyAsset(version);
        using (var storage = new RDBStorage(url, skipCompatibilityCheck: true, skipTableCreation: true))
            storage.Upgrade();

        using var reopened = new RDBStorage(url);
        var study = Optuna.LoadStudy("schema migration", reopened);
        var distributions = study.Trials[0].Distributions;

        // The deprecated Uniform/LogUniform/DiscreteUniform/IntUniform/IntLogUniform classes must have
        // been migrated to the unified {Float,Int}Distribution forms (Optuna's `migrate_new_distribution`).
        distributions["x1"].Should().BeOfType<FloatDistribution>();
        distributions["x2"].Should().BeOfType<FloatDistribution>();
        ((FloatDistribution)distributions["x2"]).Log.Should().BeTrue();
        distributions["x3"].Should().BeOfType<FloatDistribution>();
        ((FloatDistribution)distributions["x3"]).Step.Should().Be(2);
        distributions["y1"].Should().BeOfType<IntDistribution>();
        distributions["y2"].Should().BeOfType<IntDistribution>();
        ((IntDistribution)distributions["y2"]).Log.Should().BeTrue();
        distributions["z"].Should().BeOfType<CategoricalDistribution>();
    }

    // ------------------------------------------------------------------ compatibility gate

    [TestMethod]
    public void OpeningAnOldDatabaseNormally_RaisesAndDirectsToUpgrade()
    {
        // The InitAlembic fix: a pre-alembic database is stamped at the base revision (not head), so the
        // compatibility check correctly sees it as out of date and points at `storage upgrade`.
        string url = CopyAsset("0.9.0.a");
        Action open = () =>
        {
            using var _ = new RDBStorage(url);
        };
        open.Should().Throw<RuntimeError>()
            .WithMessage("*optuna storage upgrade*");
    }

    // ------------------------------------------------------------------ helpers

    private string CopyAsset(string version)
    {
        string src = Path.Combine(UpgradeAssetsDir(), $"{version}.db");
        File.Exists(src).Should().BeTrue($"the committed upgrade asset {version}.db must exist");
        string dst = Path.Combine(Path.GetTempPath(), $"optunasharp-upgrade-{version}-{Guid.NewGuid():N}.db");
        File.Copy(src, dst);
        _tempFiles.Add(dst);
        return "sqlite:///" + dst.Replace('\\', '/');
    }

    /// <summary>
    ///     Locate the committed Optuna upgrade-test assets by walking up from the test binary to the
    ///     repository root (the folder holding the solution), then into the vendored Optuna submodule.
    /// </summary>
    internal static string UpgradeAssetsDir()
    {
        string? dir = AppContext.BaseDirectory;
        while (dir is not null && !File.Exists(Path.Combine(dir, "SciSharp.OptunaSharp.sln")))
            dir = Path.GetDirectoryName(dir);
        if (dir is null)
            throw new DirectoryNotFoundException("Could not locate the repository root (SciSharp.OptunaSharp.sln).");
        return Path.Combine(
            dir, "refs", "optuna", "tests", "storages_tests", "rdb_tests", "test_upgrade_assets");
    }
}
