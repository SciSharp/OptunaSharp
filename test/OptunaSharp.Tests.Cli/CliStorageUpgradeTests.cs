using System;
using System.Collections.Generic;
using System.IO;
using AwesomeAssertions;
using OptunaSharp;
using OptunaSharp.Cli;
using OptunaSharp.Storages;

namespace OptunaSharp.Tests.Cli;

/// <summary>
///     Functional tests for the <c>optunasharp storage upgrade</c> command — the port of Optuna's
///     <c>_StorageUpgrade</c> now backed by the real alembic migration chain (Phase 4.1's deferred
///     sub-item). The command opens the RDB with the compatibility check and table creation skipped
///     (exactly as Optuna's CLI does), replays the migrations to head, and returns 0. Byte-for-byte
///     equivalence with Python's own <c>storage upgrade</c> is asserted in the interop suite; here we
///     assert the CLI wiring and end result.
/// </summary>
[TestClass]
public class CliStorageUpgradeTests
{
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

    [TestMethod]
    public void Upgrade_UpToDateStorage_Returns0_AndStaysAtHead()
    {
        string url = FreshDbUrl(out string path);
        using (var storage = new RDBStorage(url))
            storage.CreateNewStudy(new[] { StudyDirection.Minimize }, "s");

        CliProgram.Run(new[] { "storage", "upgrade", "--storage", url }).Should().Be(0);

        using var reopened = new RDBStorage(url);
        reopened.GetCurrentVersion().Should().Be("v3.2.0.a");
        // Idempotent: a second upgrade is still a no-op that returns 0.
        CliProgram.Run(new[] { "storage", "upgrade", "--storage", url }).Should().Be(0);
    }

    [TestMethod]
    public void Upgrade_OldDatabase_MigratesToHead_Returns0_AndBecomesReadable()
    {
        string url = CopyAsset("0.9.0.a", out string path);

        CliProgram.Run(new[] { "storage", "upgrade", "--storage", url }).Should().Be(0);

        // The normal constructor's compatibility check passes only if the migration truly reached head.
        using var reopened = new RDBStorage(url);
        reopened.GetCurrentVersion().Should().Be("v3.2.0.a");
        Optuna.LoadStudy("single", reopened).Trials.Should().HaveCount(1);
    }

    [TestMethod]
    public void Upgrade_WithoutStorageUrl_Returns1()
    {
        // `_check_storage_url` raises CLIUsageError when neither --storage nor OPTUNA_STORAGE is given;
        // the command handler maps that to a non-zero exit. Guarded so an ambient OPTUNA_STORAGE (which
        // would make the URL resolvable) doesn't turn this into a false failure.
        if (!string.IsNullOrEmpty(Environment.GetEnvironmentVariable("OPTUNA_STORAGE")))
            Assert.Inconclusive("OPTUNA_STORAGE is set in the environment; the no-URL path can't be exercised.");

        CliProgram.Run(new[] { "storage", "upgrade" }).Should().Be(1);
    }

    // ------------------------------------------------------------------ helpers

    private string FreshDbUrl(out string path)
    {
        path = Path.Combine(Path.GetTempPath(), $"optunasharp-cli-upg-{Guid.NewGuid():N}.db");
        _tempFiles.Add(path);
        return "sqlite:///" + path.Replace('\\', '/');
    }

    private string CopyAsset(string version, out string path)
    {
        string src = Path.Combine(UpgradeAssetsDir(), $"{version}.db");
        File.Exists(src).Should().BeTrue($"the committed upgrade asset {version}.db must exist");
        path = Path.Combine(Path.GetTempPath(), $"optunasharp-cli-upg-{version}-{Guid.NewGuid():N}.db");
        File.Copy(src, path);
        _tempFiles.Add(path);
        return "sqlite:///" + path.Replace('\\', '/');
    }

    private static string UpgradeAssetsDir()
    {
        string? dir = AppContext.BaseDirectory;
        while (dir is not null && !File.Exists(Path.Combine(dir, "SciSharp.OptunaSharp.sln")))
            dir = Path.GetDirectoryName(dir);
        if (dir is null)
            throw new DirectoryNotFoundException("Could not locate the repository root (SciSharp.OptunaSharp.sln).");
        return Path.Combine(
            dir, "src", "optuna", "tests", "storages_tests", "rdb_tests", "test_upgrade_assets");
    }
}
