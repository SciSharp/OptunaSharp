using System;
using System.IO;
using AwesomeAssertions;
using OptunaSharp;
using OptunaSharp.Cli;
using OptunaSharp.Logging;
using OptunaSharp.Storages;
using OptunaSharp.Storages.Journal;

namespace OptunaSharp.Tests.Cli;

/// <summary>
///     Unit tests for the CLI's storage resolution (<see cref="StorageResolver" />, the port of
///     <c>_check_storage_url</c> + <c>_get_storage</c>). Covers the locally-resolvable routes — SQLite
///     `RDBStorage`, journal-file, the `--storage-class` hints, the `OPTUNA_STORAGE` fallback, and the
///     two error paths. (The `redis…` route is trivially correct by inspection and needs a live server,
///     so it is exercised by the integration/parity suites, not here.)
/// </summary>
[TestClass]
public class CliStorageResolverTests
{
    private static readonly Logger Log = OptunaLogging.GetLogger("test.cli.storage");

    private static (string Url, string Path) TempSqlite()
    {
        string path = Path.Combine(Path.GetTempPath(), $"osc-storage-{Guid.NewGuid():N}.db");
        return ("sqlite:///" + path.Replace("\\", "/"), path);
    }

    private static void Cleanup(BaseStorage storage, string? path = null)
    {
        (storage as IDisposable)?.Dispose();
        if (path is not null)
            try { File.Delete(path); } catch { /* best effort */ }
    }

    [TestMethod]
    public void SqliteUrl_ResolvesToRDBStorage()
    {
        var (url, path) = TempSqlite();
        BaseStorage storage = StorageResolver.GetStorage(url, storageClass: null, Log);
        try { storage.Should().BeOfType<RDBStorage>(); }
        finally { Cleanup(storage, path); }
    }

    [TestMethod]
    public void ExistingFile_ResolvesToJournalStorage()
    {
        string path = Path.Combine(Path.GetTempPath(), $"osc-journal-{Guid.NewGuid():N}.log");
        File.WriteAllText(path, string.Empty);
        try
        {
            BaseStorage storage = StorageResolver.GetStorage(path, storageClass: null, Log);
            storage.Should().BeOfType<JournalStorage>();
        }
        finally { try { File.Delete(path); } catch { /* best effort */ } }
    }

    [TestMethod]
    public void StorageClass_RDBStorage_ForcesRDB()
    {
        var (url, path) = TempSqlite();
        BaseStorage storage = StorageResolver.GetStorage(url, storageClass: "RDBStorage", Log);
        try { storage.Should().BeOfType<RDBStorage>(); }
        finally { Cleanup(storage, path); }
    }

    [TestMethod]
    public void StorageClass_JournalFileBackend_ForcesJournal()
    {
        // A non-existent path + the JournalFileBackend hint forces journal storage (no guessing).
        string path = Path.Combine(Path.GetTempPath(), $"osc-journal-{Guid.NewGuid():N}.log");
        try
        {
            BaseStorage storage = StorageResolver.GetStorage(path, storageClass: "JournalFileBackend", Log);
            storage.Should().BeOfType<JournalStorage>();
        }
        finally { try { File.Delete(path); } catch { /* best effort */ } }
    }

    [TestMethod]
    public void OptunaStorageEnvVar_UsedAsFallback()
    {
        var (url, path) = TempSqlite();
        string? saved = Environment.GetEnvironmentVariable("OPTUNA_STORAGE");
        Environment.SetEnvironmentVariable("OPTUNA_STORAGE", url);
        try
        {
            BaseStorage storage = StorageResolver.GetStorage(storageUrl: null, storageClass: null, Log);
            try { storage.Should().BeOfType<RDBStorage>(); }
            finally { Cleanup(storage, path); }
        }
        finally { Environment.SetEnvironmentVariable("OPTUNA_STORAGE", saved); }
    }

    [TestMethod]
    public void NoStorage_ThrowsCLIUsageError()
    {
        string? saved = Environment.GetEnvironmentVariable("OPTUNA_STORAGE");
        Environment.SetEnvironmentVariable("OPTUNA_STORAGE", null);
        try
        {
            Action act = () => StorageResolver.GetStorage(storageUrl: null, storageClass: null, Log);
            act.Should().Throw<CLIUsageError>().WithMessage("Storage URL is not specified.");
        }
        finally { Environment.SetEnvironmentVariable("OPTUNA_STORAGE", saved); }
    }

    [TestMethod]
    public void UnknownStorageClass_ThrowsCLIUsageError()
    {
        var (url, _) = TempSqlite();
        Action act = () => StorageResolver.GetStorage(url, storageClass: "NoSuchBackend", Log);
        act.Should().Throw<CLIUsageError>().WithMessage("Unsupported storage class");
    }

    [TestMethod]
    public void StorageClassChoices_MatchOptuna()
        => StorageResolver.StorageClassChoices.Should().BeEquivalentTo(
            new[] { "RDBStorage", "JournalFileBackend", "JournalRedisBackend", "JournalFileStorage", "JournalRedisStorage" });
}
