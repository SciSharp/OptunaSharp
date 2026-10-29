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

    /// <summary>
    ///     A bare path that is not an existing file, or a URL of a dialect SQLAlchemy does not know, fails
    ///     <c>RDBStorage(url)</c> with SQLAlchemy's <c>ArgumentError</c>, which <c>_get_storage</c> turns into its usage
    ///     error. The CLI used to open such a path as a SQLite file instead.
    /// </summary>
    /// <param name="url">The storage string.</param>
    [TestMethod]
    [DataRow("no-such-study.db")]
    [DataRow("foo://user@host/db")]
    public void UnguessableStorage_ThrowsCLIUsageError(string url)
    {
        Action act = () => StorageResolver.GetStorage(url, storageClass: null, Log);
        act.Should().Throw<CLIUsageError>().WithMessage("Failed to guess storage class from storage_url");
    }

    /// <summary>
    ///     Only SQLAlchemy's <c>ArgumentError</c> becomes the usage error; any other failure propagates as it does from
    ///     Python's <c>_get_storage</c> (here Python's <c>float()</c> error for a bad <c>?timeout=</c>), and with
    ///     <c>--storage-class RDBStorage</c> nothing is caught at all.
    /// </summary>
    [TestMethod]
    public void OtherStorageErrors_Propagate()
    {
        var (url, _) = TempSqlite();
        Action timeout = () => StorageResolver.GetStorage(url + "?timeout=abc", storageClass: null, Log);
        timeout.Should().Throw<NumSharp.ValueError>().WithMessage("could not convert string to float: 'abc'");
        Action forced = () => StorageResolver.GetStorage("no-such-study.db", storageClass: "RDBStorage", Log);
        forced.Should().Throw<ArgumentException>().WithMessage("Could not parse SQLAlchemy URL from given URL string");
    }

    /// <summary>
    ///     A PostgreSQL URL picks the EF Core package, which the tool does not ship, and fails with Optuna's
    ///     <c>ImportError</c> text, as Python's CLI does without a PostgreSQL driver.
    /// </summary>
    [TestMethod]
    public void ServerUrl_WithoutItsPackage_RaisesOptunasImportError()
    {
        Action act = () => StorageResolver.GetStorage("postgresql://u:p@127.0.0.1:1/optuna", storageClass: null, Log);
        act.Should().Throw<NotSupportedException>()
            .WithMessage("Failed to import DB access module for the specified storage URL. Please install appropriate one.*");
    }
}
