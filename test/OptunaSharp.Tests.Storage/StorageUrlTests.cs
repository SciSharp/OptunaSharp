using System;
using System.IO;
using System.Linq;
using AwesomeAssertions;
using Microsoft.Data.Sqlite;
using Microsoft.Extensions.DependencyInjection;
using OptunaSharp;
using OptunaSharp.Extensions;
using OptunaSharp.Samplers;
using OptunaSharp.Storages;

namespace OptunaSharp.Tests.Storage;

/// <summary>
///     Review item H7 with the storage packages referenced: Core's <see cref="StorageFunctions.GetStorage(string)" />
///     and <c>AddOptunaStorage(string)</c> open the package that serves a URL's dialect. Before the fix the first
///     threw for every URL and the second registered a factory that threw on every resolution. The SQLite cases
///     run everywhere; the PostgreSQL and MySQL cases are in the EF Core suite, against live servers.
/// </summary>
[TestClass]
public class StorageUrlTests
{
    private string _path = null!;

    /// <summary>The test's SQLite URL, with forward slashes as SQLAlchemy writes Windows paths.</summary>
    private string Url => "sqlite:///" + _path.Replace('\\', '/');

    /// <summary>A fresh database path per test.</summary>
    [TestInitialize]
    public void SetUp() => _path = Path.Combine(Path.GetTempPath(), $"optunasharp-url-{Guid.NewGuid():N}.db");

    /// <summary>Deletes the test's database and its journal files.</summary>
    [TestCleanup]
    public void TearDown()
    {
        // A disposed storage's connection goes back to Microsoft.Data.Sqlite's pool, which keeps the file
        // open on Windows until the pool is cleared.
        SqliteConnection.ClearAllPools();
        foreach (string file in Directory.GetFiles(Path.GetTempPath(), Path.GetFileName(_path) + "*"))
            File.Delete(file);
    }

    /// <summary>A one-parameter objective.</summary>
    /// <param name="trial">The trial to sample.</param>
    /// <returns>The sampled value.</returns>
    private static double Objective(Trial trial) => trial.SuggestFloat("x", 0, 1);

    /// <summary>
    ///     Python's <c>get_storage("sqlite:///…")</c> returns <c>_CachedStorage(RDBStorage)</c>; the port returns a
    ///     <see cref="CachedStorage" /> over the SQLite package's <see cref="RDBStorage" />, and a study written
    ///     through it is in the file afterwards.
    /// </summary>
    [TestMethod]
    public void GetStorage_SqliteUrl_IsACacheOverTheSqlitePackagesStorage_LikePython()
    {
        BaseStorage storage = StorageFunctions.GetStorage(Url);
        RDBStorage backend = storage.Should().BeOfType<CachedStorage>()
            .Which.Backend.Should().BeOfType<RDBStorage>().Which;
        try
        {
            Optuna.CreateStudy(storage: storage, studyName: "url", sampler: new RandomSampler(0))
                .Optimize(Objective, nTrials: 3);
        }
        finally
        {
            // The cache does not own the storage, as in Python: the caller disposes it.
            backend.Dispose();
        }

        using var reopened = new RDBStorage(Url);
        Optuna.LoadStudy("url", reopened).Trials.Should().HaveCount(3);
    }

    /// <summary>
    ///     SQLAlchemy accepts a driver after the dialect, and Python opens <c>sqlite+pysqlite:///study.db</c> like
    ///     <c>sqlite:///study.db</c>. The SQLite package used to treat the whole URL as a file name; it now drops
    ///     the driver and opens the same file.
    /// </summary>
    [TestMethod]
    public void GetStorage_SqliteUrlWithADriver_OpensTheSameFile()
    {
        using (var plain = new RDBStorage(Url))
            Optuna.CreateStudy(storage: plain, studyName: "driver", sampler: new RandomSampler(0))
                .Optimize(Objective, nTrials: 2);

        BaseStorage storage = StorageFunctions.GetStorage("sqlite+pysqlite:///" + _path.Replace('\\', '/'));
        try
        {
            Optuna.LoadStudy("driver", storage).Trials.Should().HaveCount(2);
        }
        finally
        {
            ((IDisposable)((CachedStorage)storage).Backend).Dispose();
        }
    }

    /// <summary>
    ///     The H7 repro, fixed: resolving after <c>AddOptunaStorage("sqlite:///…")</c> threw NotSupportedException.
    ///     The registration is now the one <c>AddSqliteStorage(url)</c> makes: the concrete storage and
    ///     <see cref="BaseStorage" /> resolve to one instance, studies built by the factory persist to the file, and
    ///     disposing the container disposes the storage and closes its connection.
    /// </summary>
    [TestMethod]
    public void AddOptunaStorage_SqliteUrl_RegistersWhatAddSqliteStorageRegisters()
    {
        RDBStorage registered;
        using (ServiceProvider provider = new ServiceCollection()
                   .AddOptunaStorage(Url)
                   .AddOptunaStudyFactory()
                   .BuildServiceProvider())
        {
            BaseStorage storage = provider.GetRequiredService<BaseStorage>();
            registered = provider.GetRequiredService<RDBStorage>();
            storage.Should().BeSameAs(registered);

            provider.GetRequiredService<IOptunaStudyFactory>().CreateStudy("di").Optimize(Objective, nTrials: 3);
        }

        // The container owned the storage, so disposing the container closed the storage's connection.
        FluentActions.Invoking(() => registered.GetAllStudies()).Should().Throw<InvalidOperationException>();
        using var reopened = new RDBStorage(Url);
        Optuna.LoadStudy("di", reopened).Trials.Should().HaveCount(3);
    }

    /// <summary>The file and busy timeout an open SQLite storage is really using, read back from SQLite itself.</summary>
    /// <param name="storage">The open storage.</param>
    /// <returns>The main database's file (empty for an in-memory database) and <c>PRAGMA busy_timeout</c>.</returns>
    private static (string File, long BusyTimeout) Opened(RDBStorage storage)
    {
        using SqliteCommand command = storage.Connection.CreateCommand();
        command.CommandText = "SELECT file FROM pragma_database_list WHERE name = 'main'";
        string file = command.ExecuteScalar() as string ?? "";
        command.CommandText = "PRAGMA busy_timeout";
        return (file, (long)command.ExecuteScalar()!);
    }

    /// <summary>
    ///     Review item P3-3: the query string is never part of the file name. <c>sqlite:///x.db?timeout=10</c> opens
    ///     <c>x.db</c> with a 10-second busy timeout, as pysqlite does; it used to open (on Linux, create) a file named
    ///     <c>x.db?timeout=10</c>, and fail with SQLite error 14 on Windows.
    /// </summary>
    [TestMethod]
    public void RdbStorage_QueryString_IsNotPartOfTheFileName()
    {
        using (var storage = new RDBStorage(Url + "?timeout=10"))
        {
            var (file, busy) = Opened(storage);
            file.Should().Be(Path.GetFullPath(_path));
            busy.Should().Be(10_000);
            storage.Url.Should().Be(Url + "?timeout=10", "RDBStorage.url keeps the URL as written, as in Python");
        }
        File.Exists(_path).Should().BeTrue();
    }

    /// <summary>pysqlite reads six query arguments and ignores every other one, so an unknown key opens the same file.</summary>
    [TestMethod]
    public void RdbStorage_UnknownQueryArgument_IsIgnoredLikePysqlite()
    {
        using var storage = new RDBStorage(Url + "?foo=1&check_same_thread=false");
        Opened(storage).File.Should().Be(Path.GetFullPath(_path));
    }

    /// <summary>A <c>?timeout=</c> that is not a Python float fails with Python's <c>float()</c> message, before any file is opened.</summary>
    [TestMethod]
    public void RdbStorage_BadTimeout_RaisesPythonsFloatError()
    {
        Action act = () => new RDBStorage(Url + "?timeout=abc");
        act.Should().Throw<NumSharp.ValueError>().WithMessage("could not convert string to float: 'abc'");
        File.Exists(_path).Should().BeFalse();
    }

    /// <summary><c>sqlite://</c> (no path) and <c>sqlite:///:memory:</c> open a private in-memory database, as pysqlite does.</summary>
    [TestMethod]
    public void RdbStorage_NoPath_IsAnInMemoryDatabase()
    {
        foreach (string url in new[] { "sqlite://", "sqlite:///:memory:" })
        {
            using var storage = new RDBStorage(url);
            Opened(storage).File.Should().BeEmpty(url + " is in memory");
        }
    }

    /// <summary>
    ///     Optuna's <c>{SCHEMA_VERSION}</c> template is filled before the URL is parsed (<c>url.format(SCHEMA_VERSION=12)</c>),
    ///     so the file name carries the schema version.
    /// </summary>
    [TestMethod]
    public void RdbStorage_SchemaVersionTemplate_IsFilledFirst()
    {
        string templated = Path.Combine(Path.GetTempPath(), $"optunasharp-url-{Guid.NewGuid():N}");
        using (var storage = new RDBStorage("sqlite:///" + templated.Replace('\\', '/') + "-v{SCHEMA_VERSION}.db"))
            Opened(storage).File.Should().Be(Path.GetFullPath(templated + "-v12.db"));
        SqliteConnection.ClearAllPools();
        File.Delete(templated + "-v12.db");
    }

    /// <summary>
    ///     Review item P2-3: a bare path is not a URL. Python's <c>RDBStorage("study.db")</c> fails in SQLAlchemy's URL
    ///     parser, and so does the port, with the same <c>ArgumentError</c> text; a host in a SQLite URL fails pysqlite's
    ///     check the same way.
    /// </summary>
    [TestMethod]
    public void RdbStorage_BarePathOrHost_IsRefusedLikeSqlAlchemy()
    {
        Action bare = () => new RDBStorage("study.db");
        bare.Should().Throw<ArgumentException>().WithMessage("Could not parse SQLAlchemy URL from given URL string");
        Action host = () => new RDBStorage("sqlite://host/x.db");
        host.Should().Throw<ArgumentException>().WithMessage("Invalid SQLite URL: sqlite://host/x.db*");
    }

    /// <summary>
    ///     With <c>?uri=true</c> the path is a SQLite URI filename and the other query arguments are its parameters:
    ///     <c>sqlite:///file:x.db?mode=ro&amp;uri=true</c> opens <c>x.db</c> read-only, so writing to it fails.
    /// </summary>
    [TestMethod]
    public void RdbStorage_UriMode_PassesQueryArgumentsToSqlite()
    {
        using (var writable = new RDBStorage(Url))
            writable.CreateNewStudy(new[] { StudyDirection.Minimize }, "uri");
        using var readOnly = new RDBStorage("sqlite:///file:" + _path.Replace('\\', '/') + "?mode=ro&uri=true");
        readOnly.GetAllStudies().Should().ContainSingle();
        Action write = () => readOnly.CreateNewStudy(new[] { StudyDirection.Minimize }, "second");
        write.Should().Throw<SqliteException>().Which.SqliteErrorCode.Should().Be(8, "SQLITE_READONLY");
    }

    /// <summary>
    ///     A server URL is checked when it is registered, but nothing connects until the storage is resolved: the
    ///     registration finds the EF Core package by name and adds the concrete type plus the
    ///     <see cref="BaseStorage" /> forward, and only resolution reaches the (closed) server port.
    /// </summary>
    [TestMethod]
    public void AddOptunaStorage_ServerUrl_ChecksAtRegistration_AndConnectsOnResolution()
    {
        // Port 1 on the loopback refuses connections, so the resolution fails without waiting for a timeout.
        var services = new ServiceCollection().AddOptunaStorage("postgresql://user:pw@127.0.0.1:1/optuna");

        services.Select(d => d.ServiceType.FullName).Should().Equal(
            "OptunaSharp.Storages.EFCore.RDBStorage", typeof(BaseStorage).FullName);
        using ServiceProvider provider = services.BuildServiceProvider();
        Exception failure = FluentActions.Invoking(() => provider.GetRequiredService<BaseStorage>())
            .Should().Throw<Exception>().Which;
        failure.Should().NotBeAssignableTo<NotSupportedException>()
            .And.NotBeAssignableTo<ArgumentException>("the URL is valid and its package is referenced");
    }
}
