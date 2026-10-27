using System;
using AwesomeAssertions;
using Microsoft.Extensions.DependencyInjection;
using NumSharp;
using OptunaSharp.Extensions;
using OptunaSharp.Storages;

namespace OptunaSharp.Tests.Storages;

/// <summary>
///     Review item H7: storage URLs. <see cref="StorageFunctions.GetStorage(string)" /> threw for every URL, so
///     <c>AddOptunaStorage(string)</c>, which registered it, could never resolve. Both now hand a URL to the
///     storage package that serves its dialect, as Python's <c>get_storage</c> hands it to <c>RDBStorage</c>, and
///     refuse every other string with the error Python raises. This suite references no storage package, which is
///     the setup where a supported URL must fail with Optuna's <c>ImportError</c> text; the storage suite covers
///     the URLs that open, and the interop suite compares every outcome with live Optuna.
/// </summary>
[TestClass]
public class StorageUrlTests
{
    /// <summary>Optuna's <c>ImportError</c> message from <c>optuna/storages/_rdb/storage.py</c>.</summary>
    private const string ImportErrorMessage =
        "Failed to import DB access module for the specified storage URL. Please install appropriate one.";

    /// <summary>
    ///     A URL starting with <c>redis</c> raises Optuna's ValueError before anything else is checked, from both
    ///     entry points, and the DI method registers nothing.
    /// </summary>
    /// <param name="url">A URL with a lower-case <c>redis</c> prefix.</param>
    [TestMethod]
    [DataRow("redis://localhost:6379")]
    [DataRow("rediss://localhost:6379")]
    [DataRow("redis")]
    public void RedisUrls_RaiseOptunasValueError(string url)
    {
        const string message = "RedisStorage is removed at Optuna v3.1.0. Please use JournalRedisBackend instead.";
        var services = new ServiceCollection();

        FluentActions.Invoking(() => StorageFunctions.GetStorage(url))
            .Should().ThrowExactly<ValueError>().Which.Message.Should().Be(message);
        FluentActions.Invoking(() => services.AddOptunaStorage(url))
            .Should().ThrowExactly<ValueError>().Which.Message.Should().Be(message);
        services.Should().BeEmpty();
    }

    /// <summary>
    ///     A string that does not start with <c>name://</c> is not a URL to SQLAlchemy: a bare path, an empty
    ///     string, a Windows path, a scheme with a character outside <c>[\w+]</c>. It raises SQLAlchemy's
    ///     ArgumentError text as a plain <see cref="ArgumentException" />, not Optuna's ValueError.
    /// </summary>
    /// <param name="url">A string SQLAlchemy cannot parse as a URL.</param>
    [TestMethod]
    [DataRow("study.db")]
    [DataRow("")]
    [DataRow("C:/runs/study.db")]
    [DataRow("my-db://host/db")]
    [DataRow(" sqlite:///study.db")]
    public void StringsThatAreNotUrls_RaiseSqlAlchemysParseError(string url)
    {
        const string message = "Could not parse SQLAlchemy URL from given URL string";
        var services = new ServiceCollection();

        FluentActions.Invoking(() => StorageFunctions.GetStorage(url))
            .Should().ThrowExactly<ArgumentException>().Which.Message.Should().Be(message);
        FluentActions.Invoking(() => services.AddOptunaStorage(url))
            .Should().ThrowExactly<ArgumentException>().Which.Message.Should().Be(message);
        services.Should().BeEmpty();
    }

    /// <summary>
    ///     A dialect SQLAlchemy does not know raises its NoSuchModuleError text, which names the plugin with the
    ///     driver's <c>+</c> spelled <c>.</c>. Matching is case-sensitive, and <c>postgres</c> is not an alias of
    ///     <c>postgresql</c> in SQLAlchemy 2, although the EF Core package's own constructor accepts both forms.
    /// </summary>
    /// <param name="url">A URL whose dialect SQLAlchemy has no plugin for.</param>
    /// <param name="plugin">The plugin name SQLAlchemy's message reports.</param>
    [TestMethod]
    [DataRow("postgres://user:pw@localhost/optuna", "postgres")]
    [DataRow("SQLITE:///study.db", "SQLITE")]
    [DataRow("Redis://localhost:6379", "Redis")]
    [DataRow("foo+bar://host/db", "foo.bar")]
    public void UnknownDialects_RaiseSqlAlchemysPluginError(string url, string plugin)
    {
        string message = $"Can't load plugin: sqlalchemy.dialects:{plugin}";
        var services = new ServiceCollection();

        FluentActions.Invoking(() => StorageFunctions.GetStorage(url))
            .Should().ThrowExactly<ArgumentException>().Which.Message.Should().Be(message);
        FluentActions.Invoking(() => services.AddOptunaStorage(url))
            .Should().ThrowExactly<ArgumentException>().Which.Message.Should().Be(message);
        services.Should().BeEmpty();
    }

    /// <summary>
    ///     SQLAlchemy knows the <c>oracle</c> and <c>mssql</c> dialects, and Python fails on them only for want of a
    ///     driver module, with Optuna's ImportError. OptunaSharp has no package for them, so the message starts
    ///     with that text and then says which dialects the OptunaSharp packages serve.
    /// </summary>
    /// <param name="url">A URL for a SQLAlchemy dialect no OptunaSharp package serves.</param>
    /// <param name="dialect">The dialect the message must name.</param>
    [TestMethod]
    [DataRow("oracle://user:pw@localhost/optuna", "oracle")]
    [DataRow("mssql+pyodbc://user:pw@localhost/optuna", "mssql")]
    public void DialectsNoPackageServes_RaiseOptunasImportErrorText(string url, string dialect)
    {
        var services = new ServiceCollection();

        string message = FluentActions.Invoking(() => StorageFunctions.GetStorage(url))
            .Should().ThrowExactly<NotSupportedException>().Which.Message;
        message.Should().StartWith(ImportErrorMessage)
            .And.Contain($"OptunaSharp has no storage package for the '{dialect}' dialect");
        FluentActions.Invoking(() => services.AddOptunaStorage(url))
            .Should().ThrowExactly<NotSupportedException>().Which.Message.Should().Be(message);
        services.Should().BeEmpty();
    }

    /// <summary>
    ///     A supported dialect whose package the application does not reference: this suite references neither
    ///     storage package. The error is Optuna's ImportError text plus the package to add. This is the H7 repro
    ///     at the earliest point it can now fail: <c>AddOptunaStorage("sqlite:///x.db")</c> refuses at registration
    ///     instead of registering a factory that threw on every resolution.
    /// </summary>
    /// <param name="url">A URL for a dialect an OptunaSharp package serves.</param>
    /// <param name="dialect">The URL's dialect, before any <c>+driver</c>.</param>
    /// <param name="package">The package that serves the dialect.</param>
    [TestMethod]
    [DataRow("sqlite:///x.db", "sqlite", "OptunaSharp.Storage.Sqlite")]
    [DataRow("sqlite+pysqlite:///:memory:", "sqlite", "OptunaSharp.Storage.Sqlite")]
    [DataRow("postgresql://user:pw@localhost/optuna", "postgresql", "OptunaSharp.Storage.EFCore")]
    [DataRow("postgresql+psycopg2://user:pw@localhost/optuna", "postgresql", "OptunaSharp.Storage.EFCore")]
    [DataRow("mysql+pymysql://user:pw@localhost/optuna", "mysql", "OptunaSharp.Storage.EFCore")]
    [DataRow("mariadb://user:pw@localhost/optuna", "mariadb", "OptunaSharp.Storage.EFCore")]
    public void SupportedDialects_WithoutTheirPackage_NameThePackageToReference(string url, string dialect, string package)
    {
        var services = new ServiceCollection();

        string message = FluentActions.Invoking(() => StorageFunctions.GetStorage(url))
            .Should().ThrowExactly<NotSupportedException>().Which.Message;
        message.Should().Be(
            $"{ImportErrorMessage} The '{dialect}' dialect is served by the {package} package; add a reference to it.");
        FluentActions.Invoking(() => services.AddOptunaStorage(url))
            .Should().ThrowExactly<NotSupportedException>().Which.Message.Should().Be(message);
        services.Should().BeEmpty();
    }

    /// <summary>
    ///     Python's <c>get_storage(None)</c> returns a new in-memory storage. A <c>null</c> URL, such as an unset
    ///     environment variable, does the same here instead of the NullReferenceException it used to throw. The DI
    ///     method keeps refusing <c>null</c>, as every registration method does.
    /// </summary>
    [TestMethod]
    public void NullUrl_GivesANewInMemoryStorage_LikePythonsNone()
    {
        string? unset = null;

        BaseStorage first = StorageFunctions.GetStorage(unset!);
        BaseStorage second = StorageFunctions.GetStorage(unset!);

        first.Should().BeOfType<InMemoryStorage>();
        second.Should().BeOfType<InMemoryStorage>().And.NotBeSameAs(first);
        FluentActions.Invoking(() => new ServiceCollection().AddOptunaStorage(unset!))
            .Should().ThrowExactly<ArgumentNullException>().Which.ParamName.Should().Be("storageUrl");
    }
}
