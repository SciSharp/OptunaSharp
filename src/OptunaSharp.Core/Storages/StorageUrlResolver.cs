using System;
using System.Linq;
using System.Reflection;
using NumSharp;
using OptunaSharp.Storages.SqlAlchemy;

namespace OptunaSharp.Storages;

/// <summary>
///     Turns an Optuna storage URL — the string form of <c>optuna.create_study(storage=...)</c> — into the
///     OptunaSharp RDB storage that serves it. Python's <c>get_storage</c> hands such a string to
///     <c>RDBStorage(url)</c>, which fills the <c>{SCHEMA_VERSION}</c> template and lets SQLAlchemy pick the database
///     dialect and driver from the URL; this type makes the same choice between OptunaSharp's RDB storage packages.
/// </summary>
/// <remarks>
///     <para><b>Why late binding.</b> Optuna's <c>RDBStorage</c> lives in the <c>optuna</c> package itself.
///     OptunaSharp ships its RDB storages as satellite packages so that Core carries no database driver:
///     <c>OptunaSharp.Storage.Sqlite</c> serves <c>sqlite</c>, and <c>OptunaSharp.Storage.EFCore</c> serves
///     <c>postgresql</c>, <c>mysql</c> and <c>mariadb</c>. Core cannot reference them, so the resolver loads
///     the serving package's <c>RDBStorage</c> type by name. A package the application does not reference
///     cannot be loaded; that is this port's counterpart of Python's missing DB driver module, and it gets
///     Optuna's <c>ImportError</c> text, raised as <see cref="NotSupportedException" />.</para>
///     <para><b>Errors, in Python's order</b> (see <see cref="Prepare" />): the Redis refusal, then template errors,
///     then SQLAlchemy's parse error, plugin and dialect/driver lookup errors, then the missing-driver
///     <c>ImportError</c>. Everything a driver itself would reject (query arguments, an unreachable server) is raised
///     by the package's constructor.</para>
///     <para>Footgun: the scheme is matched case-sensitively and the driver must be one SQLAlchemy ships
///     (<c>sqlite+pysqlite</c>, <c>postgresql+psycopg2</c>, <c>mysql+pymysql</c>, …), as in SQLAlchemy;
///     <c>postgres://</c> and <c>SQLITE://</c> are refused.</para>
/// </remarks>
internal static class StorageUrlResolver
{
    /// <summary>
    ///     Optuna's message when <c>RDBStorage</c> cannot import the DB access module a URL needs
    ///     (<c>optuna/storages/_rdb/storage.py</c>). The resolver's <see cref="NotSupportedException" />
    ///     messages start with it verbatim and add which OptunaSharp package serves the dialect.
    /// </summary>
    internal const string ImportErrorMessage =
        "Failed to import DB access module for the specified storage URL. Please install appropriate one.";

    /// <summary>An OptunaSharp package that serves SQLAlchemy dialects with its own <c>RDBStorage</c> type.</summary>
    /// <param name="Package">The package name, which is also its assembly name.</param>
    /// <param name="TypeName">The namespace-qualified name of the package's <c>RDBStorage</c> type.</param>
    private sealed record RdbPackage(string Package, string TypeName);

    private static readonly RdbPackage SqlitePackage = new("OptunaSharp.Storage.Sqlite", "OptunaSharp.Storages.RDBStorage");
    private static readonly RdbPackage EFCorePackage = new("OptunaSharp.Storage.EFCore", "OptunaSharp.Storages.EFCore.RDBStorage");

    /// <summary>
    ///     The steps <c>RDBStorage.__init__</c> and <c>create_engine</c> take before any driver code runs: fill the
    ///     template, parse the URL, load plugins, look the dialect and driver up, and refuse a driver OptunaSharp has
    ///     no counterpart for. The storage packages call this from their URL constructors, so a URL fails the same way
    ///     whether it goes through <see cref="StorageFunctions.GetStorage(string)" /> or a package constructor.
    /// </summary>
    /// <param name="storageUrl">The storage URL as written.</param>
    /// <param name="afterTemplate">
    ///     Runs between filling the template and parsing the result, where <c>RDBStorage.__init__</c> validates its
    ///     heartbeat arguments; a package passes its option checks here so a bad option and a bad URL fail in
    ///     Python's order. <see langword="null" /> runs nothing.
    /// </param>
    /// <returns>The filled URL, the parsed URL without its plugin keys, and the dialect and driver.</returns>
    /// <exception cref="ValueError">A template error, a malformed port, or a driver name with two <c>+</c>.</exception>
    /// <exception cref="KeyError">The template names a field other than <c>SCHEMA_VERSION</c>.</exception>
    /// <exception cref="IndexError">The template has a positional field.</exception>
    /// <exception cref="AttributeError">The template reads an attribute <c>int</c> lacks.</exception>
    /// <exception cref="TypeError">The template indexes the integer.</exception>
    /// <exception cref="ArgumentException">
    ///     SQLAlchemy's <c>ArgumentError</c> (no <c>name://</c>) or <c>NoSuchModuleError</c> (an unknown dialect,
    ///     driver or plugin).
    /// </exception>
    /// <exception cref="NotSupportedException">
    ///     The dialect or driver needs a DBAPI module OptunaSharp has no counterpart for (<c>oracle</c>, <c>mssql</c>,
    ///     <c>sqlite+pysqlcipher</c>, <c>mysql+pyodbc</c>); the message starts with Optuna's <c>ImportError</c> text.
    /// </exception>
    internal static PreparedStorageUrl Prepare(string storageUrl, Action? afterTemplate = null)
    {
        string filled = StorageUrlTemplate.Fill(storageUrl);
        afterTemplate?.Invoke();
        SqlAlchemyUrl parsed = SqlAlchemyUrl.Parse(filled);
        SqlAlchemyDriver driver = SqlAlchemyRegistry.Resolve(parsed, out SqlAlchemyUrl withoutPlugins);
        if (ServingPackage(driver) is null)
        {
            string served = driver.Dialect is "sqlite" or "postgresql" or "mysql" or "mariadb"
                ? $"OptunaSharp has no counterpart of the '{driver.Driver}' driver of the '{driver.Dialect}' dialect"
                : $"OptunaSharp has no storage package for the '{driver.Dialect}' dialect";
            throw new NotSupportedException(
                $"{ImportErrorMessage} {served}: {SqlitePackage.Package} serves sqlite, and {EFCorePackage.Package} " +
                "serves postgresql, mysql and mariadb.");
        }
        return new PreparedStorageUrl(filled, withoutPlugins, driver);
    }

    /// <summary>
    ///     Validates a storage URL and returns the type of the OptunaSharp RDB storage that serves it. Nothing
    ///     is opened or connected, so callers can validate configuration early and open the storage later with
    ///     <see cref="CreateBackend" />.
    /// </summary>
    /// <param name="storageUrl">
    ///     The storage URL, for example <c>sqlite:///study.db</c> or
    ///     <c>postgresql://user:password@host:5432/optuna</c>.
    /// </param>
    /// <returns>
    ///     The serving package's <c>RDBStorage</c> type. It derives from <see cref="BaseStorage" />, implements
    ///     <see cref="ICachedStorageBackend" /> and is <see cref="IDisposable" />.
    /// </returns>
    /// <exception cref="ValueError">The URL starts with <c>redis</c> (Optuna removed RedisStorage in v3.1.0), or one of <see cref="Prepare" />'s value errors.</exception>
    /// <exception cref="KeyError">See <see cref="Prepare" />.</exception>
    /// <exception cref="IndexError">See <see cref="Prepare" />.</exception>
    /// <exception cref="AttributeError">See <see cref="Prepare" />.</exception>
    /// <exception cref="TypeError">See <see cref="Prepare" />.</exception>
    /// <exception cref="ArgumentException">The URL does not parse, or names a dialect, driver or plugin SQLAlchemy does not know.</exception>
    /// <exception cref="NotSupportedException">No OptunaSharp package serves the URL's driver, or the application does not reference the package that does.</exception>
    /// <exception cref="System.IO.FileLoadException">The serving package's assembly exists but cannot be loaded, for example because of a version conflict.</exception>
    internal static Type ResolveBackendType(string storageUrl)
    {
        // Python's get_storage tests this before RDBStorage sees the string, and case-sensitively, so
        // "Redis://" falls through to the dialect lookup below exactly as it does in Python.
        if (storageUrl.StartsWith("redis", StringComparison.Ordinal))
            throw new ValueError(
                "RedisStorage is removed at Optuna v3.1.0. Please use JournalRedisBackend instead.");
        return RdbBackendType(storageUrl);
    }

    /// <summary>
    ///     The type of the OptunaSharp RDB storage that serves a URL — what Python's single <c>RDBStorage(url)</c>
    ///     class would open — without <see cref="ResolveBackendType" />'s <c>get_storage</c> Redis check, so a
    ///     <c>redis://</c> URL fails in SQLAlchemy's dialect lookup, as <c>RDBStorage("redis://…")</c> does.
    /// </summary>
    /// <param name="storageUrl">The storage URL.</param>
    /// <returns>The serving package's <c>RDBStorage</c> type.</returns>
    /// <exception cref="ValueError">One of <see cref="Prepare" />'s value errors.</exception>
    /// <exception cref="KeyError">See <see cref="Prepare" />.</exception>
    /// <exception cref="IndexError">See <see cref="Prepare" />.</exception>
    /// <exception cref="AttributeError">See <see cref="Prepare" />.</exception>
    /// <exception cref="TypeError">See <see cref="Prepare" />.</exception>
    /// <exception cref="ArgumentException">The URL does not parse, or names a dialect, driver or plugin SQLAlchemy does not know.</exception>
    /// <exception cref="NotSupportedException">No OptunaSharp package serves the URL's driver, or the application does not reference the package that does.</exception>
    /// <exception cref="System.IO.FileLoadException">The serving package's assembly exists but cannot be loaded, for example because of a version conflict.</exception>
    internal static Type RdbBackendType(string storageUrl)
    {
        SqlAlchemyDriver driver = Prepare(storageUrl).Driver;
        RdbPackage package = ServingPackage(driver)!;

        // Loads the package's assembly by its simple name. That works for any assembly the application
        // references, signed or not, and gives null when the application does not reference it.
        Type? backendType = Type.GetType($"{package.TypeName}, {package.Package}", throwOnError: false);
        return backendType ?? throw new NotSupportedException(
            $"{ImportErrorMessage} The '{driver.Dialect}' dialect is served by the {package.Package} package; " +
            "add a reference to it.");
    }

    /// <summary>
    ///     The package that serves a dialect and driver, or <see langword="null" /> when none does. Asyncio drivers
    ///     are served: the package raises SQLAlchemy's synchronous-engine error for them, after it has checked the
    ///     URL's arguments, which is the order in which Python fails.
    /// </summary>
    /// <param name="driver">The dialect and driver.</param>
    /// <returns>The serving package, or <see langword="null" />.</returns>
    private static RdbPackage? ServingPackage(SqlAlchemyDriver driver) => driver.Dialect switch
    {
        // pysqlcipher needs SQLCipher's sqlcipher3 module; pyodbc needs an ODBC driver. Neither has an OptunaSharp
        // counterpart, so they fail like a Python install without the module.
        "sqlite" when driver.Driver != "pysqlcipher" => SqlitePackage,
        "postgresql" => EFCorePackage,
        "mysql" or "mariadb" when driver.Driver != "pyodbc" => EFCorePackage,
        _ => null,
    };

    /// <summary>
    ///     Opens the storage a URL names by calling its package's Optuna-faithful constructor,
    ///     <c>RDBStorage(string url, …)</c>, with every parameter after the URL at its default. That is the
    ///     constructor <c>new RDBStorage(url)</c> calls.
    /// </summary>
    /// <param name="backendType">The type <see cref="ResolveBackendType" /> returned for <paramref name="storageUrl" />.</param>
    /// <param name="storageUrl">The storage URL, passed to the constructor unchanged.</param>
    /// <returns>
    ///     The new storage. It owns its database connection and is <see cref="IDisposable" />: whoever holds it
    ///     disposes it.
    /// </returns>
    /// <exception cref="InvalidOperationException">
    ///     <paramref name="backendType" /> has no single constructor taking the URL followed only by optional
    ///     parameters, which means the storage package and Core come from incompatible versions.
    /// </exception>
    /// <remarks>
    ///     Whatever the constructor throws (a URL argument the driver refuses, an unreachable server, an
    ///     incompatible schema version) propagates unwrapped, exactly as from <c>new RDBStorage(url)</c>.
    /// </remarks>
    internal static BaseStorage CreateBackend(Type backendType, string storageUrl)
    {
        ConstructorInfo[] urlConstructors = backendType.GetConstructors()
            .Where(constructor =>
            {
                ParameterInfo[] parameters = constructor.GetParameters();
                return parameters.Length > 0 && parameters[0].ParameterType == typeof(string)
                                             && parameters.Skip(1).All(p => p.HasDefaultValue);
            })
            .ToArray();
        if (urlConstructors.Length != 1)
            throw new InvalidOperationException(
                $"{backendType.FullName} has {urlConstructors.Length} constructors taking a URL followed only by " +
                "optional parameters instead of exactly one; the storage package and OptunaSharp versions do not match.");

        ConstructorInfo urlConstructor = urlConstructors[0];
        object?[] arguments = urlConstructor.GetParameters()
            .Select((parameter, index) => index == 0 ? storageUrl : parameter.DefaultValue)
            .ToArray();
        // DoNotWrapExceptions: callers see the constructor's own exception, as from `new RDBStorage(url)`,
        // instead of a TargetInvocationException around it.
        return (BaseStorage)urlConstructor.Invoke(BindingFlags.DoNotWrapExceptions, binder: null, arguments, culture: null);
    }
}

/// <summary>A storage URL after the steps every RDB storage takes before its driver runs (see <see cref="StorageUrlResolver.Prepare" />).</summary>
/// <param name="Filled">The URL after <c>{SCHEMA_VERSION}</c> filling: Python's <c>RDBStorage.url</c>.</param>
/// <param name="Url">The parsed URL, without the <c>plugin</c>/<c>plugin_name</c> query keys SQLAlchemy consumes.</param>
/// <param name="Driver">The SQLAlchemy dialect and driver the URL selects.</param>
internal sealed record PreparedStorageUrl(string Filled, SqlAlchemyUrl Url, SqlAlchemyDriver Driver);
