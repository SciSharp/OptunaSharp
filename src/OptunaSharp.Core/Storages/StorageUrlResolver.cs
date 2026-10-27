using System;
using System.Linq;
using System.Reflection;
using System.Text.RegularExpressions;
using NumSharp;

namespace OptunaSharp.Storages;

/// <summary>
///     Turns an Optuna storage URL — the string form of <c>optuna.create_study(storage=...)</c> — into the
///     OptunaSharp RDB storage that serves it. Python's <c>get_storage</c> hands such a string to
///     <c>RDBStorage(url)</c>, where SQLAlchemy picks the database dialect from the URL's scheme; this type
///     makes the same choice between OptunaSharp's RDB storage packages.
/// </summary>
/// <remarks>
///     <para><b>Why late binding.</b> Optuna's <c>RDBStorage</c> lives in the <c>optuna</c> package itself.
///     OptunaSharp ships its RDB storages as satellite packages so that Core carries no database driver:
///     <c>OptunaSharp.Storage.Sqlite</c> serves <c>sqlite</c>, and <c>OptunaSharp.Storage.EFCore</c> serves
///     <c>postgresql</c>, <c>mysql</c> and <c>mariadb</c>. Core cannot reference them, so the resolver loads
///     the serving package's <c>RDBStorage</c> type by name. A package the application does not reference
///     cannot be loaded; that is this port's counterpart of Python's missing DB driver module, and it gets
///     Optuna's <c>ImportError</c> text, raised as <see cref="NotSupportedException" />.</para>
///     <para><b>Errors, in Python's order.</b>
///     <list type="number">
///         <item>A URL starting with <c>redis</c> → Optuna's "RedisStorage is removed" <see cref="ValueError" />.</item>
///         <item>A URL not starting with <c>name://</c> (a bare path, an empty string) → SQLAlchemy's
///         <c>ArgumentError</c> text, as <see cref="ArgumentException" />.</item>
///         <item>A dialect SQLAlchemy does not know (<c>postgres</c>, <c>SQLITE</c>, <c>foo</c>) → SQLAlchemy's
///         <c>NoSuchModuleError</c> text. That error is an <c>ArgumentError</c> subclass in Python, so it is an
///         <see cref="ArgumentException" /> here too.</item>
///         <item>A dialect SQLAlchemy knows but no OptunaSharp package serves (<c>oracle</c>, <c>mssql</c>), or
///         whose package is not referenced → Optuna's <c>ImportError</c> text, as
///         <see cref="NotSupportedException" />.</item>
///     </list></para>
///     <para>Footgun: the scheme is matched case-sensitively, as SQLAlchemy matches it, while the packages'
///     own constructors are more lenient (EF Core lower-cases the scheme and accepts <c>postgres</c>). A URL
///     that <c>new RDBStorage(url)</c> accepts can therefore still be refused here. The driver after <c>+</c>
///     (<c>sqlite+pysqlite</c>, <c>postgresql+psycopg2</c>) is not checked: each package accepts any driver
///     name and ignores it, where SQLAlchemy refuses a driver it does not know.</para>
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

    // SQLAlchemy's URL grammar (sqlalchemy/engine/url.py, _parse_url) opens with the driver name, made of
    // word characters and '+', followed by "://". Everything after that is optional, so a string that starts
    // this way always parses and any other string is refused before a dialect is looked up.
    private static readonly Regex DriverNamePattern = new(@"^(?<name>[\w+]+)://", RegexOptions.CultureInvariant);

    /// <summary>An OptunaSharp package that serves SQLAlchemy dialects with its own <c>RDBStorage</c> type.</summary>
    /// <param name="Package">The package name, which is also its assembly name.</param>
    /// <param name="TypeName">The namespace-qualified name of the package's <c>RDBStorage</c> type.</param>
    private sealed record RdbPackage(string Package, string TypeName);

    private static readonly RdbPackage SqlitePackage = new("OptunaSharp.Storage.Sqlite", "OptunaSharp.Storages.RDBStorage");
    private static readonly RdbPackage EFCorePackage = new("OptunaSharp.Storage.EFCore", "OptunaSharp.Storages.EFCore.RDBStorage");

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
    /// <exception cref="ValueError">The URL starts with <c>redis</c>: Optuna removed RedisStorage in v3.1.0.</exception>
    /// <exception cref="ArgumentException">The URL does not start with <c>name://</c>, or it names a dialect SQLAlchemy does not know.</exception>
    /// <exception cref="NotSupportedException">No OptunaSharp package serves the URL's dialect, or the application does not reference the package that does.</exception>
    /// <exception cref="System.IO.FileLoadException">The serving package's assembly exists but cannot be loaded, for example because of a version conflict.</exception>
    internal static Type ResolveBackendType(string storageUrl)
    {
        // Python's get_storage tests this before RDBStorage sees the string, and case-sensitively, so
        // "Redis://" falls through to the dialect lookup below exactly as it does in Python.
        if (storageUrl.StartsWith("redis", StringComparison.Ordinal))
            throw new ValueError(
                "RedisStorage is removed at Optuna v3.1.0. Please use JournalRedisBackend instead.");

        Match match = DriverNamePattern.Match(storageUrl);
        if (!match.Success)
            throw new ArgumentException("Could not parse SQLAlchemy URL from given URL string");

        string driverName = match.Groups["name"].Value;
        int plus = driverName.IndexOf('+');
        string dialect = plus < 0 ? driverName : driverName.Substring(0, plus);

        RdbPackage? package = dialect switch
        {
            "sqlite" => SqlitePackage,
            "postgresql" or "mysql" or "mariadb" => EFCorePackage,
            // SQLAlchemy's other built-in dialects. Python fails on them only for want of a driver module,
            // so they get Optuna's ImportError rather than the unknown-dialect error below.
            "oracle" or "mssql" => null,
            // SQLAlchemy loads "sqlalchemy.dialects:<dialect>.<driver>", so its message spells '+' as '.'.
            _ => throw new ArgumentException(
                $"Can't load plugin: sqlalchemy.dialects:{driverName.Replace('+', '.')}"),
        };
        if (package is null)
            throw new NotSupportedException(
                $"{ImportErrorMessage} OptunaSharp has no storage package for the '{dialect}' dialect: " +
                $"{SqlitePackage.Package} serves sqlite, and {EFCorePackage.Package} serves postgresql, " +
                "mysql and mariadb.");

        // Loads the package's assembly by its simple name. That works for any assembly the application
        // references, signed or not, and gives null when the application does not reference it.
        Type? backendType = Type.GetType($"{package.TypeName}, {package.Package}", throwOnError: false);
        return backendType ?? throw new NotSupportedException(
            $"{ImportErrorMessage} The '{dialect}' dialect is served by the {package.Package} package; " +
            "add a reference to it.");
    }

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
    ///     Whatever the constructor throws (a URL the package cannot parse, an unreachable server, an
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
