using System;
using System.IO;
using Microsoft.Data.Sqlite;
using OptunaSharp;
using OptunaSharp.Logging;
using OptunaSharp.Storages;
using OptunaSharp.Storages.Journal;

namespace OptunaSharp.Cli;

/// <summary>
///     Port of <c>optuna/cli.py: _check_storage_url</c> + <c>_get_storage</c>. Resolves a storage URL
///     (or the <c>OPTUNA_STORAGE</c> environment variable) and an optional <c>--storage-class</c> hint
///     to a <see cref="BaseStorage" />. The <c>optuna</c> command can reach every storage class from the
///     one <c>optuna</c> package; OptunaSharp splits them into satellite packages, so the CLI references
///     the two shipped ones: SQLite (<see cref="RDBStorage" />, chosen by the URL's dialect as
///     <c>get_storage</c> chooses it — see <see cref="CreateRdbStorage" />) and StackExchange.Redis
///     (<see cref="StackExchangeJournalRedisBackend" />). Journal-file storage is in Core.
/// </summary>
internal static class StorageResolver
{
    // Storage-class hint names accepted by --storage-class (Optuna's argparse choices). The last two
    // are deprecated aliases Optuna still honors but hides from the choices list.
    /// <summary>Supported storage-class names, including compatibility aliases accepted by the CLI.</summary>
    internal static readonly string[] StorageClassChoices =
    {
        nameof(RDBStorage),
        nameof(JournalFileBackend),
        "JournalRedisBackend",
        nameof(JournalFileStorage),
        "JournalRedisStorage",
    };

    /// <summary>Uses an explicit storage URL or the OPTUNA_STORAGE environment fallback, rejecting a missing configuration.</summary>
    /// <param name="storageUrl">The explicit storage URL, or null to consult OPTUNA_STORAGE.</param>
    /// <param name="logger">The logger used for environment-fallback diagnostics.</param>
    /// <returns>The explicit URL or environment-provided URL.</returns>
    /// <exception cref="CLIUsageError">Neither an explicit URL nor OPTUNA_STORAGE is available.</exception>
    internal static string CheckStorageUrl(string? storageUrl, Logger logger)
    {
        if (storageUrl is not null)
            return storageUrl;

        string? envStorage = Environment.GetEnvironmentVariable("OPTUNA_STORAGE");
        if (envStorage is not null)
        {
            logger.Warning(
                "Specifying the storage url via 'OPTUNA_STORAGE' environment variable is an experimental"
                + " feature. The interface can change in the future.");
            return envStorage;
        }

        throw new CLIUsageError("Storage URL is not specified.");
    }

    /// <summary>Resolves the storage backend selected by the CLI options and environment fallback.</summary>
    /// <param name="storageUrl">The explicit storage URL, or null to consult OPTUNA_STORAGE.</param>
    /// <param name="storageClass">An optional supported storage-class hint.</param>
    /// <param name="logger">The logger used for environment-fallback diagnostics.</param>
    /// <returns>The resolved SQLite or journal storage backend.</returns>
    /// <exception cref="CLIUsageError">The storage configuration is absent, unsupported, or cannot be inferred.</exception>
    internal static BaseStorage GetStorage(string? storageUrl, string? storageClass, Logger logger)
    {
        string url = CheckStorageUrl(storageUrl, logger);

        if (!string.IsNullOrEmpty(storageClass))
        {
            // Redis: OptunaSharp ships client-specific backends; the CLI wires the StackExchange one for
            // both the current name and the deprecated JournalRedisStorage alias.
            if (storageClass is "JournalRedisBackend" or "JournalRedisStorage")
                return new JournalStorage(new StackExchangeJournalRedisBackend(url));
            if (storageClass == nameof(JournalFileBackend))
                return new JournalStorage(new JournalFileBackend(url));
            if (storageClass == nameof(JournalFileStorage))
#pragma warning disable CS0618 // deprecated alias, honored on purpose to match Optuna's --storage-class
                return new JournalStorage(new JournalFileStorage(url));
#pragma warning restore CS0618
            if (storageClass == nameof(RDBStorage))
                return CreateRdbStorage(url);
            throw new CLIUsageError("Unsupported storage class");
        }

        if (url.StartsWith("redis", StringComparison.Ordinal))
            return new JournalStorage(new StackExchangeJournalRedisBackend(url));
        if (File.Exists(url))
            return new JournalStorage(new JournalFileBackend(url));
        try
        {
            return CreateRdbStorage(url);
        }
        // Python catches sqlalchemy.exc.ArgumentError only (NoSuchModuleError, the unknown-dialect error, is its
        // subclass); both are a plain ArgumentException in the port. Everything else propagates as in Python: a
        // database error, Optuna's ImportError, and ValueError, which derives from ArgumentException.
        catch (ArgumentException e) when (e.GetType() == typeof(ArgumentException))
        {
            throw new CLIUsageError("Failed to guess storage class from storage_url");
        }
    }

    /// <summary>
    ///     Opens the RDB storage a URL names. Python's one <c>RDBStorage</c> class serves every dialect; OptunaSharp
    ///     splits them into packages, so the URL's dialect picks the package (as <c>optuna.storages.get_storage</c>
    ///     does in Core): SQLite, which this tool references, or EF Core for PostgreSQL and MySQL, whose assembly
    ///     the tool does not ship — such a URL fails with Optuna's <c>ImportError</c> text, as a Python install
    ///     without the database driver does.
    /// </summary>
    /// <param name="url">The storage URL.</param>
    /// <returns>The open storage, uncached as Python's <c>RDBStorage(url)</c> is.</returns>
    /// <exception cref="ArgumentException">The URL does not parse, or names a dialect, driver or plugin SQLAlchemy does not know (SQLAlchemy's <c>ArgumentError</c>).</exception>
    /// <exception cref="NumSharp.ValueError">A template, port or driver-argument error, with Python's message.</exception>
    /// <exception cref="NotSupportedException">No package serves the URL's driver, or its assembly is not available.</exception>
    /// <exception cref="SqliteException">SQLite cannot open the database the URL names.</exception>
    internal static BaseStorage CreateRdbStorage(string url)
        => StorageUrlResolver.CreateBackend(StorageUrlResolver.RdbBackendType(url), url);
}
