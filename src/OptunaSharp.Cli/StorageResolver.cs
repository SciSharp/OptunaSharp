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
///     the two shipped ones: SQLite (<see cref="RDBStorage" />) and StackExchange.Redis
///     (<see cref="StackExchangeJournalRedisBackend" />). Journal-file storage is in Core.
/// </summary>
internal static class StorageResolver
{
    // Storage-class hint names accepted by --storage-class (Optuna's argparse choices). The last two
    // are deprecated aliases Optuna still honors but hides from the choices list.
    internal static readonly string[] StorageClassChoices =
    {
        nameof(RDBStorage),
        nameof(JournalFileBackend),
        "JournalRedisBackend",
        nameof(JournalFileStorage),
        "JournalRedisStorage",
    };

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
                return new RDBStorage(url);
            throw new CLIUsageError("Unsupported storage class");
        }

        if (url.StartsWith("redis", StringComparison.Ordinal))
            return new JournalStorage(new StackExchangeJournalRedisBackend(url));
        if (File.Exists(url))
            return new JournalStorage(new JournalFileBackend(url));
        try
        {
            return new RDBStorage(url);
        }
        catch (Exception e) when (e is SqliteException or ArgumentException)
        {
            throw new CLIUsageError("Failed to guess storage class from storage_url");
        }
    }
}
