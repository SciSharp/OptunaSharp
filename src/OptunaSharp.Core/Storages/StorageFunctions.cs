using NumSharp;

namespace OptunaSharp.Storages;

/// <summary>Module-level functions of <c>optuna/storages/__init__.py</c>.</summary>
public static class StorageFunctions
{
    /// <summary>
    ///     Python <c>optuna.storages.get_storage</c> — only for internal usage. It might be
    ///     deprecated in the future.
    ///
    ///     <c>null</c> maps to a fresh <see cref="InMemoryStorage" />. An RDB-style backend (any
    ///     <see cref="ICachedStorageBackend" />, e.g. the Tier X <c>OptunaSharp.Storage.Sqlite</c>
    ///     <c>RDBStorage</c>) is wrapped in a <see cref="CachedStorage" /> — mirroring Python's
    ///     <c>get_storage</c>, which returns <c>_CachedStorage(RDBStorage(...))</c>. Everything else
    ///     (in-memory, journal, an already-wrapped cache) is returned as-is.
    /// </summary>
    /// <param name="storage">Storage backend holding study and trial records.</param>
    /// <returns>A fresh in-memory backend for null, an RDB cache wrapper, or the supplied storage instance.</returns>
    public static BaseStorage GetStorage(BaseStorage? storage)
    {
        if (storage is null)
            return new InMemoryStorage();
        if (storage is ICachedStorageBackend)
            return new CachedStorage(storage);
        return storage;
    }

    /// <summary>URL-based overload mirroring Python's signature; RDB storages are Tier X packages.</summary>
    /// <param name="storageUrl">Storage URL; Core rejects URL dispatch and requires a satellite backend instance.</param>
    /// <returns>This overload always throws; use a backend instance from a storage satellite package.</returns>
    /// <exception cref="NumSharp.ValueError">The URL overload is given a removed RedisStorage URL.</exception>
    /// <exception cref="System.NotSupportedException">The URL overload is used; construct a backend from a satellite storage package.</exception>
    public static BaseStorage GetStorage(string storageUrl)
    {
        if (storageUrl.StartsWith("redis", System.StringComparison.Ordinal))
            throw new ValueError(
                "RedisStorage is removed at Optuna v3.1.0. Please use JournalRedisBackend instead.");
        throw new System.NotSupportedException(
            "URL-based storages (RDB) are provided by the OptunaSharp.Storage.* packages " +
            "(PORTING_PLAN Phase 4). Pass a BaseStorage instance instead, e.g. new JournalStorage(" +
            "new JournalFileBackend(path)).");
    }
}
