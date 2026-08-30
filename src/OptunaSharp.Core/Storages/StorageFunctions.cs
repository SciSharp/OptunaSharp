using NumSharp;

namespace OptunaSharp.Storages;

/// <summary>Module-level functions of <c>optuna/storages/__init__.py</c>.</summary>
public static class StorageFunctions
{
    /// <summary>
    ///     Python <c>optuna.storages.get_storage</c> — only for internal usage. It might be
    ///     deprecated in the future.
    ///
    ///     <c>null</c> maps to a fresh <see cref="InMemoryStorage" />. URL-string storages
    ///     (<c>sqlite:///...</c>) arrive with the Tier X RDB packages; passing a string raises
    ///     until then.
    /// </summary>
    public static BaseStorage GetStorage(BaseStorage? storage)
    {
        if (storage is null)
            return new InMemoryStorage();
        return storage;
    }

    /// <summary>URL-based overload mirroring Python's signature; RDB storages are Tier X packages.</summary>
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
