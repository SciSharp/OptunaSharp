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

    /// <summary>
    ///     Python <c>optuna.storages.get_storage</c> for a storage URL: opens the RDB storage the URL names
    ///     and wraps it in a <see cref="CachedStorage" />, exactly as Python returns
    ///     <c>_CachedStorage(RDBStorage(url))</c>. <c>sqlite:///…</c> URLs open through the
    ///     <c>OptunaSharp.Storage.Sqlite</c> package, and <c>postgresql://…</c>, <c>mysql://…</c> and
    ///     <c>mariadb://…</c> URLs through <c>OptunaSharp.Storage.EFCore</c>. The application must reference the
    ///     package that serves the URL's dialect; Core finds it by name.
    /// </summary>
    /// <param name="storageUrl">
    ///     A SQLAlchemy-style storage URL such as <c>sqlite:///study.db</c> or
    ///     <c>postgresql://user:password@host:5432/optuna</c>. <c>null</c> gives a fresh
    ///     <see cref="InMemoryStorage" />, as Python's <c>None</c> does, so a storage URL read from an unset
    ///     environment variable behaves as in Python.
    /// </param>
    /// <returns>A <see cref="CachedStorage" /> over the newly opened RDB storage, or an in-memory storage for <c>null</c>.</returns>
    /// <exception cref="ValueError">The URL starts with <c>redis</c>: Optuna removed RedisStorage in v3.1.0 (use a Redis journal backend).</exception>
    /// <exception cref="System.ArgumentException">
    ///     The URL does not start with <c>name://</c> (for example a bare file path), or it names a dialect
    ///     SQLAlchemy does not know (for example <c>postgres</c>). The messages are SQLAlchemy's.
    /// </exception>
    /// <exception cref="System.NotSupportedException">
    ///     No OptunaSharp package serves the URL's dialect (<c>oracle</c>, <c>mssql</c>), or the application does
    ///     not reference the package that does. The message starts with Optuna's <c>ImportError</c> text.
    /// </exception>
    /// <remarks>
    ///     <para>The storage package's constructor runs with its defaults, so anything it throws for this URL
    ///     (an unreachable server, an incompatible schema version) propagates unchanged.</para>
    ///     <para>Footgun: like Python's, the returned cache does not own the storage it wraps, so nothing ever
    ///     disposes the database connection. A long-running process that calls this repeatedly keeps every
    ///     connection open; dispose <c>((CachedStorage)storage).Backend</c> when done, or construct the
    ///     package's <c>RDBStorage</c> in a <c>using</c> and pass that instead. Python marks this function
    ///     "only for internal usage".</para>
    /// </remarks>
    public static BaseStorage GetStorage(string storageUrl)
    {
        // Python's single get_storage maps None to an in-memory storage before it looks at strings.
        if (storageUrl is null)
            return GetStorage((BaseStorage?)null);

        System.Type backendType = StorageUrlResolver.ResolveBackendType(storageUrl);
        return GetStorage(StorageUrlResolver.CreateBackend(backendType, storageUrl));
    }
}
