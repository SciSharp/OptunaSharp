using System;
using Microsoft.Extensions.DependencyInjection;
using OptunaSharp.Storages;

namespace OptunaSharp.Storages.Extensions;

/// <summary>
///     Microsoft.Extensions.DependencyInjection wire-up for the SQLite RDB storage
///     (<see cref="RDBStorage" />). Registers it as both its concrete type and the
///     <see cref="BaseStorage" /> abstraction, so it composes with Core's
///     <c>AddOptunaStudyFactory()</c> (which resolves <see cref="BaseStorage" />) and can be injected
///     directly.
///     <para>
///     <b>Lifetime / disposal:</b> <see cref="RDBStorage" /> owns a <c>SqliteConnection</c> and is
///     <see cref="IDisposable" /> — register it as a <b>singleton</b> (the default) so the container owns
///     it and disposes it on shutdown, and so all studies share one connection (its internal lock
///     serializes access). Do not register it transient/scoped unless you truly want a connection per
///     resolution.
///     </para>
/// </summary>
public static class SqliteRdbStorageServiceCollectionExtensions
{
    /// <summary>
    ///     Register a SQLite storage opened from a SQLAlchemy-style URL
    ///     (<c>sqlite:///relative.db</c>, <c>sqlite:////absolute.db</c>, <c>sqlite:///:memory:</c>).
    /// </summary>
    /// <param name="services">The service collection to add to.</param>
    /// <param name="url">The SQLite storage URL.</param>
    /// <param name="lifetime">The registration lifetime; defaults to <see cref="ServiceLifetime.Singleton" /> (recommended for the owned connection).</param>
    /// <returns>The same <paramref name="services" />, for chaining.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="services" /> or <paramref name="url" /> is <c>null</c>.</exception>
    public static IServiceCollection AddSqliteStorage(
        this IServiceCollection services,
        string url,
        ServiceLifetime lifetime = ServiceLifetime.Singleton)
    {
        ArgumentNullException.ThrowIfNull(services);
        ArgumentNullException.ThrowIfNull(url);
        return Add(services, _ => new RDBStorage(url), lifetime);
    }

    /// <summary>
    ///     Register a SQLite storage opened from a URL with the full <see cref="SqliteRdbOptions" /> surface
    ///     (connection-string tweaks, PRAGMAs, heartbeat, schema knobs).
    /// </summary>
    /// <param name="services">The service collection to add to.</param>
    /// <param name="url">The SQLite storage URL.</param>
    /// <param name="options">The storage and provider configuration applied before schema initialization.</param>
    /// <param name="lifetime">The registration lifetime; defaults to <see cref="ServiceLifetime.Singleton" />.</param>
    /// <returns>The same <paramref name="services" />, for chaining.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="services" />, <paramref name="url" /> or <paramref name="options" /> is <c>null</c>.</exception>
    public static IServiceCollection AddSqliteStorage(
        this IServiceCollection services,
        string url,
        SqliteRdbOptions options,
        ServiceLifetime lifetime = ServiceLifetime.Singleton)
    {
        ArgumentNullException.ThrowIfNull(services);
        ArgumentNullException.ThrowIfNull(url);
        ArgumentNullException.ThrowIfNull(options);
        return Add(services, _ => new RDBStorage(url, options), lifetime);
    }

    /// <summary>Register a pre-built <see cref="RDBStorage" /> instance as the app's <see cref="BaseStorage" /> singleton.</summary>
    /// <param name="services">The service collection to add to.</param>
    /// <param name="storage">The storage instance to share.</param>
    /// <returns>The same <paramref name="services" />, for chaining.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="services" /> or <paramref name="storage" /> is <c>null</c>.</exception>
    public static IServiceCollection AddSqliteStorage(this IServiceCollection services, RDBStorage storage)
    {
        ArgumentNullException.ThrowIfNull(services);
        ArgumentNullException.ThrowIfNull(storage);
        services.AddSingleton(storage);
        services.AddSingleton<BaseStorage>(storage);
        return services;
    }

    /// <summary>Register an <see cref="RDBStorage" /> produced by a factory (the BYO-connection / builder paths go here).</summary>
    /// <param name="services">The service collection to add to.</param>
    /// <param name="factory">Builds the storage from the resolved <see cref="IServiceProvider" />.</param>
    /// <param name="lifetime">The registration lifetime; defaults to <see cref="ServiceLifetime.Singleton" />.</param>
    /// <returns>The same <paramref name="services" />, for chaining.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="services" /> or <paramref name="factory" /> is <c>null</c>.</exception>
    public static IServiceCollection AddSqliteStorage(
        this IServiceCollection services,
        Func<IServiceProvider, RDBStorage> factory,
        ServiceLifetime lifetime = ServiceLifetime.Singleton)
    {
        ArgumentNullException.ThrowIfNull(services);
        ArgumentNullException.ThrowIfNull(factory);
        return Add(services, factory, lifetime);
    }

    /// <summary>Registers the concrete storage under the requested lifetime and forwards <see cref="BaseStorage" /> to it.</summary>
    /// <param name="services">The service collection to add to.</param>
    /// <param name="factory">The concrete-storage factory.</param>
    /// <param name="lifetime">The lifetime shared by both registrations.</param>
    /// <returns>The same <paramref name="services" />, for chaining.</returns>
    private static IServiceCollection Add(
        IServiceCollection services,
        Func<IServiceProvider, RDBStorage> factory,
        ServiceLifetime lifetime)
    {
        services.Add(new ServiceDescriptor(typeof(RDBStorage), factory, lifetime));
        services.Add(new ServiceDescriptor(typeof(BaseStorage), sp => sp.GetRequiredService<RDBStorage>(), lifetime));
        return services;
    }
}
