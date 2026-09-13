using System;
using Microsoft.Extensions.DependencyInjection;
using OptunaSharp.Storages;

namespace OptunaSharp.Storages.EFCore.Extensions;

/// <summary>
///     Microsoft.Extensions.DependencyInjection wire-up for the PostgreSQL / MySQL EF Core RDB storage
///     (<see cref="RDBStorage" />). Registers it as both its concrete type and the
///     <see cref="BaseStorage" /> abstraction, so it composes with Core's
///     <c>AddOptunaStudyFactory()</c> and can be injected directly.
///     <para>
///     <b>Lifetime / disposal:</b> <see cref="RDBStorage" /> is <see cref="IDisposable" /> (it may own a
///     <c>NpgsqlDataSource</c>) — register it as a <b>singleton</b> (the default) so the container owns
///     and disposes it. The provider (PostgreSQL vs MySQL) is chosen by the URL scheme
///     (<c>postgresql://…</c> / <c>mysql://…</c>) or by the explicit <see cref="RdbProvider" /> overload.
///     </para>
/// </summary>
public static class EFCoreRdbStorageServiceCollectionExtensions
{
    /// <summary>Register an EF Core storage opened from a SQLAlchemy-style URL; the scheme selects PostgreSQL or MySQL.</summary>
    /// <param name="services">The service collection to add to.</param>
    /// <param name="url">The database URL (<c>postgresql[+driver]://…</c> or <c>mysql[+driver]://…</c>).</param>
    /// <param name="lifetime">The registration lifetime; defaults to <see cref="ServiceLifetime.Singleton" />.</param>
    /// <returns>The same <paramref name="services" />, for chaining.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="services" /> or <paramref name="url" /> is <c>null</c>.</exception>
    public static IServiceCollection AddEFCoreStorage(
        this IServiceCollection services,
        string url,
        ServiceLifetime lifetime = ServiceLifetime.Singleton)
    {
        ArgumentNullException.ThrowIfNull(services);
        ArgumentNullException.ThrowIfNull(url);
        return Add(services, _ => new RDBStorage(url), lifetime);
    }

    /// <summary>Register an EF Core storage opened from a URL with the full <see cref="EFCoreRdbOptions" /> surface.</summary>
    /// <param name="services">The service collection to add to.</param>
    /// <param name="url">The database URL (<c>postgresql[+driver]://…</c> or <c>mysql[+driver]://…</c>).</param>
    /// <param name="options">The storage and provider configuration applied before schema initialization.</param>
    /// <param name="lifetime">The registration lifetime; defaults to <see cref="ServiceLifetime.Singleton" />.</param>
    /// <returns>The same <paramref name="services" />, for chaining.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="services" />, <paramref name="url" /> or <paramref name="options" /> is <c>null</c>.</exception>
    public static IServiceCollection AddEFCoreStorage(
        this IServiceCollection services,
        string url,
        EFCoreRdbOptions options,
        ServiceLifetime lifetime = ServiceLifetime.Singleton)
    {
        ArgumentNullException.ThrowIfNull(services);
        ArgumentNullException.ThrowIfNull(url);
        ArgumentNullException.ThrowIfNull(options);
        return Add(services, _ => new RDBStorage(url, options), lifetime);
    }

    /// <summary>Register an EF Core storage from an explicit provider and a native ADO.NET connection string.</summary>
    /// <param name="services">The service collection to add to.</param>
    /// <param name="provider">The relational database dialect (PostgreSQL or MySQL).</param>
    /// <param name="connectionString">The native provider connection string.</param>
    /// <param name="options">The storage and provider configuration, or <c>null</c> for defaults.</param>
    /// <param name="lifetime">The registration lifetime; defaults to <see cref="ServiceLifetime.Singleton" />.</param>
    /// <returns>The same <paramref name="services" />, for chaining.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="services" /> or <paramref name="connectionString" /> is <c>null</c>.</exception>
    public static IServiceCollection AddEFCoreStorage(
        this IServiceCollection services,
        RdbProvider provider,
        string connectionString,
        EFCoreRdbOptions? options = null,
        ServiceLifetime lifetime = ServiceLifetime.Singleton)
    {
        ArgumentNullException.ThrowIfNull(services);
        ArgumentNullException.ThrowIfNull(connectionString);
        return Add(services, _ => new RDBStorage(provider, connectionString, options), lifetime);
    }

    /// <summary>Register a pre-built <see cref="RDBStorage" /> instance as the app's <see cref="BaseStorage" /> singleton.</summary>
    /// <param name="services">The service collection to add to.</param>
    /// <param name="storage">The storage instance to share.</param>
    /// <returns>The same <paramref name="services" />, for chaining.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="services" /> or <paramref name="storage" /> is <c>null</c>.</exception>
    public static IServiceCollection AddEFCoreStorage(this IServiceCollection services, RDBStorage storage)
    {
        ArgumentNullException.ThrowIfNull(services);
        ArgumentNullException.ThrowIfNull(storage);
        services.AddSingleton(storage);
        services.AddSingleton<BaseStorage>(storage);
        return services;
    }

    /// <summary>Register an <see cref="RDBStorage" /> produced by a factory (the BYO data-source / builder paths go here).</summary>
    /// <param name="services">The service collection to add to.</param>
    /// <param name="factory">Builds the storage from the resolved <see cref="IServiceProvider" />.</param>
    /// <param name="lifetime">The registration lifetime; defaults to <see cref="ServiceLifetime.Singleton" />.</param>
    /// <returns>The same <paramref name="services" />, for chaining.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="services" /> or <paramref name="factory" /> is <c>null</c>.</exception>
    public static IServiceCollection AddEFCoreStorage(
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
