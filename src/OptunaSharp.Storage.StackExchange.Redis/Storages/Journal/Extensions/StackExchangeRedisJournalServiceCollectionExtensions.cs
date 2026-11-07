using System;
using Microsoft.Extensions.DependencyInjection;

namespace OptunaSharp.Storages.Journal.Extensions;

/// <summary>
///     Microsoft.Extensions.DependencyInjection wire-up for the StackExchange.Redis journal backend
///     (<see cref="StackExchangeJournalRedisBackend" />). Registers the backend, a
///     <see cref="JournalStorage" /> wrapping it, and the <see cref="BaseStorage" /> abstraction that
///     forwards to that <see cref="JournalStorage" /> — so a consumer injects <see cref="BaseStorage" />
///     (or composes with Core's <c>AddOptunaStudyFactory()</c>) and gets a Redis-backed study store.
///     <para>
///     The class name is package-unique (all three Redis backends share the
///     <c>OptunaSharp.Storages.Journal.Extensions</c> namespace, and the base <c>BaseStorage</c> /
///     <c>JournalStorage</c> registrations would otherwise be indistinguishable) so referencing more
///     than one Redis client at once does not clash. <b>Connection ownership:</b> the backend does not
///     dispose the Redis connection — when you pass an <c>IConnectionMultiplexer</c> (via the factory
///     overload) you keep its lifetime.
///     </para>
/// <para>
/// <b>Disposal.</b> The backend is <see cref="System.IDisposable" /> and <see cref="System.IAsyncDisposable" />: a
/// backend the container builds (the URL and factory overloads) is disposed with the scope that created it, or with
/// the container for a singleton, and closes the connection it opened. Scoped and transient registrations therefore
/// no longer leak a connection each. A scope disposed asynchronously (ASP.NET Core request scopes,
/// <c>AsyncServiceScope</c>, <c>await provider.DisposeAsync()</c>) closes it through <c>DisposeAsync</c> without
/// blocking a thread; a scope disposed synchronously uses <c>Dispose</c>. A pre-built backend you register is never
/// disposed by the container, and a backend never closes a client you injected into it.
/// </para>
/// </summary>
public static class StackExchangeRedisJournalServiceCollectionExtensions
{
    /// <summary>
    ///     Register a StackExchange.Redis journal storage connecting to a <c>redis://</c> /
    ///     <c>rediss://</c> URL. Registers <see cref="StackExchangeJournalRedisBackend" />,
    ///     <see cref="JournalStorage" />, and <see cref="BaseStorage" />.
    /// </summary>
    /// <param name="services">The service collection to add to.</param>
    /// <param name="url">The Redis endpoint URL.</param>
    /// <param name="useCluster">Whether the server is a Redis Cluster (changes key hash-tagging).</param>
    /// <param name="prefix">Key-namespace prefix for this study's journal keys.</param>
    /// <param name="lifetime">The registration lifetime; defaults to <see cref="ServiceLifetime.Singleton" />.</param>
    /// <param name="dedicatedThreads">Whether the backend turns on StackExchange.Redis's process-wide <c>DedicatedThreads</c>
    ///   feature flag before it connects (default <c>true</c>); see <see cref="StackExchangeJournalRedisBackend" />'s remarks.</param>
    /// <returns>The same <paramref name="services" />, for chaining.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="services" />, <paramref name="url" /> or <paramref name="prefix" /> is <c>null</c>.</exception>
    public static IServiceCollection AddStackExchangeRedisJournalStorage(
        this IServiceCollection services,
        string url,
        bool useCluster = false,
        string prefix = "",
        ServiceLifetime lifetime = ServiceLifetime.Singleton,
        bool dedicatedThreads = true)
    {
        ArgumentNullException.ThrowIfNull(services);
        ArgumentNullException.ThrowIfNull(url);
        ArgumentNullException.ThrowIfNull(prefix);
        return Add(services, new ServiceDescriptor(typeof(StackExchangeJournalRedisBackend), _ => new StackExchangeJournalRedisBackend(url, useCluster, prefix, dedicatedThreads), lifetime), lifetime);
    }

    /// <summary>Register a pre-built <see cref="StackExchangeJournalRedisBackend" /> (wrapped in a <see cref="JournalStorage" /> and exposed as <see cref="BaseStorage" />).</summary>
    /// <param name="services">The service collection to add to.</param>
    /// <param name="backend">The journal backend instance to share.</param>
    /// <returns>The same <paramref name="services" />, for chaining.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="services" /> or <paramref name="backend" /> is <c>null</c>.</exception>
    public static IServiceCollection AddStackExchangeRedisJournalStorage(
        this IServiceCollection services,
        StackExchangeJournalRedisBackend backend)
    {
        ArgumentNullException.ThrowIfNull(services);
        ArgumentNullException.ThrowIfNull(backend);
        // An instance registration: a container never disposes an instance it was handed, so this backend (and
        // any connection it owns) stays the caller's to dispose. A factory returning it would be disposed with the
        // container now that backends are IDisposable.
        return Add(services, new ServiceDescriptor(typeof(StackExchangeJournalRedisBackend), backend), ServiceLifetime.Singleton);
    }

    /// <summary>Register a <see cref="StackExchangeJournalRedisBackend" /> produced by a factory (the ConfigurationOptions / IConnectionMultiplexer paths go here).</summary>
    /// <param name="services">The service collection to add to.</param>
    /// <param name="factory">Builds the backend from the resolved <see cref="IServiceProvider" />.</param>
    /// <param name="lifetime">The registration lifetime; defaults to <see cref="ServiceLifetime.Singleton" />.</param>
    /// <returns>The same <paramref name="services" />, for chaining.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="services" /> or <paramref name="factory" /> is <c>null</c>.</exception>
    public static IServiceCollection AddStackExchangeRedisJournalStorage(
        this IServiceCollection services,
        Func<IServiceProvider, StackExchangeJournalRedisBackend> factory,
        ServiceLifetime lifetime = ServiceLifetime.Singleton)
    {
        ArgumentNullException.ThrowIfNull(services);
        ArgumentNullException.ThrowIfNull(factory);
        return Add(services, new ServiceDescriptor(typeof(StackExchangeJournalRedisBackend), factory, lifetime), lifetime);
    }

    /// <summary>Registers the backend, a <see cref="JournalStorage" /> over it, and a <see cref="BaseStorage" /> forwarder, all under one lifetime.</summary>
    /// <param name="services">The service collection to add to.</param>
    /// <param name="backend">The concrete backend's registration: a pre-built instance, or a factory under <paramref name="lifetime" />.</param>
    /// <param name="lifetime">The lifetime shared by the backend, its journal storage, and the base-storage forwarder.</param>
    /// <returns>The same <paramref name="services" />, for chaining.</returns>
    private static IServiceCollection Add(
        IServiceCollection services,
        ServiceDescriptor backend,
        ServiceLifetime lifetime)
    {
        services.Add(backend);
        // JournalStorage wraps the SAME backend instance the container holds, so the base-storage forwarder
        // and a directly-resolved backend stay consistent.
        services.Add(new ServiceDescriptor(typeof(JournalStorage),
            sp => new JournalStorage(sp.GetRequiredService<StackExchangeJournalRedisBackend>()), lifetime));
        services.Add(new ServiceDescriptor(typeof(BaseStorage), sp => sp.GetRequiredService<JournalStorage>(), lifetime));
        return services;
    }
}
