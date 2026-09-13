using System;
using Grpc.Net.Client;
using Microsoft.Extensions.DependencyInjection;
using OptunaSharp.Storages;

namespace OptunaSharp.Storages.Grpc.Extensions;

/// <summary>
///     Microsoft.Extensions.DependencyInjection wire-up for the gRPC storage-proxy <b>client</b>
///     (<see cref="GrpcStorageProxy" />) — the counterpart of the <b>server</b>-side
///     <c>AddOptunaStorageProxy</c> / <c>MapOptunaStorageProxy</c> in
///     <see cref="GrpcProxyServiceExtensions" />. Registers the proxy as both its concrete type and the
///     <see cref="BaseStorage" /> abstraction, so a client app injects <see cref="BaseStorage" /> and
///     talks to a remote Optuna storage server. DI comes from the ASP.NET Core shared framework this
///     package already references — no extra package is needed.
///     <para>
///     <b>Lifetime / disposal:</b> <see cref="GrpcStorageProxy" /> is <see cref="IDisposable" /> and owns
///     the <see cref="GrpcChannel" /> it creates — register it as a <b>singleton</b> (the default) so the
///     container disposes it (and its channel) on shutdown. The BYO-channel overloads do not transfer
///     channel ownership to the proxy; dispose such a channel yourself.
///     </para>
/// </summary>
public static class GrpcStorageProxyServiceCollectionExtensions
{
    /// <summary>Register a gRPC storage-proxy client connecting over insecure h2c to <paramref name="host" />:<paramref name="port" /> (the Optuna-identical entry point).</summary>
    /// <param name="services">The service collection to add to.</param>
    /// <param name="host">The gRPC server hostname; defaults to <c>localhost</c>.</param>
    /// <param name="port">The gRPC server port; defaults to <c>13000</c>.</param>
    /// <param name="lifetime">The registration lifetime; defaults to <see cref="ServiceLifetime.Singleton" /> (the proxy owns its channel).</param>
    /// <returns>The same <paramref name="services" />, for chaining.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="services" /> or <paramref name="host" /> is <c>null</c>.</exception>
    public static IServiceCollection AddOptunaGrpcStorageProxyClient(
        this IServiceCollection services,
        string host = "localhost",
        int port = 13000,
        ServiceLifetime lifetime = ServiceLifetime.Singleton)
    {
        ArgumentNullException.ThrowIfNull(services);
        ArgumentNullException.ThrowIfNull(host);
        return Add(services, _ => new GrpcStorageProxy(host, port), lifetime);
    }

    /// <summary>
    ///     Register a gRPC storage-proxy client on a <see cref="GrpcChannel" /> you built and own (TLS,
    ///     interceptors, a shared channel). The proxy will <b>not</b> dispose this channel — you retain its
    ///     lifetime.
    /// </summary>
    /// <param name="services">The service collection to add to.</param>
    /// <param name="channel">A configured channel to bind the client to.</param>
    /// <param name="lifetime">The registration lifetime; defaults to <see cref="ServiceLifetime.Singleton" />.</param>
    /// <returns>The same <paramref name="services" />, for chaining.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="services" /> or <paramref name="channel" /> is <c>null</c>.</exception>
    public static IServiceCollection AddOptunaGrpcStorageProxyClient(
        this IServiceCollection services,
        GrpcChannel channel,
        ServiceLifetime lifetime = ServiceLifetime.Singleton)
    {
        ArgumentNullException.ThrowIfNull(services);
        ArgumentNullException.ThrowIfNull(channel);
        return Add(services, _ => new GrpcStorageProxy(channel), lifetime);
    }

    /// <summary>Register a pre-built <see cref="GrpcStorageProxy" /> instance as the app's <see cref="BaseStorage" /> singleton.</summary>
    /// <param name="services">The service collection to add to.</param>
    /// <param name="proxy">The storage-proxy instance to share.</param>
    /// <returns>The same <paramref name="services" />, for chaining.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="services" /> or <paramref name="proxy" /> is <c>null</c>.</exception>
    public static IServiceCollection AddOptunaGrpcStorageProxyClient(this IServiceCollection services, GrpcStorageProxy proxy)
    {
        ArgumentNullException.ThrowIfNull(services);
        ArgumentNullException.ThrowIfNull(proxy);
        services.AddSingleton(proxy);
        services.AddSingleton<BaseStorage>(proxy);
        return services;
    }

    /// <summary>Register a <see cref="GrpcStorageProxy" /> produced by a factory (the address + channel-options paths go here).</summary>
    /// <param name="services">The service collection to add to.</param>
    /// <param name="factory">Builds the proxy from the resolved <see cref="IServiceProvider" />.</param>
    /// <param name="lifetime">The registration lifetime; defaults to <see cref="ServiceLifetime.Singleton" />.</param>
    /// <returns>The same <paramref name="services" />, for chaining.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="services" /> or <paramref name="factory" /> is <c>null</c>.</exception>
    public static IServiceCollection AddOptunaGrpcStorageProxyClient(
        this IServiceCollection services,
        Func<IServiceProvider, GrpcStorageProxy> factory,
        ServiceLifetime lifetime = ServiceLifetime.Singleton)
    {
        ArgumentNullException.ThrowIfNull(services);
        ArgumentNullException.ThrowIfNull(factory);
        return Add(services, factory, lifetime);
    }

    /// <summary>Registers the concrete proxy under the requested lifetime and forwards <see cref="BaseStorage" /> to it.</summary>
    /// <param name="services">The service collection to add to.</param>
    /// <param name="factory">The concrete-proxy factory.</param>
    /// <param name="lifetime">The lifetime shared by both registrations.</param>
    /// <returns>The same <paramref name="services" />, for chaining.</returns>
    private static IServiceCollection Add(
        IServiceCollection services,
        Func<IServiceProvider, GrpcStorageProxy> factory,
        ServiceLifetime lifetime)
    {
        services.Add(new ServiceDescriptor(typeof(GrpcStorageProxy), factory, lifetime));
        services.Add(new ServiceDescriptor(typeof(BaseStorage), sp => sp.GetRequiredService<GrpcStorageProxy>(), lifetime));
        return services;
    }
}
