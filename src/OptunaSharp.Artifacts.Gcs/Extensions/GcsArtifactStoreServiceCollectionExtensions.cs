using System;
using Google.Cloud.Storage.V1;
using Microsoft.Extensions.DependencyInjection;

namespace OptunaSharp.Artifacts.Extensions;

/// <summary>
///     Microsoft.Extensions.DependencyInjection wire-up for the Google Cloud Storage artifact store
///     (<see cref="GCSArtifactStore" />). Registers it as both its concrete type and the
///     <see cref="IArtifactStore" /> abstraction. The class name is package-unique (the four artifact
///     backends share the <c>OptunaSharp.Artifacts.Extensions</c> namespace) so referencing several at
///     once does not clash.
/// </summary>
public static class GcsArtifactStoreServiceCollectionExtensions
{
    /// <summary>
    ///     Register a GCS artifact store for <paramref name="bucketName" />, resolving a
    ///     <see cref="StorageClient" /> from the container when one is registered (else the store creates
    ///     one from Application Default Credentials). Registered as <see cref="GCSArtifactStore" /> and
    ///     <see cref="IArtifactStore" />.
    /// </summary>
    /// <param name="services">The service collection to add to.</param>
    /// <param name="bucketName">The destination GCS bucket for artifacts.</param>
    /// <param name="lifetime">The registration lifetime; defaults to <see cref="ServiceLifetime.Singleton" />.</param>
    /// <returns>The same <paramref name="services" />, for chaining.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="services" /> or <paramref name="bucketName" /> is <c>null</c>.</exception>
    public static IServiceCollection AddGcsArtifactStore(
        this IServiceCollection services,
        string bucketName,
        ServiceLifetime lifetime = ServiceLifetime.Singleton)
    {
        ArgumentNullException.ThrowIfNull(services);
        ArgumentNullException.ThrowIfNull(bucketName);
        // GetService (not GetRequiredService): a null client is valid — GCSArtifactStore then creates a
        // default StorageClient from ADC, matching its own constructor default.
        return Add(services, sp => new GCSArtifactStore(bucketName, sp.GetService<StorageClient>()), lifetime);
    }

    /// <summary>Register a pre-built <see cref="GCSArtifactStore" /> instance as the app's <see cref="IArtifactStore" /> singleton.</summary>
    /// <param name="services">The service collection to add to.</param>
    /// <param name="store">The artifact store instance to share.</param>
    /// <returns>The same <paramref name="services" />, for chaining.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="services" /> or <paramref name="store" /> is <c>null</c>.</exception>
    public static IServiceCollection AddGcsArtifactStore(this IServiceCollection services, GCSArtifactStore store)
    {
        ArgumentNullException.ThrowIfNull(services);
        ArgumentNullException.ThrowIfNull(store);
        services.AddSingleton(store);
        services.AddSingleton<IArtifactStore>(store);
        return services;
    }

    /// <summary>Register a <see cref="GCSArtifactStore" /> produced by a factory (full control over bucket and client).</summary>
    /// <param name="services">The service collection to add to.</param>
    /// <param name="factory">Builds the store from the resolved <see cref="IServiceProvider" />.</param>
    /// <param name="lifetime">The registration lifetime; defaults to <see cref="ServiceLifetime.Singleton" />.</param>
    /// <returns>The same <paramref name="services" />, for chaining.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="services" /> or <paramref name="factory" /> is <c>null</c>.</exception>
    public static IServiceCollection AddGcsArtifactStore(
        this IServiceCollection services,
        Func<IServiceProvider, GCSArtifactStore> factory,
        ServiceLifetime lifetime = ServiceLifetime.Singleton)
    {
        ArgumentNullException.ThrowIfNull(services);
        ArgumentNullException.ThrowIfNull(factory);
        return Add(services, factory, lifetime);
    }

    /// <summary>Registers the concrete store under the requested lifetime and forwards <see cref="IArtifactStore" /> to it.</summary>
    /// <param name="services">The service collection to add to.</param>
    /// <param name="factory">The concrete-store factory.</param>
    /// <param name="lifetime">The lifetime shared by both registrations.</param>
    /// <returns>The same <paramref name="services" />, for chaining.</returns>
    private static IServiceCollection Add(
        IServiceCollection services,
        Func<IServiceProvider, GCSArtifactStore> factory,
        ServiceLifetime lifetime)
    {
        services.Add(new ServiceDescriptor(typeof(GCSArtifactStore), factory, lifetime));
        services.Add(new ServiceDescriptor(typeof(IArtifactStore), sp => sp.GetRequiredService<GCSArtifactStore>(), lifetime));
        return services;
    }
}
