using System;
using Amazon.S3;
using Microsoft.Extensions.DependencyInjection;

namespace OptunaSharp.Artifacts.Extensions;

/// <summary>
///     Microsoft.Extensions.DependencyInjection wire-up for the Amazon S3 artifact store
///     (<see cref="Boto3ArtifactStore" />). Registers it as both its concrete type and the
///     <see cref="IArtifactStore" /> abstraction, so a consumer can inject <c>IArtifactStore</c> and
///     stay backend-agnostic. The class name is package-unique (the four artifact backends share the
///     <c>OptunaSharp.Artifacts.Extensions</c> namespace) so referencing several at once does not clash.
///     <para>
///     <b>Pairs with the AWS SDK's own DI:</b> call <c>services.AddAWSService&lt;IAmazonS3&gt;()</c> and
///     then <see cref="AddS3ArtifactStore(IServiceCollection, string, ServiceLifetime)" />, which resolves
///     that registered client — credentials/region come from your AWS options, not from this call.
///     </para>
/// </summary>
public static class S3ArtifactStoreServiceCollectionExtensions
{
    /// <summary>
    ///     Register an S3 artifact store for <paramref name="bucketName" />, resolving an
    ///     <see cref="IAmazonS3" /> client from the container when one is registered (else the store
    ///     constructs the SDK default). Registered as <see cref="Boto3ArtifactStore" /> and
    ///     <see cref="IArtifactStore" />.
    /// </summary>
    /// <param name="services">The service collection to add to.</param>
    /// <param name="bucketName">The destination S3 bucket for artifacts.</param>
    /// <param name="lifetime">The registration lifetime; defaults to <see cref="ServiceLifetime.Singleton" />.</param>
    /// <returns>The same <paramref name="services" />, for chaining.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="services" /> or <paramref name="bucketName" /> is <c>null</c>.</exception>
    public static IServiceCollection AddS3ArtifactStore(
        this IServiceCollection services,
        string bucketName,
        ServiceLifetime lifetime = ServiceLifetime.Singleton)
    {
        ArgumentNullException.ThrowIfNull(services);
        ArgumentNullException.ThrowIfNull(bucketName);
        // GetService (not GetRequiredService) for the client: a null client is valid — Boto3ArtifactStore
        // then falls back to the AWS SDK's default client construction, matching its own constructor default.
        return Add(services, sp => new Boto3ArtifactStore(bucketName, sp.GetService<IAmazonS3>()), lifetime);
    }

    /// <summary>Register a pre-built <see cref="Boto3ArtifactStore" /> instance as the app's <see cref="IArtifactStore" /> singleton.</summary>
    /// <param name="services">The service collection to add to.</param>
    /// <param name="store">The artifact store instance to share.</param>
    /// <returns>The same <paramref name="services" />, for chaining.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="services" /> or <paramref name="store" /> is <c>null</c>.</exception>
    public static IServiceCollection AddS3ArtifactStore(this IServiceCollection services, Boto3ArtifactStore store)
    {
        ArgumentNullException.ThrowIfNull(services);
        ArgumentNullException.ThrowIfNull(store);
        services.AddSingleton(store);
        services.AddSingleton<IArtifactStore>(store);
        return services;
    }

    /// <summary>Register a <see cref="Boto3ArtifactStore" /> produced by a factory (full control over bucket, client, and buffering).</summary>
    /// <param name="services">The service collection to add to.</param>
    /// <param name="factory">Builds the store from the resolved <see cref="IServiceProvider" />.</param>
    /// <param name="lifetime">The registration lifetime; defaults to <see cref="ServiceLifetime.Singleton" />.</param>
    /// <returns>The same <paramref name="services" />, for chaining.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="services" /> or <paramref name="factory" /> is <c>null</c>.</exception>
    public static IServiceCollection AddS3ArtifactStore(
        this IServiceCollection services,
        Func<IServiceProvider, Boto3ArtifactStore> factory,
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
        Func<IServiceProvider, Boto3ArtifactStore> factory,
        ServiceLifetime lifetime)
    {
        services.Add(new ServiceDescriptor(typeof(Boto3ArtifactStore), factory, lifetime));
        // Forward the abstraction to the concrete registration so both resolve the SAME instance for a
        // singleton (and consistently per-scope otherwise).
        services.Add(new ServiceDescriptor(typeof(IArtifactStore), sp => sp.GetRequiredService<Boto3ArtifactStore>(), lifetime));
        return services;
    }
}
