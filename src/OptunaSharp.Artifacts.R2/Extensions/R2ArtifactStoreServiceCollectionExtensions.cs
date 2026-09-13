using System;
using Amazon.S3;
using Microsoft.Extensions.DependencyInjection;

namespace OptunaSharp.Artifacts.Extensions;

/// <summary>
///     Microsoft.Extensions.DependencyInjection wire-up for the Cloudflare R2 artifact store
///     (<see cref="R2ArtifactStore" />). Registers it as both its concrete type and the
///     <see cref="IArtifactStore" /> abstraction, with overloads for the account-credentials path and
///     the bring-your-own-<see cref="IAmazonS3" />-client path. The class name is package-unique (the
///     four artifact backends share the <c>OptunaSharp.Artifacts.Extensions</c> namespace) so referencing
///     several at once does not clash.
/// </summary>
public static class R2ArtifactStoreServiceCollectionExtensions
{
    /// <summary>Register an R2 artifact store from a Cloudflare account id and R2 S3-API-token credentials.</summary>
    /// <param name="services">The service collection to add to.</param>
    /// <param name="accountId">The Cloudflare account id (the R2 endpoint subdomain).</param>
    /// <param name="accessKeyId">The R2 access key id.</param>
    /// <param name="secretAccessKey">The R2 secret access key.</param>
    /// <param name="bucketName">The R2 bucket that holds artifacts.</param>
    /// <param name="avoidBufCopy">Passed through to the inner <see cref="Boto3ArtifactStore" /> buffering behavior.</param>
    /// <param name="lifetime">The registration lifetime; defaults to <see cref="ServiceLifetime.Singleton" />.</param>
    /// <returns>The same <paramref name="services" />, for chaining.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="services" />, <paramref name="accountId" />, <paramref name="accessKeyId" />, <paramref name="secretAccessKey" /> or <paramref name="bucketName" /> is <c>null</c>.</exception>
    public static IServiceCollection AddR2ArtifactStore(
        this IServiceCollection services,
        string accountId,
        string accessKeyId,
        string secretAccessKey,
        string bucketName,
        bool avoidBufCopy = false,
        ServiceLifetime lifetime = ServiceLifetime.Singleton)
    {
        ArgumentNullException.ThrowIfNull(services);
        ArgumentNullException.ThrowIfNull(accountId);
        ArgumentNullException.ThrowIfNull(accessKeyId);
        ArgumentNullException.ThrowIfNull(secretAccessKey);
        ArgumentNullException.ThrowIfNull(bucketName);
        return Add(services, _ => new R2ArtifactStore(accountId, accessKeyId, secretAccessKey, bucketName, avoidBufCopy), lifetime);
    }

    /// <summary>
    ///     Register an R2 artifact store from a pre-configured, R2-endpoint <see cref="IAmazonS3" /> client
    ///     (full control over endpoint, credentials, and S3 config).
    /// </summary>
    /// <param name="services">The service collection to add to.</param>
    /// <param name="client">An S3 client already configured for the R2 endpoint.</param>
    /// <param name="bucketName">The R2 bucket that holds artifacts.</param>
    /// <param name="avoidBufCopy">Passed through to the inner <see cref="Boto3ArtifactStore" /> buffering behavior.</param>
    /// <param name="lifetime">The registration lifetime; defaults to <see cref="ServiceLifetime.Singleton" />.</param>
    /// <returns>The same <paramref name="services" />, for chaining.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="services" />, <paramref name="client" /> or <paramref name="bucketName" /> is <c>null</c>.</exception>
    public static IServiceCollection AddR2ArtifactStore(
        this IServiceCollection services,
        IAmazonS3 client,
        string bucketName,
        bool avoidBufCopy = false,
        ServiceLifetime lifetime = ServiceLifetime.Singleton)
    {
        ArgumentNullException.ThrowIfNull(services);
        ArgumentNullException.ThrowIfNull(client);
        ArgumentNullException.ThrowIfNull(bucketName);
        return Add(services, _ => new R2ArtifactStore(client, bucketName, avoidBufCopy), lifetime);
    }

    /// <summary>Register a pre-built <see cref="R2ArtifactStore" /> instance as the app's <see cref="IArtifactStore" /> singleton.</summary>
    /// <param name="services">The service collection to add to.</param>
    /// <param name="store">The artifact store instance to share.</param>
    /// <returns>The same <paramref name="services" />, for chaining.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="services" /> or <paramref name="store" /> is <c>null</c>.</exception>
    public static IServiceCollection AddR2ArtifactStore(this IServiceCollection services, R2ArtifactStore store)
    {
        ArgumentNullException.ThrowIfNull(services);
        ArgumentNullException.ThrowIfNull(store);
        services.AddSingleton(store);
        services.AddSingleton<IArtifactStore>(store);
        return services;
    }

    /// <summary>Register an <see cref="R2ArtifactStore" /> produced by a factory (full control over construction).</summary>
    /// <param name="services">The service collection to add to.</param>
    /// <param name="factory">Builds the store from the resolved <see cref="IServiceProvider" />.</param>
    /// <param name="lifetime">The registration lifetime; defaults to <see cref="ServiceLifetime.Singleton" />.</param>
    /// <returns>The same <paramref name="services" />, for chaining.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="services" /> or <paramref name="factory" /> is <c>null</c>.</exception>
    public static IServiceCollection AddR2ArtifactStore(
        this IServiceCollection services,
        Func<IServiceProvider, R2ArtifactStore> factory,
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
        Func<IServiceProvider, R2ArtifactStore> factory,
        ServiceLifetime lifetime)
    {
        services.Add(new ServiceDescriptor(typeof(R2ArtifactStore), factory, lifetime));
        services.Add(new ServiceDescriptor(typeof(IArtifactStore), sp => sp.GetRequiredService<R2ArtifactStore>(), lifetime));
        return services;
    }
}
