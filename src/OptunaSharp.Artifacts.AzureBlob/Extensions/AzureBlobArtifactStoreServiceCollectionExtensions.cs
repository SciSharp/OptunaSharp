using System;
using Azure.Core;
using Azure.Storage.Blobs;
using Microsoft.Extensions.DependencyInjection;

namespace OptunaSharp.Artifacts.Extensions;

/// <summary>
///     Microsoft.Extensions.DependencyInjection wire-up for the Azure Blob Storage artifact store
///     (<see cref="AzureBlobArtifactStore" />). Registers it as both its concrete type and the
///     <see cref="IArtifactStore" /> abstraction, with overloads for each of the store's three
///     construction paths (connection string, service URI + credential, caller-supplied container) plus
///     a path that resolves a <see cref="BlobContainerClient" /> already registered by
///     <c>Microsoft.Extensions.Azure</c>'s <c>AddAzureClients</c>. The class name is package-unique (the
///     four artifact backends share the <c>OptunaSharp.Artifacts.Extensions</c> namespace) so referencing
///     several at once does not clash.
/// </summary>
public static class AzureBlobArtifactStoreServiceCollectionExtensions
{
    /// <summary>
    ///     Register an Azure Blob artifact store built from a container resolved out of the container —
    ///     pair with <c>services.AddAzureClients(b =&gt; b.AddBlobServiceClient(...))</c> and a keyed/typed
    ///     <see cref="BlobContainerClient" />. The client is resolved with <c>GetRequiredService</c>, so
    ///     this throws at resolution time if none is registered.
    /// </summary>
    /// <param name="services">The service collection to add to.</param>
    /// <param name="lifetime">The registration lifetime; defaults to <see cref="ServiceLifetime.Singleton" />.</param>
    /// <returns>The same <paramref name="services" />, for chaining.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="services" /> is <c>null</c>.</exception>
    public static IServiceCollection AddAzureBlobArtifactStore(
        this IServiceCollection services,
        ServiceLifetime lifetime = ServiceLifetime.Singleton)
    {
        ArgumentNullException.ThrowIfNull(services);
        return Add(services, sp => new AzureBlobArtifactStore(sp.GetRequiredService<BlobContainerClient>()), lifetime);
    }

    /// <summary>Register an Azure Blob artifact store from a storage-account connection string.</summary>
    /// <param name="services">The service collection to add to.</param>
    /// <param name="connectionString">An Azure Storage connection string.</param>
    /// <param name="containerName">The blob container that holds artifacts.</param>
    /// <param name="lifetime">The registration lifetime; defaults to <see cref="ServiceLifetime.Singleton" />.</param>
    /// <returns>The same <paramref name="services" />, for chaining.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="services" />, <paramref name="connectionString" /> or <paramref name="containerName" /> is <c>null</c>.</exception>
    public static IServiceCollection AddAzureBlobArtifactStore(
        this IServiceCollection services,
        string connectionString,
        string containerName,
        ServiceLifetime lifetime = ServiceLifetime.Singleton)
    {
        ArgumentNullException.ThrowIfNull(services);
        ArgumentNullException.ThrowIfNull(connectionString);
        ArgumentNullException.ThrowIfNull(containerName);
        return Add(services, _ => new AzureBlobArtifactStore(connectionString, containerName), lifetime);
    }

    /// <summary>
    ///     Register an Azure Blob artifact store from a service endpoint and a Microsoft Entra ID
    ///     credential; a <c>null</c> credential uses <c>DefaultAzureCredential</c> (managed identity,
    ///     environment, Azure CLI, …).
    /// </summary>
    /// <param name="services">The service collection to add to.</param>
    /// <param name="serviceUri">The blob service endpoint, e.g. <c>https://&lt;account&gt;.blob.core.windows.net</c>.</param>
    /// <param name="containerName">The blob container that holds artifacts.</param>
    /// <param name="credential">A token credential; <c>null</c> selects <c>DefaultAzureCredential</c>.</param>
    /// <param name="lifetime">The registration lifetime; defaults to <see cref="ServiceLifetime.Singleton" />.</param>
    /// <returns>The same <paramref name="services" />, for chaining.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="services" />, <paramref name="serviceUri" /> or <paramref name="containerName" /> is <c>null</c>.</exception>
    public static IServiceCollection AddAzureBlobArtifactStore(
        this IServiceCollection services,
        Uri serviceUri,
        string containerName,
        TokenCredential? credential = null,
        ServiceLifetime lifetime = ServiceLifetime.Singleton)
    {
        ArgumentNullException.ThrowIfNull(services);
        ArgumentNullException.ThrowIfNull(serviceUri);
        ArgumentNullException.ThrowIfNull(containerName);
        return Add(services, _ => new AzureBlobArtifactStore(serviceUri, containerName, credential), lifetime);
    }

    /// <summary>Register an Azure Blob artifact store from a caller-supplied <see cref="BlobContainerClient" />.</summary>
    /// <param name="services">The service collection to add to.</param>
    /// <param name="container">The container client the store issues requests against.</param>
    /// <param name="lifetime">The registration lifetime; defaults to <see cref="ServiceLifetime.Singleton" />.</param>
    /// <returns>The same <paramref name="services" />, for chaining.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="services" /> or <paramref name="container" /> is <c>null</c>.</exception>
    public static IServiceCollection AddAzureBlobArtifactStore(
        this IServiceCollection services,
        BlobContainerClient container,
        ServiceLifetime lifetime = ServiceLifetime.Singleton)
    {
        ArgumentNullException.ThrowIfNull(services);
        ArgumentNullException.ThrowIfNull(container);
        return Add(services, _ => new AzureBlobArtifactStore(container), lifetime);
    }

    /// <summary>Register a pre-built <see cref="AzureBlobArtifactStore" /> instance as the app's <see cref="IArtifactStore" /> singleton.</summary>
    /// <param name="services">The service collection to add to.</param>
    /// <param name="store">The artifact store instance to share.</param>
    /// <returns>The same <paramref name="services" />, for chaining.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="services" /> or <paramref name="store" /> is <c>null</c>.</exception>
    public static IServiceCollection AddAzureBlobArtifactStore(this IServiceCollection services, AzureBlobArtifactStore store)
    {
        ArgumentNullException.ThrowIfNull(services);
        ArgumentNullException.ThrowIfNull(store);
        services.AddSingleton(store);
        services.AddSingleton<IArtifactStore>(store);
        return services;
    }

    /// <summary>Register an <see cref="AzureBlobArtifactStore" /> produced by a factory (full control over construction).</summary>
    /// <param name="services">The service collection to add to.</param>
    /// <param name="factory">Builds the store from the resolved <see cref="IServiceProvider" />.</param>
    /// <param name="lifetime">The registration lifetime; defaults to <see cref="ServiceLifetime.Singleton" />.</param>
    /// <returns>The same <paramref name="services" />, for chaining.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="services" /> or <paramref name="factory" /> is <c>null</c>.</exception>
    public static IServiceCollection AddAzureBlobArtifactStore(
        this IServiceCollection services,
        Func<IServiceProvider, AzureBlobArtifactStore> factory,
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
        Func<IServiceProvider, AzureBlobArtifactStore> factory,
        ServiceLifetime lifetime)
    {
        services.Add(new ServiceDescriptor(typeof(AzureBlobArtifactStore), factory, lifetime));
        services.Add(new ServiceDescriptor(typeof(IArtifactStore), sp => sp.GetRequiredService<AzureBlobArtifactStore>(), lifetime));
        return services;
    }
}
