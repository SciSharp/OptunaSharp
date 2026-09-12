using System;
using System.IO;
using Azure;
using Azure.Core;
using Azure.Identity;
using Azure.Storage.Blobs;

namespace OptunaSharp.Artifacts;

/// <summary>
///     An artifact backend for Azure Blob Storage. <b>OptunaSharp extension</b> — Optuna itself
///     ships no Azure backend. Backed by <c>Azure.Storage.Blobs</c>.
///
///     A blob container is the analog of an S3/GCS bucket. Three construction paths are offered:
///     a caller-supplied <see cref="BlobContainerClient" /> (the analog of Optuna's "pass your own
///     client"), a storage connection string, and a service endpoint with a Microsoft Entra ID
///     credential.
/// </summary>
/// <example>
///     <code>
///     var artifactStore = new AzureBlobArtifactStore(connectionString, "my-container");
///     ArtifactFunctions.UploadArtifact(artifactStore, filePath, trial);
///     </code>
/// </example>
public class AzureBlobArtifactStore : IArtifactStore
{
    private readonly BlobContainerClient _container;

    /// <summary>Use a caller-supplied container client (full control over auth and configuration).</summary>
    /// <param name="container">The container client retained for subsequent requests.</param>
    public AzureBlobArtifactStore(BlobContainerClient container)
    {
        _container = container;
    }

    /// <summary>Connect with a storage account connection string.</summary>
    /// <param name="connectionString">An Azure Storage connection string.</param>
    /// <param name="containerName">The name of the blob container to store artifacts.</param>
    public AzureBlobArtifactStore(string connectionString, string containerName)
    {
        _container = new BlobServiceClient(connectionString).GetBlobContainerClient(containerName);
    }

    /// <summary>
    ///     Connect to a blob service endpoint with a Microsoft Entra ID credential. When
    ///     <paramref name="credential" /> is <c>null</c> a <see cref="DefaultAzureCredential" /> is
    ///     used (managed identity, environment, Azure CLI, ... — the analog of GCS Application
    ///     Default Credentials / boto3's default credential chain).
    /// </summary>
    /// <param name="serviceUri">The blob service endpoint, e.g. <c>https://&lt;account&gt;.blob.core.windows.net</c>.</param>
    /// <param name="containerName">The name of the blob container to store artifacts.</param>
    /// <param name="credential">A token credential; defaults to <see cref="DefaultAzureCredential" />.</param>
    public AzureBlobArtifactStore(Uri serviceUri, string containerName, TokenCredential? credential = null)
    {
        var service = new BlobServiceClient(serviceUri, credential ?? new DefaultAzureCredential());
        _container = service.GetBlobContainerClient(containerName);
    }

    /// <summary>Opens a streaming download of the artifact blob.</summary>
    /// <param name="artifactId">The blob name assigned to the artifact.</param>
    /// <returns>The download stream, which the caller must dispose.</returns>
    /// <exception cref="ArtifactNotFound">Azure returns HTTP 404 for the blob or container.</exception>
    /// <exception cref="RequestFailedException">Azure rejects the request for another reason.</exception>
    public Stream OpenReader(string artifactId)
    {
        BlobClient blob = _container.GetBlobClient(artifactId);
        try
        {
            return blob.DownloadStreaming().Value.Content;
        }
        catch (RequestFailedException e) when (e.Status == 404)
        {
            throw new ArtifactNotFound(
                $"Artifact storage with container: {_container.Name}, artifact_id: {artifactId} was not found", e);
        }
    }

    /// <summary>Uploads source bytes, overwriting a blob that already uses the artifact name.</summary>
    /// <param name="artifactId">The destination blob name.</param>
    /// <param name="contentBody">The readable source stream; its lifetime remains the caller's responsibility.</param>
    /// <exception cref="RequestFailedException">Azure rejects the upload.</exception>
    /// <exception cref="IOException">Reading the source stream fails.</exception>
    public void Write(string artifactId, Stream contentBody)
    {
        BlobClient blob = _container.GetBlobClient(artifactId);
        blob.Upload(contentBody, overwrite: true);
    }

    /// <summary>Deletes the artifact blob.</summary>
    /// <param name="artifactId">The blob name to delete.</param>
    /// <exception cref="ArtifactNotFound">Azure returns HTTP 404 for the blob or container.</exception>
    /// <exception cref="RequestFailedException">Azure rejects the delete request for another reason.</exception>
    public void Remove(string artifactId)
    {
        BlobClient blob = _container.GetBlobClient(artifactId);
        try
        {
            blob.Delete();
        }
        catch (RequestFailedException e) when (e.Status == 404)
        {
            throw new ArtifactNotFound(
                $"Artifact storage with container: {_container.Name}, artifact_id: {artifactId} was not found", e);
        }
    }
}
