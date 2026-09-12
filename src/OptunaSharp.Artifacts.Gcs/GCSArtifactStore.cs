using System.IO;
using System.Net;
using Google;
using Google.Cloud.Storage.V1;

namespace OptunaSharp.Artifacts;

/// <summary>
///     An artifact backend for Google Cloud Storage (GCS).
///     Port of <c>optuna/artifacts/_gcs.py: GCSArtifactStore</c>.
///     (Marked <c>@experimental_class("3.4.0")</c> in Optuna.)
/// </summary>
/// <example>
///     <code>
///     var artifactStore = new GCSArtifactStore("my-bucket");
///     ArtifactFunctions.UploadArtifact(artifactStore, filePath, trial);
///     </code>
///     Before running, install the Google Cloud SDK and run
///     <c>gcloud auth application-default login</c> so the client can find credentials.
/// </example>
public class GCSArtifactStore : IArtifactStore
{
    private readonly string _bucketName;
    private readonly StorageClient _client;

    /// <summary>Creates a GCS artifact store using a supplied client or Application Default Credentials.</summary>
    /// <param name="bucketName">The name of the bucket to store artifacts.</param>
    /// <param name="client">
    ///     A <see cref="StorageClient" /> to use for storage operations. If not specified, a new
    ///     one is created with default settings (Application Default Credentials).
    /// </param>
    public GCSArtifactStore(string bucketName, StorageClient? client = null)
    {
        _bucketName = bucketName;
        _client = client ?? StorageClient.Create();
    }

    /// <summary>Downloads an artifact into an independent, seekable memory stream.</summary>
    /// <param name="artifactId">The GCS object name assigned to the artifact.</param>
    /// <returns>A stream positioned at zero; the caller must dispose it.</returns>
    /// <exception cref="ArtifactNotFound">GCS returns an HTTP 404 response.</exception>
    /// <exception cref="GoogleApiException">GCS rejects the download for another reason.</exception>
    public Stream OpenReader(string artifactId)
    {
        // Python downloads the whole blob (download_as_bytes) and wraps it in a BytesIO; mirror
        // that with an in-memory buffer so the returned stream is seekable and self-contained.
        var body = new MemoryStream();
        try
        {
            _client.DownloadObject(_bucketName, artifactId, body);
        }
        catch (GoogleApiException e) when (e.HttpStatusCode == HttpStatusCode.NotFound)
        {
            body.Dispose();
            throw new ArtifactNotFound(
                $"Artifact storage with bucket: {_bucketName}, artifact_id: {artifactId} was not found", e);
        }

        body.Seek(0, SeekOrigin.Begin);
        return body;
    }

    /// <summary>Uploads source bytes to the artifact's GCS object.</summary>
    /// <param name="artifactId">The destination object name.</param>
    /// <param name="contentBody">A readable source stream, owned and disposed by the caller.</param>
    /// <exception cref="GoogleApiException">GCS rejects the upload.</exception>
    /// <exception cref="IOException">Reading the source stream fails.</exception>
    public void Write(string artifactId, Stream contentBody)
    {
        // Optuna calls blob.upload_from_string(data) (content_type defaults to text/plain in
        // google-cloud-storage). The stored object's content-type is not observable through the
        // IArtifactStore round-trip, so the port passes null and lets GCS default it — a
        // deliberate, invisible divergence.
        _client.UploadObject(_bucketName, artifactId, contentType: null, source: contentBody);
    }

    /// <summary>Deletes an artifact from GCS.</summary>
    /// <param name="artifactId">The object name to delete.</param>
    /// <exception cref="GoogleApiException">The object does not exist, or GCS rejects the request for another reason.</exception>
    public void Remove(string artifactId)
    {
        // Matches Optuna: delete_blob raises when the object is missing (google's NotFound). The
        // port lets the SDK's GoogleApiException propagate rather than mapping it to ArtifactNotFound.
        _client.DeleteObject(_bucketName, artifactId);
    }
}
