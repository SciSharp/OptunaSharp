using System.IO;
using System.Net;
using Amazon.S3;
using Amazon.S3.Model;

namespace OptunaSharp.Artifacts;

/// <summary>
///     An artifact backend for Amazon S3 (and S3-compatible object stores), backed by the AWS SDK
///     for .NET. Port of <c>optuna/artifacts/_boto3.py: Boto3ArtifactStore</c>.
///
///     The Python class is named after boto3 (the AWS SDK for Python); the C# port keeps the
///     <c>Boto3ArtifactStore</c> name so the two cross-reference, even though this package is
///     <c>OptunaSharp.Artifacts.S3</c> and it is built on the AWS SDK for .NET.
/// </summary>
/// <example>
///     <code>
///     var artifactStore = new Boto3ArtifactStore("my-bucket");
///     // inside an objective:
///     ArtifactFunctions.UploadArtifact(artifactStore, filePath, trial);
///     </code>
/// </example>
public class Boto3ArtifactStore : IArtifactStore
{
    private readonly string _bucket;
    private readonly IAmazonS3 _client;
    private readonly bool _avoidBufCopy;

    /// <summary>Creates an S3 artifact store using the supplied client or the AWS default client configuration.</summary>
    /// <param name="bucketName">The name of the bucket to store artifacts.</param>
    /// <param name="client">
    ///     An S3 client to use for storage operations. If not specified, a new
    ///     <see cref="AmazonS3Client" /> is created with the ambient AWS credentials/region
    ///     (the analog of Python <c>boto3.client("s3")</c>).
    /// </param>
    /// <param name="avoidBufCopy">
    ///     If <c>true</c>, skip copying the source stream into a buffer before uploading. Mirrors
    ///     Python's <c>avoid_buf_copy</c> (there, a workaround for boto3 closing the source file
    ///     object). The port never lets the SDK close the caller's stream (see
    ///     <c>AutoCloseStream = false</c> in <see cref="Write" />), so the copy is a faithfulness
    ///     detail rather than a necessity; it defaults to <c>false</c> as in Optuna.
    /// </param>
    public Boto3ArtifactStore(string bucketName, IAmazonS3? client = null, bool avoidBufCopy = false)
    {
        _bucket = bucketName;
        _client = client ?? new AmazonS3Client();
        _avoidBufCopy = avoidBufCopy;
    }

    /// <summary>Opens the S3 object's response stream for reading.</summary>
    /// <param name="artifactId">The object key assigned to the artifact.</param>
    /// <returns>The response stream; the caller must dispose it after reading.</returns>
    /// <exception cref="ArtifactNotFound">S3 reports a missing key or an HTTP 404 response.</exception>
    /// <exception cref="AmazonS3Exception">S3 rejects the download for another reason, such as authorization or a service failure.</exception>
    public Stream OpenReader(string artifactId)
    {
        try
        {
            // IArtifactStore exposes synchronous operations. Unwrap the SDK task directly so callers
            // receive the original provider exception and can apply Optuna's missing-artifact handling.
            GetObjectResponse obj = _client
                .GetObjectAsync(new GetObjectRequest { BucketName = _bucket, Key = artifactId })
                .ConfigureAwait(false).GetAwaiter().GetResult();
            return obj.ResponseStream;
        }
        catch (AmazonS3Exception e) when (IsNotFoundError(e))
        {
            throw new ArtifactNotFound(
                $"Artifact storage with bucket: {_bucket}, artifact_id: {artifactId} was not found", e);
        }
    }

    /// <summary>Uploads bytes from the source's current position, replacing an existing object with the same key.</summary>
    /// <param name="artifactId">The destination object key.</param>
    /// <param name="contentBody">The readable source stream, which remains owned by the caller.</param>
    /// <exception cref="AmazonS3Exception">S3 rejects the upload.</exception>
    /// <exception cref="IOException">Reading the source stream fails.</exception>
    /// <exception cref="System.ObjectDisposedException">The source stream has already been disposed.</exception>
    /// <exception cref="System.NotSupportedException">The source stream does not support reading.</exception>
    public void Write(string artifactId, Stream contentBody)
    {
        Stream fsrc = contentBody;
        MemoryStream? buf = null;
        if (!_avoidBufCopy)
        {
            buf = new MemoryStream();
            contentBody.CopyTo(buf);
            buf.Seek(0, SeekOrigin.Begin);
            fsrc = buf;
        }

        try
        {
            var request = new PutObjectRequest
            {
                BucketName = _bucket,
                Key = artifactId,
                InputStream = fsrc,
                // Never let the SDK dispose the caller's stream — the caller (or the buffer copy
                // above) owns it. This is the .NET-side answer to boto3 issue #929.
                AutoCloseStream = false,
            };
            _client.PutObjectAsync(request).ConfigureAwait(false).GetAwaiter().GetResult();
        }
        finally
        {
            buf?.Dispose();
        }
    }

    /// <summary>Deletes the object; an already absent key is treated as success by S3.</summary>
    /// <param name="artifactId">The object key to delete.</param>
    /// <exception cref="AmazonS3Exception">S3 rejects the delete request.</exception>
    public void Remove(string artifactId)
    {
        // S3 delete_object is idempotent — deleting a missing key succeeds (no ArtifactNotFound),
        // matching boto3's behavior.
        _client.DeleteObjectAsync(new DeleteObjectRequest { BucketName = _bucket, Key = artifactId })
            .ConfigureAwait(false).GetAwaiter().GetResult();
    }

    /// <summary>Port of <c>optuna/artifacts/_boto3.py: _is_not_found_error</c>.</summary>
    /// <param name="e">The SDK failure to classify.</param>
    /// <returns>Whether the error denotes a missing artifact.</returns>
    private static bool IsNotFoundError(AmazonS3Exception e) =>
        e.ErrorCode == "NoSuchKey" || e.StatusCode == HttpStatusCode.NotFound;
}
