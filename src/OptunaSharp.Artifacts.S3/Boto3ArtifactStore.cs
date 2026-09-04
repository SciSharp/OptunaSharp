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

    public Stream OpenReader(string artifactId)
    {
        try
        {
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

    public void Remove(string artifactId)
    {
        // S3 delete_object is idempotent — deleting a missing key succeeds (no ArtifactNotFound),
        // matching boto3's behavior.
        _client.DeleteObjectAsync(new DeleteObjectRequest { BucketName = _bucket, Key = artifactId })
            .ConfigureAwait(false).GetAwaiter().GetResult();
    }

    /// <summary>Port of <c>optuna/artifacts/_boto3.py: _is_not_found_error</c>.</summary>
    private static bool IsNotFoundError(AmazonS3Exception e) =>
        e.ErrorCode == "NoSuchKey" || e.StatusCode == HttpStatusCode.NotFound;
}
