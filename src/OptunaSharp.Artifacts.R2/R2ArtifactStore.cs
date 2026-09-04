using System.IO;
using Amazon.S3;

namespace OptunaSharp.Artifacts;

/// <summary>
///     An artifact backend for Cloudflare R2. <b>OptunaSharp extension</b> — Optuna itself ships no
///     R2 backend.
///
///     R2 exposes an S3-compatible API and Cloudflare publishes no official .NET SDK for it; their
///     guidance is to use the AWS S3 SDK against <c>https://&lt;accountId&gt;.r2.cloudflarestorage.com</c>.
///     This store therefore delegates to <see cref="Boto3ArtifactStore" /> (the
///     <c>OptunaSharp.Artifacts.S3</c> backend) with an S3 client pre-configured for the R2 endpoint.
/// </summary>
/// <example>
///     <code>
///     var artifactStore = new R2ArtifactStore(
///         accountId: "&lt;cloudflare-account-id&gt;",
///         accessKeyId: "&lt;r2-access-key-id&gt;",
///         secretAccessKey: "&lt;r2-secret-access-key&gt;",
///         bucketName: "my-bucket");
///     ArtifactFunctions.UploadArtifact(artifactStore, filePath, trial);
///     </code>
/// </example>
public class R2ArtifactStore : IArtifactStore
{
    private readonly Boto3ArtifactStore _inner;

    /// <summary>Build an R2 store from an account id and R2 S3-API-token credentials.</summary>
    /// <param name="accountId">The Cloudflare account id (the R2 endpoint subdomain).</param>
    /// <param name="accessKeyId">The R2 access key id.</param>
    /// <param name="secretAccessKey">The R2 secret access key.</param>
    /// <param name="bucketName">The name of the R2 bucket to store artifacts.</param>
    /// <param name="avoidBufCopy">See <see cref="Boto3ArtifactStore" />.</param>
    public R2ArtifactStore(
        string accountId,
        string accessKeyId,
        string secretAccessKey,
        string bucketName,
        bool avoidBufCopy = false)
        : this(CreateClient(accountId, accessKeyId, secretAccessKey), bucketName, avoidBufCopy)
    {
    }

    /// <summary>
    ///     Build an R2 store from a pre-configured S3 client — for full control over the endpoint,
    ///     credentials, or S3 config.
    /// </summary>
    /// <param name="client">An S3 client already configured for the R2 endpoint.</param>
    /// <param name="bucketName">The name of the R2 bucket to store artifacts.</param>
    /// <param name="avoidBufCopy">See <see cref="Boto3ArtifactStore" />.</param>
    public R2ArtifactStore(IAmazonS3 client, string bucketName, bool avoidBufCopy = false)
    {
        _inner = new Boto3ArtifactStore(bucketName, client, avoidBufCopy);
    }

    private static AmazonS3Client CreateClient(string accountId, string accessKeyId, string secretAccessKey)
    {
        var config = new AmazonS3Config
        {
            ServiceURL = $"https://{accountId}.r2.cloudflarestorage.com",
            // R2 addresses buckets path-style off the account endpoint.
            ForcePathStyle = true,
            // R2 ignores the region, but the SigV4 signer needs one; Cloudflare's docs use "auto".
            AuthenticationRegion = "auto",
        };
        return new AmazonS3Client(accessKeyId, secretAccessKey, config);
    }

    public Stream OpenReader(string artifactId) => _inner.OpenReader(artifactId);

    public void Write(string artifactId, Stream contentBody) => _inner.Write(artifactId, contentBody);

    public void Remove(string artifactId) => _inner.Remove(artifactId);
}
