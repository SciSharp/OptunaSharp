using System;
using System.IO;
using System.Net;
using System.Threading;
using Amazon.Runtime;
using Amazon.S3;
using Amazon.S3.Model;
using AwesomeAssertions;
using Moq;
using OptunaSharp.Artifacts;

namespace OptunaSharp.Tests.Artifacts;

/// <summary>
///     Unit tests for the Cloudflare R2 backend (<see cref="R2ArtifactStore" />). R2 is an OptunaSharp
///     extension that delegates to <see cref="Boto3ArtifactStore" /> over an S3-compatible endpoint;
///     these confirm the wrapper delegates to the injected S3 client and surfaces the not-found mapping.
/// </summary>
[TestClass]
public class R2ArtifactStoreTests
{
    [TestMethod]
    public void Write_DelegatesToConfiguredS3Client()
    {
        PutObjectRequest? captured = null;
        var client = new Mock<IAmazonS3>();
        client.Setup(c => c.PutObjectAsync(It.IsAny<PutObjectRequest>(), It.IsAny<CancellationToken>()))
              .Callback<PutObjectRequest, CancellationToken>((r, _) => captured = r)
              .ReturnsAsync(new PutObjectResponse());
        var store = new R2ArtifactStore(client.Object, "r2-bucket");

        store.Write("art-1", new MemoryStream(new byte[] { 1, 2, 3 }));

        captured!.BucketName.Should().Be("r2-bucket");
        captured.Key.Should().Be("art-1");
    }

    [TestMethod]
    public void OpenReader_NotFound_ThrowsArtifactNotFound()
    {
        var client = new Mock<IAmazonS3>();
        client.Setup(c => c.GetObjectAsync(It.IsAny<GetObjectRequest>(), It.IsAny<CancellationToken>()))
              .ThrowsAsync(new AmazonS3Exception("nope", ErrorType.Sender, "NoSuchKey", "req", HttpStatusCode.NotFound));
        var store = new R2ArtifactStore(client.Object, "r2-bucket");

        Action act = () => store.OpenReader("missing");

        act.Should().Throw<ArtifactNotFound>();
    }

    [TestMethod]
    public void CredentialCtor_Constructs()
    {
        // The account-id/credentials ctor builds an S3 client pointed at the R2 endpoint; it must
        // construct without contacting the network.
        var store = new R2ArtifactStore("acct", "access-key", "secret-key", "r2-bucket");

        store.Should().NotBeNull();
    }
}
