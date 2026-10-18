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
///     Unit tests for the Cloudflare R2 backend (<see cref="R2ArtifactStore" />). R2 delegates to
///     <see cref="Boto3ArtifactStore" /> over an S3-compatible endpoint; these pin the R2 endpoint
///     configuration and confirm the wrapper forwards to the injected S3 client. Live round-trips are
///     in <see cref="R2IntegrationTests" /> (against the S3-compatible emulator).
/// </summary>
[TestClass]
public class R2ArtifactStoreTests
{
    [TestMethod]
    public void BuildR2Config_PinsEndpoint_Region_And_PathStyle()
    {
        AmazonS3Config cfg = R2ArtifactStore.BuildR2Config("my-account-id");

        // AWSSDK normalizes ServiceURL with a trailing slash, so pin the host with StartWith.
        cfg.ServiceURL.Should().StartWith("https://my-account-id.r2.cloudflarestorage.com");
        cfg.ForcePathStyle.Should().BeTrue();
        cfg.AuthenticationRegion.Should().Be("auto");
    }

    [TestMethod]
    public void CredentialCtor_Constructs_WithoutNetwork()
    {
        // Exercises the account-id/credentials ctor (which builds the R2 client via BuildR2Config).
        var store = new R2ArtifactStore("acct", "access-key", "secret-key", "r2-bucket");

        store.Should().NotBeNull();
    }

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
    public void Write_AvoidBufCopyTrue_ForwardsToInner()
    {
        var source = new MemoryStream(new byte[] { 7, 7 });
        PutObjectRequest? captured = null;
        var client = new Mock<IAmazonS3>();
        client.Setup(c => c.PutObjectAsync(It.IsAny<PutObjectRequest>(), It.IsAny<CancellationToken>()))
              .Callback<PutObjectRequest, CancellationToken>((r, _) => captured = r)
              .ReturnsAsync(new PutObjectResponse());
        var store = new R2ArtifactStore(client.Object, "r2-bucket", avoidBufCopy: true);

        store.Write("art-2", source);

        captured!.InputStream.Should().BeSameAs(source, "avoidBufCopy must flow through to the inner S3 store");
    }

    [TestMethod]
    public void OpenReader_ReturnsContent_OnSuccess()
    {
        byte[] data = { 5, 6, 7 };
        var client = new Mock<IAmazonS3>();
        client.Setup(c => c.GetObjectAsync(It.IsAny<GetObjectRequest>(), It.IsAny<CancellationToken>()))
              .ReturnsAsync(new GetObjectResponse { ResponseStream = new MemoryStream(data) });
        var store = new R2ArtifactStore(client.Object, "r2-bucket");

        using var reader = store.OpenReader("art-3");
        using var ms = new MemoryStream();
        reader.CopyTo(ms);

        ms.ToArray().Should().Equal(data);
    }

    [TestMethod]
    public void OpenReader_NotFound_ThrowsArtifactNotFound()
    {
        var client = new Mock<IAmazonS3>();
        client.Setup(c => c.GetObjectAsync(It.IsAny<GetObjectRequest>(), It.IsAny<CancellationToken>()))
              .ThrowsAsync(new AmazonS3Exception("err", ErrorType.Sender, "NoSuchKey", "req", HttpStatusCode.NotFound));
        var store = new R2ArtifactStore(client.Object, "r2-bucket");

        Action act = () => store.OpenReader("missing");

        act.Should().Throw<ArtifactNotFound>();
    }

    [TestMethod]
    public void Remove_DelegatesToConfiguredS3Client()
    {
        DeleteObjectRequest? captured = null;
        var client = new Mock<IAmazonS3>();
        client.Setup(c => c.DeleteObjectAsync(It.IsAny<DeleteObjectRequest>(), It.IsAny<CancellationToken>()))
              .Callback<DeleteObjectRequest, CancellationToken>((r, _) => captured = r)
              .ReturnsAsync(new DeleteObjectResponse());
        var store = new R2ArtifactStore(client.Object, "r2-bucket");

        store.Remove("art-4");

        captured!.BucketName.Should().Be("r2-bucket");
        captured.Key.Should().Be("art-4");
    }
}
