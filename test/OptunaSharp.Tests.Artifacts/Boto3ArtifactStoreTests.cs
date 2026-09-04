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
///     Unit tests for the S3 backend (<see cref="Boto3ArtifactStore" />), driven by a mocked
///     <see cref="IAmazonS3" />. They pin the real logic of the port — the not-found mapping and the
///     get/put/delete delegation — without a live bucket. Live round-trips against S3 / LocalStack
///     are a deferred integration concern (see the plan §4.5).
/// </summary>
[TestClass]
public class Boto3ArtifactStoreTests
{
    [TestMethod]
    public void OpenReader_ReturnsContent_OnSuccess()
    {
        byte[] data = { 1, 2, 3, 4, 5 };
        var client = new Mock<IAmazonS3>();
        client.Setup(c => c.GetObjectAsync(It.IsAny<GetObjectRequest>(), It.IsAny<CancellationToken>()))
              .ReturnsAsync(new GetObjectResponse { ResponseStream = new MemoryStream(data) });
        var store = new Boto3ArtifactStore("bucket", client.Object);

        using var reader = store.OpenReader("art-1");
        using var ms = new MemoryStream();
        reader.CopyTo(ms);

        ms.ToArray().Should().Equal(data);
    }

    [TestMethod]
    public void OpenReader_NoSuchKey_ThrowsArtifactNotFound()
    {
        var client = new Mock<IAmazonS3>();
        client.Setup(c => c.GetObjectAsync(It.IsAny<GetObjectRequest>(), It.IsAny<CancellationToken>()))
              .ThrowsAsync(new AmazonS3Exception("nope", ErrorType.Sender, "NoSuchKey", "req", HttpStatusCode.NotFound));
        var store = new Boto3ArtifactStore("bucket", client.Object);

        Action act = () => store.OpenReader("missing");

        act.Should().Throw<ArtifactNotFound>();
    }

    [TestMethod]
    public void OpenReader_404WithoutNoSuchKey_ThrowsArtifactNotFound()
    {
        // _is_not_found_error also treats an HTTP 404 (any error code) as not-found.
        var client = new Mock<IAmazonS3>();
        client.Setup(c => c.GetObjectAsync(It.IsAny<GetObjectRequest>(), It.IsAny<CancellationToken>()))
              .ThrowsAsync(new AmazonS3Exception("gone", ErrorType.Sender, "NoSuchBucket", "req", HttpStatusCode.NotFound));
        var store = new Boto3ArtifactStore("bucket", client.Object);

        Action act = () => store.OpenReader("missing");

        act.Should().Throw<ArtifactNotFound>();
    }

    [TestMethod]
    public void OpenReader_OtherError_Propagates()
    {
        var client = new Mock<IAmazonS3>();
        client.Setup(c => c.GetObjectAsync(It.IsAny<GetObjectRequest>(), It.IsAny<CancellationToken>()))
              .ThrowsAsync(new AmazonS3Exception("denied", ErrorType.Sender, "AccessDenied", "req", HttpStatusCode.Forbidden));
        var store = new Boto3ArtifactStore("bucket", client.Object);

        Action act = () => store.OpenReader("x");

        act.Should().Throw<AmazonS3Exception>();
    }

    [TestMethod]
    public void Write_UploadsContent_ToBucketKey()
    {
        byte[] data = { 9, 8, 7 };
        byte[]? uploaded = null;
        PutObjectRequest? captured = null;
        var client = new Mock<IAmazonS3>();
        client.Setup(c => c.PutObjectAsync(It.IsAny<PutObjectRequest>(), It.IsAny<CancellationToken>()))
              .Callback<PutObjectRequest, CancellationToken>((r, _) =>
              {
                  captured = r;
                  using var ms = new MemoryStream();
                  r.InputStream.CopyTo(ms);
                  uploaded = ms.ToArray();
              })
              .ReturnsAsync(new PutObjectResponse());
        var store = new Boto3ArtifactStore("bucket", client.Object);

        store.Write("art-2", new MemoryStream(data));

        captured!.BucketName.Should().Be("bucket");
        captured.Key.Should().Be("art-2");
        uploaded.Should().Equal(data);
    }

    [TestMethod]
    public void Remove_DeletesBucketKey()
    {
        DeleteObjectRequest? captured = null;
        var client = new Mock<IAmazonS3>();
        client.Setup(c => c.DeleteObjectAsync(It.IsAny<DeleteObjectRequest>(), It.IsAny<CancellationToken>()))
              .Callback<DeleteObjectRequest, CancellationToken>((r, _) => captured = r)
              .ReturnsAsync(new DeleteObjectResponse());
        var store = new Boto3ArtifactStore("bucket", client.Object);

        store.Remove("art-3");

        captured!.BucketName.Should().Be("bucket");
        captured.Key.Should().Be("art-3");
    }
}
