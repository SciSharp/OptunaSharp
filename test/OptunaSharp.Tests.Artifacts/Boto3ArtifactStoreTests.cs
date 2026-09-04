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
///     <see cref="IAmazonS3" /> — no live bucket. Every public ctor and method plus every branch
///     (not-found mapping, error propagation, both <c>avoidBufCopy</c> modes) is exercised. Live
///     round-trips are in <see cref="S3IntegrationTests" /> (MinIO).
/// </summary>
[TestClass]
public class Boto3ArtifactStoreTests
{
    private static AmazonS3Exception S3Error(string code, HttpStatusCode status)
        => new("err", ErrorType.Sender, code, "req-id", status);

    // ---- ctor ----

    [TestMethod]
    public void Ctor_NullClient_CreatesDefaultClient()
    {
        // The `client ?? new AmazonS3Client()` branch. AmazonS3Client construction only needs a
        // region (credentials resolve lazily at call time), so AWS_REGION makes it deterministic.
        string? prev = Environment.GetEnvironmentVariable("AWS_REGION");
        Environment.SetEnvironmentVariable("AWS_REGION", "us-east-1");
        try
        {
            var store = new Boto3ArtifactStore("bucket");
            store.Should().NotBeNull();
        }
        finally
        {
            Environment.SetEnvironmentVariable("AWS_REGION", prev);
        }
    }

    // ---- OpenReader ----

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
        client.Verify(c => c.GetObjectAsync(
            It.Is<GetObjectRequest>(r => r.BucketName == "bucket" && r.Key == "art-1"),
            It.IsAny<CancellationToken>()), Times.Once);
    }

    [TestMethod]
    public void OpenReader_NoSuchKey_ThrowsArtifactNotFound()
    {
        var client = new Mock<IAmazonS3>();
        client.Setup(c => c.GetObjectAsync(It.IsAny<GetObjectRequest>(), It.IsAny<CancellationToken>()))
              .ThrowsAsync(S3Error("NoSuchKey", HttpStatusCode.NotFound));
        var store = new Boto3ArtifactStore("bucket", client.Object);

        Action act = () => store.OpenReader("missing");

        act.Should().Throw<ArtifactNotFound>();
    }

    [TestMethod]
    public void OpenReader_404WithoutNoSuchKey_ThrowsArtifactNotFound()
    {
        // _is_not_found_error also treats any HTTP 404 as not-found (second operand of the ||).
        var client = new Mock<IAmazonS3>();
        client.Setup(c => c.GetObjectAsync(It.IsAny<GetObjectRequest>(), It.IsAny<CancellationToken>()))
              .ThrowsAsync(S3Error("NoSuchBucket", HttpStatusCode.NotFound));
        var store = new Boto3ArtifactStore("bucket", client.Object);

        Action act = () => store.OpenReader("missing");

        act.Should().Throw<ArtifactNotFound>();
    }

    [TestMethod]
    public void OpenReader_OtherError_Propagates()
    {
        var client = new Mock<IAmazonS3>();
        client.Setup(c => c.GetObjectAsync(It.IsAny<GetObjectRequest>(), It.IsAny<CancellationToken>()))
              .ThrowsAsync(S3Error("AccessDenied", HttpStatusCode.Forbidden));
        var store = new Boto3ArtifactStore("bucket", client.Object);

        Action act = () => store.OpenReader("x");

        act.Should().Throw<AmazonS3Exception>().Which.ErrorCode.Should().Be("AccessDenied");
    }

    // ---- Write ----

    [TestMethod]
    public void Write_AvoidBufCopyFalse_CopiesToBuffer_AndUploadsContent()
    {
        byte[] data = { 9, 8, 7 };
        var source = new MemoryStream(data);
        var (client, captured, uploaded) = CapturingPutClient();
        var store = new Boto3ArtifactStore("bucket", client.Object); // avoidBufCopy defaults to false

        store.Write("art-2", source);

        captured.Value!.BucketName.Should().Be("bucket");
        captured.Value.Key.Should().Be("art-2");
        captured.Value.AutoCloseStream.Should().BeFalse("the SDK must never close the caller's stream");
        captured.Value.InputStream.Should().NotBeSameAs(source, "avoidBufCopy=false uploads a buffered copy");
        uploaded.Value.Should().Equal(data);
        source.CanRead.Should().BeTrue("Write must not dispose the caller's stream");
    }

    [TestMethod]
    public void Write_AvoidBufCopyTrue_UploadsCallerStreamDirectly()
    {
        byte[] data = { 4, 4, 2 };
        var source = new MemoryStream(data);
        var (client, captured, uploaded) = CapturingPutClient();
        var store = new Boto3ArtifactStore("bucket", client.Object, avoidBufCopy: true);

        store.Write("art-3", source);

        captured.Value!.AutoCloseStream.Should().BeFalse();
        captured.Value.InputStream.Should().BeSameAs(source, "avoidBufCopy=true uploads the source stream directly");
        uploaded.Value.Should().Equal(data);
    }

    [TestMethod]
    public void Write_EmptyContent_Succeeds()
    {
        var (client, captured, uploaded) = CapturingPutClient();
        var store = new Boto3ArtifactStore("bucket", client.Object);

        store.Write("empty", new MemoryStream(Array.Empty<byte>()));

        captured.Value!.Key.Should().Be("empty");
        uploaded.Value.Should().BeEmpty();
    }

    // ---- Remove ----

    [TestMethod]
    public void Remove_DeletesBucketKey()
    {
        DeleteObjectRequest? captured = null;
        var client = new Mock<IAmazonS3>();
        client.Setup(c => c.DeleteObjectAsync(It.IsAny<DeleteObjectRequest>(), It.IsAny<CancellationToken>()))
              .Callback<DeleteObjectRequest, CancellationToken>((r, _) => captured = r)
              .ReturnsAsync(new DeleteObjectResponse());
        var store = new Boto3ArtifactStore("bucket", client.Object);

        store.Remove("art-4");

        captured!.BucketName.Should().Be("bucket");
        captured.Key.Should().Be("art-4");
    }

    /// <summary>A mock whose PutObjectAsync captures the request and the uploaded bytes.</summary>
    private static (Mock<IAmazonS3> Client, Ref<PutObjectRequest?> Request, Ref<byte[]> Uploaded) CapturingPutClient()
    {
        var request = new Ref<PutObjectRequest?>(null);
        var uploaded = new Ref<byte[]>(Array.Empty<byte>());
        var client = new Mock<IAmazonS3>();
        client.Setup(c => c.PutObjectAsync(It.IsAny<PutObjectRequest>(), It.IsAny<CancellationToken>()))
              .Callback<PutObjectRequest, CancellationToken>((r, _) =>
              {
                  request.Value = r;
                  using var ms = new MemoryStream();
                  r.InputStream.CopyTo(ms);
                  uploaded.Value = ms.ToArray();
              })
              .ReturnsAsync(new PutObjectResponse());
        return (client, request, uploaded);
    }

    private sealed class Ref<T>(T value)
    {
        public T Value = value;
    }
}
