using System;
using System.IO;
using System.Net;
using AwesomeAssertions;
using Google;
using Google.Apis.Download;
using Google.Apis.Upload;
using Google.Cloud.Storage.V1;
using Moq;
using OptunaSharp.Artifacts;

namespace OptunaSharp.Tests.Artifacts;

/// <summary>
///     Unit tests for the GCS backend (<see cref="GCSArtifactStore" />), driven by a mocked
///     <see cref="StorageClient" />. Covers success, the not-found mapping, error propagation, upload
///     delegation, and the Optuna-faithful "remove of a missing object raises" behavior. Live
///     round-trips are in <see cref="GcsIntegrationTests" /> (fake-gcs-server).
/// </summary>
[TestClass]
public class GCSArtifactStoreTests
{
    private static GoogleApiException GcsError(HttpStatusCode status)
        => new("storage", "err") { HttpStatusCode = status };

    private static Mock<StorageClient> DownloadThrows(GoogleApiException ex)
    {
        var client = new Mock<StorageClient>();
        client.Setup(c => c.DownloadObject(It.IsAny<string>(), It.IsAny<string>(), It.IsAny<Stream>(),
                          It.IsAny<DownloadObjectOptions>(), It.IsAny<IProgress<IDownloadProgress>>()))
              .Throws(ex);
        return client;
    }

    [TestMethod]
    public void OpenReader_ReturnsContent_OnSuccess()
    {
        byte[] data = { 4, 5, 6 };
        var client = new Mock<StorageClient>();
        client.Setup(c => c.DownloadObject(It.IsAny<string>(), It.IsAny<string>(), It.IsAny<Stream>(),
                          It.IsAny<DownloadObjectOptions>(), It.IsAny<IProgress<IDownloadProgress>>()))
              .Callback<string, string, Stream, DownloadObjectOptions, IProgress<IDownloadProgress>>(
                  (bucket, obj, dest, opts, prog) => dest.Write(data, 0, data.Length))
              .Returns((Google.Apis.Storage.v1.Data.Object)null!);
        var store = new GCSArtifactStore("bucket", client.Object);

        using var reader = store.OpenReader("art-1");
        using var ms = new MemoryStream();
        reader.CopyTo(ms);

        ms.ToArray().Should().Equal(data);
    }

    [TestMethod]
    public void OpenReader_NotFound_ThrowsArtifactNotFound()
    {
        var store = new GCSArtifactStore("bucket", DownloadThrows(GcsError(HttpStatusCode.NotFound)).Object);

        Action act = () => store.OpenReader("missing");

        act.Should().Throw<ArtifactNotFound>();
    }

    [TestMethod]
    public void OpenReader_OtherError_Propagates()
    {
        var store = new GCSArtifactStore("bucket", DownloadThrows(GcsError(HttpStatusCode.Forbidden)).Object);

        Action act = () => store.OpenReader("x");

        act.Should().Throw<GoogleApiException>();
    }

    [TestMethod]
    public void Write_DelegatesToUploadObject_WithNullContentType()
    {
        var client = new Mock<StorageClient>();
        client.Setup(c => c.UploadObject(It.IsAny<string>(), It.IsAny<string>(), It.IsAny<string>(),
                          It.IsAny<Stream>(), It.IsAny<UploadObjectOptions>(), It.IsAny<IProgress<IUploadProgress>>()))
              .Returns((Google.Apis.Storage.v1.Data.Object)null!);
        var store = new GCSArtifactStore("bucket", client.Object);

        store.Write("art-2", new MemoryStream(new byte[] { 1, 2, 3 }));

        client.Verify(c => c.UploadObject("bucket", "art-2", null, It.IsAny<Stream>(),
            It.IsAny<UploadObjectOptions>(), It.IsAny<IProgress<IUploadProgress>>()), Times.Once);
    }

    [TestMethod]
    public void Remove_DelegatesToDeleteObject()
    {
        var client = new Mock<StorageClient>();
        var store = new GCSArtifactStore("bucket", client.Object);

        store.Remove("art-3");

        client.Verify(c => c.DeleteObject("bucket", "art-3", It.IsAny<DeleteObjectOptions>()), Times.Once);
    }

    [TestMethod]
    public void Remove_MissingObject_PropagatesGoogleApiException()
    {
        // Matches Optuna's _gcs.py: delete_blob raises when the object is missing — the port does NOT
        // map it to ArtifactNotFound.
        var client = new Mock<StorageClient>();
        client.Setup(c => c.DeleteObject(It.IsAny<string>(), It.IsAny<string>(), It.IsAny<DeleteObjectOptions>()))
              .Throws(GcsError(HttpStatusCode.NotFound));
        var store = new GCSArtifactStore("bucket", client.Object);

        Action act = () => store.Remove("missing");

        act.Should().Throw<GoogleApiException>();
    }
}
