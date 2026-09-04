using System;
using System.IO;
using System.Net;
using AwesomeAssertions;
using Google;
using Google.Apis.Download;
using Google.Cloud.Storage.V1;
using Moq;
using OptunaSharp.Artifacts;

namespace OptunaSharp.Tests.Artifacts;

/// <summary>
///     Unit tests for the GCS backend (<see cref="GCSArtifactStore" />), driven by a mocked
///     <see cref="StorageClient" />. Live round-trips against GCS / a fake-gcs-server are a deferred
///     integration concern (see the plan §4.5).
/// </summary>
[TestClass]
public class GCSArtifactStoreTests
{
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
        var client = new Mock<StorageClient>();
        client.Setup(c => c.DownloadObject(It.IsAny<string>(), It.IsAny<string>(), It.IsAny<Stream>(),
                          It.IsAny<DownloadObjectOptions>(), It.IsAny<IProgress<IDownloadProgress>>()))
              .Throws(new GoogleApiException("storage", "not found") { HttpStatusCode = HttpStatusCode.NotFound });
        var store = new GCSArtifactStore("bucket", client.Object);

        Action act = () => store.OpenReader("missing");

        act.Should().Throw<ArtifactNotFound>();
    }

    [TestMethod]
    public void Remove_DelegatesToDeleteObject()
    {
        var client = new Mock<StorageClient>();
        var store = new GCSArtifactStore("bucket", client.Object);

        store.Remove("art-2");

        client.Verify(c => c.DeleteObject("bucket", "art-2", It.IsAny<DeleteObjectOptions>()), Times.Once);
    }
}
