using System;
using System.IO;
using System.Threading;
using AwesomeAssertions;
using Azure;
using Azure.Storage.Blobs;
using Azure.Storage.Blobs.Models;
using Moq;
using OptunaSharp.Artifacts;

namespace OptunaSharp.Tests.Artifacts;

/// <summary>
///     Unit tests for the Azure Blob backend (<see cref="AzureBlobArtifactStore" />), driven by a
///     mocked <see cref="BlobContainerClient" />/<see cref="BlobClient" />. Live round-trips against
///     Azure / Azurite are a deferred integration concern (see the plan §4.5).
/// </summary>
[TestClass]
public class AzureBlobArtifactStoreTests
{
    private static (Mock<BlobContainerClient> Container, Mock<BlobClient> Blob) NewContainer()
    {
        var blob = new Mock<BlobClient>();
        var container = new Mock<BlobContainerClient>();
        container.Setup(c => c.Name).Returns("cont");
        container.Setup(c => c.GetBlobClient(It.IsAny<string>())).Returns(blob.Object);
        return (container, blob);
    }

    [TestMethod]
    public void OpenReader_NotFound_ThrowsArtifactNotFound()
    {
        var (container, blob) = NewContainer();
        blob.Setup(b => b.DownloadStreaming(It.IsAny<BlobDownloadOptions>(), It.IsAny<CancellationToken>()))
            .Throws(new RequestFailedException(404, "BlobNotFound"));
        var store = new AzureBlobArtifactStore(container.Object);

        Action act = () => store.OpenReader("missing");

        act.Should().Throw<ArtifactNotFound>();
    }

    [TestMethod]
    public void Remove_NotFound_ThrowsArtifactNotFound()
    {
        var (container, blob) = NewContainer();
        blob.Setup(b => b.Delete(It.IsAny<DeleteSnapshotsOption>(), It.IsAny<BlobRequestConditions>(), It.IsAny<CancellationToken>()))
            .Throws(new RequestFailedException(404, "BlobNotFound"));
        var store = new AzureBlobArtifactStore(container.Object);

        Action act = () => store.Remove("missing");

        act.Should().Throw<ArtifactNotFound>();
    }

    [TestMethod]
    public void Write_DelegatesToUpload_WithOverwrite()
    {
        var (container, blob) = NewContainer();
        var store = new AzureBlobArtifactStore(container.Object);

        store.Write("art-1", new MemoryStream(new byte[] { 1, 2, 3 }));

        blob.Verify(b => b.Upload(It.IsAny<Stream>(), true, It.IsAny<CancellationToken>()), Times.Once);
    }
}
