using System;
using System.IO;
using System.Threading;
using AwesomeAssertions;
using Azure;
using Azure.Identity;
using Azure.Storage.Blobs;
using Azure.Storage.Blobs.Models;
using Moq;
using OptunaSharp.Artifacts;

namespace OptunaSharp.Tests.Artifacts;

/// <summary>
///     Unit tests for the Azure Blob backend (<see cref="AzureBlobArtifactStore" />). Covers all three
///     constructors, the not-found mapping, error propagation, and get/put/delete delegation via a
///     mocked <see cref="BlobContainerClient" />/<see cref="BlobClient" />. Live round-trips are in
///     <see cref="AzureBlobIntegrationTests" /> (Azurite).
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

    // ---- ctors ----

    [TestMethod]
    public void Ctor_ConnectionString_Constructs()
    {
        var store = new AzureBlobArtifactStore(ArtifactEmulators.AzuriteConnectionString, "cont");

        store.Should().NotBeNull();
    }

    [TestMethod]
    public void Ctor_ServiceUri_DefaultCredential_Constructs()
    {
        // credential omitted -> DefaultAzureCredential (constructed lazily, no network here).
        var store = new AzureBlobArtifactStore(
            new Uri("https://account.blob.core.windows.net"), "cont");

        store.Should().NotBeNull();
    }

    [TestMethod]
    public void Ctor_ServiceUri_ExplicitCredential_Constructs()
    {
        var store = new AzureBlobArtifactStore(
            new Uri("https://account.blob.core.windows.net"), "cont", new DefaultAzureCredential());

        store.Should().NotBeNull();
    }

    // ---- OpenReader ----

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
    public void OpenReader_OtherError_Propagates()
    {
        var (container, blob) = NewContainer();
        blob.Setup(b => b.DownloadStreaming(It.IsAny<BlobDownloadOptions>(), It.IsAny<CancellationToken>()))
            .Throws(new RequestFailedException(403, "AuthorizationFailure"));
        var store = new AzureBlobArtifactStore(container.Object);

        Action act = () => store.OpenReader("x");

        act.Should().Throw<RequestFailedException>().Which.Status.Should().Be(403);
    }

    // ---- Write ----

    [TestMethod]
    public void Write_DelegatesToUpload_WithOverwrite()
    {
        var (container, blob) = NewContainer();
        var store = new AzureBlobArtifactStore(container.Object);

        store.Write("art-1", new MemoryStream(new byte[] { 1, 2, 3 }));

        container.Verify(c => c.GetBlobClient("art-1"), Times.Once);
        blob.Verify(b => b.Upload(It.IsAny<Stream>(), true, It.IsAny<CancellationToken>()), Times.Once);
    }

    // ---- Remove ----

    [TestMethod]
    public void Remove_DeletesBlob()
    {
        var (container, blob) = NewContainer();
        var store = new AzureBlobArtifactStore(container.Object);

        store.Remove("art-2");

        blob.Verify(b => b.Delete(It.IsAny<DeleteSnapshotsOption>(), It.IsAny<BlobRequestConditions>(),
            It.IsAny<CancellationToken>()), Times.Once);
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
    public void Remove_OtherError_Propagates()
    {
        var (container, blob) = NewContainer();
        blob.Setup(b => b.Delete(It.IsAny<DeleteSnapshotsOption>(), It.IsAny<BlobRequestConditions>(), It.IsAny<CancellationToken>()))
            .Throws(new RequestFailedException(403, "AuthorizationFailure"));
        var store = new AzureBlobArtifactStore(container.Object);

        Action act = () => store.Remove("x");

        act.Should().Throw<RequestFailedException>().Which.Status.Should().Be(403);
    }
}
