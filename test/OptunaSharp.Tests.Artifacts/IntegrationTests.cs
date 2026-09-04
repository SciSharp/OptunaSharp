using System;
using System.IO;
using AwesomeAssertions;
using OptunaSharp.Artifacts;

namespace OptunaSharp.Tests.Artifacts;

/// <summary>
///     The <see cref="IArtifactStore" /> contract every backend must satisfy, asserted against a real
///     (emulated) object store: a write is durable and byte-exact on read-back, a removed artifact is
///     gone, and reading a missing artifact raises <see cref="ArtifactNotFound" />.
/// </summary>
internal static class ArtifactStoreContract
{
    public static void AssertFullLifecycle(IArtifactStore store)
    {
        string id = "it-" + Guid.NewGuid().ToString("N");
        byte[] payload = RandomBytes(1500);

        store.Write(id, new MemoryStream(payload));

        byte[] readBack;
        using (Stream r = store.OpenReader(id))
            readBack = ReadAll(r);
        readBack.Should().Equal(payload);

        store.Remove(id);

        Action reopen = () => { using Stream _ = store.OpenReader(id); };
        reopen.Should().Throw<ArtifactNotFound>();
    }

    public static void AssertUnknownIdNotFound(IArtifactStore store)
    {
        Action act = () => { using Stream _ = store.OpenReader("it-missing-" + Guid.NewGuid().ToString("N")); };
        act.Should().Throw<ArtifactNotFound>();
    }

    private static byte[] RandomBytes(int n)
    {
        var b = new byte[n];
        new Random(20260904).NextBytes(b);
        return b;
    }

    private static byte[] ReadAll(Stream s)
    {
        using var ms = new MemoryStream();
        s.CopyTo(ms);
        return ms.ToArray();
    }
}

/// <summary>
///     Integration tests for the S3 backend against a MinIO emulator (docker-compose.yml). Self-skip
///     when MinIO is not reachable. Category <c>Integration</c> so CI can include/exclude them.
/// </summary>
[TestClass]
public class S3IntegrationTests
{
    [TestMethod]
    [TestCategory("Integration")]
    public void FullLifecycle_WriteReadRemove()
    {
        ArtifactEmulators.EnsureReachable("S3/MinIO", ArtifactEmulators.MinioPort);
        ArtifactStoreContract.AssertFullLifecycle(ArtifactEmulators.NewS3Store("s3-it"));
    }

    [TestMethod]
    [TestCategory("Integration")]
    public void OpenReader_UnknownId_ThrowsArtifactNotFound()
    {
        ArtifactEmulators.EnsureReachable("S3/MinIO", ArtifactEmulators.MinioPort);
        ArtifactStoreContract.AssertUnknownIdNotFound(ArtifactEmulators.NewS3Store("s3-it"));
    }
}

/// <summary>
///     Integration tests for the Cloudflare R2 backend against a MinIO emulator (R2 is S3-compatible;
///     Cloudflare ships no emulator). Exercises <see cref="R2ArtifactStore" /> end to end.
/// </summary>
[TestClass]
public class R2IntegrationTests
{
    [TestMethod]
    [TestCategory("Integration")]
    public void FullLifecycle_WriteReadRemove()
    {
        ArtifactEmulators.EnsureReachable("R2/MinIO", ArtifactEmulators.MinioPort);
        ArtifactStoreContract.AssertFullLifecycle(ArtifactEmulators.NewR2Store("r2-it"));
    }

    [TestMethod]
    [TestCategory("Integration")]
    public void OpenReader_UnknownId_ThrowsArtifactNotFound()
    {
        ArtifactEmulators.EnsureReachable("R2/MinIO", ArtifactEmulators.MinioPort);
        ArtifactStoreContract.AssertUnknownIdNotFound(ArtifactEmulators.NewR2Store("r2-it"));
    }
}

/// <summary>
///     Integration tests for the GCS backend against a fake-gcs-server emulator (docker-compose.yml).
/// </summary>
[TestClass]
public class GcsIntegrationTests
{
    [TestMethod]
    [TestCategory("Integration")]
    public void FullLifecycle_WriteReadRemove()
    {
        ArtifactEmulators.EnsureReachable("GCS/fake-gcs-server", ArtifactEmulators.FakeGcsPort);
        ArtifactStoreContract.AssertFullLifecycle(ArtifactEmulators.NewGcsStore("gcs-it"));
    }

    [TestMethod]
    [TestCategory("Integration")]
    public void OpenReader_UnknownId_ThrowsArtifactNotFound()
    {
        ArtifactEmulators.EnsureReachable("GCS/fake-gcs-server", ArtifactEmulators.FakeGcsPort);
        ArtifactStoreContract.AssertUnknownIdNotFound(ArtifactEmulators.NewGcsStore("gcs-it"));
    }
}

/// <summary>
///     Integration tests for the Azure Blob backend against an Azurite emulator (docker-compose.yml).
/// </summary>
[TestClass]
public class AzureBlobIntegrationTests
{
    [TestMethod]
    [TestCategory("Integration")]
    public void FullLifecycle_WriteReadRemove()
    {
        ArtifactEmulators.EnsureReachable("Azure/Azurite", ArtifactEmulators.AzuritePort);
        ArtifactStoreContract.AssertFullLifecycle(ArtifactEmulators.NewAzureStore("azure-it"));
    }

    [TestMethod]
    [TestCategory("Integration")]
    public void OpenReader_UnknownId_ThrowsArtifactNotFound()
    {
        ArtifactEmulators.EnsureReachable("Azure/Azurite", ArtifactEmulators.AzuritePort);
        ArtifactStoreContract.AssertUnknownIdNotFound(ArtifactEmulators.NewAzureStore("azure-it"));
    }
}
