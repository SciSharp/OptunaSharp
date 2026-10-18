using System;
using System.IO;
using AwesomeAssertions;
using Google;
using OptunaSharp;
using OptunaSharp.Artifacts;

namespace OptunaSharp.Tests.Artifacts;

/// <summary>
///     The <see cref="IArtifactStore" /> contract every backend must satisfy, asserted against a real
///     (emulated) object store: writes are durable and byte-exact on read-back (empty, small and large
///     payloads), a re-write overwrites, nested-path ids work, a removed artifact is gone, and reading
///     a missing artifact raises <see cref="ArtifactNotFound" />.
/// </summary>
internal static class ArtifactStoreContract
{
    public static void RunAll(IArtifactStore store)
    {
        RoundTrip(store, 0);                 // empty payload
        RoundTrip(store, 1500);              // small payload
        RoundTrip(store, 2 * 1024 * 1024);   // large payload (2 MiB)
        Overwrite(store);
        NestedPathId(store);
        UnknownIdNotFound(store);
    }

    public static void RoundTrip(IArtifactStore store, int size)
    {
        string id = NewId();
        byte[] payload = RandomBytes(size);

        store.Write(id, new MemoryStream(payload));
        ReadBack(store, id).Should().Equal(payload);

        store.Remove(id);
        Reopen(store, id).Should().Throw<ArtifactNotFound>();
    }

    public static void Overwrite(IArtifactStore store)
    {
        string id = NewId();
        store.Write(id, new MemoryStream(RandomBytes(64)));
        byte[] second = RandomBytes(128);
        store.Write(id, new MemoryStream(second));

        ReadBack(store, id).Should().Equal(second, "a re-write of the same id must overwrite");
        store.Remove(id);
    }

    public static void NestedPathId(IArtifactStore store)
    {
        string id = "nested/" + Guid.NewGuid().ToString("N") + "/artifact.bin";
        byte[] payload = RandomBytes(256);
        store.Write(id, new MemoryStream(payload));
        ReadBack(store, id).Should().Equal(payload);
        store.Remove(id);
    }

    public static void UnknownIdNotFound(IArtifactStore store)
        => Reopen(store, NewId()).Should().Throw<ArtifactNotFound>();

    private static byte[] ReadBack(IArtifactStore store, string id)
    {
        using Stream r = store.OpenReader(id);
        using var ms = new MemoryStream();
        r.CopyTo(ms);
        return ms.ToArray();
    }

    private static Action Reopen(IArtifactStore store, string id)
        => () => { using Stream _ = store.OpenReader(id); };

    private static string NewId() => "it-" + Guid.NewGuid().ToString("N");

    private static byte[] RandomBytes(int n)
    {
        var b = new byte[n];
        new Random(20260904).NextBytes(b);
        return b;
    }
}

/// <summary>
///     Integration tests for the S3 backend against the S3-compatible emulator (S3Mock, docker-compose.yml).
///     Self-skip when it is not reachable. Category <c>Integration</c> so CI can include/exclude them.
/// </summary>
[TestClass]
public class S3IntegrationTests
{
    private static void Skip() => ArtifactEmulators.EnsureReachable("S3", ArtifactEmulators.S3EmulatorPort);

    [TestMethod, TestCategory("Integration")]
    public void Contract() { Skip(); ArtifactStoreContract.RunAll(ArtifactEmulators.NewS3Store("s3-it")); }

    [TestMethod, TestCategory("Integration")]
    public void RemoveMissing_IsIdempotent()
    {
        Skip();
        // S3 delete of a missing key succeeds (matches boto3) — no throw.
        Action act = () => ArtifactEmulators.NewS3Store("s3-it").Remove("it-missing-" + Guid.NewGuid().ToString("N"));
        act.Should().NotThrow();
    }

    [TestMethod, TestCategory("Integration")]
    public void AvoidBufCopy_RoundTrip()
    {
        Skip();
        var store = new Boto3ArtifactStore("s3-it", ArtifactEmulators.NewS3Client(), avoidBufCopy: true);
        ArtifactStoreContract.RoundTrip(store, 4096);
    }

    [TestMethod, TestCategory("Integration")]
    public void EndToEnd_ArtifactFunctions_UploadDownloadMeta()
    {
        Skip();
        IArtifactStore store = ArtifactEmulators.NewS3Store("s3-it");
        Study study = Optuna.CreateStudy();
        Trial trial = study.Ask();

        byte[] payload = { 1, 2, 3, 4, 5, 6, 7, 8 };
        string src = Path.Combine(Path.GetTempPath(), "osharp-src-" + Guid.NewGuid().ToString("N") + ".bin");
        string dst = Path.Combine(Path.GetTempPath(), "osharp-dst-" + Guid.NewGuid().ToString("N") + ".bin");
        try
        {
            File.WriteAllBytes(src, payload);
            string artifactId = ArtifactFunctions.UploadArtifact(store, src, trial);

            ArtifactFunctions.DownloadArtifact(store, dst, artifactId);
            File.ReadAllBytes(dst).Should().Equal(payload);

            ArtifactFunctions.GetAllArtifactMeta(trial)
                .Should().ContainSingle(m => m.ArtifactId == artifactId && m.Filename == Path.GetFileName(src));

            store.Remove(artifactId);
        }
        finally
        {
            File.Delete(src);
            File.Delete(dst);
        }
    }
}

/// <summary>
///     Integration tests for the Cloudflare R2 backend against the S3-compatible emulator (R2 speaks the S3 API;
///     Cloudflare ships no emulator). Exercises <see cref="R2ArtifactStore" /> end to end.
/// </summary>
[TestClass]
public class R2IntegrationTests
{
    private static void Skip() => ArtifactEmulators.EnsureReachable("R2 (S3 API)", ArtifactEmulators.S3EmulatorPort);

    [TestMethod, TestCategory("Integration")]
    public void Contract() { Skip(); ArtifactStoreContract.RunAll(ArtifactEmulators.NewR2Store("r2-it")); }

    [TestMethod, TestCategory("Integration")]
    public void RemoveMissing_IsIdempotent()
    {
        Skip();
        Action act = () => ArtifactEmulators.NewR2Store("r2-it").Remove("it-missing-" + Guid.NewGuid().ToString("N"));
        act.Should().NotThrow();
    }

    [TestMethod, TestCategory("Integration")]
    public void AvoidBufCopy_RoundTrip()
    {
        Skip();
        var store = new R2ArtifactStore(ArtifactEmulators.NewS3Client(), "r2-it", avoidBufCopy: true);
        ArtifactStoreContract.RoundTrip(store, 4096);
    }
}

/// <summary>
///     Integration tests for the GCS backend against a fake-gcs-server emulator (docker-compose.yml).
/// </summary>
[TestClass]
public class GcsIntegrationTests
{
    private static void Skip() => ArtifactEmulators.EnsureReachable("GCS/fake-gcs-server", ArtifactEmulators.FakeGcsPort);

    [TestMethod, TestCategory("Integration")]
    public void Contract() { Skip(); ArtifactStoreContract.RunAll(ArtifactEmulators.NewGcsStore("gcs-it")); }

    [TestMethod, TestCategory("Integration")]
    public void RemoveMissing_ThrowsGoogleApiException()
    {
        Skip();
        // Matches Optuna's _gcs.py: delete of a missing blob raises the SDK's NotFound (not mapped).
        Action act = () => ArtifactEmulators.NewGcsStore("gcs-it").Remove("it-missing-" + Guid.NewGuid().ToString("N"));
        act.Should().Throw<GoogleApiException>();
    }

    // NOTE: the GCS default-client ctor (`client ?? StorageClient.Create()`) is the single artifact
    // code path with no offline coverage: StorageClient.Create() requires real Application Default
    // Credentials (it does NOT honor STORAGE_EMULATOR_HOST at construction), so it can only be
    // exercised against live GCS. The client-provided path is covered by every test above.
}

/// <summary>
///     Integration tests for the Azure Blob backend against an Azurite emulator (docker-compose.yml).
/// </summary>
[TestClass]
public class AzureBlobIntegrationTests
{
    private static void Skip() => ArtifactEmulators.EnsureReachable("Azure/Azurite", ArtifactEmulators.AzuritePort);

    [TestMethod, TestCategory("Integration")]
    public void Contract() { Skip(); ArtifactStoreContract.RunAll(ArtifactEmulators.NewAzureStore("azure-it")); }

    [TestMethod, TestCategory("Integration")]
    public void RemoveMissing_ThrowsArtifactNotFound()
    {
        Skip();
        // Azure maps a 404 on delete to ArtifactNotFound.
        Action act = () => ArtifactEmulators.NewAzureStore("azure-it").Remove("it-missing-" + Guid.NewGuid().ToString("N"));
        act.Should().Throw<ArtifactNotFound>();
    }
}
