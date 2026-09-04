using System;
using System.Net.Sockets;
using Amazon.S3;
using Amazon.S3.Model;
using Azure.Storage.Blobs;
using Google;
using Google.Cloud.Storage.V1;
using OptunaSharp.Artifacts;

namespace OptunaSharp.Tests.Artifacts;

/// <summary>
///     Connection factories and reachability gating for the dockerized artifact-store emulators
///     defined in <c>docker-compose.yml</c> (this folder): MinIO (S3 + Cloudflare R2), Azurite
///     (Azure Blob) and fake-gcs-server (GCS).
///     <para>
///     Mirrors the interop suite's <c>PythonSession.EnsureOrInconclusive</c> pattern — an
///     integration test calls <see cref="EnsureReachable" /> first and reports
///     <see cref="Assert.Inconclusive(string)" /> when its emulator port is not reachable (so the
///     suite is safe to run without Docker), unless <c>OPTUNASHARP_ARTIFACT_EMULATORS_REQUIRED</c>
///     is truthy, in which case an absent emulator is a hard failure (for CI once wired).
///     </para>
/// </summary>
internal static class ArtifactEmulators
{
    // MinIO — S3 API endpoint (also serves the R2 tests; R2 is S3-compatible).
    public const int MinioPort = 9000;
    public const string MinioEndpoint = "http://localhost:9000";
    public const string MinioAccessKey = "minioadmin";
    public const string MinioSecretKey = "minioadmin";

    // Azurite — the well-known development connection string (account "devstoreaccount1").
    public const int AzuritePort = 10000;
    public const string AzuriteConnectionString =
        "DefaultEndpointsProtocol=http;AccountName=devstoreaccount1;" +
        "AccountKey=Eby8vdM02xNOcqFlqUwJPLlmEtlCDXJ1OUzFT50uSRZ6IFsuFq2UVErCz4I6tq/K1SZFPTOtr/KBHBeksoGMGw==;" +
        "BlobEndpoint=http://127.0.0.1:10000/devstoreaccount1;";

    // fake-gcs-server — the GCS JSON API base URI.
    public const int FakeGcsPort = 4443;
    public const string FakeGcsBaseUri = "http://localhost:4443/storage/v1/";

    /// <summary>Skip (Inconclusive) — or fail, if required — when an emulator port is not reachable.</summary>
    public static void EnsureReachable(string label, int port)
    {
        if (IsReachable("localhost", port))
            return;

        string reason =
            $"{label} emulator not reachable on localhost:{port}. Start it with: " +
            "docker compose -f test/OptunaSharp.Tests.Artifacts/docker-compose.yml up -d";
        if (Required())
            Assert.Fail(reason + " (OPTUNASHARP_ARTIFACT_EMULATORS_REQUIRED is set, so this is a hard failure).");
        Assert.Inconclusive(reason);
    }

    private static bool Required()
    {
        string? v = Environment.GetEnvironmentVariable("OPTUNASHARP_ARTIFACT_EMULATORS_REQUIRED");
        return v is "1" or "true" or "TRUE" or "yes";
    }

    private static bool IsReachable(string host, int port)
    {
        try
        {
            using var client = new TcpClient();
            return client.ConnectAsync(host, port).Wait(TimeSpan.FromSeconds(2)) && client.Connected;
        }
        catch
        {
            return false;
        }
    }

    public static IAmazonS3 NewMinioClient()
        => new AmazonS3Client(MinioAccessKey, MinioSecretKey,
            new AmazonS3Config { ServiceURL = MinioEndpoint, ForcePathStyle = true });

    public static Boto3ArtifactStore NewS3Store(string bucket)
    {
        IAmazonS3 client = NewMinioClient();
        EnsureBucket(client, bucket);
        return new Boto3ArtifactStore(bucket, client);
    }

    public static R2ArtifactStore NewR2Store(string bucket)
    {
        IAmazonS3 client = NewMinioClient();
        EnsureBucket(client, bucket);
        // Exercises the R2ArtifactStore(IAmazonS3, bucket) ctor against an S3-compatible endpoint —
        // exactly how R2 is reached (Cloudflare has no emulator; R2 speaks the S3 API).
        return new R2ArtifactStore(client, bucket);
    }

    private static void EnsureBucket(IAmazonS3 client, string bucket)
    {
        try
        {
            client.PutBucketAsync(new PutBucketRequest { BucketName = bucket }).GetAwaiter().GetResult();
        }
        catch (AmazonS3Exception e) when (e.ErrorCode is "BucketAlreadyOwnedByYou" or "BucketAlreadyExists")
        {
            // Already created by a prior run — fine.
        }
    }

    public static AzureBlobArtifactStore NewAzureStore(string container)
    {
        new BlobContainerClient(AzuriteConnectionString, container).CreateIfNotExists();
        return new AzureBlobArtifactStore(AzuriteConnectionString, container);
    }

    public static GCSArtifactStore NewGcsStore(string bucket)
    {
        StorageClient client = new StorageClientBuilder
        {
            BaseUri = FakeGcsBaseUri,
            UnauthenticatedAccess = true,
        }.Build();
        try
        {
            client.CreateBucket("test-project", bucket);
        }
        catch (GoogleApiException e) when (e.HttpStatusCode == System.Net.HttpStatusCode.Conflict)
        {
            // Already created by a prior run — fine.
        }
        return new GCSArtifactStore(bucket, client);
    }
}
