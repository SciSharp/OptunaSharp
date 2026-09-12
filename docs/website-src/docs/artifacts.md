# Artifacts

**Artifacts** are large files produced during a trial — trained model weights, generated images,
prediction dumps, logs — stored outside the study database and linked back to the trial or study by
id. The [study storage](storage.md) keeps small structured data (parameters, values, attributes);
artifacts keep the big binary blobs. OptunaSharp ports `optuna.artifacts` into **`OptunaSharp.Artifacts`**.

## The model

An **artifact store** is any backend implementing the `IArtifactStore` protocol:

```csharp
public interface IArtifactStore
{
    Stream OpenReader(string artifactId);       // read; throws ArtifactNotFound if absent
    void   Write(string artifactId, Stream contentBody);
    void   Remove(string artifactId);
}
```

You don't call these directly. The module functions in `ArtifactFunctions` handle id generation,
metadata, and the trial/study wiring:

```csharp
public static string UploadArtifact(IArtifactStore artifactStore, string filePath, object studyOrTrial,
    BaseStorage? storage = null, string? mimetype = null, string? encoding = null);
public static void DownloadArtifact(IArtifactStore artifactStore, string filePath, string artifactId);
public static List<ArtifactMeta> GetAllArtifactMeta(object studyOrTrial, BaseStorage? storage = null);
```

`studyOrTrial` accepts a `Trial`, `FrozenTrial`, or `Study`; `UploadArtifact` returns the generated
artifact id and records an `ArtifactMeta` (id, filename, MIME type, encoding) on the target.

## Uploading and downloading

```csharp
using OptunaSharp;
using OptunaSharp.Artifacts;
using OptunaSharp.Storages;

var storage = new InMemoryStorage();
var study = Optuna.CreateStudy(storage: storage);
var artifactStore = new FileSystemArtifactStore("./artifacts");

study.Optimize(trial =>
{
    double x = trial.SuggestFloat("x", -10, 10);
    double loss = Train(x);

    string modelPath = SaveModelToTempFile(x);
    ArtifactFunctions.UploadArtifact(artifactStore, modelPath, trial);   // -> artifact id, meta on the trial

    return loss;
}, nTrials: 20);

// Later: list the best trial's artifacts and fetch them back. A FrozenTrial needs the storage passed in.
foreach (ArtifactMeta meta in ArtifactFunctions.GetAllArtifactMeta(study.BestTrial, storage))
    ArtifactFunctions.DownloadArtifact(artifactStore, $"./restored/{meta.Filename}", meta.ArtifactId);
```

## Built-in backends (Core)

- **`FileSystemArtifactStore(string basePath)`** — stores artifacts as files under a directory. The
  simplest persistent option; share `basePath` on a network mount for multi-worker runs.
- **`Backoff`** — middleware that wraps another store and retries transient failures with exponential
  backoff. Wrap a flaky (e.g. network) backend:

  ```csharp
  var store = new Backoff(new FileSystemArtifactStore("./artifacts"), maxRetries: 10);
  ```

## Cloud backends (satellite packages)

The [Cloud artifacts integration guide](../integrations/artifacts/index.md) covers upload/download
semantics, retries, client ownership, and the complete setup for each provider:
[S3](../integrations/artifacts/s3.md), [GCS](../integrations/artifacts/gcs.md),
[Azure Blob](../integrations/artifacts/azure-blob.md), and [R2](../integrations/artifacts/r2.md).

Cloud stores live in separate NuGet packages so the Core library stays free of cloud SDK
dependencies — install only the one you use. All of them place their type in the same
`OptunaSharp.Artifacts` namespace, so usage is uniform.

| Backend | Type | Package | Underlying SDK |
|---------|------|---------|----------------|
| Amazon S3 | `Boto3ArtifactStore` | `OptunaSharp.Artifacts.S3` | AWS SDK for .NET (`AWSSDK.S3`) |
| Google Cloud Storage | `GCSArtifactStore` | `OptunaSharp.Artifacts.Gcs` | `Google.Cloud.Storage.V1` |
| Azure Blob Storage † | `AzureBlobArtifactStore` | `OptunaSharp.Artifacts.AzureBlob` | `Azure.Storage.Blobs` + `Azure.Identity` |
| Cloudflare R2 † | `R2ArtifactStore` | `OptunaSharp.Artifacts.R2` | `AWSSDK.S3` (R2 is S3-compatible) |

† Azure Blob and Cloudflare R2 are **OptunaSharp extensions** — Optuna ships no Azure or R2 backend.

```csharp
using OptunaSharp.Artifacts;

// Amazon S3 — ambient AWS credentials, or pass a configured IAmazonS3 client.
var s3 = new Boto3ArtifactStore("my-bucket");

// Google Cloud Storage — Application Default Credentials, or pass a StorageClient.
var gcs = new GCSArtifactStore("my-bucket");

// Azure Blob — connection string, a BlobContainerClient, or Entra ID:
var azure = new AzureBlobArtifactStore(connectionString, "my-container");
var azureAad = new AzureBlobArtifactStore(
    new Uri("https://acct.blob.core.windows.net"), "my-container");   // DefaultAzureCredential

// Cloudflare R2 — account id + R2 token, or a pre-configured IAmazonS3 client.
var r2 = new R2ArtifactStore("account-id", "access-key", "secret-key", "my-bucket");
```

Constructor reference:

```csharp
Boto3ArtifactStore(string bucketName, IAmazonS3? client = null, bool avoidBufCopy = false);
GCSArtifactStore(string bucketName, StorageClient? client = null);
AzureBlobArtifactStore(BlobContainerClient container);
AzureBlobArtifactStore(string connectionString, string containerName);
AzureBlobArtifactStore(Uri serviceUri, string containerName, TokenCredential? credential = null);
R2ArtifactStore(string accountId, string accessKeyId, string secretAccessKey, string bucketName, bool avoidBufCopy = false);
R2ArtifactStore(IAmazonS3 client, string bucketName, bool avoidBufCopy = false);
```

Each cloud store maps the SDK's "not found" (S3 `NoSuchKey`/404, a GCS/Azure 404) to `ArtifactNotFound`
on read, matching the protocol. `R2ArtifactStore` simply configures an `AWSSDK.S3` client for the R2
endpoint and delegates to `Boto3ArtifactStore`, so any S3-compatible endpoint works the same way.

## Handling a missing artifact

`OpenReader` (and `DownloadArtifact`) throw `ArtifactNotFound` (a subclass of `OptunaError`, defined in
`OptunaSharp.Artifacts`) when the id does not exist:

```csharp
try
{
    ArtifactFunctions.DownloadArtifact(store, "./out.bin", artifactId);
}
catch (ArtifactNotFound)
{
    // the artifact was removed or never existed
}
```

## Relationship to Optuna

| Optuna (Python) | OptunaSharp |
|-----------------|-------------|
| `FileSystemArtifactStore("./artifacts")` | `new FileSystemArtifactStore("./artifacts")` |
| `Boto3ArtifactStore("my-bucket")` | `new Boto3ArtifactStore("my-bucket")` |
| `GCSArtifactStore("my-bucket")` | `new GCSArtifactStore("my-bucket")` |
| `upload_artifact(artifact_store=…, file_path=…, study_or_trial=trial)` | `ArtifactFunctions.UploadArtifact(store, filePath, trial)` |
| `download_artifact(...)` / `get_all_artifact_meta(...)` | `ArtifactFunctions.DownloadArtifact(...)` / `GetAllArtifactMeta(...)` |

## See also

[Storage](storage.md) — the study store, a separate concern · [Study and Trials](study-and-trials.md) ·
the [API reference](../api/index.md).
