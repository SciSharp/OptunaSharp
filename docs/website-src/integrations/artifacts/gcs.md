# Google Cloud Storage

`OptunaSharp.Artifacts.Gcs` implements Optuna's `GCSArtifactStore` using `Google.Cloud.Storage.V1`. It stores bytes in an existing GCS bucket while Core stores artifact metadata on the trial or study. The package targets .NET 8 and .NET 10 and references the `OptunaSharp` Core package and the Google SDK. [Package definition](https://github.com/SciSharp/OptunaSharp/blob/main/src/OptunaSharp.Artifacts.Gcs/OptunaSharp.Artifacts.Gcs.csproj).

```shell
dotnet add package OptunaSharp.Artifacts.Gcs
```

## Credentials and construction

```csharp
GCSArtifactStore(string bucketName, StorageClient? client = null);
```

Omitting `client` calls `StorageClient.Create()`, which uses Application Default Credentials. The implementation's local-development example uses `gcloud auth application-default login`; deployed applications can supply their configured `StorageClient` instead. A client must be authorized for the bucket operations your workflow needs. Bucket creation and access configuration are outside the adapter. [Store implementation](https://github.com/SciSharp/OptunaSharp/blob/main/src/OptunaSharp.Artifacts.Gcs/GCSArtifactStore.cs).

```csharp
using System.IO;
using Google.Cloud.Storage.V1;
using OptunaSharp;
using OptunaSharp.Artifacts;

var client = StorageClient.Create();
var store = new GCSArtifactStore("optimization-artifacts", client);
var study = Optuna.CreateStudy();

// report.json must exist before upload.
string id = ArtifactFunctions.UploadArtifact(
    store, "report.json", study, mimetype: "application/json");

Directory.CreateDirectory("restored");
ArtifactFunctions.DownloadArtifact(store, Path.Combine("restored", $"{id}.json"), id);
```

The store retains the supplied client, exposes no disposal API, and does not dispose it. Keep application-managed client resources available for the duration of transfers. The same store can be passed to the [trial upload and metadata workflow](index.md).

## Buffering, keys, and metadata

`Write` calls `StorageClient.UploadObject(bucketName, artifactId, contentType: null, source: contentBody)`. The key is exactly the artifact ID, with no filename or prefix added. The adapter forwards the source stream without adding an upload buffer of its own. It does not pass custom upload options, generation preconditions, or MIME/encoding headers; MIME type and encoding supplied to `UploadArtifact` remain study/trial metadata. Reusing an ID replaces the object's bytes in the contract tests. [Write implementation](https://github.com/SciSharp/OptunaSharp/blob/main/src/OptunaSharp.Artifacts.Gcs/GCSArtifactStore.cs), [delegation tests](https://github.com/SciSharp/OptunaSharp/blob/main/test/OptunaSharp.Tests.Artifacts/GCSArtifactStoreTests.cs).

`OpenReader` downloads the **entire object into memory**, rewinds that `MemoryStream`, and then returns it. This is a seekable, self-contained stream, not an ongoing network response. Budget memory for the complete artifact, especially with concurrent downloads. `DownloadArtifact` disposes the buffer automatically; direct callers must do so themselves. The adapter uses synchronous SDK methods and exposes no cancellation or progress arguments. [Read implementation](https://github.com/SciSharp/OptunaSharp/blob/main/src/OptunaSharp.Artifacts.Gcs/GCSArtifactStore.cs).

## Errors and retries

A `GoogleApiException` with HTTP 404 during a read becomes `ArtifactNotFound`, preserving the SDK exception as its cause. Other download failures propagate. Removal deliberately differs: `DeleteObject` exceptions, including a 404 for an absent object, propagate as `GoogleApiException`. The backend is therefore not idempotent-on-missing in the same way as the S3 and R2 adapters. [Missing-object tests](https://github.com/SciSharp/OptunaSharp/blob/main/test/OptunaSharp.Tests.Artifacts/GCSArtifactStoreTests.cs).

The adapter adds no retry loop. Core's optional `Backoff` wraps the complete buffered download, but it also retries a missing-object delete because that failure is an SDK exception rather than `ArtifactNotFound`. Configure retries with that distinction in mind; the [shared retry guide](index.md#retries) explains total attempts, blocking waits, and upload seekability.

## Local verification boundary

The repository tests injected clients offline and runs contract tests against `fake-gcs-server`. Its emulator factory explicitly builds a client with `BaseUri` and `UnauthenticatedAccess = true`; this is a local-test configuration. The default-client path requires actual Application Default Credentials and is not covered by the offline suite. Setting an emulator host alone does not exercise that constructor in these tests. [Emulator setup](https://github.com/SciSharp/OptunaSharp/blob/main/test/OptunaSharp.Tests.Artifacts/ArtifactEmulators.cs), [integration tests and coverage note](https://github.com/SciSharp/OptunaSharp/blob/main/test/OptunaSharp.Tests.Artifacts/IntegrationTests.cs).
