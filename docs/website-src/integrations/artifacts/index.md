# Cloud artifact stores

Artifact integrations put model files, evaluation outputs, images, and other trial results in object storage. The study database keeps the artifact's metadata and ID; the artifact store keeps its bytes. Install the provider you use and pass its `IArtifactStore` to the same Core `ArtifactFunctions` API. All four providers use the `OptunaSharp.Artifacts` namespace and target .NET 8 and .NET 10.

| Integration | NuGet package | Public store | SDK |
|---|---|---|---|
| [Amazon S3 and S3-compatible storage](s3.md) | `OptunaSharp.Artifacts.S3` | `Boto3ArtifactStore` | `AWSSDK.S3` |
| [Google Cloud Storage](gcs.md) | `OptunaSharp.Artifacts.Gcs` | `GCSArtifactStore` | `Google.Cloud.Storage.V1` |
| [Azure Blob Storage](azure-blob.md) | `OptunaSharp.Artifacts.AzureBlob` | `AzureBlobArtifactStore` | `Azure.Storage.Blobs`, `Azure.Identity` |
| [Cloudflare R2](r2.md) | `OptunaSharp.Artifacts.R2` | `R2ArtifactStore` | S3 integration and `AWSSDK.S3` |

S3 and GCS port Optuna artifact backends. Azure Blob and R2 are OptunaSharp extensions. Core supplies the interface, metadata functions, local `FileSystemArtifactStore`, and `Backoff`; installing Core alone does not bring in cloud SDKs. See the [artifact concepts guide](../../docs/artifacts.md) for the local-store workflow. The provider project files linked from each integration page define their dependencies.

## Upload, discover, and restore

This example uses S3; replacing the store construction with any of the other providers leaves the workflow unchanged. Create the bucket separately and configure the SDK's credentials and region before running.

```shell
dotnet add package OptunaSharp.Artifacts.S3
```

```csharp
using System;
using System.IO;
using Amazon.S3;
using OptunaSharp;
using OptunaSharp.Artifacts;
using OptunaSharp.Storages;

using var client = new AmazonS3Client();
IArtifactStore store = new Boto3ArtifactStore("optimization-artifacts", client);
var storage = new InMemoryStorage();
var study = Optuna.CreateStudy(storage: storage);

study.Optimize(trial =>
{
    double x = trial.SuggestFloat("x", -10, 10);
    double loss = (x - 2) * (x - 2);
    string source = Path.Combine(Path.GetTempPath(), $"trial-{Guid.NewGuid():N}.txt");
    try
    {
        File.WriteAllText(source, $"x={x}; loss={loss}");
        ArtifactFunctions.UploadArtifact(store, source, trial, mimetype: "text/plain");
    }
    finally
    {
        File.Delete(source);
    }
    return loss;
}, nTrials: 10);

Directory.CreateDirectory("restored");
foreach (ArtifactMeta meta in ArtifactFunctions.GetAllArtifactMeta(
    study.BestTrial, storage))
{
    string destination = Path.Combine("restored", $"{meta.ArtifactId}-{meta.Filename}");
    ArtifactFunctions.DownloadArtifact(store, destination, meta.ArtifactId);
}
```

`UploadArtifact` returns a new GUID string. `Trial` and `Study` carry their own storage; a `FrozenTrial`, including `study.BestTrial`, requires the `storage` argument. `GetAllArtifactMeta(study)` lists artifacts attached directly to that study, not artifacts attached to its trials. Use persistent [study storage](../../docs/storage.md) when metadata must survive the process. [Core upload, metadata, and download implementation](https://github.com/SciSharp/OptunaSharp/blob/master/src/OptunaSharp.Core/Artifacts/ArtifactFunctions.cs).

## Object names and metadata

Each provider uses the artifact ID verbatim as the object key or blob name. A normal upload therefore creates an object such as `a15d5b7d-75b2-4e45-917a-2a5c5c1a8c1e`; it does not add the study name, trial number, filename, or a directory prefix. There is no prefix constructor option. The original basename, MIME type, and optional encoding are serialized in a study/trial system attribute named `artifacts:<artifactId>`. These metadata values are not passed as object HTTP headers by these adapters. [Artifact metadata implementation](https://github.com/SciSharp/OptunaSharp/blob/master/src/OptunaSharp.Core/Artifacts/ArtifactFunctions.cs).

Choose the bucket/container and the matching study storage together when restoring a study: metadata contains no provider, endpoint, or bucket name. The cloud contract tests also exercise manually supplied nested IDs, overwrite of an existing ID, and empty, 1,500-byte, and 2 MiB payloads. Those are test cases, not advertised service size limits. [Provider contract tests](https://github.com/SciSharp/OptunaSharp/blob/master/test/OptunaSharp.Tests.Artifacts/IntegrationTests.cs).

## Failure and lifetime behavior

Uploads write metadata **before** opening and transferring the file. A file or network failure can leave metadata for an object that was never successfully stored; there is no transaction spanning study storage and object storage. Calling `UploadArtifact` again generates another ID. Removing an object through `store.Remove(id)` does not remove its study/trial metadata. [Core implementation](https://github.com/SciSharp/OptunaSharp/blob/master/src/OptunaSharp.Core/Artifacts/ArtifactFunctions.cs).

`DownloadArtifact` refuses to overwrite an existing local file, opens the destination with `FileMode.CreateNew`, and disposes its reader and writer. Create the destination directory first. A failed transfer can leave a partial destination file. Direct `OpenReader` callers must dispose the returned stream. The store interface is synchronous and has no cancellation, progress, listing, signed-URL, or disposal API. The cloud adapters neither create buckets/containers nor dispose their injected SDK clients. [Store interface](https://github.com/SciSharp/OptunaSharp/blob/master/src/OptunaSharp.Core/Artifacts/ArtifactStore.cs), [download implementation](https://github.com/SciSharp/OptunaSharp/blob/master/src/OptunaSharp.Core/Artifacts/ArtifactFunctions.cs).

Missing-object behavior on reads is uniform; deletion differs:

| Provider | Missing object on read | Missing object on removal |
|---|---|---|
| S3 / R2 | `ArtifactNotFound` for `NoSuchKey` or HTTP 404 | S3 deletion succeeds when the key is absent |
| GCS | `ArtifactNotFound` for HTTP 404 | SDK `GoogleApiException` propagates |
| Azure Blob | `ArtifactNotFound` for HTTP 404 | `ArtifactNotFound` for HTTP 404 |

Other SDK errors propagate. A translated 404 may also indicate a missing bucket or container, so the exception alone does not prove that only the object is missing. Provider unit tests and emulator contracts cover the mappings described on the individual pages.

## Retries

The adapters add no retry policy of their own. Their SDK clients retain their configured behavior; Core offers an additional wrapper:

```csharp
IArtifactStore resilientStore = new Backoff(
    store, maxRetries: 4, multiplier: 2, minDelay: 0.1, maxDelay: 2);
```

Despite its name, `maxRetries` is the total number of attempts, including the first. The wrapper catches every exception except `ArtifactNotFound`, sleeps synchronously between attempts, and rethrows on the final failure. For writes it seeks the input stream back to position zero, so a retry requires a seekable stream; the file stream used by `UploadArtifact` qualifies. `OpenReader` retries opening the stream, not a failure encountered while subsequently copying a streaming response. GCS's fully buffered read finishes inside `OpenReader`, so that download is within the retry boundary. GCS's unmapped 404 on deletion is retried like any other SDK exception. Account for both SDK and wrapper retries when choosing latency budgets. [Backoff implementation](https://github.com/SciSharp/OptunaSharp/blob/master/src/OptunaSharp.Core/Artifacts/Backoff.cs).
