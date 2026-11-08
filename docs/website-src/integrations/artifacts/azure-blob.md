# Azure Blob Storage

`OptunaSharp.Artifacts.AzureBlob` provides `AzureBlobArtifactStore`, an OptunaSharp extension backed by `Azure.Storage.Blobs` and `Azure.Identity`. It uses an existing blob container for bytes and the normal Core artifact metadata workflow for trial/study associations. The package targets .NET 8 and .NET 10. [Package definition](https://github.com/SciSharp/OptunaSharp/blob/main/src/OptunaSharp.Artifacts.AzureBlob/OptunaSharp.Artifacts.AzureBlob.csproj).

```shell
dotnet add package OptunaSharp.Artifacts.AzureBlob
```

## Choose a connection path

```csharp
AzureBlobArtifactStore(BlobContainerClient container);
AzureBlobArtifactStore(string connectionString, string containerName);
AzureBlobArtifactStore(
    Uri serviceUri,
    string containerName,
    TokenCredential? credential = null);
```

The connection-string overload creates a `BlobServiceClient` and selects the container. The URI overload accepts a **blob service endpoint**, such as `https://<account>.blob.core.windows.net`, and a separate container name. It uses the supplied `TokenCredential`, or `DefaultAzureCredential` when omitted. Inject a `BlobContainerClient` when your application needs direct control over client construction and options. None of the constructors creates a container. [Constructor implementation](https://github.com/SciSharp/OptunaSharp/blob/main/src/OptunaSharp.Artifacts.AzureBlob/AzureBlobArtifactStore.cs).

```csharp
using System;
using System.IO;
using OptunaSharp;
using OptunaSharp.Artifacts;

// Configure an identity with access to this existing container.
var store = new AzureBlobArtifactStore(
    new Uri("https://exampleaccount.blob.core.windows.net"),
    "optimization-artifacts");
var study = Optuna.CreateStudy();

// model.zip must exist before upload.
string id = ArtifactFunctions.UploadArtifact(store, "model.zip", study);
Directory.CreateDirectory("restored");
ArtifactFunctions.DownloadArtifact(store, Path.Combine("restored", $"{id}.zip"), id);
```

For a connection string, read it from application configuration and pass it explicitly:

```csharp
string connectionString = Environment.GetEnvironmentVariable("AZURE_STORAGE_CONNECTION_STRING")
    ?? throw new InvalidOperationException("Configure AZURE_STORAGE_CONNECTION_STRING.");
var connectionStringStore = new AzureBlobArtifactStore(
    connectionString, "optimization-artifacts");
```

This environment lookup is application code; the artifact store does not automatically read that variable. Authentication and service/client configuration stay with the Azure SDK. The wrapper has no disposal API and does not dispose an injected container client. [Store implementation](https://github.com/SciSharp/OptunaSharp/blob/main/src/OptunaSharp.Artifacts.AzureBlob/AzureBlobArtifactStore.cs).

## Requests and object names

Every operation resolves `_container.GetBlobClient(artifactId)`. The blob name is therefore exactly the artifact ID, not the uploaded filename or a study-specific path. `Write` invokes `Upload(contentBody, overwrite: true)`, so a repeated ID overwrites the blob. The wrapper does not add a full-file memory buffer; SDK upload behavior is delegated to `BlobClient`. It does not supply upload headers, lease conditions, tier, or version options. Filename, MIME type, and encoding from Core's `UploadArtifact` are metadata in study storage, not blob headers set by this adapter. [Implementation](https://github.com/SciSharp/OptunaSharp/blob/main/src/OptunaSharp.Artifacts.AzureBlob/AzureBlobArtifactStore.cs), [overwrite delegation test](https://github.com/SciSharp/OptunaSharp/blob/main/test/OptunaSharp.Tests.Artifacts/AzureBlobArtifactStoreTests.cs).

`OpenReader` returns `DownloadStreaming().Value.Content`. `DownloadArtifact` copies and disposes that response stream; a direct caller must dispose it. All three store methods are synchronous and expose no cancellation or progress argument. An upload/download is not transactional with study metadata; see the [shared artifact lifecycle](index.md#failure-and-lifetime-behavior).

## Errors and retries

Reads and deletes both translate `RequestFailedException` with status 404 to `ArtifactNotFound`, preserving the original exception. Other Azure failures propagate, including status 403. Deleting a missing blob therefore throws; the adapter uses `Delete`, not `DeleteIfExists`. [Error mapping tests](https://github.com/SciSharp/OptunaSharp/blob/main/test/OptunaSharp.Tests.Artifacts/AzureBlobArtifactStoreTests.cs).

The adapter adds no retry loop. A configured Azure client can control SDK behavior, and Core's `Backoff` can wrap the store. The wrapper immediately rethrows `ArtifactNotFound`; its read retry boundary ends once a response stream has been returned. A later failure copying the response is outside that boundary. See [retry behavior](index.md#retries).

## What the tests cover

Unit tests cover all constructor paths, upload with overwrite, blob selection, and read/delete error propagation. Azurite integration tests exercise actual bytes, empty and 2 MiB payloads, nested IDs, replacement, removal, and missing-object behavior. The test setup explicitly creates the container before constructing the store. These are emulator checks rather than live Azure identity/authorization tests. [Emulator factory](https://github.com/SciSharp/OptunaSharp/blob/main/test/OptunaSharp.Tests.Artifacts/ArtifactEmulators.cs), [contract tests](https://github.com/SciSharp/OptunaSharp/blob/main/test/OptunaSharp.Tests.Artifacts/IntegrationTests.cs).
