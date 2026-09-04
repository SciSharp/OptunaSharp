# OptunaSharp.Artifacts.AzureBlob

**Azure Blob Storage** artifact store for [OptunaSharp](https://github.com/SciSharp/OptunaSharp).

> **OptunaSharp extension** — Optuna itself ships no Azure backend. Backed by the official
> [Azure.Storage.Blobs](https://www.nuget.org/packages/Azure.Storage.Blobs) and
> [Azure.Identity](https://www.nuget.org/packages/Azure.Identity) clients. A blob container is the
> analog of an S3/GCS bucket.

This is an **optional satellite package**: OptunaSharp Core stays pure-managed and does not depend on
it.

## Install

```bash
dotnet add package OptunaSharp.Artifacts.AzureBlob
```

## Usage

```csharp
using OptunaSharp.Artifacts;

// Connection string:
var artifactStore = new AzureBlobArtifactStore(connectionString, "my-container");

// Or Microsoft Entra ID (DefaultAzureCredential when the credential is omitted):
var byAad = new AzureBlobArtifactStore(
    new Uri("https://<account>.blob.core.windows.net"), "my-container");

// Or a caller-supplied BlobContainerClient (full control):
var byClient = new AzureBlobArtifactStore(containerClient);

ArtifactFunctions.UploadArtifact(artifactStore, filePath, trial);
```

## License

MIT. OptunaSharp is a derivative of Optuna (© Preferred Networks, Inc.) — see `LICENSE`.
