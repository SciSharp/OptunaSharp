# OptunaSharp.Artifacts.R2

**Cloudflare R2** artifact store for [OptunaSharp](https://github.com/SciSharp/OptunaSharp).

> **OptunaSharp extension** — Optuna itself ships no R2 backend. R2 exposes an S3-compatible API and
> Cloudflare publishes no official .NET SDK for it, so this package reuses
> `OptunaSharp.Artifacts.S3` (the AWS SDK for .NET) pre-configured for the R2 endpoint.

## Install

```bash
dotnet add package OptunaSharp.Artifacts.R2
```

## Usage

```csharp
using OptunaSharp.Artifacts;

var artifactStore = new R2ArtifactStore(
    accountId: "<cloudflare-account-id>",
    accessKeyId: "<r2-access-key-id>",
    secretAccessKey: "<r2-secret-access-key>",
    bucketName: "my-bucket");
ArtifactFunctions.UploadArtifact(artifactStore, filePath, trial);
```

Or pass a pre-configured `IAmazonS3` client for full control over endpoint and credentials.

## License

MIT. OptunaSharp is a derivative of Optuna (© Preferred Networks, Inc.) — see `LICENSE`.
