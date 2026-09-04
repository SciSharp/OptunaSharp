# OptunaSharp.Artifacts.Gcs

**Google Cloud Storage (GCS)** artifact store for
[OptunaSharp](https://github.com/SciSharp/OptunaSharp) — a port of Optuna's
`optuna.artifacts.GCSArtifactStore`, backed by the official
[Google.Cloud.Storage.V1](https://www.nuget.org/packages/Google.Cloud.Storage.V1) client.

This is an **optional satellite package**: OptunaSharp Core stays pure-managed and does not depend on
it. Install it only when you store artifacts in GCS.

## Install

```bash
dotnet add package OptunaSharp.Artifacts.Gcs
```

## Usage

```csharp
using OptunaSharp.Artifacts;

var artifactStore = new GCSArtifactStore("my-bucket");
// inside an objective:
ArtifactFunctions.UploadArtifact(artifactStore, filePath, trial);
```

Before running, install the Google Cloud SDK and run `gcloud auth application-default login` so the
client can find credentials, or pass your own `StorageClient`.

## License

MIT. OptunaSharp is a derivative of Optuna (© Preferred Networks, Inc.) — see `LICENSE`.
