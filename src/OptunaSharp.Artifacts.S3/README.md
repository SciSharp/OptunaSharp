# OptunaSharp.Artifacts.S3

Amazon **S3** artifact store for [OptunaSharp](https://github.com/SciSharp/OptunaSharp) — a port of
Optuna's `optuna.artifacts.Boto3ArtifactStore`, backed by the official
[AWS SDK for .NET](https://www.nuget.org/packages/AWSSDK.S3) (`AWSSDK.S3`).

This is an **optional satellite package**: OptunaSharp Core stays pure-managed and does not depend on
it. Install it only when you store artifacts in S3.

## Install

```bash
dotnet add package OptunaSharp.Artifacts.S3
```

## Usage

```csharp
using OptunaSharp.Artifacts;

var artifactStore = new Boto3ArtifactStore("my-bucket");
// inside an objective:
ArtifactFunctions.UploadArtifact(artifactStore, filePath, trial);
```

Pass your own `IAmazonS3` client to control credentials, region, or endpoint — including any
S3-compatible store. (For Cloudflare R2 specifically, see `OptunaSharp.Artifacts.R2`.)

## License

MIT. OptunaSharp is a derivative of Optuna (© Preferred Networks, Inc.) — see `LICENSE`.
