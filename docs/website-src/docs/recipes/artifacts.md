# Artifacts

[Artifacts](../artifacts.md) let you attach **large files** to a trial — model snapshots, generated
images, chemical structures, prediction dumps — that are too big for the study database. The study
[storage](../storage.md) keeps the small structured data (params, values, attributes); an *artifact
store* keeps the blobs, linked back to the trial by id. This recipe walks the two standard setups.

> This is the how-to. The [Artifacts guide](../artifacts.md) documents the full `IArtifactStore`
> protocol, every backend, and the `ArtifactFunctions` surface.

## Scenario 1: SQLite + local file system

For a run that stays on one machine, pair the SQLite [`RDBStorage`](rdb.md) with a
`FileSystemArtifactStore` pointed at any directory. Inside the objective, write a file, upload it, and
remember its id on the trial so you can find it later:

```csharp
using OptunaSharp;
using OptunaSharp.Artifacts;
using OptunaSharp.Storages;

using var storage = new RDBStorage("sqlite:///example.db");
var artifactStore = new FileSystemArtifactStore("./artifacts");   // any directory

var study = Optuna.CreateStudy(storage: storage, studyName: "test_study", loadIfExists: true);

double Objective(Trial trial)
{
    double x = trial.SuggestFloat("x", -10, 10);

    // Produce a file for this trial (a model snapshot, an image, a structure dump, …).
    string filePath = GenerateExample(x);
    string artifactId = ArtifactFunctions.UploadArtifact(artifactStore, filePath, trial);
    trial.SetUserAttr("artifact_id", artifactId);   // store the id in the RDB for later reference

    return (x - 2) * (x - 2);
}

study.Optimize(Objective, nTrials: 100);

// Download the artifact attached to the best trial.
var bestArtifactId = (string)study.BestTrial.UserAttrs["artifact_id"]!;
ArtifactFunctions.DownloadArtifact(artifactStore, "./best_artifact.bin", bestArtifactId);
```

Alternatively, list a trial's artifacts without tracking ids yourself — `GetAllArtifactMeta` returns
the `ArtifactMeta` (id, filename, MIME type) recorded on each upload. A `FrozenTrial` needs the
storage passed in:

```csharp
foreach (ArtifactMeta meta in ArtifactFunctions.GetAllArtifactMeta(study.BestTrial, storage))
    ArtifactFunctions.DownloadArtifact(artifactStore, $"./restored/{meta.Filename}", meta.ArtifactId);
```

## Scenario 2: remote database + cloud object store

When optimization is distributed across nodes, a local directory won't do. Keep the study in a remote
database ([`OptunaSharp.Storage.EFCore`](../../integrations/storage/ef-core.md) for PostgreSQL/MySQL)
and the artifacts in cloud object storage. Install `OptunaSharp.Artifacts.S3` and swap the artifact
store for `Boto3ArtifactStore` — **everything else in the objective is identical**:

```csharp
using OptunaSharp.Artifacts;   // OptunaSharp.Artifacts.S3

// Ambient AWS credentials, or pass a configured IAmazonS3 client.
var artifactStore = new Boto3ArtifactStore("example-bucket");
```

Optuna's storage and artifact modules resolve race conditions when many workers upload in parallel, so
reads and writes stay transparent even though the params live in the database and the files live in
the object store. GCS, Azure Blob, and Cloudflare R2 stores work the same way — see the
[cloud artifacts guide](../../integrations/artifacts/index.md).

## A note on large scientific data

Optuna's original tutorial optimizes chemical adsorption structures, saving each candidate structure
as an artifact. The pattern is exactly Scenario 1: whenever a trial produces data too large or too
binary for the RDB — a molecule, an image, a model checkpoint — write it to a file and upload it. For
small structures it's fine to serialize into a [user attribute](attributes.md) instead; the artifact
module is what you reach for once the data outgrows the database.

## Relationship to Optuna

| Optuna (Python) | OptunaSharp |
|-----------------|-------------|
| `from optuna.artifacts import FileSystemArtifactStore` | `new FileSystemArtifactStore("./artifacts")` |
| `upload_artifact(artifact_store=…, file_path=…, study_or_trial=trial)` | `ArtifactFunctions.UploadArtifact(store, filePath, trial)` |
| `download_artifact(artifact_store=…, file_path=…, artifact_id=…)` | `ArtifactFunctions.DownloadArtifact(store, filePath, artifactId)` |
| `Boto3ArtifactStore(client=…, bucket_name="example_bucket")` | `new Boto3ArtifactStore("example-bucket")` |
| `trial.set_user_attr("artifact_id", artifact_id)` | `trial.SetUserAttr("artifact_id", artifactId)` |

## See also

[Artifacts guide](../artifacts.md) · [Storage](../storage.md) · [User attributes](attributes.md) ·
the [cloud artifacts integration](../../integrations/artifacts/index.md).
