# OptunaSharp.Tests.Artifacts

Tests for the cloud artifact backends — `OptunaSharp.Artifacts.{S3,Gcs,AzureBlob,R2}`. Two layers.

## Unit tests (default — no external dependencies)

Mock-based (Moq) tests of each store's real logic: the not-found → `ArtifactNotFound` mapping and the
get/put/delete delegation, with the cloud SDK clients mocked (no network).

```bash
dotnet test test/OptunaSharp.Tests.Artifacts -c Release
```

## Integration tests (`Integration` category — dockerized emulators)

Full `Write`/`OpenReader`/`Remove` round-trips against **local, credential-free emulators** — no cloud
account needed:

| Backend | Store | Emulator | Image |
|---|---|---|---|
| S3 | `Boto3ArtifactStore` | MinIO | `minio/minio` |
| Cloudflare R2 | `R2ArtifactStore` | MinIO (R2 is S3-compatible) | `minio/minio` |
| Azure Blob | `AzureBlobArtifactStore` | Azurite (Microsoft's official emulator) | `mcr.microsoft.com/azure-storage/azurite` |
| GCS | `GCSArtifactStore` | fake-gcs-server | `fsouza/fake-gcs-server` |

```bash
# 1. start the emulators
docker compose -f test/OptunaSharp.Tests.Artifacts/docker-compose.yml up -d

# 2. run the integration tests
dotnet test test/OptunaSharp.Tests.Artifacts -c Release --filter TestCategory=Integration

# 3. stop the emulators
docker compose -f test/OptunaSharp.Tests.Artifacts/docker-compose.yml down -v
```

When an emulator port is not reachable the integration tests **self-skip** (`Assert.Inconclusive`), so
a plain `dotnet test` without Docker is safe. Set `OPTUNASHARP_ARTIFACT_EMULATORS_REQUIRED=1` to turn
an absent emulator into a hard failure instead (for CI once wired). To exclude the integration tests
explicitly:

```bash
dotnet test test/OptunaSharp.Tests.Artifacts -c Release --filter "TestCategory!=Integration"
```
