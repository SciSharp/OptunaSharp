# OptunaSharp.Tests.PackageParity — the isolated, package-consuming parity gate

This project answers a question the source-referenced suites cannot: **does the OptunaSharp we actually
ship work, standalone, against its real external dependency graph?** It binds to the **packaged**
`OptunaSharp.nupkg` (never `src/`), runs in a **clean Docker container**, and demonstrates byte-for-byte
/ proximity parity of the packaged sampler against **real Optuna driving real ML packages** (xgboost,
lightgbm, scikit-learn).

## How isolation is enforced

- **PackageReference, not ProjectReference.** `OptunaSharp` resolves from a local folder feed
  (`NuGet.config` → `feed/`), so the tests bind to the shipped assembly + its declared deps
  (`NumSharp`, `NumSharp.Interop.OpenBLAS`) — there is no compile path to `src/`.
- **Its own `Directory.Build.props`** stops MSBuild's upward search, so the repo-root props (the
  NumSharp package version, Core references, strong-naming) never apply here — the packaged deps
  (`NumSharp`, `NumSharp.Interop.OpenBLAS`) restore from nuget.org.
- **Container build context is this directory only** (`Dockerfile` + `.dockerignore`), so `src/` is not
  present anywhere in the image. CI additionally sparse-checks-out only this folder.

## How parity works (two-phase replay)

A gist's real objective (xgboost training, etc.) isn't portable to C#, and the packaged OptunaSharp
can't run xgboost. But a sampler's trajectory is fixed by `(seed, search space, value sequence)` alone,
so:

1. **`gen_oracle.py`** runs the real study (real Optuna + real xgboost/lightgbm/sklearn) and records
   each trial's params + the value the ML package produced → `oracle/<name>.json`.
2. **`RealPackageParityTests`** replays that trajectory against the packaged OptunaSharp via `Ask`/`Tell`:
   it makes the identical `suggest_*` calls (consuming the sampler RNG) and asserts each suggestion
   matches what real Optuna produced, then `Tell`s the real recorded value so the next `Ask` sees the
   same history. **RandomSampler** is asserted **bit-exact** (xgboost/lightgbm, full spaces incl. int
   params); **TPE** to **tolerance** over a continuous-only sklearn-SVM space.

`PackageSmokeTests` needs no Python — it just proves the packaged assembly restores, runs, and carries
the SciSharp strong-name key.

## Run it

```bash
# Local (needs: dotnet, python with `pip install optuna xgboost lightgbm scikit-learn`)
cp <path>/OptunaSharp.<ver>.nupkg feed/
export PACKAGE_PARITY_ORACLE_DIR="$PWD/oracle" PACKAGE_PARITY_REQUIRE=1
python gen_oracle.py
dotnet test -c Release -f net10.0 -p:OptunaSharpPackageVersion=<ver>

# Fully isolated (Docker Compose) — mirrors CI
OPTUNASHARP_VERSION=<ver> docker compose up --build --abort-on-container-exit --exit-code-from package-parity
docker compose down -v
```

The CI pipeline is `.github/workflows/package-parity.yml` (build → package → deploy → test), each stage
isolated.

## docker compose + real external services

The gate is orchestrated with **docker compose**, the same way the source-backed integration suites
stand up real, credential-free services (`OptunaSharp.Tests.Storage` → Redis;
`OptunaSharp.Tests.Artifacts` → MinIO/Azurite/fake-gcs emulators). `docker-compose.yml` here **reuses**
the Redis service definition via `include:` (no duplication) and runs the package-parity test container
alongside it.

One execution-model difference matters: those suites run `dotnet test` **on the host** and reach
services at `localhost:<port>`; this gate runs the tests **inside a container** (so `src/` is never
present), so it reaches services by their compose **service name** (`redis:6379`). Only
endpoint-configurable backends compose cleanly into that model — `JournalRedisBackend` honours
`OPTUNASHARP_REDIS_URL`, so Redis is wired now. The artifact emulators are still hardcoded to
`localhost` in `ArtifactEmulators.cs`, so they compose in identically **once that helper honours an
endpoint env var** — the one change needed to add packaged S3/GCS/Azure artifact parity here.

Redis is up and reachable from the test container today; the **packaged** `JournalRedis` /
artifact-store **parity tests** that consume it land once the satellite packages
(`OptunaSharp.Storage.Redis`, `OptunaSharp.Artifacts.*`) are packed into `feed/` alongside Core.

## Dependency graph restores from nuget.org

`OptunaSharp.Core` depends on `NumSharp` and `NumSharp.Interop.OpenBLAS` (the latter's `<Module>`
initializer force-loads `OpenBlasEngine`). Both are **published to nuget.org** (`0.70.0`), so the
packaged OptunaSharp restores its **entire** dependency graph from nuget.org — exactly as any downstream
consumer would — and the OpenBLAS native LAPACK binaries arrive via the package's `runtimes/<rid>/native`
+ `buildTransitive` machinery. The local feed carries only the freshly-built `OptunaSharp.<version>.nupkg`.

> Historical note: this gate originally **failed at restore** (`NU1101: Unable to find package
> NumSharp.Interop.OpenBLAS`) because OpenBLAS wasn't published and nuget.org's NumSharp was a stale
> `0.60.0` without `np.linalg` — a real finding this isolated gate surfaced. Resolved when NumSharp +
> NumSharp.Interop.OpenBLAS `0.70.0` shipped to nuget.org.
