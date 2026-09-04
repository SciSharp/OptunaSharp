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
  conditional sibling-NumSharp/Core references, strong-naming) never apply here.
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
artifact-store **parity tests** that consume it land once (1) the OpenBLAS blocker below is resolved so
the packaged assembly loads, and (2) the satellite packages (`OptunaSharp.Storage.Redis`,
`OptunaSharp.Artifacts.*`) are packed into `feed/` alongside Core.

## ⚠️ Known blocker this gate surfaces (as of the first run)

The gate currently **fails at restore** with:

```
NU1101: Unable to find package NumSharp.Interop.OpenBLAS.
```

This is a **real finding, not a harness bug**: `OptunaSharp.Core` hard-depends on
`NumSharp.Interop.OpenBLAS 0.60.0` (and its `<Module>` initializer force-loads `OpenBlasEngine`), but
that package **is not published to nuget.org** (`OptunaSharp.Core.csproj` line ~69 documents this). So
the packaged OptunaSharp is **not restorable by any consumer that lacks the sibling NumSharp checkout** —
including this container and, once CI runs on a remote, the existing build/interop jobs. The gate will
go green once the OpenBLAS dependency is published or made optional (the satellite-package pattern the
project's own `CLAUDE.md` prescribes for foreign-runtime deps).
