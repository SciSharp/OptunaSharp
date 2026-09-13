# OptunaSharp.Tests.PackageParity — the isolated, package-consuming parity gate

This project answers a question the source-referenced suites cannot: **do the packages we actually ship
work, standalone, against their real external dependency graphs?** It binds to the **packaged** NuGets —
**all 20 of them** (`OptunaSharp` Core + every satellite) — from a local feed (never `src/`), runs in a
**clean Docker container**, and asserts three things: every shipped package restores + loads + carries the
SciSharp strong-name identity; the packaged sampler shows byte-for-byte / proximity parity against **real
Optuna driving real ML packages** (xgboost, lightgbm, scikit-learn); and a service-backed satellite (the
packaged Redis journal backend) round-trips a study through **real Redis**.

## How isolation is enforced

- **PackageReference, not ProjectReference.** Every OptunaSharp package resolves from a local folder feed
  (`NuGet.config` source-maps `OptunaSharp*` → `feed/`), so the tests bind to the shipped assemblies + each
  package's declared deps (from nuget.org) — there is no compile path to `src/`. The 20th package,
  `OptunaSharp.Cli`, is a `dotnet tool` (no `lib/` to reference), so it is installed from the feed and run.
- **Its own `Directory.Build.props`** stops MSBuild's upward search, so the repo-root props (package
  versions, references, strong-naming) never apply here — the packaged third-party deps restore from nuget.org.
- **`OptunaSharp.Storage.Grpc`** carries a `FrameworkReference` to `Microsoft.AspNetCore.App` (its server
  rides Kestrel), so the project supplies that shared framework — exactly what a consumer of it must do.
- **Local convenience:** the `_PackAllOptunaSharpPackagesIntoLocalFeed` MSBuild target (in the `.csproj`,
  hooked `BeforeTargets=Restore`) packs every `src/*` package into `feed/` on first build / any src change,
  so a clean checkout builds with no manual step and no cryptic `NU1301`/`NU1101`. It self-disables where
  there is no `src/` (the CI container), which consumes the pipeline-deployed feed instead.
- **Container build context is this directory only** (`Dockerfile` + `.dockerignore`), so `src/` is not
  present anywhere in the image. CI additionally sparse-checks-out only this folder.

## What it covers

1. **Every shipped package restores + loads + is ours** (`PackagedAssembliesSmokeTests`, no Python/oracle) —
   one case per library NuGet (19): the packaged assembly restores against its real declared deps and loads,
   carrying the SciSharp strong-name token. `PackagedCliToolSmokeTests` covers the 20th (the `optunasharp`
   `dotnet tool`): it installs from the feed and runs. This is the "does the shipped artifact work at all"
   gate — the exact class of failure that once surfaced as `NU1101` (unpublished `NumSharp.Interop.OpenBLAS`).
2. **Sampler parity vs real Optuna + real ML** (`PackageSmokeTests` + `RealPackageParityTests`) — the
   two-phase replay described next.
3. **A service-backed satellite works end-to-end** (`PackagedRedisJournalParityTests`, `RequiresRedis`) —
   the packaged `OptunaSharp.Storage.StackExchange.Redis` journal backend persists a study to **real Redis**
   and replays it exactly on reload through a fresh connection. Self-skips locally without Redis; required
   under compose (which sets `OPTUNASHARP_REDIS_URL`).

## How sampler parity works (two-phase replay)

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
# Local — the build AUTO-PACKS every src/ package into feed/ on first build / whenever any src changes
# (the _PackAllOptunaSharpPackagesIntoLocalFeed MSBuild target in the .csproj, hooked BeforeTargets=Restore),
# so no manual `cp` of a .nupkg is needed (first build packs all 20 — a one-time cost; warm builds skip).
# The smoke + Cli-tool tests then need nothing else — no python, no oracle (the Redis test self-skips):
dotnet test -c Release -f net10.0 --filter "TestCategory!=RequiresOracle"

# To ALSO run the real-package parity tests, generate the oracle first
# (needs python: pip install optuna xgboost lightgbm scikit-learn):
export PACKAGE_PARITY_ORACLE_DIR="$PWD/oracle" PACKAGE_PARITY_REQUIRE=1
python gen_oracle.py
dotnet test -c Release -f net10.0

# Fully isolated (Docker Compose) — mirrors CI (here the feed is pre-dropped, not auto-packed: the
# container has no src/ tree, so the auto-pack target self-disables and CI's packaged artifact is used)
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

The **packaged Redis journal parity** (`PackagedRedisJournalParityTests`) consumes that Redis **now** — all
satellite packages are packed into `feed/` alongside Core. The **artifact-store** parity (packaged
S3/GCS/Azure) still waits on the one change noted above: `ArtifactEmulators.cs` honouring an endpoint env
var so the MinIO/Azurite/fake-gcs emulators compose in by service name.

## Dependency graph restores from nuget.org

`OptunaSharp.Core` depends on `NumSharp` and `NumSharp.Interop.OpenBLAS` (the latter's `<Module>`
initializer force-loads `OpenBlasEngine`); the satellites add their own (AWSSDK, Azure, Google.Cloud,
StackExchange.Redis, ML.NET, …). All are **published to nuget.org**, so each packaged OptunaSharp restores
its **entire** dependency graph from nuget.org — exactly as any downstream consumer would — and OpenBLAS's
native LAPACK binaries arrive via the package's `runtimes/<rid>/native` + `buildTransitive` machinery. The
local feed carries only the freshly-built OptunaSharp family (all 20 `.nupkg`s).

> Historical note: this gate originally **failed at restore** (`NU1101: Unable to find package
> NumSharp.Interop.OpenBLAS`) because OpenBLAS wasn't published and nuget.org's NumSharp was a stale
> `0.60.0` without `np.linalg` — a real finding this isolated gate surfaced. Resolved when NumSharp +
> NumSharp.Interop.OpenBLAS `0.70.0` shipped to nuget.org.
