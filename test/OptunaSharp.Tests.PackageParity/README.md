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

# Fully isolated (Docker) — mirrors CI
docker build -t optunasharp-pkgparity -f Dockerfile . --build-arg OPTUNASHARP_VERSION=<ver>
docker run --rm optunasharp-pkgparity
```

The CI pipeline is `.github/workflows/package-parity.yml` (build → package → deploy → test), each stage
isolated.

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
