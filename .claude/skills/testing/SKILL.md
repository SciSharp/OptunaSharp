---
name: testing
description: How OptunaSharp is tested end-to-end — the six suites (unit, live-Optuna interop, isolated package-parity, Redis + cloud-artifact integration, benchmark), how to run each, the byte-exact parity framework (StudySnapshot / ParityScenario / Ask-Tell replay, bit-exact vs erf-tolerance), the RNG discipline that makes parity possible, test categories and the "green ≠ exercised" self-skip semantics, docker-compose external services, the 3-OS CI matrix, and the prerequisites (NumSharp+OpenBLAS from NuGet, Python+optuna, Docker). Use when running, adding, or debugging any OptunaSharp test, deciding where a test belongs, understanding why a suite skipped, or wiring CI.
keywords: testing, run tests, dotnet test, parity, byte-exact, interop, pythonnet, docker-compose, integration test, test categories, self-skip, Inconclusive, RequiresPython, package parity, oracle replay, StudySnapshot, ParityScenario
keywords-sparse: how to run the tests, why did the test skip, add a parity test, which suite, test the package, live optuna, redis test, artifact emulator, bit-exact, numsharp nuget
---

# testing — how OptunaSharp validates itself

OptunaSharp's north star is **1:1 behavioral parity with Optuna 5.x**, so testing is parity-driven:
tests assert against values produced by **real Optuna** (run live through pythonnet, or replayed from
a captured trajectory), and under a fixed seed the match is **bit-identical** — because every sampler
consumes `np.random.RandomState` through `LazyRandomState` on NumSharp's `MT19937`, never
`System.Random`. That RNG discipline is what makes byte-for-byte differential testing possible; the
`CategoricalChoice` Python-equality rule is the other load-bearing invariant. Break either and parity
tests go red.

## 0. Prerequisites

Core references NumSharp + NumSharp.Interop.OpenBLAS as **NuGet packages** (from nuget.org, version
`$(NumSharpVersion)` in `Directory.Build.props`), so a plain `dotnet build`/`dotnet test` restores
everything — **no submodule init, no native staging**. The OpenBLAS package ships the LAPACK natives
under `runtimes/<rid>/native/` and its `buildTransitive` targets copy them to the test/executable
output automatically, so `np.linalg` (cholesky/eigh/solve → GP, CMA-ES, NSGA-III) is bit-identical to
NumPy at runtime with no setup. Per-suite prereqs: **Python + optuna 5.0.0.dev** (interop) and
**Docker** (service/container suites). The NumSharp source is vendored reference-only at `refs/NumSharp`
(`git submodule update --init refs/NumSharp`) for browsing — it is not part of the build.

## 1. The suites

| Project (in `test/`) | What it proves | Runs for real when… | In `.sln` |
|---|---|---|---|
| **OptunaSharp.Tests** | Unit + differential parity of the pure-managed surface — samplers, pruners, distributions, storages, GP, terminator, importance. The dependable core. | always | yes |
| **OptunaSharp.Tests.Interop** | OptunaSharp matches **live Optuna in-process** (pythonnet) — the differential gate. | a Python w/ `optuna` is discoverable | yes |
| **OptunaSharp.Tests.PackageParity** | The **shipped `.nupkg`**, restored in an isolated container (no `src/`), replays real optuna + xgboost/lightgbm/sklearn oracles. | a packed feed + generated oracles exist | yes |
| **OptunaSharp.Tests.Storage** | `JournalRedisBackend` against a **real Redis** — on-wire encoding interchangeable Python↔C#. | Redis reachable | yes |
| **OptunaSharp.Tests.Artifacts** | S3 / R2 / Azure / GCS artifact stores against **local emulators**. | MinIO/Azurite/fake-gcs reachable | yes |
| **OptunaSharp.Benchmark** | BenchmarkDotNet harness (perf; not a test suite). | never under `dotnet test` | yes |

## 2. Running each suite

```bash
# Unit — the always-real gate (CI filter). Both TFMs.
dotnet test test/OptunaSharp.Tests/OptunaSharp.Tests.csproj -c Release \
  --filter "TestCategory!=OpenBugs&TestCategory!=HighMemory&TestCategory!=RequiresPython"

# Interop — needs live optuna; pin the interpreter for pythonnet
PYTHONNET_PYDLL=/path/to/libpython3.12.so \
  dotnet test test/OptunaSharp.Tests.Interop/OptunaSharp.Tests.Interop.csproj -c Release -f net10.0

# Redis integration — bring the service up first
docker compose -f test/OptunaSharp.Tests.Storage/docker-compose.yml up -d
dotnet test test/OptunaSharp.Tests.Storage -c Release --filter TestCategory=Integration
docker compose -f test/OptunaSharp.Tests.Storage/docker-compose.yml down -v

# Artifact emulators — same shape
docker compose -f test/OptunaSharp.Tests.Artifacts/docker-compose.yml up -d
dotnet test test/OptunaSharp.Tests.Artifacts -c Release --filter TestCategory=Integration

# Package-parity — fully isolated, tests run INSIDE the container (see §6)
OPTUNASHARP_VERSION=<ver> docker compose \
  -f test/OptunaSharp.Tests.PackageParity/docker-compose.yml up --build --exit-code-from package-parity
```

## 3. Green ≠ exercised — the self-skip rule

The parity/integration suites **self-skip to `Inconclusive`** (not fail, not pass) when their
prerequisite is missing — so "the run was green" can mean "the real checks quietly skipped." Only
**OptunaSharp.Tests always asserts**. To tell them apart, read the **Skipped / Inconclusive count** in
the output. To make a missing prerequisite a **hard red** instead (CI does this), set the suite's
`*_REQUIRED` env var:

| Suite | Skips unless… | Force-fail env var |
|---|---|---|
| Interop | Python+optuna found | `OPTUNASHARP_PYTHONNET_REQUIRE_ENGINE=1` |
| Storage (Redis) | Redis reachable | `OPTUNASHARP_REDIS_REQUIRED=1` |
| Artifacts | emulators reachable | `OPTUNASHARP_ARTIFACT_EMULATORS_REQUIRED=1` |
| PackageParity | oracles generated | `PACKAGE_PARITY_REQUIRE=1` |

## 4. Test categories

Typed attributes in `test/OptunaSharp.Tests/TestCategory.cs` (keeps CI filter strings stable):
`[OpenBugs]` (known-failing / not-yet-ported — **CI excludes**), `[Misaligned]` (a documented,
intentional divergence — the test passes), `[RequiresPython]` (live optuna; self-skips), `[HighMemory]`
(**CI excludes**), `[WindowsOnly]` (excluded off-Windows). String categories used elsewhere:
`[TestCategory("Integration")]` (service-backed) and `[TestCategory("RequiresOracle")]` (package-parity
replay). The interop project uses the string form `[TestCategory("RequiresPython")]` (it doesn't
reference the attribute).

## 5. The parity framework (the core methodology)

Live under `test/OptunaSharp.Tests.Interop/`, reusable across suites:

- **`OptunaInteropTestBase`** — one embedded CPython engine per process; **GIL discipline** (every
  `PyObject` touch, incl. `Dispose`, inside `using (Py.GIL())`); helpers `PyExec`/`PyFloat`/`PyStudyState`.
  `_oss_dump_state` serializes the full Python study to JSON (floats via `repr`, so bits survive).
- **`StudySnapshot`** — a complete, comparison-ready study state captured from both sides. `AssertParity`
  compares **bit-exact** (`BitConverter.DoubleToInt64Bits`, NaN==NaN): every trial's number/state,
  params, `params_internal` (`to_internal_repr`), distributions JSON, user/system attrs (structural,
  Python-repr), intermediate values, datetimes.
- **`ParityScenario`** — a seeded differential-fuzz spec (search space of every distribution shape +
  per-trial ops: NaN/±inf reports, unicode/nested attrs, constraints, complete/fail/prune, enqueues),
  run **identically** on OptunaSharp and on Python (the spec crosses as JSON via scope binding, both
  sides derive objective values with the same IEEE op order), then `AssertParity`.
- **Bit-exact vs erf-tolerance.** Assert **bit-exact** for RNG-deterministic samplers — RandomSampler,
  GridSampler, BruteForceSampler, QMCSampler, CmaEsSampler, NSGA-II/III, PartialFixed. Assert to a small
  **tolerance** (~1e-9 continuous, exact branch/state/param-set, non-finite exact) for **TPESampler /
  GPSampler**, which route densities through the FreeBSD `erf` port (~1–3 ULP from ucrt `math.erf`).
- **Ask/Tell replay** (package-parity) — when the objective isn't portable (real xgboost, etc.), Python
  runs it and records the trajectory; the packaged OptunaSharp replays it via `Ask`→suggest→assert→`Tell`,
  reproducing the same suggestions. Bit-exact for RandomSampler (value-independent), tolerance for TPE.
- **Gotchas that cost parity:** scalar `x**2` = `Math.Pow(x,2.0)` (never `x*x`); TPE over int params with
  repeats diverges via NumPy x86-simd-sort tie order (documented, excluded); JSON floats round-trip via
  `repr`↔invariant-parse.

## 6. External services via docker-compose

Each service-backed suite has a `docker-compose.yml` standing up **real, credential-free** services:
Storage → `redis:7-alpine`; Artifacts → Adobe S3Mock (S3+R2; `minio/minio` is no longer pullable), Azurite (Azure), fake-gcs-server (GCS);
PackageParity → its own test container (+ reused Redis via `include:`). A `Session`/`Emulators` helper
gates reachability (self-skip, or hard-fail under the `*_REQUIRED` env). **Two execution models:** the
source suites run `dotnet test` **on the host** reaching `localhost:<port>`; the package-parity gate
runs the test **inside a container** reaching services by **compose name** — so only endpoint-
configurable backends compose into the container model (Redis via `OPTUNASHARP_REDIS_URL` does;
`ArtifactEmulators.cs` still hardcodes `localhost`).

## 7. What CI runs

`.github/workflows/`: **build-and-release**:
- `test`: unit tests across windows/ubuntu/macos × net8/net10, with the exclusion filter.
- `satellite-tests`: every other `test/OptunaSharp.Tests.*` suite, on ubuntu and windows only. The macOS
  arm64 runner has no ML.NET LightGBM native and can't run Linux containers.
  - On ubuntu it brings up the Redis and artifact emulators with docker compose, and Testcontainers
    brings up PostgreSQL and MySQL. The `*_REQUIRED` switches are set there, so a missing service fails.
  - It installs the vendored `optuna` for the live CLI parity cases.
- `interop-test`: live optuna, `REQUIRE_ENGINE=1`. On Linux it also installs the OS-independent parity
  backends, so those suites run instead of self-skipping:
  - BoTorch, pycma 4.4.4, shap, lightgbm and grpcio;
  - the PostgreSQL and MySQL drivers, plus a Redis server;
  - optunahub, plus the registry submodule;
  - the JANUS reference.
- `package`: on every run, `pack_all.sh` packs all 20 packages, then the `verify_strong_name.cs` and
  `verify_packages.cs` gates run.
- The release pipeline runs on `v*` tags.

The other workflows:
- **docs**: DocFX compiles Core. The build job gates PRs; a separate `deploy` job deploys from master.
- **benchmark**: build-smoke always; full run manual.
- **package-parity**: build→package→deploy→test-in-container.
- **documentation**: the in-code doc gate.
- **examples**: compiles every example.

Runner images are pinned (`ubuntu-24.04` / `windows-2025` / `macos-26`) because the bit-exact suites
depend on them. Core-building jobs just restore NumSharp + NumSharp.Interop.OpenBLAS from nuget.org, with
no submodule init or native staging. The satellite job inits `refs/optuna` for its upgrade-test
databases.

## 8. Adding a test — Definition of Done

1. **Read the Optuna source** (`refs/optuna`) and **run it** to capture exact behavior — tests assert
   against real Optuna output, ideally live.
2. Match signatures/names/order/defaults, edge cases (NaN, single-value dists, dynamic spaces,
   None/categorical), and **verbatim error messages/types**.
3. Respect the invariants: `LazyRandomState` (never `System.Random`), `CategoricalChoice` Python
   equality, Core-stays-pure-managed (foreign runtime → satellite package + its own service suite).
4. Reproducibility under a fixed seed must be **bit-identical** to Python where the path is deterministic
   (that's why the RNG rule exists); use the tolerance path only for the erf-routed samplers.
5. Put it in the right suite (§1), tag it (§4), keep the unit filter green, document any intentional
   divergence with `[Misaligned]`.

To port tests from real-world GitHub Optuna usage specifically, use the **`interop-test-porting`** skill.

## 9. Current blockers to know

- **TPE parity over integer-parameter spaces** with repeats is not byte-comparable (NumPy simd-sort tie
  order) — assert RandomSampler bit-exact there, or a continuous-only space for TPE.
