# OptunaSharp Project Instructions

OptunaSharp is a **.NET port of [Optuna](https://github.com/optuna/optuna)** — the define-by-run
hyperparameter-optimization framework — targeting **1-to-1 API and behavioral compatibility with
Optuna 5.x**. It is a [SciSharp STACK](https://github.com/SciSharp) project and builds on
[NumSharp](https://github.com/SciSharp/NumSharp) for numerical work.

> **Status: core surface ported.** The build/test/package/CI pipeline has been green from day one, and
> the port has grown well past scaffolding: the define-by-run core (Study / Trial / Distributions /
> Storages), the sampler suite (Random / Grid / TPE / CMA-ES / GP / QMC / BruteForce / NSGA-II / III,
> plus beyond-Optuna AGE-MOEA-2 / SMS-EMOA / LM-CMA / JANUS), all pruners, importance (fANOVA / MDI /
> PED-ANOVA / SHAP), terminator, the visualization data layer, and optional satellite packages
> (SQLite / EF Core / gRPC / Redis storage, cloud artifacts, Plotly viz, LightGBM / ML.NET / sklearn
> integration, an `optunasharp` CLI, and pythonnet bridges to OptunaHub / BoTorch / PyCma). This file
> (`CLAUDE.md`) still documents the **infrastructure**; see [`docs/PORTING_PLAN.md`](../docs/PORTING_PLAN.md)
> for the phase-by-phase algorithm status.

**The port is governed by [`docs/PORTING_PLAN.md`](../docs/PORTING_PLAN.md)** — the authoritative
roadmap (phases, package map, per-phase gates, timeline, risk register). Read it before porting any
Optuna module. This file (`CLAUDE.md`) documents the **infrastructure**; the plan documents **what
to build and in what order**.

---

## Reference sources

- **Optuna (the source of truth):** a full clone lives at `K:/source/optuna` (pinned in the plan) and
  is **vendored in-repo as a pinned git submodule at `refs/optuna`** (`github.com/optuna/optuna` @
  `01ddd17`, the v5.0.0 release) — alongside the other reference-only submodules, so the reference source restores
  reproducibly from a clean checkout (`git submodule update --init refs/optuna`). Use it as the
  authoritative reference for API behavior, edge cases, defaults, and error messages. **Run actual
  Python/Optuna first, observe, then replicate in C#.**
- **rustuna (accelerator reference):** `K:/source/rustuna` — a Rust Optuna port, useful for the
  SQLite schema and Sobol direction numbers (see the plan).
- **NumSharp (the numerical foundation + the infra template):** consumed from **NuGet** (`NumSharp` +
  `NumSharp.Interop.OpenBLAS`, version single-sourced as `$(NumSharpVersion)` in
  `Directory.Build.props`), and **vendored reference-only as a pinned git submodule at `refs/NumSharp`**
  (`SciSharp/NumSharp` @ the matching `v$(NumSharpVersion)` release tag — `git submodule update --init
  refs/NumSharp`) for offline browsing / LLM cross-referencing; it is NOT part of the build. A local
  sibling `K:/source/NumSharp` is also fine for reference. OptunaSharp's whole stack is modeled on
  NumSharp's; when in doubt about an infra decision, look at how NumSharp does it.

Run real Optuna to observe behavior:

```bash
python -c "import optuna; optuna.logging.set_verbosity(optuna.logging.WARNING); \
  s=optuna.create_study(); s.optimize(lambda t:(t.suggest_float('x',-10,10)-2)**2, n_trials=20); \
  print(s.best_params, s.best_value)"
```

---

## Core principles (Optuna is the source of truth)

1. **Match Optuna exactly.** Behavior, defaults, parameter names/order, edge cases (NaN,
   single-value distributions, dynamic search spaces, `None`/categorical choices), and **verbatim
   error messages** where Optuna raises typed errors.
2. **Match Optuna's implementation structure**, not just its behavior — so a future reader can diff
   the C# against the Python.
3. **Breaking changes are OK** when they bring us closer to Optuna.
4. **Test from Optuna output.** Tests assert against values produced by running real Optuna — ideally
   **live** through the interop harness (see below).

Four load-bearing rules from the plan that infra must respect (full detail in
[`docs/PORTING_PLAN.md`](../docs/PORTING_PLAN.md) §2):

- **RNG discipline.** Every sampler consumes `np.random.RandomState` through a `LazyRandomState`
  port (built on NumSharp's `MT19937`). **Never `System.Random`.** This is what makes differential
  testing bit-exact against Python.
- **Categorical value semantics.** `CategoricalChoice` implements **Python equality** (`True == 1`,
  `1.0 == 1`, `nan != nan`). This is the one place C# object equality silently diverges — pin it with
  tests.
- **Package optionality mirrors Optuna's.** `OptunaSharp` (Core) stays **pure managed (NumSharp
  only)**. Anything needing a foreign runtime (ML.NET, TorchSharp, Redis, SQLite, cloud SDKs, gRPC)
  is a **separate NuGet package** — the `NumSharp.Interop.OpenBLAS` pattern (see the plan's package
  map).
- **Threading.** Trial-level parallelism (`n_jobs`, distributed workers) is supported and tested;
  numeric kernels stay single-threaded (NumSharp house rule). The design (threads, locks and their
  order, per-layer guarantees, the rules for objectives, samplers and storages) is
  [`docs/THREADING.md`](../docs/THREADING.md).

---

## Repository layout

```
OptunaSharp/
├── src/
│   ├── OptunaSharp.Core/            # THE package — OptunaSharp (net8.0;net10.0). The pure-managed library.
│   │   ├── OptunaSharp.Core.csproj  #   packable; PackageIcon; NuGet NumSharp + OpenBLAS reference
│   │   ├── Assembly/Properties.cs   #   InternalsVisibleTo (must carry PublicKey) + friend-key const
│   │   ├── Version.cs               #   OptunaSharp.OptunaSharpVersion (the version constant)
│   │   └── README.md                #   packed into the .nupkg
│   └── OptunaSharp.*/               # optional satellite packages — Storage.{Sqlite,EFCore,Grpc,Redis,…}, Visualization.Plotly, Integration.{LightGbm,ML.NET}, Interop.{pythonnet,BoTorch,PyCma}, Artifacts.{S3,Gcs,AzureBlob,R2}, Cli
├── refs/                            # reference-only pinned submodules (NOT built) — optuna @ v5.0.0 (01ddd17, the porting reference), NumSharp @ v0.70.0, optunahub @ v0.5.0, optuna-integration @ v5.0.0, shap, scikit-learn
├── test/
│   ├── OptunaSharp.Tests/           # unit tests — MSTest v3 + AwesomeAssertions
│   ├── OptunaSharp.Tests.Interop/   # LIVE-Optuna parity harness (pythonnet)
│   └── OptunaSharp.Benchmark/       # BenchmarkDotNet harness (template; grows with the ops)
├── docs/
│   ├── PORTING_PLAN.md              # the governing conversion roadmap
│   ├── THREADING.md                 # the multithreading design: threads, lock order, guarantees, rules
│   └── website-src/                 # DocFX site (docfx.json, conceptual docs, generated API ref)
├── .github/
│   ├── workflows/                   # build-and-release · package-parity · docs · documentation · examples · benchmark
│   ├── scripts/                     # pack_all.sh · verify_strong_name.cs · verify_packages.cs · check_documentation.cs
│   ├── ISSUE_TEMPLATE/ · PULL_REQUEST_TEMPLATE.md · FUNDING.yml · dependabot.yml
├── Directory.Build.props            # strong-naming + shared package metadata + version
├── Open.snk                         # Microsoft's published OSS key (shared across the SciSharp STACK)
├── NuGet.Config                     # nuget.org for every id; OptunaSharp* source-mapped to the package-parity feed/ (IDEs read only this file)
├── .editorconfig / .gitattributes / .gitignore / .git-blame-ignore-revs
├── LICENSE (MIT) · SECURITY.md · CITATION.cff · CONTRIBUTING.md · CODE_OF_CONDUCT.md
├── README.md
└── SciSharp.OptunaSharp.sln
```

### Projects

| Project | TFMs | Kind | Key deps | Purpose |
|---------|------|------|----------|---------|
| `src/OptunaSharp.Core` | net8.0; net10.0 | library, **packable** (`PackageId=OptunaSharp`) | NumSharp | The library. `RootNamespace`/`AssemblyName` = `OptunaSharp`. Emits XML docs. |
| `test/OptunaSharp.Tests` | net8.0; net10.0 | MSTest exe, `IsPackable=false` | MSTest 4.5.2, AwesomeAssertions 9.6.0 | Unit tests. |
| `test/OptunaSharp.Tests.Interop` | net8.0; net10.0 | MSTest exe, `IsPackable=false` | MSTest, AwesomeAssertions, **pythonnet 3.2.1** (Python 3.11–3.15) | Byte/behavior parity vs **live Optuna** in-process. |
| `test/OptunaSharp.Benchmark` | net8.0; net10.0 | exe, `IsPackable=false` | BenchmarkDotNet 0.15.8 | Perf harness (template). |

The solution (`SciSharp.OptunaSharp.sln`) holds these four and every satellite and test project; NumSharp is a **NuGet
package dependency** (see below), not a solution member.

---

## The NumSharp reference — NuGet (`NumSharp` + `NumSharp.Interop.OpenBLAS`)

`OptunaSharp.Core` depends on **NumSharp** and its native LAPACK/CBLAS backend
**NumSharp.Interop.OpenBLAS**, both consumed as **NuGet packages from nuget.org** (SciSharp STACK) —
two plain `PackageReference`s in `OptunaSharp.Core.csproj`, versioned by the single-sourced
`$(NumSharpVersion)` property in `Directory.Build.props` (currently `0.70.0`; bump that one value to
move to a newer release). **Build and CI restore these from nuget.org, always** — there is no submodule
init and no OpenBLAS-native staging step anymore.

- **OpenBLAS ships the natives.** `NumSharp.Interop.OpenBLAS` carries the prebuilt scipy-openblas
  binaries NumPy itself bundles, under `runtimes/<rid>/native/`; the package's `buildTransitive`
  targets copy the right RID's natives to the consuming executable/test output automatically (no
  `fetch_openblas.py`). So `np.linalg` (cholesky / eigh / solve / matrix_rank / svd / qr) computes
  through LAPACK and is **bit-identical to NumPy** (verified byte-for-byte vs numpy 2.4.2) — the basis
  for the byte-exact linalg-backed samplers (NSGA-III's hyperplane solve, CMA-ES's eigendecomposition,
  the GP stack's Cholesky). OpenBLAS depends on the same NumSharp version, so pack records both as
  direct dependencies.
- **The backend is force-loaded.** OpenBLAS self-registers from its own `[ModuleInitializer]`, but the
  CLR runs a referenced assembly's initializer only when the assembly is first touched.
  `src/OptunaSharp.Core/Internal/OpenBlasActivation.cs` touches `OpenBlasEngine.LapackAvailable` from
  Core's own `[ModuleInitializer]` to force the load; without it `np.linalg.cholesky` throws
  `OpenBlasMissingBackendException`.
- **Reference-only source at `refs/NumSharp`.** The matching NumSharp source is vendored as a pinned
  submodule at `refs/NumSharp` (`SciSharp/NumSharp` @ the `v$(NumSharpVersion)` release tag), for
  offline browsing / LLM cross-referencing — **NOT built, never on the build path**. Restore it with
  `git submodule update --init refs/NumSharp`; a local sibling `K:/source/NumSharp` also works for
  reference. It does NOT need NumSharp's own nested submodules (numpy/pocketfft/OpenBLAS-source).

---

## Strong naming

Every assembly is strong-named via `Directory.Build.props` (repo-wide) with **`Open.snk`** at the
repo root — Microsoft's *published* open-source key (596 bytes, sha256
`b897629e0b20090c9219ac80392c037fa4cfcc2bdc21d3c45d9bf74b8df0f671`, public key token
**`cc7b13ffcd2ddd51`**). It is the same key `netstandard`, NumSharp, and TensorFlow.NET use, so it
keeps OptunaSharp's identity consistent with the SciSharp STACK.

**This is IDENTITY, not authenticity** — the private half is public by design, so `InternalsVisibleTo`
is *not* an access control. Real authenticity would be Authenticode + NuGet author signing.

Consequences to know:
- **Every `InternalsVisibleTo` MUST carry `PublicKey=`** — a keyless one is a hard `CS1726` from a
  signed assembly. The literal lives once in `OptunaSharpFriendKey.PublicKey`
  (`src/OptunaSharp.Core/Assembly/Properties.cs`). Current friends: `OptunaSharp.Tests`,
  `OptunaSharp.Tests.Interop`, `OptunaSharp.DotNetRunScript`.
- **`Open.snk` is pinned `binary` in `.gitattributes`** — without that, `* text=auto eol=lf` would
  rewrite CRLF bytes inside the key and silently corrupt it. The CI "Key is present and intact" step
  asserts its size + sha256.
- The gate `.github/scripts/verify_strong_name.cs` reads PE metadata straight out of each packed
  `.nupkg` and fails unless every `lib/**/*.dll` carries token `cc7b13ffcd2ddd51` with a **real**
  (not delay-signed) signature — and fails if it finds nothing to check (no vacuous pass).

Opt a project out with `<SignAssembly>false</SignAssembly>` (props is imported before the body, so
the project wins).

---

## Build & test

```bash
# Build everything
dotnet build SciSharp.OptunaSharp.sln -c Release

# Unit tests (what per-push CI runs — both TFMs, minus known-failing/heavy/python/slow categories;
# drop `&TestCategory!=Slow` for the full run: ~10 min per TFM instead of ~2)
dotnet test test/OptunaSharp.Tests/OptunaSharp.Tests.csproj -c Release \
  --filter "TestCategory!=OpenBugs&TestCategory!=HighMemory&TestCategory!=RequiresPython&TestCategory!=Slow"

# Live-Optuna interop (needs a discoverable Python with `pip install optuna`)
dotnet test test/OptunaSharp.Tests.Interop/OptunaSharp.Tests.Interop.csproj -c Release

# Pack the OptunaSharp package
dotnet pack src/OptunaSharp.Core/OptunaSharp.Core.csproj -c Release -o packages

# Pack the WHOLE family exactly as CI and the release do (all 20 packages, one version), then gate it
bash .github/scripts/pack_all.sh artifacts/nuget 0.1.0-local.1
dotnet run .github/scripts/verify_strong_name.cs -- artifacts/nuget
dotnet run .github/scripts/verify_packages.cs -- artifacts/nuget --version 0.1.0-local.1 \
  --commit "$(git rev-parse HEAD)" --source-repository "$(git remote get-url origin)"
```

`global.json` pins the SDK to the GA 10.0 line (`rollForward: latestFeature`, `allowPrerelease: false`):
the newest installed 10.0.x builds everything, and a preview or .NET 11 SDK is never picked up.

### Test categories

Typed attributes in `test/OptunaSharp.Tests/TestCategory.cs` (so CI filter strings stay stable):

| Attribute | Meaning | CI |
|-----------|---------|----|
| `[OpenBugs]` | Known-failing reproduction / not-yet-ported behavior. Remove when it passes. | **excluded** |
| `[Misaligned]` | Intentional, documented difference from Optuna (the test passes). | runs |
| `[RequiresPython]` | Needs live Optuna via pythonnet; self-skips (Inconclusive) when absent. | interop job only |
| `[HighMemory]` | Resource-heavy. | **excluded** |
| `[WindowsOnly]` | Windows-only API. | excluded off-Windows |
| `[Slow]` | ~10 s+ each (REMBO/SAASBO claim runs; the BoTorch qEHVI family in the interop suite, `[TestCategory("Slow")]`). Runtime only — never to hide a failure. | **excluded** on push/PR; ubuntu `Test (slow)` step on `v*` tags + dispatch |

MSTest attribute namespaces are supplied globally via `GlobalUsings.cs` in each test project (`global
using Microsoft.VisualStudio.TestTools.UnitTesting;`), so test files don't repeat the `using`. The
`[RequiresPython]` *attribute* lives in `OptunaSharp.Tests`; the interop project (which doesn't
reference it) uses the string form `[TestCategory("RequiresPython")]`.

---

## Testing infrastructure

### Unit tests (`OptunaSharp.Tests`)

The suite now spans distributions, samplers, pruners, storages, importance, terminator, visualization
data, and the sklearn integration — all asserted against values produced by real Optuna. The original
`ScaffoldingTests.cs` remains as the assembly-identity smoke test: the assembly loads and is
strong-named with token `cc7b13ffcd2ddd51`, the version metadata is present, and the NumSharp
dependency resolves and runs (`np.arange(6).reshape(2,3)`). Keep adding tests derived from real Optuna
output as modules land.

### Live-Optuna interop (`OptunaSharp.Tests.Interop`)

Runs **real CPython + Optuna embedded in the test process** via pythonnet — the analogue of
`NumSharp.Tests.Interop` (which runs live NumPy). Optuna is asked to compute an answer *right now* and
OptunaSharp is asserted to match it.

- **`PythonSession`** — one embedded engine per process (`[AssemblyInitialize]`). Discovery honors
  `PYTHONNET_PYDLL` first, else probes `python3`/`python` on PATH for the shared library, then
  `import optuna` to fail fast. Exposes `EnsureOrInconclusive()`.
- A machine without a usable Python/Optuna reports **Inconclusive** (never red) — **unless**
  `OPTUNASHARP_PYTHONNET_REQUIRE_ENGINE` is truthy, which makes an absent engine a hard failure. CI
  sets it after installing Python, so a discovery break goes red instead of green-by-skipping.
- **GIL discipline:** anything touching a `PyObject` (including its `Dispose`) must run under
  `using (Py.GIL())`. Compute the C# side first, then open the GIL for the Python call + comparison.

Run it:

```bash
cd test/OptunaSharp.Tests.Interop
dotnet test -f net10.0                                   # needs `pip install optuna`
PYTHONNET_PYDLL=/path/to/libpython3.12.so dotnet test -f net10.0   # pin the interpreter
```

**This is the current differential-parity gate** until the committed-corpus oracle exists (see the
plan's §4 validation strategy: a future `OptunaSharp.Tests.Oracle` will replay committed JSONL corpora
with **no Python at test time**, bit-comparing where the path is deterministic — everything on
MT19937 — the same model as NumSharp's differential-fuzz pipeline, with a `MisalignedRegistry` ledger
of excused divergences).

### Benchmark (`OptunaSharp.Benchmark`)

A BenchmarkDotNet console (`BenchmarkSwitcher.FromAssembly`). Today it holds placeholder benchmarks
that exercise the OptunaSharp→NumSharp chain to prove the harness runs. As samplers/pruners/study land,
add `[Benchmark]` methods measuring them (and compare against live Optuna where useful). Run:

```bash
dotnet run --project test/OptunaSharp.Benchmark/OptunaSharp.Benchmark.csproj -c Release -f net8.0 -- --filter '*'
# quick pass: append --job short
```

---

## Packaging

- The release ships **20 packages**: Core (`OptunaSharp`) plus every `src/OptunaSharp.*` satellite
  (`OptunaSharp.Cli` is a `dotnet tool`). `.github/scripts/pack_all.sh` packs them all with one version;
  CI runs it on every push, PR and tag.
- `dotnet pack src/OptunaSharp.Core/OptunaSharp.Core.csproj -c Release -o packages` produces
  `OptunaSharp.<version>.nupkg` with `lib/net8.0/` + `lib/net10.0/` assemblies, each with its `.xml` docs
  and a separate portable `.pdb` (SourceLinked), plus `README.md` and the **`PackageIcon`**
  (`docs/website-src/images/optunasharp-icon.png`), and records the `NumSharp` dependency.
- `GeneratePackageOnBuild` is **not** set on Core, so a plain `dotnet build` doesn't auto-pack; CI
  packs explicitly. Package output path is `packages/` at the repo root (gitignored).
- Shared metadata (Authors/Company/Copyright/RepositoryUrl/`PackageLicenseExpression=MIT`) and the
  version (`VersionPrefix`/`AssemblyVersion`/`FileVersion`, default `0.1.0`) live in
  `Directory.Build.props`. CI stamps `-p:Version`/`-p:AssemblyVersion` from the pushed `v*` tag, and
  from `<VersionPrefix>-ci.<run number>` on every other run.
- `OptunaSharpVersion.Version` is read at run time from the assembly's informational version, minus its
  `+commit` suffix. It is not a `const`, so it always equals the shipped package version, and so does the
  `version_info.library_version` row the SQLite and EF Core storages write.
- **`RepositoryUrl` names the canonical public repo (`SciSharp/OptunaSharp`); SourceLink names the repo
  the build ran in** (git `origin`). The two agree only for a release built in the canonical repo, which
  is the one place a publish is allowed (`verify_packages.cs` reports `publishable`).

---

## Documentation (DocFX)

The site source is `docs/website-src/`, built with **DocFX** and deployed to GitHub Pages by
`.github/workflows/docs.yml`.

- `docfx.json` — `metadata` generates the API reference by compiling
  `src/OptunaSharp.Core/OptunaSharp.Core.csproj` into `api/*.yml` (gitignored; regenerated each
  build); `build` renders the markdown with the `default`+`modern` templates; OptunaSharp branding
  (`images/optunasharp-logo.png`, favicon `images/optunasharp-icon.png`), git-contribute links,
  sitemap, and the `ExtractSearchIndex` postprocessor.
- Conceptual pages: `docs/website-src/docs/{intro,getting-started,roadmap}.md` (+ `toc.yml`), landing
  `index.md`, `api/index.md`, `filterConfig.yml`.
- `scripts/generate-llms-txt.sh` emits an [llms.txt](https://llmstxt.org/) index of the built site.

Build it locally. The metadata step compiles Core, which restores NumSharp + NumSharp.Interop.OpenBLAS
from NuGet — no submodule init or weaver setup needed.

```bash
docfx docs/website-src/docfx.json --serve          # http://localhost:8080
```

Output goes to `docs/website/` (gitignored). Verified to build **0 warnings / 0 errors** with DocFX
2.78.4.

---

## CI/CD

The main workflows in `.github/workflows/` (modeled on NumSharp's live pipeline). Env `DOTNET_NOWARN`
suppresses noisy XML-doc warnings.

Rules every workflow follows:
- **Runner images are pinned:** `ubuntu-24.04`, `windows-2025`, `macos-26`, never `-latest`. The
  bit-exact suites depend on the image's libm and CPU dispatch, and a `-latest` label migrates on
  GitHub's schedule (ubuntu-latest began moving to 26.04 on 2026-10-19). Bump the labels deliberately,
  in every workflow at once.
- **GA SDKs only:** `actions/setup-dotnet` installs `8.0.x` + `10.0.x`, and `global.json` selects the 10.0
  GA line. Never add `dotnet-quality: 'preview'`: it used to install the 8.0/10.0 **RC** SDKs.
- **Node-24 action majors:** checkout@v7, setup-dotnet@v6, cache@v6, upload-artifact@v7,
  download-artifact@v8, setup-python@v7, Pages v5/v6, action-gh-release@v3. Dependabot bumps them as
  one grouped PR.
- **Least privilege:** `contents: read` at the top; a job that writes declares only that write.

### `build-and-release.yml`
- **`test`** — matrix `windows/ubuntu/macos`, .NET 8 + 10; builds `OptunaSharp.Tests` and runs both
  TFMs with `TestCategory!=OpenBugs&!=HighMemory&!=RequiresPython&!=Slow` (+ `!=WindowsOnly` off-Windows);
  a `Test (slow)` step runs the `Slow` category on Linux for `v*` tags and manual dispatch (dispatch input
  `run_slow`, default on; off skips ~2 h of BoTorch parity when re-checking an OS leg). The `setup`
  job decides the matrix per trigger: push = ubuntu+windows × both TFMs, PR = net10.0 only,
  tags/dispatch = +macOS.
- **`satellite-tests`** — ubuntu + windows on every trigger, never macOS: the arm64 runner has no ML.NET
  LightGBM native and cannot host the Linux service containers. It runs every other `test/OptunaSharp.Tests.*` suite (Storage,
  Artifacts, Cli, Hosting, Importance.*, Integration.*, Benchmark.VolleyEval); Interop and
  PackageParity have their own jobs. On Linux it starts Redis and the S3Mock/Azurite/fake-gcs emulators
  with docker compose, Testcontainers starts PostgreSQL and MySQL, and the `*_REQUIRED` switches make a
  missing service red. Elsewhere those cases self-skip. The job installs the vendored Optuna for the
  live `optuna` CLI parity cases and initializes `refs/optuna`, whose upgrade-test databases the Storage
  and Cli suites read.
- **`interop-test`** — matrix; installs Python 3.12 + Optuna, resolves `PYTHONNET_PYDLL`, builds and
  tests `OptunaSharp.Tests.Interop` with `OPTUNASHARP_PYTHONNET_REQUIRE_ENGINE=1`.
  - Like BoTorch, the OS-independent parity backends are gated on Linux only: one OS suffices when the
    SAME Python library runs on both sides. The pinned backends:
    - `cma==4.4.4` (pymoo's `cma==3.2.2` breaks under NumPy 2);
    - `shap==0.52.0`, `lightgbm`, `grpcio` + `protobuf`;
    - `psycopg2-binary` / `PyMySQL[rsa]` for the EF Core cross-language suites (Testcontainers);
    - `redis`, plus a compose Redis server;
    - the vendored `optunahub` + `refs/optunahub-registry`;
    - the JANUS reference: `hongyuanyu/JANUS` @ `53e372f`, via `JANUS_REF_PATH`.
  - On every OS, `sqlalchemy==2.0.45`: the storage-URL port reproduces that release's driver registry
    (a bare `postgresql://` means psycopg2; SQLAlchemy 2.1 switched it to psycopg 3).
  - Before this, ~60 cases per TFM were Inconclusive in CI.
- **`package`** — runs on **every push, PR and tag** (it replaced `verify-signing`, which packed a
  throwaway Core only). It checks that `Open.snk` is intact, then `pack_all.sh` packs all 20 packages at
  the tag version or `<VersionPrefix>-ci.<run>`. Two gates follow: `verify_strong_name.cs` (every
  assembly really signed) and `verify_packages.cs`. The second requires one package per packable
  project, every version stamped (nuspec, family dependencies, assembly and informational versions),
  the license, icon and readme present, no stale "early scaffolding" text, and PDBs SourceLinked to the
  commit with deterministic `/_/` paths. It outputs `publishable`. The job uploads the `nuget-packages`
  artifact.
- **Release pipeline** (on `v*` tags): `validate-release` (needs every test job and `package`; the tag
  must be SemVer, on `master`, and equal to the packed version) → `create-release` (the only
  `contents: write` job; GitHub Release with every package plus checksums) and `publish-nuget`.
  `publish-nuget` runs in the `nuget` environment, where an approval gate can be added. It soft-skips
  without `secrets.NUGETAPIKEY` and refuses a non-publishable feed (a release from the private mirror).
  It pushes Core first.

### `docs.yml`
DocFX (pinned 2.78.4) build → GitHub Pages. The `build` job runs for PRs too, as a gate (metadata must
compile). A separate `deploy` job holds the `github-pages` environment and the Pages/OIDC permissions,
and runs only off-PR with `vars.ENABLE_PAGES == 'true'`: Pages restricts that environment to the default
branch, so a PR job naming it would be rejected. Triggers on `src/**`, `docs/website-src/**`,
`Directory.Build.*`, `global.json`, and the workflow file.

### `package-parity.yml`
Build → package (packs every `src/*`, strong-name gate) → deploy → test, each isolated: the test
container sees only `test/OptunaSharp.Tests.PackageParity` plus the feed, and replays real Optuna,
xgboost, lightgbm and sklearn oracles against the PACKAGED OptunaSharp. Runs on changes to `src/**`,
the test project, the build props, `global.json`, `NuGet.Config`, `Open.snk` or the strong-name script.

### `documentation.yml`
The in-code documentation gate: `check_documentation.cs` over `src/`, then every src project built with
the doc-comment warnings as errors, on both TFMs.

### `benchmark.yml`
`build-smoke` compiles the harness on every relevant push/PR; a full **short** run is manual-only
(`workflow_dispatch`) — a real perf suite isn't meaningful until algorithms exist.

### `examples.yml`
Compile gate for `examples/`: builds every file-based app there (`dotnet build <file>.cs`, which also
builds its `#:project` references), including `examples/optunahub/`. It runs on ubuntu for pushes and PRs
that touch `examples/**`, `src/**`, `Directory.Build.*`, `NuGet.Config` or `global.json`. It keeps going after a
failure and reports every example that doesn't compile. Examples are built, not run, because several
need ffmpeg, Redis or Python.

### Other
`.github/dependabot.yml` (weekly nuget + one grouped github-actions PR), issue forms
(`ISSUE_TEMPLATE/{bug,feature,config}.yml` — the feature form centers "Optuna behavior as source of
truth"), `PULL_REQUEST_TEMPLATE.md`, `FUNDING.yml`.

---

## Scripting with `dotnet run` (.NET 10 file-based apps)

Internal members are exposed to a `dotnet run` script that overrides its assembly name to the declared
friend `OptunaSharp.DotNetRunScript`. A script **under the repo root** inherits `Directory.Build.props`
and is signed automatically; one run from elsewhere must pass the signing properties itself:

```csharp
#:project src/OptunaSharp.Core/OptunaSharp.Core.csproj
#:property AssemblyName=OptunaSharp.DotNetRunScript
#:property PublishAot=false
#:property SignAssembly=true
#:property AssemblyOriginatorKeyFile=K:/source/OptunaSharp/Open.snk
```

---

## Deferred / not-ported infrastructure

- **Committed differential-parity CORPUS oracle** (NumSharp's Fuzz Soak analog) — deferred until there
  is an algorithm surface to snapshot. The **live-Optuna interop suite is the differential gate** in
  the meantime. The plan's §4 specifies the future `OptunaSharp.Tests.Oracle` +
  `test/oracle/gen_optuna_oracle.py`.
- **NumSharp's `tools/` weaver + leak analyzer** are NOT ported — they are specific to NumSharp's
  unmanaged `NDArray` disposal model, which OptunaSharp has no analog for.
- **Satellite packages** — no longer deferred: the 19 `src/OptunaSharp.*` satellites exist, follow the
  `NumSharp.Interop.OpenBLAS` optional-package pattern so Core stays pure-managed, and ship with Core in
  every release (see Packaging).

---

## Adding a ported feature — Definition of Done

Follow the plan's phase order (`docs/PORTING_PLAN.md`). For each unit of work:

1. **Read the Optuna source** at `K:/source/optuna` and **run it** to capture exact behavior.
2. **Implement** matching the Python's signatures (names/order/defaults), behavior, edge cases, and
   verbatim error messages/types.
3. Respect the ground rules: `LazyRandomState` (never `System.Random`), `CategoricalChoice` Python
   equality, Core-stays-pure-managed (foreign deps → satellite package).
4. **Test from real Optuna** — unit tests and (where practical) a live interop parity test.
   Reproducibility under a fixed seed must be **bit-identical** to Python (that's why the RNG rule
   exists).
5. Keep `dotnet build` and the unit-test filter green; document any intentional divergence with
   `[Misaligned]` (and, later, the `MisalignedRegistry`).

---

## GitHub (issues, CI inspection) via `gh`

In this environment the `gh` CLI is authenticated through a keyring; call it with the ambient tokens
unset:

```bash
env -u GH_TOKEN -u GITHUB_TOKEN gh <args>
# e.g. inspect CI once a remote exists:
env -u GH_TOKEN -u GITHUB_TOKEN gh run list --repo SciSharp/OptunaSharp --limit 10
```

File issues on `SciSharp/OptunaSharp` following the bug/feature templates (feature issues must include
the equivalent Optuna Python snippet + output as the source of truth).

---

## Conventions

- **Branch from `master`**; releases are tagged `v*` from `master`.
- **Conventional-commit prefixes** as in the history (`feat`, `fix`, `chore`, `docs`, with optional
  scope: `chore(infra):`, `docs(readme):`, `feat(branding):`). Extensive commit bodies for future
  reference.
- Follow `.editorconfig` (the SciSharp house style); match the naming/comment density of surrounding
  code. Name public types after their Optuna counterparts so the two cross-reference.
- **Remotes:** the canonical public repository is `github.com/SciSharp/OptunaSharp` (what
  `RepositoryUrl` names, and the only place a release may publish from); CI also runs on a private
  mirror, from which `publish-nuget` refuses to push (`verify_packages.cs` reports it non-publishable).

---

## Paper References

> The three source-paper Markdown files below are embedded whole. They are the `paper-gateway` extractions (converter 1.1.0 / PyMuPDF + OCR) of the samplers' source PDFs, audited 2026-09-13 (`docs/papers/extracted/EXTRACTION-AUDIT.md`). Prose, algorithm pseudocode and tables are faithful; **display equations degrade** in the linear text (worst in AGE-MOEA-2 — read the linked `pages/*.png` for any formula). Image links resolve to the local, git-ignored `docs/papers/paper-gateway/library/`. Each paper keeps its own `#` title as its divider.

<!-- ============================================================================== -->
<!-- PAPER REFERENCE: JANUS: Online Jacobian-Aligned Infill for Black-Box Optimization  (source: docs/papers/extracted/JANUS.md) -->
<!-- ============================================================================== -->

# JANUS: Online Jacobian-Aligned Infill for Black-Box Optimization

Source: [JANUS-Online-Jacobian-Aligned-Infill-for-Black-Box-Optimization-72e55dcbd3.pdf](file:///K:/source/OptunaSharp/docs/papers/paper-gateway/downloads/JANUS-Online-Jacobian-Aligned-Infill-for-Black-Box-Optimization-72e55dcbd3.pdf)

SHA-256: `5204e8c7ed00c503f00942ce662291a8666a372e29d4a5df3282c681b2fc4440`

> Extraction is heuristic. Full-page images preserve drawings, graphs and visual layout; searchable text is extracted separately.

<a id="page-1"></a>

<!-- Page 1 -->

## JANUS: ONLINE JACOBIAN-ALIGNED INFILL FOR BLACK-BOX OPTIMIZATION

Hongyuan Yu1, Pufan Xu2∗, Jiaojiao Yi1, Yiding Tian1, Mingrui Sun1, Jiayuan Lu1

Changyuan Wen1

1Multimedia Department, Xiaomi Inc. 2School of Integrated Circuits, Tsinghua University
\{yuhongyuan, yijiaojiao, tianyiding,
sunmingrui1, lujiayuan, wenchangyuan\}@xiaomi.com
xpf22@mails.tsinghua.edu.cn

## ABSTRACT

Population optimizers such as CMA-ES, DE, and multi-objective evolutionary
algorithms drive search mainly through selection signals that are scalar or rank
based: such a signal indicates that one candidate outperforms another, but not
the local direction responsible for the improvement. JANUS (Jacobian-Aligned
Newton-Unified Search) is a plug-and-play infill module that extracts this missing
local geometric signal without replacing the host optimizer. It estimates a local
Jacobian from the recent evaluation trace; the same Jacobian yields both a damped
Gauss–Newton exploitation candidate and a trace-preserving exploration metric,
reserving a fraction of the host’s per-generation candidate slots for geometry-
guided infill rather than spending evaluations on top of the host’s budget. Unlike
MetaBBO methods, JANUS needs no offline training or task distribution, estimating
this geometry on the fly from the current run alone, while the host keeps full
control of selection, survival, covariance adaptation, and step-size control. Under
same-protocol comparisons, JANUS improves the CMA-ES host on 11–15/16
BBOB functions across d ∈\{30, 100, 500\}. It also attains the best mean error on
13 of the 16 functions at d=500 in the complete NN-BBO/MetaBBO baseline
comparison, with no training cost, and yields a 936× geometric-mean improvement
over the host on a d=1000 BBOB subset. On structured and multi-objective tasks,
JANUS gives the best mean cost on 1135-dimensional UAV path planning (−12.8%
vs. the strongest baseline), and it improves SMS-EMOA/AGE-MOEA2 hosts on
12/38 multi-objective tasks with zero significant regressions. Code is available at

https://github.com/hongyuanyu/JANUS.

## 1 INTRODUCTION

Black-box optimization (BBO) appears in hyperparameter tuning, simulation calibration, controller
design, and scientific workflows where gradients are unavailable or unreliable (Bergstra &amp; Bengio,

## arXiv:2608.22862v1 \[cs.NE\] 24 Aug 2026

2012; Hansen, 2016; Ma et al., 2023). Population optimizers are successful partly because they avoid
brittle modeling assumptions: they rank candidates by scalar fitness and adapt from selection statistics.
Yet scalar rankings are a deliberately thin interface. A ranking indicates that one sample is better, but
it does not reveal the local direction responsible for the improvement, nor how the next generation
of samples should be shaped. We test this gap across scalar BBOB, structured UAV planning, and
multi-objective optimization (Fig. 1).

Population hosts leave local geometry unused. We identify a rank-geometry gap between two
families of derivative-free methods. On one side, population hosts (e.g., CMA-ES, DE) adapt search
distributions using selection signals that are scalar or rank based. They are sample-efficient and robust,
but they discard the local residual geometry that classical trust-region methods exploit. On the other
side, derivative-free trust-region methods (NEWUOA, BOBYQA, DFO-LS) build local quadratic or
least-squares models from the trace. These methods replace the host optimizer entirely, however, and

∗Corresponding author.

1

<details>
<summary>Page 1: complete visual reference</summary>

![Full source page 1](K:/source/OptunaSharp/docs/papers/paper-gateway/library/5204e8c7ed00c503/pages/page-0001.png)

</details>

> Quality note: Rotated or vertical text retains geometry and requires visual order review.
> Quality note: Mathematical notation or superscripts require visual review; linear text does not establish fraction, exponent or equation semantics.

<a id="page-2"></a>

<!-- Page 2 -->

A. BBOB wins vs CMA-ES B. UAV (d=1135) C. MOO leaders

### 20.0 Low

16 15 15

### 17.5 J-SMS 2.0

14 15/28

15.0

12 11 +12.8%

### 12.5 SMS 3.1

10

10.0

## 8 Many

Wins / 16

6 7.5 J-AGE2 2.0

4 5.0

2 2.5 AGE2 3.6

0 0.0
30D 100D 500D Rank 1 Cost 0 1 2 3 4
terrains gain (%) Avg. rank

![Figure 1: JANUS at a glance. Headline BBOB, UAV, and MOO results.](K:/source/OptunaSharp/docs/papers/paper-gateway/library/5204e8c7ed00c503/figures/p0002-figure-01.png)

Figure 1: JANUS at a glance. Headline BBOB, UAV, and MOO results.

spend a large fraction of the budget on model maintenance. Restricted-covariance variants such as
LM-CMA (Loshchilov, 2014) and VkD-CMA (Akimoto &amp; Hansen, 2016) partially address scalability
by learning a restricted covariance basis, but they still learn this basis from host selection statistics
rather than from the local residual Jacobian itself. We show (Prop. 1) that a covariance basis fixed
independently of the local residual Jacobian J cannot, in general, align its exploration metric with
the inverse Hessian T = (J⊤J + λI)−1 of the local residual model.

Closing the gap with one Jacobian, two uses. JANUS turns the recent search trace into residual
coordinates and estimates a local Jacobian J. If evaluations expose components, for example

Pm
f(x) = 1m i=1 ri(x), (1)

p
we use normalized residual coordinates such as si(x) = ri(x); if only scalar values are observed,
JANUS instead derives residual coordinates from local linear or quadratic fits on the recent window.
Explicit components are useful but not required: what matters is a local residual map s(x) whose
Jacobian summarizes how recent changes in x affect the local objective model.

The central observation is that the same J answers two questions usually handled separately: no
prior method derives both a bounded exploitation candidate and a trace-preserving exploration metric
from the same online computation. Earlier per-component arg min assembly can be viewed as a
block-diagonal special case restricted to exploitation: it assembles the best fragment from each
component, but discards the coupling across coordinates and says nothing about population spread.
JANUS keeps the full J⊤J and reuses it for both exploitation and exploration.

JANUS is implemented as infill, not a replacement optimizer: it asks the host for native candidates,
adds a few bounded Jacobian-aligned candidates, evaluates the union, and lets the host perform
its usual selection, survival, covariance, and step-size updates. Because JANUS only contributes
candidates, it can augment many population-based search algorithms without meta-training or task-
specific tuning, while preserving the host optimizer’s safeguards.

On-instance, not meta-learning. JANUS estimates the residual Jacobian online from the current
black-box problem and discards it after the run: there is no training distribution, no learned policy,
and no parameter transfer between problems. This trade-off is deliberate. MetaBBO methods (Ma
et al., 2023; Lange et al., 2023; Chen et al., 2024; Li et al., 2025) can outperform on-instance methods
when the test task matches the training distribution, but they introduce a generalization axis and
require offline meta-training; JANUS occupies the complementary regime, working on any single
task without prior training. Whether JANUS’s local-geometry signal could be augmented with a
learned prior over residual structures is left for future work. We validate this across three problem
classes, four host optimizers, and one industrial deployment. Our contributions are:

1\. We show (Prop. 1) that restricted-covariance variants (LM-CMA, VkD-CMA) cannot, in general,
align a host-fixed covariance basis with the local curvature induced by a residual Jacobian.
2\. We introduce online construction of residual coordinates to extract reusable local geometry
from black-box search traces, using either observed components or coordinates from a local
linear or quadratic fit; from one estimated Jacobian, we derive two bounded infill mechanisms: a
damped Gauss–Newton exploitation candidate and a trace-preserving inverse Hessian exploration
distribution.

2

<details>
<summary>Page 2: complete visual reference</summary>

![Full source page 2](K:/source/OptunaSharp/docs/papers/paper-gateway/library/5204e8c7ed00c503/pages/page-0002.png)

</details>

> Quality note: Rotated or vertical text retains geometry and requires visual order review.
> Quality note: Mathematical notation or superscripts require visual review; linear text does not establish fraction, exponent or equation semantics.

<a id="page-3"></a>

<!-- Page 3 -->

3\. Under same-protocol comparisons, JANUS variants improve over the CMA-ES host on 11–15/16
BBOB functions across d ∈\{30, 100, 500\} (Table 11, Table 3) and achieve the best mean cost
on 28-terrain UAV path planning (−12.8% vs the strongest baseline). The same mechanism
improves both SMS-EMOA and AGE-MOEA2 hosts on 12/38 multi-objective tasks with zero
significant regressions, and transfers to a DE host (86–96% win rate).

## 2 RELATED WORK

Evolution strategies and covariance adaptation. CMA-ES adapts a Gaussian search distribution
from selected samples and remains one of the strongest general-purpose black-box optimizers (Hansen,
2016). Scalable variants such as LM-CMA (Loshchilov, 2014) and VkD-CMA (Akimoto &amp; Hansen,
2016\) restrict the covariance basis to reduce cost in high dimension, but still learn geometry from
selection statistics. FOCAL (Shir et al., 2010) modifies the adaptation rule itself to recover inverse-
Hessian structure near the optimum. JANUS is complementary: it estimates residual geometry
externally and injects candidates without changing the host update rule.

Model-based derivative-free optimization. Classical DFO methods build local interpolation,
trust-region, or least-squares models from black-box evaluations, including NEWUOA, BOBYQA,
COBYLA, and DFO-LS (Powell, 2006; Powell et al., 2009; Cartis et al., 2019). They are most
effective when model maintenance is affordable and the local model can serve as the entire optimizer.
JANUS does not maintain a trust region, choose incumbents, or replace the search distribution; it
imports only the residual-model signal into a population loop, so the host retains global search
pressure while JANUS spends a small infill budget on geometry-aware proposals.

Natural-gradient and learned optimizer views. Natural evolution strategies and information-
geometric optimization interpret distribution updates as gradient steps on a statistical manifold (Wier-
stra et al., 2014; Ollivier et al., 2017); CMA-ES itself can be viewed through approximate natural-
gradient adaptation (Akimoto et al., 2010). JANUS shares the idea that proposals should encode
useful geometry, but its geometry source is a residual normal matrix J⊤J rather than an expected-
fitness gradient estimated from rankings. MetaBBO methods learn optimizers or operators across task
distributions (Ma et al., 2023; Lange et al., 2023; Chen et al., 2024; Li et al., 2025); JANUS instead
estimates geometry on the current instance only, avoiding offline meta-training and task-distribution
shift.

Multi-objective optimization and infill. Multi-objective evolutionary algorithms maintain conver-
gence and diversity through nondominated sorting, hypervolume selection, decomposition, reference
vectors, or geometry-aware survival (Deb et al., 2002; Beume et al., 2007; Zhang &amp; Li, 2007; Deb &amp;
Jain, 2014; Cheng et al., 2016); expensive MOO also studies infill criteria such as expected hyper-
volume improvement (Emmerich et al., 2006). JANUS does not replace survival or reference-vector
logic. Attached to SMS-EMOA or AGE-MOEA2 (Panichella, 2022)-style hosts, it only changes how
a small fraction of offspring are proposed, using the objective vector itself as residual coordinates.
Surrogate-assisted evolutionary algorithms (Jin, 2011) instead build global surrogates (GP, RBF) to
pre-screen candidates; JANUS uses only a local fit on the recent window, scaling to d&gt;1000.

## 3 METHOD

Table 1 gives the symbols needed to read this section; App. A.2 keeps the full notation and variant
list.

### 3.1 HOST INTERFACE

JANUS assumes only a population-style ask/evaluate/tell loop. At generation g, a host such as CMA-
ES samples candidates (e.g., xk ∼¯x + σN(0, C)), evaluates them, and updates its internal state from
selected or surviving samples. JANUS does not change this loop: it reads a recent evaluation window,
proposes a small number of additional candidates, and returns all evaluated points to the host, so
selection pressure, survival logic, covariance adaptation, and step-size control remain host-defined. A
host baseline and its JANUS-augmented counterpart always share the same total evaluation budget;

3

<details>
<summary>Page 3: complete visual reference</summary>

![Full source page 3](K:/source/OptunaSharp/docs/papers/paper-gateway/library/5204e8c7ed00c503/pages/page-0003.png)

</details>

> Quality note: Mathematical notation or superscripts require visual review; linear text does not establish fraction, exponent or equation semantics.

<a id="page-4"></a>

<!-- Page 4 -->

Table 1: Core notation used in the method.

Symbol Meaning

x, f(x) Candidate and evaluated black-box value.
W Recent evaluated window used for local fitting.
r(x) Observed components/objective vector when available.
s(x) Residual coordinates, observed or scalar-derived.
J Online residual Jacobian fitted on W.
G = J⊤J Gauss–Newton Hessian approximation; metric for GN exploitation
and CMIX exploration.
λ, ρ Damping coefficient and CMIX candidate fraction.
GN / CMIX Bounded Newton candidate / inverse Hessian mixing.
JANUS-GN / JANUS Exploitation-only variant / GN plus CMIX variant.

JANUS reallocates part of each generation’s candidate slots to infill proposals, rather than drawing
evaluations on top of the budget.

### 3.2 ONLINE RESIDUAL COORDINATES

The only object JANUS needs is a local residual map s(z) ∈Rm on a recent window W =
\{(xi, fi)\}ni=1. We first express candidates in host-normalized coordinates, zi = D−1(xi −µ), using
the current local scale. JANUS then constructs s in one of three ways:

• Observed components: if evaluations expose nonnegative per-scenario, per-constraint, or per-p
objective terms, use normalized residual coordinates such as si(x) = ri(x); otherwise use the
shifted or standardized component value.
• Linear scalar mode: if only scalar values are observed, fit ˆf(z) = a⊤z + b on W and use
s1(z) = a⊤z after normalization. This gives a stable slope estimate for the GN candidate.
• Quadratic scalar mode: fit ˆf(z) = z⊤Az + b⊤z + c, decompose A = Pp i λiuiu⊤i , and use
si = \|λi\| u⊤i z as curvature-aligned coordinates.

The scalar modes are not fallbacks; they handle the common case where the benchmark exposes
only scalar values. In these modes s is a locally fitted pseudo-residual coordinate system, not a
Gauss–Newton decomposition of the objective: the linear mode drops the intercept b (encoding onlyp
the local slope direction), and the quadratic mode uses \|λi\|, discarding the curvature sign, so ∥s∥2

does not reproduce ˆf. It is a heuristic slope- and curvature-aligned frame for generating bounded
candidates, not an exact residual factorization of f. Linear mode is conservative and mainly supports
exploitation; quadratic mode supplies multiple curvature directions for CMIX but can be noisy on
strongly multimodal landscapes.

Given residual-coordinate values on W, JANUS fits s(xi) ≈b + Jzi by ridge regression and forms
G = J⊤J, the Gauss–Newton approximation to the Hessian of the local residual model ∥s(x)∥2.
The regression is intentionally local: old samples are discarded, inputs are host-normalized, and ridge
regularization suppresses unsupported directions. Unlike global surrogate optimization, JANUS does
not require the model to predict the objective across the whole box — only a local coordinate system
that produces a bounded candidate and a temporary proposal metric. Survival remains the host’s
decision.

### 3.3 EXPLOITATION: DAMPED GAUSS–NEWTON INFILL

Let xb be the best point in the current window and sb = s(xb). JANUS forms the damped Gauss–
Newton step
∆GN = −(G + λI)−1J⊤sb, (2)
and proposes xb + η∆GN. To avoid destabilizing CMA-style step-size control, the proposal is
bounded-injected following the clipping rule of Hansen (2011). In whitened coordinates, its step is√
clipped to ∥ye∥≤κ d. Thus the injected candidate has the scale of a normal host sample, while
its direction comes from the residual Jacobian rather than from historical-best reuse or random
perturbation.

4

<details>
<summary>Page 4: complete visual reference</summary>

![Full source page 4](K:/source/OptunaSharp/docs/papers/paper-gateway/library/5204e8c7ed00c503/pages/page-0004.png)

</details>

> Quality note: Mathematical notation or superscripts require visual review; linear text does not establish fraction, exponent or equation semantics.

<a id="page-5"></a>

<!-- Page 5 -->

Algorithm 1 JANUS: online metric-guided infill

Input: Host A, window n, infill ratio ρ, damping λ

1: while budget remains do
2: Ask A for native candidates; reserve ρ slots for infill
3: Fit J from window W; form G = J⊤J
4: GN candidate: −(G + λI)−1J⊤sb; bounded inject
5: CMIX candidates: sample from trace-normalized (G + λI)−1

6: Evaluate all candidates; host performs selection/survival
7: end while

### 3.4 EXPLORATION: INVERSE HESSIAN CMIX

For exploration, JANUS inverts the same Gauss–Newton curvature G to form a regularized inverse-
Hessian covariance,

T = (G + λmixI)−1, C ←(1 −w)C + wtr(C) (3)
tr(T) T.

The trace factor preserves the total sampling energy of C and changes only its shape: directions that
are locally flat under G receive more variance, while sharp directions receive less. Setting w = 0
recovers JANUS-GN, the conservative exploitation-only fallback used when the curvature estimate is
unstable (Sec. 5.4).

### 3.5 DIMENSION-ADAPTIVE MIXING WEIGHT

w = wbase · min(m, \|W\|), w ∈\[wmin, wmax\]. (4)
d

The numerator measures how much independent residual information is available, while the de-
nominator reflects the ambient dimension over which that information must be spread. The bounds
\[wmin, wmax\] prevent the metric from either disappearing or dominating the host covariance. This
rule is fixed across all experiments.

### 3.6 ALTERNATIVE DESIGNS AND SCOPE

Standalone damped GN or DFO-LS spends the whole budget maintaining a local model, whereas
JANUS contributes only bounded candidates and leaves selection and scale control to the host; the
empirical gap is large, with standalone local models much weaker than host+JANUS under the same
high-dimensional BBOB budgets (App. A.3). Restricted-covariance hosts such as LM-CMA and
VkD-CMA are complementary rather than substitutes: their covariance basis is learned from host
selection statistics, while JANUS derives directions from the residual Jacobian (Prop. 1; App. A.3,
Table 9). Learning the mixing weight would instead add an offline task distribution and a new
generalization axis; the fixed rule in Eq. 4, by contrast, is shared across BBOB, UAV, and MOO.

## 4 THEORETICAL PROPERTIES

Proposition 1 (A fixed rank-k basis cannot express arbitrary residual-aligned metrics). Let Chost =
BMB⊤be a purely rank-k host covariance whose basis B ∈Rd×k, k &lt; d, has orthonormal
columns and is fixed by host adaptation statistics independently of the local residual Jacobian
J ∈Rm×d. Let T = (J⊤J + λI)−1 be the inverse Hessian metric induced by J (λ &gt; 0), and let
P⊥= I −BB⊤be the orthogonal projector onto span(B)⊥. Then no restricted-covariance update
of the form C′ = (1 −α)Chost + αBM ′B⊤can equal T; quantitatively,

√

d −k

∥C′ −T∥F ≥∥P⊥TP⊥∥F ≥

σmax(J)2 + λ &gt; 0,

where σmax(J) is the largest singular value of J. The bound is independent of M, M ′, α and of the
host block, so no choice of restricted-covariance parameters removes it. The gap ∥P⊥TP⊥∥F is
exactly the residual-aligned curvature that lives in span(B)⊥, which the restricted family sets to zero
regardless of its parameters.

5

<details>
<summary>Page 5: complete visual reference</summary>

![Full source page 5](K:/source/OptunaSharp/docs/papers/paper-gateway/library/5204e8c7ed00c503/pages/page-0005.png)

</details>

> Quality note: Mathematical notation or superscripts require visual review; linear text does not establish fraction, exponent or equation semantics.

<a id="page-6"></a>

<!-- Page 6 -->

Proposition 2 (Damped GN descends on the fitted residual model). Let ˆs(xb + ∆) = sb + J∆and
ˆf(∆) = ∥sb + J∆∥22. For λ &gt; 0, ∆GN = −(J⊤J + λI)−1J⊤sb satisfies

∇ˆf(0)⊤∆GN = −2s⊤b J(J⊤J + λI)−1J⊤sb ≤0,
strictly whenever J⊤sb ̸= 0.
Proposition 3 (Bounded injection and trace-preserving CMIX). If an injected point is clipped in√ √
host-whitened coordinates to ∥ye∥≤κ d, then ∥xe −¯x∥≤κσλmax(C1/2) d. Moreover, for
T ≻0 and C′ = (1−w)C + w tr(C)T/ tr(T), tr(C′) = tr(C).

These statements explain why JANUS is infill rather than a replacement optimizer. Prop. 1’s scope
is restricted-basis hosts (k &lt; d), true of LM-CMA/VkD-CMA by construction; for a full-rank
host such as vanilla CMA-ES, span(B) = Rd and the proposition makes no structural claim, so
JANUS’s advantage there is empirical rather than structural — learning a general full covariance
from rank-based updates involves O(d2) covariance parameters and can therefore be sample- and
computation-intensive in high dimensions, while JANUS forms the residual-aligned metric from
the fitted J in one step (App. C.2 separates the structural regime, illustrated by Table 9, from this
convergence-speed regime). Prop. 2 is local and model-relative: the injected point descends on the
fitted residual chart, but the chart may not describe disconnected basins. The failure mode is bounded
rather than open-ended: under the Prop. 3 clip, the proposal displacement of a misspecified chart stays
on the host sampling scale, although its objective value is not guaranteed to improve. Prop. 3 gives this
safety interface: injected points stay on the host sampling scale and CMIX reshapes direction without
changing covariance trace. A blockwise assembly rule is recovered as a degenerate zero-coupling GN
case. Full proofs are in App. C.2.

Local versus global exploration. The inverse Hessian metric can reallocate variance among di-
rections represented in the recent trace, but cannot identify disconnected basins outside the current
sampling region; such exploration still requires host diversity, restarts, or another global mecha-
nism. This predicts why CMIX is strongest when the trace contains coherent local curvature, while
JANUS-GN is safer when basin discovery or noisy pseudo-residuals dominate.

## 5 EXPERIMENTS

All experiments report final objective values or hypervolume under fixed budgets and indepen-
dent seeds. Appendix A.1 gives the full search spaces, evaluation budgets, run counts, and metric
conventions for BBOB, UAV, and MOO.

### 5.1 SCALAR BBOB: GEOMETRY FROM SCALAR EVALUATIONS (d ∈\{30, 100, 500\})

BBOB represents the most challenging setting for our claim: the benchmark exposes only scalar
values, so any useful geometry must be recovered from the recent trace rather than observed directly.
We use a MetaBox/COCO-compatible BBOB protocol (Ma et al., 2023): search domain \[−5, 5\]d,
budget 20,000 function evaluations, 30 independent runs, on the 16 held-out BBOB functions. The
CMA-ES host comparison in Fig. 2 is the controlled headline: same functions, seeds, population size,
initialization, box constraints, and budget, with and without JANUS infill. JANUS-GN injects only
the Gauss–Newton candidate, while JANUS additionally enables CMIX.

Host-controlled comparison (headline). Fig. 2 compares JANUS variants against their CMA-ES
host under the identical \[−5, 5\]d protocol. With the BBOB-specific fuse cadence (fuse every=50,
tuned once for d ∈\{30, 100\}; App. A shows all hyperparameters stay within 1.5× of default under
a one-at-a-time sweep), JANUS wins on 11/16 functions at d=30 (11W/1T/4L), 15/16 at d=100
(15W/1T/0L), and 15/16 at d=500 (geometric-mean ratio 0.11), with 0 significant regressions at
d ∈\{100, 500\} (Table 11). The few host-controlled losses concentrate on functions such as f22
and f23, where progress depends on global basin discovery beyond the current residual model.
Bonferroni-corrected paired Wilcoxon tests (App. B) confirm the effect grows with dimension and is
not attributable to chance. JANUS-GN is the recommended default when CMIX’s inverse Hessian
estimate is noisy.

Comparison with model-based DFO and host-agnostic validation. Against model-based
derivative-free optimizers (NEWUOA, BOBYQA, COBYLA (Powell, 2006; Powell et al., 2009),

6

<details>
<summary>Page 6: complete visual reference</summary>

![Full source page 6](K:/source/OptunaSharp/docs/papers/paper-gateway/library/5204e8c7ed00c503/pages/page-0006.png)

</details>

> Quality note: Mathematical notation or superscripts require visual review; linear text does not establish fraction, exponent or equation semantics.

<a id="page-7"></a>

<!-- Page 7 -->

![Vector artwork, page 7, region 1](K:/source/OptunaSharp/docs/papers/paper-gateway/library/5204e8c7ed00c503/figures/p0007-figure-01.png)

Table 2: UAV main comparison at 2,500 FEs (d=1135). Lower is better; full 12-method, 28-terrain
table and path visualizations are in App. B.3.

![Mixed artwork, page 7, region 2](K:/source/OptunaSharp/docs/papers/paper-gateway/library/5204e8c7ed00c503/figures/p0007-figure-02.png)

Method Mean Median Rank-1

JANUS 9,600 6,644 15
DE 11,006 9,261 1
JDE21 11,017 8,579 1
RLDEAFL 11,411 8,197 0
CMA-ES 14,486 10,908 0

JANUS vs CMA-ES host: per-function log10  ratio across dimensions worse

30D 0.94 0.1 0.69 0.95 0.90 0.0 5e-10 0.1 2.24 0.79 0.67 0.62 0.96 0.76 0.87 1.18 8
JANUS-GN

6

30D 1.87 2.05 1.04 1.52 1.58 10.3 0.16 2.82 9.43 5e2 3.68 1.31 0.83 0.76 1.00 1.53
JANUS 4

100D 0.86 0.34 0.59 0.96 0.93 0.38 0.76 0.56 0.50 0.80 0.60 0.97 0.97 1.25 1.03 0.97 2
JANUS-GN

0

100D 3.37 0.51 1.32 1.01 0.96 0.35 0.22 1.18 1.12 0.67 8.78 0.21 0.92 1.25 1.00 0.29
JANUS −2

500D 0.69 0.49 0.29 0.1 2e-4 0.1 0.74 3e-3 0.19 0.0 0.59 0.31 6e-3 1.05 1.00 0.96 −4

log10 (JANUS/CMA−ES)

JANUS-GN

−6

500D 0.70 0.50 0.29 0.1 1e-4 0.1 0.56 5e-3 0.22 0.0 0.61 0.31 6e-3 1.03 0.99 0.97
JANUS −8

f4 f6 f7 f8 f9 f10 f11 f12 f13 f14 f18 f19 f20 f22 f23 f24 better
Figure 2: Per-function BBOB host comparison across d ∈ \{30, 100, 500\}. Color encodes
log10(JANUS variant/CMA-ES host) under the same protocol; blue=JANUS better, red=worse.

DFO-LS (Cartis et al., 2019)) at d=30/100 (30 seeds; App. A.3, Table 8(a)), the rerun CMA-ES host
has geometric-mean final values 0.8/20.1, JANUS-GN improves these to 0.1/14.7, and standalone
DFO is far worse (1351–2419 at d=30, 2112–5477 at d=100), leaving model construction costs
behind the population-plus-local-geometry combination. To test whether the metric principle is
CMA-specific, we also attach JANUS infill to a DE host (30 seeds, 16 functions, d ∈\{30, 100, 500\}):
JANUS-DE beats vanilla DE on 86%/91%/96% of runs with geometric-mean ratios 0.86/0.58/0.59
(Table 8(b)), confirming metric-guided infill is not CMA-ES-specific.

NN-BBO baselines (same-protocol, d=500). Table 3 reports same-protocol comparisons against
8 NN-BBO / MetaBBO baselines: Random Search, GLHF (Li et al., 2024), SYMBOL (Chen et al.,
2024), LES (Lange et al., 2023), RLEPSO (Yin et al., 2021), GLEET (Ma et al., 2024), LDE (Sun
et al., 2021), SynCMA (Zhang et al., 2024), and DiBO (Yun et al., 2025) (compact 4-baseline slice
shown; full 10-baseline table in App. B.1, Table 12). B2Opt (Li et al., 2025) is omitted at d=500
because its policy network is fixed at d=10 and cannot accept 500-dimensional inputs. Full JANUS
attains the best mean error on 13 of the 16 functions in the complete baseline comparison (App. B.1,
Table 12; the compact slice below shows 9 blue cells, as the JANUS-GN ablation column can hold the
best cell and f20 lacks baseline coverage). The geometric-mean final error is 248 for JANUS, versus
8.6×104 for CMA-ES and 1.0×104 for the best NN-BBO baseline (LDE)—a &gt; 40× reduction
over the strongest learned baseline. Among the external baselines the only per-function wins are
f24 (RLEPSO) and f23 (SYMBOL), both narrow margins where the residual Jacobian carries little
exploitable curvature.

### 5.2 HIGH-DIMENSIONAL UAV PATH PLANNING (d=1135)

We next test whether the same local-geometry reuse helps in a high-dimensional structured application:
the UAV problem optimizes a 1135-dimensional spherical-coordinate flight path under terrain-
dependent costs, and JANUS estimates residual coordinates from the observed optimization trace
rather than requiring analytic derivatives. We evaluate on 28 held-out terrains with 30 runs each (840
final values per method) under a short 2,500-FE budget. Baselines span traditional optimizers (DE,
PSO, RS), adaptive variants (JDE21 (Brest et al., 2021), CMA-ES, SAHLPSO (Tao et al., 2021)),
and external MetaBBO methods (RLDEAFL (Guo et al., 2025), GLEET, GLHF, LES).

Table 2 shows that JANUS achieves the best mean cost (9,600) and median (6,644), 12.8% below
the strongest external baseline (DE at 11,006); JDE21 has the tightest IQR among the top methods
(2,795 vs JANUS’s 9,579), though at a higher mean and median. The observed UAV gains are
consistent with a regime in which the local residual coordinates are more informative and stable than

7

<details>
<summary>Page 7: complete visual reference</summary>

![Full source page 7](K:/source/OptunaSharp/docs/papers/paper-gateway/library/5204e8c7ed00c503/pages/page-0007.png)

</details>

> Quality note: Rotated or vertical text retains geometry and requires visual order review.
> Quality note: Mathematical notation or superscripts require visual review; linear text does not establish fraction, exponent or equation semantics.

<a id="page-8"></a>

<!-- Page 8 -->

Table 3: BBOB d=500: mean final error, 15 held-out functions with baseline coverage (\[−5, 5\]d,
20,000 FEs; f20 omitted, no baseline data). NN-BBO baselines and JANUS rerun under our protocol
(10 seeds); JANUS-GN from the host-controlled run (30 seeds). Blue=best, green=second. Lower
is better. Full 10-baseline comparison in App. B.1, Table 12. The Wins row counts best (blue) cells
within this reduced 7-column slice, in which the ablation column JANUS-GN can hold the best cell;
this differs from the “13/16” figure in the text, which counts functions where full JANUS is best
among all baselines over the complete 16-function set. Sig.: Bonferroni-corrected wins/ties/losses vs
JANUS (p &lt; 0.05), by a paired Wilcoxon signed-rank test over the 15 per-function final errors (each
function’s mean over the first 10 JANUS seeds, paired against the corresponding baseline mean); the
pairing unit is the function identity, so this is a cross-function comparison of aggregated final errors
rather than a within-function seed test.

![Vector artwork, page 8, region 1](K:/source/OptunaSharp/docs/papers/paper-gateway/library/5204e8c7ed00c503/figures/p0008-figure-01.png)

Host NN-BBO / MetaBBO baselines Ours

Func CMA-ES RS SYMBOL RLEPSO LDE JANUS-GN JANUS

f4 4.2e4 3.7e5 1.2e5 2.5e4 1.6e4 1.6e4 5992
f6 1.8e6 1.6e7 5.6e6 9.2e5 9.9e5 7103 2113
f7 1.5e4 3.9e4 1.8e4 1.2e4 4824 – 988
f8 5.9e8 5.9e8 4.7e7 1.5e7 1.3e7 3072 1426
f9 6.1e8 4.1e8 2.4e6 9.5e5 1.8e5 508.7 1323
f10 3.3e7 2.8e8 8.9e7 1.8e7 1.1e7 1799 1.5e6
f11 1.2e4 5833 3099 2715 3141 1934 4037
f12 8.2e9 3.0e10 4.8e9 1.9e9 1.5e9 1.8e6 1.0e4
f13 1.1e4 1.5e4 8913 5772 5347 4728 88.6
f14 267 655 176 82.5 73.0 1.5e5 0.12
f18 73.9 140 70.3 54.7 40.6 157.6 14.5
f19 3006 2096 24.2 22.4 12.3 0.34 11.7
f22 84.1 86.4 81.2 63.6 62.2 11.6 2.87
f23 1.65 1.64 1.46 1.65 1.64 11.1 1.66
f24 2.3e4 2.1e4 9315 7426 7660 7448 7435

Wins 0 0 1 1 0 5 9
Sig. vs JANUS 15/1/0 14/2/0 – – – 0/16/0 –

the scalar-derived coordinates used on BBOB, so inverse Hessian exploration is more reliable under
the short budget.

The terrain-level results are heterogeneous: JANUS ranks first on 15/28 terrains (best among all
12 compared methods, including JANUS-GN; App. B.3 reports the complementary count against
external baselines only), while other methods lead the remaining terrains under the short budget,
typically where early corridor discovery matters more than local refinement. UAV exposes more
meaningful residual structure than scalar BBOB, but the terrain distribution spans both regimes, so
App. B.3 reports the full per-terrain audit rather than only the aggregate mean. BO-like sanity checks
(sequential ET-BO, RF-SMAC-like (Hutter et al., 2011), TuRBO-like (Eriksson et al., 2019) batched)
on a terrain subset give mean costs of 13,900–14,600 (App. A.3, Table 8(c)), substantially worse
than JANUS (9,600), consistent with the difficulty of surrogate modeling at d=1135 with only 2,500
evaluations.

Long-budget validation. Extending the same terrain protocol to 10,000 and 20,000 FEs (App. B.3,
Table 16), JANUS still leads at 10k (8,488 vs CMA-ES 8,587, −1.2%); at 20k the methods converge
to near-parity (8,019 vs 8,023). Thus JANUS is best interpreted as an early-budget accelerator — the
regime of interest when evaluations cost minutes to hours — and the 20,000-FE result suggests that
JANUS does not cause a persistent degradation in this UAV setting.

Figure 3 shows the mechanism behind the UAV aggregate numbers: the convergence panels show
JANUS separating from CMA-ES and the traditional baselines before the full budget is exhausted,
while the 3D path panels show the corresponding effect in the native trajectory space—locally
informed candidates steer the host toward smoother, lower-cost corridors rather than merely improving
a scalar score at the end of the run.

JANUS-GN follows the same early descent trend but more conservatively. The full JANUS variant
gains more when CMIX has a stable local metric to exploit; JANUS-GN wins several individual

8

<details>
<summary>Page 8: complete visual reference</summary>

![Full source page 8](K:/source/OptunaSharp/docs/papers/paper-gateway/library/5204e8c7ed00c503/pages/page-0008.png)

</details>

> Quality note: Mathematical notation or superscripts require visual review; linear text does not establish fraction, exponent or equation semantics.

<a id="page-9"></a>

<!-- Page 9 -->

4 350 400450 JANUS
4×10 3×104 300 350 JANUS-GN
4 250 300
3×10 4 200 200250 DE

150

4 2×10 100 100150 JDE21
2×10 50 50 PSO

900 900

700800 700800 CMAES

4
4 10 600 600 SAHLPSO
10 100 500 100 500

Best-so-far cost

Best-so-far cost

3 200 300 400 300400 y 200 300 400 300400 y RS
6×10 x 500 600 700 100200 x 500 600 700 100200 StartGoal
0 1000 2000 0 1000 2000 800 900 800 900
Function evaluations Function evaluations Threats

(a) T1 convergence (b) T17 convergence (c) T1 paths (d) T17 paths

![Figure 3: UAV convergence and path quality on two representative terrains. JANUS separates early and produces smooth low-cost paths; JANUS-GN follows more conservatively, while other baselines often plateau higher under the 2,500-FE budget.](K:/source/OptunaSharp/docs/papers/paper-gateway/library/5204e8c7ed00c503/figures/p0009-figure-01.png)

Figure 3: UAV convergence and path quality on two representative terrains. JANUS separates early
and produces smooth low-cost paths; JANUS-GN follows more conservatively, while other baselines
often plateau higher under the 2,500-FE budget.

Table 4: MOO main results. 20,000 evaluations, 10 runs/task (180 low-, 200 many-objective), unified
post-hoc HV reference points. Blue cell=best, green cell=second per regime. +/≈/−: Mann–
Whitney significant wins/ties/losses vs strongest host.

![Vector artwork, page 9, region 2](K:/source/OptunaSharp/docs/papers/paper-gateway/library/5204e8c7ed00c503/figures/p0009-figure-02.png)

Regime Method Avg. Rank Norm. HV +/≈/−

Low JANUS-SMS 5% 2.00 0.994 3/15/0
(2–3 obj.) JANUS-SMS 2% 2.50 0.993 –
SMS-EMOA 3.11 0.972 –
NSGA-II 3.61 0.967 –
MOEA/D 5.00 0.926 –
Random 6.50 0.756 –
CMA-ES 8.11 0.711 –

Many JANUS-AGE2 5% 2.00 0.972 9/11/0
(5/8/10 obj.) JANUS-AGE2 2% 2.30 0.973 –
AGE-MOEA2 3.65 0.915 –
NSGA-III 5.00 0.888 –
RVEA 6.40 0.865 –
NSGA-II 6.50 0.917 –
MOEA/D 7.45 0.820 –

terrains in App. B.3 where corridor discovery matters more than metric reshaping under the short
budget.

### 5.3 MULTI-OBJECTIVE OPTIMIZATION (38 TASKS)

We then test whether JANUS can improve proposal generation without interfering with Pareto
survival, treating each objective as a residual coordinate and generating a small fraction (ρ=5%)
of offspring from the inverse Hessian metric while leaving host survival unchanged. For the low-
objective suite (18 tasks, 2–3 objectives over ZDT, DTLZ, WFG), we pair JANUS with SMS-EMOA
(hypervolume-based survival); for the many-objective suite (20 tasks, 5/8/10 objectives), we pair
JANUS with AGE-MOEA2 (geometry-aware survival). All experiments use 20,000 evaluations, 10
runs, and unified post-hoc hypervolume reference points.

Headline: 12/38 significant wins, 0 regressions, regime-dependent advantage. Table 4 summa-
rizes the leaderboard. JANUS-SMS 5% and JANUS-AGE2 5% each achieve the best average rank
(2.00) in their respective regime. Pairwise Mann–Whitney U tests (p &lt; 0.05) against the strongest
host baseline show 3/18 and 9/20 significant wins respectively (12/38 total; App. B.2 additionally
reports raw best-average-rank win counts, higher because they are not gated by a significance test).
No statistically significant regressions were detected under these tests and the Bonferroni correction
— a property that matters more than raw win rate here, since infill mechanisms that reshape proposal
distributions risk degrading the host where geometry is unhelpful. The remaining tasks are statistically
tied.

Regime-dependent gains. The MOO results are intentionally reported by regime rather than pooled
into a single headline; the rank matrix (App. B.2, Fig. 4) localizes the advantage. JANUS-SMS is
strongest on WFG/DTLZ, where a few metric-guided candidates quickly improve the hypervolume-

9

<details>
<summary>Page 9: complete visual reference</summary>

![Full source page 9](K:/source/OptunaSharp/docs/papers/paper-gateway/library/5204e8c7ed00c503/pages/page-0009.png)

</details>

> Quality note: Rotated or vertical text retains geometry and requires visual order review.
> Quality note: Mathematical notation or superscripts require visual review; linear text does not establish fraction, exponent or equation semantics.

<a id="page-10"></a>

<!-- Page 10 -->

Table 5: Component ablation across representative domains. Ratios are relative to the base host; lower
is better for BBOB/UAV and higher is better for MOO.

![Vector artwork, page 10, region 1](K:/source/OptunaSharp/docs/papers/paper-gateway/library/5204e8c7ed00c503/figures/p0010-figure-01.png)

Variant GN CMIX BBOB ↓ UAV ↓ MOO ↑

Base host 1.00× 1.00× 1.00×
GN only ✓ 0.85× 0.91× 1.02×
CMIX only ✓ 0.92× 0.88× 0.97×
JANUS ✓ ✓ 0.83× 0.89× 1.01×

dominated region; JANUS-AGE2 leads at 5/8 objectives, where objective-vector residuals expose
useful local geometry that geometry-aware survival exploits. At 10 objectives the gap narrows for both
hosts because survival is already strong, mirroring the scalar BBOB boundary: JANUS helps when
candidate geometry is a bottleneck, and its effect shrinks once host survival has saturated the diversity-
convergence trade-off. JANUS is thus most valuable as a default-on augmentation for hosts facing
moderate-dimensional (m ≤10) Pareto fronts, where a small infill fraction improves convergence
at negligible risk. Runtime: with CUDA-accelerated metric estimation, JANUS-SMS/AGE2 5%
are 1.46×/1.22× slower than SMS-EMOA/AGE-MOEA2 (App. B.2, Fig. 5)—unlikely to be the
bottleneck when evaluation dominates wall-clock time.

### 5.4 ABLATION: TWO USES OF ONE JACOBIAN

Finally, we isolate the two uses of J via a component ablation over representative scalar, UAV, and
MOO settings (Table 5). GN improves early exploitation, CMIX is strongest when curvature is stable
enough to guide exploration, and full JANUS gives the best scalar result while remaining competitive
on structured and multi-objective tasks.

Practical recommendation: use JANUS (CMIX+GN, tuned fuse cadence) as the scalar default; fall
back to JANUS-GN when the trace is noisy, curvature unstable, or the objective multimodal.

### 5.5 IMPLEMENTATION AND PROTOCOL CONTROLS

JANUS is a candidate generator around the host ask/tell interface. At each generation the host proposes
its ordinary offspring; JANUS reads the most recent evaluated window, fits the residual chart, and
contributes a few additional candidates, all evaluated before the host’s standard tell/update step. It
never overwrites the host covariance, restart logic, archive, or survival operator, so the implementation
is portable across CMA-ES, DE, SMS-EMOA, and AGE-MOEA2-style loops. In CMA-style hosts√
we clip the proposed displacement in host-whitened coordinates to ∥· ∥≤ d; DE and MOO hosts
use analogous box and step-length constraints, keeping candidates on the host sampling scale and
preventing an ill-conditioned Jacobian from dominating adaptation or survival. The chart is rebuilt
more often for scalar objectives, where pseudo-residuals are less stable than observed components;
for UAV/MOO, coordinates are observed directly and CMIX is used more aggressively (App. A).

Operating regime. JANUS helps most when the recent trace holds a stable local metric the host
does not already encode: the effect grows with dimension on BBOB, transfers to a proprietary 1135D
ISP task (22% error reduction vs. CMA-ES at 5,000 evaluations, App. C.1), and is strongest on
UAV, where the objective decomposition yields meaningful residual coordinates. The same evidence
marks its limits: JANUS-GN is safer on multimodal functions, short UAV budgets can leave curvature
unstable before CMIX has enough trace, and in MOO the gain is largest before host survival saturates
diversity (App. C). All comparisons share seeds, budgets, initialization, and protocol (App. A.1).

## 6 CONCLUSION

Population optimizers discard the local geometry already in their evaluation trace; JANUS recovers
it through one online residual Jacobian, fed back as bounded infill without touching host selection,
covariance adaptation, or step-size control. With hyperparameters shared across tasks (bar the BBOB
fuse cadence; App. A), the same mechanism improves CMA-ES on 11–15/16 BBOB functions, attains
the best mean UAV cost, and yields 12/38 significant MOO wins with no significant regressions.
Under the evaluated protocols this pairing outperformed either standalone variant.

10

<details>
<summary>Page 10: complete visual reference</summary>

![Full source page 10](K:/source/OptunaSharp/docs/papers/paper-gateway/library/5204e8c7ed00c503/pages/page-0010.png)

</details>

> Quality note: Mathematical notation or superscripts require visual review; linear text does not establish fraction, exponent or equation semantics.

<a id="page-11"></a>

<!-- Page 11 -->

## REFERENCES

Youhei Akimoto and Nikolaus Hansen. Projection-based restricted covariance matrix adaptation
for high dimension. In Proceedings of the Genetic and Evolutionary Computation Conference
(GECCO), pp. 197–204, 2016.

Youhei Akimoto, Yuichi Nagata, Isao Ono, and Shigenobu Kobayashi. Bidirectional relation between
cma evolution strategies and natural evolution strategies. In Parallel Problem Solving from Nature,
PPSN XI, pp. 154–163. Springer, 2010.

James Bergstra and Yoshua Bengio. Random search for hyper-parameter optimization. Journal of
Machine Learning Research, 13(10):281–305, 2012.

Nicola Beume, Boris Naujoks, and Michael Emmerich. Sms-emoa: Multiobjective selection based
on dominated hypervolume. European Journal of Operational Research, 181(3):1653–1669, 2007.

Janez Brest, Mirjam Sepesy Mauˇcec, and Borko Boˇskovi´c. Self-adaptive differential evolution
algorithm with population size reduction for single objective bound-constrained optimization:
Algorithm j21. In 2021 IEEE Congress on Evolutionary Computation (CEC), pp. 817–824. IEEE,
2021.

Coralia Cartis, Jan Fiala, Benjamin Marber, and Lindon Roberts. Improving the flexibility and ro-
bustness of model-based derivative-free optimization solvers. ACM Transactions on Mathematical
Software, 45(3):1–41, 2019.

Jiacheng Chen, Zeyuan Ma, Hongshu Guo, Yining Ma, Jie Zhang, and Yue-Jiao Gong. SYMBOL:
Generating flexible black-box optimizers through symbolic equation learning. In The Twelfth
International Conference on Learning Representations, 2024.

Ran Cheng, Yaochu Jin, Markus Olhofer, and Bernhard Sendhoff. A reference vector guided
evolutionary algorithm for many-objective optimization. IEEE Transactions on Evolutionary
Computation, 20(5):773–791, 2016.

Kalyanmoy Deb and Himanshu Jain. An evolutionary many-objective optimization algorithm
using reference-point-based nondominated sorting approach, part i: solving problems with box
constraints. IEEE Transactions on Evolutionary Computation, 18(4):577–601, 2014.

Kalyanmoy Deb, Amrit Pratap, Sameer Agarwal, and T. Meyarivan. A fast and elitist multiobjective
genetic algorithm: Nsga-ii. IEEE Transactions on Evolutionary Computation, 6(2):182–197, 2002.

Michael T. M. Emmerich, Kyriakos C. Giannakoglou, and Boris Naujoks. Single- and multiobjective
evolutionary optimization assisted by gaussian random field metamodels. IEEE Transactions on
Evolutionary Computation, 10(4):421–439, 2006.

David Eriksson, Michael Pearce, Jacob Gardner, Ryan D. Turner, and Matthias Poloczek. Scalable
global optimization via local Bayesian optimization. In Advances in Neural Information Processing
Systems, volume 32, 2019.

Hongshu Guo, Sijie Ma, Zechuan Huang, Yuzhi Hu, Zeyuan Ma, Xinglin Zhang, and Yue-Jiao Gong.
Reinforcement learning-based self-adaptive differential evolution through automated landscape
feature learning. In Proceedings of the Genetic and Evolutionary Computation Conference, pp.
1117–1126, 2025.

Nikolaus Hansen. Injecting external solutions into cma-es. Research Report RR-7748, INRIA,
October 2011.

Nikolaus Hansen. The cma evolution strategy: A tutorial. arXiv preprint arXiv:1604.00772, 2016.

Frank Hutter, Holger H. Hoos, and Kevin Leyton-Brown. Sequential model-based optimization for
general algorithm configuration. In Learning and Intelligent Optimization (LION 5), pp. 507–523,
2011.

Yaochu Jin. Surrogate-assisted evolutionary computation: Recent advances and future challenges.
Swarm and Evolutionary Computation, 1(2):61–70, 2011.

11

<details>
<summary>Page 11: complete visual reference</summary>

![Full source page 11](K:/source/OptunaSharp/docs/papers/paper-gateway/library/5204e8c7ed00c503/pages/page-0011.png)

</details>

<a id="page-12"></a>

<!-- Page 12 -->

Robert Tjarko Lange, Tom Schaul, Yutian Chen, Tom Zahavy, Valentin Dalibard, Chris Lu, Satinder
Singh, and Sebastian Flennerhag. Discovering evolution strategies via meta-black-box optimization.
In International Conference on Learning Representations, 2023.

Xiaobin Li, Kai Wu, Yujian Betterest Li, Xiaoyu Zhang, Handing Wang, and Jing Liu. Pretrained opti-
mization model for zero-shot black box optimization. Advances in Neural Information Processing
Systems, 37:14283–14324, 2024.

Xiaobin Li, Kai Wu, Xiaoyu Zhang, and Handing Wang. B2opt: Learning to optimize black-box
optimization with little budget. Proceedings of the AAAI Conference on Artificial Intelligence, 39
(17):18502–18510, Apr. 2025.

Ilya Loshchilov. A computationally efficient limited memory cma-es for large scale optimization. In
Proceedings of the Genetic and Evolutionary Computation Conference, pp. 397–404, 2014.

Zeyuan Ma, Hongshu Guo, Jiacheng Chen, Zhenrui Li, Guojun Peng, Yue-Jiao Gong, Yining Ma,
and Zhiguang Cao. Metabox: A benchmark platform for meta-black-box optimization with
reinforcement learning. Advances in Neural Information Processing Systems, 36:10775–10795,
2023.

Zeyuan Ma, Jiacheng Chen, Hongshu Guo, Yining Ma, and Yue-Jiao Gong. Auto-configuring
exploration-exploitation tradeoff in evolutionary computation via deep reinforcement learning. In
Proceedings of the Genetic and Evolutionary Computation Conference, 2024.

Yann Ollivier, Ludovic Arnold, Anne Auger, and Nikolaus Hansen. Information-geometric opti-
mization algorithms: A unifying picture via invariance principles. Journal of Machine Learning
Research, 18(18):1–65, 2017.

Annibale Panichella. An improved pareto front modeling algorithm for large-scale many-objective
optimization. In Proceedings of the Genetic and Evolutionary Computation Conference, pp.
565–573, 2022.

Michael J. D. Powell. The newuoa software for unconstrained optimization without derivatives.
Large-Scale Nonlinear Optimization, pp. 255–297, 2006.

Michael JD Powell et al. The bobyqa algorithm for bound constrained optimization without derivatives.
Cambridge NA Report NA2009/06, University of Cambridge, Cambridge, 26(26-46):1, 2009.

Ofer M. Shir, Jonathan Roslund, and Herschel Rabitz. Forced optimal covariance adaptive learning:
modified cma-es for efficient hessian determination. In Proceedings of the 12th Annual Conference
on Genetic and Evolutionary Computation, GECCO ’10, pp. 421–422, New York, NY, USA, 2010.
Association for Computing Machinery. ISBN 9781450300728. doi: 10.1145/1830483.1830563.
URL https://doi.org/10.1145/1830483.1830563.

Jianyong Sun, Xin Liu, Thomas B¨ack, and Zongben Xu. Learning adaptive differential evolution
algorithm from optimization experiences by policy gradient. IEEE Transactions on Evolutionary
Computation, 25(4):666–680, 2021.

Xinmin Tao, Xiangke Li, Wei Chen, Tian Liang, Yetong Li, Jie Guo, and Lin Qi. Self-adaptive two
roles hybrid learning strategies-based particle swarm optimization. Information Sciences, 578:
457–481, 2021.

Daan Wierstra, Tom Schaul, Jan Peters, and J¨urgen Schmidhuber. Natural evolution strategies.
Journal of Machine Learning Research, 15:949–980, 2014.

Shiyuan Yin, Yi Liu, GuoLiang Gong, Huaxiang Lu, and Wenchang Li. Rlepso: Reinforcement
learning based ensemble particle swarm optimizer. In Proceedings of the 2021 4th International
Conference on Algorithms, Computing and Artificial Intelligence, pp. 1–6, 2021.

Taeyoung Yun, Kiyoung Om, Jaewoo Lee, Sujin Yun, and Jinkyoo Park. Posterior inference with
diffusion models for high-dimensional black-box optimization. In Forty-second International
Conference on Machine Learning, 2025. URL https://openreview.net/forum?id=EXds2NBOoq.

12

<details>
<summary>Page 12: complete visual reference</summary>

![Full source page 12](K:/source/OptunaSharp/docs/papers/paper-gateway/library/5204e8c7ed00c503/pages/page-0012.png)

</details>

<a id="page-13"></a>

<!-- Page 13 -->

Qingfu Zhang and Hui Li. Moea/d: A multiobjective evolutionary algorithm based on decomposition.
IEEE Transactions on Evolutionary Computation, 11(6):712–731, 2007.

Zhengfei Zhang, Yunyue Wei, and Yanan Sui. An invariant information geometric method for high-
dimensional online optimization. In 6th Annual Learning for Dynamics &amp; Control Conference, pp.
641–653. PMLR, 2024.

13

<details>
<summary>Page 13: complete visual reference</summary>

![Full source page 13](K:/source/OptunaSharp/docs/papers/paper-gateway/library/5204e8c7ed00c503/pages/page-0013.png)

</details>

<a id="page-14"></a>

<!-- Page 14 -->

## JANUS: Online Jacobian-Aligned Infill

## Supplementary Material

## A HYPERPARAMETERS AND REPRODUCIBILITY

Table 6 lists the JANUS hyperparameters held fixed in all experiments. The adaptive mixing weight
w (Eq. 4) is derived only from problem dimensions. One exception is the BBOB fuse cadence:
fuse every is tuned once for the d ∈\{30, 100\} scalar setting (fuse every=50) rather than reused
verbatim at d = 500; every other hyperparameter in Table 6 is shared unchanged across every
experiment in this paper.

For completeness, the protocols used by the main paper are: BBOB uses 16 held-out functions,
d ∈\{30, 100, 500\}, 20,000 FEs, and 30 runs; UAV uses 28 held-out terrains, 2,500 FEs, and 30 runs;
MOO uses pymoo hosts, 20,000 evaluations, 10 runs, and unified post-hoc hypervolume reference
points.

A.1 SEARCH SPACES AND EVALUATION PROTOCOLS

BBOB. We use a MetaBox/COCO-compatible protocol: 16 held-out functions, d ∈\{30, 100, 500\},
\[−5, 5\]d, 20,000 FEs, and 30 runs. The CMA-ES host comparison in Fig. 2 matches functions,
dimensions, seeds, population size, initialization, box constraints, and budget. Non-ABOM baselines
use this setting whenever supported. ABOM is treated only as a reported reference because its paper
states a wider box and does not provide public source code.

UAV path planning. UAV uses a 1135-dimensional spherical-coordinate path parameterization
decoded to Cartesian trajectories before evaluation. We use 28 held-out terrains, 30 runs, and a
2,500-FE budget; the long-budget check repeats the protocol at 10,000 and 20,000 FEs. Table 18
keeps per-terrain results because the terrain distribution is heterogeneous.

Multi-objective optimization. MOO experiments use pymoo tasks and host survival operators:
SMS-EMOA-style selection for low-objective tasks and AGE-MOEA2 (Panichella, 2022)-style
survival for many-objective tasks. Each task uses 20,000 evaluations and 10 runs. Hypervolume uses
unified post-hoc reference points, and JANUS uses the objective vector as the residual coordinate
system.

Table 6: JANUS hyperparameters shared across all experiments. The sole per-setting exception is the
BBOB fuse cadence (assembly period), tuned once for the scalar BBOB setting (d ∈\{30, 100\}) and
reused verbatim elsewhere.

Symbol Value Meaning

window 300 Sliding window of recent (x, s) pairs for Jacobian regression
µ 10−3 Ridge regularization for residual-gradient regression
λ \[1, 10\] Levenberg–Marquardt damping (stable across range)
κ 1.0 Bounded-injection clip factor
wbase 0.5 Base mixing weight before dimension scaling
\[wmin, wmax\] \[0.02, 0.6\] Clip range for the adaptive mixing weight
assembly period 4–6 gen† Generations between Jacobian re-estimation (†50 for the scalar BBOB
setting)
ρ (MOO) 0.05 Fraction of offspring allocated to JANUS infill

Sensitivity sweep. We vary each hyperparameter in Table 6 one at a time (window size, damping
λ, CMIX damping λmix, infill ratio ρ, and the mixing-weight bounds \[wmin, wmax\]) across 18
configurations on the d=100 BBOB suite. Most configurations stay within 1.5× of the default
geometric-mean final error, and no single setting collapses the method; the largest sensitivity is to the
window size, since a window that is too short destabilizes the ridge-regression Jacobian estimate and

14

<details>
<summary>Page 14: complete visual reference</summary>

![Full source page 14](K:/source/OptunaSharp/docs/papers/paper-gateway/library/5204e8c7ed00c503/pages/page-0014.png)

</details>

> Quality note: Mathematical notation or superscripts require visual review; linear text does not establish fraction, exponent or equation semantics.

<a id="page-15"></a>

<!-- Page 15 -->

## a window that is too long mixes stale curvature into J. This supports using one fixed hyperparameter set across BBOB, UAV, and MOO rather than tuning per benchmark.

## A.2 NOTATION AND VARIANTS

## See Table 7 for notation and Table 7 for implementation variants.

## Table 7: Notation and JANUS instantiations.

Notation

Symbol Meaning

x ∈X Candidate solution in the native search space
f(x) Scalar black-box objective or scalarized cost
r(x) Observed component/objective vector when available
s(x) Residual coordinates used for local Jacobian fitting
W Recent evaluated window for local fitting
J Local residual Jacobian from current window
G = J⊤J Normal metric reused for exploitation and exploration
λ Damping/regularization
ρ Fraction of candidates allocated to CMIX

Instantiations

Name Mechanism

JANUS-GN Bounded damped Gauss–Newton infill from G
JANUS JANUS-GN plus inverse-curvature candidate mixing (CMIX)
JANUS-SMS JANUS infill attached to SMS-EMOA survival
JANUS-AGE2 JANUS infill attached to AGE-MOEA2 survival
JANUS-DE JANUS infill attached to differential evolution
Random infill Same infill budget without G

Where used

All benchmarks
BBOB/UAV
Structured/MOO tasks
All
Online adaptation
Metric construction
JANUS infill
Numerical stability
Exploration budget

Role in experiments

Exploitation-only mode
Full scalar/UAV mode
Low-objective MOO host
Many-objective MOO host
Non-CMA host check
Metric-free control

Default host. Scalar BBOB and UAV use CMA-ES unless otherwise stated; JANUS-GN and JANUS differ only by whether CMIX candidates

are enabled. MOO variants use the host named in the method label.

## A.3 EXTENDED BASELINE COMPARISONS

Table 8 (referenced from the main paper, Sec. 5.1 and Sec. 5.2) adds three checks: classical local
modeling on BBOB, JANUS-DE as a non-CMA host check, and BO-style surrogate modeling on
1135-dimensional UAV.

Restricted-covariance baselines (LM-CMA, VkD-CMA). Table 9 isolates the structural predic-
tion of Prop. 1 by comparing two restricted-covariance hosts—LM-CMA (LPCMA) (Loshchilov,
2014\) and VkD-CMA (VLPCMA) (Akimoto &amp; Hansen, 2016)—against a full-rank CMA-ES host on
the 16 held-out BBOB functions at d ∈\{30, 100\}, 10 seeds, 5,000 FEs, all under this single shared
protocol. It reports geometric-mean final error (value minus the known optimum) at 5,000 FEs over
10 seeds, and is therefore separate from the DFO panel of Table 8, which reports geometric-mean final
value at 20,000 FEs over 30 seeds. LPCMA and VLPCMA are the canonical restricted-covariance
baselines for Prop. 1: both adapt covariance in a fixed low-dimensional basis, so the proposition
predicts they should underperform a full-rank host whenever the residual Jacobian carries curva-
ture outside that basis. Empirically, LPCMA achieves geometric-mean 67.4 (d=30) and 4.98×103
(d=100) and VLPCMA achieves 2.33×103 and 1.96×104; both are worse than the full-rank CMA-
ES host (14.8 and 332) at every dimension, and the gap widens from 4.6× at 30D to 15.0× at 100D
for LPCMA. The observed ordering is compatible with the structural limitation described by Prop. 1
— restricted-covariance hosts cannot express an arbitrary residual-aligned metric — although the
proposition addresses expressiveness and does not by itself predict optimization performance, the
direction of the gap, or its growth with dimension; those remain empirical observations. (JANUS’s
own advantage over a full-rank host is a separate, convergence-speed claim, quantified on real 30-seed
data in the DFO panel of Table 8, not here.)

## 15

<details>
<summary>Page 15: complete visual reference</summary>

![Full source page 15](K:/source/OptunaSharp/docs/papers/paper-gateway/library/5204e8c7ed00c503/pages/page-0015.png)

</details>

> Quality note: Mathematical notation or superscripts require visual review; linear text does not establish fraction, exponent or equation semantics.

<a id="page-16"></a>

<!-- Page 16 -->

![Vector artwork, page 16, region 1](K:/source/OptunaSharp/docs/papers/paper-gateway/library/5204e8c7ed00c503/figures/p0016-figure-01.png)

Table 8: Extended baseline checks (Sec. 5.1). Lower is better except win rate. Blue cell=best, green
cell=second.

(a) BBOB DFO comparison, geometric-mean final value.

Method d=30 d=100

JANUS-GN 0.1 14.7
CMA-ES host 0.8 20.1
BOBYQA 1351 2112
COBYLA 1447 2524
DFO-LS 1864 2641
NEWUOA 2419 5477

(b) JANUS-DE vs. vanilla DE, host-agnostic check.

Metric d=30 d=100 d=500

JANUS-DE win rate 86% 91% 96%
JANUS-DE geo ratio 0.86 0.58 0.59

(c) UAV BO-like sanity check, d=1135, 2,500 FEs, 3 terrains.

Method Mean cost

JANUS 9,600
ET-BO 13,923
RF-SMAC 14,022
TuRBO 14,634
Random 17,701

![Table 9: BBOB restricted- vs. full-rank host comparison: 16 held-out functions, 10 seeds, 5,000 FEs, (−5, 5)d. Geometric-mean final error across 16 functions, and per-function wins (count of functions on which each method attains the lowest error). Both LPCMA (LM-CMA) and VLPCMA (VkD-CMA), which adapt covariance in a fixed low-dimensional basis, underperform the full-rank CMA-ES host at d ∈{30, 100}. This is compatible with the structural limitation of Prop. 1 — restricted- covariance hosts cannot express an arbitrary residual-aligned metric — though the proposition concerns expressiveness and does not by itself predict this performance ordering. The full-rank CMA-ES host is 4.6× better than LM-CMA at 30D and 15.0× better at 100D. Blue cell=best, green cell=second-best.](K:/source/OptunaSharp/docs/papers/paper-gateway/library/5204e8c7ed00c503/figures/p0016-figure-02.png)

Table 9: BBOB restricted- vs. full-rank host comparison: 16 held-out functions, 10 seeds, 5,000 FEs,
\[−5, 5\]d. Geometric-mean final error across 16 functions, and per-function wins (count of functions on
which each method attains the lowest error). Both LPCMA (LM-CMA) and VLPCMA (VkD-CMA),
which adapt covariance in a fixed low-dimensional basis, underperform the full-rank CMA-ES
host at d ∈\{30, 100\}. This is compatible with the structural limitation of Prop. 1 — restricted-
covariance hosts cannot express an arbitrary residual-aligned metric — though the proposition
concerns expressiveness and does not by itself predict this performance ordering. The full-rank
CMA-ES host is 4.6× better than LM-CMA at 30D and 15.0× better at 100D. Blue cell=best, green
cell=second-best.

d=30 d=100

Method Geo-Mean Wins Geo-Mean Wins

CMA-ES (host, full-rank) 14.8 16 332 16
LM-CMA (LPCMA) 67.4 0 4.98e3 0
VkD-CMA (VLPCMA) 2.33e3 0 1.96e4 0
Random Search 3.63e3 0 1.90e4 0

Readout. Classical DFO builds useful local scalar models, but these standalone methods spend many
evaluations maintaining local interpolation or trust-region models and do not scale well under the
tested BBOB budgets. This does not mean that local modeling is unhelpful; rather, JANUS uses
local geometry as a bounded infill signal on top of a population host instead of replacing the host.
JANUS-DE further shows that this signal is not specific to CMA-ES adaptation. The UAV BO-like
sanity check is included for the same reason: at d=1135 and 2,500 evaluations, generic surrogates
struggle to learn a useful global model, whereas JANUS only asks for a local metric from the recent
trace.

## B PER-DOMAIN EVIDENCE

B.1 BBOB PER-FUNCTION RESULTS

Takeaway. BBOB is strong in aggregate, but the per-function tables are important because scalar
BBOB is the least favorable setting for JANUS: the benchmark exposes only scalar values, so residual
coordinates must be estimated from the recent trace. Table 10 gives the compact per-function ratio
view. The results therefore test whether pseudo-residual geometry is sufficient to improve a population
optimizer, and where it becomes unreliable.

16

<details>
<summary>Page 16: complete visual reference</summary>

![Full source page 16](K:/source/OptunaSharp/docs/papers/paper-gateway/library/5204e8c7ed00c503/pages/page-0016.png)

</details>

> Quality note: Mathematical notation or superscripts require visual review; linear text does not establish fraction, exponent or equation semantics.

<a id="page-17"></a>

<!-- Page 17 -->

## Table 10: Dense BBOB function strip. Entries are log10(JANUS/ABOM) final-value ratios; negative cells favor JANUS.

Function 30D 100D 500D Wins Mean log-ratio Verdict

F4 -0.75 -1.08 -0.23 3/3 -0.69 robust win
F6 -9.39 -4.54 -0.30 3/3 -4.74 robust win
F7 -1.84 -2.11 -0.67 3/3 -1.54 robust win
F8 -1.53 -1.57 -1.66 3/3 -1.59 robust win
F9 -3.25 -3.05 -2.04 3/3 -2.78 robust win
F10 -3.30 -3.48 -0.52 3/3 -2.43 robust win
F11 -3.59 -3.11 -0.18 3/3 -2.29 robust win
F12 -7.41 -9.37 -2.06 3/3 -6.28 robust win
F13 -1.40 -2.37 -0.82 3/3 -1.53 robust win
F14 -2.39 -5.17 -1.60 3/3 -3.05 robust win
F18 -3.28 -1.65 -0.34 3/3 -1.75 robust win
F19 1.17 0.80 -0.20 1/3 0.59 hard case
F20 ref¡0 ref¡0 -1.84 1/3 -1.84 hard case
F22 -0.93 -1.12 0.34 2/3 -0.57 mostly win
F23 1.02 0.02 -0.00 1/3 0.35 hard case
F24 -0.15 -0.59 -0.03 3/3 -0.26 robust win

Readout. Negative cells favor JANUS; red/amber cells mark the locality boundary. The full table
below keeps the per-function mean±std audit at d = 30 and d = 100, because the variance matters for
distinguishing systematic gains from unstable runs. With the tuned fuse cadence, JANUS (CMIX+GN
on pyCMA host) wins 11/16 at d=30 and 15/16 at d=100; JANUS-GN provides a more conservative
fallback that is competitive on per-function wins but has higher overall mean due to instability on a
few ill-conditioned functions.

![Table 11: BBOB d = 30 and d = 100: mean±std; lower is better. 30D JANUS uses tuned configu- ration (fuse every=50, 10 seeds); JANUS-GN 30D and all 100D from the host-controlled run (30 seeds). Blue cell=best, green cell=second-best. W/T/L: wins/ties/losses vs CMA-ES per function (mean comparison).](K:/source/OptunaSharp/docs/papers/paper-gateway/library/5204e8c7ed00c503/figures/p0017-figure-01.png)

Table 11: BBOB d = 30 and d = 100: mean±std; lower is better. 30D JANUS uses tuned configu-
ration (fuse every=50, 10 seeds); JANUS-GN 30D and all 100D from the host-controlled run (30
seeds). Blue cell=best, green cell=second-best. W/T/L: wins/ties/losses vs CMA-ES per function
(mean comparison).

![Vector artwork, page 17, region 2](K:/source/OptunaSharp/docs/papers/paper-gateway/library/5204e8c7ed00c503/figures/p0017-figure-02.png)

d = 30 d = 100

Func JANUS-GN JANUS CMA-ES JANUS-GN JANUS CMA-ES

35.12 70.01 66.59 979 1014 2685
f4 ±7.39 ±14.79 ±13.86 ±78.20 ±74.55 ±689

2.44e-07 2.99e-08 5.11e-08 3140 3416 1.80e+04
f6 ±3.8e-08 ±3.7e-08 ±7.4e-08 ±659 ±1031 ±2845

0.276 8.36 9.82 229 301 5101
f7 ±0.53 ±2.84 ±8.33 ±84.01 ±95.92 ±1841

3.15 13.13 16.75 2101 1922 3075
f8 ±2.26 ±1.60 ±12.93 ±3846 ±3465 ±4851

1.71 11.50 11.69 792 670 9.69e+04
f9 ±1.31 ±1.70 ±1.27 ±1751 ±1817 ±2.8e+05

2.62e+04 134 122 2.06e+07 3.28e+07 1.08e+08
f10 ±8.9e+04 ±107 ±123 ±4.3e+06 ±7.0e+06 ±2.3e+07

9.18e-06 38.79 60.02 2.09e+05 2.41e+05 1.04e+06
f11 ±1.1e-05 ±52.69 ±87.47 ±2.2e+04 ±2.8e+04 ±8.6e+05

0.025 0.543 0.0273 1.35e+05 4.35e+05 4.59e+08
f12 ±0.044 ±1.08 ±0.075 ±1.2e+05 ±8.1e+05 ±3.0e+08

7.82e-03 0.602 0.498 6.66 7.09 105
f13 ±2.6e-03 ±0.85 ±0.48 ±2.41 ±2.23 ±29.86

2.35e-06 4.70e-07 4.75e-07 150 165 1376
f14 ±4.7e-07 ±2.5e-07 ±2.0e-07 ±39.76 ±42.95 ±443

1.33 0.272 0.391 87.35 91.36 213
f18 ±2.36 ±0.35 ±0.24 ±32.66 ±28.72 ±45.14

0.559 3.09 3.41 7.86 7.93 19.63
f19 ±0.85 ±1.74 ±2.01 ±0.36 ±0.27 ±21.75

-37.97 1.86 1.87 -32.33 -33.10 -2.83
f20 ±4.61 ±0.2 ±0.19 ±2.58 ±2.37 ±1.20
f22 86.57 86.57 86.57 86.57 86.57 86.57

2.93 3.13 3.23 4.84 4.88 7.30
f23 ±0.4 ±0.33 ±0.5 ±0.34 ±0.34 ±0.57

41.77 119 135 888 882 1055
f24 ±8.40 ±79.99 ±65.53 ±77.19 ±85.20 ±27.55

W/T/L vs CMA-ES 11/1/4 11/1/4 – 15/1/0 15/1/0 –

Readout. The dense tables match the strip plot. JANUS variants dominate well-conditioned and weakly
multimodal functions, while losses concentrate on f19/f23-like landscapes where basin discovery or
weak residual structure matters more than local conditioning. The host-controlled comparison shows

## 17

<details>
<summary>Page 17: complete visual reference</summary>

![Full source page 17](K:/source/OptunaSharp/docs/papers/paper-gateway/library/5204e8c7ed00c503/pages/page-0017.png)

</details>

> Quality note: Mathematical notation or superscripts require visual review; linear text does not establish fraction, exponent or equation semantics.

<a id="page-18"></a>

<!-- Page 18 -->

Table 12: BBOB d=500, full 10-baseline comparison: mean final error, 16 held-out functions
(\[−5, 5\]d, 20,000 FEs). NN-BBO baselines and JANUS rerun under our protocol (10 seeds). JANUS-
GN shown from the host-controlled run (30 seeds). Blue cell=best, green cell=second. Lower is
better. B2Opt: N/A (network fixed at 10D). Sig. row: Bonferroni-corrected wins/ties/losses vs JANUS
(p &lt; 0.05), by a paired Wilcoxon signed-rank test over the 16 per-function final errors (each func-
tion’s mean over the first 10 JANUS seeds, paired against the corresponding baseline mean); the
pairing unit is the function identity, so this is a cross-function comparison of aggregated final errors
rather than a within-function seed test.

![Vector artwork, page 18, region 1](K:/source/OptunaSharp/docs/papers/paper-gateway/library/5204e8c7ed00c503/figures/p0018-figure-01.png)

Host NN-BBO / MetaBBO baselines Ours

Func CMA-ES RS GLHF SYMBOL LES RLEPSO GLEET LDE SynCMA DiBO JANUS-GN JANUS

f4 4.2e4 3.7e5 4.7e5 1.2e5 2.4e5 2.5e4 3.1e4 1.6e4 3.4e4 2.1e5 1.6e4 5992
f6 1.8e6 1.6e7 1.8e7 5.6e6 9.3e6 9.2e5 1.5e6 9.9e5 5.8e5 9.8e6 7103 2113
f7 1.5e4 3.9e4 4.7e4 1.8e4 2.3e4 1.2e4 1.2e4 4824 1.5e4 3.5e4 – 988
f8 5.9e8 5.9e8 7.1e8 4.7e7 5.1e7 1.5e7 3.1e7 1.3e7 9.5e7 5.1e8 3072 1426
f9 6.1e8 4.1e8 4.7e8 2.4e6 4543 9.5e5 4.4e6 1.8e5 7.0e6 4.6e8 508.7 1323
f10 3.3e7 2.8e8 3.7e8 8.9e7 2.1e8 1.8e7 2.5e7 1.1e7 4.1e7 2.2e8 1799 1.5e6
f11 1.2e4 5833 9115 3099 5136 2715 2746 3141 3437 6575 1934 4037
f12 8.2e9 3.0e10 4.8e10 4.8e9 1.1e10 1.9e9 2.5e9 1.5e9 5.5e9 1.9e10 1.8e6 1.0e4
f13 1.1e4 1.5e4 1.5e4 8913 1.3e4 5772 6709 5347 9840 1.3e4 4728 88.6
f14 267 655 838 176 1458 82.5 112 73.0 252 633 1.5e5 0.12
f18 73.9 140 171 70.3 371 54.7 57.9 40.6 69.5 115 157.6 14.5
f19 3006 2096 2488 24.2 2500 22.4 39.9 12.3 45.3 2392 0.34 11.7
f20 – – – – – – – – – – 16.9 17.3
f22 84.1 86.4 86.5 81.2 1184 63.6 69.1 62.2 75.5 86.0 11.6 2.87
f23 1.65 1.64 1.85 1.46 1202 1.65 1.50 1.64 1.64 1.65 11.1 1.66
f24 2.3e4 2.1e4 2.2e4 9315 8878 7426 8514 7660 9434 2.1e4 7448 7435

Wins 0 0 0 1 0 1 0 0 0 0 5 9
Sig. vs JANUS 15/1/0 14/2/0 – – – – – – – – 0/16/0 –

the same locality boundary from a cleaner angle: at 500D, JANUS is behind or tied with the rerun
CMA-ES host on f22/f23.

With the tuned fuse cadence, low-dimensional CMIX on the pyCMA host is competitive with JANUS-
GN on per-function wins (11/16 at 30D). Without tuning, CMIX can overweight a noisy inverse
Hessian estimate before the recent trace defines stable local directions; GN-only injection is then
the safer fallback. ABOM uses a wider BBOB box and has no public source code, so ABOM
ratios are reported-reference comparisons under an explicitly stated JANUS protocol, not same-code
reproduction claims. The qualitative lesson is still useful: a local Jacobian can describe the current
basin, but it cannot point to a disconnected one. These cases motivate restarts, host-level diversity, or
conservative GN-only injection instead of simply increasing CMIX.

Practical implication. The BBOB appendix supports JANUS (CMIX+GN, tuned fuse cadence)
as the scalar default across d ∈\{30, 100, 500\}; fall back to JANUS-GN when the trace is noisy,
low-dimensional with unstable curvature, or visibly multimodal. This is also why we report both
JANUS-GN and JANUS throughout the scalar tables rather than presenting only the full method.

Full d=500 NN-BBO/MetaBBO comparison. The main paper’s Table 3 reports a compact 4-
baseline slice (RS, SYMBOL, RLEPSO, LDE) of the same-protocol d=500 comparison for space.
Table 12 gives the full 10-baseline comparison, including GLHF, LES, GLEET, SynCMA, and
DiBO. We use “NN-BBO/MetaBBO” as an umbrella label for these learned or configured optimizers,
but they differ in mechanism: Random Search is a non-learning reference; RLEPSO and GLEET
are reinforcement-learning-trained policies; LDE and GLHF are meta-trained/learned update rules;
SYMBOL is a symbolically learned optimizer; LES is a learned evolution strategy; DiBO is a
diffusion/posterior-based method; and SynCMA is a configured CMA variant. They are grouped only
because all are trained or configured off-line and then run on the held-out functions without further
adaptation; the comparison is same-protocol but not a claim that these methods share a training or
generalization setting.

d=1000 extension. To check whether the 500D trend continues at higher dimension, we evaluate
JANUS-GN against the CMA-ES host on a five-function representative BBOB subset at d=1000,
using the same \[−5, 5\]d box, 20,000-FE budget, and 30 seeds as the main protocol. JANUS improves
over the host on all five functions, reducing the geometric-mean final error from 1.19 × 109 to
1.27 × 106, a 936× improvement. This is consistent with the 30D→500D trend in Table 11: as

18

<details>
<summary>Page 18: complete visual reference</summary>

![Full source page 18](K:/source/OptunaSharp/docs/papers/paper-gateway/library/5204e8c7ed00c503/pages/page-0018.png)

</details>

> Quality note: Mathematical notation or superscripts require visual review; linear text does not establish fraction, exponent or equation semantics.

<a id="page-19"></a>

<!-- Page 19 -->

ambient dimension grows, host covariance adaptation from rank-based selection alone becomes
harder, while the residual-Jacobian estimate remains a local, low-cost signal that does not need to
scale with d in the same way.

B.2 MOO SUPPLEMENTARY RESULTS

Takeaway. The MOO gains are regime-dependent rather than uniform. Unlike scalar BBOB, MOO
provides natural objective vectors, so the residual coordinates are not synthetic. JANUS-SMS is
strongest in low-objective hypervolume selection, while JANUS-AGE2 is strongest in the many-
objective setting where AGE-style survival benefits from metric-guided candidates.

Low-objective rank by \#obj Many-objective rank by \#obj

8

JANUS-AGE2 5% 1.7 2.1 2.1

JANUS-SMS 5% 1.8 2.5

7

JANUS-AGE2 2% 2.0 1.9 3.0

JANUS-SMS 2% 2.3 2.8 6

AGE-MOEA2 4.0 3.9 3.1

5

SMS-EMOA 3.2 3.0 NSGA-III 6.2 4.4 4.6

4

RVEA 6.7 6.3 6.3

NSGA-II 3.5 3.8 3

Avg. rank (lower = better)

NSGA-II 5.5 7.4 6.4

2

MOEA/D 4.4 6.2

SMS-EMOA 6.7 7.0 7.3 1

2 3 5 8 10
\# objectives \# objectives

![Figure 4: MOO ranks by objective count. JANUS-SMS leads low-objective tasks; JANUS-AGE2 leads many-objective tasks.](K:/source/OptunaSharp/docs/papers/paper-gateway/library/5204e8c7ed00c503/figures/p0019-figure-01.png)

Figure 4: MOO ranks by objective count. JANUS-SMS leads low-objective tasks; JANUS-AGE2
leads many-objective tasks.

Readout. JANUS-SMS leads on 2-objective tasks, where hypervolume contribution is most dis-
criminative and a small number of geometry-guided candidates can quickly improve the dominated
region. JANUS-AGE2 leads at 5 and 8 objectives, where many-objective survival has more difficulty
preserving useful search directions from ranking alone. Table 13 reports the corresponding grouped
leaderboard. At 10 objectives the gap narrows because AGE-MOEA2 survival is already strong,
leaving less room for proposal-side geometry to change the final ranking.

Table 13: MOO performance by objective count. Top three methods are shown for each group; JANUS
variants lead the low-objective SMS regime and the many-objective AGE2 regime. Blue cell=best,
green cell=second-best per group.

![Vector artwork, page 19, region 2](K:/source/OptunaSharp/docs/papers/paper-gateway/library/5204e8c7ed00c503/figures/p0019-figure-02.png)

Group Method Tasks Avg. rank Wins Norm. HV

Low-objective tasks
2 JANUS-SMS 5% 12 1.75 8 1.000
JANUS-SMS 2% 12 2.33 1 0.996
SMS-EMOA 12 3.17 0 0.974
3 JANUS-SMS 5% 6 2.50 2 0.982
JANUS-SMS 2% 6 2.83 3 0.986
SMS-EMOA 6 3.00 0 0.967

Many-objective tasks
5 JANUS-AGE2 5% 6 1.67 2 0.997
JANUS-AGE2 2% 6 2.00 2 0.983
AGE-MOEA2 6 4.00 1 0.954
8 JANUS-AGE2 2% 7 1.86 3 0.975
JANUS-AGE2 5% 7 2.14 2 0.959
AGE-MOEA2 7 3.86 0 0.928
10 JANUS-AGE2 5% 7 2.14 2 0.965
JANUS-AGE2 2% 7 3.00 1 0.963
AGE-MOEA2 7 3.14 2 0.869

Ablation readout. The five-arm ablation is reported in the main paper. Here we keep the broader
rank and runtime panels, which show that the zero-regression pattern and modest CUDA overhead
hold beyond the compact main-table summary.

19

<details>
<summary>Page 19: complete visual reference</summary>

![Full source page 19](K:/source/OptunaSharp/docs/papers/paper-gateway/library/5204e8c7ed00c503/pages/page-0019.png)

</details>

> Quality note: Rotated or vertical text retains geometry and requires visual order review.
> Quality note: Mathematical notation or superscripts require visual review; linear text does not establish fraction, exponent or equation semantics.

<a id="page-20"></a>

<!-- Page 20 -->

A. JANUS vs JANUS-GN B. Significance test C. Runtime (CUDA)

2.50 2.00

16 0 regressions (−)Signif. +
2.25 Tied 1.75

14

2.00 1.50 1.46x

12

1.75 1.25 1.22x

10

no overhead

1.50 1.00

8

\# tasks

1.25 6 0.75

parity

Slowdown vs host

JANUS / JANUS-GN

1.00 4 0.50

0.75 2 0.25

0.50 0 0.00
30D 100D 500D Low Many J-SMS J-AGE2
(J-SMS) (J-AGE2) 5% 5%

![Figure 5: Ablation, significance, and cost. (A) JANUS/JANUS-GN ratio. (B) Significance: 0 regressions. (C) CUDA runtime: 1.46×/1.22×.](K:/source/OptunaSharp/docs/papers/paper-gateway/library/5204e8c7ed00c503/figures/p0020-figure-01.png)

Figure 5: Ablation, significance, and cost. (A) JANUS/JANUS-GN ratio. (B) Significance: 0
regressions. (C) CUDA runtime: 1.46×/1.22×.

CMIX fraction and runtime. Table 14 sweeps the fraction of offspring allocated to CMIX infill
from 0.05 to 0.60. The sweep shows that the best fixed fraction is often larger than the conservative
default, but the curve is not monotone on every problem. We therefore use a conservative fraction
in the main experiments rather than tuning it per benchmark. Table 15 reports wall-clock overhead:
with CUDA-accelerated metric estimation, JANUS-SMS 5% is 1.46× slower than SMS-EMOA and
JANUS-AGE2 5% is 1.22× slower than AGE-MOEA2. This overhead is the price of online Jacobian
estimation; it is acceptable when evaluations dominate runtime, but may matter for benchmarks with
sub-millisecond evaluation cost.
Table 14: CMIX candidate fraction ablation (mean HV; higher is better). Blue cell=best per row,
green cell=second-best.

Problem 0.05 0.10 0.20 0.30 0.40 0.50 0.60

DTLZ4 d12 n3 72.7 72.9 72.9 72.8 72.9 72.9 72.9
WFG1 d12 n2 8.10 9.23 9.64 9.50 9.63 9.51 9.86
WFG2 d12 n3 117 119 121 124 131 127 128
WFG4 d12 n2 7.02 7.51 7.47 7.99 7.71 8.07 8.74
ZDT1 d30 n2 6.70 6.75 6.81 6.94 7.09 7.07 7.10
ZDT2 d30 n2 5.85 5.90 5.88 5.91 6.12 6.29 6.27

ALL 36.2 36.9 37.3 37.8 39.1 38.4 38.9

![Table 15: Runtime overhead with CUDA-accelerated metric estimation. Slowdown is the ratio of mean wall-clock time per independent run for the JANUS variant to its host baseline (lower is better; 1.00× = no overhead). Values match Fig. 5C.](K:/source/OptunaSharp/docs/papers/paper-gateway/library/5204e8c7ed00c503/figures/p0020-figure-02.png)

Table 15: Runtime overhead with CUDA-accelerated metric estimation. Slowdown is the ratio of
mean wall-clock time per independent run for the JANUS variant to its host baseline (lower is better;
1.00× = no overhead). Values match Fig. 5C.

JANUS variant Host baseline Tasks Runs Slowdown

JANUS-SMS 5% SMS-EMOA 18 180 1.46×
JANUS-AGE2 5% AGE-MOEA2 20 200 1.22×

Per-task winner summary. Across all 38 MOO tasks, JANUS variants win 16/38 (10/18 low-
objective and 6/20 many-objective) by best average rank, not gated by a significance test; the main
paper’s headline “12/38 significant wins” is the subset of these that additionally passes a Mann–
Whitney test at p &lt; 0.05 against the strongest host baseline. The few non-JANUS wins concentrate
on simple ZDT instances with disconnected Pareto fronts and on 10-objective tasks where AGE-
MOEA2’s survival is already highly competitive. This distribution is useful for interpreting the
method: JANUS is not a universal dominance trick, but an infill mechanism that helps when the host
leaves exploitable local geometry unused. Practical implication. The MOO appendix shows that
JANUS is not uniformly better by construction; its gains appear where the host survival operator
leaves room for geometry-guided proposal generation. In practice, this suggests using JANUS as a
host augmentation and monitoring whether the host is already saturating diversity preservation before
increasing the CMIX fraction.

B.3 UAV PER-TERRAIN RESULTS

Takeaway. UAV is the clearest structured-objective case because the objective decomposes into
meaningful path-quality components and the dimensionality is high enough that blind covariance

20

<details>
<summary>Page 20: complete visual reference</summary>

![Full source page 20](K:/source/OptunaSharp/docs/papers/paper-gateway/library/5204e8c7ed00c503/pages/page-0020.png)

</details>

> Quality note: Rotated or vertical text retains geometry and requires visual order review.
> Quality note: Mathematical notation or superscripts require visual review; linear text does not establish fraction, exponent or equation semantics.

<a id="page-21"></a>

<!-- Page 21 -->

learning is difficult. JANUS has the best aggregate cost and wins many terrains, but the terrain panel
also exposes real failures under the short 2,500-FE budget.

Table 16: UAV budget sweep over 28 terrains and 30 runs. Lower is better.

Budget JANUS CMA-ES Gain

2,500 9,600 14,486 −33.7%
10,000 8,488 8,587 −1.2%
20,000 8,019 8,023 −0.05%

![Table 17: UAV path planning: 28 held-out terrains, 30 runs, 2,500 FEs (840 values/method). Blue cell=best, green cell=second-best. Ratio is mean/JANUS (lower=closer to JANUS). Sig. column: Bonferroni-corrected Mann–Whitney U wins/ties/losses vs JANUS (p < 0.05/28 across 28 terrains). The test is unpaired because the CMA-ES baseline runs do not share seeds with the JANUS runs.](K:/source/OptunaSharp/docs/papers/paper-gateway/library/5204e8c7ed00c503/figures/p0021-figure-01.png)

Table 17: UAV path planning: 28 held-out terrains, 30 runs, 2,500 FEs (840 values/method). Blue
cell=best, green cell=second-best. Ratio is mean/JANUS (lower=closer to JANUS). Sig. column:
Bonferroni-corrected Mann–Whitney U wins/ties/losses vs JANUS (p &lt; 0.05/28 across 28 terrains).
The test is unpaired because the CMA-ES baseline runs do not share seeds with the JANUS runs.

![Vector artwork, page 21, region 2](K:/source/OptunaSharp/docs/papers/paper-gateway/library/5204e8c7ed00c503/figures/p0021-figure-02.png)

Mean Ratio Terrains Sig. vs
\# Method Family Median IQR
±Std vs JANUS rank 1 JANUS

9,600
1 JANUS Ours ±5,966 6,644 9,579 1.00 15 –

11,006
2 DE Traditional ±3,865 9,261 7,412 1.15 1 24/3/1

11,017
3 JDE21 Adaptive 8,579 2,795 1.15 1 –
±5,604

11,317
4 JANUS-GN Ablation 9,508 9,815 1.18 8 0/28/0
±5,373

11,411
5 RLDEAFL MetaBBO 8,197 9,227 1.19 0 –
±5,818

11,603
6 GLEET MetaBBO ±7,085 7,455 9,808 1.21 1 –

11,792
7 PSO Traditional ±4,454 9,156 8,223 1.23 0 –

13,891
8 RS Traditional 10,394 9,169 1.45 0 23/5/0
±6,859

14,486
9 CMA-ES Adaptive 10,908 9,956 1.51 0 24/3/1
±6,559

14,865
10 SAHLPSO Adaptive ±6,885 11,265 9,716 1.55 0 –

20,354
11 GLHF MetaBBO ±5,918 20,921 7,137 2.12 0 –

26,968
12 LES MetaBBO 26,220 20,365 2.81 2 –
±14,006

Readout. JANUS is most useful in the early-budget regime. At 2,500 evaluations, the local metric
gives the host a faster route toward feasible, low-cost path shapes, as also reflected in the final-value
distribution in Fig. 6. At longer budgets, CMA-ES catches up and the advantage converges toward
parity, so JANUS should be interpreted as an early-budget accelerator rather than an asymptotic
replacement. This is the regime we care about most for expensive simulation-driven planning.

UAV final-value distributions (840 runs)

LES

GLHF

SAHLPSO

CMAES

RS

JANUS-GN

DE

PSO

JDE21

RLDEAFL

GLEET

JANUS

4
10

Final path cost

![Figure 6: UAV final-value distributions over 840 runs.](K:/source/OptunaSharp/docs/papers/paper-gateway/library/5204e8c7ed00c503/figures/p0021-figure-03.png)

Figure 6: UAV final-value distributions over 840 runs.

Readout. JANUS has the best center of mass, while several baselines occasionally find strong terrain-
specific paths. The distribution plot is therefore not meant to claim pointwise dominance; it shows that
JANUS shifts the bulk of outcomes downward while still leaving a nontrivial tail of terrain-specific
failures. This motivates the terrain-level audit rather than relying only on aggregate averages.

21

<details>
<summary>Page 21: complete visual reference</summary>

![Full source page 21](K:/source/OptunaSharp/docs/papers/paper-gateway/library/5204e8c7ed00c503/pages/page-0021.png)

</details>

> Quality note: Mathematical notation or superscripts require visual review; linear text does not establish fraction, exponent or equation semantics.

<a id="page-22"></a>

<!-- Page 22 -->

![Complete page reference; figure boundary uncertain. Full source page; figure boundaries require review.](K:/source/OptunaSharp/docs/papers/paper-gateway/library/5204e8c7ed00c503/pages/page-0022.png)

3D path visualizations. The main paper now includes the representative T1/T17 convergence
and path panels. Figure 7 therefore keeps only additional terrain-level path examples: T9/T41 are
hard cases where non-JANUS baselines can find safer early paths, while T25/T55 are representative
JANUS wins. This avoids duplicating the main figure while keeping the visual evidence aligned with
the terrain-level audit.

Threats Threats Threats Threats
Start Start Start Start
Goal Goal Goal Goal
JANUS (16209) JANUS (7046) JANUS (5539) JANUS (6444)
GN-CMA (16840) GN-CMA (7097) GN-CMA (5607) GN-CMA (6524)
DE (17102) DE (10231) DE (8648) DE (7616)
JDE21 (18304) JDE21 (8693) JDE21 (7168) JDE21 (7758)
CMAES (17475) CMAES (10224) CMAES (7310) CMAES (8736)

35 35 30 50
30 30 25 40
25 25 20 30
20 15
20 15 20
15 100 10 10
100 50 50 0
50 0 0 −10
0 −50 −50 −20

900 900 900 900

800 800 800 800

700 700 700 700

600 600 600 600

100 500 100 500 100 500 100 500

200 300 400 300400 y 200 300 400 300400 y 200 300 400 300400 y 200 300 400 300400 y

500 500 500 500

x 600 700 100200 x 600 700 100200 x 600 700 100200 x 600 700 100200
800 800 800 800

(a) T9: hard (b) T25: JANUS win (c) T41: hard (d) T55: JANUS win
Figure 7: Additional UAV 3D path visualizations on four terrains (T9/T25/T41/T55). JANUS often
finds smooth, low-cost trajectories, but T9/T41 show that this behavior is not universal under the
short 2,500-FE budget.

Terrain audit. Table 18 is intentionally a compact audit rather than a visual centerpiece: it keeps
every held-out terrain visible and makes the significant wins, ties, and the one significant loss against
the CMA-ES host easy to locate. We keep the non-wins in the same table as the wins because they
explain when CMIX should be used cautiously: the one true failure occurs where CMIX’s covariance
reshaping actively undoes a strong exploitation signal.

Table 18: UAV terrain audit. Every held-out terrain is kept; gain compares JANUS with the
CMA-ES host under the identical 2,500-FE protocol. Cell color reflects statistical significance
(Bonferroni-corrected Mann–Whitney U across 28 terrains, consistent with Table 17’s Sig. column):
green=significant win, amber=not significant (tie), red=significant loss.

Terrain JANUS CMA-ES Gain Terrain JANUS CMA-ES Gain

T1 7035 10062 +30.1% T29 15998 10607 -50.8%
T3 9716 17753 +45.3% T31 6044 18533 +67.4%
T5 5974 19695 +69.7% T33 7696 10066 +23.5%
T7 5834 18681 +68.8% T35 5304 7547 +29.7%
T9 23155 19402 -19.3% T37 10520 18362 +42.7%
T11 21077 18970 -11.1% T39 10395 17266 +39.8%
T13 7527 10261 +26.6% T41 6621 9257 +28.5%
T15 6528 8369 +22.0% T43 6119 16738 +63.4%
T17 6584 20170 +67.4% T45 5975 8912 +32.9%
T19 22186 19407 -14.3% T47 8285 9291 +10.8%
T21 7515 10439 +28.0% T49 5690 8906 +36.1%
T23 5807 8322 +30.2% T51 9329 11420 +18.3%
T25 11623 18375 +36.7% T53 7572 10028 +24.5%
T27 15966 27321 +41.6% T55 6728 21444 +68.6%

Readout. Under this host-controlled comparison (optimization budget, terrain set, and number of
runs matched across methods as in the BBOB and MOO tables; the statistical test is the Bonferroni-
corrected unpaired Mann–Whitney U across 28 terrains, chosen because these runs are not seed-
paired), JANUS significantly beats the CMA-ES host on 24/28 terrains, ties on 3 (T9, T11, T19; the
raw mean favors CMA-ES but the difference is not significant), and loses significantly on exactly 1
(T29). This matches Table 17’s aggregate Sig. column for CMA-ES (24/3/1) exactly. On all 4 of these
non-win terrains, JANUS-GN (exploitation only) ranks first among all 12 compared methods and
clearly beats CMA-ES, while the full JANUS variant falls to 5th–9th place; the GN signal itself is
strong there, but CMIX’s covariance reshaping is harmful, most severely on T29 (JANUS-GN 6,063
vs. full JANUS 15,998, CMA-ES 10,607, p &lt; 10−5). This is consistent with JANUS-GN ranking
first on 8 terrains overall in the main paper’s “ranks first on 15/28” count (computed against all 12
methods): these 4 terrains are among the 8, confirming that CMIX should be used cautiously when the
recent trace has not yet produced a stable curvature estimate. This distinction matters operationally:
GN injection is a safer fallback when the path distribution is still changing rapidly, while CMIX is
most useful after the recent window contains coherent path variations.

22

<details>
<summary>Page 22: complete visual reference</summary>

![Full source page 22](K:/source/OptunaSharp/docs/papers/paper-gateway/library/5204e8c7ed00c503/pages/page-0022.png)

</details>

> Quality note: Mathematical notation or superscripts require visual review; linear text does not establish fraction, exponent or equation semantics.

<a id="page-23"></a>

<!-- Page 23 -->

## C NEGATIVE RESULTS AND OPERATING BOUNDARIES

Operating regime. Sec. 5.5 summarizes the operating regime found across experiments; we do not
repeat it here. The mechanism-level detail worth adding is that, in the high-dimensional BBOB and
UAV regimes where JANUS helps most, CMIX is not simply adding more candidates: it reallocates
the same sampling scale toward directions that the fitted residual model regards as locally flat or
under-explored.

Takeaway. JANUS is a local-geometry method. The inverse Hessian metric T = (G + λI)−1
captures the landscape near the current mean, but cannot reveal disconnected basins outside the
reachable radius or outside the trust region of the local Jacobian estimate. The failure cases summa-
rized in Table 19 are therefore expected rather than incidental. JANUS is strongest when the recent
trace gives a reliable metric, while JANUS-GN is safer when basin discovery, unstable curvature, or
noisy pseudo-residuals dominate. These negative cases are included to make the method’s operating
boundary explicit.

Table 19: Negative evidence summary: cases where JANUS underperforms or ties.

Regime Evidence Interpretation

BBOB 30D/100D CMIX loses more often than JANUS- Scalar pseudo-residual curvature can be noisy in low dimension.
GN

BBOB 500D Host-control losses on f22/f23 Local metric helps broadly but not when basin discovery domi-
nates.
ABOM reference f19/f20 anomalous ratios Reported-reference comparisons are interpreted qualitatively.
UAV short budget 1 significant host-control loss (T29), 3 CMIX can undo a strong GN signal before curvature stabilizes.
ties (T9/11/19); JANUS-GN ranks first
on all 4

MOO saturation 4 tied tasks Survival already preserves diversity, limiting proposal-geometry
gains.

How to read the failures. The BBOB losses do not show that the GN candidate is harmful in
general. They show that scalar pseudo-residual curvature can be misleading when the useful move
is global, not local. The host-controlled comparison points mainly to f22/f23; the ABOM reported-
reference ratios add a few anomalous cases that we treat qualitatively instead of as same-protocol
failures.

The UAV losses have a different cause. The residual structure is meaningful, but the 2,500-FE budget
can be too short for a stable curvature estimate. The MOO ties occur when the host survival operator
already preserves enough diversity, leaving little room for proposal-side geometry. These failure
modes call for different responses: restarts for disconnected basins, GN-only fallback for unstable
curvature, and conservative CMIX fractions when the host is already strong.

Practical use pattern. A practical use pattern follows from this boundary. For scalar objectives, use
JANUS-GN or a small CMIX fraction until the recent window produces stable curvature directions.
Increase CMIX only when the host is spending many evaluations in a locally coherent region. For
structured objectives or multi-objective tasks, where residual coordinates are observed rather than
fitted, CMIX can be enabled earlier. In all cases, the host should keep ownership of restarts, boundary
handling, and diversity preservation. JANUS is intentionally not a trust-region replacement and not a
learned optimizer: it is a small, local, bounded proposal mechanism whose value is highest when
evaluations are expensive enough that reusing local geometry matters.

C.1 INDUSTRIAL VALIDATION: ISP CALIBRATION (d=1135)

To complement the synthetic benchmarks, we report a single deployment on a proprietary image
signal processing (ISP) calibration task. The objective is the mean reproduction angle error (MRAE)
of an auto white balance (AWB) module across m=111 test scenes; the search space is d=1135 ISP
parameters. Each evaluation runs the full ISP pipeline on all scenes, making repeated large-scale
experimentation infeasible. We therefore report a single run per method under a fixed 5,000-evaluation
budget.

23

<details>
<summary>Page 23: complete visual reference</summary>

![Full source page 23](K:/source/OptunaSharp/docs/papers/paper-gateway/library/5204e8c7ed00c503/pages/page-0023.png)

</details>

> Quality note: Mathematical notation or superscripts require visual review; linear text does not establish fraction, exponent or equation semantics.

<a id="page-24"></a>

<!-- Page 24 -->

Table 20: ISP calibration (AWB, d=1135, m=111 scenes): running-best MRAE. Lower is better.
Single run per method, 5,000 evaluations. Blue cell=best, green cell=second-best per column.

Method @1000 @2000 @3000 @4000 @5000

CMA-ES (baseline) 1.529 1.276 1.177 1.127 1.056
JANUS-GN (exploit only) 1.154 0.968 0.918 0.866 0.837
JANUS (GN + CMIX) 1.200 0.993 0.897 0.863 0.819

Readout. Both JANUS variants reduce MRAE by ∼21–22% relative to CMA-ES at @5000. The full
JANUS variant (GN+CMIX) trails the GN-only variant early (@1000) because it allocates sampling
energy to exploration, but overtakes it from @3000 onward and achieves the lowest final error (0.819
vs. 0.837). The CMIX advantage continues to grow at @5000 with no sign of reversal, consistent
with the delayed benefit of CMIX observed in the controlled ablation (Sec. 5.4). Because this is a
single run on a proprietary pipeline, we treat it as deployment evidence rather than a statistically
controlled comparison.

C.2 DERIVATION NOTES

Residual coordinates to GN. For component observations, JANUS starts from the form f(x) =p
m−1 Pi ri(x) and defines si(x) = ri(x), so f(x) = m−1∥s(x)∥2. This is the standard least-
squares view behind Gauss–Newton. Scalar-only objectives do not expose true components, but
JANUS can still construct a local residual coordinate system by fitting linear or quadratic models to
the recent trace. The resulting residual map is only local, but that is sufficient for an infill module
whose candidates are clipped to the host sampling region. With J = ∂s/∂x, the local first- and
second-order approximations are

∇f(x) ≈2mJ⊤s(x), HGN(x) ≈2mJ⊤J.

Ignoring the common factor 2/m and adding Levenberg–Marquardt damping gives the injected
exploitation candidate
(J⊤J + λI)∆x = −J⊤s.
Thus explicit residual components are useful but not required: scalar JANUS uses the same algebra
with a fitted local residual chart.

Black-box Jacobian estimation. JANUS estimates J only from the recent search trace. Let XW be
the current window, let Ur be the local PCA basis, and write each centered sample as z = U ⊤r (x −¯x).
For residual observations S and standardized design matrix Z, ridge regression gives
ˆB = (Z⊤Z + ρI)−1Z⊤S,

where columns of ˆB are residual slopes in PCA coordinates. Mapping back through Ur gives the
ambient Jacobian estimate used by both GN and CMIX. The PCA restriction is not a modeling
assumption about the true objective; it is a variance-control device that prevents underdetermined
high-dimensional regressions from dominating the host optimizer.

CMIX metric and dimension scaling. CMIX forms the inverse Hessian metric T = (J⊤J+λI)−1,
normalizes its trace to match the host covariance scale, and mixes it with the host covariance only
through a bounded fraction. A fixed mixing weight is too small in very high dimension and too
aggressive in low dimension, so JANUS uses

min(m, \|W\|)
w = wbase , w ∈\[wmin, wmax\].
d

For UAV, m = 111, \|W\| = 300, and d = 1135, yielding w ≈0.049. This explains why the UAV
experiments require a dimension-aware CMIX weight: a fixed 3 × 10−6-style fraction would make
the metric contribution effectively invisible.

Proof of Proposition 1. Write the SVD J = UΣV ⊤with Σ = diag(σ1, . . . , σr), r = rank(J). The
inverse Hessian metric is T = (J⊤J+λI)−1 = V (Σ⊤Σ+λI)−1V ⊤, so each right-singular direction
vi is an eigenvector of T with eigenvalue (σ2i + λ)−1, and directions in ker(J) are eigenvectors
with eigenvalue λ−1. Since every eigenvalue satisfies (σ2i + λ)−1 ≥(σmax(J)2 + λ)−1 and λ−1 ≥

24

<details>
<summary>Page 24: complete visual reference</summary>

![Full source page 24](K:/source/OptunaSharp/docs/papers/paper-gateway/library/5204e8c7ed00c503/pages/page-0024.png)

</details>

> Quality note: Mathematical notation or superscripts require visual review; linear text does not establish fraction, exponent or equation semantics.

<a id="page-25"></a>

<!-- Page 25 -->

(σmax(J)2 + λ)−1, the metric is uniformly positive definite:

1
T ⪰ (5)
σmax(J)2 + λ I.

The restricted covariance annihilates span(B)⊥from both sides. Any restricted-covariance up-
date C′ = (1 −α)Chost + αBM ′B⊤with Chost = BMB⊤can be written C′ = B (1 −
α)M + αM ′B⊤=: B fMB⊤, whose range and corange lie in span(B). With B having or-
thonormal columns, P⊥= I −BB⊤is the orthogonal projector onto span(B)⊥and B⊤P⊥=
B⊤−(B⊤B)B⊤= 0. Hence

P⊥C′P⊥= (P⊥B)fM(B⊤P⊥) = 0 (6)
regardless of M, M ′, and α: the restricted family has no degrees of freedom in the complement of
its own basis. Note that this argument uses only that C′ is spanned by B; it does not require any
right-singular vector of J to lie exactly in span(B)⊥.

A projection-based Frobenius bound. Since P⊥is an orthogonal projector, the map A 7→P⊥AP⊥
is a Frobenius-norm contraction (∥P⊥AP⊥∥F ≤∥A∥F ). Applying it to A = C′ −T and using
equation 6,
∥C′ −T∥F ≥∥P⊥(C′ −T)P⊥∥F = ∥P⊥TP⊥∥F . (7)
It remains to lower-bound ∥P⊥TP⊥∥F . Because T is symmetric, congruence by P⊥preserves the
Loewner order, so equation 5 gives P⊥TP⊥⪰(σmax(J)2 + λ)−1P⊥. For positive semidefinite
matrices X ⪰Y ⪰0 we have ∥X∥F ≥∥Y ∥F (as ∥X∥2p √ F −∥Y ∥2F = ⟨X −Y, X + Y ⟩≥0), and
∥P⊥∥F = rank(P⊥) = d −k. Therefore

√
∥P⊥∥F d −k
∥C′ −T∥F ≥∥P⊥TP⊥∥F ≥ (8)
σmax(J)2 + λ = σmax(J)2 + λ &gt; 0,

strictly positive because k &lt; d. In particular C′ ̸= T. The bound has the correct units of an inverse
curvature (1/σ2), scales with the dimension deficit d −k of the restricted basis, and does not depend
on M, M ′, α. The middle quantity ∥P⊥TP⊥∥F is the exact residual-aligned curvature of T restricted
to span(B)⊥, which the host sets identically to zero. □

Remark. The proposition does not claim that LM-CMA or VkD-CMA underperform JANUS in
absolute terms; their host adaptation may be sufficient when the trace itself concentrates in span(B).
The claim is the structural one stated in the introduction: restricted-covariance hosts cannot, in general,
express the residual-aligned curvature that JANUS derives from J. The empirical question of when
this gap matters is taken up in Sec. 5.

Scope: purely rank-k hosts. Prop. 1 assumes a purely rank-k covariance Chost = BMB⊤with k &lt; d:
the annihilation identity equation 6 relies on C′ having range and corange inside span(B). If a
restricted-covariance implementation adds an isotropic floor ϵI or a full-rank diagonal term (as some
practical LM-CMA/VkD-CMA variants do for numerical stability), then C′ no longer annihilates
span(B)⊥and the exact-mismatch statement weakens to an O(ϵ)-approximate one; the proposition
should be read as governing the rank-k low-rank correction, not every implementation detail. The
hypothesis also fails for a full-rank host such as vanilla CMA-ES, where span(B) = Rd, P⊥= 0,
and the bound is vacuously zero; there the proposition makes no structural claim. This distinction
separates two questions that should not be conflated. Table 9 addresses the structural one—whether a
restricted-covariance host can match a full-rank host at all—and answers it in the negative, consistent
with Prop. 1. Whether JANUS additionally improves on a full-rank host is a separate, empirical
convergence-speed question, and is not structural: learning a general full covariance from rank-based
updates involves O(d2) covariance parameters and can therefore become sample- and computation-
intensive in high dimensions (Hansen, 2016), whereas JANUS computes T = (J⊤J + λI)−1 from
the fitted J in one step and mixes it in immediately. That convergence-speed comparison is quantified
separately, on real 30-seed data, in the DFO panel of Table 8.

Propositions restated with full proofs. Propositions 5–7 restate Propositions 2 and 3 from the
main paper (the latter split into two parts) so that complete proofs can be given below. Proposition 4
is the only new result in this section.
Proposition 4 (Blockwise assembly is a degenerate GN case). Partition coordinates into groups and
constrain the fitted normal matrix J⊤J to be block diagonal under this partition. Then the damped

25

<details>
<summary>Page 25: complete visual reference</summary>

![Full source page 25](K:/source/OptunaSharp/docs/papers/paper-gateway/library/5204e8c7ed00c503/pages/page-0025.png)

</details>

> Quality note: Mathematical notation or superscripts require visual review; linear text does not establish fraction, exponent or equation semantics.

<a id="page-26"></a>

<!-- Page 26 -->

GN solve (J⊤J + λI)∆= −J⊤s decomposes into independent block solves. Per-component
best-fragment assembly is recovered as a zero-coupling, zero-damping, discrete-search limit of this
blockwise update.

Proposition 5 (Damped GN descends on the fitted residual model). For ˆs(xb + ∆) = sb + J∆and
ˆf(∆) = ∥sb + J∆∥22, the JANUS step ∆GN = −(J⊤J + λI)−1J⊤sb satisfies ∇ˆf(0)⊤∆GN ≤0
for λ &gt; 0, strictly whenever J⊤sb ̸= 0.

Proposition 6 (Bounded injection controls normalized step length). If an injected point is clipped in√ √
host-whitened coordinates to ∥ye∥≤κ d, then ∥xe −¯x∥≤κσλmax(C1/2) d.

Proposition 7 (Trace-preserving covariance mix). For T ≻0 and C′ = (1−w)C+w tr(C)T/ tr(T),
tr(C′) = tr(C).

Proof of Proposition 4. Let the coordinate partition be G = \{G1, . . . , Gq\} and suppose the fitted
normal matrix has no cross-block terms: J⊤J = blkdiag(G⊤1 G1, . . . , G⊤q Gq). The damped system
(J⊤J + λI)∆= −J⊤s is then block diagonal, so each ∆Gj depends only on the residual slopes
and residual values assigned to block Gj. If damping is taken to zero and the continuous block
step is restricted to the finite set of historical fragments available for that block, the update reduces
to choosing the fragment that minimizes the corresponding component residual. This is the per-
component assembly rule. The condition is deliberately restrictive: once off-diagonal blocks are kept,
JANUS can express interactions among coordinate groups that assembly discards.

Proof of Proposition 5. For the fitted model ˆf(∆) = ∥sb + J∆∥22, ∇ˆf(0) = 2J⊤sb. Substituting
∆GN = −(J⊤J + λI)−1J⊤sb gives

∇ˆf(0)⊤∆GN = −2(J⊤sb)⊤(J⊤J + λI)−1(J⊤sb).
Because J⊤J + λI ≻0 for λ &gt; 0, the quadratic form is nonnegative, and it is positive unless
J⊤sb = 0. Therefore the GN proposal is a descent direction for the fitted residual objective whenever
the fitted local gradient is nonzero. This is not a global improvement theorem: if the fitted residual
chart is inaccurate outside the window, the true objective can still fail to improve.

Proof of Proposition 6. Host-whitened coordinates are defined by ye = C−1/2(xe −¯x)/σ, equiva-
lently xe −¯x = σC1/2ye. Hence

√
∥xe −¯x∥≤σ∥C1/2∥2∥ye∥≤κσλmax(C1/2) d.

Thus clipping controls the injected step in exactly the coordinate system used by the host distribution.
The proposition does not claim that JANUS leaves the stochastic step-size path unchanged; it claims
the weaker and needed safety property that injected candidates are scale-compatible with ordinary
samples.

Proof of Proposition 7. Let T ′ = tr(C)T/ tr(T). Then tr(T ′) = tr(C) by linearity of trace, and
therefore
tr((1 −w)C + wT ′) = (1 −w) tr(C) + w tr(T ′) = tr(C).
CMIX can rotate or anisotropically reshape the proposal covariance through T ′, but it cannot increase
the total covariance trace. These results are local: T can reallocate variance among plausible nearby
directions, but disconnected basin discovery still requires host diversity, restarts, or another global
exploration mechanism.

Implementation notes. JANUS is implemented as a bounded candidate generator, not as a replace-
ment optimizer. The host proposes its ordinary offspring; JANUS reads the recent evaluated window,
fits the residual chart, and contributes a small number of GN/CMIX candidates. All candidates
are evaluated before the host performs its standard tell/update step. Selection, survival, covariance
adaptation, step-size adaptation, archive maintenance, and restart logic remain owned by the host.

In CMA-style hosts, a proposed displacement is transformed into host-whitened coordinates and√
clipped to the same d scale as ordinary samples. In DE and MOO hosts, we apply analogous box
and step-length constraints before evaluation. This keeps JANUS proposals on the host sampling
scale and prevents one ill-conditioned Jacobian estimate from dominating adaptation.

For scalar objectives, the residual chart is rebuilt frequently from local scalar fits with standardized
design matrices, ridge regularization, and rank caps for the quadratic chart. For UAV, component costs

26

<details>
<summary>Page 26: complete visual reference</summary>

![Full source page 26](K:/source/OptunaSharp/docs/papers/paper-gateway/library/5204e8c7ed00c503/pages/page-0026.png)

</details>

> Quality note: Mathematical notation or superscripts require visual review; linear text does not establish fraction, exponent or equation semantics.

<a id="page-27"></a>

<!-- Page 27 -->

define a more meaningful residual space; for MOO, the objective vector itself provides the residual
coordinates. Changing the host requires only the ask/tell interface, the available residual coordinates,
and the fixed hyperparameters in Table 6. JANUS-GN is the conservative fallback because it uses the
same local Jacobian for one bounded exploitation candidate without reshaping the host covariance.

27

<details>
<summary>Page 27: complete visual reference</summary>

![Full source page 27](K:/source/OptunaSharp/docs/papers/paper-gateway/library/5204e8c7ed00c503/pages/page-0027.png)

</details>

<!-- ============================================================================== -->
<!-- PAPER REFERENCE: NSGA-III (Deb & Jain, Part I)  (source: docs/papers/extracted/NSGA-III.md) -->
<!-- ============================================================================== -->

# An Evolutionary Many-Objective Optimization Algorithm Using Reference-point Based Non-dominated Sorting Approach, Part I: Solving Problems with Box Constraints

Source: [An Evolutionary Many-Objective Optimization Algorithm Using Reference-Point-Based Nondominated Sorting - NSGA-III.pdf](file:///K:/source/OptunaSharp/docs/papers/An%20Evolutionary%20Many-Objective%20Optimization%20Algorithm%20Using%20Reference-Point-Based%20Nondominated%20Sorting%20-%20NSGA-III.pdf)

SHA-256: `4c240f7368799f9dbf77ea85c1c86cc524597291811922b422b31fcb7fc6f7a6`

> Extraction is heuristic. Full-page images preserve drawings, graphs and visual layout; searchable text is extracted separately.

<a id="page-1"></a>

<!-- Page 1 -->

This article has been accepted for publication in a future issue of this journal, but has not been fully edited. Content may change prior to final publication.

1

## An Evolutionary Many-Objective Optimization

## Algorithm Using Reference-point Based

## Non-dominated Sorting Approach, Part I: Solving Problems with Box Constraints

Kalyanmoy Deb, Fellow, IEEE and Himanshu Jain

Abstract—Having developed multi-objective optimization al-
gorithms using evolutionary optimization methods and demon-
strated their niche on various practical problems involving
mostly two and three objectives, there is now a growing need
for developing evolutionary multi-objective optimization (EMO)
algorithms for handling many-objective (having four or more
objectives) optimization problems. In this paper, we recognize a
few recent efforts and discuss a number of viable directions for
developing a potential EMO algorithm for solving many-objective
optimization problems. Thereafter, we suggest a reference-point
based many-objective NSGA-II (we call it NSGA-III) that em-
phasizes population members which are non-dominated yet close
to a set of supplied reference points. The proposed NSGA-III is
applied to a number of many-objective test problems having two
to 15 objectives and compared with two versions of a recently
suggested EMO algorithm (MOEA/D). While each of the two
MOEA/D methods works well on different classes of problems,
the proposed NSGA-III is found to produce satisfactory results
on all problems considered in this study. This paper presents
results on unconstrained problems and the sequel paper considers
constrained and other specialties in handling many-objective
optimization problems.

Index Terms—Many-objective optimization, evolutionary com-
putation, large dimension, NSGA-III, non-dominated sorting,
multi-criterion optimization.

I. INTRODUCTION

Evolutionary multi-objective optimization (EMO) method-
ologies have amply shown their niche in finding a set of
well-converged and well-diversified non-dominated solutions
in different two and three-objective optimization problems
since the beginning of nineties. However, in most real-world
problems involving multiple stake-holders and functionalities,
there often exists many optimization problems that involve
four or more objectives, sometimes demanding to have 10 to
15 objectives \[1\], \[2\]. Thus, it is not surprising that handling a
large number of objectives had been one of the main research
activities in EMO for the past few years. Many-objective

K. Deb is with Department of Electrical and Computer Engineering.
Michigan State University, 428 S. Shaw Lane, 2120 EB, East Lansing, MI
48824, USA, e-mail: kdeb@egr.msu.edu (see http://www.egr.msu.edu/˜kdeb).
Prof. Deb is also a visiting professor at Aalto University School of Business,
Finland and University of Sk¨ovde, Sweden.

H. Jain is with Eaton Corporation, Pune 411014, India, email: himan-
shu.j689@gmail.com.

Copyright (c) 2012 IEEE. Personal use of this material is permitted.
However, permission to use this material for any other purposes must be
obtained from the IEEE by sending a request to pubs-permissions@ieee.org.

problems pose a number of challenges to any optimization
algorithm, including an EMO. First and foremost, the propor-
tion of non-dominated solutions in a randomly chosen set of
objective vectors becomes exponentially large with an increase
of number of objectives. Since the non-dominated solutions
occupy most of the population slots, any elite-preserving
EMO faces a difficulty in accommodating adequate number
of new solutions in the population. This slows down the
search process considerably \[3\], \[4\]. Second, implementation
of a diversity-preservation operator (such as the crowding
distance operator \[5\] or clustering operator \[6\]) becomes a
computationally expensive operation. Third, visualization of
a large-dimensional front becomes a difficult task, thereby
causing difficulties in subsequent decision-making tasks and in
evaluating the performance of an algorithm. For this purpose,
the performance metrics (such as hyper-volume measure \[7\] or
other metrics \[3\], \[8\]) are either computationally too expensive
or may not be meaningful.

An important question to ask is then ‘Are EMOs useful for
many-objective optimization problems?’. Although the third
difficulty related to visualization and performance measures
mentioned above cannot be avoided, some algorithmic changes
to the existing EMO algorithms may be possible to address
the first two concerns. In this paper, we review some of
the past efforts \[9\], \[10\], \[11\], \[12\], \[13\], \[14\] in devising
many-objective EMOs and outline some viable directions
for designing efficient many-objective EMO methodologies.
Thereafter, we propose a new method that uses the framework
of NSGA-II procedure \[5\] but works with a set of supplied
or predefined reference points and demonstrate its efficacy in
solving two to 15-objective optimization problems. In this
paper, we introduce the framework and restrict to solving
unconstrained problems of various kinds, such as having
normalized, scaled, convex, concave, disjointed, and focusing
on a part of the Pareto-optimal front. Practice may offer a
number of such properties to exist in a problem. Therefore, an
adequate test of an algorithm for these eventualities remains
as an important task. We compare the performance of the
proposed NSGA-III with two versions of an existing many-
objective EMO (MOEA/D \[10\]), as the method is somewhat
similar to the proposed method. Interesting insights about
working of both versions of MOEA/D and NSGA-III are
revealed. The proposed NSGA-III is also evaluated for its
use in a few other interesting multi-objective optimization and

Copyright (c) 2013 IEEE. Personal use is permitted. For any other purposes, permission must be obtained from the IEEE by emailing pubs-permissions@ieee.org.

<details>
<summary>Page 1: complete visual reference</summary>

![Full source page 1](K:/source/OptunaSharp/docs/papers/paper-gateway/library/4c240f7368799f9d/pages/page-0001.png)

</details>

<a id="page-2"></a>

<!-- Page 2 -->

This article has been accepted for publication in a future issue of this journal, but has not been fully edited. Content may change prior to final publication.

2

decision-making tasks. In the sequel of this paper, we suggest
an extension of the proposed NSGA-III for handling many-
objective constrained optimization problems and a few other
special and challenging many-objective problems.

In the remainder of this paper, we first discuss the diffi-
culties in solving many-objective optimization problems and
then attempt to answer the question posed above about the
usefulness of EMO algorithms in handling many objectives.
Thereafter, in Section III, we present a review of some of
the past studies on many-objective optimization including a
recently proposed method MOEA/D \[10\]. Then, in Section IV,
we outline our proposed NSGA-III procedure in detail. Results
on normalized DTLZ test problems up to 15 objectives using
NSGA-III and two versions of MOEA/D are presented next
in Section V-A. Results on scaled version of DTLZ problems
suggested here are shown next. Thereafter in subsequent
sections, NSGA-III procedure is tested on different types
of many-objective optimization problems. Finally, NSGA-III
is applied to two practical problems involving three and
nine objective problems in Section VII. Conclusions of this
extensive study are drawn in Section VIII.

II. MANY-OBJECTIVE PROBLEMS

Loosely, many-objective problems are defined as problems
having four or more objectives. Two and three-objective prob-
lems fall into a different class as the resulting Pareto-optimal
front, in most cases, in its totality can be comprehensively
visualized using graphical means. Although a strict upper
bound on the number of objectives for a many-objective
optimization problem is not so clear, except a few occasions
\[15\], most practitioners are interested in a maximum of 10
to 15 objectives. In this section, first, we discuss difficulties
that an existing EMO algorithm may face in handling many-
objective problems and investigate if EMO algorithms are
useful at all in handling a large number of objectives.

A. Difficulties in Handling Many Objectives

It has been discussed elsewhere \[4\], \[16\] that the current
state-of-the-art EMO algorithms that work under the principle
of domination \[17\] may face the following difficulties:

1\) A large fraction of population is non-dominated: It is

well-known \[3\], \[16\] that with an increase in number
of objectives, an increasingly larger fraction of a ran-
domly generated population becomes non-dominated.
Since most EMO algorithms emphasize non-dominated
solutions in a population, in handling many-objective
problems, there is not much room for creating new
solutions in a generation. This slows down the search
process and therefore the overall EMO algorithm be-
comes inefficient.
2\) Evaluation of diversity measure becomes computation-

ally expensive: To determine the extent of crowding of
solutions in a population, the identification of neigh-
bors becomes computationally expensive in a large-
dimensional space. Any compromise or approximation
in diversity estimate to make the computations faster

may cause an unacceptable distribution of solutions at
the end.
3\) Recombination operation may be inefficient: In a many-

objective problem, if only a handful of solutions are
to be found in a large-dimensional space, solutions are
likely to be widely distant from each other. In such a
population, the effect of recombination operator (which
is considered as a key search operator in an EMO)
becomes questionable. Two distant parent solutions are
likely to produce offspring solutions that are also distant
from parents. Thus, special recombination operators
(mating restriction or other schemes) may be necessary
for handling many-objective problems efficiently.
4\) Representation of trade-off surface is difficult: It is

intuitive to realize that to represent a higher dimensional
trade-off surface, exponentially more points are needed.
Thus, a large population size is needed to represent the
resulting Pareto-optimal front. This causes difficulty for
a decision-maker to comprehend and make an adequate
decision to choose a preferred solution.
5\) Performance metrics are computationally expensive to

compute: Since higher-dimensional sets of points are
to be compared against each other to establish the
performance of one algorithm against another, a larger
computational effort is needed. For example, computing
hyper-volume metric requires exponentially more com-
putations with the number of objectives \[18\], \[19\].
6\) Visualization is difficult: Finally, although it is not a

matter related to optimization directly, eventually visu-
alization of a higher-dimensional trade-off front may be
difficult for many-objective problems.
The first three difficulties can only be alleviated by certain
modifications to existing EMO methodologies. The fourth,
fifth, and sixth difficulties are common to all many-objective
optimization problems and we do not address them adequately
here.

B. EMO Methodologies for Handling Many-Objective Prob-
lems

Before we discuss the possible remedies for three difficulties
mentioned above, here we highlight two different many-
objective problem classes for which existing EMO method-
ologies can still be used.

First, existing EMO algorithms may still be useful in finding
a preferred subset of solutions (a partial set) from the complete
Pareto-optimal set. Although the preferred subset will still be
many-dimensional, since the targeted solutions are focused in
a small region on the Pareto-optimal front, most of the above
difficulties will be alleviated by this principle. A number of
MCDM-based EMO methodologies are already devised for
this purpose and results on as large as 10-objective problems
have shown to perform well \[20\], \[21\], \[22\], \[23\].

Second, many problems in practice, albeit having many
objectives, often degenerate to result in a low-dimensional
Pareto-optimal front \[4\], \[11\], \[24\], \[25\]. In such problems,
identification of redundant objectives can be integrated with an
EMO to find the Pareto-optimal front that is low-dimensional.

Copyright (c) 2013 IEEE. Personal use is permitted. For any other purposes, permission must be obtained from the IEEE by emailing pubs-permissions@ieee.org.

<details>
<summary>Page 2: complete visual reference</summary>

![Full source page 2](K:/source/OptunaSharp/docs/papers/paper-gateway/library/4c240f7368799f9d/pages/page-0002.png)

</details>

<a id="page-3"></a>

<!-- Page 3 -->

This article has been accepted for publication in a future issue of this journal, but has not been fully edited. Content may change prior to final publication.

3

Since the ultimate front is as low as two or three-dimensional,
existing EMO methodologies should work well in handling
such problems. A previous study of NSGA-II with a principal
component analysis (PCA) based procedure \[4\] was able to
solve as large as 50-objective problems having a two-objective
Pareto-optimal front.

C. Two Ideas for a Many-Objective EMO

Keeping in mind the first three difficulties associated with
a domination-based EMO procedure, two different strategies
can be considered to alleviate the difficulties:

1\) Use of a special domination principle: The first difficulty

mentioned above can be alleviated by using a special
domination principle that will adaptively discretize the
Pareto-optimal front and find a well-distributed set of
points. For example, the use of ϵ-domination principle
\[26\], \[27\] will make all points within ϵ distance from a
set of Pareto-optimal points ϵ-dominated and hence the
process will generate a finite number of Pareto-optimal
points as target. Such a consideration will also alleviate
the second difficulty of diversity preservation. The third
difficulty can be taken care of by the use of a mating
restriction scheme or a special recombination scheme
in which near-parent solutions are emphasized (such as
SBX with a large distribution index \[28\]). Other special
domination principles \[29\], \[30\] can also be used for this
purpose. Aguirre and Tanaka \[31\] and Sato et al. \[32\]
suggested the use of a subset of objectives for dominance
check and using a different combination of objectives in
every generation. The use of fixed cone-domination \[33\],
\[34\] or variable cone-domination \[35\] principles can also
be tried. These studies were made in the context of low-
dimensional problems and their success in solving many-
objective optimization is yet to be established.
2\) Use of a predefined multiple targeted search: It has

been increasingly clear that it is too much to expect
from a single population-based optimization algorithm
to have convergence of its population near the Pareto-
optimal front and simultaneously its distributed uni-
formly around the entire front in a large-dimensional
problem. One way to handle such many-objective opti-
mization problems would be to aid the diversity mainte-
nance issue by some external means. This principle can
directly address the second difficulty mentioned above.
Instead of searching the entire search space for Pareto-
optimal solutions, multiple predefined targeted searches
can be set by an algorithm. Since optimal points are
found corresponding to each of the targeted search task,
the first difficulty of dealing with a large non-dominated
set is also alleviated. The recombination issue can be
addressed by using a mating restriction scheme in which
two solutions from neighboring targets are participated
in the recombination operation. Our proposed algorithm
(NSGA-III) is based on this principle and thus we
discuss this aspect in somewhat more detail. We suggest
two different ways to implement the predefined multiple
targeted search principle:

a) A set of predefined search directions spanning

the entire Pareto-optimal front can be specified
beforehand and multiple searches can be performed
along each direction. Since the search directions
are widely distributed, the obtained optimal points
are also likely to be widely distributed on the
Pareto-optimal front in most problems. Recently
proposed MOEA/D procedure \[10\] uses this con-
cept.
b) Instead of multiple search directions, multiple pre-

defined reference points can be specified for this
purpose. Thereafter, points corresponding to each
reference point can be emphasized to find set of
widely distributed set of Pareto-optimal points. A
few such implementations were proposed recently
\[36\], \[37\], \[38\], \[14\], and this paper suggests
another approach extending and perfecting the al-
gorithm proposed in the first reference \[36\].

III. EXISTING MANY-OBJECTIVE OPTIMIZATION

ALGORITHMS

Garza-Fabre, Pulido and Coello \[16\] suggested three single-
objective measures by using differences in individual objective
values between two competing parents and showed that in 5 to
50-objective DTLZ1, DTLZ3 and DTLZ6 problems the con-
vergence property can get enhanced compared to a number of
existing methods including the usual Pareto-dominance based
EMO approaches. Purshouse and Fleming \[39\] clearly showed
that diversity preservation and achieving convergence near the
Pareto-optimal front are two contradictory goals and usual
genetic operators are not adequate to attain both goals simul-
taneously, particularly for many-objective problems. Another
study \[40\] extends NSGA-II by adding diversity-controlling
operators to solve six to 20-objective DTLZ2 problems.
K¨oppen and Yoshida \[41\] claimed that NSGA-II procedure
in its originality is not suitable for many-objective optimiza-
tion problems and suggested a number of metrics that can
potentially replace NSGA-II’s crowding distance operator for
better performance. Based on simulation studies on two to
15-objective DTLZ2, DTLZ3 and DTLZ6 problems, they
suggested to use a substitute assignment distance measure as
the best strategy. Hadka and Reed \[42\] suggested an ensemble-
based EMO procedure which uses a suitable recombination
operator adaptively chosen from a set of eight to 10 different
pre-defined operators based on their generation-wise success
rate in a problem. It also uses ϵ-dominance concept and an
adaptive population sizing approach that is reported to solve
up to eight-objective test problems successfully. Bader and
Zitzler \[43\] have suggested a fast procedure for computing
sample-based hyper-volume and devised an algorithm to find a
set of trade-off solutions for maximizing the hyper-volume. A
growing literature on approximate hyper-volume computation
\[44\], \[18\], \[45\] may make such an approach practical for
solving many-objective problems.

The above studies analyze and extend previously-suggested
evolutionary multi-objective optimization algorithms for their
suitability to solving many-objective problems. In most cases,

Copyright (c) 2013 IEEE. Personal use is permitted. For any other purposes, permission must be obtained from the IEEE by emailing pubs-permissions@ieee.org.

<details>
<summary>Page 3: complete visual reference</summary>

![Full source page 3](K:/source/OptunaSharp/docs/papers/paper-gateway/library/4c240f7368799f9d/pages/page-0003.png)

</details>

> Quality note: Mathematical notation or superscripts require visual review; linear text does not establish fraction, exponent or equation semantics.

<a id="page-4"></a>

<!-- Page 4 -->

This article has been accepted for publication in a future issue of this journal, but has not been fully edited. Content may change prior to final publication.

4

the results are promising and the suggested algorithms must
be tested on other more challenging problems than the usual
normalized test problems such as DTLZ problems. They
must also be tried on real-world problems. In the following
paragraphs, we describe a recently proposed algorithm which
fits well with our description of a many-objective optimization
algorithm given in Section II-C and closely matches with our
proposed algorithm.

MOEA/D \[10\] uses a predefined set of weight vectors to
maintain a diverse set of trade-off solutions. For each weight
vector, the resulting problem is called a sub-problem. To start
with, every population member (with size same as the number
of weight vectors) is associated with a weight vector randomly.
Thereafter, two solutions from neighboring weight vectors
(defined through a niching parameter (T )) are mated and an
offspring solution is created. The offspring is then associated
with one or more weight vectors based on a performance
metric. Two metrics are suggested in the study. A penalized
distance measure of a point from the ideal point is formed by
weighted sum (weight θ is another algorithmic parameter) of
perpendicular distance (d2) from the reference direction and
distance (d1) along the reference direction:

PBI(x, w) = d1 + θd2. (1)

We call this procedure here as MOEA/D-PBI. The second
approach suggested is the use of Tchebycheff metric using
a utopian point z∗and the weight vector w:

M
TCH(x, w, z∗) = max wi\|fi(x) −z∗i \|. (2)
i=1

In reported simulations \[10\], the ideal point was used as z∗

and zero-weight scenario is handled by using a small number.
We call this procedure as MOEA/D-TCH here. An external
population maintains the non-dominated solutions. The first
two difficulties mentioned earlier are negotiated by using an
explicit set of weight vectors to find points and the third
difficulty is alleviated by using a mating restriction scheme.
Simulation results were shown for two and three-objective test
problems only and it was concluded that MOEA/D-PBI is
better for three-objective problems than MOEA/D-TCH and
the performance of MOEA/D-TCH improved with an objective
normalization process using population minimum and maxi-
mum objective values. Both versions of MOEA/D require to
set a niching parameter (T ). Based on some simulation results
on two and three-objective problems, authors suggested the
use of a large fraction of population size as T . Additionally,
MOEA/D-PBI requires an appropriate setting of an additional
parameter – penalty parameter θ, for which authors have
suggested a value of 5.

A later study by the developers of MOEA/D suggested
the use of differential evolution (DE) to replace genetic
recombination and mutation operators. Also, further modifi-
cations were done in defining neighborhood of a particular
solution and in replacing parents in a given neighborhood
by the corresponding offspring solutions \[46\]. We call this
method as MOEA/D-DE here. Results on a set of mostly
two and a three-objective linked problems \[47\] showed better
performance with MOEA/D-DE compared to other algorithms.

As mentioned, MOEA/D is a promising approach for many-
objective optimization as it addresses some of the difficulties
mentioned above well, but the above-mentioned MOEA/D
studies did not quite explore their suitability to a large number
of objectives. In this paper, we apply them to problems having
up to 15 objectives and evaluate their applicability to truly
many-objective optimization problems and reveal interesting
properties of these algorithms.

Another recent study \[14\] follows our description of a
many-objective optimization procedure. The study extends
the NSGA-II procedure to suggest a hybrid NSGA-II (HN
algorithm) for handling three and four-objective problems.
Combined population members are projected on a hyper-
plane and a clustering operation is performed on the hyper-
plane to select a desired number of clusters which is user-
defined. Thereafter, based on the diversity of the population,
either a local search operation on a random cluster member
is used to move the solution closer to the Pareto-optimal
front or a diversity enhancement operator is used to choose
population members from all clusters. Since no targeted and
distributed search is used, the approach is more generic than
MOEA/D or the procedure suggested in this paper. However,
the efficiency of HN algorithm for problems having more than
four objectives is yet to be investigated to suggest its use for
many-objective problems, in general. We now describe our
proposed algorithm.

IV. PROPOSED ALGORITHM: NSGA-III

The basic framework of the proposed many-objective
NSGA-II (or NSGA-III) remains similar to the original
NSGA-II algorithm \[5\] with significant changes in its selec-
tion mechanism. But unlike in NSGA-II, the maintenance of
diversity among population members in NSGA-III is aided by
supplying and adaptively updating a number of well-spread
reference points. For completeness, we first present a brief
description of the original NSGA-II algorithm.

Let us consider t-th generation of NSGA-II algorithm.
Suppose the parent population at this generation is Pt and
its size is N, while the offspring population created from Pt
is Qt having N members. The first step is to choose the
best N members from the combined parent and offspring
population Rt = Pt ∪Qt (of size 2N), thus allowing to
preserve elite members of the parent population. To achieve
this, first, the combined population Rt is sorted according to
different non-domination levels (F1, F2 and so on). Then, each
non-domination level is selected one at a time to construct a
new population St, starting from F1, until the size of St is
equal to N or for the first time exceeds N. Let us say the last
level included is the l-th level. Thus, all solutions from level
(l + 1) onwards are rejected from the combined population
Rt. In most situations, the last accepted level (l-th level) is
only accepted partially. In such a case, only those solutions
that will maximize the diversity of the l-th front are chosen. In
NSGA-II, this is achieved through a computationally efficient
yet approximate niche-preservation operator which computes
the crowding distance for every last level member as the
summation of objective-wise normalized distance between two

Copyright (c) 2013 IEEE. Personal use is permitted. For any other purposes, permission must be obtained from the IEEE by emailing pubs-permissions@ieee.org.

<details>
<summary>Page 4: complete visual reference</summary>

![Full source page 4](K:/source/OptunaSharp/docs/papers/paper-gateway/library/4c240f7368799f9d/pages/page-0004.png)

</details>

> Quality note: Mathematical notation or superscripts require visual review; linear text does not establish fraction, exponent or equation semantics.

<a id="page-5"></a>

<!-- Page 5 -->

This article has been accepted for publication in a future issue of this journal, but has not been fully edited. Content may change prior to final publication.

5

neighboring solutions. Thereafter, the solutions having larger
crowding distance values are chosen. Here, we replace the
crowding distance operator with the following approaches
(subsections IV-A to IV-E).

A. Classification of Population into Non-dominated Levels

The above procedure of identifying non-dominated fronts
using the usual domination principle \[17\] is also used in
NSGA-III. All population members from non-dominated front
level 1 to level l are first included in St. If \|St\| = N, no further
operations are needed and the next generation is started with
Pt+1 = St. For \|St\| &gt; N, members from one to (l −1)
fronts are already selected, that is, Pt+1 = ∪l−1i=1Fi, and the
remaining (K = N −\|Pt+1\|) population members are chosen
from the last front Fl. We describe the remaining selection
process in the following subsections.

B. Determination of Reference Points on a Hyper-Plane

As indicated before, NSGA-III uses a predefined set of
reference points to ensure diversity in obtained solutions.
The chosen reference points can either be predefined in a
structured manner or supplied preferentially by the user. We
shall present results of both methods in the results section later.
In the absence of any preference information, any predefined
structured placement of reference points can be adopted, but in
this paper we use Das and Dennis’s \[48\] systematic approach1

that places points on a normalized hyper-plane – a (M −1)-
dimensional unit simplex – which is equally inclined to all
objective axes and has an intercept of one on each axis. If p
divisions are considered along each objective, the total number
of reference points (H) in an M-objective problem is given
by: "

!M + p −1

H = . (3)

p

For example, in a three-objective problem (M = 3), the
reference points are created on a triangle with apex at (1, 0, 0),
(0, 1, 0) and (0, 0, 1). If four divisions (p = 4) are chosen for$
each objective axis, H = \#3+4−14 or 15 reference points will
be created. For clarity, these reference points are shown in
Figure 1. In the proposed NSGA-III, in addition to emphasiz-
ing non-dominated solutions, we also emphasize population
members which are in some sense associated with each
of these reference points. Since the above-created reference
points are widely distributed on the entire normalized hyper-
plane, the obtained solutions are also likely to be widely
distributed on or near the Pareto-optimal front. In the case of a
user-supplied set of preferred reference points, ideally the user
can mark H points on the normalized hyper-plane or indicate
any H, M-dimensional vectors for the purpose. The proposed
algorithm is likely to find near Pareto-optimal solutions cor-
responding to the supplied reference points, thereby allowing
this method to be used more from the point of view of a
combined application of decision-making and many-objective
optimization. The procedure is presented in Algorithm 1.

1Any other structured distribution with or without a biasing on some part
of the Pareto-optimal front can be used as well.

f3

1

Reference
point Normalized
hyperplane

Reference
line

## f1 1 1 f2

Ideal point

![Fig. 1. 15 reference points are shown on a normalized reference plane for a three-objective problem with p = 4.](K:/source/OptunaSharp/docs/papers/paper-gateway/library/4c240f7368799f9d/figures/p0005-figure-01.png)

Fig. 1. 15 reference points are shown on a normalized reference plane for
a three-objective problem with p = 4.

Algorithm 1 Generation t of NSGA-III procedure
Input: H structured reference points Zs or supplied aspira-

tion points Za, parent population Pt
Output: Pt+1

1: St = ∅, i = 1

2: Qt = Recombination+Mutation(Pt)

3: Rt = Pt ∪Qt
4: (F1, F2, . . .) = Non-dominated-sort(Rt)

5: repeat

6: St = St ∪Fi and i = i + 1
7: until \|St\| ≥N

8: Last front to be included: Fl = Fi
9: if \|St\| = N then

10: Pt+1 = St, break

11: else
12: Pt+1 = ∪l−1j=1Fj
13: Points to be chosen from Fl: K = N −\|Pt+1\|

14: Normalize objectives and create reference set Zr:
Normalize(fn, St, Zr, Zs, Za)

15: Associate each member s of St with a reference point:
\[π(s), d(s)\] =Associate(St, Zr) % π(s): closest
reference point, d: distance between s and π(s)

16: %Compute niche count of reference point j ∈Zr: ρj =

s∈St/Fl ((π(s) = j) ? 1 : 0)

17: Choose K members one at a time from Fl to construct
Pt+1: Niching(K, ρj, π, d, Zr, Fl, Pt+1)
18: end if

C. Adaptive Normalization of Population Members

First, the ideal point of the population St is determined
by identifying the minimum value (zmini ), for each objective
function i = 1, 2, . . . , M in ∪tτ=0Sτ and by constructing the
ideal point ¯z = (zmin1 , zmin2 , . . . , zminM ). Each objective value
of St is then translated by subtracting objective fi by zmini ,
so that the ideal point of translated St becomes a zero vector.
We denote this translated objective as f ′i(x) = fi(x) −zmini .
Thereafter, the extreme point in each objective axis is identi-
fied by finding the solution (x ∈St) that makes the following
achievement scalarizing function minimum with weight vector

Copyright (c) 2013 IEEE. Personal use is permitted. For any other purposes, permission must be obtained from the IEEE by emailing pubs-permissions@ieee.org.

<details>
<summary>Page 5: complete visual reference</summary>

![Full source page 5](K:/source/OptunaSharp/docs/papers/paper-gateway/library/4c240f7368799f9d/pages/page-0005.png)

</details>

> Quality note: Mathematical notation or superscripts require visual review; linear text does not establish fraction, exponent or equation semantics.

<a id="page-6"></a>

<!-- Page 6 -->

![Complete page reference; figure boundary uncertain. Full source page; figure boundaries require review.](K:/source/OptunaSharp/docs/papers/paper-gateway/library/4c240f7368799f9d/pages/page-0006.png)

This article has been accepted for publication in a future issue of this journal, but has not been fully edited. Content may change prior to final publication.

6

w being the axis direction:

M
ASF(x, w) = maxi=1 f ′i(x)/wi, for x ∈St. (4)

For wi = 0, we replace it with a small number 10−6. For i-th
translated objective direction f ′i, this will result in an extreme
objective vector, zi,max. These M extreme vectors are then
used to constitute a M-dimensional linear hyper-plane. The
intercept ai of the i-th objective axis and the linear hyper-
plane can then be computed (see Figure 2) and the objective
functions can be normalized as follows:

f n f ′i(x) = fi(x) −zmini
i (x) = , for i = 1, 2, . . ., M.
ai −zmini ai −zmini

(5)
Note that the intercept on each normalized objective axis is
now at f ni = 1 and a hyper-plane constructed with these
intercept points will make %M i = 1.

i=1 f n

Algorithm 2 Normalize(fn, St, Zr, Zs/Za) procedure
Input: St, Zs (structured points) or Za (supplied points)
Output: f n, Zr (reference points on normalized hyper-plane)

1: for j = 1 to M do
2: Compute ideal point: zminj = mins∈St fj(s)

3: Translate objectives: f ′j(s) = fj(s) −zminj ∀s ∈St
4: Compute extreme points: zj,max = s :
argmins∈StASF(s, wj), where wj = (ϵ, . . . , ϵ)T

ϵ = 10−6, and wjj = 1

5: end for

6: Compute intercepts aj for j = 1, . . . , M
7: Normalize objectives (f n) using Equation 5

8: if Za is given then
9: Map each (aspiration) point on normalized hyper-plane
using Equation 5 and save the points in the set Zr

10: else
11: Zr = Zs

12: end if

D. Association Operation

## f’3 After normalizing each objective adaptively based on the

## 3

extent of members of St in the objective space, next we need to
z3,max associate each population member with a reference point. Forthis purpose, we define a reference line corresponding to each
 2 reference point on the hyper-plane by joining the reference
point with the origin. Then, we calculate the perpendicular
a3 distance of each population member of St from each of the
 1 a2 reference lines. The reference point whose reference line isclosest to a population member in the normalized objective
z2,max space is considered associated with the population member.
 0 This is illustrated in Figure 3. The procedure is presented in

## 0

## a1 1 1.5 f3

## 2 z1,max

## 2 1 f’1 0 1f’ 2

Fig. 2. Procedure of computing intercepts and then forming the hyper-plane 0.5
from extreme points are shown for a three-objective problem.

In the case of structured reference points (H of them), the
original reference points calculated using Das and Dennis’s
\[48\] approach already lie on this normalized hyper-plane. In
the case of preferred reference points by the user, the reference
points are simply mapped on to the above-constructed normal-
ized hyper-plane using equation 5. Since the normalization
procedure and the creation of the hyper-plane is done at each
generation using extreme points ever found from the start of
the simulation, the proposed NSGA-III procedure adaptively
maintains a diversity in the space spanned by the members
of St at every generation. This enables NSGA-III to solve
problems having a Pareto-optimal front whose objective values
may be differently scaled. The procedure is also described in
Algorithm 2.

0

0

0.5

f2

1 0
1 0.5
1.5 1.5 f1

Fig. 3. Association of population members with reference points is illustrated.

Algorithm 3.

E. Niche-Preservation Operation

It is worth noting that a reference point may have one
or more population members associated with it or need not
have any population member associated with it. We count the

Copyright (c) 2013 IEEE. Personal use is permitted. For any other purposes, permission must be obtained from the IEEE by emailing pubs-permissions@ieee.org.

<details>
<summary>Page 6: complete visual reference</summary>

![Full source page 6](K:/source/OptunaSharp/docs/papers/paper-gateway/library/4c240f7368799f9d/pages/page-0006.png)

</details>

> Quality note: Mathematical notation or superscripts require visual review; linear text does not establish fraction, exponent or equation semantics.

<a id="page-7"></a>

<!-- Page 7 -->

This article has been accepted for publication in a future issue of this journal, but has not been fully edited. Content may change prior to final publication.

7

Algorithm 3 Associate(St, Zr) procedure
Input: Zr, St
Output: π(s ∈St), d(s ∈St)

1: for each reference point z ∈Zr do
2: Compute reference line w = z

3: end for

4: for each s ∈St do
5: for each w ∈Zr do

6: Compute d⊥(s, w) = s −wT s/∥w∥

7: end for
8: Assign π(s) = w : argminw∈Zrd⊥(s, w)

9: Assign d(s) = d⊥(s, π(s))
10: end for

number of population members from Pt+1 = St/Fl that are
associated with each reference point. Let us denote this niche
count as ρj for the j-th reference point. We now devise a new
niche-preserving operation as follows. First, we identify the
reference point set Jmin = \{j : argminjρj\} having minimum
ρj. In case of multiple such reference points, one (¯j ∈Jmin)
is chosen at random.

If ρ¯j = 0 (meaning that there is no associated Pt+1 member
to the reference point ¯j), there can be two scenarios with ¯j
in set Fl. First, there exists one or more members in front
Fl that are already associated with the reference point ¯j. In
this case, the one having the shortest perpendicular distance
from the reference line is added to Pt+1. The count ρ¯j is then
incremented by one. Second, the front Fl does not have any
member associated with the reference point ¯j. In this case, the
reference point is excluded from further consideration for the
current generation.

In the event of ρ¯j ≥1 (meaning that already one mem-
ber associated with the reference point exists in St/Fl), a
randomly2 chosen member, if exists, from front Fl that is
associated with the reference point ¯j is added to Pt+1. The
count ρ¯j is then incremented by one. After niche counts are
updated, the procedure is repeated for a total of K times to fill
all vacant population slots of Pt+1. The procedure is presented
in Algorithm 4.

F. Genetic Operations to Create Offspring Population

After Pt+1 is formed, it is then used to create a new
offspring population Qt+1 by applying usual genetic operators.
In NSGA-III, we have already performed a careful elitist selec-
tion of solutions and attempted to maintain diversity among
solutions by emphasizing solutions closest to the reference
line of each reference point. Also, as we shall describe in
Section V, for a computationally fast procedure, we have
set N almost equal to H, thereby giving equal importance
to each population member. For all these reasons, we do
not employ any explicit selection operation with NSGA-III.
The population Qt+1 is constructed by applying the usual
crossover and mutation operators by randomly picking parents
from Pt+1. However, to create offspring solutions closer to

2The point closest to the reference point or using any other diversity
preserving criterion can also be used.

Algorithm 4 Niching(K, ρj, π, d, Zr, Fl, Pt+1) procedure
Input: K, ρj, π(s ∈St), d(s ∈St), Zr, Fl
Output: Pt+1

1: k = 1
2: while k ≤K do

3: Jmin = \{j : argminj∈Zrρj\}
4: ¯j = random(Jmin)

5: I¯j = \{s : π(s) = ¯j, s ∈Fl\}

6: if I¯j ̸= ∅then
7: if ρ¯j = 0 then &amp; '

8: Pt+1 = Pt+1 ∪ s : argmins∈I¯jd(s)

9: else

10: Pt+1 = Pt+1 ∪random(I¯j)
11: end if

12: ρ¯j = ρ¯j + 1, Fl = Fl\\s

13: k = k + 1
14: else

15: Zr = Zr/\{¯j\}
16: end if

17: end while

parent solutions (to take care of third difficulty mentioned in
Section II-A), we suggest using a relatively larger value of
distribution index in the SBX operator.

G. Computational Complexity of One Generation of NSGA-III

The non-dominated sorting (line 4 in Algorithm 1) of a
population of size 2N having M-dimensional objective vec-
tors require O(N logM−2 N) computations \[49\]. Identification
of ideal point in line 2 of Algorithm 2 requires a total
of O(MN) computations. Translation of objectives (line 3)
requires O(MN) computations. However, identification of
extreme points (line 4) require O(M 2N) computations. De-
termination of intercepts (line 6) requires one matrix inversion
of size M × M, requiring O(M 3) operations. Thereafter,
normalization of a maximum of 2N population members
(line 7) require O(N) computations. Line 8 of Algorithm 2
requires O(MH) computations. All operations in Algorithm 3
in associating a maximum of 2N population members to
H reference points would require O(MNH) computations.
Thereafter, in the niching procedure in Algorithm 4, line 3 will
require O(H) comparisons. Assuming that L = \|Fl\|, line 5
requires O(L) checks. Line 8 in the worst case requires O(L)
computations. Other operations have smaller complexity. How-
ever, the above computations in the niching algorithm need to
be performed a maximum of L times, thereby requiring larger
of O(L2) or O(LH) computations. In the worst case scenario
(St = F1, that is, the first non-dominated front exceeds the
population size), L ≤2N. In all our simulations, we have used
N ≈H and N &gt; M. Considering all the above considerations
and computations, the overall worst-case complexity of one
generation of NSGA-III is O(N 2 logM−2 N) or O(N 2M),
whichever is larger.

Copyright (c) 2013 IEEE. Personal use is permitted. For any other purposes, permission must be obtained from the IEEE by emailing pubs-permissions@ieee.org.

<details>
<summary>Page 7: complete visual reference</summary>

![Full source page 7](K:/source/OptunaSharp/docs/papers/paper-gateway/library/4c240f7368799f9d/pages/page-0007.png)

</details>

> Quality note: Mathematical notation or superscripts require visual review; linear text does not establish fraction, exponent or equation semantics.

<a id="page-8"></a>

<!-- Page 8 -->

This article has been accepted for publication in a future issue of this journal, but has not been fully edited. Content may change prior to final publication.

8

H. Parameter-Less Property of NSGA-III

Like in NSGA-II, NSGA-III algorithm does not require to
set any new parameter other than the usual genetic parameters
such as the population size, termination parameter, crossover
and mutation probabilities and their associated parameters.
The number of reference points H is not an algorithmic
parameter, as this is entirely at the disposal of the user. The
population size N is dependent on H, as N ≈H. The
location of the reference points is similarly dependent on the
preference information that the user is interested to achieve in
the obtained solutions.

We shall now present simulation results of NSGA-III for
various many-objective scenarios and compare its performance
with MOEA/D and classical methods.

V. RESULTS

In this section, we provide the simulation results of NSGA-
III on three to 15-objective optimization problems. Since our
method has a framework similar to MOEA/D in that both types
of algorithms require a set of user-supplied reference points
or weight vectors, we compare our proposed method with
different versions of MOEA/D (codes from MOEA/D web-
site \[50\] are used). The original MOEA/D study proposed two
procedures (MOEA/D-PBI and MOEA/D-TCH), but did not
solve four or more objective problems. Here, along with our
algorithm, we investigate the performance of these MOEA/D
algorithms on three to 15-objective problems.

As a performance metric, we have chosen the inverse
generational distance (IGD) metric \[51\], \[47\] which as a
single metric which can provide a combined information
about the convergence and diversity of the obtained solutions.
Since reference points or reference directions are supplied in
NSGA-III and MOEA/D algorithms, respectively, and since
in this section we show the working of these methods on test
problems for which the exact Pareto-optimal surface is known,
we can exactly locate the targeted Pareto-optimal points in
the normalized objective space. We compute these targeted
points and call them a set Z. For any algorithm, we obtain
the final non-dominated points in the objective space and call
them the set A. Now, we compute the IGD metric as the
average Euclidean distance of points in set Z with their nearest
members of all points in set A:

∑\|Z\| \|A\|
IGD(A, Z) = 1 minj=1 d(zi, aj), (6)

\|Z\| i=1

where d(zi, aj) = ∥zi −aj∥2. It is needless to write that a set
with a smaller IGD value is better. If no solution associated
with a reference point is found, the IGD metric value for the
set will be large. For each case, 20 different runs from different
initial populations are performed and best, median and worst
IGD performance values are reported. For all algorithms, the
population members from the final generation are presented
and used for computing the performance measure.

Table I shows the number of chosen reference points (H) for
different sizes of a problem. The population size N for NSGA-

III is set as the smallest multiple of four3 higher than H. For
MOEA/D procedures, we use a population size, N ′ = H, as
suggested by their developers. For three-objective problems,

TABLE I
NUMBER OF REFERENCE POINTS/DIRECTIONS AND CORRESPONDING
POPULATION SIZES USED IN NSGA-III AND MOEA/D ALGORITHMS.

No. of Ref. pts./ NSGA-III MOEA/D
objectives Ref. dirn. popsize popsize
(M) (H) (N) (N′)
3 91 92 91
5 210 212 210
8 156 156 156
10 275 276 275
15 135 136 135

C(3−1+12, 12)

we have used p = 12 in order to obtain H = 12 or
91 reference points (refer to equation 3). For five-objective
problems, we have used p = 6, so that H = 210 reference
points are obtained. Note that as long as p ≥M is not chosen,
no intermediate point will be created by Das and Dennis’s
systematic approach. For eight-objective problems, even if we
use p = 8 (to have at least one intermediate reference point),
it requires 5,040 reference points. To avoid such a situation,
we use two layers of reference points with small values of p.
On the boundary layer, we use p = 3, so that 120 points are
created. On the inside layer, we use p = 2, so that 36 points
are created. All H = 156 points are then used as reference
points for eight-objective problems. We illustrate this scenario
in Figure 4 for a three-objective problem using p = 2 for
boundary layer and p = 1 for the inside layer. For 10-objective

![Fig. 4. The concept for two-layered reference points (with six points on the boundary layer (p = 2) and three points on the inside layer (p = 1)) is shown for a three-objective problem, but are implemented for eight or more objectives in the simulations here.](K:/source/OptunaSharp/docs/papers/paper-gateway/library/4c240f7368799f9d/figures/p0008-figure-01.png)

Fig. 4. The concept for two-layered reference points (with six points on
the boundary layer (p = 2) and three points on the inside layer (p = 1)) is
shown for a three-objective problem, but are implemented for eight or more
objectives in the simulations here.

problems as well, we use p = 3 and p = 2 for boundary and
inside layers, respectively, thereby requiring a total of H =
220 + 55 or 275 reference points. Similarly, for 15-objective
problems, we use p = 2 and p = 1 for boundary and inside
layers, respectively, thereby requiring H = 120 + 15 or 135
reference points.

3Since no tournament selection is used in NSGA-III, a factor of two would
be adequate as well, but as we re-introduce tournament selection in the
constrained NSGA-III in the sequel paper \[52\], we keep population size as a
multiple of four here as well to have a unified algorithm.

Copyright (c) 2013 IEEE. Personal use is permitted. For any other purposes, permission must be obtained from the IEEE by emailing pubs-permissions@ieee.org.

<details>
<summary>OCR transcript from a graphic region; inspect the source image for layout</summary>

```text
Boundary Layer
```

</details>

<details>
<summary>OCR transcript from a graphic region; inspect the source image for layout</summary>

```text
Inside Layer \
```

</details>

<details>
<summary>OCR transcript from a graphic region; inspect the source image for layout</summary>

```text
1 f
```

</details>

<details>
<summary>OCR transcript from a graphic region; inspect the source image for layout</summary>

```text
a
```

</details>

<details>
<summary>OCR transcript from a graphic region; inspect the source image for layout</summary>

```text
0.5
```

</details>

<details>
<summary>OCR transcript from a graphic region; inspect the source image for layout</summary>

```text
0
0 0
```

</details>

<details>
<summary>OCR transcript from a graphic region; inspect the source image for layout</summary>

```text
0.5 0.5
```

</details>

<details>
<summary>OCR transcript from a graphic region; inspect the source image for layout</summary>

```text
f, f,
11
```

</details>

<details>
<summary>Complete alternate PDF text layer</summary>

```text
Copyright (c) 2013 IEEE. Personal use is permitted. For any other purposes, permission must be obtained from the IEEE by emailing pubs-permissions@ieee.org.
This article has been accepted for publication in a future issue of this journal, but has not been fully edited. Content may change prior to final publication.
8
H. Parameter-Less Property of NSGA-III
Like in NSGA-II, NSGA-III algorithm does not require to
set any new parameter other than the usual genetic parameters
such as the population size, termination parameter, crossover
and mutation probabilities and their associated parameters.
The number of reference points H is not an algorithmic
parameter, as this is entirely at the disposal of the user. The
population size N is dependent on H, as N ≈H. The
location of the reference points is similarly dependent on the
preference information that the user is interested to achieve in
the obtained solutions.
We shall now present simulation results of NSGA-III for
various many-objective scenarios and compare its performance
with MOEA/D and classical methods.
V. RESULTS
In this section, we provide the simulation results of NSGA-
III on three to 15-objective optimization problems. Since our
method has a framework similar to MOEA/D in that both types
of algorithms require a set of user-supplied reference points
or weight vectors, we compare our proposed method with
different versions of MOEA/D (codes from MOEA/D web-
site [50] are used). The original MOEA/D study proposed two
procedures (MOEA/D-PBI and MOEA/D-TCH), but did not
solve four or more objective problems. Here, along with our
algorithm, we investigate the performance of these MOEA/D
algorithms on three to 15-objective problems.
As a performance metric, we have chosen the inverse
generational distance (IGD) metric [51], [47] which as a
single metric which can provide a combined information
about the convergence and diversity of the obtained solutions.
Since reference points or reference directions are supplied in
NSGA-III and MOEA/D algorithms, respectively, and since
in this section we show the working of these methods on test
problems for which the exact Pareto-optimal surface is known,
we can exactly locate the targeted Pareto-optimal points in
the normalized objective space. We compute these targeted
points and call them a set Z. For any algorithm, we obtain
the ﬁnal non-dominated points in the objective space and call
them the set A. Now, we compute the IGD metric as the
average Euclidean distance of points in set Z with their nearest
members of all points in set A:
IGD(A, Z) = 1
|Z|
|Z|
(
i=1
|A|
min
j=1 d(zi, aj),
(6)
where d(zi, aj) = ∥zi −aj∥2. It is needless to write that a set
with a smaller IGD value is better. If no solution associated
with a reference point is found, the IGD metric value for the
set will be large. For each case, 20 different runs from different
initial populations are performed and best, median and worst
IGD performance values are reported. For all algorithms, the
population members from the ﬁnal generation are presented
and used for computing the performance measure.
Table I shows the number of chosen reference points (H) for
different sizes of a problem. The population size N for NSGA-
III is set as the smallest multiple of four3 higher than H. For
MOEA/D procedures, we use a population size, N ′ = H, as
suggested by their developers. For three-objective problems,
TABLE I
NUMBER OF REFERENCE POINTS/DIRECTIONS AND CORRESPONDING
POPULATION SIZES USED IN NSGA-III AND MOEA/D ALGORITHMS.
No. of
Ref. pts./
NSGA-III
MOEA/D
objectives
Ref. dirn.
popsize
popsize
(M)
(H)
(N)
(N′)
3
91
92
91
5
210
212
210
8
156
156
156
10
275
276
275
15
135
136
135
we have used p = 12 in order to obtain H =
#3−1+12
12
$
or
91 reference points (refer to equation 3). For ﬁve-objective
problems, we have used p = 6, so that H = 210 reference
points are obtained. Note that as long as p ≥M is not chosen,
no intermediate point will be created by Das and Dennis’s
systematic approach. For eight-objective problems, even if we
use p = 8 (to have at least one intermediate reference point),
it requires 5,040 reference points. To avoid such a situation,
we use two layers of reference points with small values of p.
On the boundary layer, we use p = 3, so that 120 points are
created. On the inside layer, we use p = 2, so that 36 points
are created. All H = 156 points are then used as reference
points for eight-objective problems. We illustrate this scenario
in Figure 4 for a three-objective problem using p = 2 for
boundary layer and p = 1 for the inside layer. For 10-objective
Fig. 4.
The concept for two-layered reference points (with six points on
the boundary layer (p = 2) and three points on the inside layer (p = 1)) is
shown for a three-objective problem, but are implemented for eight or more
objectives in the simulations here.
problems as well, we use p = 3 and p = 2 for boundary and
inside layers, respectively, thereby requiring a total of H =
220 + 55 or 275 reference points. Similarly, for 15-objective
problems, we use p = 2 and p = 1 for boundary and inside
layers, respectively, thereby requiring H = 120 + 15 or 135
reference points.
3Since no tournament selection is used in NSGA-III, a factor of two would
be adequate as well, but as we re-introduce tournament selection in the
constrained NSGA-III in the sequel paper [52], we keep population size as a
multiple of four here as well to have a uniﬁed algorithm.

```

</details>

<details>
<summary>Page 8: complete visual reference</summary>

![Full source page 8](K:/source/OptunaSharp/docs/papers/paper-gateway/library/4c240f7368799f9d/pages/page-0008.png)

</details>

> Quality note: Text produced by local English OCR; verify wording, mathematics and symbols against the page image.
> Quality note: Image-region OCR is separated from prose; graphic/table transcript semantics require visual review.
> Quality note: Mathematical notation or superscripts require visual review; linear text does not establish fraction, exponent or equation semantics.
> Quality note: An alternate native text view contains characters absent from the ordered view. It is retained separately and indexed, with no inferred position; inspect both views and the page image.
> Quality note: Replacement characters detected in extracted text; inspect fonts/symbols against the page image.

<a id="page-9"></a>

<!-- Page 9 -->

This article has been accepted for publication in a future issue of this journal, but has not been fully edited. Content may change prior to final publication.

9

Table II presents other NSGA-III and MOEA/D parameters
used in this study. MOEA/D requires additional parameters

TABLE II
PARAMETER VALUES USED IN NSGA-III AND TWO VERSIONS OF

MOEA/D. n IS THE NUMBER OF VARIABLES.

Parameters NSGA-III MOEA/D
SBX probability \[28\], pc 1 1
Polynomial mutation prob. \[3\], pm 1/n 1/n
ηc \[28\] 30 20
ηm\[28\] 20 20

which we have set according to the suggestions given by their
developers. The neighborhood size T is set as 20 for both
approaches and additionally the penalty parameter θ for the
MOEA/D-PBI approach is set as 5.

A. Normalized Test Problems

To start with, we use three to 15-objective DTLZ1, DTLZ2,
DTLZ3 and DTLZ4 problems \[53\]. The number of variables
are (M + k −1), where M is number of objectives and k = 5
for DTLZ1, while k = 10 for DTLZ2, DTLZ3 and DTLZ4. In
all these problems, the corresponding Pareto-optimal fronts lie
in fi ∈\[0, 0.5\] for DTLZ1 problem, or \[0, 1\] for other DTLZ
problems. Since they have an identical range of values for each
objective, we call these problems ‘normalized test problems’
in this study. Table III indicates the maximum number of
generations used for each test problem.

Figure 5 shows NSGA-III obtained front for the three-
objective DTLZ1 problem. This particular run is associated
with the median value of IGD performance metric. All 91
points are well distributed on the entire Pareto-optimal set.
Results with the MOEA/D-PBI are shown in Figure 6. It is
clear that MOEA/D-PBI is also able to find a good distribution
of points similar to that of NSGA-III. However, Figure 7 shows
that MOEA/D-TCH is unable to find a uniform distribution of
points. Such a distribution was also reported in the original
MOEA/D study \[10\].

Table III shows that for the DTLZ1 problem NSGA-III
performs slightly better in terms of the IGD metric, followed
by MOEA/D-PBI. For the five-objective DTLZ1 problem,
MOEA/D performs better than NSGA-III, but in 8, 10, and 15-
objective problems, NSGA-III performs better. MOEA/D-TCH
consistently does not perform well in all higher dimensions
of the problem. This observation is similar to that concluded
in the original MOEA/D study \[10\] based on two and three-
objective problems.

For DTLZ2 problems, the performance of MOEA/D-PBI
is consistently better than NSGA-III, however NSGA-III
performs better than MOEA/D-TCH approach. Figures 8, 9
and 10 show the distribution of obtained points for NSGA-
III, MOEA/D-PBI, and MOEA/D-TCH algorithms on three-
objective DTLZ2 problem, respectively. Figures show that the
performances of NSGA-III and MOEA/D-PBI are comparable
to each other, whereas the performance of MOEA/D-TCH is
poor.

Figure 11 shows the variation of the IGD metric value
with function evaluations for NSGA-III and MOEA/D-PBI
approaches for the eight-objective DTLZ2 problem. Average

IGD metric value for 20 runs are plotted. It is clear that both
approaches are able to reduce the IGD value with elapse of
function evaluations.

Similar observation is made for the DTLZ3 problem. This
problem introduces a number of local Pareto-optimal fronts
that provide a stiff challenge for algorithms to come close
to the global Pareto-optimal front. While the performance of
NSGA-III and MOEA/D-PBI are similar, with a slight edge
for MOEA/D-PBI, the performance of MOEA/D-TCH is poor.
However, in some runs, MOEA/D-PBI is unable to get close
to the Pareto-optimal front, as evident from a large value of
IGD metric value. Figure 12 shows variation of the average
IGD metric of 20 runs with function evaluations for NSGA-
III and MOEA/D-PBI approaches. NSGA-III manages to find
a better IGD value than MOEA/D-PBI approach after about
80,000 function evaluations.

Problem DTLZ4 has a biased density of points away from
fM = 0, however the Pareto-optimal front is identical to
that in DTLZ2. The difference in the performances between
NSGA-III and MOEA/D-PBI is clear from this problem.
Both MOEA/D algorithms are not able to find an adequate
distribution of points, whereas NSGA-III algorithm performs
as it did in other problems. Figures 13, 14 and 15 show the
obtained distribution of points on the three-objective DTLZ4
problem. The algorithms are unable to find near f3 = 0 Pareto-
optimal points, whereas the NSGA-III is able to find a set
of well-distributed points on the entire Pareto-optimal front.
These plots are made using the median performed run in
each case. Table III clearly show that the IGD metric values
for NSGA-III algorithm are better than those of MOEA/D
algorithms. Figure 16 shows the value path plot of all obtained
solutions for the 10-objective DTLZ4 problem by NSGA-III.
A spread of solutions over fi ∈\[0, 1\] for all 10 objectives and
a trade-off among them are clear from the plot. In contrast,
Figure 17 shows the value path plot for the same problem
obtained using MOEA/D-PBI. The figure clearly shows that
MOEA/D-PBI is not able to find a widely distributed set of
points for the 10-objective DTLZ4 problem. Points having
larger values of objectives f1 to f7 are not found by MOEA/D-
PBI approach.

The right-most column of Table III presents the performance
of the recently proposed MOEA/D-DE approach \[46\]. The
approach uses differential evolution (DE) instead of SBX and
polynomial mutation operators. Two additional parameters are
introduced in the MOEA/D-DE procedure. First, a maximum
bound (nr) on the number of weight vectors that a child can
be associated to is introduced and is set as nr = 2. Second,
for choosing a mating partner of a parent, a probability (δ)
is set for choosing a neighboring partner and the probability
(1 −δ) for choosing any other population member. Authors
suggested to use δ = 0.9. We use the same values of these
parameters in our study. Table III indicates that this version
of MOEA/D does not perform well on the normalized DTLZ
problems, however it performs better than MOEA/D-PBI on
DTLZ4 problem. Due to its poor performance in general in
these problems, we do not apply it any further.

In the above runs with MOEA/D, we have used SBX
recombination parameter index ηc = 20 (as indicated in

Copyright (c) 2013 IEEE. Personal use is permitted. For any other purposes, permission must be obtained from the IEEE by emailing pubs-permissions@ieee.org.

<details>
<summary>Page 9: complete visual reference</summary>

![Full source page 9](K:/source/OptunaSharp/docs/papers/paper-gateway/library/4c240f7368799f9d/pages/page-0009.png)

</details>

> Quality note: Mathematical notation or superscripts require visual review; linear text does not establish fraction, exponent or equation semantics.

<a id="page-10"></a>

<!-- Page 10 -->

This article has been accepted for publication in a future issue of this journal, but has not been fully edited. Content may change prior to final publication.

10

TABLE III
BEST, MEDIAN AND WORST IGD VALUES OBTAINED FOR NSGA-III AND TWO VERSIONS OF MOEA/D ON M-OBJECTIVE DTLZ1, DTLZ2, DTLZ3

AND DTLZ4 PROBLEMS. BEST PERFORMANCE IS SHOWN IN BOLD.

Problem M MaxGen NSGA-III
DTLZ1 4.880 × 10−4

3 400 1.308 × 10−3

4.880 × 10−3

5.116 × 10−4

5 600 9.799 × 10−4

1.979 × 10−3

2.044 × 10−3

8 750 3.979 × 10−3

8.721 × 10−3

2.215 × 10−3

10 1000 3.462 × 10−3

6.869 × 10−3

2.649 × 10−3

15 1500 5.063 × 10−3

1.123 × 10−2

DTLZ2 1.262 × 10−3

3 250 1.357 × 10−3

2.114 × 10−3

4.254 × 10−3

5 350 4.982 × 10−3

5.862 × 10−3

1.371 × 10−2

8 500 1.571 × 10−2

1.811 × 10−2

1.350 × 10−2

10 750 1.528 × 10−2

1.697 × 10−2

1.360 × 10−2

15 1000 1.726 × 10−2

2.114 × 10−2

DTLZ3 9.751 × 10−4

3 1000 4.007 × 10−3

6.665 × 10−3

3.086 × 10−3

5 1000 5.960 × 10−3

1.196 × 10−2

1.244 × 10−2

8 1000 2.375 × 10−2

9.649 × 10−2
8.849 × 10−3

10 1500 1.188 × 10−2

2.083 × 10−2

1.401 × 10−2

15 2000 2.145 × 10−2

4.195 × 10−2

DTLZ4 2.915 × 10−4

3 600 5.970 × 10−4

4.286 × 10−1

9.849 × 10−4

5 1000 1.255 × 10−3

1.721 × 10−3

5.079 × 10−3

8 1250 7.054 × 10−3

6.051 × 10−1

5.694 × 10−3

10 2000 6.337 × 10−3

1.076 × 10−1

7.110 × 10−3

15 3000 3.431 × 10−1

1.073

MOEA/D-PBI MOEA/D-TCH MOEA/D-DE
4.095 × 10−4 3.296 × 10−2 5.470 × 10−3

1.495 × 10−3 3.321 × 10−2 1.778 × 10−2

4.743 × 10−3 3.359 × 10−2 3.394 × 10−1

3.179 × 10−4 1.124 × 10−1 2.149 × 10−2

6.372 × 10−4 1.129 × 10−1 2.489 × 10−2

1.635 × 10−3 1.137 × 10−1 3.432 × 10−2

3.914 × 10−3 1.729 × 10−1 3.849 × 10−2

6.106 × 10−3 1.953 × 10−1 4.145 × 10−2

8.537 × 10−3 2.094 × 10−1 4.815 × 10−2

3.872 × 10−3 2.072 × 10−1 4.253 × 10−2

5.073 × 10−3 2.147 × 10−1 4.648 × 10−2

6.130 × 10−3 2.400 × 10−1 4.908 × 10−2

1.236 × 10−2 3.237 × 10−1 8.048 × 10−2

1.431 × 10−2 3.438 × 10−1 8.745 × 10−2

1.692 × 10−2 3.634 × 10−1 1.008 × 10−1

5.432 × 10−4 7.499 × 10−2 3.849 × 10−2

6.406 × 10−4 7.574 × 10−2 4.562 × 10−2

8.006 × 10−4 7.657 × 10−2 6.069 × 10−2

1.219 × 10−3 2.935 × 10−1 1.595 × 10−1

1.437 × 10−3 2.945 × 10−1 1.820 × 10−1

1.727 × 10−3 2.953 × 10−1 1.935 × 10−1

3.097 × 10−3 5.989 × 10−1 3.003 × 10−1

3.763 × 10−3 6.301 × 10−1 3.194 × 10−1

5.198 × 10−3 6.606 × 10−1 3.481 × 10−1

2.474 × 10−3 7.002 × 10−1 2.629 × 10−1

2.778 × 10−3 7.266 × 10−1 2.873 × 10−1

3.235 × 10−3 7.704 × 10−1 3.337 × 10−1

5.254 × 10−3 1.000 3.131 × 10−1

6.005 × 10−3 1.084 3.770 × 10−1

9.409 × 10−3 1.120 4.908 × 10−1

9.773 × 10−4 7.602 × 10−2 5.610 × 10−2

3.426 × 10−3 7.658 × 10−2 1.439 × 10−1

9.113 × 10−3 7.764 × 10−2 8.887 × 101

1.129 × 10−3 2.938 × 10−1 1.544 × 10−1

2.213 × 10−3 2.948 × 10−1 2.115 × 10−1

6.147 × 10−3 2.956 × 10−1 8.152 × 101

6.459 × 10−3 6.062 × 10−1 2.607 × 10−1

1.948 × 10−2 6.399 × 10−1 3.321 × 10−1

1.123 6.808 × 10−1 3.923
2.791 × 10−3 7.174 × 10−1 2.549 × 10−1

4.319 × 10−3 7.398 × 10−1 2.789 × 10−1

1.010 8.047 × 10−1 2.998 × 10−1

4.360 × 10−3 1.029 2.202 × 10−1

1.664 × 10−2 1.073 3.219 × 10−1

1.260 1.148 4.681 × 10−1

2.929 × 10−1 2.168 × 10−1 3.276 × 10−2

4.280 × 10−1 3.724 × 10−1 6.049 × 10−2

5.234 × 10−1 4.421 × 10−1 3.468 × 10−1

1.080 × 10−1 3.846 × 10−1 1.090 × 10−1

5.787 × 10−1 5.527 × 10−1 1.479 × 10−1

7.348 × 10−1 7.491 × 10−1 4.116 × 10−1

5.298 × 10−1 6.676 × 10−1 2.333 × 10−1

8.816 × 10−1 9.036 × 10−1 3.333 × 10−1

9.723 × 10−1 1.035 7.443 × 10−1

3.966 × 10−1 7.734 × 10−1 2.102 × 10−1

9.203 × 10−1 9.310 × 10−1 2.885 × 10−1

1.077 1.039 6.422 × 10−1

5.890 × 10−1 1.056 4.500 × 10−1

1.133 1.162 6.282 × 10−1

1.249 1.220 8.477 × 10−1

Copyright (c) 2013 IEEE. Personal use is permitted. For any other purposes, permission must be obtained from the IEEE by emailing pubs-permissions@ieee.org.

<details>
<summary>Page 10: complete visual reference</summary>

![Full source page 10](K:/source/OptunaSharp/docs/papers/paper-gateway/library/4c240f7368799f9d/pages/page-0010.png)

</details>

> Quality note: Mathematical notation or superscripts require visual review; linear text does not establish fraction, exponent or equation semantics.

<a id="page-11"></a>

<!-- Page 11 -->

This article has been accepted for publication in a future issue of this journal, but has not been fully edited. Content may change prior to final publication.

11

![Fig. 6. Obtained solutions by MOEA/D-PBI for DTLZ1.](K:/source/OptunaSharp/docs/papers/paper-gateway/library/4c240f7368799f9d/figures/p0011-figure-02.png)

0.6 f3 0.6 f3 0.6 f3 0.6 f3

0.6f3 0.5 0.6 f3
0.5 0.5 0.5 0.4 0.5
0.4 0.3 0.4
0.4 0.3

0.4 0.2 0.4

0.2 0.1 0.2
0.1
0.3 0

0.3 0 0.3

0 0 0 0

0 0

0.2 0.25 0.25

f2 f1 0.2 0.25 0 0.2 0.25 0.25

0.5 0.5 f2

0.1 0.25 f2 0.5 0.5 f1

0.5 f1

0.1 0.5 0.1

0

0

0.25 0.5 0 0

f1 0.25 f2 0 0

0.5 0 0.25 0.5 0.25 0.5
f1 0.5 0.25 f2 f1 0.5 0.25 f2
0 0

![Fig. 7. Obtained solutions by MOEA/D-TCH for DTLZ1.](K:/source/OptunaSharp/docs/papers/paper-gateway/library/4c240f7368799f9d/figures/p0011-figure-01.png)

![Fig. 5. Obtained solutions by NSGA-III for DTLZ1.](K:/source/OptunaSharp/docs/papers/paper-gateway/library/4c240f7368799f9d/figures/p0011-figure-03.png)

![Fig. 9. Obtained solutions by MOEA/D-PBI for DTLZ2.](K:/source/OptunaSharp/docs/papers/paper-gateway/library/4c240f7368799f9d/figures/p0011-figure-05.png)

Fig. 5. Obtained solutions by NSGA-III for
DTLZ1. Fig. 6. Obtained solutions by MOEA/D-PBI for Fig. 7. Obtained solutions by MOEA/D-TCH for
DTLZ1. DTLZ1.

f3 1.2 f3 1.2 f3

1.2 f3 f3

1 1

1 1 1 1

f3 0.5

0.5 0.5

0.8 0.8 0.8

0

0 0 0

0.6 00 0 0 0 0.6

0.6 0.5 0.5

0.5 0.5 0.5 0.5 f2 f1

f2 f1 f2 f1 1 1

0.4 1 1 1 1 0.4

0.4

0.2 0.2

0.2

0 0

0 0

0.5 0.8 1 0 1
f1 0.6 f2 0 0.5
1 0.2 0.4 0.5 0.8 1 f1 0.5 f2
0 f1 0.4 0.6 f2 1
1 0 0.2 0

![Fig. 8. Obtained solutions by NSGA-III for DTLZ2.](K:/source/OptunaSharp/docs/papers/paper-gateway/library/4c240f7368799f9d/figures/p0011-figure-04.png)

![Fig. 10. Obtained solutions by MOEA/D-TCH for DTLZ2.](K:/source/OptunaSharp/docs/papers/paper-gateway/library/4c240f7368799f9d/figures/p0011-figure-06.png)

![Fig. 11. Variation of IGD metric value with NSGA-III and MOEA/D-PBI for DTLZ2.](K:/source/OptunaSharp/docs/papers/paper-gateway/library/4c240f7368799f9d/figures/p0011-figure-07.png)

![Fig. 12. Variation of IGD metric value with NSGA-III and MOEA/D-PBI for DTLZ3.](K:/source/OptunaSharp/docs/papers/paper-gateway/library/4c240f7368799f9d/figures/p0011-figure-08.png)

Fig. 8. Obtained solutions by NSGA-III for
DTLZ2. Fig. 9. Obtained solutions by MOEA/D-PBI for Fig. 10. Obtained solutions by MOEA/D-TCH for
DTLZ2. DTLZ2.

0 3

10 10

NSGA-III NSGA-III
MOEA/D-PBI MOEA/D-PBI

2

10

-1

10

1

10

IGD

IGD

0

10

-2

10

-1

10

-3 -2

10 10

0 2 4 6 8 0 5 10 15
Function Evaluations x 104 Function Evaluations x 104

Fig. 11. Variation of IGD metric value with NSGA-III and MOEA/D-PBI Fig. 12. Variation of IGD metric value with NSGA-III and MOEA/D-PBI
for DTLZ2. for DTLZ3.

Table II) mainly because this value was chosen in the original
MOEA/D study \[10\]. Next, we investigate MOEA/D-PBI’s
performance with ηc = 30 (which was used with NSGA-III).
Table IV shows that the performance of MOEA/D does not
change much with the above change in ηc value.

Next, we apply NSGA-III and MOEA/D-PBI approaches
to two WFG test problems. The parameter settings are the
same as before. Figures 18 and 19 show the obtained points
on WFG6 problem. The convergence is slightly better with
NSGA-III approach. Similarly, Figures 20 and 21 show a
similar performance comparison of the above two approaches
for WFG7 problem. Table V shows the performances of

the two approaches up to 15-objective WFG6 and WFG7
problems.

Based on the results on three to 15-objective normalized
DTLZ and WFG test problems, it can be concluded that (i)
MOEA/D-PBI performs consistently better than MOEA/D-
TCH approach, (ii) MOEA/D-PBI performs best in some
problems, whereas the proposed NSGA-III approach performs
best in some other problems, and (iii) in a non-uniformly
distributed Pareto-optimal front (like in the DTLZ4 problem),
both MOEA/D approaches fail to maintain a good distribution
of points, while NSGA-III performs well.

Copyright (c) 2013 IEEE. Personal use is permitted. For any other purposes, permission must be obtained from the IEEE by emailing pubs-permissions@ieee.org.



<details>
<summary>Page 11: complete visual reference</summary>

![Full source page 11](K:/source/OptunaSharp/docs/papers/paper-gateway/library/4c240f7368799f9d/pages/page-0011.png)

</details>

> Quality note: Image-region OCR is separated from prose; graphic/table transcript semantics require visual review.
> Quality note: Rotated or vertical text retains geometry and requires visual order review.
> Quality note: Mathematical notation or superscripts require visual review; linear text does not establish fraction, exponent or equation semantics.

<a id="page-12"></a>

<!-- Page 12 -->

This article has been accepted for publication in a future issue of this journal, but has not been fully edited. Content may change prior to final publication.

12

![Fig. 14. Obtained solutions by MOEA/D-PBI for DTLZ4.](K:/source/OptunaSharp/docs/papers/paper-gateway/library/4c240f7368799f9d/figures/p0012-figure-01.png)

1.2 f3 f3 f3 f3 1.2 f3

1.2

1 1 f3

1 1 1 1

0.5

0.5 0.5

0.8 0.8 0.8

0 0

0 0 0 0

0 0 0.6 0

0.6 0.5 0.6

0.5 0.5 0.5 0.5 0.5
f2 1 1 f1 0.4 f2 1 f1 f2 f1

0.4 1 1 1

0.4

0.2

0.2 0.2

0

0

0

0 0

1 0.5 0
0.5 0.6 0.8
0.4 f2 f1 0.8 1 0.5 1
f1 1 0 0.2 1 0.4 0.6 f2 f1 0.6 f20.8
0.2 1 0.2 0.4
0 0

![Fig. 15. Obtained solutions by MOEA/D-TCH for DTLZ4.](K:/source/OptunaSharp/docs/papers/paper-gateway/library/4c240f7368799f9d/figures/p0012-figure-02.png)

![Fig. 13. Obtained solutions by NSGA-III for DTLZ4.](K:/source/OptunaSharp/docs/papers/paper-gateway/library/4c240f7368799f9d/figures/p0012-figure-03.png)

Fig. 13. Obtained solutions by NSGA-III for
DTLZ4. Fig. 14. Obtained solutions by MOEA/D-PBI for Fig. 15. Obtained solutions by MOEA/D-TCH for
DTLZ4. DTLZ4.

![Fig. 16. NSGA-III Solutions are shown using 10-objective value path format for DTLZ4.](K:/source/OptunaSharp/docs/papers/paper-gateway/library/4c240f7368799f9d/figures/p0012-figure-04.png)

1 1

0.8 0.8

0.6 0.6

0.4 0.4

Objective Value

Objective Value

0.2 0.2

0 0
1 2 3 4 5 6 7 8 9 10 1 2 3 4 5 6 7 8 9 10

Objective No. Objective No.

![Fig. 17. MOEA/D-PBI Solutions are shown using 10-objective value path format for DTLZ4.](K:/source/OptunaSharp/docs/papers/paper-gateway/library/4c240f7368799f9d/figures/p0012-figure-05.png)

Fig. 16. NSGA-III Solutions are shown using 10-objective value path Fig. 17. MOEA/D-PBI Solutions are shown using 10-objective value path
format for DTLZ4. format for DTLZ4.

![Fig. 19. MOEA/D-PBI solutions are shown on three-objective WFG6 problem.](K:/source/OptunaSharp/docs/papers/paper-gateway/library/4c240f7368799f9d/figures/p0012-figure-06.png)

1.2 f3 f3 1.2 f3 f3 1.2 f3 f3 1.2 f3 f3

1 1 1 1

1 0.5 1 0.5 1 1 0.5

0.5

0.8 0 0.8 00 0 0.8 0 0.8 0

0 0 0 0 0
0.5 0.5 0.6 0.5 f2 f1 0.5 0.5 0.5 0.5 0.5
0.6 f2 1 1 f1 1 1 0.6 f2 1 1 f1 0.6 f2 1 1 f1

0.4

0.4 0.4 0.4

0.2

0.2 0.2 0.2

0

0 0 0 0

0 0.5 0 0

0.5 1 f1 f2 1 0.5 1 0.5 1

f1 1 0.5 f2 1 0.5 f1 1 0.5 f2 f1 1 0.5 f2
0 0 0 0

Fig. 18. NSGA-III solutions are Fig. 19. MOEA/D-PBI solutions Fig. 20. NSGA-III solutions are Fig. 21. MOEA/D-PBI solutions
shown on three-objective WFG6 prob- are shown on three-objective WFG6 shown on three-objective WFG7 prob- are shown on three-objective WFG7
lem. problem. lem. problem.

Copyright (c) 2013 IEEE. Personal use is permitted. For any other purposes, permission must be obtained from the IEEE by emailing pubs-permissions@ieee.org.

<details>
<summary>Page 12: complete visual reference</summary>

![Full source page 12](K:/source/OptunaSharp/docs/papers/paper-gateway/library/4c240f7368799f9d/pages/page-0012.png)

</details>

> Quality note: Rotated or vertical text retains geometry and requires visual order review.

<a id="page-13"></a>

<!-- Page 13 -->

This article has been accepted for publication in a future issue of this journal, but has not been fully edited. Content may change prior to final publication.

13

TABLE IV

IGD VALUES OF MOEA/D-PBI APPROACH WITH ηc = 30.

Function M MaxGen NSGA-III
DTLZ1 4.880 × 10−4

3 400 1.308 × 10−3

4.880 × 10−3

5.116 × 10−4

5 600 9.799 × 10−4

1.979 × 10−3

2.044 × 10−3

8 750 3.979 × 10−3

8.721 × 10−3

2.215 × 10−3

10 1000 3.462 × 10−3

6.869 × 10−3

2.649 × 10−3

15 1500 5.063 × 10−3

1.123 × 10−2

DTLZ2 1.262 × 10−3

3 250 1.357 × 10−3

2.114 × 10−3

4.254 × 10−3

5 350 4.982 × 10−3

5.862 × 10−3

1.371 × 10−2

8 500 1.571 × 10−2

1.811 × 10−2

1.350 × 10−2

10 750 1.528 × 10−2

1.697 × 10−2

1.360 × 10−2

15 1000 1.726 × 10−2

2.114 × 10−2

DTLZ3 9.751 × 10−4

3 1000 4.007 × 10−3

6.665 × 10−3

3.086 × 10−3

5 1000 5.960 × 10−3

1.196 × 10−2

1.244 × 10−2

8 1000 2.375 × 10−2

9.649 × 10−2
8.849 × 10−3

10 1500 1.188 × 10−2

2.083 × 10−2

1.401 × 10−2

15 2000 2.145 × 10−2

4.195 × 10−2
DTLZ4 2.915 × 10−4

3 600 5.970 × 10−4

4.286 × 10−1

9.849 × 10−4

5 1000 1.255 × 10−3

1.721 × 10−3

5.079 × 10−3

8 1250 7.054 × 10−3

6.051 × 10−1

5.694 × 10−3

10 2000 6.337 × 10−3

1.076 × 10−1
7.110 × 10−3

15 3000 3.431 × 10−1
1.073

TABLE V
IGD VALUES FOR NSGA-III AND MOEA/D-PBI APPROACHES ON THREE

MOEA/D-PBI TO 15-OBJECTIVE WFG PROBLEMS.
4.047 × 10−4

1.797 × 10−3 Problem M MaxGen NSGA-III MOEA/D-PBI
5.699 × 10−3 WFG6 4.828 × 10−3 1.015 × 10−2

2.849 × 10−4 3 400 1.224 × 10−2 3.522 × 10−2

5.717 × 10−4 5.486 × 10−2 1.066 × 10−1

1.647 × 10−3 5.065 × 10−3 8.335 × 10−3

4.053 × 10−3 5 750 1.965 × 10−2 4.230 × 10−2

6.839 × 10−3 4.474 × 10−2 1.058 × 10−1

1.127 × 10−2 1.009 × 10−2 1.757 × 10−2

4.057 × 10−3 8 1500 2.922 × 10−2 5.551 × 10−2

5.251 × 10−3 7.098 × 10−2 1.156 × 10−1

6.300 × 10−3 1.060 × 10−2 9.924 × 10−3

1.301 × 10−2 10 2000 2.491 × 10−2 4.179 × 10−2

1.653 × 10−2 6.129 × 10−2 1.195 × 10−1

2.695 × 10−2 1.368 × 10−2 1.513 × 10−2

4.535 × 10−4 15 3000 2.877 × 10−2 6.782 × 10−2

5.778 × 10−4 6.970 × 10−2 1.637 × 10−1

8.049 × 10−4 WFG7 2.789 × 10−3 1.033 × 10−2

1.096 × 10−3 3 400 3.692 × 10−3 1.358 × 10−2

1.208 × 10−3 4.787 × 10−3 1.926 × 10−2

1.370 × 10−3 8.249 × 10−3 8.780 × 10−3

2.793 × 10−3 5 750 9.111 × 10−3 1.101 × 10−2

3.615 × 10−3 1.050 × 10−2 1.313 × 10−2

5.231 × 10−3 2.452 × 10−2 1.355 × 10−2

2.286 × 10−3 8 1500 2.911 × 10−2 1.573 × 10−2

2.433 × 10−3 6.198 × 10−2 2.626 × 10−2

2.910 × 10−3 3.228 × 10−2 1.041 × 10−2

5.292 × 10−3 10 2000 4.292 × 10−2 1.218 × 10−2

5.952 × 10−3 9.071 × 10−2 1.490 × 10−2

8.413 × 10−3 3.457 × 10−2 7.552 × 10−3

1.001 × 10−3 15 3000 5.450 × 10−2 1.063 × 10−2

5.080 × 10−3 8.826 × 10−2 2.065 × 10−2

1.154 × 10−2

7.191 × 10−4

2.344 × 10−3

6.118 × 10−3

6.285 × 10−3

2.032 × 10−2

1.133
2.848 × 10−3

6.110 × 10−3

4.834 × 10−1

4.647 × 10−3

1.110 × 10−2

1.271

TABLE VI
1.092 × 10−1 BEST, MEDIAN AND WORST IGD AND CONVERGENCE METRIC VALUES
4.277 × 10−1 OBTAINED FOR NSGA-III AND CLASSICAL GENERATIVE METHODS FOR

5.235 × 10−1 THREE-OBJECTIVE DTLZ1 AND DTLZ2 PROBLEMS.

2.342 × 10−1

4.384 × 10−1

7.347 × 10−1

Prob.

FE NSGA-III Generative Method
3.821 × 10−1 IGD GD IGD GD

8.015 × 10−1 4.880×10−4 4.880×10−4 6.400×10−2 1.702×101

9.686 × 10−1 1.308×10−3 6.526×10−4 8.080×10−2 1.808×101

5.083 × 10−1

36,400

DTLZ1

4.880×10−3 7.450×10−4 1.083×10−1 1.848×101

8.321 × 10−1

1.024 1.262×10−3 1.264×10−3 1.113×10−3 9.678×10−5
9.406 × 10−1 1.357×10−3 1.270×10−3 6.597×10−3 1.019×10−4

22,750

DTLZ2

1.157 2.114×10−3 1.274×10−3 9.551×10−3 1.082×10−4
1.219

Copyright (c) 2013 IEEE. Personal use is permitted. For any other purposes, permission must be obtained from the IEEE by emailing pubs-permissions@ieee.org.

<details>
<summary>Page 13: complete visual reference</summary>

![Full source page 13](K:/source/OptunaSharp/docs/papers/paper-gateway/library/4c240f7368799f9d/pages/page-0013.png)

</details>

> Quality note: Rotated or vertical text retains geometry and requires visual order review.
> Quality note: Mathematical notation or superscripts require visual review; linear text does not establish fraction, exponent or equation semantics.

<a id="page-14"></a>

<!-- Page 14 -->

This article has been accepted for publication in a future issue of this journal, but has not been fully edited. Content may change prior to final publication.

14

B. Classical Generative Method

For solving many-objective optimization problems, we re-
quire a set of reference points – either supplied by the user
or systematically constructed as discussed earlier. One may
then ponder how the proposed NSGA-III method will compare
with a classical generative method in which multiple scalarized
single-objective optimization problems can be formulated for
each of the preferred or structured reference points and solved
independently. For this purpose, we minimize the PBI-metric
for each case. Along the reference direction obtained by
joining the ideal point (z∗) to the supplied reference point (¯z)
dictated by a w-vector (a unit vector w = (¯z −z∗)/\|¯z −z∗\|),
the distance (d1) along the w-direction and the distance (d2)
perpendicular to the w-direction are computed for any point
x. A weighted-sum of these two directions is then minimized,
as follows:

Minimizex d1+θd2 = wT f(x)+θ∥f(x)−wT f(x)w∥. (7)

The parameter θ is set to 5 for all reference points \[10\]. In
other words, the above minimization process is likely to find
the intersection point between the reference direction w and
the Pareto-optimal surface, if there exists an intersection. Since
the ideal point (z∗) is required to be known for this method,
we assume origin to be the ideal vector for this problem.

To make a fair comparison, for H reference points, we allo-
cate a maximum of T = FENSGA-III/H (where FENSGA-III is
the total number of solution evaluations needed by NSGA-
III) function evaluations to each optimization process. For
three-objective DTLZ1 and DTLZ2 problems, FENSGA-III =
92×400 or 36,800 was required to find 91 solutions. Therefore,
we allocate 36,800/91 or 405 function evaluations to each
optimization run by fmincon routine of Matlab. A random
solution is used for initialization. Figures 22 and 23 show the
final solutions obtained by this generative method.

The DTLZ1 problem has multiple local fronts and Matlab’s
fmincon routine could not find a Pareto-optimal solution
every time with the allotted number of function evaluations.
Table VI shows the IGD and GD metric (average distance
of obtained points from the closest reference points, a metric
opposite in sense to IGD metric) values. For three-objective
DTLZ2 problem, the generative method is able to find a well-
distributed set of points, as shown in the figure. However, the
table indicates that the distribution of points is not as good
as that obtained by NSGA-III with the number of function
evaluations restricted in this study. The inherent parallel search
of NSGA-III constitutes a more efficient optimization.

C. Scaled Test Problems

To investigate the algorithm’s performance on problems
having differently scaled objective values, we consider DTLZ1
and DTLZ2 problems again, but now we modify them as
follows. Objective fi is multiplied by a factor 10i. To illustrate,
objectives f1, f2 and f3 for the three-objective scaled DTLZ1
problem are multiplied by 100, 101 and 102, respectively.

To handle different scaling of objectives and make the
distances (d1 and d2) along and perpendicular to reference

directions in MOEA/D-PBI approach, we normalize the objec-
tive values using the procedure suggested for MOEA/D-TCH
approach in the original MOEA/D study \[10\]. We also use the
code from MOEA/D web-site \[50\] to obtain the points. Fig-
ures 24, 25, and 26 show the obtained distribution of points us-
ing NSGA-III, MOEA/D-PBI and MOEA/D-TCH approaches,
respectively. It is clear that both MOEA/D approaches with
objective normalization are not able to handle the scaling
involved in the objectives adequately, whereas NSGA-III’s
operators, normalization of objectives, and adaptive hyper-
plane construction process are able to negotiate the scaling of
the objectives quite well. In the scaled problems, IGD metric is
computed by first normalizing the objective values using the
ideal and nadir points of the exact Pareto-optimal front and
then computing the IGD value using the reference points as
before. Resulting IGD values are shown in Table VII. Similar

TABLE VII
BEST, MEDIAN AND WORST IGD VALUES FOR THREE-OBJECTIVE SCALED

DTLZ1 AND DTLZ2 PROBLEMS USING NSGA-III AND MOEA/D
ALGORITHMS. A SCALING FACTOR OF 10i, i = 1, 2, . . . , M, IS USED.

Prob.

MaxGen

NSGA-III MOEA/D MOEA/D
-PBI -TCH

3.853×10−4 2.416×10−2 8.047×10−2

1.214×10−3 2.054×10−1 2.051×10−1

400

DTLZ1

1.103×10−2 4.113×10−1 2.704×10−1

1.347×10−3 4.749×10−2 2.350×10−1

2.069×10−3 3.801×10−1 5.308×10−1

250

DTLZ2

5.284×10−3 5.317×10−1 5.321×10−1

performance is also observed for the scaled DTLZ2 problem,
as shown in Figures 27, 28, and 29 and results are tabulated
in Table VII.

It is interesting to note that the MOEA/D-PBI algorithm
that worked so well in the normalized test problems in the
previous subsection did not perform well for the scaled version
of the same problems. Practical problems are far from being
normalized and the objectives are usually scaled differently.
An efficient optimization algorithm must handle different
scaling of objectives as effectively as shown by NSGA-III.

Next, Table VIII shows the IGD values of the obtained
solutions for 5, 8, 10, and 15-objective scaled DTLZ1 and
DTLZ2 problems. Although difficult to visualize, a small IGD
value in each case refers to a well-distributed set of points.
Due to the poor performance of MOEA/D algorithms in scaled
problems, we did not apply them further to the above higher-
dimensional scaled problems.

D. Convex DTLZ2 Problem

DTLZ1 problem has a linear Pareto-optimal front and
DTLZ2 to DTLZ4 problems have concave Pareto-optimal
fronts. To investigate the performance of NSGA-III and both
versions of MOEA/D on scalable problems having a con-
vex Pareto-optimal front, we suggest a new problem based
on DTLZ2 problem. After the objective values (fi, i =
1, 2, . . ., M) are computed using the original DTLZ2 problem

Copyright (c) 2013 IEEE. Personal use is permitted. For any other purposes, permission must be obtained from the IEEE by emailing pubs-permissions@ieee.org.

<details>
<summary>Page 14: complete visual reference</summary>

![Full source page 14](K:/source/OptunaSharp/docs/papers/paper-gateway/library/4c240f7368799f9d/pages/page-0014.png)

</details>

> Quality note: Rotated or vertical text retains geometry and requires visual order review.
> Quality note: Mathematical notation or superscripts require visual review; linear text does not establish fraction, exponent or equation semantics.

<a id="page-15"></a>

<!-- Page 15 -->

This article has been accepted for publication in a future issue of this journal, but has not been fully edited. Content may change prior to final publication.

15

![Fig. 22. Solutions obtained by classical generative method for DTLZ1. Only the points obtained close to the true Pareto-optimal front are shown.](K:/source/OptunaSharp/docs/papers/paper-gateway/library/4c240f7368799f9d/figures/p0015-figure-01.png)

0.6 f3

1.2 f3 f3

0.6 f3 1
0.5 0.5

1

0.4 0.5
0.4 0.3 0.8
0.2 0
0.1 0 0
0.3 0 0.6
0.5 0.5

0 0 f2 1 f1

1

0.2 0.25 0.25 0.4

f2 f1

0.5 0.5 0.2

0.1

0

0 0

0

0.25 0.5 0.5 0.8 1
f1 0.5 0.25 f2 f1 0.4 0.6 f2
0 1 0.2
0

![Fig. 25. Obtained solutions by MOEA/D-PBI for scaled DTLZ1.](K:/source/OptunaSharp/docs/papers/paper-gateway/library/4c240f7368799f9d/figures/p0015-figure-04.png)

Fig. 22. Solutions obtained by classical generative method for DTLZ1.

![Fig. 23. Solutions obtained by classical generative method for DTLZ2.](K:/source/OptunaSharp/docs/papers/paper-gateway/library/4c240f7368799f9d/figures/p0015-figure-02.png)

Fig. 23. Solutions obtained by classical generative method for DTLZ2.

Only the points obtained close to the true Pareto-optimal front are shown.

60 f3 f3 60 f3 60 f3 f3

60 f3 60
60 50
50 50 40
50 50 40
40 30
30 20

40

40 20 40 20 10
10 0
0 30 0

30 0 30 0 0

0 0 2.5 0.25

0.25 0 20 f2
20 2.5 20 2.5 f1

f2 f1 0.25 5 0.5

5 6 0.5 f2 5 f1 10

0.5

10 10

0

0 0

00 0 0.25 5
0.25 f1 2.5 f2 5 0.25 5 f1 2.5 f2
0.5 0 f1 0.5 2.5 f2 0.5
0 0

![Fig. 24. Obtained solutions by NSGA-III for scaled DTLZ1.](K:/source/OptunaSharp/docs/papers/paper-gateway/library/4c240f7368799f9d/figures/p0015-figure-03.png)

![Fig. 26. Obtained solutions by MOEA/D-TCH for scaled DTLZ1.](K:/source/OptunaSharp/docs/papers/paper-gateway/library/4c240f7368799f9d/figures/p0015-figure-05.png)

![Fig. 27. Obtained solutions by NSGA-III for scaled DTLZ2.](K:/source/OptunaSharp/docs/papers/paper-gateway/library/4c240f7368799f9d/figures/p0015-figure-06.png)

Fig. 24. Obtained solutions by NSGA-III for
scaled DTLZ1. Fig. 25. Obtained solutions by MOEA/D-PBI for Fig. 26. Obtained solutions by MOEA/D-TCH for
scaled DTLZ1. scaled DTLZ1.

f3 f3

120 f3 120 f3 f3 120

100 100 f3

100 100 100 100

50

50 50

80 80 0 80

0 0 0 0
60 0 0 60 5 0.5 0 0
5 0.5 f2 f1 60
f2 f1 10 1 5 0.5
40 10 1 40 f2 f1

10 1

40

20 20

20

0 0

0 0

10 12
0.5 6 f28 0.5 12 00 0.2 f2 10
f1 4 8 10 0.4 0.6 5
1 0 2 f1 1 4 6 f2 f1 0.8 1 0
0 2

![Fig. 28. Obtained solutions by MOEA/D-PBI for scaled DTLZ2.](K:/source/OptunaSharp/docs/papers/paper-gateway/library/4c240f7368799f9d/figures/p0015-figure-07.png)

![Fig. 29. Obtained solutions by MOEA/D-TCH for scaled DTLZ2.](K:/source/OptunaSharp/docs/papers/paper-gateway/library/4c240f7368799f9d/figures/p0015-figure-08.png)

Fig. 27. Obtained solutions by NSGA-III for Fig. 29. Obtained solutions by MOEA/D-TCH for
scaled DTLZ2. Fig. 28. Obtained solutions by MOEA/D-PBI for scaled DTLZ2.
scaled DTLZ2.

## description, we map them as follows:

## fi ← f 4i , i = 1, 2, . . ., M −1,

## fM ← f 2M.

## Thus, the Pareto-optimal surface is given as

## M−1( )

## fM + fi = 1. (8)

i=1

## Figure 30 shows the obtained points on a three-objective convex Pareto-optimal front problem with H = 91 reference

points. As the figure shows, the Pareto-optimal surface is
almost flat at the edges, but changes sharply in the intermediate
region. Although the reference points are uniformly placed
on a normalized hyper-plane making an equal angle to all
objective axes, the distribution of points on the Pareto-optimal
front is not uniform. Despite such a nature of the front, NSGA-
III is able to find a widely distributed set of points on the
edges and also in the intermediate region of the surface. Note
that the range of each objective for the Pareto-optimal set is
identical (within \[0,1\]), hence MOEA/D-PBI is expected to

Copyright (c) 2013 IEEE. Personal use is permitted. For any other purposes, permission must be obtained from the IEEE by emailing pubs-permissions@ieee.org.

<details>
<summary>Page 15: complete visual reference</summary>

![Full source page 15](K:/source/OptunaSharp/docs/papers/paper-gateway/library/4c240f7368799f9d/pages/page-0015.png)

</details>

> Quality note: Mathematical notation or superscripts require visual review; linear text does not establish fraction, exponent or equation semantics.

<a id="page-16"></a>

<!-- Page 16 -->

This article has been accepted for publication in a future issue of this journal, but has not been fully edited. Content may change prior to final publication.

![Fig. 31. Obtained solutions by MOEA/D-PBI for the convex Pareto-optimal front problem.](K:/source/OptunaSharp/docs/papers/paper-gateway/library/4c240f7368799f9d/figures/p0016-figure-01.png)

16

f3 1.2 f3 f3 f3

f3 1.2 1.2 1.2 f3

1.2

1 1 1 1 1

1

0.8 0.8 0.8

0.8 0.5

0.8 0.6 0.6 0.6

0.6 0.4 0.6 0.4 0

0.4 0

0.2 0

0.2 0.5

0.4 0.2 f2 0.5
0 0.4 0 1 1 f1

0 0 0 0
0.2 0.5 f2 f1 0.5 0.5 0
1 1 0.5 f1 0
0.2 f2 1 1

00 0.5
0.5 f2 1 0 f1 1.2
f1 0.5 0 1 0.8 1
1 0.5 1 0.4 0.6 f2
0 f1 1 0.5 f2 0.2
0 0

![Fig. 32. Obtained solutions by MOEA/D-TCH for the convex Pareto-optimal front problem.](K:/source/OptunaSharp/docs/papers/paper-gateway/library/4c240f7368799f9d/figures/p0016-figure-02.png)

![Fig. 30. Obtained solutions by NSGA-III for the convex Pareto-optimal front problem.](K:/source/OptunaSharp/docs/papers/paper-gateway/library/4c240f7368799f9d/figures/p0016-figure-03.png)

Fig. 30. Obtained solutions by NSGA-III for the
convex Pareto-optimal front problem. Fig. 31. Obtained solutions by MOEA/D-PBI for Fig. 32. Obtained solutions by MOEA/D-TCH for
the convex Pareto-optimal front problem. the convex Pareto-optimal front problem.

TABLE VIII
BEST, MEDIAN AND WORST IGD VALUES FOR SCALED M-OBJECTIVE

DTLZ1 AND DTLZ2 PROBLEMS.

Problem M Scaling MaxGen NSGA-III
factor
DTLZ1 3.853 × 10−4

3 10i 400 1.214 × 10−3

1.103 × 10−2

1.099 × 10−3

5 10i 600 2.500 × 10−3

3.921 × 10−2

4.659 × 10−3

8 3i 750 1.051 × 10−2

1.167 × 10−1

3.403 × 10−3

10 2i 1000 5.577 × 10−3

3.617 × 10−2

3.450 × 10−3

15 1.2i 1500 6.183 × 10−3

1.367 × 10−2

DTLZ2 1.347 × 10−3

3 10i 250 2.069 × 10−3

5.284 × 10−3

1.005 × 10−2

5 10i 350 2.564 × 10−2

8.430 × 10−2

1.582 × 10−2

8 3i 500 1.788 × 10−2

2.089 × 10−2

2.113 × 10−2

10 3i 750 3.334 × 10−2

2.095 × 10−1

2.165 × 10−2

15 2i 1000 2.531 × 10−2

4.450 × 10−2

perform better. As shown in Figure 31, a nicely distributed
set of points are found in the middle part of the front, but the
algorithm fails to find points on the boundary of the Pareto-
optimal front. This happens due to the very small slope of the
surface at the boundary region. The use of penalty parameter
value θ = 5 finds a non-boundary point to have a better PBI
metric value for the boundary reference directions. A larger
value of θ may allow MOEA/D-PBI to find the exact boundary
points, thereby requiring to do a parametric study with θ.
However, the advantage of no additional parameter in NSGA-
III becomes clear from this problem. MOEA/D-TCH performs
poorly again, as shown in Figure 32.

Table IX shows the best, median and worst IGD values of
20 independent runs (obtained using equation 6) for all three
methods. Clearly, NSGA-III performs the best by finding a set
of points having an order of magnitude smaller IGD values in
three, five, eight, 10, and 15-objective versions of the convex
DTLZ2 problem.

TABLE IX
BEST, MEDIAN AND WORST IGD VALUES FOR THE CONVEX DTLZ2

PROBLEM.

M MaxGen NSGA-III MOEA/D-PBI MOEA/D-TCH
2.603 × 10−3 3.050 × 10−2 6.835 × 10−2

3 250 4.404 × 10−3 3.274 × 10−2 6.871 × 10−2

8.055 × 10−3 3.477 × 10−2 6.912 × 10−2

7.950 × 10−3 1.082 × 10−1 1.211 × 10−1

5 750 1.341 × 10−2 1.103 × 10−1 1.223 × 10−1

1.917 × 10−2 1.117 × 10−1 1.235 × 10−1

2.225 × 10−2 2.079 × 10−1 2.235 × 10−1

8 2000 2.986 × 10−2 2.086 × 10−1 2.396 × 10−1

4.234 × 10−2 2.094 × 10−1 2.490 × 10−1

7.389 × 10−2 2.117 × 10−1 2.373 × 10−1

10 4000 9.126 × 10−2 2.119 × 10−1 2.472 × 10−1

1.051 × 10−1 2.120 × 10−1 2.514 × 10−1

2.169 × 10−2 3.681 × 10−1 3.851 × 10−1

15 4500 2.769 × 10−2 3.682 × 10−1 3.911 × 10−1

4.985 × 10−2 3.683 × 10−1 3.972 × 10−1

To demonstrate the distribution of obtained points on a
higher-objective problem, we show the obtained points of
the median performed run of the 15-objective convex DTLZ2
problem on a value path using NSGA-III and MOEA/D-PBI
approaches in Figures 33 and 34, respectively. It is clear from
the plots that NSGA-III is able to find a good spread of
solutions in the entire range of Pareto-optimal front (fi ∈\[0, 1\]
for all i), but MOEA/D-PBI is unable to find solutions having
larger objective values.

Before we evaluate the proposed NSGA-III approach fur-
ther, we highlight a couple of properties of MOEA/D-TCH and
MOEA/D-PBI approaches that become clear form the above
simulation results:

1\) As observed in three-objective results with MOEA/D-

TCH above (Figure 7, for example), it ends up gen-
erating a non-uniform distribution of points. This is
due to the way the Tchebysheff metric is constructed.
Moreover, multiple weight vectors produce an identical

Copyright (c) 2013 IEEE. Personal use is permitted. For any other purposes, permission must be obtained from the IEEE by emailing pubs-permissions@ieee.org.

<details>
<summary>Page 16: complete visual reference</summary>

![Full source page 16](K:/source/OptunaSharp/docs/papers/paper-gateway/library/4c240f7368799f9d/pages/page-0016.png)

</details>

> Quality note: Mathematical notation or superscripts require visual review; linear text does not establish fraction, exponent or equation semantics.

<a id="page-17"></a>

<!-- Page 17 -->

This article has been accepted for publication in a future issue of this journal, but has not been fully edited. Content may change prior to final publication.

17

![Fig. 33. NSGA-III Solutions are shown using 15-objective value path format for the convex problem.](K:/source/OptunaSharp/docs/papers/paper-gateway/library/4c240f7368799f9d/figures/p0017-figure-02.png)

1 1

0.8 0.8

0.6 0.6

0.4 0.4

Objective Value

Objective Value

0.2 0.2

0 0
1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15

Objective No. Objective No.

![Fig. 34. MOEA/D-PBI Solutions are shown using 10-objective value path format for the convex problem.](K:/source/OptunaSharp/docs/papers/paper-gateway/library/4c240f7368799f9d/figures/p0017-figure-01.png)

Fig. 33. NSGA-III Solutions are shown using 15-objective value path Fig. 34. MOEA/D-PBI Solutions are shown using 10-objective value path
format for the convex problem. format for the convex problem.

extreme solution, thereby wasting in computational ef-
forts.
2\) Interestingly, depending on the parameter θ, MOEA/D-

PBI is potentially capable of generating dominated solu-
tions. This certainly results in a waste of computational
effort.
Although special care can be taken to reduce the chance of
above, they are important for one to be aware while working
with MOEA/D-PBI or MOEA/D-TCH approaches.

VI. FURTHER INVESTIGATIONS OF NSGA-III

Having performed well on all test problems in above sec-
tions, we now investigate NSGA-III’s performance in certain
special types of many-objective problem solving tasks.

A. Finding a Preferred Part of Pareto-front

In many-objective optimization problems, the user may
not always be interested in finding the entire Pareto-optimal
front. Practically speaking, an user is often interested in a
preferred part of the Pareto-optimal front. In such a scenario,
the user may represent his/her preferred region using a few
representative reference points (or aspiration points). The aim
in such a many-objective optimization task is to find the
Pareto-optimal points that are in some sense closest to the
supplied reference points.

Recall from Section IV-B that the proposed NSGA-III
algorithm can also be applied with a set of H user-supplied
reference points, instead of Das and Dennis’s structured refer-
ence points used so far. Here, we demonstrate the working of
NSGA-III for scaled DTLZ1 and DTLZ2 problems for a few
user-supplied preference information. In such a case, in order
to determine the right normalized hyper-plane, by default, we
supply M additional reference points, one at each objective
axis at an intercept of unity. The presence of these extreme
points will attempt to keep extreme Pareto-optimal points,
thereby forming a proper normalized hyper-plane needed to
have appropriate scaling of the objectives.

For both scaled DTLZ1 and DTLZ2 problems, we supply
10 reference points in the middle of the normalized hyper-
plane (scenario 1, as shown in Figure 35). As mentioned, three
additional reference points (1, 0, 0)T, (0, 1, 0)T and (0, 0, 1)T

are also included for the execution of the algorithm, thereby
making a total of 13 supplied reference points. Figure 37

![Fig. 35. Reference points used for Scenario 1 on normalized hyper- plane.](K:/source/OptunaSharp/docs/papers/paper-gateway/library/4c240f7368799f9d/figures/p0017-figure-03.png)

![Fig. 36. Reference points used for Scenario 2 on normalized hyper- plane.](K:/source/OptunaSharp/docs/papers/paper-gateway/library/4c240f7368799f9d/figures/p0017-figure-04.png)

Fig. 35. Reference points used Fig. 36. Reference points used
for Scenario 1 on normalized hyper- for Scenario 2 on normalized hyper-
plane. plane.

shows the points obtained by NSGA-III on the scaled Pareto-
optimal front of DTLZ1 problem. Figure 38 shows 10 obtained
points for the scaled DTLZ2 problem for scenario 1. Next, we
supply a set of 10 reference points on a different part of the
Pareto-optimal, as shown as scenario 2 in Figure 36. Figure 38
shows the obtained points on the scaled DTLZ2 problem. In
both cases, a set of 10 near-Pareto-optimal points are obtained
close to where the reference points were supplied.

Best, median and worst IGD values for the obtained points
for three and 10-objective scaled DTLZ1 and DTLZ2 problems
are shown in Table X. The IGD values are computed by
normalizing the obtained objective values by theoretical ideal
and nadir points and by projecting 10 supplied reference
points on the normalized Pareto-optimal front. A small IGD
value in each case ensures the convergence and diversity of
the obtained solutions. The success of NSGA-III in finding
just 10 Pareto-optimal points on 10-objective scaled problems
indicates its potential use in finding a handful of preferred
solutions in a many-objective practical optimization problem.

B. Randomly generated preferred reference points

The supplied preferred reference points considered in the
previous subsection are structured. In fact, they were created
using Das and Dennis’s \[48\] structured reference point creation
procedure. Here, we investigate NSGA-III’s ability to deal
with a set of randomly created reference points. The procedure
used here is the same as in previous subsection – M additional
axis points are included for NSGA-III to adaptively create its
hyper-plane.

Copyright (c) 2013 IEEE. Personal use is permitted. For any other purposes, permission must be obtained from the IEEE by emailing pubs-permissions@ieee.org.

<details>
<summary>OCR transcript from a graphic region; inspect the source image for layout</summary>

```text
15 1 5
```

</details>

<details>
<summary>OCR transcript from a graphic region; inspect the source image for layout</summary>

```text
0.5
```

</details>

<details>
<summary>OCR transcript from a graphic region; inspect the source image for layout</summary>

```text
0.5
```

</details>

<details>
<summary>OCR transcript from a graphic region; inspect the source image for layout</summary>

```text
0
0 0 0
0 [}
```

</details>

<details>
<summary>OCR transcript from a graphic region; inspect the source image for layout</summary>

```text
0.5 0.5
```

</details>

<details>
<summary>OCR transcript from a graphic region; inspect the source image for layout</summary>

```text
0.5 0.5
```

</details>

<details>
<summary>OCR transcript from a graphic region; inspect the source image for layout</summary>

```text
f,2 11 f.! f2 11 f1
```

</details>

<details>
<summary>Page 17: complete visual reference</summary>

![Full source page 17](K:/source/OptunaSharp/docs/papers/paper-gateway/library/4c240f7368799f9d/pages/page-0017.png)

</details>

> Quality note: Text produced by local English OCR; verify wording, mathematics and symbols against the page image.
> Quality note: Image-region OCR is separated from prose; graphic/table transcript semantics require visual review.
> Quality note: Rotated or vertical text retains geometry and requires visual order review.
> Quality note: Mathematical notation or superscripts require visual review; linear text does not establish fraction, exponent or equation semantics.

<a id="page-18"></a>

<!-- Page 18 -->

![Complete page reference; figure boundary uncertain. Full source page; figure boundaries require review.](K:/source/OptunaSharp/docs/papers/paper-gateway/library/4c240f7368799f9d/pages/page-0018.png)

This article has been accepted for publication in a future issue of this journal, but has not been fully edited. Content may change prior to final publication.

18

f3 f3

60 f3 f3 120

f3 120

60 100
50 50 100 100

100 f3

40 50 50
40 30 80

80 0

20 0 0 0
10 0 0

30 60 60 5 0.5

0 5 0.5 f1
f2 f1 f2

0 0 10 1 10 1

20 40 40

2.5 0.25

f2 f1

10 5 0.5 20 20

0 0 0

0 0 0

5 10 10

0.25 0.5 f2 0.5 f2
f1 2.5 f2 f1 5 f1 5

0.5 1 1
0 0 0

Fig. 37. Preferred solutions (scenario 1) obtained Fig. 38. Preferred solutions (scenario 1) obtained Fig. 39. Preferred solutions (scenario 2) obtained
by NSGA-III for three-objective DTLZ1. by NSGA-III for three-objective DTLZ2. by NSGA-III for three-objective DTLZ2.

60 f3 f3 f3 f3

60 120 100

50 40

100 50

40 20

80

0 0

30 0 0

0 0 60 5 0.5

20 2.5 0.25 f2 10 1 f1

f2 f1 40

5 0.5

10 20

0 0
0 0 10

5 f2

0.25 f2 0.5 5
f1 2.5 f1

0.5 1
0 0

Fig. 40. Preferred solutions obtained by NSGA-III for three-objective Fig. 41. Preferred solutions obtained by NSGA-III for three-objective
DTLZ1. DTLZ2.

For decision-making purposes, only a few trade-off points
can be analyzed, even for a many-objective problem. Here,
we investigate if the proposed NSGA-III is able to find only
five Pareto-optimal points in a preferred region for three
and 10-objective scaled DTLZ1 and DTLZ2 problems. Five
random points are supplied in the intermediate region of the
normalized hyper-plane (within zi ∈\[0.4, 0.6\] for all i).
Figures 40 and 41 show the obtained points on scaled three-
objective DTLZ1 and DTLZ2 problems, respectively. Table XI
shows the IGD values for three and 10-objective problems.
In each case, a small IGD value indicates that NSGA-III
is adequate to successfully find the desired solutions. This
application shows promise in applying NSGA-III to many-
objective problems (10-objective or like) with a handful (five
or like) of preferred reference points – a matter which is of
great importance to practitioners.

C. Small population size

Previous subsections have shown that NSGA-III can work
with a small number of reference points in order to find a
few Pareto-optimal points in a preferred region. In previous
sections, we have used a population size that is almost equal to
the number of reference points. A natural question then arises:
‘Can NSGA-III work well with a small population size?’. We
investigate this aspect here.

Table XII tabulates the number of reference points (H)
and corresponding layer-wise p and the maximum number of
generations considered for different many-objective DTLZ2
problems. We perform this study for three, five, and 10-
objective scaled DTLZ2 problems. Identical scaling factors as
those shown in Table VIII are used here. In each case, the pop-
ulation size (N) is always kept as the smallest number multiple
of four and greater than the number of reference points (H).
A larger number of generations is kept for higher objective
problems. The chosen maximum number of generations are
also tabulated in the table.

For the three-objective DTLZ2 problem, when we are
interested in finding 10 well-distributed Pareto-optimal points,
we find that a population of size 12 is adequate to find all
10 Pareto-optimal points after 250 generations (Figure 42).
For the 10-objective problem, if we desire 65 reference points
(with p = 2), we find that a population of size 68 is adequate to
find all 65 solutions after 1,000 generations. The best, median
and worst IGD values are shown in the table for three, five
and 10-objective DTLZ2 problems. These results show that
NSGA-III can work with a small population size due to its
focus and balanced emphasis for finding near Pareto-optimal
points corresponding to a supplied set of reference points.

Copyright (c) 2013 IEEE. Personal use is permitted. For any other purposes, permission must be obtained from the IEEE by emailing pubs-permissions@ieee.org.

<details>
<summary>Page 18: complete visual reference</summary>

![Full source page 18](K:/source/OptunaSharp/docs/papers/paper-gateway/library/4c240f7368799f9d/pages/page-0018.png)

</details>

> Quality note: Mathematical notation or superscripts require visual review; linear text does not establish fraction, exponent or equation semantics.

<a id="page-19"></a>

<!-- Page 19 -->

![Complete page reference; figure boundary uncertain. Full source page; figure boundaries require review.](K:/source/OptunaSharp/docs/papers/paper-gateway/library/4c240f7368799f9d/pages/page-0019.png)

This article has been accepted for publication in a future issue of this journal, but has not been fully edited. Content may change prior to final publication.

19

TABLE X
BEST, MEDIAN AND WORST IGD VALUES FOR SCALED DTLZ1 AND

DTLZ2 PROBLEMS WITH 10 REFERENCE POINTS.

Function M N MaxGen NSGA-III
DTLZ1 4.945 × 10−4

3 28 750 4.076 × 10−3

1.898 × 10−2

1.671 × 10−3

(Scenario 1) 10 100 2000 5.792 × 10−3

(Figure 35) 1.212 × 10−1

DTLZ1 2.742 × 10−4

3 28 750 2.875 × 10−3

1.741 × 10−2

4.194 × 10−3

(Scenario 2) 10 100 2000 7.393 × 10−3

(Figure 36) 5.799 × 10−2

DTLZ2 6.459 × 10−4

3 28 250 1.473 × 10−3

3.883 × 10−3

2.840 × 10−3

(Scenario 1) 10 100 1500 4.907 × 10−3

(Figure 35) 1.074 × 10−2

DTLZ2 1.327 × 10−3

3 28 250 2.706 × 10−3

8.099 × 10−3

1.576 × 10−2

(Scenario 2) 10 100 1500 2.548 × 10−2

(Figure 36) 5.893 × 10−2

TABLE XI
BEST, MEDIAN AND WORST IGD VALUES FOR SCALED DTLZ1 AND
DTLZ2 PROBLEMS WITH RANDOMLY SUPPLIED REFERENCE POINTS.

Problem M N MaxGen NSGA-III
DTLZ1 1.621 × 10−3

3 28 600 9.870 × 10−3

3.186 × 10−1

3.244 × 10−3

10 100 2000 8.227 × 10−3

1.688 × 10−2

DTLZ2 5.763 × 10−4

3 28 250 1.951 × 10−3

6.766 × 10−3

1.317 × 10−2

10 100 1500 2.463 × 10−2

6.439 × 10−2

D. Nadir Point Estimation

In multi-objective optimization problems, the nadir point
is important to be found for various reasons. A nadir point
is constructed from the worst objective values for the entire
Pareto-optimal set. Thus, along with the ideal point, nadir
point can be used to normalize the objective functions for
a systematic and reliable execution of many classical and
non-classical optimization methods. In another study, a nadir
point estimation strategy was proposed based on a bilevel
evolutionary approach \[54\]. In that study, three to 20-objective
DTLZ1 and DTLZ2 problems were considered for estimating
the nadir point. Since studies in the previous two subsections
have shown amply that NSGA-III can be used to find a few
Pareto-optimal solutions even in 10-objective problems, here
we apply NSGA-III to find only the extreme Pareto-optimal
points so that the nadir point can be estimated accurately.

For this purpose, we suggest using M reference points, one
on each objective axis, located at unity. We apply NSGA-III to

TABLE XII
MINIMUM POPULATION SIZE AND BEST, MEDIAN AND WORST IGD

VALUES OF THE OBTAINED SOLUTIONS FOR THE SCALED DTLZ2

PROBLEM.

M (pB, pI) H MaxGen N NSGA-III
1.464 × 10−3

3 (3,0) 10 250 12 4.206 × 10−3

1.524 × 10−1

4.952 × 10−3

5 (2,1) 20 500 20 8.537 × 10−3

1.195 × 10−1

8.641 × 10−3

10 (2,1) 65 1,000 68 1.155 × 10−2

1.842 × 10−2

f3 f3

120

100

100

50

80

0

0 0
60 5 0.5
f2 10 1 f1

40

20

0

0

0.5

f1 10 12
1 6 8
0 2 4 f2

Fig. 42. Obtained points using NSGA-III for the scaled DTLZ2 with a small
population size of 12.

three to 20-objective DTLZ1 and DTLZ2 problems and record
the overall function evaluations needed to find the true nadir
point with the following termination condition. When the error
value (E) computed as below

\*
\+ M !znad "2
+( −zest
E = , i i , (9)
i=1 znadi −z∗i

is less than a threshold (0.01, used here), the run is terminated.
Here, z∗, znad and zest are the ideal point, the true nadir
point, and the estimated nadir point by NSGA-III, respectively.
Table XIII presents the results and compares them with the
evaluations needed to find the nadir point with an identical
termination condition in the previous study \[54\]. In most cases,
NSGA-III requires smaller number of function evaluations
than the previous procedure to locate the nadir point reliably.

Figure 43 shows the value path plot of obtained points
for estimating the nadir point for the 10-objective DTLZ2
problems. Understandably, the points near the extreme of the
Pareto-optimal front are found. From this plot, the nadir point
of the problem is estimated to be (1, 1, 1, 1, 1, 1, 1, 1, 1, 1)T.

VII. TWO PROBLEMS FROM PRACTICE

After solving a number of test problems, we now apply
NSGA-III to a couple of engineering design optimization
problems. The first problem has three objectives and the
second one has nine objectives.

Copyright (c) 2013 IEEE. Personal use is permitted. For any other purposes, permission must be obtained from the IEEE by emailing pubs-permissions@ieee.org.

<details>
<summary>Page 19: complete visual reference</summary>

![Full source page 19](K:/source/OptunaSharp/docs/papers/paper-gateway/library/4c240f7368799f9d/pages/page-0019.png)

</details>

> Quality note: Mathematical notation or superscripts require visual review; linear text does not establish fraction, exponent or equation semantics.

<a id="page-20"></a>

<!-- Page 20 -->

![Complete page reference; figure boundary uncertain. Full source page; figure boundaries require review.](K:/source/OptunaSharp/docs/papers/paper-gateway/library/4c240f7368799f9d/pages/page-0020.png)

This article has been accepted for publication in a future issue of this journal, but has not been fully edited. Content may change prior to final publication.

20

TABLE XIII
NADIR POINT ESTIMATION BY NSGA-III IN THREE TO 20-OBJECTIVE

DTLZ1 AND DTLZ2 PROBLEMS.

Problem M NSGA-III NSGA-II \[54\]
DTLZ1 3 7, 400 18, 800
16, 660 26, 500
72, 180 45, 700
5 25, 000 35, 300
62, 900 58, 400
152, 040 107, 100
10 58, 640 239, 800
259, 760 274, 200
483, 880 358, 000
15 356, 460 750, 000
589, 140 1, 100, 000
1, 200, 000 1, 800, 000
20 274, 800 2, 600, 000
956, 960 3, 500, 000
1, 600, 000 4, 000, 000
DTLZ2 3 1, 368 4, 100
1, 920 4, 900
2, 532 5, 500
5 4, 320 9, 400
6, 900 11, 400
13, 000 14, 200
10 31, 400 77, 600
52, 840 92, 800
76, 080 128, 000
15 153, 540 346, 500
213, 060 429, 000
316, 860 810, 000
20 309, 440 1, 089, 000
589, 520 1, 415, 000
734, 400 2, 102, 000

1

0.8

0.6

0.4

Objective Value

0.2

0
1 2 3 4 5 6 7 8 9 10

Objective No.

Fig. 43. A value path plot for the 10-objective DTLZ2 problem shows that
NSGA-III is able to find the extreme objective values (zero and one) in each
objective.

A. Crash-worthiness Design of Vehicles for Complete Trade-
off Front

This problem aims at structural optimization of the frontal
structure of vehicle for crash-worthiness \[55\]. Thickness of
five reinforced members around the frontal structure are
chosen as design variables, while mass of vehicle, deceler-
ation during the full frontal crash (which is proportional to
biomechanical injuries caused to the occupants) and toe board
intrusion in the offset-frontal crash (which accounts for the
structural integrity of the vehicle) are taken as objectives.
Mathematical formulation for the three objectives can be found
in the original study \[55\].

All objectives are to be minimized. For this problem, we
choose p = 16 so that there are H = C(3−1+16, 16) or 153

structured reference points. The reference points are initialized
on the entire normalized hyper-plane in the three-objective
space. NSGA-III is applied with 156 population size and
run for 200 generations. Other parameters are the same as
before. Figure 44 shows the obtained points with an open
circle. The non-uniform scaling of objectives is clear from

Fig. 44. 60 solutions are found on the entire front on the three-objective
crush-worthiness problem.

the figure. Although 153 reference points were used, only 60
of them have representative solutions on the final front. Other
93 reference do not correspond to any Pareto-optimal point,
thereby indicating possible ‘holes’ in the Pareto-optimal front
correspond to remaining reference points. The ideal point (z∗)
and the nadir point (znad) are estimated from the obtained
front.

To investigate the nature of the true Pareto-optimal front,
next, we create 7,381 reference points zref,i (i = 1, . . . , 7381)
with p = 120. Using the ideal and nadir points obtained as
above, we normalize the objectives and then solve the achieve-
ment scalarization function (ASF) (equation 4) corresponding
to each weight vector wi = zref,i. Each ASF is minimized
using Matlab’s fmincon routine. Resulting 7,381 points are
collected and dominated points are removed. We noticed that
only 4,450 points are non-dominated. These non-dominated
points are shown in Figure 44 with a dot. It is clear that
the NSGA-III obtained set of 60 points are widely-distributed
on the entire front. Interestingly, 4,450 trade-off points reveal
the complex nature of Pareto-optimal front (having holes and
varying density of points) for this practical problem and the
ability of NSGA-III to find a few widely-distributed points on
the entire front.

To investigate the closeness of NSGA-III points with the
classically optimized solutions, we compute the convergence
metric (average distance of NSGA-III points from the closest
fmincon-optimized points) and the minimum, median and
maximum convergence metric values are found be 0.0019,
0.0022, and 0.0026, respectively. Since these values are small,
they indicate that obtained NSGA-III solutions are close to the
classically optimized front. This problem demonstrates the use
of NSGA-III in finding a representative set of points on the
entire Pareto-optimal front on problems having practical diffi-
culties (such as differences in scaling of objectives, potential
holes in the Pareto-optimal front, etc.).

Copyright (c) 2013 IEEE. Personal use is permitted. For any other purposes, permission must be obtained from the IEEE by emailing pubs-permissions@ieee.org.

<details>
<summary>OCR transcript from a graphic region; inspect the source image for layout</summary>

```text
0.25 f f24a)
```

</details>

<details>
<summary>OCR transcript from a graphic region; inspect the source image for layout</summary>

```text
0.2 (SEL
```

</details>

<details>
<summary>OCR transcript from a graphic region; inspect the source image for layout</summary>

```text
0.15 .
```

</details>

<details>
<summary>OCR transcript from a graphic region; inspect the source image for layout</summary>

```text
0.1 Rata
```

</details>

<details>
<summary>OCR transcript from a graphic region; inspect the source image for layout</summary>

```text
0.05 2
```

</details>

<details>
<summary>OCR transcript from a graphic region; inspect the source image for layout</summary>

```text
1660
```

</details>

<details>
<summary>OCR transcript from a graphic region; inspect the source image for layout</summary>

```text
0 1680
6 7 f
8 f, 9 10 1 100
```

</details>

<details>
<summary>Complete alternate PDF text layer</summary>

```text
Copyright (c) 2013 IEEE. Personal use is permitted. For any other purposes, permission must be obtained from the IEEE by emailing pubs-permissions@ieee.org.
This article has been accepted for publication in a future issue of this journal, but has not been fully edited. Content may change prior to final publication.
20
TABLE XIII
NADIR POINT ESTIMATION BY NSGA-III IN THREE TO 20-OBJECTIVE
DTLZ1 AND DTLZ2 PROBLEMS.
Problem
M
NSGA-III
NSGA-II [54]
DTLZ1
3
7, 400
18, 800
16, 660
26, 500
72, 180
45, 700
5
25, 000
35, 300
62, 900
58, 400
152, 040
107, 100
10
58, 640
239, 800
259, 760
274, 200
483, 880
358, 000
15
356, 460
750, 000
589, 140
1, 100, 000
1, 200, 000
1, 800, 000
20
274, 800
2, 600, 000
956, 960
3, 500, 000
1, 600, 000
4, 000, 000
DTLZ2
3
1, 368
4, 100
1, 920
4, 900
2, 532
5, 500
5
4, 320
9, 400
6, 900
11, 400
13, 000
14, 200
10
31, 400
77, 600
52, 840
92, 800
76, 080
128, 000
15
153, 540
346, 500
213, 060
429, 000
316, 860
810, 000
20
309, 440
1, 089, 000
589, 520
1, 415, 000
734, 400
2, 102, 000
1
2
3
4
5
6
7
8
9
10
0
0.2
0.4
0.6
0.8
1
Objective Value
Objective No.
Fig. 43.
A value path plot for the 10-objective DTLZ2 problem shows that
NSGA-III is able to ﬁnd the extreme objective values (zero and one) in each
objective.
A. Crash-worthiness Design of Vehicles for Complete Trade-
off Front
This problem aims at structural optimization of the frontal
structure of vehicle for crash-worthiness [55]. Thickness of
ﬁve reinforced members around the frontal structure are
chosen as design variables, while mass of vehicle, deceler-
ation during the full frontal crash (which is proportional to
biomechanical injuries caused to the occupants) and toe board
intrusion in the offset-frontal crash (which accounts for the
structural integrity of the vehicle) are taken as objectives.
Mathematical formulation for the three objectives can be found
in the original study [55].
All objectives are to be minimized. For this problem, we
choose p = 16 so that there are H =
#3−1+16
16
$
or 153
structured reference points. The reference points are initialized
on the entire normalized hyper-plane in the three-objective
space. NSGA-III is applied with 156 population size and
run for 200 generations. Other parameters are the same as
before. Figure 44 shows the obtained points with an open
circle. The non-uniform scaling of objectives is clear from
Fig. 44.
60 solutions are found on the entire front on the three-objective
crush-worthiness problem.
the ﬁgure. Although 153 reference points were used, only 60
of them have representative solutions on the ﬁnal front. Other
93 reference do not correspond to any Pareto-optimal point,
thereby indicating possible ‘holes’ in the Pareto-optimal front
correspond to remaining reference points. The ideal point (z∗)
and the nadir point (znad) are estimated from the obtained
front.
To investigate the nature of the true Pareto-optimal front,
next, we create 7,381 reference points zref,i (i = 1, . . . , 7381)
with p = 120. Using the ideal and nadir points obtained as
above, we normalize the objectives and then solve the achieve-
ment scalarization function (ASF) (equation 4) corresponding
to each weight vector wi = zref,i. Each ASF is minimized
using Matlab’s fmincon routine. Resulting 7,381 points are
collected and dominated points are removed. We noticed that
only 4,450 points are non-dominated. These non-dominated
points are shown in Figure 44 with a dot. It is clear that
the NSGA-III obtained set of 60 points are widely-distributed
on the entire front. Interestingly, 4,450 trade-off points reveal
the complex nature of Pareto-optimal front (having holes and
varying density of points) for this practical problem and the
ability of NSGA-III to ﬁnd a few widely-distributed points on
the entire front.
To investigate the closeness of NSGA-III points with the
classically optimized solutions, we compute the convergence
metric (average distance of NSGA-III points from the closest
fmincon-optimized points) and the minimum, median and
maximum convergence metric values are found be 0.0019,
0.0022, and 0.0026, respectively. Since these values are small,
they indicate that obtained NSGA-III solutions are close to the
classically optimized front. This problem demonstrates the use
of NSGA-III in ﬁnding a representative set of points on the
entire Pareto-optimal front on problems having practical difﬁ-
culties (such as differences in scaling of objectives, potential
holes in the Pareto-optimal front, etc.).

```

</details>

<details>
<summary>Page 20: complete visual reference</summary>

![Full source page 20](K:/source/OptunaSharp/docs/papers/paper-gateway/library/4c240f7368799f9d/pages/page-0020.png)

</details>

> Quality note: Text produced by local English OCR; verify wording, mathematics and symbols against the page image.
> Quality note: Image-region OCR is separated from prose; graphic/table transcript semantics require visual review.
> Quality note: Rotated or vertical text retains geometry and requires visual order review.
> Quality note: Mathematical notation or superscripts require visual review; linear text does not establish fraction, exponent or equation semantics.
> Quality note: An alternate native text view contains characters absent from the ordered view. It is retained separately and indexed, with no inferred position; inspect both views and the page image.
> Quality note: Replacement characters detected in extracted text; inspect fonts/symbols against the page image.

<a id="page-21"></a>

<!-- Page 21 -->

This article has been accepted for publication in a future issue of this journal, but has not been fully edited. Content may change prior to final publication.

21

B. Car Cab Design with Preference Information

Next, we consider a vehicle performance optimization prob-
lem having 11 decision variables involving dimensions of the
car body and bounds on natural frequencies. The problem
involves nine objectives on ‘roominess’ of the car, fuel econ-
omy, acceleration time, and road noise at different speeds.
Mathematical formulation can be found elsewhere \[56\].

In this problem, we demonstrate NSGA-III’s ability to
use preference information in finding a few preferred non-
dominated solutions on a part of the Pareto-optimal front,
instead of on the complete front. We apply NSGA-III with
only 10 reference points in the intermediate part of the Pareto-
optimal front. For this purpose, we first create 10 random
points on the entire unit hyper-plane and then ‘shrink’ them
within 25% around the centroid of the unit hyper-plane. As
suggested in Section VI-A, nine extreme reference points are
also included for a better normalization purpose. A population
of size 100 is used and NSGA-III is run for 4,000 generations.
20 independent runs are made with different sets of reference
points and resulting solutions are compared with the classi-
cal achievement scalarizing function approach implemented
with Matlab’s fmincon routine. To have good representative
solutions in the chosen intermediate part, fmincon routine is
run for 20,000 random reference points initialized in the same
intermediate range one at a time. A domination check of
final solutions make 11,280 of them non-dominated to each
other. Each set of 10 solutions from NSGA-III runs is then
compared against these 11,280 fmincon solutions and the GD
metric value is computed. The solutions from the specific
NSGA-III run corresponding to the median GD metric value
is shown against the silhouette of 11,280 fmincon solutions on
a value path plot in Figure 45 to make a visual comparison.
The trade-off information and spread of obtained NSGA-III
solutions are clear from the plot. While the large number of
fmincon solutions provide a range of objective values for the
chosen range of reference points, only one out of 10 NSGA-
III solutions has two objective values (f4 and f9) outside the
range. Other nine solutions quite nicely represent relevant part
of the Pareto-optimal front. With NSGA-III’s ability to find
just 10 preferred trade-off solutions on a nine-objective non-
dominated front should remain as a significant step towards a
convenient post-optimal multi-criterion decision-making task.

![Fig. 45. A value path plot for the nine-objective car cab design problem shows that NSGA-III is able to find as few as 10 preferred solutions within the silhouette of solutions found by classical means.](K:/source/OptunaSharp/docs/papers/paper-gateway/library/4c240f7368799f9d/figures/p0021-figure-01.png)

Fig. 45. A value path plot for the nine-objective car cab design problem
shows that NSGA-III is able to find as few as 10 preferred solutions within
the silhouette of solutions found by classical means.

To quantify the convergence of NSGA-III solutions, the
best, median and worst GD metric values against 11,280 fmin-
con solutions on the normalized objective space are computed
as follows: 0.0245, 0.0408, and 0.0566. These values are
small enough to indicate the closeness of obtained NSGA-III
solutions from the fmincon solutions. The ability of NSGA-III
to find a handful of trade-off points in a preferred part of a
nine-objective practical optimization problem having different
scaling of objectives shows a promise of its use in practical
many-objective optimization problems.

VIII. CONCLUSIONS

In this paper, we have suggested a reference point based ap-
proach to an earlier-proposed NSGA-II framework for solving
many-objective optimization problems. The proposed NSGA-
III approach has been applied to three to 15-objective existing
and new test problems and to three and nine-objective practical
problems. The test problems involve fronts that have convex,
concave, disjointed, differently scaled, biased density of points
across the front, and multimodality involving multiple local
fronts where an optimization algorithm can get stuck to.
In all such problems, the proposed NSGA-III approach has
been able to successfully find a well-converged and well-
diversified set of points repeatedly over multiple runs. The
performance scaling to 15-objectives is achieved mainly due
to the aid in diversity preservation by supplying a set of well-
distributed reference points. In higher-dimensional problems,
EMO algorithms face with an increasingly difficult task of
maintaining diversity as well as in converging to the Pareto-
optimal front. The supply of a set of reference points and
NSGA-III’s efficient niching methodology in finding a Pareto-
optimal solution associated to each reference point has made
the diversity preservation of obtained solutions in as large as
15-objectives possible.

The performance of NSGA-III has been compared with
several versions of a recently proposed MOEA/D procedure.
Although different MOEA/Ds have shown their working on
different problems, no single version is able to solve all
problems efficiently. Having solved all problems well by the
proposed NSGA-III procedure, there is another advantage of
it that is worth mentioning here. Unlike MOEA/D versions,
NSGA-III procedure does not require any additional parameter
to be set.

Furthermore. NSGA-III has been tested for its ability to
solve a number of different types of many-objective problem
solving tasks. It has been demonstrated that NSGA-III is
able to work with a small number of user-supplied struc-
tured or randomly assigned reference points, thereby making
the method suitable for a many-objective preference-based
optimization-cum-decision-making approach. It has also been
shown that the NSGA-III procedure can be used to quickly es-
timate the nadir point in many-objective optimization problems
compared to another EMO-based method proposed earlier. For
a practical application, only a few handful trade-off points
are required for decision-making in multi- and many-objective
optimization problems. It has been shown that NSGA-III can
be used to find only a few points (20 in five-objective and

Copyright (c) 2013 IEEE. Personal use is permitted. For any other purposes, permission must be obtained from the IEEE by emailing pubs-permissions@ieee.org.

<details>
<summary>OCR transcript from a graphic region; inspect the source image for layout</summary>

```text
A
```

</details>

<details>
<summary>OCR transcript from a graphic region; inspect the source image for layout</summary>

```text
> 06 Se FF
: 04
Vy VV / 5Objective Number6 7 ON
```

</details>

<details>
<summary>Page 21: complete visual reference</summary>

![Full source page 21](K:/source/OptunaSharp/docs/papers/paper-gateway/library/4c240f7368799f9d/pages/page-0021.png)

</details>

> Quality note: Text produced by local English OCR; verify wording, mathematics and symbols against the page image.
> Quality note: Image-region OCR is separated from prose; graphic/table transcript semantics require visual review.
> Quality note: Mathematical notation or superscripts require visual review; linear text does not establish fraction, exponent or equation semantics.

<a id="page-22"></a>

<!-- Page 22 -->

This article has been accepted for publication in a future issue of this journal, but has not been fully edited. Content may change prior to final publication.

22

## 65 in a 10-objective problem) with a small population size, thereby reducing the computational efforts.

The proposed NSGA-III approach with a supply of a set
of reference points has a similar implicit principle to the
hyper-grid based archiving techniques, such as adaptive grid
algorithm \[57\], \[58\] and ϵ-MOEA \[59\], \[60\]. Grid based
methods require exponentially more grids to be considered
with an increase in objectives, whereas ϵ-domination concept,
although reduce the burden of finding an appropriate diver-
sity in solutions somewhat, still poses enough difficulties in
finding well-distributed Pareto-optimal solutions for a large-
dimensional problem. However, it remains as an interesting
future study to compare NSGA-III’s performance with an ϵ-
domination based algorithm for many-objective optimization
problems. NSGA-III’s performance has been found to be much
better than a classical generating method in many-objective
problems. However, a more thorough study using population-
based aggregation methods \[61\], \[62\] in which multiple search
directions are used simultaneously is worth pursuing.

Results on many different problems with box constraints
used in this paper have clearly shown promise for NSGA-
III’s further application in other challenging problems. In
the sequel of this paper \[52\], we have extended NSGA-III
to solve constrained many-objective optimization problems
having generic inequality and equality constraints and also
suggested an adaptive NSGA-III approach for adding and
deleting the supplied reference points to maximize the number
of obtained Pareto-optimal solutions in a computationally effi-
cient manner. Nevertheless, this paper has addressed the issues
and difficulties related to solving many-objective optimization
problems, reviewed some past studies in this direction, sug-
gested a viable many-objective evolutionary algorithm, and
demonstrated proof-of-principle results on many problems
including two engineering design problems.

## REFERENCES

\[1\] O. Chikumbo, E. Goodman, and K. Deb, “Approximating a multi-

dimensional pareto front for a land use management problem: A
modified MOEA with an epigenetic silencing metaphor,” in Proceedings
of Congress on Evolutionary Computation (CEC-2012), 2012, pp. 1–8.
\[2\] C. A. C. Coello and G. B. Lamont, Applications of Multi-Objective

Evolutionary Algorithms. World Scientific, 2004.
\[3\] K. Deb, Multi-objective optimization using evolutionary algorithms.

Chichester, UK: Wiley, 2001.
\[4\] K. Deb and D. Saxena, “Searching for Pareto-optimal solutions through

dimensionality reduction for certain large-dimensional multi-objective
optimization problems,” in Proceedings of the World Congress on
Computational Intelligence (WCCI-2006), 2006, pp. 3352–3360.
\[5\] K. Deb, S. Agrawal, A. Pratap, and T. Meyarivan, “A fast and elitist

multi-objective genetic algorithm: NSGA-II,” IEEE Transactions on
Evolutionary Computation, vol. 6, no. 2, pp. 182–197, 2002.
\[6\] E. Zitzler, M. Laumanns, and L. Thiele, “SPEA2: Improving the

strength Pareto evolutionary algorithm for multiobjective optimization,”
in Evolutionary Methods for Design Optimization and Control with
Applications to Industrial Problems, K. C. Giannakoglou, D. T. Tsahalis,
J. P´eriaux, K. D. Papailiou, and T. Fogarty, Eds. Athens, Greece:
International Center for Numerical Methods in Engineering (CIMNE),
2001, pp. 95–100.
\[7\] E. Zitzler and L. Thiele, “Multiobjective evolutionary algorithms: A

comparative case study and the strength Pareto approach,” IEEE Trans-
actions on Evolutionary Computation, vol. 3, no. 4, pp. 257–271, 1999.
\[8\] E. Zitzler, L. Thiele, M. Laumanns, C. M. Fonseca, and V. G. Fonseca,

“Performance assessment of multiobjective optimizers: An analysis and
review,” IEEE Transactions on Evolutionary Computation, vol. 7, no. 2,
pp. 117–132, 2003.

\[9\] E. J. Hughes, “Evolutionary many-objective optimisation: Many once

or one many?” in IEEE Congress on Evolutionary Computation (CEC-
2005), 2005, pp. 222–227.
\[10\] Q. Zhang and H. Li, “MOEA/D: A multiobjective evolutionary algorithm

based on decomposition,” Evolutionary Computation, IEEE Transactions
on, vol. 11, no. 6, pp. 712–731, 2007.
\[11\] D. K. Saxena, J. A. Duro, A. Tiwari, K. Deb, and Q. Zhang, “Ob-

jective reduction in many-objective optimization: Linear and nonlinear
algorithms,” IEEE Transactions on Evolutionary Computation, in press.
\[12\] J. A. L´opez and C. A. C. Coello, “Some techniques to deal with many-

objective problems,” in Proceedings of the 11th Annual Conference
Companion on Genetic and Evolutionary Computation Conference.
New York: ACM, 2009, pp. 2693–2696.
\[13\] H. Ishibuchi, N. Tsukamoto, and Y. Nojima, “Evolutionary many-

objective optimization: A short review,” in Proceedings of Congress on
Evolutionary Computation (CEC-2008), 2008, pp. 2424–2431.
\[14\] K. Sindhya, K. Miettinen, and K. Deb, “A hybrid framework for

evolutionary multi-objective optimization,” IEEE Transactions on Evo-
lutionary Computation, in press.
\[15\] H. Nakayama, K. Kaneshige, S. Takemoto, and Y. Watada, “An applica-

tion of a multiobjective programming technique to construction accuracy
control of cable-stayed bridges,” European Journal of Operations Re-
search, vol. 87, no. 3, pp. 731–738, 1995.
\[16\] M. Garza-Fabre, G. T. Pulido, and C. A. C. Coello, “Ranking methods

for many-objective optimization,” in Proceedings of Mexican Interna-
tional Conference on Artificial Intelligence (MICAI-2009). Springer,
2009, pp. 633–645.
\[17\] V. Chankong and Y. Y. Haimes, Multiobjective Decision Making Theory

and Methodology. New York: North-Holland, 1983.
\[18\] C. M. Fonseca, L. Paquete, and M. L´opez-Ib´anez, “An improved dimen-

sion sweep algorithm for the hypervolume indicator,” in Proceedings of
the 2006 Congress on Evolutionary Computation (CEC’06). IEEE
Press, Piscataway, NJ, 2006, pp. 1157–1163.
\[19\] L. While, P. Hingston, L. Barone, and S. Huband, “A faster algorithm

for calculating hypervolume,” IEEE Transactions on Evolutionary Com-
putation, vol. 10, no. 1, pp. 29–38, 2006.
\[20\] K. Deb, J. Sundar, N. Uday, and S. Chaudhuri, “Reference point based

multi-objective optimization using evolutionary algorithms,” Interna-
tional Journal of Computational Intelligence Research (IJCIR), vol. 2,
no. 6, pp. 273–286, 2006.
\[21\] K. Deb and A. Kumar, “Interactive evolutionary multi-objective op-

timization and decision-making using reference direction method,” in
Proceedings of the Genetic and Evolutionary Computation Conference
(GECCO-2007). New York: The Association of Computing Machinery
(ACM), 2007, pp. 781–788.
\[22\] ——, “Light beam search based multi-objective optimization using evo-

lutionary algorithms,” in Proceedings of the Congress on Evolutionary
Computation (CEC-07), 2007, pp. 2125–2132.
\[23\] U. K. Wickramasinghe and X. Li, “A distance metric for evolutionary

many-objective optimization algorithms using user-preferences,” in Pro-
ceedings of Artificial Intelligence conference (AI-2009), (LNAI 5866).
Heidelberg, Germany: Springer-Verlag, 2009, pp. 443–453.
\[24\] D. Brockhoff and E. Zitzler, “Dimensionality Reduction in Multiob-

jective Optimization: The Minimum Objective Subset Problem,” in
Operations Research Proceedings 2006, K. H. Waldmann and U. M.
Stocker, Eds. Springer, 2007, pp. 423–429.
\[25\] H. K. Singh, A. Isaacs, and T. Ray, “A Pareto corner search evolutionary

algorithm and dimensionality reduction in many-objective optimization
problems,” IEEE Transactions on Evolutionary Computation, vol. 15,
no. 4, pp. 539–556, 2011.
\[26\] M. Laumanns, L. Thiele, K. Deb, and E. Zitzler, “Combining con-

vergence and diversity in evolutionary multi-objective optimization,”
Evolutionary Computation, vol. 10, no. 3, pp. 263–282, 2002.
\[27\] D. Hadka and P. Reed, “Diagnostic assessment of search controls and

failure modes in many-objective evolutionary optimization,” Evolution-
ary Computation Journal, in press.
\[28\] K. Deb and R. B. Agrawal, “Simulated binary crossover for continuous

search space,” Complex Systems, vol. 9, no. 2, pp. 115–148, 1995.
\[29\] M. Farina and P. Amato, “A fuzzy definition of ‘optimality’ for many-

criteria decision-making and optimization problems,” IEEE Trans. on
Sys. Man and Cybern., vol. 34, no. 3, pp. 315–326, 2004.
\[30\] X. Zou, Y. Chen, M. Liu, and L. Kang, “A new evolutionary algorithm

for solving many-objective optimization problems,” IEEE Transactions
on Systems, Man, and Cybernetics – Part B: Cybernetics, vol. 38, no. 5,
pp. 1402–1412, 2008.

Copyright (c) 2013 IEEE. Personal use is permitted. For any other purposes, permission must be obtained from the IEEE by emailing pubs-permissions@ieee.org.

<details>
<summary>Page 22: complete visual reference</summary>

![Full source page 22](K:/source/OptunaSharp/docs/papers/paper-gateway/library/4c240f7368799f9d/pages/page-0022.png)

</details>

> Quality note: Mathematical notation or superscripts require visual review; linear text does not establish fraction, exponent or equation semantics.

<a id="page-23"></a>

<!-- Page 23 -->

This article has been accepted for publication in a future issue of this journal, but has not been fully edited. Content may change prior to final publication.

23

\[31\] H. Aguirre and K. Tanaka, “Many-objective optimization by space par-

titioning and adaptive epsilon-ranking on mnk-landscapes,” in Proceed-
ings of 5th International Conference on Evolutionary Multi-Criterion
Optimization (EMO 2009), 2009, pp. 407–422, lecture Notes in Com-
puter Science 5467.
\[32\] H. Sato, H. E. Aguirre, and K. Tanaka, “Pareto partial dominance

moea in many-objective optimization,” in Proceedings of Congress on
Computational Intelligence (CEC-2010), 2010, pp. 1–8.
\[33\] K. Miettinen, Nonlinear Multiobjective Optimization. Boston: Kluwer,
1999\.
\[34\] J. Branke, T. Kauβler, and H. Schmeck, “Guidance in evolution-

ary multi-objective optimization,” Advances in Engineering Software,
vol. 32, pp. 499–507, 2001.
\[35\] G. Eichfelder, “Optimal elements in vector optimization with a variable

ordering structure,” Journal of Optimization Theory and Applications,
vol. 151, no. 2, pp. 217–240, 2011.
\[36\] K. Deb and H. Jain, “Handling many-objective problems using an

improved NSGA-II procedure,” in Proceedings of World Congress on
Computational Intelligence (WCCI-2012), 2012, pp. 1–8.
\[37\] Q. Zhang, H. Li, D. Maringer, and E. Tsang, “MOEA/D with NBI-style

Tchebycheff approach for portfolio management,” in IEEE Congress on
Evolutionary Computation, 2010.
\[38\] A. Zhou, Q. Zhang, and Y. Jin, “Approximating the set of pareto optimal

solutions in both the decision and objective spaces by an estimation of
distribution algorithm,” IEEE Transaction on Evolutionary Computation,
vol. 13, no. 5, pp. 1167–1189, 2009.
\[39\] R. C. Purshouse and P. J. Fleming, “On the evolutionary optimization

of many conflicting objectives,” IEEE Transactions on Evolutionary
Computation, vol. 11, no. 6, 2007.
\[40\] S. F. Adra and P. J. Fleming, “Diversity management in evolutionary

many-objective optimization,” IEEE Transactions on Evolutionary Com-
putation, vol. 15, no. 2, pp. 183–195, 2011.
\[41\] M. K¨oppen and K. Yoshida, “Substitute distance assignments in nsga-ii

for handling many-objective optimization problems,” in Proceedings of
Fourth International Conference on Multi-Objective Optimization, EMO
2007 (LNCS 4403). Heidelberg, Germany: Springer-Verlag, 2007, pp.
727–741.
\[42\] D. Hadka and P. Reed, “Borg: An auto-adaptive many-objective evolu-

tionary computing framework,” Evolutionary Computation, p. in press,
2011\.
\[43\] J. Bader and E. Zitzler, “Hype: An algorithm for fast hypervolume-

based many-objective optimization,” Evolutionary Computation Journal,
vol. 19, no. 1, pp. 45–76, 2011.
\[44\] J. Bader, K. Deb, and E. Zitzler, “Faster hypervolume-based search using

Monte Carlo sampling,” in Proceedings of Multiple Criteria Decision
Making (MCDM 2008). Heidelberg: Springer, 2010, pp. 313–326,
lNEMS-634.
\[45\] L. Bradstreet, L. While, and L. Barone, “A fast incremental hypervolume

algorithm,” IEEE Transactions on Evolutionary Computation, vol. 12,
no. 6, pp. 714–723, 2008.
\[46\] H. Li and Q. Zhang, “Multiobjective optimization problems with compli-

cated Pareto sets, MOEA/D and NSGA-II,” IEEE Trans on Evolutionary
Computation, vol. 12, no. 2, pp. 284–302, 2009.
\[47\] Q. Zhang, A. Zhou, S. Z. Zhao, P. N. Suganthan, W. Liu, and S. Tiwari,

“Multiobjective optimization test instances for the cec-2009 special
session and competition,” Singapore: Nanyang Technological University,
Tech. Rep., 2008.
\[48\] I. Das and J. Dennis, “Normal-boundary intersection: A new method

for generating the Pareto surface in nonlinear multicriteria optimization
problems,” SIAM Journal of Optimization, vol. 8, no. 3, pp. 631–657,
1998\.
\[49\] H. T. Kung, F. Luccio, and F. P. Preparata, “On finding the maxima of

a set of vectors,” Journal of the Association for Computing Machinery,
vol. 22, no. 4, pp. 469–476, 1975.
\[50\] Q. Zhang, “MOEA/D homepage,” http://dces.essex.ac.uk/staff/zhang/

webofmoead.htm.
\[51\] D. V. Veldhuizen and G. B. Lamont, “Multiobjective evolutionary

algorithm research: A history and analysis,” Dayton, OH: Department of
Electrical and Computer Engineering, Air Force Institute of Technology,
Tech. Rep. TR-98-03, 1998.
\[52\] K. Deb and H. Jain, “An improved NSGA-II procedure for many-

objective optimization, Part II: Handling constraints and extending to
an adaptive approach,” Indian Institute of Technology Kanpur, Tech.
Rep. 2012010, 2012.
\[53\] K. Deb, L. Thiele, M. Laumanns, and E. Zitzler, “Scalable test problems

for evolutionary multi-objective optimization,” in Evolutionary Multi-

objective Optimization, A. Abraham, L. Jain, and R. Goldberg, Eds.
London: Springer-Verlag, 2005, pp. 105–145.
\[54\] K. Deb, K. Miettinen, and S. Chaudhuri, “Towards an estimation of

nadir objective vector using a hybrid of evolutionary and local search
approaches,” IEEE Transactions on Evolutionary Computation, in press.
\[55\] X. Liao, Q. Li, W. Zhang, and X. Yang, “Multiobjective optimization

for crash safety design of vehicle using stepwise regression model,”
Structural and Multidisciplinary Optimization, vol. 35, pp. 561–569,
2008\.
\[56\] K. Deb, S. Gupta, D. Daum, J. Branke, A. Mall, and D. Padmanabhan,

“Reliability-based optimization using evolutionary algorithms,” IEEE
Trans. on Evolutionary Computation, vol. 13, no. 5, pp. 1054–1074,
2009\.
\[57\] J. D. Knowles and D. W. Corne, “Quantifying the effects of objective

space dimension in evolutionary multiobjective optimization,” in Pro-
ceedings of the Fourth International Conference on Evolutionary Multi-
Crietrion Optimization, 2007, pp. 757–771.
\[58\] D. W. Corne, N. R. Jerram, J. D. Knowles, and M. J. Oates, “PESA-II:

Region-based selection in evolutionary multiobjective optimization,” in
Proceedings of the Genetic and Evolutionary Computation Conference
(GECCO-2001). San Mateo, CA: Morgan Kaufmann Publishers, 2001,
pp. 283–290.
\[59\] M. Laumanns, G. Rudolph, and H. P. Schwefel, “A spatial predator-

prey approach to multi-objective optimization: A preliminary study,” in
Proceedings of the Parallel Problem Solving from Nature, V, 1998, pp.
241–249.
\[60\] K. Deb, M. Mohan, and S. Mishra, “Towards a quick computation

of well-spread Pareto-optimal solutions,” in Proceedings of the Sec-
ond Evolutionary Multi-Criterion Optimization (EMO-03) Conference
(LNCS 2632), 2003, pp. 222–236.
\[61\] H. Ishibuchi and T. Murata, “A multi-objective genetic local search al-

gorithm and its application to flowshop scheduling,” IEEE Transactions
on Systems, Man and Cybernetics – Part C: Applications and reviews,
vol. 28, no. 3, pp. 392–403, 1998.
\[62\] A. Jaszkiewicz, “Genetic local search for multiple objective combinato-

rial optimization,” European Journal of Operational Research, vol. 137,
no. 1, pp. 50–71, 2002.

![Raster artwork, page 23, region 1](K:/source/OptunaSharp/docs/papers/paper-gateway/library/4c240f7368799f9d/figures/p0023-figure-01.png)

Kalyanmoy Deb Kalyanmoy Deb is the Koenig
Endowed Chair Professor at the Department of Elec-
trical and Computer Engineering at Michigan State
University (MSU), East Lansing, USA. Prior to this
position, he was at Indian Institute of Technology
Kanpur in India. Prof. Deb’s main research interests
are in evolutionary optimization algorithms and their
application in optimization and machine learning.
He is largely known for his seminal research in
Evolutionary Multi-Criterion Optimization. He was
awarded the prestigious ‘Infosys Prize’ in 2012,
‘TWAS Prize’ in Engineering Sciences in 2012, ‘CajAstur Mamdani Prize’ in
2011, ‘JC Bose National Fellowship’ in 2011, ‘Distinguished Alumni Award’
from IIT Kharagpur in 2011, ’Edgeworth-Pareto’ award in 2008, Shanti
Swarup Bhatnagar Prize in Engineering Sciences in 2005, ‘Thomson Citation
Laureate Award’ from Thompson Reuters. His 2002 IEEE-TEC NSGA-II
paper is now judged as the Most Highly Cited paper and a Current Classic
by Thomson Reuters having more than 4,000+ citations. He is a fellow of
IEEE. He has written two text books on optimization and more than 340
international journal and conference research papers. He is in the editorial
board on 18 major international journals, including IEEE TEC.

Himanshu Jain Himanshu Jain was born in Nehtaur,
India, on December 6, 1989. He received his dual
Bachelor’s and Master’s degree in Mechanical Engi-
neering from Indian Institute of Technology Kanpur
in 2012. He was a member of the Kanpur Ge-
netic Algorithms Laboratory (KanGAL) from 2009.
He is currently working as an Engineer at Eaton
Corporation, Pune. His research interests include
evolutionary computation and machine learning.

Copyright (c) 2013 IEEE. Personal use is permitted. For any other purposes, permission must be obtained from the IEEE by emailing pubs-permissions@ieee.org.

<details>
<summary>OCR transcript from a graphic region; inspect the source image for layout</summary>

```text
Xia rs 3 i.
RS
```

</details>

<details>
<summary>OCR transcript from a graphic region; inspect the source image for layout</summary>

```text
caf
```

</details>

![Raster artwork, page 23, region 2](K:/source/OptunaSharp/docs/papers/paper-gateway/library/4c240f7368799f9d/figures/p0023-figure-02.png)

<details>
<summary>Page 23: complete visual reference</summary>

![Full source page 23](K:/source/OptunaSharp/docs/papers/paper-gateway/library/4c240f7368799f9d/pages/page-0023.png)

</details>

> Quality note: Text produced by local English OCR; verify wording, mathematics and symbols against the page image.
> Quality note: Image-region OCR is separated from prose; graphic/table transcript semantics require visual review.
> Quality note: Mathematical notation or superscripts require visual review; linear text does not establish fraction, exponent or equation semantics.

<!-- ============================================================================== -->
<!-- PAPER REFERENCE: AGE-MOEA-2 / AGE-MOEA-II (Panichella, GECCO'22)  (source: docs/papers/extracted/AGE-MOEA-2.md) -->
<!-- ============================================================================== -->

# An Improved Pareto Front Modeling Algorithm for Large-scale Many-Objective Optimization

Source: [126__Panichella-2022-AGE-MOEA-II.pdf](file:///K:/source/OptunaSharp/docs/papers/genetic/nsga-iii-successors/126__Panichella-2022-AGE-MOEA-II.pdf)

SHA-256: `72851351a07ce0300b52d3eae0a544a5a5fe8a89f9cc86cd4aea4b46b4d26501`

> Extraction is heuristic. Full-page images preserve drawings, graphs and visual layout; searchable text is extracted separately.

<a id="page-1"></a>

<!-- Page 1 -->

## An Improved Pareto Front Modeling Algorithm for Large-scale Many-Objective Optimization

Annibale Panichella
Delft University of Technology
Delft, The Netherlands
a.panichella@tudelft.nl

## ABSTRACT

A key idea in many-objective optimization is to approximate the
optimal Pareto front using a set of representative non-dominated
solutions. The produced solution set should be close to the opti-
mal front (convergence) and well-diversified (diversity). Recent
studies have shown that measuring both convergence and diver-
sity depends on the shape (or curvature) of the Pareto front. In
recent years, researchers have proposed evolutionary algorithms
that model the shape of the non-dominated front to define environ-
mental selection strategies that adapt to the underlying geometry.
This paper proposes a novel method for non-dominated front mod-
eling using the Newton-Raphson iterative method for roots finding.
Second, we compute the distance (diversity) between each pair of
non-dominated solutions using geodesics, which are generaliza-
tions of the distance on Riemann manifolds (curved topological
spaces). We have introduced an evolutionary algorithm within the
Adaptive Geometry Estimation based MOEA (AGE-MOEA) frame-
work, which we called AGE-MOEA-II. Computational experiments
with 17 problems from the WFG and SMOP benchmarks show that
AGE-MOEA-II outperforms its predecessor AGE-MOEA as well
as other state-of-the-art many-objective algorithms, i.e., NSGA-III,
MOEA/D, VaEA, and LMEA.

## CCS CONCEPTS

• Theory of computation ! Evolutionary algorithms.

## KEYWORDS

Many-objective optimization, Geodesics, Newton-Raphson method

ACM Reference Format:
Annibale Panichella. 2022. An Improved Pareto Front Modeling Algorithm
for Large-scale Many-Objective Optimization. In Genetic and Evolutionary
Computation Conference (GECCO ’22), July 9–13, 2022, Boston, MA, USA. 
ACM, New York, NY, USA, 9 pages. https://doi.org/10.1145/3512290.3528732

## 1 INTRODUCTION

Many engineering and real-world problems are inherently multi-
objective as they require addressing multiple objectives \[4, 16, 24\].
However, there is no single solution that can satisfy all objectives
if they are conflicting. In these scenarios, there are multiple Pareto

This work is licensed under a Creative Commons Attribution International 4.0 License.

GECCO ’22, July 9–13, 2022, Boston, MA, USA
© 2022 Copyright held by the owner/author(s).
ACM ISBN 978-1-4503-9237-2/22/07.
https://doi.org/10.1145/3512290.3528732

optimal trade-offs between the various objectives, which are critical
for the decision-makers when choosing the solution that best fits
their constraints. Thus, finding these Pareto optimal trade-offs is the
primary goal of multi- and many-objective optimization algorithms.
Many-objective Evolutionary algorithms (MOEAs) have been
successfully used in literature to solve multi and many-objective
optimization problems (MOPs) with conflicting objectives. Their
goal is twofold: (1) produce a set of trade-offs that is as close to
the true Pareto front as possible (convergence); and (2) obtain well-
distributed trade-offs (diversity) for a more complete and informed
decision-making process. The success of MOEAs strongly depends
on their ability to balance convergence and diversity.
Over the last two decades, several MOEAs have been proposed
in the literature. They differ in the main mechanisms used to select
new solutions for the next generations (environmental selection)
such that both convergence and diversity are promoted. For ex-
ample, Pareto-based MOEAs like NSGA-II \[14\] divide candidate
solutions into several non-dominance fronts (for convergence) and
then use a distance function (e.g., crowding distance \[14\]) to pro-
mote more diverse solutions within the same convergence level
(front). Reference-points based MOEAs (e.g., NSGA-III \[13\]) use a
set of well-distributed reference points (diversity) and select the
best solution for each of the reference points (convergence). Finally,
indicator-based MOEAs (e.g., IBEA \[41\]) rely on quality indica-
tors, such as the hypervolume (HV), that measure both solutions
diversity and convergence. Other widely-used quality indicators
include the Averaged Hausdorffdistance (Δ?) \[32\], the inverted
generational distance plus (IGD+) \[20\], and R2 \[7\].
As highlighted in recent studies \[25, 29, 36\], these well-known
MOEAs use fixed diversity and convergence methods that achieve
different performances depending on the shape/geometry of the
Pareto front to approximate. For example, Das and Dennis’s sys-
tematic approach \[11\] used in NSGA-III and MOEA/D places the
reference points on a normalized hyper-plane, which is optimal for
flat Pareto fronts. The angular-based distance used in VaEA \[23\] and
LMEA \[40\] is instead effective on spherical fronts since the distance
on a sphere can be better measured by angles.
Hence, recent research has been devoted to designing adaptive
environmental selection strategies depending on the geometry of
the Pareto front. To this aim, fitting methods are used to model
the set of non-dominated solutions produced during the evolution-
ary process. In particular, Martinez et al. \[25\], and Tian et al. \[36\]
have applied non-linear fitting methods that can effectively model
the shape formed by the non-dominated solution set. However,
non-linear fitting methods like the Levenberg-Marquardt (LM) al-
gorithm \[36\] incur a substantial overhead, which increases dra-
matically with the population size and the number of objectives.

565

<details>
<summary>Page 1: complete visual reference</summary>

![Full source page 1](K:/source/OptunaSharp/docs/papers/paper-gateway/library/72851351a07ce030/pages/page-0001.png)

</details>

> Quality note: Mathematical notation or superscripts require visual review; linear text does not establish fraction, exponent or equation semantics.

<a id="page-2"></a>

<!-- Page 2 -->

GECCO ’22, July 9–13, 2022, Boston, MA, USA

Instead, AGE-MOEA \[29\] is a recently proposed MOEA that uses sim-
ple heuristics to model the shape of the Pareto front. In particular,
it uses one single non-dominated point (the central point), as a
reference throughout the search. Although simple, the single-point-
based heuristic has been shown to be effective while incurring a
negligible computational cost.
In this paper, we observe two areas of improvement in the overall
AGE-MOEA framework. First, the accuracy of the front-modeling
heuristic used in \[29\] strongly depends on the single reference
point being used. This can lead to non-accurate modeling of the non-
dominated front with the corresponding inaccurate computation
of converge and diversity. Second, the distances used to compute
the diversity among non-dominated solutions are generalizations
of the Euclidean distance (?-norm) that do not capture the front’s
topological structure (i.e., the curvature).
This paper introduces two novel strategies (1) to model the
non-dominated front and (2) to measure the distance among non-
dominated solutions. In particular, we use the Newton-Raphson
iterative method for root finding \[38\], which is much more robust
and less sensitive to the choice of the non-dominated point used
as reference (as empirically shown in Section 2.2). Second, we use
geodesic \[27\] to measure the distance between non-dominated so-
lutions on a curved manifold. In differential geometry, a geodesic
is the shortest path between two points on a curved manifold, e.g.,
the great circle on a sphere. Hence, we rely on these “curved” paths
on the front models to better measure solution diversity.
We have incorporated the Newton-Raphson iterative method
and the geodesic distance into the AGE-MOEA framework. We named
the resulting evolutionary algorithm AGE-MOEA-II. To assess the
performance of the proposed MOEA, we conducted an empiri-
cal study with 17 test problems from two benchmarks, namely
WFG \[18\] and SMOP \[37\]. These benchmarks have been widely
used in literature since they contain problems with different search
challenges and Pareto front shapes. Our study considers these test
problems with a different number of objective functions "=3, 6,
and 9. Experimental results indicate that AGE-MOEA-II achieves
higher hypervolume (HV) values than its predecessor AGE-MOEA
and outperforms other state-of-the-art MOEAs, namely NSGA-III,
MOEA/D, LMEA, and VaEA.
The remainder of the paper is organized as follows. Section 2
introduces basic concepts on Pareto-front modeling techniques
and many-objective problems. Section 3 introduces the novel front-
modeling and techniques as well as the geodesic distance. Section 4
details the empirical study we have carried out to assess the perfor-
mance of AGE-MOEA-II. Finally, Section 5 concludes the paper.

## 2 PRELIMINARIES

Multi-objective optimization problems are very common in real-
world engineering problems \[24\]. Without loss of generality, a
multi-objective optimization problem can be defined as follows:

&lt;8=퐹(G) = \[51(G), 52(G), . . . , 5" (G)\]) (1)

subject to G 2 ⌦✓R:

where G = (G1, . . . ,G:)) is a :-dimensional solution (or decision)
vector from the feasible region ⌦⇢R:; 5 : ⌦! R" is an objective
vector with " conflicting objective functions (or simply objectives).

A. Panichella

⇣ ⌘
A= cos c−42 , sin c−42

1 /max=(1,1)

$
0.5 B=\#cos c8 , sin c8

Second Objective

$
C=\#cos c16, sin c16

0
0 0.5 1

First Objective

![Figure 1: Example of a non-dominated front with ?=2 shape and two objectives. /<0G denotes the nadir point.](K:/source/OptunaSharp/docs/papers/paper-gateway/library/72851351a07ce030/figures/p0002-figure-01.png)

Figure 1: Example of a non-dominated front with ?=2 shape
and two objectives. /&lt;0G denotes the nadir point.

Multi-objective problems with more than three objectives (" &gt; 3)
are often referred to as many-objective in the literature.
A solution G dominates another solution \~ (G ≺\~), if 58 (G) ≤
58 (\~) 858 2 퐹and there exists at least one objective 59 2 퐹such that
59 (G) &lt; 59 (\~). A Pareto optimal solution G⇤2 ⌦is a solution that
is non-dominated by any other solution in ⌦, i.e., ö \~ 2 ⌦such
that \~ ≺G⇤. The set of all Pareto optimal solutions is called Pareto
optimal solution set (PS), while the corresponding objective vectors
form the so-called Pareto front (PF).

### 2.1 Pareto front modelling

In recent years, researchers have proposed different front modeling
techniques with the goal of defining diversity measures that are
specific or adaptive w.r.t. the shape (curvature) of the Pareto front.
The modeling is done using the following family of manifolds:
⇣ ⌘1/?
0?1 + · · · + 0?" = 1 (2)

where08 are the coordinates (objective values) of the non-dominated
solution 퐴, and ? is the parameter that control the front’s curvature.
Equation 2 is also referred to as \|\|·\|\|? norm or simply !? space. !? is
a generalization of the Euclidean (!2) distance \[34\] and defines the
so-called unit circle, as shown in Figure 1. In particular, points A, B,
and C are all at the same Euclidean distance from the origin of the
axes ($) that is equal to one (i.e., 3(퐴,$) = \|\|퐴−$\|\|2 = 1). Hence,
!2 accurately models the Pareto front that forms a (hyper) sphere.
If the Pareto front forms a flat hyper-plane, the best fitting norm
is ? = 1. Instead, for fronts that correspond to hyperbolic mani-
folds, the corresponding norm has ? &lt; 1 values. Hence, modeling
the Pareto front requires finding the value of ? that minimizes the
fitting error, i.e., the difference between the actual front and the cur-
vature obtained with the !? norm. To this aim, multiple numerical
methods can be used to solve this non-linear fitting problem.
GFM-MOEA by Tian et al. \[36\] uses a generalized version of the
!? family of manifolds:

8&gt;&gt;&gt;&gt;&lt; ?q

21 · 0?1,1 + · · · + 2" · 0?1," = 1
. . . (3)

&gt;&gt;&gt;&gt;: ?q

21 · 0?;,1 + · · · + 2" · 0?;," = 1

566

<details>
<summary>Page 2: complete visual reference</summary>

![Full source page 2](K:/source/OptunaSharp/docs/papers/paper-gateway/library/72851351a07ce030/pages/page-0002.png)

</details>

> Quality note: Rotated or vertical text retains geometry and requires visual order review.
> Quality note: Mathematical notation or superscripts require visual review; linear text does not establish fraction, exponent or equation semantics.

<a id="page-3"></a>

<!-- Page 3 -->

![Complete page reference; figure boundary uncertain. Full source page; figure boundaries require review.](K:/source/OptunaSharp/docs/papers/paper-gateway/library/72851351a07ce030/pages/page-0003.png)

An Improved Pareto Front Modeling Algorithm for Large-scale Many-Objective Optimization GECCO ’22, July 9–13, 2022, Boston, MA, USA

Algorithm 1: Pseudo-code of AGE-MOEA \[29\]

Input: ": Number of objectives
\# : Population size
Result: Final population %

1 begin

2 %  −RANDOM-POPULATION(\# )

3 while not (stop\_condition) do

4 &amp;  −GENERATE-OFFSPRING(%)

## 5 F −FAST-NONDOMINATED-SORT(% –&amp;)

## 6 F −NORMALIZE(F)

7 ?  −GET-GEOMETRY(F1, ") /\* Equation 4 \*/

8 3  −1 /\* First non-dominated rank \*/

9 while \| % \| + \| F3 \|6 \# do

## 10 SURVIVAL-SCORE(F3, 3, ?)

11 %  −% – F3

12 3  −3 + 1

## 13 SORT(F3) /\* by survival scores \*/

14 %  −% – F3 \[1 : (\# −\|% \|)\]

15 return %

where 08,9 denotes 9-th objective value for the 8-the non-dominated
solution. Solving such a system of equations requires finding the
constants ? and 21, . . . ,2" that minimize the fitting error. Given
this set of non-linear equations, GFM-MOEA uses the Levenberg-
Marquardt (LM) algorithm. The fitting process in \[36\] is particularly
expensive since it has an overall complexity of $(⌧0"2(" + \#)),
where ⌧0 is the number of iterations of the LM algorithm, \# is the
population size, while " is the number of objectives. Besides, the
fitting method must be re-applied in every  generations based
on a non-dominated set of solutions produced in the latest genera-
tion/iteration of the evolutionary process.
While GFM-MOEA uses the entire non-dominated front to find the
best fitting !? manifold, AGE-MOEA \[29\] uses one single point (called
central point) to determine the curvature ? of the approximated
Pareto front. This reduces the cost of modeling the front to $(").
This much lower computational cost allows to re-compute the ?-
value in each generation.
The pseudo-code of AGE-MOEA is reported in Algorithm 1 \[29\].
There are three key ingredients in AGE-MOEA \[29\]: (1) front normal-
ization; (2) determining the curvature ? of the first non-dominated
front; (3) environmental selection based on proximity and diversity,
both computed using the fitted ?-value. We elaborate on these key
ingredients in the following paragraphs.
Normalization (line 6 in Algorithm 1). First, the fast non-domina-
ted sorting algorithm (from NSGA-II \[12\]) groups the population
(parents and offspring) in sub-sequent non-dominated fronts. Then,
all fronts are normalized by scaling the objective scores within
the range \[0; 1\] using the same procedure of NSGA-III \[5\]. The
normalization is computed by (i) scaling the ideal point /&lt;8= to
the origin of the axes, and (ii) diving the objective values based on
the hyperplane intercepting the extreme points in the front \[29\].
Front modeling (line 7 in Algorithm 1). Once the first front is
normalized, its curvature is computed using the following formula:

;&gt;6(")
? = ⇣Õ" ⌘ (4)
;&gt;6(") −;&gt;6 8=1 ⇠8

where " is the number of objectives; ⇠is the objective vector for
the non-dominated solution closer to the central point within the
non-dominated front.

Environmental selection (line 10 in Algorithm 1). The popu-
lation for the next generation is selected by computing the survival
score, which combines two sub-scores: convergence and diversity.
The convergence score for each non-dominated solution 퐴is com-
puted using the distance between as !? of the vector \|\|퐴\|\|?, also
called Minkowski distance \[34\]. The diversity for a non-dominated
solution 퐴is computed using the formula:

38E4AB8C\~(퐴, F1) = min \|\|퐴−⌫\|\|? (5)
⌫&lt;퐴2F1

In other words, the diversity of the solution 퐴2 F1 is computed
as the minimum distance (!? norm) of 퐴to all other solutions in
F1 using the estimated ? norm.

### 2.2 Limitations of Existing Modeling Methods

The first limitation in the original AGE-MOEA framework regards the
fitting accuracy of Equation 4. The accuracy strongly depends on the
non-dominated solution used as the reference one. To better explain
this limitation, let us consider the graphical example depicted in
Figure 1. The plot shows three non-dominated points A, B, and C,
forming a circular bi-dimensional front (manifold with ?=2). If we
apply Equation 4 for the different non-dominated points, we obtain
the following results:

• For point A, ? = 0.495074054167603
• For point B, ? = 1.628072268133888
• For point C, ? = 1.305032304384445

In the example of Figure 1, the exact solution is ? = 2. Hence, the
accuracy of computing/approximating the value ? strongly depends
on how close the chosen reference point is to the theoretical center
of the front. Only if we apply Equation 4 to the theoretical middlep p
point ⇠⇤= (1/ 2, 1/ 2), we obtain the correct value ?=2.
Finally, the diversity is computed in AGE-MOEA using the \|\| · \|\|?
norm for the estimated geometry ?. For example, for a flat front
(?=1), the distance between two non-dominated solutions 퐴and ⌫
is computed using the Manhattan distance, i.e., 3(퐴, ⌫) = \|\|퐴−⌫\|\|1.
From a geometric point of view, the shortest distance between
two points in a flat space (? = 1) is the length of the straight line
between them computed with the Euclidean distance (or \|\| · \|\|2
norm). Similarly, the distance between two points 퐴and ⌫on a
sphere (3-d manifold with ?=2 geometry) is the great circle distance,
which is the length of the chord of the great circle between the two
points. For example, the great circle distance between points 퐴and⇣ ⌘
⌫in Figure 1 is the arc length ! = \\ · A = 2 −2c8 \* 1 ⇡0.8219,
where \\ is the angle delimited by the two points, and A is the ray
of the circle. Recall that the circle is the geometry (shape) of the
non-dominated front after normalization.

## 3 IMPROVED FRONT MODELLING MOEA

This paper introduces (i) a new accurate yet fast method to com-
pute the curvature of the non-dominated front, and (2) a new
diversity metrics for the non-dominated solutions based on Rie-
mannian geometry. We called the resulting evolutionary algorithm
AGE-MOEA-II. The key differences with its predecessor AGE-MOEA
are highlighted in bold-face in Algorithm 2. We discuss them in the
following sections.

567

<details>
<summary>Page 3: complete visual reference</summary>

![Full source page 3](K:/source/OptunaSharp/docs/papers/paper-gateway/library/72851351a07ce030/pages/page-0003.png)

</details>

> Quality note: Mathematical notation or superscripts require visual review; linear text does not establish fraction, exponent or equation semantics.

<a id="page-4"></a>

<!-- Page 4 -->

![Complete page reference; figure boundary uncertain. Full source page; figure boundaries require review.](K:/source/OptunaSharp/docs/papers/paper-gateway/library/72851351a07ce030/pages/page-0004.png)

GECCO ’22, July 9–13, 2022, Boston, MA, USA A. Panichella

### 3.1 Newton-Raphson Method

This paper proposes a more precise method to determine the curva-
ture (L? geometry) of the manifold formed by the non-dominated
front. In particular, we propose to use the Newton-Raphson method,
which is fast for approximating the roots (zeros) of real-valued func-
tions of the form 5 (G) = 0 \[19\]. Let 5 (G) be a continuous differential
function; the Newton-Raphson method iteratively approximates
the root of 5 (G) using the formula:

G=+1 = G= + 5 (G=) (6)
5 0(G=)

where G=+1 is the root approximation at the = + 1 iteration, while
G= is the approximation produced in the previous iteration G=. The
iterative method starts with an initial user-provided point G0 and
terminates when the difference between two consecutive approxi-
mations is lower than a given threshold n, i.e., when \|G=+1 −G=\| ≤n.
In our context, we want to determine the value ? that solves the
following non-linear equation:

vut "
? ’ $? = 1
\#08 −I&lt;8=8 (7)

8=1

where 08 and I&lt;8=8 are the 8-th coordinates of the non-dominated
point 퐴= \{01, . . . ,0"\} and the ideal point I&lt;8=. We consider the
case where the ideal point I&lt;8= is the origin of the axes (I&lt;8=8 =
$, 88 = 1 . . . "), which can be easily obtained by normalizing the
non-dominated fronts within the interval \[0; 1\]. Hence, our problem
consists of finding the root of the function (after simplification):

" !
’
5 (?) = ;&gt;6 (08)? (8)
8=1

The Newton-Raphson iterative formula for the function 5 (?) in
Equation 8 for a generic non-dominated point 퐴= \{01, . . . ,0"\} is:

" " \#
’
;&gt;6 (08)?=
8=1

?=+1 = ?= + " " \# (9)
’

(08)?= · ;&gt;6 (08)
8=1

with ?= and ?=+1 being the approximated roots obtained at the
iterations= and=+1, respectively. To initialize the Newton-Raphson
method, we have chosen the seed ?0 = 1, which corresponds to a
flat manifold (with no curvature). To make Equation 9 well-defined
within the interval ? 2 \[0, +1), we consider (08)?= · ;&gt;6 (08) = 0
when 08 = 0 since it corresponds to the limit limG!0G · ;&gt;6(G).

We have chosen the Newton-Raphson method over other numeri-
cal root-finding approaches (such as the bisection \[31\] or secant \[2\]
methods) because of its fast convergence \[17\] and our function
5 (?) is differentiable. In our preliminary analysis, we found that
Equation 9 requires, on average, 3 iterations to produce an error of
\|?=+1−?=\| ≤0.001 for the benchmark problems we have considered
in our empirical evaluation (See Section 4).

3.1.1 Graphical example. To better explain the advantages of the
Newton-Raphson method over the heuristic used in \[29\], let us
consider the graphical example depicted in Figure 1. The plot shows
three non-dominated points A, B, and C forming a circular bi-
dimensional front (manifold with ?=2). If we use Equation 9, we
obtain the following approximation of the value ? when considering

Algorithm 2: AGE-MOEA-II: An improved front modeling-
based MOEA

Input: ": Number of objectives
\# : Population size
Result: Final population %

1 begin

2 %  −RANDOM-POPULATION(\# )

3 while not (stop\_condition) do

4 &amp;  −GENERATE-OFFSPRING(%)

## 5 F −FAST-NONDOMINATED-SORT(% –&amp;)

## 6 F −NORMALIZE(F)

7 ?  −NEWTON-RAPHSON(F1) /\* Eq. 9 \*/

8 3  −1 /\* First non-dominated rank \*/

9 while \| % \| + \| F3 \|6 \# do

## 10 F? −MANIFOLD-PROJECTION(F3, ?)

11 ⇡ −GEODESIC-DIV(F?, ?) /\* Eq. 15 \*/

## 12 SURVIVAL-SCORE(D, F, 3, ?)

13 %  −% – F3

14 3  −3 + 1

## 15 SORT(F3) /\* by survival scores \*/

16 %  −% – F3 \[1 : (\# −\|% \|)\]

17 return %

the different non-dominated points, setting the initial seed ?0 = 1
and with only four iterations:

• For point A, ?4 = 1.999983149170659
• For point B, ?4 = 1.999999999879300
• For point C, ?4 = 1.999979682971426

Instead, if we use the equation used in \[29\] for the point closer
to the bisector of the first quadrant (e.g., point B), we obtain the
value ? = 1.628072268133887 (as shown in Section 2.2). Hence,
the accuracy of the original formula used in AGE-MOEA depends on
how close the reference point (i.e., the one used to approximate the
front curvature) is to the bisector. Instead, Equation 9 achieves a
very accurate approximation (with error &lt;0.0001) in only four iter-
ations, independently from which non-dominated point is actually
considered as the reference one.

3.1.2 Complexity. The computational complexity of the Newton-
Raphson iterative method is $(" ⇥⌧), where " is the number
of objectives while ⌧is the number of iterations. In practice, the
number of required iterations is very small (maximum 4-5 iterations)
given the fast convergence of the method.

### 3.2 Calculating Diversity with Geodesics

In differential geometry, the minimum distance between two points
on a non-Euclidean topological space is computed busing geodesics,
which generalize of the straight line between two points \[22\].
Geodesics are well-known geometry constructs for defining dis-
tances and metrics on Riemannian topologies \[27\]. Without loss of
generality, the distance between two points 퐴and ⌫on a manifold
is defined by the following integral:

π ⌫

55 3C
3 (퐴, ⌫) = infW 55W0(C) (10)

퐴

where W is a piecewise smooth curve (geodesic) joining 퐴and ⌫.
In other words, the distance is the infimum of the lengths of all
piecewise smooth paths between the two points.

568

<details>
<summary>Page 4: complete visual reference</summary>

![Full source page 4](K:/source/OptunaSharp/docs/papers/paper-gateway/library/72851351a07ce030/pages/page-0004.png)

</details>

> Quality note: Mathematical notation or superscripts require visual review; linear text does not establish fraction, exponent or equation semantics.

<a id="page-5"></a>

<!-- Page 5 -->

An Improved Pareto Front Modeling Algorithm for Large-scale Many-Objective Optimization GECCO ’22, July 9–13, 2022, Boston, MA, USA

A=(0,1) A=(0,1) A=(0,1)

1 1 1

⇠?

C(0.5, 0.5) ⌘C? C=(0.5,0.5)

0.5 0.5 0.5

C=(0.5,0.5)

⇠?

Second Objective

Second Objective

Second Objective

B=(1,0) B=(1,0) B=(1,0)

0 0 0
0 0.5 1 0 0.5 1 0 0.5 1

First Objective First Objective First Objective

(a) Distance approximation for ? = 1 (b) Distance approximation for ? = 2 (c) Distance approximation for ? = 0.5

![Figure 2: Graphical examples on how approximating the distance between two points on fronts with different curvatures.](K:/source/OptunaSharp/docs/papers/paper-gateway/library/72851351a07ce030/figures/p0005-figure-01.png)

Figure 2: Graphical examples on how approximating the distance between two points on fronts with different curvatures.

To simplify the computation of the distance between two points
on a generic manifold, we propose using the approximation proce-
dure described in the following subsection.

3.2.1 Geodesic Approximation. Let us consider a generic normal-
ized set of non-dominated solutions F and the curvature of ? com-
puted using Equation 9. To approximate the distance between two
generic points 퐴= \{01, . . . ,0"\} and ⌫= \{11, . . . ,1"\} 2 F, we first
compute the middle point ⇠, i.e., the point in the middle of the
straight line connecting the two points:

7

⇢01 + 11 , . . . , 0" + 1"
⇠= (11)
2 2

Then, we compute the projection ⇠? of the middle point ⇠into
the estimated unit-manifold (e.g., unit circle for ?=2) using the
following equation:

vut "
’

⇠? = ⇠÷ ? (28)? (12)
8=1

Finally, the geodesic distance between the two points 퐴and ⌫
on the manifold with the estimated ? is approximated as follow:

63(퐴, ⌫) = \|\|퐴−⇠?\|\|2 + \|\|⌫−⇠?\|\|2 (13)

where \|\|퐴−⇠?\|\|2 is the Euclidean distance between 퐴and ⇠?; and
\|\|⌫−⇠?\|\|2 is the Euclidean distance between ⌫and ⇠?.

3.2.2 Graphical example. . Figure 2 graphically shows how the
geodesic distance is computed for fronts with three different cur-
vatures, i.e., ? 2 \{0.5, 1, 2\}. Let us consider the case ? = 2 shown
in Figure 2a as an example. In this case, the shape/geometry of
the front is the unit circle depicted in the figure. Let us assume
we want to compute the geodesic distance between the two non-
dominated points 퐴= (0, 1) and ⌫= (1, 0). The midpoint computed
using Equation 11 is ⇠= (0.5, 0.5), while its projection on thep p
unit circle is ⇠? = (1/ 2, 1/ 2) obtained by applying Equation 12.
Then, the geodesic distance is approximated by Equation 13 as
3(퐴, ⌫) ⇡0.7653 + 0.7653 ⇡1.5306. This approximated value is
very close to great-circle distance ! = \\ · A = ?8/2 ⇡1.5708. While
Equation 13 approximates the exact geodesic distance, it provides a
better measure of the distance on the unit-circle than simply relying

on the Euclidean distance 퐴and ⌫since \|\|퐴−⌫\|\|2 ⇡1.4142. Notice
that the distance \|\| · \|\|2 is used in the original AGE-MOEA for fronts
with curvature ?=2.

3.2.3 Diversity Measure. Let F be the set of non-dominated so-
lutions after normalization, and let ? be the curvature/geometry
computed using Equation 9. First, we project all non-dominated
points onto the estimated front manifold with curvature ?:

vut "
’

퐴? = 퐴÷ ? (08)?, 8퐴2 F (14)
8=1

The diversity for each non-dominated point 퐴2 F is computed
using the geodesic distance in Equation 13 between its projected
point 퐴? and all other projected points of the front. Algorithm 2
denotes the projected front with ⇡, where ⇡= \[\{퐴?\}, 8퐴2 F.
More formally, the geodesic diversity of each non-dominated point
퐴is computed as:

64&gt;34B82\_38E(퐴, F1) = min 63 (퐴? −⌫?) (15)
⌫?&lt;퐴?2⇡

where 63 is the geodesic distance computed using Equation 13. In
other words, the diversity of the solution 퐴2 F1 is computed as
the minimum geodesic distance between its projection 퐴? and all
other projected points in ⇡.

3.2.4 Complexity. The computational complexity of the geodesic
distance (Equation 13) is $(" ⇥\# 2). In particular, for each pair
퐴and ⌫, it requires $(") operations to compute the Euclidean
distance between 퐴and the middle point ⇠? and other $(") oper-
ations for the Euclidean distance between 퐴and ⇠?.

## 4 EMPIRICAL STUDY

To assess the performance of AGE-MOEA-II, we performed a com-
putational experiment to compare our MOEA with several state-of-
the-art MOEAs on multiple benchmark problems.

### 4.1 Empirical Setting

4.1.1 Test Problems. We have selected two test benchmarks, namely
WFG \[18\] and SMOP \[37\]. WFG is a well-known test benchmark

569

<details>
<summary>Page 5: complete visual reference</summary>

![Full source page 5](K:/source/OptunaSharp/docs/papers/paper-gateway/library/72851351a07ce030/pages/page-0005.png)

</details>

> Quality note: Rotated or vertical text retains geometry and requires visual order review.
> Quality note: Mathematical notation or superscripts require visual review; linear text does not establish fraction, exponent or equation semantics.

<a id="page-6"></a>

<!-- Page 6 -->

![Complete page reference; figure boundary uncertain. Full source page; figure boundaries require review.](K:/source/OptunaSharp/docs/papers/paper-gateway/library/72851351a07ce030/pages/page-0006.png)

GECCO ’22, July 9–13, 2022, Boston, MA, USA A. Panichella

with 9 test problems with different properties, such as convex
(WFG1, WFG2), degenerate (WFG3), concave (WFG5, WFG6, WFG7,
and WFG8), and multimodal (WFG2, WFG9) fronts. SMOP has been
recently proposed by Tian et al. \[37\] and contains eight large-scale
sparse test problems with different shape and optimization chal-
lenges, such as deceptive objectives (SMOP2, SMOP3, SMOP8),
sub-optima (SMOP1, SMOP4, SMOP6, and SMOP7), and epistasis
(SMOP7 and SMOP8). We have chosen these two benchmarks since
they cover different real-world scenarios \[18, 37\], and can be con-
figured with a different number of objectives and decision variables.
To assess the performance of the selected MOEAs, we considered
the test problems with &lt; 2 \{3, 6, 9\} objectives. Besides, we set the
number of decision variables ⇡= 100 for all test problems in both
WFG and SMOP benchmarks.

4.1.2 Algorithms. Five state-of-the-art MOEAs are compared with
AGE-MOEA-II, namely AGE-MOEA \[29\], NSGA-III \[13\], MOEA/D \[28\],
VaEA \[23\], and LMEA \[40\]. AGE-MOEA \[29\] is an intuitive baseline
since it is the direct predecessor of AGE-MOEA-II. The main dif-
ferences between the two variants are related to how the non-
dominated front is modeled and how the distance/diversity be-
tween the different solutions is computed. NSGA-III uses reference
points and Pareto-ranking to compute diversity and convergence of
the non-dominated fronts. MOEA/D \[28\] relies on a well-distributed
set of weighted vectors and aggregation functions to produce a
well-diversified set of non-dominated solutions. VaEA \[23\] uses an
angle-based selection strategy that promotes only a few solutions
within a given minimum angle distance. Finally, LMEA \[40\] is an
MOEA designed for large-scale problems. It clusters decision vari-
ables into convergence-related and diversity-related ones. Then,
the environmental selection is based on both the clustering method
and the diversity of the generated solutions measured using an
angle-based distance. We have selected these baselines since they
use different strategies to balance convergence and diversity (e.g.,
reference point, Pareto-based, decomposition, angle-based distance,
and clustering methods).

4.1.3 Performance metrics. We run each MOEA 30 times on each
test problem to account for their stochastic nature. We stored the
non-dominated front produced at the end of the search by each
MOEA on each test problem. To assess the performance of the
different MOEAs, we employed the hypervolume (HV) quality indi-
cator \[33\]. HV takes values in \[0, 1\], and it measures the size (por-
tion) of the feasible region dominated by the set of non-dominated
points produced by an MOEA (e.g., AGE-MOEA-II). Hence, a large
HV value indicates better results. For calculating the HV scores, we
consider the nadir point /&lt;0G of the true Pareto front as the refer-
ence point. We have chosen the HV as the quality indicator since
it simultaneously measures both convergence and diversity of the
produced non-dominated fronts. To analyze the results, we report
the median and the interquartile range (IQR) of the HV values pro-
duced by each MOEA across the 30 independent runs. Furthermore,
we apply the Wilcoxon rank-sum test \[10\] to assess the significance
of the difference (if any) between AGE-MOEA-II and the baselines
(other MOEAs). For this test, we use the significance level U = 0.05.

4.1.4 Paremeter settings. For all MOEAs, we used the same parame-
ter settings for the shared parameters and genetic operators. Table 1

Table 1: Parameter values used for all MOEAS. " denotes the
number of objectives.

Parameters "=3 "=6 "=9
Population size 200 300 400
Number of Solution Evaluations 40,000 60,000 80,000
Number of iterations 200
SBX probability \[1\] ?2 = 1
SBX distributed index \[1\] \[2 = 30
Polynomial mutation probability \[12\] ?&lt; = 1/=
Mutation distributed index \[12\] \[&lt; = 20

details the applied genetic operators as well as the corresponding
parameter values \[1, 9, 12, 13, 21\]. Notice that for both mutation
and crossover, we use the probability values and distributed indexes
suggested in the literature \[1, 12\]. MOEA/D and NSGA-III require to
set the reference points. To this aim, we have used Das and Dennis’s
systematic approach \[11, 13\] and set the number of reference points
to the maximum value possible given the population size, that is
190, 273, and 330 for "=3, 6, and 9, respectively. For the stopping
condition, we set the maximum number of solution evaluations (SE)
proportional to the population size. In particular, we set SE=\# ⇥200,
where \# is the population size.

4.1.5 Implementation. We have implemented AGE-MOEA-II in Mat-
lab within the PlatEMO framework \[35\]. We have selected this
framework because it contains the original implementation of
AGE-MOEA, and it implements all MOEAs used in our experiment as
well as the benchmark problems. Besides, PlatEMO is public avail-
able on GitHub1, and it can be easily extended with new MOEAs.
The source code of AGE-MOEA-II is also publicly available at:
https://zenodo.org/record/6462859

All experiments were conducted on the same machine with the
following characteristics: 6-Core Intel Core i7 processor running at
3.2GHz with 16GB RAM.

### 4.2 Empirical Results

Table 2 reports the median and the IQR values for the HV indica-
tor achieved by the different MOEAs over 30 independent runs.
The table highlights the results with H and M when the HV val-
ues achieved by AGE-MOEA-II are statistically better (larger) or
worse (smaller) than the baseline MOEA according to the Wilcoxon
rank-sum test \[10\]. In addition, Table 3 counts the number of test
problems for which AGE-MOEA-II significantly outperforms (or is
outperformed by) a baseline (e.g., AGE-MOEA).
From the tables, we can observe that AGE-MOEA-II achieves a
significantly higher HV than NSGA-III in 15 out of 17 test problems
when " = 3 and " = 6. The 15 test problems with a significant
difference are the same independently of the number of objectives.
When increasing the number of objectives to " = 9, AGE-MOEA-II
achieves a significantly better HV value for an additional test prob-
lem, namely WFG3. It is worth noting that WFG3 has a degenerate
true Pareto front with ? = 1 curvature. Our results suggest that
AGE-MOEA-II outperforms NSGA-III for degenerate fronts only for
many objectives. Remarkable are the differences when considering
a larger number of objectives. We observe 2-3 test problems with

1https://github.com/BIMK/PlatEMO

570

<details>
<summary>Page 6: complete visual reference</summary>

![Full source page 6](K:/source/OptunaSharp/docs/papers/paper-gateway/library/72851351a07ce030/pages/page-0006.png)

</details>

> Quality note: Mathematical notation or superscripts require visual review; linear text does not establish fraction, exponent or equation semantics.

<a id="page-7"></a>

<!-- Page 7 -->

![Complete page reference; figure boundary uncertain. Full source page; figure boundaries require review.](K:/source/OptunaSharp/docs/papers/paper-gateway/library/72851351a07ce030/pages/page-0007.png)

An Improved Pareto Front Modeling Algorithm for Large-scale Many-Objective Optimization GECCO ’22, July 9–13, 2022, Boston, MA, USA

Table 2: HV values (median and IQR) yielded by the AGE-MOEA-II and the baselines on the SMOP and WFG benchmarks with "=3,6,
and 9 objectives. Best values are highlighted in grey color.

Problem " NSGA-III AGE-MOEA

SMOP1 3 7.9383e-1 (1.04e-2) H 8.1416e-1 (1.14e-2) H
SMOP2 3 4.8527e-1 (5.86e-2) H 6.2387e-1 (4.70e-2) H
SMOP3 3 1.3283e-1 (1.03e-2) H 1.6283e-1 (1.52e-2) H
SMOP4 3 9.1571e-1 (1.44e-2) H 9.6175e-1 (6.80e-3) H
SMOP5 3 8.3630e-1 (2.01e-3) H 8.5338e-1 (1.29e-3) M
SMOP6 3 9.6060e-1 (1.74e-3) H 9.7041e-1 (7.76e-4) H
SMOP7 3 2.1646e-1 (5.14e-2) H 2.6638e-1 (2.22e-2) H
SMOP8 3 0.0000e+0 (0.00e+0) 0.0000e+0 (0.00e+0)
WFG1 3 3.8954e-1 (1.20e-2) H 4.4376e-1 (1.56e-2) H
WFG2 3 8.5378e-1 (6.59e-3) H 8.7808e-1 (6.75e-3) H
WFG3 3 3.2949e-1 (7.75e-3) 3.2793e-1 (4.03e-3)
WFG4 3 4.8723e-1 (2.88e-3) H 5.1364e-1 (4.19e-3) H
WFG5 3 4.7312e-1 (5.02e-3) H 5.0431e-1 (2.70e-3) H
WFG6 3 4.5648e-1 (7.10e-3) H 4.9439e-1 (5.54e-3) H
WFG7 3 4.9211e-1 (8.22e-3) H 5.2730e-1 (4.16e-3) H
WFG8 3 4.5709e-1 (4.49e-3) H 4.8521e-1 (4.81e-3)
WFG9 3 4.4233e-1 (1.14e-2) H 4.6916e-1 (1.48e-2) H

SMOP1 6 9.0587e-1 (1.22e-2) H 9.7475e-1 (3.62e-3) H
SMOP2 6 9.4595e-2 (1.73e-2) H 7.1181e-1 (6.13e-2) H
SMOP3 6 3.7803e-2 (8.58e-3) H 4.0438e-1 (2.41e-2) H
SMOP4 6 9.4333e-1 (2.54e-2) H 9.9851e-1 (4.15e-4) H
SMOP5 6 9.6938e-1 (5.80e-3) H 9.9247e-1 (9.81e-4) H
SMOP6 6 9.9446e-1 (2.78e-3) H 9.9989e-1 (2.10e-5) H
SMOP7 6 4.7756e-3 (2.69e-3) H 2.9255e-1 (6.06e-2) H
SMOP8 6 0.0000e+0 (0.00e+0) 0.0000e+0 (0.00e+0)
WFG1 6 2.7449e-1 (2.23e-3) H 2.8373e-1 (4.04e-3) H
WFG2 6 8.4769e-1 (2.59e-2) H 9.2756e-1 (6.01e-3) H
WFG3 6 0.0000e+0 (0.00e+0) 0.0000e+0 (0.00e+0)
WFG4 6 6.7253e-1 (1.15e-2) H 7.4670e-1 (5.48e-3) H
WFG5 6 6.1381e-1 (1.02e-2) H 7.1530e-1 (5.40e-3) H
WFG6 6 5.7755e-1 (1.24e-2) H 7.0568e-1 (1.00e-2) H
WFG7 6 5.4299e-1 (1.71e-2) H 7.6026e-1 (1.30e-2) H
WFG8 6 6.2718e-1 (1.11e-2) H 7.0843e-1 (6.29e-3) H
WFG9 6 5.3602e-1 (1.88e-2) H 6.5562e-1 (1.40e-2) H

SMOP1 9 9.1194e-1 (1.36e-2) H 9.9652e-1 (6.51e-4) H
SMOP2 9 1.6797e-1 (1.55e-2) H 8.3050e-1 (3.26e-2) H
SMOP3 9 2.5468e-2 (7.19e-3) H 5.1486e-1 (3.35e-2) H
SMOP4 9 9.6621e-1 (1.23e-2) H 9.9997e-1 (9.00e-6) H
SMOP5 9 9.8093e-1 (1.65e-2) H 9.9997e-1 (6.00e-6) M
SMOP6 9 9.9263e-1 (5.52e-3) H 1.0000e+0 (1.00e-6)
SMOP7 9 2.8711e-3 (3.03e-3) H 4.0556e-1 (7.24e-2) H
SMOP8 9 0.0000e+0 (0.00e+0) 0.0000e+0 (0.00e+0)
WFG1 9 2.4892e-1 (5.08e-3) H 3.2931e-1 (7.91e-3) H
WFG2 9 8.9753e-1 (2.67e-2) H 9.3605e-1 (1.13e-2) H
WFG3 3 3.3299e-1 (7.53e-3) H 3.4988e-1 (3.84e-3)
WFG4 9 6.8611e-1 (2.26e-2) H 7.8342e-1 (9.12e-3) H
WFG5 9 5.8525e-1 (1.80e-2) H 7.4103e-1 (1.08e-2) H
WFG6 9 5.4269e-1 (1.43e-2) H 7.3866e-1 (8.99e-3) H
WFG7 9 5.7171e-1 (1.25e-2) H 8.0721e-1 (1.58e-2) H
WFG8 9 5.9656e-1 (1.55e-2) H 7.3638e-1 (8.60e-3) H
WFG9 9 5.2598e-1 (3.83e-2) H 6.9463e-1 (1.48e-2) H

MOEA/D VaEA

7.6983e-1 (2.66e-2) H 7.8485e-1 (8.47e-3) H
3.3598e-1 (1.14e-1) H 5.1073e-1 (4.52e-2) H
5.8745e-2 (2.64e-2) H 1.3495e-1 (7.01e-3) H
8.9440e-1 (2.30e-2) H 9.3308e-1 (8.98e-3) H
8.4850e-1 (3.12e-3) H 8.3946e-1 (1.20e-3) H
9.5560e-1 (3.25e-3) H 9.5990e-1 (1.19e-3) H
3.2481e-2 (2.26e-2) H 2.1337e-1 (3.19e-2) H
0.0000e+0 (0.00e+0) 0.0000e+0 (0.00e+0)
4.1812e-1 (3.83e-2) H 3.8184e-1 (7.88e-3) H
7.4573e-1 (4.27e-2) H 8.5574e-1 (6.20e-3) H
2.1978e-1 (3.51e-2) H 3.0509e-1 (6.90e-3) H
4.4788e-1 (7.35e-3) H 4.9490e-1 (3.45e-3) H
4.2370e-1 (1.45e-2) H 4.7992e-1 (3.52e-3) H
4.1391e-1 (1.26e-2) H 4.5843e-1 (4.89e-3) H
3.1497e-1 (2.99e-2) H 4.8442e-1 (1.06e-2) H
4.2197e-1 (1.45e-2) H 4.6605e-1 (5.52e-3) H
2.9674e-1 (2.97e-2) H 4.4047e-1 (1.03e-2) H

9.5277e-1 (1.98e-2) H 8.3600e-1 (1.68e-2) H
5.1517e-1 (1.26e-1) H 9.7716e-2 (1.78e-2) H
6.8498e-2 (5.51e-2) H 2.5869e-2 (1.41e-2) H
9.5481e-1 (1.14e-2) H 9.7842e-1 (1.46e-3) H
9.0342e-1 (1.96e-2) H 9.7330e-1 (2.12e-3) H
9.7485e-1 (5.59e-3) H 9.9890e-1 (2.13e-4) H
4.3521e-2 (4.33e-2) H 5.0929e-4 (4.20e-4) H
0.0000e+0 (0.00e+0) 0.0000e+0 (0.00e+0)
2.7277e-1 (2.19e-2) H 2.7240e-1 (1.51e-3) H
6.9661e-1 (3.28e-2) H 8.3469e-1 (1.15e-2) H
0.0000e+0 (0.00e+0) 0.0000e+0 (0.00e+0)
5.3248e-1 (6.44e-2) H 6.7395e-1 (9.97e-3) H
5.8198e-1 (3.58e-2) H 6.0171e-1 (1.20e-2) H
4.7854e-1 (6.33e-2) H 5.7803e-1 (8.70e-3) H
4.9288e-1 (4.48e-2) H 6.0440e-1 (1.58e-2) H
3.9970e-1 (6.94e-2) H 6.2668e-1 (1.49e-2) H
4.8193e-1 (1.23e-1) H 5.0751e-1 (2.24e-2) H

7.6769e-1 (4.36e-2) H 8.2945e-1 (2.60e-2) H
4.8073e-1 (4.09e-2) H 1.6800e-1 (2.51e-2) H
7.5394e-2 (5.23e-2) H 2.5798e-2 (8.67e-3) H
8.6900e-1 (7.73e-3) H 9.9844e-1 (1.72e-4) H
8.8156e-1 (5.79e-3) H 9.9937e-1 (1.04e-4) H
9.2078e-1 (7.62e-3) H 9.9995e-1 (1.70e-5) H
1.3145e-1 (9.49e-2) H 0.0000e+0 (0.00e+0) H
0.0000e+0 (0.00e+0) 0.0000e+0 (0.00e+0)
4.7066e-1 (6.14e-2) M 2.3761e-1 (2.84e-3) H
6.6769e-1 (3.70e-2) H 8.1996e-1 (1.58e-2) H
2.4602e-1 (3.61e-2) H 3.2881e-1 (4.36e-3) H
3.1054e-1 (3.17e-2) H 7.0813e-1 (8.11e-3) H
4.3496e-1 (4.22e-2) H 6.1813e-1 (1.47e-2) H
1.8879e-1 (1.58e-2) H 5.7646e-1 (1.64e-2) H
2.3464e-1 (9.53e-2) H 6.6157e-1 (1.40e-2) H
1.6326e-1 (5.12e-2) H 6.3040e-1 (1.17e-2) H
2.5318e-1 (2.44e-1) H 5.5946e-1 (2.11e-2) H

LMEA AGEMOEA-II

0.0000e+0 (1.48e-4) H 8.2104e-1 (8.81e-3)
0.0000e+0 (0.00e+0) H 6.7715e-1 (4.73e-2)
0.0000e+0 (0.00e+0) H 1.7365e-1 (5.93e-3)
3.1649e-1 (1.41e-2) H 9.7360e-1 (3.94e-3)
3.3764e-1 (2.03e-2) H 8.5254e-1 (1.36e-3)
7.6192e-1 (7.81e-3) H 9.7307e-1 (4.87e-4)
0.0000e+0 (0.00e+0) H 2.8582e-1 (3.43e-2)
0.0000e+0 (0.00e+0) 0.0000e+0 (0.00e+0)
0.0000e+0 (0.00e+0) H 4.6969e-1 (1.27e-2)
5.0972e-1 (2.17e-2) H 8.8293e-1 (9.80e-3)
9.6791e-2 (1.79e-3) H 3.3085e-1 (7.55e-3)
2.7076e-1 (1.06e-2) H 5.1827e-1 (2.82e-3)
1.9892e-1 (2.83e-3) H 5.0723e-1 (2.95e-3)
1.4230e-1 (3.31e-3) H 4.9923e-1 (5.31e-3)
3.2986e-1 (9.64e-2) H 5.3535e-1 (3.91e-3)
4.9750e-1 (7.76e-3) M 4.8645e-1 (6.75e-3)
3.5208e-1 (1.14e-2) H 4.8294e-1 (9.85e-3)

3.7740e-2 (1.06e-2) H 9.8920e-1 (9.81e-4)
6.4543e-4 (1.22e-3) H 9.6290e-1 (8.46e-3)
0.0000e+0 (5.33e-6) H 5.4054e-1 (1.62e-2)
8.5187e-1 (2.91e-2) H 9.9991e-1 (1.60e-5)
8.4851e-1 (1.85e-2) H 9.9369e-1 (1.12e-3)
9.7433e-1 (8.39e-3) H 9.9989e-1 (1.20e-5)
0.0000e+0 (0.00e+0) H 5.2498e-1 (4.26e-2)
0.0000e+0 (0.00e+0) 0.0000e+0 (0.00e+0)
0.0000e+0 (0.00e+0) H 2.9030e-1 (3.68e-3)
5.0834e-1 (1.98e-2) H 9.4392e-1 (8.92e-3)
0.0000e+0 (0.00e+0) 0.0000e+0 (0.00e+0)
3.3950e-1 (1.61e-2) H 7.5855e-1 (5.54e-3)
3.1960e-1 (7.52e-3) H 7.4264e-1 (6.38e-3)
7.8580e-1 (5.30e-1) M 7.3071e-1 (7.15e-3)
5.1953e-1 (1.72e-2) H 7.9293e-1 (6.03e-3)
7.0121e-1 (1.07e-2) H 7.2105e-1 (7.20e-3)
4.1141e-1 (2.22e-2) H 7.0852e-1 (1.43e-2)

6.7208e-2 (2.19e-2) H 9.9918e-1 (9.60e-5)
2.1433e-3 (2.96e-3) H 9.9585e-1 (1.26e-3)
0.0000e+0 (2.44e-6) H 6.3187e-1 (1.06e-2)
8.9264e-1 (1.93e-2) H 1.0000e+0 (1.00e-6)
8.9361e-1 (2.49e-2) H 9.9985e-1 (2.60e-5)
9.8272e-1 (4.63e-3) H 1.0000e+0 (1.00e-6)
0.0000e+0 (0.00e+0) H 7.2868e-1 (5.93e-2)
0.0000e+0 (0.00e+0) 0.0000e+0 (0.00e+0)
1.5701e-2 (1.80e-2) H 3.8569e-1 (1.58e-2)
4.7675e-1 (1.22e-2) H 9.5272e-1 (7.45e-3)
1.0350e-1 (1.60e-3) H 3.4847e-1 (5.14e-3)
3.4294e-1 (1.37e-2) H 8.0922e-1 (7.27e-3)
3.2565e-1 (5.19e-3) H 7.9309e-1 (8.74e-3)
8.4480e-1 (1.13e-2) M 7.7980e-1 (1.16e-2)
5.6795e-1 (1.74e-2) H 8.5471e-1 (6.94e-3)
7.4579e-1 (1.12e-2) H 7.5891e-1 (8.66e-3)
4.6368e-1 (1.82e-2) H 7.6207e-1 (2.11e-2)

at least one order of magnitude difference in the median HV val-
ues achieved by the two MOEAs (e.g., SMOP7 with "=6). Table 2
indicates that AGE-MOEA-II outperforms its predecessor AGE-MOEA
in the large majority of test problems. In particular, this applies in
13, 15, and 13 problems for " = 3, " = 6, and " = 9, respectively.

There is only one test problem, namely SMPO5, for which AGE-MOEA
produces significantly better HV values than AGE-MOEA-II. How-
ever, this happens only for " = 3 and " = 9 while there is no
significant difference for " = 6. However, we notice that the dif-
ference in the median HV is negligible in practice, being 0.0008 on

571

<details>
<summary>Page 7: complete visual reference</summary>

![Full source page 7](K:/source/OptunaSharp/docs/papers/paper-gateway/library/72851351a07ce030/pages/page-0007.png)

</details>

> Quality note: Mathematical notation or superscripts require visual review; linear text does not establish fraction, exponent or equation semantics.

<a id="page-8"></a>

<!-- Page 8 -->

GECCO ’22, July 9–13, 2022, Boston, MA, USA A. Panichella

Table 3: Number of test problems in which an algorithm A
statistically outperforms (&lt;) another algorithm B w.r.t. the
HV metric according to the Wilcoxon test.

Comparison M=3 M=6 M=9
AGE-MOEA-II &lt; NSGA-III 15 15 16
NSGA-III &lt; AGE-MOEA-II - - -
AGE-MOEA-II &lt; AGE-MOEA 13 15 13
AGE-MOEA &lt; AGE-MOEA-II 1 - 1
AGE-MOEA-II &lt; MOEA/D 16 15 15
MOEA/D &lt; AGE-MOEA-II 1 - 1
AGE-MOEA-II &lt; VaEA 16 15 16
VaEA &lt; AGE-MOEA-II - - -
AGE-MOEA-II &lt; LMEA 15 14 15
LMEA &lt; AGE-MOEA-II 1 1 1

average for " = 3 and " = 6. For WFG3, there is no significant
difference between AGE-MOEA-II and AGE-MOEA, independently of
the number of objectives ". This result is expected since the front
for WFG3 is degenerate (e.g., one single line), for which the dif-
ference in the numerical methods applied in AGE-MOEA-II and its
predecessor is minimal.
With regards to MOEA/D, Table 2 reveals that AGE-MOEA-II signif-
icantly outperforms the latter in most test problems independently
of the number of objectives. In detail, AGE-MOEA-II yielded signif-
icantly higher HV values than MOEA/D in 16 out of 17 problems
for "=3, and in 15 problems for both "=6 and "=9. For some
problems, the difference is very remarkable, e.g., for SMOP7 with
"=3, where there is one order of magnitude difference in the HV
values produced by the two MOEAs. In particular, AGE-MOEA-II
produced a median HV⇡0.54 against an HV⇡0.04 for MOEA/D. The
true Pareto front for SMOP7 has a ? = 2 curvature and multiple
local optima.We generally observe large differences for other de-
ceptive and multimodal problems such as SMOP3 and WFG8. Vice
versa, there is only one test problem (namely WFG1 with "=9), for
which the baseline outperforms AGE-MOEA-II.
From Tables 2-3, AGE-MOEA-II achieves significantly larger HV
values than VaEA in 16 out of 17 test problems for "=3 and "=9,
respectively. For "=6, the significance holds in 15 test problems.
The largest difference can be observed for SMOP7 with "=9, for
which the proposed MOEA obtains an average HV⇡0.72 while VaEA
achieved an average HV=0. Vice versa, in none of the test problems,
the baseline could outperform AGE-MOEA-II.
Particularly relevant are the results of the comparison between
AGE-MOEA-II and LMEA. The latter is an MOEA suitably designed
for large-scale problems, e.g., the SMOP benchmark. As reported in
Table 3, AGE-MOEA-II statistically outperformed LMEA in 15 out of
17 test problems for "=3 and "=9. Instead, for "=6, the statistical
significance holds for 14 out of 17 test problems. There are various
test problems (e.g., SMOP3 and SMOP7 with any ") for which
LMEA produced a zero HV value while AGE-MOEA achieved non-zero
values. That indicates that for certain problems (e.g., WFG1 with
" = 2), LMEA struggles to generate non-dominated solutions that
reach at least the nadir point of the true Pareto. For " 2 \{3, 6, 9\},
there is only one test problem where LMEA significantly outperforms
AGE-MOEA-II, but the test problem for which this happens varies
when increasing the number of objectives.

0.7

0.65

0.6

0.55

0.5

0.45

HV Values

0.4

0.35

AGE-MOEA
0.3 NSGA-III
MOEA/D
0.25 VaEA
LMEA
0.2 AGE-MOEA-II

1 2 3 4 5 6
Number of Evaluations 104

![Figure 3: Median HV values achieved by the different MOEAs for WFG9 with "=6 objectives.](K:/source/OptunaSharp/docs/papers/paper-gateway/library/72851351a07ce030/figures/p0008-figure-01.png)

Figure 3: Median HV values achieved by the different
MOEAs for WFG9 with "=6 objectives.

4.2.1 Graphical comparison. To graphically compare the MOEAs,
Figure 3 depicts the median HV values achieved for WFG9 with
"=6 and ⇡=100, using the parameters described in Table 1. The
median HV values achieved by AGE-MOEA-II are larger than those
achieved by all other MOEAs when varying the number of fitness
evaluations (FEs). The delta between AGE-MOEA-II and its predeces-
sor AGE-MOEA increases with a larger number of FEs. Furthermore,
the delta with the other MOEAs remains very large through the
generations. Finally, AGE-MOEA yielded the second-best HV results.
This confirms the better effectiveness/performance of MOEAs that
use front modeling techniques.

## 5 CONCLUSION AND FUTURE WORK

We have introduced an accurate yet fast method to model the ge-
ometry/curvature of the non-dominated fronts. By leveraging the
Newton-Raphson method, the novel technique estimates the ac-
tual geometry more accurately within a few iterations, allowing
its usage with negligible overhead. Finally, we introduce a novel
method to compute the diversity among the generated solutions us-
ing the geodesic distance, which is a generalization of the Euclidean
distance for curved Riemannian manifolds. We have implemented
these new methods within the AGE-MOEA framework, obtaining a
new MOEA, which we called AGE-MOEA-II.
We conducted an empirical study to assess the performance of
AGE-MOEA-II against its predecessor AGE-MOEA and other four state-
of-the-art MOEAs, namely NSGA-III, MOEA/D, LMEA, and VaEA. We
selected 17 test problems from two benchmarks, namely WFG and
SMOP, that are characterized by different front shapes and opti-
mization challenges. The achieved results show that AGE-MOEA-II
significantly outperforms the other MOEAs w.r.t. the HV indicator,
also when increasing the number of objectives from " = 3 to 9.
In our future work, we aim to assess the impact of using different
normalization methods \[5\], different numerical methods to approx-
imate the geodesic distance, as well as incorporating front model-
ing methods into other MOEA categories, such as decomposition-
based \[39\] and clustering-based \[6, 40\] MOEAs. Furthermore, we
plan to apply AGE-MOEA-II to real-world engineering problems,
such as regression testing \[3, 4, 26\], test case generation \[15, 30\],
and multi-objective predictive models \[8\].

572

<details>
<summary>Page 8: complete visual reference</summary>

![Full source page 8](K:/source/OptunaSharp/docs/papers/paper-gateway/library/72851351a07ce030/pages/page-0008.png)

</details>

> Quality note: Rotated or vertical text retains geometry and requires visual order review.
> Quality note: Mathematical notation or superscripts require visual review; linear text does not establish fraction, exponent or equation semantics.

<a id="page-9"></a>

<!-- Page 9 -->

An Improved Pareto Front Modeling Algorithm for Large-scale Many-Objective Optimization GECCO ’22, July 9–13, 2022, Boston, MA, USA

## REFERENCES

\[1\] Ram Bhushan Agrawal, K Deb, and RB Agrawal. 1995. Simulated binary crossover
for continuous search space. Complex systems 9, 2 (1995), 115–148.
\[2\] Myron B Allen III and Eli L Isaacson. 2011. Numerical analysis for applied science.
Vol. 35. John wiley &amp; sons.
\[3\] Christian Birchler, Nicolas Ganz, Sajad Khatiri, Alessio Gambi, and Sebastiano
Panichella. 2022. Cost-effective simulation-based test selection in self-driving
cars software with SDC-Scissor. In 29th IEEE International Conference on Software
Analysis, Evolution, and Reengineering, Honolulu, USA (online), 15-18 March 2022.
ZHAW Zürcher Hochschule für Angewandte Wissenschaften.
\[4\] Christian Birchler, Sajad Khatiri, Pouria Derakhshanfar, Sebastiano Panichella,
and Annibale Panichella. 2021. Single and Multi-objective Test Cases Prioriti-
zation for Self-driving Cars in Virtual Environments. https://doi.org/10.48550/
ARXIV.2107.09614
\[5\] Julian Blank, Kalyanmoy Deb, and Proteek Chandan Roy. 2019. Investigating the
normalization procedure of NSGA-III. In International Conference on Evolutionary
Multi-Criterion Optimization. Springer, 229–240.
\[6\] Anton Bouter, Ngoc Hoang Luong, Cees Witteveen, Tanja Alderliesten, and
Peter AN Bosman. 2017. The multi-objective real-valued gene-pool optimal
mixing evolutionary algorithm. In Proceedings of the Genetic and Evolutionary
Computation Conference. 537–544.
\[7\] Dimo Brockhoff, Tobias Wagner, and Heike Trautmann. 2012. On the properties
of the R2 indicator. In Proceedings of the 14th annual conference on Genetic and
evolutionary computation. 465–472.
\[8\] Gerardo Canfora, Andrea De Lucia, Massimiliano Di Penta, Rocco Oliveto, Anni-
bale Panichella, and Sebastiano Panichella. 2013. Multi-objective cross-project
defect prediction. In 2013 IEEE Sixth International Conference on Software Testing,
Verification and Validation. IEEE, 252–261.
\[9\] Ran Cheng, Miqing Li, Ye Tian, Xingyi Zhang, Shengxiang Yang, Yaochu Jin,
and Xin Yao. 2017. A benchmark test suite for evolutionary many-objective
optimization. Complex &amp; Intelligent Systems 3, 1 (01 Mar 2017), 67–81. https:
//doi.org/10.1007/s40747-017-0039-7
\[10\] W. J. Conover. 1998. Practical Nonparametric Statistics (3rd edition ed.). Wiley.
\[11\] Indraneel Das and John E Dennis. 1998. Normal-boundary intersection: A new
method for generating the Pareto surface in nonlinear multicriteria optimization
problems. SIAM Journal on Optimization 8, 3 (1998), 631–657.
\[12\] Kalyanmoy Deb. 2001. Multi-objective optimization using evolutionary algorithms.
Vol. 16. John Wiley &amp; Sons.
\[13\] K. Deb and H. Jain. 2014. An Evolutionary Many-Objective Optimization Al-
gorithm Using Reference-Point-Based Nondominated Sorting Approach, Part I:
Solving Problems With Box Constraints. IEEE Transactions on Evolutionary Com-
putation 18, 4 (Aug 2014), 577–601. https://doi.org/10.1109/TEVC.2013.2281535
\[14\] Kalyanmoy Deb, Amrit Pratap, Sameer Agarwal, and TAMT Meyarivan. 2002. A
fast and elitist multiobjective genetic algorithm: NSGA-II. IEEE transactions on
evolutionary computation 6, 2 (2002), 182–197.
\[15\] Pouria Derakhshanfar, Xavier Devroey, Andy Zaidman, Arie Van Deursen, and
Annibale Panichella. 2020. Good things come in threes: Improving search-based
crash reproduction with helper objectives. In 2020 35th IEEE/ACM International
Conference on Automated Software Engineering (ASE). IEEE, 211–223.
\[16\] Giuseppe Di Domenico, Dror Weisman, Annibale Panichella, Dolev Roitman, and
Ady Arie. 2022. Large-Scale Inverse Design of a Planar On-Chip Mode Sorter.
ACS Photonics (2022).
\[17\] JC Ehiwario and SO Aghamie. 2014. Comparative study of bisection, Newton-
Raphson and secant methods of root-finding problems. IOSR Journal of Engineer-
ing 4, 4 (2014), 01–07.
\[18\] Simon Huband, Philip Hingston, Luigi Barone, and Lyndon While. 2006. A
review of multiobjective test problems and a scalable test problem toolkit. IEEE
Transactions on Evolutionary Computation 10, 5 (2006), 477–506.
\[19\] Eugene Isaacson and Herbert Bishop Keller. 2012. Analysis of numerical methods.
Courier Corporation.
\[20\] Hisao Ishibuchi, Hiroyuki Masuda, Yuki Tanigaki, and Yusuke Nojima. 2015.
Modified distance calculation in generational distance and inverted generational
distance. In International conference on evolutionary multi-criterion optimization.

Springer, 110–125.
\[21\] H. Jain and K. Deb. 2014. An Evolutionary Many-Objective Optimization Algo-
rithm Using Reference-Point Based Nondominated Sorting Approach, Part II:
Handling Constraints and Extending to an Adaptive Approach. IEEE Transactions
on Evolutionary Computation 18, 4 (Aug 2014), 602–622. https://doi.org/10.1109/
TEVC.2013.2281534
\[22\] John M Lee. 2018. Introduction to Riemannian manifolds. Springer.
\[23\] Zhi-Zhong Liu, Yong Wang, and Pei-Qiu Huang. 2020. AnD: A many-objective
evolutionary algorithm with angle-based selection and shift-based density esti-
mation. Information Sciences 509 (2020), 400–419.
\[24\] R Timothy Marler and Jasbir S Arora. 2004. Survey of multi-objective optimization
methods for engineering. Structural and multidisciplinary optimization 26, 6
(2004), 369–395.
\[25\] Saúl Zapotecas Martínez, Víctor A Sosa Hernández, Hernán Aguirre, Kiyoshi
Tanaka, and Carlos A Coello Coello. 2014. Using a family of curves to approxi-
mate the Pareto front of a multi-objective optimization problem. In International
Conference on Parallel Problem Solving from Nature. Springer, 682–691.
\[26\] Mitchell Olsthoorn and Annibale Panichella. 2021. Multi-objective test case
selection through linkage learning-based crossover. In International Symposium
on Search Based Software Engineering. Springer, 87–102.
\[27\] John Oprea. 2019. Differential geometry and its applications. Vol. 59. American
Mathematical Soc.
\[28\] Suat Özdemir, A Attea Bara’a, and Önder A Khalil. 2013. Multi-objective evo-
lutionary algorithm based on decomposition for energy efficient coverage in
wireless sensor networks. Wireless personal communications 71, 1 (2013), 195–215.
\[29\] Annibale Panichella. 2019. An adaptive evolutionary algorithm based on non-
euclidean geometry for many-objective optimization. In Proceedings of the Genetic
and Evolutionary Computation Conference. 595–603.
\[30\] Annibale Panichella, Fitsum Meshesha Kifetew, and Paolo Tonella. 2015. Refor-
mulating branch coverage as a many-objective optimization problem. In 2015
IEEE 8th international conference on software testing, verification and validation
(ICST). IEEE, 1–10.
\[31\] Alfio Quarteroni, Riccardo Sacco, and Fausto Saleri. 2010. Numerical mathematics.
Vol. 37. Springer Science &amp; Business Media.
\[32\] Günter Rudolph, Oliver Schütze, Christian Grimme, Christian Domínguez-
Medina, and Heike Trautmann. 2016. Optimal averaged Hausdorffarchives
for bi-objective problems: theoretical and numerical results. Computational
Optimization and Applications 64, 2 (2016), 589–618.
\[33\] Ke Shang, Hisao Ishibuchi, Linjun He, and Lie Meng Pang. 2020. A survey on
the hypervolume indicator in evolutionary multiobjective optimization. IEEE
Transactions on Evolutionary Computation 25, 1 (2020), 1–20.
\[34\] Anthony C Thompson and Anthony C Thompson. 1996. Minkowski geometry.
Vol. 63. Cambridge University Press.
\[35\] Ye Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin. 2017. PlatEMO: A MATLAB
Platform for Evolutionary Multi-Objective Optimization. IEEE Computational
Intelligence Magazine 12 (2017), 73–87. Issue 4.
\[36\] Ye Tian, Xingyi Zhang, Ran Cheng, Cheng He, and Yaochu Jin. 2018. Guiding
Evolutionary Multio-bjective Optimization With Generic Front Modeling. IEEE
transactions on cybernetics (2018).
\[37\] Ye Tian, Xingyi Zhang, Chao Wang, and Yaochu Jin. 2019. An evolutionary
algorithm for large-scale sparse multiobjective optimization problems. IEEE
Transactions on Evolutionary Computation 24, 2 (2019), 380–393.
\[38\] Tjalling J Ypma. 1995. Historical development of the Newton–Raphson method.
SIAM review 37, 4 (1995), 531–551.
\[39\] Qingfu Zhang and Hui Li. 2007. MOEA/D: A multiobjective evolutionary algo-
rithm based on decomposition. IEEE Transactions on evolutionary computation
11, 6 (2007), 712–731.
\[40\] Xingyi Zhang, Ye Tian, Ran Cheng, and Yaochu Jin. 2016. A decision variable
clustering-based evolutionary algorithm for large-scale many-objective optimiza-
tion. IEEE Transactions on Evolutionary Computation 22, 1 (2016), 97–112.
\[41\] Eckart Zitzler and Simon Künzli. 2004. Indicator-based selection in multiobjec-
tive search. In International conference on parallel problem solving from nature.
Springer, 832–842.

573

<details>
<summary>Page 9: complete visual reference</summary>

![Full source page 9](K:/source/OptunaSharp/docs/papers/paper-gateway/library/72851351a07ce030/pages/page-0009.png)

</details>
