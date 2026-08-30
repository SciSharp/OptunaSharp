# OptunaSharp Project Instructions

OptunaSharp is a **.NET port of [Optuna](https://github.com/optuna/optuna)** — the define-by-run
hyperparameter-optimization framework — targeting **1-to-1 API and behavioral compatibility with
Optuna 5.x**. It is a [SciSharp STACK](https://github.com/SciSharp) project and builds on
[NumSharp](https://github.com/SciSharp/NumSharp) for numerical work.

> **Status: scaffolding.** This repository currently contains the *infrastructure* only — solution,
> projects, strong-naming, tests, pythonnet interop harness, benchmark harness, packaging, DocFX
> docs, and CI/CD. **No Optuna algorithms are ported yet.** The pipeline is green from day one so the
> port can grow incrementally against it.

**The port is governed by [`docs/PORTING_PLAN.md`](../docs/PORTING_PLAN.md)** — the authoritative
roadmap (phases, package map, per-phase gates, timeline, risk register). Read it before porting any
Optuna module. This file (`CLAUDE.md`) documents the **infrastructure**; the plan documents **what
to build and in what order**.

---

## Reference sources

- **Optuna (the source of truth):** a full clone lives at `K:/source/optuna` (pinned in the plan).
  Use it as the authoritative reference for API behavior, edge cases, defaults, and error messages.
  **Run actual Python/Optuna first, observe, then replicate in C#.**
- **rustuna (accelerator reference):** `K:/source/rustuna` — a Rust Optuna port, useful for the
  SQLite schema and Sobol direction numbers (see the plan).
- **NumSharp (the numerical foundation + the infra template):** `K:/source/NumSharp`. OptunaSharp's
  whole stack is modeled on NumSharp's; when in doubt about an infra decision, look at how NumSharp
  does it.

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
  numeric kernels stay single-threaded (NumSharp house rule).

---

## Repository layout

```
OptunaSharp/
├── src/
│   └── OptunaSharp.Core/            # THE package — OptunaSharp (net8.0;net10.0). Empty scaffolding today.
│       ├── OptunaSharp.Core.csproj  #   packable; PackageIcon; conditional NumSharp reference
│       ├── Assembly/Properties.cs   #   InternalsVisibleTo (must carry PublicKey) + friend-key const
│       ├── Version.cs               #   OptunaSharp.OptunaSharpVersion (the only public symbol so far)
│       └── README.md                #   packed into the .nupkg
├── test/
│   ├── OptunaSharp.Tests/           # unit tests — MSTest v3 + AwesomeAssertions
│   ├── OptunaSharp.Tests.Interop/   # LIVE-Optuna parity harness (pythonnet)
│   └── OptunaSharp.Benchmark/       # BenchmarkDotNet harness (template; grows with the ops)
├── docs/
│   ├── PORTING_PLAN.md              # the governing conversion roadmap
│   └── website-src/                 # DocFX site (docfx.json, conceptual docs, generated API ref)
├── .github/
│   ├── workflows/                   # build-and-release · docs · benchmark
│   ├── scripts/verify_strong_name.cs
│   ├── ISSUE_TEMPLATE/ · PULL_REQUEST_TEMPLATE.md · FUNDING.yml · dependabot.yml
├── Directory.Build.props            # strong-naming + shared package metadata + version
├── Open.snk                         # Microsoft's published OSS key (shared across the SciSharp STACK)
├── NuGet.Config                     # nuget.org only (reproducible restore)
├── .editorconfig / .gitattributes / .gitignore / .git-blame-ignore-revs
├── LICENSE (MIT) · SECURITY.md · CITATION.cff · CONTRIBUTING.md · CODE_OF_CONDUCT.md
├── README.md
└── SciSharp.OptunaSharp.sln
```

### Projects

| Project | TFMs | Kind | Key deps | Purpose |
|---------|------|------|----------|---------|
| `src/OptunaSharp.Core` | net8.0; net10.0 | library, **packable** (`PackageId=OptunaSharp`) | NumSharp | The library. `RootNamespace`/`AssemblyName` = `OptunaSharp`. Emits XML docs. |
| `test/OptunaSharp.Tests` | net8.0; net10.0 | MSTest exe, `IsPackable=false` | MSTest 3.8.3, AwesomeAssertions 9.3.0 | Unit tests. |
| `test/OptunaSharp.Tests.Interop` | net8.0; net10.0 | MSTest exe, `IsPackable=false` | MSTest, AwesomeAssertions, **pythonnet 3.0.5** | Byte/behavior parity vs **live Optuna** in-process. |
| `test/OptunaSharp.Benchmark` | net8.0; net10.0 | exe, `IsPackable=false` | BenchmarkDotNet 0.15.8 | Perf harness (template). |

The solution (`SciSharp.OptunaSharp.sln`) contains only these four projects; NumSharp is referenced
by path (see below) but is **not** a solution member.

---

## The NumSharp reference — "csproj for now, nuget in future"

`OptunaSharp.Core` depends on NumSharp, resolved **conditionally on `Exists(...)`** in the csproj:

- **Local dev** — when a sibling `NumSharp` checkout exists at `../../../NumSharp/src/NumSharp.Core/`
  (i.e. `K:/source/NumSharp` beside `K:/source/OptunaSharp`), it is referenced by **ProjectReference**
  so changes flow live.
- **Standalone / CI** — otherwise it falls back to a **PackageReference** on `NumSharp 0.60.0`.

The two are mutually exclusive, so exactly one applies. Packing always records a clean `NumSharp`
package dependency either way (verified). **CI uses the NuGet fallback** — CI runners have no sibling
NumSharp checkout, which also sidesteps the cross-repo build friction below.

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

# Unit tests (what CI runs — both TFMs, minus known-failing/heavy/python categories)
dotnet test test/OptunaSharp.Tests/OptunaSharp.Tests.csproj -c Release \
  --filter "TestCategory!=OpenBugs&TestCategory!=HighMemory&TestCategory!=RequiresPython"

# Live-Optuna interop (needs a discoverable Python with `pip install optuna`)
dotnet test test/OptunaSharp.Tests.Interop/OptunaSharp.Tests.Interop.csproj -c Release

# Pack the OptunaSharp package
dotnet pack src/OptunaSharp.Core/OptunaSharp.Core.csproj -c Release -o packages
```

### Local-build gotcha: the sibling-NumSharp two-hop P2P

When a sibling NumSharp checkout is present, building OptunaSharp compiles `NumSharp.Core` via the
ProjectReference — which triggers **NumSharp's NDScope IL weaver** and its `GeneratePackageOnBuild`.
That is slow and pollutes `NumSharp/packages/`. For fast, side-effect-free local builds pass NumSharp's
own escape hatches (they propagate to the P2P dependency as global properties):

```bash
dotnet build SciSharp.OptunaSharp.sln -c Release -p:SkipNDScopeWeave=true -p:GeneratePackageOnBuild=false
```

There is also a **cold-build race** in the two-hop cross-repo P2P (`OptunaSharp.Tests[.Interop] →
OptunaSharp.Core → NumSharp.Core`): on a first build it can briefly resolve `NumSharp.Core` in
**Debug** and fail copying `NumSharp.dll` (MSB3030 "not found"). It **self-heals on retry**. CI never
hits it (NuGet fallback). Cause: the "Any CPU" (sln) vs "AnyCPU" (project) config-flow quirk.

### Test categories

Typed attributes in `test/OptunaSharp.Tests/TestCategory.cs` (so CI filter strings stay stable):

| Attribute | Meaning | CI |
|-----------|---------|----|
| `[OpenBugs]` | Known-failing reproduction / not-yet-ported behavior. Remove when it passes. | **excluded** |
| `[Misaligned]` | Intentional, documented difference from Optuna (the test passes). | runs |
| `[RequiresPython]` | Needs live Optuna via pythonnet; self-skips (Inconclusive) when absent. | interop job only |
| `[HighMemory]` | Resource-heavy. | **excluded** |
| `[WindowsOnly]` | Windows-only API. | excluded off-Windows |

MSTest attribute namespaces are supplied globally via `GlobalUsings.cs` in each test project (`global
using Microsoft.VisualStudio.TestTools.UnitTesting;`), so test files don't repeat the `using`. The
`[RequiresPython]` *attribute* lives in `OptunaSharp.Tests`; the interop project (which doesn't
reference it) uses the string form `[TestCategory("RequiresPython")]`.

---

## Testing infrastructure

### Unit tests (`OptunaSharp.Tests`)

Smoke tests today (`ScaffoldingTests.cs`): the assembly loads and is strong-named with token
`cc7b13ffcd2ddd51`, the version metadata is present, and the NumSharp dependency resolves and runs
(`np.arange(6).reshape(2,3)`). As modules land, add tests derived from real Optuna output.

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

- `dotnet pack src/OptunaSharp.Core/OptunaSharp.Core.csproj -c Release -o packages` produces
  `OptunaSharp.<version>.nupkg` with `lib/net8.0/` + `lib/net10.0/` assemblies (+ embedded `.pdb`),
  `README.md`, and the **`PackageIcon`** (`docs/website-src/images/optunasharp-icon.png`, packed via
  Core.csproj), and records the `NumSharp` dependency.
- `GeneratePackageOnBuild` is **not** set on Core, so a plain `dotnet build` doesn't auto-pack; CI
  packs explicitly. Package output path is `packages/` at the repo root (gitignored).
- Shared metadata (Authors/Company/Copyright/RepositoryUrl/`PackageLicenseExpression=MIT`) and the
  version (`VersionPrefix`/`AssemblyVersion`/`FileVersion`, default `0.1.0`) live in
  `Directory.Build.props`; CI overrides them from the pushed git tag.

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

Build it locally. The metadata step compiles Core, which (with a sibling NumSharp) would run
NumSharp's weaver/pack — suppress that by exporting the flags as environment variables (MSBuild reads
env vars as properties). **`-p:` is not a DocFX CLI flag.**

```bash
export SkipNDScopeWeave=true GeneratePackageOnBuild=false
docfx docs/website-src/docfx.json --serve          # http://localhost:8080
```

Output goes to `docs/website/` (gitignored). Verified to build **0 warnings / 0 errors** with DocFX
2.78.4.

---

## CI/CD

Three workflows in `.github/workflows/` (modeled on NumSharp's live pipeline). Env `DOTNET_NOWARN`
suppresses noisy XML-doc warnings.

### `build-and-release.yml`
- **`test`** — matrix `windows/ubuntu/macos`, .NET 8 + 10; builds `OptunaSharp.Tests` and runs both
  TFMs with `TestCategory!=OpenBugs&!=HighMemory&!=RequiresPython` (+ `!=WindowsOnly` off-Windows).
- **`verify-signing`** — asserts `Open.snk` size+sha256, packs a throwaway `OptunaSharp`, runs
  `verify_strong_name.cs` on it. Runs on **every push and PR** (a release-only gate is how "shipped
  unsigned" hides).
- **`interop-test`** — matrix; installs Python 3.12 + Optuna, resolves `PYTHONNET_PYDLL`, builds and
  tests `OptunaSharp.Tests.Interop` with `OPTUNASHARP_PYTHONNET_REQUIRE_ENGINE=1`.
- **Release pipeline** (on `v*` tags): `validate-release` (tag must be on `master`) →
  `build-nuget` (`-t:Rebuild` + pack + strong-name gate) → `create-release` + `publish-nuget` (uses
  `secrets.NUGETAPIKEY`).

### `docs.yml`
DocFX build → GitHub Pages. PRs build the docs as a gate (metadata must compile); only `master`
deploys. Triggers on `src/**`, `docs/website-src/**`, and the workflow file.

### `benchmark.yml`
`build-smoke` compiles the harness on every relevant push/PR; a full **short** run is manual-only
(`workflow_dispatch`) — a real perf suite isn't meaningful until algorithms exist.

### Other
`.github/dependabot.yml` (weekly nuget + github-actions), issue forms
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
- **Satellite packages** (`OptunaSharp.Storage.*`, `.Gp`, `.Importance.Forests`, `.Visualization`,
  `.Cli`, `.Artifacts.*`) do not exist yet — they arrive with their phases (plan §3 package map),
  each following the `NumSharp.Interop.OpenBLAS` optional-package pattern so Core stays pure-managed.

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
- **Nothing is pushed to any remote yet** — the repo is local-only (`git init` on `master`, no
  remote) until it's published.
