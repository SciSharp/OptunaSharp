# OptunaSharp — Dependency-Injection Delivery Plan

**Status: core surface delivered; remaining scope finalized below.** The Microsoft.Extensions.DependencyInjection
wire-up has shipped for Core and every DI-friendly satellite (commits `532fb08`, `1ff443c`, `cbc317b`,
`920d862`), with an all-packages "no-clash" host test project (`d71b98f`). This document is the authoritative
record of **what DI means in OptunaSharp**, **what is delivered**, **the rule for deciding what belongs in the
container**, and **the finalized to-do for the remaining gaps**.

> **DI here is a .NET-ergonomics layer, not an Optuna-parity feature.** Optuna's API is functional and
> define-by-run (`Optuna.CreateStudy(...)`); nothing about DI changes how any component behaves. Its only
> justification is consumer ergonomics on the .NET side: lifetime/disposal management, config-driven
> construction, host/worker integration, and swappability. A consumer who never touches the container pays
> nothing but a transitive reference.

---

## 1. Principles

1. **Optional and additive.** Every `Add*` is an opt-in extension in a `<namespace>.Extensions` namespace. The
   functional API (`new TPESampler(...)`, `Optuna.CreateStudy(...)`) remains the primary surface.
2. **Register the service, not the config.** A type belongs in the container when its construction pulls in
   *other services* (clients, connections, backends, evaluators) — i.e. it is a decorator, a composer, or a
   client-consumer. A type whose construction is dominated by per-run *value config* (hyperparameters,
   thresholds, seeds, search spaces, datasets) does **not** — it is created with `new` at the call site or
   built by a factory. See §4 for the decision rule.
3. **Core stays pure-managed.** The only dependency DI adds to Core is
   `Microsoft.Extensions.DependencyInjection.Abstractions` — pure-managed, BCL-adjacent, **not** a foreign
   runtime — so Core's "no foreign runtime" guarantee holds.
4. **TFM-matched versions.** The DI-abstractions version tracks the target framework: the 8.0.x line on
   `net8.0`, the 10.0.x line on `net10.0`. A `net8.0` consumer never has .NET 10-era `Microsoft.Extensions.*`
   dragged in. See §7.

---

## 2. Conventions (the shipped design)

- **Namespace:** each package exposes `<primary-type-full-namespace>.Extensions`
  (e.g. `OptunaSharp.Extensions`, `OptunaSharp.Artifacts.Extensions`,
  `OptunaSharp.Storages.Journal.Extensions`).
- **Package-unique class names.** Several packages deliberately share a namespace — the four artifact stores
  all use `OptunaSharp.Artifacts.Extensions`, the three Redis backends all use
  `OptunaSharp.Storages.Journal.Extensions`. A duplicate `ServiceCollectionExtensions` type name across two
  referenced assemblies is a hard `CS0433`, so each class is named uniquely
  (`S3ArtifactStoreServiceCollectionExtensions`, `StackExchangeRedisJournalServiceCollectionExtensions`, …).
- **Register concrete + abstraction.** Each `Add*` registers the concrete type and forwards the base
  abstraction (`BaseStorage` / `IArtifactStore` / `BaseSampler` / `BaseImportanceEvaluator` / …) to it, so a
  consumer can inject either.
- **Overloads.** A convenience overload (mirrors the primary constructor), an instance overload, and a
  `Func<IServiceProvider,T>` factory overload, each taking `ServiceLifetime lifetime = Singleton`.
- **Lazy for foreign runtimes.** The Interop-sampler convenience overloads register a *factory*, so embedded
  CPython is initialized on first resolution, not at registration.
- **gRPC uses the framework.** `OptunaSharp.Storage.Grpc` takes DI from the ASP.NET Core shared framework it
  already references — no `Microsoft.Extensions.DependencyInjection.Abstractions` package reference.

---

## 3. What is delivered

| Package | Namespace | Extensions (abstraction registered) |
|---|---|---|
| Core | `OptunaSharp.Extensions` | `AddOptunaSampler` (`BaseSampler`), `AddOptunaPruner` (`BasePruner`), `AddOptunaStorage` (`BaseStorage`; instance / factory / URL), `AddOptunaStudyFactory` → `IOptunaStudyFactory` + `OptunaStudyOptions` |
| Artifacts.S3 / Gcs / AzureBlob / R2 | `OptunaSharp.Artifacts.Extensions` | `AddS3/Gcs/AzureBlob/R2ArtifactStore` (`IArtifactStore`) |
| Storage.Sqlite | `OptunaSharp.Storages.Extensions` | `AddSqliteStorage` (`BaseStorage`) |
| Storage.EFCore | `OptunaSharp.Storages.EFCore.Extensions` | `AddEFCoreStorage` (`BaseStorage`) |
| Storage.Grpc (client) | `OptunaSharp.Storages.Grpc.Extensions` | `AddOptunaGrpcStorageProxyClient` (`BaseStorage`) |
| Storage.Grpc (server) | `OptunaSharp.Storages.Grpc` | `AddOptunaStorageProxy` / `MapOptunaStorageProxy` (pre-existing) |
| Storage.{StackExchange,NRedisStack,ServiceStack}.Redis | `OptunaSharp.Storages.Journal.Extensions` | `Add{…}RedisJournalStorage` (backend + `JournalStorage` + `BaseStorage`) |
| Interop.BoTorch / PyCma / Pymoo | `OptunaSharp.Interop.{BoTorch,PyCma,Pymoo}.Extensions` | `AddBoTorch/PyCma/PymooSampler` (`BaseSampler`, lazy) |
| Interop.pythonnet | `OptunaSharp.Interop.PythonNet.Extensions` | `AddOptunaHubSampler` (`BaseSampler`), `AddOptunaHubPruner` (`BasePruner`), lazy |
| Integration.LightGbm | `OptunaSharp.Integration.LightGbm.Extensions` | `AddLightGbmTrainer` (`ILightGbmTrainer` + `ILightGbmBoosterStore`), `AddLightGbm{Fanova,MeanDecreaseImpurity}ImportanceEvaluator` |
| Integration.ML.NET | `OptunaSharp.Integration.MLNet.Extensions` | `AddFastForest*/FastTree*/FastTreeTweedie*ImportanceEvaluator`, generic `AddMLNetImportanceEvaluator<T>`, `AddMLNetEstimator` (`ICrossValidationEstimator`, factory) |
| Visualization.Plotly | `OptunaSharp.Visualization.Plotly.Extensions` | `AddOptunaPlotlyVisualization` → `IOptunaPlotlyVisualizer` facade over the static `PlotlyVisualization` |

**Verification:** `test/OptunaSharp.Tests.Hosting` references all 18 DI packages at once — the single
compilation is the `CS0433` no-clash proof — and `AllPackagesDiHostTests` (4 tests, both TFMs) asserts exact
descriptor counts (coexistence), resolves the offline-constructible services, proves all four artifact stores
resolve via DI-provided clients, and confirms the Interop samplers register lazily (no Python). Unit suite
824/824 on `net8.0` and `net10.0`.

---

## 4. The decision rule — what belongs in the container

Classify by **constructor shape**, distinguishing the *kind* of each parameter:

- **Service dependency** — an abstraction/interface/client you would resolve from the container
  (`IArtifactStore`, `BaseStorage`, `BaseJournalBackend`, `IAmazonS3`, `ILightGbmTrainer`,
  `BaseImprovementEvaluator`, …). Presence ⇒ **DI citizen**.
- **Value config** — primitives, enums, collections, delegates, thresholds, seeds. Volume ⇒
  **over-parameterized**, create with `new`.
- **Per-run data** — a `Study`, a dataset/`IDataView`/`Dataset`, param distributions, a search space. Presence
  ⇒ **not a singleton**; if it also has a service dependency, register a **factory**.

> **Footgun corrected:** a sampler's `independentSampler` / `crossover` / `mutation` parameters look like
> service dependencies but are *inline strategies* (`new RandomSampler(seed)`, `new SBXCrossover(...)`) — you
> do not resolve a crossover from the container. Samplers are therefore **over-parameterized**, not DI-heavy.

### Bucket A — DI-heavy (register as services)

| Type | Service dependency | Role |
|---|---|---|
| `Backoff` | `IArtifactStore` | retry decorator |
| `CachedStorage` | `BaseStorage` | decorator |
| `JournalStorage` | `BaseJournalBackend` | wrapper (wired for Redis; **not** for the file backend) |
| `PatientPruner` | `BasePruner` | decorator |
| `Terminator` | `BaseImprovementEvaluator` + `BaseErrorEvaluator` | composer |
| `MedianErrorEvaluator` | `BaseImprovementEvaluator` | composer |
| *(shipped)* cloud artifact stores, `RDBStorage`, `GrpcStorageProxy`, Redis backends | clients / connections | client-consumer |

### Bucket B — over-parameterized (create with `new`, not the container)

Ctor dominated by per-run value config; the generic `AddOptunaSampler/Pruner/Storage` instance/factory seams
already cover the rare "I want it in the container" case.

- **Samplers (all native):** `JanusSampler` (26 ctor params), `JanusDeSampler` (19), the Janus cmix variants
  (16–17), `TPESampler` / `CmaEsSampler` (14), `NSGAIII`(+CPT) (13–14), `RveaSampler` (14), `NSGAIISampler` /
  `PsoSampler` (11), `MoeaDSampler` (10), `SmsEmoaSampler` / `DifferentialEvolutionSampler` / `VkdCmaSampler`
  (8), `AgeMoea2Sampler` / `LmCmaSampler` (7), `GPSampler` / `QMCSampler` (6), `GridSampler` /
  `PartialFixedSampler` / `BruteForceSampler` / `RandomSampler` (data / 1–2 knobs).
- **Pruners:** `PercentilePruner` (5), `MedianPruner` / `HyperbandPruner` / `SuccessiveHalvingPruner` /
  `ThresholdPruner` (4), `WilcoxonPruner` (2), `NopPruner` (0). *(`PatientPruner` is Bucket A.)*
- **Core importance evaluators:** `FanovaImportanceEvaluator`, `MeanDecreaseImpurityImportanceEvaluator`,
  `PedAnovaImportanceEvaluator`, `ShapleyImportanceEvaluator` — 3 value knobs each (light).
- **Terminator leaf evaluators:** `EMMREvaluator` (4), `RegretBoundEvaluator` (3),
  `BestValueStagnationEvaluator` (1), `StaticErrorEvaluator` (1), `CrossValidationErrorEvaluator` (0).

### Bucket C — per-run operations (factory registration, never a singleton)

Carry per-run data **and** an injectable dependency ⇒ register with a `Func<IServiceProvider,T>` that resolves
the dependency while the caller supplies the data.

| Type | Injectable dep | Per-run data |
|---|---|---|
| `OptunaSearchCV` | `ICrossValidationEstimator` | `Study` + param distributions |
| `LightGBMTuner` / `LightGBMTunerCV` | `ILightGbmTrainer` | `Dataset` + `Study` + params |
| `MLNetEstimator` *(shipped, factory-only)* | `MLContext` | `IDataView` + pipeline + scorer |
| `SklearnEstimator` | `IRegressor` | data + `BaseCrossValidator` + scorer |

### Excluded — not DI services

Domain/data (`Study`, `Trial`, `FrozenTrial`, distributions, gRPC proto messages), logging
(`Logger`, `StreamLogHandler`), internal mechanisms (`HeartbeatThread`, `BaseJournalFileLock` impls),
sampler-internal strategies (`*ChildGenerationStrategy`, `*ElitePopulationSelectionStrategy`, `BaseCrossover`
/ `BaseMutation` impls), callbacks (`MaxTrialsCallback`, `RetryFailedTrialCallback`, `TerminatorCallback`,
`PruningCallback`, …), and static facades (`PlotlyVisualization`, `GrpcProxyServer`).

---

## 5. Coverage audit (source of truth: Optuna module `__all__`)

Cross-checking Optuna's own public API against our DI surface shows the delivered set skews toward the
*beyond-Optuna satellites* (cloud artifact stores, ML.NET/LightGBM off-label evaluators) while missing several
of Optuna's *canonical, top-level, heavily-used* components. These are the genuine gaps to close.

| Optuna canonical component | Ported as | In DI? |
|---|---|---|
| `FileSystemArtifactStore`, `Backoff` | same | **No** (cloud stores are; local + decorator are not) |
| `JournalFileStorage` / file backend | `JournalFileBackend` + `JournalStorage` | **No** (Redis journal is) |
| `FanovaImportanceEvaluator`, `MeanDecreaseImpurityImportanceEvaluator`, `PedAnovaImportanceEvaluator` | same | **No** (off-label ML.NET/LightGBM ones are) |
| `Terminator` + improvement/error evaluators | same | **No** (whole category) |
| `SklearnEstimator` (`ICrossValidationEstimator`) | same | **No** (`MLNetEstimator` is) |

---

## 6. Finalized scope — the to-do

### 6.1 Add to DI (genuine service gaps)

1. **`OptunaSharp.Artifacts` (Core):** `AddFileSystemArtifactStore(string baseDir)` and a **decorator**
   `AddArtifactStoreBackoff(...)` that wraps the already-registered `IArtifactStore`. Requires a Core
   `Microsoft.Extensions.DependencyInjection.Abstractions` reference and a new
   `OptunaSharp.Artifacts.Extensions` in Core (the satellites already use that namespace — keep class names
   unique).
2. **`OptunaSharp.Storages.Journal` (Core):** `AddJournalFileStorage(string filePath, ...)` — build a
   `JournalFileBackend`, wrap it in a `JournalStorage`, forward `BaseStorage` (mirrors the Redis extensions).
3. **`OptunaSharp.Terminator` (Core):** `AddOptunaTerminator(...)` — compose a registered
   `BaseImprovementEvaluator` + `BaseErrorEvaluator` into a `Terminator` (`BaseTerminator`), plus light
   `Add*` for the evaluators consumers want to inject.
4. **`OptunaSharp.Integration.Sklearn` (Core):** `AddSklearnEstimator(Func<IServiceProvider,SklearnEstimator>)`
   — factory-only (per-run data), mirroring `AddMLNetEstimator`.
5. **Core importance parity (light):** `AddOptunaFanovaImportanceEvaluator` /
   `AddOptunaMeanDecreaseImpurityImportanceEvaluator` / `AddOptunaPedAnovaImportanceEvaluator` /
   `AddOptunaShapleyImportanceEvaluator` (`BaseImportanceEvaluator`). These are light value-config, but the
   off-label satellite evaluators already have `Add*`; adding the canonical ones removes the inconsistency.

All of the above register concrete + abstraction with the standard convenience / instance / factory
overloads, and follow the TFM-matched version rule (Core is `net8.0`-clean at 8.0.x).

### 6.2 Stay OUT of DI (create with `new`, or via factory)

- **Samplers and pruners:** no per-type `Add*`. Configured per study; the generic `AddOptunaSampler` /
  `AddOptunaPruner` (instance / factory) already cover container registration for the rare case.
- **Terminator leaf evaluators and Core importance value knobs:** creatable with `new`; the light `Add*` in
  §6.1 is the only concession, for parity.
- **Per-run operations** (`OptunaSearchCV`, `LightGBMTuner` / `LightGBMTunerCV`): factory registration only if
  a consumer asks — never singletons.

---

## 7. Versioning & packaging

- **Single-sourced, TFM-matched.** `$(MicrosoftExtensionsDIVersion)` in `Directory.Build.props` resolves to
  the 8.0.x line on `net8.0` and the 10.0.x line on `net10.0` (per-inner-build `$(TargetFramework)`).
- **NU1605 floor rule.** A direct reference below a transitive requirement is a package downgrade
  (warnings-as-errors). Where a third-party lib forced a higher DI line on `net8.0`, the lib itself is
  version-split per TFM so `net8.0` stays on the 8.0.x line:
  - `Artifacts.AzureBlob`: `Azure.Storage.Blobs 12.26.0` / `Azure.Identity 1.13.2` on `net8.0`.
  - `Storage.StackExchange.Redis`: `StackExchange.Redis 2.13.17` on `net8.0` (the entire 3.x line is
    .NET 10-tied).
  - `Storage.NRedisStack`: `NRedisStack 1.5.0` on `net8.0`.
  - `Storage.EFCore`: **EF Core 8** (`Npgsql.EntityFrameworkCore.PostgreSQL 8.0.11` / `Pomelo 8.0.3`) on
    `net8.0`, via a small `#if NET8_0` fork of two Npgsql-9-only calls (`NpgsqlDbContextOptionsBuilder.MapEnum`
    → model `HasPostgresEnum`; `NpgsqlDataSource.ReloadTypes` → connection `ReloadTypes`). Schema is
    EF-version-independent (created by `RdbSchema` raw SQL).
  - **Result: all 18 DI packages resolve DI abstractions 8.0.2 on `net8.0`, 10.0.12 on `net10.0`.**
- **Caveat:** the EF Core 8 enum path is compile-verified; confirm its byte-exact schema behavior with the
  live-PostgreSQL / MySQL Testcontainers integration tests before release.

---

## 8. Testing

- **`test/OptunaSharp.Tests.Hosting`** — the all-packages host: coexistence (descriptor counts), offline
  resolution, artifact stores via DI-provided clients, lazy Interop-sampler registration. Every new §6.1
  `Add*` must be added to `ConfigureEverything` and its descriptor count asserted.
- **`test/OptunaSharp.Tests/DependencyInjectionTests.cs`** — Core composition-root behavior
  (`IOptunaStudyFactory`, option defaults, override precedence, transient lifetime, the Plotly facade).
- The host project is intentionally **not** wired into CI (it pulls the entire dependency graph); add a
  single-OS gate if a continuous no-clash check is wanted.

---

## 9. Non-goals & rationale

- **No per-sampler / per-pruner `Add*`.** They are per-study value config; a container singleton is
  meaningless and the generic seams suffice. (Bucket B.)
- **No DI for domain, logging, callbacks, operators, or static facades.** Not services.
- **Core does not take a foreign runtime.** Only the managed DI abstractions.
- **DI never changes behavior.** It is packaging/ergonomics; the functional API is authoritative.

---

## Appendix A — full type inventory by bucket

- **In DI (delivered):** 4 cloud artifact stores · Sqlite/EFCore RDB · gRPC client (+ server) · 3 Redis
  journal backends · BoTorch/PyCma/Pymoo/OptunaHub samplers · OptunaHub pruner · 8 satellite importance
  evaluators · `MlNetLightGbmTrainer` · `MLNetEstimator` · Plotly facade · Core sampler/pruner/storage seams +
  `IOptunaStudyFactory`.
- **Bucket A (add — DI-heavy):** `FileSystemArtifactStore`, `Backoff`, file-`JournalStorage`, `Terminator`,
  `MedianErrorEvaluator`, `SklearnEstimator` (factory), plus already-shipped decorators/consumers.
  *(`CachedStorage`, `PatientPruner` are DI-shaped but internal/rare — optional.)*
- **Bucket B (create with `new`):** 26 native samplers; `MedianPruner`, `PercentilePruner`,
  `HyperbandPruner`, `SuccessiveHalvingPruner`, `ThresholdPruner`, `WilcoxonPruner`, `NopPruner`; Core
  importance evaluators (light `Add*` in §6.1 for parity); terminator leaf evaluators.
- **Bucket C (factory, per-run):** `OptunaSearchCV`, `LightGBMTuner`, `LightGBMTunerCV`, `MLNetEstimator`
  (done), `SklearnEstimator`.
- **Excluded (not services):** domain/data types, `Logger`/`ILogHandler`, `IRegressor`/`RandomForestRegressor`,
  heartbeat threads, journal file locks, GA operator strategies, callbacks, static facades.

## Appendix B — delivery commits

| Commit | Scope |
|---|---|
| `532fb08` | DI wire-up for Core + all DI-friendly packages (`<ns>.Extensions`) |
| `1ff443c` | TFM-match the DI abstractions (8.0.x `net8.0` / 10.0.x `net10.0`) |
| `cbc317b` | Version-split `net8.0` third-party libs (Azure / Redis) off the .NET 10 line |
| `920d862` | `Storage.EFCore` uses EF Core 8 on `net8.0` (`#if NET8_0` enum/reload fork) |
| `d71b98f` | `OptunaSharp.Tests.Hosting` — all-packages DI host proving no clash + resolution |
