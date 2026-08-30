# OptunaSharp — Optuna → C# Conversion Plan

**Source of truth:** `optuna/optuna` @ `f6898da` (v5.0.0.dev, cloned at `K:/source/optuna`, vendored as a pinned submodule at `src/optuna`) — 24,474 core SLOC
(tokenizer-counted, comments/docstrings excluded). Secondary reference: `optuna/rustuna` @ `8acb2dd`
(`K:/source/rustuna`) — used ONLY where it accelerates (SQLite schema, Sobol tables, static-type designs);
Optuna's Python behavior always wins on conflict.

**Ordering directives (two, composed):**
1. Core / infra / storage / persistence before features; features by difficulty, easiest first.
2. **Native before external:** everything implementable with **BCL + NumSharp.Core only** (Tier N,
   Phases 0–3) is completed before anything pulling a third-party NuGet, native runtime, or network
   service (Tier X, Phases 4–5). Target: **almost all features**.

**The tier rule (what counts as "external"):** a work item is Tier X iff it requires any dependency
beyond the .NET BCL and `NumSharp.Core` — third-party managed packages (ML.NET, StackExchange.Redis,
Grpc.Net, System.CommandLine, YamlDotNet, Plotly.NET, Microsoft.Data.Analysis, EF Core), native
binaries (Microsoft.Data.Sqlite's e_sqlite3, TorchSharp, `NumSharp.Interop.OpenBLAS`), or external
services (Redis, PostgreSQL/MySQL, S3/GCS, gRPC servers). Everything else is Tier N.
`NumSharp.Interop.OpenBLAS` is first-party but bundles native binaries → Tier X by rule; the native
tier must be fully functional without it (it appears only as an optional byte-exactness upgrade).

**Tier N headline (the release claim this ordering buys):** `OptunaSharp` Core ships **Optuna's
entire algorithm surface — every sampler incl. TPE/NSGA/CMA-ES/GP, every pruner, MO machinery,
PED-ANOVA, terminators, in-memory + journal-file persistence — pure managed, zero dependencies
beyond NumSharp**. Cross-language persistence is ALREADY in this tier: Optuna's journal-file format
is JSON-lines over plain file IO, so Python Optuna and OptunaSharp can share a live journal file
with no external package on either side.

---

## 1. Goals & non-goals

**Goals**
- API-compatible .NET port: an Optuna user reads Optuna's docs and writes the same call in C#.
- Behavior parity proven the NumSharp way: Python Optuna is the **oracle**; committed corpora are
  replayed bit-exact in CI (no Python at test time). NumSharp's byte-identical MT19937 `RandomState`
  makes sampler sequences reproducible bit-for-bit.
- Storage-level interop: journal-file (Tier N) and SQLite `.db` / gRPC (Tier X) studies open in
  Python Optuna and vice versa.
- NumSharp is the numeric backbone; plain C# where arrays add nothing.

**Non-goals (explicit cuts)**
- `optuna.integration.*` (LightGBM/PyTorch/Keras callbacks) — framework adapters, not ports; revisit
  per-framework (ML.NET/TorchSharp) after v1.
- Interactive visualization rendering in Core — the **data layer** (pure-numpy `_get_*_info`
  functions) is Tier N; rendering is a Tier X package.
- OptunaHub, Dashboard, `optuna.testing` (we build our own test infra).

---

## 2. Ground rules

1. **Match Optuna exactly** — run the Python first, observe, replicate; edge cases (NaN, single-value
   distributions, dynamic search spaces, `None` choices) matter. Verbatim error messages where Optuna
   raises typed errors.
2. **RNG discipline:** every sampler consumes `np.random.RandomState` through a `LazyRandomState`
   port. Never `System.Random`. This is what makes differential testing bit-exact.
3. **Categorical value semantics:** `CategoricalChoiceType = None|bool|int|float|str` maps to a
   `CategoricalChoice` value type implementing **Python equality** (`True == 1`, `1.0 == 1` — see
   `distributions.py:509` note). Pinned by tests; the one place C# object equality silently diverges.
4. **Tier N is dependency-pure:** Core references NumSharp.Core and nothing else. Even logging is a
   self-contained seam (port of Optuna's own `logging.py` verbosity API); the
   `Microsoft.Extensions.Logging` adapter is Tier X. `trials_dataframe` returns a plain typed row
   table in Core; the `Microsoft.Data.Analysis` adapter is Tier X.
5. **Package optionality mirrors Optuna's:** anything needing a foreign runtime is a separate NuGet
   package — the NumSharp.Interop.OpenBLAS pattern.
6. **Threading:** trial-level parallelism (`n_jobs`, distributed workers) is supported and tested;
   numeric kernels stay single-threaded (NumSharp house rule).
7. **Every phase ends green:** its oracle tier committed + replayed in CI, unit tests passing, and —
   for storage phases — a live Python↔C# interop script (the `verify_npy_interop.py` model).

---

## 3. Package map

| Tier | Package | Contents | Deps beyond BCL |
|---|---|---|---|
| N | `OptunaSharp` (Core) | distributions, trial, study, ALL samplers (Random/Grid/PartialFixed/BruteForce/TPE/QMC/NSGA-II/NSGA-III/CMA-ES/**GP**), all 9 pruners, transform, MO machinery, PED-ANOVA, terminators (incl. GP-based), InMemory + **journal-file** storage, filesystem artifacts, viz data layer, own managed `eigh`(Jacobi)/`cholesky`/L-BFGS-B | **NumSharp.Core only** |
| X | `OptunaSharp.Storage.Sqlite` | RDB storage, Optuna-compatible schema + alembic version pin | Microsoft.Data.Sqlite |
| X | `OptunaSharp.Storage.EFCore` | same schema on PostgreSQL/MySQL | EF Core + providers |
| X | `OptunaSharp.Storage.Redis` | journal Redis backend | StackExchange.Redis |
| X | `OptunaSharp.Storage.Grpc` | storage proxy client + server | Grpc.Net, Google.Protobuf |
| X | `OptunaSharp.Importance.Forests` | fANOVA, MDI | ML.NET (managed-CART exact port = later Tier N option) |
| X | `OptunaSharp.Gp.Torch` | TorchSharp GP backend — ONLY if the native GP gate fails, or as alt backend | TorchSharp |
| X | `OptunaSharp.Artifacts.S3` / `.Gcs` | cloud artifact stores | AWSSDK / Google.Cloud.Storage |
| X | `OptunaSharp.Visualization.Plotly` | rendering over the Core data layer | Plotly.NET |
| X | `OptunaSharp.Cli` | `optuna` CLI | System.CommandLine, YamlDotNet |
| X | `OptunaSharp.Extensions` | M.E.L logging adapter, Data.Analysis adapter, OpenBLAS byte-exactness switches | various |

---

## 4. Validation strategy (built in Phase 0, extended every phase)

Copy of NumSharp's differential-fuzz pipeline, adapted:

- `test/oracle/gen_optuna_oracle.py` — pinned `optuna==<release>` + pinned clone SHA; generates JSONL
  corpora: `(seed, search_space, observed_trials) → expected (params | pruning decision | importance
  values | storage bytes)` per feature tier.
- `OptunaSharp.Tests.Oracle` — replays corpora with **no Python at test time**; bit-compare where the
  path is deterministic (everything on MT19937), tolerance-compare only where the native tier
  legitimately diverges (managed `eigh`/`cholesky` vs LAPACK — documented per case, upgradeable to
  byte-exact via the Tier X OpenBLAS switch).
- `OptunaSharp.Tests.Interop` (scaffolded) — live gates run by hand/nightly: Python writes study →
  C# continues it → Python reads back (journal file in Tier N; SQLite/gRPC in Tier X).
- `MisalignedRegistry` — the documented-divergence ledger, same rules as NumSharp: excused, never silent.

---

# TIER N — pure managed (BCL + NumSharp only)

## Phase 0 — Core framework

Prerequisite to everything; order within the phase is the dependency order.

| # | Work item | Optuna source | py SLOC | est. C# | Notes |
|---|---|---|--:|--:|---|
| 0.1 | Exceptions, enums, own logging seam | `exceptions.py`, `logging.py`, `study/_study_direction.py`, `trial/_state.py` | ~200 | 450 | `TrialPruned`, `DuplicatedStudyError`…; logging = self-contained port of Optuna's verbosity API (no M.E.L — that adapter is 5.4) |
| 0.2 | **Distributions** + JSON round-trip | `distributions.py` | 381 | 900 | 9 classes incl. deprecated aliases; `to_internal_repr`/`to_external_repr`; `distribution_to_json`/`from_json` (storage depends on it); `CategoricalChoice` Python-equality (§2.3) |
| 0.3 | `LazyRandomState` | `samplers/_lazy_random_state.py` | 24 | 60 | wraps NumSharp `np.random.RandomState` |
| 0.4 | **SearchSpaceTransform** | `_transform.py` | 198 | 450 | float64 bounds matrix, one-hot categorical encode/decode — the numeric boundary of the whole system |
| 0.5 | Trial family | `trial/*` | 853 | 1,900 | suggest_* API, `report`/`should_prune`, attrs, `FrozenTrial`, `create_trial` |
| 0.6 | **BaseStorage contract** + InMemoryStorage | `storages/_base.py`, `_in_memory.py` | 879 | 1,900 | the contract every backend implements; thread-safe in-memory impl; conformance suite born here |
| 0.7 | Sampler contract + RandomSampler + search-space intersection | `samplers/_base.py`, `_random.py`, `search_space/*` | 727 | 1,500 | `infer_relative_search_space`/`sample_relative`/`sample_independent`; `IntersectionSearchSpace` |
| 0.8 | **Study** + optimize loop + ask/tell | `study/study.py`, `_optimize.py`, `_tell.py`, `_dataframe.py` (plain table) | 1,256 | 2,700 | callbacks, `n_jobs` threading, enqueue, best_trial(s), directions |
| 0.9 | **Walking skeleton gate** | — | — | — | quadratic objective E2E: create_study → optimize(RandomSampler) → best_trial **bit-identical to Python** for shared seed; oracle harness v1 lands here |

**Phase 0 ≈ 4.5k py → ~9.9k C#. Estimate: 5–6 weeks.**

## Phase 1 — Native persistence & infra (pure file/memory)

| # | Work item | Optuna source | py SLOC | est. C# | Difficulty | Notes |
|---|---|---|--:|--:|---|---|
| 1.1 | **Journal storage**: file backend, `OpenLock`/`SymlinkLock` → `FileStream` locks, snapshot/replay | `storages/journal/*` minus `_redis.py` | ~650 | 1,400 | 🟢 | JSON-lines op log, byte-compatible with Python's format. **Interop gate (Tier N!):** Python appends → C# replays → C# appends → Python replays, same file |
| 1.2 | Cached storage layer | `storages/_cached_storage.py` | ~300 | 600 | 🟢 | read-through cache (used later by RDB too) |
| 1.3 | Heartbeat + failed-trial retry | `storages/_heartbeat.py`, `_callbacks.py` | ~250 | 550 | 🟢 | `HeartbeatThread`, `RetryFailedTrialCallback`, `fail_stale_trials` |
| 1.4 | Artifacts: protocol + FileSystem + backoff + upload/download | `artifacts/*` minus boto3/GCS | ~250 | 550 | 🟢 | cloud backends are 5.3 |
| 1.5 | Copy/delete/summaries API | `copy_study`, `get_all_study_summaries`, delete | ~200 | 450 | 🟢 | composes on storage contract |

**Phase 1 ≈ 1.65k py → ~3.55k C#. Estimate: 3–4 weeks.**
Exit gate: journal-file Python↔C# live interop green — **cross-language persistence with zero external deps on either side**.

## Phase 2 — Native features, easiest first

| # | Feature | Optuna source | py SLOC | est. C# | Difficulty | Depends on |
|---|---|---|--:|--:|---|---|
| 2.1 | Pruners batch 1: `Nop`, `Threshold`, `Patient`, `Median`, `Percentile` | `pruners/*` | ~300 | 700 | 🟢 trivial | 0.5, 0.8 |
| 2.2 | Pruners batch 2: `SuccessiveHalving` (ASHA), `Hyperband` | `pruners/_successive_halving.py`, `_hyperband.py` | ~250 | 650 | 🟢 | 2.1 |
| 2.3 | `WilcoxonPruner` | `pruners/_wilcoxon.py` | ~100 | 300 | 🟢 | self-contained signed-rank test (no scipy — verified vs scipy in oracle gen) |
| 2.4 | `PartialFixedSampler`, `GridSampler` | `samplers/_partial_fixed.py`, `_grid.py` | ~400 | 900 | 🟢 | 0.7 |
| 2.5 | `BruteForceSampler` | `samplers/_brute_force.py` | ~480 | 1,000 | 🟢+ | tree enumeration over dynamic spaces; the categorical-NaN quirk (`_is_nan`) |
| 2.6 | Terminator framework + non-GP evaluators (`StaticError`, `BestValueStagnation`, `MedianError`, `CrossValidationError`) | `terminator/*` minus regret/EMMR | ~300 | 700 | 🟢 | 0.8 |
| 2.7 | Importance framework + **PED-ANOVA** | `importance/__init__.py`, `_ped_anova/*` | ~350 | 800 | 🟢+ | pure-numpy Parzen; NumSharp throughout |
| 2.8 | **TPE stage 1 — the default sampler, univariate** | `_tpe/_erf.py`, `_truncnorm.py`, `parzen_estimator.py`, `probability_distributions.py`, `sampler.py` core | ~800 | 2,600 | 🟡 | 0.4; port `_erf`/`_truncnorm` exactly (they exist so scipy isn't needed); **bit-exact oracle vs Python TPE** is the flagship parity claim |
| 2.9 | TPE stage 2 — `multivariate`, `group`, `constant_liar`, weights | `_tpe/sampler.py` rest | ~300 | 900 | 🟡 | 2.8 |
| 2.10 | **MO machinery**: non-dominated sort, WFG hypervolume, HSSP, box decomposition, Pareto front | `study/_multi_objective.py`, `_hypervolume/*` | ~400 | 1,100 | 🟡 | NumSharp array code |
| 2.11 | TPE stage 3 — MOTPE + constraints | `_tpe/sampler.py` MO split | ~150 | 450 | 🟡 | 2.9, 2.10 |
| 2.12 | **QMC sampler** (Sobol + Halton, scrambling) | `samplers/_qmc.py` + scipy behavior | ~172 | 1,600 | 🟡 | Sobol direction numbers lifted from `rustuna_sampler/src/qmc/sobol/direction_numbers.rs`; scrambled sequences oracle-pinned vs scipy |
| 2.13 | **NSGA-II** (6 crossovers, polynomial mutation, elite selection, constraints) | `samplers/nsgaii/*`, `_ga/_base.py` | ~1,000 | 2,400 | 🟡 | 2.10; crossover object-array site → `CategoricalChoice[]` |
| 2.14 | NSGA-III (reference points, niching) | `samplers/_nsgaiii/*` | ~390 | 950 | 🟡+ | 2.13 |
| 2.15 | Managed numeric primitives for 2.16/3.2: symmetric `eigh` (cyclic Jacobi), unblocked `cholesky`, `solve_triangular` | — (NumSharp.Core has no managed eigen/cholesky) | — | 500 | 🟡 | small dims (CMA-ES: search-space rank; GP: ≤ trial count) → O(n³) managed is fine. Tolerance oracle vs LAPACK; **byte-exact upgrade switch lives in Tier X (5.5)** |
| 2.16 | **CMA-ES** — port the external `cmaes` package (CMA, sep-CMA, IPOP/BIPOP restarts, margin, lr-adapt) + `CmaEsSampler` | `samplers/_cmaes.py` + `CyberAgentAILab/cmaes` (~1.8k py) | ~2,300 | 4,500 | 🟠 | 2.15. Trajectory oracle is tolerance-based on the native `eigh` (byte-exact under 5.5). Optimizer state → own JSON (pickle non-goal, `[Misaligned]`) |

**Phase 2 ≈ 7.7k py → ~20.1k C#. Estimate: 14–17 weeks.**
Milestone after 2.11: **"Optuna default experience" complete** — TPE + MedianPruner bit-exact with
Python, persisted to a journal file Python can open. Zero external deps.

## Phase 3 — Native heavy tail

| # | Feature | Optuna source | py SLOC | est. C# | Difficulty | Notes |
|---|---|---|--:|--:|---|---|
| 3.1 | Visualization **data layer** | `visualization/_*.py` info builders | ~1,500 | 2,500 | 🟡 | pure numpy → NumSharp; rendering pkg is 5.2 |
| 3.2 | **GP stack, native path**: `gp.py` (Matérn-5/2 + MLL fit via hand-derived gradients), `acqf.py` (LogEI/qLogEI/LogPI), `optim_mixed`, `batched_lbfgsb` (already pure numpy), `prior`, `search_space` → `GPSampler` | `_gp/*`, `samplers/_gp/sampler.py` | 1,766 | 8,500 | 🔴 | **Decision gate first (§7), prototyped during Phase 2.** Uses 2.15 cholesky/solve_triangular. Port order: search_space → prior → kernel+MLL → L-BFGS-B → acqf → optim_mixed → sampler |
| 3.3 | GP terminators: `RegretBoundEvaluator` (default), `EMMREvaluator` | `terminator/improvement/*` | ~230 | 700 | 🔴 | rides 3.2 |

**Phase 3 ≈ 3.5k py → ~11.7k C#. Estimate: 10–13 weeks (GP is 8–10).**
Exit gate = **Tier N complete**: the full Optuna algorithm surface, pure managed, NumSharp-only.

---

# TIER X — external dependencies & integrations (last)

## Phase 4 — External storage & transport

| # | Work item | Optuna source | py SLOC | est. C# | Package | Notes |
|---|---|---|--:|--:|---|---|
| 4.1 | **RDB storage on SQLite, Optuna-compatible schema** | `storages/_rdb/*` | 2,339 | 3,000 | `.Storage.Sqlite` | hand SQL on Microsoft.Data.Sqlite mirroring the SQLAlchemy models + `alembic_version` pin so Python opens our `.db`. Reference: `rustuna_storage/sqlite3.rs`. **Interop gate:** Python↔C# same-file round-trip. Reuses 1.2 cache + 0.6 conformance suite |
| 4.2 | EF Core providers (PostgreSQL/MySQL) | (schema from 4.1) | — | 1,500 | `.Storage.EFCore` | conformance suite + testcontainers |
| 4.3 | Redis journal backend | `storages/journal/_redis.py` | ~200 | 450 | `.Storage.Redis` | thin: 1.1 already carries the journal semantics |
| 4.4 | gRPC storage proxy (client + server) | `storages/_grpc/*` | 1,719 | 2,400 | `.Storage.Grpc` | reuse Optuna's `.proto` verbatim; wire-compat gate: C# client ↔ Python `run_grpc_proxy_server` |
| 4.5 | Cloud artifacts (S3, GCS) | `artifacts/_boto3.py`, `_gcs.py` | ~100 | 300 | `.Artifacts.*` | protocol landed in 1.4 |

**Phase 4 ≈ 4.4k py → ~7.65k C#. Estimate: 5–6 weeks.**

## Phase 5 — External integrations & upgrades

| # | Work item | py SLOC | est. C# | Package | Notes |
|---|---|--:|--:|---|---|
| 5.1 | **fANOVA + MDI importance** via ML.NET FastForest | 930 | 4,000 | `.Importance.Forests` | tolerance oracle (forests not tree-identical to sklearn), divergence documented; exact managed-CART port stays on the backlog as a Tier N follow-up |
| 5.2 | Visualization rendering | — | 1,500 | `.Visualization.Plotly` | over the 3.1 data layer |
| 5.3 | CLI | 806 | 1,300 | `.Cli` | System.CommandLine + YamlDotNet |
| 5.4 | Adapters: M.E.L logging, `Microsoft.Data.Analysis` `trials_dataframe` | — | 400 | `.Extensions` | thin bridges over Core seams |
| 5.5 | **Byte-exactness upgrades via `NumSharp.Interop.OpenBLAS`**: CMA-ES `eigh` → LAPACK `syevd`, GP `cholesky` → `potrf` | — | 200 | `.Extensions` | flips the 2.15 primitives to the backend seam → CMA-ES/GP oracles upgrade from tolerance to byte-exact (NumSharp's linalg_parity model) |
| 5.6 | `OptunaSharp.Gp.Torch` (contingency) | — | (2,000) | `.Gp.Torch` | ONLY if the §7 native gate failed — or as an alternative backend for large-trial GP perf |
| 5.7 | Framework pruning-callback adapters (ML.NET, TorchSharp) | — | (backlog) | — | the `optuna.integration` analog; per-framework, post-v1 |

**Phase 5 ≈ 1.7k py → ~7.4k C#. Estimate: 6–8 weeks.**

---

## 7. GP autodiff decision gate (opens 3.2; prototyped during Phase 2)

The only genuinely hard porting problem. Preference order **reflects the native-first directive**:

1. **Hand-derived analytic gradients** (pure managed — keeps GP in Tier N): Matérn-5/2
   marginal-likelihood gradients are closed-form; Optuna's own `Matern52Kernel(torch.autograd.Function)`
   already writes the backward by hand — evidence the surface is small. Optimizer = ported
   `batched_lbfgsb` (already numpy). Risk: derivation errors → finite-difference checks in the oracle.
2. **TorchSharp** (`OptunaSharp.Gp.Torch`, Tier X): fallback if (1) stalls; isolated to the optional
   package, ~600 MB native dep.
3. Micro reverse-AD library: only if (1) proves error-prone AND (2) unacceptable.

Prototype task (1 week, during Phase 2): fit a 1-D GP with hand gradients + ported L-BFGS-B; compare
log-marginal-likelihood trajectory vs Python. Green → GP stays native; red → 5.6.

---

## 8. Cross-cutting workstreams

- **Oracle corpus growth** — every feature lands with its tier; nightly soak sweeps seeds.
- **Docs** — DocFX site (scaffolded); per-API pages mirror Optuna's doc structure.
- **Benchmarks** — `OptunaSharp.Benchmark` (scaffolded): suggest-throughput, storage ops/s,
  sampler wall-clock vs Python Optuna AND vs Rustuna.
- **NumSharp feedback loop** — gaps found get fixed upstream (done already: `logaddexp`, `nextafter`,
  `interp`; watch the `erf` family in 2.8). Managed `eigh`/`cholesky` (2.15) live in OptunaSharp
  first; graduating them into NumSharp.Core as managed linalg fallbacks is a candidate upstream
  contribution.

---

## 9. Timeline & totals

| Phase | Tier | Scope | est. C# SLOC | Duration |
|---|---|---|--:|---|
| 0 | N | Core framework | ~9.9k | 5–6 wk |
| 1 | N | Native persistence (journal/heartbeat/artifacts) | ~3.6k | 3–4 wk |
| 2 | N | Native features easy→hard (pruners → TPE → … → CMA-ES) | ~20.1k | 14–17 wk |
| 3 | N | Native heavy tail (viz data, GP) | ~11.7k | 10–13 wk |
| — | — | **Tier N complete — "pure managed full Optuna"** | **~45.3k** | **~7.5–9 mo** |
| 4 | X | External storage (SQLite/EFCore/Redis/gRPC/cloud) | ~7.7k | 5–6 wk |
| 5 | X | External integrations (forests, rendering, CLI, adapters, exactness) | ~7.4k | 6–8 wk |
| **Total** | | | **~60.4k** | **~10–12.5 mo** |

Tier X has no dependency on Tier N's *late* items (4.1 needs only Phase 0/1 contracts) — with a second
contributor, Phase 4 can run concurrently with Phases 2–3 and the total compresses toward ~9 months.

Checkpoints: **M1** (end P0) walking skeleton bit-exact · **M2** (end P1) **cross-language journal
persistence, zero external deps** · **M3** (after 2.11) default-experience parity — TPE + pruning,
the announceable release · **M4** (end P3) Tier N complete: full algorithm surface pure-managed ·
**M5** (end P4) Python-interoperable SQLite/gRPC · **M6** (end P5) near-full Optuna.

---

## 10. Risk register

| Risk | Impact | Mitigation |
|---|---|---|
| GP gradients wrong/slow | 3.2 slips | decision-gate prototype in Phase 2; TorchSharp fallback (5.6) |
| Managed `eigh`/`cholesky` drift vs LAPACK | CMA-ES/GP oracles only tolerance-exact in Tier N | documented per-case; 5.5 OpenBLAS switch restores byte-exactness; small-dim Jacobi is deterministic |
| fANOVA forest not tree-identical to sklearn | importance values differ | ML.NET + tolerance oracle + documented divergence; exact CART port as Tier N follow-up |
| Journal-file format drift across Optuna versions | Tier N interop breaks on their upgrades | format pinned to Optuna release; interop gate runs against pinned + latest |
| SQLAlchemy schema drift | Tier X interop breaks | schema pinned to alembic revision; same dual-version gate |
| Python-equality of `CategoricalChoice` edge cases | silent wrong index → wrong param | dedicated property tests (`True/1`, `1.0/1`, `nan`, mixed lists) from 0.2 |
| Pickle-serialized sampler state (CMA-ES attrs) | cross-language studies can't resume mid-CMA | own JSON state, `[Misaligned]`, migration notes |
| `frompyfunc`/object-array sites | none — audited: 1 site (NSGA-II crossover) + 1 intermediate | plain C# arrays (dtype audit, 2026-08-29) |
