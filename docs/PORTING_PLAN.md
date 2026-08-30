# OptunaSharp — Optuna → C# Conversion Plan

**Source of truth:** `optuna/optuna` @ `f6898da` (v5.0.0.dev, cloned at `K:/source/optuna`) — 24,474 core SLOC
(tokenizer-counted, comments/docstrings excluded). Secondary reference: `optuna/rustuna` @ `8acb2dd`
(`K:/source/rustuna`) — used ONLY where it accelerates (SQLite schema, Sobol tables, static-type designs);
Optuna's Python behavior always wins on conflict.

**Ordering directive:** everything core / infra / storage / persistence is ported FIRST (Phases 0–1),
then features strictly by difficulty, easiest first (Phases 2–3). Target: **almost all features** —
only Python-ecosystem integrations and interactive dashboards are out of scope.

---

## 1. Goals & non-goals

**Goals**
- API-compatible .NET port: an Optuna user reads Optuna's docs and writes the same call in C#.
- Behavior parity proven the NumSharp way: Python Optuna is the **oracle**; committed corpora are
  replayed bit-exact in CI (no Python at test time). NumSharp's byte-identical MT19937 `RandomState`
  makes sampler sequences reproducible bit-for-bit.
- Storage-level interop: a study written by OptunaSharp opens in Python Optuna and vice versa
  (`.db` SQLite files, journal logs, gRPC proxy protocol).
- NumSharp is the numeric backbone (`np.random`, array math in samplers); plain C# where arrays add
  nothing (bookkeeping, storage, orchestration).

**Non-goals (explicit cuts)**
- `optuna.integration.*` (LightGBM/PyTorch/Keras callbacks) — framework adapters, not ports; revisit
  per-framework (ML.NET/TorchSharp) after v1.
- Interactive visualization rendering — we port the **data layer** (the pure-numpy `_get_*_info`
  functions); rendering is an optional `Plotly.NET`/ScottPlot package later.
- OptunaHub, Dashboard, `optuna.testing` (we build our own test infra).

---

## 2. Ground rules

1. **Match Optuna exactly** — run the Python first, observe, replicate; edge cases (NaN, single-value
   distributions, dynamic search spaces, `None` choices) matter. Verbatim error messages where Optuna
   raises typed errors.
2. **RNG discipline:** every sampler consumes `np.random.RandomState` through a `LazyRandomState`
   port. Never `System.Random`. This is what makes differential testing bit-exact.
3. **Categorical value semantics:** `CategoricalChoiceType = None|bool|int|float|str` maps to a
   `CategoricalChoice` value type implementing **Python equality** (`True == 1`, `1.0 == 1`,
   `nan != nan` but `is`-identity matches — see `distributions.py:509` note). Pinned by tests; this is
   the one place C# object equality silently diverges.
4. **Package optionality mirrors Optuna's:** Core stays pure managed (NumSharp only). Anything needing
   a foreign runtime (ML.NET forests, TorchSharp, Redis, cloud SDKs) is a separate NuGet package —
   the NumSharp.Interop.OpenBLAS pattern.
5. **Threading:** trial-level parallelism (`n_jobs`, distributed workers) is supported and tested;
   numeric kernels stay single-threaded (NumSharp house rule).
6. **Every phase ends green:** its oracle tier committed + replayed in CI, its unit tests passing,
   and — for storage phases — a live Python↔C# interop script (the `verify_npy_interop.py` model).

---

## 3. Package map

| Package | Contents | Foreign deps |
|---|---|---|
| `OptunaSharp` (Core) | distributions, trial, study, samplers (all pure-managed), pruners, transform, MO machinery, PED-ANOVA, terminator (non-GP), InMemory + Journal-file storage, artifacts (filesystem) | NumSharp only |
| `OptunaSharp.Storage.Sqlite` | RDB storage, Optuna-compatible schema + Alembic version table | Microsoft.Data.Sqlite |
| `OptunaSharp.Storage.EFCore` | same schema on PostgreSQL/MySQL | EF Core + providers |
| `OptunaSharp.Storage.Redis` | journal Redis backend | StackExchange.Redis |
| `OptunaSharp.Storage.Grpc` | storage proxy client + server | Grpc.Net, Google.Protobuf |
| `OptunaSharp.Importance.Forests` | fANOVA, MDI | ML.NET (or managed CART) |
| `OptunaSharp.Gp` | GP sampler + acqf + GP terminators | TorchSharp **or** managed AD (decision gate §7) |
| `OptunaSharp.Artifacts.S3` / `.Gcs` | cloud artifact stores | AWSSDK / Google.Cloud.Storage |
| `OptunaSharp.Visualization` | plot-info data layer (+ optional Plotly.NET rendering) | none / Plotly.NET |
| `OptunaSharp.Cli` | `optuna` CLI | System.CommandLine |

---

## 4. Validation strategy (built in Phase 0, extended every phase)

Copy of NumSharp's differential-fuzz pipeline, adapted:

- `test/oracle/gen_optuna_oracle.py` — pinned `optuna==<release>` + pinned clone SHA; generates JSONL
  corpora: `(seed, search_space, observed_trials) → expected (params | pruning decision | importance
  values | storage bytes)` per feature tier.
- `OptunaSharp.Tests.Oracle` — replays corpora with **no Python at test time**; bit-compare where the
  path is deterministic (everything on MT19937), tolerance-compare only where Optuna itself is
  platform-dependent (documented per case).
- `OptunaSharp.Tests.Interop` (already scaffolded) — live gates run by hand/nightly: Python writes
  study → C# continues it → Python reads back (SQLite, journal, gRPC).
- `MisalignedRegistry` — the documented-divergence ledger, same rules as NumSharp: excused, never
  silent.

---

## Phase 0 — Core framework (port first)

Everything below is prerequisite to any feature. Order within the phase is the dependency order.

| # | Work item | Optuna source | py SLOC | est. C# | Notes |
|---|---|---|--:|--:|---|
| 0.1 | Exceptions, enums, logging seam | `exceptions.py`, `logging.py`, `study/_study_direction.py`, `trial/_state.py` | ~200 | 400 | `TrialPruned`, `DuplicatedStudyError`…; logging → `Microsoft.Extensions.Logging` abstraction with Optuna-style verbosity API |
| 0.2 | **Distributions** + JSON round-trip | `distributions.py` | 381 | 900 | 9 classes incl. deprecated aliases; `to_internal_repr`/`to_external_repr`; `distribution_to_json`/`from_json` (storage depends on it); `CategoricalChoice` Python-equality (§2.3) |
| 0.3 | `LazyRandomState` | `samplers/_lazy_random_state.py` | 24 | 60 | wraps NumSharp `np.random.RandomState` |
| 0.4 | **SearchSpaceTransform** | `_transform.py` | 198 | 450 | float64 bounds matrix, one-hot categorical encode/decode — the numeric boundary of the whole system |
| 0.5 | Trial family | `trial/*` (`_trial.py`, `_frozen.py`, `_fixed.py`, `create_trial`) | 853 | 1,900 | suggest_* API, `report`/`should_prune`, system/user attrs, `FrozenTrial` |
| 0.6 | **BaseStorage contract** + InMemoryStorage | `storages/_base.py`, `_in_memory.py`, `_heartbeat.py` interfaces | 879 | 1,900 | the contract every backend implements; thread-safe in-memory impl |
| 0.7 | Sampler contract + RandomSampler + search-space intersection | `samplers/_base.py`, `_random.py`, `search_space/*` | 727 | 1,500 | `infer_relative_search_space`/`sample_relative`/`sample_independent`; `IntersectionSearchSpace` |
| 0.8 | **Study** + optimize loop + ask/tell | `study/study.py`, `_optimize.py`, `_tell.py`, `_dataframe.py` (as plain table) | 1,256 | 2,700 | callbacks, `n_jobs` threading, enqueue, best_trial(s), directions; `trials_dataframe` → typed row table (Data.Analysis adapter optional) |
| 0.9 | **Walking skeleton gate** | — | — | — | quadratic objective E2E: create_study → optimize(RandomSampler) → best_trial **bit-identical to Python** for shared seed; oracle harness v1 lands here |

**Phase 0 total ≈ 4.5k py → ~9.8k C#. Estimate: 5–6 weeks.**
Exit gate: skeleton E2E bit-exact; storage-contract conformance suite (reused by every later backend).

---

## Phase 1 — Persistence & infra completion (still "first" per directive)

| # | Work item | Optuna source | py SLOC | est. C# | Difficulty | Notes |
|---|---|---|--:|--:|---|---|
| 1.1 | **Journal storage** (file backend + locks + snapshot/replay) | `storages/journal/*` (minus Redis) | ~650 | 1,400 | 🟢 | append-only op log; `JournalFileOpenLock`/`SymlinkLock` → `FileStream` locks; replay determinism gate |
| 1.2 | **RDB storage on SQLite, Optuna-compatible schema** | `storages/_rdb/*` (models, storage, alembic version pin) | 2,339 | 3,000 | 🟡 | Microsoft.Data.Sqlite, hand-written SQL mirroring the SQLAlchemy models + `alembic_version` row so Python Optuna opens our files. Reference: `rustuna_storage/sqlite3.rs` (already schema-compatible). **Interop gate:** Python↔C# same-file round-trip |
| 1.3 | Cached storage layer | `storages/_cached_storage.py` | ~300 | 600 | 🟢 | read-through cache used by RDB |
| 1.4 | Heartbeat + failed-trial retry | `storages/_heartbeat.py`, `_callbacks.py` | ~250 | 550 | 🟢 | `HeartbeatThread`, `RetryFailedTrialCallback`, `fail_stale_trials` |
| 1.5 | Redis journal backend | `storages/journal/_redis.py` | ~200 | 450 | 🟡 | `OptunaSharp.Storage.Redis`; StackExchange.Redis; interop gate vs Python redis journal |
| 1.6 | gRPC storage proxy (client + server) | `storages/_grpc/*` | 1,719 | 2,400 | 🟡 | reuse Optuna's `.proto` verbatim → C# `StorageServiceClient`/server; wire-compat gate: C# client ↔ Python `run_grpc_proxy_server` |
| 1.7 | EF Core providers (PostgreSQL/MySQL) | (schema from 1.2) | — | 1,500 | 🟡 | `OptunaSharp.Storage.EFCore`; conformance suite + testcontainers |
| 1.8 | Artifacts: protocol + FileSystem + backoff + upload/download | `artifacts/*` (minus boto3/GCS) | ~250 | 550 | 🟢 | S3/GCS stubs deferred to 3.6 |
| 1.9 | Copy/delete/summaries API | `study/_tell.py` remnants, `copy_study`, `get_all_study_summaries` | ~200 | 450 | 🟢 | composes on storage contract |

**Phase 1 total ≈ 5.9k py → ~10.9k C#. Estimate: 6–7 weeks.**
Exit gate: all storage backends pass ONE shared conformance suite + live Python interop scripts green.

---

## Phase 2 — Features, easiest first

Strict easy→hard order, dependencies noted. Each item ships with its oracle tier.

| # | Feature | Optuna source | py SLOC | est. C# | Difficulty | Depends on |
|---|---|---|--:|--:|---|---|
| 2.1 | Pruners batch 1: `Nop`, `Threshold`, `Patient`, `Median`, `Percentile` | `pruners/*` | ~300 | 700 | 🟢 trivial | 0.5, 0.8 |
| 2.2 | Pruners batch 2: `SuccessiveHalving` (ASHA), `Hyperband` | `pruners/_successive_halving.py`, `_hyperband.py` | ~250 | 650 | 🟢 | 2.1 |
| 2.3 | `WilcoxonPruner` | `pruners/_wilcoxon.py` | ~100 | 300 | 🟢 | self-contained signed-rank test (no scipy — verify vs scipy in oracle gen) |
| 2.4 | `PartialFixedSampler`, `GridSampler` | `samplers/_partial_fixed.py`, `_grid.py` | ~400 | 900 | 🟢 | 0.7 |
| 2.5 | `BruteForceSampler` | `samplers/_brute_force.py` | ~480 | 1,000 | 🟢+ | tree enumeration over dynamic spaces; the categorical-NaN quirk (`_is_nan`) |
| 2.6 | Terminator framework + non-GP evaluators (`StaticError`, `BestValueStagnation`, `MedianError`, `CrossValidationError`) | `terminator/*` minus regret/EMMR | ~300 | 700 | 🟢 | 0.8 |
| 2.7 | Importance framework + **PED-ANOVA** | `importance/__init__.py`, `_ped_anova/*` | ~350 | 800 | 🟢+ | pure-numpy Parzen; NumSharp throughout |
| 2.8 | **TPE stage 1 — the default sampler, univariate** | `_tpe/_erf.py`, `_truncnorm.py`, `parzen_estimator.py`, `probability_distributions.py`, `sampler.py` (core) | ~800 | 2,600 | 🟡 | 0.4; port `_erf`/`_truncnorm` exactly (they exist so scipy isn't needed); **bit-exact oracle vs Python TPE** is the flagship parity claim |
| 2.9 | TPE stage 2 — `multivariate`, `group`, `constant_liar`, weights | `_tpe/sampler.py` (rest) | ~300 | 900 | 🟡 | 2.8 |
| 2.10 | **MO machinery**: non-dominated sort, WFG hypervolume, HSSP, box decomposition, Pareto front | `study/_multi_objective.py`, `_hypervolume/*` | ~400 | 1,100 | 🟡 | NumSharp array code |
| 2.11 | TPE stage 3 — MOTPE + constraints | `_tpe/sampler.py` (MO split) | ~150 | 450 | 🟡 | 2.9, 2.10 |
| 2.12 | **QMC sampler** (Sobol + Halton, scrambling) | `samplers/_qmc.py` + scipy behavior | ~172 | 1,600 | 🟡 | Sobol direction numbers lifted from `rustuna_sampler/src/qmc/sobol/direction_numbers.rs`; scrambled sequences oracle-pinned vs scipy |
| 2.13 | **NSGA-II** (6 crossovers, polynomial mutation, elite selection, constraints) | `samplers/nsgaii/*`, `_ga/_base.py` | ~1,000 | 2,400 | 🟡 | 2.10; crossover object-array site → `CategoricalChoice[]` |
| 2.14 | NSGA-III (reference points, niching) | `samplers/_nsgaiii/*` | ~390 | 950 | 🟡+ | 2.13 |
| 2.15 | **CMA-ES** — port the external `cmaes` package (CMA, sep-CMA, IPOP/BIPOP restarts, margin, lr-adapt) + `CmaEsSampler` | `samplers/_cmaes.py` + `CyberAgentAILab/cmaes` (~1.8k py) | ~2,300 | 4,500 | 🟠 | NumSharp linalg (`eigh` via OpenBLAS interop or power iteration); sampler's attr-serialized optimizer state must round-trip Python's pickle format → **deviation:** own JSON state (documented, `[Misaligned]`) |

**Phase 2 total ≈ 7.7k py → ~19.6k C#. Estimate: 13–16 weeks.**
Milestone after 2.11: **"Optuna default experience" complete** — `create_study(); optimize()` runs TPE
+ MedianPruner bit-exact with Python, persisted to SQLite Python can open.

---

## Phase 3 — Heavy tail (hardest last)

| # | Feature | Optuna source | py SLOC | est. C# | Difficulty | Notes |
|---|---|---|--:|--:|---|---|
| 3.1 | Visualization **data layer** | `visualization/_*.py` info builders | ~1,500 | 2,500 | 🟡 | pure numpy → NumSharp; rendering pkg optional later |
| 3.2 | CLI | `cli.py` | 806 | 1,300 | 🟡 | System.CommandLine; YAML via YamlDotNet |
| 3.3 | **fANOVA + MDI importance** | `importance/_fanova/*`, `_mean_decrease_impurity.py` | 930 | 4,000 | 🟠 | decision gate: ML.NET FastForest behind `OptunaSharp.Importance.Forests` (fast, not tree-identical → tolerance oracle) vs managed CART port (slow, exact). Start with ML.NET + document divergence; revisit exact port if demanded |
| 3.4 | **GP stack**: `gp.py` (Matérn-5/2 + MLL fit), `acqf.py` (LogEI/qLogEI/LogPI), `optim_mixed`, `batched_lbfgsb`, `prior`, `search_space` → `GPSampler` | `_gp/*`, `samplers/_gp/sampler.py` | 1,766 | 8,500 | 🔴 | **Decision gate first (§7).** Port order: search_space → prior → kernel+MLL → L-BFGS-B → acqf → optim_mixed → sampler. `batched_lbfgsb.py` is already pure numpy (head start) |
| 3.5 | GP terminators: `RegretBoundEvaluator` (default), `EMMREvaluator` | `terminator/improvement/*` | ~230 | 700 | 🔴 | rides 3.4 |
| 3.6 | Cloud artifacts (S3, GCS) | `artifacts/_boto3.py`, `_gcs.py` | ~100 | 300 | 🟢 | AWSSDK / Google.Cloud.Storage packages |

**Phase 3 total ≈ 5.3k py → ~17.3k C#. Estimate: 12–15 weeks (of which GP is 8–10).**

---

## 7. GP autodiff decision gate (opens Phase 3.4; prototype during Phase 2)

The only genuinely hard porting problem. Options, in preference order:

1. **Hand-derived analytic gradients** (pure managed, NumSharp): Matérn-5/2 marginal-likelihood
   gradients are closed-form; Optuna's own `Matern52Kernel(torch.autograd.Function)` already writes
   the backward by hand — evidence the surface is small enough. Port `batched_lbfgsb` (already numpy)
   as the optimizer. Risk: derivation errors → mitigate with finite-difference checks in the oracle.
2. **TorchSharp** behind `OptunaSharp.Gp`: fastest to correct, ~600 MB native dependency, isolated to
   the optional package (NumSharp.Interop.OpenBLAS precedent). Fallback if (1) stalls.
3. Micro reverse-AD library: only if (1) proves error-prone AND (2) unacceptable.

Prototype task (1 week, during Phase 2): fit a 1-D GP with hand gradients + ported L-BFGS-B; compare
log-marginal-likelihood trajectory vs Python. Green → option 1; red → option 2.

---

## 8. Cross-cutting workstreams (run alongside all phases)

- **Oracle corpus growth** — every feature lands with its tier; nightly soak sweeps seeds (fuzz-soak model).
- **Docs** — DocFX site (scaffolded); per-API pages mirror Optuna's doc structure.
- **Benchmarks** — `OptunaSharp.Benchmark` (scaffolded): suggest-throughput, storage ops/s,
  sampler wall-clock vs Python Optuna AND vs Rustuna (their published benchmark shapes).
- **NumSharp feedback loop** — gaps discovered get fixed upstream in NumSharp
  (already done: `logaddexp`, `nextafter`, `interp`; watch for `erf`-family needs in 2.8).

---

## 9. Timeline & totals

| Phase | Scope | est. C# SLOC | Duration (1 experienced dev) |
|---|---|--:|---|
| 0 | Core framework | ~9.8k | 5–6 wk |
| 1 | Persistence & infra | ~10.9k | 6–7 wk |
| 2 | Features easy→hard | ~19.6k | 13–16 wk |
| 3 | Heavy tail | ~17.3k | 12–15 wk |
| **Total** | **~23.4k py ported** | **~57.6k** | **~9–11 months** (≈6.5 mo to end of Phase 2) |

Checkpoints: **M1** (end P0) walking skeleton bit-exact · **M2** (end P1) Python-interoperable
persistence · **M3** (after 2.11) default-experience parity (TPE+pruning, the announceable release) ·
**M4** (end P2) all pure-managed samplers · **M5** (end P3) near-full Optuna.

---

## 10. Risk register

| Risk | Impact | Mitigation |
|---|---|---|
| GP gradients wrong/slow | Phase 3.4 slips | decision-gate prototype in Phase 2; TorchSharp fallback |
| fANOVA forest not tree-identical to sklearn | importance values differ | ML.NET + tolerance oracle + documented divergence; exact CART port as follow-up |
| CMA-ES `eigh` parity | state trajectories drift | OpenBLAS-interop `eigh` (LAPACK `syevd`, byte-exact per NumSharp gates); oracle over full ask/tell trajectories |
| SQLAlchemy schema drift across Optuna versions | interop breaks on their upgrades | pin schema to Alembic revision; interop gate runs against pinned + latest Python Optuna |
| Python-equality of `CategoricalChoice` edge cases | silent wrong index → wrong param | dedicated property tests (`True/1`, `1.0/1`, `nan`, mixed lists) from day 0 (0.2) |
| Pickle-serialized sampler state (CMA-ES attrs) | cross-language studies can't resume mid-CMA | own JSON state format, `[Misaligned]`, documented in migration notes |
| `frompyfunc`/object-array sites | none — audited: 1 site (NSGA-II crossover) + 1 intermediate | plain C# arrays (see dtype audit, 2026-08-29) |
