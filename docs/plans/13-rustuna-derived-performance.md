# Plan 13 — Rustuna-derived performance: compatible speedups first, opt-in paths where results would change

> **Status: DESIGN (2026-11-08).** Nothing here is built yet. A scratch prototype of the CMA-ES part exists (session
> `5417a82e`, a patched copy of `d0558f5`) and is measured below; Phase 1 ports it. Background, benchmarks and the
> full Rustuna read-through: catalog note `optunasharp/notes/rustuna-comparison-2026-11-08.md`.

---

## 1. The request and the decisions (owner, 2026-11-08)

The request: "Write a plan for rustuna's optimization's integration, with evaluation for dual-impl recommendation
(user chooses if we use Optuna's path or Rustuna)."

[Rustuna](https://github.com/optuna/rustuna) is the Optuna team's Rust port (0.2.1 on PyPI). On its announcement's
workloads it is 3-60x faster than OptunaSharp, which is itself 3-5x faster than Optuna 5.0 (§2.1). Most of the gap is
implementation, not anything Optuna compatibility requires: a scratch prototype took CMA-ES at 100,000 trials from
812 s to 26 s with byte-identical results (§2.3).

| Question | Answer |
|---|---|
| Default when a Rustuna-style path would change results or storage contents | "We would prefer rustuna unless it affects results." So a Rustuna-style change that keeps every result becomes the only implementation; one that changes results needs a measurement of how different (§2.4) and a decision per item |
| CMA-ES keeping the optimizer it just told (Rustuna's live object) | Opt-in; Optuna's path stays the default (§5.1) |
| TPE fast numerics in candidate scoring | Decide after the exact fused kernel exists and can be measured (§5.2) |
| Separate trial queue (Rustuna's `TrialQueue`) | Deferred. The owner asked whether Rustuna has one: it does (§5.3) |
| Trial discarding (Rustuna's `apply_discard`) | Deferred: "what's best + don't break Optuna compatibility". §5.4 proposes compacting finished trials in memory instead, which keeps every API result |
| Where a user flips a switch | Deferred. The CMA-ES choice above was made on a page that showed a constructor parameter; §5.5 proposes that shape for every switch |

## 2. Evidence

### 2.1 Where we stand

i9-13900K, Windows 11, Release, every process opted out of power throttling; OptunaSharp `d0558f5`, Rustuna 0.2.1,
Optuna 5.0.0; 40 float parameters in [-10, 10], seed 0, default samplers.

| Workload | OptunaSharp | Rustuna 0.2.1 | Optuna 5.0 |
|---|---|---|---|
| TPE, 1,000 trials | 9.94 s | 1.36 s | 35.2 s |
| MOTPE (2 objectives), 1,000 trials | 36.6 s | 6.30 s | 163 s |
| TPE, 1,000 trials, 4 threads | 9.29 s shared / 2.77 s PerWorker | 0.40 s | 32.5 s |
| CMA-ES, 10,000 trials | 18.3 s | 1.59 s | 56.9 s |
| CMA-ES, 100,000 trials | 812 s | 13.6 s | not run |
| NSGA-II, 10,000 trials | 3.51 s | 0.82 s | 44.2 s |
| NSGA-II, 100,000 trials | 62.9 s | 19.4 s | not run |

Rustuna is API-compatible, not result-compatible: its seeded trials differ from Optuna's from the first trial (its
own RNG, `rand::StdRng`), while OptunaSharp's are identical to Optuna's.

### 2.2 Where the time goes (dotnet-trace, `dotnet-sampled-thread-time`)

| Workload | Profile |
|---|---|
| CMA-ES, 10k | 88 % per ask: eigendecomposition 40 % (the optimizer is rebuilt from its pickle every ask and loses its cached decomposition), restore 21 %, generation scan 17 % (O(n) per ask) |
| NSGA-II, 10k | ~66 % in `Trial.SuggestFloat`: `InMemoryStorage.SetTrialParam` clones the trial's dictionaries on every suggest (Optuna's copy-on-write), O(d²) per trial; GC pauses 22 % |
| TPE, 1k | `MixtureOfProductDistribution.LogPdf` 58 %, with 4.4 GB of candidate × kernel × dimension temporaries per 1,000 trials |
| MOTPE, 300 trials (3.87 s) | the trial split 0.93 s (24 %: `FastNonDominationRank` 0.53 s, `UniqueRows` 0.39 s), recomputed for each of the 40 parameters; `SetTrialParam` 0.89 s (23 %); scoring 0.95 s |

### 2.3 The CMA-ES prototype (scratch copy of `d0558f5`)

`Samplers/CmaEsSampler.cs` changed in three ways, and one line of `SearchSpace/IntersectionSearchSpace.cs`:

1. **Optimizer cache.** The optimizer deserialized from the latest stored state is kept, keyed by the trial that holds
   that state (a finished trial cannot change). Later asks of the generation reuse it, eigendecomposition included.
   After every tell it is dropped, so the next ask deserializes the state just stored, as Optuna does.
2. **Folded history.** Instead of copying and filtering every trial on every ask, the sampler folds the leading run of
   finished trials once, in trial order, into a completed list, a by-generation index and the latest state holder. Each
   ask classifies only the trials after the first unfinished one.
3. **Zero-copy reads.** The sampler and `IntersectionSearchSpace.Calculate` read `Study.GetTrialsView()`, the window the
   genetic samplers already use, instead of a full copy of the history.

| CMA-ES, 40-D, seed 0 | `d0558f5` | prototype | prototype, `OPENBLAS_NUM_THREADS=1` | Rustuna 0.2.1 |
|---|---|---|---|---|
| 10,000 trials | 17.7-18.9 s | 3.3-5.2 s | 2.56-2.59 s | 1.4-1.65 s |
| 100,000 trials | 812 s, 3.4 GB peak | 33.0-34.2 s, 31 GB allocated, 15 gen-2 GCs | 26.0 s | 13.4-13.9 s |

**Byte-identical**, three ways:
- Five seeded studies were hashed over every trial's number, state, values, params and system attributes (the stored
  `cma:optimizer:N` pickle chunks included): 40-D CMA 3,000 trials, SepCMA, with-margin, lr-adapt, and 8-D where
  every 7th trial fails. The hashes equal the unmodified build's.
- 86/86 CMA-ES, pickle and search-space unit tests pass, as on the unmodified build.
- 29/29 live-Python interop tests pass, including C# and Python workers taking turns on one study, with the engine
  required.

**The trap.** Reusing the *told* object instead of reloading the stored state moved seeded results. Optuna's pickle
stores only the upper triangle of the covariance matrix and mirrors it on load; the told matrix keeps the
floating-point asymmetry of the update. SepCMA, which stores its diagonal exactly, did not move.

**OpenBLAS.** At `nJobs: 1`, `TrialExecutorSnapshot.BlasThreadsFor` leaves OpenBLAS at its default thread count, and
the trace shows about three cores of unmanaged time beside the one optimizing thread. With `OPENBLAS_NUM_THREADS=1`
the unmodified 3,000-trial study ran in 5.6 s instead of 8.3-9.0 s, with the same hashes.

### 2.4 How different the result-changing paths are (measured 2026-11-08)

**CMA-ES keeping the told optimizer**, against the compatible cache, 10 seeds × 3,000 trials:
- 40-D sphere: the first trial that differs is trial 32 (generation 3) in every seed. The first difference can already
  be large (a parameter can change sign): a last-bit change in the covariance can flip an eigenvector's sign, so the
  same random draw maps to a different point of the same distribution. Best value, told / compatible: median 0.79,
  range 0.51-1.48. Same quality, a different run.
- 10-D Rastrigin: the first difference is at trial 22, 4e-14 to 7e-12 relative, and the best value is identical in all
  10 seeds.
- Storage is unchanged: the pickle is still written every generation, so Python workers and resuming still work;
  they just don't reproduce the C# run exactly.
- Speed with single-threaded OpenBLAS: 10k 2.57 → 2.18 s, 100k 26.0 → 21.9 s (about 15 %). With the default thread
  count the gap is larger (10k 4.7-5.2 → 2.6 s, 100k 33.0 → 25.3 s), because each skipped eigendecomposition is a
  multi-threaded one.

**TPE fast numerics**, simulated: noise of a chosen size was added to every acquisition value, drawn from its own
counter so the sampler's random stream was untouched.
- TPE, 40-D multivariate, 1,000 trials, 3 seeds (990 decisions each): noise of 1e-12, 1e-9 and even 1e-6 changed no
  decision. The smallest gap between the best and second-best candidate was 3.4e-4 (median 1.2; acquisition values
  up to 89).
- MOTPE, 40 parameters, 300 trials, 2 seeds (11,600 decisions each): identical with 1e-12 and 1e-9; 1e-6 changed
  decisions (first difference at trial 99 and trial 244). Smallest gaps 1.6e-7 and 3.8e-7.
- A reordered sum here errs by about 1e-13, so these runs had a margin of 5-9 orders of magnitude. That is not a
  guarantee: one near-tie in a long run flips one decision, and from there the run differs from Python. Noise in
  candidate *sampling* (not tested) would change stored parameters in the last bit.

**Trial queue** (no numbers): in one process it is identical, since Rustuna also creates the WAITING trial at enqueue
time and first-in-first-out equals Optuna's lowest-number-first pop. Across processes, an in-memory queue is private;
a SQLite queue is shared and pops atomically.

**Trial discarding** (no numbers): sampling is unchanged when only trials no sampler needs are dropped (CMA-ES reads
only the latest state holder and the current generation). `study.Trials`, the trials dataframe and the best trial (if
the best was dropped) change.

## 3. The rule: one implementation or two

The owner's rule, applied to each change:

1. **Does the Rustuna-style path change anything observable?** Observable means any of:
   - seeded parameters or values, compared with Python Optuna;
   - bytes written to storage;
   - what public APIs return (`Study.Trials`, the best trial, attributes, counts);
   - behavior across processes or with Python workers sharing the storage;
   - which inputs raise errors, and when.
2. **No** → one implementation. The faster path replaces the Optuna-structured code; there is nothing for a user to
   choose between. The gate is §6's "no observable change".
3. **Yes** → measure how different (trajectory divergence, quality over seeds, cross-process semantics), then the owner
   decides per item. A path that changes results defaults to Optuna's behavior so far (§1); the other path is a switch
   on the component it changes (§5.5).
4. **Build a second path only if it pays.** It needs a measured gain on a §7 workload and no compatible change that
   gets most of it. Otherwise it is not built: two paths mean two code paths to test and document forever.

## 4. Inventory and verdicts

"Same results" items are a single implementation (rule 2). Rustuna references are to `K:/source/rustuna` at `8acb2dd`.

| ID | Change | Rustuna does | OptunaSharp today | Evidence | Verdict |
|---|---|---|---|---|---|
| S1 | CMA-ES optimizer cache, dropped after each tell | keeps the live `cmaes.CMA` object (`rustuna_pyo3/src/sampler/cmaes.rs`) | restores from the pickle every ask | §2.3 (measured) | same results: build (Phase 1) |
| S2 | Folded history for CMA-ES | keeps the generation's trial ids; `StudyCache` folds with a cursor | full copy + scan every ask | §2.3 (measured) | same results: build (Phase 1) |
| S3 | Zero-copy history reads in the search spaces and the remaining samplers | samplers read `&PersistedTrial` under the storage lock | `IntersectionSearchSpace`, `GroupDecomposedSearchSpace`, QMC and CMA-ES copy the list per call | §2.3 for the intersection space | same results: build (Phases 1, 3) |
| S4 | In-place writes to a running trial, snapshots built only when read | `set_trial_param` inserts in place; `get_trials` clones for callers | `InMemoryStorage.SetTrialParam` clones the trial's dictionaries per suggest | §2.2: 66 % NSGA-II, 23 % MOTPE, 12 % CMA-ES | same results: build (Phase 2) |
| S5 | MOTPE split cache | `split_cache` per (completed trial numbers, gamma) | split recomputed per parameter; only the HSSP is cached (as in Optuna) | §2.2: 24 % of MOTPE | same results: build (Phase 3) |
| S6 | TPE exact fused kernel | accumulates each candidate into one n-kernel vector with precomputed kernel constants | builds candidate × kernel × dimension matrices | §2.2: 58 % of TPE, 4.4 GB per 1k trials | same results if NumPy's operation order is kept: build (Phase 3) |
| S7 | Interning of parameter names and distributions | interned keys; categorical labels stored once per study | a new key string per suggest when computed; distribution objects per trial | memory: CMA-ES history 2.5 GB at 100k | same results: build (Phase 2) |
| S8 | Bulk attribute writes (`SetUserAttrs`) | `set_trial_attrs` / `set_study_attrs` in bulk | one storage call per attribute | not measured | added API: build (Phase 4) |
| S9 | Lazy attribute decoding in database storages | attributes are strings (no JSON) | JSON parsed per attribute at load | Rustuna's motive: 100,000 `json.loads` to load 10k trials × 10 attrs | same values; errors surface on access: evaluate (Phase 4) |
| S10 | Incremental reads in `CachedStorage` | `CachedStorage` wrapper returns references after a sync | `GetTrialsPrefixView` falls back to a full copy | not measured | same results: build (Phase 4) |
| S11 | OpenBLAS thread policy at `nJobs: 1` | n/a | default thread count | §2.3: 8.3-9.0 → 5.6 s, same hashes | same results; GP's large matrices need measuring first (Phase 5) |
| S12 | Compact storage of finished trials | `PersistedTrial` keeps internal values only | per-trial boxed dictionaries | 2.5 GB at 100k CMA-ES | same results: the compatible answer to discarding (Phase 6) |
| D1 | CMA-ES reuses the told optimizer | yes | no | §2.4: results differ from generation 3; ~15 % faster | **opt-in, Optuna default** (owner) (Phase 7) |
| D2 | TPE fast numerics in scoring | its own numerics | NumPy's | §2.4: no decision changed in measured runs; gain unknown | **decide after S6** (owner) |
| D3 | Separate trial queue | `TrialQueue`: in-memory, SQLite3, directory; Redis planned | WAITING trials in storage | §2.4 | **deferred** (owner) |
| D4 | Trial discarding | `apply_discard` + `discard_trials` | none | §2.4 | **deferred** (owner); S12 first |
| — | Process-local CMA-ES state (no pickle) | yes | no | after S1, the pickle is ~2 % of the 10k profile | rejected: breaks resuming and mixed fleets for ~2 % |
| — | ChaCha RNG | `rand::StdRng` | MT19937 | the RNG never shows in a profile | rejected: loses parity, gains nothing |
| — | String-only attributes | yes | JSON values | | rejected: an API break; S9 gets the gain |
| — | Seeding CMA-ES once instead of per ask | yes | per ask (Optuna) | negligible | rejected |
| — | Dropping pruners, callbacks, timeout, `n_jobs`, per-trial logging | yes | no | the log line is already skipped unless INFO is on | rejected |
| — | Parallel sampling under one shared sampler | fine-grained locks; the GIL released while sampling | `SamplerConcurrency.Shared` serializes; `PerWorker` samples in parallel | 4-thread TPE: 0.40 s vs 2.77 s PerWorker | no new work: `PerWorker` is the existing switch, and plan 12's no-duplicate guarantee rests on the shared lock |

Already done before this plan: the genetic samplers fold their generations incrementally
(`BaseGASampler.GenerationTracker`, O(new trials) per ask), which is Rustuna's `generation_to_numbers`.

### 4.1 Notes on the single-implementation items

- **S1-S3 (Phase 1).** Port the prototype to HEAD (NumSharp 0.80.0 since), without its bisection switches and its
  first-version leftovers (the string-keyed cache, the generation memo). Edge cases to test:
  - trials that finish out of order wait in the tail and fold in trial order;
  - a FAIL trial that told a generation (the old state is restored again);
  - `considerPrunedTrials` (pruned trials folded as value-rewritten copies);
  - warm start (`sourceTrials`);
  - a new study on a storage that reuses ids (the fold resets when the history no longer starts with the folded trials);
  - `SamplerConcurrency.PerWorker` (the clone gets an empty cache);
  - a stuck RUNNING trial (the fold stops there; speed degrades toward today's, results stay right).
- **S4 (Phase 2).** Keep a mutable draft for each running trial inside `InMemoryStorage`. A write changes the draft in
  O(1) and marks it dirty; a read (`GetTrial`, `GetAllTrials`, a `TrialsPrefixView` index) publishes an immutable
  `FrozenTrial` snapshot from a dirty draft, under the storage lock, and caches it until the next write. A sequential
  trial does 40 writes and a few reads, so O(d) instead of O(d²).
  - The view then takes the storage lock only for a dirty running trial, which keeps plan 12's readers of running
    trials (`HandedOutParams`) correct.
  - The invariant is the one Optuna's copy-on-write protects: a snapshot a caller holds never changes. Tests assert it
    for every path that hands one out.
- **S5 (Phase 3).** Cache the identities of the split, not snapshots: constant liar re-reads running trials, whose
  parameters change between suggests. Key it by a per-study counter of finished trials that `InMemoryStorage` bumps,
  so the key costs O(1); other storages fall back to keying by the list of trial numbers and states.
- **S6 (Phase 3).** Fuse `MixtureOfProductDistribution.LogPdf` / `AddContinuousLogPdf` into per (candidate, kernel)
  loops without the big temporaries. Byte identity requires NumPy's order:
  - the batched truncated-normal log-pdf (`_truncnorm.logpdf(...).sum(axis=-1)` in `probability_distributions.py`)
    computes each term as `(_norm_logpdf(z) - log_gauss_mass) - log(scale)`, with NumPy's `z**2` on an array (a
    square, not `pow`), then sums over the continuous dimensions with NumPy's pairwise summation over a contiguous
    axis; replicate that per (candidate, kernel) on a stack buffer with `MathUtil.PairwiseSum`;
  - precompute only the terms NumPy computes per kernel anyway (log mass, log scale); the same value computed once
    instead of 24 times has the same bits;
  - Rustuna's binned bandwidths (`binned_sigmas`, O(n) instead of a sort) are probably bit-identical: test that, adopt
    if so.
  - If some kernel cannot be made byte-identical, keep the old one for it and move the fast one into D2.
- **S9 (Phase 4).** Measure first: load time and allocations of a 10k-trial SQLite study with CMA-ES pickles. The only
  observable difference is when malformed JSON raises (on access instead of on load); document it, or validate
  cheaply at load.
- **S11 (Phase 5).** Measure `GPSampler` (500 and 1,000 trials) and CMA-ES (10, 40, 100 dimensions) with OpenBLAS at 1
  thread and at the default. Results are identical either way, so this is a resource policy: choose from the numbers
  (one thread at `nJobs: 1`, or by matrix size). `TrialExecutorOptions.BlasThreads` already lets a user override it.
- **S12 (Phase 6).** Store finished trials compactly while every API still returns the same `FrozenTrial` contents:
  - parameter values as one internal-representation `double[]` per trial against a per-study parameter index;
  - distributions and names shared (S7);
  - attributes as raw JSON until read (S9);
  - snapshots materialized on read and cached weakly.
  Measure the memory of a 100k-trial CMA-ES study before and after. This is the compatible answer to the owner's
  "what's best + don't break Optuna compatibility" for discarding.

## 5. The switches (two paths)

### 5.1 D1: `CmaEsSampler(..., reuseToldOptimizer: false)`, decided

The owner chose opt-in with Optuna's path as the default.
- **When true,** after a tell the told optimizer becomes the cache for the state it just stored, so the next asks skip
  the restore. It is still written to storage exactly as before, so resuming and Python workers work, but no worker
  reproduces the C# run.
- **API.** The parameter goes after every Optuna parameter, so positional calls ported from Python keep their meaning.
  The XML doc states the consequence in its first sentence: seeded runs stop matching Python Optuna from generation 3.
- **Tests:**
  - default-path parity is unchanged (the Phase 0 gates);
  - with the switch on: convergence over seeds on sphere and Rastrigin (§2.4's numbers as the bar), the mixed-fleet
    interop case still completes, and a failing teller still restores the old state;
  - a guard test that a default-constructed sampler stays on Optuna's path.

### 5.2 D2: TPE fast numerics, decide after S6

Phase 3 builds S6. Then measure two things:
- the time scoring still takes;
- a fast scoring kernel (vectorized exp and log, reordered sums), with candidate sampling left exact.

Rerun §2.4's noise harness, extended to MOTPE with 1,000 trials and to constraints, and ask the owner with both numbers.
If built, it is `TPESampler(..., numerics: TpeNumerics.Exact)` with the default the owner picks.

### 5.3 D3: separate trial queue, deferred

Rustuna's design, for the record:
- A `TrialQueue` trait (`rustuna_core/src/trial_queue.rs`) with an in-memory default.
- `SQLite3TrialQueue`: a namespaced `trial_queue` table with an atomic pop.
- `DirectoryTrialQueue`: `pending/` and `processing/` files.
- A Redis backend, planned.
- `enqueue_trial` still creates the WAITING trial in storage, numbered at enqueue time, then pushes its id. `ask` pops
  from the queue instead of scanning storage.
- `add_trial` with WAITING is not queued.

Recommendation when the owner returns to it: opt-in, with Optuna's storage-backed WAITING trials as the default:
- the gain in memory is nil (our WAITING pop is already O(1) through a cursor);
- on database storages it saves one query per ask;
- a SQLite or Redis queue fixes Optuna's duplicate pops on SQLite.

The risk of a default in-memory queue is private enqueues across processes.

### 5.4 D4: trial discarding, deferred

S12 comes first because it saves memory without changing any API result. Discarding is worth reopening only if S12's
measured memory is not enough. If it is reopened, the compatible slice is sampler-driven discarding of what no
sampler or API caller can need, which is hard to define for the public `Study.Trials`. Anything beyond that changes
`Study.Trials`, so it would be opt-in on `InMemoryStorage`.

### 5.5 Where switches live (proposed; the owner deferred the general question)

- **On the component they change:** a constructor parameter or options property of the sampler, storage or queue.
  Nothing process-wide: a process default would also reach libraries in the same process (the study `OptunaSearchCV`
  creates).
- **Named for the behavior, not for Rustuna:** `reuseToldOptimizer`, `TpeNumerics.Fast`, `DiscardTrials`.
- **Defaults chosen per item by the owner;** so far Optuna's behavior (§1).
- **One documentation page,** "Performance and compatibility", lists every switch: what it changes, whether seeded runs
  still match Python, and whether the storage can still be shared with Python workers.

## 6. Phases and gates

| Phase | Content | Depends on |
|---|---|---|
| 0 | Gates and benchmarks: long-horizon live-parity cases and the benchmark harness (below) | — |
| 1 | S1, S2, S3 for CMA-ES and the intersection search space | 0 |
| 2 | S4 in-place writes behind snapshots, S7 interning | 0 |
| 3 | S5 MOTPE split cache, S6 TPE exact kernel, S3 for the remaining samplers (QMC) and `GroupDecomposedSearchSpace`; then the D2 measurement and question | 0, 2 |
| 4 | S10 `CachedStorage` incremental reads, S9 lazy attributes (after measuring), S8 bulk attribute API | 0 |
| 5 | S11 OpenBLAS policy (measure, then choose) | 1 |
| 6 | S12 compact finished trials | 2 |
| 7 | D1 `reuseToldOptimizer` | 1 |
| 8 | Docs: the "Performance and compatibility" page; README Performance rerun; THREADING.md §6 (the new sampler state is guarded by the sampler lock); catalog | each phase |

**Gate A: no observable change** (every S item):
- The live-parity suites: unit, interop with the engine required, package parity.
- Phase 0 adds [Slow] interop cases that run Python Optuna and compare every trial, stored pickles included:
  - CMA-ES 40-D for 3,000 trials, and the failing-trial study;
  - TPE 40-D for 1,000 trials;
  - MOTPE 300 trials;
  - NSGA-II 2,000 trials.
  Python is the oracle on every CI OS, so no hash is committed (the bit-exact suites depend on each image's libm).
- During development: a before/after hash probe of seeded studies, built from two trees. `dotnet run probe.cs` re-runs
  a stale build after a referenced project changes, so build explicitly and check that the DLL holds the new code.

**Gate B: concurrency.**
- `ConcurrencyGun` and plan 12's parallel-read tests stay green.
- Each phase that adds state to a sampler or storage adds `Fire` / `ReadInLockstep` cases for it.

**Gate C: the gain is real.**
- The §7 workloads before and after, in Release, with power throttling opted out and the OpenBLAS thread count stated.
- Every number goes into this plan and, at the end, the README.

**Gate D: two paths** (D items).
- Both paths are tested; the default path keeps Gate A.
- A guard test pins the default.

## 7. Targets (to measure, not promises)

| Workload | Today (`d0558f5`) | After the S items (expected) | Rustuna 0.2.1 |
|---|---|---|---|
| CMA-ES, 10k / 100k | 18.3 / 812 s | 2.6 / 26 s measured in the prototype with single-threaded OpenBLAS; lower after S4 and S12 | 1.6 / 13.6 s |
| NSGA-II, 10k / 100k | 3.5 / 62.9 s | about half, if S4 removes most of the 66 % | 0.8 / 19.4 s |
| MOTPE, 1k | 36.6 s | about half, from S5 (24 %) and S4 (23 %) | 6.3 s |
| TPE, 1k | 9.9 s | unknown until S6; the scoring is 58 % | 1.4 s |

Rustuna will stay ahead on TPE, where its acquisition math is native and ours follows NumPy's order. It will also stay
ahead wherever Optuna's protocol costs time (the pickle per generation, per-trial system attributes); keeping that
protocol is what lets Python workers share OptunaSharp's studies.

## 8. Risks

- **S4 aliasing.** A `FrozenTrial` handed out and then changed would silently break samplers and users. Tests must
  cover every path that hands out a snapshot, under concurrency.
- **Index memory.** The folds and caches hold references the storage already holds; they must reset on a new study and
  must not pin discarded or deleted studies.
- **Database storages.** S1-S3 save sampler work there too, but the per-ask history read stays O(n) until S10.
- **Two paths rot.** Every D item doubles a code path; rule 4 builds one only with a measured gain, and Gate D keeps
  both tested.
- **Stale measurements.** Throttling (EcoQoS), Debug builds of file-based probes and stale probe builds have each
  produced wrong numbers in this repository. Gate C's conditions exist because of them.

## 9. Open questions for the owner

- D3 (trial queue) and D4 (discarding) after S12's memory numbers.
- D2's default once S6 and its fast variant are measured.
- §5.5's switch shape for future switches; D1 uses a constructor parameter.
