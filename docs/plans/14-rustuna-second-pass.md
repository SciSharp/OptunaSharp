# Plan 14 — Rustuna, second pass: compatible speedups, each gated byte for byte

> **Status: BUILT (2026-11-09).** Six steps on `main`: A `adda96c`, B `db4e46c`, C `a279254`, D `45fadd1`,
> E `2dc35cf`, F `14b4698`. None of them changes a seeded trial or a stored byte, so none is a switch. §4 lists the
> byte-for-byte gates of every affected area, §5 the final before/after proof. §8 is the concurrency review that
> followed: no threading defect, new concurrency tests, and a fix to the TPE cache for studies recreated under a
> reused id. The comparison behind the plan: catalog note `optunasharp/notes/rustuna-comparison-2026-11-08.md`,
> section "Second pass".

---

## 1. The request

After plan 13 the owner asked (2026-11-09): "Keep comparing, what other rustuna optimizations with backwards
compatability have been achieved and we can use it aswell". The answer was a second read of Rustuna's `main`
(`d0ec7db`, 2026-11-04) against OptunaSharp at `f4eca0c` (plan 13 with S8 and S12), with new profiles of the plan 13
workloads. Then:

> 1. Each affected area must have byte-for-byte parity test coverage
> 2. Improve performance
>
> Implement.

Plan 13's rule carries over: a change that keeps every result is the only implementation. Nothing here changes a
result, so nothing here is opt-in. "Byte for byte" is checked three ways:

- **Against the code it replaced.** Unit tests pin each new path to the old one, value bits and, for random draws, the
  generator's next state.
- **Across whole studies.** A hash probe (scratchpad `p14/hashprobe`, session `5417a82e`) runs 36 seeded studies over
  every sampler family the changes reach, directly or through `RandomSampler` and `SearchSpaceTransform`, and hashes
  every trial's number, state, parameter names and value bits, and objective-value bits. It was built against the
  `f4eca0c` snapshot and against each step; every hash stayed equal.
- **Against live Optuna 5.0** (pythonnet, Python 3.12.12, the vendored `refs/optuna`), where OptunaSharp's parity with
  it is bit-exact: random search, NSGA-II, NSGA-III, CMA-ES, QMC, and the journal and SQLite storages. TPE's live
  parity stays what it was before this plan: the same trajectory, with values compared within the documented `erf`
  tolerance. TPE's byte-for-byte gates are therefore the unit pins and the hash probe.

## 2. Inventory and verdicts

"Where it showed" is the state at `f4eca0c`. Profile shares are from stopwatch-instrumented scratch copies where
marked; `dotnet-trace`'s sampled thread time misplaces whole seconds (§6).

| ID | Change | From | Where it showed | Verdict |
|---|---|---|---|---|
| A | Random draws and search-space transforms without NumSharp arrays on per-parameter paths | Found while comparing; Rust has no finalizers | GC pauses: 47 % of a 40-parameter random search, 41 % of NSGA-II; random 10k gen-0/1 collections 154/153 | Built, `adda96c` |
| B1 | TPE kernel widths in O(n), exact | Rustuna's `binned_sigmas`, made exact by verifying each bin | The width argsort: 24 % of TPE 1k (stopwatch) | Built, `db4e46c` |
| B2 | Finished trials' internal values cached across asks | Rustuna `ddd9c56`, `e7a22d7`, `a51e6db` | History conversion: 14 % of TPE 1k (stopwatch) | Built, `db4e46c` |
| B3 | No `exp` call where its result is exactly +0.0 | OptunaSharp's own, from measured `Math.Exp` costs | The log-sum-exp of the candidate scoring | Built, `db4e46c` |
| C | QMC's sample-id key computed once per search space | OptunaSharp's own, from an allocation trace | Most of QMC's 1.58 GB allocated per 10k trials | Built, `a279254` |
| D | The journal file backend keeps sparse line offsets | Rustuna `82b13d8` | One dictionary entry per log line: 160 MB for a 4.3M-line journal | Built, `45fadd1` |
| E | SQLite's cached refresh as `UNION ALL` | Rustuna `d46ace5` | The OR form scans `ix_trials_study_id`: 2.6 ms against 64 µs per refresh at 20k trials | Built, `2dc35cf` |
| F | A categorical weight's log taken once per (kernel, choice) | Rustuna `198e484` cuts the same logs, with other numerics | One `log` per candidate and kernel | Built, `14b4698` |
| — | Rustuna's binned sigmas as written, its categorical numerics, `log_diff_cdf` (`a4d0bc3`) | Rustuna | | Not taken: different bits |
| — | A `(study_id, number)` index (part of `d46ace5`) | Rustuna | | Not taken: changes Optuna's schema |
| — | `UNION ALL` in the EF Core storage | | PostgreSQL and MySQL combine ORed index ranges | Not changed |
| — | Candidates scored as ordered arrays (`93c2abe`); the copies left in model construction (rest of `fa4af9a`) | Rustuna | Second-pass profile: the dictionary path of `ParzenEstimator.LogPdf` 9 % of TPE 1k, `Transform` 9 % of MOTPE | Open (§7) |
| — | NSGA-II per-trial precomputation (`946fdf1`, `bd72444`, `cbecbf3`, `ed090c4`) | Rustuna | After this plan: NSGA-II 10k 0.75 s warm, 1.4 s with the JIT; Rustuna 0.2.1 0.82 s | Open (§7) |
| — | CMA-ES: an in-memory copy instead of the restore after each tell | OptunaSharp's own | `RestoreOptimizer` 6 % and the hex decode 4 % of a traced CMA-ES 20k run | Open (§7): byte-safe only if the copy pickles exactly like a restore |
| — | `apply_discard` / `discard_trials` | Rustuna | | Not compatible (plan 13, D4); plan 13's S12 cut the memory instead |

## 3. What each step does

**A — NumSharp temporaries (`adda96c`).** Every NumSharp `NDArray` has a finalizer. An array nobody disposes
survives its first collection on the finalization queue and is promoted, so the sampler paths that built small arrays
per suggestion turned almost every gen-0 collection into a gen-1. The new `Internal/NumPyDraws` reads the generator's
bit generator directly (`BitGenerator.NextDouble` under its lock, what NumSharp's own scalar and fill paths do):
`Rand`, `RandomSample` and `Uniform` create no array; `RandInt` and `Normal` still call NumSharp and dispose the
result. `Uniform` keeps NumPy's order: the range, its finiteness check before any draw, then `low + range * U`.
`SearchSpaceTransform` gained `TransformValues` and `UntransformValues(ReadOnlySpan<double>)`, and `Untransform`
disposes its `astype` copy. `RandomSampler` draws each bound row through `NumPyDraws.Uniform` instead of building five
arrays per suggestion; the NSGA-II family (child generation, crossover, every crossover operator, polynomial mutation),
CMA-ES, QMC and TPE's candidate sampling use the helpers. Every draw is the same call on the same generator in the
same order.

**B — TPE (`db4e46c`).** (1) `ParzenEstimator.BinnedRawSigmas`: a width is the larger distance to the two sorted
neighbours, clipped up to `minsigma`, so binning the centres at width `minsigma` decides most widths without a sort.
Rustuna trusts its binning and can differ in the last bits; this version proves its answer: bin assignment is
monotone, so bins are contiguous runs of the sorted order; a bin's extremes are taken at the stable sort's positions;
every bin is checked to span at most `minsigma`. Anything else falls back to the sort (`SortedRawSigmas`).
(2) `TpeObservationCache`: each finished trial's internal value per (parameter name, distribution), or the fact that it
lacks the name; running trials (constant liar) are never cached; another study or storage, or more than 256 columns,
starts over. (3) The log-sum-exp stores 0.0 below `ExpIsZeroBelow` (-746) instead of calling `Math.Exp`, which returns
exactly +0.0 below about -745.1332 on every runtime and platform CI uses; the zero still enters the pairwise sum.

**C — QMC (`a279254`).** The sample-id attribute is `"qmc:" + sha256(json.dumps(qmc_vars))`, rebuilt per trial from
every distribution's repr. `QmcIdKey` keeps it for the last search space and reuses it while the next call passes the
very same distribution objects in the same order. Reference identity decides, because two categoricals can be equal
by Python's rules (`True == 1`) and still print differently.

**D — Journal file backend (`45fadd1`).** Instead of one offset per log line (Python's `_log_number_offset`), the
backend keeps the furthest line start any read established and one checkpoint every 4,096 lines. A sync still seeks
straight to the next unread line; an earlier start seeks to the nearest checkpoint and skips forward, as Python skips
lines when it reads from byte 0. A line's successor is recorded only after the line decodes.

**E — SQLite refresh (`2dc35cf`).** `CachedStorage` asks for `trial_id > watermark` plus the unfinished trials at or
below it. Optuna's `... AND (trial_id > ? OR trial_id IN (...))` makes SQLite walk the study's whole index; the two
halves joined by `UNION ALL`, each answered through the primary key, return the same rows, and the final
`ORDER BY trial_id` the same order. Only studies with a running trial below the last finished one take this query
(parallel workers, or a crashed worker's trial, which then costs on every refresh).

**F — TPE categorical (`14b4698`).** A categorical dimension added `log(weights[k, x])` per candidate and kernel. The
log of a (kernel, choice) column is now taken the first time a candidate draws that choice and reused. `Math.Log` is
deterministic and the terms enter the same buffer elements in the same order, so every sum keeps its bits.

## 4. Byte-for-byte gates, per affected area

| Area | Unit (OptunaSharp against the code it replaced) | Live Optuna 5.0 |
|---|---|---|
| Random draws (`NumPyDraws`) | `NumPyDrawsTests`: every helper against the NumSharp call it replaces, value bits and the generator's next state; scalar and vector uniforms, every bound shape `RandomSampler` meets, row-by-row against the array uniform, non-finite ranges, `randint` seed ranges, Gaussians with the cached polar value, NSGA-II's mixed draw sequence | through every suite below |
| `SearchSpaceTransform` | `SearchSpaceTransformTests`: `TransformValues` = `Transform`, `UntransformValues` = `Untransform` to the bit (mixed space, with and without the unit cube, clamped and tied coordinates); the `NDArray` overload still reads float32/int64 input | through the sampler suites |
| `RandomSampler` | `RandomSamplerTests`: 60 trials × 3 seeds over every distribution kind replay the pre-plan-14 array draw | `RandomSamplerParityTests.LongRun_AllDistributionKinds_IsBitIdentical_ToLiveOptuna`: 300 trials × 2 seeds, every value compared through Python's `repr` |
| NSGA-II family, NSGA-III, CMA-ES | the draw-sequence pins above | NSGA-II (every crossover, polynomial mutation, constraints, a 2,000-trial 40-parameter run), NSGA-III, CMA-ES and the mixed fleet, all green |
| TPE kernel widths | `TpeBinnedSigmasTests`: binned against sorted widths bit for bit over uniform, tied, clustered, on-edge, one-ulp-off-edge and floor-spaced centres, 2 to 1,500 centres, ten domains, both endpoint settings; asserts the fast path answers more than 90 % of uniform inputs. Scratch fuzz: 800,000 more cases, 0 mismatches | TPE suite (trajectory within the `erf` tolerance), including the 1,000-trial 40-parameter and MOTPE 300-trial runs |
| TPE observation cache | `TpeObservationCacheTests`: a warm cache against one emptied before every suggestion, same trials: dynamic ranges, a conditional parameter, pruned trials, both modes, plain and compact storage, MOTPE, constant liar with running trials, an ever-changing range, a sampler reused for a second study | `TpeSamplerParityTests.DynamicRangesConditionalAndPrunedTrials`, `ConstantLiarWithRunningTrials_AskTell` |
| TPE `exp` skip | `TpeExpUnderflowTests`: `Math.Exp` gives +0.0 for the 100,000 doubles below the bound and 200,000 inputs down to -1e6, `-inf` and `double.MinValue`, on every CI platform | TPE suite |
| TPE categorical logs | `TpeExactKernelTests.CategoricalLogPdf_OneLogPerKernelAndChoice_MatchesThePerCandidateLogsToTheBit`: 1, 2 and 4 categorical dimensions, 2-7 choices, weights with exact zeros (`-inf` terms), 1-40 kernels, against the per-candidate logs | TPE suite |
| QMC sample-id key | `QmcSamplerTests.QmcIdKey_Memoized_EqualsAFreshComputation`, including equal-by-value categoricals that print differently | `QmcSamplerParityTests.LongRun_40Parameters_300Trials_FullState_IsBitIdentical`: Sobol and Halton, scrambled and not, the whole study state with the `qmc:<sha256>` counter |
| Journal file backend | `JournalOffsetCheckpointTests`: reads from 13 positions around the checkpoints equal a full read, cold and warm; a reader that stopped half way resumes exactly; 12 concurrent readers; an undecodable record fails every read that reaches it; a partial last line waits | `JournalStateParityTests.LargeJournal_PythonWrites_CSharpReplaysAndContinues_SharedBackendCatchesUp_FullState`: Python writes 250 trials × 40 parameters; a C# storage replays it, two share a backend while one writes 30 more; the lagging storage, a cold one and Python's replay give the identical state |
| SQLite refresh | `SqliteRdbStorageTests.CachedRefresh_WithRunningTrialsBelowTheWatermark_EqualsAFreshRead`: after every step the cached view equals a fresh storage's read | `SqliteRdbInteropParityTests.RunningTrialBelowTheWatermark_CachedRefresh_MatchesPythonsRead`: C#'s cached view equals Python's fresh read field for field |
| Every sampler family | hash probe: 36 seeded studies equal before and after each step | |

## 5. Final before/after proof

`scratchpad/p14/final-proof.sh` (session `5417a82e`): the plan 13 proof program (`proof/Program.cs`) built twice in
Release, against the `f4eca0c` snapshot (before: plan 13 with S8 and S12) and against `14b4698` (after). 40 float
parameters in [-10, 10], seed 0, sampler defaults, `InMemoryStorage`; objective `sum((x - 1)^2)`, or `sum(x^2)` and
`sum((x - 2)^2)` for two objectives. i9-13900K, Windows 11, .NET 10, power throttling opted out, OpenBLAS at its
default thread count. Each workload ran before, after, before, after, back to back, 2026-11-09 14:23-14:27, with other
sessions loading the machine to about 30 % CPU. Rep 0 includes the JIT; rep 1 is a second study in the same process.
Ranges span the two runs. The hash covers every trial's parameters and values by their bits; every pair was equal.

| Workload | Before rep 0 / rep 1 | After rep 0 / rep 1 | Speed-up, rep 1 | Allocated, rep 1 | gen-0/1/2, rep 1 | GC pauses, rep 1 |
|---|---|---|---|---|---|---|
| Random, 10,000 | 4.31-4.46 / 3.52-3.61 s | 0.81-0.82 / 0.50-0.52 s | 6.8-7.2× | 2,306 → 654 MB | 130/129/2 → 37/9/1 | 1.63-1.72 → 0.14 s |
| QMC, 10,000 | 2.31-2.38 / 0.99-1.01 s | 0.82-0.87 / 0.43-0.45 s | 2.2-2.3× | 1,584 → 472 MB | 89/17/1 → 27/9/1 | 0.20-0.21 → 0.12-0.13 s |
| NSGA-II, 10,000 | 2.64-2.71 / 1.51-1.57 s | 1.40-1.45 / 0.75 s | 2.0-2.1× | 1,234 → 710 MB | 70/69/2 → 40/10/1 | 0.66-0.68 → 0.16 s |
| TPE, 1,000 | 4.75-4.85 / 3.86-3.91 s | 3.18-3.30 / 2.47 s | 1.6× | 1,569 → 1,026 MB | 87/4/0 → 56/3/0 | 0.06 → 0.04 s |
| TPE, 1,000, `compactFinishedTrials: true` (one pair) | 4.28 / 3.39 s | 3.43 / 2.43 s | 1.4× | 1,595 → 1,051 MB | 88/2/0 → 58/2/0 | 0.04 → 0.02 s |
| MOTPE, 1,000 (rep 0) | 12.70-12.82 s | 10.44-10.52 s | 1.2× | 3,418 → 2,832 MB | 190/3/1 → 157/1/0 | 0.18 → 0.09-0.10 s |
| CMA-ES, 10,000 (three runs; rep 0 medians 3.66 → 3.64 s) | 3.64-3.99 / 2.09-2.28 s | 3.61-4.68 / 2.10-2.21 s | unchanged | 1,744 → 1,678 MB | 99/28/4 → 93/23-24/4 | 0.26-0.28 → 0.24-0.25 s |
| NSGA-II, 100,000 (rep 0, one pair) | 17.19 s | 8.60 s | 2.0× | 12,438 → 7,240 MB | 704/698/10 → 408/108/7 | 7.12 → 1.81 s |
| CMA-ES, 100,000 (rep 0, one pair) | 23.88 s | 22.11 s | within noise | 17,419 → 16,766 MB | 969/248/21 → 913/211/19 | 2.74 → 2.29 s |

The categorical-heavy scenario of the hash probe (TPE, 500 trials, 20 five-choice categoricals and 20 floats; two
pairs, same hash `E015C07493B6830A`): 2.5 and 2.7 s → 1.8 and 1.8 s.

The storage steps were measured at their commits with their own probes (scratchpad `p14/journalmem`, `p14/sqlgrowth2`):

| Storage workload | Before | After | Same records |
|---|---|---|---|
| Journal file backend: memory kept after a full read of a 4.3M-line, 1.09 GB journal | 160.3 MB | 0.1 MB | yes |
| Journal file backend: full read / next sync | 5.48-7.36 s / 0.28-0.35 ms | 5.22-5.26 s / 0.34-0.36 ms | yes |
| SQLite through `CachedStorage`, 8,000 trials with one trial left running from the start | 13.0 s | 9.6 s | yes |

Against Rustuna 0.2.1 (plan 13 §2.1, measured on a quieter day): NSGA-II 10k 0.82 s and 100k 19.4 s, TPE 1k 1.36 s,
MOTPE 1k 6.30 s, CMA-ES 10k 1.59 s and 100k 13.6 s. NSGA-II at 100k (8.6 s, JIT included) is now below Rustuna's
time, and so is a warm NSGA-II 10k (0.75 s); with the JIT, NSGA-II 10k takes 1.4 s. TPE, MOTPE and CMA-ES remain
behind; part of that gap is the compatibility OptunaSharp keeps (TPE scores candidates in NumPy's order, and CMA-ES
pickles its state into the storage every generation so Python workers can share the study).

The README's sampler table (benchmark 14's C# probes, three fresh processes per configuration, built against
`f4eca0c` and `14b4698`, interleaved, 2026-11-09 14:29-14:31, load 16-57 %), medians:

| Probe | Before | After |
|---|---|---|
| TPE, 6 parameters, 500 trials | 0.96 s | 0.61 s |
| TPE multivariate, 6 parameters, 500 trials | 0.97 s | 0.66 s |
| TPE, 100 parameters, 150 trials | 1.8 s | 1.6 s |
| GP, 6 parameters, 100 trials | 3.52 s | 3.43 s |
| CMA-ES, 6 parameters, 3,000 trials | 0.83 s | 0.74 s |
| CMA-ES, 200 parameters, 100 trials | 1.1 s | 1.0 s |
| NSGA-II, 6 parameters, 3,000 trials | 0.65 s | 0.31 s |
| QMC, 6 parameters, 3,000 trials | 0.36 s | 0.13 s |
| Random, 6 parameters, 5,000 trials | 0.78 s | 0.15 s |
| Random, 1,000 parameters, 200 trials | 2.6 s | 0.6 s |
| Random, 4 parameters, 50,000 trials: each block of 10,000 | 0.39-0.77 s (the first 1.08-1.19 s) | 0.08-0.25 s |

## 6. Findings along the way

- **Sampled thread time hid TPE's stages.** `dotnet-trace`'s `dotnet-sampled-thread-time` charged EventPipe's thread
  suspensions to `PollGC`, `Monitor.Enter` and `memmove`. Stopwatches around TPE's stages in a scratch copy gave the
  real split before step B: scoring 1.74 s (40 %), estimator construction 1.34 s (31 %, of which the width argsort
  1.05 s), history conversion 0.60 s (14 %). After B1 and B2: construction 0.54 s, conversion 0.15 s.
- **`Math.Exp` has three speeds here:** 5 ns for a normal result, 16 ns on its underflow-to-zero path, 63 ns for a
  subnormal result. MOTPE's scoring is dominated by subnormal results, which an exact port has to compute.
- **A flaky journal lock test.** `JournalConcurrencyTests.Lock_WithoutAGracePeriod_WaitsForALeftBehindLockUntilItIsRemoved`
  failed once on net10.0 under load and passed on three reruns; the lock code is untouched.
- **Out-of-tree builds.** The schema-upgrade live cases (`SqliteRdbUpgradeInteropParityTests`) find the repository
  root from the test binaries, so they cannot run from a build with `--artifacts-path` outside the repository. They
  exercise the migration chain, which step E does not touch.

## 7. Open

- TPE: score candidates as ordered arrays instead of name-keyed dictionaries (Rustuna `93c2abe`), and drop the copies
  left in estimator construction (the rest of `fa4af9a`). Both are the same arithmetic.
- NSGA-II: precompute each trial's values and constraints, build parameters only for the selected parents, update the
  joint search space in place (Rustuna `946fdf1`, `bd72444`, `cbecbf3`, `ed090c4`).
- CMA-ES: replace the restore after each tell by an in-memory copy that applies the pickle round trip's normalizations
  (C mirrored from its upper triangle, B and D dropped). About 10 % of a traced run by sampled thread time, so to be
  measured with a stopwatch first; it must serialize to the same `CmaPickle.ToHex` as a restore.
- QMC, scrambled Sobol: `QMCSampler` takes the point at `sample_id` by generating every point up to it
  (`ScrambledSobol.Random(d, sampleId + 1, seed)`), so a study costs O(n²) points. Computing the one point directly
  would be O(1) per trial; it has to stay bit-exact with scipy's scrambled `Sobol.fast_forward(...).random(1)`.
- NumSharp temporaries were traced (allocation ticks by creating frame, scratch `p14/nd/AnalyzeNd.cs`) on the five
  plan 13 workloads only: random, QMC, NSGA-II, TPE, CMA-ES. GP, DE, PSO, the NES family and the other samplers were
  not traced.

## 8. Concurrency review (2026-11-09, after the build)

The owner asked: "Did we not break the concurrency?" Everything plan 14 added that more than one thread can reach, and
what guards it:

| State | Reached from | Guard | Tests |
|---|---|---|---|
| `TpeObservationCache` (a `TPESampler` field) | `GetInternalRepr` ← `BuildParzenEstimator` ← `Sample` ← `SampleRelative` / `SampleIndependent` only | The sampler monitor (`BaseSampler.SyncRoot`), held for every study-driven call, also when several studies share the sampler. A `PerWorker` instance gets its own cache: the field initializer runs for the private copy constructor, which does not copy it | New: `WorkerInstances_OwnTheirCaches_AndThePrototypesStaysUnused`, `SharedByEightWorkers_EveryCachedRowIsItsTrialsOwnValue` (eight workers, constant liar; every cached row equals its trial's conversion to the bit). Existing: `SamplerParallelReadTests` and `SamplerSharingConcurrencyCoverageTests` (`tpe`, `tpe-independent`) |
| `QMCSampler._lastKeyNames` / `_lastKeyDistributions` / `_lastKeyQmcId` | `QmcIdKey` ← `FindSampleId` ← `SampleQmc` ← `SampleRelative` only | The sampler monitor; QMC has no worker instances, so it is always shared and called one at a time | Existing: `SamplerParallelReadTests`, `SamplerSharingConcurrencyCoverageTests` (`qmc`, `qmc-scrambled`); `QmcIdKey_Memoized_EqualsAFreshComputation` |
| `JournalFileBackend` offsets (`_latestLogNumber`, `_latestLogOffset`, `_offsetCheckpoints`) | `ReadLogs`, from any number of storages sharing the backend | `_offsetLock` on every access; each read carries its own byte position; a line start is recorded only after the line decodes, so the furthest start advances one line at a time and each checkpoint is appended once, in order (index check) | New: `ReadersSyncingWhileOneWriterAppends_AcrossCheckpoints_AllReadExactly` (one writer and six syncing readers on one backend across three checkpoints). Existing: `ConcurrentReaders_FromScatteredPositions_AllReadExactly`, `JournalConcurrencyTests` |
| `Internal/NumPyDraws` | Every sampler that draws | No state. Each draw or fill holds `BitGenerator.@lock`, the lock NumSharp's own primitives take and the one NumSharp asks callers outside its `Generator` to hold | Through the suites above |
| Exact widths, the `exp` skip, categorical logs, `TransformValues` / `UntransformValues`, the SQLite query | | No shared state; the SQLite query runs under the storage's one-connection lock as before | |

Runs on the fixed tree: the unit suite with CI's filter, 2172 passed on net8.0 and on net10.0 (one Windows self-skip
each); every concurrency-focused class (`OptunaSharp.Tests.Concurrency`, `OptimizeConcurrencyTests`,
`SamplerNJobsSweepTests`, `JournalConcurrencyTests`, `JournalOffsetCheckpointTests`, `TpeObservationCacheTests`,
`QmcSamplerTests`) three more times in a row, 342/342 each time; live TPE, compact-storage, importance and
battle-tested parity, 83/83; the storage suite's gRPC, SQLite and journal-contract classes, 114/114.

**Found by the review, not a threading bug: recreated studies.** `TpeObservationCache` reset only when the storage
object or the study id changed. SQLite reuses the highest deleted study and trial ids, and a gRPC proxy (or a custom
storage) passes them through as one storage object, so a reused `TPESampler` on a study deleted and recreated under the
same id read the deleted study's values for the cached trial numbers. Optuna has no such cache, and TPE before plan 14
re-read every trial, so this was a regression. The storage suite's new
`GrpcStorageProxyTests.Tpe_ReusedOnARecreatedStudyUnderTheSameId_SamplesAsUnderANewId` reproduced it: the same run
with the deleted study's id reused suggested different parameters from the first TPE trial on. The fix applies
`PerStudyState`'s `TrialAnchor` rule per row: each cached row keeps its trial's storage id and start time, and every
call checks the trials it is about to read, starting over on a mismatch. The cache also holds its storage weakly now,
as `PerStudyState` does, so a long-lived sampler does not keep the last storage alive. New unit tests:
`LiveStudy_NeverStartsOverOnATrialsIdentity` (no false alarm in either TPE mode, plain or compact storage, constant
liar with running trials), `TrialNumbersCarryingOtherTrials_StartTheCacheOver`, `Cache_DoesNotKeepTheStorageAlive`.
Mutation checks: with the identity check disabled, or the storage held strongly, those tests fail. The 36-study hash
probe is unchanged, and TPE / MOTPE times are unchanged within noise (TPE 1k warm: 2.68-2.90 s before the fix,
2.58-3.04 s after; allocations equal).
