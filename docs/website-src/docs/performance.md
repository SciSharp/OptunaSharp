# Performance and Compatibility

OptunaSharp reproduces Optuna's seeded results: the same seed, sampler and objective give the same trials, value for
value, as Python Optuna 5.0. Speed work keeps that promise. A technique that leaves every result unchanged is simply
built in; one that would change results is a switch on the component it changes, off by default, and this page lists
each one.

## What is fast, and why results stay the same

The optimizations below came from comparing OptunaSharp with [Rustuna](https://github.com/optuna/rustuna), the Optuna
team's Rust port (plan 13 in the repository, `docs/plans/13-rustuna-derived-performance.md`). None of them is
observable: seeded studies hash identically, every trial's parameters and values by their bits, before and after.

| Area | What changed | Effect |
|---|---|---|
| `CmaEsSampler` | The optimizer restored from the stored state is kept for the rest of its generation, and finished trials are scanned once instead of on every ask | 40 parameters, 100,000 trials: about 35 times faster |
| `InMemoryStorage` | A running trial nobody has read is written in place; a record any read returned is copied before its next write, so what a caller holds never changes | Suggestions no longer copy the trial's dictionaries |
| `InMemoryStorage` | Identical distributions and parameter names are stored once per study | Less memory for long studies |
| `TPESampler` | The scoring kernel computes per (kernel, dimension) what it used to compute per candidate, on vector lanes that keep the scalar summation order, with pooled buffers | With the next row: 40 parameters, 1,000 trials about twice as fast, and 2.6 times with two objectives |
| `TPESampler` | Independent sampling (the default for several objectives) reuses the trial's below/above split for every parameter | One split per trial instead of one per parameter |
| `QMCSampler`, genetic samplers, search spaces | History read through a zero-copy view instead of a copied list | Per-ask cost no longer grows with the study |
| `CachedStorage` (around every RDB storage) | Cached trials kept in number order instead of sorted on every read | |
| SQLite and EF Core storages | Each distinct distribution text parsed once per load | SQLite, 10,000-trial cold load: 2.1 s to 1.5 s, 60 % less memory |

Measured on an Intel Core i9-13900K under Windows 11 with .NET 10 Release builds, 40 float parameters, seed 0, the
same program built against the code before and after; each pair ran back to back on a machine that was also running
other work, so compare within a row. The first number of a pair includes the .NET JIT warming up.

| Study | Before | After | Same results |
|---|---|---|---|
| TPE, 1,000 trials | 10.5 s | 5.0 s | yes |
| TPE (two objectives), 1,000 trials | 47.4 s | 18.1 s | yes |
| CMA-ES, 10,000 trials | 25.4 s | 4.5 s | yes |
| CMA-ES, 100,000 trials | 988 s | 27.5 s | yes |
| NSGA-II, 10,000 trials | 3.8 s | 2.9 s | yes |
| NSGA-II, 100,000 trials | 73.2 s | 22.2 s | yes |
| QMC, 10,000 trials | 4.8 s | 2.5 s | yes |
| Random, 10,000 trials | 5.8 s | 4.7 s | yes |

### Second comparison

A second read of Rustuna and new profiles of the same studies (plan 14, `docs/plans/14-rustuna-second-pass.md`)
found more changes that leave every result unchanged. Each one is pinned by tests that compare it with the code it
replaced bit for bit, and by the live-Optuna parity tests of the samplers and storages it touches.

| Area | What changed | Effect |
|---|---|---|
| `RandomSampler`, the NSGA-II family, `CmaEsSampler`, `QMCSampler`, `TPESampler` | Random draws and search-space transforms no longer create NumSharp arrays for each parameter. A NumSharp array has a finalizer, so every dropped one survived a garbage collection and pulled the next collection into generation 1 | Random search 5 to 7 times faster, NSGA-II about twice |
| `TPESampler` | Kernel widths in linear time where that provably gives the sorted result, and the sort otherwise; finished trials' values reused across asks instead of converted again | Estimator construction and history reading about 2.5 and 4 times faster |
| `TPESampler` | No `exp` call where its result is exactly zero; a categorical weight's logarithm taken once per kernel and choice instead of once per candidate | 20 categorical and 20 float parameters, 500 trials: 2.6 s to 1.8 s |
| `QMCSampler` | The sample-id key (a SHA-256 of the search space) computed once per search space instead of once per trial | Most of QMC's allocations; with the first row, QMC 2.2 times faster |
| `JournalFileBackend` | One line offset kept every 4,096 lines instead of one per line | A 4.3-million-line journal: 160 MB to 0.1 MB kept in memory |
| SQLite storage | A cached refresh with a running trial below the last finished one asks with `UNION ALL` instead of `OR`, which SQLite answers through the primary key | 8,000 trials with one trial left running: 13.0 s to 9.6 s |

Measured as above, the "before" column being the code after the first comparison, on a machine running other work
at about 30 % load. Times include the .NET JIT warming up:

| Study | Before | After | Same results |
|---|---|---|---|
| Random, 10,000 trials | 4.4 s | 0.8 s | yes |
| QMC, 10,000 trials | 2.3 s | 0.8 s | yes |
| NSGA-II, 10,000 trials | 2.7 s | 1.4 s | yes |
| NSGA-II, 100,000 trials | 17.2 s | 8.6 s | yes |
| TPE, 1,000 trials | 4.8 s | 3.2 s | yes |
| TPE (two objectives), 1,000 trials | 12.8 s | 10.5 s | yes |

A second study in the same process, past the JIT, gains more: random search 7 times, QMC 2.2 times, NSGA-II 2 times,
TPE 1.6 times. CMA-ES is unchanged within the noise.

## Start-up time

OptunaSharp ships IL, so a process compiles each method with the .NET JIT the first time it calls it. In a short
process that is most of the work: a program that runs five TPE trials spends about 340 ms of its 400 ms of CPU in the
JIT. Two settings in the project of the application that uses OptunaSharp shorten that, and neither changes a result:

| Setting in your application's project | What it does | A five-trial TPE process |
|---|---|---|
| `<PublishReadyToRun>true</PublishReadyToRun>`, then `dotnet publish -r <runtime identifier>` | Compiles the application and its libraries, OptunaSharp and NumSharp included, ahead of time | Time in the JIT 337 → 108 ms, the five trials 265 → 163 ms, CPU 406 → 281 ms |
| `<TieredPGO>false</TieredPGO>` | Skips the profile-guided recompilation of hot methods, which a short process never pays back | Time in the JIT 337 → 319 ms, CPU 406 → 375 ms |

36 seeded studies, across every sampler family, hash identically (every trial's parameters and values, by their
bits) whether the program is published ReadyToRun or as IL. A long study gains little from either setting: the JIT's
work is done once.

The `optunasharp` command-line tool starts one short process per command, so it ships with `TieredPGO` off. An `ask`
and a `tell` (two processes) on a SQLite study driven by TPE took 1.79 s with PGO and 1.58 s without, and 1.44 s
published ReadyToRun; the tool package itself is IL, because a .NET tool package serves every platform.
`<InvariantGlobalization>true</InvariantGlobalization>` saved another 10 % of a cycle by not loading ICU, but it makes
every culture invariant, which changes how an application formats and parses text, so the tool does not set it.

Measured on the i9-13900K above with .NET 10, medians of 15 interleaved runs (the command-line tool: 10 cycles), on a
machine running other work at 60–100 % load.

## Switches that change results

| Switch | What it does | Seeded runs still match Python? | Can Python workers share the storage? |
|---|---|---|---|
| `new CmaEsSampler(..., reuseToldOptimizer: true)` | Keeps the optimizer that just told a generation instead of restoring the state it stored; about 20-25 % faster on 40 parameters | No: from about the third generation (Optuna's pickle stores only the covariance's upper triangle, so a restored optimizer differs in the last bits) | Yes: the state is still stored every generation exactly as Optuna stores it |
| `TrialExecutorOptions.BlasThreads = 1` with `nJobs: 1` | Runs OpenBLAS on one thread; CMA-ES was 12 % faster at 40 parameters and 24-35 % faster at 100 | Up to about 40 parameters, yes; at 100 parameters no, because OpenBLAS's threaded kernels sum in another order (NumPy's results depend on the thread count the same way) | Yes |

With `nJobs` above 1 OptunaSharp already runs OpenBLAS on one thread automatically: parallel runs reseed every trial,
so they are not reproducible anyway, and several workers in linear algebra at once would otherwise oversubscribe the
cores (see [Easy Parallelization](key-features/parallelization.md)).

## Memory: compact finished trials

`new InMemoryStorage(compactFinishedTrials: true)` stores each trial compactly once it finishes. Its parameters
become one internal value each, stored against a parameter schema that the study's trials share, and its
distributions come from that schema. Its attributes become two arrays, with long ASCII strings stored as bytes:
CMA-ES's optimizer state, a hexadecimal string, takes two digits per byte. Empty dictionaries take no memory.
Results do not change: each pair below gave the same trials (equal hashes of every parameter and value), and the
live-Optuna parity tests pass on it bit for bit.

| Study (40 parameters, seed 0) | Retained by the finished study | Peak working set | Time |
|---|---|---|---|
| CMA-ES, 100,000 trials | 1,287 MB to 257 MB | 1,846 MB to 458 MB | 23.2 s to 22.1 s |
| NSGA-II, 100,000 trials | 671 MB to 88 MB | 751 MB to 167 MB | 15.7 s to 14.6 s |
| CMA-ES, 10,000 trials | 130 MB to 27 MB | 372 MB to 137 MB | 2.28 s to 2.31 s |
| TPE, 1,000 trials | 13.4 MB to 6.9 MB | 109 MB to 92 MB | 3.8 s to 3.4 s |
| Random, 10,000 trials | 64 MB to 6.7 MB | 200 MB to 91 MB | 3.5 s to 3.5 s |

The times are the same within the machine's noise: CMA-ES over 10,000 trials averaged 2.28 s against 2.31 s over
six steady runs each, and the large studies spend less time in the garbage collector (CMA-ES over 100,000 trials:
2.7 s of pauses against 1.4 s). Allocation stays within 6 %. It is a switch rather than the default because a
finished trial no longer holds dictionaries:

- `Params`, `Distributions`, `UserAttrs` and `SystemAttrs` of a finished trial are read-only views, with the same
  entries in the same order. A cast to `OrderedDict` fails, and a write through `IDictionary` throws.
- Numbers and long strings are rebuilt on each read, so two reads are equal but are not the same object.
- `ParamsDict`, the other `…Dict` properties and `IntermediateValues` still return real dictionaries. The first such
  read gives the trial its dictionary back, and it keeps it, so a change made through it stays; that trial then
  loses its saving.
- A finished trial no longer shares its dictionaries with the running record it was copied from.
- Only `InMemoryStorage` has the switch; journal and database storages keep their records as before.

## Writing several attributes at once

`trial.SetUserAttrs(attrs)`, `study.SetUserAttrs(attrs)` and the storage methods `SetTrialUserAttrs`,
`SetTrialSystemAttrs`, `SetStudyUserAttrs` and `SetStudySystemAttrs` write a dictionary of attributes in one step.
Optuna's singular names take a dictionary too, as overloads (`trial.SetUserAttr(attrs)`, `storage.SetTrialUserAttr(id,
attrs)`). The result is what one call per entry would leave. The in-memory storage takes its lock once, the journal
appends ordinary one-attribute records in one batch (Python Optuna reads them), and the databases use one
transaction. A finished or unknown trial gets nothing written.

## Thread safety

None of the caches changes what concurrent callers see. A record read from an in-memory storage, through any read, is
never changed afterwards; the samplers' caches are covered by the sampler's lock like the rest of a sampler's state,
and each `SamplerConcurrency.PerWorker` worker has its own. The repository's `docs/THREADING.md` has the details.
