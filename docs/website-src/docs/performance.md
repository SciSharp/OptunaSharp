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

## Switches that change results

| Switch | What it does | Seeded runs still match Python? | Can Python workers share the storage? |
|---|---|---|---|
| `new CmaEsSampler(..., reuseToldOptimizer: true)` | Keeps the optimizer that just told a generation instead of restoring the state it stored; about 20-25 % faster on 40 parameters | No: from about the third generation (Optuna's pickle stores only the covariance's upper triangle, so a restored optimizer differs in the last bits) | Yes: the state is still stored every generation exactly as Optuna stores it |
| `TrialExecutorOptions.BlasThreads = 1` with `nJobs: 1` | Runs OpenBLAS on one thread; CMA-ES was 12 % faster at 40 parameters and 24-35 % faster at 100 | Up to about 40 parameters, yes; at 100 parameters no, because OpenBLAS's threaded kernels sum in another order (NumPy's results depend on the thread count the same way) | Yes |

With `nJobs` above 1 OptunaSharp already runs OpenBLAS on one thread automatically: parallel runs reseed every trial,
so they are not reproducible anyway, and several workers in linear algebra at once would otherwise oversubscribe the
cores (see [Easy Parallelization](key-features/parallelization.md)).

## Thread safety

None of the caches changes what concurrent callers see. A record read from an in-memory storage, through any read, is
never changed afterwards; the samplers' caches are covered by the sampler's lock like the rest of a sampler's state,
and each `SamplerConcurrency.PerWorker` worker has its own. The repository's `docs/THREADING.md` has the details.
