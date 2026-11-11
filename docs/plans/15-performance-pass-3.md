# Plan 15 — third performance pass: speed and memory gaps (2026-11-11)

> **Status: IN PROGRESS.** A four-hour working session, 2026-11-11 05:44 → 09:44 JST, on `main` after `b7ae717`.
> This file is its central record: the rules, the baseline survey, the backlog, the progress log and the results.

## 1. The request

Owner, 2026-11-11: "Now lets do 4 hours of performance impact and gaps, striving to speed up and reduce memory".

## 2. Rules (carried over from plans 13 and 14)

- **Results do not change.** The owner's standing rule is "prefer rustuna unless it affects results"; plan 14 added "each
  affected area must have byte-for-byte parity test coverage". A change ships only when every seeded study gives the
  same trials, bit for bit. A technique that would change results is measured and listed for the owner, not built.
- **Gates for each change:** (1) a unit test pinning the new path to the code it replaces where the replaced code is a
  numeric kernel; (2) the hash probe (36 seeded studies across every sampler family, hashed over every trial's
  parameters and values by their bits) built against a `git archive` snapshot of the starting commit and against the
  tree, with equal lines; (3) the area's existing unit and live-Optuna suites.
- **Measuring on this machine:** every probe opts out of Windows power throttling (EcoQoS moves background console
  processes to the E-cores), is built in Release and prints `IsJITOptimizerDisabled`; before/after runs are
  interleaved because other sessions share the machine; GC counts and allocated bytes are trusted over wall time.

## 3. Baseline survey

Probe `survey` (session scratchpad `4852dc9e`, `survey/Program.cs`): one workload per sampler family on
`InMemoryStorage`, 10 float parameters in [-10, 10] (GP: 6), seed 0, sampler defaults, objective `sum((x - 1)^2)`
or `sum(x^2)` and `sum((x - 2)^2)` for two objectives; BruteForce exhausts a 4,320-point mixed grid. Built in Release
against `b7ae717`, power throttling opted out; the warm repetition (a second study in the same process). "Second
half's share" is the share of the time spent on the second half of the trials: about 0.5 when a trial costs the same
throughout, above that when its cost grows with the history (0.75 for a cost proportional to the history).

| Workload | Trials | Warm time | Second half's share | Allocated | Per trial | gen-0/1/2 | GC pauses |
|---|---|---|---|---|---|---|---|
| random | 10,000 | 0.402 s | 0.53 | 195 MB | 20.0 KB | 10/3/0 | 0.04 s |
| tpe | 1,000 | 0.814 s | 0.71 | 351 MB | 359.4 KB | 19/1/0 | 0.01 s |
| tpe-mv | 1,000 | 0.809 s | 0.71 | 351 MB | 359.4 KB | 19/1/0 | 0.01 s |
| motpe | 400 | 0.501 s | 0.73 | 171 MB | 437.8 KB | 9/1/0 | 0.00 s |
| cmaes | 5,000 | 0.465 s | 0.55 | 217 MB | 44.4 KB | 12/3/0 | 0.05 s |
| cmaes-sep | 5,000 | 0.249 s | 0.53 | 138 MB | 28.3 KB | 7/2/0 | 0.04 s |
| qmc-sobol | 4,000 | 2.900 s | 0.54 | 963 MB | 246.5 KB | 147/115/113 | 0.08 s |
| qmc-halton | 4,000 | 1.192 s | 0.51 | 645 MB | 165.1 KB | 39/38/1 | 0.49 s |
| nsga2 | 5,000 | 0.162 s | 0.49 | 145 MB | 29.7 KB | 8/2/0 | 0.03 s |
| nsga3 | 3,000 | 0.092 s | 0.49 | 89 MB | 30.4 KB | 4/1/0 | 0.01 s |
| sms-emoa | 2,000 | 0.266 s | 0.64 | 275 MB | 140.8 KB | 15/1/0 | 0.02 s |
| age-moea2 | 2,000 | 0.078 s | 0.58 | 65 MB | 33.3 KB | 3/1/0 | 0.01 s |
| moead | 2,000 | 0.320 s | 0.68 | 165 MB | 84.5 KB | 9/8/0 | 0.04 s |
| rvea | 2,000 | 0.057 s | 0.52 | 62 MB | 31.7 KB | 3/0/0 | 0.01 s |
| de | 3,000 | 0.594 s | 0.72 | 333 MB | 113.7 KB | 18/2/0 | 0.05 s |
| pso | 3,000 | 0.651 s | 0.69 | 417 MB | 142.3 KB | 23/2/0 | 0.04 s |
| snes | 3,000 | 0.421 s | 0.72 | 505 MB | 172.4 KB | 28/2/0 | 0.04 s |
| xnes | 2,000 | 0.274 s | 0.66 | 286 MB | 146.4 KB | 15/0/0 | 0.02 s |
| openai-es | 3,000 | 0.453 s | 0.71 | 547 MB | 186.7 KB | 30/1/0 | 0.04 s |
| lm-cma | 3,000 | 0.432 s | 0.67 | 345 MB | 117.8 KB | 19/2/0 | 0.04 s |
| vkd-cma | 2,000 | 0.184 s | 0.63 | 141 MB | 72.2 KB | 7/1/0 | 0.02 s |
| janus | 2,000 | 0.163 s | 0.64 | 216 MB | 110.6 KB | 12/0/0 | 0.02 s |
| gp | 60 | 0.631 s | 0.82 | 425 MB | 7,253.3 KB | 37/29/24 | 0.04 s |
| bruteforce | 4,320 | 12.200 s | — | 13,967 MB | 3,310.7 KB | 779/199/1 | 1.04 s |

Reading: QMC (both scrambled engines) re-did per trial work that depends only on the seed — Sobol even generated
every earlier point — and BruteForce allocates 3.3 MB per trial. TPE, the default sampler, costs 0.8 ms and 360 KB per
trial at 1,000 trials and grows with the history; GP allocates 7 MB per trial with a gen-2 collection every few trials.

## 4. Backlog

Open after plan 14 (its §7):

1. TPE: score candidates as ordered arrays instead of name-keyed dictionaries, drop the copies left in estimator
   construction.
2. NSGA-II: precompute each trial's values and constraints, build parameters only for the selected parents, update the
   joint search space in place.
3. CMA-ES: an in-memory copy instead of the restore after each tell (must serialize to the same pickle).
4. QMC, scrambled Sobol: the point at `sample_id` is found by generating every earlier point, O(n²) per study.
5. NumSharp temporaries left to the finalizer in samplers plan 14 did not trace (GP, DE, PSO, the NES family, …).

## 5. Progress log

- 05:44 — session starts; harness copied from session `5417a82e` (hash probe, proof program, power-throttling opt-out).
- 05:48 — baseline survey (§3).
- 05:55 — **step A, QMC engines.** `ScrambledSobol.Create` / `ScrambledHalton.Create` build an immutable engine per
  `(d, seed)` (the shift and scrambled direction numbers; the Owen digit permutations) and `QMCSampler` keeps it across
  trials. Sobol's point at `sample_id` is now computed directly: `shift XOR` the direction numbers of the bits set in
  `gray(sample_id)`, the same integer state stepping reaches, so the same double. Halton's shuffle arrays are disposed
  when copied out. Warm 4,000 trials: Sobol 2.900 → 0.150 s (963 → 58 MB, gen-2 113 → 0), Halton 1.192 → 0.161 s
  (645 → 58 MB, GC pauses 0.49 → 0.02 s); same hashes. Gates: new `ScrambledSobolTests` (direct point = stepped row
  over 4,100 indexes × 5 `(d, seed)`; `Create`+`Generate` = `Random`; the past-the-end guard) and `ScrambledHaltonTests`
  (engine = the replaced per-call algorithm, written out, over 11 indexes × 4 `(dim, seed)`); QMC / GP / TuRBO / SAASBO
  unit tests 246/246; live QMC and Sobol parity 20/20; hash probe 36/36 identical.
- 06:10 — a kernel bench (`kbench`, a friend assembly) times `ParzenEstimator` construction and `LogPdf` at 100, 1,000
  and 3,000 kernels for 10 floats, a mixed space (floats, log floats, ints) and 10 ints, and hashes every output by its
  bits. Finding: the **stepped (int) branch** was the slow one — 10 ints at 100 kernels took 1.8 ms and 976 KB per call,
  against 0.1 ms and 7 KB for 10 floats — because its three `np.unique`-style dedups sorted with LINQ over up to
  24 × n_kernels grid cells. The sampled profiler could not say where the continuous branch spent its time: a call-free
  loop's time is charged to the next safe point (memset and GC-poll frames with no GC running), so the parts were timed
  on their own and the JIT's code read instead.
- 06:38 — **step B, TPE log density** (`397e60f`). (1) The stepped branch moved to `AddDiscreteLogPdf`: the dedups hash
  (`UniqueInverse`, first-occurrence order, NumPy's `==` grouping: ±0 together, every NaN alone) and every buffer is
  rented. Same values: the unique set and its size are unchanged, the log mass is elementwise and picks its `erf` path
  from the central count alone, and a group's members differ at most in the sign of a zero, which no log mass sees.
  (2) The continuous accumulation (`AccumulateContinuous`) keeps each candidate's sums for a vector-wide kernel block in
  a register across the dimensions and loads through `Vector.LoadUnsafe` at one offset (mu and sigma copied in as grid
  rows; every length checked once up front): the JIT had emitted ~60 instructions per vector step, mostly bounds checks,
  around one division. `-(z*z) / 2` became `* 0.5` (both are the correctly rounded half, the same bits). Same-process A/B,
  best of five: 2.0–2.4× faster, same bits. Per the owner's rule (2026-11-11) the vector path and a separate scalar path
  sit in an explicit if/else. Kernel bench per `LogPdf` call (24 candidates, 10 parameters), base → tree: floats
  K=1,000 868 → 721 µs; mixed K=1,000 1,814 → 959 µs (415 → 20 KB); ints K=1,000 5,293 → 1,846 µs (1.9 MB → 18 KB),
  K=100 1,838 → 1,203 µs (976 → 4 KB); identical hash over all nine configurations. Survey, 1,000 trials, warm:
  `tpe` 1.44 → 1.15 s; new `tpe-int` (10 ints) 4.08 → 2.39 s, 1,301 → 337 MB, gen-2 3 → 0; new `tpe-mixed`
  1.51 → 1.02 s, 532 → 352 MB; same study hashes. Gates: new `TpeDiscreteKernelTests` (stepped `LogPdf` against the
  replaced branch over sorted dedups, both sides of the `erf` switch; `UniqueInverse` against the sorted dedup's grouping
  and representatives; both accumulate branches against the replaced loop; the up-front length check) with the plan 13
  `TpeExactKernelTests`; unit CI filter 2,190/2,190 on net10.0 and net8.0; live TPE / PED-ANOVA / truncnorm parity
  35/35; hash probe 36/36 identical.

- 06:40 — `motpe` profile: Optuna 5 resolves `multivariate=None` to True for one objective and False for several (the
  port does the same), so MOTPE samples each parameter on its own; its cost is in the same kernels. BruteForce (12 s,
  14 GB for 4,320 points) rebuilds its tree from every trial per parameter and per finished trial, which is Optuna's
  design; an incremental tree is possible but was judged too risky for this pass (backlog).
- 06:45 — `gp` at 150 trials: 11.0 s, 37.6 MB allocated per trial, **782 gen-2 collections**; the finalizer thread was
  busy for 12 of 12.5 s (trimming the array pools after every gen-2 collection, and finalizing NumSharp blocks), and
  `LinAlg.SolveTriangular` with a matrix right-hand side solved one column at a time through a captured accessor, two
  arrays per column.
- 07:02 — **step C, GP linear algebra and fit scratch.** (1) `LinAlg.SolveTriangular(matrix)` sweeps all right-hand
  sides at once (`SweepTriangular`: row `i` minus a coefficient times row `j`, vectorized across the right-hand sides,
  with a separate scalar branch per the owner's rule); a right solve works on the transposed block. Every element still
  gets `TriSolveVec`'s sequence. (2) `LinAlg.CholeskySolve(L, B, left)` runs both triangular solves of a Cholesky solve
  on one block, no intermediate matrix and no materialized `Lᵀ` (solving with `Lᵀ` stored upper reads the coefficients
  that sweeping `L` transposed reads, in the same direction); the GP's fit (`C⁻¹`), `Posterior` (`V`) and
  `SolveCholesky` use it. (3) `LinAlg.SolveTriangular(a, b, upper, transposeA)` for vectors reads `a` transposed in
  place; the per-point posterior, `CacheMatrix` and the vector `SolveCholesky` no longer transpose the factor per call.
  (4) `LossAndGrad` keeps its nine n × n buffers on the regressor across the fit's evaluations (`FitScratch`, dropped
  when the fit ends so a cached regressor does not keep them). Survey (base → tree, same hashes): `gp` 150 trials
  15.6 → 13.2 s (machine loaded; 11.0 → 9.7 s after (1) alone), 5,509 → 1,408 MB (−74%), gen-2 collections
  795 → 143; 60 trials 0.71 → 0.56 s, 425 → 142 MB. Gates: `LinAlgTests` +3 (the matrix solve against the replaced
  one-at-a-time code over every triangle, side, size 1–17 and 1–33 right-hand sides; `CholeskySolve` against the two
  spelled-out solves; the transposed vector solve against a materialized transpose), 15/15; GP-family probe (GPSampler
  float, mixed, deterministic, constrained, two objectives; TuRBO; SAASBO-MAP; REMBO) 8/8 identical; hash probe 36/36;
  unit CI filter 2,193/2,193 on net10.0 and net8.0; live GP / terminator / TuRBO / SAASBO / REMBO / LinAlg parity
  70/70. Not found: the source of the NumSharp `UInt32` blocks the GP run left to the finalizer (not `SeedSequence`,
  which keeps a managed pool, nor `randint`, which defaults to int64). Committed `8588074`.
- 07:07 — **step D, the forked regressor.** `Samplers/ModelBased/Gp/GPRegressor.cs` (TuRBO's fit, and the fork's
  posterior) takes step C's changes line for line: the fit's `FitScratch` (released when `FitInner` ends, and by
  `TurboGpFit` after its own L-BFGS-B), `CholeskySolve` for `C⁻¹`, `V` and `SolveCholesky`, the in-place transposed
  vector solve for the per-point posterior, `CacheMatrix` and the vector `SolveCholesky`. GP-family probe with TuRBO at
  120 trials and REMBO at 60 (base → tree): all 8 hashes identical; TuRBO 4.64 → 4.28 s, 1,857 → 1,215 MB; REMBO
  unchanged (its embedded GP does not go through this fit); the GPSampler rows show step C (e.g. 26 trials 52 → 24 MB,
  two objectives 197 → 121 MB). Gates: GP-family, LinAlg and terminator unit tests 266/266 (net10.0); live parity
  70/70. Committed `874c64b`.
- 07:16 — **step E, BruteForce tree rebuilds.** Profile (12.4 s): `PopulateTree` 76% — paths 3.5 s, `AddPath` 3.8 s
  of which `EnumerateCandidates` 1.2 s (Python-faithful decimal arithmetic: a float repr, a decimal parse and a
  correctly rounded conversion per candidate, for every newly expanded node, about a thousand per rebuild). (1) One path
  buffer serves every trial of a rebuild (`AddPath` walks a path once and keeps nothing). (2) `CandidatesFor` caches the
  candidate lists per rebuild, keyed by the progression's exact bits and kind (`ChoicesArgs.Equals` treats -0.0 as 0.0
  and ignores `IsInt`, so it is not the key); a progression without a step throws on every call and is never cached.
  Survey `bruteforce` (4,320 points, all of them): 13.9 → 11.6 s (base runs ranged 13.9–17.6 s on the loaded
  machine), 13,967 → 5,448 MB (−61%), gen-0 collections 779 → 307; same hash. Gates: two new `BruteForceSamplerTests`
  (the cache equals a fresh enumeration by bits and returns the same list again; -0.0 / 0.0 and int / float are
  separate keys; the stepless progression throws twice and caches nothing), BruteForce unit tests 26/26; live
  BruteForce parity 13/13; hash probe 36/36. Left for later: caching each finished trial's path across rebuilds (the
  path of a finished trial never changes; 28% of the time). Committed `6418e0b`.
- 07:25 — **step F, TPE's trial split.** A split bench (`sbench`) on studies of 1,000 and 3,000 completed trials with
  many ties: 175–190 µs and 155 KB per split at 1,000 trials, 705–735 µs and 476 KB at 3,000 — about 15% of a TPE
  trial. (1) `GetInfeasibleTrialScore` returns the empty loop's 0.0 for a trial that records no constraint
  (`ConstrainedOptimization.HasConstraints`: the two tests the dictionary's construction makes) instead of building an
  empty constraints dictionary per trial per split, two thirds of the bytes. (2) The partition lists are pre-sized
  and both sides built at their final sizes. (3) The single-objective sort sorts (value, position) structs that
  implement `IComparable<T>` (the order of LINQ's stable `OrderBy` / `OrderByDescending`, with `double.CompareTo`), so
  the generic sort calls `CompareTo` directly; a struct `IComparer<int>` passed to `Span.Sort` is boxed and gained
  nothing. (4) Both sides are sorted by number on primitive keys (`Array.Sort(int[], FrozenTrial[])`; numbers are
  unique). Split bench: 130 µs / 52 KB at 1,000 trials, 553–576 µs / 154 KB at 3,000; same split hash. Survey,
  interleaved: `tpe` 1,000 trials 0.94–0.99 → 0.86–0.88 s (all of plan 15 against the base), 358 → 300 KB per trial;
  `motpe` 400 trials time unchanged, 440 → 405 KB per trial. Gates: new `TpeTrialSplitSortTests` (the split against
  the LINQ code written out, shuffled histories with ties, both zeros, both infinities and NaN, both directions, every
  seventh group size; constraints in both formats still score infeasible); TPE, constraint and split unit tests
  117/117; hash probe 36/36. Committed `59b69dd`.
- 07:29 — **step G, Parzen estimator construction.** Each numerical parameter computed its centers (logs, for the log
  kinds) and widths into arrays of their own and then copied both into arrays one longer for the prior kernel. The
  centers are now written straight into the prior-extended array and the widths computed into its slots:
  `ComputeSigmas`, `SortedRawSigmas` and `BinnedRawSigmas` gain span-filling cores (the binned one returns whether it
  wrote, and every refusing check runs before its first write), `StableArgsort` a span overload; the array forms stay
  as thin wrappers for `TpeBinnedSigmasTests`. Kernel bench: estimator build K=1,000 259–275 → 181 KB, K=3,000
  775–822 → 540 KB; output hash unchanged. Survey, all of plan 15 against the base: `tpe` 1,000 trials
  0.92 → 0.78 s, 358 → 261 KB per trial; `tpe-mixed` 1.32 → 1.05 s, 544 → 258 KB per trial; same hashes. Gates:
  TPE, Parzen and PED-ANOVA unit tests 122/122; kernel-bench hash; hash probe 36/36. Committed `eaed03a`. Unit CI
  filter through step G: 2,199/2,199 (1 skipped) on net10.0 and net8.0; documentation gate (`check_documentation.cs`,
  5,840 declarations) 0 issues; Core built with the documentation warnings as errors, both TFMs: 0 errors.
- 07:45 — **step H, the evolution-strategy samplers' attribute keys.** SNES, xNES and OpenAI-ES spent most of their
  per-trial allocation (147–187 KB) rebuilding one string: `AttrKeyGeneration` was an expression-bodied
  `AttrPrefix + "generation"` read for every trial of the history by `GetSolutionTrials` and `CountRunningMembers` on
  every ask. The two keys are now built once per sampler (`AttrPrefix` is a constant per sampler type), and the
  completed-trials list is sized for the history. DE, PSO, LM-CMA and VkD-CMA concatenate `const` prefixes, which
  the compiler folds; CMA-ES hands its keys to its trial history once, and the GA samplers read theirs once per newly
  folded trial: none of those allocate per scan. Survey (machine loaded by other sessions, base and tree back to
  back): SNES 3,000 trials 1.06 → 0.89 s, 173 → 59 KB per trial; xNES 2,000 0.90 → 0.86 s, 148 → 73 KB; OpenAI-ES
  3,000 1.08 → 0.89 s, 188 → 61 KB; same hashes. Gates: ES, parallel-read and nJobs sampler tests 248/248; hash
  probe 36/36 (SNES, xNES and OpenAI-ES studies included).

## 6. Findings and notes for the owner

- **NumSharp `NDIter` / `NDExpr` / `np.evaluate` (asked 2026-11-11).** `NDExpr` compiles an elementwise tree over
  `NDIter` operands into one IL kernel with a SIMD body (numexpr-style `np.evaluate`, no intermediate arrays), and its
  generator keeps multiplications and additions unfused, so an elementwise tree would keep its bits. No hot path in
  Core runs on `NDArray` arithmetic any more, though: plans 13 and 14 moved TPE, the RNG draws and the samplers' kernels
  to loops over `double[]`; what still goes through NumSharp is LAPACK (`LinAlg`, OpenBLAS), the MT19937 generator and
  a few small array steps (CMA-ES, NSGA-III); `Hypervolume.cs`'s `np.` mentions are comments, and the GP stack, DE, PSO
  and the NES family are plain managed code (151 `new double[]` sites in `Gp/`, one NumSharp line per ES/DE/PSO file).
  So there is no chain of `NDArray` temporaries for `np.evaluate` to fuse today. A reduction tree (`NDExpr.Sum`) would
  need checking against NumPy's pairwise order before any use.
- **QA-84 (parity, not performance; not changed).** TPE sums each candidate's continuous log terms over the
  dimensions left to right, `((0 + t0) + t1) + …`; Optuna's `.sum(axis=-1)` runs over a C-contiguous last axis, which
  NumPy sums pairwise (eight accumulators) from eight elements on. Measured with Optuna 5.0.0's own `_truncnorm.logpdf`
  and NumPy 2.4.2: the pairwise model matches NumPy in 7,200 of 7,200 (candidate, kernel) entries for 8, 10 and 16
  dimensions, the left-to-right sum differs in 31–54% of them. A ULP-level difference in the log densities of studies
  with eight or more continuous parameters, the class of the documented `erf` divergence; fixing it changes results, so
  it is recorded with an `[OpenBugs]` repro rather than changed here.
