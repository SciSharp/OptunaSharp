# Plan 15 — third performance pass: speed and memory gaps (2026-11-11)

> **Status: DONE.** A four-hour working session, 2026-11-11 05:44 → 09:44 JST, on `main` after `b7ae717`: seventeen
> steps and one regression fix (`fb18c44`..`17a3448`), every seeded result unchanged; QA-84 recorded (`96abe26`);
> a Windows race in the journal file lock found (§6). Results in §7. The owner's decisions of 2026-11-12 on the two
> open items (the lock race fixed `07af6a7`, QA-84 fixed through NumSharp `cb512df`, NumSharp's two costs pinned as
> OpenBugs on its journey5 `e9c38c2f`) are in §8; that work is on the local branch `journey1`. **Closed 2026-11-12**
> (owner: "document the remaining and stop"): QA-84's cost stays until NumSharp fixes its pins; the remaining work is
> listed in §9.
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

Open after plan 14 (its §7), with what this pass did about each:

1. TPE: score candidates as ordered arrays instead of name-keyed dictionaries, drop the copies left in estimator
   construction. — **The copies are gone (step G).** Ordered-array scoring was not attempted: the kernels (steps B, F)
   were the larger costs.
2. NSGA-II: precompute each trial's values and constraints, build parameters only for the selected parents, update the
   joint search space in place. — Open (NSGA-II costs 30 µs per trial).
3. CMA-ES: an in-memory copy instead of the restore after each tell (must serialize to the same pickle). — Open.
4. QMC, scrambled Sobol: the point at `sample_id` is found by generating every earlier point, O(n²) per study. —
   **Done (step A).**
5. NumSharp temporaries left to the finalizer in samplers plan 14 did not trace (GP, DE, PSO, the NES family, …). —
   **Traced:** GP, DE, PSO and the NES family are plain managed code; their per-trial memory was LOH churn (GP,
   steps C and D) and history scans (steps H and I). Open: an unidentified source of NumSharp `UInt32` blocks the GP
   run leaves to the finalizer. Ruled out since: the optimizers' entropy-seeded `RandomState()` (its `SeedSequence`
   state is not an `NDArray` left behind: ~6.2 KB managed either way, step M's note). The per-ask draws that dropped
   `NDArray`s (DE, MOEA/D, CMA-ES and others; the allocation sampler listed `UnmanagedStorage`, `ArraySlice<T>` and
   `Disposer<T>` for them) read through `NumPyDraws` since steps P and R; MOEA/D's gen-1 collections fell 8 → 0.

New, found by this pass and left open:

6. BruteForce: cache each finished trial's path across rebuilds (a finished trial's path never changes; building
   paths is 28% of its time). The per-rebuild categorical cache (`catCache`, first trial seen per name after
   filtering) has to be reproduced exactly.
7. DE: it deserializes its JSON optimizer state on every ask; a cached copy would need cloning before each mutation.
   — **Done (steps M, N, O)** without cloning: the cache holds the parsed *state*, from which every `FromState` builds a
   new optimizer by copying; done for DE, LM-CMA, PSO, VkD-CMA, SNES, xNES and OpenAI-ES.
8. Multi-objective split (MOTPE): `lvals` through nested LINQ, `UniqueRows` with a delegate sort. The sort is unstable
   and its representative among equal rows (±0) reaches the hypervolume arithmetic, so changing it is not provably
   bit-identical; the LINQ part is.
9. Parity, not performance: QA-84 (§6).
10. The study core's per-suggest cost (random search, 20 KB per trial): about 40 % is the in-memory storage's
    copy-on-write (plan 13 S4) copying a running trial's parameter and distribution dictionaries on each suggest after
    the sampler has read the record, and 20 % is closures and delegates (`Locked(() => …)`, `Trial.Suggest`). Every
    sampler pays it; changing the copy-on-write needs its own correctness argument.
11. TPE's remaining bytes are the Parzen estimators' own per-kernel arrays (66 % of a TPE trial's allocation is
    `double[]`); pooling them means reworking the estimator's ownership, not a local change.
12. MOEA/D scans every completed trial's `SystemAttrs` by string key twice per ask (the offspring count and the new
    offspring); an incremental count would make it O(new trials).

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
- 07:36 — **step H, the evolution-strategy samplers' attribute keys.** SNES, xNES and OpenAI-ES spent most of their
  per-trial allocation (147–187 KB) rebuilding one string: `AttrKeyGeneration` was an expression-bodied
  `AttrPrefix + "generation"` read for every trial of the history by `GetSolutionTrials` and `CountRunningMembers` on
  every ask. The two keys are now built once per sampler (`AttrPrefix` is a constant per sampler type), and the
  completed-trials list is sized for the history. DE, PSO, LM-CMA and VkD-CMA concatenate `const` prefixes, which
  the compiler folds; CMA-ES hands its keys to its trial history once, and the GA samplers read theirs once per newly
  folded trial: none of those allocate per scan. Survey (machine loaded by other sessions, base and tree back to
  back): SNES 3,000 trials 1.06 → 0.89 s, 173 → 59 KB per trial; xNES 2,000 0.90 → 0.86 s, 148 → 73 KB; OpenAI-ES
  3,000 1.08 → 0.89 s, 188 → 61 KB; same hashes. Gates: ES, parallel-read and nJobs sampler tests 248/248; hash
  probe 36/36 (SNES, xNES and OpenAI-ES studies included). Committed `7be356d`.
- 07:39 — **step I, the other samplers' history lists.** DE, PSO, LM-CMA, VkD-CMA, JANUS and JANUS-DE collect their
  completed trials into a list grown from empty on every ask; it is now sized for the history (the same contents in
  the same order). Survey, back to back, same hashes: DE 115 → 91 KB per trial, PSO 143 → 120, LM-CMA 119 → 95,
  VkD-CMA 73 → 60, JANUS 112 → 98; wall times within the loaded machine's noise. DE's remaining per-trial cost is
  deserializing its JSON optimizer state on every ask (backlog). Gates: their unit tests 103/103; hash probe 36/36.
  Committed `a1fbe9a`.
- 07:45 — **A regression in step F, found and fixed.** Re-measuring the split bench for a planned step J (the
  multi-objective `lvals` as plain loops) showed the **two-objective split 10–20% slower than the base** — step F's
  own measurements had shown it too, and I had put it down to noise. Cause: the multi-objective path builds both sides
  in the history's order, so step F's `SortByNumber` copied two already-ordered lists out and back where the old
  in-place delegate sort barely moved anything. Fix: `SortByNumber` returns at once when the list is already strictly
  ascending by number (sorting would leave every trial in place). Best of five alternating runs, base → fixed:
  one objective 165.9 → 125.5 µs (1,000 trials), 693.6 → 551.7 µs (3,000); two objectives 540.3 → 489.8 µs,
  1,505.2 → 1,320.1 µs; same split hash. Step J itself was dropped: the loop version measured ~250 µs slower at
  3,000 trials than the LINQ it replaced, for a reason not found (the property it reads is a plain field). Lesson:
  re-run a step's A/B in both orders before calling a difference noise. Gates: split, TPE and multi-objective unit
  tests 134/134; hash probe 36/36.
- 08:00 — checkpoint: §7 written for `d521e16` (`bd2b9f4`); work continues. A new probe, an **in-process allocation
  sampler** (the runtime's `GCAllocationTick` events counted by type through an `EventListener`, `ALLOC_TICKS=1`,
  scratch `common/AllocTicks.cs`), showed where each workload's bytes go. Two sinks shared by several samplers: copies of
  the whole history (`FrozenTrial[]`: DE 40 %, MOEA/D 13 %, SMS-EMOA 13 %, TPE 12 %) and dictionaries over every
  completed trial rebuilt per ask (`Dictionary<int, FrozenTrial>`: SMS-EMOA 37 %, MOEA/D 38 %, plus their buckets).
- 08:12 — **step K, TPE's history reads** (`439c26c`). TPE read the filtered history twice per trial in multivariate
  mode and twice per *parameter* in independent mode (MOTPE's default), a fresh copy each time, and copied both split
  groups per parameter. `Study.GetTrialsCachedReadOnly(states)` shares one filtered list per trial snapshot and state set
  (memo keyed by the snapshot's identity and a state bit mask; `CachedAllTrials` became a property whose setter drops
  it, so a tell drops both); `Study.FilterByState` filters by bit mask into a list allocated at its final size;
  `TpeSplitCache.SplitShared` returns the cached groups without copies. Same history: both reads filter the same snapshot,
  and a handed-out record's state never changes (the storages change a shallow copy, plan 13 S4). MOTPE 386.6 → 353.3 KB
  per trial (base 440.3), TPE about −4 KB per trial: .NET 10's `Where().ToList()` already avoids most growth garbage, so
  the gain is the sharing. Gates: new `StudyTrialFilterTests` (13 state sets incl. values outside the enum, the memo's
  sharing and invalidation, `SplitShared` = `Split`); unit tests 526/526; hash probe 36/36; live TPE / multi-objective
  parity 74 passed, 1 skipped.
- 08:24 — **step L, steady-state population lookups** (`eca033a`). `Internal/TrialsByNumber`: a binary search over the
  number-ordered history (one allocation-free order check per ask; a list out of order gets the replaced dictionary
  itself, later record winning). SMS-EMOA and MOEA/D keep their populations as trial numbers instead of storage ids (the
  same records within a study; a recreated study is discarded first by the anchor check), and MOEA/D's new-offspring
  `OrderBy` runs only when the history is not ascending. SMS-EMOA 141.8 → 80.9 KB per trial, MOEA/D 85.5 → 54.8; cold
  runs 15–20 % faster, warm runs 5–12 %. **A trap found on the way:** a first version (hand-written loops, the list
  walked through `IReadOnlyList`) made MOEA/D's 2,000-trial run 0.54–0.57 s against 0.36–0.49 s, while 4,000 trials
  were even and `DOTNET_TC_QuickJitForLoops=0` made the tree equal or faster on every repetition: new loops run as
  unoptimized tier-0 code through the first few thousand asks, while the LINQ they replace ships precompiled
  (ReadyToRun). The final form keeps LINQ's `Where().ToList()` and walks a span in methods marked
  `AggressiveOptimization`. Probably also the reason step J measured slower. Gates: new `TrialsByNumberTests`; unit tests
  660/660 and 35/35 (recreated studies); hash probe 36/36.
- 08:29 — **step M, DE's optimizer state** (`8c1a1f7`, backlog 7). DE parsed its stored JSON population on every ask;
  the text changes once per generation. A one-entry cache keyed by the whole text, safe because `De.FromState` copies
  every array out of the state. DE 267 → 241 MB (base 335); time within noise (the parse was cheap). Measured and
  dropped on the way: the optimizers' OS-entropy `np.random.RandomState()` before the reseed that precedes every draw
  costs ~13 µs against 6–13 µs seeded, both ~6.2 KB, same stream after the reseed (scratch `rsbench`): ~2 % of a DE ask.
  Gates: hash probe 36/36; DE / JANUS-DE / nJobs / reuse tests 179/179.
- 08:33 — **step N, the same cache for LM-CMA, PSO and VkD-CMA** (`dbe86e5`), each `FromState` checked for aliasing.
  PSO 351 → 280 MB, LM-CMA 279 → 220, VkD-CMA 117 → 103 (base 419 / 348 / 143). Gates: hash probe 36/36; their tests
  254/254.
- 08:36 — **step O, the same cache for SNES, xNES and OpenAI-ES** (`e889d56`). SNES 174 → 167 MB, xNES 143 → 125,
  OpenAI-ES 178 → 169 (small states; step H was the large cut). Gates: hash probe 36/36; their tests 222/222;
  documentation gate (5,855 declarations) 0 issues, Core's strict documentation build 0 errors.
- 08:41 — **step P, draws that left arrays to the finalizer** (`bbefda2`). 21 per-ask or per-parameter sites still
  cast a NumSharp draw (`(int)rng.randint(…)`, `(double)rng.rand()`), each dropping a finalizable `NDArray`; they read
  through plan 14's `NumPyDraws` now (`RandInt` is the same call with the array disposed, `Rand` is pinned to `rand()`
  by `NumPyDrawsTests`): DE and `De`, LM-CMA, MOEA/D and its offspring, BruteForce, Grid, GPSampler, CMA-ES, the ES
  base class and both GP search spaces. Left: NSGA-III-CPT, SAASBO, `OptunaSearchCV` (not hot). Gates: hash probe
  36/36; GP-family probe 8/8; their unit tests 460/460; live Grid and BruteForce parity 24/24.
- 08:48 — **step Q, the state filter compiled optimized** (`cd33077`). The final survey read SMS-EMOA and MOEA/D
  slower at the second repetition (0.37 → 0.50 s and 0.37 → 0.45 s, machine loaded). Four repetitions per process,
  alternating: the tree is ahead from the cold run on (SMS-EMOA cold + second 1.20 / 1.16 → 1.04 / 1.01 s, all four
  1.76 / 1.69 → 1.55 / 1.52 s; MOEA/D cold + second 1.19 → 0.94 s); only the second repetition is about 10 % behind
  while the tree's code tiers up. `Study.FilterByState` (step K) now indexes a span and carries
  `AggressiveOptimization`, which took SMS-EMOA's cold run from 0.56–0.58 to 0.53–0.54 s; the second repetition's gap
  stayed, so it comes from tiering elsewhere. Gates: hash probe 36/36; 630/630 unit tests.
- 08:55 — **step R, Gaussian draws disposed** (`17a3448`), the other half of step P: `NumPyDraws.Randn` is the same
  `rng.randn(n)` call with its array disposed, now used by CMA-ES and SepCMA (every sampled point), VkD-CMA, the ES
  noise matrix, JANUS, SAASBO's NUTS, REMBO and TuRBO / TuRBO-m (10 sites; JANUS-MOO draws through a NumSharp
  `Generator` and stays as it was). Measured on CMA-ES 5,000 trials: no visible change in bytes, collections (12/3/0)
  or time; disposing spares only the finalizer's work, which never showed as collections here. Kept as hygiene.
  Gates (probe DLLs checked fresh after a first build failed on the JANUS-MOO site): hash probe 36/36; GP-family probe
  8/8; their unit tests 343/343.

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
  it is recorded with an `[OpenBugs]` repro rather than changed here (`96abe26`). Fixed 2026-11-12 (`cb512df`, §8).
- **An allocation sampler beats the CPU profiler for finding memory sinks.** An `EventListener` on
  `Microsoft-Windows-DotNETRuntime` with the GC keyword at Verbose receives `GCAllocationTick` (one per ~100 KB, naming
  the type of the object that crossed the threshold); counted per type over a run, the shares are proportional to the
  bytes. It found steps K, L and P in minutes, where the sampled CPU profiler charges call-free loops to safe points.
  No stacks (the callback runs on the dispatch thread), but the types point at the code.
- **Hand-written loops replacing LINQ can be slower for the first few thousand calls.** Framework LINQ ships
  precompiled (ReadyToRun) and optimized; a new loop in OptunaSharp starts as unoptimized tier-0 code, and with tiered
  PGO it can take thousands of calls to reach its final tier. A 2,000-trial study can end before that: step L's first
  version measured 40 % slower on MOEA/D's 2,000-trial run, even at 4,000 trials, and equal or faster with
  `DOTNET_TC_QuickJitForLoops=0`. Compare a rewrite with warm-up included (rep 1) and warm (rep 3), and when it must be
  a loop, keep it small and mark it `AggressiveOptimization`, or keep the LINQ.
- **The entropy-seeded generators are not the `UInt32` blocks.** The optimizers' `np.random.RandomState()` (as cmaes'
  `__setstate__` does) before the reseed that precedes every draw costs ~13 µs against 6–13 µs for a seeded one, and
  both allocate ~6.2 KB managed; the stream after the reseed is identical (scratch `rsbench`). Not changed.
- **A journal-lock flake, cause found, not changed (an owner decision).** Two of about eight full net10.0 runs of
  this pass, both on a machine loaded by other test processes, failed
  `JournalConcurrencyTests.Lock_WithoutAGracePeriod_WaitsForALeftBehindLockUntilItIsRemoved` with
  `UnauthorizedAccessException: Access to the path '…\optunasharp-journal-cc-….log.lock' is denied` (once at
  `d521e16`, once at `cd33077`); the test alone, its class three times in a row (21/21 each) and the other runs pass.
  Nothing in this pass touches the journal. Cause: `JournalFileOpenLock.Acquire` creates the lock file with
  `FileMode.CreateNew` and treats only an "already exists" `IOException` as contention; on Windows, creating a file
  whose deletion is pending (the holder deleted it while another handle, e.g. a waiter's `GetLastWriteTimeUtc`, is still
  open) fails with ACCESS_DENIED, which .NET raises as `UnauthorizedAccessException`, so the waiter fails instead of
  retrying. It is a faithful port: Optuna's `os.open(O_CREAT | O_EXCL)` raises `PermissionError` in the same window
  and catches only `FileExistsError`. Treating ACCESS_DENIED on an existing-or-deleting lock file as contention would
  depart from Optuna to fix a Windows race; recorded for the owner. Decided and fixed 2026-11-12 (`07af6a7`, §8).

## 7. Results

Final survey, every workload, base `b7ae717` → tree `cd33077` (all of plan 15 but step R), back to back per workload,
warm repetition (the second study in the process), the same probe and settings as §3. The machine was shared with
other sessions and, for this run, with the full unit suites, so allocations and collections are the dependable columns;
times are compared pairwise. Every study hash is the same. Step R (`17a3448`, Gaussian draws disposed) was measured on
its own: no visible change in CMA-ES's bytes, collections or time, same hashes.

| Workload | Trials | Warm time | Allocated | Per trial | gen-0/1/2 | GC pauses | Hash |
|---|---|---|---|---|---|---|---|
| random | 10000 | 0.335 → 0.347 s | 192 → 190 MB | 19.7 → 19.5 KB | 10/3/0 → 10/3/0 | 0.05 → 0.05 s | same |
| tpe | 1000 | 0.909 → 0.753 s | 351 → 252 MB | 359.4 → 258.0 KB | 19/1/0 → 14/1/0 | 0.01 → 0.01 s | same |
| tpe-int | 1000 | 2.915 → 1.382 s | 1,301 → 242 MB | 1,332.2 → 247.8 KB | 72/2/0 → 13/1/0 | 0.04 → 0.01 s | same |
| tpe-mixed | 1000 | 1.260 → 0.877 s | 532 → 249 MB | 544.8 → 255.0 KB | 29/2/0 → 13/1/0 | 0.02 → 0.01 s | same |
| motpe | 400 | 0.522 → 0.543 s | 172 → 138 MB | 440.3 → 353.3 KB | 9/1/0 → 7/1/0 | 0.00 → 0.00 s | same |
| cmaes | 5000 | 0.757 → 0.800 s | 223 → 223 MB | 45.7 → 45.7 KB | 12/3/0 → 12/3/0 | 0.05 → 0.05 s | same |
| cmaes-sep | 5000 | 0.649 → 0.497 s | 143 → 143 MB | 29.3 → 29.3 KB | 7/2/0 → 7/2/0 | 0.04 → 0.03 s | same |
| qmc-sobol | 4000 | 2.530 → 0.136 s | 962 → 58 MB | 246.3 → 14.8 KB | 158/128/127 → 3/1/0 | 0.15 → 0.01 s | same |
| qmc-halton | 4000 | 1.283 → 0.140 s | 644 → 58 MB | 164.9 → 14.8 KB | 39/38/2 → 3/1/0 | 0.59 → 0.01 s | same |
| nsga2 | 5000 | 0.490 → 0.561 s | 158 → 162 MB | 32.4 → 33.2 KB | 8/2/0 → 9/2/0 | 0.03 → 0.03 s | same |
| nsga3 | 3000 | 0.316 → 0.341 s | 101 → 101 MB | 34.5 → 34.5 KB | 5/0/0 → 5/0/0 | 0.01 → 0.01 s | same |
| sms-emoa | 2000 | 0.412 → 0.431 s | 276 → 159 MB | 141.3 → 81.4 KB | 15/1/0 → 8/1/0 | 0.02 → 0.01 s | same |
| age-moea2 | 2000 | 0.277 → 0.249 s | 72 → 72 MB | 36.9 → 36.9 KB | 4/2/0 → 4/2/0 | 0.02 → 0.02 s | same |
| moead | 2000 | 0.380 → 0.435 s | 167 → 95 MB | 85.5 → 48.6 KB | 9/8/0 → 5/0/0 | 0.04 → 0.01 s | same |
| rvea | 2000 | 0.186 → 0.182 s | 68 → 68 MB | 34.8 → 34.8 KB | 3/0/0 → 3/0/0 | 0.01 → 0.01 s | same |
| de | 3000 | 1.059 → 1.010 s | 336 → 241 MB | 114.7 → 82.3 KB | 18/3/0 → 13/2/0 | 0.05 → 0.03 s | same |
| pso | 3000 | 1.003 → 1.032 s | 419 → 280 MB | 143.0 → 95.6 KB | 24/4/1 → 15/2/0 | 0.04 → 0.04 s | same |
| snes | 3000 | 0.870 → 0.650 s | 508 → 167 MB | 173.4 → 57.0 KB | 28/2/0 → 9/1/0 | 0.04 → 0.03 s | same |
| xnes | 2000 | 0.787 → 0.603 s | 288 → 125 MB | 147.5 → 64.0 KB | 16/1/0 → 6/0/0 | 0.03 → 0.01 s | same |
| openai-es | 3000 | 1.065 → 0.707 s | 550 → 169 MB | 187.7 → 57.7 KB | 30/2/0 → 9/1/0 | 0.04 → 0.03 s | same |
| lm-cma | 3000 | 1.135 → 0.920 s | 348 → 220 MB | 118.8 → 75.1 KB | 19/2/0 → 12/2/0 | 0.04 → 0.03 s | same |
| vkd-cma | 2000 | 0.651 → 0.466 s | 143 → 103 MB | 73.2 → 52.7 KB | 8/1/0 → 5/1/0 | 0.02 → 0.01 s | same |
| janus | 2000 | 0.487 → 0.476 s | 218 → 192 MB | 111.6 → 98.3 KB | 12/0/0 → 10/0/0 | 0.02 → 0.02 s | same |
| gp | 60 | 0.629 → 0.521 s | 425 → 137 MB | 7,253.3 → 2,338.1 KB | 38/33/24 → 21/20/17 | 0.02 → 0.01 s | same |
| bruteforce | 4320 | 12.291 → 8.742 s | 13,965 → 5,446 MB | 3,310.2 → 1,290.9 KB | 779/199/1 → 304/62/1 | 0.98 → 0.38 s | same |

- `sms-emoa` and `moead` read slower here because the column times the second repetition, the one in which the tree's
  code is still tiering up (step Q). Over four repetitions per process the tree is ahead from the cold run on:
  SMS-EMOA all four 1.76 / 1.69 → 1.55 / 1.52 s, MOEA/D cold + second 1.19 → 0.94 s.
- `motpe`: at `d521e16` its time moved within noise (six runs per side in both orders: base best 0.517 s, tree best
  0.474 s); step K then cut its allocation 386.6 → 353.3 KB per trial.
- Larger runs measured during the steps: `gp` 150 trials 15.6 → 13.2 s, 5,509 → 1,408 MB, gen-2 collections
  795 → 143 (step C); TuRBO 120 trials 4.64 → 4.28 s, 1,857 → 1,215 MB (step D); the TPE split at 3,000 trials
  693.6 → 551.7 µs (one objective) and 1,505.2 → 1,320.1 µs (two objectives), best of five (step F and its fix).
- `moead`'s gen-1 collections fell 8 → 0 with step P: its per-parameter mutation draws had left one
  finalizable array each to the finalizer, which promoted them past gen 0.
- Unchanged by design: random, CMA-ES, NSGA-II/III, AGE-MOEA-2 and RVEA kept their per-trial bytes (CMA-ES and the
  GA samplers were already 30–45 KB per trial after plans 13 and 14); their rows move within noise, allocations
  included (the base itself measured 158–161 MB for `nsga2` across this pass's surveys).

Gates over the final tree `17a3448`: unit CI filter 2,209 passed, 1 skipped, 0 failed on net10.0 and on net8.0.
Earlier snapshots: `dbe86e5` and `bbefda2` 2,209 passed, 1 skipped, 0 failed on both TFMs; `cd33077` net10.0 2,208
passed, 1 skipped, 1 failed (the §6 journal-lock flake, under three concurrent test and survey processes), net8.0
2,209 passed, 1 skipped, 0 failed. Documentation gate (5,855 declarations) 0 issues and Core built with the
documentation warnings as errors 0 errors. Hash probe 36/36 and GP-family probe 8/8 identical to the `b7ae717` snapshot after every step.
Live-Optuna suites of the touched areas at their steps: TPE / PED-ANOVA / truncnorm 35, GP family 70, BruteForce 13,
QMC 20, ES / JANUS / VkD-CMA reference parity 39 + 6 skipped without the JANUS reference checkout, TPE /
multi-objective 74 + 1 skipped (step K), Grid and BruteForce 24 (step P).

## 8. Follow-ups: the owner's decisions (2026-11-12)

Asked through design pages after the pass; the work lives on the local branch `journey1` (the owner's plan: work there,
reset `main` to `origin/main`; nothing pushed).

- **Journal-lock race: "Test fix + bounded retry" (`07af6a7`).** The owner first asked what holds the file and whether
  anything but a retry would do. Measured: nothing holds it. A lock path deleted IN PLACE stays "delete pending" while
  the deleting handle lives, and an exclusive create of a pending name is ACCESS_DENIED; releases never do that (C# and
  Python Optuna rename the lock away, then unlink it), the flaky test did (`File.Delete` of its stale lock). Two threads
  taking one lock 40,000 times each: in-place delete 5,070 denials idle and 14,277 under load, rename-then-delete 0
  and 0. Fix: the test removes its stale lock by rename-then-delete, and on Windows `JournalFileOpenLock.Acquire` waits
  out ACCESS_DENIED like contention while the denials run unbroken for at most 1 s
  (`JournalFileLockProtocol.MaxDeniedStreakMilliseconds`), then rethrows; off Windows unchanged. A documented departure
  from Optuna's error path (its waiters raise `PermissionError` there); new tests `Lock_DeletedInPlace_IsWaitedOutOnWindows`
  (`[Misaligned]`) and `Lock_PersistentlyDenied_IsReportedAfterTheBound`. Journal unit tests 98/98.
- **QA-84: "Call NumSharp directly" (`cb512df`).** The owner asked whether NumSharp would do instead of porting
  NumPy's order: it does. NumSharp's `np.sum(axis=-1)` equals NumPy 2.4.2 at every row length 1–300 (the recursive
  split past 128 included), but its pairwise kernel is private, so the TPE kernel now stores every term in NumPy's
  `(n_samples, n_kernels, n_cont)` layout (rented) and calls `np.sum(terms, axis: -1, @out: sums)` on the wrapped
  buffers. Seeded results: the 36-study hash probe and the survey's TPE studies unchanged (none samples eight or more
  continuous dimensions jointly). Cost, 1,000 trials: `tpe` 0.75–0.81 → 1.20–1.34 s (+60–66%), `tpe-mixed` +27–41%,
  `motpe` +39% (one dimension per call), `tpe-int` unchanged; allocation +6–25 MB. Not built, the owner's call: below
  eight continuous dimensions NumPy's order IS left to right, so a bit-identical in-register path could skip NumSharp
  there (MOTPE's whole +39%).
- **NumSharp OpenBugs for that cost ("create OpenBugs at journey5 … for the slowness or memory overallocation",
  NumSharp `e9c38c2f`, `test/NumSharp.Tests/OpenBugs.Consumers.OptunaSharp.cs`).** Memory: nothing to pin; the call
  allocates a constant 1.7–1.9 KB managed per call (the earlier "output-proportional garbage" was a cold-JIT
  measurement). Two speed pins, same-process ratios with their batches alternating:
  - `SumOverShortRows_CostsAFewLongRowSums`: 2,400 rows of 10 against the same values as 24 rows of 1,000, limit 5.0;
    NumSharp 6.1–7.4, NumPy 3.3–3.5. NumSharp pays 17.6 ns per output row (NumPy 7.9) through `ForEach`'s one
    inner-loop call per row, though its elements are cheaper (0.04 against 0.25 ns).
  - `SumIntoOut_CostsAboutAFreshSum`: `out=` against the fresh call at `[24, 3000, 1]`, limit 1.25; NumSharp
    1.84–2.19, NumPy 1.02–1.04: `ReduceIntoOut` computes a fresh result and copies it into `out`.
  - Calibration traps: a flat-sum reference moved with the process (NumSharp's flat sum is 4–5 µs in the test host,
    7.1–7.5 µs in a console probe); timed one route after the other, a hybrid CPU's slow stretch let the first version
    pass with the bug present (alternating batches fixed it); the `out=` pin is Release-only (Debug 1.05–1.90).
- **QA-84's cost: "Keep; wait for NumSharp" (second round, same day).** Offered: a bit-identical in-register sum below
  eight continuous dimensions (NumPy's own order there is left to right from `-0.0`, and the reduction's identity add
  makes it equal to the old sum; `cb512df`'s tests already assert that), a full port of NumPy's order, or keeping it.
  Kept: TPE keeps calling `np.sum(axis: -1, out:)` for every call, so Optuna's default independent TPE, which sums ONE
  continuous dimension per call through NumSharp's `out=` path, stays 27–66% slower until NumSharp fixes the two pins.
  Even fixed, a NumSharp call keeps a fixed per-call cost and the stored terms, so the shortcut stays the way to
  recover all of it (§9, item 1).
- **Next performance work: "document the remaining and stop".** No further steps; §9 lists what is left.

## 9. Remaining work (documented at the owner's request, 2026-11-12)

Nothing below is scheduled. Each item names its measured cost and the rule a change must keep; every one is a
performance change that must leave seeded results bit-identical (the 36-study hash probe and the GP-family probe,
plus the area's live-Optuna suite), unless it says otherwise.

1. **QA-84's cost in TPE** (§8). Today: `tpe` 1,000 trials 1.20–1.34 s against 0.75–0.81 s before `cb512df`,
   `tpe-mixed` +27–41%, `motpe` +39%, allocation +6–25 MB. Owner decision: wait for NumSharp (its journey5 pins
   `SumOverShortRows_CostsAFewLongRowSums` and `SumIntoOut_CostsAboutAFreshSum`; after a fix, bump NumSharp and
   re-measure). The full recovery is the in-register sum below eight continuous dimensions, provably bit-identical
   there.
2. **The study core's per-suggest cost** (§4 item 10). Random search spends 19.5 KB per trial: about 40% is the
   in-memory storage's copy-on-write of a running trial's parameter and distribution dictionaries on each suggest,
   about 20% closures and delegates. Every sampler pays it. Rule: copy-on-write is plan 13 S4's design for concurrent
   readers; changing it needs its own correctness argument and the concurrency suite.
3. **BruteForce: cache finished trials' paths across rebuilds** (§4 item 6). Path building is 28% of its time; 4,320
   trials take 8.7 s and 5,446 MB (1.29 MB per trial). Rule: the per-rebuild categorical cache (`catCache`, the first
   trial seen per name after filtering) must be reproduced exactly.
4. **NSGA-II precompute** (§4 item 2): each trial's values and constraints once, parameters only for the selected
   parents, the joint search space updated in place (the sampler costs about 30 µs per trial; 5,000 trials 0.56 s,
   33 KB per trial).
5. **MOEA/D's `SystemAttrs` scans** (§4 item 12): two scans of every completed trial per ask (the offspring count and
   the new offspring); an incremental count makes them O(new trials). 2,000 trials: 0.44 s, 48.6 KB per trial.
6. **TPE's estimator arrays** (§4 item 11): about 66% of a TPE trial's allocation (258 KB per trial at the end of the
   pass) is the Parzen estimators' per-kernel `double[]`; pooling them means reworking who owns and returns them.
7. **The multi-objective split** (§4 item 8): `lvals` through nested LINQ (the LINQ part is provably bit-identical
   to replace) and `UniqueRows` with an unstable delegate sort (its representative among equal rows, ±0, reaches the
   hypervolume arithmetic, so changing it is not provably bit-identical).
8. **CMA-ES: keep the optimizer between tells** (§4 item 3) instead of restoring it from the stored state after each
   tell (5,000 trials 0.80 s, 45.7 KB per trial). Rule: the stored state must stay the pickle cmaes 0.13.1 reads, and a
   cached optimizer must be dropped when another worker tells.
9. **An unidentified source of NumSharp `UInt32` blocks** that the GP run leaves to the finalizer (§4 item 5): not the
   entropy-seeded `RandomState()` (measured); find it with the `GCAllocationTick` sampler (§6).
10. **TPE scoring through ordered arrays** instead of name-keyed dictionaries (§4 item 1): not attempted; the kernels
    were the larger costs.
