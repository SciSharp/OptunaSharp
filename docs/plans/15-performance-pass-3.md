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
