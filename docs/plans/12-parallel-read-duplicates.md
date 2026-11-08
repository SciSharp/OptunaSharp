# Plan 12 — Parallel reads never hand one point to two trials

> **Status: DONE (2026-11-08).** The owner asked for test coverage, per sampler, that parallel reads produce no
> duplicate samples, and chose on 2026-11-08 to fix the samplers that did. Every Core sampler now gives the trials
> that ask at once different points; the four Python bridges are measured and pinned, not changed.
> [`docs/THREADING.md`](../THREADING.md) §6.3 describes what the code does.

---

## 1. The request and the decisions (owner, 2026-11-08)

The request: "Add test coverage to each sampler confirming parallel read doesn't produce duplicate samples."

A survey (§3) found samplers that did produce them, and the owner chose:

| Question | Answer |
|---|---|
| The samplers that hand one point to several trials asked at once | "Fix all six, then confirm": the fixes act only while other trials run, so sequential runs and JANUS's byte parity stay identical, and every Core sampler then gets a test that asserts no collision |
| BruteForce, whose strict mode hands running points out again on purpose, as Optuna's does | "Fix it too (diverge from Optuna)": concurrent asks never share a point in either mode; the test is `[Misaligned]` |
| The Python bridges (OptunaHub, BoTorch, PyCma, pymoo) | "Core plus the four bridges": interop tests that measure and pin each bridge's repeats under parallel reads |

## 2. What is measured

- **A parallel read:** a round of eight trials that ask and suggest on the same history before any of them is told.
  `test/OptunaSharp.Tests/Concurrency/ParallelReads.cs` makes it two ways:
  - `Fire`: eight dedicated threads of a `ConcurrencyGun`, released together, with a barrier between the reads and
    the tells. The interleaving is whatever the threads take.
  - `ReadInLockstep`: one thread asks the eight trials, then each samples its relative parameters before any of them
    stores one, then they suggest and are told. This is the interleaving where every read lands before any write, and
    it repeats exactly, so its counts can be pinned.
- **A collision:** one new point received by two or more trials of one round. Only parallel reads can make one.
- **A history repeat:** a point a trial of an earlier round already had. A sampler that re-proposes finished points
  does so one trial at a time too (`ReadOneAtATime`, the sequential baseline).

## 3. The survey before the fixes (8 rounds of 8, collisions)

| Sampler | Collisions | Why |
|---|---|---|
| `LmCmaSampler` | 8–10 | a Rademacher candidate is one of few points per generation, and concurrent asks drew the same one |
| `RandomEmbeddingSampler` (REMBO) | up to 21 | concurrent asks maximized the same deterministic acquisition |
| `JanusSampler`, `JanusDeSampler` | 7 (and 30, 29 of 120 under `Optimize(nJobs: 16)`, from storage.md) | the population slot is the trial number modulo the population size, and the trials past the population wrapped onto taken slots |
| `PsoSampler` | 1–9 | the particle was the count of finished trials of the generation, the same for every concurrent ask |
| `DifferentialEvolutionSampler` | up to 2 | the same target for every concurrent ask; then, with distinct targets, the same donors or a clamp onto the same corner |
| `TurboSampler` | every startup round | the startup Sobol row was the count of finished trials |
| `SaasboSampler` (found after the decision) | 7 | the same, for its startup design |
| NSGA-II, NSGA-III, NSGA-III-CPT (found after the decision) | 1 in one run of 12–20 | two children built from the same parents and crossover mask, with no mutation |
| `BruteForceSampler` | strict mode 3–8 per 88 trials of a 90-point space, default 0–1 | strict mode weighs running points as unexplored on purpose; both modes see a running trial one parameter late |

Every other Core sampler gave 0, threaded and in lockstep, and with per-worker instances for the samplers that take
part in `PerWorker`.

## 4. The fixes

The rule for all of them: act only where a trial asked one at a time would not. All but JANUS look at the *other
running trials*, which a sequential run has none of; JANUS hands each slot of a generation out once, which a
sequential run, asking consecutive trials, does anyway. So a sequential run never skips a slot, never redraws and never
lies: seeded results, Optuna parity and JANUS's byte parity with its reference are unchanged.

| Sampler | Fix | Shared helper |
|---|---|---|
| JANUS (CMA and DE hosts) | `JanusOptimizer.AskSlot(trialNumber)` hands each slot of a generation to one trial, running or finished: the reference's slot when free, else the next free one; with none left it returns null and the trial is sampled independently | — |
| PSO, DE | the first particle or target that no finished or running trial of the generation holds, read in one storage read with the stamps; a point a running trial holds is redrawn, up to 32 times; with neither, an independent sample without the independent-sampling warning | `ConcurrentAsks.OtherCompleteOrRunningTrials`, `HandedOutPoints`, `OverrunTrials` |
| LM-CMA | a Rademacher candidate a running trial of the generation holds is redrawn, up to 32 times; then an independent sample | `HandedOutPoints`, `OverrunTrials` |
| TuRBO, SAASBO | the first startup Sobol row no completed or running trial holds, stamped (`turbo:init` with slot and restart, `saasbo:init`); rows past the design are the scrambled sequence's next points, which leave the first ones unchanged | `ConcurrentAsks.FirstFreeIndex` |
| REMBO | the acquisition is fitted with the running trials' stored latents as constant lies (the worst observed value), and the low-dimensional proposer re-asks while its point equals a pending latent | — |
| NSGA-II/III (and CPT), AGE-MOEA-2 (and CPT), RVEA, SMS-EMOA, MOEA/D, and the JANUS multi-objective samplers through their hosts | a child equal to one a running trial holds is generated again, up to 32 times; then an independent sample | `HandedOutParams` (`BaseGASampler.HandedOutChildren`) |
| BruteForce | the values handed to each running trial are recorded, so the tree sees every running path in full; a trial still choosing counts as taking one point below its path (`CountFreeUnexplored`, `CountFreeEnds`); the trials choosing the same parameter at the same node are not subtracted, so the first of them does not yield its point; with nothing unexplored free, a trial re-evaluates a finished point only in a branch that still has one | — |

The records (`HandedOutPoints`, `HandedOutParams`, BruteForce's handed-out values, JANUS's slot holders) live in the
sampler instance and are guarded by its monitor; entries leave with `AfterTrial`, or when the trial is no longer
running (JANUS's when the next generation is drawn).

**A lesson from JANUS.** Its first fix freed a slot once the trial holding it was no longer running. Every volley
passed, because a round of 8 releases a population of 7 together, but `Optimize(nJobs: 16)` repeated 41–65 points in
120 trials: a trial asked while its generation waited for the rest of its trials received a finished trial's
candidate again. Hence the population-spanning volley and the whole `Optimize` runs in the tests. The one sequential
difference of the final rule: after a JANUS trial fails, its generation is never told (the C# port batches in ask
order), and the trials past its slots are sampled independently instead of evaluating its candidates again.

## 5. Tests

- `test/OptunaSharp.Tests/Concurrency/SamplerParallelReadTests.cs`: every Core sampler, threaded (`Fire`) and in
  lockstep, asserting no collision, and no history repeat for the samplers that never re-propose a finished point
  (CMA-ES and VkD-CMA may repeat one point, through Optuna's per-trial seeding, about once in 65,535 pairs); SAASBO
  with a light NUTS budget; per-worker instances; JANUS, JANUS-DE, DE, PSO and LM-CMA with a population of 12, so a
  generation spans two rounds; whole `Optimize(nJobs: 16)` runs in which no sampler that never re-proposes a
  finished point evaluates a point twice; Grid and BruteForce (both modes, `[Misaligned]`) over 88 of 90 points; DE
  and PSO with more trials than their population, sampled independently without warnings; and the sequential
  baseline that shows the genetic samplers, DE, JANUS-DE, PSO and LM-CMA repeat finished points without any
  parallelism.
- `test/OptunaSharp.Tests.Interop/BridgeSamplerParallelReadTests.cs` (the harness is compiled in from the unit
  suite): lockstep counts pinned exactly, PyCma's compared bit for bit with `optuna_integration.PyCmaSampler` read the
  same way; threaded volleys asserting what holds in every interleaving. BoTorch is in the slow category.
- Before the tests were written, 100–400 volleys per changed sampler collided in none; the threaded and lockstep
  survey of every sampler gave 0 collisions.

## 6. Measured under `Optimize(nJobs: 16)`

Whole repeated points in 120 trials (60 for the Gaussian-process samplers), in memory, three runs, against one trial
at a time:

| Sampler | `nJobs: 16` | One at a time |
|---|---|---|
| Random, TPE, CMA-ES, VkD-CMA, QMC, OpenAI-ES, SNES, JANUS, JANUS-DE, GP, TuRBO, REMBO | 0 | 0 |
| DE, PSO | 0–3, 1 | 2, 1 |
| LM-CMA | 7–13 | 9 |
| NSGA-II, NSGA-III, the JANUS multi-objective sampler | 10–25, 11–25, 18–24 | 16, 14, 16 |
| SMS-EMOA, MOEA/D, AGE-MOEA-2, RVEA | 2–3, 0–1, 1–4, 1–6 | 7, 4, 7, 4 |
| Grid, BruteForce (default, strict), 90 points | 104–107, 91–101, 97–101 trials to cover them | — |

## 7. Measured on the bridges (not changed)

| Bridge | Lockstep | Threaded |
|---|---|---|
| PyCma | 49 collisions, exactly Optuna's points | 0–1 collisions |
| pymoo NSGA-II, AGE-MOEA-2 | 42 collisions | 0–2 collisions, and 7–8 offspring re-handed after told trials evaluated them |
| OptunaHub `random_search`, `pso` | 0 | 0 |
| OptunaHub `differential_evolution` | 49 | 49 (it ignores running trials) |
| BoTorch, with and without `considerRunningTrials` | 0 | 0 |

## 8. Not done (follow-ups, not decided)

- **Two processes on one storage** are steered apart only by the stamped indices (particles, targets, design rows):
  JANUS's slots, the redraws and BruteForce's record are per instance.
- **Grid at the end of the grid** keeps Optuna's rule: once every id is finished or running, any id, a running one
  included, until the study stops. The BruteForce fix could apply.
- **The pymoo bridge under threads** re-hands offspring that told trials already evaluated; doling by a stamped
  index, as PSO and DE do, would stop it.
- **MOEA/D** gives concurrent asks the same subproblem (the count of completed offspring); their children differ, so
  this costs search quality, not duplicates.
- **BruteForce near the end of the space** counts an unexpanded branch as one point and a running trial still
  choosing at a fresh node as taking its whole branch, so when the running trials hold nearly every unexplored point
  a trial may re-evaluate a finished point while one is still free.
- **xNES degenerates on its own** (found by the whole-run test, not a concurrency issue): one trial at a time, on the
  3-D quadratic of the tests (`x`, `y` in [-10, 10], `z` in [0, 1], optimum on `z = 0`), 6 of 100 seeds went wrong
  within 120 trials. Seed 18 clamped candidates onto the box's corners and repeated 5 of them. Seeds 24, 31, 64 and
  77 failed the study with `ValueError: nan is invalid value` (trials 113, 106, 85, 99), and seed 42 with
  `LinAlgError: Singular matrix` (trial 92). SNES and OpenAI-ES were clean over the same 100 seeds. xNES is left out
  of the whole-run test until this is fixed.
