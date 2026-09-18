# TuRBO-m Porting Plan — multi-trust-region trust-region Bayesian optimization

**Reference:** David Eriksson, Michael Pearce, Jacob Gardner, Ryan D. Turner, Matthias Poloczek,
*Scalable Global Optimization via Local Bayesian Optimization*, NeurIPS 2019
([arXiv:1910.01739](https://arxiv.org/abs/1910.01739)), §2 "Trust region Bayesian optimization" +
supplement §C/§D. No Optuna/OptunaHub counterpart; the operational details (init, re-init, the batch
competition) come from the paper + the authors' reference code (`uber-research/TuRBO`, `turbo_m.py`) and the
BoTorch TuRBO tutorial's multi-region variant.

**Status.** TuRBO-1 is DONE and paper-faithful (`Samplers/Turbo/{TurboOptimizer,TurboSampler,TurboGpFit}.cs`;
commits `19c2e07` → `1c13785` → `88f049b`). TuRBO-m — the multi-trust-region bandit — is the last remaining
Expansion-Plan **P2** item (`plans/OPTUNASHARP_EXPANSION_PLAN.md` §5). This document plans it by reproducing
the exact techniques we used for TuRBO-1, **in the same order**, and calls out the one place TuRBO-m must
diverge (minimal state persistence).

---

## 1. The algorithm, in theory (§2 + §D)

TuRBO-1 runs **one** trust region at a time, restarting it sequentially on collapse. TuRBO-m runs **m** trust
regions **simultaneously** and lets an implicit multi-armed bandit decide which region to spend each
evaluation on.

- **m independent regions.** Each trust region `TR_ℓ`, `ℓ ∈ {1,…,m}`, is a hyperrectangle of base side
  length `L_ℓ ≤ L_max` with its **own** local GP, its **own** incumbent `x⋆_ℓ`, and its **own**
  success/failure counters. Everything inside a single region is *exactly* TuRBO-1.
- **The bandit is Thompson sampling across regions (the one genuinely new mechanic).** To pick the `i`-th of
  the batch's `q` candidates, draw **one** posterior realization from *each* region's local GP over that
  region's candidate set, `f_ℓ^(i) ∼ GP_ℓ(μ_ℓ, k_ℓ)`, then take the global optimum across all regions and
  their candidates:

  ```
  x_i ∈ argmin_ℓ  argmin_{x ∈ TR_ℓ}  f_ℓ^(i)
  ```

  i.e. concatenate a Thompson sample from every region and take the single best. The winning candidate comes
  from whichever region's draw looked best, so promising regions naturally win more evaluations — that *is*
  the bandit (no separate arm-selection rule). (The paper minimizes `f`; our scores are
  maximization-oriented, so this is a global **argmax** over the concatenated sampled scores.)
- **Per-region resize (§D), separate counters.** A batch is a **success for `TR_ℓ`** iff `q_ℓ > 0` candidates
  were drawn from it *and* at least one improves `TR_ℓ`'s **own** incumbent; otherwise (all `q_ℓ` worse) it is
  a **failure** and `q_ℓ` is added to that region's failure counter. Regions that got `q_ℓ = 0` candidates in
  a batch are untouched. Then the usual TuRBO-1 schedule per region: `τ_succ` successes ⇒ `L ← min(2L,L_max)`;
  `τ_fail` failures ⇒ `L ← L/2`; reset counters on any resize; `L < L_min` ⇒ **discard the region and
  re-initialize** a fresh one in its slot (so `m` regions stay live).
- **Sequential case (`q = 1`, our setting).** Each trial produces exactly one candidate from exactly one
  **winning** region. Only that region has `q_ℓ = 1` (all others `q_ℓ = 0`), so **only the winner's counters
  update** that trial. This makes the per-region bookkeeping collapse to "TuRBO-1, applied to the subsequence
  of trials that were assigned to this region."
- **Initialization.** Each region is seeded independently with `n_init = 2d` scrambled-Sobol points before its
  GP can participate in the competition.
- **Why m>1 helps (§3.7 / Fig 6).** Restarts land in *distant* basins of *varying* quality; with `m` regions
  alive at once, the bandit concentrates the budget on the best basin while still exploring the others. This
  is the paper's motivation for TuRBO-m over TuRBO-1.

### Honest scope (like the TuRBO-1 GP head-to-head decision)

The budget is **divided** across `m` regions (each pays a `2d` init tax and gets fewer exploitation steps), so
TuRBO-m is **not** uniformly better: it wins on multimodal/deceptive problems with several basins and can tie
or slightly lose on easy/unimodal ones — which is exactly why the paper reports TuRBO-1 **and** TuRBO-5 **and**
TuRBO-20 by regime. The behavioral test must therefore pick a regime where `m>1` robustly wins, **or** assert
the weaker-but-robust claim (every region gets used, the bandit concentrates on the best, no severe
regression) — the same discipline that made us *not* assert "TuRBO-1 beats global GP" at low `d`.

---

## 2. The one required departure from TuRBO-1: minimal state persistence

TuRBO-1 persists **nothing** — its restart boundaries reconstruct from the completed-trial value sequence via
a deterministic left-fold, because restarts are **sequential contiguous blocks** (`TurboOptimizer.
ReconstructState`).

TuRBO-m **interleaves** the `m` regions: trial `t`'s region is the *stochastic-but-seeded competition winner*,
a function of the `m` fitted GPs, hence of which prior trials landed in which region. **A trial's region label
is therefore not a function of the value sequence alone.** Replaying all labels from scratch each trial costs
`O(n·m)` GP competitions repeated over `n` trials = `O(n²·m)` GP fits — prohibitive.

**Solution — persist the per-trial region label (one small int), nothing more.** At propose time, after the
competition picks the winning region `ℓ*`, record it on the trial exactly as CMA-ES/REMBO persist their state:

```csharp
study.Storage.SetTrialSystemAttr(trial.TrialId, "turbo:tr", ℓ*);   // read back via FrozenTrial.SystemAttrs
```

Then each region's `L`/counters/restart boundaries **fold cheaply** from `(labels, values)` — TuRBO-1's fold,
run per region slot over that slot's labeled subsequence. Total cost is back to `O(n·m)` (one `m`-way
competition per trial). This is a **middle ground**: TuRBO-1 persists nothing, CMA-ES persists a full
serialized blob, TuRBO-m persists just the labels (compact, resumable, parallel-worker-safe — each worker
writes only its own trial's label).

- **Init / re-init labels are deterministic**, so they need no persistence in principle: the initial fill
  assigns slot `= t / (2d)` while `t < m·2d`, and a collapsed region's forced `2d` re-init burst is derivable
  from that region's fold. Only the *BO-phase competition winners* strictly require a stored label; persisting
  every label is simplest and still `O(1)`/trial.
- **Alternatives considered:** a shadow-study per region (the REMBO precedent) or a full serialized blob (the
  CMA-ES precedent). The label-fold is preferred — it is the smallest change from TuRBO-1 and keeps the
  per-region mechanics literally identical.

---

## 3. Techniques from TuRBO-1, reproduced in the same order

For each step: **[what we did for TuRBO-1] → [the TuRBO-m adaptation]**.

1. **Read the paper + reference impl, observe first.** TuRBO-1: §2/§C/§D/§E + the BoTorch tutorial; measured
   real runs before calibrating. → TuRBO-m: §2 multi-TR + §D per-region counters; fetch the authors'
   `turbo_m.py` and the BoTorch multi-region tutorial for the init/re-init/competition details the paper is
   terse on; run a small live probe if a reference is runnable, else probe our own m-TR prototype.
2. **Per-TR mechanics as a pure class (`TurboOptimizer`).** TuRBO-1: the state fold, length-scale-weighted TR
   box, masked-Sobol candidate set, single-TR joint Thompson argmax, `SelectCenterIndex`, `TurboTrState`. →
   **REUSE UNCHANGED, once per region.** This is the analog of TuRBO-1 reusing the GP stack: **zero new
   per-region mechanic code** — a region *is* a `TurboOptimizer` fold over its labeled subsequence.
3. **§C bounded local-GP fit (`TurboGpFit`).** TuRBO-1: bound-constrained MLL fit over the shared
   `GPRegressor` via a boxed L-BFGS-B, GPSampler untouched. → **REUSE UNCHANGED**, called once per region
   (`m` fits per trial).
4. **The Thompson step.** TuRBO-1: `ThompsonArgmax` — one joint draw `mean + L·z` over a single TR's
   candidates, take the argmax. → **NEW cross-region competition** `TurboMOptimizer.SelectAcrossTrustRegions`:
   for each *ready* region, build its candidate set (`TurboOptimizer.GenerateCandidates`) and draw its joint
   posterior sample (`mean + L·z` on that region's `PosteriorJoint`), then take the **global argmax across the
   concatenation of every region's sampled candidate scores** → return `(winnerRegion, winnerCandidate)`. It
   reuses the exact per-region joint-Cholesky sample math; only the outer "compete across regions" is new.
5. **The public sampler `SampleRelative` pipeline.** TuRBO-1: fold → init-phase vs BO-phase → fit GP → center
   → TR box → candidates → Thompson → unnormalize. → **Extend `TurboSampler` (the `m>1` branch):** read the
   stored labels → partition completed trials into `m` slots → per-slot fold (`state_ℓ`, `x⋆_ℓ`, box_ℓ) →
   decide the init-fill (which slot still needs its next `2d` Sobol point, or which collapsed slot is
   re-initializing) → for every ready slot fit its GP + build candidates + draw its sample → **compete** →
   propose the winner's candidate → **persist the winner label**. `m == 1` **delegates to the existing,
   untouched TuRBO-1 path** (byte-identical; no label needed).
6. **Reuse the native GP stack unchanged** (`GPRegressor.PosteriorJoint`/`LengthScales`, `GpSearchSpace`,
   `ScrambledSobol`, `LinAlg`). → Same, `m` fits/trial.
7. **RNG discipline** (NumSharp MT19937; per-trial `Mix(baseSeed, trial.Number)`; per-restart Sobol seed
   masked `& 0x7FFFFFFF` because numpy `SeedSequence` rejects negative entropy). → Per-slot init Sobol seed
   `Mix(baseSeed ^ salt, slot·K + restartCount) & 0x7FFFFFFF`; per-trial competition RNG
   `Mix(baseSeed, trial.Number)`, with each region consuming the stream in a **fixed slot order `0..m-1`** so
   the competition reproduces bit-for-bit given `(seed, labels)`.
8. **State reconstructed, not persisted.** TuRBO-1: fold over the value sequence, no blob. → **Reconstructed
   from the stored labels** (§2 above) — the deliberate minimal-persistence departure; per-slot `L`/counters
   fold exactly as in TuRBO-1.
9. **Paper-faithful defaults + named default helpers** (`DefaultFailureTolerance(d)=d`,
   `DefaultNumCandidates(d)=min(100d,5000)`, `L_init=0.8`, `L_min=2⁻⁷`, `L_max=1.6`, `τ_succ=3`,
   strict-success, §C box). → Same per-region defaults; **new `numTrustRegions` ctor arg**, default `1`
   (backward-compatible TuRBO-1); document `numTrustRegions: 5` as the paper's common "TuRBO-5". The per-region
   `τ_fail = ⌈d/q⌉ = d` and `n_init = 2d` are unchanged.
10. **Two-layer tests** (unit mechanics + end-to-end behavioral, thresholds calibrated from probes). →
    - **Unit** (`TurboMOptimizerTests`): the cross-region competition returns the global argmax across regions;
      per-slot independent fold from labels; init-fill order; a collapsed slot's forced re-init; determinism
      under a fixed seed.
    - **End-to-end** (`TurboSamplerTests`, `m>1`): every slot is actually used (the bandit exercises all
      levers); the bandit **concentrates** proposals on the region holding the best basin; regions restart
      independently; reproducibility; `m>1` helps on a chosen multi-basin problem (regime-picked & seed-robust)
      **or** the weaker no-regression claim; the §D per-region success/failure rule (only the winner updates)
      asserted at the fold level.
11. **Validation: probe → calibrate honestly → both TFMs → full suite green.** → Same. Measure the m-TR
    dynamics first (are all slots used? does `m>1` help on the target problem? seed spread), then set robust
    thresholds; **guarantee `m=1` stays byte-identical** (the existing 45 Turbo tests must pass untouched).
12. **Full documentation.** TuRBO-1: consequence-conveying XML docs on every member + the `methods/turbo.md`
    DocFX page. → Same rigor, with the new footguns called out (the label persistence & why; the divided
    budget; the `numTrustRegions` default; per-slot re-init). Extend `methods/turbo.md` with a **"TuRBO-m"**
    section (it is the same sampler behind `numTrustRegions`), including the fenced mechanism diagram's
    multi-region variant.
13. **Commit + memory + plan update.** Extensive commit message; update memory `turbo-1-port.md` (or a new
    `turbo-m-port.md`); mark Expansion-Plan §5 TuRBO-m **DONE**.

---

## 4. Proposed files

| File | Change | Notes |
|---|---|---|
| `Samplers/Turbo/TurboMOptimizer.cs` | **new** | cross-region competition (`SelectAcrossTrustRegions`) + label→slot partition + init-fill / re-init helpers; each slot is a reused `TurboTrState`/`TurboOptimizer` fold |
| `Samplers/TurboSampler.cs` | **extend** | `numTrustRegions` ctor arg; `m>1` branch in `SampleRelative`; label persistence via `SetTrialSystemAttr`; **`m==1` path untouched** |
| `Samplers/Turbo/TurboOptimizer.cs`, `TurboGpFit.cs` | **reuse** | unchanged — per-region building blocks |
| `test/…/Samplers/TurboMOptimizerTests.cs` | **new** | unit competition/fold/init/restart tests |
| `test/…/Samplers/TurboSamplerTests.cs` | **extend** | end-to-end `m>1` behavioral tests |
| `docs/website-src/docs/methods/turbo.md` | **extend** | TuRBO-m section + multi-region diagram |
| `plans/OPTUNASHARP_EXPANSION_PLAN.md` | **update** | §5 status → TuRBO-m DONE |

---

## 5. Open design decisions (settle during implementation)

- **Init-fill order:** fill slot-by-slot (`slot = t / 2d` while `t < m·2d`) — recommended, fully deterministic —
  vs. round-robin. Either is valid; slot-by-slot mirrors the authors' "initialize each TR independently."
- **Collapsed-region re-init interleaving:** on collapse, force that slot's fresh `2d` Sobol init **immediately**
  (labels deterministic), then resume the competition — recommended (keeps `m` live, avoids a starved slot).
- **`numTrustRegions` default:** `1` (backward-compatible TuRBO-1) with TuRBO-5 opt-in — recommended — vs. a
  multi-TR default that would change existing behavior.
- **Label key & read path:** `trial.SystemAttrs["turbo:tr"]`; confirm the value round-trips through storage
  types (int) and that parallel workers stay consistent (they do — per-trial write).
- **`m=1` equivalence guarantee:** branch on `numTrustRegions == 1` to the existing path so TuRBO-1 is provably
  untouched (existing tests are the guard).

## 6. Definition of Done

- `m=1` **byte-identical** to current TuRBO-1 — the existing 45 Turbo tests pass unchanged.
- `m>1`: unit (competition, per-slot fold, init, restart, determinism) + end-to-end behavioral (all levers
  used, bandit concentrates, multi-basin help / no-regression, reproducibility) — both TFMs; full unit suite
  green with no regressions.
- Full XML docs + `methods/turbo.md` TuRBO-m section; Expansion-Plan §5 marked DONE.
- Extensive commit; memory updated.
