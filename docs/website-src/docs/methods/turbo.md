# TuRBO (2019)

**Scalable Global Optimization via Local Bayesian Optimization** — David Eriksson, Michael Pearce, Jacob
Gardner, Ryan D. Turner, Matthias Poloczek, NeurIPS 2019. arXiv:
[1910.01739](https://arxiv.org/abs/1910.01739).

TuRBO (Trust-Region Bayesian Optimization) fixes the failure mode of global [GP-BO](gp.md) in higher
dimension: a single global surrogate, fit under stationary-kernel assumptions, over-explores and wastes
the budget. TuRBO instead keeps a **local** GP inside a **trust region** around the incumbent, proposes by
Thompson sampling within that region, and grows/shrinks/restarts the region from the observed progress.
OptunaSharp ships **both variants behind one sampler**: **TuRBO-1** (a single trust region, the default) and
**TuRBO-m** (`m` trust regions run simultaneously with an implicit multi-armed bandit across them), selected
by the `numTrustRegions` constructor argument.

## The idea

Work in the normalized box `Ω = [0, 1]^d`. TuRBO runs a sequence of *local* optimization runs
(**restarts**); each restart maintains one trust region and a local GP fit only to that restart's own
observations. A restart begins with a space-filling Sobol initialization, then repeats one loop — fit →
place the region → propose → resize — until the region collapses and a fresh restart begins.

```
   each trial
      │
      ▼
  reconstruct state from history  ─▶  L, success/failure counters, restart boundary   (nothing persisted)
      │
      ├─ still initializing (< nInit trials this restart)
      │     └─▶ propose the next scrambled-Sobol point in [0, 1]^d
      │
      └─ Bayesian-optimization phase:
            1. fit the local GP to this restart's points     (Matérn-5/2 ARD, §C-boxed hyperparameters)
            2. place the trust region: centre x⋆, sides  L_i = λ_i · L / (Π_j λ_j)^{1/d}
            3. candidates: scrambled Sobol in the region, each coord perturbed w.p. min(20/d, 1)
            4. Thompson: one JOINT draw  f̃ = mean + L·z  over the candidates → take the argmax
      │
      ▼
  evaluate, then the NEXT trial's fold resizes the region:
      success (strictly improves x⋆): τ_succ in a row  ⇒  L ← min(2L, L_max)
      failure:                        τ_fail in a row  ⇒  L ← L / 2
      L < L_min                                        ⇒  discard the region, restart (fresh Sobol, new x⋆)
```

**Trust region.** A hyperrectangle centered at the incumbent `x⋆` — the best *observation* on a noise-free
objective, or the training point with the best posterior *mean* when the objective is noisy (paper §2), so a
lucky noise draw can't anchor the region on a spurious point. It has a base side length `L`, and each
dimension's side is rescaled by the fitted GP length scale while holding the total volume at `L^d`:

```
L_i = λ_i · L / (Π_j λ_j)^{1/d}
```

so flatter directions (longer length scales) get wider sides. `L` starts at `L_init`. If `L` ever grew large
enough to cover the whole box, TuRBO would reduce to standard global BO — shrinking it is precisely what
keeps the local model accurate where the budget is being spent.

**Candidates by Thompson sampling.** Draw a scrambled Sobol' set inside the trust region, perturbing only a
random subset of coordinates per candidate (each with probability `min(20/d, 1)`, at least one) and holding
the rest at the incumbent — so in high dimension candidates stay near `x⋆` along most axes. Then draw one
realization of the local GP posterior over the candidate set and take its argmin (argmax of the score):

```
x_next ∈ argmin_{x ∈ TR} f̃,   f̃ ∼ GP_posterior
```

The draw is the **joint** posterior over the candidate set (`f̃ = mean + L·z`, `cov = L·Lᵀ`), not independent
per-point marginals — so the realization respects the correlations a tight region induces, which is what
lets Thompson sampling explore coherently instead of chasing per-point noise.

**Resize & restart.** A *success* is a candidate that strictly improves `x⋆`, a *failure* one that does not.
After `τ_succ` consecutive successes `L ← min(L_max, 2L)`; after `τ_fail` consecutive failures `L ← L/2`;
counters reset on any resize. When `L` falls below `L_min` the region has collapsed — TuRBO **discards it
and restarts** a fresh local run (new Sobol initialization, new incumbent). This restart-from-collapse is
what keeps TuRBO globally effective while each run stays local: successive restarts land in different basins
(the paper's §3.7 shows their final centres spread across the space), and Optuna keeps the global best across
all of them.

**Defaults** (paper supplement §C/§D): `L_init = 0.8`, `L_min = 2⁻⁷ ≈ 0.0078`, `L_max = 1.6`, `τ_succ = 3`,
`τ_fail = ⌈d/q⌉ = d` (batch `q = 1`), strict-improvement successes, `min(100·d, 5000)` Thompson candidates,
and a local GP fitted inside the §C hyperparameter box (`λ ∈ [0.005, 2]`, `s² ∈ [0.05, 20]`,
`σ² ∈ [0.0005, 0.1]`). The authors' reference code / BoTorch tutorial differ in two knobs — `τ_succ = 10` and
`τ_fail = max(4, d)` (a low-dimension guard, identical for `d ≥ 4`) — both reachable via constructor
arguments (see the knob reference below).

## In OptunaSharp

TuRBO is **absent from Optuna core**, so — like [LM-CMA](lm-cma.md),
[Differential Evolution](differential-evolution.md), and the [OpenAI-ES](openai-es.md) family — it is a
*beyond-core*, convergence-validated sampler with no Optuna counterpart. OptunaHub does have a TuRBO, by the Optuna
team ([`samplers/turbo`](https://hub.optuna.org/samplers/turbo/), `TuRBOSampler`, since 2025-09); OptunaSharp ports
the paper instead and does not follow that package, so a study moved over from it gets other defaults and other
trials, even with the same seed:

| OptunaHub `TuRBOSampler` | OptunaSharp `TurboSampler` |
|---|---|
| `n_trust_region=5` | `numTrustRegions = 1` (TuRBO-1; pass 5 for TuRBO-5) |
| `n_startup_trials=4` | `nInit`: `2·d` Sobol points per (re)start |
| `failure_tolerance=5` | `failureTolerance`: the paper's `d` |
| `success_tolerance=3`, `init_length=0.8`, `max_length=1.6`, `min_length=0.5**7` | `successTolerance = 3`, `lengthInit = 0.8`, `lengthMax = 1.6`, `lengthMin = 1/128`: the same values |
| `seed`, `independent_sampler`, `deterministic_objective`, `warn_independent_sampling` | the same names in C# casing; plus `numCandidates`, `successMargin`, `minimumNoise` |

The sampler **reuses the native GP stack**
that [GP-BO](gp.md) ships (`GPRegressor`: Matérn-5/2 ARD, the hand-derived marginal-likelihood gradient,
managed L-BFGS-B, OpenBLAS Cholesky), so no new surrogate machinery is introduced — the new code is the
trust-region bookkeeping, the Sobol-in-TR candidate set, and the Thompson draw.

`TurboSampler` drives an internal `TurboOptimizer` through the Optuna ask/tell protocol:

```csharp
using OptunaSharp;
using OptunaSharp.Samplers;

var study = Optuna.CreateStudy(sampler: new TurboSampler(seed: 42));
study.Optimize(t =>
{
    double s = 0;
    for (int i = 0; i < 10; i++) { double x = t.SuggestFloat($"x{i}", -5, 5); s += x * x; }
    return s;
}, nTrials: 200);
```

**Constructor knobs.** Every value in the schedule is overridable; the defaults are the paper's, with the
reference-code / BoTorch alternative noted where they differ (`src` cites the paper section):

```
knob                    default              controls
──────────────────────  ───────────────────  ──────────────────────────────────────────────────────────
seed                    OS entropy           reproducible trajectory on a platform
nInit                   2·d                  Sobol init points per restart before the GP takes over
numCandidates           min(100·d, 5000)     §D · Thompson candidate-set size (O(n³) draw — see below)
lengthInit              0.8                  §D · initial base side length L on a fresh restart
lengthMin               2⁻⁷ ≈ 0.0078         §D · restart floor: L below this ⇒ discard + re-initialize
lengthMax               1.6                  §D · cap on L under expansion
successTolerance        3                    §D · consecutive successes ⇒ double L   (BoTorch: 10)
failureTolerance        ⌈d/q⌉ = d            §D · consecutive failures ⇒ halve L     (ref code: max(4, d))
successMargin           0.0 (strict)         §D · relative gain to count as a success (BoTorch: 1e-3)
minimumNoise            5e-4                 §C · GP noise-variance floor & §C-box lower bound
deterministicObjective  false                §2 · noise-free ⇒ fix noise + centre on best observation
numTrustRegions         1 (TuRBO-1)          §2 · m simultaneous trust regions (m>1 ⇒ TuRBO-m; see below)
independentSampler      RandomSampler        categorical / single-valued / init fallback
```

Four properties worth knowing:

- **Continuous / integer only.** The trust-region box is a continuous construct, so the relative search
  space is Float/Int (Int normalized then snapped); categorical and single-valued parameters go to the
  independent sampler, as in [`CmaEsSampler`](cma-es.md). Single-objective only.
- **State is reconstructed, not persisted.** TuRBO stores no optimizer blob — the region length, resize
  counters, and restart boundaries are recomputed each trial by a deterministic fold over the completed
  trials' objective sequence, so a study resumes from storage with no serialized state and parallel workers
  stay consistent without a shared optimizer object.
- **Paper-faithful local GP (§C).** The local GP is fitted under the paper's hard hyperparameter box
  (`λ ∈ [0.005, 2]`, `s² ∈ [0.05, 20]`, `σ² ∈ [0.0005, 0.1]`) — a *bound-constrained* marginal-likelihood
  fit that reuses the [GP-BO](gp.md) stack unchanged, driving it through a boxed L-BFGS-B (`GPSampler`'s own
  *unconstrained* soft-prior fit is left untouched). The noise floor defaults to the §C `0.0005` (not the
  stack's `1e-6`), which deliberately caps precision on a truly noise-free objective; pass
  `deterministicObjective: true` for the paper's noise-free treatment (fix the noise at the floor and centre
  the region on the best observation rather than the best posterior mean), and/or lower `minimumNoise` if you
  need tighter interpolation.
- **Cost & candidate count.** `numCandidates` defaults to the paper's `min(100·d, 5000)` (§D). The Thompson
  draw forms the joint posterior over the candidate set and Choleskys it (`O(numCandidates³)` per trial) — a
  dense managed draw, where the paper leans on BoTorch's scalable sampler — so at high `d` this default is
  genuinely heavy (a 5000-wide Cholesky per trial). Lower `numCandidates` (e.g. a few hundred) to trade
  discretization fidelity for speed when per-trial overhead dominates.

**When to reach for it.** TuRBO's edge over global [GP-BO](gp.md) shows on higher-dimensional and
multimodal objectives, where a single global GP over-explores — the trust region + restart concentrates the
budget locally and re-seeds when a basin is exhausted. On smooth low-dimensional problems plain
[`GPSampler`](gp.md) is already excellent and simpler. Two dials matter most in practice: give each restart
enough budget (and `nInit`) to actually exploit its region before it collapses, and drop `numCandidates` if
the per-trial `O(n³)` draw becomes the bottleneck.

## TuRBO-m: the multi-region bandit

TuRBO-1 runs **one** trust region, restarting it sequentially on collapse. The paper's full method (§2) runs
**`m`** trust regions **simultaneously** and lets an implicit multi-armed bandit decide which region spends
each evaluation. Set `numTrustRegions = m > 1` to activate it (the common choice is `5`, the paper's
"TuRBO-5"); `numTrustRegions = 1` is TuRBO-1, a provably byte-identical code path.

Each region is **exactly TuRBO-1** — its own local GP, incumbent `x⋆_ℓ`, base length `L_ℓ`, and
success/failure counters. The only new mechanic is the competition **across** regions:

```
   each trial
      │
      ▼
  read the persisted per-trial region labels, partition the completed trials into the m regions,
  fold each region independently (TuRBO-1 over its own labelled subsequence)
      │
      ├─ any region still initializing (< nInit trials this run)?
      │     └─▶ INIT-FILL the lowest such region: propose its next Sobol point   (fills slot-by-slot at
      │          startup; forces a collapsed region's fresh nInit re-init burst)  → label the trial with it
      │
      └─ every region ready ⇒ THE BANDIT COMPETES:
            for each region ℓ:  fit GP_ℓ, build its candidate set, draw ONE joint Thompson sample f̃_ℓ
            un-standardize each region's draw to shared objective units  (each region standardizes by
                                                                          its OWN mean/std — required!)
            winner = argmax over ALL regions and candidates   ⇒  x_i ∈ argmin_ℓ argmin_{x∈TR_ℓ} f̃_ℓ
            → propose the winner's candidate, label the trial with the winning region
```

Because the winner is whichever region's Thompson draw looked best, promising regions naturally win more
evaluations — that emergent allocation *is* the bandit (there is no separate arm-selection rule). Each region
resizes and restarts on its **own** counters (paper §D: a batch is a success for region ℓ only if a point
drawn from it beats ℓ's own incumbent), independently of the others, so a collapsed region is discarded and
re-initialized in place while the rest keep going — `m` regions stay live throughout.

**The one required departure from TuRBO-1: a persisted label.** TuRBO-1 persists nothing because its restart
boundaries reconstruct from the value sequence alone (restarts are contiguous blocks). TuRBO-m *interleaves*
the regions — a trial's region is the *seeded competition winner*, a function of which prior trials landed in
which region, **not** of the values alone — so replaying it from scratch would cost `O(n²·m)` GP fits.
Instead each trial records its winning region as one small system attr (`"turbo:tr"`); every region then
folds cheaply (`O(n·m)` total) from `(labels, values)`, exactly TuRBO-1 over its labelled subsequence. This
is the middle ground between TuRBO-1 (persists nothing) and CMA-ES (persists a full serialized blob); each
worker writes only its own trial's label, so it stays parallel-worker-safe and resumable.

**Honest scope.** The budget is **divided** across the `m` regions (each pays a `2·d` init tax and gets fewer
exploitation steps), so TuRBO-m is **not** uniformly better than TuRBO-1: it wins on multimodal/deceptive
problems with several basins (§3.7 — restarts land in distant basins of varying quality, and the bandit
concentrates on the best) and can tie or slightly lose on easy/unimodal ones. This is exactly why the paper
reports TuRBO-1 **and** TuRBO-5 **and** TuRBO-20 by regime, and why our tests assert the robust claims (every
region is used, the bandit concentrates on one region far beyond the uniform `1/m` share, regions restart
independently, no severe regression) rather than a fragile "`m>1` always beats `m=1`".

```csharp
// TuRBO-5: five trust regions competing on a multimodal objective with several basins.
var study = Optuna.CreateStudy(sampler: new TurboSampler(seed: 42, numTrustRegions: 5));
study.Optimize(t =>
{
    double s = 100;
    for (int i = 0; i < 10; i++)
    {
        double x = t.SuggestFloat($"x{i}", -5, 5);
        s += x * x - 10 * Math.Cos(2 * Math.PI * x);   // Rastrigin — many local basins
    }
    return s;
}, nTrials: 400);
```

Everything else — the §C-boxed local GP fit, the length-scale-weighted region, the Sobol-in-TR candidate set,
the joint Thompson draw, and the per-region resize schedule — is the unchanged TuRBO-1 machinery above, reused
once per region.

## Validation

Behavioral, not bit-exact: GP fitting + Thompson sampling are stochastic and, like the rest of the GP
stack, tolerance-reproducible vs BoTorch rather than byte-for-byte (the documented GP boundary). Beyond
convergence (the sampler reaches the sphere optimum and drives the multimodal Ackley far below a random
baseline), the unit tests assert the paper's own claims directly: the length-scale-weighted box has volume
exactly `L^d` (§2); a large `L` degenerates the trust region to the whole `[0,1]^d` box (§2's "equivalent to
global BO"); the resize schedule expands at `τ_succ`, shrinks at `τ_fail = d`, and restarts below `L_min`
(§D); a strict improvement — not a tie — counts as a success (§D); the local GP's fitted hyperparameters land
inside the §C box; the noisy-case center follows the best posterior mean rather than the best observation
(§2); and the Thompson draw is the exact joint posterior sample `mean + L·z` (§E), reproduced
element-for-element from the seed.

For **TuRBO-m** the tests add the multi-region claims: the cross-region competition returns the global argmax
across regions **after un-standardizing** each region's draw to shared units (a strong region and a
low-variance region are otherwise compared on incompatible z-scales); the init-fill picks the lowest region
still initializing; and end-to-end over a real run every region is used, the bandit concentrates one region's
share far above the uniform `1/m`, regions restart independently, the labels partition the trials disjointly
(§D — only the winner's region owns a trial), and a fixed seed reproduces the whole multi-region trajectory.

## Relationship to the lineage

- **From [GP-BO (2012)](gp.md):** reuses the identical Matérn-5/2 ARD GP surrogate and its fit; replaces the
  global acquisition optimization with a local trust region + Thompson sampling.
- **TuRBO-1 ↔ TuRBO-m:** the paper's full method maintains `m` trust regions with an implicit multi-armed
  bandit across them (`numTrustRegions = m`); TuRBO-1 (`numTrustRegions = 1`, the default) is the
  single-region special case and the recommended choice for smooth or unimodal problems, while TuRBO-m earns
  its divided budget on multimodal landscapes with several basins.

## See also

[GP-BO (2012)](gp.md) · [CMA-ES (2001)](cma-es.md) · [LM-CMA (2015)](lm-cma.md) ·
[OpenAI-ES (2017)](openai-es.md) · [Samplers](../samplers.md) · the [API reference](../../api/index.md).
