# TuRBO (2019)

**Scalable Global Optimization via Local Bayesian Optimization** — David Eriksson, Michael Pearce, Jacob
Gardner, Ryan D. Turner, Matthias Poloczek, NeurIPS 2019. arXiv:
[1910.01739](https://arxiv.org/abs/1910.01739).

TuRBO (Trust-Region Bayesian Optimization) fixes the failure mode of global [GP-BO](gp.md) in higher
dimension: a single global surrogate, fit under stationary-kernel assumptions, over-explores and wastes
the budget. TuRBO instead keeps a **local** GP inside a **trust region** around the incumbent, proposes by
Thompson sampling within that region, and grows/shrinks/restarts the region from the observed progress.
OptunaSharp ships **TuRBO-1** — the single-trust-region variant.

## The idea

Work in the normalized box `Ω = [0, 1]^d`. TuRBO runs a sequence of *local* optimization runs; each run
maintains a trust region and a local GP fit only to that run's observations.

**Trust region.** A hyperrectangle centered at the incumbent `x⋆` (the best observation so far). It has a
base side length `L`, and each dimension's side is rescaled by the fitted GP length scale while holding
the total volume at `L^d`:

```
L_i = λ_i · L / (Π_j λ_j)^{1/d}
```

so flatter directions (longer length scales) get wider sides. `L` starts at `L_init`.

**Candidates by Thompson sampling.** Draw a scrambled Sobol' set inside the trust region, perturbing only a
random subset of coordinates per candidate (each with probability `min(20/d, 1)`, at least one) and holding
the rest at the incumbent — so in high dimension candidates stay near `x⋆` along most axes. Then draw one
realization of the local GP posterior over the candidate set and take its argmin (argmax of the score):

```
x_next ∈ argmin_{x ∈ TR} f̃,   f̃ ∼ GP_posterior
```

**Resize & restart.** A *success* is a candidate that improves `x⋆`, a *failure* one that does not. After
`τ_succ` consecutive successes `L ← min(L_max, 2L)`; after `τ_fail` consecutive failures `L ← L/2`;
counters reset on any resize. When `L` falls below `L_min` the region has collapsed — TuRBO **discards it
and restarts** a fresh local run (new Sobol initialization, new incumbent). This restart-from-collapse is
what keeps TuRBO globally effective while each run stays local.

**Defaults** (BoTorch TuRBO tutorial, except `τ_succ` from the paper supplement): `L_init = 0.8`,
`L_min = 2⁻⁷ ≈ 0.0078`, `L_max = 1.6`, `τ_succ = 3`, `τ_fail = ⌈max(4/q, d/q)⌉ = max(4, d)` (batch `q = 1`).

## In OptunaSharp

TuRBO is **absent from Optuna core**, so — like [LM-CMA](lm-cma.md),
[Differential Evolution](differential-evolution.md), and the [OpenAI-ES](openai-es.md) family — it is a
*beyond-core*, convergence-validated sampler with no Optuna counterpart. It **reuses the native GP stack**
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

Constructor knobs cover the whole schedule — `nInit` (per-restart Sobol initialization, default `2·d`),
`numCandidates`, `lengthInit` / `lengthMin` / `lengthMax`, `successTolerance` / `failureTolerance`,
`minimumNoise`, `deterministicObjective` — plus the shared `independentSampler`, `warnIndependentSampling`,
and `seed`.

Three properties worth knowing:

- **Continuous / integer only.** The trust-region box is a continuous construct, so the relative search
  space is Float/Int (Int normalized then snapped); categorical and single-valued parameters go to the
  independent sampler, as in [`CmaEsSampler`](cma-es.md). Single-objective only.
- **State is reconstructed, not persisted.** TuRBO stores no optimizer blob — the region length, resize
  counters, and restart boundaries are recomputed each trial by a deterministic fold over the completed
  trials' objective sequence, so a study resumes from storage with no serialized state.
- **Cost & candidate count.** The Thompson draw forms the joint posterior over the candidate set and
  Choleskys it (`O(numCandidates³)` per trial), so the default caps `numCandidates` at `min(100·d, 500)` to
  keep per-trial overhead near the GP-fit cost. The paper uses `min(100·d, 5000)` with BoTorch's scalable
  posterior sampler; raise `numCandidates` toward that for harder problems if you can afford it.

**When to reach for it.** TuRBO's edge over global [GP-BO](gp.md) shows on higher-dimensional and
multimodal objectives, where a single global GP over-explores — the trust region + restart concentrates the
budget locally and re-seeds when a basin is exhausted. On smooth low-dimensional problems plain
[`GPSampler`](gp.md) is already excellent and simpler.

## Validation

Behavioral, not bit-exact: GP fitting + Thompson sampling are stochastic and, like the rest of the GP
stack, tolerance-reproducible vs BoTorch rather than byte-for-byte (the documented GP boundary). The trust
-region mechanics (expand / shrink / restart, the length-scale-weighted box, the masked Sobol candidate
set, the Thompson argmax) are unit-tested directly; end-to-end the sampler converges on the sphere and
drives the multimodal Ackley well below a random baseline.

## Relationship to the lineage

- **From [GP-BO (2012)](gp.md):** reuses the identical Matérn-5/2 ARD GP surrogate and its fit; replaces the
  global acquisition optimization with a local trust region + Thompson sampling.
- **TuRBO-m (not yet ported):** the paper's full method maintains `m` trust regions with an implicit
  multi-armed bandit across them; TuRBO-1 is the single-region special case and the recommended default for
  most problems.

## See also

[GP-BO (2012)](gp.md) · [CMA-ES (2001)](cma-es.md) · [LM-CMA (2015)](lm-cma.md) ·
[OpenAI-ES (2017)](openai-es.md) · [Samplers](../samplers.md) · the [API reference](../../api/index.md).
