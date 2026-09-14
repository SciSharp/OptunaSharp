# SNES (2014)

**Natural Evolution Strategies** (Separable NES, §5.1, Algorithm 7) — Daan Wierstra, Tom Schaul, Tobias
Glasmachers, Yi Sun, Jan Peters, Jürgen Schmidhuber, JMLR 15 (2014) 949–980.
[JMLR page](https://jmlr.org/papers/v15/wierstra14a.html).

SNES is the `O(d)` diagonal member of the **Natural Evolution Strategy (NES)** family — the practical
middle ground between the covariance-free [OpenAI-ES](openai-es.md) and the full-covariance
[xNES](xnes.md). It adapts a Gaussian with a *diagonal* covariance: one mean and one standard deviation
per coordinate, updated by natural-gradient steps.

## The idea

NES parameterizes a search distribution and ascends the **natural** gradient of the expected fitness —
the steepest-ascent direction in the Fisher-Rao metric, which makes the update invariant to linear
reparameterizations. Restricting the distribution to a *separable* (diagonal) Gaussian collapses the
per-sample cost to `O(d)`, so SNES scales to high-dimensional spaces where a full covariance is too
expensive.

Each generation samples `sₖ ∼ N(0, I)`, forms candidates `zₖ = μ + σ ⊙ sₖ`, ranks them, and turns the
ranks into **utility weights** `u₁ ≥ … ≥ u_λ` (the CMA-derived weighting, shifted to sum to zero so the
utilities are a built-in fitness baseline). The natural-gradient steps are then (Algorithm 7):

```
∇μ = Σₖ uₖ · sₖ                         μ ← μ + η_μ · σ ⊙ ∇μ
∇σ = Σₖ uₖ · (sₖ² − 1)                  σ ← σ ⊙ exp(η_σ/2 · ∇σ)
```

The mean moves additively (scaled by the current σ); each coordinate's σ moves *multiplicatively* through
the exponential map, so it can both shrink toward and grow away from the optimum as the landscape warrants.
The paper's defaults scale with the dimension: `λ = 4 + ⌊3 ln d⌋`, `η_μ = 1`, and
`η_σ = (3 + ln d)/(5√d)` — the separable case can afford a σ learning rate a factor of `d/3` larger than
[xNES](xnes.md)'s, which is what makes SNES adapt quickly.

## In OptunaSharp

SNES is **absent from Optuna core**, so — like [LM-CMA](lm-cma.md),
[Differential Evolution](differential-evolution.md), and the [JANUS](janus.md) family — it is a
*beyond-core* sampler validated by convergence and a live-numpy parity gate for its natural-gradient step,
not against an Optuna counterpart. Randomness flows through the NumPy-exact MT19937 `RandomState` (never
`System.Random`), so a seeded study is byte-reproducible.

`SnesSampler` drives an internal `SnesOptimizer` through the Optuna ask/tell protocol the same way
`CmaEsSampler` drives `Cma`. Categorical and single-valued parameters are delegated to the independent
sampler.

```csharp
using OptunaSharp;
using OptunaSharp.Samplers;

var study = Optuna.CreateStudy(sampler: new SnesSampler(seed: 42));
study.Optimize(t =>
{
    double s = 0;
    for (int i = 0; i < 10; i++) { double x = t.SuggestFloat($"x{i}", -5, 5); s += x * x; }
    return s;
}, nTrials: 500);
```

Constructor knobs: `sigma` (initial per-coordinate scale), `etaMu`, `etaSigma` (`null` ⇒ the Table-1
default), and `popsize` (`null` ⇒ the Table-1 default), plus the shared `nStartupTrials`,
`independentSampler`, and `warnIndependentSampling`.

**When to reach for it.** Because σ is adapted per coordinate, SNES converges to high precision on
*separable* (axis-aligned) problems — far tighter than the fixed-σ [OpenAI-ES](openai-es.md) — while
staying `O(d)`. Its blind spot is the mirror image: on strongly **non-separable** problems (a rotated or
curved valley such as Rosenbrock) the diagonal covariance cannot represent the correlation between
coordinates, and [xNES](xnes.md) or [`CmaEsSampler`](cma-es.md) will do better.

## Relationship to the lineage

- **To [OpenAI-ES](openai-es.md):** SNES adds a per-coordinate σ (and its natural-gradient update) on top
  of OpenAI-ES's mean-only adaptation, and uses the classic NES rank *utilities* rather than centered
  ranks + Adam.
- **To [xNES](xnes.md):** SNES is xNES with the covariance constrained to be diagonal — trading xNES's
  rotation invariance for an `O(d)` cost and a larger, faster σ learning rate.
- **To [CMA-ES](cma-es.md):** NES is the natural-gradient view CMA-ES's rank-µ update approximates; the
  separable NES is its diagonal analogue, akin to `sep-CMA-ES`.

## See also

[OpenAI-ES (2017)](openai-es.md) · [xNES (2014)](xnes.md) · [CMA-ES (2001)](cma-es.md) ·
[Samplers](../samplers.md) · the [API reference](../../api/index.md).
