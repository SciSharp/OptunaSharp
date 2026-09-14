# OpenAI-ES (2017)

**Evolution Strategies as a Scalable Alternative to Reinforcement Learning** — Tim Salimans, Jonathan Ho,
Xi Chen, Szymon Sidor, Ilya Sutskever, OpenAI, 2017. arXiv:
[1703.03864](https://arxiv.org/abs/1703.03864).

OpenAI-ES is the simplest member of the **Natural Evolution Strategy (NES)** family and the isotropic
sibling of [SNES](snes.md) and [xNES](xnes.md). It optimizes a search distribution `N(θ, σ²I)` by
estimating the natural gradient of the *mean* θ from a population of random perturbations — no covariance,
no per-coordinate scale, just one global step size — which is exactly what makes it embarrassingly
parallel and robust on noisy, high-dimensional objectives.

## The idea

Blur the objective with Gaussian noise, `η(θ) = E_{ε∼N(0,I)}[f(θ + σε)]`, and ascend it. The score-function
estimator of `∇θ η` is a randomized finite difference:

```
∇θ η(θ) = (1/σ) · E_{ε∼N(0,I)}[ f(θ + σε) · ε ]
```

so one generation samples `ε₁…εₙ`, evaluates `Fᵢ = f(θ + σεᵢ)`, and steps
`θ ← θ + α · 1/(nσ) · Σᵢ Fᵢ εᵢ` (Algorithm 1 of the paper). OptunaSharp follows the widely-used
`estool` / `evolution-strategies-starter` recipe, which turns that bare estimator into a practical
optimizer with three refinements from §2.1:

- **Antithetic (mirrored) sampling** — the population is drawn as pairs `ε, −ε`, halving the estimator's
  variance (the population size is therefore even).
- **Centered-rank fitness shaping** — raw returns are replaced by their ranks mapped to `[-0.5, 0.5]`,
  then z-scored, making the step invariant to any monotone rescaling of the objective and immune to
  outliers.
- **Adam** on the estimated gradient (rather than plain SGD) for a well-conditioned step, with σ and the
  learning rate slowly annealed toward floors.

Because the whole update is finite-difference-like, its cost scales with the *intrinsic* dimension of the
problem, not the raw parameter count — the property the paper exploited to train large neural-network
policies across thousands of CPUs communicating only scalar returns.

## In OptunaSharp

OpenAI-ES is **absent from Optuna core**, so — like [LM-CMA](lm-cma.md),
[Differential Evolution](differential-evolution.md), and the [JANUS](janus.md) family — it is a
*beyond-core* sampler validated by convergence and by a live-numpy parity gate for its natural-gradient
math, not against an Optuna counterpart. Randomness flows through the NumPy-exact MT19937 `RandomState`
(never `System.Random`), so a seeded study is byte-reproducible.

`OpenAiEsSampler` drives an internal `OpenEsOptimizer` through the Optuna ask/tell protocol the same way
`CmaEsSampler` drives `Cma`: continuous Float/Int parameters form the relative search space, a generation
is spread across consecutive trials, and the optimizer state round-trips through trial system attrs.
Categorical and single-valued parameters are delegated to the independent sampler.

```csharp
using OptunaSharp;
using OptunaSharp.Samplers;

var study = Optuna.CreateStudy(sampler: new OpenAiEsSampler(seed: 42));
study.Optimize(t =>
{
    double s = 0;
    for (int i = 0; i < 8; i++) { double x = t.SuggestFloat($"x{i}", -5, 5); s += x * x; }
    return s;
}, nTrials: 600);
```

Constructor knobs cover the whole recipe — `sigma`, `learningRate`, `popsize`, `mirrored`, `weightDecay`,
`sigmaDecay` / `sigmaLimit`, `learningRateDecay` / `learningRateLimit`, and the Adam `beta1` / `beta2` /
`adamEpsilon` — plus the shared `nStartupTrials`, `independentSampler`, and `warnIndependentSampling`.

**Convergence character.** With a single global σ that is only slowly annealed, OpenAI-ES settles to a
*neighborhood* of the optimum within a typical Optuna budget rather than to machine precision — the
natural regime for noisy or rugged objectives. Two consequences worth knowing:

- **Prefer [SNES](snes.md) or [`CmaEsSampler`](cma-es.md)** when you need fine convergence on a smooth
  objective — a per-coordinate (SNES) or full (CMA-ES) covariance refines far tighter.
- **`weightDecay` is off by default.** In the transformed `[0,1]` search box the origin is a *corner*, so
  a nonzero L2 pull would bias the search toward it rather than toward a neutral center — set it only if
  you know your problem wants that regularization.

## Relationship to the lineage

- **To [SNES](snes.md) / [xNES](xnes.md):** all three are NES — they ascend the natural gradient of a
  Gaussian search distribution. OpenAI-ES fixes the covariance to `σ²I` (mean-only adaptation); SNES adds
  a per-coordinate σ; xNES adds a full covariance. OpenAI-ES swaps the classic NES rank *utilities* for
  centered ranks + an Adam step, the choices that made it scale in the RL setting.
- **To [CMA-ES](cma-es.md):** NES is the natural-gradient view that CMA-ES's rank-µ update approximates;
  OpenAI-ES is the minimal, covariance-free point of that family.

## See also

[SNES (2014)](snes.md) · [xNES (2014)](xnes.md) · [CMA-ES (2001)](cma-es.md) ·
[Differential Evolution (1997)](differential-evolution.md) · [Samplers](../samplers.md) ·
the [API reference](../../api/index.md).
