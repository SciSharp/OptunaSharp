# xNES (2014)

**Natural Evolution Strategies** (Exponential NES, §4, Algorithm 4) — Daan Wierstra, Tom Schaul, Tobias
Glasmachers, Yi Sun, Jan Peters, Jürgen Schmidhuber, JMLR 15 (2014) 949–980.
[JMLR page](https://jmlr.org/papers/v15/wierstra14a.html).

xNES is the **full-covariance** member of the **Natural Evolution Strategy (NES)** family — the sibling of
the covariance-free [OpenAI-ES](openai-es.md) and the diagonal [SNES](snes.md). It adapts a Gaussian
`N(μ, σ²·B·Bᵀ)` and is the NES most closely related to [CMA-ES](cma-es.md)'s rank-µ update.

## The idea

xNES factors the covariance as `A = σ·B` — a global step size `σ` and a unit-determinant transform `B` —
and updates `B` through the **matrix exponential** of a symmetric natural-gradient direction. That
exponential parameterization is what keeps the update invariant to linear transformations of the search
space (rotations included), the property a diagonal method like [SNES](snes.md) lacks.

Each generation samples `sₖ ∼ N(0, I)`, forms candidates `zₖ = μ + σ·(B·sₖ)`, ranks them, and computes the
NES utilities `u₁ ≥ … ≥ u_λ`. From those it assembles a mean gradient, a full "covariance" gradient, and
splits the latter into a step-size part (its trace) and a shape part (traceless):

```
∇δ = Σₖ uₖ · sₖ                         ∇M = Σₖ uₖ · (sₖ sₖᵀ − I)
∇σ = tr(∇M)/d                           ∇B = ∇M − ∇σ·I

μ ← μ + η_μ · σ · (B·∇δ)
σ ← σ · exp(η_σ/2 · ∇σ)                 B ← B · exp(η_B/2 · ∇B)
```

The `exp(·)` on `B` is a *matrix* exponential of the symmetric direction `∇B`, computed from its
eigendecomposition (`exp(QΛQᵀ) = Q exp(Λ) Qᵀ`). The paper's defaults are `λ = 4 + ⌊3 ln d⌋`, `η_μ = 1`,
and `η_σ = η_B = (9 + 3 ln d)/(5 d √d)`.

## In OptunaSharp

xNES is **absent from Optuna core**, so — like [LM-CMA](lm-cma.md),
[Differential Evolution](differential-evolution.md), and the [JANUS](janus.md) family — it is a
*beyond-core* sampler validated by convergence, not against an Optuna counterpart. Randomness flows through
the NumPy-exact MT19937 `RandomState` and the matrix exponential / reconstruction solve run on the
OpenBLAS-backed linear algebra (`eigh` / `solve`) already used by [CMA-ES](cma-es.md) and the GP stack, so
a seeded study is byte-reproducible.

`XnesSampler` drives an internal `XnesOptimizer` through the Optuna ask/tell protocol the same way
`CmaEsSampler` drives `Cma`; its `d×d` transform `B` is chunked across trial system attrs when the state
exceeds the storage limit. Categorical and single-valued parameters are delegated to the independent
sampler.

```csharp
using OptunaSharp;
using OptunaSharp.Samplers;

// Rosenbrock's curved valley is non-separable — xNES's full covariance follows it, SNES cannot.
var study = Optuna.CreateStudy(sampler: new XnesSampler(seed: 42));
study.Optimize(t =>
{
    double x = t.SuggestFloat("x", -5, 5);
    double y = t.SuggestFloat("y", -5, 5);
    return (x - 1) * (x - 1) + 100 * (y - x * x) * (y - x * x);
}, nTrials: 600);
```

Constructor knobs: `sigma`, `etaMu`, `etaSigma` / `etaB` (`null` ⇒ the Table-1 defaults), and `popsize`,
plus the shared `nStartupTrials`, `independentSampler`, and `warnIndependentSampling`.

**When to reach for it.** xNES captures correlations between parameters that the diagonal
[SNES](snes.md) cannot — so it is the NES to use on rotated / correlated objectives — at `O(d²)`–`O(d³)`
cost per generation. That said, OptunaSharp already ships mature full-covariance adaptation:
[`CmaEsSampler`](cma-es.md) is the better-tuned default for most correlated continuous problems,
[VkD-CMA](vkd-cma.md) restricts the covariance to a rank-k + diagonal form for high dimension, and
[SNES](snes.md) is cheaper still when coordinates are roughly independent. xNES is provided for
completeness of the NES story and its clean connection to CMA-ES.

**Implementation note.** The JMLR Algorithm-4 box writes the sample as `μ + σ·Bᵀ·s` but the mean update as
`μ + η·σB·∇δ`; those agree only when `B` is symmetric (a transpose slip in the linearized box).
OptunaSharp uses `B` consistently in both — `z = μ + σ·B·s` and `μ ← μ + η_μ·σ·B·∇δ` — matching the
reference pybrain / evosax implementations, so the sample map and the mean-gradient map are the same
operator `σB`.

## Relationship to the lineage

- **To [SNES](snes.md):** xNES generalizes SNES's diagonal covariance to a full one, regaining rotation
  invariance at a higher cost (and a smaller σ learning rate).
- **To [OpenAI-ES](openai-es.md):** xNES adds the full covariance `B` on top of OpenAI-ES's mean-only,
  isotropic model, and keeps the classic NES rank utilities.
- **To [CMA-ES](cma-es.md):** the natural-gradient update of xNES and CMA-ES's rank-µ update are the same
  idea derived two ways; the main differences are xNES's exponential parameterization of `B` and CMA-ES's
  evolution paths (which xNES omits).

## See also

[SNES (2014)](snes.md) · [OpenAI-ES (2017)](openai-es.md) · [CMA-ES (2001)](cma-es.md) ·
[VkD-CMA (2016)](vkd-cma.md) · [Samplers](../samplers.md) · the [API reference](../../api/index.md).
