# REMBO (2016)

**Bayesian Optimization in a Billion Dimensions via Random Embeddings** — Ziyu Wang, Frank Hutter,
Masrour Zoghi, David Matheson, Nando de Freitas, JAIR 55 (2016) 361–387.
[doi:10.1613/jair.4806](https://doi.org/10.1613/jair.4806).

REMBO (Random EMbedding Bayesian Optimization) attacks the one thing plain [GP-BO](gp.md) cannot do:
scale to a high-dimensional search space. Its insight is that many real problems — hyperparameter tuning
especially — have a **low effective dimensionality**: only a few parameters actually move the objective,
but you do not know which. REMBO draws a *random* linear embedding into a small space and runs an ordinary
Bayesian optimizer there, so the surrogate only ever sees a handful of dimensions no matter how many real
parameters the study has. OptunaSharp ships it as a **meta-sampler** that wraps any inner sampler.

## The idea

Let the study have `D` real parameters. REMBO fixes, once, a random Gaussian matrix `A ∈ ℝ^{D×d}` with
i.i.d. `N(0, 1)` entries, where `d` (the *embedding dimension*) is chosen to be at least the guessed
**effective** dimensionality `d_e`. It then optimizes the low-dimensional function

```
g(y) = f(p_X(A · y)),   y ∈ Y = [−√d, √d]^d
```

with an inner Bayesian optimizer over the small box `Y`. Theorem 2 guarantees that, with probability one,
some `y⋆` maps to the true optimum whenever `d ≥ d_e`; Theorem 3 justifies the box `[−√d, √d]^d` (the
paper's practical choice, `ε = log(d)/√d`).

**The box projection `p_X`.** The real box is rescaled to `X = [−1, 1]^D`. When `A · y` lands outside `X`,
REMBO clips it per coordinate before evaluating — `p_X(z) = argmin_{u∈X} ‖u − z‖` is exactly per-dimension
clamping for a box. OptunaSharp then maps the clipped `[−1, 1]` value back onto each parameter's real
`[low, high]`, honoring integer, step, and log scaling through the same de-normalization the rest of the
library uses.

**Invariance.** Because the embedding is random and linear, REMBO is invariant to *adding* unimportant
dimensions (Theorem 4) and to *rotating* the objective (Theorem 6) — unlike axis-aligned variable-selection
methods, it does not require the important directions to line up with the coordinate axes.

**Raising the success rate.** A single embedding's box captures the optimum only with some probability;
with probability `δ` it misses. The paper's §3.1 gives two levers, both available here: run `k`
**interleaved** independent embeddings (the miss probability drops to `δ^k`), and/or increase `d` beyond
`d_e` (more embeddings of dimension `d_e` live inside a larger `d`, though a larger `d` also slows the inner
optimizer). Interleaving is the more reliable lever and, in practice, the one to reach for.

## In OptunaSharp

REMBO is **absent from Optuna core**, so — like [TuRBO](turbo.md), [LM-CMA](lm-cma.md), and the
[OpenAI-ES](openai-es.md) family — it is a *beyond-core*, convergence-validated sampler with no Optuna
counterpart. Randomness flows through the NumPy-exact MT19937 `RandomState` (never `System.Random`), so a
seeded study is reproducible under sequential optimization.

`RandomEmbeddingSampler` is a **meta-sampler**: the study keeps the user's `D` real parameters, while REMBO
maintains `d` latent coordinates internally. The inner optimizer runs against a private in-memory *shadow
study* whose trials are the latent coordinates of the real study's finished trials; each proposed latent
point is mapped back to the real parameters and its latent coordinates are stored in the trial's system
attributes so the shadow can be rebuilt after resumption.

```csharp
using OptunaSharp;
using OptunaSharp.Samplers;

// A 2-effective-dimension objective hidden inside 25 real parameters.
var study = Optuna.CreateStudy(
    sampler: new RandomEmbeddingSampler(nEmbeddingDims: 3, nEmbeddings: 3, seed: 42));
study.Optimize(t =>
{
    double sum = 0;
    for (int i = 0; i < 25; i++)
    {
        double x = t.SuggestFloat($"x{i}", -5, 5);
        if (i < 2) { double e = x - (i == 0 ? 2.0 : -3.0); sum += e * e; }  // only x0, x1 matter
    }
    return sum;
}, nTrials: 240);
```

Constructor knobs:

- **`nEmbeddingDims` (d)** — the one knob that matters, and the only required argument. Set it to at least
  the guessed effective dimensionality. There is no universal default because it encodes a belief about the
  problem; larger values raise the chance the box captures the optimum but slow convergence.
- **`nEmbeddings` (k)** — the number of interleaved independent embeddings (§3.1), default 1. Trials are
  assigned to embeddings round-robin. Prefer `k ≥ 3–4` for reliability (see *Validation*).
- **`kernel`** — which of REMBO's two kernels (§3.2) the inner optimizer uses (see the next paragraph). The
  default is `LowDimensional`.
- **`innerSampler`** — for the low-dimensional kernel, the Bayesian optimizer run in the latent box; defaults
  to a [`GPSampler`](gp.md) (the paper's choice). Any history-reconstructing sampler works —
  [`TPESampler`](tpe.md) is a cheaper pure-managed alternative. (Samplers that carry an optimizer blob across
  trials, like CMA-ES/ES, are not intended here; and the high-dimensional kernel ignores this argument.)
- **`nStartupTrials` / `numCandidates` / `minimumNoise` / `deterministicObjective`** — the high-dimensional
  kernel's GP knobs (random-startup length, projected candidates scored per proposal, GP noise floor,
  noise-free flag).
- **`independentSampler`** — handles parameters REMBO does not embed (single-valued always; categorical under
  the low-dimensional kernel); defaults to [`RandomSampler`](random.md).
- **`warnIndependentSampling`**, **`seed`**.

**The two kernels (§3.2).** REMBO can measure the inner GP's similarity in two spaces:

- **`LowDimensional`** (default) — the inner sampler models the objective as a function of the latent point
  `y` directly (a GP in `d` dimensions). Cheap and the paper's choice for very high extrinsic `D`, but it can
  waste effort on the large part of the box `Y` that projects onto the boundary of the real box (two far-apart
  `y`'s can map to nearly the same point).
- **`HighDimensional`** — the GP measures similarity in the projected real parameter space `p_X(A·y)`, so
  `y`'s that map to the same clipped point are treated as identical. This removes the boundary
  over-exploration (making a *single* embedding much more reliable), and — because the GP now lives in the
  real parameter space — it also **embeds categorical parameters** (the §3.2 categorical/Hamming kernel, via
  scale-and-round). It fits a GP in the real (embedded) space and maximizes expected improvement over the
  reachable subspace, so it suits moderate extrinsic `D`. It reuses the native GP stack (`GPRegressor`),
  exactly as [`TurboSampler`](turbo.md) does.

Properties worth knowing:

- **What is embedded.** Float and integer parameters are always embedded (integer, step, and log scaling
  included). Categorical parameters are embedded only under the `HighDimensional` kernel; under
  `LowDimensional` they — and single-valued parameters — go to the independent sampler, as in
  [`CmaEsSampler`](cma-es.md) / [`TurboSampler`](turbo.md). Single-objective only.
- **Every trial is embedded.** REMBO produces all real parameters of a trial from one latent point sampled
  once (through `SampleIndependent`), so even the very first trial's observation feeds the inner optimizer —
  there is no first-trial gap.
- **A single embedding is fragile — interleave, or use the high-dimensional kernel.** With `k = 1` the box may
  simply not contain the optimum (a true *box miss*, fixed only by interleaving / a larger `d`), and — under
  the low-dimensional kernel — the inner GP can additionally *over-explore* the region of `Y` that projects
  onto the boundary of `X` (the §3.2 caveat). The `HighDimensional` kernel removes the over-exploration
  failure; interleaved embeddings (§3.1) remove the box-miss failure. Both are inherent trade-offs, not
  defects.

**When to reach for it.** REMBO shines when `D` is large but only a few parameters matter and you cannot say
which — the regime where plain [GP-BO](gp.md) over-explores and degrades. On genuinely low-dimensional
problems, use [`GPSampler`](gp.md) or [`TurboSampler`](turbo.md) directly; on high-dimensional problems
where *all* dimensions matter, REMBO does not fail catastrophically but has no structure to exploit.

## Validation

Behavioral, not bit-exact (REMBO has no Optuna counterpart, and the inner GP is stochastic). The geometry —
the embedding matrix's determinism and `N(0,1)` entries, the latent box, the round-robin run assignment, and
the box-projection `x = p_X(A·y)` with integer/step/log/categorical de-normalization — is unit-tested
directly. End to end, the tests mirror the paper's own findings: a **single** embedding is fragile, so the
convergence gates use **interleaved** embeddings (or the high-dimensional kernel), which reliably recover a
2-effective-dimension optimum hidden in a 25-dimensional space and beat plain random search over the full
space; determinism holds under a fixed seed for sequential optimization.

**The paper's claims are proved directly.** A dedicated claims suite audits the theory against this
implementation: *Theorem 2* — the effective optimum is reachable through the random embedding (constructed
explicitly, and the effective submatrix is non-singular over a large ensemble); *Theorem 3* — the box
`Y = [−√d, √d]^d` captures the optimum with a rate that rises with `d` (≈67 % at `d = d_e = 2`, ≈100 % by
`d = 6`); *Theorem 4* — invariance to adding unimportant dimensions, which for this implementation is an
*exact, deterministic* identity (per-name-seeded rows make the trajectory for `D = 2` and `D = 30` bitwise
identical), and its corollary, exact independence of the extrinsic dimension `D` (the "billion dimensions"
claim); *Lemma 5 / Theorem 6* — rotation invariance (the algebra `f_R(R⁻¹A·y) = f(A·y)`, plus convergence on
a non-axis-aligned effective subspace where variable-selection methods fail); *Section 3.1* — interleaved
embeddings raise the box-capture probability (`δ^k`); and *Section 3.2* — the high-dimensional kernel escapes
a stall the low-dimensional kernel suffers from over-exploring the clipped boundary.

**Deliberate scope note.** OptunaSharp implements REMBO's core (Algorithm 2), both kernels of §3.2 (the
low-dimensional kernel over the latent `y` and the high-dimensional kernel over `p_X(A·y)`, the latter also
carrying categoricals via the §3.2 scale-and-round / categorical kernel), and the §3.1 interleaving lever.
The one piece not ported is the adaptive length-scale scheme (Algorithm 3), which tempers the inner GP's
length-scale bound when it exploits too much: it would require injecting bounds into the GP fit, and the
high-dimensional kernel addresses the same clipped-boundary over-exploration from a different angle, so
reliability comes from the kernel choice and interleaving instead.

## Relationship to the lineage

- **To [GP-BO (2012)](gp.md):** REMBO is standard Bayesian optimization run inside a randomly embedded
  low-dimensional box; the only change to the loop is the multiplication by `A` and the box projection.
- **To [TuRBO (2019)](turbo.md):** both scale BO to higher dimension, but by opposite means — TuRBO keeps a
  *local* trust region in the full space, while REMBO keeps a *global* search in a random *subspace*. They
  are complementary; TuRBO is the stronger general high-dimensional baseline, REMBO the method of choice when
  a genuine low effective dimension is suspected.
- **Successors (not ported):** HeSBO (hashing embeddings) and ALEBO refine REMBO's projection and box
  pathologies; REMBO is the readable canonical base.

## See also

[GP-BO (2012)](gp.md) · [TuRBO (2019)](turbo.md) · [TPE (2011)](tpe.md) · [CMA-ES (2001)](cma-es.md) ·
[Partial-Fixed](partial-fixed.md) · [Samplers](../samplers.md) · the [API reference](../../api/index.md).
