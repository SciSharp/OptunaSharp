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
- **`innerSampler`** — the Bayesian optimizer run in the latent box; defaults to a [`GPSampler`](gp.md)
  (the paper's choice). Any history-reconstructing sampler works — [`TPESampler`](tpe.md) is a cheaper
  pure-managed alternative. (Samplers that carry an optimizer blob across trials, like CMA-ES/ES, are not
  intended as the inner sampler.)
- **`independentSampler`** — handles parameters REMBO does not embed (categorical, single-valued); defaults
  to [`RandomSampler`](random.md).
- **`warnIndependentSampling`**, **`seed`**.

Three properties worth knowing:

- **Float / integer are embedded; categorical is delegated.** A categorical cannot be recovered from a
  single projected scalar, so categorical and single-valued parameters go to the independent sampler, as in
  [`CmaEsSampler`](cma-es.md) / [`TurboSampler`](turbo.md). Integer, step, and log scaling *are* embedded.
  Single-objective only.
- **Every trial is embedded.** REMBO produces all real parameters of a trial from one latent point sampled
  once (through `SampleIndependent`), so even the very first trial's observation feeds the inner optimizer —
  there is no first-trial gap.
- **A single embedding is fragile — interleave.** With `k = 1` the box may simply not contain the optimum,
  and the inner GP can settle on a boundary-projected point (the paper's §3.2 caveat: the low-dimensional
  kernel "may waste time exploring in the region of the embedding outside of `X`"). This is inherent to the
  method, not a defect; interleaved embeddings are the fix.

**When to reach for it.** REMBO shines when `D` is large but only a few parameters matter and you cannot say
which — the regime where plain [GP-BO](gp.md) over-explores and degrades. On genuinely low-dimensional
problems, use [`GPSampler`](gp.md) or [`TurboSampler`](turbo.md) directly; on high-dimensional problems
where *all* dimensions matter, REMBO does not fail catastrophically but has no structure to exploit.

## Validation

Behavioral, not bit-exact (REMBO has no Optuna counterpart, and the inner GP is stochastic). The geometry —
the embedding matrix's determinism and `N(0,1)` entries, the latent box, the round-robin run assignment, and
the box-projection `x = p_X(A·y)` with integer/step/log de-normalization — is unit-tested directly. End to
end, the tests mirror the paper's own findings: a **single** embedding is fragile (it may miss the optimum),
so the convergence gates use **interleaved** embeddings, which reliably recover a 2-effective-dimension
optimum hidden in a 25-dimensional space and beat plain random search over the full space; determinism holds
under a fixed seed for sequential optimization.

**Deliberate scope note.** OptunaSharp implements REMBO's core (Algorithm 2) plus the §3.1 interleaving
lever. The paper's adaptive length-scale scheme (Algorithm 3), which tempers the inner GP for the embedding's
clipped plateaus, is *not* injected into the pluggable inner sampler; reliability comes from interleaving
instead. The high-dimensional (Hamming) kernel for categorical spaces (§3.2) is likewise not ported —
categoricals are delegated to the independent sampler.

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
