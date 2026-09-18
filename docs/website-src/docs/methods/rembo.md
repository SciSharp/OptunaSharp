# REMBO (2016)

**Bayesian Optimization in a Billion Dimensions via Random Embeddings** — Ziyu Wang, Frank Hutter,
Masrour Zoghi, David Matheson, Nando de Freitas, JAIR 55 (2016) 361–387.
[doi:10.1613/jair.4806](https://doi.org/10.1613/jair.4806).

REMBO (Random EMbedding Bayesian Optimization) attacks the one thing plain [GP-BO](gp.md) cannot do: scale
to a high-dimensional search space. Its insight is that many real problems — hyperparameter tuning
especially — have a **low effective dimensionality**: only a few parameters actually move the objective, but
you do not know which. REMBO draws a *random* linear embedding into a small space and runs an ordinary
Bayesian optimizer there, so the surrogate only ever sees a handful of dimensions no matter how many real
parameters the study has. OptunaSharp ships it as a **meta-sampler** that wraps any inner optimizer.

## The idea

Let the study have `D` real parameters. REMBO fixes, once, a random Gaussian matrix `A ∈ ℝ^{D×d}` with
i.i.d. `N(0, 1)` entries, where `d` (the *embedding dimension*) is chosen to be at least the guessed
**effective** dimensionality `d_e`. It then optimizes the low-dimensional function

```
g(y) = f( p_X(A·y) ),    y ∈ Y = [−√d, √d]^d
```

with an inner Bayesian optimizer over the small box `Y`. Theorem 2 guarantees that, with probability one,
some `y⋆` maps to the true optimum whenever `d ≥ d_e`. Theorem 3 bounds *how large* `Y` must be to still
contain that optimizer (a ball of radius `√(d_e)/ε · ‖x⋆‖`, w.p. `1 − ε`); the specific box `[−√d, √d]^d` is
**not** from Theorem 3 — the authors note it "is not useful for this choice" — but from a simulation study,
taking `ε = log(d)/√d` in their empirical `‖B⁻¹x‖∞` bound.

```
   each trial
      │
      ▼
  assign an embedding:  run = trial.Number mod k        (k independent interleaved embeddings, §3.1)
      │
      ▼
  sample ONE latent point y ∈ Y = [−√d, √d]^d      (cached — every real parameter of the trial maps from it)
      │
      ├─ LowDimensional kernel:  the inner sampler (a GP by default) proposes y from this
      │        embedding's latent history — a GP in the d-dimensional latent space
      │
      └─ HighDimensional kernel: fit a GP in the REAL parameter space over this embedding's history,
      │        then maximize expected improvement over projected candidates → take the argmax y   (§3.2)
      ▼
  map every real parameter through the embedding:   xᵢ = de-normalize( p_X( Aᵢ · y ) )            (§3)
      │        Aᵢ = parameter i's projection row (d i.i.d. N(0,1), seeded by (run, name))
      │        p_X = clip A·y into the rescaled box [−1, 1]^D, per coordinate  (the box projection)
      │        float/int → affine · log · step de-normalize;  categorical → scale-and-round        (§3.2)
      ▼
  evaluate f(x); the observation feeds THIS embedding's inner optimizer for its next trial
```

**The box projection `p_X`.** The real box is rescaled to `X = [−1, 1]^D`. When `A·y` lands outside `X`,
REMBO clips it per coordinate before evaluating — `p_X(z) = argmin_{u∈X} ‖u − z‖` is exactly per-dimension
clamping for a box. OptunaSharp then maps the clipped `[−1, 1]` value back onto each parameter's real
`[low, high]`, honoring integer, step, and log scaling through the same de-normalization the rest of the
library uses. Because `A·y` for a latent point near the edge of `Y` is typically well outside `[−1, 1]`, much
of `Y` maps onto the *boundary* of `X` — a fact that shapes the two kernels below.

**The two kernels (§3.2).** REMBO can measure the inner GP's similarity in either of two spaces, and the
choice is the main reliability dial:

- **Low-dimensional kernel** (default) — the inner sampler models `g` as a function of the latent point `y`
  directly, a GP in `d` dimensions: `k^d_ℓ(y₁, y₂) = K_SE((y₁ − y₂)/ℓ)`. Cheap — and the only feasible option
  at very high extrinsic `D`, where the high-dimensional kernel's `D`-shaped GP is out of reach. Its weakness
  is the boundary: two far-apart `y`'s that both project onto (nearly) the same clipped point look *distant*
  to this kernel, so it can waste budget re-exploring the flat region of `Y` outside `X`.
- **High-dimensional kernel** — the GP measures similarity in the *projected* real space,
  `k^D_ℓ(y₁, y₂) = K_SE((p_X(A·y₁) − p_X(A·y₂))/ℓ)`. Since `p_X(A·y)` is precisely the real parameter vector
  REMBO evaluates, this is simply a GP over the real (embedded) parameters, with the acquisition (expected
  improvement) maximized over the reachable subspace by scoring projected candidate `y`'s. The two `y`'s
  above now look *identical* — the boundary over-exploration disappears — and because the GP lives in the
  real parameter space it also **embeds categorical parameters** (the §3.2 categorical / Hamming kernel, via
  the GP's categorical equality kernel). It fits a GP in `D`-shaped inputs, so it suits *moderate* extrinsic
  `D`; it reuses the native GP stack (`GPRegressor`) exactly as [`TurboSampler`](turbo.md) does.

**Raising the success rate (§3.1).** A single embedding's box captures the optimum only with some
probability; with probability `δ` it misses. Two levers, both available here, drive `δ` down: run `k`
**interleaved** independent embeddings (the miss probability across all of them is `δ^k`), and/or increase
`d` beyond `d_e` (more `d_e`-dimensional embeddings live inside a larger `d`, though a larger `d` also slows
the inner optimizer). Interleaving is the more reliable lever and the one to reach for.

**Invariances.** Because the embedding is random and linear, REMBO is invariant to *adding* unimportant
dimensions (Theorem 4) and to *rotating* the objective (Theorem 6) — unlike axis-aligned variable-selection
methods, it does not require the important directions to line up with the coordinate axes. (In this
implementation the first is even an *exact* identity — see [Validation](#validation).)

**Two kinds of single-embedding failure.** A `k = 1` run can fall short two ways, with two different fixes: a
*box miss* (no `y ∈ Y` maps to the optimum — cured by interleaving or a larger `d`, §3.1), and, under the
low-dimensional kernel, *boundary over-exploration* (cured by the high-dimensional kernel, §3.2). They are
inherent trade-offs of the method, not implementation defects.

## In OptunaSharp

REMBO is **absent from Optuna core**, so — like [TuRBO](turbo.md), [LM-CMA](lm-cma.md), and the
[OpenAI-ES](openai-es.md) family — it is a *beyond-core*, convergence-validated sampler with no Optuna
counterpart. Randomness flows through the NumPy-exact MT19937 `RandomState` (never `System.Random`), so a
seeded study is reproducible under sequential optimization. The high-dimensional kernel **reuses the native
GP stack** ([GP-BO](gp.md)'s `GPRegressor` + `LogEi` + normalized search space) unchanged — no new surrogate
machinery is introduced.

`RandomEmbeddingSampler` is a **meta-sampler**: the study keeps the user's `D` real parameters, while REMBO
maintains `d` latent coordinates internally. Under the low-dimensional kernel the inner optimizer runs
against a private in-memory *shadow study* whose trials are the latent coordinates of the real study's
finished trials; each proposed latent point is mapped back to the real parameters, and its latent coordinates
are stored in the trial's system attributes so the shadow can be rebuilt after resumption.

```csharp
using OptunaSharp;
using OptunaSharp.Samplers;

// A 2-effective-dimension objective hidden inside 25 real parameters. Interleaved embeddings (k = 3)
// make a robust single call; the optimum (x0, x1) = (2, -3) is off-center, so the box must capture it.
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

**Constructor knobs.** The one that matters is `nEmbeddingDims`; the rest tune the two levers and (for the
high-dimensional kernel) the GP (`src` cites the paper section):

```
knob                     default                    controls
───────────────────────  ─────────────────────────  ─────────────────────────────────────────────────────
nEmbeddingDims (d)        (required)                 embedding dimension; set ≥ the guessed effective dim d_e
nEmbeddings (k)           1                          §3.1 · interleaved independent embeddings (round-robin)
kernel                   LowDimensional             §3.2 · LowDimensional (GP over y) | HighDimensional
innerSampler             GPSampler                  low-dim kernel: the BO run in the latent box (TPE works too)
independentSampler       RandomSampler              parameters REMBO does not embed (see below)
warnIndependentSampling  true                       warn when a parameter is sampled independently
seed                     OS entropy                 reproducible trajectory (sequential) + fixes the matrix A
   ── high-dimensional kernel only ──
nStartupTrials           10                          random latent points before an embedding's GP takes over
numCandidates            min(5000, max(1000, 200·d)) §3.3 · projected candidates scored per proposal
minimumNoise             GP-stack default            GP observation-noise-variance floor
deterministicObjective   false                       treat the objective as noise-free
```

Properties worth knowing:

- **What is embedded.** Float and integer parameters are always embedded (integer, step, and log scaling
  included). Categorical parameters are embedded only under the **high-dimensional** kernel; under the
  low-dimensional kernel they — and single-valued parameters — go to the `independentSampler`, as in
  [`CmaEsSampler`](cma-es.md) / [`TurboSampler`](turbo.md). Single-objective only.
- **Every trial is embedded.** REMBO produces all real parameters of a trial from one latent point sampled
  once (through `SampleIndependent`), so even the very first trial's observation feeds the inner optimizer —
  there is no first-trial gap.
- **The matrix is generated, not stored.** Each parameter's projection row is `d` i.i.d. `N(0,1)` values
  drawn from a stream seeded by `(seed, run, name)` — a pure function of identity. It therefore survives
  parameter-discovery order, study resumption, and parallel workers with nothing persisted, while remaining
  distributionally identical to the paper's single matrix `A`.
- **`d` is required.** It encodes a belief about the problem's effective dimensionality and has no honest
  universal default; larger `d` raises the box-capture rate but slows the inner optimizer.

**When to reach for it.** REMBO shines when `D` is large but only a few parameters matter and you cannot say
which — the regime where plain [GP-BO](gp.md) over-explores and degrades. Prefer the **high-dimensional
kernel** (or `k ≥ 3` interleaving) at moderate `D` for reliability; keep the **low-dimensional kernel** for
very large `D` where a `D`-shaped GP is too costly. On genuinely low-dimensional problems, use
[`GPSampler`](gp.md) or [`TurboSampler`](turbo.md) directly; on high-dimensional problems where *all*
dimensions matter, REMBO does not fail catastrophically but has no structure to exploit.

## Validation

Behavioral, not bit-exact (REMBO has no Optuna counterpart, and the inner GP is stochastic). The geometry —
the embedding matrix's determinism and `N(0,1)` entries, the latent box, the round-robin run assignment, and
the box projection `x = p_X(A·y)` with integer / step / log / categorical de-normalization — is unit-tested
directly. End to end, the tests mirror the paper's own findings: a single embedding is fragile, so the
convergence gates use **interleaved** embeddings (or the high-dimensional kernel), which recover a
2-effective-dimension optimum hidden in a 25-dimensional space and beat plain random search over the full
space; determinism holds under a fixed seed for sequential optimization.

**The paper's claims are proved directly.** A dedicated claims suite audits the theory against this
implementation:

- **Theorem 2** — the effective optimum is reachable through the random embedding (its latent preimage is
  constructed explicitly, and the effective submatrix is non-singular over a large ensemble — its
  "probability 1" full rank).
- **Theorem 3 (box capture)** — the box `Y = [−√d, √d]^d` contains the optimizer's preimage increasingly
  often as `d` grows (≈ 67 % at `d = d_e = 2`, ≈ 99.5 % at `d = 4`, ≈ 100 % by `d = 6`) — the empirical
  capture behind Theorem 3's `1 − ε` guarantee and the §3.1 "increase `d`" lever.
- **Theorem 4** — invariance to adding unimportant dimensions. Because rows are seeded by name, this is an
  *exact, deterministic* identity here: the trajectory for `D = 2` and `D = 30` is bitwise identical. Its
  corollary — independence of the extrinsic dimension `D`, the "billion dimensions" claim — is likewise exact
  (`D = 50` and `D = 500` yield an identical run).
- **Lemma 5 / Theorem 6** — rotation invariance: the algebra `f_R(R⁻¹A·y) = f(A·y)`, plus convergence on a
  non-axis-aligned effective subspace (a diagonal direction) where variable-selection methods fail.
- **Section 3.1** — interleaved embeddings raise the box-capture probability (`δ^k`); the `k` embeddings are
  nested here, so capture is provably monotone in `k`.
- **Section 3.2** — the high-dimensional kernel escapes a stall the low-dimensional kernel suffers from
  over-exploring the clipped boundary (on a seed where the low-dimensional kernel stalls one effective
  coordinate at ≈ 4, the high-dimensional kernel converges).

## Differences from the paper

REMBO's *algorithmic* core is faithful — the random Gaussian embedding, the box `Y = [−√d, √d]^d`, the
projection `g(y) = f(p_X(A·y))`, both §3.2 kernel structures (low-dimensional over `y`; high-dimensional over
`p_X(A·y)`), the §3.2 categorical Hamming mechanism, the §3.1 interleaving and "increase `d`" levers, and the
invariances. The differences are concentrated in the *GP machinery*, where OptunaSharp reuses its own stack
rather than re-deriving the paper's:

- **Embedding matrix generation.** The paper draws one `D×d` matrix `A` up front. OptunaSharp generates each
  parameter's row lazily from a `(seed, run, name)`-seeded stream — distributionally identical i.i.d.
  `N(0,1)`, but order-independent, resumption-safe, and concurrency-safe with nothing persisted (and it makes
  Theorem 4 / D-independence *exact* identities here rather than merely holding in distribution).
- **Surrogate kernel.** The paper uses an isotropic **squared-exponential** kernel with a single length scale
  (Def. 7). OptunaSharp reuses the stack's **Matérn-5/2 ARD** kernel (per-coordinate length scales). The
  §3.2 categorical kernel is likewise the stack's Hamming-mismatch categorical kernel (per-coordinate
  indicator, ARD-weighted, in the Matérn-5/2 form) rather than the paper's `exp(−λ²·h²)` with a single `λ`.
- **Acquisition.** The paper maximizes **Expected Improvement** with DIRECT + CMA-ES ("tens of thousands of
  evaluations"). OptunaSharp maximizes **log-EI** (same argmax) — under the low-dimensional kernel by the
  inner sampler's mixed Sobol + gradient acquisition optimizer, and under the high-dimensional kernel by
  argmax over a scrambled-Sobol candidate set (default `min(5000, max(1000, 200·d))`, fewer than "tens of
  thousands", and a discrete scan rather than a continuous DIRECT/CMA-ES optimization).
- **Length-scale adaptation (Algorithm 3) — not implemented.** The paper's scheme shrinks the length-scale
  *upper bound* when the search exploits too much (`[L, U] = [0.01, 50]`, `t_σ = 0.002`, re-fit every 20
  iterations or after 5 exploiting steps). OptunaSharp instead fits hyperparameters by the stack's soft-prior
  marginal likelihood; the high-dimensional kernel addresses the same clipped-boundary over-exploration from
  a different angle, so reliability comes from the kernel choice and interleaving.
- **Startup and generality.** A short random startup (`nStartupTrials`, default 10) precedes each embedding's
  GP — the paper's Algorithm 2 leaves the initial design unspecified — and the low-dimensional kernel accepts
  *any* inner sampler (e.g. [`TPESampler`](tpe.md)), a generalization beyond the paper's GP-only loop.
- **Integration.** The paper is a standalone BO loop; OptunaSharp exposes REMBO as a define-by-run
  meta-sampler (the low-dimensional inner optimizer runs against a private in-memory *shadow study* of latent
  trials), and flips the sign for OptunaSharp's minimize-by-default convention. Neither changes the algorithm.

## Relationship to the lineage

- **To [GP-BO (2012)](gp.md):** REMBO is standard Bayesian optimization run inside a randomly embedded
  low-dimensional box; the only change to the loop is the multiplication by `A` and the box projection. The
  high-dimensional kernel is literally GP-BO's GP over the real parameters, restricted to the embedded
  subspace.
- **To [TuRBO (2019)](turbo.md):** both scale BO to higher dimension, but by opposite means — TuRBO keeps a
  *local* trust region in the full space, while REMBO keeps a *global* search in a random *subspace*. They
  are complementary; TuRBO is the stronger general high-dimensional baseline, REMBO the method of choice when
  a genuine low effective dimension is suspected. (REMBO's high-dimensional kernel reuses the same GP stack
  TuRBO does.)
- **Successors (not ported):** HeSBO (hashing embeddings) and ALEBO refine REMBO's projection and box
  pathologies; REMBO is the readable canonical base.

## See also

[GP-BO (2012)](gp.md) · [TuRBO (2019)](turbo.md) · [TPE (2011)](tpe.md) · [CMA-ES (2001)](cma-es.md) ·
[Partial-Fixed](partial-fixed.md) · [Samplers](../samplers.md) · the [API reference](../../api/index.md).
