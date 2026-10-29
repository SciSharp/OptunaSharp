# Samplers

A **sampler** decides which hyperparameter values to try next, given the history of the study. It is
the search strategy: random, grid, Bayesian (TPE / Gaussian process), evolutionary (CMA-ES, NSGA),
quasi-Monte-Carlo, or exhaustive. OptunaSharp ports Optuna's entire sampler surface into
**`OptunaSharp.Samplers`**, reproducing each algorithm's output bit-for-bit under a fixed seed wherever
the numeric path is deterministic.

## Choosing a sampler

```csharp
using OptunaSharp;
using OptunaSharp.Samplers;

var study = Optuna.CreateStudy(sampler: new TPESampler(seed: 42));
```

The default is `TPESampler` for single-objective studies and `NSGAIISampler` for multi-objective
ones — the same defaults as Optuna.

| Sampler | Best for | Constructor (key parameters) |
|---------|----------|------------------------------|
| `RandomSampler` | Baselines, cheap high-dimensional search | `(int? seed = null)` |
| `GridSampler` | Small, fully-enumerable grids | `(IReadOnlyDictionary<string, IReadOnlyList<CategoricalChoice>> searchSpace, int? seed = null)` |
| `TPESampler` | The general-purpose default; mixed spaces | `(nStartupTrials: 10, nEiCandidates: 24, multivariate: false, group: false, constantLiar: true, seed: null, …)` |
| `CmaEsSampler` | Continuous, moderately high-dimensional | `(nStartupTrials: 1, useSeparableCma: false, withMargin: false, lrAdapt: false, restartStrategy: null, seed: null, …)` |
| `GPSampler` | Expensive objectives, small trial budgets | `(nStartupTrials: 10, deterministicObjective: false, seed: null, …)` |
| `QMCSampler` | Low-discrepancy space filling | `(qmcType: "sobol", scramble: false, seed: null, …)` |
| `NSGAIISampler` | Multi-objective (2–3 objectives) | `(populationSize: 50, crossover: null, mutation: null, seed: null, …)` |
| `NSGAIIISampler` | Multi-objective (many objectives) | `(populationSize: 50, referencePoints: null, dividingParameter: 3, seed: null, …)` |
| `BruteForceSampler` | Exhaustive define-by-run search | `(int? seed = null, bool avoidPrematureStop = false)` |
| `PartialFixedSampler` | Fix some params, search the rest | `(IReadOnlyDictionary<string, object?> fixedParams, BaseSampler baseSampler)` |

Pass `seed:` for reproducibility. All RNG flows through a `LazyRandomState` built on NumSharp's
MT19937, never `System.Random` — this is what makes runs bit-identical to Python.

> **Reproducibility under parallelism.** A seeded sampler reproduces a run bit-for-bit only with
> `nJobs: 1`. When `study.Optimize(..., nJobs: n)` with `n != 1` spawns more than one worker, each
> worker **reseeds** its sampler — so parallel workers don't draw the same parameters — exactly as
> Optuna does, which makes a multi-worker run *not* reproducible. For distributed optimization across
> processes, for the same reason use either `seed: null` or a **different** seed value per process.

## Capabilities, cost, and budget

The nine samplers OptunaSharp ports 1-to-1 from Optuna support the following, reproducing Optuna's own
capability matrix. Read the marks as **✓** supported · **▲** works, but inefficiently · **✗** errors or
has no interface.

### Parameter types and search space

| Sampler | Float | Int | Categorical | Conditional | Multivariate |
|---------|:-----:|:---:|:-----------:|:-----------:|:------------:|
| `RandomSampler` | ✓ | ✓ | ✓ | ✓ | ▲ |
| `TPESampler` | ✓ | ✓ | ✓ | ✓ | ✓ |
| `GPSampler` | ✓ | ✓ | ✓ | ▲ | ✓ |
| `CmaEsSampler` | ✓ | ✓ | ▲ | ▲ | ✓ |
| `NSGAIISampler` | ▲ | ▲ | ✓ | ▲ | ▲ |
| `NSGAIIISampler` | ▲ | ▲ | ✓ | ▲ | ▲ |
| `GridSampler` | ✓ † | ✓ | ✓ | ▲ | ▲ |
| `QMCSampler` | ✓ | ✓ | ▲ | ▲ | ▲ |
| `BruteForceSampler` | ✓ † | ✓ | ✓ | ✓ | ▲ |

† `✗` for an infinite (unbounded) float domain.

### Optimization modes

| Sampler | Pruning | Multi-objective | Constrained | Batch | Distributed |
|---------|:-------:|:---------------:|:-----------:|:-----:|:-----------:|
| `RandomSampler` | ✓ | ✓ | ▲ | ✓ | ✓ |
| `TPESampler` | ✓ | ✓ | ✓ | ✓ | ✓ |
| `GPSampler` | ▲ | ✓ | ✓ | ✓ | ✓ |
| `CmaEsSampler` | ▲ | ✗ | ▲ | ✓ | ✓ |
| `NSGAIISampler` | ✗ ‡ | ✓ ‡ | ✓ | ✓ | ✓ |
| `NSGAIIISampler` | ✗ ‡ | ✓ ‡ | ✓ | ✓ | ✓ |
| `GridSampler` | ✓ | ✓ | ▲ | ✓ | ✓ |
| `QMCSampler` | ✓ | ✓ | ▲ | ✓ | ✓ |
| `BruteForceSampler` | ✓ | ✓ | ▲ | ✓ | ✓ |

‡ The NSGA samplers are multi-objective by design: `✗`/`✓` shown is for multi-objective use, but each
degrades to `▲` on a single-objective study.

### Cost and recommended budget

| Sampler | Time complexity per trial | Recommended budget (#trials) |
|---------|---------------------------|------------------------------|
| `RandomSampler` | O(*d*) | as many as you like |
| `TPESampler` | O(*dn* log *n*) | 100–1000 |
| `GPSampler` | O(*n*³) | up to ~500 |
| `CmaEsSampler` | O(*d*³) | 1000–10000 |
| `NSGAIISampler` | O(*mp*²) | 100–10000 |
| `NSGAIIISampler` | O(*mp*²) | 100–10000 |
| `GridSampler` | O(*dn*) | number of combinations |
| `QMCSampler` | O(*dn*) | as many as you like |
| `BruteForceSampler` | O(*dn*) | number of combinations |

where *d* is the search-space dimension, *n* the number of finished trials, *m* the number of
objectives, and *p* the population size. The complexities are for the sampling step in
single-objective optimization (the NSGA rows are the multi-objective exception); OptunaSharp also spends
O(*d*) to call the suggest methods and O(*n*) to collect finished trials, which the table omits.
The recommended budget includes any `nStartupTrials` a sampler takes, and depends on the number of
parameters and objectives.

Beyond these nine, OptunaSharp ports many more samplers — CMA-ES variants (separable, with-margin,
LM-CMA, VkD-CMA), the natural-evolution-strategy family (OpenAI-ES, SNES, xNES), trust-region and
random-embedding Bayesian optimization (TuRBO, REMBO, SAASBO), the JANUS infill family, and additional
MOEAs (SMS-EMOA, AGE-MOEA-2, MOEA/D, RVEA, PSO, Differential Evolution) — each documented on its own
[Methods](methods/index.md) page. Optuna's table also lists two samplers that live outside Core:
**AutoSampler** (not ported; reachable through the [OptunaHub bridge](../integrations/python/optunahub.md))
and **BoTorchSampler** (O(*n*³) per trial, 10–100 trials; available through the
[BoTorch bridge](../integrations/python/botorch.md)).

## The workhorses

### TPESampler

Tree-structured Parzen Estimator — Optuna's default. It models good and bad trials with kernel density
estimates and samples where the ratio favors improvement. Notable options:

```csharp
new TPESampler(
    nStartupTrials: 10,        // random trials before TPE kicks in
    nEiCandidates: 24,         // candidates scored per suggestion
    multivariate: false,       // model parameters jointly instead of independently
    group: false,              // with multivariate, model co-occurring params as groups
    constantLiar: true,        // penalize params of still-running trials (distributed runs)
    constraintsFunc: null,     // constrained optimization
    seed: 42);
```

`multivariate: true` (optionally with `group: true`) captures parameter interactions;
`constraintsFunc` enables constrained optimization; the sampler also drives MOTPE for
multi-objective studies.

### CmaEsSampler

Covariance Matrix Adaptation Evolution Strategy, for continuous spaces. One class covers all four
backends: full-covariance CMA (default), separable CMA (`useSeparableCma: true`), CMA-with-margin for
discrete/mixed spaces (`withMargin: true`), and warm-starting from prior trials
(`sourceTrials:`). Restart strategies and learning-rate adaptation (`lrAdapt: true`) are supported.

### GPSampler

Gaussian-process Bayesian optimization for expensive objectives with small budgets. The full stack —
kernel fitting, log-expected-improvement and its multi-objective / constrained / batched variants, and
the mixed continuous-discrete acquisition optimizer — is implemented natively (no TorchSharp). Set
`deterministicObjective: true` for noise-free objectives; `constraintsFunc:` for constraints.

### NSGA-II and NSGA-III

Genetic algorithms for multi-objective optimization. `NSGAIISampler` uses non-dominated sorting with
crowding distance; `NSGAIIISampler` uses reference points and suits many-objective problems. Both let
you plug in a crossover and mutation operator from `OptunaSharp.Samplers.Nsgaii`:

```csharp
using OptunaSharp.Samplers;
using OptunaSharp.Samplers.Nsgaii;

var sampler = new NSGAIISampler(
    populationSize: 64,
    crossover: new SBXCrossover(),
    mutation: new PolynomialMutation(eta: 20.0),
    seed: 42);
```

Available crossovers (all in `OptunaSharp.Samplers.Nsgaii`): `UniformCrossover`, `BLXAlphaCrossover`,
`SBXCrossover`, `VSBXCrossover`, `SPXCrossover`, `UNDXCrossover`; mutation: `PolynomialMutation`.
`NSGAIIISampler` additionally accepts `referencePoints` and `dividingParameter`.

### GridSampler and BruteForceSampler

`GridSampler` enumerates an explicit grid you provide and stops when it is exhausted
(`IsExhausted(study)` reports this). `BruteForceSampler` enumerates a *define-by-run* space discovered
from the objective itself — no grid needed — and is ideal for exhaustively covering a small
conditional space.

### QMCSampler and RandomSampler

`RandomSampler` samples independently and uniformly — a strong baseline. `QMCSampler` fills the space
with a low-discrepancy sequence (`qmcType: "sobol"` (default) or `"halton"`); `scramble: true` is
supported for both engines (scrambled Sobol' via the linear-matrix-scramble, scrambled Halton via
Owen's digit permutations), each bit-identical to Optuna/scipy.

### PartialFixedSampler

Wraps another sampler and pins a subset of parameters to fixed values, letting the base sampler search
the remainder — useful for freezing a known-good parameter while tuning others. The base sampler's RNG
continues uninterrupted across the fixed parameters.

## Writing a custom sampler

Derive from the abstract `BaseSampler` and implement the three-method contract Optuna uses:

```csharp
public abstract OrderedDict<string, BaseDistribution> InferRelativeSearchSpace(Study study, FrozenTrial trial);
public abstract OrderedDict<string, object?> SampleRelative(Study study, FrozenTrial trial, OrderedDict<string, BaseDistribution> searchSpace);
public abstract object? SampleIndependent(Study study, FrozenTrial trial, string paramName, BaseDistribution paramDistribution);
// optional lifecycle hooks: BeforeTrial, AfterTrial, ReseedRng
```

`SampleRelative` handles parameters modeled jointly (from the inferred [search space](search-spaces.md));
`SampleIndependent` handles the rest one at a time. Genetic samplers derive from `BaseGASampler`.

## Relationship to Optuna

| Optuna (Python) | OptunaSharp |
|-----------------|-------------|
| `TPESampler(multivariate=True, seed=42)` | `new TPESampler(multivariate: true, seed: 42)` |
| `CmaEsSampler(with_margin=True)` | `new CmaEsSampler(withMargin: true)` |
| `GPSampler(deterministic_objective=True)` | `new GPSampler(deterministicObjective: true)` |
| `NSGAIISampler(population_size=64)` | `new NSGAIISampler(populationSize: 64)` |
| `QMCSampler(qmc_type="sobol", scramble=True)` | `new QMCSampler(qmcType: "sobol", scramble: true)` |
| `QMCSampler(qmc_type="halton", scramble=True)` | `new QMCSampler(qmcType: "halton", scramble: true)` |

## Divergences

These are the documented, tested differences from Optuna:

- **TPESampler** rides the error-function path, so its float outputs match Python only to a few ULP
  (bit-exact off that path); certain argsort ties and repeated discrete observations in the
  multivariate estimator are order-sensitive.
- **CmaEsSampler**'s with-margin variant is tolerance-bounded rather than bit-exact. Its optimizer state
  is Optuna's own pickle, so C# and Python workers can share one CMA-ES study in any order; see
  [mixed fleets](storage.md#concurrency-workers-studies-and-languages-on-one-journal).
- **GridSampler** may order not-yet-visited grid points differently under concurrency (cosmetic).
- **NSGAIIISampler** can differ on near-tie reference-point selection; **UNDXCrossover** uses a managed
  QR and diverges for 1-D search spaces.

All are marked in the source and pinned by tests.

## See also

[Python sampler integrations](../integrations/python/index.md):
[BoTorch](../integrations/python/botorch.md), [PyCma](../integrations/python/pycma.md),
[pymoo](../integrations/python/pymoo.md), and [OptunaHub](../integrations/python/optunahub.md).

[Study and Trials](study-and-trials.md) · [Distributions](distributions.md) ·
[Search Spaces](search-spaces.md) · [Pruners](pruners.md) ·
[Multi-objective Optimization](multi-objective.md) · the [API reference](../api/index.md).
