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

#### Conditional parameters (OptunaSharp extension)

Define-by-run objectives often suggest a parameter only under a branch — `beta1` only when
`optimizer == "adam"`. Optuna's GA samplers evolve the *intersection* search space (the parameters every
completed trial has), so such a parameter drops out after the first trial that lacks it and is then
resampled at random on every trial: no crossover, no mutation, no inheritance. `ConditionalParams`
(in `OptunaSharp.SearchSpace`) declares the activation structure, so the sampler keeps those parameters
in the relative search space and evolves them whenever they are active — and leaves them out entirely,
never blended and never counted, whenever they are not:

```csharp
using OptunaSharp.SearchSpace;

var rules = new ConditionalParams()
    .When("optimizer", "adam", "adamw").Activates("beta1", "beta2")   // rules on one activator OR together
    .When("optimizer", "sgd").Activates("momentum")
    .When("n_layers", v => (long)v! >= 3).Activates("width_3");        // a predicate for ranges

var sampler = new NSGAIIISampler(populationSize: 32, seed: 0, conditionalParams: rules);

study.Optimize(t =>
{
    string optimizer = t.SuggestCategorical("optimizer", new[] { "adam", "adamw", "sgd" });
    if (optimizer is "adam" or "adamw")
    {
        t.SuggestFloat("beta1", 0.5, 0.999);
        t.SuggestFloat("beta2", 0.9, 0.9999);
    }
    else
    {
        t.SuggestFloat("momentum", 0.0, 0.99);
    }
    // ...
});
```

How it behaves: a conditional parameter joins the relative search space once any completed trial has
suggested it (as long as its distribution never changes); crossover runs over the parameters the selected
parents share, a parameter only some parents carry is inherited from one of them, and every conditional
parameter that is inactive for the child is pruned before mutation, so it neither counts toward the
default `1 / n_params` mutation probability nor reaches the trial. The objective's own `Suggest*` calls
stay the final word, as always in define-by-run. Rules for the same activator OR together, rules for
different activators AND together, activators may themselves be conditional (chains), and cycles are
rejected. Values compare with Python `==` semantics (`1 == 1.0 == true`, strings ordinal); a predicate
receives the boxed `long` / `double` / raw choice value. The sampler warns once per parameter when a
completed trial contradicts the rules (present while inactive, or absent while active) — usually a sign
the objective's branches and the declaration drifted apart. Both `NSGAIISampler` and `NSGAIIISampler`
accept `conditionalParams`; with none declared (the default) they stay bit-identical to Optuna. There is
no Optuna analog for the GA samplers (for TPE, Optuna's own answer is `TPESampler(group: true)`).

`examples/ConditionalParameters.cs` prints the effect generation by generation: the same seeded NSGA-III
study with and without the rules, where the branch-only parameters converge on their optima with rules
and sit at the random-draw level without. `ConditionalParamsProofTests` pins the three guarantees: every
rule kind drives the sampler's own proposals, a parameter is learned while active, and a deactivated
parameter leaves the child and the RNG state bit-identical to a world in which it never existed.
`ConditionalParamsGateTests` is the null control — the effect happens only when you turn it on: with no
rules, an empty `ConditionalParams`, or rules that always admit, the study is byte-for-byte the Optuna
baseline (an empty rule set is a bit-exact no-op even on a branchy study, both samplers, copy-type and
arithmetic crossover); and activation is per-parameter — declaring one branch-only parameter leaves an
undeclared sibling exactly at its no-rules random-draw level.

##### Dormant-value memory (opt-in)

By default a reactivating branch is re-sampled from scratch: when `optimizer` flips back to `adam` and no
selected parent happened to carry `beta1`, `beta1` is drawn at random and everything learned about it last
time is lost. Set `PreserveDormantValues = true` on the rule set to keep it instead:

```csharp
var rules = new ConditionalParams { PreserveDormantValues = true }
    .When("optimizer", "adam").Activates("beta1")
    .When("optimizer", "sgd").Activates("momentum");
```

While a branch is inactive its parameter's last value is parked in a latent side channel (a trial system
attribute) — never suggested, never scored, never crossed or mutated, so it stays out of every equation
and does not change while dormant — and a reactivating branch **resumes from that value instead of a random
draw**, then evolves normally from there. The hyperparameter list stays fluid, but toggling a branch off
and on no longer throws its tuning away. This is a genuine divergence (Optuna re-samples), so it is off
unless you ask for it. Measured on a 256-trial two-branch study, it removes 40–60% of the random
reactivations; `ConditionalParamsDormantMemoryTests` pins the resume/park/no-op behavior and that effect.

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
- **CmaEsSampler** persists optimizer state as JSON, not Python `pickle`, so a study cannot be resumed
  *mid-CMA* across languages; the with-margin variant is tolerance-bounded rather than bit-exact.
- **GridSampler** may order not-yet-visited grid points differently under concurrency (cosmetic).
- **NSGAIIISampler** can differ on near-tie reference-point selection; **UNDXCrossover** uses a managed
  QR and diverges for 1-D search spaces.
- **ConditionalParams** (NSGA-II/III) is an OptunaSharp extension with no Optuna analog, off by default;
  its `PreserveDormantValues` mode is a deliberate divergence — a reactivating conditional branch resumes
  its parked value instead of the random re-sample Optuna would do. A parameter that is inactive for a
  child is math-neutral — out of selection, ranking, niching, the tournament, mutation, and the crossover
  operator (including the multi-parent operators' partial-carrier draw) — proven bit-for-bit against a
  "never existed" world (`ConditionalParamsMathNeutralityTests`). The single residual: the conditional
  crossover rejects out-of-box children and redraws, so a box-overshooting operator (**vSBXCrossover**) can
  let a parameter that is *active on a rejected attempt* shift the redraw stream — an active-parameter
  effect on a discarded attempt, not a deactivated one; rare and absent from the default `UniformCrossover`.

All are marked in the source and pinned by tests.

## See also

[Study and Trials](study-and-trials.md) · [Distributions](distributions.md) ·
[Search Spaces](search-spaces.md) · [Pruners](pruners.md) ·
[Multi-objective Optimization](multi-objective.md) · the [API reference](../api/index.md).
