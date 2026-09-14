# Efficient optimization algorithms

OptunaSharp makes hyperparameter search efficient two ways: **samplers** that steer which parameter
values to try next, and **pruners** that stop unpromising trials early.

## Sampling algorithms

A sampler narrows the search using the history of tried parameters and their results, steering toward
regions that give better objective values (see [`BaseSampler`](../samplers.md) for the mechanism).
OptunaSharp ships:

| Sampler | Class | Idea |
|---------|-------|------|
| Grid Search | `GridSampler` | Every point of a user grid |
| Exhaustive Search | `BruteForceSampler` | Every point of a finite define-by-run space |
| Random Search | `RandomSampler` | Independent random draws |
| Tree-structured Parzen Estimator | `TPESampler` | Density-ratio model (**the default**) |
| CMA-ES | `CmaEsSampler` | Covariance-matrix adaptation evolution strategy |
| Gaussian process | `GPSampler` | Bayesian optimization with a GP surrogate |
| Partial-fixed | `PartialFixedSampler` | Pin some params, sample the rest with another sampler |
| NSGA-II | `NSGAIISampler` | Multi-objective genetic algorithm |
| Quasi-Monte-Carlo | `QMCSampler` | Low-discrepancy sequences |

The [Samplers guide](../samplers.md) documents them all (and OptunaSharp's beyond-Optuna additions);
the [Methods](../methods/index.md) section is the per-algorithm deep dive.

### Switching samplers

The default is `TPESampler`. Pass any sampler to `CreateStudy`:

```csharp
using OptunaSharp;
using OptunaSharp.Samplers;

var study = Optuna.CreateStudy();
Console.WriteLine($"Sampler is {study.Sampler.GetType().Name}");   // TPESampler

study = Optuna.CreateStudy(sampler: new RandomSampler());
Console.WriteLine($"Sampler is {study.Sampler.GetType().Name}");   // RandomSampler

study = Optuna.CreateStudy(sampler: new CmaEsSampler());
Console.WriteLine($"Sampler is {study.Sampler.GetType().Name}");   // CmaEsSampler
```

## Pruning algorithms

A [pruner](../pruners.md) automatically stops unpromising trials early (automated early stopping). It
is for single-objective optimization. OptunaSharp ships:

| Pruner | Class |
|--------|-------|
| Median stopping rule | `MedianPruner` |
| No-op (never prunes) | `NopPruner` |
| Patience wrapper | `PatientPruner` |
| Percentile rule | `PercentilePruner` |
| Asynchronous Successive Halving | `SuccessiveHalvingPruner` |
| Hyperband | `HyperbandPruner` |
| Threshold | `ThresholdPruner` |
| Wilcoxon signed-rank test | `WilcoxonPruner` |

Most examples use `MedianPruner`, but `SuccessiveHalvingPruner` and `HyperbandPruner` generally
outperform it.

### Activating a pruner

Report intermediate values during iterative training and check `ShouldPrune()`; throw `TrialPruned`
when it says so:

```csharp
using OptunaSharp;
using OptunaSharp.Pruners;

double Objective(Trial trial)
{
    double alpha = trial.SuggestFloat("alpha", 1e-5, 1e-1, log: true);

    double error = 1.0;
    for (int step = 0; step < 100; step++)
    {
        error = TrainOneStepAndValidate(alpha, step);   // your iterative training here
        trial.Report(error, step);                       // monitor the intermediate value

        if (trial.ShouldPrune())                          // decide whether to stop early
            throw new TrialPruned();
    }
    return error;
}

var study = Optuna.CreateStudy(pruner: new MedianPruner());   // the median stopping rule
study.Optimize(Objective, nTrials: 20);
```

Pruned trials stop before finishing all iterations; OptunaSharp records them as `Pruned` and logs
`"Trial <n> pruned."`.

## Which sampler and pruner should you use?

For non-deep-learning tasks, Optuna's benchmarks suggest:

- With `RandomSampler`, pair `MedianPruner`.
- With `TPESampler`, pair `HyperbandPruner`.

For deep learning, a rough guide (Ozaki et al., 2020):

| Parallel compute | Categorical / conditional params | Recommended |
|------------------|-----------------------------------|-------------|
| Limited | No | TPE; GP-EI if the space is low-dimensional and continuous |
| Limited | Yes | TPE; GP-EI if low-dimensional and continuous |
| Sufficient | No | CMA-ES, Random Search |
| Sufficient | Yes | Random Search or a genetic algorithm |

For framework-native pruning, see the training integrations — e.g. the
[LightGBM pruning callback](../../integrations/training/lightgbm.md) prunes without changing your
training loop.

## Relationship to Optuna

| Optuna (Python) | OptunaSharp |
|-----------------|-------------|
| `optuna.samplers.RandomSampler()` / `CmaEsSampler()` | `new RandomSampler()` / `new CmaEsSampler()` |
| `study.sampler.__class__.__name__` | `study.Sampler.GetType().Name` |
| `optuna.pruners.MedianPruner()` | `new MedianPruner()` |
| `trial.report(v, step)` / `trial.should_prune()` | `trial.Report(v, step)` / `trial.ShouldPrune()` |
| `raise optuna.TrialPruned()` | `throw new TrialPruned()` |

## See also

[Samplers](../samplers.md) · [Pruners](../pruners.md) · [Methods](../methods/index.md) ·
[Wilcoxon pruner recipe](../recipes/wilcoxon-pruner.md) · the [API reference](../../api/index.md).
