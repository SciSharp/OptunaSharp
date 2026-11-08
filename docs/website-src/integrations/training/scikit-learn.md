# Native scikit-learn and OptunaSearchCV

OptunaSharp includes a managed regression backend that reproduces a focused part of scikit-learn:
`RandomForestRegressor`, cross-validation splitting, regression scoring and `OptunaSearchCV`.
These APIs live in the core **`OptunaSharp`** package; they do not load Python or require scikit-learn
to be installed.

`OptunaSearchCV` lives in `OptunaSharp.Integration`; the native estimator, forest, splitters and
scorers live in `OptunaSharp.Integration.Sklearn`. The native backend implements
`ICrossValidationEstimator`, the same contract used by the [ML.NET integration](ml-net.md).

## Run a complete search

```shell
dotnet add package OptunaSharp
```

```csharp
using System;
using System.Collections.Generic;
using OptunaSharp.Distributions;
using OptunaSharp.Integration;
using OptunaSharp.Integration.Sklearn;

var x = new double[30, 2];
var y = new double[30];
for (int i = 0; i < 30; i++)
{
    x[i, 0] = i / 29.0;
    x[i, 1] = i % 3;
    y[i] = 2 * x[i, 0] + 0.2 * x[i, 1];
}

var estimator = new SklearnEstimator(
    new RandomForestRegressor(nEstimators: 20, randomState: 42),
    x, y,
    cv: new KFold(nSplits: 3, shuffle: true, randomState: 42),
    scorer: Scorers.NegMeanSquaredError);

var search = new OptunaSearchCV(estimator,
    new Dictionary<string, BaseDistribution>
    {
        ["max_depth"] = new IntDistribution(2, 8),
        ["min_samples_leaf"] = new IntDistribution(1, 4)
    },
    nTrials: 10,
    randomState: 42,
    refit: true,
    returnTrainScore: true);

search.Fit();
Console.WriteLine($"Best validation score: {search.BestScore}");
double[] predictions = estimator.Predict(new double[,] { { 0.5, 1.0 } });
Console.WriteLine(predictions[0]);
```

The estimator owns its training data and validation strategy, so `Fit()` has no `X` or `y` arguments.
Each trial configures an unfitted estimator with the sampled Python-style parameter names, then fits
fresh models on the folds. With `refit: true`, the best configuration is trained on all supplied data.
Call `Predict` and `Score` on the concrete estimator after refit; `OptunaSearchCV` itself does not
forward prediction methods.

## Search behavior and results

`OptunaSearchCV` always maximizes the mean validation score. Use negative loss, such as
`Scorers.NegMeanSquaredError`, when smaller raw errors are better. Without a supplied study it creates
a maximize study with `TPESampler`; `randomState` seeds a generator that derives the sampler seed.
A supplied study must also maximize. Its sampler is used as configured.

Defaults are 10 trials, one job, no timeout, refit enabled and training scores disabled. `nTrials: null`
removes the count limit; `timeout` is in seconds. `nJobs: -1` uses the processor count. Custom
estimators used with parallel jobs must support concurrent `CrossValidate` calls without sharing
mutable fitted models. A seed does not make parallel trial ordering deterministic.

After fitting, inspect `Study`, `NSplits`, `BestParams`, `BestScore`, `BestTrial`, `BestIndex`, `Trials`,
`NTrials`, `UserAttrs` and `RefitTimeSeconds`. `CvResults` collects trial user-attribute columns,
including per-split test scores, mean/std test scores and fit/score timings; train-score columns
appear when requested. Its column names are taken from the first trial's user attributes, so it is
not the complete scikit-learn `cv_results_` schema with ranks and parameter columns.

Scores use NaN-aware mean/std aggregation. Fold errors in the native backend receive `errorScore`
(default `double.NaN`); an all-NaN objective cannot become a completed best trial. A numeric fallback
score participates in optimization like any other score, so choose it consistently with maximization.
The search also reports fold scores to the terminator. `callbacks` run after trials, and `@catch`
specifies exception types for which the study continues. `verbose` is accepted but reserved.

## Native estimator scope

The supplied forest is a regression implementation using bootstrap-aggregated squared-error CART.
Its public constructor and `WithParams` cover:

| Parameter names | Meaning |
|---|---|
| `n_estimators`, `max_depth` | Tree count and maximum depth |
| `min_samples_split`, `min_samples_leaf` | Split/leaf sample thresholds; `WithParams` also accepts fractions |
| `min_weight_fraction_leaf` | Minimum weighted fraction per leaf |
| `max_features` | All features, a count, a fraction, `"sqrt"` or `"log2"` |
| `min_impurity_decrease` | Minimum improvement required to split |
| `bootstrap`, `max_samples` | Bootstrap enablement and draw size |
| `random_state` | Forest random seed |
| `criterion` | Only `"squared_error"` is implemented |

`max_leaf_nodes` is accepted by parameter assignment but a non-null value fails at fit: the best-first
builder is not implemented. User-provided sample weights are not part of `Fit(double[,], double[])`.
There is no general Python estimator wrapper, classifier backend, pipeline transformer or complete
scikit-learn API in this namespace.

The forest casts feature values to float32, builds its trees with the ported scikit-learn RNG and
splitter rules, and averages predictions in tree order. This matters when comparing reference values;
feeding a different preprocessing result or changing data order changes the problem.

## Folds and scorers

`SklearnEstimator` defaults to `KFold(5)` with no shuffle and to the estimator's own R² score.
`KFold` requires at least two folds and no more folds than samples. Give `randomState` only when
`shuffle: true`. Shuffling changes fold membership, while emitted train/test indices remain sorted.

`PredefinedSplits` accepts caller-supplied `(int[] Train, int[] Test)` pairs. Use it, or implement
`BaseCrossValidator`, when fold membership must follow groups or time boundaries. The supplied
regression splitters do not add a classifier/stratified branch. `CrossValidation.CheckCv` accepts
`null`, a fold count, a validator instance or a sequence of index pairs.

`Scorers` provides `Default`, `R2` and `NegMeanSquaredError`. `CheckScoring` resolves `null`, `"r2"`
and `"neg_mean_squared_error"`. A custom `Scorer` delegate receives a fitted `IRegressor`, feature
matrix and targets and must return a score where higher is better. The estimator uses that same
configured scorer for its post-refit `Score` method.

## Implement an estimator backend

`ICrossValidationEstimator` separates model training from the Optuna search:

```csharp
public interface ICrossValidationEstimator
{
    int GetNumSplits();
    IReadOnlyList<CrossValidationFoldResult> CrossValidate(
        IReadOnlyDictionary<string, object?> parameters,
        bool returnTrainScore);
    void Refit(IReadOnlyDictionary<string, object?> parameters);
}
```

An implementation owns the training data, split policy and metric. `CrossValidate` applies the
parameters, fits an independent model on each training split, and returns exactly one
`CrossValidationFoldResult(TestScore, TrainScore, FitTimeSeconds, ScoreTimeSeconds)` per fold.
`TrainScore` should be null when it was not requested. Report loss metrics negated. The backend owns
its fold-level error policy and any preprocessing needed to avoid fitting transforms on validation
data.

`Refit` trains the best parameters on the full training data and retains the model for prediction.
The search invokes it only when refit is enabled. Keep prediction as an API of the concrete backend,
as `SklearnEstimator.Predict` and `MLNetEstimator.Transform` do.

For a managed regressor that fits the native backend, implementing `IRegressor` is another option:
`WithParams` must return an unfitted configured clone, `Fit` trains it, and `Predict`/`Score` evaluate
it. `SklearnEstimator` then supplies the fold and scoring machinery.

The search does not implement Python's `partial_fit` pruning (`enable_pruning`/`max_iter`) or
`subsample` path. An iterative trainer can use OptunaSharp's pruning callback in its own training
loop; setting a study pruner does not inject intermediate training reports into this search.

## Validation

The [committed oracle tests](https://github.com/SciSharp/OptunaSharp/blob/main/test/OptunaSharp.Tests/Integration/Sklearn/SklearnOracleParityTests.cs)
use values captured from scikit-learn 1.8.0 and Python's OptunaSearchCV integration. They compare
forest predictions, metrics, folds, cross-validation scores and complete searches using integer
parameter distributions. Those exact comparisons are pinned to x64 and become inconclusive on other
architectures; Python is not needed to run the committed-oracle tests.

[Component tests](https://github.com/SciSharp/OptunaSharp/blob/main/test/OptunaSharp.Tests/Integration/Sklearn/SklearnComponentTests.cs)
and [search tests](https://github.com/SciSharp/OptunaSharp/blob/main/test/OptunaSharp.Tests/Integration/Sklearn/OptunaSearchCVSklearnTests.cs)
check the managed usage and supported boundaries. Timing values are measured wall-clock results,
not parity values. This evidence covers the implemented regression subset and tested configurations,
not arbitrary scikit-learn estimators or platforms.

Source: [search](https://github.com/SciSharp/OptunaSharp/blob/main/src/OptunaSharp.Core/Integration/OptunaSearchCV.cs),
[estimator contract](https://github.com/SciSharp/OptunaSharp/blob/main/src/OptunaSharp.Core/Integration/ICrossValidationEstimator.cs),
[native backend](https://github.com/SciSharp/OptunaSharp/blob/main/src/OptunaSharp.Core/Integration/Sklearn/SklearnEstimator.cs),
[forest](https://github.com/SciSharp/OptunaSharp/blob/main/src/OptunaSharp.Core/Integration/Sklearn/Ensemble/RandomForestRegressor.cs).
