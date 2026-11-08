# ML.NET

**`OptunaSharp.Integration.ML.NET`** provides two integration surfaces: `MLNetEstimator` for
cross-validation search over ML.NET pipelines, and six alternative parameter-importance evaluators
backed by ML.NET tree ensembles. It targets .NET 8 and .NET 10 and references `Microsoft.ML` and
`Microsoft.ML.FastTree`. LightGBM has its [own integration package](lightgbm.md).

```bash
dotnet add package OptunaSharp.Integration.ML.NET
```

## Search an ML.NET pipeline

`MLNetEstimator` bundles an `MLContext`, an `IDataView`, a pipeline factory, a scorer, and fixed
cross-validation splits. `OptunaSearchCV` in Core samples the parameter distributions and asks the
estimator to evaluate each candidate. The scorer must return a value where **higher is better**.

This is a complete console example. Integer distributions return boxed `long` values, so the
factory converts them before assigning ML.NET's integer options.

```csharp
using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;
using OptunaSharp.Distributions;
using OptunaSharp.Integration;

var ml = new MLContext(seed: 0);
var rows = Enumerable.Range(0, 180).Select(i => new RegressionRow
{
    Features = new[] { i / 180f, (i % 17) / 17f },
    Label = 2 * (i / 180f) - (i % 17) / 17f
});
IDataView data = ml.Data.LoadFromEnumerable(rows);

IEstimator<ITransformer> BuildPipeline(IReadOnlyDictionary<string, object?> parameters)
    => ml.Regression.Trainers.FastTree(
        labelColumnName: "Label",
        featureColumnName: "Features",
        numberOfLeaves: Convert.ToInt32(parameters["leaves"]),
        numberOfTrees: 20,
        learningRate: Convert.ToDouble(parameters["learning_rate"]));

double Score(MLContext context, ITransformer model, IDataView validation)
    => context.Regression.Evaluate(model.Transform(validation)).RSquared;

var estimator = new MLNetEstimator(ml, data, BuildPipeline, Score, folds: 3, seed: 0);
var space = new Dictionary<string, BaseDistribution>
{
    ["leaves"] = new IntDistribution(4, 24),
    ["learning_rate"] = new FloatDistribution(0.05, 0.3, log: true)
};
var search = new OptunaSearchCV(estimator, space, nTrials: 5, randomState: 0);
search.Fit();

Console.WriteLine($"Best cross-validation R²: {search.BestScore}");
IDataView predictions = estimator.Predict(data);
ml.Model.Save(estimator.BestModel!, data.Schema, "best-model.zip");

public sealed class RegressionRow
{
    [VectorType(2)]
    public float[] Features { get; set; } = Array.Empty<float>();
    public float Label { get; set; }
}
```

The final prediction call above uses training data only to demonstrate the API. Evaluate model
quality on a separate held-out dataset. For a loss scorer, return its negative; a caller-supplied
`Study` must also have `StudyDirection.Maximize`.

## Data, folds, errors, and refitting

| Option or operation | Current behavior |
|---|---|
| `folds` | Defaults to 5; values below 2 throw `NumSharp.ValueError` |
| `seed` | Passed to ML.NET's fold splitter; separate from `OptunaSearchCV.randomState` and trainer seeds |
| `samplingKeyColumnName` | Forwarded to ML.NET's `CrossValidationSplit`; key related rows consistently when choosing a split |
| Pipeline factory | Called once per fold for each candidate, and again for the final refit; return a fresh pipeline |
| `errorScore` | Fit/scoring exceptions become this fold score (default `NaN`); factory exceptions occur outside that handler |
| `returnTrainScore` | Also scores each fitted fold model on its training split |
| `refit: true` | Fits the winning configuration on the full data; stores it in `BestModel` |
| `Predict` / `Transform` / `Score` | Operate on the refitted model; throw before refit |

Fold definitions are created in the estimator constructor and reused across candidates. Folds run
sequentially within `CrossValidate`; trial-level parallelism is controlled by `OptunaSearchCV.nJobs`.
When enabling parallel trials, ensure your pipeline factory, scorer, and captured application state
can be invoked concurrently. Refit happens after the search.

`OptunaSearchCV` aggregates fold results using a NaN-aware mean. A mixture of valid and failed folds
can therefore produce a score from fewer folds. Inspect trial attributes and `CvResults` when
diagnosing failures. Refitting is not covered by the per-fold `errorScore` handler.

The [native scikit-learn-style guide](scikit-learn.md) describes the shared `OptunaSearchCV` result
surface and `ICrossValidationEstimator` contract. This ML.NET backend uses ML.NET's trainers and
splits; it does not promise numeric parity with Python scikit-learn. It also does not implement
incremental `partial_fit` pruning; see [training-loop pruning](pruning.md).

The fold error handler also catches `TrialPruned` thrown by fitting or scoring and converts it to
`errorScore`. Putting a pruning callback inside this adapter's pipeline therefore does not by
itself mark the Optuna trial pruned. A custom `ICrossValidationEstimator` that supports pruning
must allow that exception to reach `Study.Optimize`.

## Parameter importance backends

These evaluators fit a surrogate model to **completed optimization trials**, not to your original
training data. Pass them to `ImportanceFunctions.GetParamImportances`.

| Evaluators | Namespace | Ensemble and interpretation |
|---|---|---|
| `FastForestFanovaImportanceEvaluator`, `FastForestMeanDecreaseImpurityImportanceEvaluator` | `OptunaSharp.Importance.Forests` | Bagged random forest; same model family as canonical Optuna importance, different fitted trees and values |
| `FastTreeFanovaImportanceEvaluator`, `FastTreeMeanDecreaseImpurityImportanceEvaluator` | `OptunaSharp.Importance.Booster` | Experimental gradient-boosted surrogate; residual trees change the meaning of the importance |
| `FastTreeTweedieFanovaImportanceEvaluator`, `FastTreeTweedieMeanDecreaseImpurityImportanceEvaluator` | `OptunaSharp.Importance.Booster` | Experimental Tweedie-loss booster; no Optuna value-parity claim |

```csharp
using OptunaSharp.Importance;
using OptunaSharp.Importance.Forests;

var importance = ImportanceFunctions.GetParamImportances(
    search.Study, new FastForestFanovaImportanceEvaluator(seed: 0));
```

Each evaluator accepts `(nTrees = 64, maxDepth = 64, seed = null)`, its corresponding complete
trainer options object, or an `Action<Options>` that modifies reproducible defaults. For example:

```csharp
var evaluator = new FastForestFanovaImportanceEvaluator(options =>
{
    options.NumberOfTrees = 100;
    options.FeatureFraction = 0.7;
    options.Seed = 0;
    options.NumberOfThreads = 1;
});
```

The evaluator owns the label and feature column names. Weight and row-group columns do not exist
in its transformed-trial schema; setting them is rejected. Full options otherwise control the
trainer, so keep a fixed seed and one training thread when you need reproducibility. The fANOVA
and MDI calculations reuse Core's implementation, but ML.NET grows different trees. Core's
`FanovaImportanceEvaluator` and `MeanDecreaseImpurityImportanceEvaluator` remain the canonical
scikit-learn-compatible implementations; the ML.NET satellite is optional.

## Implementation and verification

The [estimator](https://github.com/SciSharp/OptunaSharp/blob/main/src/OptunaSharp.Integration.ML.NET/Sklearn/MLNetEstimator.cs)
defines fold handling and refit behavior. The
[end-to-end test](https://github.com/SciSharp/OptunaSharp/blob/main/test/OptunaSharp.Tests.Integration.ML.NET/MLNetEstimatorEndToEndTests.cs)
fits a real FastTree pipeline, searches its parameters, refits, predicts, and scores. The
[search tests](https://github.com/SciSharp/OptunaSharp/blob/main/test/OptunaSharp.Tests.Integration.ML.NET/OptunaSearchCVTests.cs)
cover direction validation, reproducibility, result attributes, and refit options.
