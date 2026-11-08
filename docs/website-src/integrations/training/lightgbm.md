# LightGBM

**`OptunaSharp.Integration.LightGbm`** integrates LightGBM through ML.NET's native LightGBM trainer.
It supplies stepwise tuners, a metric-specific pruning callback, and two experimental importance
evaluators. All types described here use `OptunaSharp.Integration.LightGbm`. No Python runtime is
required; the package targets .NET 8 and .NET 10 and references `Microsoft.ML.LightGbm`.

```bash
dotnet add package OptunaSharp.Integration.LightGbm
```

## Tune against a validation set

`Dataset` holds a rectangular feature matrix (`float[][]` or `double[,]`) and optional `double[]`
labels. The rectangular-double overload copies features into floats. `LightGBMTuner` evaluates
candidates on its validation sets and accumulates all tuning stages in one `Study`.

Supply labels explicitly: an unlabeled dataset becomes zero targets in the default trainer.
When several validation sets are supplied, all are scored, but **the last validation set** drives
optimization. Keep `validNames` aligned with `validSets`.

Use a single scalar metric string such as `"l2"` or `"auc"`. Although score lookup accepts a
metric sequence and selects its last element, direction detection currently recognizes only
scalar strings. For example, a sequence containing `"auc"` incorrectly selects minimization.

```csharp
using System;
using System.Collections.Generic;
using System.Linq;
using OptunaSharp.Integration.LightGbm;

Dataset MakeData(int first, int count)
{
    var x = Enumerable.Range(first, count)
        .Select(i => new[] { (i % 31) / 31f, (i % 13) / 13f }).ToArray();
    var y = x.Select(row => (double)(2 * row[0] - row[1])).ToArray();
    return new Dataset(x, y);
}

var parameters = new Dictionary<string, object?>
{
    ["objective"] = "regression",
    ["metric"] = "l2",
    ["verbosity"] = -1
};
var tuner = new LightGBMTuner(
    parameters, MakeData(0, 180), validSets: new[] { MakeData(180, 60) },
    numBoostRound: 30, optunaSeed: 42);
tuner.Run();

Console.WriteLine($"Best validation loss: {tuner.BestScore}");
foreach (var parameter in tuner.BestParams)
    Console.WriteLine($"{parameter.Key} = {parameter.Value}");
Booster best = tuner.GetBestBooster();
```

The default `Booster.Model` is an ML.NET `ITransformer` when a model is retained, not a Python
`lightgbm.Booster`. Use ML.NET to transform data or serialize that model. The static
`LightGBMTuner.Train(...)` convenience method constructs the tuner, runs every stage, and returns
the best booster. It is a full tuning operation, not a single model fit.

## Stepwise search

| Stage | Parameters | Sampler | Default trial budget |
|---|---|---|---|
| Feature fraction | `feature_fraction` in `[0.4, 1.0]` | Grid | 7 |
| Leaves | `num_leaves` | TPE | 20 |
| Bagging | `bagging_fraction`, `bagging_freq` | TPE | 10 |
| Fine feature fraction | Grid around the current best fraction, clipped to `[0.4, 1.0]` | Grid | Up to 6 |
| Regularization | `lambda_l1`, `lambda_l2` | TPE | 20 |
| Minimum leaf data | `min_child_samples` in `{5, 10, 25, 50, 100}` | Grid | 5 |

`Run()` applies this order. Individual `TuneFeatureFraction`, `TuneNumLeaves`, `TuneBagging`,
`TuneFeatureFractionStage2`, `TuneRegularizationFactors`, and `TuneMinDataInLeaf` methods are also
public. Their trial budgets account for already complete or pruned trials belonging to that stage,
so resuming a study does not blindly repeat the whole budget. Each stage replaces `Study.Sampler`.

`optunaSeed` seeds the stage samplers. `timeBudget` limits total tuning time in seconds, checked
between trials; it cannot interrupt an ML.NET fit already running. `numBoostRound` defaults to
1,000 per trial, so choose it deliberately. `sampleSize` keeps the **last** N training rows, rather
than taking a random subsample. Avoid `sampleSize` with grouped ranking data: the current reduced
dataset path drops group information. `feature_pre_filter` is forced to `false` for leaf-size tuning.

## Cross-validation and model persistence

`LightGBMTunerCV` uses the same search with fold-averaged validation metrics:

```csharp
var cv = new LightGBMTunerCV(
    parameters, MakeData(0, 240), numBoostRound: 30,
    nFold: 3, seed: 0, optunaSeed: 42, returnCvBooster: true);
cv.Run();
CvBooster folds = cv.GetBestBooster();
```

`seed` controls fold splitting; `optunaSeed` controls search sampling. Set `returnCvBooster: true`
to enable `GetBestBooster` for the CV tuner. Its result retains the separate fold models; it does
not refit one model on all rows.

Supply `study` to resume stored trials; its direction must agree with the metric. Supply `modelDir`
to recover boosters across tuner instances or processes. The default trainer writes ML.NET model
files plus JSON score metadata. These files are not Python pickle files and are separate from the
study database. All participating workers need access to the same model directory.

Use a separate directory for each study: persisted model names are based on trial numbers alone.
Sharing one directory between different studies can overwrite their models.

`GetBestBooster` first returns the current in-memory winner. If the best stored trial belongs to a
different run, it needs a persisted model; without `modelDir` or the corresponding files it throws.
`BestScore` is an infinity sentinel before completed trials exist, and `BestParams` then contains
defaults overlaid with supplied parameters. Do not treat those values as a completed tuning result.

## What the default trainer implements

The search stages follow Optuna's tuner; the fit and score layer is an adapter over ML.NET. Its
current behavior is narrower than the complete Python LightGBM API:

| Capability | Current adapter behavior |
|---|---|
| Regression | Fits L2 regression; other regression objectives and unknown objective names fall back to that task |
| Binary classification | Binary LightGBM trainer; computes binary error, log loss, and AUC metrics |
| Multiclass | Handles `multiclass`, `multiclassova`, and their recognized aliases |
| Ranking | Handles ranking objective aliases through ML.NET ranking; requires positive query-group sizes summing to the dataset row count |
| Ranking folds | Split whole query groups so a query is not divided between training and validation |
| Other folds | Row splits, optionally shuffled; the accepted `stratified` flag is not consumed by the default fold builder |
| Training duration | Fixed boosting-round count; no per-iteration early-stopping callback |
| CV results | Final fold-mean metric values, not a complete per-iteration curve or standard-deviation series |
| Dataset weights | Accepted and stored, but not consumed by the default trainer |
| Feature names | Accepted; default training uses positional feature vectors |

Supported score names are `l2`, `l1`, `rmse`, and `mape` for regression; `binary_logloss`,
`binary_error`, and `auc` for binary classification; `multi_logloss` and `multi_error` for
multiclass; and `ndcg@k` / `map@k` for ranking. Configure ranking with scalar `"ndcg"` or `"map"`
and `eval_at`; the tuner uses its first cutoff. An unsupported metric can fail score lookup even
when the objective falls back to regression.

The default trainer catches `ArgumentNullException` from fitting and substitutes a constant
predictor's metrics, accommodating degenerate ML.NET tree fits. Such a trial may complete with
`Booster.Model == null`. Check for a model before prediction or serialization; a completed trial
does not guarantee a retained transformer.

Poisson, gamma, Tweedie, quantile, and custom callable objectives therefore should not be described
as exact native LightGBM implementations through this default adapter. Likewise, `stratified: true`
does not currently create stratified classification folds. Use a custom trainer when those semantics
matter.

Implement `ILightGbmTrainer.Train(TrainRequest)` and `Cv(CvRequest)` to replace training and scoring
while retaining the search. Implement `ILightGbmBoosterStore` as well if you use `modelDir`; supplying
a directory with a trainer that cannot persist boosters fails at construction. `optunaCallbacks`
are callbacks after Optuna trials, not callbacks after boosting iterations.

## Metric-specific pruning

`LightGBMPruningCallback(trial, metric, validName = "valid_0", reportInterval = 1)` consumes a list
of `LightGbmEvaluationResult` values via `Call(iteration, results, isCv: false)`. It finds the
requested dataset/metric, checks the metric direction against the study, reports its score, and
throws `TrialPruned` when appropriate.

```csharp
// Inside an objective whose training engine exposes its own iteration metrics:
var callback = new LightGBMPruningCallback(trial, metric: "l2", reportInterval: 5);
callback.Call(iteration, new[]
{
    new LightGbmEvaluationResult("valid_0", "l2", validationLoss, IsHigherBetter: false)
});
```

The snippet assumes `trial`, `iteration`, and `validationLoss` from that engine. ML.NET's default
LightGBM trainer does not invoke this callback. For an ordinary scalar metric loop use Core's
[`PruningCallback`](pruning.md). Use a positive report interval: reporting occurs when
`(iteration + 1) % reportInterval == 0`. CV selects dataset name `valid`; metric lookup accepts
both the bare metric name and a `valid ` prefix. Missing metrics and inconsistent directions throw
`NumSharp.ValueError`.

## Experimental importance evaluators

`LightGbmFanovaImportanceEvaluator` and `LightGbmMeanDecreaseImpurityImportanceEvaluator` fit
LightGBM to completed optimization trials and reuse Core's importance calculations. They accept
`(nTrees = 64, maxDepth = 64, seed = null)`, complete `LightGbmRegressionTrainer.Options`, or an
`Action<Options>` to modify defaults. The evaluators control label/feature column names and reject
weight/group columns absent from the trial schema.

These are **experimental alternatives**: boosted trees fit residuals, unlike the bagged forest
used by Optuna's canonical importance. Their values are not equivalent to Optuna's fANOVA/MDI
values. Use Core's canonical evaluators for that contract, or the
[ML.NET FastForest alternatives](ml-net.md#parameter-importance-backends) when choosing that model
family intentionally.

## Implementation and verification

The [tuner engine](https://github.com/SciSharp/OptunaSharp/blob/main/src/OptunaSharp.Integration.LightGbm/Tuner/LightGbmBaseTuner.cs),
[trainer adapter](https://github.com/SciSharp/OptunaSharp/blob/main/src/OptunaSharp.Integration.LightGbm/Internal/MlNetLightGbmTrainer.cs),
and [dataset](https://github.com/SciSharp/OptunaSharp/blob/main/src/OptunaSharp.Integration.LightGbm/Tuner/Dataset.cs)
define the behavior above. The
[integration tests](https://github.com/SciSharp/OptunaSharp/tree/main/test/OptunaSharp.Tests.Integration.LightGbm)
exercise the tuning schedule, real training, multiclass/ranking support, persistence, importance,
and pruning. Schedule parity uses a deterministic scorer to isolate the search from the training
engine; it is not evidence that ML.NET and Python LightGBM fit identical models.
