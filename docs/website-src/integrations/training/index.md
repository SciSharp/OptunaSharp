# Model training

Connect an estimator or a training loop to OptunaSharp through the integration that matches how it
trains. Cross-validation search, stepwise LightGBM tuning, and iteration-level pruning have different
contracts; they are not interchangeable callbacks.

| Integration | Use it for | Package | Python required? |
|---|---|---|---|
| [Native scikit-learn-style search](scikit-learn.md) | `OptunaSearchCV`, the managed random-forest regressor, fold splitting, and scoring | `OptunaSharp` (Core) | No |
| [ML.NET](ml-net.md) | Tune an ML.NET pipeline over an `IDataView`; use ML.NET forest importance backends | `OptunaSharp.Integration.ML.NET` | No |
| [LightGBM](lightgbm.md) | Stepwise `LightGBMTuner` / `LightGBMTunerCV`, metric callbacks, and alternative importance backends | `OptunaSharp.Integration.LightGbm` | No |
| [Training-loop pruning](pruning.md) | Report validation metrics from an iterative loop you control | `OptunaSharp` (Core) | No |

`OptunaSearchCV` always maximizes a score: negate a loss such as mean squared error. A LightGBM
tuner derives its study direction from a scalar metric name; use the
[documented metric form](lightgbm.md#tune-against-a-validation-set). A pruning callback uses the direction
of the study that owns its trial. Choose the same metric and direction throughout a run.

Both ML.NET packages target .NET 8 and .NET 10 and bring their trainer dependencies transitively.
They do not initialize CPython. For algorithms that execute in Python, see the
[Python bridges](../python/index.md).

## Extending the training integration

Implement `ICrossValidationEstimator` when you can fit and score independent folds. Its
`CrossValidate` method returns `CrossValidationFoldResult` values; `Refit` trains with the winning
parameters. `OptunaSearchCV` owns trial sampling, score aggregation, trial attributes, and the final
refit. See the [cross-validation contract](scikit-learn.md).

Use `PruningCallback` inside a loop that emits intermediate validation metrics. The callback does
not add an iteration hook to a trainer that has none. The repository does not ship separate
TorchSharp, TensorFlow.NET, Keras, XGBoost, or CatBoost adapter packages; an application can use the
generic callback where it controls training.
