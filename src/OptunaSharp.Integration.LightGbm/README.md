# OptunaSharp.Integration.LightGbm

The **LightGBM integration** for [OptunaSharp](https://github.com/SciSharp/OptunaSharp) — a .NET port
of [Optuna](https://github.com/optuna/optuna), part of the
[SciSharp STACK](https://github.com/SciSharp). Backed by
[ML.NET](https://github.com/dotnet/machinelearning)'s native LightGBM. This is the analog of Optuna's
`optuna.integration.lightgbm`.

One package per heavy native dependency: `Microsoft.ML.LightGbm` ships the native LightGBM binaries
per-RID, so this ships **separately from Core** (the `NumSharp.Interop.OpenBLAS` optional-package
pattern) and Core stays pure managed.

## Today: LightGBM importance backends (⚠️ experimental / off-label)

> Optuna's importance module uses **only** `sklearn.ensemble.RandomForestRegressor` (bagging). **There
> is no boosting anywhere in Optuna's importance.** LightGBM is a histogram **gradient booster** — its
> trees fit the ensemble **residual**, not the objective — so running fANOVA/MDI over a LightGBM
> ensemble measures a **different, non-standard quantity** with **no value-parity to Optuna** (weak
> ranking agreement at best).

| Evaluator | Namespace | Optuna parity |
|---|---|---|
| `LightGbmFanovaImportanceEvaluator` | `OptunaSharp.Integration.LightGbm` | ⚠ off-label, ranking only |
| `LightGbmMeanDecreaseImpurityImportanceEvaluator` | `OptunaSharp.Integration.LightGbm` | ⚠ off-label, ranking only |

**Prefer** the canonical, value-exact `FanovaImportanceEvaluator` /
`MeanDecreaseImpurityImportanceEvaluator` in Core (a pure-managed, bit-exact port of scikit-learn's
`RandomForestRegressor`), or the faithful ML.NET FastForest backend
(`OptunaSharp.Importance.Forests.ML.NET`). Use the LightGBM importance backends only for
experimentation / consistency with a LightGBM pipeline.

```csharp
using OptunaSharp;
using OptunaSharp.Importance;            // ImportanceFunctions, canonical evaluators
using OptunaSharp.Integration.LightGbm;  // the experimental LightGBM importance backends

var study = Optuna.CreateStudy(sampler: new RandomSampler(seed: 42));
study.Optimize(t => { double x = t.SuggestFloat("x", -10, 10); return x * x; }, nTrials: 100);

// EXPERIMENTAL — ranking only, no Optuna value-parity:
var imp = ImportanceFunctions.GetParamImportances(
    study, new LightGbmFanovaImportanceEvaluator(seed: 0));

// For Optuna parity, use the canonical Core evaluator instead:
var canonical = ImportanceFunctions.GetParamImportances(study, new FanovaImportanceEvaluator(seed: 0));
```

The evaluators take `nTrees` (default 64), `maxDepth` (default 64), and `seed` (default `null`). The
LightGBM ensemble is fitted single-threaded from the given `seed` for reproducibility. The fANOVA walk
and MDI formula are shared with Core (validated against Optuna's own `test_tree.py`); only the
underlying tree ensemble differs.

### Full control: the ML.NET trainer options

Each evaluator also has an overload taking the **complete `LightGbmRegressionTrainer.Options`**, so the
entire LightGBM surface is controllable — learning rate, number of leaves, bin count, categorical
handling, and the nested `Booster` (GBDT / DART / GOSS):

```csharp
using Microsoft.ML.Trainers.LightGbm;

var evaluator = new LightGbmFanovaImportanceEvaluator(new LightGbmRegressionTrainer.Options
{
    NumberOfIterations = 300,
    NumberOfLeaves = 63,
    LearningRate = 0.05,               // absent from the simple ctor
    MaximumBinCountPerFeature = 255,
    Seed = 0,
    NumberOfThreads = 1,               // set Seed + a single thread for a reproducible result
});
var imp = ImportanceFunctions.GetParamImportances(study, evaluator);
```

`LabelColumnName` / `FeatureColumnName` are managed by the evaluator and overwritten; every other field
is used as given. The `(nTrees, maxDepth, seed)` constructor is this overload fed a reproducible
default options object.

Or **start from those reproducible defaults and tweak** just a few knobs with an
`Action<LightGbmRegressionTrainer.Options>` — no need to restate the defaults:

```csharp
var evaluator = new LightGbmFanovaImportanceEvaluator(
    o => { o.LearningRate = 0.05; o.Seed = 0; o.NumberOfThreads = 1; });
```

## The LightGBM **tuner** — the Optuna-justified half (Phase 6.1)

LightGBM's real identity in Optuna is a **tuner**, not importance — and it is ported here:
[`optuna_integration.lightgbm`](https://optuna.readthedocs.io/en/stable/reference/integration.html)'s
**`LightGBMTuner`** / **`LightGBMTunerCV`** (and the `train` drop-in). It is a **stepwise** tuner that
sweeps the core LightGBM hyperparameters in a fixed, well-tested order — `feature_fraction`, then
`num_leaves`, then `bagging_fraction`/`bagging_freq`, then a fine `feature_fraction` pass, then
`lambda_l1`/`lambda_l2`, then `min_child_samples` — each step a short Optuna study over a fresh
`GridSampler` or `TPESampler`, holding the previously-found best fixed.

```csharp
using OptunaSharp.Integration.LightGbm;

var dtrain = new Dataset(xTrain, yTrain);       // features (float[][] or double[,]) + labels
var dval   = new Dataset(xVal,   yVal);

var lgbmParams = new Dictionary<string, object?>
{
    ["objective"] = "regression",
    ["metric"]    = "l2",
};

var tuner = new LightGBMTuner(lgbmParams, dtrain, validSets: new[] { dval },
    numBoostRound: 100, optunaSeed: 42);
tuner.Run();

var best        = tuner.BestParams;             // the tuned LightGBM parameters
double score    = tuner.BestScore;              // the best validation metric
var booster     = tuner.GetBestBooster();       // the best trial's trained model

// Or the drop-in one-liner (the analog of optuna.integration.lightgbm.train):
var booster2 = LightGBMTuner.Train(lgbmParams, dtrain, validSets: new[] { dval }, optunaSeed: 42);
```

**`LightGBMTunerCV`** is the cross-validation variant — same stepwise search, scored by k-fold CV on
the training set (the analog of `lightgbm.cv`).

### What is faithful, and what is an adapter

The **stepwise search** — which parameters, in which order, with which sampler and seed — is a faithful
port of `optuna_integration.lightgbm` and rides OptunaSharp's own bit-exact `GridSampler` / `TPESampler`.
It is validated **against live Optuna's tuner** by a committed synthetic-scorer oracle: with the training
call replaced by a deterministic score, the C# tuner reproduces Optuna's exact trial sequence — the
grid- and integer-valued parameters bit-for-bit, the TPE-sampled continuous parameters to the erf
tolerance (see `test/OptunaSharp.Tests.Integration.LightGbm`).

The **training + metric evaluation** is an **adapter** over ML.NET's native LightGBM (this is a Phase 6
framework-integration adapter — ML.NET's LightGBM is not Python's, so there is no bit-parity there). The
first cut maps the `binary` and regression objectives. To plug a different engine — or a custom scorer —
implement `ILightGbmTrainer` and pass it as the `trainer` argument; the default is `MlNetLightGbmTrainer`.

> **Not yet ported:** cross-process booster sharing via `model_dir` (the Python version pickles boosters;
> ML.NET model persistence is different) — run tuning in a single process; and the ranking objectives
> (`ndcg`/`map`) in the default trainer. `feature_name` / `categorical_feature` and the LightGBM-internal
> `feval` / `callbacks` / `keep_training_booster` kwargs are Python-binding specifics and are omitted.

## License

MIT — see the repository `LICENSE`.
