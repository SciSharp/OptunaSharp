# OptunaSharp.Importance.Forests

Random-forest **parameter-importance evaluators** for
[OptunaSharp](https://github.com/SciSharp/OptunaSharp) — a .NET port of
[Optuna](https://github.com/optuna/optuna), part of the [SciSharp STACK](https://github.com/SciSharp).

This satellite package ports Optuna's two forest-based importance evaluators:

| OptunaSharp | Optuna |
|---|---|
| `OptunaSharp.Importance.FanovaImportanceEvaluator` | `optuna.importance.FanovaImportanceEvaluator` |
| `OptunaSharp.Importance.MeanDecreaseImpurityImportanceEvaluator` | `optuna.importance.MeanDecreaseImpurityImportanceEvaluator` |

- **fANOVA** — the functional-ANOVA decomposition of
  [Hutter et al., ICML 2014](http://proceedings.mlr.press/v32/hutter14.html): fit a random forest that
  predicts the objective from the parameters, then attribute the forest's prediction variance to each
  parameter.
- **Mean Decrease Impurity (MDI)** — the random forest's built-in feature importances (the analogue of
  scikit-learn's `RandomForestRegressor.feature_importances_`).

It ships **separately from Core** because it pulls a foreign runtime — the random forest is fitted by
[ML.NET](https://github.com/dotnet/machinelearning)'s `FastForest` (native FastTree) rather than
scikit-learn. Core stays pure-managed; this is the `NumSharp.Interop.OpenBLAS` optional-package
pattern.

## Usage

```csharp
using OptunaSharp;
using OptunaSharp.Importance;

var study = Optuna.CreateStudy(sampler: new RandomSampler(seed: 42));
study.Optimize(t =>
{
    double x = t.SuggestFloat("x", -10, 10);
    long   y = t.SuggestInt("y", 0, 10);
    return x * x + y;
}, nTrials: 100);

// fANOVA
var fanova = ImportanceFunctions.GetParamImportances(
    study, evaluator: new FanovaImportanceEvaluator(seed: 0));

// Mean Decrease Impurity
var mdi = ImportanceFunctions.GetParamImportances(
    study, evaluator: new MeanDecreaseImpurityImportanceEvaluator(seed: 0));
```

`GetParamImportances` lives in Core (`OptunaSharp.Importance`); pass either evaluator as the
`evaluator` argument. The default evaluator when none is passed is the pure-managed
`PedAnovaImportanceEvaluator` (Core, no ML.NET). Both evaluators here accept `nTrees` (default 64),
`maxDepth` (default 64), and `seed` (default `null`).

## Divergence from Python Optuna (documented)

Optuna backs both evaluators with **scikit-learn's `RandomForestRegressor`**. This package backs them
with **ML.NET's `FastForest`**, which is not tree-identical to scikit-learn:

- it bins features and grows trees leaf-wise (not depth-wise);
- thresholds are float32;
- for MDI the node impurities are recomputed by routing the training set through the fitted trees
  (ML.NET does not expose per-node bootstrap statistics), i.e. training-set variance rather than the
  per-tree bootstrap impurity scikit-learn stores.

Consequently the **importance values differ numerically** from Python Optuna — the ranking of a
clearly-dominant parameter is robust, exact values are not. This is the plan's Tier X
`tolerance oracle` divergence. The fANOVA marginal-variance algorithm (a port of
`optuna/importance/_fanova/_tree.py`) and the MDI formula are themselves faithful ports, validated
bit-for-bit against Optuna's own `test_tree.py` on a hand-built tree; only the underlying forest
differs. An exact managed-CART port that would restore byte-for-byte parity remains a Tier N backlog
item.

For reproducibility the forest is fitted single-threaded from the given `seed`.

## License

MIT — see the repository `LICENSE`.
