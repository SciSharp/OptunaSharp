# OptunaSharp.Integration.Shap

The **SHAP integration** for [OptunaSharp](https://www.nuget.org/packages/OptunaSharp) — a .NET port
of [Optuna](https://github.com/optuna/optuna). Part of the [SciSharp STACK](https://github.com/SciSharp).

Ships **`ShapleyImportanceEvaluator`**, a Shapley (SHAP) parameter-importance evaluator — a .NET
analogue of [`optuna_integration.shap`](https://github.com/optuna/optuna-integration) (which Optuna
also re-exports as `optuna.integration.ShapleyImportanceEvaluator`). It fits a random forest that
predicts the objective values of completed trials from their parameter configurations, then scores
each parameter by its **mean absolute SHAP value**.

```csharp
using OptunaSharp;
using OptunaSharp.Integration.Shap;

var study = Study.Create(direction: StudyDirection.Minimize);
study.Optimize(trial =>
{
    var x = trial.SuggestFloat("x", -5, 5);
    var y = trial.SuggestFloat("y", -5, 5);
    return x * x + y * y;
}, nTrials: 50);

var importances = ImportanceFunctions.GetParamImportances(
    study, evaluator: new ShapleyImportanceEvaluator(seed: 0));
// { "x": ..., "y": ... }  (descending by importance)
```

## Byte-exact, and pure-managed

The Python original depends on the third-party `shap` and `scikit-learn` packages. This port depends
on **nothing but OptunaSharp Core** — the Tree SHAP walk is pure arithmetic:

- the surrogate is Core's **bit-exact port of scikit-learn's `RandomForestRegressor`** (the same
  forest the Mean Decrease Impurity evaluator uses), and
- the tree-path-dependent Tree SHAP walk is a faithful port of
  [`shap/cext/tree_shap.h`](https://github.com/shap/shap) (Lundberg et al.,
  [arXiv:1802.03888](https://arxiv.org/abs/1802.03888)).

With a fixed `seed`, the importances match Optuna's `ShapleyImportanceEvaluator` (SHAP 0.52.0,
scikit-learn) run through pythonnet to within a few ULP.

It ships as a **separate package**, mirroring where Optuna keeps it, so Core's
`OptunaSharp.Importance` stays a 1-to-1 mirror of `optuna.importance` — the canonical
`FanovaImportanceEvaluator`, `MeanDecreaseImpurityImportanceEvaluator`, and
`PedAnovaImportanceEvaluator` live in Core.
