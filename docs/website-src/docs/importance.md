# Parameter Importance

After a study, **parameter importance** tells you which hyperparameters actually drove the objective
and which barely mattered — so you can widen the ranges that count and fix the ones that don't.
OptunaSharp ports `optuna.importance` into **`OptunaSharp.Importance`**.

## Computing importances

```csharp
using OptunaSharp;
using OptunaSharp.Collections;   // OrderedDict
using OptunaSharp.Importance;

OrderedDict<string, double> importances = ImportanceFunctions.GetParamImportances(study);

foreach (var (name, importance) in importances)   // ordered most-important first
    Console.WriteLine($"{name}: {importance:P1}");
```

The full signature:

```csharp
public static OrderedDict<string, double> GetParamImportances(
    Study study,
    BaseImportanceEvaluator? evaluator = null,           // default: PedAnovaImportanceEvaluator
    IReadOnlyList<string>? @params = null,               // restrict to these parameters
    Func<FrozenTrial, double>? target = null,            // custom target (e.g. an objective index)
    bool normalize = true);                              // importances sum to 1
```

With `normalize: true` (the default) the importances sum to 1 and read as fractions of the total.
Restrict the analysis with `@params`, or point it at a different quantity — a specific objective in a
multi-objective study, or a trial duration — with `target`.

## Available evaluators

All four of Optuna's importance evaluators are ported, and all live in **Core** — the forest-based
ones use a **pure-managed, bit-exact port of scikit-learn's `RandomForestRegressor`** as the
surrogate, so none of them needs a foreign runtime:

| Evaluator | Namespace | Ports |
|-----------|-----------|-------|
| `PedAnovaImportanceEvaluator` *(default)* | `OptunaSharp.Importance.PedAnova` | `optuna.importance.PedAnovaImportanceEvaluator` |
| `FanovaImportanceEvaluator` | `OptunaSharp.Importance` | `optuna.importance.FanovaImportanceEvaluator` |
| `MeanDecreaseImpurityImportanceEvaluator` | `OptunaSharp.Importance` | `optuna.importance.MeanDecreaseImpurityImportanceEvaluator` |
| `ShapleyImportanceEvaluator` | `OptunaSharp.Importance.Shap` | `optuna.integration.ShapleyImportanceEvaluator` |

> **Which parameters are assessed.** With the default `PedAnovaImportanceEvaluator` and
> `params: null`, *all* parameters that appear in completed trials are assessed, including conditional
> ones. The three forest/SHAP evaluators assess only parameters that are present in **all** of the
> completed trials. Passing `params` restricts any evaluator to the listed parameters.

### PED-ANOVA (the default evaluator)

The default evaluator is `PedAnovaImportanceEvaluator` (namespace `OptunaSharp.Importance.PedAnova`),
a fast, sampler-agnostic method based on the Pearson-divergence ANOVA of the distribution of good
trials against all trials. It needs no surrogate forest.

```csharp
using OptunaSharp.Importance.PedAnova;

var evaluator = new PedAnovaImportanceEvaluator(
    targetQuantile: 0.1,      // "good" trials = the top 10%
    regionQuantile: 1.0,      // baseline region = all trials
    evaluateOnLocal: true);   // measure importance within the good region

var importances = ImportanceFunctions.GetParamImportances(study, evaluator);
```

`targetQuantile` sets what counts as a good trial; `evaluateOnLocal` focuses the measurement on the
promising region rather than the whole space.

### fANOVA

`FanovaImportanceEvaluator` (namespace `OptunaSharp.Importance`) implements the fANOVA algorithm of
[Hutter et al. (ICML 2014)](http://proceedings.mlr.press/v32/hutter14.html): it fits a random forest
that predicts the objective from the parameter configurations, then decomposes the forest's
prediction variance across the parameters. The more accurate the forest, the more reliable the
importances — an exploration-oriented sampler such as `RandomSampler` is recommended.

```csharp
using OptunaSharp.Importance;

var evaluator = new FanovaImportanceEvaluator(nTrees: 64, maxDepth: 64, seed: 0);
var importances = ImportanceFunctions.GetParamImportances(study, evaluator);
```

Because the surrogate is a bit-exact port of scikit-learn's `RandomForestRegressor`, a fixed `seed`
reproduces Python Optuna's importance values to about 1 ULP.

### Mean Decrease Impurity (MDI)

`MeanDecreaseImpurityImportanceEvaluator` (namespace `OptunaSharp.Importance`) fits the same random
forest and reads each tree's stored per-node impurity decrease and weighted sample counts — a port of
`sklearn.ensemble.RandomForestRegressor.feature_importances_`.

```csharp
using OptunaSharp.Importance;

var evaluator = new MeanDecreaseImpurityImportanceEvaluator(nTrees: 64, maxDepth: 64, seed: 0);
var importances = ImportanceFunctions.GetParamImportances(study, evaluator);
```

Like fANOVA, under a fixed `seed` the importances match Python Optuna to about 1 ULP.

### Shapley (SHAP)

`ShapleyImportanceEvaluator` (namespace `OptunaSharp.Importance.Shap`) fits the same random forest and
computes each parameter's importance as the **mean absolute SHAP value** over the trials, using a
tree-path-dependent Tree SHAP walk. Optuna ships this evaluator in the separate `optuna_integration`
package because *there* it depends on the third-party `shap` and `scikit-learn` Python libraries;
OptunaSharp's port is pure arithmetic (no `erf`, no LAPACK), so it lives in Core with the other
evaluators.

```csharp
using OptunaSharp.Importance.Shap;

var evaluator = new ShapleyImportanceEvaluator(nTrees: 64, maxDepth: 64, seed: 0);
var importances = ImportanceFunctions.GetParamImportances(study, evaluator);
```

Two behaviors mirror Optuna: the constructor raises an `ExperimentalWarning` (the class is
`@experimental_class("3.0.0")`), and on a **multi-objective** study it throws a `ValueError` unless
you pass a `target` (for example `target: t => t.Values[0]`). Under a fixed `seed` the importances are
byte-for-byte identical to Optuna's `ShapleyImportanceEvaluator` (SHAP 0.52.0) on x64/x86 hosts.

### ML.NET forest backends (optional package)

The `OptunaSharp.Integration.ML.NET` package adds forest-based evaluators that swap the native
surrogate for an **ML.NET** trainer. The faithful drop-ins are
`FastForestFanovaImportanceEvaluator` and `FastForestMeanDecreaseImpurityImportanceEvaluator`
(namespace `OptunaSharp.Importance.Forests`); experimental FastTree / FastTree-Tweedie boosters
(namespace `OptunaSharp.Importance.Booster`) are also provided. These are an off-label extension
beyond Optuna — the four Core evaluators above are the parity surface.

## Writing a custom evaluator

Derive from `BaseImportanceEvaluator`:

```csharp
public abstract OrderedDict<string, double> Evaluate(
    Study study, IReadOnlyList<string>? @params = null, Func<FrozenTrial, double>? target = null);
```

## Relationship to Optuna

| Optuna (Python) | OptunaSharp |
|-----------------|-------------|
| `optuna.importance.get_param_importances(study)` | `ImportanceFunctions.GetParamImportances(study)` |
| `PedAnovaImportanceEvaluator(target_quantile=0.1)` | `new PedAnovaImportanceEvaluator(targetQuantile: 0.1)` |
| `FanovaImportanceEvaluator(seed=0)` | `new FanovaImportanceEvaluator(seed: 0)` |
| `MeanDecreaseImpurityImportanceEvaluator(seed=0)` | `new MeanDecreaseImpurityImportanceEvaluator(seed: 0)` |
| `optuna.integration.ShapleyImportanceEvaluator(seed=0)` | `new ShapleyImportanceEvaluator(seed: 0)` |
| `get_param_importances(study, evaluator=e, normalize=False)` | `GetParamImportances(study, e, normalize: false)` |

Under a fixed seed the forest-based evaluators (fANOVA, MDI, SHAP) reproduce Optuna's importance
values byte-for-byte or to about 1 ULP — their scikit-learn surrogate is a bit-exact port. PED-ANOVA
is computed with plain numeric operations and matches Optuna closely; where the underlying quantities
ride the error-function path, the documented erf tolerance applies. In every case the **ranking** is
the intended, stable output.

## See also

[ML.NET importance backends](../integrations/training/ml-net.md#parameter-importance-backends) ·
[LightGBM importance backends](../integrations/training/lightgbm.md#experimental-importance-evaluators) —
alternative trainer-backed evaluators with different numeric behavior from Core's canonical forest.

[Study and Trials](study-and-trials.md) · [Visualization](visualization.md) — plot importances ·
the [API reference](../api/index.md).
