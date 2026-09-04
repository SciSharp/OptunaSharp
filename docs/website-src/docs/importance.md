# Parameter Importance

After a study, **parameter importance** tells you which hyperparameters actually drove the objective
and which barely mattered — so you can widen the ranges that count and fix the ones that don't.
OptunaSharp ports `optuna.importance` into **`OptunaSharp.Importance`**.

## Computing importances

```csharp
using OptunaSharp;
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

## PED-ANOVA (the default evaluator)

The default and only built-in evaluator is `PedAnovaImportanceEvaluator` (namespace
`OptunaSharp.Importance.PedAnova`), a fast, sampler-agnostic method based on the Pearson-divergence
ANOVA of the distribution of good trials against all trials.

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

> **Roadmap.** Optuna's forest-based evaluators (`FanovaImportanceEvaluator`,
> `MeanDecreaseImpurityImportanceEvaluator`) depend on a random-forest backend and are planned for a
> later phase. PED-ANOVA is the evaluator available today.

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
| `PedAnovaImportanceEvaluator(baseline_quantile=0.1)` | `new PedAnovaImportanceEvaluator(targetQuantile: 0.1)` |
| `get_param_importances(study, evaluator=e, normalize=False)` | `GetParamImportances(study, e, normalize: false)` |

The importance *values* on the PED-ANOVA path are computed with plain numeric operations and match
Optuna closely; where the underlying quantities ride the error-function path, the documented erf
tolerance applies. The **ranking** is the intended, stable output.

## See also

[Study and Trials](study-and-trials.md) · [Visualization](visualization.md) — plot importances ·
the [API reference](../api/index.md).
