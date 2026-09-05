# OptunaSharp.Importance.Booster.ML.NET

> ⚠️ **Experimental / non-Optuna.** Read this section before using the package.

**Gradient-boosted-tree backends** (ML.NET `FastTree` and `FastTreeTweedie`) for the forest-based
parameter-importance evaluators of [OptunaSharp](https://github.com/SciSharp/OptunaSharp) — a .NET
port of [Optuna](https://github.com/optuna/optuna), part of the
[SciSharp STACK](https://github.com/SciSharp).

## This is off-label — Optuna has no boosting importance

Optuna's importance module (`optuna.importance`) uses **only**
`sklearn.ensemble.RandomForestRegressor` (bagging) for fANOVA and Mean-Decrease-Impurity. **There is
no gradient boosting anywhere in Optuna's importance.** These evaluators fit boosted trees instead:

- **Bagging / random forest** (what fANOVA/MDI were designed for): independent trees, each fits the
  target `y`, ensemble = **average**.
- **Gradient boosting** (this package — `FastTree` = MART, `FastTreeTweedie` = Tweedie-loss MART):
  sequential trees, each fits the **residual** of the ensemble-so-far, ensemble = **weighted sum**.
  The trees model residuals, not the objective, so running fANOVA/MDI over them means something
  **different** — a non-standard quantity with **no basis in Optuna**.

Consequently these evaluators have **no value-parity with Python Optuna** and only weak ranking
agreement. **Prefer** the canonical, value-exact evaluators in Core, or the faithful ML.NET FastForest
backend:

| Evaluator | Package | Model | Optuna parity |
|---|---|---|---|
| `FanovaImportanceEvaluator` / `MeanDecreaseImpurityImportanceEvaluator` | **Core** (`OptunaSharp`) | native CART, bit-exact scikit-learn | **value-exact (~1 ULP)** |
| `FastForest…ImportanceEvaluator` | `OptunaSharp.Importance.Forests.ML.NET` | ML.NET FastForest (bagging) | ranking-only, faithful model |
| `FastTree…` / `FastTreeTweedie…` | **this package** | ML.NET gradient boosting | ⚠ off-label, weak ranking only |

## Evaluators

- `FastTreeFanovaImportanceEvaluator`, `FastTreeMeanDecreaseImpurityImportanceEvaluator`
- `FastTreeTweedieFanovaImportanceEvaluator`, `FastTreeTweedieMeanDecreaseImpurityImportanceEvaluator`

All live in namespace `OptunaSharp.Importance.Booster` and take `nTrees` (default 64), `maxDepth`
(default 64), and `seed` (default `null`).

## Usage

```csharp
using OptunaSharp;
using OptunaSharp.Importance;          // ImportanceFunctions, canonical evaluators
using OptunaSharp.Importance.Booster;  // the experimental booster backends (this package)

var study = Optuna.CreateStudy(sampler: new RandomSampler(seed: 42));
study.Optimize(t => { double x = t.SuggestFloat("x", -10, 10); return x * x; }, nTrials: 100);

// EXPERIMENTAL booster backend — ranking only, no Optuna value-parity:
var imp = ImportanceFunctions.GetParamImportances(
    study, new FastTreeFanovaImportanceEvaluator(seed: 0));

// For Optuna parity, use the canonical Core evaluator instead:
var canonical = ImportanceFunctions.GetParamImportances(study, new FanovaImportanceEvaluator(seed: 0));
```

### Full control: the ML.NET trainer options

Each evaluator also has an overload taking the **complete ML.NET trainer options**
(`FastTreeRegressionTrainer.Options` / `FastTreeTweedieTrainer.Options`), so the entire boosting
surface is controllable — learning rate, shrinkage, bagging, pruning, optimization algorithm, early
stopping, and (for Tweedie) the variance-power `Index`:

```csharp
using Microsoft.ML.Trainers.FastTree;

var evaluator = new FastTreeFanovaImportanceEvaluator(new FastTreeRegressionTrainer.Options
{
    NumberOfTrees = 300,
    NumberOfLeaves = 32,
    LearningRate = 0.05,          // absent from the simple ctor
    Shrinkage = 1.0,
    MinimumExampleCountPerLeaf = 5,
    Seed = 0,
    NumberOfThreads = 1,          // set Seed + a single thread for a reproducible result
});
var imp = ImportanceFunctions.GetParamImportances(study, evaluator);
```

`LabelColumnName` / `FeatureColumnName` are managed by the evaluator and overwritten; every other field
is used as given. (Of the trainer's four column-name fields, only those two can bind — the importance
schema has no weight or group column, so `ExampleWeightColumnName` / `RowGroupColumnName` are
unsupported and throw a clear `ArgumentException` if set.) The `(nTrees, maxDepth, seed)` constructor is
this overload fed a reproducible default options object.

Or **start from those reproducible defaults and tweak** just a few knobs with an
`Action<FastTreeRegressionTrainer.Options>` (or `...FastTreeTweedieTrainer.Options`):

```csharp
var evaluator = new FastTreeFanovaImportanceEvaluator(
    o => { o.LearningRate = 0.05; o.Seed = 0; o.NumberOfThreads = 1; });
```

## Implementation

The fANOVA marginal-variance walk and the MDI formula are **shared with Core** (validated bit-for-bit
against Optuna's own `test_tree.py`); only the underlying tree ensemble differs. Each ML.NET tree is
adapted into Core's flat sklearn-like tree by a pre-order DFS, and the training set is routed through
each tree to recompute per-node impurity and counts (ML.NET exposes no per-node statistics). The
ensemble is fitted single-threaded from the given `seed` for reproducibility. It pulls a foreign
runtime (native FastTree), so it ships **separately from Core** — the `NumSharp.Interop.OpenBLAS`
optional-package pattern. No new native dependency beyond the FastForest backend's.

## License

MIT — see the repository `LICENSE`.
