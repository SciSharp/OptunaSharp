# OptunaSharp.Importance.Forests.ML.NET

An **ML.NET FastForest backend** for the forest-based parameter-importance evaluators of
[OptunaSharp](https://github.com/SciSharp/OptunaSharp) — a .NET port of
[Optuna](https://github.com/optuna/optuna), part of the [SciSharp STACK](https://github.com/SciSharp).

## Two backends

OptunaSharp ships fANOVA and Mean-Decrease-Impurity importance in **two forms**:

| Evaluator | Where | Random forest | Optuna parity |
|---|---|---|---|
| `FanovaImportanceEvaluator` | **Core** (`OptunaSharp`) | pure-managed, **bit-exact port of scikit-learn's `RandomForestRegressor`** | **value-exact (~1 ULP)** |
| `MeanDecreaseImpurityImportanceEvaluator` | **Core** (`OptunaSharp`) | same native CART | **value-exact (~1 ULP)** |
| `FastForestFanovaImportanceEvaluator` | **this package** | ML.NET `FastForest` | ranking-only (values diverge) |
| `FastForestMeanDecreaseImpurityImportanceEvaluator` | **this package** | ML.NET `FastForest` | ranking-only (values diverge) |

The **canonical** evaluators live in Core and are pure-managed (NumSharp only): they fit a
byte-for-byte reproduction of scikit-learn's CART (`our_rand_r`, the MSE criterion, the introsort
partitioner, the best-splitter, the depth-first builder, and bootstrap bagging on NumSharp's
NumPy-exact MT19937), so with a fixed seed their importances **match Python Optuna value-for-value**
(the residual is ~1 ULP from numpy's pairwise float reductions). **You do not need this package for
Optuna parity.**

**This package** is an *alternative backend*: it fits the forest with
[ML.NET](https://github.com/dotnet/machinelearning)'s `FastForest` instead. Use it if you specifically
want ML.NET's random forest (e.g. consistency with an existing ML.NET pipeline). It pulls a foreign
runtime (native FastTree), so it ships **separately from Core** — the
`NumSharp.Interop.OpenBLAS` optional-package pattern.

## Usage

```csharp
using OptunaSharp;
using OptunaSharp.Importance;          // canonical evaluators (Core)
using OptunaSharp.Importance.Forests;  // FastForest* backend (this package)

var study = Optuna.CreateStudy(sampler: new RandomSampler(seed: 42));
study.Optimize(t => { double x = t.SuggestFloat("x", -10, 10); return x * x; }, nTrials: 100);

// Canonical (Core, no package needed) — matches Optuna value-for-value:
var fanova = ImportanceFunctions.GetParamImportances(study, new FanovaImportanceEvaluator(seed: 0));

// ML.NET FastForest backend (this package):
var ffFanova = ImportanceFunctions.GetParamImportances(
    study, new FastForestFanovaImportanceEvaluator(seed: 0));
```

Both accept `nTrees` (default 64), `maxDepth` (default 64), and `seed` (default `null`). The default
`GetParamImportances` evaluator (when none is passed) is the pure-managed
`PedAnovaImportanceEvaluator` (Core).

### Full control: the ML.NET trainer options

Each evaluator also has an overload that takes the **complete `FastForestRegressionTrainer.Options`**,
so every knob of the ML.NET forest is controllable — not just three:

```csharp
using Microsoft.ML.Trainers.FastTree;

var evaluator = new FastForestFanovaImportanceEvaluator(new FastForestRegressionTrainer.Options
{
    NumberOfTrees = 200,
    NumberOfLeaves = 50,
    FeatureFraction = 0.7,              // per-split feature subsampling — absent from the simple ctor
    MinimumExampleCountPerLeaf = 5,
    MaximumBinCountPerFeature = 512,
    Seed = 0,
    NumberOfThreads = 1,               // set Seed + a single thread for a reproducible result
});
var imp = ImportanceFunctions.GetParamImportances(study, evaluator);
```

`LabelColumnName` / `FeatureColumnName` are managed by the evaluator (bound to the transformed-trial
schema) and are overwritten; **every other field is used as given**. The `(nTrees, maxDepth, seed)`
constructor is exactly this overload fed a reproducible default options object.

## Divergence from Python Optuna (this package's backend)

ML.NET's `FastForest` is not tree-identical to scikit-learn's random forest — it bins features and
grows trees leaf-wise (not depth-wise), thresholds are float32, and for MDI the node impurities are
recomputed by routing the training set through the fitted trees (ML.NET exposes no per-node
statistics). Consequently the **importance values differ numerically** from Python Optuna — the
ranking of a clearly-dominant parameter is robust, exact values are not. The fANOVA marginal-variance
algorithm and the MDI formula themselves are shared with Core (and validated bit-for-bit against
Optuna's own `test_tree.py`); only the underlying forest differs. For reproducibility the forest is
fitted single-threaded from the given `seed`.

## License

MIT — see the repository `LICENSE`.
