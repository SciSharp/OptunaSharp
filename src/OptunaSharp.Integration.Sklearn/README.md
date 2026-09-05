# OptunaSharp.Integration.Sklearn

The **scikit-learn integration** for [OptunaSharp](https://github.com/SciSharp/OptunaSharp) — a .NET
port of [Optuna](https://github.com/optuna/optuna). The .NET analogue of
`optuna_integration.sklearn`.

Unlike the other `OptunaSharp.Integration.*` satellites, this one pulls **no foreign runtime**. It is a
pure-managed, **byte-exact** port of the scikit-learn pieces that `OptunaSearchCV` drives, built on
Core's already-bit-exact scikit-learn CART:

- **`RandomForestRegressor`** — fit + predict + score, reproducing `sklearn.ensemble.RandomForestRegressor`'s
  exact per-tree RNG (tree seeds, bootstrap sample counts, and the splitter's `our_rand_r` seed) over the
  full parameter surface (`n_estimators`, `max_depth`, `min_samples_split`/`_leaf`,
  `min_weight_fraction_leaf`, `max_features`, `min_impurity_decrease`, `bootstrap`, `max_samples`).
- **`KFold`** / **`check_cv`** — the contiguous k-fold splitter (with the MT19937 `shuffle` path).
- **`cross_validate`** — the per-fold fit + score loop.
- **`r2`** / **`neg_mean_squared_error`** scorers (`sklearn.metrics.r2_score` / `mean_squared_error`).

## End-to-end byte-exact hyperparameter search

`SklearnEstimator` implements the Core `ICrossValidationEstimator` seam, so the Core `OptunaSearchCV`
runs **end-to-end byte-exact** against Python's `optuna_integration.OptunaSearchCV`:

```csharp
using OptunaSharp.Distributions;
using OptunaSharp.Integration;
using OptunaSharp.Integration.Sklearn;

// X: double[n, d] features, y: double[n] targets
var forest = new RandomForestRegressor(randomState: 0);
var estimator = new SklearnEstimator(forest, X, y, cv: new KFold(5));

var dists = new Dictionary<string, BaseDistribution>
{
    ["n_estimators"]     = new IntDistribution(10, 100),
    ["max_depth"]        = new IntDistribution(2, 32),
    ["min_samples_split"] = new IntDistribution(2, 16),
};

var search = new OptunaSearchCV(estimator, dists, nTrials: 20, randomState: 42).Fit();

Console.WriteLine(search.BestScore);                 // == study_.best_value
Console.WriteLine(search.BestParams["max_depth"]);   // == study_.best_params
var yPred = estimator.Predict(Xtest);                // the refit best forest
```

With integer / categorical `param_distributions`, TPE samples the same values on both sides
(bit-exact), so the search visits identical configurations and every CV score matches Python
bit-for-bit. (Float `param_distributions` sample within TPE's erf-tolerance, as elsewhere in OptunaSharp.)

Value parity is host-pinned (the float32 forest accumulation + the platform `np.int_` bootstrap width);
the interop suite asserts it on x64.

Part of the [SciSharp STACK](https://github.com/SciSharp).
