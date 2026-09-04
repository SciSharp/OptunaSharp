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

## Planned (Phase 5.7): the LightGBM **tuner** — the Optuna-justified half

LightGBM's real identity in Optuna is a **tuner**, not importance:
[`optuna.integration.lightgbm`](https://optuna.readthedocs.io/en/stable/reference/integration.html)'s
**`LightGBMTuner`** / **`LightGBMTunerCV`** — a stepwise tuner that sweeps the core LightGBM
hyperparameters (`feature_fraction`, `num_leaves`, `bagging`, `min_child_samples`, regularization) in a
fixed, well-tested order using Optuna under the hood, as a drop-in for `lightgbm.train`.

This is the **Optuna-justified** addition and is scoped to **this package** (one package per heavy
native dep), to be implemented under **Phase 5.7 (framework integrations)** — see
`docs/PORTING_PLAN.md`. It is **not** part of the current importance-backends work; the importance
evaluators above are an OptunaSharp-only extension. Until then, this package hosts the importance
backends only.

## License

MIT — see the repository `LICENSE`.
