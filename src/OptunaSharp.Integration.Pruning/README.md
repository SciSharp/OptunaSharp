# OptunaSharp.Integration.Pruning

The **named framework pruning callbacks** for [OptunaSharp](https://github.com/SciSharp/OptunaSharp) —
a .NET port of [Optuna](https://github.com/optuna/optuna), part of the
[SciSharp STACK](https://github.com/SciSharp). Faithful ports of the remaining
[`optuna_integration`](https://optuna-integration.readthedocs.io/) pruning callbacks.

Each callback reports the current validation metric to the trial and prunes an unpromising trial via the
study's pruner — matching Optuna's per-callback metric lookup, step semantics, warnings and verbatim
messages.

## Pure managed — one package, not one per framework

Unlike the ML.NET / LightGBM integrations, **none of these callbacks needs a foreign ML runtime.** In
Python the training framework invokes the callback and hands it the per-iteration evaluation data; here
the callback consumes that data modeled as plain C# structures and is **driven from your own training
loop** — the same convention as OptunaSharp's `LightGBMPruningCallback` (no .NET GBM/DL trainer exposes
a managed per-iteration hook). Because there is no runtime dependency, they all ship in **one
pure-managed package** (depending only on Core), one source file per Optuna module so the C# diffs
against the Python — rather than fragmenting into a premature heavy-runtime package per framework.

| Callback | Optuna module | Convention |
|---|---|---|
| `CatBoostPruningCallback` | `catboost` | **deferred** — `AfterIteration` returns `false`, then `CheckPruned` raises |
| `XGBoostPruningCallback` | `xgboost` | immediate — `AfterIteration` raises `TrialPruned` |
| `KerasPruningCallback` | `keras` | immediate — `OnEpochEnd` (with a report `interval`) |
| `TFKerasPruningCallback` | `tfkeras` | immediate — `OnEpochEnd` |
| `TensorFlowPruningHook` | `tensorflow` | immediate — `AfterRun` (⚠ deprecated, prefer `KerasPruningCallback`) |
| `PyTorchIgnitePruningHandler` | `pytorch_ignite` | immediate — `Call` |
| `PyTorchLightningPruningCallback` | `pytorch_lightning` | immediate + DDP — `OnValidationEnd` / `CheckPruned` |
| `SkorchPruningCallback` | `skorch` | immediate — `OnEpochEnd` |
| `FastAIV2PruningCallback` (alias `FastAIPruningCallback`) | `fastaiv2` | two-stage — `AfterEpoch` (`CancelFitException`) then `AfterFit` (`TrialPruned`) |

## Usage

Each callback wraps a `Trial` plus its framework-specific configuration (metric / monitor / observation
key). You invoke it from the loop that produces the per-step metric, and it reports + prunes.

```csharp
using OptunaSharp;
using OptunaSharp.Integration;   // the named pruning callbacks

var study = Optuna.CreateStudy(
    direction: StudyDirection.Minimize,
    pruner: new MedianPruner());

study.Optimize(trial =>
{
    // XGBoost-style: raise TrialPruned as soon as the trial is unpromising.
    var pruning = new XGBoostPruningCallback(trial, observationKey: "validation-error");
    var evalsLog = new XGBoostEvalsLog();

    for (int round = 0; round < 100; round++)
    {
        double validationError = TrainOneRound();          // your engine
        evalsLog.Add("validation", "error", validationError);
        pruning.AfterIteration(round, evalsLog);           // reports + throws TrialPruned when unpromising
    }
    return FinalError();
}, nTrials: 50);
```

The **deferred** (CatBoost) convention returns `false` instead of throwing, so an engine whose callback
cannot propagate an exception can stop cleanly; you then call `CheckPruned()` after training:

```csharp
var pruning = new CatBoostPruningCallback(trial, metric: "Logloss");
for (int iteration = 1; !stopped; iteration++)
{
    var info = new CatBoostEvaluationInfo(iteration, metricHistory);   // your per-iteration history
    if (!pruning.AfterIteration(info))
        break;                                                          // stop, without throwing
}
pruning.CheckPruned();   // raises TrialPruned if the trial was pruned
```

The Keras / tf.keras / PyTorch Lightning callbacks emit the same missing-metric `UserWarning` Optuna
does (via OptunaSharp's logging seam) and skip the step when the monitored metric is absent.

## See also

- **`PruningCallback`** in `OptunaSharp.Integration.ML.NET` — the framework-agnostic
  `report` → `should_prune` → raise helper (plus the deferred `AfterIteration` / `CheckPruned`), for any
  iterative / epoch-based loop where you already have a scalar metric.
- **`LightGBMPruningCallback`** in `OptunaSharp.Integration.LightGbm` — the faithful named LightGBM
  callback (metric / valid-name / report-interval lookup + the study-direction consistency check), fed
  the per-iteration evaluation log.

## License

MIT — see the repository `LICENSE`.
