# Pruners

A **pruner** stops an unpromising trial early, before it wastes a full training budget. During a trial
you report intermediate values (a validation loss per epoch, say); the pruner compares the trial's
trajectory against the others seen so far and decides whether to abandon it. OptunaSharp ports
`optuna.pruners` into **`OptunaSharp.Pruners`**.

## The pruning loop

Pruning is a contract between your objective and the pruner: you `Report` and then check
`ShouldPrune`.

```csharp
using OptunaSharp;

var study = Optuna.CreateStudy(pruner: new MedianPruner(nStartupTrials: 5, nWarmupSteps: 10));

study.Optimize(trial =>
{
    for (int epoch = 0; epoch < 100; epoch++)
    {
        double loss = TrainOneEpoch(trial, epoch);
        trial.Report(loss, epoch);
        if (trial.ShouldPrune())
            throw new TrialPruned();
    }
    return Validate(trial);
}, nTrials: 50);
```

`Optimize` catches `TrialPruned` and records the trial with state `Pruned`. The default pruner (when
you pass none) is `MedianPruner`, matching Optuna.

## The pruners

| Pruner | Strategy | Constructor |
|--------|----------|-------------|
| `MedianPruner` | Prune if worse than the median of past trials at this step | `(nStartupTrials: 5, nWarmupSteps: 0, intervalSteps: 1, nMinTrials: 1)` |
| `PercentilePruner` | Prune below a chosen percentile | `(double percentile, nStartupTrials: 5, nWarmupSteps: 0, intervalSteps: 1, nMinTrials: 1)` |
| `SuccessiveHalvingPruner` | Asynchronous Successive Halving (ASHA) | `(minResource: null, reductionFactor: 4, minEarlyStoppingRate: 0, bootstrapCount: 0)` |
| `HyperbandPruner` | Hyperband over ASHA brackets | `(minResource: 1, maxResource: null, reductionFactor: 3, bootstrapCount: 0)` |
| `ThresholdPruner` | Prune outside a fixed value range | `(lower: null, upper: null, nWarmupSteps: 0, intervalSteps: 1)` |
| `PatientPruner` | Delay another pruner until patience runs out | `(BasePruner? wrappedPruner, int patience, double minDelta = 0.0)` |
| `WilcoxonPruner` | Wilcoxon signed-rank test over per-instance scores | `(pThreshold: 0.1, nStartupSteps: 2)` |
| `NopPruner` | Never prunes | `()` |

## Notes on the main pruners

- **`MedianPruner`** is a special case of `PercentilePruner` at the 50th percentile — the sensible
  default. `nStartupTrials` skips pruning until enough trials exist to compare against;
  `nWarmupSteps` protects the noisy early steps of each trial; `intervalSteps` limits how often the
  check runs.
- **`SuccessiveHalvingPruner`** (ASHA) promotes only the top `1/reductionFactor` of trials to the next
  resource rung, so most trials stop cheaply. Good when you can express training progress as a
  resource budget.
- **`HyperbandPruner`** runs several ASHA brackets with different starting resources and routes trials
  between them, trading off aggressive vs. conservative early stopping automatically. It typically
  outperforms a single ASHA configuration.
- **`ThresholdPruner`** is unconditional: prune as soon as a reported value falls below `lower` or
  above `upper`. Useful for hard feasibility limits (e.g. loss that diverged to NaN/∞).
- **`PatientPruner`** wraps another pruner and suppresses its verdict until the trial has failed to
  improve for `patience` steps — a guard against pruning on transient dips.
- **`WilcoxonPruner`** compares a trial against the best using a statistical test over per-instance
  results; report the per-instance scores through the trial's intermediate values.

## Setting or disabling pruning

```csharp
var aggressive = Optuna.CreateStudy(pruner: new SuccessiveHalvingPruner(reductionFactor: 3));
var never      = Optuna.CreateStudy(pruner: new NopPruner());   // disable pruning entirely
```

## Writing a custom pruner

Derive from `BasePruner` and implement one method:

```csharp
public abstract bool Prune(Study study, FrozenTrial trial);   // true -> the trial should stop
```

`trial.IntermediateValues` gives you the reported trajectory; `study.GetTrials(...)` gives the history
to compare against.

## Relationship to Optuna

| Optuna (Python) | OptunaSharp |
|-----------------|-------------|
| `MedianPruner(n_startup_trials=5, n_warmup_steps=10)` | `new MedianPruner(nStartupTrials: 5, nWarmupSteps: 10)` |
| `SuccessiveHalvingPruner(reduction_factor=3)` | `new SuccessiveHalvingPruner(reductionFactor: 3)` |
| `HyperbandPruner(min_resource=1, max_resource=81)` | `new HyperbandPruner(minResource: 1, maxResource: 81)` |
| `trial.report(loss, step)` / `trial.should_prune()` | `trial.Report(loss, step)` / `trial.ShouldPrune()` |

## See also

[Study and Trials](study-and-trials.md) · [Samplers](samplers.md) · [Terminators](terminators.md) ·
the [API reference](../api/index.md).
