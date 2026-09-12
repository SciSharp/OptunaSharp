# Training-loop pruning

`OptunaSharp.Integration.PruningCallback` connects an iterative training loop to a trial's pruner.
It ships in **`OptunaSharp` (Core)** and has no training-framework or Python dependency. It neither
trains a model nor selects the metric: your loop supplies a scalar validation value and a step.

```bash
dotnet add package OptunaSharp
```

## Immediate pruning

`Report(value, step)` calls `Trial.Report` and `Trial.ShouldPrune`. When the pruner rejects the
trial, it throws `TrialPruned`. Let that exception reach `Study.Optimize`, which records the trial
as pruned and continues the study.

This complete example uses a small iterative optimization loop as a stand-in for an epoch-based
trainer. Replace the update and validation loss with your model's training and evaluation calls.

```csharp
using OptunaSharp;
using OptunaSharp.Integration;
using OptunaSharp.Pruners;

var study = Optuna.CreateStudy(
    direction: StudyDirection.Minimize,
    pruner: new MedianPruner(nStartupTrials: 5, nWarmupSteps: 3));

study.Optimize(trial =>
{
    var pruning = new PruningCallback(trial);
    double learningRate = trial.SuggestFloat("learning_rate", 0.01, 0.5, log: true);
    double weight = 0;
    double loss = 4;
    for (int epoch = 0; epoch < 30; epoch++)
    {
        weight -= learningRate * 2 * (weight - 2);
        loss = (weight - 2) * (weight - 2);
        pruning.Report(loss, epoch);
    }
    return loss;
}, nTrials: 20);
```

Use zero-based, non-negative steps and comparable metrics at the same step across trials. The
underlying trial keeps the first value reported at a duplicate step. Intermediate reporting and
`ShouldPrune` do not support multi-objective studies.

## Deferred pruning

Some training engines require their callbacks to return a continuation flag instead of throwing.
`AfterIteration(value, step)` returns `false` when training should stop and stores the pruning
decision. Always call `CheckPruned()` after leaving that loop:

```csharp
double Objective(Trial trial)
{
    var pruning = new PruningCallback(trial);
    double rate = trial.SuggestFloat("rate", 0.01, 0.5, log: true);
    double weight = 0;
    double loss = 4;
    for (int step = 0; step < 30; step++)
    {
        weight -= rate * 2 * (weight - 2);
        loss = (weight - 2) * (weight - 2);
        if (!pruning.AfterIteration(loss, step))
            break;
    }
    pruning.CheckPruned();
    return loss;
}
```

Without `CheckPruned`, returning a loss would mark the shortened run complete. `Pruned` exposes the
stored decision. Create a fresh callback for each trial; it retains state and has no reset method.

## Framework hooks and limits

The application must invoke the callback. An ML.NET `Fit` call does not acquire intermediate
pruning merely because a `PruningCallback` exists. In particular, the ML.NET LightGBM adapter does
not expose a managed per-boosting-iteration hook. Its
[`LightGBMPruningCallback`](lightgbm.md#metric-specific-pruning) also needs caller-supplied metrics.

`OptunaSearchCV` does not implement Python's `partial_fit`-based `enable_pruning` / `max_iter` path.
Use this helper inside an estimator you control, or implement the estimator's training contract
explicitly. A subclass can read the protected `Trial` and override `FormatPrunedMessage(int step)`
to use framework terminology such as “epoch.”

## Implementation and verification

The [callback implementation](https://github.com/SciSharp/OptunaSharp/blob/master/src/OptunaSharp.Core/Integration/PruningCallback.cs)
contains both reporting styles. The
[callback tests](https://github.com/SciSharp/OptunaSharp/blob/master/test/OptunaSharp.Tests/PruningCallbackTests.cs)
cover immediate and deferred exceptions, a no-op check when no pruning occurred, and pruning
inside a real `Study.Optimize` loop. See the [Pruners guide](../../docs/pruners.md) for choosing
the decision rule.
