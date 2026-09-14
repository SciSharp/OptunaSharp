# Callback for `Study.Optimize`

A **callback** is invoked after every trial finishes, receiving the `Study` and the resulting
`FrozenTrial`. Use one to log to an experiment tracker, checkpoint state, or — as here — implement a
custom stopping rule. In OptunaSharp a callback is an `Action<Study, FrozenTrial>`, passed to
`Optimize` through the `callbacks` parameter.

## Stop after N pruned trials in a row

This stateful callback counts consecutive pruned trials and calls `study.Stop()` once the count
reaches a threshold. Because it holds state across calls, model it as a small class and pass its
`Invoke` method:

```csharp
using OptunaSharp;

// Stops the study once `threshold` trials are pruned back-to-back.
sealed class StopWhenTrialKeepBeingPrunedCallback
{
    private readonly int _threshold;
    private int _consecutivePrunedCount;

    public StopWhenTrialKeepBeingPrunedCallback(int threshold) => _threshold = threshold;

    // Signature matches Action<Study, FrozenTrial>: run after every finished trial.
    public void Invoke(Study study, FrozenTrial trial)
    {
        if (trial.State == TrialState.Pruned)
            _consecutivePrunedCount++;      // another prune in the streak
        else
            _consecutivePrunedCount = 0;    // a non-pruned trial resets the streak

        if (_consecutivePrunedCount >= _threshold)
            study.Stop();                   // ask the optimize loop to exit after running trials finish
    }
}
```

The objective prunes every trial after the first five (`trial.Number` is zero-based):

```csharp
double Objective(Trial trial)
{
    if (trial.Number > 4)
        throw new TrialPruned();            // Optimize catches this and records the trial as Pruned

    return trial.SuggestFloat("x", 0, 1);
}
```

Wire the callback in with a threshold of `2` — the study stops once two trials are pruned in a row, so
it finishes after 7 trials even though we asked for 10:

```csharp
var callback = new StopWhenTrialKeepBeingPrunedCallback(2);
var study = Optuna.CreateStudy();
study.Optimize(Objective, nTrials: 10, callbacks: new Action<Study, FrozenTrial>[] { callback.Invoke });
```

`study.Stop()` only takes effect inside an objective or a callback; it lets the currently running
trials finish and then returns from `Optimize`.

## Relationship to Optuna

| Optuna (Python) | OptunaSharp |
|-----------------|-------------|
| a callable `callback(study, trial)` | an `Action<Study, FrozenTrial>` |
| `trial.state == TrialState.PRUNED` | `trial.State == TrialState.Pruned` |
| `study.stop()` | `study.Stop()` |
| `study.optimize(obj, n_trials=10, callbacks=[cb])` | `study.Optimize(obj, nTrials: 10, callbacks: new Action<Study, FrozenTrial>[] { cb.Invoke })` |

## See also

[Study and Trials](../study-and-trials.md) · [Pruners](../pruners.md) ·
[Ask-and-tell interface](ask-and-tell.md) · the [API reference](../../api/index.md).
