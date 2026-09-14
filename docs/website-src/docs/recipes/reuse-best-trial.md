# Re-use the best trial

After tuning, you often want to run the objective *once more* with the best hyperparameters — to
compute extra metrics, to retrain on the full dataset, or to analyze the winning model in detail.
`Study.BestTrial` is a [`FrozenTrial`](../study-and-trials.md#trial-types) whose stored parameters can
be *replayed*: passing it to a function that calls `Suggest*` returns the recorded values instead of
sampling new ones.

## Tune, then re-evaluate

First a normal optimization:

```csharp
using OptunaSharp;

double Objective(BaseTrial trial)
{
    double c = trial.SuggestFloat("C", 1e-7, 10.0, log: true);
    return TrainAndScore(c);          // accuracy on the test split
}

var study = Optuna.CreateStudy(direction: StudyDirection.Maximize);
study.Optimize(Objective, nTrials: 10);

Console.WriteLine(study.BestTrial.Value);   // the best accuracy
```

Note the objective takes **`BaseTrial`**, the shared base of both `Trial` (the live handle) and
`FrozenTrial` (a stored snapshot). That one signature lets the *same* method be driven by the study
during tuning **and** be replayed from a frozen trial afterward.

Now define a second function that reproduces the winning model and computes more metrics. It shares
the `Suggest*` calls, so replaying `study.BestTrial` rebuilds exactly the best configuration:

```csharp
(double acc, double f1, double recall, double precision) DetailedObjective(BaseTrial trial)
{
    double c = trial.SuggestFloat("C", 1e-7, 10.0, log: true);   // replayed from the frozen trial
    var model = Train(c);
    return (model.Accuracy, model.F1, model.Recall, model.Precision);
}

// Pass the frozen best trial — its stored C is returned by SuggestFloat above.
var metrics = DetailedObjective(study.BestTrial);
```

## `BestTrial` behaves differently from a live trial

A `FrozenTrial` is not an active trial, so some methods are inert on it — in particular
`ShouldPrune()` always returns `false` and `Report(...)` does nothing. That's exactly what you want
when replaying: there is no sampler or pruner in the loop, just the stored parameters.

## Multi-objective studies

For a [multi-objective](multi-objective.md) study there is no single best trial;
`Study.BestTrials` returns the list of `FrozenTrial`s on the Pareto front. Replay any of them the same
way:

```csharp
foreach (FrozenTrial t in study.BestTrials)
    Analyze(DetailedObjective(t));
```

## Relationship to Optuna

| Optuna (Python) | OptunaSharp |
|-----------------|-------------|
| `study.best_trial` | `study.BestTrial` |
| `detailed_objective(study.best_trial)` (objective takes a trial) | `DetailedObjective(study.BestTrial)` (objective takes `BaseTrial`) |
| `study.best_trial.value` | `study.BestTrial.Value` |
| `study.best_trials` (Pareto front) | `study.BestTrials` |

## See also

[Study and Trials](../study-and-trials.md) · [Multi-objective optimization](multi-objective.md) ·
[Specify hyperparameters manually](specify-params.md) · the [API reference](../../api/index.md).
