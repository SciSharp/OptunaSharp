# User-defined pruner

A [pruner](../pruners.md) decides when to stop an unpromising trial early, based on the intermediate
values it has reported. This recipe shows how to write your own by subclassing `BasePruner` — here, an
aggressive pruner that stops any trial in *last place* among the completed trials at the same step.

## The pruner interface

Subclass `BasePruner` and implement one method:

```csharp
public abstract bool Prune(Study study, FrozenTrial trial);   // true => prune this trial
```

Inside it you have the whole study: reach other trials with `study.GetTrials(...)`, and read a trial's
reported values from `FrozenTrial.IntermediateValues` (a step→value map) and `FrozenTrial.LastStep`.

## Example: a `LastPlacePruner`

The pruner starts considering a trial only after a warmup of `warmupSteps` steps and `warmupTrials`
completed trials, then prunes it if its latest score is worse than *every* completed trial's score at
the same step.

```csharp
using System;
using System.Linq;
using OptunaSharp;
using OptunaSharp.Pruners;

public sealed class LastPlacePruner : BasePruner
{
    private readonly int _warmupSteps;
    private readonly int _warmupTrials;

    public LastPlacePruner(int warmupSteps, int warmupTrials)
    {
        _warmupSteps = warmupSteps;
        _warmupTrials = warmupTrials;
    }

    public override bool Prune(Study study, FrozenTrial trial)
    {
        int? step = trial.LastStep;      // null until this trial has reported a value
        if (step is null)
            return false;

        double thisScore = trial.IntermediateValues[step.Value];

        // Scores from other completed trials at the same step.
        var completed = study.GetTrials(deepcopy: false, states: new[] { TrialState.Complete });
        var otherScores = completed
            .Where(t => t.IntermediateValues.ContainsKey(step.Value))
            .Select(t => t.IntermediateValues[step.Value])
            .OrderBy(s => s)
            .ToList();

        // Prune once past warmup, and only if strictly worse than all peers.
        if (step.Value >= _warmupSteps && otherScores.Count > _warmupTrials)
        {
            if (thisScore < otherScores[0])
            {
                Console.WriteLine($"prune() True: Trial {trial.Number}, Step {step}, Score {thisScore}");
                return true;
            }
        }
        return false;
    }
}
```

Confirm it works with a maximizing objective that reports a score at each of five steps:

```csharp
double Objective(Trial trial)
{
    string loss  = trial.SuggestCategorical("loss", new[] { "hinge", "log_loss", "perceptron" });
    double alpha = trial.SuggestFloat("alpha", 1e-5, 1e-3, log: true);

    double score = 0;
    for (int step = 0; step < 5; step++)
    {
        score = TrainOneStepAndScore(loss, alpha, step);   // your model here
        trial.Report(score, step);
        if (trial.ShouldPrune())
            throw new TrialPruned();
    }
    return score;
}

var pruner = new LastPlacePruner(warmupSteps: 1, warmupTrials: 5);
var study = Optuna.CreateStudy(direction: StudyDirection.Maximize, pruner: pruner);
study.Optimize(Objective, nTrials: 50);
```

When `ShouldPrune()` returns `true`, throw `TrialPruned`; OptunaSharp records the trial as `Pruned`
and takes its *last reported* intermediate value as the trial's value.

## Relationship to Optuna

| Optuna (Python) | OptunaSharp |
|-----------------|-------------|
| `class MyPruner(optuna.pruners.BasePruner)` | `class MyPruner : BasePruner` |
| `def prune(self, study, trial) -> bool` | `public override bool Prune(Study study, FrozenTrial trial)` |
| `trial.last_step` | `trial.LastStep` |
| `trial.intermediate_values[step]` | `trial.IntermediateValues[step]` |
| `study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,))` | `study.GetTrials(deepcopy: false, states: new[] { TrialState.Complete })` |
| `raise optuna.TrialPruned()` | `throw new TrialPruned()` |

## See also

[Pruners](../pruners.md) · [Wilcoxon pruner](wilcoxon-pruner.md) ·
[User-defined sampler](user-defined-sampler.md) · the [API reference](../../api/index.md).
