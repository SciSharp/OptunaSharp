# Terminators

A **terminator** stops the *whole study* once more trials are unlikely to help — as opposed to a
[pruner](pruners.md), which stops a single unpromising trial. It compares the potential for further
improvement against the noise in your evaluation, and calls it done when improvement no longer clears
the noise. OptunaSharp ports `optuna.terminator` into **`OptunaSharp.Terminator`**.

> **Deprecated in Optuna.** The `optuna.terminator` module is deprecated as of Optuna v4.9.0 and is
> scheduled for removal in Optuna v6.0.0, with its functionality planned to move to
> [hub.optuna.org](https://hub.optuna.org). OptunaSharp ports it for parity and reproduces the
> deprecation: `Terminator`, `TerminatorCallback`, and `ReportCrossValidationScores` emit a warning
> when used. It remains functional; prefer a fixed trial budget or a [pruner](pruners.md) for new code.

## How termination is decided

A `Terminator` combines two evaluators:

- an **improvement evaluator** — estimates the remaining room to improve (a regret bound); and
- an **error evaluator** — estimates the statistical noise floor of your objective.

When the estimated improvement drops below the estimated error (after a minimum number of trials),
the study should stop.

```csharp
public Terminator(
    BaseImprovementEvaluator? improvementEvaluator = null,  // default: RegretBoundEvaluator
    BaseErrorEvaluator? errorEvaluator = null,              // default: CrossValidationErrorEvaluator
    int minNTrials = 20);

public abstract bool ShouldTerminate(Study study);         // the decision
```

## Using it in a run

The usual way is `TerminatorCallback`, which checks after each trial and stops the study:

```csharp
using OptunaSharp;
using OptunaSharp.Terminator;

var callback = new TerminatorCallback();   // wraps a default Terminator
study.Optimize(Objective, nTrials: 400, callbacks: new[] { callback.Invoke });
// The study stops early once the terminator is satisfied, even though nTrials allows 400.
```

Or query a `Terminator` directly in your own loop:

```csharp
var terminator = new Terminator(minNTrials: 50);
while (!terminator.ShouldTerminate(study))
{
    Trial trial = study.Ask();
    study.Tell(trial, Evaluate(trial));
}
```

## Improvement evaluators

| Evaluator | Idea | Constructor |
|-----------|------|-------------|
| `RegretBoundEvaluator` (default) | Gaussian-process regret bound over the top trials | `(topTrialsRatio: 0.5, minNTrials: 20, seed: null)` |
| `BestValueStagnationEvaluator` | Stop after N trials with no new best | `(maxStagnationTrials: 30)` |
| `EMMREvaluator` | Expected-minimum model regret (a GP criterion) | `(deterministicObjective: false, delta: 0.1, minNTrials: 2, seed: null)` |

`RegretBoundEvaluator` and `EMMREvaluator` build on the same Gaussian-process stack as the
[`GPSampler`](samplers.md); their output tracks Optuna to a documented tolerance (see below).

## Error evaluators

| Evaluator | Idea | Constructor |
|-----------|------|-------------|
| `CrossValidationErrorEvaluator` (default) | Scale of the best trial's cross-validation scores | `()` |
| `StaticErrorEvaluator` | A fixed error floor you supply | `(double constant)` |
| `MedianErrorEvaluator` | Median improvement × a ratio | `(BaseImprovementEvaluator pairedImprovementEvaluator, warmUpTrials: 10, nInitialTrials: 20, thresholdRatio: 0.01)` |

To use cross-validation error, report the per-fold scores during the trial:

```csharp
TerminatorFunctions.ReportCrossValidationScores(trial, new[] { 0.91, 0.89, 0.90, 0.92, 0.88 });
```

## Relationship to Optuna

| Optuna (Python) | OptunaSharp |
|-----------------|-------------|
| `Terminator(min_n_trials=50)` | `new Terminator(minNTrials: 50)` |
| `study.optimize(obj, callbacks=[TerminatorCallback()])` | `study.Optimize(obj, callbacks: new[] { new TerminatorCallback().Invoke })` |
| `RegretBoundEvaluator(top_trials_ratio=0.5)` | `new RegretBoundEvaluator(topTrialsRatio: 0.5)` |
| `report_cross_validation_scores(trial, scores)` | `TerminatorFunctions.ReportCrossValidationScores(trial, scores)` |

**Divergence.** The GP-based evaluators (`RegretBoundEvaluator`, `EMMREvaluator`) inherit the GP
stack's tolerance boundary: the managed kernel fit drifts the hyperparameters by ~1e-3, which
propagates through the criterion, so their values match Optuna to a few 1e-3 relative rather than
bit-for-bit. The termination *decision* agrees with Optuna in tests within that tolerance.

## See also

[Pruners](pruners.md) — early-stopping individual trials · [Samplers](samplers.md) — the GP stack the
evaluators reuse · the [API reference](../api/index.md).
