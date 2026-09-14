# Ask-and-tell interface

`Optimize` runs the loop for you. The **ask-and-tell** interface hands the loop back: you call
`study.Ask()` to get a trial, evaluate it however you like, and call `study.Tell(...)` to report the
result. This recipe shows the three situations where that control pays off.

## 1. Apply OptunaSharp with minimal changes to existing code

Suppose you already have a training script that ends with a validation score. Rather than wrapping it
in a `Func<Trial, double>`, keep your own loop and only *ask* for parameters and *tell* the result:

```csharp
using OptunaSharp;

var study = Optuna.CreateStudy(direction: StudyDirection.Maximize);

const int nTrials = 10;
for (int i = 0; i < nTrials; i++)
{
    Trial trial = study.Ask();                       // a live Trial (not a FrozenTrial)

    double c      = trial.SuggestFloat("C", 1e-7, 10.0, log: true);
    string solver = trial.SuggestCategorical("solver", new[] { "lbfgs", "saga" });

    double valAccuracy = TrainAndScore(c, solver);   // your existing code, unchanged

    study.Tell(trial, valAccuracy);                  // finish the trial with its objective value
}
```

`Ask` creates a trial that can sample; `Tell` finishes it. Your training code never has to move into
an objective function.

To **prune** in an ask/tell loop, report intermediate values and pass an explicit state to `Tell`:

```csharp
Trial trial = study.Ask();
double alpha = trial.SuggestFloat("alpha", 0.0, 1.0);

bool pruned = false;
for (int step = 0; step < nIter; step++)
{
    double intermediate = TrainOneStep(alpha, step);
    trial.Report(intermediate, step);
    if (trial.ShouldPrune()) { pruned = true; break; }
}

if (pruned)
    study.Tell(trial, state: TrialState.Pruned);     // record the prune…
else
    study.Tell(trial, FinalScore(alpha));            // …or the final value
```

> `Tell` also accepts a trial *number* instead of the trial object: `study.Tell(trial.Number, y)` is
> equivalent to `study.Tell(trial, y)`.

## 2. Define-and-run

Ask/tell supports both *define-by-run* (above) and *define-and-run*: declare the distributions up
front and pass them to `Ask`, which pre-samples every one. The returned trial already holds the
values in `Params`.

```csharp
using OptunaSharp;
using OptunaSharp.Collections;
using OptunaSharp.Distributions;

var distributions = new OrderedDict<string, BaseDistribution>
{
    ["C"]      = new FloatDistribution(1e-7, 10.0, log: true),
    ["solver"] = new CategoricalDistribution(new CategoricalChoice[] { "lbfgs", "saga" }),
};

var study = Optuna.CreateStudy(direction: StudyDirection.Maximize);
for (int i = 0; i < 10; i++)
{
    Trial trial = study.Ask(distributions);          // both params already sampled

    double c      = (double)trial.Params["C"]!;
    object? solver = trial.Params["solver"];

    study.Tell(trial, TrainAndScore(c, solver));
}
```

## 3. Batch optimization

Because you decide when to evaluate, you can ask for a whole batch of trials, evaluate them together
(vectorized, or across parallel workers), and tell them back. Store each trial's number so you can
finish it after the batch runs:

```csharp
using OptunaSharp;
using OptunaSharp.Samplers;

const int batchSize = 10;
var study = Optuna.CreateStudy(sampler: new CmaEsSampler());

for (int round = 0; round < 3; round++)               // 3 batches x 10 = 30 trials
{
    var numbers = new List<int>();
    var xs = new List<double>();
    var ys = new List<double>();
    for (int i = 0; i < batchSize; i++)
    {
        Trial trial = study.Ask();
        numbers.Add(trial.Number);
        xs.Add(trial.SuggestFloat("x", -10, 10));
        ys.Add(trial.SuggestFloat("y", -10, 10));
    }

    double[] objectives = BatchedObjective(xs, ys);   // evaluate the whole batch at once

    for (int i = 0; i < batchSize; i++)
        study.Tell(numbers[i], objectives[i]);
}
```

> **Tips.** `TPESampler` takes a `constantLiar` option (on by default) that discourages running
> trials from clustering on the same region — useful when many workers ask before any tells. For
> `CmaEsSampler`, setting the population size to a multiple of the batch size lines up the algorithm
> with your batches. See the [Samplers guide](../samplers.md).

## Relationship to Optuna

| Optuna (Python) | OptunaSharp |
|-----------------|-------------|
| `trial = study.ask()` | `Trial trial = study.Ask()` |
| `study.tell(trial, value)` | `study.Tell(trial, value)` |
| `study.tell(trial.number, value)` | `study.Tell(trial.Number, value)` |
| `study.tell(trial, state=TrialState.PRUNED)` | `study.Tell(trial, state: TrialState.Pruned)` |
| `study.ask(distributions)` (define-and-run) | `study.Ask(distributions)` |
| `trial.params["C"]` | `trial.Params["C"]` |

## See also

[Study and Trials](../study-and-trials.md) · [Samplers](../samplers.md) · [Pruners](../pruners.md) ·
[Command-line interface](cli.md) (ask/tell from the shell) · the [API reference](../../api/index.md).
