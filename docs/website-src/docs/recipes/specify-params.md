# Specify hyperparameters manually

You usually have a few configurations worth trying first — sensible defaults, or settings that worked
on a similar problem. OptunaSharp gives you two ways to feed them in:

1. **`Study.EnqueueTrial`** — hand OptunaSharp the parameter sets and let it *evaluate* them (before
   it starts sampling on its own).
2. **`Study.AddTrial`** — register sets you have *already evaluated*, so the sampler takes their
   results into account without re-running them.

Both use the parameter names below with a placeholder objective; substitute your real training code.

## First scenario: have OptunaSharp evaluate your hyperparameters

`EnqueueTrial` queues a configuration to be evaluated next. Queued trials run before sampled ones, so
your defaults are tried first:

```csharp
using OptunaSharp;
using OptunaSharp.Collections;
using OptunaSharp.Pruners;

double Objective(Trial trial)
{
    double baggingFraction  = trial.SuggestFloat("bagging_fraction", 0.4, 1.0 + 1e-12);
    long   baggingFreq      = trial.SuggestInt("bagging_freq", 0, 7);
    long   minChildSamples  = trial.SuggestInt("min_child_samples", 5, 100);
    return TrainAndScore(baggingFraction, baggingFreq, minChildSamples);
}

var study = Optuna.CreateStudy(direction: StudyDirection.Maximize, pruner: new MedianPruner());

// Try the defaults, then a variant with a larger bagging frequency, before sampling.
study.EnqueueTrial(new OrderedDict<string, object?>
{
    ["bagging_fraction"] = 1.0, ["bagging_freq"] = 0, ["min_child_samples"] = 20,
});
study.EnqueueTrial(new OrderedDict<string, object?>
{
    ["bagging_fraction"] = 0.75, ["bagging_freq"] = 5, ["min_child_samples"] = 20,
});

study.Optimize(Objective, nTrials: 100, timeout: 600);
```

## Second scenario: reuse already-evaluated hyperparameters

If you have *results* for those sets, register them as completed trials with `AddTrial`. The sampler
then models them as history — no re-evaluation. Build each completed trial with
`FrozenTrial.CreateTrial`, supplying the `@params`, their `distributions`, and the observed `value`:

```csharp
using OptunaSharp;
using OptunaSharp.Collections;
using OptunaSharp.Distributions;
using OptunaSharp.Pruners;

var study = Optuna.CreateStudy(direction: StudyDirection.Maximize, pruner: new MedianPruner());

var distributions = new OrderedDict<string, BaseDistribution>
{
    ["bagging_fraction"]  = new FloatDistribution(0.4, 1.0 + 1e-12),
    ["bagging_freq"]      = new IntDistribution(0, 7),
    ["min_child_samples"] = new IntDistribution(5, 100),
};

study.AddTrial(FrozenTrial.CreateTrial(
    value: 0.94,
    @params: new OrderedDict<string, object?>
    {
        ["bagging_fraction"] = 1.0, ["bagging_freq"] = 0L, ["min_child_samples"] = 20L,
    },
    distributions: distributions));

study.AddTrial(FrozenTrial.CreateTrial(
    value: 0.95,
    @params: new OrderedDict<string, object?>
    {
        ["bagging_fraction"] = 0.75, ["bagging_freq"] = 5L, ["min_child_samples"] = 20L,
    },
    distributions: distributions));

study.Optimize(Objective, nTrials: 100, timeout: 600);
```

`CreateTrial` validates that every parameter lies inside its declared distribution, so a typo in a
bound fails fast rather than corrupting the study.

## Relationship to Optuna

| Optuna (Python) | OptunaSharp |
|-----------------|-------------|
| `study.enqueue_trial({"bagging_freq": 5, …})` | `study.EnqueueTrial(new OrderedDict<string, object?> { ["bagging_freq"] = 5, … })` |
| `optuna.trial.create_trial(params=…, distributions=…, value=0.94)` | `FrozenTrial.CreateTrial(@params: …, distributions: …, value: 0.94)` |
| `optuna.distributions.FloatDistribution(0.4, 1.0 + 1e-12)` | `new FloatDistribution(0.4, 1.0 + 1e-12)` |
| `optuna.distributions.IntDistribution(0, 7)` | `new IntDistribution(0, 7)` |
| `study.add_trial(trial)` | `study.AddTrial(trial)` |

## See also

[Study and Trials](../study-and-trials.md) · [Distributions](../distributions.md) ·
[Re-use the best trial](reuse-best-trial.md) · the [API reference](../../api/index.md).
