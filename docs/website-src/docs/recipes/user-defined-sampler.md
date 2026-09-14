# User-defined sampler

Writing your own sampler lets you experiment with a new sampling algorithm, add a task-specific
heuristic, or wrap another optimization library behind OptunaSharp's interface. This recipe explains
the sampler contract and implements a small **Simulated Annealing** sampler as an example.

## The sampler interface

A sampler decides the value of each parameter a trial suggests. Subclass
[`BaseSampler`](../samplers.md) and implement its three abstract methods:

| Method | Role |
|--------|------|
| `InferRelativeSearchSpace(study, trial)` | Return the parameters to sample *jointly* (their correlations matter) |
| `SampleRelative(study, trial, searchSpace)` | Sample those joint parameters at once |
| `SampleIndependent(study, trial, name, dist)` | Sample one parameter on its own (anything not in the relative space) |

At the start of a trial, `InferRelativeSearchSpace` runs; `SampleRelative` then samples that space;
and during the objective any parameter outside it falls through to `SampleIndependent`.

> **RNG discipline.** Built-in samplers draw randomness from `LazyRandomState` (a port of NumPy's
> `RandomState`), which is what makes seeded runs bit-identical to Python. Use it in your own sampler
> too — `(double)_rng.Rng.rand()` for a uniform in `[0, 1)`, `_rng.Rng.uniform(low, high)` for a
> range — rather than `System.Random`.

## Example: a Simulated Annealing sampler

```csharp
using System;
using OptunaSharp;
using OptunaSharp.Collections;
using OptunaSharp.Distributions;
using OptunaSharp.Internal;      // LazyRandomState
using OptunaSharp.SearchSpace;   // SearchSpaceFunctions
using OptunaSharp.Samplers;

// A sampler driven by Simulated Annealing over the relative (intersection) search space.
public sealed class SimulatedAnnealingSampler : BaseSampler
{
    private readonly LazyRandomState _rng = new();
    private double _temperature;
    private FrozenTrial? _currentTrial;   // the accepted SA state

    public SimulatedAnnealingSampler(double temperature = 100) => _temperature = temperature;

    public override OrderedDict<string, object?> SampleRelative(
        Study study, FrozenTrial trial, OrderedDict<string, BaseDistribution> searchSpace)
    {
        if (searchSpace.Count == 0)
            return new OrderedDict<string, object?>();

        // 1. Transition probability, using the trial evaluated just before this one.
        var trials = study.GetTrials(deepcopy: false);
        FrozenTrial prevTrial = trials[^2];
        double probability = 1.0;
        if (_currentTrial is not null && prevTrial.Value > _currentTrial.Value)
            probability = Math.Exp((_currentTrial.Value!.Value - prevTrial.Value!.Value) / _temperature);
        _temperature *= 0.9;   // cool down each step

        // 2. Accept the previous point as the new current state with that probability.
        if ((double)_rng.Rng.rand() < probability)
            _currentTrial = prevTrial;

        // 3. Sample each parameter from a neighborhood around the current state.
        var sampled = new OrderedDict<string, object?>();
        foreach (var (name, dist) in searchSpace)
        {
            // This toy sampler only supports plain SuggestFloat (no step > 1, no log).
            if (dist is not FloatDistribution fd || (fd.Step is not null && fd.Step != 1) || fd.Log)
                throw new NotImplementedException(
                    "Only SuggestFloat() with `step` null or 1.0 and `log` false is supported.");

            double current = (double)_currentTrial!.Params[name]!;
            double width = (fd.High - fd.Low) * 0.1;
            double low  = Math.Max(current - width, fd.Low);
            double high = Math.Min(current + width, fd.High);
            sampled[name] = (double)_rng.Rng.uniform(low, high);
        }
        return sampled;
    }

    // The rest is boilerplate, unrelated to the SA algorithm.
    public override OrderedDict<string, BaseDistribution> InferRelativeSearchSpace(Study study, FrozenTrial trial)
        => SearchSpaceFunctions.IntersectionSearchSpaceOf(study.GetTrials(deepcopy: false));

    public override object? SampleIndependent(
        Study study, FrozenTrial trial, string paramName, BaseDistribution paramDistribution)
        => new RandomSampler().SampleIndependent(study, trial, paramName, paramDistribution);
}
```

Use it exactly like a built-in sampler:

```csharp
double Objective(Trial trial)
{
    double x = trial.SuggestFloat("x", -10, 10);
    double y = trial.SuggestFloat("y", -5, 5);
    return x * x + y;
}

var study = Optuna.CreateStudy(sampler: new SimulatedAnnealingSampler());
study.Optimize(Objective, nTrials: 100);

Console.WriteLine($"Best value:  {study.BestValue}");
Console.WriteLine($"Best params: x={study.BestParams["x"]}, y={study.BestParams["y"]}");
```

In the **first** trial, `SampleIndependent` handles the parameters:
`SearchSpaceFunctions.IntersectionSearchSpaceOf` can't infer a relative space until at least one trial
has completed, so it returns empty and each parameter is sampled independently. From the second trial
on, `SampleRelative` takes over.

## Relationship to Optuna

| Optuna (Python) | OptunaSharp |
|-----------------|-------------|
| `class MySampler(optuna.samplers.BaseSampler)` | `class MySampler : BaseSampler` |
| `sample_relative`, `sample_independent`, `infer_relative_search_space` | `SampleRelative`, `SampleIndependent`, `InferRelativeSearchSpace` |
| `optuna.search_space.intersection_search_space(study.get_trials(deepcopy=False))` | `SearchSpaceFunctions.IntersectionSearchSpaceOf(study.GetTrials(deepcopy: false))` |
| `self._rng.uniform(low, high)` (`np.random.RandomState`) | `_rng.Rng.uniform(low, high)` (`LazyRandomState`) |
| `raise NotImplementedError(...)` | `throw new NotImplementedException(...)` |

## See also

[Samplers](../samplers.md) · [Distributions](../distributions.md) · [Search Spaces](../search-spaces.md) ·
[User-defined pruner](user-defined-pruner.md) · the [API reference](../../api/index.md).
