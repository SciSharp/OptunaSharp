# Multi-objective optimization

Real problems often trade off competing goals — a model that is both *accurate* and *small*, both
*fast* and *cheap*. A multi-objective study optimizes several objectives at once and returns the set
of best trade-offs (the **Pareto front**) rather than a single winner. This recipe mirrors Optuna's
tutorial, which minimizes a model's size while maximizing its accuracy.

> This is the how-to. For the concepts — dominance, the Pareto front, constrained MO, choosing a
> sampler — see the [Multi-objective Optimization guide](../multi-objective.md).

## Define the objective

A multi-objective objective returns an `IReadOnlyList<double>`, one value per objective, in the order
of the `directions` you declare:

```csharp
using OptunaSharp;

IReadOnlyList<double> Objective(Trial trial)
{
    long  nLayers = trial.SuggestInt("n_layers", 1, 3);
    double lr     = trial.SuggestFloat("lr", 1e-5, 1e-1, log: true);

    var model = BuildAndTrain(nLayers, lr);   // your model here
    double size     = model.ParameterCount;   // minimize this…
    double accuracy = model.ValidationAccuracy; // …and maximize this
    return new[] { size, accuracy };
}
```

## Run it

Pass `directions` (plural) instead of `direction`. Here we minimize size and maximize accuracy:

```csharp
var study = Optuna.CreateStudy(
    directions: new[] { StudyDirection.Minimize, StudyDirection.Maximize });

study.Optimize(Objective, nTrials: 30, timeout: 300);

Console.WriteLine($"Number of finished trials: {study.Trials.Count}");
```

The default sampler for a multi-objective study is `NSGAIISampler`. For sample-efficient runs
(`nTrials` below ~1000) the [Samplers guide](../samplers.md) recommends `TPESampler` or `GPSampler`,
both of which also support constrained multi-objective optimization.

## Read the Pareto front

`Study.BestTrials` returns the `FrozenTrial`s on the Pareto front — the non-dominated trade-offs. Each
trial's `Values` holds its objective vector, so you can pick the one that suits you (say, the highest
accuracy):

```csharp
using System.Linq;

Console.WriteLine($"Trials on the Pareto front: {study.BestTrials.Count}");

FrozenTrial best = study.BestTrials.MaxBy(t => t.Values![1]);   // Values[1] = accuracy
Console.WriteLine($"  number: {best.Number}");
Console.WriteLine($"  params: {string.Join(", ", best.Params)}");
Console.WriteLine($"  values: {string.Join(", ", best.Values!)}");
```

## Visualize

The [visualization](../visualization.md) layer has a Pareto-front plot. Compute the data with the Core
builder, or render it directly with the [Plotly.NET satellite](../../integrations/visualization/plotly.md):

```csharp
using OptunaSharp.Visualization;                 // Core data layer
ParetoFrontInfo info = ParetoFront.GetInfo(study, targetNames: new[] { "size", "accuracy" });

using OptunaSharp.Visualization.Plotly;          // OptunaSharp.Visualization.Plotly package
var fig = PlotlyVisualization.PlotParetoFront(study, targetNames: new[] { "size", "accuracy" });
```

## Relationship to Optuna

| Optuna (Python) | OptunaSharp |
|-----------------|-------------|
| `optuna.create_study(directions=["minimize", "maximize"])` | `Optuna.CreateStudy(directions: new[] { StudyDirection.Minimize, StudyDirection.Maximize })` |
| objective returns a tuple `(flops, accuracy)` | objective returns `IReadOnlyList<double>` |
| `study.best_trials` | `study.BestTrials` |
| `max(study.best_trials, key=lambda t: t.values[1])` | `study.BestTrials.MaxBy(t => t.Values![1])` |
| `optuna.visualization.plot_pareto_front(study, …)` | `PlotlyVisualization.PlotParetoFront(study, …)` |

## See also

[Multi-objective Optimization guide](../multi-objective.md) · [Samplers](../samplers.md) ·
[Visualization](../visualization.md) · [Re-use the best trial](reuse-best-trial.md) ·
the [API reference](../../api/index.md).
