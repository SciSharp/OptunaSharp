# Quick visualization

OptunaSharp ports Optuna's visualization suite so you can analyze a study visually — the optimization
history, learning curves, parameter relationships, and importances. Install
**`OptunaSharp.Visualization.Plotly`** for ready-made interactive charts (`PlotlyVisualization`), or
consume the Core [data-layer builders](../visualization.md) and render however you like.

## A study to visualize

```csharp
using OptunaSharp;
using OptunaSharp.Samplers;
using OptunaSharp.Pruners;
using OptunaSharp.Visualization.Plotly;   // OptunaSharp.Visualization.Plotly package

const int seed = 13;
var study = Optuna.CreateStudy(
    direction: StudyDirection.Maximize,
    sampler: new TPESampler(seed: seed),
    pruner: new MedianPruner());
study.Optimize(Objective, nTrials: 30, timeout: 300);
```

## Plot functions

Each returns a Plotly.NET `GenericChart` you can display or export:

```csharp
// Objective value and best-so-far over trials.
var history   = PlotlyVisualization.PlotOptimizationHistory(study);

// The reported intermediate values (learning curves) per trial.
var learning  = PlotlyVisualization.PlotIntermediateValues(study);

// All parameters and the objective as parallel axes…
var parallel  = PlotlyVisualization.PlotParallelCoordinate(study);
// …or a chosen subset.
var parallel2 = PlotlyVisualization.PlotParallelCoordinate(study, params: new[] { "lr", "n_layers" });

// Objective over parameter pairs.
var contour   = PlotlyVisualization.PlotContour(study, params: new[] { "lr", "n_layers" });

// Objective vs. each parameter.
var slice     = PlotlyVisualization.PlotSlice(study, params: new[] { "lr", "n_layers" });

// Parameter importances as bars.
var importance = PlotlyVisualization.PlotParamImportances(study);

// Importance with respect to a custom target — e.g. which params drive trial duration.
var durImp    = PlotlyVisualization.PlotParamImportances(
    study, target: t => t.Duration!.Value.TotalSeconds, targetName: "duration");

// Empirical distribution of the objective values.
var edf       = PlotlyVisualization.PlotEdf(study);

// Trials ranked over parameter pairs, colored by objective.
var rank      = PlotlyVisualization.PlotRank(study);

// Start/finish timeline of the trials.
var timeline  = PlotlyVisualization.PlotTimeline(study);
```

The [Visualization guide](../visualization.md) lists every builder and the info record it returns; the
[Parameter Importance guide](../importance.md) covers the importance evaluators behind
`PlotParamImportances`.

## Customizing a figure

Each plot returns an editable Plotly.NET chart, so you can adjust titles, axes, and layout with the
Plotly.NET API — or pass a `PlotlyChartCustomization` to the customization overload:

```csharp
var fig = PlotlyVisualization.PlotIntermediateValues(study)
    .WithTitle("Hyperparameter optimization for FashionMNIST classification");
```

## Related tools

- **Multi-objective**: the Pareto-front plot is covered in the
  [multi-objective recipe](../recipes/multi-objective.md) (`PlotParetoFront`).
- **Optuna Dashboard**: persist the study to an [RDB backend](../recipes/rdb.md) and the Python
  `optuna-dashboard` tool renders the same history, importances, and relationships in a live web UI —
  it reads OptunaSharp's SQLite files directly.

## Relationship to Optuna

| Optuna (Python) | OptunaSharp |
|-----------------|-------------|
| `optuna.visualization.plot_optimization_history(study)` | `PlotlyVisualization.PlotOptimizationHistory(study)` |
| `plot_contour(study, params=["lr", "n_layers"])` | `PlotlyVisualization.PlotContour(study, params: new[] { "lr", "n_layers" })` |
| `plot_param_importances(study, target=lambda t: t.duration.total_seconds(), target_name="duration")` | `PlotlyVisualization.PlotParamImportances(study, target: t => t.Duration!.Value.TotalSeconds, targetName: "duration")` |
| `plot_slice` / `plot_edf` / `plot_rank` / `plot_timeline` | `PlotSlice` / `PlotEdf` / `PlotRank` / `PlotTimeline` |

## See also

[Visualization guide](../visualization.md) · [Parameter Importance](../importance.md) ·
[Plotly.NET integration](../../integrations/visualization/plotly.md) ·
[Multi-objective recipe](../recipes/multi-objective.md) · the [API reference](../../api/index.md).
