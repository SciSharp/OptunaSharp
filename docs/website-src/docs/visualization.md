# Visualization

OptunaSharp ports Optuna's visualization suite as a **data layer**: each plot is a builder that
computes the exact numbers Optuna would hand to Plotly — axes, series, colors, hover text — returned as
a strongly-typed *info record*. You can render with the implemented
[Plotly.NET integration](../integrations/visualization/plotly.md), or consume these records in your
own charting library. This split keeps Core free of a plotting dependency while still guaranteeing
the figures are computed identically to Optuna. Everything lives in **`OptunaSharp.Visualization`**.

The builders below return info records (the `_get_*_info` analog). Install
`OptunaSharp.Visualization.Plotly` for ready-made interactive figures; its
[integration guide](../integrations/visualization/plotly.md) covers chart methods and HTML output.

## Using a builder

```csharp
using OptunaSharp;
using OptunaSharp.Visualization;

var study = Optuna.CreateStudy();
study.Optimize(t => { double x = t.SuggestFloat("x", -10, 10); return (x - 2) * (x - 2); }, nTrials: 100);

List<OptimizationHistoryInfo> history = OptimizationHistory.GetInfoList(study);
foreach (OptimizationHistoryInfo info in history)
    // info.ValuesInfo carries the raw and best-so-far series, error bars, feasibility, etc.
    Render(info);
```

## The builders

Each builder is a `static` class; its entry method returns a `public sealed record` info type.

| Builder | Entry method | Returns | Shows |
|---------|--------------|---------|-------|
| `OptimizationHistory` | `GetInfoList(study, …)` | `OptimizationHistoryInfo` (list) | Objective value and best-so-far over trials |
| `IntermediateValues` | `GetPlotInfo(study)` | `IntermediatePlotInfo` | Reported intermediate values per trial |
| `Edf` | `GetInfo(study, …)` | `EdfInfo` | Empirical distribution of objective values |
| `SlicePlot` | `GetPlotInfo(study, …)` | `SlicePlotInfo` | Objective vs. each parameter |
| `Contour` | `GetInfo(study, …)` | `ContourInfo` | Objective over parameter pairs |
| `Rank` | `GetInfo(study, …)` | `RankPlotInfo` | Trials ranked over parameter pairs |
| `ParallelCoordinate` | `GetInfo(study, …)` | `ParallelCoordinateInfo` | All params + objective as parallel axes |
| `ParamImportances` | `GetInfo` / `GetInfos(study, …)` | `ImportancesInfo` | Parameter importances as bars |
| `ParetoFront` | `GetInfo(study, …)` | `ParetoFrontInfo` | The Pareto front (multi-objective) |
| `HypervolumeHistory` | `GetInfo(study, referencePoint)` | `HypervolumeHistoryInfo` | Dominated hypervolume over trials |
| `Timeline` | `GetInfo(study, nRecentTrials?)` | `TimelineInfo` | Trial start/finish timeline |
| `TerminatorImprovement` | `GetInfo(study, …)` | `ImprovementInfo` | Improvement vs. error over trials |

Most builders accept optional `@params` (which parameters to include), `target`
(`Func<FrozenTrial, double>` — pick an objective in a multi-objective study), and `targetName` (the
axis label). `HypervolumeHistory` requires a `referencePoint`; `SlicePlot` and `IntermediateValues`
use `GetPlotInfo` rather than `GetInfo`.

## Building your own chart

The info records expose everything a renderer needs. For example, `OptimizationHistoryInfo` carries
the trial numbers, the raw `ValuesInfo` series, the best-so-far series, optional error bars, and a
per-point feasibility/completeness `ValueState` — enough to draw the standard history chart in any
library, or to export as CSV/JSON.

## Relationship to Optuna

| Optuna (Python) | OptunaSharp |
|-----------------|-------------|
| `optuna.visualization._optimization_history._get_optimization_history_info_list(study)` | `OptimizationHistory.GetInfoList(study)` |
| `_get_pareto_front_info(study)` | `ParetoFront.GetInfo(study)` |
| `_get_hypervolume_history_info(study, ref_point)` | `HypervolumeHistory.GetInfo(study, referencePoint)` |
| `plot_optimization_history(study)` (rendering) | `PlotlyVisualization.PlotOptimizationHistory(study)` in the Plotly satellite |

The info builders are **bit-exact** against live Optuna wherever the path avoids the error function
(and within the documented erf tolerance where it doesn't).

**Divergences.** `ParallelCoordinate` may compute the line draw *order* differently on tied ranks
(cosmetic — it only affects which lines paint first); `Rank`'s RGB color values come from Plotly's
colorscale sampling and are deferred to the rendering layer (the data layer exposes the pure color
*indices* and an infeasibility mask instead).

## See also

[Study and Trials](study-and-trials.md) · [Parameter Importance](importance.md) ·
[Multi-objective Optimization](multi-objective.md) · the [API reference](../api/index.md).
