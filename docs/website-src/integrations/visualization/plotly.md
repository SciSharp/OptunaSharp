# Plotly visualization

`OptunaSharp.Visualization.Plotly` renders interactive charts from studies using Plotly.NET. It is the implemented rendering companion to Core's [visualization data layer](../../docs/visualization.md): Core computes public info records, and this package turns them into `Plotly.NET.GenericChart` figures. Normal use needs no Python runtime. The package targets .NET 8 and .NET 10 and references `OptunaSharp`, `Plotly.NET`, and `Plotly.NET.CSharp`. [Package definition](https://github.com/SciSharp/OptunaSharp/blob/main/src/OptunaSharp.Visualization.Plotly/OptunaSharp.Visualization.Plotly.csproj).

```shell
dotnet add package OptunaSharp.Visualization.Plotly
```

## Create and export a chart

```csharp
using System.IO;
using OptunaSharp;
using OptunaSharp.Samplers;
using OptunaSharp.Visualization.Plotly;
using Plotly.NET;
using Plotly.NET.CSharp;

var study = Optuna.CreateStudy(sampler: new RandomSampler(0));
study.Optimize(trial =>
{
    double x = trial.SuggestFloat("x", -10, 10);
    double y = trial.SuggestFloat("y", -10, 10);
    return (x - 2) * (x - 2) + (y + 3) * (y + 3);
}, nTrials: 50);

GenericChart chart = PlotlyVisualization.PlotOptimizationHistory(study);
chart.SaveHtml("history.html");
File.WriteAllText("history.json", GenericChart.toJson(chart));
// Optional for an interactive local session:
chart.Show();
```

`SaveHtml` and `Show` are Plotly.NET.CSharp extensions; `Show` opens a browser. `GenericChart.toJson` exports the figure representation, useful for inspection or integration with a Plotly frontend. Generating the chart or writing its JSON does not launch a browser. `IsAvailable` is always `true` because Plotly.NET is a required package dependency; it is not a browser or display-environment probe. [Public facade](https://github.com/SciSharp/OptunaSharp/blob/main/src/OptunaSharp.Visualization.Plotly/PlotlyVisualization.cs), [JSON inspection tests](https://github.com/SciSharp/OptunaSharp/blob/main/test/OptunaSharp.Tests/Visualization/Plotly/PlotlyChartCustomizationTests.cs).

HTML output still needs a working plotly.js loading configuration in the browser. This package exposes Plotly.NET's chart and HTML configuration rather than implementing its own hosting or image-export service. It does not add a server-side PNG, SVG, or PDF export API or image-renderer dependency. The `staticPlot` configuration flag disables chart interaction; it does not convert the figure to a bitmap. [Package dependencies](https://github.com/SciSharp/OptunaSharp/blob/main/src/OptunaSharp.Visualization.Plotly/OptunaSharp.Visualization.Plotly.csproj), [configuration surface](https://github.com/SciSharp/OptunaSharp/blob/main/src/OptunaSharp.Visualization.Plotly/PlotlyChartCustomization.cs).

## All supported charts

The public facade has **12 chart families**, plus `IsAvailable`. Each family has a customization overload. History and EDF also accept `IReadOnlyList<Study>` to compare studies. [Complete public API](https://github.com/SciSharp/OptunaSharp/blob/main/src/OptunaSharp.Visualization.Plotly/PlotlyVisualization.cs).

| Method | What it shows | Inputs and behavior to know |
|---|---|---|
| `PlotOptimizationHistory` | Trial values and running best | One study or several; `errorBar: true` aggregates histories. An explicit `target` removes the default best-so-far series. |
| `PlotEdf` | Empirical distribution of objective values | One study or several; accepts a scalar `target` and `targetName`. |
| `PlotIntermediateValues` | Values reported at intermediate steps | Uses running, pruned, and completed trials with `Trial.Report` values; steps are sorted. |
| `PlotHypervolumeHistory` | Accumulated dominated hypervolume | Requires a multi-objective study and a reference point with one value per objective; accounts for minimize/maximize directions and feasibility. |
| `PlotTimeline` | Trial durations, colored by state | Optional `nRecentTrials` must be positive. |
| `PlotParamImportances` | Horizontal importance bars | Accepts an evaluator and parameter selection; produces one group per objective when no scalar target is supplied. The current Core default is PED-ANOVA. |
| `PlotSlice` | Target value against each parameter | One subplot per selected parameter; markers colored by trial number. |
| `PlotContour` | Target contours across parameter pairs | Select two parameters for one plot, or more for a matrix; accepts a scalar target. |
| `PlotRank` | Parameter pairs colored by target rank | Two parameters or a matrix; rank colors use the `RdYlBu_r` scale. |
| `PlotParallelCoordinate` | Objective and parameter axes joined by trial lines | Supports conditional/missing parameters; multi-objective studies without a target show objective axes and use Pareto-rank coloring. |
| `PlotParetoFront` | Feasible Pareto-optimal and dominated trials | Two or three plotted targets; `includeDominatedTrials` defaults to `true`; higher-dimensional studies need a target projection. |
| `PlotTerminatorImprovement` | Estimated improvement potential and optional error | Single-objective only; optional evaluators, `plotError`, and rendering threshold `minNTrials` (default 20). |

Parameter-importance charts call Core's `ImportanceFunctions.GetParamImportances`, which defaults to `PedAnovaImportanceEvaluator`. Select fANOVA explicitly when that is the estimator you want:

```csharp
using OptunaSharp.Importance;

PlotlyVisualization.PlotParamImportances(
    study, evaluator: new FanovaImportanceEvaluator(seed: 0))
    .SaveHtml("importances.html");
```

The estimator determines which parameters and trials contribute; rendering does not train a different estimator behind that API. [Importance default](https://github.com/SciSharp/OptunaSharp/blob/main/src/OptunaSharp.Core/Importance/ImportanceFunctions.cs), [plot data construction](https://github.com/SciSharp/OptunaSharp/blob/main/src/OptunaSharp.Core/Visualization/ParamImportances.cs).

## Targets, constraints, and conditional parameters

For scalar charts such as history, EDF, slice, contour, and rank, select an objective when the study is multi-objective:

```csharp
// multiObjectiveStudy is an existing study with at least two objectives.
PlotlyVisualization.PlotSlice(
    multiObjectiveStudy,
    @params: new[] { "learning_rate", "depth" },
    target: trial => trial.Values![0],
    targetName: "Validation loss")
    .SaveHtml("loss-slice.html");
```

Without that target, the scalar data builders reject a multi-objective study. Parameter selections must name parameters present in the study's relevant completed trials. Missing or nonfinite values are handled by each data builder, so a chart is not a promise to display every stored trial. Parallel-coordinate plots have a dedicated multi-objective path and place missing parameter values on a labeled `None` position. Their renderer uses ordinary scatter-line traces and axis annotations, rather than Plotly's `parcoords` trace. Infeasible lines are dotted. [Argument validation](https://github.com/SciSharp/OptunaSharp/blob/main/src/OptunaSharp.Core/Visualization/VisualizationUtils.cs), [parallel-coordinate data](https://github.com/SciSharp/OptunaSharp/blob/main/src/OptunaSharp.Core/Visualization/ParallelCoordinate.cs), [renderer](https://github.com/SciSharp/OptunaSharp/blob/main/src/OptunaSharp.Visualization.Plotly/Renderers/ParallelCoordinatePlot.cs).

`PlotParetoFront` accepts `targets: trial => new[] { trial.Values![0], trial.Values![2] }` to project a larger study to two dimensions; give matching `targetNames`. Pareto membership is computed from the study's full objective vectors before projection. The projection must yield two or three plotted values. Empty studies with a custom projection also require `targetNames`. Stored trial constraints are used for feasibility; the optional `constraintsFunc` parameter follows a deprecated upstream path and emits a warning. [Pareto data construction and validation](https://github.com/SciSharp/OptunaSharp/blob/main/src/OptunaSharp.Core/Visualization/ParetoFront.cs).

For hypervolume, pass the reference point in the study's original objective units and order. The builder transforms maximizing dimensions internally; points outside the reference region do not increase the plotted hypervolume. [Hypervolume implementation](https://github.com/SciSharp/OptunaSharp/blob/main/src/OptunaSharp.Core/Visualization/HypervolumeHistory.cs).

## Terminator improvement has evaluation costs

`PlotTerminatorImprovement` runs its improvement evaluator for trial prefixes; it is not simply plotting values already stored on every trial. The default evaluator is `RegretBoundEvaluator`. With `plotError: true`, the default error evaluator expects cross-validation scores reported during objectives through `TerminatorFunctions.ReportCrossValidationScores(trial, scores)`. An absent score record raises `ValueError`. A `BestValueStagnationEvaluator` instead gets a zero-valued default error evaluator. Supplying explicit improvement/error evaluators changes this calculation. [Improvement data builder](https://github.com/SciSharp/OptunaSharp/blob/main/src/OptunaSharp.Core/Visualization/TerminatorImprovement.cs), [cross-validation error evaluator](https://github.com/SciSharp/OptunaSharp/blob/main/src/OptunaSharp.Core/Terminator/ErrorEval.cs).

`minNTrials` controls the faded initial chart segment and y-axis window; it is passed to the renderer, not to the data evaluator. It does not skip early-prefix computation or configure a study's stopping behavior. [Terminator renderer](https://github.com/SciSharp/OptunaSharp/blob/main/src/OptunaSharp.Visualization.Plotly/Renderers/TerminatorImprovementPlot.cs).

## Customize the figure and HTML

Use `Plotly.NET` for native fluent methods such as `WithConfig`, `WithLayout`, `WithSize`, and `WithTitle`. OptunaSharp adds `WithDisplayOptions` and `WithCustomization`, plus overloads accepting the same customization bundle:

```csharp
var config = new Config();
config.SetValue("staticPlot", true);

var layout = new Layout();
layout.SetValue("width", 900);
layout.SetValue("height", 500);
layout.SetValue("title", "Search progress");

var customization = new PlotlyChartCustomization
{
    Config = config,
    Layout = layout,
    DisplayOptions = new DisplayOptions()
};

PlotlyVisualization.PlotOptimizationHistory(study, customization)
    .SaveHtml("custom-history.html");
```

The bundle applies layout first, then configuration, then HTML display options. Layout merges with the generated layout: unmodified axes/title keys survive, and caller values win on conflicts. A null bundle member is skipped. `DisplayOptions` is attached to the HTML representation and is not included in figure JSON. These customizations leave the data traces unchanged. Plotly.NET's layout merge can mutate a chart in place; render separate charts or capture the canonical JSON before customizing when comparing variants. [Extensions](https://github.com/SciSharp/OptunaSharp/blob/main/src/OptunaSharp.Visualization.Plotly/PlotlyChartExtensions.cs), [customization and mutation tests](https://github.com/SciSharp/OptunaSharp/blob/main/test/OptunaSharp.Tests/Visualization/Plotly/PlotlyChartCustomizationAllPlotsTests.cs).

## Data-only use and verification

When another chart library is a better fit, call Core builders such as `OptimizationHistory.GetInfoList(study)` or `ParetoFront.GetInfo(study)` directly. Their public records carry calculated series, axes, feasibility, and labels without a Plotly dependency. The Plotly facade currently accepts studies and calls those builders itself; its individual renderer classes are internal. [Visualization data guide](../../docs/visualization.md), [facade implementation](https://github.com/SciSharp/OptunaSharp/blob/main/src/OptunaSharp.Visualization.Plotly/PlotlyVisualization.cs).

The repository tests canonical figure structures and customization behavior, and has live-Optuna comparison suites for Plotly data and meaningful layout keys. Those comparisons normalize Python Plotly templates and typed-array encodings; some paths use numerical tolerances or omit nondeterministic timestamps. This is evidence about figure specifications, not a guarantee of identical pixels across browsers. Parallel-coordinate line order can differ on tied ranks, and GP-based improvement calculations have their documented numerical tolerance. [Render comparison harness](https://github.com/SciSharp/OptunaSharp/blob/main/test/OptunaSharp.Tests.Interop/PlotlyRenderParityBase.cs), [customization tests](https://github.com/SciSharp/OptunaSharp/blob/main/test/OptunaSharp.Tests/Visualization/Plotly/PlotlyChartCustomizationAllPlotsTests.cs).
