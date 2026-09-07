# OptunaSharp.Visualization.Plotly

The **plotly visualization backend** for [OptunaSharp](https://github.com/SciSharp/OptunaSharp) — a
raw-C# port of `optuna.visualization` (the 13 `plot_*` functions) built on
[Plotly.NET](https://plotly.net/), over OptunaSharp's Phase 3.1 visualization **data** layer.

Figures are plotly.js figures whose spec matches live Optuna's `figure.to_dict()`, so what you see is
what Optuna draws. **Pure managed** — Plotly.NET pulls only managed dependencies (no native binaries).

```csharp
using OptunaSharp;
using OptunaSharp.Visualization.Plotly;
using Plotly.NET.CSharp;   // .Show() / .SaveHtml()

var study = Optuna.CreateStudy();
study.Optimize(t => { var x = t.SuggestFloat("x", -10, 10); return (x - 2) * (x - 2); }, nTrials: 50);

PlotlyVisualization.PlotOptimizationHistory(study).SaveHtml("history.html");
PlotlyVisualization.PlotOptimizationHistory(study).Show();   // opens a browser
```

### Customizing (golden-standard Plotly.NET config)

Every `Plot*` returns a Plotly.NET `GenericChart`, so configure it the idiomatic way. Most of the
fluent surface ships in **Plotly.NET core** (`using Plotly.NET;`) — `.WithConfig`, `.WithLayout`,
`.WithSize`, `.WithTitle`, … This package adds the two things it's missing: `.WithDisplayOptions` and
`.WithCustomization` (a one-call bundle), plus a `customization` **overload** on each `Plot*`:

```csharp
using Plotly.NET;                       // .WithConfig / .WithSize / .WithLayout (Plotly.NET core)
using OptunaSharp.Visualization.Plotly; // .WithDisplayOptions / .WithCustomization + the overloads

var config = new Config();  config.SetValue("staticPlot", true);
var layout = new Layout();  layout.SetValue("width", 900);   // merged onto Optuna's title/axes

// fluent:
PlotlyVisualization.PlotOptimizationHistory(study).WithConfig(config).WithSize(900, 500);

// or the one-shot overload:
PlotlyVisualization.PlotOptimizationHistory(
    study, new PlotlyChartCustomization { Config = config, Layout = layout });
```

## Status

**Complete** — all 12 `optuna.visualization` plot functions are ported (each with a golden-standard
Plotly.NET customization overload) and byte-exact vs live Optuna:

`PlotOptimizationHistory` · `PlotEdf` · `PlotIntermediateValues` · `PlotHypervolumeHistory` ·
`PlotTimeline` · `PlotParamImportances` · `PlotSlice` · `PlotTerminatorImprovement` · `PlotContour` ·
`PlotRank` · `PlotParallelCoordinate` · `PlotParetoFront` — plus `IsAvailable`.

Part of the [SciSharp STACK](https://github.com/SciSharp).
