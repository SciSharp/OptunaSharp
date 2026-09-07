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

## Status

Incremental — renderers land per `docs/VISUALIZATION_PLOTLY_PLAN.md`. Currently available:

- `PlotOptimizationHistory`

Part of the [SciSharp STACK](https://github.com/SciSharp).
