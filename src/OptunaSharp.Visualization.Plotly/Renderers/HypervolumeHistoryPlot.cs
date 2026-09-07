using System.Collections.Generic;
using System.Linq;
using OptunaSharp.Visualization.Plotly.Internal;
using Plotly.NET;

namespace OptunaSharp.Visualization.Plotly.Renderers;

/// <summary>
///     Renders <c>optuna.visualization.plot_hypervolume_history</c> — the plotly half of
///     <c>_hypervolume_history.py</c>. A single <c>lines+markers</c> scatter of the running
///     hypervolume (the MO / reference-point validations live in the Core data layer).
/// </summary>
internal static class HypervolumeHistoryPlot
{
    public static GenericChart Render(HypervolumeHistoryInfo info)
    {
        var layout = Fig.Layout(
            ("title", "Hypervolume History Plot"),
            ("xaxis", Fig.Obj(("title", "Trial"))),
            ("yaxis", Fig.Obj(("title", "Hypervolume"))));

        var trace = Fig.Trace("scatter",
            ("x", info.TrialNumbers.ToArray()),
            ("y", info.Values.ToArray()),
            ("mode", "lines+markers"));

        return Fig.Figure(new[] { trace }, layout);
    }
}
