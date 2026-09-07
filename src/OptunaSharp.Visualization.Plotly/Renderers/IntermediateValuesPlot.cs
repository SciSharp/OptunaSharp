using System;
using System.Collections.Generic;
using System.Linq;
using OptunaSharp.Visualization.Plotly.Internal;
using Plotly.NET;

namespace OptunaSharp.Visualization.Plotly.Renderers;

/// <summary>
///     Renders <c>optuna.visualization.plot_intermediate_values</c> — the plotly half of
///     <c>_intermediate_values.py</c>. One <c>lines+markers</c> trace per trial (grey when infeasible),
///     with the <c>maxdisplayed:10</c> default marker.
/// </summary>
internal static class IntermediateValuesPlot
{
    public static GenericChart Render(IntermediatePlotInfo info)
    {
        var layout = Fig.Layout(
            ("title", "Intermediate Values Plot"),
            ("xaxis", Fig.Obj(("title", "Step"))),
            ("yaxis", Fig.Obj(("title", "Intermediate Value"))),
            ("showlegend", false));

        if (info.TrialInfos.Count == 0)
            return Fig.Figure(Array.Empty<Trace>(), layout);

        var traces = new List<Trace>();
        foreach (var tinfo in info.TrialInfos)
        {
            var marker = tinfo.Feasible
                ? Fig.Obj(("maxdisplayed", 10))
                : Fig.Obj(("maxdisplayed", 10), ("color", "#CCCCCC"));

            traces.Add(Fig.Trace("scatter",
                ("x", tinfo.SortedIntermediateValues.Select(p => p.Step).ToArray()),
                ("y", tinfo.SortedIntermediateValues.Select(p => p.Value).ToArray()),
                ("mode", "lines+markers"),
                ("marker", marker),
                ("name", $"Trial{tinfo.TrialNumber}")));
        }

        return Fig.Figure(traces, layout);
    }
}
