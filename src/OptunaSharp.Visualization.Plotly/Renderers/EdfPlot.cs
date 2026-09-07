using System;
using System.Collections.Generic;
using OptunaSharp.Visualization.Plotly.Internal;
using Plotly.NET;

namespace OptunaSharp.Visualization.Plotly.Renderers;

/// <summary>
///     Renders <c>optuna.visualization.plot_edf</c> — the plotly half of <c>_edf.py</c>. One
///     <c>lines</c> step-function per study over the shared x-grid; the y-axis is pinned to
///     <c>[0, 1]</c> (Optuna's <c>update_yaxes</c>) only when there is data.
/// </summary>
internal static class EdfPlot
{
    public static GenericChart Render(EdfInfo info, string targetName)
    {
        var yaxis = Fig.Obj(("title", "Cumulative Probability"));
        var layout = Fig.Layout(
            ("title", "Empirical Distribution Function Plot"),
            ("xaxis", Fig.Obj(("title", targetName))),
            ("yaxis", yaxis));

        if (info.Lines.Count == 0)
            return Fig.Figure(Array.Empty<Trace>(), layout);

        var traces = new List<Trace>();
        foreach (var line in info.Lines)
            traces.Add(Fig.Trace("scatter",
                ("x", info.XValues),
                ("y", line.YValues),
                ("name", line.StudyName),
                ("mode", "lines")));

        // figure.update_yaxes(range=[0, 1]) — merged onto the y-axis that already carries the title.
        yaxis.SetValue("range", new double[] { 0.0, 1.0 });
        return Fig.Figure(traces, layout);
    }
}
