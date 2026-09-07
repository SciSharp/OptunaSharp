using System.Collections.Generic;
using System.Linq;
using OptunaSharp.Visualization.Plotly.Internal;
using Plotly.NET;
using DObj = DynamicObj.DynamicObj;

namespace OptunaSharp.Visualization.Plotly.Renderers;

/// <summary>
///     Renders <c>optuna.visualization.plot_pareto_front</c> — the plotly half of <c>_pareto_front.py</c>.
///     A 2D <c>scatter</c> (or 3D <c>scatter3d</c> with a <c>scene</c>) of the trials' objective values:
///     grey infeasible, Blues-colored dominated/feasible, Reds-colored best trials — each colored by
///     trial number, with the JSON hovertext.
/// </summary>
internal static class ParetoFrontPlot
{
    public static GenericChart Render(ParetoFrontInfo info)
    {
        var traces = new List<Trace>();
        if (!info.HasConstraints)
        {
            traces.Add(Scatter(info, info.NonBestTrialsWithValues, "%{text}<extra>Trial</extra>", dominated: true, infeasible: false));
            traces.Add(Scatter(info, info.BestTrialsWithValues, "%{text}<extra>Best Trial</extra>", dominated: false, infeasible: false));
        }
        else
        {
            traces.Add(Scatter(info, info.InfeasibleTrialsWithValues, "%{text}<extra>Infeasible Trial</extra>", dominated: false, infeasible: true));
            traces.Add(Scatter(info, info.NonBestTrialsWithValues, "%{text}<extra>Feasible Trial</extra>", dominated: true, infeasible: false));
            traces.Add(Scatter(info, info.BestTrialsWithValues, "%{text}<extra>Best Trial</extra>", dominated: false, infeasible: false));
        }

        Layout layout = info.NTargets == 2
            ? Fig.Layout(
                ("title", "Pareto-front Plot"),
                ("xaxis", Fig.Obj(("title", info.TargetNames[info.AxisOrder[0]]))),
                ("yaxis", Fig.Obj(("title", info.TargetNames[info.AxisOrder[1]]))))
            : Fig.Layout(
                ("title", "Pareto-front Plot"),
                // xaxis_title etc. are plotly magic-underscore shortcuts for scene.xaxis.title.text.
                ("scene", Fig.Obj(
                    ("xaxis", Fig.Obj(("title", info.TargetNames[info.AxisOrder[0]]))),
                    ("yaxis", Fig.Obj(("title", info.TargetNames[info.AxisOrder[1]]))),
                    ("zaxis", Fig.Obj(("title", info.TargetNames[info.AxisOrder[2]]))))));

        return Fig.Figure(traces, layout);
    }

    private static Trace Scatter(
        ParetoFrontInfo info,
        IReadOnlyList<(FrozenTrial Trial, IReadOnlyList<double> Values)> trialsWithValues,
        string hovertemplate, bool dominated, bool infeasible)
    {
        var marker = Marker(trialsWithValues.Select(t => t.Trial).ToList(), info.IncludeDominatedTrials, dominated, infeasible);

        var entries = new List<(string, object?)>
        {
            ("x", trialsWithValues.Select(t => t.Values[info.AxisOrder[0]]).ToArray()),
            ("y", trialsWithValues.Select(t => t.Values[info.AxisOrder[1]]).ToArray()),
        };
        if (info.NTargets == 3)
            entries.Add(("z", trialsWithValues.Select(t => t.Values[info.AxisOrder[2]]).ToArray()));
        entries.Add(("text", trialsWithValues.Select(t => VisualizationUtils.MakeHovertext(t.Trial)).ToArray()));
        entries.Add(("mode", "markers"));
        entries.Add(("hovertemplate", hovertemplate));
        entries.Add(("marker", marker));
        entries.Add(("showlegend", false));

        return Fig.Trace(info.NTargets == 2 ? "scatter" : "scatter3d", entries.ToArray());
    }

    private static DObj Marker(List<FrozenTrial> trials, bool includeDominated, bool dominated, bool infeasible)
    {
        if (infeasible)
            return Fig.Obj(("color", "#cccccc"));

        if (dominated)
            return Fig.Obj(
                ("line", Fig.Obj(("width", 0.5), ("color", "Grey"))),
                ("color", trials.Select(t => t.Number).ToArray()),
                ("colorscale", ColorScales.Blues),
                ("colorbar", Fig.Obj(("title", "Trial"))));

        return Fig.Obj(
            ("line", Fig.Obj(("width", 0.5), ("color", "Grey"))),
            ("color", trials.Select(t => t.Number).ToArray()),
            ("colorscale", ColorScales.Reds),
            ("colorbar", Fig.Obj(
                ("title", "Best Trial"),
                ("x", includeDominated ? (object)1.1 : 1),
                ("xpad", 40))));
    }
}
