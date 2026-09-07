using System;
using System.Collections.Generic;
using System.Linq;
using OptunaSharp.Internal;
using OptunaSharp.Visualization.Plotly.Internal;
using Plotly.NET;
using DObj = DynamicObj.DynamicObj;

namespace OptunaSharp.Visualization.Plotly.Renderers;

/// <summary>
///     Renders <c>optuna.visualization.plot_rank</c> — the plotly half of <c>_rank.py</c>. A scatter of
///     the (x-param, y-param) values colored by target rank on the <c>RdYlBu_r</c> scale (grey for
///     infeasible), a single plot for two params or the N×N <see cref="SubplotGrid" /> (spacing
///     <c>0.08/P</c>), plus the trailing hidden colorbar trace with quantile tick labels.
/// </summary>
internal static class RankPlot
{
    public static GenericChart Render(RankPlotInfo info)
    {
        var layout = Fig.Layout(("title", $"Rank ({info.TargetName})"));
        int p = info.Params.Count;
        if (p == 0)
            return Fig.Figure(Array.Empty<Trace>(), layout);

        var traces = new List<Trace>();

        if (p == 2)
        {
            var sub = info.SubPlotInfos[0][0];
            traces.Add(RankScatter(sub, info.TargetName, info.HasCustomTarget));

            var xaxis = new DObj();
            var yaxis = new DObj();
            layout.SetValue("xaxis", xaxis);
            layout.SetValue("yaxis", yaxis);
            SetRangeType(xaxis, sub.XAxis);
            xaxis.SetValue("title", info.Params[0]);
            SetRangeType(yaxis, sub.YAxis);
            yaxis.SetValue("title", info.Params[1]);
        }
        else
        {
            var grid = SubplotGrid.Init(layout, p, p, 0.08 / p, 0.08 / p);
            for (int xi = 0; xi < p; xi++)
            {
                for (int yi = 0; yi < p; yi++)
                {
                    int n = grid.Cell(yi + 1, xi + 1);
                    var sub = info.SubPlotInfos[yi][xi];
                    var scatter = RankScatter(sub, info.TargetName, info.HasCustomTarget);
                    scatter.SetValue("xaxis", SubplotGrid.XAnchor(n));
                    scatter.SetValue("yaxis", SubplotGrid.YAnchor(n));
                    traces.Add(scatter);

                    SetRangeType(grid.XAxis(n), sub.XAxis);
                    SetRangeType(grid.YAxis(n), sub.YAxis);
                    if (xi == 0)
                        grid.YAxis(n).SetValue("title", info.Params[yi]);
                    if (yi == p - 1)
                        grid.XAxis(n).SetValue("title", info.Params[xi]);
                }
            }
        }

        traces.Add(ColorbarTrace(info));
        return Fig.Figure(traces, layout);
    }

    private static Trace RankScatter(RankSubplotInfo sub, string targetName, bool hasCustomTarget)
    {
        object?[] xs = sub.XAxis.IsCat
            ? sub.Xs.Select(x => (object?)PythonRepr.Str(x)).ToArray()
            : sub.Xs.ToArray();
        object?[] ys = sub.YAxis.IsCat
            ? sub.Ys.Select(y => (object?)PythonRepr.Str(y)).ToArray()
            : sub.Ys.ToArray();

        var colors = new string[sub.ColorIdxs.Length];
        for (int j = 0; j < colors.Length; j++)
            colors[j] = sub.Infeasible[j] ? PlotlyColors.InfeasibleRgb : PlotlyColors.SampleRgb(sub.ColorIdxs[j]);

        var hovertext = new string[sub.Trials.Count];
        for (int j = 0; j < hovertext.Length; j++)
            hovertext[j] = HoverText(sub.Trials[j], sub.Zs[j], targetName, hasCustomTarget);

        return Fig.Trace("scatter",
            ("x", xs),
            ("y", ys),
            ("marker", Fig.Obj(
                ("color", colors),
                ("line", Fig.Obj(("width", 0.5), ("color", "Grey"))))),
            ("mode", "markers"),
            ("showlegend", false),
            ("hovertemplate", "%{hovertext}<extra></extra>"),
            ("hovertext", hovertext));
    }

    private static string HoverText(FrozenTrial trial, double targetValue, string targetName, bool hasCustomTarget)
    {
        var lines = new List<string> { $"Trial #{trial.Number}" };
        foreach (var kv in trial.Params)
            lines.Add($"{kv.Key}: {PythonRepr.Str(kv.Value)}");
        lines.Add($"<b>{targetName}: {PythonRepr.FloatRepr(targetValue)}</b>");
        if (hasCustomTarget && trial.Values is { } values)
            for (int i = 0; i < values.Count; i++)
                lines.Add($"Objective #{i}: {PythonRepr.FloatRepr(values[i])}");
        return string.Join("<br>", lines);
    }

    private static Trace ColorbarTrace(RankPlotInfo info)
    {
        var sorted = (double[])info.Zs.Clone();
        Array.Sort(sorted);
        double[] coloridxs = { 0, 0.25, 0.5, 0.75, 1 };
        string[] rankText = { "min.", "25%", "50%", "75%", "max." };
        var ticktext = new string[coloridxs.Length];
        for (int i = 0; i < coloridxs.Length; i++)
            ticktext[i] = $"{rankText[i]} ({PlotlyColors.FormatG(PlotlyColors.QuantileSorted(sorted, coloridxs[i]))})";

        return Fig.Trace("scatter",
            ("x", new object?[] { null }),
            ("y", new object?[] { null }),
            ("mode", "markers"),
            ("marker", Fig.Obj(
                ("colorscale", PlotlyColors.RdYlBuRScale()),
                ("showscale", true),
                ("cmin", 0),
                ("cmax", 1),
                ("colorbar", Fig.Obj(
                    ("thickness", 10),
                    ("tickvals", coloridxs),
                    ("ticktext", ticktext))))),
            ("hoverinfo", "none"),
            ("showlegend", false));
    }

    private static void SetRangeType(DObj axis, RankAxisInfo ax)
    {
        axis.SetValue("range", new[] { ax.Range.Min, ax.Range.Max });
        if (ax.IsCat)
            axis.SetValue("type", "category");
        if (ax.IsLog)
        {
            axis.SetValue("range", new[] { Math.Log10(ax.Range.Min), Math.Log10(ax.Range.Max) });
            axis.SetValue("type", "log");
        }
    }
}
