using System;
using System.Collections.Generic;
using System.Linq;
using OptunaSharp.Logging;
using OptunaSharp.Visualization.Plotly.Internal;
using Plotly.NET;
using DObj = DynamicObj.DynamicObj;

namespace OptunaSharp.Visualization.Plotly.Renderers;

/// <summary>
///     Renders <c>optuna.visualization.plot_contour</c> — the plotly half of <c>_contour.py</c>. Two
///     params → a single contour + feasible/infeasible scatter overlays; N params → the N×N
///     <see cref="SubplotGrid" /> with a per-cell contour + feasible scatter (diagonals get an empty
///     scatter; the infeasible overlay is dropped in the grid, matching Optuna), <c>showscale</c> on the
///     first contour only, and per-axis range/type(log/category)/title.
/// </summary>
internal static class ContourPlot
{
    private static readonly Logger Logger = OptunaLogging.GetLogger("optuna.visualization._contour");

    public static GenericChart Render(ContourInfo info)
    {
        int p = info.SortedParams.Count;
        if (p <= 1)
            return Fig.Figure(Array.Empty<Trace>(), Fig.Layout(("title", "Contour Plot")));
        return p == 2 ? RenderSingle(info) : RenderGrid(info);
    }

    private static GenericChart RenderSingle(ContourInfo info)
    {
        var sub = info.SubPlotInfos[0][0];
        var (contour, feasible, infeasible) = Subplot(sub, info.ReverseScale, info.TargetName);

        var xaxis = new DObj();
        var yaxis = new DObj();
        var layout = Fig.Layout(("title", "Contour Plot"), ("xaxis", xaxis), ("yaxis", yaxis));

        SetRangeType(xaxis, sub.XAxis);
        xaxis.SetValue("title", info.SortedParams[0]);
        SetRangeType(yaxis, sub.YAxis);
        yaxis.SetValue("title", info.SortedParams[1]);

        return Fig.Figure(new[] { contour, feasible, infeasible }, layout);
    }

    private static GenericChart RenderGrid(ContourInfo info)
    {
        int p = info.SortedParams.Count;
        var layout = Fig.Layout(("title", "Contour Plot"));
        var grid = SubplotGrid.Init(layout, p, p);

        var traces = new List<Trace>();
        bool showscale = true;

        for (int xi = 0; xi < p; xi++)
        {
            for (int yi = 0; yi < p; yi++)
            {
                int n = grid.Cell(yi + 1, xi + 1);   // add_trace(row=y_i+1, col=x_i+1)
                var sub = info.SubPlotInfos[yi][xi];

                if (xi == yi)   // x_param == y_param — an empty scatter on the diagonal
                {
                    traces.Add(Fig.Trace("scatter",
                        ("xaxis", SubplotGrid.XAnchor(n)), ("yaxis", SubplotGrid.YAnchor(n))));
                }
                else
                {
                    var (contour, feasible, _) = Subplot(sub, info.ReverseScale, info.TargetName);
                    contour.SetValue("showscale", showscale);
                    if (showscale) showscale = false;
                    Anchor(contour, n);
                    Anchor(feasible, n);
                    traces.Add(contour);
                    traces.Add(feasible);   // the infeasible overlay is not added in the grid.
                }

                SetRangeType(grid.XAxis(n), sub.XAxis);
                SetRangeType(grid.YAxis(n), sub.YAxis);
                if (xi == 0)
                    grid.YAxis(n).SetValue("title", info.SortedParams[yi]);
                if (yi == p - 1)
                    grid.XAxis(n).SetValue("title", info.SortedParams[xi]);
            }
        }

        return Fig.Figure(traces, layout);
    }

    private static void Anchor(Trace trace, int n)
    {
        trace.SetValue("xaxis", SubplotGrid.XAnchor(n));
        trace.SetValue("yaxis", SubplotGrid.YAnchor(n));
    }

    /// <summary>Python <c>update_*axes(range=…)</c> then category / log10 range + <c>type</c>.</summary>
    private static void SetRangeType(DObj axis, ContourAxisInfo ax)
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

    /// <summary>Python <c>_get_contour_subplot</c> — (contour, feasible scatter, infeasible scatter).</summary>
    private static (Trace Contour, Trace Feasible, Trace Infeasible) Subplot(
        SubContourInfo info, bool reverseScale, string targetName)
    {
        var xIndices = info.XAxis.Indices;
        var yIndices = info.YAxis.Indices;

        if (xIndices.Count < 2 || yIndices.Count < 2)
            return (Fig.Trace("contour"), Fig.Trace("scatter"), Fig.Trace("scatter"));
        if (info.ZValues.Count == 0)
        {
            Logger.Warning(
                $"Contour plot will not be displayed because `{info.XAxis.Name}` and " +
                $"`{info.YAxis.Name}` cannot co-exist in `trial.params`.");
            return (Fig.Trace("contour"), Fig.Trace("scatter"), Fig.Trace("scatter"));
        }

        var fx = new List<object>();
        var fy = new List<object>();
        var ix = new List<object>();
        var iy = new List<object>();
        for (int i = 0; i < info.XAxis.Values.Count; i++)
        {
            object? xv = info.XAxis.Values[i], yv = info.YAxis.Values[i];
            if (xv is null || yv is null)
                continue;
            if (info.Constraints[i]) { fx.Add(xv); fy.Add(yv); }
            else { ix.Add(xv); iy.Add(yv); }
        }

        int ny = yIndices.Count, nx = xIndices.Count;
        var z = new double[ny][];
        for (int r = 0; r < ny; r++)
        {
            z[r] = new double[nx];
            for (int c = 0; c < nx; c++)
                z[r][c] = double.NaN;
        }
        foreach (var kv in info.ZValues)
            z[kv.Key.Y][kv.Key.X] = kv.Value;

        var contour = Fig.Trace("contour",
            ("x", xIndices.ToArray()),
            ("y", yIndices.ToArray()),
            ("z", z),
            ("colorbar", Fig.Obj(("title", targetName))),
            ("colorscale", ColorScales.Blues),
            ("connectgaps", true),
            ("contours", Fig.Obj(("coloring", "heatmap"))),
            ("hoverinfo", "none"),
            ("line", Fig.Obj(("smoothing", 1.3))),
            ("reversescale", reverseScale));

        return (contour, CreateScatter(fx, fy, true), CreateScatter(ix, iy, false));
    }

    private static Trace CreateScatter(List<object> x, List<object> y, bool isFeasible)
        => Fig.Trace("scatter",
            ("x", x.ToArray()),
            ("y", y.ToArray()),
            ("marker", Fig.Obj(
                ("line", Fig.Obj(("width", 2.0), ("color", "Gray"))),
                ("color", isFeasible ? "black" : "#cccccc"))),
            ("mode", "markers"),
            ("name", isFeasible ? "Feasible Trial" : "Infeasible Trial"),
            ("showlegend", false));
}
