using System;
using System.Collections.Generic;
using System.Linq;
using OptunaSharp.Distributions;
using OptunaSharp.Internal;
using OptunaSharp.Visualization.Plotly.Internal;
using Plotly.NET;
using DObj = DynamicObj.DynamicObj;

namespace OptunaSharp.Visualization.Plotly.Renderers;

/// <summary>
///     Renders <c>optuna.visualization.plot_slice</c> — the plotly half of <c>_slice.py</c>. One
///     subplot per parameter (a single plot, or a 1×N <see cref="SubplotGrid" /> with shared y-axes):
///     feasible trials as markers colored by trial number on the Blues scale (with a colorbar, shown
///     once), infeasible trials grey, plus per-axis log / categorical (with <c>repr</c> labels) typing.
/// </summary>
internal static class SlicePlotRenderer
{
    public static GenericChart Render(SlicePlotInfo info)
    {
        var layout = Fig.Layout(("title", "Slice Plot"));
        int n = info.Subplots.Count;
        if (n == 0)
            return Fig.Figure(Array.Empty<Trace>(), layout);

        if (n == 1)
        {
            var sub = info.Subplots[0];
            var trace = GenerateSubplot(sub, showscale: null);
            var xaxis = new DObj();
            var yaxis = new DObj();
            layout.SetValue("xaxis", xaxis);
            layout.SetValue("yaxis", yaxis);
            xaxis.SetValue("title", sub.ParamName);
            yaxis.SetValue("title", info.TargetName);
            ApplyXType(xaxis, sub);
            return Fig.Figure(trace, layout);
        }

        var grid = SubplotGrid.Init(layout, 1, n);
        var traces = new List<Trace>();
        bool showscale = true;
        for (int col = 1; col <= n; col++)
        {
            var sub = info.Subplots[col - 1];
            var trace = GenerateSubplot(sub, showscale);
            if (showscale)
                showscale = false;
            foreach (var t in trace)
            {
                t.SetValue("xaxis", SubplotGrid.XAnchor(col));
                t.SetValue("yaxis", SubplotGrid.YAnchor(col));
                traces.Add(t);
            }

            grid.XAxis(col).SetValue("title", sub.ParamName);
            if (col == 1)
                grid.YAxis(col).SetValue("title", info.TargetName);
            ApplyXType(grid.XAxis(col), sub);
        }

        if (n > 3)
            layout.SetValue("width", 300 * n);

        return Fig.Figure(traces, layout);
    }

    private static void ApplyXType(DObj xaxis, SliceSubplotInfo sub)
    {
        if (!sub.IsNumerical)
        {
            xaxis.SetValue("type", "category");
            xaxis.SetValue("categoryorder", "array");
            xaxis.SetValue("categoryarray", CategoricalLabels(sub.XLabels!));
        }
        else if (sub.IsLog)
        {
            xaxis.SetValue("type", "log");
        }
    }

    private static List<Trace> GenerateSubplot(SliceSubplotInfo sub, bool? showscale)
    {
        var fx = new List<object?>();
        var fy = new List<double>();
        var fnum = new List<int>();
        var ix = new List<object?>();
        var iy = new List<double>();

        for (int i = 0; i < sub.X.Count; i++)
        {
            object? x = sub.X[i];
            if (sub.IsNumerical && x is null)
                continue;
            if (sub.Constraints[i]) { fx.Add(x); fy.Add(sub.Y[i]); fnum.Add(sub.TrialNumbers[i]); }
            else { ix.Add(x); iy.Add(sub.Y[i]); }
        }

        object?[] feasibleX, infeasibleX;
        double[] feasibleY, infeasibleY;
        int[] feasibleC;
        if (sub.IsNumerical)
        {
            feasibleX = fx.ToArray();
            feasibleY = fy.ToArray();
            feasibleC = fnum.ToArray();
            infeasibleX = ix.ToArray();
            infeasibleY = iy.ToArray();
        }
        else
        {
            (feasibleX, feasibleY, feasibleC) = CategoricalPlotValues(sub, fx, fy, fnum);
            (infeasibleX, infeasibleY, _) = CategoricalPlotValues(sub, ix, iy, new List<int>(new int[ix.Count]));
        }

        var feasibleMarker = Fig.Obj(
            ("line", Fig.Obj(("width", 0.5), ("color", "Grey"))),
            ("color", feasibleC),
            ("colorscale", ColorScales.Blues),
            ("colorbar", Fig.Obj(("title", "Trial"), ("x", 1.0), ("xpad", 40))));
        if (showscale is { } s)
            feasibleMarker.SetValue("showscale", s);

        var trace = new List<Trace>
        {
            Fig.Trace("scatter",
                ("x", feasibleX),
                ("y", feasibleY),
                ("mode", "markers"),
                ("name", "Feasible Trial"),
                ("marker", feasibleMarker),
                ("showlegend", false)),
        };

        if (infeasibleX.Length > 0)
            trace.Add(Fig.Trace("scatter",
                ("x", infeasibleX),
                ("y", infeasibleY),
                ("mode", "markers"),
                ("name", "Infeasible Trial"),
                ("marker", Fig.Obj(("color", "#cccccc"))),
                ("showlegend", false)));

        return trace;
    }

    // Python _get_categorical_plot_values — regroup points into x_labels order, x = repr(label).
    private static (object?[] X, double[] Y, int[] C) CategoricalPlotValues(
        SliceSubplotInfo sub, List<object?> xs, List<double> ys, List<int> nums)
    {
        var pointsDict = new Dictionary<CategoricalChoice, List<(double Y, int Num)>>();
        for (int i = 0; i < xs.Count; i++)
        {
            var key = CategoricalChoice.From(xs[i]);
            if (!pointsDict.TryGetValue(key, out var list))
                pointsDict[key] = list = new List<(double, int)>();
            list.Add((ys[i], nums[i]));
        }

        var vx = new List<object?>();
        var vy = new List<double>();
        var vc = new List<int>();
        foreach (var label in sub.XLabels!)
        {
            if (!pointsDict.TryGetValue(label, out var pts))
                continue;
            foreach (var (y, num) in pts)
            {
                vx.Add(PythonRepr.Repr(label.Value));
                vy.Add(y);
                vc.Add(num);
            }
        }

        return (vx.ToArray(), vy.ToArray(), vc.ToArray());
    }

    private static string[] CategoricalLabels(IReadOnlyList<CategoricalChoice> xLabels)
        => xLabels.Select(l => PythonRepr.Repr(l.Value)).ToArray();
}
