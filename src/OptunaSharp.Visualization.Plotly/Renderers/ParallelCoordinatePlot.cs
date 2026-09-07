using System;
using System.Collections.Generic;
using System.Linq;
using OptunaSharp.Visualization.Plotly.Internal;
using Plotly.NET;

namespace OptunaSharp.Visualization.Plotly.Renderers;

/// <summary>
///     Renders <c>optuna.visualization.plot_parallel_coordinate</c> — the plotly half of
///     <c>_parallel_coordinate.py</c> (the version rewritten off <c>go.Parcoords</c> onto plain
///     <c>go.Scatter</c> lines). One line per trial (drawn in <c>draw_order</c>, colored on the Blues
///     scale, dashed when infeasible), the hidden colorbar trace, per-axis vertical shapes, and the
///     tick-label annotations.
/// </summary>
internal static class ParallelCoordinatePlot
{
    public static GenericChart Render(ParallelCoordinateInfo info)
    {
        var layout = Fig.Layout(("title", Fig.Obj(
            ("text", "Parallel Coordinate Plot"), ("y", 1), ("yanchor", "bottom"), ("yref", "paper"))));

        if (info.DimsParams.Count == 0 || info.DimsObjectives[0].Values.Count == 0)
            return Fig.Figure(Array.Empty<Trace>(), layout);

        var dimensions = info.DimsObjectives.Concat(info.DimsParams).ToList();
        var (colorMin, colorMax) = GetColorRange(info.ColorValues, info.IsRankColor);
        var normalized = dimensions.Select(Normalize).ToList();
        var colors = GetLineColors(info.ColorValues, colorMin, colorMax, info.ReverseScale);
        var xValues = Enumerable.Range(0, dimensions.Count).ToArray();

        var traces = new List<Trace>();
        foreach (int trialId in info.DrawOrder)
        {
            bool isFeasible = info.Feasibility is null || info.Feasibility[trialId];
            traces.Add(Fig.Trace("scatter",
                ("x", xValues),
                ("y", normalized.Select(nd => nd[trialId]).ToArray()),
                ("customdata", dimensions
                    .Select(dim => new object[] { dim.Label, FormatDimensionValue(dim, dim.Values[trialId]) })
                    .ToArray()),
                ("mode", "lines"),
                ("line", Fig.Obj(
                    ("color", colors[trialId]),
                    ("dash", isFeasible ? "solid" : "dot"),
                    ("width", 1.2))),
                ("opacity", 0.5),
                ("showlegend", false),
                ("hovertemplate", "%{customdata[0]}: %{customdata[1]}<extra></extra>")));
        }

        var colorbar = Fig.Obj(("title", info.ColorbarTitle));
        if (info.IsRankColor)
            colorbar.SetValue("tickvals", Enumerable.Range(0, (int)info.ColorValues.Max() + 1).ToArray());
        if (info.Feasibility is not null)
        {
            traces.Add(FeasibilityLegend("solid", "Feasible"));
            traces.Add(FeasibilityLegend("dot", "Infeasible"));
        }

        traces.Add(Fig.Trace("scatter",
            ("x", new object?[] { null, null }),
            ("y", new object?[] { null, null }),
            ("mode", "markers"),
            ("marker", Fig.Obj(
                ("color", new[] { colorMin, colorMax }),
                ("cmin", colorMin),
                ("cmax", colorMax),
                ("colorscale", ColorScales.Blues),
                ("reversescale", info.ReverseScale),
                ("showscale", true),
                ("colorbar", colorbar))),
            ("showlegend", false),
            ("hoverinfo", "skip")));

        var shapes = xValues.Select(axisId => (object)Fig.Obj(
            ("type", "line"),
            ("x0", axisId), ("x1", axisId), ("y0", 0), ("y1", 1),
            ("line", Fig.Obj(("color", "#888"), ("width", 1))),
            ("layer", "below"))).ToArray();

        layout.SetValue("xaxis", Fig.Obj(
            ("tickmode", "array"),
            ("tickvals", xValues),
            ("ticktext", dimensions.Select(d => d.Label).ToArray()),
            ("range", new[] { -0.2, dimensions.Count - 0.8 }),
            ("showgrid", false),
            ("zeroline", false)));
        layout.SetValue("yaxis", Fig.Obj(("range", new[] { -0.08, 1.08 }), ("visible", false), ("fixedrange", true)));
        layout.SetValue("annotations", GetAxisAnnotations(dimensions));
        layout.SetValue("shapes", shapes);
        layout.SetValue("hovermode", "closest");
        layout.SetValue("plot_bgcolor", "white");
        layout.SetValue("legend", Fig.Obj(
            ("orientation", "h"), ("x", 1), ("xanchor", "right"), ("y", 1), ("yanchor", "bottom")));

        return Fig.Figure(traces, layout);
    }

    private static Trace FeasibilityLegend(string dash, string name)
        => Fig.Trace("scatter",
            ("x", new object?[] { null }),
            ("y", new object?[] { null }),
            ("mode", "lines"),
            ("line", Fig.Obj(("color", "#2F3B52"), ("dash", dash), ("width", 2))),
            ("name", name),
            ("showlegend", true),
            ("hoverinfo", "skip"));

    private static (double Min, double Max) GetColorRange(IReadOnlyList<double> values, bool isRank)
    {
        if (isRank)
            return (-0.5, values.Max() + 0.5);
        double min = values.Min(), max = values.Max();
        return min == max ? (min - 0.5, max + 0.5) : (min, max);
    }

    private static double[] Normalize(DimensionInfo dim)
    {
        var (min, max) = dim.Range;
        if (min == max)
            return dim.Values.Select(_ => 0.5).ToArray();
        return dim.Values.Select(v => (v - min) / (max - min)).ToArray();
    }

    private static string[] GetLineColors(IReadOnlyList<double> values, double min, double max, bool reverse)
    {
        var norm = new double[values.Count];
        for (int i = 0; i < values.Count; i++)
            norm[i] = min == max ? 0.5 : (values[i] - min) / (max - min);
        if (reverse)
            for (int i = 0; i < norm.Length; i++)
                norm[i] = 1.0 - norm[i];
        return norm.Select(PlotlyColors.BluesLineColor).ToArray();
    }

    private static string FormatDimensionValue(DimensionInfo dim, double value)
    {
        for (int i = 0; i < dim.Tickvals.Count; i++)
            if (IsClose(value, dim.Tickvals[i]))
                return dim.Ticktext[i];
        return PlotlyColors.FormatG(dim.IsLog ? Math.Pow(10, value) : value, 3);
    }

    private static object[] GetAxisAnnotations(List<DimensionInfo> dimensions)
    {
        var annotations = new List<object>();
        for (int axisId = 0; axisId < dimensions.Count; axisId++)
        {
            var dim = dimensions[axisId];
            double[] tickValues;
            string[] tickText;
            if (dim.Tickvals.Count > 0)
            {
                tickValues = dim.Tickvals.ToArray();
                tickText = dim.Ticktext.Count > 0
                    ? dim.Ticktext.ToArray()
                    : tickValues.Select(v => PlotlyColors.FormatG(v, 3)).ToArray();
            }
            else if (dim.Range.Min == dim.Range.Max)
            {
                tickValues = new[] { dim.Range.Min };
                tickText = new[] { PlotlyColors.FormatG(dim.Range.Min, 3) };
            }
            else
            {
                tickValues = Linspace(dim.Range.Min, dim.Range.Max, 5);
                tickText = tickValues.Select(v => PlotlyColors.FormatG(v, 3)).ToArray();
            }

            var normalizedTicks = Normalize(dim with { Values = tickValues });
            for (int i = 0; i < tickValues.Length; i++)
                annotations.Add(Fig.Obj(
                    ("x", axisId),
                    ("y", normalizedTicks[i]),
                    ("text", tickText[i]),
                    ("showarrow", false),
                    ("xanchor", "right"),
                    ("xshift", -4),
                    ("font", Fig.Obj(("color", "#555"), ("size", 10)))));
        }
        return annotations.ToArray();
    }

    /// <summary>NumPy <c>np.linspace(min, max, num)</c> — endpoint forced exact.</summary>
    private static double[] Linspace(double min, double max, int num)
    {
        var result = new double[num];
        double step = (max - min) / (num - 1);
        for (int i = 0; i < num; i++)
            result[i] = min + i * step;
        result[num - 1] = max;
        return result;
    }

    private static bool IsClose(double a, double b)
        => Math.Abs(a - b) <= 1e-9 * Math.Max(Math.Abs(a), Math.Abs(b));
}
