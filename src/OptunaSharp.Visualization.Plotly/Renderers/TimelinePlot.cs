using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using OptunaSharp.Visualization.Plotly.Internal;
using Plotly.NET;

namespace OptunaSharp.Visualization.Plotly.Renderers;

/// <summary>
///     Renders <c>optuna.visualization.plot_timeline</c> — the plotly half of <c>_timeline.py</c>. One
///     horizontal <c>bar</c> trace per <see cref="TrialState" /> (states walked in name order, matching
///     Optuna's <c>sorted(TrialState, key=name)</c>), with COMPLETE split into a grey INFEASIBLE bar
///     and the blue feasible bar. Each bar's <c>base</c> is the start time and its <c>x</c> is the run
///     duration in milliseconds.
/// </summary>
internal static class TimelinePlot
{
    private static readonly IReadOnlyDictionary<string, string> ColorMap = new Dictionary<string, string>
    {
        ["COMPLETE"] = "blue",
        ["FAIL"] = "red",
        ["PRUNED"] = "orange",
        ["RUNNING"] = "green",
        ["WAITING"] = "gray",
    };

    // sorted(TrialState, key=lambda x: x.name) => COMPLETE, FAIL, PRUNED, RUNNING, WAITING.
    private static readonly TrialState[] StatesByName =
    {
        TrialState.Complete, TrialState.Fail, TrialState.Pruned, TrialState.Running, TrialState.Waiting,
    };

    /// <summary>Converts extracted visualization data into the corresponding Plotly figure.</summary>
    /// <param name="info">Validated plot data produced by the corresponding Core visualization extractor.</param>
    /// <returns>A chart containing the plot's traces, axes, and labels.</returns>
    public static GenericChart Render(TimelineInfo info)
    {
        var traces = new List<Trace>();
        foreach (var state in StatesByName)
        {
            string name = StateName(state);
            if (state == TrialState.Complete)
            {
                var infeasible = info.Bars.Where(b => b.State == state && b.Infeasible).ToList();
                var feasible = info.Bars.Where(b => b.State == state && !b.Infeasible).ToList();
                PlotBars(infeasible, "#cccccc", "INFEASIBLE", traces);
                PlotBars(feasible, ColorMap[name], name, traces);
            }
            else
            {
                var bars = info.Bars.Where(b => b.State == state).ToList();
                PlotBars(bars, ColorMap[name], name, traces);
            }
        }

        var layout = Fig.Layout(
            ("title", "Timeline Plot"),
            ("xaxis", Fig.Obj(("title", "Datetime"), ("type", "date"))),
            ("yaxis", Fig.Obj(("title", "Trial"))),
            ("showlegend", true));

        return Fig.Figure(traces, layout);
    }

    /// <summary>Appends a horizontal duration-bar trace for a nonempty state group.</summary>
    /// <param name="bars">Trial lifetimes sharing one state or feasibility group.</param>
    /// <param name="color">Plotly marker color for this group.</param>
    /// <param name="name">State or feasibility label used in the legend and hover text.</param>
    /// <param name="traces">Destination trace list; appended to only when bars is nonempty.</param>
    private static void PlotBars(IReadOnlyList<TimelineBarInfo> bars, string color, string name, List<Trace> traces)
    {
        if (bars.Count == 0)
            return;

        traces.Add(Fig.Trace("bar",
            ("name", name),
            ("x", bars.Select(b => (b.Complete - b.Start).TotalSeconds * 1000.0).ToArray()),
            ("y", bars.Select(b => b.Number).ToArray()),
            ("base", bars.Select(b => Iso(b.Start)).ToArray()),
            ("text", bars.Select(b => b.Hovertext).ToArray()),
            ("hovertemplate", "%{text}<extra>" + name + "</extra>"),
            ("orientation", "h"),
            ("marker", Fig.Obj(("color", color))),
            ("textposition", "none")));
    }

    /// <summary>Python <c>datetime.isoformat()</c> — microseconds only when non-zero.</summary>
    /// <param name="dt">Local wall-clock timestamp for the timeline bar.</param>
    /// <returns>Offset-free ISO timestamp with microseconds only when nonzero.</returns>
    private static string Iso(DateTime dt)
        => dt.Ticks % TimeSpan.TicksPerSecond == 0
            ? dt.ToString("yyyy-MM-ddTHH:mm:ss", CultureInfo.InvariantCulture)
            : dt.ToString("yyyy-MM-ddTHH:mm:ss.ffffff", CultureInfo.InvariantCulture);

    /// <summary>Formats a trial state using the uppercase labels expected by the timeline legend.</summary>
    /// <param name="state">Trial state associated with the bar group.</param>
    /// <returns>Uppercase state name.</returns>
    private static string StateName(TrialState state) => state switch
    {
        TrialState.Complete => "COMPLETE",
        TrialState.Fail => "FAIL",
        TrialState.Pruned => "PRUNED",
        TrialState.Running => "RUNNING",
        TrialState.Waiting => "WAITING",
        _ => state.ToString().ToUpperInvariant(),
    };
}
