using System;
using System.Collections.Generic;
using System.Linq;
using OptunaSharp.Internal;
using OptunaSharp.Logging;
using OptunaSharp.Visualization.Plotly.Internal;
using Plotly.NET;

namespace OptunaSharp.Visualization.Plotly.Renderers;

/// <summary>
///     Renders <c>optuna.visualization.plot_terminator_improvement</c> — the plotly half of
///     <c>_terminator_improvement.py</c>. A blue <c>markers+lines</c> improvement trace (its first
///     <c>min_n_trials+1</c> points drawn at a lighter opacity), an optional second full-opacity
///     segment, and a red error trace (an empty scatter when errors are not computed). The y-axis range
///     is Optuna's padded window over the post-<c>min_n_trials</c> trials.
/// </summary>
internal static class TerminatorImprovementPlot
{
    private const double PaddingRatioY = 0.05;
    private const double Opacity = 0.25;

    private static readonly Logger Logger =
        OptunaLogging.GetLogger("optuna.visualization._terminator_improvement");

    public static GenericChart Render(ImprovementInfo info, int minNTrials)
    {
        var yaxis = Fig.Obj(("title", "Terminator Improvement"));
        var layout = Fig.Layout(
            ("title", "Terminator Improvement Plot"),
            ("xaxis", Fig.Obj(("title", "Trial"))),
            ("yaxis", yaxis));

        int nTrials = info.TrialNumbers.Count;
        if (nTrials == 0)
        {
            Logger.Warning("There are no complete trials.");
            return Fig.Figure(Array.Empty<Trace>(), layout);
        }

        var traces = new List<Trace>
        {
            // Lighter color until the number of trials reaches min_n_trials; legend only if that is all.
            ImprovementScatter(
                info.TrialNumbers.Take(minNTrials + 1).ToArray(),
                info.Improvements.Take(minNTrials + 1).ToArray(),
                Opacity,
                nTrials <= minNTrials),
        };

        if (nTrials > minNTrials)
            traces.Add(ImprovementScatter(
                info.TrialNumbers.Skip(minNTrials).ToArray(),
                info.Improvements.Skip(minNTrials).ToArray(),
                1.0,
                true));

        traces.Add(ErrorScatter(info.TrialNumbers, info.Errors));

        var (lo, hi) = YRange(info, minNTrials);
        yaxis.SetValue("range", new[] { lo, hi });
        return Fig.Figure(traces, layout);
    }

    private static Trace ImprovementScatter(int[] x, double[] y, double opacity, bool showlegend)
    {
        string blue = $"rgba(99, 110, 250, {PythonRepr.FloatRepr(opacity)})";
        return Fig.Trace("scatter",
            ("x", x),
            ("y", y),
            ("mode", "markers+lines"),
            ("marker", Fig.Obj(("color", blue))),
            ("line", Fig.Obj(("color", blue))),
            ("name", "Terminator Improvement"),
            ("showlegend", showlegend),
            ("legendgroup", "improvement"));
    }

    private static Trace ErrorScatter(IReadOnlyList<int> trialNumbers, IReadOnlyList<double>? errors)
    {
        if (errors is null)
            return Fig.Trace("scatter");   // Python go.Scatter() — a bare {type:"scatter"}.

        return Fig.Trace("scatter",
            ("x", trialNumbers.ToArray()),
            ("y", errors.ToArray()),
            ("mode", "markers+lines"),
            ("name", "Error"),
            ("marker", Fig.Obj(("color", "rgb(239, 85, 59)"))),
            ("line", Fig.Obj(("color", "rgb(239, 85, 59)"))));
    }

    private static (double Low, double High) YRange(ImprovementInfo info, int minNTrials)
    {
        double minValue = info.Improvements.Min();
        if (info.Errors is not null)
            minValue = Math.Min(minValue, info.Errors.Min());

        double maxValue = info.TrialNumbers.Count > minNTrials
            ? info.Improvements.Skip(minNTrials).Max()
            : info.Improvements.Max();
        if (info.Errors is not null)
            maxValue = Math.Max(maxValue, info.Errors.Max());

        double padding = (maxValue - minValue) * PaddingRatioY;
        return (minValue - padding, maxValue + padding);
    }
}
