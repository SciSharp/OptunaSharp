using System.Collections.Generic;
using System.Linq;
using OptunaSharp.Logging;
using OptunaSharp.Visualization.Plotly.Internal;
using Plotly.NET;
using DObj = DynamicObj.DynamicObj;

namespace OptunaSharp.Visualization.Plotly.Renderers;

/// <summary>
///     Renders <c>optuna.visualization.plot_optimization_history</c> — the
///     <c>_get_optimization_history_plot(info_list, target_name)</c> half of
///     <c>optuna/visualization/_optimization_history.py</c>. Consumes the Phase 3.1 data layer
///     (<see cref="OptimizationHistory.GetInfoList(System.Collections.Generic.IReadOnlyList{Study}, System.Func{FrozenTrial, double}, string, bool)" />).
/// </summary>
internal static class OptimizationHistoryPlot
{
    private static readonly Logger Logger =
        OptunaLogging.GetLogger("optuna.visualization._optimization_history");

    public static GenericChart Render(IReadOnlyList<OptimizationHistoryInfo> infoList, string targetName)
    {
        var layout = Fig.Layout(
            ("title", "Optimization History Plot"),
            ("xaxis", Fig.Obj(("title", "Trial"))),
            ("yaxis", Fig.Obj(("title", targetName))));

        var traces = new List<Trace>();
        foreach (var info in infoList)
        {
            var trialNumbers = info.TrialNumbers;
            var valuesInfo = info.ValuesInfo;
            var bestValuesInfo = info.BestValuesInfo;
            var states = valuesInfo.States;
            var values = valuesInfo.Values;

            // infeasible_trial_numbers = [n for n, s in zip(trial_numbers, states) if s == Infeasible]
            var infeasibleTrialNumbers = new List<int>();
            for (int i = 0; i < trialNumbers.Count; i++)
                if (states[i] == ValueState.Infeasible)
                    infeasibleTrialNumbers.Add(trialNumbers[i]);

            DObj? errorY;
            List<int> feasibleTrialNumbers;
            List<double> feasibleTrialValues;
            List<double> infeasibleTrialValues;

            if (valuesInfo.Stds is null)
            {
                errorY = null;
                feasibleTrialNumbers = new List<int>();
                for (int i = 0; i < trialNumbers.Count; i++)
                    if (states[i] == ValueState.Feasible)
                        feasibleTrialNumbers.Add(trialNumbers[i]);
                // NB: Optuna indexes `values` by trial NUMBER (values_info.values[num]) — replicated verbatim.
                feasibleTrialValues = feasibleTrialNumbers.Select(num => values[num]).ToList();
                infeasibleTrialValues = infeasibleTrialNumbers.Select(num => values[num]).ToList();
            }
            else
            {
                if (states.Contains(ValueState.Infeasible) || states.Contains(ValueState.Incomplete))
                    Logger.Warning(
                        "Your study contains infeasible trials. In optimization history plot, " +
                        "error bars are calculated for only feasible trial values.");
                errorY = Fig.Obj(("type", "data"), ("array", valuesInfo.Stds.ToArray()), ("visible", true));
                feasibleTrialNumbers = trialNumbers.ToList();
                feasibleTrialValues = values.ToList();
                infeasibleTrialValues = new List<double>();
            }

            traces.Add(Fig.Trace("scatter",
                ("x", feasibleTrialNumbers.ToArray()),
                ("y", feasibleTrialValues.ToArray()),
                ("error_y", errorY),                       // null => key omitted (matches error_y=None)
                ("mode", "markers"),
                ("name", valuesInfo.LabelName)));

            if (bestValuesInfo is not null)
            {
                traces.Add(Fig.Trace("scatter",
                    ("x", trialNumbers.ToArray()),
                    ("y", bestValuesInfo.Values.ToArray()),
                    ("name", bestValuesInfo.LabelName),
                    ("mode", "lines")));

                if (bestValuesInfo.Stds is not null)
                {
                    var bestValues = bestValuesInfo.Values;
                    var bestStds = bestValuesInfo.Stds;
                    var upper = new double[trialNumbers.Count];
                    var lower = new double[trialNumbers.Count];
                    for (int i = 0; i < trialNumbers.Count; i++)
                    {
                        upper[i] = bestValues[i] + bestStds[i];
                        lower[i] = bestValues[i] - bestStds[i];
                    }

                    traces.Add(Fig.Trace("scatter",
                        ("x", trialNumbers.ToArray()),
                        ("y", upper),
                        ("mode", "lines"),
                        ("line", Fig.Obj(("width", 0.01))),
                        ("showlegend", false)));

                    traces.Add(Fig.Trace("scatter",
                        ("x", trialNumbers.ToArray()),
                        ("y", lower),
                        ("mode", "none"),
                        ("showlegend", false),
                        ("fill", "tonexty"),
                        ("fillcolor", "rgba(255,0,0,0.2)")));
                }
            }

            traces.Add(Fig.Trace("scatter",
                ("x", infeasibleTrialNumbers.ToArray()),
                ("y", infeasibleTrialValues.ToArray()),
                ("error_y", errorY),
                ("mode", "markers"),
                ("name", "Infeasible Trial"),
                ("marker", Fig.Obj(("color", "#cccccc"))),
                ("showlegend", false)));
        }

        return Fig.Figure(traces, layout);
    }
}
