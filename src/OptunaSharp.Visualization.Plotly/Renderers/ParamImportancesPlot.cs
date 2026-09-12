using System;
using System.Collections.Generic;
using System.Linq;
using OptunaSharp.Distributions;
using OptunaSharp.Internal;
using OptunaSharp.Visualization.Plotly.Internal;
using Plotly.NET;

namespace OptunaSharp.Visualization.Plotly.Renderers;

/// <summary>
///     Renders <c>optuna.visualization.plot_param_importances</c> — the plotly half of
///     <c>_param_importances.py</c>. One horizontal <c>bar</c> per objective (empty infos skipped),
///     with the per-bar hovertemplate <c>"{param} ({DistributionClass}): {importance}&lt;extra&gt;&lt;/extra&gt;"</c>
///     — the raw importance formatted with CPython's <c>repr(float)</c> (Core's <see cref="PythonRepr" />).
/// </summary>
internal static class ParamImportancesPlot
{
    /// <summary>Converts extracted visualization data into the corresponding Plotly figure.</summary>
    /// <param name="infos">Importance results in objective order.</param>
    /// <param name="study">Study whose recorded trials are visualized.</param>
    /// <returns>A chart containing the plot's traces, axes, and labels.</returns>
    public static GenericChart Render(IReadOnlyList<ImportancesInfo> infos, Study study)
    {
        var layout = Fig.Layout(
            ("title", "Hyperparameter Importances"),
            ("xaxis", Fig.Obj(("title", "Hyperparameter Importance"))),
            ("yaxis", Fig.Obj(("title", "Hyperparameter"))));

        var traces = new List<Trace>();
        foreach (var info in infos)
        {
            if (info.ImportanceValues.Count == 0)
                continue;

            traces.Add(Fig.Trace("bar",
                ("x", info.ImportanceValues.ToArray()),
                ("y", info.ParamNames.ToArray()),
                ("name", info.TargetName),
                ("text", info.ImportanceLabels.ToArray()),
                ("textposition", "outside"),
                ("cliponaxis", false),
                ("hovertemplate", HoverTemplate(info, study)),
                ("orientation", "h")));
        }

        return Fig.Figure(traces, layout);
    }

    /// <summary>Builds per-parameter hover text including distribution class and importance.</summary>
    /// <param name="info">Validated plot data produced by the corresponding Core visualization extractor.</param>
    /// <param name="study">Study whose recorded trials are visualized.</param>
    /// <returns>Hover templates aligned with the importance parameter names.</returns>
    /// <exception cref="System.InvalidOperationException">A parameter has no distribution in the study history.</exception>
    private static string[] HoverTemplate(ImportancesInfo info, Study study)
    {
        var result = new string[info.ParamNames.Count];
        for (int i = 0; i < result.Length; i++)
        {
            string paramName = info.ParamNames[i];
            string className = GetDistribution(paramName, study).GetType().Name;
            // f"{param_name} ({class_name}): {importance}<extra></extra>" — {importance} is repr(float).
            result[i] = $"{paramName} ({className}): {PythonRepr.FloatRepr(info.ImportanceValues[i])}<extra></extra>";
        }
        return result;
    }

    /// <summary>Python <c>_get_distribution(param_name, study)</c> — the first trial that carries it.</summary>
    /// <param name="paramName">Parameter whose first recorded distribution supplies hover metadata.</param>
    /// <param name="study">Study whose recorded trials are visualized.</param>
    /// <returns>The first distribution recorded under this parameter name.</returns>
    /// <exception cref="System.InvalidOperationException">No trial records the requested parameter.</exception>
    private static BaseDistribution GetDistribution(string paramName, Study study)
    {
        foreach (var trial in study.GetTrials(deepcopy: false))
            if (trial.Distributions.TryGetValue(paramName, out var dist))
                return dist;
        throw new InvalidOperationException($"No distribution found for parameter '{paramName}'.");
    }
}
