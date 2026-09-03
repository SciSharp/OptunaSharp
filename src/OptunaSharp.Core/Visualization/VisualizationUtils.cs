using System;
using System.Collections.Generic;
using System.Linq;
using NumSharp;
using OptunaSharp.Collections;
using OptunaSharp.Distributions;
using OptunaSharp.Internal;
using OptunaSharp.Logging;

namespace OptunaSharp.Visualization;

/// <summary>
///     Shared helpers for the visualization data layer. Port of the pure (non-plotly) helpers in
///     <c>optuna/visualization/_utils.py</c>. The plotly-only members (<c>is_available</c>,
///     <c>COLOR_SCALE</c>) belong to the rendering package (Tier X) and are intentionally omitted;
///     <c>_make_json_compatible</c>/<c>_make_hovertext</c> land with their first consumer (timeline).
/// </summary>
internal static class VisualizationUtils
{
    private static readonly Logger Logger = OptunaLogging.GetLogger("optuna.visualization._utils");

    /// <summary>Python <c>_check_plot_args</c>.</summary>
    internal static void CheckPlotArgs(
        IReadOnlyList<Study> studies, Func<FrozenTrial, double>? target, string targetName)
    {
        if (target is null && studies.Any(s => s.IsMultiObjective()))
            throw new ValueError(
                "If the `study` is being used for multi-objective optimization, " +
                "please specify the `target`.");

        if (target is not null && targetName == "Objective Value")
            OptunaWarnings.Warn(
                "`target` is specified, but `target_name` is the default value, 'Objective Value'.");
    }

    /// <inheritdoc cref="CheckPlotArgs(System.Collections.Generic.IReadOnlyList{Study}, System.Func{FrozenTrial, double}, string)" />
    internal static void CheckPlotArgs(Study study, Func<FrozenTrial, double>? target, string targetName)
        => CheckPlotArgs(new[] { study }, target, targetName);

    /// <summary>Python <c>_is_log_scale</c>.</summary>
    internal static bool IsLogScale(IReadOnlyList<FrozenTrial> trials, string param)
    {
        foreach (var trial in trials)
        {
            if (!trial.Params.ContainsKey(param))
                continue;
            var dist = trial.Distributions[param];
            return dist switch
            {
                FloatDistribution f => f.Log,
                IntDistribution i => i.Log,
                _ => false,
            };
        }
        return false;
    }

    /// <summary>Python <c>_is_numerical</c>.</summary>
    internal static bool IsNumerical(IReadOnlyList<FrozenTrial> trials, string param)
    {
        foreach (var trial in trials)
        {
            if (!trial.Params.ContainsKey(param))
                continue;
            var dist = trial.Distributions[param];
            switch (dist)
            {
                case IntDistribution:
                case FloatDistribution:
                    return true;
                case CategoricalDistribution c:
                    // NOTE: Although it is a bit odd to do so, we keep it as is only for visualization.
                    // Python: all values are int or float (and not bool). None/str/bool are non-numerical.
                    return c.Choices.All(v => v.Value is long or double);
                default:
                    throw new InvalidOperationException("Should not reach.");
            }
        }
        return true;
    }

    /// <summary>
    ///     Python <c>_filter_nonfinite</c>. Drops trials whose target value is not finite (NaN, +inf
    ///     or -inf); logs a warning per omitted trial when <paramref name="withMessage" /> is true.
    /// </summary>
    internal static List<FrozenTrial> FilterNonfinite(
        IReadOnlyList<FrozenTrial> trials,
        Func<FrozenTrial, double>? target = null,
        bool withMessage = true)
    {
        // target is None only for single-objective studies (callers sanitize via CheckPlotArgs),
        // so t.Value is available; a missing value is treated as non-finite and filtered out.
        target ??= t => t.Value ?? double.NaN;

        var filtered = new List<FrozenTrial>();
        foreach (var trial in trials)
        {
            double value = target(trial);
            if (!double.IsFinite(value))
            {
                if (withMessage)
                    Logger.Warning(
                        $"Trial {trial.Number} is omitted in visualization " +
                        "because its objective value is inf or nan.");
            }
            else
            {
                filtered.Add(trial);
            }
        }

        return filtered;
    }

    /// <summary>Python <c>_is_reverse_scale</c>.</summary>
    internal static bool IsReverseScale(Study study, Func<FrozenTrial, double>? target)
        => target is not null || study.Direction == StudyDirection.Minimize;

    /// <summary>
    ///     Python <c>_make_json_compatible</c>: return the value unchanged if it is JSON-serializable,
    ///     else its <c>str()</c>.
    /// </summary>
    internal static object? MakeJsonCompatible(object? value)
    {
        try
        {
            PyJson.Dumps(value);
            return value;
        }
        catch (TypeError)
        {
            return PythonRepr.Str(value);
        }
    }

    /// <summary>
    ///     Python <c>_make_hovertext(trial)</c>: <c>json.dumps({number, values, params, [user_attrs]},
    ///     indent=2)</c> with newlines turned into <c>&lt;br&gt;</c>.
    /// </summary>
    internal static string MakeHovertext(FrozenTrial trial)
    {
        var userAttrs = new OrderedDict<string, object?>();
        foreach (var (key, value) in trial.UserAttrs)
            userAttrs[key] = MakeJsonCompatible(value);

        var dict = new OrderedDict<string, object?>
        {
            { "number", trial.Number },
            { "values", trial.Values },
            { "params", trial.ParamsDict },
        };
        if (userAttrs.Count > 0)
            dict["user_attrs"] = userAttrs;

        return PyJson.DumpsIndented(dict, 2).Replace("\n", "<br>");
    }
}
