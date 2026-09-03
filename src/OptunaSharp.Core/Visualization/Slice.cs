using System;
using System.Collections.Generic;
using System.Linq;
using NumSharp;
using OptunaSharp.Distributions;
using OptunaSharp.Logging;

namespace OptunaSharp.Visualization;

/// <summary>
///     One parameter's slice subplot data. Port of <c>_SliceSubplotInfo</c> in
///     <c>optuna/visualization/_slice.py</c>. <see cref="X" /> holds the external parameter values
///     (a boxed <c>double</c>/<c>long</c>/<c>string</c>/<c>bool</c>/<c>null</c>), one per trial that
///     has the parameter.
/// </summary>
public sealed record SliceSubplotInfo(
    string ParamName,
    IReadOnlyList<object?> X,
    IReadOnlyList<double> Y,
    IReadOnlyList<int> TrialNumbers,
    bool IsLog,
    bool IsNumerical,
    IReadOnlyList<bool> Constraints,
    IReadOnlyList<CategoricalChoice>? XLabels);

/// <summary>
///     The data backing a slice plot: one subplot per parameter. Port of <c>_SlicePlotInfo</c>.
/// </summary>
public sealed record SlicePlotInfo(string TargetName, IReadOnlyList<SliceSubplotInfo> Subplots);

/// <summary>
///     Data layer for <c>optuna.visualization.plot_slice</c> — the pure <c>_get_slice_plot_info</c>/
///     <c>_get_slice_subplot_info</c> half of <c>_slice.py</c>. Named <c>SlicePlot</c> (not
///     <c>Slice</c>) to avoid a collision with <see cref="NumSharp.Slice" />.
/// </summary>
public static class SlicePlot
{
    private static readonly Logger Logger = OptunaLogging.GetLogger("optuna.visualization._slice");

    /// <summary>Python <c>_get_slice_plot_info(study, params, target, target_name)</c>.</summary>
    public static SlicePlotInfo GetPlotInfo(
        Study study,
        IReadOnlyList<string>? @params = null,
        Func<FrozenTrial, double>? target = null,
        string targetName = "Objective Value")
    {
        VisualizationUtils.CheckPlotArgs(study, target, targetName);

        var trials = VisualizationUtils.FilterNonfinite(
            study.GetTrials(deepcopy: false, states: new[] { TrialState.Complete }), target: target);

        if (trials.Count == 0)
        {
            Logger.Warning("Your study does not have any completed trials.");
            return new SlicePlotInfo(targetName, new List<SliceSubplotInfo>());
        }

        var allParams = new HashSet<string>(trials.SelectMany(t => t.Params.Keys));

        // First-seen distribution per parameter (mirrors the Python dict-insertion behavior).
        var distributions = new Dictionary<string, BaseDistribution>();
        foreach (var trial in trials)
            foreach (var (paramName, distribution) in trial.Distributions)
                if (!distributions.ContainsKey(paramName))
                    distributions[paramName] = distribution;

        var xLabels = new Dictionary<string, IReadOnlyList<CategoricalChoice>>();
        foreach (var (paramName, distribution) in distributions)
            if (distribution is CategoricalDistribution cat)
                xLabels[paramName] = cat.Choices;

        List<string> sortedParams;
        if (@params is null)
        {
            sortedParams = allParams.OrderBy(p => p, StringComparer.Ordinal).ToList();
        }
        else
        {
            foreach (var inputName in @params)
                if (!allParams.Contains(inputName))
                    throw new ValueError($"Parameter {inputName} does not exist in your study.");
            sortedParams = @params.Distinct().OrderBy(p => p, StringComparer.Ordinal).ToList();
        }

        var subplots = sortedParams
            .Select(param => GetSliceSubplotInfo(
                trials,
                param,
                target,
                logScale: VisualizationUtils.IsLogScale(trials, param),
                numerical: distributions[param] is not CategoricalDistribution,
                xLabels: xLabels.GetValueOrDefault(param)))
            .ToList();

        return new SlicePlotInfo(targetName, subplots);
    }

    /// <summary>Python <c>_get_slice_subplot_info(trials, param, target, log_scale, numerical, x_labels)</c>.</summary>
    private static SliceSubplotInfo GetSliceSubplotInfo(
        IReadOnlyList<FrozenTrial> trials,
        string param,
        Func<FrozenTrial, double>? target,
        bool logScale,
        bool numerical,
        IReadOnlyList<CategoricalChoice>? xLabels)
    {
        target ??= t => t.Value ?? double.NaN;

        var x = new List<object?>();
        var y = new List<double>();
        var trialNumbers = new List<int>();
        var constraints = new List<bool>();

        foreach (var t in trials)
        {
            if (!t.Params.TryGetValue(param, out var paramValue))
                continue;
            x.Add(paramValue);
            y.Add(target(t));
            trialNumbers.Add(t.Number);
            constraints.Add(t.Constraints.Values.All(v => v <= 0.0));
        }

        return new SliceSubplotInfo(param, x, y, trialNumbers, logScale, numerical, constraints, xLabels);
    }
}
