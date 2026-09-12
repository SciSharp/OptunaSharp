using System;
using System.Collections.Generic;
using System.Linq;
using NumSharp;
using OptunaSharp.Distributions;
using OptunaSharp.Internal;
using OptunaSharp.Logging;

namespace OptunaSharp.Visualization;

/// <summary>
///     One axis of a contour subplot. Port of <c>_AxisInfo</c> in
///     <c>optuna/visualization/_contour.py</c>. <see cref="Indices" /> is the sorted unique axis
///     values (with the padded numeric endpoints prepended/appended for numerical axes);
///     <see cref="Values" /> is the per-trial value (numeric, string, or <c>null</c> when the
///     parameter is absent).
/// </summary>
/// <param name="Name">The parameter name identifying this axis.</param>
/// <param name="Range">The padded minimum and maximum positions used as the axis bounds.</param>
/// <param name="IsLog">Whether the axis uses logarithmic scaling.</param>
/// <param name="IsCat">Whether the axis represents discrete categorical labels.</param>
/// <param name="Indices">The sorted distinct axis positions, including any synthetic boundary positions.</param>
/// <param name="Values">The per-trial values in the same order as the plot's trial sequence.</param>
public sealed record ContourAxisInfo(
    string Name,
    (double Min, double Max) Range,
    bool IsLog,
    bool IsCat,
    IReadOnlyList<object> Indices,
    IReadOnlyList<object?> Values);

/// <summary>
///     One (x-param, y-param) contour cell: the axis metadata, the sparse grid <c>z</c> values keyed
///     by (x-index, y-index), and per-trial feasibility. Port of <c>_SubContourInfo</c>.
/// </summary>
/// <param name="XAxis">The metadata describing the horizontal parameter axis.</param>
/// <param name="YAxis">The metadata describing the vertical parameter axis.</param>
/// <param name="ZValues">Target values keyed by the corresponding pair of axis-position indices.</param>
/// <param name="Constraints">One flag per trial; true means every recorded constraint is satisfied.</param>
public sealed record SubContourInfo(
    ContourAxisInfo XAxis,
    ContourAxisInfo YAxis,
    IReadOnlyDictionary<(int X, int Y), double> ZValues,
    IReadOnlyList<bool> Constraints);

/// <summary>
///     The data backing a contour plot: the sorted parameters and the (possibly N×N) grid of
///     subplots. Port of <c>_ContourInfo</c>.
/// </summary>
/// <param name="SortedParams">The parameter names in the deterministic order used to lay out subplots.</param>
/// <param name="SubPlotInfos">The row/column matrix of pairwise parameter subplot data.</param>
/// <param name="ReverseScale">Whether the renderer should reverse its objective-value color scale.</param>
/// <param name="TargetName">The display label for the selected objective or custom target.</param>
public sealed record ContourInfo(
    IReadOnlyList<string> SortedParams,
    IReadOnlyList<IReadOnlyList<SubContourInfo>> SubPlotInfos,
    bool ReverseScale,
    string TargetName);

/// <summary>
///     Data layer for <c>optuna.visualization.plot_contour</c> — the pure <c>_get_contour_info</c>/
///     <c>_get_contour_subplot_info</c>/<c>_get_axis_info</c> half of <c>_contour.py</c>.
/// </summary>
public static class Contour
{
    private static readonly Logger Logger = OptunaLogging.GetLogger("optuna.visualization._contour");

    private const double PaddingRatio = 0.05;

    /// <summary>Python <c>_get_contour_info(study, params, target, target_name)</c>.</summary>
    /// <param name="study">The study supplying trial history, objective directions, and storage.</param>
    /// <param name="params">Parameter names to include; null selects the parameters supported by this operation.</param>
    /// <param name="target">A selector for a scalar trial value; null uses the single objective value.</param>
    /// <param name="targetName">The display label for the selected objective or custom target.</param>
    /// <returns>The subplot matrix, axis metadata, and sparse target grid for the selected parameters.</returns>
    /// <exception cref="NumSharp.ValueError">A requested parameter is absent from the plotted completed trials, or a multi-objective study has no scalar target selector.</exception>
    public static ContourInfo GetInfo(
        Study study,
        IReadOnlyList<string>? @params = null,
        Func<FrozenTrial, double>? target = null,
        string targetName = "Objective Value")
    {
        VisualizationUtils.CheckPlotArgs(study, target, targetName);

        var trials = VisualizationUtils.FilterNonfinite(
            study.GetTrials(deepcopy: false, states: new[] { TrialState.Complete }), target: target);

        var allParams = new HashSet<string>(trials.SelectMany(t => t.Params.Keys));
        List<string> sortedParams;
        if (trials.Count == 0)
        {
            Logger.Warning("Your study does not have any completed trials.");
            sortedParams = new List<string>();
        }
        else if (@params is null)
        {
            sortedParams = allParams.OrderBy(p => p, StringComparer.Ordinal).ToList();
        }
        else
        {
            if (@params.Count <= 1)
                Logger.Warning("The length of params must be greater than 1.");
            foreach (var inputName in @params)
                if (!allParams.Contains(inputName))
                    throw new ValueError($"Parameter {inputName} does not exist in your study.");
            sortedParams = @params.Distinct().OrderBy(p => p, StringComparer.Ordinal).ToList();
        }

        List<IReadOnlyList<SubContourInfo>> subPlotInfos;
        if (sortedParams.Count == 2)
        {
            var sub = GetContourSubplotInfo(study, trials, sortedParams[0], sortedParams[1], target);
            subPlotInfos = new List<IReadOnlyList<SubContourInfo>> { new List<SubContourInfo> { sub } };
        }
        else
        {
            subPlotInfos = new List<IReadOnlyList<SubContourInfo>>();
            foreach (var yParam in sortedParams)
            {
                var row = new List<SubContourInfo>();
                foreach (var xParam in sortedParams)
                    row.Add(GetContourSubplotInfo(study, trials, xParam, yParam, target));
                subPlotInfos.Add(row);
            }
        }

        return new ContourInfo(
            sortedParams,
            subPlotInfos,
            VisualizationUtils.IsReverseScale(study, target),
            targetName);
    }

    /// <summary>Python <c>_get_contour_subplot_info(study, trials, x_param, y_param, target)</c>.</summary>
    /// <param name="study">The study supplying trial history, objective directions, and storage.</param>
    /// <param name="trials">Trial snapshots to inspect; their order is preserved unless the operation explicitly ranks them.</param>
    /// <param name="xParam">The parameter assigned to the horizontal axis.</param>
    /// <param name="yParam">The parameter assigned to the vertical axis.</param>
    /// <param name="target">A selector for a scalar trial value; null uses the single objective value.</param>
    /// <returns>The two axis descriptions, sparse target grid, and per-trial feasibility flags.</returns>
    private static SubContourInfo GetContourSubplotInfo(
        Study study,
        IReadOnlyList<FrozenTrial> trials,
        string xParam,
        string yParam,
        Func<FrozenTrial, double>? target)
    {
        var xaxis = GetAxisInfo(trials, xParam);
        var yaxis = GetAxisInfo(trials, yParam);

        var empty = new Dictionary<(int, int), double>();
        if (xParam == yParam)
            return new SubContourInfo(xaxis, yaxis, empty, Array.Empty<bool>());
        if (xaxis.Indices.Count < 2)
        {
            Logger.Warning($"Param {xParam} unique value length is less than 2.");
            return new SubContourInfo(xaxis, yaxis, empty, Array.Empty<bool>());
        }
        if (yaxis.Indices.Count < 2)
        {
            Logger.Warning($"Param {yParam} unique value length is less than 2.");
            return new SubContourInfo(xaxis, yaxis, empty, Array.Empty<bool>());
        }

        var xIndexOf = BuildIndexLookup(xaxis.Indices);
        var yIndexOf = BuildIndexLookup(yaxis.Indices);

        var zValues = new Dictionary<(int X, int Y), double>();
        for (int i = 0; i < trials.Count; i++)
        {
            var trial = trials[i];
            if (!trial.Params.ContainsKey(xParam) || !trial.Params.ContainsKey(yParam))
                continue;
            object xValue = xaxis.Values[i]!;
            object yValue = yaxis.Values[i]!;
            int xi = xIndexOf[xValue];
            int yi = yIndexOf[yValue];

            double value = target is null ? trial.Value!.Value : target(trial);

            if (!zValues.TryGetValue((xi, yi), out double existing) || target is not null)
            {
                // When target function is present, we can't be sure what the z-value
                // represents and therefore we don't know how to select the best one.
                zValues[(xi, yi)] = value;
            }
            else
            {
                zValues[(xi, yi)] = study.Direction == StudyDirection.Minimize
                    ? Math.Min(existing, value)
                    : Math.Max(existing, value);
            }
        }

        var constraints = trials.Select(SatisfyConstraints).ToList();
        return new SubContourInfo(xaxis, yaxis, zValues, constraints);
    }

    /// <summary>Returns true when every recorded trial constraint is nonpositive.</summary>
    /// <param name="trial">The trial whose recorded state and values are used.</param>
    /// <returns>True when every recorded trial constraint is nonpositive.</returns>
    private static bool SatisfyConstraints(FrozenTrial trial)
        => trial.Constraints.Values.All(x => x <= 0.0);

    /// <summary>Python <c>_get_axis_info(trials, param_name)</c>.</summary>
    /// <param name="trials">Trial snapshots to inspect; their order is preserved unless the operation explicitly ranks them.</param>
    /// <param name="paramName">The parameter name identifying the distribution or values to process.</param>
    /// <returns>The axis bounds, scale, categorical positions, and per-trial values.</returns>
    private static ContourAxisInfo GetAxisInfo(IReadOnlyList<FrozenTrial> trials, string paramName)
    {
        bool isNumerical = VisualizationUtils.IsNumerical(trials, paramName);

        var values = new List<object?>();
        foreach (var t in trials)
        {
            bool has = t.Params.TryGetValue(paramName, out var v);
            if (isNumerical)
                values.Add(has ? v : null);
            else
                values.Add(has ? PythonRepr.Str(v) : null);
        }

        double minValue = 0.0;
        double maxValue = 0.0;
        bool isLog;
        bool isCat;

        if (VisualizationUtils.IsLogScale(trials, paramName))
        {
            (minValue, maxValue) = NumericMinMax(values);
            double padding = (Math.Log10(maxValue) - Math.Log10(minValue)) * PaddingRatio;
            minValue = Math.Pow(10, Math.Log10(minValue) - padding);
            maxValue = Math.Pow(10, Math.Log10(maxValue) + padding);
            isLog = true;
            isCat = false;
        }
        else if (isNumerical)
        {
            (minValue, maxValue) = NumericMinMax(values);
            double padding = (maxValue - minValue) * PaddingRatio;
            minValue -= padding;
            maxValue += padding;
            isLog = false;
            isCat = false;
        }
        else
        {
            // Categorical: the span is over the distinct labels (excluding the missing-value None).
            var unique = new HashSet<object?>(values);
            int span = unique.Count - 1;
            if (unique.Contains(null))
                span -= 1;
            double padding = span * PaddingRatio;
            minValue = -padding;
            maxValue = span + padding;
            isLog = false;
            isCat = true;
        }

        var indices = SortedUnique(values, isNumerical);

        if (indices.Count < 2)
            return new ContourAxisInfo(paramName, (minValue, maxValue), isLog, isCat, indices, values);

        if (isNumerical)
        {
            indices.Insert(0, minValue);
            indices.Add(maxValue);
        }

        return new ContourAxisInfo(paramName, (minValue, maxValue), isLog, isCat, indices, values);
    }

    /// <summary>Returns the smallest and largest numeric value in the supplied sequence.</summary>
    /// <param name="values">The ordered values to aggregate or format.</param>
    /// <returns>The smallest and largest numeric value in the supplied sequence.</returns>
    private static (double Min, double Max) NumericMinMax(IReadOnlyList<object?> values)
    {
        double min = double.PositiveInfinity, max = double.NegativeInfinity;
        foreach (var v in values)
        {
            if (v is null)
                continue;
            double d = Convert.ToDouble(v, System.Globalization.CultureInfo.InvariantCulture);
            if (d < min) min = d;
            if (d > max) max = d;
        }
        return (min, max);
    }

    /// <summary>Python <c>sorted(set([v for v in values if v is not None]))</c>.</summary>
    /// <param name="values">The ordered values to aggregate or format.</param>
    /// <param name="isNumerical">Whether axis values should be sorted and compared numerically.</param>
    /// <returns>The distinct axis values in numeric or ordinal display order.</returns>
    private static List<object> SortedUnique(IReadOnlyList<object?> values, bool isNumerical)
    {
        var distinct = values.Where(v => v is not null).Select(v => v!).Distinct().ToList();
        if (isNumerical)
            distinct = distinct
                .OrderBy(v => Convert.ToDouble(v, System.Globalization.CultureInfo.InvariantCulture))
                .ToList();
        else
            distinct = distinct.OrderBy(v => (string)v, StringComparer.Ordinal).ToList();
        return distinct;
    }

    /// <summary>Map each index entry to its position (Python <c>list.index</c>), first-occurrence.</summary>
    /// <param name="indices">The ordered positions or sample-row indices to use.</param>
    /// <returns>A mapping from each axis position to its index in the ordered axis grid.</returns>
    private static Dictionary<object, int> BuildIndexLookup(IReadOnlyList<object> indices)
    {
        var map = new Dictionary<object, int>();
        for (int i = 0; i < indices.Count; i++)
            if (!map.ContainsKey(indices[i]))
                map[indices[i]] = i;
        return map;
    }
}
