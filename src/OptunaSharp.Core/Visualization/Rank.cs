using System;
using System.Collections.Generic;
using System.Linq;
using NumSharp;
using OptunaSharp.Logging;

namespace OptunaSharp.Visualization;

/// <summary>
///     One axis of a rank subplot. Port of <c>_AxisInfo</c> in <c>optuna/visualization/_rank.py</c>
///     (the rank variant has no <c>indices</c>/<c>values</c>, unlike the contour one).
/// </summary>
/// <param name="Name">The parameter name identifying this axis.</param>
/// <param name="Range">The padded minimum and maximum positions used as the axis bounds.</param>
/// <param name="IsLog">Whether the axis uses logarithmic scaling.</param>
/// <param name="IsCat">Whether the axis represents discrete categorical labels.</param>
public sealed record RankAxisInfo(
    string Name,
    (double Min, double Max) Range,
    bool IsLog,
    bool IsCat);

/// <summary>
///     One (x-param, y-param) rank scatter. Port of <c>_RankSubplotInfo</c>. Instead of Optuna's
///     plotly-derived RGB <c>colors</c>, the pure-numpy <see cref="ColorIdxs" /> (the scaled ranks in
///     [0, 1], one per plotted trial) and the <see cref="Infeasible" /> mask are exposed; the Tier X
///     rendering package (5.2) maps these onto the <c>RdYlBu_r</c> colorscale (and greys the
///     infeasible ones), which is the plotly dependency Optuna's own TODO wants removed from the data
///     layer.
/// </summary>
/// <param name="XAxis">The metadata describing the horizontal parameter axis.</param>
/// <param name="YAxis">The metadata describing the vertical parameter axis.</param>
/// <param name="Xs">The horizontal parameter value of each plotted trial.</param>
/// <param name="Ys">The vertical parameter value of each plotted trial.</param>
/// <param name="Trials">The trial snapshots aligned with the subplot's coordinate and color arrays.</param>
/// <param name="Zs">The target values aligned with the plotted trials.</param>
/// <param name="ColorIdxs">The average-rank color coordinates normalized to the unit interval.</param>
/// <param name="Infeasible">Flags identifying trials whose constraints are violated.</param>
public sealed record RankSubplotInfo(
    RankAxisInfo XAxis,
    RankAxisInfo YAxis,
    IReadOnlyList<object?> Xs,
    IReadOnlyList<object?> Ys,
    IReadOnlyList<FrozenTrial> Trials,
    double[] Zs,
    double[] ColorIdxs,
    bool[] Infeasible);

/// <summary>
///     The data backing a rank plot. Port of <c>_RankPlotInfo</c> — with the plotly RGB
///     <c>colors</c> replaced by the pure <see cref="ColorIdxs" /> (see <see cref="RankSubplotInfo" />).
/// </summary>
/// <param name="Params">The parameter names in subplot-layout order.</param>
/// <param name="SubPlotInfos">The row/column matrix of pairwise parameter subplot data.</param>
/// <param name="TargetName">The display label for the selected objective or custom target.</param>
/// <param name="Zs">The target values aligned with the plotted trials.</param>
/// <param name="ColorIdxs">The average-rank color coordinates normalized to the unit interval.</param>
/// <param name="HasCustomTarget">Whether target values came from a caller-supplied selector.</param>
public sealed record RankPlotInfo(
    IReadOnlyList<string> Params,
    IReadOnlyList<IReadOnlyList<RankSubplotInfo>> SubPlotInfos,
    string TargetName,
    double[] Zs,
    double[] ColorIdxs,
    bool HasCustomTarget);

/// <summary>
///     Data layer for <c>optuna.visualization.plot_rank</c> — the pure half of <c>_rank.py</c>
///     (<c>_get_rank_info</c>/<c>_get_rank_subplot_info</c>/<c>_get_axis_info</c>/
///     <c>_get_order_with_same_order_averaging</c>). The RGB colour conversion
///     (<c>_convert_color_idxs_to_scaled_rgb_colors</c>, which calls plotly's
///     <c>sample_colorscale</c>) is intentionally NOT ported — it is Tier X rendering (5.2). This
///     builder exposes the colormap input (<c>ColorIdxs</c>) instead.
/// </summary>
public static class Rank
{
    private static readonly Logger Logger = OptunaLogging.GetLogger("optuna.visualization._rank");

    private const double PaddingRatio = 0.05;

    /// <summary>Python <c>_get_rank_info(study, params, target, target_name)</c>.</summary>
    /// <param name="study">The study supplying trial history, objective directions, and storage.</param>
    /// <param name="params">Parameter names to include; null selects the parameters supported by this operation.</param>
    /// <param name="target">A selector for a scalar trial value; null uses the single objective value.</param>
    /// <param name="targetName">The display label for the selected objective or custom target.</param>
    /// <returns>The pairwise parameter subplots, target values, and normalized average-rank color coordinates.</returns>
    /// <exception cref="NumSharp.ValueError">A requested parameter is absent from completed trials, or a multi-objective study has no scalar target selector.</exception>
    public static RankPlotInfo GetInfo(
        Study study,
        IReadOnlyList<string>? @params = null,
        Func<FrozenTrial, double>? target = null,
        string targetName = "Objective Value")
    {
        VisualizationUtils.CheckPlotArgs(study, target, targetName);

        // NOTE: rank does NOT filter non-finite trials (unlike slice/contour).
        var trials = study.GetTrials(deepcopy: false, states: new[] { TrialState.Complete });

        var allParams = new HashSet<string>(trials.SelectMany(t => t.Params.Keys));
        List<string> resolvedParams;
        if (trials.Count == 0)
        {
            Logger.Warning("Your study does not have any completed trials.");
            resolvedParams = new List<string>();
        }
        else if (@params is null)
        {
            resolvedParams = allParams.OrderBy(p => p, StringComparer.Ordinal).ToList();
        }
        else
        {
            foreach (var inputName in @params)
                if (!allParams.Contains(inputName))
                    throw new ValueError($"Parameter {inputName} does not exist in your study.");
            resolvedParams = @params.ToList();  // rank keeps the given order (no sort/dedupe).
        }

        if (resolvedParams.Count == 0)
            Logger.Warning("params is an empty list.");

        bool hasCustomTarget = target is not null;
        var effectiveTarget = target ?? (t => t.Value ?? double.NaN);

        double[] targetValues = trials.Select(effectiveTarget).ToArray();
        double[] rawRanks = OrderWithSameOrderAveraging(targetValues);
        double[] colorIdxs = trials.Count >= 2
            ? rawRanks.Select(r => r / (trials.Count - 1)).ToArray()
            : new[] { 0.5 };

        List<IReadOnlyList<RankSubplotInfo>> subPlotInfos;
        if (resolvedParams.Count == 2)
        {
            var sub = GetRankSubplotInfo(trials, targetValues, colorIdxs, resolvedParams[0], resolvedParams[1]);
            subPlotInfos = new List<IReadOnlyList<RankSubplotInfo>> { new List<RankSubplotInfo> { sub } };
        }
        else
        {
            subPlotInfos = new List<IReadOnlyList<RankSubplotInfo>>();
            foreach (var yParam in resolvedParams)
            {
                var row = new List<RankSubplotInfo>();
                foreach (var xParam in resolvedParams)
                    row.Add(GetRankSubplotInfo(trials, targetValues, colorIdxs, xParam, yParam));
                subPlotInfos.Add(row);
            }
        }

        return new RankPlotInfo(resolvedParams, subPlotInfos, targetName, targetValues, colorIdxs, hasCustomTarget);
    }

    /// <summary>Python <c>_get_rank_subplot_info(trials, target_values, colors, x_param, y_param)</c>.</summary>
    /// <param name="trials">Trial snapshots to inspect; their order is preserved unless the operation explicitly ranks them.</param>
    /// <param name="targetValues">The target value associated with each supplied trial.</param>
    /// <param name="colorIdxs">The normalized rank-color coordinate associated with each trial.</param>
    /// <param name="xParam">The parameter assigned to the horizontal axis.</param>
    /// <param name="yParam">The parameter assigned to the vertical axis.</param>
    /// <returns>The selected parameter pair's coordinates, trials, rank colors, and infeasibility flags.</returns>
    private static RankSubplotInfo GetRankSubplotInfo(
        IReadOnlyList<FrozenTrial> trials,
        double[] targetValues,
        double[] colorIdxs,
        string xParam,
        string yParam)
    {
        var xaxis = GetAxisInfo(trials, xParam);
        var yaxis = GetAxisInfo(trials, yParam);

        var filteredIds = new List<int>();
        for (int idx = 0; idx < trials.Count; idx++)
            if (trials[idx].Params.ContainsKey(xParam) && trials[idx].Params.ContainsKey(yParam))
                filteredIds.Add(idx);

        var filteredTrials = filteredIds.Select(i => trials[i]).ToList();
        var xs = filteredTrials.Select(t => t.Params[xParam]).ToList();
        var ys = filteredTrials.Select(t => t.Params[yParam]).ToList();
        var zs = filteredIds.Select(i => targetValues[i]).ToArray();
        var subColorIdxs = filteredIds.Select(i => colorIdxs[i]).ToArray();
        // Optuna greys the infeasible trials in the RGB colours; here we expose the per-trial flag.
        var infeasible = filteredTrials
            .Select(t => t.Constraints.Values.Any(c => c > 0.0))
            .ToArray();

        return new RankSubplotInfo(xaxis, yaxis, xs, ys, filteredTrials, zs, subColorIdxs, infeasible);
    }

    /// <summary>Python <c>_get_axis_info(trials, param_name)</c> (rank variant, no indices/values).</summary>
    /// <param name="trials">Trial snapshots to inspect; their order is preserved unless the operation explicitly ranks them.</param>
    /// <param name="paramName">The parameter name identifying the distribution or values to process.</param>
    /// <returns>The padded axis bounds and its logarithmic/categorical classification.</returns>
    private static RankAxisInfo GetAxisInfo(IReadOnlyList<FrozenTrial> trials, string paramName)
    {
        bool isNumerical = VisualizationUtils.IsNumerical(trials, paramName);

        var values = new List<object?>();
        foreach (var t in trials)
        {
            bool has = t.Params.TryGetValue(paramName, out var v);
            if (isNumerical)
                values.Add(has ? v : null);
            else
                values.Add(has ? Internal.PythonRepr.Str(v) : null);
        }

        double minValue;
        double maxValue;
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

        return new RankAxisInfo(paramName, (minValue, maxValue), isLog, isCat);
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

    /// <summary>
    ///     Python <c>_get_order_with_same_order_averaging(data)</c>: each element's rank is the mean
    ///     of the positions its value occupies in the sorted array (ties share the averaged rank).
    /// </summary>
    /// <param name="data">The target values whose ranks are averaged across ties.</param>
    /// <returns>The order-based rank of each value, averaged across equal values.</returns>
    private static double[] OrderWithSameOrderAveraging(double[] data)
    {
        var order = new double[data.Length];
        var sorted = (double[])data.Clone();
        Array.Sort(sorted);  // finite/±inf match np.sort; NaN (only from custom targets) is an edge case.
        for (int i = 0; i < data.Length; i++)
        {
            double d = data[i];
            long sum = 0;
            int count = 0;
            for (int j = 0; j < sorted.Length; j++)
                if (sorted[j] == d)
                {
                    sum += j;
                    count++;
                }
            order[i] = (double)sum / count;
        }
        return order;
    }
}
