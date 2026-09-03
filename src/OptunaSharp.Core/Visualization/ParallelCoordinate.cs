using System;
using System.Collections.Generic;
using System.Linq;
using NumSharp;
using OptunaSharp.Distributions;
using OptunaSharp.Internal;
using OptunaSharp.Logging;

namespace OptunaSharp.Visualization;

/// <summary>
///     One axis (objective or parameter) of a parallel-coordinate plot. Port of <c>_DimensionInfo</c>
///     in <c>optuna/visualization/_parallel_coordinate.py</c>. <see cref="Values" /> is the per-trial
///     numeric position on this axis (categorical params are mapped to integer codes; missing values
///     get a synthetic below-range position).
/// </summary>
public sealed record DimensionInfo(
    string Label,
    IReadOnlyList<double> Values,
    (double Min, double Max) Range,
    bool IsLog,
    bool IsCat,
    IReadOnlyList<double> Tickvals,
    IReadOnlyList<string> Ticktext,
    bool HasMissing);

/// <summary>
///     The data backing a parallel-coordinate plot. Port of <c>_ParallelCoordinateInfo</c>. For a
///     multi-objective study (<c>target</c> = null) the lines are coloured by Pareto rank
///     (<see cref="IsRankColor" />).
/// </summary>
public sealed record ParallelCoordinateInfo(
    IReadOnlyList<DimensionInfo> DimsObjectives,
    IReadOnlyList<DimensionInfo> DimsParams,
    bool ReverseScale,
    string ColorbarTitle,
    IReadOnlyList<double> ColorValues,
    bool IsRankColor,
    IReadOnlyList<int> DrawOrder,
    IReadOnlyList<bool>? Feasibility);

/// <summary>
///     Data layer for <c>optuna.visualization.plot_parallel_coordinate</c> — the pure
///     <c>_get_parallel_coordinate_info</c> half of <c>_parallel_coordinate.py</c>.
/// </summary>
public static class ParallelCoordinate
{
    private static readonly Logger Logger =
        OptunaLogging.GetLogger("optuna.visualization._parallel_coordinate");

    private const string MissingLabel = "None";
    private static readonly object Missing = new();

    /// <summary>Python <c>_get_parallel_coordinate_info(study, params, target, target_name)</c>.</summary>
    public static ParallelCoordinateInfo GetInfo(
        Study study,
        IReadOnlyList<string>? @params = null,
        Func<FrozenTrial, double>? target = null,
        string targetName = "Objective Value")
    {
        var customTarget = target;
        bool hasCustomTarget = target is not null;
        bool isMultiObjective = target is null && study.IsMultiObjective();
        if (!isMultiObjective)
            VisualizationUtils.CheckPlotArgs(study, target, targetName);

        var trials = study.GetTrials(deepcopy: false, states: new[] { TrialState.Complete });
        if (isMultiObjective)
            trials = trials
                .Where(t => t.Values is not null
                            && t.Values.Count == study.Directions.Count
                            && t.Values.All(double.IsFinite))
                .ToList();
        else
            trials = VisualizationUtils.FilterNonfinite(trials, target: target);

        var allParams = new HashSet<string>(trials.SelectMany(t => t.Params.Keys));
        if (@params is not null)
        {
            foreach (var inputName in @params)
                if (!allParams.Contains(inputName))
                    throw new ValueError($"Parameter {inputName} does not exist in your study.");
            allParams = new HashSet<string>(@params);
        }
        var sortedParams = allParams.OrderBy(p => p, StringComparer.Ordinal).ToList();

        List<DimensionInfo> objectiveDims;
        double[] colorValues;
        int[] drawOrder;
        string colorbarTitle;
        bool reverseScale;

        if (isMultiObjective)
        {
            var directions = study.Directions;
            var metricNames = study.MetricNames;
            var objectiveLabels = metricNames
                ?? Enumerable.Range(0, directions.Count).Select(i => $"Objective {i}").ToList();

            objectiveDims = objectiveLabels
                .Select((label, objectiveId) => BuildNumericalDimension(
                    label, trials.Select(t => t.Values![objectiveId]).ToList()))
                .ToList();

            var lossValues = trials
                .Select(t => t.Values!
                    .Select((v, i) => directions[i] == StudyDirection.Maximize ? -v : v)
                    .ToArray())
                .ToArray();
            int[] ranks = MultiObjective.FastNonDominationRank(lossValues);
            colorValues = ranks.Select(r => (double)r).ToArray();
            drawOrder = ArgsortDescending(ranks);  // np.argsort(ranks)[::-1] — see the [Misaligned] note.
            colorbarTitle = "Pareto Rank";
            reverseScale = true;
        }
        else
        {
            var effectiveTarget = target ?? (t => t.Value ?? double.NaN);
            var objectives = trials.Select(effectiveTarget).ToList();
            objectiveDims = new List<DimensionInfo> { BuildNumericalDimension(targetName, objectives) };
            colorValues = objectives.ToArray();
            drawOrder = Enumerable.Range(0, trials.Count).ToArray();
            colorbarTitle = targetName;
            reverseScale = VisualizationUtils.IsReverseScale(study, customTarget);
        }

        var dimsObjectives = objectiveDims.Count > 0
            ? objectiveDims
            : new List<DimensionInfo> { BuildNumericalDimension(targetName, new List<double>()) };

        if (trials.Count == 0)
        {
            Logger.Warning("Your study does not have any completed trials.");
            return new ParallelCoordinateInfo(
                dimsObjectives, new List<DimensionInfo>(), reverseScale, colorbarTitle,
                colorValues, isMultiObjective, drawOrder, null);
        }

        var feasibility = GetFeasibility(trials);

        var numericCatParamsIndices = new List<int>();
        var dims = new List<DimensionInfo>();
        for (int position = 0; position < sortedParams.Count; position++)
        {
            int dimIndex = dimsObjectives.Count + position;
            string pName = sortedParams[position];
            var values = trials.Select(t => t.Params.TryGetValue(pName, out var v) ? v : Missing).ToList();
            bool isCategorical = trials.Any(
                t => t.Params.ContainsKey(pName) && t.Distributions[pName] is CategoricalDistribution);
            bool hasMissing = values.Any(v => ReferenceEquals(v, Missing));

            DimensionInfo dim;
            if (VisualizationUtils.IsLogScale(trials, pName))
                dim = BuildLogDimension(pName, values, hasMissing);
            else if (isCategorical)
                dim = BuildCategoricalDimension(trials, pName, values, hasMissing, dimIndex, numericCatParamsIndices);
            else
                dim = BuildNumericalParamDimension(pName, values, hasMissing);

            dims.Add(dim);
        }

        var plotDims = dims;
        if (numericCatParamsIndices.Count > 0)
        {
            var combined = dimsObjectives.Concat(plotDims).ToList();
            int[] idx = LexsortByNumericCatParams(combined, numericCatParamsIndices);

            var updated = combined
                .Select(d => d with { Values = idx.Select(i => d.Values[i]).ToList() })
                .ToList();
            dimsObjectives = updated.Take(dimsObjectives.Count).ToList();
            plotDims = updated.Skip(dimsObjectives.Count).ToList();
            colorValues = idx.Select(i => colorValues[i]).ToArray();
            if (feasibility is not null)
                feasibility = idx.Select(i => feasibility[i]).ToList();
            if (isMultiObjective)
            {
                var cv = colorValues;
                drawOrder = Enumerable.Range(0, trials.Count)
                    .OrderByDescending(i => cv[i])  // stable descending, like Python sorted(reverse=True).
                    .ToArray();
            }
        }

        return new ParallelCoordinateInfo(
            dimsObjectives, plotDims, reverseScale, colorbarTitle,
            colorValues, isMultiObjective, drawOrder, feasibility);
    }

    // -------------------------------------------------------------------- dimension builders

    private static DimensionInfo BuildLogDimension(string pName, List<object?> values, bool hasMissing)
    {
        var validValues = values.Where(v => !ReferenceEquals(v, Missing)).Select(ToDouble).ToList();
        var logValid = validValues.Select(Math.Log10).ToList();
        double minValue = logValid.Min();
        double maxValue = logValid.Max();

        var tickvals = new List<double>();
        for (int t = (int)Math.Ceiling(minValue); t <= (int)Math.Floor(maxValue); t++)
            tickvals.Add(t);
        if (!tickvals.Contains(minValue))
            tickvals.Insert(0, minValue);
        if (!tickvals.Contains(maxValue))
            tickvals.Add(maxValue);

        double? missingValue = hasMissing ? GetMissingValue(minValue, maxValue) : null;
        var transformed = values
            .Select(v => ReferenceEquals(v, Missing) ? missingValue!.Value : Math.Log10(ToDouble(v)))
            .ToList();
        if (missingValue is not null)
            tickvals.Insert(0, missingValue.Value);

        var ticktext = (hasMissing ? new List<string> { MissingLabel } : new List<string>())
            .Concat(tickvals.Skip(hasMissing ? 1 : 0).Select(x => MathUtil.FormatG3(Math.Pow(10, x))))
            .ToList();

        return new DimensionInfo(
            TruncateLabel(pName), transformed,
            (missingValue ?? minValue, maxValue), true, false, tickvals, ticktext, hasMissing);
    }

    private static DimensionInfo BuildCategoricalDimension(
        IReadOnlyList<FrozenTrial> trials, string pName, List<object?> values, bool hasMissing,
        int dimIndex, List<int> numericCatParamsIndices)
    {
        var validValues = values.Where(v => !ReferenceEquals(v, Missing)).ToList();
        var vocab = new Dictionary<object?, int>();
        List<string> ticktext;

        if (VisualizationUtils.IsNumerical(trials, pName))
        {
            foreach (var v in validValues.OrderBy(ToDouble))
                VocabId(vocab, v);
            ticktext = vocab.Keys.OrderBy(ToDouble).Select(PythonRepr.Str).ToList();
            numericCatParamsIndices.Add(dimIndex);
        }
        else
        {
            foreach (var v in validValues)
                VocabId(vocab, v);
            ticktext = vocab.Keys.OrderBy(k => vocab[k]).Select(PythonRepr.Str).ToList();
        }

        int offset = hasMissing ? 1 : 0;
        var categoricalValues = values
            .Select(v => ReferenceEquals(v, Missing) ? 0.0 : vocab[v] + offset)
            .ToList();
        if (hasMissing)
            ticktext = ticktext.Select(text => text == MissingLabel ? $"{text} (value)" : text).ToList();

        var tickvals = Enumerable.Range(0, vocab.Count + offset).Select(i => (double)i).ToList();
        var fullTicktext = (hasMissing ? new List<string> { MissingLabel } : new List<string>())
            .Concat(ticktext).ToList();

        return new DimensionInfo(
            TruncateLabel(pName), categoricalValues,
            (categoricalValues.Min(), categoricalValues.Max()), false, true, tickvals, fullTicktext, hasMissing);
    }

    private static DimensionInfo BuildNumericalParamDimension(string pName, List<object?> values, bool hasMissing)
    {
        var validValues = values.Where(v => !ReferenceEquals(v, Missing)).Select(ToDouble).ToList();
        double minValue = validValues.Min();
        double maxValue = validValues.Max();
        double? missingValue = hasMissing ? GetMissingValue(minValue, maxValue) : null;

        var numericalValues = values
            .Select(v => ReferenceEquals(v, Missing) ? missingValue!.Value : ToDouble(v))
            .ToList();

        var validTicks = minValue == maxValue
            ? new List<double> { minValue }
            : MathUtil.Linspace(minValue, maxValue, 5).ToList();

        var tickvals = missingValue is not null
            ? new List<double> { missingValue.Value }.Concat(validTicks).ToList()
            : new List<double>();
        var ticktext = hasMissing
            ? new List<string> { MissingLabel }.Concat(validTicks.Select(MathUtil.FormatG3)).ToList()
            : new List<string>();

        return new DimensionInfo(
            TruncateLabel(pName), numericalValues,
            (missingValue ?? minValue, maxValue), false, false, tickvals, ticktext, hasMissing);
    }

    // -------------------------------------------------------------------- helpers

    private static DimensionInfo BuildNumericalDimension(string label, List<double> values)
    {
        var range = values.Count > 0 ? (values.Min(), values.Max()) : (0.0, 0.0);
        return new DimensionInfo(label, values, range, false, false, new List<double>(), new List<string>(), false);
    }

    private static IReadOnlyList<bool>? GetFeasibility(IReadOnlyList<FrozenTrial> trials)
    {
        if (!trials.Any(t => t.Constraints.Count > 0))
            return null;
        return trials
            .Select(t => t.Constraints.Count > 0 && t.Constraints.Values.All(v => v <= 0.0))
            .ToList();
    }

    private static double GetMissingValue(double minValue, double maxValue)
        => minValue == maxValue ? minValue - 1.0 : minValue - (maxValue - minValue) * 0.12 / 0.88;

    private static string TruncateLabel(string label)
        => label.Length < 20 ? label : $"{label.Substring(0, 17)}...";

    private static int VocabId(Dictionary<object?, int> vocab, object? key)
    {
        if (!vocab.TryGetValue(key, out int id))
        {
            id = vocab.Count;
            vocab[key] = id;
        }
        return id;
    }

    private static double ToDouble(object? v)
        => Convert.ToDouble(v, System.Globalization.CultureInfo.InvariantCulture);

    /// <summary>
    ///     NumPy <c>np.argsort(ranks)[::-1]</c>. Ascending stable argsort, then reversed. NumPy's
    ///     argsort is an unstable quicksort, so for tied ranks the tie order can differ — a documented
    ///     divergence, cosmetic (it only reorders which lines are drawn first).
    /// </summary>
    private static int[] ArgsortDescending(int[] ranks)
        => Enumerable.Range(0, ranks.Length)
            .OrderBy(i => ranks[i])
            .Reverse()
            .ToArray();

    /// <summary>
    ///     NumPy <c>np.lexsort([keys][::-1])</c> over the numeric-categorical param columns: a stable
    ///     multi-key ascending sort of the trial indices, primary key = the first numeric-cat param.
    /// </summary>
    private static int[] LexsortByNumericCatParams(List<DimensionInfo> combined, List<int> numericCatIndices)
    {
        IOrderedEnumerable<int> ordered = Enumerable.Range(0, combined[numericCatIndices[0]].Values.Count)
            .OrderBy(i => combined[numericCatIndices[0]].Values[i]);
        for (int k = 1; k < numericCatIndices.Count; k++)
        {
            int col = numericCatIndices[k];
            ordered = ordered.ThenBy(i => combined[col].Values[i]);
        }
        return ordered.ToArray();
    }
}
