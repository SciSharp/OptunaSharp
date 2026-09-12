using System;
using System.Collections.Generic;
using System.Linq;
using OptunaSharp.Internal;
using OptunaSharp.Logging;

namespace OptunaSharp.Visualization;

/// <summary>
///     Feasibility/completeness state of a trial in the optimization-history data. Port of
///     <c>_ValueState</c> in <c>optuna/visualization/_optimization_history.py</c>.
/// </summary>
public enum ValueState
{
    /// <summary>The trial completed with every recorded constraint satisfied.</summary>
    Feasible = 0,
    /// <summary>The trial completed with a positive constraint violation.</summary>
    Infeasible = 1,
    /// <summary>The trial does not supply a completed objective value for the history curve.</summary>
    Incomplete = 2,
}

/// <summary>
///     One series of values (with optional error-bar standard deviations). Port of
///     <c>_ValuesInfo</c>.
/// </summary>
/// <param name="Values">The per-trial values in the same order as the plot's trial sequence.</param>
/// <param name="Stds">The standard deviation at each plotted position, or null when error bars are absent.</param>
/// <param name="LabelName">The legend label identifying this objective-value sequence.</param>
/// <param name="States">The feasibility/completion classification aligned with the plotted values.</param>
public sealed record ValuesInfo(
    IReadOnlyList<double> Values,
    IReadOnlyList<double>? Stds,
    string LabelName,
    IReadOnlyList<ValueState> States);

/// <summary>
///     The data backing one study's optimization-history plot. Port of
///     <c>_OptimizationHistoryInfo</c>.
/// </summary>
/// <param name="TrialNumbers">The study-local trial numbers aligned with the value or estimate sequences.</param>
/// <param name="ValuesInfo">The per-trial objective values, labels, and feasibility states.</param>
/// <param name="BestValuesInfo">The running best-value curve, or null when that curve is not available.</param>
public sealed record OptimizationHistoryInfo(
    IReadOnlyList<int> TrialNumbers,
    ValuesInfo ValuesInfo,
    ValuesInfo? BestValuesInfo);

/// <summary>
///     Data layer for <c>optuna.visualization.plot_optimization_history</c> — the pure
///     <c>_get_optimization_history_info_list</c> half of <c>_optimization_history.py</c>.
/// </summary>
public static class OptimizationHistory
{
    private static readonly Logger Logger =
        OptunaLogging.GetLogger("optuna.visualization._optimization_history");

    /// <inheritdoc cref="GetInfoList(System.Collections.Generic.IReadOnlyList{Study}, System.Func{FrozenTrial, double}, string, bool)" />
    /// <param name="study">The study whose trial values form the history.</param>
    /// <param name="target">A scalar target selector; null uses the study's single objective.</param>
    /// <param name="targetName">The label for the objective-value curve.</param>
    /// <param name="errorBar">Whether to use the history aggregation path that also computes standard deviations.</param>
    /// <returns>The history descriptors built for this study.</returns>
    /// <exception cref="NumSharp.ValueError">A multi-objective study is supplied without a scalar target selector.</exception>
    public static List<OptimizationHistoryInfo> GetInfoList(
        Study study,
        Func<FrozenTrial, double>? target = null,
        string targetName = "Objective Value",
        bool errorBar = false)
        => GetInfoList(new[] { study }, target, targetName, errorBar);

    /// <summary>Python <c>_get_optimization_history_info_list(study, target, target_name, error_bar)</c>.</summary>
    /// <param name="studies">The studies to compare, in the order their data should appear.</param>
    /// <param name="target">A selector for a scalar trial value; null uses the single objective value.</param>
    /// <param name="targetName">The display label for the selected objective or custom target.</param>
    /// <param name="errorBar">Whether histories are aggregated across studies with mean and standard-deviation error bars.</param>
    /// <returns>The per-study objective histories and available running-best curves, optionally aggregated with error bars.</returns>
    /// <exception cref="NumSharp.ValueError">A multi-objective study is supplied without a scalar target selector.</exception>
    public static List<OptimizationHistoryInfo> GetInfoList(
        IReadOnlyList<Study> studies,
        Func<FrozenTrial, double>? target = null,
        string targetName = "Objective Value",
        bool errorBar = false)
    {
        VisualizationUtils.CheckPlotArgs(studies, target, targetName);

        var infoList = new List<OptimizationHistoryInfo>();
        foreach (var study in studies)
        {
            var trials = study.GetTrials(deepcopy: false);
            string labelName = studies.Count == 1 ? targetName : $"{targetName} of {study.StudyName}";
            var values = new List<double>();
            var valueStates = new List<ValueState>();
            foreach (var trial in trials)
            {
                if (trial.State != TrialState.Complete)
                {
                    values.Add(double.NaN);
                    valueStates.Add(ValueState.Incomplete);
                    continue;
                }

                valueStates.Add(trial.Constraints.Values.All(x => x <= 0.0)
                    ? ValueState.Feasible
                    : ValueState.Infeasible);
                values.Add(target is not null ? target(trial) : trial.Value ?? double.NaN);
            }

            ValuesInfo? bestValuesInfo;
            if (target is not null)
            {
                // We don't calculate best for user-defined target function since we cannot tell
                // which direction is better.
                bestValuesInfo = null;
            }
            else
            {
                List<double> bestValues;
                if (study.Direction == StudyDirection.Minimize)
                {
                    var feasibleBestValues = values.Zip(valueStates,
                        (v, s) => s == ValueState.Feasible ? v : double.PositiveInfinity);
                    bestValues = MinimumAccumulate(feasibleBestValues);
                }
                else
                {
                    var feasibleBestValues = values.Zip(valueStates,
                        (v, s) => s == ValueState.Feasible ? v : double.NegativeInfinity);
                    bestValues = MaximumAccumulate(feasibleBestValues);
                }

                string bestLabelName = studies.Count == 1 ? "Best Value" : $"Best Value of {study.StudyName}";
                bestValuesInfo = new ValuesInfo(bestValues, null, bestLabelName, valueStates);
            }

            infoList.Add(new OptimizationHistoryInfo(
                trials.Select(t => t.Number).ToList(),
                new ValuesInfo(values, null, labelName, valueStates),
                bestValuesInfo));
        }

        if (infoList.Count == 0)
            Logger.Warning("There are no studies.");

        int feasibleTrialCount = infoList.Sum(info => info.ValuesInfo.States.Count(s => s == ValueState.Feasible));
        int infeasibleTrialCount = infoList.Sum(info => info.ValuesInfo.States.Count(s => s == ValueState.Infeasible));
        if (feasibleTrialCount + infeasibleTrialCount == 0)
        {
            Logger.Warning("There are no complete trials.");
            infoList.Clear();
        }

        if (!errorBar)
            return infoList;

        // When error_bar=True, a list of 0 or 1 element is returned.
        if (infoList.Count == 0)
            return new List<OptimizationHistoryInfo>();
        if (feasibleTrialCount == 0)
        {
            Logger.Warning("There are no feasible trials.");
            return new List<OptimizationHistoryInfo>();
        }

        int maxNumTrial = infoList.SelectMany(info => info.TrialNumbers).Max() + 1;

        (List<int> TrialNumbers, ValuesInfo ValuesInfo) Aggregate(string label, bool useBestValue)
        {
            var values = new List<double>[maxNumTrial];
            var states = new List<ValueState>[maxNumTrial];
            for (int i = 0; i < maxNumTrial; i++)
            {
                values[i] = new List<double>();
                states[i] = new List<ValueState>();
            }

            foreach (var info in infoList)
            {
                var valuesInfo = useBestValue ? info.BestValuesInfo! : info.ValuesInfo;
                foreach (var (n, v, s) in info.TrialNumbers
                             .Zip(valuesInfo.Values, (n, v) => (n, v))
                             .Zip(valuesInfo.States, (nv, s) => (nv.n, nv.v, s)))
                {
                    if (!double.IsInfinity(v))
                    {
                        if (!useBestValue && s == ValueState.Feasible)
                            values[n].Add(v);
                        else if (useBestValue)
                            values[n].Add(v);
                    }
                    states[n].Add(s);
                }
            }

            var trialNumbersUnion = new List<int>();
            var valueStates = new List<ValueState>();
            var valueMeans = new List<double>();
            var valueStds = new List<double>();
            for (int i = 0; i < maxNumTrial; i++)
            {
                if (states[i].Count > 0 && states[i].Contains(ValueState.Feasible))
                {
                    valueStates.Add(ValueState.Feasible);
                    trialNumbersUnion.Add(i);
                    valueMeans.Add(MathUtil.Mean(values[i].ToArray()));
                    valueStds.Add(MathUtil.Std(values[i].ToArray()));
                }
                else
                {
                    valueStates.Add(ValueState.Infeasible);
                }
            }

            return (trialNumbersUnion, new ValuesInfo(valueMeans, valueStds, label, valueStates));
        }

        var (ebTrialNumbers, ebValuesInfo) = Aggregate(targetName, useBestValue: false);
        ValuesInfo? ebBestValuesInfo = null;
        if (target is null)
            ebBestValuesInfo = Aggregate("Best Value", useBestValue: true).ValuesInfo;

        return new List<OptimizationHistoryInfo>
        {
            new(ebTrialNumbers, ebValuesInfo, ebBestValuesInfo),
        };
    }

    /// <summary>NumPy <c>np.minimum.accumulate</c> — running minimum (no NaN in the inputs here).</summary>
    /// <param name="values">The ordered values to aggregate or format.</param>
    /// <returns>The running minimum of the input values.</returns>
    private static List<double> MinimumAccumulate(IEnumerable<double> values)
    {
        var result = new List<double>();
        double acc = double.PositiveInfinity;
        bool first = true;
        foreach (double v in values)
        {
            acc = first ? v : Math.Min(acc, v);
            first = false;
            result.Add(acc);
        }
        return result;
    }

    /// <summary>NumPy <c>np.maximum.accumulate</c> — running maximum (no NaN in the inputs here).</summary>
    /// <param name="values">The ordered values to aggregate or format.</param>
    /// <returns>The running maximum of the input values.</returns>
    private static List<double> MaximumAccumulate(IEnumerable<double> values)
    {
        var result = new List<double>();
        double acc = double.NegativeInfinity;
        bool first = true;
        foreach (double v in values)
        {
            acc = first ? v : Math.Max(acc, v);
            first = false;
            result.Add(acc);
        }
        return result;
    }
}
