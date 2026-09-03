using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using OptunaSharp.Importance;
using OptunaSharp.Logging;

namespace OptunaSharp.Visualization;

/// <summary>
///     The data backing one objective's hyperparameter-importance plot (ascending by importance, for
///     a bottom-to-top bar chart). Port of <c>_ImportancesInfo</c> in
///     <c>optuna/visualization/_param_importances.py</c>.
/// </summary>
public sealed record ImportancesInfo(
    IReadOnlyList<double> ImportanceValues,
    IReadOnlyList<string> ParamNames,
    IReadOnlyList<string> ImportanceLabels,
    string TargetName);

/// <summary>
///     Data layer for <c>optuna.visualization.plot_param_importances</c> — the pure
///     <c>_get_importances_info</c>/<c>_get_importances_infos</c> half of <c>_param_importances.py</c>.
///     Rides <see cref="ImportanceFunctions.GetParamImportances" />, so the importance values inherit
///     PED-ANOVA's erf/ucrt few-ULP tolerance (documented on <c>TruncNorm</c>); ordering, names and
///     labels are exact.
/// </summary>
public static class ParamImportances
{
    private static readonly Logger Logger =
        OptunaLogging.GetLogger("optuna.visualization._param_importances");

    /// <summary>Python <c>_get_importances_info(study, evaluator, params, target, target_name)</c>.</summary>
    public static ImportancesInfo GetInfo(
        Study study,
        BaseImportanceEvaluator? evaluator = null,
        IReadOnlyList<string>? @params = null,
        System.Func<FrozenTrial, double>? target = null,
        string targetName = "Objective Value")
    {
        VisualizationUtils.CheckPlotArgs(study, target, targetName);

        var trials = VisualizationUtils.FilterNonfinite(
            study.GetTrials(deepcopy: false, states: new[] { TrialState.Complete }), target: target);

        if (trials.Count == 0)
        {
            Logger.Warning("Study instance does not contain completed trials.");
            return new ImportancesInfo(
                new List<double>(), new List<string>(), new List<string>(), targetName);
        }

        var importances = ImportanceFunctions.GetParamImportances(
            study, evaluator: evaluator, @params: @params, target: target);

        // dict(reversed(list(importances.items()))): descending → ascending for the bar chart.
        var reversed = importances.Reverse().ToList();
        var importanceValues = reversed.Select(kv => kv.Value).ToList();
        var paramNames = reversed.Select(kv => kv.Key).ToList();
        var importanceLabels = importanceValues
            .Select(val => val >= 0.01 ? val.ToString("F2", CultureInfo.InvariantCulture) : "<0.01")
            .ToList();

        return new ImportancesInfo(importanceValues, paramNames, importanceLabels, targetName);
    }

    /// <summary>
    ///     Python <c>_get_importances_infos(study, evaluator, params, target, target_name)</c> — one
    ///     <see cref="ImportancesInfo" /> for single-objective / an explicit <paramref name="target" />,
    ///     or one per objective for a multi-objective study with <paramref name="target" /> = null.
    /// </summary>
    public static List<ImportancesInfo> GetInfos(
        Study study,
        BaseImportanceEvaluator? evaluator = null,
        IReadOnlyList<string>? @params = null,
        System.Func<FrozenTrial, double>? target = null,
        string targetName = "Objective Value")
    {
        var metricNames = study.MetricNames;
        if (target is not null || !study.IsMultiObjective())
        {
            string tn = metricNames is not null && target is null ? metricNames[0] : targetName;
            return new List<ImportancesInfo> { GetInfo(study, evaluator, @params, target, tn) };
        }

        int nObjectives = study.Directions.Count;
        IReadOnlyList<string> targetNames = metricNames
            ?? Enumerable.Range(0, nObjectives).Select(i => $"{targetName} {i}").ToList();

        var infos = new List<ImportancesInfo>();
        for (int objectiveId = 0; objectiveId < targetNames.Count; objectiveId++)
        {
            int oid = objectiveId;
            infos.Add(GetInfo(study, evaluator, @params, t => t.Values![oid], targetNames[objectiveId]));
        }
        return infos;
    }
}
