using System;
using System.Collections.Generic;
using System.Linq;
using NumSharp;
using OptunaSharp.Collections;
using OptunaSharp.Distributions;
using OptunaSharp.Internal;
using OptunaSharp.SearchSpace;

namespace OptunaSharp.Importance;

/// <summary>
///     Abstract parameter importance evaluator. Port of
///     <c>optuna/importance/_base.py: BaseImportanceEvaluator</c>.
/// </summary>
public abstract class BaseImportanceEvaluator
{
    /// <summary>
    ///     Evaluate parameter importances based on completed trials in the given study.
    ///
    ///     <para>This method is not meant to be called by library users. See
    ///     <see cref="ImportanceFunctions.GetParamImportances" /> for how a concrete evaluator
    ///     should implement this method.</para>
    /// </summary>
    /// <param name="study">An optimized study.</param>
    /// <param name="params">
    ///     A list of names of parameters to assess. If <c>null</c>, all parameters that are present
    ///     in all of the completed trials are assessed.
    /// </param>
    /// <param name="target">
    ///     A function to specify the value to evaluate importances. If it is <c>null</c> and
    ///     <paramref name="study" /> is being used for single-objective optimization, the objective
    ///     values are used. Can also be used for other trial attributes, such as the duration, like
    ///     <c>target: t =&gt; t.Duration!.Value.TotalSeconds</c>.
    /// </param>
    /// <returns>
    ///     A dictionary where the keys are parameter names and the values are assessed importances.
    /// </returns>
    public abstract OrderedDict<string, double> Evaluate(
        Study study,
        IReadOnlyList<string>? @params = null,
        Func<FrozenTrial, double>? target = null);
}

/// <summary>
///     Port of the module-level helpers in <c>optuna/importance/_base.py</c>. The PED-ANOVA
///     evaluator (Phase 2.7) uses only <see cref="SortDictByImportance" />; the remaining helpers
///     are used by the fANOVA + Mean Decrease Impurity evaluators (Phase 5.1, satellite package
///     <c>OptunaSharp.Importance.Forests</c>) and are reached across the friend boundary.
/// </summary>
internal static class ImportanceBase
{
    /// <summary>Python <c>_sort_dict_by_importance(param_importances)</c>.</summary>
    internal static OrderedDict<string, double> SortDictByImportance(OrderedDict<string, double> paramImportances)
    {
        // Python: dict(reversed(sorted(items, key=lambda x: x[1]))). `sorted` is ascending and
        // stable (ties keep insertion order); the whole sequence is then reversed — so the result
        // is descending by value, and ties fall in reverse insertion order.
        var ordered = paramImportances
            .Select((kv, index) => (kv, index))
            .OrderBy(t => t.kv.Value)
            .ThenBy(t => t.index)
            .Reverse();
        var result = new OrderedDict<string, double>(paramImportances.Count);
        foreach (var (kv, _) in ordered)
            result[kv.Key] = kv.Value;
        return result;
    }

    /// <summary>Python <c>_get_distributions(study, params)</c>.</summary>
    internal static OrderedDict<string, BaseDistribution> GetDistributions(
        Study study, IReadOnlyList<string>? @params)
    {
        var completedTrials = study.GetTrials(deepcopy: false, states: new[] { TrialState.Complete });
        CheckEvaluateArgs(completedTrials, @params);

        if (@params is null)
            return SearchSpaceFunctions.IntersectionSearchSpaceOf(study.GetTrials(deepcopy: false));

        // Compute the search space based on the subset of trials containing all parameters.
        OrderedDict<string, BaseDistribution>? distributions = null;
        foreach (var trial in completedTrials)
        {
            var trialDistributions = trial.DistributionsDict;
            if (!@params.All(name => trialDistributions.ContainsKey(name)))
                continue;

            if (distributions is null)
            {
                distributions = new OrderedDict<string, BaseDistribution>();
                foreach (var (name, dist) in trialDistributions)
                {
                    if (@params.Contains(name))
                        distributions[name] = dist;
                }
                continue;
            }

            if (distributions.Any(kv => !trialDistributions[kv.Key].Equals(kv.Value)))
                throw new ValueError(
                    "Parameters importances cannot be assessed with dynamic search spaces if " +
                    $"parameters are specified. Specified parameters: {PythonRepr.ListRepr(@params)}.");
        }

        System.Diagnostics.Debug.Assert(distributions is not null);
        // Python: dict(sorted(distributions.items(), key=lambda x: x[0])).
        var sorted = new OrderedDict<string, BaseDistribution>(distributions!.Count);
        foreach (var kv in distributions.OrderBy(kv => kv.Key, StringComparer.Ordinal))
            sorted[kv.Key] = kv.Value;
        return sorted;
    }

    /// <summary>Python <c>_check_evaluate_args(completed_trials, params)</c>.</summary>
    internal static void CheckEvaluateArgs(
        IReadOnlyList<FrozenTrial> completedTrials, IReadOnlyList<string>? @params)
    {
        // The Python TypeErrors ("Parameters must be specified as a list" / "…by their names with
        // strings") are unreachable here — the C# signature already constrains params to a list of
        // strings. Only the ValueError branch can fire.
        if (@params is not null && @params.Count > 0)
        {
            bool atLeastOneTrial = completedTrials.Any(
                trial => @params.All(p => trial.DistributionsDict.ContainsKey(p)));
            if (!atLeastOneTrial)
                throw new ValueError(
                    "Study must contain completed trials with all specified parameters. " +
                    $"Specified parameters: {PythonRepr.ListRepr(@params)}.");
        }
    }

    /// <summary>Python <c>_get_filtered_trials(study, params, target)</c>.</summary>
    internal static List<FrozenTrial> GetFilteredTrials(
        Study study, IReadOnlyCollection<string> @params, Func<FrozenTrial, double>? target)
    {
        var trials = study.GetTrials(deepcopy: false, states: new[] { TrialState.Complete });
        return trials
            .Where(trial =>
                @params.All(p => trial.Params.ContainsKey(p)) &&
                double.IsFinite(target is not null ? target(trial) : trial.Value!.Value))
            .ToList();
    }

    /// <summary>Python <c>_param_importances_to_dict(params, param_importances)</c> — array form.</summary>
    internal static OrderedDict<string, double> ParamImportancesToDict(
        IReadOnlyCollection<string> @params, double[] paramImportances)
    {
        var result = new OrderedDict<string, double>(@params.Count);
        int i = 0;
        foreach (var name in @params)
            result[name] = paramImportances[i++];
        return result;
    }

    /// <summary>
    ///     Python <c>_param_importances_to_dict(params, param_importances)</c> — scalar form
    ///     (<c>np.broadcast_to(scalar, (len(params),))</c>).
    /// </summary>
    internal static OrderedDict<string, double> ParamImportancesToDict(
        IReadOnlyCollection<string> @params, double paramImportance)
    {
        var result = new OrderedDict<string, double>(@params.Count);
        foreach (var name in @params)
            result[name] = paramImportance;
        return result;
    }

    /// <summary>Python <c>_get_trans_params(trials, trans)</c>.</summary>
    internal static double[][] GetTransParams(IReadOnlyList<FrozenTrial> trials, SearchSpaceTransform trans)
    {
        var transParams = new double[trials.Count][];
        for (int i = 0; i < trials.Count; i++)
            transParams[i] = trans.Transform(trials[i].Params).ToArray<double>();
        return transParams;
    }

    /// <summary>Python <c>_get_target_values(trials, target)</c>.</summary>
    internal static double[] GetTargetValues(IReadOnlyList<FrozenTrial> trials, Func<FrozenTrial, double>? target)
    {
        var values = new double[trials.Count];
        for (int i = 0; i < trials.Count; i++)
            values[i] = target is not null ? target(trials[i]) : trials[i].Value!.Value;
        return values;
    }
}
