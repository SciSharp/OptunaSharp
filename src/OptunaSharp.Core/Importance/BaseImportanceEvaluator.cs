using System;
using System.Collections.Generic;
using System.Linq;
using OptunaSharp.Collections;

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
///     Port of the module-level helpers in <c>optuna/importance/_base.py</c> used by the concrete
///     importance evaluators shipped in Tier N. (The fANOVA/MDI-only helpers in that module land
///     with those evaluators in Phase 5.1.)
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
}
