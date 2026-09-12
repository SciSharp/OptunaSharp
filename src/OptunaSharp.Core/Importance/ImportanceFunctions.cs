using System;
using System.Collections.Generic;
using OptunaSharp.Collections;
using OptunaSharp.Importance.PedAnova;
using OptunaSharp.Internal;

namespace OptunaSharp.Importance;

/// <summary>
///     Port of the module-level entry point in <c>optuna/importance/__init__.py</c>.
/// </summary>
public static class ImportanceFunctions
{
    /// <summary>
    ///     Evaluate parameter importances (<see cref="PedAnovaImportanceEvaluator" /> by default)
    ///     based on completed trials in the given study.
    ///
    ///     <para>The parameter importances are returned as a dictionary where the keys consist of
    ///     parameter names and their values importances. The importances are represented by
    ///     non-negative floating point numbers, where higher values mean that the parameters are
    ///     more important. The returned dictionary is ordered by its values in a descending order.
    ///     By default, the sum of the importance values are normalized to 1.0.</para>
    ///
    ///     <para>With the default evaluator, <see cref="PedAnovaImportanceEvaluator" />,
    ///     <paramref name="params" /> = <c>null</c> assesses all parameters that appear in completed
    ///     trials, including conditional parameters. Other evaluators assess only parameters that
    ///     are present in all of the completed trials. If <paramref name="params" /> is specified,
    ///     only the specified parameters are assessed.</para>
    ///
    ///     <para>If <paramref name="params" /> is specified as an empty list, an empty dictionary is
    ///     returned.</para>
    /// </summary>
    /// <param name="study">An optimized study.</param>
    /// <param name="evaluator">
    ///     An importance evaluator object that specifies which algorithm to base the importance
    ///     assessment on. Defaults to <see cref="PedAnovaImportanceEvaluator" />.
    /// </param>
    /// <param name="params">A list of names of parameters to assess.</param>
    /// <param name="target">
    ///     A function that returns the value used to evaluate importances. If <c>null</c>, objective
    ///     values are used for single-objective optimization.
    /// </param>
    /// <param name="normalize">
    ///     Whether the sum of the importance values should be normalized to 1.0. Defaults to
    ///     <c>true</c>.
    /// </param>
    /// <returns>
    ///     A dictionary where the keys are parameter names and the values are assessed importances.
    /// </returns>
    public static OrderedDict<string, double> GetParamImportances(
        Study study,
        BaseImportanceEvaluator? evaluator = null,
        IReadOnlyList<string>? @params = null,
        Func<FrozenTrial, double>? target = null,
        bool normalize = true)
    {
        evaluator ??= new PedAnovaImportanceEvaluator();

        var res = evaluator.Evaluate(study, @params: @params, target: target);
        if (normalize)
        {
            double s = 0.0;  // Python builtin sum(): sequential over the descending-ordered values.
            foreach (var value in res.Values)
                s += value;

            var result = new OrderedDict<string, double>(res.Count);
            if (s == 0.0)
            {
                int nParams = res.Count;
                foreach (var param in res.Keys)
                    result[param] = 1.0 / nParams;
            }
            else
            {
                foreach (var (param, value) in res)
                    result[param] = value / s;
            }
            return result;
        }

        // Optuna raises an ExperimentalWarning (warn_experimental_argument) on the non-default
        // normalize=false path. OptunaSharp targets the Optuna 5.x surface and treats it as stable,
        // so that warning is intentionally omitted (deliberate divergence).
        return res;
    }
}
