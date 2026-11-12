using System;
using System.Collections.Generic;
using OptunaSharp.Internal;

namespace OptunaSharp.Integration.LightGbm;

/// <summary>
///     One entry of a LightGBM per-iteration evaluation log — the C# analogue of a member of LightGBM's
///     <c>env.evaluation_result_list</c>: <c>(&lt;dataset_name&gt;, &lt;metric_name&gt;, &lt;score&gt;,
///     &lt;is_higher_better&gt;)</c>.
/// </summary>
/// <param name="ValidName">The validation dataset name (e.g. <c>valid_0</c>, or <c>valid</c> for cv).</param>
/// <param name="Metric">
///     The metric name. LightGBM v4.0.0+ prefixes it with <c>"valid "</c>; both the bare and prefixed
///     forms are accepted, matching <c>_find_evaluation_result</c>.
/// </param>
/// <param name="Score">The metric value at the current iteration.</param>
/// <param name="IsHigherBetter">Whether a higher value of the metric is better.</param>
public readonly record struct LightGbmEvaluationResult(
    string ValidName, string Metric, double Score, bool IsHigherBetter);

/// <summary>
///     Callback for LightGBM to prune unpromising trials. A .NET port of
///     <c>optuna_integration.lightgbm.LightGBMPruningCallback</c>.
///
///     <para>
///     Each boosting iteration, it locates the entry for the target validation name + metric in the
///     iteration's evaluation log, checks that the metric's optimization direction is consistent with the
///     study's direction, reports the value to the <see cref="Trial" />, and raises
///     <see cref="TrialPruned" /> when the study's pruner decides the trial is unpromising.
///     </para>
///
///     <para>
///     <b>.NET note (Phase-6 adapter).</b> In Python the LightGBM engine invokes the callback with the
///     iteration's <c>evaluation_result_list</c>. ML.NET's native LightGBM (<c>Microsoft.ML.LightGbm</c>)
///     exposes no managed per-iteration callback, so you invoke <see cref="Call" /> yourself from your
///     training loop, passing the iteration index and the evaluation log for that iteration. For a
///     framework-agnostic helper that takes just a scalar metric, see
///     <see cref="OptunaSharp.Integration.PruningCallback" />.
///     </para>
/// </summary>
/// <seealso href="https://optuna-integration.readthedocs.io/en/stable/reference/generated/optuna_integration.LightGBMPruningCallback.html">Official metric-reporting and pruning contract.</seealso>
public class LightGBMPruningCallback
{
    private readonly Trial _trial;
    private readonly string _validName;
    private readonly string _metric;
    private readonly int _reportInterval;

    /// <param name="trial">
    ///     A <see cref="Trial" /> corresponding to the current evaluation of the objective function.
    /// </param>
    /// <param name="metric">
    ///     An evaluation metric for pruning, e.g. <c>binary_error</c> and <c>multi_error</c>.
    /// </param>
    /// <param name="validName">
    ///     The name of the target validation. If omitted, <c>valid_0</c> is used which is the default name
    ///     of the first validation. Ignored when <c>isCv = true</c> is passed to <see cref="Call" />.
    /// </param>
    /// <param name="reportInterval">
    ///     Check whether the trial should report intermediate values for pruning every n-th boosting
    ///     iteration. By default <c>reportInterval=1</c> and reporting is performed after every iteration.
    ///     Note that the pruning itself is performed according to the interval definition of the pruner.
    /// </param>
    /// <summary>Creates a callback that reports the selected validation metric and asks Optuna whether to prune.</summary>
    /// <exception cref="System.ArgumentNullException"><paramref name="trial" /> or <paramref name="metric" /> or <paramref name="validName" /> is null.</exception>
    public LightGBMPruningCallback(
        Trial trial, string metric, string validName = "valid_0", int reportInterval = 1)
    {
        _trial = trial ?? throw new ArgumentNullException(nameof(trial));
        _metric = metric ?? throw new ArgumentNullException(nameof(metric));
        _validName = validName ?? throw new ArgumentNullException(nameof(validName));
        _reportInterval = reportInterval;
    }

    // Port of `_find_evaluation_result`: the first entry whose valid-name matches and whose metric matches
    // either the bare name or the LightGBM v4.0.0+ "valid "-prefixed name.
    /// <summary>Finds the requested validation-set and metric entry, including cross-validation naming rules.</summary>
    /// <param name="targetValidName">The validation-set name whose metric result is required.</param>
    /// <param name="evaluationResultList">The per-validation-set metric results supplied by the training callback.</param>
    /// <returns>The matching metric result, or null when it is absent.</returns>
    private LightGbmEvaluationResult? FindEvaluationResult(
        string targetValidName, IReadOnlyList<LightGbmEvaluationResult> evaluationResultList)
    {
        foreach (var result in evaluationResultList)
        {
            if (result.ValidName != targetValidName ||
                (result.Metric != "valid " + _metric && result.Metric != _metric))
                continue;
            return result;
        }

        return null;
    }

    /// <summary>
    ///     Report the current iteration's metric to the trial and prune if unpromising. Port of the Python
    ///     callback's <c>__call__(env)</c>.
    /// </summary>
    /// <param name="iteration">
    ///     The zero-based boosting iteration (the analogue of <c>env.iteration</c>). Reporting only happens
    ///     when <c>(iteration + 1) % reportInterval == 0</c>.
    /// </param>
    /// <param name="evaluationResultList">The evaluation log for this iteration (the analogue of <c>env.evaluation_result_list</c>).</param>
    /// <param name="isCv">
    ///     <c>true</c> when this callback is driving a cross-validation run, in which case the target
    ///     validation name is <c>valid</c> (as LightGBM's <c>cv</c> reports) rather than
    ///     the validation name supplied to the constructor.
    /// </param>
    /// <exception cref="NumSharp.ValueError">
    ///     Thrown when the target validation name + metric is not present in
    ///     <paramref name="evaluationResultList" />, or when the metric's direction is inconsistent with the
    ///     study's direction (verbatim Optuna messages).
    /// </exception>
    /// <exception cref="TrialPruned">Thrown when the pruner decides the trial should be pruned.</exception>
    /// <exception cref="System.DivideByZeroException">The callback was constructed with a report interval of zero.</exception>
    public void Call(
        int iteration, IReadOnlyList<LightGbmEvaluationResult> evaluationResultList, bool isCv = false)
    {
        if ((iteration + 1) % _reportInterval != 0)
            return;

        string targetValidName = isCv ? "valid" : _validName;

        var evaluationResult = FindEvaluationResult(targetValidName, evaluationResultList);
        if (evaluationResult is null)
            throw new NumSharp.ValueError(
                $"The entry associated with the validation name \"{targetValidName}\" and the metric " +
                $"name \"{_metric}\" is not found in the evaluation result list " +
                $"{FormatEvaluationResultList(evaluationResultList)}.");

        var result = evaluationResult.Value;

        // The intermediate values must be consistent with the objective in terms of study direction.
        if (result.IsHigherBetter)
        {
            if (_trial.Study.Direction != StudyDirection.Maximize)
                throw new NumSharp.ValueError(
                    "The intermediate values are inconsistent with the objective values in terms of " +
                    "study directions. Please specify a metric to be minimized for " +
                    "LightGBMPruningCallback.");
        }
        else
        {
            if (_trial.Study.Direction != StudyDirection.Minimize)
                throw new NumSharp.ValueError(
                    "The intermediate values are inconsistent with the objective values in terms of " +
                    "study directions. Please specify a metric to be maximized for " +
                    "LightGBMPruningCallback.");
        }

        _trial.Report(result.Score, iteration);

        if (_trial.ShouldPrune())
            throw new TrialPruned($"Trial was pruned at iteration {iteration}.");
    }

    /// <summary>Formats evaluation results into a Python-style diagnostic list.</summary>
    /// <param name="list">The callback metric results formatted into the diagnostic.</param>
    /// <returns>The formatted list of validation metric records.</returns>
    private static string FormatEvaluationResultList(IReadOnlyList<LightGbmEvaluationResult> list)
    {
        // Python's str(list of tuples): every element in repr, so scores read 100.0 and 1e-05 whatever the current
        // culture (C#'s default formatting gave 100, 1E-05 and, under de-DE, 0,5), and a name holding a single quote is
        // double-quoted as Python's repr switches quotes (QA-76). A cross-validation result prints the four fields this
        // record carries; Python's has a fifth, the standard deviation, which LightGbmEvaluationResult does not hold.
        var parts = new List<string>(list.Count);
        foreach (var r in list)
            parts.Add($"({PythonRepr.Repr(r.ValidName)}, {PythonRepr.Repr(r.Metric)}, {PythonRepr.Repr(r.Score)}, " +
                      $"{PythonRepr.Repr(r.IsHigherBetter)})");
        return "[" + string.Join(", ", parts) + "]";
    }
}
