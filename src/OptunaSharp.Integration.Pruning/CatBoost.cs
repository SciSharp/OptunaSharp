using System;
using System.Collections.Generic;
using System.Text;
using NumSharp;
using OptunaSharp.Internal;

namespace OptunaSharp.Integration;

/// <summary>
///     The per-iteration information CatBoost hands its <c>after_iteration</c> callback — the C#
///     analogue of the <c>SimpleNamespace</c> in <c>optuna_integration.CatBoostPruningCallback</c>:
///     an iteration counter plus the running history of losses per validation set and metric.
/// </summary>
/// <remarks>
///     For example, CatBoost's <c>SimpleNamespace(iteration=2, metrics={'learn': {'Logloss': [0.6,
///     0.5]}, 'validation': {'Logloss': [0.7, 0.6], 'AUC': [0.8, 0.9]}})</c> is modeled as
///     <c>new CatBoostEvaluationInfo(2, metrics)</c> where <c>metrics["validation"]["Logloss"] = [0.7,
///     0.6]</c>. The callback reads the <b>last</b> element of the targeted history (the current
///     iteration's score).
/// </remarks>
public sealed class CatBoostEvaluationInfo
{
    /// <param name="iteration">
    ///     CatBoost's 1-based iteration counter (<c>info.iteration</c>); the reported step is
    ///     <c>iteration - 1</c>.
    /// </param>
    /// <param name="metrics">
    ///     Per-validation-set, per-metric history of losses (<c>info.metrics</c>): validation name
    ///     (e.g. <c>learn</c> / <c>validation</c> / <c>validation_1</c>) to metric name (e.g.
    ///     <c>Logloss</c> / <c>AUC</c>) to the list of scores observed so far.
    /// </param>
    public CatBoostEvaluationInfo(
        int iteration,
        IReadOnlyDictionary<string, IReadOnlyDictionary<string, IReadOnlyList<double>>> metrics)
    {
        Iteration = iteration;
        Metrics = metrics ?? throw new ArgumentNullException(nameof(metrics));
    }

    /// <summary>CatBoost's 1-based iteration counter (<c>info.iteration</c>).</summary>
    public int Iteration { get; }

    /// <summary>Per-validation-set, per-metric history of losses (<c>info.metrics</c>).</summary>
    public IReadOnlyDictionary<string, IReadOnlyDictionary<string, IReadOnlyList<double>>> Metrics { get; }

    /// <summary>
    ///     Reproduce CPython's <c>str(SimpleNamespace(...))</c> so the callback's <c>ValueError</c>
    ///     messages match Optuna verbatim: <c>namespace(iteration=&lt;n&gt;, metrics=&lt;dict&gt;)</c>.
    /// </summary>
    public override string ToString()
    {
        var sb = new StringBuilder("namespace(iteration=");
        sb.Append(Iteration.ToString(System.Globalization.CultureInfo.InvariantCulture));
        sb.Append(", metrics={");
        bool firstValid = true;
        foreach (var validEntry in Metrics)
        {
            if (!firstValid)
                sb.Append(", ");
            sb.Append(PythonRepr.StringRepr(validEntry.Key)).Append(": {");
            bool firstMetric = true;
            foreach (var metricEntry in validEntry.Value)
            {
                if (!firstMetric)
                    sb.Append(", ");
                sb.Append(PythonRepr.StringRepr(metricEntry.Key)).Append(": ")
                  .Append(PythonRepr.ListRepr(metricEntry.Value));
                firstMetric = false;
            }
            sb.Append('}');
            firstValid = false;
        }
        return sb.Append("})").ToString();
    }
}

/// <summary>
///     Callback for CatBoost to prune unpromising trials. A .NET port of
///     <c>optuna_integration.catboost.CatBoostPruningCallback</c>.
///
///     <para>
///     Once per CatBoost iteration, it reads the current score for the target validation set + metric,
///     reports it to the <see cref="Trial" />, and — if the study's pruner decides the trial is
///     unpromising — records that the trial should be pruned. As in Optuna,
///     <see cref="TrialPruned" /> is <b>not</b> raised from <see cref="AfterIteration" /> (CatBoost's
///     callback contract cannot propagate an exception): <see cref="AfterIteration" /> returns
///     <c>false</c> to stop training, and you must call <see cref="CheckPruned" /> after training to
///     raise <see cref="TrialPruned" />.
///     </para>
///
///     <para>
///     <b>.NET note (Phase-6 adapter).</b> In Python CatBoost invokes this callback, handing it a
///     <c>SimpleNamespace</c> of the iteration's metric history. There is no .NET CatBoost with a
///     managed per-iteration callback, so you invoke <see cref="AfterIteration" /> yourself from your
///     training loop, passing a <see cref="CatBoostEvaluationInfo" /> for the current iteration. For a
///     framework-agnostic helper that takes just a scalar metric, see
///     <see cref="OptunaSharp.Integration.PruningCallback" />.
///     </para>
/// </summary>
public class CatBoostPruningCallback
{
    private readonly Trial _trial;
    private readonly string _metric;
    private readonly string _validName;
    private bool _pruned;
    private string _message = "";

    /// <param name="trial">
    ///     A <see cref="Trial" /> corresponding to the current evaluation of the objective function.
    /// </param>
    /// <param name="metric">
    ///     An evaluation metric for pruning, e.g. <c>Logloss</c> and <c>AUC</c>.
    /// </param>
    /// <param name="evalSetIndex">
    ///     The index of the target validation dataset. If you set only one <c>eval_set</c>, leave it
    ///     <c>null</c> (the target validation name is <c>validation</c>). If you set multiple datasets,
    ///     pass the index (e.g. <c>0</c> or <c>1</c>) so the target validation name becomes
    ///     <c>validation_&lt;index&gt;</c>.
    /// </param>
    public CatBoostPruningCallback(Trial trial, string metric, int? evalSetIndex = null)
    {
        const string defaultValidName = "validation";
        _trial = trial ?? throw new ArgumentNullException(nameof(trial));
        _metric = metric ?? throw new ArgumentNullException(nameof(metric));
        _validName = evalSetIndex is null
            ? defaultValidName
            : defaultValidName + "_" + evalSetIndex.Value.ToString(System.Globalization.CultureInfo.InvariantCulture);
    }

    /// <summary><c>true</c> once <see cref="AfterIteration" /> has decided the trial should be pruned.</summary>
    public bool Pruned => _pruned;

    /// <summary>
    ///     Report the current iteration's metric value for Optuna pruning. Port of the Python
    ///     callback's <c>after_iteration(info)</c>.
    /// </summary>
    /// <param name="info">
    ///     The current iteration's evaluation information (the analogue of CatBoost's
    ///     <c>SimpleNamespace</c>).
    /// </param>
    /// <returns>
    ///     <c>false</c> if the trial should be pruned — stop training and then call
    ///     <see cref="CheckPruned" />. Otherwise <c>true</c>, meaning training continues.
    /// </returns>
    /// <exception cref="NumSharp.ValueError">
    ///     Thrown when the target validation name or the metric name is not present in
    ///     <paramref name="info" /> (verbatim Optuna messages).
    /// </exception>
    public bool AfterIteration(CatBoostEvaluationInfo info)
    {
        int step = info.Iteration - 1;
        if (!info.Metrics.TryGetValue(_validName, out var metrics))
            throw new ValueError(
                $"The entry associated with the validation name \"{_validName}\" " +
                $"is not found in the evaluation result list {info}.");

        if (!metrics.TryGetValue(_metric, out var history))
            throw new ValueError(
                $"The entry associated with the metric name \"{_metric}\" " +
                $"is not found in the evaluation result list {info}.");

        double currentScore = history[history.Count - 1];
        _trial.Report(currentScore, step);
        if (_trial.ShouldPrune())
        {
            _message = $"Trial was pruned at iteration {step}.";
            _pruned = true;
            return false;
        }

        return true;
    }

    /// <summary>
    ///     Raise <see cref="TrialPruned" /> manually if the CatBoost optimization was pruned. Call this
    ///     once after training, as with Optuna's <c>CatBoostPruningCallback.check_pruned</c>.
    /// </summary>
    /// <exception cref="TrialPruned">Thrown when the trial was pruned.</exception>
    public void CheckPruned()
    {
        if (_pruned)
            throw new TrialPruned(_message);
    }
}
