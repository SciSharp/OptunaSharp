using System;
using System.Collections.Generic;
using NumSharp;

namespace OptunaSharp.Integration;

/// <summary>
///     One entry of an XGBoost per-iteration evaluation history. In a plain training run this is a
///     single score (<c>scores[-1]</c> in Python); in a cross-validation run XGBoost reports
///     <c>[mean, stddev]</c> across the folds and Optuna keeps only the mean (<c>scores[-1][0]</c>).
///     Both are modeled here: <see cref="Value" /> is the number reported to the trial, and
///     <see cref="CvStd" /> carries the dropped cross-validation fold stddev (<c>null</c> for a plain
///     run). Build these with <see cref="XGBoostEvalsLog.Add" /> / <see cref="XGBoostEvalsLog.AddCv" />.
/// </summary>
/// <param name="Value">The score reported to the trial (the CV fold mean, when applicable).</param>
/// <param name="CvStd">The cross-validation fold stddev XGBoost also reports, or <c>null</c> for a plain run.</param>
public readonly record struct XGBoostEvalScore(double Value, double? CvStd = null);

/// <summary>
///     An XGBoost per-iteration evaluation log — the C# analogue of XGBoost's <c>evals_log</c>: a
///     <c>{dataset: {metric: [scores over iterations]}}</c> mapping that accumulates one score per
///     dataset+metric each boosting round. Pass the same instance to
///     <see cref="XGBoostPruningCallback.AfterIteration" /> every round, appending the round's scores
///     first (as XGBoost's <c>evals_log</c> grows in place).
/// </summary>
public sealed class XGBoostEvalsLog
{
    // Insertion-ordered {dataset: {metric: history}} — order is irrelevant to the callback (it indexes
    // by the "dataset-metric" observation key), but preserving it keeps the model faithful to XGBoost's.
    private readonly Dictionary<string, Dictionary<string, List<XGBoostEvalScore>>> _log = new();

    /// <summary>Append a plain (non-CV) score for <paramref name="dataset" /> + <paramref name="metric" /> (XGBoost's <c>scores.append(value)</c>).</summary>
    public XGBoostEvalsLog Add(string dataset, string metric, double score)
        => Append(dataset, metric, new XGBoostEvalScore(score));

    /// <summary>Append a cross-validation score for <paramref name="dataset" /> + <paramref name="metric" /> — the fold <paramref name="mean" /> (reported) and <paramref name="stddev" /> (kept for fidelity, dropped by the callback), i.e. XGBoost's <c>scores.append([mean, stddev])</c>.</summary>
    public XGBoostEvalsLog AddCv(string dataset, string metric, double mean, double stddev)
        => Append(dataset, metric, new XGBoostEvalScore(mean, stddev));

    private XGBoostEvalsLog Append(string dataset, string metric, XGBoostEvalScore score)
    {
        if (dataset is null)
            throw new ArgumentNullException(nameof(dataset));
        if (metric is null)
            throw new ArgumentNullException(nameof(metric));
        if (!_log.TryGetValue(dataset, out var byMetric))
            _log[dataset] = byMetric = new Dictionary<string, List<XGBoostEvalScore>>();
        if (!byMetric.TryGetValue(metric, out var history))
            byMetric[metric] = history = new List<XGBoostEvalScore>();
        history.Add(score);
        return this;
    }

    // The datasets/metrics/histories, in insertion order — the callback flattens these to
    // "dataset-metric" keys (the analogue of iterating evals_log.items() then metrics.items()).
    internal IEnumerable<(string Dataset, string Metric, List<XGBoostEvalScore> Scores)> Entries()
    {
        foreach (var dataset in _log)
            foreach (var metric in dataset.Value)
                yield return (dataset.Key, metric.Key, metric.Value);
    }
}

/// <summary>
///     Callback for XGBoost to prune unpromising trials. A .NET port of
///     <c>optuna_integration.xgboost.XGBoostPruningCallback</c> (the modern
///     <c>xgb.callback.TrainingCallback</c> form).
///
///     <para>
///     Each boosting round it flattens the evaluation log to <c>{dataset-metric: score}</c>, reports
///     the score under the observation key to the <see cref="Trial" />, and raises
///     <see cref="TrialPruned" /> when the study's pruner decides the trial is unpromising.
///     </para>
///
///     <para>
///     <b>.NET note (Phase-6 adapter).</b> In Python XGBoost invokes the callback with the round's
///     <c>evals_log</c>. There is no .NET XGBoost with a managed per-iteration callback, so you invoke
///     <see cref="AfterIteration" /> yourself from your training loop, passing the accumulating
///     <see cref="XGBoostEvalsLog" />. For a framework-agnostic helper that takes just a scalar metric,
///     see <see cref="OptunaSharp.Integration.PruningCallback" />.
///     </para>
/// </summary>
public class XGBoostPruningCallback
{
    private readonly Trial _trial;
    private readonly string _observationKey;

    /// <param name="trial">
    ///     A <see cref="Trial" /> corresponding to the current evaluation of the objective function.
    /// </param>
    /// <param name="observationKey">
    ///     An evaluation metric for pruning, e.g. <c>validation-error</c> and <c>validation-merror</c>.
    ///     When using the scikit-learn API the index number of <c>eval_set</c> must be included, e.g.
    ///     <c>validation_0-error</c>. The key is <c>&lt;dataset&gt;-&lt;metric&gt;</c>.
    /// </param>
    public XGBoostPruningCallback(Trial trial, string observationKey)
    {
        _trial = trial ?? throw new ArgumentNullException(nameof(trial));
        _observationKey = observationKey ?? throw new ArgumentNullException(nameof(observationKey));
    }

    /// <summary>
    ///     Report the round's metric to the trial and prune if unpromising. Port of the Python
    ///     callback's <c>after_iteration(model, epoch, evals_log)</c>.
    /// </summary>
    /// <param name="epoch">The zero-based boosting round (the analogue of <c>epoch</c>).</param>
    /// <param name="evalsLog">The evaluation log accumulated so far (the analogue of <c>evals_log</c>).</param>
    /// <returns>
    ///     <c>false</c> — matching Optuna, the callback never asks XGBoost to stop training itself
    ///     (pruning is signalled by the thrown <see cref="TrialPruned" />).
    /// </returns>
    /// <exception cref="NumSharp.KeyError">Thrown when <c>observationKey</c> is not present in <paramref name="evalsLog" />.</exception>
    /// <exception cref="TrialPruned">Thrown when the pruner decides the trial should be pruned.</exception>
    public bool AfterIteration(int epoch, XGBoostEvalsLog evalsLog)
    {
        if (evalsLog is null)
            throw new ArgumentNullException(nameof(evalsLog));

        // Flatten the evaluation history to a {dataset-metric: score} layout. The CV-fold stddev
        // XGBoost appends alongside the mean (Python's scores[-1][0]) is carried on XGBoostEvalScore
        // and dropped here — Value is the mean in a CV run and the plain score otherwise.
        var evaluationResults = new Dictionary<string, double>();
        foreach (var (dataset, metric, scores) in evalsLog.Entries())
            evaluationResults[dataset + "-" + metric] = scores[scores.Count - 1].Value;

        if (!evaluationResults.TryGetValue(_observationKey, out double currentScore))
            throw new KeyError(_observationKey);

        _trial.Report(currentScore, epoch);
        if (_trial.ShouldPrune())
            throw new TrialPruned($"Trial was pruned at iteration {epoch}.");

        // The training should not stop.
        return false;
    }
}
