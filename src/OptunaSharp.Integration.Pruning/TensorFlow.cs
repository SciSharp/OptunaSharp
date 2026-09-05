using System;
using System.Collections.Generic;
using NumSharp;

namespace OptunaSharp.Integration;

/// <summary>
///     TensorFlow (Estimator) hook to prune unpromising trials. A .NET port of
///     <c>optuna_integration.tensorflow.TensorFlowPruningHook</c>.
///
///     <para>
///     On a step-interval it reads the newest evaluation summary, reports the metric to the
///     <see cref="Trial" />, and raises <see cref="TrialPruned" /> when the study's pruner decides the
///     trial is unpromising.
///     </para>
///
///     <para>
///     <b>Deprecated, mirroring Optuna:</b> TensorFlow recommends migrating from Estimator to Keras
///     APIs — prefer <see cref="KerasPruningCallback" />.
///     </para>
///
///     <para>
///     <b>.NET note (Phase-6 adapter).</b> In Python this is a <c>SessionRunHook</c>: <c>begin</c> /
///     <c>before_run</c> fetch the global-step tensor and <c>after_run</c> receives its value. Those
///     session-tensor mechanics have no .NET analogue, so they collapse into the <c>globalStep</c>
///     argument of <see cref="AfterRun" />, which you invoke yourself from your training loop. Reading
///     the eval summaries (Python's <c>read_eval_metrics(estimator.eval_dir())</c>) is supplied as the
///     <c>readEvalMetrics</c> delegate, called only when the step timer triggers.
///     </para>
/// </summary>
[Obsolete(
    "Deprecated in v4.1.0. This feature will be removed in v6.0.0. Use KerasPruningCallback instead, " +
    "because TensorFlow recommends migrating from Estimator to Keras APIs.")]
public class TensorFlowPruningHook
{
    private readonly Trial _trial;
    private readonly string _metric;
    private readonly SecondOrStepTimer _timer;
    private long _currentSummaryStep = -1;

    /// <param name="trial">
    ///     A <see cref="Trial" /> corresponding to the current evaluation of the objective function.
    /// </param>
    /// <param name="metric">An evaluation metric for pruning, e.g. <c>accuracy</c> and <c>loss</c>.</param>
    /// <param name="runEverySteps">An interval (in steps) to watch the summary file.</param>
    public TensorFlowPruningHook(Trial trial, string metric, int runEverySteps)
    {
        _trial = trial ?? throw new ArgumentNullException(nameof(trial));
        _metric = metric ?? throw new ArgumentNullException(nameof(metric));
        _timer = new SecondOrStepTimer(runEverySteps);
    }

    /// <summary>
    ///     Port of the Python hook's <c>after_run</c>: on the step interval, read the newest evaluation
    ///     summary, report its metric, and prune if unpromising.
    /// </summary>
    /// <param name="globalStep">The current global step (the analogue of <c>run_values.results</c>).</param>
    /// <param name="readEvalMetrics">
    ///     Reads the evaluation summaries as an ascending-by-step ordered list (the analogue of
    ///     <c>read_eval_metrics(estimator.eval_dir())</c> — an <c>OrderedDict</c> sorted by step), where
    ///     each value maps metric name to its value (<c>null</c> when the metric had no value).
    ///     Invoked only when the step timer triggers.
    /// </param>
    /// <exception cref="NumSharp.KeyError">Thrown when <c>metric</c> is absent from the newest summary.</exception>
    /// <exception cref="TrialPruned">Thrown when the pruner decides the trial should be pruned.</exception>
    public void AfterRun(
        long globalStep,
        Func<IReadOnlyList<KeyValuePair<long, IReadOnlyDictionary<string, double?>>>> readEvalMetrics)
    {
        if (readEvalMetrics is null)
            throw new ArgumentNullException(nameof(readEvalMetrics));

        // Get eval metrics every n steps.
        IReadOnlyList<KeyValuePair<long, IReadOnlyDictionary<string, double?>>>? evalMetrics;
        if (_timer.ShouldTriggerForStep(globalStep))
        {
            _timer.UpdateLastTriggeredStep(globalStep);
            evalMetrics = readEvalMetrics();
        }
        else
        {
            evalMetrics = null;
        }

        if (evalMetrics is not null && evalMetrics.Count > 0)
        {
            // next(reversed(eval_metrics)): the last (newest) entry of the ordered summaries.
            var newest = evalMetrics[evalMetrics.Count - 1];
            long summaryStep = newest.Key;
            var latestEvalMetrics = newest.Value;
            // If there exists a new evaluation summary.
            if (summaryStep > _currentSummaryStep)
            {
                if (!latestEvalMetrics.TryGetValue(_metric, out double? currentScore))
                    throw new KeyError(_metric);
                _trial.Report(currentScore ?? double.NaN, (int)summaryStep);
                _currentSummaryStep = summaryStep;
            }
            if (_trial.ShouldPrune())
                throw new TrialPruned($"Trial was pruned at iteration {_currentSummaryStep}.");
        }
    }
}

/// <summary>
///     The step-interval subset of TensorFlow's <c>tf.estimator.SecondOrStepTimer</c>
///     (<c>every_secs=None, every_steps=N</c>) that <see cref="TensorFlowPruningHook" /> uses to decide
///     when to read the eval summaries.
/// </summary>
internal sealed class SecondOrStepTimer
{
    private readonly long _everySteps;
    private long? _lastTriggeredStep;

    internal SecondOrStepTimer(long everySteps)
    {
        _everySteps = everySteps;
    }

    internal bool ShouldTriggerForStep(long step)
    {
        if (_lastTriggeredStep is null)
            return true;
        if (_lastTriggeredStep == step)
            return false;
        return step >= _lastTriggeredStep.Value + _everySteps;
    }

    internal void UpdateLastTriggeredStep(long step)
    {
        _lastTriggeredStep = step;
    }
}
