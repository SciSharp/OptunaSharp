using System;
using System.Collections.Generic;
using NumSharp;

namespace OptunaSharp.Integration;

/// <summary>
///     The .NET analogue of fastai's <c>CancelFitException</c> — the signal
///     <see cref="FastAIV2PruningCallback.AfterEpoch" /> raises to stop the fit early (fastai catches
///     it internally). The real <see cref="TrialPruned" /> is raised afterwards from
///     <see cref="FastAIV2PruningCallback.AfterFit" />, matching Optuna.
/// </summary>
public class CancelFitException : Exception
{
    /// <summary>Create a <see cref="CancelFitException" />.</summary>
    public CancelFitException() { }

    /// <summary>Create a <see cref="CancelFitException" /> with a message.</summary>
    public CancelFitException(string message) : base(message) { }
}

/// <summary>
///     FastAI (v2) callback to prune unpromising trials. A .NET port of
///     <c>optuna_integration.fastaiv2.FastAIV2PruningCallback</c>.
///
///     <para>
///     After each epoch it reports the monitored metric to the <see cref="Trial" /> and, if the study's
///     pruner decides the trial is unpromising, raises <see cref="CancelFitException" /> to stop the fit
///     (as fastai's <c>TrackerCallback</c> does); the real <see cref="TrialPruned" /> is then raised
///     from <see cref="AfterFit" /> — the two-stage flow Optuna uses so fastai can unwind cleanly first.
///     </para>
///
///     <para>
///     <b>.NET note (Phase-6 adapter).</b> In Python fastai invokes <c>after_epoch</c> / <c>after_fit</c>
///     and <c>TrackerCallback</c> resolves the monitor's column index into <c>recorder.final_record</c>.
///     Here you invoke <see cref="AfterEpoch" /> / <see cref="AfterFit" /> yourself from your training
///     loop, passing the epoch's final record and the metric names aligned with it (the analogue of
///     <c>recorder.metric_names[1:]</c>). For a framework-agnostic helper that takes just a scalar
///     metric, see <see cref="OptunaSharp.Integration.PruningCallback" />.
///     </para>
/// </summary>
public class FastAIV2PruningCallback
{
    private readonly Trial _trial;
    private readonly string _monitor;

    /// <param name="trial">
    ///     A <see cref="Trial" /> corresponding to the current evaluation of the objective function.
    /// </param>
    /// <param name="monitor">
    ///     An evaluation metric for pruning, e.g. <c>valid_loss</c> or <c>accuracy</c>.
    /// </param>
    public FastAIV2PruningCallback(Trial trial, string monitor = "valid_loss")
    {
        _trial = trial ?? throw new ArgumentNullException(nameof(trial));
        _monitor = monitor ?? throw new ArgumentNullException(nameof(monitor));
    }

    /// <summary>
    ///     Report the epoch's monitored metric and, if the trial should be pruned, raise
    ///     <see cref="CancelFitException" /> to stop the fit. Port of the Python callback's
    ///     <c>after_epoch</c> (which relies on <c>TrackerCallback</c> for the monitor's column index).
    /// </summary>
    /// <param name="finalRecord">
    ///     The epoch's final record — the loss/metric values (the analogue of
    ///     <c>recorder.final_record</c>).
    /// </param>
    /// <param name="metricNames">
    ///     The names aligned with <paramref name="finalRecord" /> (the analogue of
    ///     <c>recorder.metric_names[1:]</c>, i.e. excluding the leading <c>epoch</c> column) — used to
    ///     locate <c>monitor</c>'s column, as <c>TrackerCallback</c> does.
    /// </param>
    /// <param name="epoch">The zero-based epoch index (the analogue of <c>self.epoch</c>).</param>
    /// <exception cref="NumSharp.ValueError">Thrown when <c>monitor</c> is not among <paramref name="metricNames" /> (as Python's <c>list.index</c> raises).</exception>
    /// <exception cref="CancelFitException">Thrown when the pruner decides the trial should be pruned.</exception>
    public void AfterEpoch(IReadOnlyList<double> finalRecord, IReadOnlyList<string> metricNames, int epoch)
    {
        if (finalRecord is null)
            throw new ArgumentNullException(nameof(finalRecord));
        if (metricNames is null)
            throw new ArgumentNullException(nameof(metricNames));

        // TrackerCallback sets self.idx = list(recorder.metric_names[1:]).index(self.monitor).
        int idx = IndexOf(metricNames, _monitor);
        if (idx < 0)
            throw new ValueError($"'{_monitor}' is not in list.");

        _trial.Report(finalRecord[idx], epoch);
        if (_trial.ShouldPrune())
            throw new CancelFitException();
    }

    /// <summary>
    ///     Raise <see cref="TrialPruned" /> if the trial should be pruned. Port of the Python callback's
    ///     <c>after_fit</c> — called after the fit has been cancelled, to surface the pruning to Optuna.
    /// </summary>
    /// <param name="epoch">The zero-based epoch index at which the fit stopped (the analogue of <c>self.epoch</c>).</param>
    /// <exception cref="TrialPruned">Thrown when the pruner decides the trial should be pruned.</exception>
    public void AfterFit(int epoch)
    {
        if (_trial.ShouldPrune())
            throw new TrialPruned($"Trial was pruned at epoch {epoch}.");
    }

    private static int IndexOf(IReadOnlyList<string> names, string monitor)
    {
        for (int i = 0; i < names.Count; i++)
            if (names[i] == monitor)
                return i;
        return -1;
    }
}

/// <summary>
///     Alias for <see cref="FastAIV2PruningCallback" />, matching Optuna's
///     <c>FastAIPruningCallback = FastAIV2PruningCallback</c> export so either name can be used.
/// </summary>
public sealed class FastAIPruningCallback : FastAIV2PruningCallback
{
    /// <inheritdoc cref="FastAIV2PruningCallback(Trial, string)" />
    public FastAIPruningCallback(Trial trial, string monitor = "valid_loss") : base(trial, monitor) { }
}
