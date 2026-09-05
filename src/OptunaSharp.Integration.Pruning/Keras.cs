using System;
using System.Collections.Generic;
using OptunaSharp.Internal;

namespace OptunaSharp.Integration;

/// <summary>
///     Keras callback to prune unpromising trials. A .NET port of
///     <c>optuna_integration.keras.KerasPruningCallback</c>.
///
///     <para>
///     At the end of each epoch (subject to <see cref="OnEpochEnd" />'s interval), it reads the
///     monitored metric from the epoch logs, reports it to the <see cref="Trial" />, and raises
///     <see cref="TrialPruned" /> when the study's pruner decides the trial is unpromising. If the
///     monitored metric is absent, it emits a warning and skips (matching Optuna).
///     </para>
///
///     <para>
///     <b>.NET note (Phase-6 adapter).</b> In Python Keras invokes <c>on_epoch_end</c> with the epoch
///     logs. You invoke <see cref="OnEpochEnd" /> yourself from your training loop, passing the epoch's
///     metric logs. For a framework-agnostic helper that takes just a scalar metric, see
///     <see cref="OptunaSharp.Integration.PruningCallback" />.
///     </para>
/// </summary>
public class KerasPruningCallback
{
    private readonly Trial _trial;
    private readonly string _monitor;
    private readonly int _interval;

    /// <param name="trial">
    ///     A <see cref="Trial" /> corresponding to the current evaluation of the objective function.
    /// </param>
    /// <param name="monitor">
    ///     An evaluation metric for pruning, e.g. <c>val_loss</c> and <c>val_accuracy</c>.
    /// </param>
    /// <param name="interval">
    ///     Check whether the trial should be pruned every n-th epoch. By default <c>interval=1</c> and
    ///     pruning is performed after every epoch. Increase it to run several epochs before applying
    ///     pruning.
    /// </param>
    public KerasPruningCallback(Trial trial, string monitor, int interval = 1)
    {
        _trial = trial ?? throw new ArgumentNullException(nameof(trial));
        _monitor = monitor ?? throw new ArgumentNullException(nameof(monitor));
        _interval = interval;
    }

    /// <summary>
    ///     Report the epoch's monitored metric and prune if unpromising. Port of the Python callback's
    ///     <c>on_epoch_end(epoch, logs)</c>.
    /// </summary>
    /// <param name="epoch">The zero-based epoch index.</param>
    /// <param name="logs">
    ///     The epoch's evaluation logs (metric name to value). May be <c>null</c> (treated as empty).
    /// </param>
    /// <exception cref="TrialPruned">Thrown when the pruner decides the trial should be pruned.</exception>
    public void OnEpochEnd(int epoch, IReadOnlyDictionary<string, double>? logs = null)
    {
        if ((epoch + 1) % _interval != 0)
            return;

        double? currentScore = logs is not null && logs.TryGetValue(_monitor, out double v) ? v : null;
        if (currentScore is null)
        {
            OptunaWarnings.Warn(
                $"The metric '{_monitor}' is not in the evaluation logs for pruning. " +
                "Please make sure you set the correct metric name.");
            return;
        }

        _trial.Report(currentScore.Value, epoch);
        if (_trial.ShouldPrune())
            throw new TrialPruned($"Trial was pruned at epoch {epoch}.");
    }
}
