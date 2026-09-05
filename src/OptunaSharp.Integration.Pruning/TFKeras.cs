using System;
using System.Collections.Generic;
using OptunaSharp.Internal;

namespace OptunaSharp.Integration;

/// <summary>
///     tf.keras callback to prune unpromising trials. A .NET port of
///     <c>optuna_integration.tfkeras.TFKerasPruningCallback</c>.
///
///     <para>
///     At the end of each epoch it reads the monitored metric from the epoch logs, reports it to the
///     <see cref="Trial" />, and raises <see cref="TrialPruned" /> when the study's pruner decides the
///     trial is unpromising. If the monitored metric is absent, it emits a warning and skips (matching
///     Optuna). This mirrors <see cref="KerasPruningCallback" /> but without a report interval, as in
///     Optuna.
///     </para>
///
///     <para>
///     <b>.NET note (Phase-6 adapter).</b> You invoke <see cref="OnEpochEnd" /> yourself from your
///     training loop, passing the epoch's metric logs (in Python tf.keras invokes it). For a
///     framework-agnostic helper that takes just a scalar metric, see
///     <see cref="OptunaSharp.Integration.PruningCallback" />.
///     </para>
/// </summary>
public class TFKerasPruningCallback
{
    private readonly Trial _trial;
    private readonly string _monitor;

    /// <param name="trial">
    ///     A <see cref="Trial" /> corresponding to the current evaluation of the objective function.
    /// </param>
    /// <param name="monitor">
    ///     An evaluation metric for pruning, e.g. <c>val_loss</c> or <c>val_acc</c>.
    /// </param>
    public TFKerasPruningCallback(Trial trial, string monitor)
    {
        _trial = trial ?? throw new ArgumentNullException(nameof(trial));
        _monitor = monitor ?? throw new ArgumentNullException(nameof(monitor));
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
        double? currentScore = logs is not null && logs.TryGetValue(_monitor, out double v) ? v : null;
        if (currentScore is null)
        {
            OptunaWarnings.Warn(
                $"The metric '{_monitor}' is not in the evaluation logs for pruning. " +
                "Please make sure you set the correct metric name.");
            return;
        }

        // Report current score and epoch to Optuna's trial.
        _trial.Report(currentScore.Value, epoch);

        // Prune trial if needed.
        if (_trial.ShouldPrune())
            throw new TrialPruned($"Trial was pruned at epoch {epoch}.");
    }
}
