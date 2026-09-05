using System;
using System.Collections.Generic;
using NumSharp;

namespace OptunaSharp.Integration;

/// <summary>
///     Skorch callback to prune unpromising trials. A .NET port of
///     <c>optuna_integration.skorch.SkorchPruningCallback</c>.
///
///     <para>
///     At the end of each epoch it reads the monitored metric from the last row of the net's history,
///     reports it to the <see cref="Trial" /> at that epoch, and raises <see cref="TrialPruned" /> when
///     the study's pruner decides the trial is unpromising.
///     </para>
///
///     <para>
///     <b>.NET note (Phase-6 adapter).</b> In Python skorch invokes <c>on_epoch_end(net)</c>; here you
///     invoke <see cref="OnEpochEnd" /> yourself from your training loop, passing the net's history (one
///     dictionary per epoch). For a framework-agnostic helper that takes just a scalar metric, see
///     <see cref="OptunaSharp.Integration.PruningCallback" />.
///     </para>
/// </summary>
public class SkorchPruningCallback
{
    private readonly Trial _trial;
    private readonly string _monitor;

    /// <param name="trial">
    ///     A <see cref="Trial" /> corresponding to the current evaluation of the objective function.
    /// </param>
    /// <param name="monitor">
    ///     An evaluation metric for pruning, e.g. <c>val_loss</c> or <c>val_acc</c>. The metrics are
    ///     obtained from the net's history rows, so the names depend on how that history is formatted.
    /// </param>
    public SkorchPruningCallback(Trial trial, string monitor)
    {
        _trial = trial ?? throw new ArgumentNullException(nameof(trial));
        _monitor = monitor ?? throw new ArgumentNullException(nameof(monitor));
    }

    /// <summary>
    ///     Report the latest epoch's monitored metric and prune if unpromising. Port of the Python
    ///     callback's <c>on_epoch_end(net)</c>.
    /// </summary>
    /// <param name="history">
    ///     The net's history — one dictionary of metric name to value per epoch (the analogue of
    ///     <c>net.history</c>). If empty, the call is a no-op.
    /// </param>
    /// <exception cref="NumSharp.KeyError">Thrown when <c>monitor</c> is not present in the last history row.</exception>
    /// <exception cref="TrialPruned">Thrown when the pruner decides the trial should be pruned.</exception>
    public void OnEpochEnd(IReadOnlyList<IReadOnlyDictionary<string, double>> history)
    {
        if (history is null || history.Count == 0)
            return;

        int epoch = history.Count - 1;
        var lastRow = history[epoch];
        if (!lastRow.TryGetValue(_monitor, out double currentScore))
            throw new KeyError(_monitor);

        _trial.Report(currentScore, epoch);
        if (_trial.ShouldPrune())
            throw new TrialPruned($"Trial was pruned at epoch {epoch}.");
    }
}
