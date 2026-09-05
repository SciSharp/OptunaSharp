using System;
using System.Collections.Generic;
using NumSharp;

namespace OptunaSharp.Integration;

/// <summary>
///     PyTorch Ignite handler to prune unpromising trials. A .NET port of
///     <c>optuna_integration.pytorch_ignite.PyTorchIgnitePruningHandler</c>.
///
///     <para>
///     When invoked, it reads the named metric from the (evaluator) engine's metrics, reports it to the
///     <see cref="Trial" /> at the trainer's current epoch, and raises <see cref="TrialPruned" /> when
///     the study's pruner decides the trial is unpromising. As in Optuna, the reported step is the
///     <i>trainer</i>'s epoch while the metric comes from the engine the handler is attached to.
///     </para>
///
///     <para>
///     <b>.NET note (Phase-6 adapter).</b> In Python Ignite invokes the handler with the engine; here
///     you invoke <see cref="Call" /> yourself (e.g. from an epoch-completed handler), passing the
///     engine's metrics and the trainer's epoch. For a framework-agnostic helper that takes just a
///     scalar metric, see <see cref="OptunaSharp.Integration.PruningCallback" />.
///     </para>
/// </summary>
public class PyTorchIgnitePruningHandler
{
    private readonly Trial _trial;
    private readonly string _metric;

    /// <param name="trial">
    ///     A <see cref="Trial" /> corresponding to the current evaluation of the objective function.
    /// </param>
    /// <param name="metric">A name of metric for pruning, e.g. <c>accuracy</c> and <c>loss</c>.</param>
    public PyTorchIgnitePruningHandler(Trial trial, string metric)
    {
        _trial = trial ?? throw new ArgumentNullException(nameof(trial));
        _metric = metric ?? throw new ArgumentNullException(nameof(metric));
    }

    /// <summary>
    ///     Report the engine's metric to the trial at the trainer's epoch and prune if unpromising.
    ///     Port of the Python handler's <c>__call__(engine)</c>.
    /// </summary>
    /// <param name="engineMetrics">
    ///     The engine's metrics (the analogue of <c>engine.state.metrics</c>) — the handler reads
    ///     <c>metric</c> from here.
    /// </param>
    /// <param name="trainerEpoch">The trainer's current epoch (the analogue of <c>trainer.state.epoch</c>).</param>
    /// <exception cref="NumSharp.KeyError">Thrown when <c>metric</c> is not present in <paramref name="engineMetrics" />.</exception>
    /// <exception cref="TrialPruned">Thrown when the pruner decides the trial should be pruned.</exception>
    public void Call(IReadOnlyDictionary<string, double> engineMetrics, int trainerEpoch)
    {
        if (engineMetrics is null)
            throw new ArgumentNullException(nameof(engineMetrics));
        if (!engineMetrics.TryGetValue(_metric, out double score))
            throw new KeyError(_metric);

        _trial.Report(score, trainerEpoch);
        if (_trial.ShouldPrune())
            throw new TrialPruned($"Trial was pruned at {trainerEpoch} epoch.");
    }
}
