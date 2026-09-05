using System;
using System.Collections;
using System.Collections.Generic;
using System.Globalization;
using NumSharp;
using OptunaSharp.Collections;
using OptunaSharp.Internal;
using OptunaSharp.Storages;

namespace OptunaSharp.Integration;

/// <summary>
///     The distributed-data-parallel (DDP) coordination a <see cref="PyTorchLightningPruningCallback" />
///     needs on the DDP path — the C# analogue of the PyTorch Lightning <c>Trainer</c> bits it reaches:
///     <c>trainer.is_global_zero</c>, <c>trainer.strategy.broadcast(...)</c> and <c>trainer.should_stop</c>.
///     Non-distributed training never needs this. There is no PyTorch Lightning on .NET, so you supply
///     it against your own distributed runtime when driving the DDP path.
/// </summary>
public interface ILightningDistributedTrainer
{
    /// <summary>Whether this is the global-zero (rank 0) process (<c>trainer.is_global_zero</c>).</summary>
    bool IsGlobalZero { get; }

    /// <summary>The trainer's <c>should_stop</c> flag, read and written across the world processes.</summary>
    bool ShouldStop { get; set; }

    /// <summary>Broadcast a stop decision from global-zero to every process (<c>trainer.strategy.broadcast</c>).</summary>
    bool Broadcast(bool value);
}

/// <summary>
///     PyTorch Lightning callback to prune unpromising trials. A .NET port of
///     <c>optuna_integration.pytorch_lightning.PyTorchLightningPruningCallback</c>.
///
///     <para>
///     At the end of each validation it reads the monitored metric from the trainer's callback metrics,
///     reports it to the <see cref="Trial" />, and raises <see cref="TrialPruned" /> when the study's
///     pruner decides the trial is unpromising. For distributed data parallel (DDP) training, pruning is
///     coordinated across processes through the trial's system attributes and surfaced by
///     <see cref="CheckPruned" /> (call it right after fitting), exactly as in Optuna.
///     </para>
///
///     <para>
///     <b>.NET note (Phase-6 adapter).</b> In Python Lightning invokes <c>on_fit_start</c> /
///     <c>on_validation_end</c> with the <c>Trainer</c> and <c>LightningModule</c>; here you invoke
///     <see cref="OnFitStart" /> / <see cref="OnValidationEnd" /> yourself from your training loop,
///     passing the pieces they read (sanity-check flag, callback metrics, current epoch, and — on the
///     DDP path — an <see cref="ILightningDistributedTrainer" />). For a framework-agnostic helper that
///     takes just a scalar metric, see <see cref="OptunaSharp.Integration.PruningCallback" />.
///     </para>
/// </summary>
public class PyTorchLightningPruningCallback
{
    // Key names of Trial.system_attrs, verbatim from Optuna (so a Python and a C# process coordinating
    // over the same RDB storage read/write the same attributes).
    private const string EpochKey = "ddp_pl:epoch";
    private const string IntermediateValueKey = "ddp_pl:intermediate_value";
    private const string PrunedKey = "ddp_pl:pruned";

    private readonly Trial _trial;

    /// <param name="trial">
    ///     A <see cref="Trial" /> corresponding to the current evaluation of the objective function.
    /// </param>
    /// <param name="monitor">
    ///     An evaluation metric for pruning, e.g. <c>val_loss</c> or <c>val_acc</c>. The metrics are
    ///     obtained from the trainer's callback metrics, so the names depend on how those are formatted.
    /// </param>
    public PyTorchLightningPruningCallback(Trial trial, string monitor)
    {
        _trial = trial ?? throw new ArgumentNullException(nameof(trial));
        Monitor = monitor ?? throw new ArgumentNullException(nameof(monitor));
    }

    /// <summary>The monitored evaluation metric.</summary>
    public string Monitor { get; }

    /// <summary>Whether the DDP backend is in use (set by <see cref="OnFitStart" />).</summary>
    public bool IsDdpBackend { get; private set; }

    /// <summary>
    ///     Port of the Python callback's <c>on_fit_start</c>. Records whether DDP is in use and, for DDP,
    ///     verifies the study uses RDB storage and initializes the intermediate-value system attribute on
    ///     the global-zero process.
    /// </summary>
    /// <param name="isDistributed">
    ///     Whether the trainer runs distributed (the analogue of
    ///     <c>trainer._accelerator_connector.is_distributed</c>).
    /// </param>
    /// <param name="isGlobalZero">Whether this is the global-zero process (the analogue of <c>trainer.is_global_zero</c>).</param>
    /// <exception cref="NumSharp.ValueError">Thrown (on the DDP path) when the study does not use RDB storage.</exception>
    public void OnFitStart(bool isDistributed, bool isGlobalZero = true)
    {
        IsDdpBackend = isDistributed;
        if (!IsDdpBackend)
            return;

        // Python additionally requires PyTorch Lightning >= 1.6.0 in DDP; there is no PyTorch Lightning
        // on .NET, so that version gate has no analogue here.

        // DDP needs shared, persistent storage so worker processes see each other's writes. Python
        // checks isinstance(storage, _CachedStorage) and isinstance(storage._backend, RDBStorage); in
        // OptunaSharp the RDB backends (Sqlite / EF Core) are CachedStorage over ICachedStorageBackend
        // and live in satellite packages, so Core matches the intent by that seam (in-memory storage is
        // not a CachedStorage and so is correctly rejected).
        if (_trial.Study.Storage is not CachedStorage cached || cached.Backend is not ICachedStorageBackend)
            throw new ValueError(
                "optuna_integration.PyTorchLightningPruningCallback" +
                " supports only optuna.storages.RDBStorage in DDP.");

        // Intermediate values are stored directly in the backend storage because they are not properly
        // propagated to the main process due to cached storage.
        if (isGlobalZero)
            _trial.Storage.SetTrialSystemAttr(_trial.TrialId, IntermediateValueKey, new OrderedDict<string, double>());
    }

    /// <summary>
    ///     Report the validation's monitored metric and prune if unpromising. Port of the Python
    ///     callback's <c>on_validation_end</c>.
    /// </summary>
    /// <param name="sanityChecking">
    ///     Whether the trainer is running its sanity check (the analogue of <c>trainer.sanity_checking</c>);
    ///     when <c>true</c> the call is a no-op, avoiding a spurious report at epoch 0.
    /// </param>
    /// <param name="callbackMetrics">
    ///     The trainer's callback metrics (the analogue of <c>trainer.callback_metrics</c>) — the monitor
    ///     is read from here. If the monitor is absent, a warning is emitted and the call is a no-op.
    /// </param>
    /// <param name="currentEpoch">The module's current epoch (the analogue of <c>pl_module.current_epoch</c>).</param>
    /// <param name="distributed">
    ///     The DDP coordination seam, required when <see cref="IsDdpBackend" /> is <c>true</c> and ignored
    ///     otherwise.
    /// </param>
    /// <exception cref="TrialPruned">Thrown (non-DDP) when the pruner decides the trial should be pruned.</exception>
    public void OnValidationEnd(
        bool sanityChecking,
        IReadOnlyDictionary<string, double>? callbackMetrics,
        int currentEpoch,
        ILightningDistributedTrainer? distributed = null)
    {
        // Trainer calls on_validation_end for the sanity check too; skip it to avoid reporting twice at
        // epoch 0.
        if (sanityChecking)
            return;

        double? currentScore = callbackMetrics is not null && callbackMetrics.TryGetValue(Monitor, out double v)
            ? v
            : null;
        if (currentScore is null)
        {
            OptunaWarnings.Warn(
                $"The metric '{Monitor}' is not in the evaluation logs for pruning. " +
                "Please make sure you set the correct metric name.");
            return;
        }

        int epoch = currentEpoch;

        // Determine if the trial should be terminated in a single process.
        if (!IsDdpBackend)
        {
            _trial.Report(currentScore.Value, epoch);
            if (!_trial.ShouldPrune())
                return;
            throw new TrialPruned($"Trial was pruned at epoch {epoch}.");
        }

        // Determine if the trial should be terminated in a DDP.
        if (distributed is null)
            throw new InvalidOperationException(
                "OnValidationEnd requires an ILightningDistributedTrainer when IsDdpBackend is true.");

        bool shouldStop = false;
        if (distributed.IsGlobalZero)
        {
            _trial.Report(currentScore.Value, epoch);
            shouldStop = _trial.ShouldPrune();

            // Update the intermediate value in the storage.
            var systemAttrs = _trial.Storage.GetTrialSystemAttrs(_trial.TrialId);
            var intermediateValues = CopyIntermediateValues(
                systemAttrs.TryGetValue(IntermediateValueKey, out var stored) ? stored : null);
            intermediateValues[epoch.ToString(CultureInfo.InvariantCulture)] = currentScore.Value;
            _trial.Storage.SetTrialSystemAttr(_trial.TrialId, IntermediateValueKey, intermediateValues);
        }

        // Terminate every process if any world process decides to stop.
        shouldStop = distributed.Broadcast(shouldStop);
        distributed.ShouldStop = distributed.ShouldStop || shouldStop;
        if (!shouldStop)
            return;

        if (distributed.IsGlobalZero)
        {
            // Update system_attr from the global-zero process.
            _trial.Storage.SetTrialSystemAttr(_trial.TrialId, PrunedKey, true);
            _trial.Storage.SetTrialSystemAttr(_trial.TrialId, EpochKey, epoch);
        }
    }

    /// <summary>
    ///     Raise <see cref="TrialPruned" /> manually if the trial was pruned during DDP training. Port of
    ///     the Python callback's <c>check_pruned</c> — call it right after fitting. Intermediate values
    ///     are replayed from the trial's system attributes (they are not propagated between processes due
    ///     to the storage cache). If no DDP backend storage is present, this does nothing.
    /// </summary>
    /// <exception cref="TrialPruned">Thrown when the trial was pruned.</exception>
    public void CheckPruned()
    {
        // Confirm the storage is not in-memory in case this is called in a non-distributed situation by
        // mistake (in-memory storage is not a CachedStorage).
        if (_trial.Study.Storage is not CachedStorage cached)
            return;

        var systemAttrs = cached.Backend.GetTrialSystemAttrs(_trial.TrialId);
        object? isPruned = systemAttrs.TryGetValue(PrunedKey, out var p) ? p : null;
        object? intermediateValues = systemAttrs.TryGetValue(IntermediateValueKey, out var iv) ? iv : null;

        // Confirm the DDP backend is used in case this is called from a non-DDP situation by mistake.
        if (intermediateValues is null)
            return;

        foreach (var (epoch, score) in ReadIntermediateValues(intermediateValues))
            _trial.Report(score, epoch);

        if (IsTruthy(isPruned))
        {
            object? epoch = systemAttrs.TryGetValue(EpochKey, out var e) ? e : null;
            throw new TrialPruned(
                $"Trial was pruned at epoch {Convert.ToInt64(epoch, CultureInfo.InvariantCulture)}.");
        }
    }

    // Copy the stored intermediate-value mapping into a fresh insertion-ordered dict so a subsequent
    // write serializes deterministically (json coerces the int epoch keys to strings, as Python's does).
    private static OrderedDict<string, double> CopyIntermediateValues(object? stored)
    {
        var result = new OrderedDict<string, double>();
        foreach (var (epoch, score) in ReadIntermediateValues(stored))
            result[epoch.ToString(CultureInfo.InvariantCulture)] = score;
        return result;
    }

    // The stored intermediate-value mapping round-trips through JSON, so its concrete type depends on the
    // backend (an OrderedDict/Dictionary of string epoch -> numeric score). Read it generically.
    private static IEnumerable<KeyValuePair<int, double>> ReadIntermediateValues(object? stored)
    {
        switch (stored)
        {
            case null:
                yield break;
            case IEnumerable<KeyValuePair<string, object?>> generic:
                foreach (var kv in generic)
                    yield return new KeyValuePair<int, double>(ParseEpoch(kv.Key), ToDouble(kv.Value));
                break;
            case IEnumerable<KeyValuePair<string, double>> genericDouble:
                foreach (var kv in genericDouble)
                    yield return new KeyValuePair<int, double>(ParseEpoch(kv.Key), kv.Value);
                break;
            case IDictionary dict:
                foreach (DictionaryEntry entry in dict)
                    yield return new KeyValuePair<int, double>(
                        ParseEpoch(Convert.ToString(entry.Key, CultureInfo.InvariantCulture) ?? ""),
                        ToDouble(entry.Value));
                break;
        }
    }

    private static int ParseEpoch(string key) => int.Parse(key, CultureInfo.InvariantCulture);

    private static double ToDouble(object? value) => Convert.ToDouble(value, CultureInfo.InvariantCulture);

    private static bool IsTruthy(object? value) => value switch
    {
        null => false,
        bool b => b,
        _ => true,
    };
}
