using System;
using System.Collections.Generic;
using AwesomeAssertions;
using NumSharp;
using OptunaSharp.Integration;
using OptunaSharp.Pruners;
using static OptunaSharp.Tests.Integration.Pruning.PruningTestHelpers;

namespace OptunaSharp.Tests.Integration.Pruning;

/// <summary>
///     Tests for <see cref="PyTorchLightningPruningCallback" /> — the .NET port of
///     <c>optuna_integration.pytorch_lightning.PyTorchLightningPruningCallback</c>. The single-process
///     (non-DDP) path is the primary, fully exercised route: report the monitored metric on validation
///     end and raise <see cref="TrialPruned" /> when unpromising, skipping the sanity check and warning
///     on a missing metric. The DDP path requires a distributed runtime .NET lacks; its
///     in-memory-storage guard and the <see cref="PyTorchLightningPruningCallback.CheckPruned" /> no-op
///     on non-RDB storage are asserted here (they match Optuna's own DDP-requires-RDBStorage rule).
/// </summary>
[TestClass]
public class PyTorchLightningPruningCallbackTests
{
    private static Study MinStudy(BasePruner pruner)
        => Optuna.CreateStudy(direction: StudyDirection.Minimize, pruner: pruner);

    [TestMethod]
    public void OnValidationEnd_NonDdp_Prunes()
    {
        var study = MinStudy(new ThresholdPruner(upper: 0.5));
        var trial = study.Ask();
        var cb = new PyTorchLightningPruningCallback(trial, "val_loss");

        Action act = () => cb.OnValidationEnd(
            sanityChecking: false,
            callbackMetrics: new Dictionary<string, double> { ["val_loss"] = 0.9 },
            currentEpoch: 4);

        act.Should().Throw<TrialPruned>().WithMessage("*epoch 4*");
    }

    [TestMethod]
    public void OnValidationEnd_NonDdp_NoPrune_ReportsScore()
    {
        var study = MinStudy(new NopPruner());
        var trial = study.Ask();
        var cb = new PyTorchLightningPruningCallback(trial, "val_loss");

        cb.OnValidationEnd(false, new Dictionary<string, double> { ["val_loss"] = 0.3 }, currentEpoch: 2);

        var frozen = study.Tell(trial, 0.0);
        frozen.IntermediateValues.Should().ContainKey(2).WhoseValue.Should().Be(0.3);
    }

    [TestMethod]
    public void OnValidationEnd_SanityChecking_IsNoop()
    {
        var study = MinStudy(new ThresholdPruner(upper: 0.5));
        var trial = study.Ask();
        var cb = new PyTorchLightningPruningCallback(trial, "val_loss");

        // sanity_checking -> skip entirely (no report even though 0.9 would prune).
        Action act = () => cb.OnValidationEnd(
            sanityChecking: true, new Dictionary<string, double> { ["val_loss"] = 0.9 }, currentEpoch: 0);

        act.Should().NotThrow();
        study.Tell(trial, 0.0).IntermediateValues.Should().BeEmpty();
    }

    [TestMethod]
    public void OnValidationEnd_MissingMetric_WarnsAndSkips()
    {
        var study = MinStudy(new NopPruner());
        var trial = study.Ask();
        var cb = new PyTorchLightningPruningCallback(trial, "val_loss");

        IReadOnlyList<string> warnings = Array.Empty<string>();
        Action act = () => warnings = CaptureWarnings(
            () => cb.OnValidationEnd(false, new Dictionary<string, double> { ["train_loss"] = 0.1 }, 0));

        act.Should().NotThrow();
        warnings.Should().ContainSingle()
            .Which.Should().Be(
                "The metric 'val_loss' is not in the evaluation logs for pruning. " +
                "Please make sure you set the correct metric name.");
        study.Tell(trial, 0.0).IntermediateValues.Should().BeEmpty();
    }

    [TestMethod]
    public void CheckPruned_OnInMemoryStorage_IsNoop()
    {
        var study = MinStudy(new NopPruner());
        var trial = study.Ask();
        var cb = new PyTorchLightningPruningCallback(trial, "val_loss");

        // In-memory storage is not a CachedStorage, so check_pruned returns immediately (as in Optuna).
        Action act = () => cb.CheckPruned();

        act.Should().NotThrow();
    }

    [TestMethod]
    public void OnFitStart_NonDistributed_SetsIsDdpBackendFalse()
    {
        var study = MinStudy(new NopPruner());
        var trial = study.Ask();
        var cb = new PyTorchLightningPruningCallback(trial, "val_loss");

        cb.OnFitStart(isDistributed: false);

        cb.IsDdpBackend.Should().BeFalse();
    }

    [TestMethod]
    public void OnFitStart_Distributed_WithNonRdbStorage_Throws()
    {
        var study = MinStudy(new NopPruner());   // default in-memory storage — not RDB
        var trial = study.Ask();
        var cb = new PyTorchLightningPruningCallback(trial, "val_loss");

        Action act = () => cb.OnFitStart(isDistributed: true);

        act.Should().Throw<ValueError>().WithMessage("*supports only optuna.storages.RDBStorage in DDP*");
    }
}
