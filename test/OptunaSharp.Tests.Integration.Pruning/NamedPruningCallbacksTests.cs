using System;
using System.Collections.Generic;
using System.Linq;
using AwesomeAssertions;
using NumSharp;
using OptunaSharp.Integration;
using OptunaSharp.Pruners;
using static OptunaSharp.Tests.Integration.Pruning.PruningTestHelpers;

namespace OptunaSharp.Tests.Integration.Pruning;

/// <summary>
///     Usage / parity tests for the named framework pruning callbacks ported from
///     <c>optuna_integration</c> (CatBoost, XGBoost, Keras, tf.keras, TensorFlow, PyTorch Ignite, skorch,
///     fastai v2). The deterministic <see cref="ThresholdPruner" /> (prune once a reported value crosses
///     a bound) and <see cref="NopPruner" /> let these assert the ported control flow — metric lookup,
///     step semantics, report-interval, warnings and verbatim messages — without depending on other
///     trials. These are Phase-6 adapters: each callback is fed the framework's per-iteration evaluation
///     data directly (no .NET trainer exposes a managed per-iteration hook).
/// </summary>
[TestClass]
public class NamedPruningCallbacksTests
{
    private static Study MinStudy(BasePruner pruner)
        => Optuna.CreateStudy(direction: StudyDirection.Minimize, pruner: pruner);

    // ---- CatBoost (deferred: after_iteration returns false, check_pruned raises) ----

    [TestMethod]
    public void CatBoost_AfterIteration_Defers_ThenCheckPrunedRaises()
    {
        var study = MinStudy(new ThresholdPruner(upper: 0.5));
        var trial = study.Ask();
        var cb = new CatBoostPruningCallback(trial, "Logloss");

        var info = new CatBoostEvaluationInfo(
            iteration: 1, Metrics(("validation", MetricHistory(("Logloss", new[] { 0.9 })))));

        cb.AfterIteration(info).Should().BeFalse();   // step 0, value 0.9 > 0.5 -> prune, WITHOUT throwing
        cb.Pruned.Should().BeTrue();

        Action check = () => cb.CheckPruned();
        check.Should().Throw<TrialPruned>().WithMessage("*pruned at iteration 0*");
    }

    [TestMethod]
    public void CatBoost_AfterIteration_ReportsLastOfHistory_AtIterationMinusOne()
    {
        var study = MinStudy(new NopPruner());
        var trial = study.Ask();
        var cb = new CatBoostPruningCallback(trial, "Logloss");

        // iteration=2 -> step 1; the reported score is the LAST element of the history (0.1).
        var info = new CatBoostEvaluationInfo(
            iteration: 2, Metrics(("validation", MetricHistory(("Logloss", new[] { 0.9, 0.1 })))));

        cb.AfterIteration(info).Should().BeTrue();
        cb.Pruned.Should().BeFalse();

        var frozen = study.Tell(trial, 0.0);
        frozen.IntermediateValues.Should().ContainKey(1).WhoseValue.Should().Be(0.1);
    }

    [TestMethod]
    public void CatBoost_MissingValidName_Throws_WithSimpleNamespaceRepr()
    {
        var study = MinStudy(new NopPruner());
        var trial = study.Ask();
        // eval_set_index 1 -> target validation name "validation_1", which the info does not carry.
        var cb = new CatBoostPruningCallback(trial, "Logloss", evalSetIndex: 1);
        var info = new CatBoostEvaluationInfo(
            iteration: 1, Metrics(("validation", MetricHistory(("Logloss", new[] { 0.9 })))));

        Action act = () => cb.AfterIteration(info);

        // Verbatim message including the SimpleNamespace repr of the info.
        act.Should().Throw<ValueError>()
            .WithMessage("*validation_1*not found*namespace(iteration=1, metrics={'validation': {'Logloss': [0.9]}})*");
    }

    [TestMethod]
    public void CatBoost_MissingMetric_Throws()
    {
        var study = MinStudy(new NopPruner());
        var trial = study.Ask();
        var cb = new CatBoostPruningCallback(trial, "AUC");
        var info = new CatBoostEvaluationInfo(
            iteration: 1, Metrics(("validation", MetricHistory(("Logloss", new[] { 0.9 })))));

        Action act = () => cb.AfterIteration(info);

        act.Should().Throw<ValueError>().WithMessage("*metric name*AUC*not found*");
    }

    // ---- XGBoost (immediate: after_iteration raises) ----

    [TestMethod]
    public void XGBoost_AfterIteration_Prunes_RaisesTrialPruned()
    {
        var study = MinStudy(new ThresholdPruner(upper: 0.5));
        var trial = study.Ask();
        var cb = new XGBoostPruningCallback(trial, "validation-error");
        var log = new XGBoostEvalsLog().Add("validation", "error", 0.9);

        Action act = () => cb.AfterIteration(0, log);

        act.Should().Throw<TrialPruned>().WithMessage("*iteration 0*");
    }

    [TestMethod]
    public void XGBoost_AfterIteration_NoPrune_ReturnsFalse()
    {
        var study = MinStudy(new NopPruner());
        var trial = study.Ask();
        var cb = new XGBoostPruningCallback(trial, "validation-error");
        var log = new XGBoostEvalsLog().Add("validation", "error", 0.1);

        cb.AfterIteration(0, log).Should().BeFalse();   // never asks XGBoost to stop itself
    }

    [TestMethod]
    public void XGBoost_ObservationKeyMissing_Throws()
    {
        var study = MinStudy(new NopPruner());
        var trial = study.Ask();
        var cb = new XGBoostPruningCallback(trial, "validation-merror");
        var log = new XGBoostEvalsLog().Add("validation", "error", 0.1);

        Action act = () => cb.AfterIteration(0, log);

        act.Should().Throw<KeyError>();
    }

    [TestMethod]
    public void XGBoost_CvMode_ReportsFoldMean_DropsStddev()
    {
        var study = MinStudy(new NopPruner());
        var trial = study.Ask();
        var cb = new XGBoostPruningCallback(trial, "cv-error");
        var log = new XGBoostEvalsLog().AddCv("cv", "error", mean: 0.2, stddev: 0.05);

        cb.AfterIteration(0, log);

        var frozen = study.Tell(trial, 0.0);
        frozen.IntermediateValues[0].Should().Be(0.2);   // the mean, not the stddev
    }

    [TestMethod]
    public void XGBoost_Flatten_PicksObservationKeyAcrossDatasets()
    {
        var study = MinStudy(new NopPruner());
        var trial = study.Ask();
        var cb = new XGBoostPruningCallback(trial, "validation-error");
        var log = new XGBoostEvalsLog()
            .Add("train", "error", 0.1)
            .Add("validation", "error", 0.7);

        cb.AfterIteration(0, log);

        var frozen = study.Tell(trial, 0.0);
        frozen.IntermediateValues[0].Should().Be(0.7);   // "validation-error", not "train-error"
    }

    // ---- Keras (interval + missing-metric warning) ----

    [TestMethod]
    public void Keras_OnEpochEnd_Prunes()
    {
        var study = MinStudy(new ThresholdPruner(upper: 0.5));
        var trial = study.Ask();
        var cb = new KerasPruningCallback(trial, "val_loss");

        Action act = () => cb.OnEpochEnd(0, new Dictionary<string, double> { ["val_loss"] = 0.9 });

        act.Should().Throw<TrialPruned>().WithMessage("*epoch 0*");
    }

    [TestMethod]
    public void Keras_Interval_OnlyReportsOnMultiples()
    {
        var study = MinStudy(new NopPruner());
        var trial = study.Ask();
        var cb = new KerasPruningCallback(trial, "val_loss", interval: 2);

        cb.OnEpochEnd(0, new Dictionary<string, double> { ["val_loss"] = 0.9 });   // (0+1)%2 != 0 -> skip
        cb.OnEpochEnd(1, new Dictionary<string, double> { ["val_loss"] = 0.8 });   // (1+1)%2 == 0 -> report

        var frozen = study.Tell(trial, 0.0);
        frozen.IntermediateValues.Keys.Should().BeEquivalentTo(new[] { 1 });
        frozen.IntermediateValues[1].Should().Be(0.8);
    }

    [TestMethod]
    public void Keras_MissingMetric_WarnsAndSkips()
    {
        var study = MinStudy(new NopPruner());
        var trial = study.Ask();
        var cb = new KerasPruningCallback(trial, "val_loss");

        IReadOnlyList<string> warnings = Array.Empty<string>();
        Action act = () => warnings = CaptureWarnings(
            () => cb.OnEpochEnd(0, new Dictionary<string, double> { ["accuracy"] = 0.9 }));

        act.Should().NotThrow();
        warnings.Should().ContainSingle()
            .Which.Should().Be(
                "The metric 'val_loss' is not in the evaluation logs for pruning. " +
                "Please make sure you set the correct metric name.");
        study.Tell(trial, 0.0).IntermediateValues.Should().BeEmpty();
    }

    // ---- tf.keras (no interval) ----

    [TestMethod]
    public void TFKeras_OnEpochEnd_Prunes()
    {
        var study = MinStudy(new ThresholdPruner(upper: 0.5));
        var trial = study.Ask();
        var cb = new TFKerasPruningCallback(trial, "val_loss");

        Action act = () => cb.OnEpochEnd(3, new Dictionary<string, double> { ["val_loss"] = 0.9 });

        act.Should().Throw<TrialPruned>().WithMessage("*epoch 3*");
    }

    [TestMethod]
    public void TFKeras_MissingMetric_WarnsAndSkips()
    {
        var study = MinStudy(new NopPruner());
        var trial = study.Ask();
        var cb = new TFKerasPruningCallback(trial, "val_acc");

        IReadOnlyList<string> warnings = Array.Empty<string>();
        Action act = () => warnings = CaptureWarnings(
            () => cb.OnEpochEnd(0, new Dictionary<string, double> { ["loss"] = 0.1 }));

        act.Should().NotThrow();
        warnings.Should().ContainSingle().Which.Should().Contain("The metric 'val_acc' is not in the evaluation logs");
        study.Tell(trial, 0.0).IntermediateValues.Should().BeEmpty();
    }

    // ---- TensorFlow estimator hook (deprecated) ----

#pragma warning disable CS0618 // TensorFlowPruningHook is intentionally deprecated; tested for parity.
    [TestMethod]
    public void TensorFlow_AfterRun_Triggers_ReportsAndPrunes()
    {
        var study = MinStudy(new ThresholdPruner(upper: 0.5));
        var trial = study.Ask();
        var hook = new TensorFlowPruningHook(trial, "loss", runEverySteps: 1);
        var summaries = new List<KeyValuePair<long, IReadOnlyDictionary<string, double?>>>
        {
            new(0L, new Dictionary<string, double?> { ["loss"] = 0.9 }),
        };

        Action act = () => hook.AfterRun(0, () => summaries);

        act.Should().Throw<TrialPruned>().WithMessage("*iteration 0*");
    }

    [TestMethod]
    public void TensorFlow_AfterRun_RespectsStepInterval()
    {
        var study = MinStudy(new NopPruner());
        var trial = study.Ask();
        var hook = new TensorFlowPruningHook(trial, "loss", runEverySteps: 5);

        int reads = 0;
        IReadOnlyList<KeyValuePair<long, IReadOnlyDictionary<string, double?>>> Read()
        {
            reads++;
            return new List<KeyValuePair<long, IReadOnlyDictionary<string, double?>>>
            {
                new(reads - 1, new Dictionary<string, double?> { ["loss"] = 0.1 }),
            };
        }

        hook.AfterRun(0, Read);   // last-triggered null -> triggers            (reads == 1)
        hook.AfterRun(1, Read);   // 1 <  0 + 5          -> no trigger          (reads == 1)
        hook.AfterRun(2, Read);   // 2 <  0 + 5          -> no trigger          (reads == 1)
        hook.AfterRun(5, Read);   // 5 >= 0 + 5          -> triggers            (reads == 2)

        reads.Should().Be(2);
    }

    [TestMethod]
    public void TensorFlow_AfterRun_NoneMetric_ReportsNaN()
    {
        var study = MinStudy(new NopPruner());
        var trial = study.Ask();
        var hook = new TensorFlowPruningHook(trial, "loss", runEverySteps: 1);
        var summaries = new List<KeyValuePair<long, IReadOnlyDictionary<string, double?>>>
        {
            new(0L, new Dictionary<string, double?> { ["loss"] = null }),
        };

        hook.AfterRun(0, () => summaries);

        var frozen = study.Tell(trial, 0.0);
        double.IsNaN(frozen.IntermediateValues[0]).Should().BeTrue();
    }
#pragma warning restore CS0618

    // ---- PyTorch Ignite ----

    [TestMethod]
    public void Ignite_Call_Prunes_MessageUsesEpochWordOrder()
    {
        var study = MinStudy(new ThresholdPruner(upper: 0.5));
        var trial = study.Ask();
        var handler = new PyTorchIgnitePruningHandler(trial, "loss");

        Action act = () => handler.Call(new Dictionary<string, double> { ["loss"] = 0.9 }, trainerEpoch: 2);

        // Optuna's message is "Trial was pruned at {epoch} epoch." (epoch before the word).
        act.Should().Throw<TrialPruned>().WithMessage("*at 2 epoch*");
    }

    [TestMethod]
    public void Ignite_Call_MissingMetric_Throws()
    {
        var study = MinStudy(new NopPruner());
        var trial = study.Ask();
        var handler = new PyTorchIgnitePruningHandler(trial, "loss");

        Action act = () => handler.Call(new Dictionary<string, double> { ["accuracy"] = 0.9 }, 0);

        act.Should().Throw<KeyError>();
    }

    // ---- skorch ----

    [TestMethod]
    public void Skorch_OnEpochEnd_Prunes_AtLastEpoch()
    {
        var study = MinStudy(new ThresholdPruner(upper: 0.5));
        var trial = study.Ask();
        var cb = new SkorchPruningCallback(trial, "val_loss");
        var history = new IReadOnlyDictionary<string, double>[]
        {
            new Dictionary<string, double> { ["val_loss"] = 0.2 },
            new Dictionary<string, double> { ["val_loss"] = 0.9 },
        };

        Action act = () => cb.OnEpochEnd(history);

        act.Should().Throw<TrialPruned>().WithMessage("*epoch 1*");   // epoch = len(history) - 1
    }

    [TestMethod]
    public void Skorch_OnEpochEnd_EmptyHistory_IsNoop()
    {
        var study = MinStudy(new ThresholdPruner(upper: 0.5));
        var trial = study.Ask();
        var cb = new SkorchPruningCallback(trial, "val_loss");

        Action act = () => cb.OnEpochEnd(Array.Empty<IReadOnlyDictionary<string, double>>());

        act.Should().NotThrow();
    }

    [TestMethod]
    public void Skorch_OnEpochEnd_MissingMonitor_Throws()
    {
        var study = MinStudy(new NopPruner());
        var trial = study.Ask();
        var cb = new SkorchPruningCallback(trial, "val_loss");
        var history = new IReadOnlyDictionary<string, double>[]
        {
            new Dictionary<string, double> { ["train_loss"] = 0.5 },
        };

        Action act = () => cb.OnEpochEnd(history);

        act.Should().Throw<KeyError>();
    }

    // ---- fastai v2 (two-stage: CancelFitException then TrialPruned) ----

    [TestMethod]
    public void FastAI_AfterEpoch_Prunes_RaisesCancelFit_ThenAfterFitRaisesTrialPruned()
    {
        var study = MinStudy(new ThresholdPruner(upper: 0.5));
        var trial = study.Ask();
        var cb = new FastAIV2PruningCallback(trial, "valid_loss");

        Action epoch = () => cb.AfterEpoch(new[] { 0.9 }, new[] { "valid_loss" }, epoch: 0);
        epoch.Should().Throw<CancelFitException>();   // fastai stops the fit via CancelFitException

        Action fit = () => cb.AfterFit(0);
        fit.Should().Throw<TrialPruned>().WithMessage("*epoch 0*");   // then the real TrialPruned
    }

    [TestMethod]
    public void FastAI_AfterEpoch_UsesMonitorColumnIndex()
    {
        var study = MinStudy(new NopPruner());
        var trial = study.Ask();
        var cb = new FastAIV2PruningCallback(trial, "accuracy");

        // metric_names[1:] aligned with final_record: monitor "accuracy" is column index 1.
        cb.AfterEpoch(new[] { 0.3, 0.95 }, new[] { "valid_loss", "accuracy" }, epoch: 0);

        var frozen = study.Tell(trial, 0.0);
        frozen.IntermediateValues[0].Should().Be(0.95);
    }

    [TestMethod]
    public void FastAI_AfterEpoch_MonitorNotInNames_Throws()
    {
        var study = MinStudy(new NopPruner());
        var trial = study.Ask();
        var cb = new FastAIV2PruningCallback(trial, "valid_loss");

        Action act = () => cb.AfterEpoch(new[] { 0.9 }, new[] { "accuracy" }, epoch: 0);

        act.Should().Throw<ValueError>().WithMessage("*valid_loss*not in list*");
    }

    [TestMethod]
    public void FastAI_AliasType_IsFastAIV2PruningCallback_WithDefaultMonitor()
    {
        var study = MinStudy(new NopPruner());
        var trial = study.Ask();
        var cb = new FastAIPruningCallback(trial);   // default monitor "valid_loss"

        cb.Should().BeAssignableTo<FastAIV2PruningCallback>();

        cb.AfterEpoch(new[] { 0.42 }, new[] { "valid_loss" }, epoch: 0);
        study.Tell(trial, 0.0).IntermediateValues[0].Should().Be(0.42);
    }
}
