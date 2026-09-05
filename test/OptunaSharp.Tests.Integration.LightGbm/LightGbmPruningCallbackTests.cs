using System;
using AwesomeAssertions;
using NumSharp;
using OptunaSharp.Integration.LightGbm;
using OptunaSharp.Pruners;

namespace OptunaSharp.Tests.Integration.LightGbm;

/// <summary>
///     Tests for <see cref="LightGBMPruningCallback" /> — the .NET port of
///     <c>optuna_integration.lightgbm.LightGBMPruningCallback</c>. These assert the ported control flow
///     (evaluation-result lookup, the study-direction consistency check with its verbatim messages, the
///     report interval, and report → prune → raise). This is a Phase-6 adapter: the callback is fed the
///     per-iteration evaluation log directly (ML.NET's LightGBM exposes no managed per-iteration hook).
/// </summary>
[TestClass]
public class LightGbmPruningCallbackTests
{
    private static Study NewStudy(StudyDirection direction, BasePruner pruner)
        => Optuna.CreateStudy(direction: direction, pruner: pruner);

    private static LightGbmEvaluationResult Loss(double score, string validName = "valid_0")
        => new(validName, "binary_logloss", score, IsHigherBetter: false);

    [TestMethod]
    public void Prunes_WhenValueCrossesThreshold_RaisesTrialPruned()
    {
        // Minimize + ThresholdPruner(upper): a reported value above the upper bound prunes at step 0.
        var study = NewStudy(StudyDirection.Minimize, new ThresholdPruner(upper: 0.5));
        var trial = study.Ask();
        var callback = new LightGBMPruningCallback(trial, "binary_logloss");

        Action act = () => callback.Call(0, new[] { Loss(0.9) });

        act.Should().Throw<TrialPruned>().WithMessage("*pruned at iteration 0*");
    }

    [TestMethod]
    public void DoesNotPrune_WhenValueWithinThreshold()
    {
        var study = NewStudy(StudyDirection.Minimize, new ThresholdPruner(upper: 0.5));
        var trial = study.Ask();
        var callback = new LightGBMPruningCallback(trial, "binary_logloss");

        Action act = () => callback.Call(0, new[] { Loss(0.1) });

        act.Should().NotThrow();
    }

    [TestMethod]
    public void DirectionConsistency_HigherBetterMetricInMinimizeStudy_Throws()
    {
        var study = NewStudy(StudyDirection.Minimize, new NopPruner());
        var trial = study.Ask();
        var callback = new LightGBMPruningCallback(trial, "auc");

        Action act = () => callback.Call(
            0, new[] { new LightGbmEvaluationResult("valid_0", "auc", 0.8, IsHigherBetter: true) });

        // A higher-better metric in a MINIMIZE study: the fix is to supply a to-be-minimized metric.
        act.Should().Throw<ValueError>().WithMessage("*inconsistent*minimized*");
    }

    [TestMethod]
    public void DirectionConsistency_LowerBetterMetricInMaximizeStudy_Throws()
    {
        var study = NewStudy(StudyDirection.Maximize, new NopPruner());
        var trial = study.Ask();
        var callback = new LightGBMPruningCallback(trial, "binary_logloss");

        Action act = () => callback.Call(0, new[] { Loss(0.3) });

        // A lower-better metric in a MAXIMIZE study: the fix is to supply a to-be-maximized metric.
        act.Should().Throw<ValueError>().WithMessage("*inconsistent*maximized*");
    }

    [TestMethod]
    public void MissingValidNameOrMetric_Throws()
    {
        var study = NewStudy(StudyDirection.Minimize, new NopPruner());
        var trial = study.Ask();
        var callback = new LightGBMPruningCallback(trial, "binary_logloss", validName: "valid_1");

        // The log only carries valid_0, so the target valid_1 is absent.
        Action act = () => callback.Call(0, new[] { Loss(0.3, validName: "valid_0") });

        act.Should().Throw<ValueError>().WithMessage("*valid_1*binary_logloss*not found*");
    }

    [TestMethod]
    public void MetricNameMatch_AcceptsLightGbmV4ValidPrefix()
    {
        // LightGBM v4.0.0+ prefixes the metric with "valid "; the lookup accepts both forms.
        var study = NewStudy(StudyDirection.Minimize, new NopPruner());
        var trial = study.Ask();
        var callback = new LightGBMPruningCallback(trial, "binary_logloss");

        callback.Call(
            0, new[] { new LightGbmEvaluationResult("valid_0", "valid binary_logloss", 0.3, IsHigherBetter: false) });

        var frozen = study.Tell(trial, 0.0);
        frozen.IntermediateValues.Should().ContainKey(0).WhoseValue.Should().Be(0.3);
    }

    [TestMethod]
    public void ReportInterval_OnlyReportsOnMultiples()
    {
        var study = NewStudy(StudyDirection.Minimize, new NopPruner());
        var trial = study.Ask();
        var callback = new LightGBMPruningCallback(trial, "binary_logloss", reportInterval: 2);

        // iteration 0 -> (0+1)%2 == 1 -> skipped; iteration 1 -> (1+1)%2 == 0 -> reported.
        callback.Call(0, new[] { Loss(0.9) });
        callback.Call(1, new[] { Loss(0.8) });

        var frozen = study.Tell(trial, 0.0);
        frozen.IntermediateValues.Keys.Should().BeEquivalentTo(new[] { 1 });
        frozen.IntermediateValues[1].Should().Be(0.8);
    }

    [TestMethod]
    public void CvMode_UsesValidTargetName()
    {
        var study = NewStudy(StudyDirection.Minimize, new NopPruner());
        var trial = study.Ask();
        var callback = new LightGBMPruningCallback(trial, "binary_logloss");

        // In cv mode the target validation name is "valid" (as lightgbm.cv reports), regardless of validName.
        callback.Call(0, new[] { Loss(0.2, validName: "valid") }, isCv: true);

        var frozen = study.Tell(trial, 0.0);
        frozen.IntermediateValues.Should().ContainKey(0).WhoseValue.Should().Be(0.2);
    }
}
