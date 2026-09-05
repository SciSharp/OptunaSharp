using System;
using System.Linq;
using AwesomeAssertions;
using OptunaSharp.Integration;
using OptunaSharp.Pruners;
using OptunaSharp.Samplers;

namespace OptunaSharp.Tests;

/// <summary>
///     Tests for the framework-agnostic <see cref="PruningCallback" /> — the shared essence of Optuna's
///     per-framework pruning callbacks, for iterative / epoch-based .NET training loops. The deterministic
///     <see cref="ThresholdPruner" /> (prune once a reported value crosses a bound) lets these assert both
///     the immediate (raise) and deferred (return-false + check) conventions without needing prior trials.
/// </summary>
[TestClass]
public class PruningCallbackTests
{
    [TestMethod]
    public void Report_RaisesTrialPruned_WhenPrunerPrunes()
    {
        var study = Optuna.CreateStudy(direction: StudyDirection.Minimize, pruner: new ThresholdPruner(upper: 0.5));
        var trial = study.Ask();
        var callback = new PruningCallback(trial);

        callback.Report(0.1, 0);         // within threshold -> keep going
        callback.Pruned.Should().BeFalse();

        Action act = () => callback.Report(0.9, 1);   // crosses the upper bound -> prune

        act.Should().Throw<TrialPruned>().WithMessage("*iteration 1*");
        callback.Pruned.Should().BeTrue();
    }

    [TestMethod]
    public void Report_DoesNotRaise_WhenPrunerKeepsTrial()
    {
        var study = Optuna.CreateStudy(direction: StudyDirection.Minimize, pruner: new NopPruner());
        var trial = study.Ask();
        var callback = new PruningCallback(trial);

        Action act = () =>
        {
            callback.Report(0.9, 0);
            callback.Report(0.8, 1);
        };

        act.Should().NotThrow();
        callback.Pruned.Should().BeFalse();
    }

    [TestMethod]
    public void AfterIteration_DefersPruning_ThenCheckPrunedRaises()
    {
        var study = Optuna.CreateStudy(direction: StudyDirection.Minimize, pruner: new ThresholdPruner(upper: 0.5));
        var trial = study.Ask();
        var callback = new PruningCallback(trial);

        callback.AfterIteration(0.1, 0).Should().BeTrue();     // keep going, no throw
        callback.AfterIteration(0.9, 1).Should().BeFalse();    // prune, but WITHOUT raising
        callback.Pruned.Should().BeTrue();

        Action check = () => callback.CheckPruned();
        check.Should().Throw<TrialPruned>().WithMessage("*iteration 1*");
    }

    [TestMethod]
    public void CheckPruned_IsNoop_WhenNeverPruned()
    {
        var study = Optuna.CreateStudy(direction: StudyDirection.Minimize, pruner: new NopPruner());
        var trial = study.Ask();
        var callback = new PruningCallback(trial);

        callback.AfterIteration(0.9, 0).Should().BeTrue();
        callback.Pruned.Should().BeFalse();

        Action check = () => callback.CheckPruned();
        check.Should().NotThrow();
    }

    [TestMethod]
    public void WorksInsideOptimizeLoop_PrunesUnpromisingTrials()
    {
        // End-to-end usage: an objective that reports its (constant) value each epoch through the callback.
        // With ThresholdPruner(upper: 0.5), any trial whose value exceeds 0.5 is pruned at step 0; the rest
        // complete. Deterministic — no dependence on other trials.
        var study = Optuna.CreateStudy(
            direction: StudyDirection.Minimize,
            sampler: new TPESampler(seed: 0),
            pruner: new ThresholdPruner(upper: 0.5));

        study.Optimize(trial =>
        {
            double x = trial.SuggestFloat("x", 0.0, 1.0);
            var callback = new PruningCallback(trial);
            for (int step = 0; step < 5; step++)
                callback.Report(x, step);
            return x;
        }, nTrials: 25);

        var trials = study.Trials;
        trials.Should().Contain(t => t.State == TrialState.Pruned);
        trials.Should().Contain(t => t.State == TrialState.Complete);

        foreach (var t in trials.Where(t => t.State == TrialState.Pruned))
            ((double)t.Params["x"]!).Should().BeGreaterThan(0.5);
        foreach (var t in trials.Where(t => t.State == TrialState.Complete))
            ((double)t.Params["x"]!).Should().BeLessThanOrEqualTo(0.5);
    }
}
