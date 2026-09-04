using System;
using System.Collections.Generic;
using System.Linq;
using AwesomeAssertions;
using NumSharp;
using OptunaSharp.Collections;
using OptunaSharp.Distributions;
using OptunaSharp.Samplers;
using OptunaSharp.Terminator;
using OptunaSharp.Visualization;

namespace OptunaSharp.Tests.Visualization;

/// <summary>
///     Unit tests for the terminator-improvement visualization data layer
///     (<see cref="TerminatorImprovement.GetInfo" />) — PORTING_PLAN Phase 3.3. Covers the control
///     flow (multi-objective guard, default improvement/error-evaluator selection, incomplete-trial
///     accumulation); the GP numeric parity of the improvement values lives in the interop
///     <c>GpTerminatorParityTests</c> / <c>VisualizationParityTests</c>.
/// </summary>
[TestClass]
public class TerminatorImprovementTests
{
    [TestMethod]
    public void GetInfo_MultiObjective_Throws()
    {
        var study = Optuna.CreateStudy(directions: new[] { StudyDirection.Minimize, StudyDirection.Minimize });
        Action act = () => TerminatorImprovement.GetInfo(study);
        act.Should().Throw<ValueError>()
            .WithMessage("*does not support multi-objective*");
    }

    [TestMethod]
    public void GetInfo_DefaultRegretBound_NoError()
    {
        // Default improvement evaluator is RegretBoundEvaluator (GP); get_error defaults to false, so
        // Errors is null and every completed prefix yields one finite improvement.
        var study = Optuna.CreateStudy(direction: StudyDirection.Minimize, sampler: new RandomSampler(0));
        for (int i = 0; i < 3; i++)
        {
            var t = study.Ask();
            double x = t.SuggestFloat("x", -5.0, 5.0);
            study.Tell(t, x * x);
        }

        var info = TerminatorImprovement.GetInfo(study);

        info.TrialNumbers.Should().Equal(0, 1, 2);
        info.Improvements.Should().HaveCount(3);
        info.Improvements.Should().OnlyContain(v => double.IsFinite(v));
        info.Errors.Should().BeNull();
    }

    [TestMethod]
    public void GetInfo_BestValueStagnation_DefaultsToStaticError()
    {
        // A BestValueStagnationEvaluator improvement evaluator selects StaticErrorEvaluator(0) as the
        // default error evaluator (so no cross-validation scores are needed). get_error=true → the
        // errors list is populated, all zeros.
        var study = Optuna.CreateStudy(direction: StudyDirection.Maximize, sampler: new RandomSampler(0));
        foreach (double v in new[] { 0.0, 1.0, 2.0, 3.0 })
            study.AddTrial(FrozenTrial.CreateTrial(value: v));

        var info = TerminatorImprovement.GetInfo(
            study, getError: true, improvementEvaluator: new BestValueStagnationEvaluator(maxStagnationTrials: 30));

        info.TrialNumbers.Should().Equal(0, 1, 2, 3);
        info.Improvements.Should().HaveCount(4);
        info.Errors.Should().NotBeNull();
        info.Errors!.Should().Equal(0.0, 0.0, 0.0, 0.0);
    }

    [TestMethod]
    public void GetInfo_SkipsLeadingIncompleteTrials()
    {
        // Trials before the first COMPLETE one contribute no data point (the completed-prefix is
        // still empty). Uses the cheap non-GP BestValueStagnationEvaluator.
        var study = Optuna.CreateStudy(direction: StudyDirection.Maximize, sampler: new RandomSampler(0));
        study.AddTrials(new[]
        {
            FrozenTrial.CreateTrial(state: TrialState.Fail),             // number 0 — no value
            FrozenTrial.CreateTrial(value: 1.0),                         // number 1 — first complete
            FrozenTrial.CreateTrial(value: 2.0),                         // number 2
        });

        var info = TerminatorImprovement.GetInfo(
            study, improvementEvaluator: new BestValueStagnationEvaluator());

        info.TrialNumbers.Should().Equal(1, 2);
        info.Improvements.Should().HaveCount(2);
        info.Errors.Should().BeNull();
    }
}
