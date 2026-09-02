using System;
using System.Collections.Generic;
using System.Linq;
using AwesomeAssertions;
using NumSharp;
using OptunaSharp.Collections;
using OptunaSharp.Distributions;
using OptunaSharp.Samplers;
using OptunaSharp.Terminator;

namespace OptunaSharp.Tests.Terminator;

/// <summary>
///     Port of Optuna's own <c>tests/terminator_tests/*</c> for the Phase 2.6 surface — the
///     terminator framework and its non-GP evaluators (<see cref="StaticErrorEvaluator" />,
///     <see cref="BestValueStagnationEvaluator" />, <see cref="MedianErrorEvaluator" />,
///     <see cref="CrossValidationErrorEvaluator" />). The GP-based improvement evaluators
///     (<see cref="RegretBoundEvaluator" />, <see cref="EMMREvaluator" />) are placeholders until
///     PORTING_PLAN Phase 3.3, so their <c>Evaluate</c> paths are exercised only where Optuna's
///     tests construct them without evaluating.
/// </summary>
[TestClass]
public class TerminatorTests
{
    // Python `_StaticImprovementEvaluator` from tests/terminator_tests/test_terminator.py.
    private sealed class StaticImprovementEvaluator : BaseImprovementEvaluator
    {
        private readonly double _constant;
        public StaticImprovementEvaluator(double constant) => _constant = constant;
        public override double Evaluate(IReadOnlyList<FrozenTrial> trials, StudyDirection studyDirection)
            => _constant;
    }

    // Python `_DeterministicTerminator` from tests/terminator_tests/test_callback.py.
    private sealed class DeterministicTerminator : BaseTerminator
    {
        private readonly int _terminationTrialNumber;
        public DeterministicTerminator(int terminationTrialNumber) => _terminationTrialNumber = terminationTrialNumber;

        public override bool ShouldTerminate(Study study)
        {
            var trials = study.GetTrials(states: new[] { TrialState.Complete });
            int latestNumber = trials.Max(t => t.Number);
            return latestNumber >= _terminationTrialNumber;
        }
    }

    private static FrozenTrial CvTrial(double value, params double[] cvScores)
        => FrozenTrial.CreateTrial(
            value: value,
            @params: new OrderedDict<string, object?>(),
            distributions: new OrderedDict<string, BaseDistribution>(),
            systemAttrs: new OrderedDict<string, object?>
            {
                [CrossValidationErrorEvaluator.CrossValidationScoresKey] = cvScores.Select(s => (object?)s).ToList(),
            });

    // ------------------------------------------------------------------ Terminator (test_terminator.py)

    [TestMethod]
    public void Terminator_Init()
    {
        // A positive `min_n_trials` does not raise.
        _ = new OptunaSharp.Terminator.Terminator(minNTrials: 1);

        // A non-positive `min_n_trials` raises ValueError.
        Action nonPositive = () => new OptunaSharp.Terminator.Terminator(minNTrials: 0);
        nonPositive.Should().Throw<ValueError>();

        _ = new OptunaSharp.Terminator.Terminator(new BestValueStagnationEvaluator(), minNTrials: 1);
    }

    [TestMethod]
    public void Terminator_ShouldTerminate()
    {
        var study = Optuna.CreateStudy(sampler: new RandomSampler(0));

        // Add dummy trial because Terminator needs at least one trial.
        var trial = study.Ask();
        study.Tell(trial, 0.0);

        // Improvement is greater than error.
        new OptunaSharp.Terminator.Terminator(
                new StaticImprovementEvaluator(3), new StaticErrorEvaluator(0), minNTrials: 1)
            .ShouldTerminate(study).Should().BeFalse();

        // Improvement is less than error.
        new OptunaSharp.Terminator.Terminator(
                new StaticImprovementEvaluator(-1), new StaticErrorEvaluator(0), minNTrials: 1)
            .ShouldTerminate(study).Should().BeTrue();

        // Improvement is less than error, but the number of trials is less than `min_n_trials`.
        new OptunaSharp.Terminator.Terminator(
                new StaticImprovementEvaluator(-1), new StaticErrorEvaluator(0), minNTrials: 2)
            .ShouldTerminate(study).Should().BeFalse();

        // Improvement is equal to error.
        new OptunaSharp.Terminator.Terminator(
                new StaticImprovementEvaluator(0), new StaticErrorEvaluator(0), minNTrials: 1)
            .ShouldTerminate(study).Should().BeFalse();
    }

    // ------------------------------------------------------------------ TerminatorCallback (test_callback.py)

    [TestMethod]
    public void TerminatorCallback_StopsStudy()
    {
        const int terminationTrialNumber = 10;
        var callback = new TerminatorCallback(new DeterministicTerminator(terminationTrialNumber));

        var study = Optuna.CreateStudy(sampler: new RandomSampler(0));
        study.Optimize(_ => 0.0, nTrials: 100, callbacks: new Action<Study, FrozenTrial>[] { callback });

        study.Trials.Count.Should().Be(terminationTrialNumber + 1);
    }

    // ------------------------------------------------------------------ error evaluators (test_erroreval.py)

    [DataTestMethod]
    [DataRow(StudyDirection.Minimize)]
    [DataRow(StudyDirection.Maximize)]
    public void CrossValidationEvaluator(StudyDirection direction)
    {
        var study = Optuna.CreateStudy(direction: direction);
        double sign = direction == StudyDirection.Minimize ? 1.0 : -1.0;
        study.AddTrials(new[]
        {
            CvTrial(sign * 2.0, 1.0, -1.0),   // Second best trial with 1.0 var.
            CvTrial(sign * 1.0, 2.0, -2.0),   // Best trial with 4.0 var.
        });

        var evaluator = new CrossValidationErrorEvaluator();
        double serror = evaluator.Evaluate(study.Trials, study.Direction);

        const double expectedScale = 1.5;
        serror.Should().Be(Math.Sqrt(4.0 * expectedScale));
    }

    [DataTestMethod]
    [DataRow(StudyDirection.Minimize)]
    [DataRow(StudyDirection.Maximize)]
    public void CrossValidationEvaluator_WithoutCvScores(StudyDirection direction)
    {
        var study = Optuna.CreateStudy(direction: direction);
        // Note that the CV score is not reported with the system attr.
        study.AddTrial(FrozenTrial.CreateTrial(
            @params: new OrderedDict<string, object?>(),
            distributions: new OrderedDict<string, BaseDistribution>(),
            value: 0.0));

        var evaluator = new CrossValidationErrorEvaluator();
        Action act = () => evaluator.Evaluate(study.Trials, study.Direction);
        act.Should().Throw<ValueError>();
    }

    [DataTestMethod]
    [DataRow(StudyDirection.Minimize)]
    [DataRow(StudyDirection.Maximize)]
    public void ReportCrossValidationScores(StudyDirection direction)
    {
        var scores = new[] { 1.0, 2.0 };

        var study = Optuna.CreateStudy(direction: direction, sampler: new RandomSampler(0));
        var trial = study.Ask();
        TerminatorFunctions.ReportCrossValidationScores(trial, scores);
        study.Tell(trial, 0.0);

        var stored = study.Trials[0].SystemAttrs[CrossValidationErrorEvaluator.CrossValidationScoresKey];
        ((List<object?>)stored!).Select(Convert.ToDouble).Should().Equal(1.0, 2.0);
    }

    [DataTestMethod]
    [DataRow(StudyDirection.Minimize)]
    [DataRow(StudyDirection.Maximize)]
    public void ReportCrossValidationScores_IllegalLength(StudyDirection direction)
    {
        var study = Optuna.CreateStudy(direction: direction, sampler: new RandomSampler(0));
        var trial = study.Ask();
        Action act = () => TerminatorFunctions.ReportCrossValidationScores(trial, new[] { 1.0 });
        act.Should().Throw<ValueError>();
    }

    [DataTestMethod]
    [DataRow(StudyDirection.Minimize)]
    [DataRow(StudyDirection.Maximize)]
    public void StaticEvaluator(StudyDirection direction)
    {
        var study = Optuna.CreateStudy(direction: direction);
        study.AddTrial(CvTrial(2.0, 1.0, -1.0));

        var evaluator = new StaticErrorEvaluator(constant: 100.0);
        evaluator.Evaluate(study.Trials, study.Direction).Should().Be(100.0);
    }

    // ------------------------------------------------------------------ BestValueStagnation (test_evaluator.py)

    private static double EvalStagnation(BestValueStagnationEvaluator evaluator, StudyDirection direction, params double[] values)
        => evaluator.Evaluate(values.Select(v => FrozenTrial.CreateTrial(value: v)).ToList(), direction);

    [TestMethod]
    public void BestValueStagnation_InvalidArgument()
    {
        Action act = () => new BestValueStagnationEvaluator(maxStagnationTrials: -1);
        act.Should().Throw<ValueError>();
    }

    [TestMethod]
    public void BestValueStagnation_Evaluate()
    {
        var evaluator = new BestValueStagnationEvaluator(maxStagnationTrials: 1);

        // Monotonical improvement (best step is the latest element).
        EvalStagnation(evaluator, StudyDirection.Maximize, 0, 1, 2).Should().Be(1.0);
        EvalStagnation(evaluator, StudyDirection.Minimize, 2, 1, 0).Should().Be(1.0);
        // Jagged improvement (best step is the second element).
        EvalStagnation(evaluator, StudyDirection.Maximize, 0, 1, 0).Should().Be(0.0);
        EvalStagnation(evaluator, StudyDirection.Minimize, 1, 0, 1).Should().Be(0.0);
        // Flat improvement (best step is the first element).
        EvalStagnation(evaluator, StudyDirection.Maximize, 0, 0, 0).Should().Be(-1.0);
        EvalStagnation(evaluator, StudyDirection.Minimize, 0, 0, 0).Should().Be(-1.0);
    }

    [TestMethod]
    public void BestValueStagnation_EvaluateWithNoTrial()
    {
        // Python parametrizes this over RegretBoundEvaluator too; that evaluator rides the GP
        // stack (Phase 3.3), so only the native BestValueStagnationEvaluator is exercised here.
        Action act = () => new BestValueStagnationEvaluator().Evaluate(new List<FrozenTrial>(), StudyDirection.Maximize);
        act.Should().Throw<ValueError>();
    }

    // ------------------------------------------------------------------ MedianErrorEvaluator (test_median_erroreval.py)

    [TestMethod]
    public void MedianErrorEvaluator_Validation()
    {
        Action warmUp = () => new MedianErrorEvaluator(new EMMREvaluator(), warmUpTrials: -1);
        warmUp.Should().Throw<ValueError>();

        Action nInitial = () => new MedianErrorEvaluator(new EMMREvaluator(), nInitialTrials: 0);
        nInitial.Should().Throw<ValueError>();

        Action threshold = () => new MedianErrorEvaluator(new EMMREvaluator(), thresholdRatio: 0.0);
        threshold.Should().Throw<ValueError>();
    }

    [TestMethod]
    public void MedianErrorEvaluator_NotEnoughTrials_DoesNotTerminate()
    {
        var evaluator = new MedianErrorEvaluator(
            new BestValueStagnationEvaluator(), warmUpTrials: 0, nInitialTrials: 5, thresholdRatio: 2.0);

        var trials = Enumerable.Range(0, 3).Select(i => FrozenTrial.CreateTrial(value: i)).ToList();

        // Fewer than warm_up + n_initial complete trials → -sys.float_info.min (do not terminate).
        evaluator.Evaluate(trials, StudyDirection.Maximize).Should().Be(-2.2250738585072014e-308);
    }

    [TestMethod]
    public void MedianErrorEvaluator_PairedWithBestValueStagnation()
    {
        var study = Optuna.CreateStudy(direction: StudyDirection.Maximize, sampler: new RandomSampler(0));
        foreach (double v in new[] { 0.0, 1.0, 2.0, 3.0 })
            study.AddTrial(FrozenTrial.CreateTrial(value: v));

        var evaluator = new MedianErrorEvaluator(
            new BestValueStagnationEvaluator(maxStagnationTrials: 30),
            warmUpTrials: 0, nInitialTrials: 3, thresholdRatio: 2.0);

        // BestValueStagnation over monotone-increasing slices returns 30 for every prefix, so the
        // median criterion is 30 and the threshold is min(float_max, 30 * 2.0) = 60.
        evaluator.Evaluate(study.Trials, study.Direction).Should().Be(60.0);

        // The threshold is cached on first evaluation.
        evaluator.Evaluate(study.Trials, study.Direction).Should().Be(60.0);
    }
}
