using System;
using System.Collections.Generic;
using AwesomeAssertions;
using NumSharp;
using OptunaSharp.Collections;
using OptunaSharp.Distributions;
using OptunaSharp.Internal;
using OptunaSharp.Terminator;

namespace OptunaSharp.Tests.Terminator;

/// <summary>
///     Port of Optuna's own <c>tests/terminator_tests/improvement_tests/{test_evaluator,
///     test_emmr_evaluator}.py</c> for the GP-based improvement evaluators
///     (<see cref="RegretBoundEvaluator" />, <see cref="EMMREvaluator" />) — PORTING_PLAN Phase 3.3.
///     These ride the native GP stack, so the value assertions here only check finiteness / sentinel
///     paths (matching Optuna's own unit tests); numeric parity vs live Optuna lives in the interop
///     <c>GpTerminatorParityTests</c>.
/// </summary>
[TestClass]
public class GpTerminatorTests
{
    private static FrozenTrial FloatTrial(double value, string name, double low, double high, double param)
        => FrozenTrial.CreateTrial(
            value: value,
            @params: new OrderedDict<string, object?> { [name] = param },
            distributions: new OrderedDict<string, BaseDistribution> { [name] = new FloatDistribution(low, high) });

    // ------------------------------------------------------------------ RegretBoundEvaluator

    [TestMethod]
    public void RegretBound_Evaluate()
    {
        // Python test_regret_bound_evaluate — a single trial, maximization; just runs end-to-end.
        var trials = new List<FrozenTrial> { FloatTrial(0, "a", -1.0, 1.0, 0.0) };
        double regret = new RegretBoundEvaluator().Evaluate(trials, StudyDirection.Maximize);
        double.IsFinite(regret).Should().BeTrue();
    }

    [TestMethod]
    public void RegretBound_EvaluateWithNoTrial()
    {
        // Python test_evaluate_with_no_trial (the RegretBoundEvaluator branch).
        Action act = () => new RegretBoundEvaluator().Evaluate(new List<FrozenTrial>(), StudyDirection.Maximize);
        act.Should().Throw<ValueError>();
    }

    [TestMethod]
    public void RegretBound_EvaluateWithEmptyIntersectionSearchSpace()
    {
        // Python test_evaluate_with_empty_intersection_search_space.
        var trials = new List<FrozenTrial>
        {
            FrozenTrial.CreateTrial(
                value: 0,
                @params: new OrderedDict<string, object?>(),
                distributions: new OrderedDict<string, BaseDistribution>()),
        };
        Action act = () => new RegretBoundEvaluator().Evaluate(trials, StudyDirection.Maximize);
        act.Should().Throw<ValueError>();
    }

    // ------------------------------------------------------------------ EMMREvaluator

    [TestMethod]
    public void Emmr_Evaluate()
    {
        // Python test_emmr_evaluate — min_n_trials=3, three trials, maximization.
        var trials = new List<FrozenTrial>();
        for (int i = 0; i < 3; i++)
            trials.Add(FloatTrial(i, "a", -1.0, 10.0, i));

        double criterion = new EMMREvaluator(minNTrials: 3).Evaluate(trials, StudyDirection.Maximize);
        double.IsFinite(criterion).Should().BeTrue();
        (criterion < double.MaxValue).Should().BeTrue();
    }

    [TestMethod]
    public void Emmr_InvalidArgument()
    {
        // Python test_emmr_evaluate_with_invalid_argument.
        Action act = () => new EMMREvaluator(minNTrials: 1);
        act.Should().Throw<ValueError>();
    }

    [TestMethod]
    public void Emmr_InsufficientTrials()
    {
        // Python test_emmr_evaluate_with_insufficient_trials — 0 then 1 complete trial (< min_n_trials
        // default 2) returns sys.float_info.max * MARGIN_FOR_NUMARICAL_STABILITY (do not terminate).
        var evaluator = new EMMREvaluator();
        var trials = new List<FrozenTrial>();
        for (int i = 0; i < 2; i++)
        {
            double criterion = evaluator.Evaluate(trials, StudyDirection.Maximize);
            criterion.Should().Be(double.MaxValue * 0.1);
            trials.Add(FrozenTrial.CreateTrial(
                value: 0,
                @params: new OrderedDict<string, object?>(),
                distributions: new OrderedDict<string, BaseDistribution>()));
        }
    }

    [TestMethod]
    public void Emmr_EmptyIntersectionSearchSpace()
    {
        // Python test_emmr_evaluate_with_empty_intersection_search_space — three empty-space trials
        // warn (UserWarning) and return sys.float_info.max * MARGIN (the dim==0 branch).
        var trials = new List<FrozenTrial>();
        for (int i = 0; i < 3; i++)
            trials.Add(FrozenTrial.CreateTrial(
                value: 0,
                @params: new OrderedDict<string, object?>(),
                distributions: new OrderedDict<string, BaseDistribution>()));

        double criterion = 0;
        string? warning = CaptureWarning(
            WarningCategory.UserWarning,
            () => criterion = new EMMREvaluator().Evaluate(trials, StudyDirection.Maximize));

        warning.Should().NotBeNull();
        criterion.Should().Be(double.MaxValue * 0.1);
    }

    private static string? CaptureWarning(WarningCategory wanted, Action action)
    {
        string? captured = null;
        void Handler(string msg, WarningCategory cat)
        {
            if (cat == wanted && captured is null)
                captured = msg;
        }

        OptunaWarnings.WarningRaised += Handler;
        try
        {
            action();
        }
        finally
        {
            OptunaWarnings.WarningRaised -= Handler;
        }
        return captured;
    }
}
