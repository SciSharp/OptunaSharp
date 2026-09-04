using System.Collections.Generic;
using System.Linq;
using AwesomeAssertions;
using OptunaSharp.Samplers;
using OptunaSharp.Terminator;

namespace OptunaSharp.Tests.Interop;

/// <summary>
///     Live parity for the Phase 2.6 terminator surface — the framework plus its non-GP
///     evaluators. Identical deterministic trial histories are driven through real Optuna's
///     <c>optuna.terminator</c> and through OptunaSharp, and the results are required to match:
///     the full <c>should_terminate</c> decision sequence of a <see cref="Terminator" /> (with a
///     <see cref="BestValueStagnationEvaluator" /> improvement evaluator + its default
///     <c>StaticErrorEvaluator(0)</c>), the scaled cross-validation error, and the median-error
///     threshold. The GP-based improvement evaluators (RegretBound/EMMR) are covered separately in
///     <see cref="GpTerminatorParityTests" /> (Phase 3.3, tolerance parity). Values here are set
///     through <c>tell</c>, so the result is RNG-free and the float comparisons are bit-exact.
/// </summary>
[TestClass]
public class TerminatorParityTests : OptunaInteropTestBase
{
    // Deterministic, RNG-free jagged value/CV formulas — identical integer/IEEE arithmetic on
    // both sides (i >= 0, so the Python and C# `%` agree).
    private static double Value(int i) => ((i * 37 + 11) % 41) / 3.0 - 6.0;
    private static double[] CvScores(int i) => new[] { Value(i) + 0.11, Value(i) - 0.07, Value(i) + 0.03 };

    private const string PyHelpers =
        "import warnings, optuna.terminator\n" +
        "def _oss_val(i):\n" +
        "    return ((i * 37 + 11) % 41) / 3.0 - 6.0\n" +
        "def _oss_cv(i):\n" +
        "    return [_oss_val(i) + 0.11, _oss_val(i) - 0.07, _oss_val(i) + 0.03]\n";

    [TestMethod]
    [TestCategory("RequiresPython")]
    [DataRow("minimize")]
    [DataRow("maximize")]
    public void TerminatorDecisions_Match_LiveOptuna(string direction)
    {
        PythonSession.EnsureOrInconclusive();

        const int maxStag = 3;
        const int minNTrials = 5;
        const int nTrials = 15;

        // --- C# side -------------------------------------------------------------------
        var csDir = direction == "maximize" ? StudyDirection.Maximize : StudyDirection.Minimize;
        var study = Optuna.CreateStudy(direction: csDir, sampler: new RandomSampler(0));
        var terminator = new OptunaSharp.Terminator.Terminator(
            new BestValueStagnationEvaluator(maxStagnationTrials: maxStag), minNTrials: minNTrials);
        var decisions = new List<string>();
        for (int i = 0; i < nTrials; i++)
        {
            var t = study.Ask();
            study.Tell(t, Value(i));
            decisions.Add(terminator.ShouldTerminate(study) ? "1" : "0");
        }
        string csharp = string.Join(",", decisions);

        // --- Python side ----------------------------------------------------------------
        PyExec(
            PyHelpers +
            "def _oss_run_terminator(direction, max_stag, min_n, n):\n" +
            "    study = optuna.create_study(direction=direction, sampler=optuna.samplers.RandomSampler(seed=0))\n" +
            "    with warnings.catch_warnings():\n" +
            "        warnings.simplefilter('ignore')\n" +
            "        term = optuna.terminator.Terminator(\n" +
            "            improvement_evaluator=optuna.terminator.BestValueStagnationEvaluator(max_stagnation_trials=max_stag),\n" +
            "            min_n_trials=min_n)\n" +
            "        decisions = []\n" +
            "        for i in range(n):\n" +
            "            t = study.ask()\n" +
            "            study.tell(t, _oss_val(i))\n" +
            "            decisions.append('1' if term.should_terminate(study) else '0')\n" +
            "    return ','.join(decisions)\n" +
            $"result = _oss_run_terminator('{direction}', {maxStag}, {minNTrials}, {nTrials})\n");
        string python = PyStr("result");

        csharp.Should().Be(python, "every Terminator.should_terminate decision must match live Optuna");
    }

    [TestMethod]
    [TestCategory("RequiresPython")]
    [DataRow("minimize")]
    [DataRow("maximize")]
    public void CrossValidationError_Matches_LiveOptuna(string direction)
    {
        PythonSession.EnsureOrInconclusive();

        const int nTrials = 12;

        // --- C# side -------------------------------------------------------------------
        var csDir = direction == "maximize" ? StudyDirection.Maximize : StudyDirection.Minimize;
        var study = Optuna.CreateStudy(direction: csDir, sampler: new RandomSampler(0));
        for (int i = 0; i < nTrials; i++)
        {
            var t = study.Ask();
            TerminatorFunctions.ReportCrossValidationScores(t, CvScores(i));
            study.Tell(t, Value(i));
        }
        double csharp = new CrossValidationErrorEvaluator().Evaluate(study.Trials, study.Direction);

        // --- Python side ----------------------------------------------------------------
        PyExec(
            PyHelpers +
            "def _oss_run_cve(direction, n):\n" +
            "    study = optuna.create_study(direction=direction, sampler=optuna.samplers.RandomSampler(seed=0))\n" +
            "    with warnings.catch_warnings():\n" +
            "        warnings.simplefilter('ignore')\n" +
            "        for i in range(n):\n" +
            "            t = study.ask()\n" +
            "            optuna.terminator.report_cross_validation_scores(t, _oss_cv(i))\n" +
            "            study.tell(t, _oss_val(i))\n" +
            "        return float(optuna.terminator.CrossValidationErrorEvaluator().evaluate(study.trials, study.direction))\n" +
            $"result = _oss_run_cve('{direction}', {nTrials})\n");
        double python = PyFloat("result");

        csharp.Should().Be(python, "scaled cross-validation error must be bit-exact vs live Optuna");
    }

    [TestMethod]
    [TestCategory("RequiresPython")]
    [DataRow("minimize")]
    [DataRow("maximize")]
    public void MedianError_Matches_LiveOptuna(string direction)
    {
        PythonSession.EnsureOrInconclusive();

        const int warmUp = 2;
        const int nInitial = 8;
        const double thresholdRatio = 0.5;
        const int nTrials = 15;

        // --- C# side -------------------------------------------------------------------
        var csDir = direction == "maximize" ? StudyDirection.Maximize : StudyDirection.Minimize;
        var study = Optuna.CreateStudy(direction: csDir, sampler: new RandomSampler(0));
        for (int i = 0; i < nTrials; i++)
        {
            var t = study.Ask();
            study.Tell(t, Value(i));
        }
        var evaluator = new MedianErrorEvaluator(
            new BestValueStagnationEvaluator(), warmUpTrials: warmUp, nInitialTrials: nInitial, thresholdRatio: thresholdRatio);
        double csharp = evaluator.Evaluate(study.Trials, study.Direction);

        // --- Python side ----------------------------------------------------------------
        PyExec(
            PyHelpers +
            "def _oss_run_median(direction, warm, ninit, ratio, n):\n" +
            "    study = optuna.create_study(direction=direction, sampler=optuna.samplers.RandomSampler(seed=0))\n" +
            "    with warnings.catch_warnings():\n" +
            "        warnings.simplefilter('ignore')\n" +
            "        for i in range(n):\n" +
            "            t = study.ask()\n" +
            "            study.tell(t, _oss_val(i))\n" +
            "        ev = optuna.terminator.MedianErrorEvaluator(\n" +
            "            paired_improvement_evaluator=optuna.terminator.BestValueStagnationEvaluator(),\n" +
            "            warm_up_trials=warm, n_initial_trials=ninit, threshold_ratio=ratio)\n" +
            "        return float(ev.evaluate(study.trials, study.direction))\n" +
            $"result = _oss_run_median('{direction}', {warmUp}, {nInitial}, {thresholdRatio}, {nTrials})\n");
        double python = PyFloat("result");

        csharp.Should().Be(python, "median-error threshold must be bit-exact vs live Optuna");
    }
}
