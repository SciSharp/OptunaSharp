using System;
using System.Collections.Generic;
using System.Linq;
using AwesomeAssertions;
using OptunaSharp.Samplers;

namespace OptunaSharp.Tests.Interop;

/// <summary>
///     Live parity for the derived multi-objective Pareto set (audit follow-up). The full-state
///     fuzz asserts every trial's values/states agree, but not the DERIVED <c>best_trials</c>
///     (Pareto front) selection — the non-dominated sort in <c>Internal/MultiObjective.cs</c>
///     (the first half of plan 2.10). With a seeded RandomSampler both sides produce identical
///     trials, so the set of Pareto-optimal trial numbers must match Optuna's exactly.
/// </summary>
[TestClass]
public class MultiObjectiveBestTrialsParityTests : OptunaInteropTestBase
{
    [TestMethod]
    [TestCategory("RequiresPython")]
    [DataRow(0)]
    [DataRow(7)]
    [DataRow(2026)]
    public void BestTrials_ParetoSet_Matches_LiveOptuna(int seed)
    {
        PythonSession.EnsureOrInconclusive();

        var csStudy = Optuna.CreateStudy(
            studyName: "mo",
            directions: new[] { StudyDirection.Minimize, StudyDirection.Maximize },
            sampler: new RandomSampler(seed));
        csStudy.Optimize((Func<Trial, IReadOnlyList<double>>)(t =>
        {
            double x = t.SuggestFloat("x", 0, 5);
            double y = t.SuggestFloat("y", 0, 5);
            return new[] { x + y, x - y };
        }), nTrials: 30);
        var csBest = string.Join(",", csStudy.BestTrials.Select(t => t.Number).OrderBy(n => n));

        PyExec(
            "def _oss_obj(t):\n" +
            "    x = t.suggest_float('x', 0, 5)\n" +
            "    y = t.suggest_float('y', 0, 5)\n" +
            "    return x + y, x - y\n" +
            "study = optuna.create_study(study_name='mo', directions=['minimize', 'maximize'], " +
            $"sampler=optuna.samplers.RandomSampler(seed={seed}))\n" +
            "study.optimize(_oss_obj, n_trials=30)\n" +
            "best = ','.join(str(n) for n in sorted(t.number for t in study.best_trials))\n");
        string pyBest = PyStr("best");

        csBest.Should().Be(pyBest, $"the Pareto best_trials set must match live Optuna (seed {seed})");
    }
}
