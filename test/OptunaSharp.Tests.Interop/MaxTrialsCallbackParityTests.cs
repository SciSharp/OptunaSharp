using System;
using AwesomeAssertions;

namespace OptunaSharp.Tests.Interop;

/// <summary>
///     Live-Optuna parity for <see cref="MaxTrialsCallback" /> (plan 4.9.1). The callback's stop
///     decision is count-driven (not RNG-driven), so real Optuna and OptunaSharp must run the exact
///     same number of trials before <c>study.stop()</c> ends a would-be 10-trial run: both the
///     default COMPLETE-count and an explicit PRUNED-count.
/// </summary>
[TestClass]
public class MaxTrialsCallbackParityTests : OptunaInteropTestBase
{
    // Typed helper so the objective binds to the single-objective Optimize overload (a bare
    // `_ => throw ...` lambda is ambiguous between the double and IReadOnlyList<double> overloads).
    private static double Prune(Trial _) => throw new TrialPruned();

    [TestMethod]
    [TestCategory("RequiresPython")]
    public void MaxTrialsCallback_CompleteCount_MatchesOptuna()
    {
        PyExec(
            "study = optuna.create_study()\n" +
            "study.optimize(lambda _: 1.0, n_trials=10, callbacks=[optuna.study.MaxTrialsCallback(5)])\n" +
            "n = len(study.trials)\n");
        long pyN = PyLong("n");

        var csStudy = Optuna.CreateStudy();
        csStudy.Optimize(
            _ => 1.0,
            nTrials: 10,
            callbacks: new[] { (Action<Study, FrozenTrial>)new MaxTrialsCallback(5) });

        pyN.Should().Be(5);
        csStudy.Trials.Count.Should().Be((int)pyN);
    }

    [TestMethod]
    [TestCategory("RequiresPython")]
    public void MaxTrialsCallback_PrunedCount_MatchesOptuna()
    {
        PyExec(
            "from optuna.trial import TrialState\n" +
            "def _obj(t):\n" +
            "    raise optuna.TrialPruned()\n" +
            "study = optuna.create_study()\n" +
            "study.optimize(_obj, n_trials=10, callbacks=[optuna.study.MaxTrialsCallback(5, states=(TrialState.PRUNED,))])\n" +
            "n = len(study.trials)\n");
        long pyN = PyLong("n");

        var csStudy = Optuna.CreateStudy();
        csStudy.Optimize(
            Prune,
            nTrials: 10,
            callbacks: new[] { (Action<Study, FrozenTrial>)new MaxTrialsCallback(5, new[] { TrialState.Pruned }) });

        pyN.Should().Be(5);
        csStudy.Trials.Count.Should().Be((int)pyN);
    }
}
