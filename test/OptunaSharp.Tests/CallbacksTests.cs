using System;
using AwesomeAssertions;

namespace OptunaSharp.Tests;

/// <summary>
///     <see cref="MaxTrialsCallback" /> (plan 4.9.1) — a port of Optuna's own
///     <c>tests/test_callbacks.py: test_stop_with_MaxTrialsCallback</c>, plus the two C#-specific
///     semantics the two-constructor design pins: the default counts only
///     <see cref="TrialState.Complete" /> (Optuna's <c>states=(TrialState.COMPLETE,)</c>), and an
///     explicit <c>null</c> counts every state (Optuna's <c>states=None</c>).
/// </summary>
[TestClass]
public class CallbacksTests
{
    // The suggest-free objectives below never touch the sampler, so a plain create_study is enough.
    // Typed helper so the objective binds to the single-objective Optimize overload (a bare
    // `_ => throw ...` lambda is ambiguous between the double and IReadOnlyList<double> overloads).
    private static double Prune(Trial _) => throw new TrialPruned();

    // optuna/tests/test_callbacks.py: first half — stop once 5 trials COMPLETE (of a 10-trial run).
    [TestMethod]
    public void MaxTrialsCallback_StopsAfterNCompleteTrials()
    {
        var study = Optuna.CreateStudy();

        study.Optimize(
            _ => 1.0,
            nTrials: 10,
            callbacks: new[] { (Action<Study, FrozenTrial>)new MaxTrialsCallback(5) });

        study.Trials.Count.Should().Be(5);
    }

    // optuna/tests/test_callbacks.py: second half — count PRUNED trials (pruned_objective raises
    // TrialPruned), stop once 5 are pruned.
    [TestMethod]
    public void MaxTrialsCallback_CountsPrunedTrials()
    {
        var study = Optuna.CreateStudy();

        study.Optimize(
            Prune,
            nTrials: 10,
            callbacks: new[] { (Action<Study, FrozenTrial>)new MaxTrialsCallback(5, new[] { TrialState.Pruned }) });

        study.Trials.Count.Should().Be(5);
    }

    // The default (single-argument ctor) counts ONLY complete trials, so an all-pruning objective
    // never trips the limit — all 10 trials run.
    [TestMethod]
    public void MaxTrialsCallback_DefaultStatesIgnoresNonCompleteTrials()
    {
        var study = Optuna.CreateStudy();

        study.Optimize(
            Prune,
            nTrials: 10,
            callbacks: new[] { (Action<Study, FrozenTrial>)new MaxTrialsCallback(5) });

        study.Trials.Count.Should().Be(10);
    }

    // Explicit null == Optuna's states=None: count trials in every state. Here every trial prunes,
    // so the study stops at 5.
    [TestMethod]
    public void MaxTrialsCallback_NullStatesCountsEveryState()
    {
        var study = Optuna.CreateStudy();

        study.Optimize(
            Prune,
            nTrials: 10,
            callbacks: new[] { (Action<Study, FrozenTrial>)new MaxTrialsCallback(5, states: null) });

        study.Trials.Count.Should().Be(5);
    }
}
