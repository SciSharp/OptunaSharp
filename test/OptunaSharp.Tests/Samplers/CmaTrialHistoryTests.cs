using System;
using System.Collections.Generic;
using System.Linq;
using AwesomeAssertions;
using OptunaSharp.Collections;
using OptunaSharp.Distributions;
using OptunaSharp.Samplers.CmaEs;
using OptunaSharp.Storages;

namespace OptunaSharp.Tests.Samplers;

/// <summary>
///     <see cref="CmaTrialHistory" />, the incremental history <c>CmaEsSampler</c> reads on every ask (Rustuna plan 13,
///     item S2). Every assertion compares it with a naive re-implementation of Optuna's per-ask scans
///     (<c>_get_trials</c>, <c>_restore_optimizer</c>'s backward scan, <c>_get_solution_trials</c>) over the same
///     history, so "same trials, same order" is checked at every step of randomly evolving studies: trials created,
///     finished out of order, failed, pruned with and without intermediate values, stuck running.
/// </summary>
[TestClass]
public class CmaTrialHistoryTests
{
    private const string GenerationKey = "cma:generation";
    private const string OptimizerPrefix = "cma:optimizer";

    /// <summary>The sampler's completed-trial filter, as <c>CmaEsSampler.AsSamplerTrial</c> applies it.</summary>
    /// <param name="considerPruned">Whether pruned trials with reported values count.</param>
    /// <returns>The filter: the trial, a value-rewritten copy of a pruned trial, or null.</returns>
    private static Func<FrozenTrial, FrozenTrial?> Filter(bool considerPruned) => t =>
    {
        if (t.State == TrialState.Complete)
            return t;
        if (t.State == TrialState.Pruned && t.IntermediateValues.Count > 0 && considerPruned)
        {
            var copy = t.DeepCopy();
            copy.Value = t.IntermediateValues[t.IntermediateValues.Keys.Max()];
            return copy;
        }
        return null;
    };

    /// <summary>A trial snapshot with the given state, optional generation stamp and optional optimizer state.</summary>
    /// <param name="number">Trial number (also its position).</param>
    /// <param name="trialId">Storage id.</param>
    /// <param name="state">Trial state.</param>
    /// <param name="generation">Generation stamp, or null for none.</param>
    /// <param name="holdsState">Whether to add an optimizer-state chunk.</param>
    /// <param name="intermediate">Intermediate values (for pruned trials), or null.</param>
    /// <returns>A new snapshot.</returns>
    private static FrozenTrial Make(int number, int trialId, TrialState state, long? generation, bool holdsState,
        OrderedDict<long, double>? intermediate = null)
    {
        var systemAttrs = new OrderedDict<string, object?>();
        if (generation is not null)
            systemAttrs[GenerationKey] = generation.Value;
        if (holdsState)
            systemAttrs[OptimizerPrefix + ":0"] = $"state-of-{number}";
        double? value = state == TrialState.Complete ? number * 0.5 : null;
        return new FrozenTrial(number, state, value, DateTime.Now, state.IsFinished() ? DateTime.Now : null,
            new OrderedDict<string, object?> { ["x"] = (double)number },
            new OrderedDict<string, BaseDistribution> { ["x"] = new FloatDistribution(-1000, 1000) },
            new OrderedDict<string, object?>(), systemAttrs,
            intermediate ?? new OrderedDict<long, double>(), trialId);
    }

    /// <summary>The naive scans, from scratch, over the current history.</summary>
    /// <param name="history">The history.</param>
    /// <param name="filter">The completed-trial filter.</param>
    /// <returns>The completed trials in order.</returns>
    private static List<FrozenTrial> NaiveCompleted(IReadOnlyList<FrozenTrial> history, Func<FrozenTrial, FrozenTrial?> filter)
        => history.Select(filter).Where(t => t is not null).Select(t => t!).ToList();

    /// <summary>Asserts the history's three outputs equal the naive scans' (ids, values and order).</summary>
    /// <param name="history">The incremental history after a refresh.</param>
    /// <param name="list">The history list it was refreshed with.</param>
    /// <param name="filter">The completed-trial filter.</param>
    /// <param name="step">Step number, for the failure message.</param>
    private static void AssertMatchesNaive(CmaTrialHistory history, IReadOnlyList<FrozenTrial> list,
        Func<FrozenTrial, FrozenTrial?> filter, int step)
    {
        var completed = NaiveCompleted(list, filter);
        history.CompletedCount.Should().Be(completed.Count, $"step {step}: the completed count");

        FrozenTrial? naiveHolder = completed.LastOrDefault(t => t.SystemAttrs.Keys.Any(k => k.StartsWith(OptimizerPrefix, StringComparison.Ordinal)));
        (history.LatestStateHolder?.TrialId).Should().Be(naiveHolder?.TrialId, $"step {step}: the latest state holder");

        var generations = completed.Select(t => t.SystemAttrs.TryGetValue(GenerationKey, out var g) ? Convert.ToInt64(g) : -1)
            .Distinct().ToList();
        generations.Add(999);  // a generation nobody has
        foreach (long g in generations)
        {
            if (g < int.MinValue || g > int.MaxValue)
                continue;
            var naive = completed.Where(t => (t.SystemAttrs.TryGetValue(GenerationKey, out var s) ? Convert.ToInt64(s) : -1) == g).ToList();
            var actual = history.SolutionTrials((int)g);
            actual.Select(t => (t.TrialId, t.Value)).Should().Equal(naive.Select(t => (t.TrialId, t.Value)),
                $"step {step}: generation {g}'s trials, in trial order, with pruned values rewritten");
        }
    }

    /// <summary>
    ///     600 random steps of a study: new trials, attributes written while running, trials finished out of order as
    ///     complete, failed or pruned. After each step the incremental history must equal the naive scans exactly.
    /// </summary>
    /// <param name="seed">Seed of the step generator.</param>
    /// <param name="considerPruned">Whether pruned trials with reported values count, as with <c>considerPrunedTrials</c>.</param>
    [TestMethod]
    [DataRow(1, false)]
    [DataRow(2, true)]
    [DataRow(3, false)]
    [DataRow(4, true)]
    [DataRow(5, true)]
    public void RandomlyEvolvingStudies_MatchOptunasPerAskScans_AtEveryStep(int seed, bool considerPruned)
    {
        var rng = new Random(seed);
        var filter = Filter(considerPruned);
        var history = new CmaTrialHistory(filter, GenerationKey, OptimizerPrefix);
        var storage = new InMemoryStorage();
        var list = new List<FrozenTrial>();
        var unfinished = new List<int>();

        for (int step = 0; step < 600; step++)
        {
            double roll = rng.NextDouble();
            if (roll < 0.45 || unfinished.Count == 0)
            {
                // A new trial: running (or occasionally waiting, as an enqueued trial).
                int number = list.Count;
                var state = rng.NextDouble() < 0.1 ? TrialState.Waiting : TrialState.Running;
                list.Add(Make(number, 1000 + number, state, generation: rng.Next(0, 4), holdsState: false));
                unfinished.Add(number);
            }
            else if (roll < 0.55)
            {
                // A running trial gains attributes while running (in the tail, so re-read each refresh).
                int number = unfinished[rng.Next(unfinished.Count)];
                list[number] = Make(number, 1000 + number, list[number].State, generation: rng.Next(0, 4), holdsState: rng.NextDouble() < 0.3);
            }
            else
            {
                // Finish an unfinished trial, not necessarily the oldest: out-of-order completion.
                int index = rng.Next(unfinished.Count);
                int number = unfinished[index];
                unfinished.RemoveAt(index);
                double kind = rng.NextDouble();
                var state = kind < 0.6 ? TrialState.Complete : kind < 0.8 ? TrialState.Fail : TrialState.Pruned;
                OrderedDict<long, double>? intermediate = null;
                if (state == TrialState.Pruned && rng.NextDouble() < 0.7)
                    intermediate = new OrderedDict<long, double> { [1] = rng.NextDouble(), [rng.Next(2, 9)] = rng.NextDouble() };
                // Copy-on-write, as the storage does: a new snapshot replaces the old one.
                list[number] = Make(number, 1000 + number, state, generation: rng.Next(0, 4), holdsState: rng.NextDouble() < 0.25, intermediate);
            }

            history.Refresh(list, storage, studyId: 0);
            AssertMatchesNaive(history, list, filter, step);
        }
    }

    /// <summary>
    ///     A trial left running by a dead worker stops the fold: everything after it stays in the per-refresh tail, with
    ///     correct results; once it is failed (as <c>fail_stale_trials</c> does), the fold takes everything.
    /// </summary>
    [TestMethod]
    public void AStuckRunningTrial_KeepsLaterTrialsInTheTail_AndFoldsThemOnceItFinishes()
    {
        var filter = Filter(considerPruned: false);
        var history = new CmaTrialHistory(filter, GenerationKey, OptimizerPrefix);
        var storage = new InMemoryStorage();
        var list = new List<FrozenTrial> { Make(0, 0, TrialState.Running, generation: 0, holdsState: false) };
        for (int n = 1; n <= 40; n++)
            list.Add(Make(n, n, TrialState.Complete, generation: n / 10, holdsState: n % 10 == 0));

        history.Refresh(list, storage, 0);
        AssertMatchesNaive(history, list, filter, step: 0);

        // The dead worker's trial is failed (as fail_stale_trials does); the fold then takes everything.
        list[0] = Make(0, 0, TrialState.Fail, generation: 0, holdsState: false);
        history.Refresh(list, storage, 0);
        AssertMatchesNaive(history, list, filter, step: 1);
        history.WasReset.Should().BeFalse();
    }

    /// <summary>
    ///     With <c>considerPrunedTrials</c>, a pruned trial's value-rewritten copy is made once, when the trial is folded,
    ///     and the same copy is returned afterwards; Optuna made a new one on every ask.
    /// </summary>
    [TestMethod]
    public void APrunedTrialsCopy_IsMadeOnceWhenFolded_AndReusedByLaterRefreshes()
    {
        int copies = 0;
        Func<FrozenTrial, FrozenTrial?> counting = t =>
        {
            if (t.State == TrialState.Pruned)
                copies++;
            return Filter(considerPruned: true)(t);
        };
        var history = new CmaTrialHistory(counting, GenerationKey, OptimizerPrefix);
        var storage = new InMemoryStorage();
        var list = new List<FrozenTrial>
        {
            Make(0, 0, TrialState.Pruned, generation: 2, holdsState: false, new OrderedDict<long, double> { [3] = 7.5, [9] = 1.25 }),
        };

        history.Refresh(list, storage, 0);
        var first = history.SolutionTrials(2).Single();
        first.Value.Should().Be(1.25, "the value at the largest step, as Optuna rewrites it");
        for (int i = 0; i < 5; i++)
            history.Refresh(list, storage, 0);

        copies.Should().Be(1, "a folded trial is classified once");
        history.SolutionTrials(2).Single().Should().BeSameAs(first);
    }

    /// <summary>
    ///     The fold starts over, and says so, whenever the history it describes is gone: first use, another storage
    ///     object, another study id, other trials at the folded positions, or a history shorter than the fold. The
    ///     sampler drops its optimizer cache on that signal, because trial ids from another storage can repeat.
    /// </summary>
    [TestMethod]
    public void WasReset_IsTrueOnTheFirstRefresh_AnotherStorageOrStudy_OrAnotherHistory()
    {
        var filter = Filter(considerPruned: false);
        var history = new CmaTrialHistory(filter, GenerationKey, OptimizerPrefix);
        var storageA = new InMemoryStorage();
        var list = Enumerable.Range(0, 5).Select(n => Make(n, n, TrialState.Complete, 0, n == 4)).ToList();

        history.Refresh(list, storageA, 0);
        history.WasReset.Should().BeTrue("the first refresh folds from scratch");
        history.Refresh(list, storageA, 0);
        history.WasReset.Should().BeFalse("the same history continues");

        history.Refresh(list, new InMemoryStorage(), 0);
        history.WasReset.Should().BeTrue("trial ids of another storage mean nothing here");
        history.Refresh(list, storageA, 0);
        history.WasReset.Should().BeTrue("back to the first storage: the fold was replaced, so it starts over");
        history.Refresh(list, storageA, 7);
        history.WasReset.Should().BeTrue("another study");

        // Same storage and study id, but the folded positions now hold other trials (a new study under the id).
        var other = Enumerable.Range(0, 5).Select(n => Make(n, 500 + n, TrialState.Complete, 1, false)).ToList();
        history.Refresh(other, storageA, 7);
        history.WasReset.Should().BeTrue("the last folded trial's id changed");
        AssertMatchesNaive(history, other, filter, step: 0);

        // A shorter history than the fold (the study was replaced by a smaller one) also starts over.
        history.Refresh(other.Take(2).ToList(), storageA, 7);
        history.WasReset.Should().BeTrue();
        AssertMatchesNaive(history, other.Take(2).ToList(), filter, step: 1);
    }

    /// <summary>
    ///     A state holder that finished after a still-running trial sits in the tail, and is the latest holder: Optuna's
    ///     backward scan over the completed trials finds it first.
    /// </summary>
    [TestMethod]
    public void AHolderInTheTail_IsLaterThanEveryFoldedHolder()
    {
        var filter = Filter(considerPruned: false);
        var history = new CmaTrialHistory(filter, GenerationKey, OptimizerPrefix);
        var storage = new InMemoryStorage();
        var list = new List<FrozenTrial>
        {
            Make(0, 0, TrialState.Complete, 0, holdsState: true),
            Make(1, 1, TrialState.Running, 1, holdsState: false),
            Make(2, 2, TrialState.Complete, 1, holdsState: true),
        };

        history.Refresh(list, storage, 0);
        history.LatestStateHolder!.TrialId.Should().Be(2, "trial 2 finished after the stuck trial 1 and holds a state");

        list[2] = Make(2, 2, TrialState.Complete, 1, holdsState: true);
        list[1] = Make(1, 1, TrialState.Complete, 1, holdsState: false);
        history.Refresh(list, storage, 0);
        history.LatestStateHolder!.TrialId.Should().Be(2);
        AssertMatchesNaive(history, list, filter, step: 1);
    }

    /// <summary>The constructor and <see cref="CmaTrialHistory.Refresh" /> reject null arguments up front.</summary>
    [TestMethod]
    public void NullArguments_AreRejected()
    {
        Action noFilter = () => _ = new CmaTrialHistory(null!, GenerationKey, OptimizerPrefix);
        noFilter.Should().Throw<ArgumentNullException>();
        var history = new CmaTrialHistory(Filter(false), GenerationKey, OptimizerPrefix);
        Action noHistory = () => history.Refresh(null!, new InMemoryStorage(), 0);
        noHistory.Should().Throw<ArgumentNullException>();
        Action noStorage = () => history.Refresh(new List<FrozenTrial>(), null!, 0);
        noStorage.Should().Throw<ArgumentNullException>();
    }
}
