using System;
using System.Collections.Generic;
using System.Linq;
using AwesomeAssertions;
using OptunaSharp.Collections;
using OptunaSharp.Distributions;
using OptunaSharp.Internal;
using OptunaSharp.Samplers;

namespace OptunaSharp.Tests;

/// <summary>
///     Plan 15 (docs/plans/15-performance-pass-3.md), step K: the cached history read filters by state through a bit mask
///     into a list allocated at its final size, TPE shares one read-only filtered list per trial snapshot
///     (<see cref="Study.GetTrialsCachedReadOnly" />), and TPE reads its split groups without copies
///     (<see cref="TpeSplitCache.SplitShared" />). Each must return exactly the records, in exactly the order, of the code it
///     replaces: <c>Where(t =&gt; states.Contains(t.State)).ToList()</c> and <see cref="TpeSplitCache.Split" />.
/// </summary>
[TestClass]
public class StudyTrialFilterTests
{
    /// <summary>
    ///     Histories mixing every state, and values outside the enum, filtered by every kind of state set (arrays and
    ///     other collections, duplicates, empty, a value a mask cannot hold): the same references in the same order as the
    ///     LINQ filter, in a new list on every call.
    /// </summary>
    /// <param name="seed">Seeds the history.</param>
    [TestMethod]
    [DataRow(1)]
    [DataRow(2)]
    [DataRow(3)]
    public void FilterByState_ReturnsTheLinqFiltersRecords_InOrder(int seed)
    {
        var rng = new Random(seed);
        TrialState[] pool =
        {
            TrialState.Running, TrialState.Complete, TrialState.Pruned, TrialState.Fail, TrialState.Waiting,
            (TrialState)7, (TrialState)30, (TrialState)31, (TrialState)40,
        };
        var trials = new List<FrozenTrial>();
        for (int i = 0; i < 200; i++)
            trials.Add(Trial(i, pool[rng.Next(pool.Length)]));

        var sets = new List<IReadOnlyCollection<TrialState>>
        {
            Array.Empty<TrialState>(),
            new[] { TrialState.Complete },
            new[] { TrialState.Complete, TrialState.Pruned },
            new[] { TrialState.Pruned, TrialState.Complete, TrialState.Complete },
            new[] { TrialState.Running },
            new[] { TrialState.Complete, TrialState.Pruned, TrialState.Running },
            new[] { TrialState.Running, TrialState.Complete, TrialState.Pruned, TrialState.Fail, TrialState.Waiting },
            new[] { (TrialState)7, (TrialState)30 },
            new[] { (TrialState)31 },
            new[] { (TrialState)40, TrialState.Complete },
            new[] { (TrialState)(-1), TrialState.Pruned },
            new List<TrialState> { TrialState.Fail, TrialState.Waiting },
            new HashSet<TrialState> { TrialState.Complete, (TrialState)40 },
        };

        foreach (IReadOnlyCollection<TrialState> states in sets)
        {
            string label = string.Join(",", states.Select(s => (int)s));
            List<FrozenTrial> expected = trials.Where(t => states.Contains(t.State)).ToList();
            List<FrozenTrial> actual = Study.FilterByState(trials, states);

            actual.Should().HaveCount(expected.Count, label);
            for (int i = 0; i < expected.Count; i++)
                ReferenceEquals(actual[i], expected[i]).Should().BeTrue($"{label}: position {i}");
            ReferenceEquals(actual, trials).Should().BeFalse(label);
            ReferenceEquals(Study.FilterByState(trials, states), actual).Should().BeFalse($"{label}: a new list per call");
        }
    }

    /// <summary>
    ///     Within one trial snapshot, read-only reads of the same states share one list, whose records are those of the
    ///     copying read; a read of other states, or a snapshot dropped as a tell drops it, gives a fresh list with the
    ///     current records; a state set no mask can hold is filtered every time.
    /// </summary>
    /// <exception cref="TrialPruned">Never escapes: the objective raises it for every fifth trial, and the optimize loop records that trial as pruned.</exception>
    [TestMethod]
    public void GetTrialsCachedReadOnly_SharesOneListPerSnapshotAndStateSet()
    {
        OptunaSharp.Logging.OptunaLogging.SetVerbosity(OptunaSharp.Logging.LogLevel.Error);
        Study study = Optuna.CreateStudy(sampler: new RandomSampler(seed: 0));
        study.Optimize(t =>
        {
            double x = t.SuggestFloat("x", -1, 1);
            if (t.Number % 5 == 1)
                throw new TrialPruned();
            if (t.Number % 7 == 3)
                return double.NaN;   // a failed trial
            return x * x;
        }, nTrials: 40);

        var completeOrPruned = new[] { TrialState.Complete, TrialState.Pruned };
        study.ThreadLocal.CachedAllTrials = null;

        IReadOnlyList<FrozenTrial> first = study.GetTrialsCachedReadOnly(completeOrPruned);
        List<FrozenTrial> copying = study.GetTrialsInternal(deepcopy: false, states: completeOrPruned, useCache: true);
        first.Should().HaveCount(copying.Count);
        for (int i = 0; i < copying.Count; i++)
            ReferenceEquals(first[i], copying[i]).Should().BeTrue($"position {i}");
        first.Select(t => t.State).Should().Contain(TrialState.Pruned).And.Contain(TrialState.Complete);

        // The same set, in any order: the same list.
        ReferenceEquals(study.GetTrialsCachedReadOnly(completeOrPruned), first).Should().BeTrue();
        ReferenceEquals(study.GetTrialsCachedReadOnly(new[] { TrialState.Pruned, TrialState.Complete }), first).Should().BeTrue();

        // Another set: its own list, and the memo then answers for that set.
        IReadOnlyList<FrozenTrial> complete = study.GetTrialsCachedReadOnly(new[] { TrialState.Complete });
        complete.Should().OnlyContain(t => t.State == TrialState.Complete);
        complete.Should().HaveCount(study.GetTrials(deepcopy: false).Count(t => t.State == TrialState.Complete));
        IReadOnlyList<FrozenTrial> again = study.GetTrialsCachedReadOnly(completeOrPruned);
        ReferenceEquals(again, first).Should().BeFalse("the memo holds the last set only");
        again.Should().Equal(first, ReferenceEquals);

        // A set no mask can hold: filtered every time, never memoized.
        var odd = new[] { (TrialState)40 };
        IReadOnlyList<FrozenTrial> oddFirst = study.GetTrialsCachedReadOnly(odd);
        oddFirst.Should().BeEmpty();
        ReferenceEquals(study.GetTrialsCachedReadOnly(odd), oddFirst).Should().BeFalse();

        // A tell drops the snapshot, and with it the memo: the next read sees the new trial.
        IReadOnlyList<FrozenTrial> before = study.GetTrialsCachedReadOnly(completeOrPruned);
        Trial asked = study.Ask();
        asked.SuggestFloat("x", -1, 1);
        study.Tell(asked, 0.25);
        IReadOnlyList<FrozenTrial> after = study.GetTrialsCachedReadOnly(completeOrPruned);
        ReferenceEquals(after, before).Should().BeFalse();
        after.Should().HaveCount(before.Count + 1);
        after[^1].Number.Should().Be(asked.Number);
    }

    /// <summary>
    ///     The copy-free split returns the very groups <see cref="TpeSplitCache.Split" /> returns, computing the split once
    ///     and handing the same lists out again while the same records are split.
    /// </summary>
    [TestMethod]
    public void SplitShared_ReturnsSplitsGroups_AndReusesThem()
    {
        Study study = Optuna.CreateStudy();
        var rng = new Random(5);
        var trials = new List<FrozenTrial>();
        for (int i = 0; i < 50; i++)
            trials.Add(Trial(i, TrialState.Complete, rng.Next(10) * 0.5));

        var cache = new TpeSplitCache();
        var (sharedBelow, sharedAbove) = cache.SplitShared(study, trials, 12);
        var (copiedBelow, copiedAbove) = new TpeSplitCache().Split(study, trials, 12);
        sharedBelow.Should().Equal(copiedBelow, ReferenceEquals);
        sharedAbove.Should().Equal(copiedAbove, ReferenceEquals);

        var (againBelow, againAbove) = cache.SplitShared(study, trials, 12);
        ReferenceEquals(againBelow, sharedBelow).Should().BeTrue();
        ReferenceEquals(againAbove, sharedAbove).Should().BeTrue();
        var (splitBelow, _) = cache.Split(study, trials, 12);
        ReferenceEquals(splitBelow, sharedBelow).Should().BeFalse("Split still copies");
        splitBelow.Should().Equal(sharedBelow, ReferenceEquals);
        cache.Computed.Should().Be(1);
    }

    /// <summary>A finished-looking trial record with the given number, state and value, and nothing else.</summary>
    /// <param name="number">The trial number, also used as its id.</param>
    /// <param name="state">The state to record; values outside the enum are allowed, the filter must cope with them.</param>
    /// <param name="value">The objective value.</param>
    /// <returns>The record.</returns>
    private static FrozenTrial Trial(int number, TrialState state, double value = 1.0)
    {
        return new FrozenTrial(number, state, value, DateTime.UnixEpoch, DateTime.UnixEpoch,
            new OrderedDict<string, object?>(), new OrderedDict<string, BaseDistribution>(),
            new OrderedDict<string, object?>(), new OrderedDict<string, object?>(), new OrderedDict<long, double>(), number);
    }
}
