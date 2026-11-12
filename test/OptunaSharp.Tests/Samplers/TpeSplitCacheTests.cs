using System;
using System.Collections.Generic;
using System.Linq;
using AwesomeAssertions;
using OptunaSharp.Distributions;
using OptunaSharp.Internal;
using OptunaSharp.Samplers;
using OptunaSharp.Storages;

namespace OptunaSharp.Tests.Samplers;

/// <summary>
///     Plan 13 (docs/plans/13-rustuna-derived-performance.md), item S5: TPE reuses its last below/above split while the
///     same trial records are split again, which independent TPE does once per parameter; and item S3 for TPE: the
///     constant-liar history read skips the trial being sampled without reading its record. The cache returns exactly
///     what <see cref="TpeTrialSplit.SplitTrials" /> returns, and it is reused only for the very same records of the
///     same study with the same size of the preferred group.
/// </summary>
[TestClass]
public class TpeSplitCacheTests
{
    /// <summary>
    ///     One- and two-objective histories with completed, pruned, failed-constraint and running trials: the cache's
    ///     split equals <see cref="TpeTrialSplit.SplitTrials" /> for every size of the preferred group, and splitting the
    ///     same records again reuses it.
    /// </summary>
    /// <param name="seed">Seeds the history.</param>
    /// <param name="objectives">The number of objectives of the history's study.</param>
    [TestMethod]
    [DataRow(1, 2)]
    [DataRow(2, 2)]
    [DataRow(3, 2)]
    [DataRow(1, 1)]
    [DataRow(2, 1)]
    [DataRow(3, 1)]
    public void TheCachedSplit_IsSplitTrials_AndTheSameRecordsReuseIt(int seed, int objectives)
    {
        Study study = History(seed, objectives, out List<FrozenTrial> trials);
        var cache = new TpeSplitCache();
        int computed = 0;
        for (int nBelow = 0; nBelow <= trials.Count(t => t.State != TrialState.Running); nBelow++)
        {
            var (below, above) = cache.Split(study, trials, nBelow);
            cache.Computed.Should().Be(++computed, "a new size of the preferred group is a new split");
            var (expectedBelow, expectedAbove) = TpeTrialSplit.SplitTrials(study, trials, nBelow);
            below.Select(t => t.Number).Should().Equal(expectedBelow.Select(t => t.Number), $"below, n_below={nBelow}");
            above.Select(t => t.Number).Should().Equal(expectedAbove.Select(t => t.Number), $"above, n_below={nBelow}");

            var (againBelow, againAbove) = cache.Split(study, trials.ToList(), nBelow);
            cache.Computed.Should().Be(computed, "the same records, in a new list, reuse the split");
            againBelow.Select(t => t.Number).Should().Equal(below.Select(t => t.Number));
            againAbove.Select(t => t.Number).Should().Equal(above.Select(t => t.Number));
        }
    }

    /// <summary>
    ///     Anything but the very same records misses: one record more or less, the records in another order, a running
    ///     trial whose record was replaced by a write after it was read, the same history shape in another study.
    /// </summary>
    [TestMethod]
    public void AnythingButTheSameRecords_SplitsAfresh()
    {
        Study study = History(4, 2, out List<FrozenTrial> trials);
        var cache = new TpeSplitCache();
        int nBelow = 3;
        cache.Split(study, trials, nBelow);
        cache.Computed.Should().Be(1);

        cache.Split(study, trials.Take(trials.Count - 1).ToList(), nBelow);
        cache.Computed.Should().Be(2, "one record fewer");
        cache.Split(study, Enumerable.Reverse(trials).ToList(), nBelow);
        cache.Computed.Should().Be(3, "another order");
        cache.Split(study, trials, nBelow);
        cache.Computed.Should().Be(4, "back to the original records: the cache holds the last split only");

        // A running trial written after it was read gets a new record (in-memory item S4 copies a handed-out record).
        FrozenTrial running = trials.First(t => t.State == TrialState.Running);
        study.Storage.SetTrialUserAttr(running.TrialId, "written", 1L);
        List<FrozenTrial> reread = study.GetTrials(deepcopy: false,
            states: new[] { TrialState.Complete, TrialState.Pruned, TrialState.Running });
        reread.Should().NotContain(t => ReferenceEquals(t, running));
        cache.Split(study, reread, nBelow);
        cache.Computed.Should().Be(5, "the running trial's record changed");

        Study other = History(4, 2, out List<FrozenTrial> otherTrials);
        cache.Split(other, otherTrials, nBelow);
        cache.Computed.Should().Be(6, "another study's records, even with the same values");
    }

    /// <summary>The lists the cache returns are the caller's: changing them leaves the next reuse intact.</summary>
    [TestMethod]
    public void TheReturnedLists_AreCopies()
    {
        Study study = History(5, 2, out List<FrozenTrial> trials);
        var cache = new TpeSplitCache();
        var (below, above) = cache.Split(study, trials, 4);
        int[] belowNumbers = below.Select(t => t.Number).ToArray();
        int[] aboveNumbers = above.Select(t => t.Number).ToArray();
        below.Clear();
        above.Reverse();

        var (againBelow, againAbove) = cache.Split(study, trials, 4);
        cache.Computed.Should().Be(1);
        againBelow.Select(t => t.Number).Should().Equal(belowNumbers);
        againAbove.Select(t => t.Number).Should().Equal(aboveNumbers);
    }

    /// <summary>Null arguments are refused before anything is cached.</summary>
    [TestMethod]
    public void NullArguments_AreRefused()
    {
        Study study = History(6, 2, out List<FrozenTrial> trials);
        var cache = new TpeSplitCache();
        Action noStudy = () => cache.Split(null!, trials, 1);
        Action noTrials = () => cache.Split(study, null!, 1);
        noStudy.Should().Throw<ArgumentNullException>();
        noTrials.Should().Throw<ArgumentNullException>();
        cache.Computed.Should().Be(0);
    }

    /// <summary>
    ///     A multi-objective study sampled by independent TPE splits its history once per trial, not once per parameter:
    ///     every parameter of a trial reads the same records. Constant liar is on (Optuna 5's default), so this also shows
    ///     the read leaves the trial being sampled out without touching its record: a read of it would change its record
    ///     at the next suggestion and make every parameter split afresh.
    /// </summary>
    [TestMethod]
    public void IndependentMultiObjectiveTpe_SplitsOncePerTrial_NotOncePerParameter()
    {
        var sampler = new TPESampler(seed: 3, nStartupTrials: 5);
        var storage = new InMemoryStorage();
        Study study = Optuna.CreateStudy(storage: storage, sampler: sampler,
            directions: new[] { StudyDirection.Minimize, StudyDirection.Minimize });
        const int trials = 25, parameters = 8;
        study.Optimize(t =>
        {
            double a = 0, b = 0;
            for (int i = 0; i < parameters; i++)
            {
                double v = t.SuggestFloat($"x{i}", -5, 5);
                a += v * v;
                b += (v - 1) * (v - 1);
            }
            return new[] { a, b };
        }, nTrials: trials);

        sampler.SplitCache.Computed.Should().Be(trials - 5, "one split per trial after the startup trials");
        study.Trials.Should().HaveCount(trials).And.OnlyContain(t => t.State == TrialState.Complete);
    }

    /// <summary>
    ///     With another worker's trial running, constant liar still sees it, and still leaves out the trial being sampled:
    ///     the history the sampler splits is every finished trial plus the other running one.
    /// </summary>
    [TestMethod]
    public void ConstantLiar_SeesOtherRunningTrials_AndLeavesOutTheCurrentOne()
    {
        var sampler = new TPESampler(seed: 4, nStartupTrials: 3);
        Study study = Optuna.CreateStudy(storage: new InMemoryStorage(), sampler: sampler,
            directions: new[] { StudyDirection.Minimize, StudyDirection.Minimize });
        study.Optimize(t => new[] { t.SuggestFloat("x", 0, 1), t.SuggestFloat("y", 0, 1) }, nTrials: 6);

        Trial otherWorker = study.Ask();
        otherWorker.SuggestFloat("x", 0, 1);
        Trial current = study.Ask();
        int before = sampler.SplitCache.Computed;
        current.SuggestFloat("x", 0, 1);
        current.SuggestFloat("y", 0, 1);

        sampler.SplitCache.Computed.Should().Be(before + 1, "both parameters of the current trial split the same records");
        // The running trial the other worker holds is in the history; the current trial is not: 6 finished + 1 running.
        var (below, above) = TpeTrialSplit.SplitTrials(study,
            study.GetTrials(deepcopy: false, states: new[] { TrialState.Complete, TrialState.Pruned, TrialState.Running })
                .Where(t => t.Number != current.Number).ToList(), TpeTrialSplit.DefaultGammaMultiObjective(6));
        (below.Count + above.Count).Should().Be(7);
        above.Should().Contain(t => t.Number == otherWorker.Number);
        study.Tell(otherWorker, new[] { 0.5, 0.5 });
        study.Tell(current, new[] { 0.5, 0.5 });
    }

    /// <summary>
    ///     A two-objective history with a pruned trial that reported a number does not split, as Optuna's TPE fails on it:
    ///     the pruned-trial score reads <c>study.direction</c>, which raises for a multi-objective study (QA-79). The cache
    ///     stores nothing then, so its previous entry is still reused.
    /// </summary>
    [TestMethod]
    public void AMultiObjectivePrunedReport_RaisesLikeOptuna_AndIsNotCached()
    {
        Study study = History(7, 2, out List<FrozenTrial> trials);
        var cache = new TpeSplitCache();
        cache.Split(study, trials, 3);
        cache.Computed.Should().Be(1);

        // A finished trial's reports cannot change, so a new pruned trial carries the report.
        int id = study.Storage.CreateNewTrial(study.StudyId);
        study.Storage.SetTrialParam(id, "x", 0.5, new FloatDistribution(0, 1));
        study.Storage.SetTrialIntermediateValue(id, 3, 0.25);
        study.Storage.SetTrialStateValues(id, TrialState.Pruned);
        List<FrozenTrial> reread = study.GetTrials(deepcopy: false,
            states: new[] { TrialState.Complete, TrialState.Pruned, TrialState.Running });

        Action split = () => cache.Split(study, reread, 3);
        split.Should().ThrowExactly<NumSharp.RuntimeError>().WithMessage(
            "A single direction cannot be retrieved from a multi-objective study. Consider using Study.directions to " +
            "retrieve a list containing all directions.");
        cache.Computed.Should().Be(1, "a split that throws is not stored");
        cache.Split(study, trials, 3);
        cache.Computed.Should().Be(1, "the previous records still reuse their split");
    }

    /// <summary>A worker instance for <see cref="SamplerConcurrency.PerWorker" /> gets a split cache of its own.</summary>
    [TestMethod]
    public void AWorkerInstance_HasItsOwnSplitCache()
    {
        var sampler = new TPESampler(seed: 1);
        var worker = (TPESampler)sampler.CreateWorkerInstance()!;
        worker.SplitCache.Should().NotBeSameAs(sampler.SplitCache);
    }

    /// <summary>
    ///     A one- or two-objective study on an in-memory storage with 14 finished trials (completed, pruned with and
    ///     without intermediate values, completed but infeasible) and 2 running ones, and the records TPE's constant-liar
    ///     read returns for it.
    /// </summary>
    /// <param name="seed">Seeds the values.</param>
    /// <param name="objectives">1 (minimize) or 2 (minimize, maximize).</param>
    /// <param name="trials">The study's completed, pruned and running records, by trial number.</param>
    /// <returns>The study.</returns>
    /// <remarks>
    ///     A two-objective history's pruned trials report NaN: it is the only report a multi-objective TPE split accepts,
    ///     because a number makes Optuna read <c>study.direction</c>, which raises (QA-79), while NaN scores
    ///     <c>(-step, inf)</c> before the direction is read. A one-objective history's reports are numbers.
    /// </remarks>
    private static Study History(int seed, int objectives, out List<FrozenTrial> trials)
    {
        var rng = new Random(seed);
        Study study = Optuna.CreateStudy(storage: new InMemoryStorage(),
            directions: objectives == 1
                ? new[] { StudyDirection.Minimize }
                : new[] { StudyDirection.Minimize, StudyDirection.Maximize });
        var storage = study.Storage;
        for (int n = 0; n < 16; n++)
        {
            int id = storage.CreateNewTrial(study.StudyId);
            storage.SetTrialParam(id, "x", rng.NextDouble(), new FloatDistribution(0, 1));
            if (n % 8 == 7)
                continue;  // stays running
            if (n % 5 == 4)
            {
                if (n % 2 == 0)
                {
                    double report = rng.NextDouble();  // drawn either way, so both histories share their other values
                    storage.SetTrialIntermediateValue(id, n, objectives == 1 ? report : double.NaN);
                }
                storage.SetTrialStateValues(id, TrialState.Pruned);
                continue;
            }
            if (n % 6 == 5)
                storage.SetTrialSystemAttr(id, "constraints", new List<object?> { 0.5 + rng.NextDouble() });
            // Values on a small grid, so ranks and ties are plentiful.
            double first = Math.Round(rng.NextDouble() * 4) / 4, second = Math.Round(rng.NextDouble() * 4) / 4;
            storage.SetTrialStateValues(id, TrialState.Complete,
                objectives == 1 ? new[] { first } : new[] { first, second });
        }
        trials = study.GetTrials(deepcopy: false,
            states: new[] { TrialState.Complete, TrialState.Pruned, TrialState.Running });
        return study;
    }
}
