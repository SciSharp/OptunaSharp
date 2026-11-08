using System.Collections.Generic;
using System.Linq;
using AwesomeAssertions;
using OptunaSharp.Collections;
using OptunaSharp.Distributions;
using OptunaSharp.Samplers;
using OptunaSharp.SearchSpace;
using OptunaSharp.Storages;

namespace OptunaSharp.Tests.Samplers;

/// <summary>
///     Plan 13 (docs/plans/13-rustuna-derived-performance.md), item S3 for the QMC sampler and the group-decomposed
///     search space: both now scan the history through <c>Study.GetTrialsForSampler</c> (the zero-copy view in memory)
///     instead of a copied, filtered list, and QMC stops at the first match instead of sorting. Each test builds a history
///     with trials finished out of order and in every state, and compares the result with Optuna's read, done the way
///     Optuna does it: copy, filter by state, sort by number.
/// </summary>
[TestClass]
public class ZeroCopyHistoryReadTests
{
    /// <summary>
    ///     QMC's relative search space is the distributions of the lowest-numbered COMPLETE or PRUNED trial, whatever
    ///     order the trials finished in and whatever states lie before it.
    /// </summary>
    /// <param name="firstFinishedState">The state the lowest-numbered finished trial ends in.</param>
    [TestMethod]
    [DataRow(TrialState.Complete)]
    [DataRow(TrialState.Pruned)]
    public void Qmc_TheSearchSpaceComesFromTheLowestNumberedFinishedTrial(TrialState firstFinishedState)
    {
        var sampler = new QMCSampler(seed: 1, warnIndependentSampling: false);
        Study study = Optuna.CreateStudy(storage: new InMemoryStorage(), sampler: sampler);
        BaseStorage storage = study.Storage;

        int running = storage.CreateNewTrial(study.StudyId);  // number 0, never finishes
        storage.SetTrialParam(running, "a", 0.5, new FloatDistribution(0, 1));
        int failed = storage.CreateNewTrial(study.StudyId);   // number 1, fails
        storage.SetTrialParam(failed, "b", 0.5, new FloatDistribution(0, 1));
        storage.SetTrialStateValues(failed, TrialState.Fail);
        int late = storage.CreateNewTrial(study.StudyId);     // number 2, finishes after number 3
        storage.SetTrialParam(late, "x", 0.5, new FloatDistribution(0, 1));
        storage.SetTrialParam(late, "y", 2, new IntDistribution(0, 5));
        int early = storage.CreateNewTrial(study.StudyId);    // number 3, finishes first
        storage.SetTrialParam(early, "z", 0.5, new FloatDistribution(0, 2));
        storage.SetTrialStateValues(early, TrialState.Complete, new[] { 1.0 });
        storage.SetTrialStateValues(late, firstFinishedState, firstFinishedState == TrialState.Complete ? new[] { 2.0 } : null);

        FrozenTrial current = storage.GetTrial(storage.CreateNewTrial(study.StudyId));
        OrderedDict<string, BaseDistribution> space = sampler.InferRelativeSearchSpace(study, current);

        // Optuna: sorted(study._get_trials(deepcopy=False, states=(COMPLETE, PRUNED)), key=number)[0].distributions.
        FrozenTrial expected = study.GetTrials(deepcopy: false, states: new[] { TrialState.Complete, TrialState.Pruned })
            .OrderBy(t => t.Number).First();
        expected.TrialId.Should().Be(late);
        space.Keys.Should().Equal(expected.Distributions.Keys);
        foreach (var (name, distribution) in expected.Distributions)
            space[name].Should().BeSameAs(distribution);
    }

    /// <summary>
    ///     Before any trial finished, QMC still takes the union of the running trials' distributions, as Optuna does for
    ///     the initial trial, and an empty space when nothing is running either.
    /// </summary>
    [TestMethod]
    public void Qmc_BeforeAnyTrialFinished_UsesTheRunningTrials()
    {
        // Two studies: the running-trial read uses the per-thread cache, which lives until the next Ask, so one study
        // per question keeps each answer about the history built for it.
        var empty = new QMCSampler(seed: 2, warnIndependentSampling: false);
        Study bare = Optuna.CreateStudy(storage: new InMemoryStorage(), sampler: empty);
        FrozenTrial nothing = bare.Storage.GetTrial(bare.Storage.CreateNewTrial(bare.StudyId));
        empty.InferRelativeSearchSpace(bare, nothing).Should().BeEmpty();

        var sampler = new QMCSampler(seed: 2, warnIndependentSampling: false);
        Study study = Optuna.CreateStudy(storage: new InMemoryStorage(), sampler: sampler);
        BaseStorage storage = study.Storage;
        int first = storage.CreateNewTrial(study.StudyId);
        storage.SetTrialParam(first, "x", 0.5, new FloatDistribution(0, 1));
        int other = storage.CreateNewTrial(study.StudyId);
        storage.SetTrialParam(other, "x", 0.5, new FloatDistribution(0, 1));
        storage.SetTrialParam(other, "y", 0.5, new FloatDistribution(0, 1));
        FrozenTrial current = storage.GetTrial(storage.CreateNewTrial(study.StudyId));
        sampler.InferRelativeSearchSpace(study, current).Keys.Should().Equal("x", "y");
    }

    /// <summary>
    ///     The group-decomposed search space over a history with every state, parameters that come and go, and trials
    ///     finished out of order equals Optuna's decomposition of the copied, filtered history: the same groups, in the
    ///     same order, holding the same distribution objects, with and without pruned trials and with and without the
    ///     per-trial cache.
    /// </summary>
    /// <param name="includePruned">Whether pruned trials count.</param>
    /// <param name="useCache">Whether the read may use the per-thread cache.</param>
    [TestMethod]
    [DataRow(false, false)]
    [DataRow(false, true)]
    [DataRow(true, false)]
    [DataRow(true, true)]
    public void GroupDecomposed_EqualsOptunasDecompositionOfTheFilteredHistory(bool includePruned, bool useCache)
    {
        Study study = Optuna.CreateStudy(storage: new InMemoryStorage());
        BaseStorage storage = study.Storage;
        string[][] paramSets =
        {
            new[] { "a", "b" }, new[] { "a", "c" }, new[] { "d" }, new[] { "a", "b", "e" }, new[] { "c" },
            new[] { "f", "a" }, new[] { "b" }, new[] { "e", "d" },
        };
        TrialState[] finalStates =
        {
            TrialState.Complete, TrialState.Pruned, TrialState.Fail, TrialState.Complete, TrialState.Running,
            TrialState.Pruned, TrialState.Complete, TrialState.Complete,
        };
        var ids = new List<int>();
        for (int n = 0; n < paramSets.Length; n++)
        {
            int id = storage.CreateNewTrial(study.StudyId);
            ids.Add(id);
            foreach (string name in paramSets[n])
                storage.SetTrialParam(id, name, 0.5, new FloatDistribution(0, 1));
        }
        // Finish in reverse order, so the finishing order disagrees with the trial numbers.
        for (int n = paramSets.Length - 1; n >= 0; n--)
        {
            if (finalStates[n] == TrialState.Running)
                continue;
            storage.SetTrialStateValues(ids[n], finalStates[n], finalStates[n] == TrialState.Complete ? new[] { 1.0 } : null);
        }

        var space = new GroupDecomposedSearchSpace(includePruned);
        SearchSpaceGroup actual = space.Calculate(study, useCache);

        var expected = new SearchSpaceGroup();
        TrialState[] states = includePruned ? new[] { TrialState.Complete, TrialState.Pruned } : new[] { TrialState.Complete };
        foreach (FrozenTrial trial in study.GetTrials(deepcopy: false, states: states))
            expected.AddDistributions(trial.DistributionsDict);

        actual.SearchSpaces.Should().HaveCount(expected.SearchSpaces.Count);
        for (int g = 0; g < expected.SearchSpaces.Count; g++)
        {
            actual.SearchSpaces[g].Keys.Should().Equal(expected.SearchSpaces[g].Keys, $"group {g}");
            foreach (var (name, distribution) in expected.SearchSpaces[g])
                actual.SearchSpaces[g][name].Should().BeSameAs(distribution);
        }

        // A second call over the same history is the same decomposition, as Optuna's re-adding every trial is.
        SearchSpaceGroup again = space.Calculate(study, useCache);
        again.SearchSpaces.Select(s => string.Join(",", s.Keys))
            .Should().Equal(expected.SearchSpaces.Select(s => string.Join(",", s.Keys)));
    }
}
