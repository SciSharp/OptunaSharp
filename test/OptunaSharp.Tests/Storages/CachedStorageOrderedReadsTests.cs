using System;
using System.Collections.Generic;
using System.Linq;
using AwesomeAssertions;
using OptunaSharp.Distributions;
using OptunaSharp.Samplers;
using OptunaSharp.Storages;

namespace OptunaSharp.Tests.Storages;

/// <summary>
///     Plan 13 (docs/plans/13-rustuna-derived-performance.md), item S10: <see cref="CachedStorage" /> keeps its cached
///     trials in number order as well, so <see cref="CachedStorage.GetAllTrials" /> no longer sorts the cache on every read
///     and <see cref="CachedStorage.GetTrialsPrefixView" /> wraps it instead of copying. The backend here is a fake
///     database: an in-memory store that returns fresh records on every read, as a database does, and can hide trials
///     from a synchronization, as another worker's trial is hidden until a sync sees it.
/// </summary>
[TestClass]
public class CachedStorageOrderedReadsTests
{
    private static readonly StudyDirection[] Minimize = { StudyDirection.Minimize };

    /// <summary>
    ///     Over a history finished out of order and in every state, every read returns the trials the backend holds, in
    ///     number order, with and without each state filter, and with deep copies.
    /// </summary>
    [TestMethod]
    public void GetAllTrials_ReturnsTheBackendsTrialsInNumberOrder_ForEveryFilter()
    {
        var backend = new FakeDatabase();
        var cached = new CachedStorage(backend);
        int studyId = cached.CreateNewStudy(Minimize, "ordered");
        var ids = new List<int>();
        for (int n = 0; n < 30; n++)
        {
            int id = cached.CreateNewTrial(studyId);
            ids.Add(id);
            cached.SetTrialParam(id, "x", n / 30.0, new FloatDistribution(0, 1));
            if (n % 3 == 0)
                _ = cached.GetAllTrials(studyId, deepcopy: false);  // syncs land between the creates
        }
        TrialState[] finals = { TrialState.Complete, TrialState.Pruned, TrialState.Fail, TrialState.Running };
        for (int n = 29; n >= 0; n--)
        {
            TrialState state = finals[n % finals.Length];
            if (state != TrialState.Running)
                cached.SetTrialStateValues(ids[n], state, state == TrialState.Complete ? new[] { (double)n } : null);
        }

        foreach (TrialState[]? states in new[]
                 {
                     null, new[] { TrialState.Complete }, new[] { TrialState.Complete, TrialState.Pruned },
                     new[] { TrialState.Running }, new[] { TrialState.Waiting },
                 })
        {
            foreach (bool deepcopy in new[] { false, true })
            {
                List<FrozenTrial> actual = cached.GetAllTrials(studyId, deepcopy, states);
                List<FrozenTrial> expected = backend.GetAllTrials(studyId, deepcopy: true, states);
                actual.Select(Describe).Should().Equal(expected.Select(Describe),
                    $"states [{(states is null ? "all" : string.Join(",", states))}], deepcopy {deepcopy}");
            }
        }
    }

    /// <summary>
    ///     With the cache contiguous, the view is a zero-copy window holding the very records the uncopied read returns;
    ///     a later synchronization that refreshes an unfinished trial is seen through it, as the view's contract says.
    /// </summary>
    [TestMethod]
    public void TheView_IsZeroCopy_AndSeesARefreshedUnfinishedTrial()
    {
        var backend = new FakeDatabase();
        var cached = new CachedStorage(backend);
        int studyId = cached.CreateNewStudy(Minimize, "view");
        int done = cached.CreateNewTrial(studyId);
        cached.SetTrialStateValues(done, TrialState.Complete, new[] { 1.0 });
        int running = cached.CreateNewTrial(studyId);

        IReadOnlyList<FrozenTrial> view = cached.GetTrialsPrefixView(studyId);
        view.Should().BeOfType<TrialsPrefixView>();
        List<FrozenTrial> list = cached.GetAllTrials(studyId, deepcopy: false);
        view.Should().HaveCount(2);
        view[0].Should().BeSameAs(list[0]);
        view[1].Should().BeSameAs(list[1]);

        // Another worker writes the running trial; the next synchronization brings the fresher record in.
        backend.SetTrialParam(running, "y", 0.5, new FloatDistribution(0, 1));
        _ = cached.GetAllTrials(studyId, deepcopy: false);
        view[1].Params.Should().ContainKey("y");
        view[0].Should().BeSameAs(list[0], "a finished record is never replaced");
    }

    /// <summary>
    ///     A trial number not cached yet (another worker's trial a sync has not seen, while this process cached a later
    ///     one) leaves a gap: reads fall back to sorting the cache and the view to a copied list, both still in number
    ///     order; once a sync brings the missing trial in, the view is zero-copy again and holds every trial.
    /// </summary>
    [TestMethod]
    public void AGap_FallsBackToTheSortedCopy_UntilASyncFillsIt()
    {
        var backend = new FakeDatabase();
        var cached = new CachedStorage(backend);
        int studyId = cached.CreateNewStudy(Minimize, "gap");
        cached.CreateNewTrial(studyId);  // number 0
        cached.CreateNewTrial(studyId);  // number 1
        int other = backend.CreateNewTrial(studyId);  // number 2, another worker's, not seen yet
        backend.Hidden.Add(other);
        cached.CreateNewTrial(studyId);  // number 3, cached locally beside a gap at 2

        cached.GetAllTrials(studyId, deepcopy: false).Select(t => t.Number).Should().Equal(0, 1, 3);
        IReadOnlyList<FrozenTrial> fallback = cached.GetTrialsPrefixView(studyId);
        fallback.Should().NotBeOfType<TrialsPrefixView>();
        fallback.Select(t => t.Number).Should().Equal(0, 1, 3);

        backend.Hidden.Remove(other);
        IReadOnlyList<FrozenTrial> view = cached.GetTrialsPrefixView(studyId);
        view.Should().BeOfType<TrialsPrefixView>();
        view.Select(t => t.Number).Should().Equal(0, 1, 2, 3);
        cached.GetAllTrials(studyId, deepcopy: false).Select(t => t.Number).Should().Equal(0, 1, 2, 3);
    }

    /// <summary>
    ///     The genetic samplers read this view on every ask: a seeded NSGA-II study on the cached fake database is the
    ///     same study, trial by trial, as on an in-memory storage, and so is CMA-ES, which reads through the per-trial
    ///     cached list on this storage.
    /// </summary>
    /// <param name="samplerName"><c>nsga2</c> or <c>cmaes</c>.</param>
    [TestMethod]
    [DataRow("nsga2")]
    [DataRow("cmaes")]
    public void SeededStudies_OnTheCachedStorage_MatchTheInMemoryRun(string samplerName)
    {
        string Run(BaseStorage storage)
        {
            BaseSampler sampler = samplerName == "nsga2"
                ? new NSGAIISampler(populationSize: 10, seed: 7)
                : new CmaEsSampler(seed: 7);
            Study study = samplerName == "nsga2"
                ? Optuna.CreateStudy(storage: storage, sampler: sampler,
                    directions: new[] { StudyDirection.Minimize, StudyDirection.Minimize })
                : Optuna.CreateStudy(storage: storage, sampler: sampler);
            if (samplerName == "nsga2")
                study.Optimize(t => new[] { t.SuggestFloat("x", -2, 2), 1 - t.SuggestFloat("y", -2, 2) }, nTrials: 120);
            else
                study.Optimize(t => Math.Pow(t.SuggestFloat("x", -2, 2), 2) + Math.Pow(t.SuggestFloat("y", -2, 2), 2), nTrials: 120);
            return string.Join("\n", study.Trials.Select(Describe));
        }

        Run(new CachedStorage(new FakeDatabase())).Should().Be(Run(new InMemoryStorage()));
    }

    /// <summary>A record's number, state, parameters (with value bits) and objective values, as one line.</summary>
    /// <param name="trial">The record.</param>
    /// <returns>The description.</returns>
    private static string Describe(FrozenTrial trial)
        => $"{trial.Number}|{trial.State}|" +
           string.Join(",", trial.Params.Select(kv => $"{kv.Key}={(kv.Value is double d ? BitConverter.DoubleToInt64Bits(d) : kv.Value)}")) +
           "|" + string.Join(",", (trial.Values ?? (IReadOnlyList<double>)Array.Empty<double>()).Select(BitConverter.DoubleToInt64Bits));

    /// <summary>
    ///     A database stand-in for <see cref="CachedStorage" />: an in-memory store whose reads for the cache return
    ///     deep copies, as rows read from a database are new objects, and which can hide trials from a sync.
    /// </summary>
    private sealed class FakeDatabase : InMemoryStorage, ICachedStorageBackend
    {
        /// <summary>Trial ids a synchronization does not return yet, as another worker's uncommitted trials.</summary>
        internal HashSet<int> Hidden { get; } = new();

        /// <inheritdoc />
        public FrozenTrial CreateNewTrialFull(int studyId, FrozenTrial? templateTrial = null)
            => GetTrial(CreateNewTrial(studyId, templateTrial)).DeepCopy();

        /// <inheritdoc />
        public void SetTrialParamWithCachedDistribution(
            int trialId, string paramName, double paramValueInternal, BaseDistribution distribution,
            BaseDistribution? cachedDistribution)
            => SetTrialParam(trialId, paramName, paramValueInternal, distribution);

        /// <inheritdoc />
        public int GetBestTrialId(int studyId, StudyDirection direction) => GetBestTrial(studyId).TrialId;

        /// <inheritdoc />
        public List<FrozenTrial> GetTrialsPartial(
            int studyId, IReadOnlyCollection<TrialState>? states, IReadOnlyCollection<int> includedTrialIds,
            int trialIdGreaterThan)
            => GetAllTrials(studyId, deepcopy: true, states)
                .Where(t => !Hidden.Contains(t.TrialId)
                            && (includedTrialIds.Contains(t.TrialId) || t.TrialId > trialIdGreaterThan))
                .ToList();

        /// <inheritdoc />
        public void RecordHeartbeat(int trialId)
        {
        }

        /// <inheritdoc />
        public List<int> GetStaleTrialIds(int studyId) => new();

        /// <inheritdoc />
        public int? GetHeartbeatInterval() => null;

        /// <inheritdoc />
        public Action<Study, FrozenTrial>? GetHeartbeatStaleTrialCallback() => null;
    }
}
