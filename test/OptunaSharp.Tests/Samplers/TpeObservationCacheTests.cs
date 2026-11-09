using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;
using AwesomeAssertions;
using OptunaSharp.Collections;
using OptunaSharp.Distributions;
using OptunaSharp.Samplers;
using OptunaSharp.Samplers.Tpe;
using OptunaSharp.Storages;

namespace OptunaSharp.Tests.Samplers;

/// <summary>
///     Plan 14: TPE keeps finished trials' internal parameter values in a <see cref="TpeObservationCache" /> instead of
///     converting the whole history on every ask. Each test runs the same seeded study twice, once with the cache warm
///     (as sampling uses it) and once with it emptied before every suggestion (so every observation is converted from the
///     trial again, as before the cache), and asserts every trial is identical to the bit: dynamic ranges (one name, many
///     distributions), conditional parameters (finished trials lacking a name), pruned trials, constant liar with trials
///     still running, both TPE modes, two objectives and compact storage.
/// </summary>
[TestClass]
public class TpeObservationCacheTests
{
    /// <summary>Every trial's number, state, parameters (value bits) and values (bits), one line per trial.</summary>
    /// <param name="study">The finished study.</param>
    /// <returns>The study's trials as text.</returns>
    private static string Dump(Study study)
    {
        var sb = new StringBuilder();
        foreach (var t in study.Trials)
        {
            sb.Append(t.Number).Append(' ').Append(t.State).Append(' ');
            foreach (var (k, v) in t.Params)
                sb.Append(k).Append('=').Append(v is double d ? BitConverter.DoubleToInt64Bits(d).ToString("X16") : v?.ToString()).Append(';');
            if (t.Values is not null)
                foreach (double v in t.Values)
                    sb.Append(' ').Append(BitConverter.DoubleToInt64Bits(v).ToString("X16"));
            sb.Append('\n');
        }
        return sb.ToString();
    }

    /// <summary>
    ///     The objective with a dynamic range, a conditional parameter and pruning; <paramref name="beforeSuggest" /> runs
    ///     before each suggestion (the cold run empties the cache there).
    /// </summary>
    /// <param name="t">The trial.</param>
    /// <param name="beforeSuggest">Called before each suggestion.</param>
    /// <returns>The objective value.</returns>
    /// <exception cref="TrialPruned">Every seventh trial is pruned after reporting its value.</exception>
    private static double Objective(Trial t, Action beforeSuggest)
    {
        double span = 5.0 + t.Number % 3;
        beforeSuggest();
        double x = t.SuggestFloat("x", -span, span);
        beforeSuggest();
        string c = t.SuggestCategorical("c", new[] { "a", "b", "c" });
        double y = 0.0;
        if (c == "a")
        {
            beforeSuggest();
            y = t.SuggestFloat("y", 1e-3, 10.0, log: true);
        }
        beforeSuggest();
        long k = t.SuggestInt("k", 0, 20, step: 2);
        double value = x * x + y + k / 10.0 + (c == "b" ? 0.0 : 1.0);
        t.Report(value, 0);
        if (t.Number % 7 == 3)
            throw new TrialPruned();
        return value;
    }

    /// <summary>The study run warm and cold must be identical, for both TPE modes, with plain and compact storage.</summary>
    /// <param name="multivariate">Whether TPE models the parameters jointly.</param>
    /// <param name="compact">Whether the storage keeps finished trials compact (plan 13, item S12).</param>
    [TestMethod]
    [DataRow(false, false)]
    [DataRow(true, false)]
    [DataRow(false, true)]
    [DataRow(true, true)]
    public void WarmCache_EqualsColdCache_DynamicConditionalPruned(bool multivariate, bool compact)
    {
        string Run(bool cold)
        {
            var sampler = new TPESampler(seed: 21, multivariate: multivariate, nStartupTrials: 5);
            var study = Optuna.CreateStudy(storage: new InMemoryStorage(compactFinishedTrials: compact), sampler: sampler);
            Action before = cold ? () => sampler.ObservationCache.Reset() : () => { };
            study.Optimize(t => Objective(t, before), nTrials: 120);
            return Dump(study);
        }

        Run(cold: false).Should().Be(Run(cold: true));
    }

    /// <summary>Two objectives (MOTPE), warm and cold, must be identical.</summary>
    [TestMethod]
    public void WarmCache_EqualsColdCache_MultiObjective()
    {
        string Run(bool cold)
        {
            var sampler = new TPESampler(seed: 5, nStartupTrials: 5);
            var study = Optuna.CreateStudy(sampler: sampler, directions: new[] { StudyDirection.Minimize, StudyDirection.Minimize });
            study.Optimize(t =>
            {
                double a = 0, b = 0;
                for (int i = 0; i < 6; i++)
                {
                    if (cold)
                        sampler.ObservationCache.Reset();
                    double v = t.SuggestFloat("x" + i, -3, 3);
                    a += v * v;
                    b += (v - 1) * (v - 1);
                }
                return new[] { a, b };
            }, nTrials: 100);
            return Dump(study);
        }

        Run(cold: false).Should().Be(Run(cold: true));
    }

    /// <summary>
    ///     Constant liar with trials still running (ask three, then tell three): running trials must never be cached, and
    ///     warm and cold runs must be identical in both TPE modes.
    /// </summary>
    /// <param name="multivariate">Whether TPE models the parameters jointly.</param>
    [TestMethod]
    [DataRow(false)]
    [DataRow(true)]
    public void WarmCache_EqualsColdCache_ConstantLiarWithRunningTrials(bool multivariate)
    {
        string Run(bool cold)
        {
            var sampler = new TPESampler(seed: 9, multivariate: multivariate, constantLiar: true, nStartupTrials: 4);
            var study = Optuna.CreateStudy(sampler: sampler);
            for (int round = 0; round < 25; round++)
            {
                var batch = new List<Trial>();
                for (int i = 0; i < 3; i++)
                {
                    var t = study.Ask();
                    if (cold)
                        sampler.ObservationCache.Reset();
                    t.SuggestFloat("x", -5, 5);
                    if (cold)
                        sampler.ObservationCache.Reset();
                    t.SuggestCategorical("c", new[] { "u", "v", "w" });
                    batch.Add(t);
                }
                foreach (var t in batch)
                {
                    var p = study.Storage.GetTrial(t.TrialId).Params;
                    study.Tell(t, (double)p["x"]! * (double)p["x"]! + ((string)p["c"]! == "u" ? 0.0 : 1.0));
                }
            }
            return Dump(study);
        }

        Run(cold: false).Should().Be(Run(cold: true));
    }

    /// <summary>
    ///     A range that changes on every trial makes a new column per distribution; the cache must stay within
    ///     <see cref="TpeObservationCache.MaxColumns" /> by starting over, and the study must equal the cold run.
    /// </summary>
    [TestMethod]
    public void EverChangingRange_StaysBounded_AndEqualsColdCache()
    {
        int maxSeen = 0;
        string Run(bool cold)
        {
            var sampler = new TPESampler(seed: 4, nStartupTrials: 3);
            var study = Optuna.CreateStudy(sampler: sampler);
            study.Optimize(t =>
            {
                if (cold)
                    sampler.ObservationCache.Reset();
                double x = t.SuggestFloat("x", 0, 1 + t.Number);
                maxSeen = Math.Max(maxSeen, sampler.ObservationCache.ColumnCount);
                return x;
            }, nTrials: 400);
            return Dump(study);
        }

        Run(cold: false).Should().Be(Run(cold: true));
        maxSeen.Should().BeLessThanOrEqualTo(TpeObservationCache.MaxColumns);
    }

    /// <summary>
    ///     One sampler reused for a second study (trial numbers start over) gives the second study exactly what a new sampler
    ///     with the same seed state would: the cache must start over when the study changes.
    /// </summary>
    [TestMethod]
    public void SecondStudy_StartsTheCacheOver()
    {
        var reused = new TPESampler(seed: 8, nStartupTrials: 3);
        var first = Optuna.CreateStudy(sampler: reused);
        first.Optimize(t => Math.Pow(t.SuggestFloat("x", -2, 2) - 1, 2), nTrials: 40);
        var second = Optuna.CreateStudy(sampler: reused);
        second.Optimize(t => Math.Pow(t.SuggestFloat("x", -2, 2) + 1, 2), nTrials: 40);

        // The same sequence with the cache emptied before every suggestion: identical, so nothing carried over.
        var cold = new TPESampler(seed: 8, nStartupTrials: 3);
        var coldFirst = Optuna.CreateStudy(sampler: cold);
        coldFirst.Optimize(t => { cold.ObservationCache.Reset(); return Math.Pow(t.SuggestFloat("x", -2, 2) - 1, 2); }, nTrials: 40);
        var coldSecond = Optuna.CreateStudy(sampler: cold);
        coldSecond.Optimize(t => { cold.ObservationCache.Reset(); return Math.Pow(t.SuggestFloat("x", -2, 2) + 1, 2); }, nTrials: 40);

        Dump(first).Should().Be(Dump(coldFirst));
        Dump(second).Should().Be(Dump(coldSecond));
    }

    /// <summary>
    ///     A live study never makes the cache start over on a trial's identity: every finished trial it reads again is the
    ///     trial its row was read from. A false alarm would not change results, only throw the cache away, so this guards
    ///     the speed-up: both TPE modes, plain and compact storage, dynamic ranges, a conditional parameter, pruned trials,
    ///     then constant liar with trials still running (asked three at a time, told three at a time).
    /// </summary>
    /// <param name="multivariate">Whether TPE models the parameters jointly.</param>
    /// <param name="compact">Whether the storage keeps finished trials compact (plan 13, item S12).</param>
    [TestMethod]
    [DataRow(false, false)]
    [DataRow(true, false)]
    [DataRow(false, true)]
    [DataRow(true, true)]
    public void LiveStudy_NeverStartsOverOnATrialsIdentity(bool multivariate, bool compact)
    {
        var sampler = new TPESampler(seed: 21, multivariate: multivariate, nStartupTrials: 5, constantLiar: true);
        var study = Optuna.CreateStudy(storage: new InMemoryStorage(compactFinishedTrials: compact), sampler: sampler);
        study.Optimize(t => Objective(t, () => { }), nTrials: 80);
        for (int round = 0; round < 10; round++)
        {
            var batch = new List<Trial>();
            for (int i = 0; i < 3; i++)
            {
                var t = study.Ask();
                t.SuggestFloat("x", -5, 5);
                t.SuggestCategorical("c", new[] { "a", "b", "c" });
                batch.Add(t);
            }
            foreach (var t in batch)
                study.Tell(t, 1.0);
        }

        sampler.ObservationCache.IdentityRestarts.Should().Be(0, "every trial number still carries the trial cached under it");
        sampler.ObservationCache.ColumnCount.Should().BeGreaterThan(0, "the run read its history through the cache");
    }

    /// <summary>
    ///     The identity rule on its own. The end-to-end case (a study deleted and recreated under a reused SQLite id behind
    ///     a gRPC proxy) is <c>GrpcStorageProxyTests.Tpe_ReusedOnARecreatedStudyUnderTheSameId_SamplesAsUnderANewId</c> in
    ///     the storage suite. With the same storage object and study id, a trial carrying a cached number must be the trial
    ///     the row was read from: the same storage id with a later start (SQLite reuses trial ids too) or another storage
    ///     id makes the cache start over before anything is read, while an unknown start time is not a difference.
    /// </summary>
    [TestMethod]
    public void TrialNumbersCarryingOtherTrials_StartTheCacheOver()
    {
        var sampler = new TPESampler(seed: 3, nStartupTrials: 3, multivariate: false);
        var study = Optuna.CreateStudy(sampler: sampler);
        double Objective1(Trial t) => Math.Pow(t.SuggestFloat("x", -2, 2) - 1, 2);
        study.Optimize(Objective1, nTrials: 30);
        TpeObservationCache cache = sampler.ObservationCache;
        var names = new[] { "x" };
        var distributions = new BaseDistribution[] { new FloatDistribution(-2, 2) };
        IReadOnlyList<FrozenTrial> live = study.GetTrials(deepcopy: false);

        TpeObservationCache.Column column = cache.ColumnsFor(study, live, names, distributions)[0];
        cache.IdentityRestarts.Should().Be(0, "the live trials are the ones the rows were read from");
        // The last trial finished after the last sampling, so no ask has read (or cached) it yet.
        live.Where(t => t.Number < live.Count - 1).Should()
            .OnlyContain(t => column.StateOf(t.Number) == TpeObservationCache.Present, "every row read stays cached");

        cache.ColumnsFor(study, Rebuilt(live, idShift: 0, startShift: null), names, distributions);
        cache.IdentityRestarts.Should().Be(0, "an unknown start time on one side is not a difference");

        column = cache.ColumnsFor(study, Rebuilt(live, idShift: 0, startShift: TimeSpan.FromSeconds(5)), names, distributions)[0];
        cache.IdentityRestarts.Should().Be(1, "the same storage ids started later: a recreated study under reused ids");
        live.Should().OnlyContain(t => column.StateOf(t.Number) == TpeObservationCache.Unknown, "no old row may be read");

        // Sampling fills the cache again from the live trials, which are consistent with it.
        study.Optimize(Objective1, nTrials: 5);
        cache.IdentityRestarts.Should().Be(1);
        column = cache.ColumnsFor(study, Rebuilt(study.GetTrials(deepcopy: false), idShift: 1000, startShift: TimeSpan.Zero), names, distributions)[0];
        cache.IdentityRestarts.Should().Be(2, "other storage ids under the cached numbers: another study's trials");
        column.StateOf(0).Should().Be(TpeObservationCache.Unknown);
    }

    /// <summary>
    ///     Copies of <paramref name="trials" /> with the same numbers and the storage ids and start times changed, the way
    ///     a study recreated under a reused id presents its own trials; parameters are left out (the identity check reads
    ///     only the number, the storage id and the start time).
    /// </summary>
    /// <param name="trials">The trials to copy.</param>
    /// <param name="idShift">Added to each storage id.</param>
    /// <param name="startShift">Added to each start time; <see langword="null" /> gives the copies no start time.</param>
    /// <returns>The copies, in the same order.</returns>
    private static List<FrozenTrial> Rebuilt(IReadOnlyList<FrozenTrial> trials, int idShift, TimeSpan? startShift)
        => trials.Select(t =>
        {
            DateTime? start = startShift is { } shift ? t.DatetimeStart!.Value + shift : null;
            return new FrozenTrial(
                t.Number, start is null ? TrialState.Waiting : TrialState.Complete, start is null ? null : 1.0,
                start, start, new OrderedDict<string, object?>(), new OrderedDict<string, BaseDistribution>(),
                new OrderedDict<string, object?>(), new OrderedDict<string, object?>(), new OrderedDict<long, double>(),
                t.TrialId + idShift);
        }).ToList();

    /// <summary>
    ///     The cache holds the storage it answers for weakly, as <c>PerStudyState</c> holds its storages: a long-lived
    ///     sampler (a DI singleton, say) must not keep the last storage it served, and every trial in it, alive.
    /// </summary>
    [TestMethod]
    public void Cache_DoesNotKeepTheStorageAlive()
    {
        var cache = new TpeObservationCache();
        WeakReference storage = AnswerForAStudyThatGoesAway(cache);
        GC.Collect();
        GC.WaitForPendingFinalizers();
        GC.Collect();
        storage.IsAlive.Should().BeFalse("only the cache referred to the storage, and it holds it weakly");
    }

    /// <summary>
    ///     Makes <paramref name="cache" /> answer for a study whose storage nothing else keeps; not inlined, so no local of
    ///     this frame outlives the call.
    /// </summary>
    /// <param name="cache">The cache under test.</param>
    /// <returns>A weak reference to the study's storage.</returns>
    [MethodImpl(MethodImplOptions.NoInlining)]
    private static WeakReference AnswerForAStudyThatGoesAway(TpeObservationCache cache)
    {
        var storage = new InMemoryStorage();
        var study = Optuna.CreateStudy(storage: storage);
        cache.ColumnsFor(study, Array.Empty<FrozenTrial>(), new[] { "x" }, new BaseDistribution[] { new FloatDistribution(0, 1) });
        cache.ColumnCount.Should().Be(1);
        return new WeakReference(storage);
    }

    /// <summary>
    ///     <see cref="SamplerConcurrency.PerWorker" />: every worker instance owns its observation cache, so workers never
    ///     share one, and the study's own sampler (the prototype) samples nothing while the workers run, so its cache
    ///     stays empty (THREADING.md §6).
    /// </summary>
    [TestMethod]
    public void WorkerInstances_OwnTheirCaches_AndThePrototypesStaysUnused()
    {
        var prototype = new TPESampler(seed: 2, nStartupTrials: 4, multivariate: false);
        var a = (TPESampler)prototype.CreateWorkerInstance()!;
        var b = (TPESampler)prototype.CreateWorkerInstance()!;
        a.ObservationCache.Should().NotBeSameAs(b.ObservationCache);
        a.ObservationCache.Should().NotBeSameAs(prototype.ObservationCache);
        b.ObservationCache.Should().NotBeSameAs(prototype.ObservationCache);

        var study = Optuna.CreateStudy(sampler: prototype);
        study.Optimize(t => Math.Pow(t.SuggestFloat("x", -2, 2), 2) + t.SuggestFloat("y", -2, 2), nTrials: 64, nJobs: 8,
            executorOptions: new TrialExecutorOptions { SamplerConcurrency = SamplerConcurrency.PerWorker });
        study.Trials.Should().HaveCount(64).And.OnlyContain(t => t.State == TrialState.Complete);
        prototype.ObservationCache.ColumnCount.Should().Be(0, "only the workers' own instances sampled");
    }

    /// <summary>
    ///     One TPE instance shared by eight workers (the default <see cref="SamplerConcurrency.Shared" />), constant liar
    ///     on, so each ask reads finished and running trials while other trials finish: afterwards every row the cache holds
    ///     is exactly the value its finished trial converts to, or that trial's lack of the parameter, and no identity
    ///     restart happened. Sampling runs under the sampler's monitor; a race on the cache would leave a torn or misplaced
    ///     row here.
    /// </summary>
    [TestMethod]
    public void SharedByEightWorkers_EveryCachedRowIsItsTrialsOwnValue()
    {
        var sampler = new TPESampler(seed: 17, nStartupTrials: 6, multivariate: false, constantLiar: true);
        var study = Optuna.CreateStudy(sampler: sampler);
        study.Optimize(t => Objective(t, () => { }), nTrials: 240, nJobs: 8);

        TpeObservationCache cache = sampler.ObservationCache;
        cache.IdentityRestarts.Should().Be(0);
        List<FrozenTrial> finished = study.GetTrials(deepcopy: false).Where(t => t.State.IsFinished()).ToList();
        int checkedRows = 0;
        foreach (var (name, distribution) in finished.SelectMany(t => t.Distributions).Select(kv => (kv.Key, kv.Value)).Distinct().ToList())
        {
            TpeObservationCache.Column column = cache.ColumnsFor(study, finished, new[] { name }, new[] { distribution })[0];
            foreach (FrozenTrial trial in finished)
            {
                byte state = column.StateOf(trial.Number);
                if (state == TpeObservationCache.Present)
                {
                    trial.Params.ContainsKey(name).Should().BeTrue($"trial {trial.Number} has a cached {name}");
                    BitConverter.DoubleToInt64Bits(column.ValueOf(trial.Number)).Should().Be(
                        BitConverter.DoubleToInt64Bits(distribution.ToInternalRepr(trial.Params[name])),
                        $"trial {trial.Number}'s {name} under {distribution}");
                    checkedRows++;
                }
                else if (state == TpeObservationCache.Absent)
                {
                    trial.Params.ContainsKey(name).Should().BeFalse($"trial {trial.Number} is cached as lacking {name}");
                    checkedRows++;
                }
            }
        }
        checkedRows.Should().BeGreaterThan(500, "the run cached most of its history");
    }
}
