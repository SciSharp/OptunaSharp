using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using AwesomeAssertions;
using OptunaSharp.Samplers;
using OptunaSharp.Storages;
using OptunaSharp.Storages.Journal;

namespace OptunaSharp.Tests.Samplers;

/// <summary>
///     The optimizer cache and incremental history of <see cref="CmaEsSampler" /> (Rustuna plan 13, phase 1), tested
///     through the sampler. That the trials themselves equal Optuna's is the live interop suite's job
///     (<c>CmaEsSamplerParityTests</c>); these pin the cache's contract (one restore per generation, a failed teller's
///     state restored again, nothing reused across storages) and the history read
///     (<c>Study.GetTrialsForSampler</c>).
/// </summary>
[TestClass]
public class CmaEsSamplerCacheTests
{
    /// <summary>A shifted sphere over <paramref name="dim" /> float parameters in [-5, 5].</summary>
    /// <param name="dim">Number of parameters.</param>
    /// <param name="shift">Where the optimum is, on every axis.</param>
    /// <returns>The objective.</returns>
    private static Func<Trial, double> Sphere(int dim, double shift) => t =>
    {
        double v = 0;
        for (int i = 0; i < dim; i++)
        {
            double x = t.SuggestFloat($"x{i}", -5, 5) - shift;
            v += x * x;
        }
        return v;
    };

    /// <summary>Whether the trial stored an optimizer state, i.e. it told a generation.</summary>
    /// <param name="trial">A finished trial.</param>
    /// <returns>True when a <c>cma:optimizer:*</c> chunk is present.</returns>
    private static bool HoldsState(FrozenTrial trial)
        => trial.SystemAttrs.Keys.Any(k => k.StartsWith("cma:optimizer:", StringComparison.Ordinal));

    /// <summary>The trial's stored optimizer state, chunks concatenated in order.</summary>
    /// <param name="trial">A trial that told a generation.</param>
    /// <returns>The stored hex pickle.</returns>
    private static string StoredState(FrozenTrial trial)
        => string.Concat(trial.SystemAttrs.Keys.Where(k => k.StartsWith("cma:optimizer:", StringComparison.Ordinal))
            .OrderBy(k => int.Parse(k["cma:optimizer:".Length..]))
            .Select(k => (string)trial.SystemAttrs[k]!));

    /// <summary>
    ///     In a sequential run each stored state is restored exactly once, at the ask after the tell that stored it;
    ///     every other ask of the generation reuses that object. Optuna restores on every ask (199 times here).
    /// </summary>
    [TestMethod]
    public void ASequentialRun_RestoresEachStoredStateOnce_AtTheAskAfterItsTell()
    {
        var sampler = new CmaEsSampler(seed: 3, popsize: 8);
        var study = Optuna.CreateStudy(sampler: sampler);
        study.Optimize(Sphere(5, 0.7), nTrials: 200);

        int tellers = study.Trials.Count(HoldsState);
        tellers.Should().BeGreaterThan(20);
        int expected = HoldsState(study.Trials[^1]) ? tellers - 1 : tellers;
        sampler.RestoreCount.Should().Be(expected);
    }

    /// <summary>
    ///     A trial that told a generation and then failed never becomes the latest state holder, so the next ask
    ///     restores the previous state again (the cache was dropped at the tell) and tells the same solutions: the state
    ///     it stores equals the failed teller's, byte for byte. The restore count keeps its one-per-tell contract.
    /// </summary>
    /// <exception cref="InvalidOperationException">
    ///     Thrown by the objective on purpose for every other trial that told a generation, and caught by
    ///     <c>Optimize</c>, so it never leaves the test.
    /// </exception>
    [TestMethod]
    public void AFailedTeller_ItsGenerationIsToldAgainFromThePreviousState_IdenticalState()
    {
        var sampler = new CmaEsSampler(seed: 5, popsize: 6);
        var study = Optuna.CreateStudy(sampler: sampler);
        var sphere = Sphere(4, -1.2);
        int tellersSeen = 0;
        study.Optimize(t =>
        {
            double v = sphere(t);
            // The trial told a generation if its sampling stored a state; every other such trial fails afterwards, so
            // the failed-teller path is exercised for sure (a fixed failure pattern can miss every teller).
            bool told = study.Storage.GetTrialSystemAttrs(t.TrialId).ContainsKey("cma:optimizer:0");
            if (told && tellersSeen++ % 2 == 0)
                throw new InvalidOperationException("planned failure");
            return v;
        }, nTrials: 300, @catch: new[] { typeof(InvalidOperationException) });

        var trials = study.Trials;
        int failedTellers = 0;
        for (int i = 0; i + 1 < trials.Count; i++)
        {
            if (trials[i].State != TrialState.Fail || !HoldsState(trials[i]))
                continue;
            failedTellers++;
            HoldsState(trials[i + 1]).Should().BeTrue($"trial {i + 1} tells the generation trial {i} told before failing");
            StoredState(trials[i + 1]).Should().Be(StoredState(trials[i]),
                $"the same previous state and the same solutions give the same state (trials {i} and {i + 1})");
        }
        failedTellers.Should().BeGreaterThan(0, "the scenario must exercise a failed teller");

        // One restore follows each tell, at the next ask, with one exception: when no completed trial holds a state yet
        // (the very first teller failed), the next ask starts a fresh optimizer instead of restoring one.
        int expected = 0;
        bool holderExists = false;
        for (int i = 0; i + 1 < trials.Count; i++)
        {
            if (!HoldsState(trials[i]))
                continue;
            if (trials[i].State == TrialState.Complete)
                holderExists = true;
            if (holderExists)
                expected++;
        }
        sampler.RestoreCount.Should().Be(expected);
    }

    /// <summary>
    ///     One sampler used for a second study on another in-memory storage. Trial ids restart at 0 there, so the second
    ///     study's state holders reuse the first study's ids; the history starts over on the new storage and drops the
    ///     cached optimizer, so every state is restored from the second study's own storage. A stale cache would answer
    ///     with the first study's optimizer once the ids collide, and the count would come out one short.
    /// </summary>
    [TestMethod]
    public void ASecondStudyOnAnotherStorage_NeverReusesTheFirstStudysOptimizer()
    {
        var sampler = new CmaEsSampler(seed: 1, popsize: 6);
        var first = Optuna.CreateStudy(storage: new InMemoryStorage(), sampler: sampler);
        first.Optimize(Sphere(3, 1.0), nTrials: 30);
        int afterFirst = sampler.RestoreCount;

        var second = Optuna.CreateStudy(storage: new InMemoryStorage(), sampler: sampler);
        second.Optimize(Sphere(3, -2.0), nTrials: 80);

        int tellers = second.Trials.Count(HoldsState);
        int expected = HoldsState(second.Trials[^1]) ? tellers - 1 : tellers;
        (sampler.RestoreCount - afterFirst).Should().Be(expected);
        second.Trials.Skip(1).Should().OnlyContain(t => t.SystemAttrs.ContainsKey("cma:generation"),
            "every trial after the startup trial is sampled by CMA-ES from the second study's own state");
    }

    /// <summary>
    ///     <c>Study.GetTrialsForSampler</c> reads an in-memory study through its zero-copy view, and any other storage
    ///     through Optuna's read (here a journal file), so a database-backed storage gets no extra round trip. Both
    ///     return the same trials in the same order.
    /// </summary>
    [TestMethod]
    public void GetTrialsForSampler_IsTheZeroCopyViewInMemory_AndOptunasReadElsewhere()
    {
        var inMemory = Optuna.CreateStudy(storage: new InMemoryStorage());
        inMemory.Optimize(t => t.SuggestFloat("x", 0, 1), nTrials: 5);
        inMemory.Storage.HasZeroCopyTrialsView.Should().BeTrue();
        var view = inMemory.GetTrialsForSampler(useCache: false);
        view.Should().BeOfType<TrialsPrefixView>();
        view.Select(t => t.Number).Should().Equal(Enumerable.Range(0, 5));

        string path = Path.Combine(Path.GetTempPath(), $"optunasharp-sampler-read-{Guid.NewGuid():N}.log");
        try
        {
            var journal = Optuna.CreateStudy(storage: new JournalStorage(new JournalFileBackend(path)));
            journal.Optimize(t => t.SuggestFloat("x", 0, 1), nTrials: 5);
            journal.Storage.HasZeroCopyTrialsView.Should().BeFalse();
            var read = journal.GetTrialsForSampler(useCache: false);
            read.Should().NotBeOfType<TrialsPrefixView>();
            read.Select(t => t.Number).Should().Equal(Enumerable.Range(0, 5));
        }
        finally
        {
            File.Delete(path);
        }
    }

    /// <summary>
    ///     The default is Optuna's path: a sampler built without the opt-in produces the same trials as one built with
    ///     <c>reuseToldOptimizer: false</c>, and restores once per tell (the default-path contract above).
    /// </summary>
    [TestMethod]
    public void ReuseToldOptimizer_IsOffByDefault()
    {
        var byDefault = new CmaEsSampler(seed: 21, popsize: 6);
        var first = Optuna.CreateStudy(sampler: byDefault);
        first.Optimize(Sphere(4, 0.4), nTrials: 120);
        var second = Optuna.CreateStudy(sampler: new CmaEsSampler(seed: 21, popsize: 6, reuseToldOptimizer: false));
        second.Optimize(Sphere(4, 0.4), nTrials: 120);

        second.Trials.Select(t => (t.Params["x0"], t.Params["x3"], t.Value))
            .Should().Equal(first.Trials.Select(t => (t.Params["x0"], t.Params["x3"], t.Value)));
        int tellers = first.Trials.Count(HoldsState);
        byDefault.RestoreCount.Should().Be(HoldsState(first.Trials[^1]) ? tellers - 1 : tellers);
    }

    /// <summary>
    ///     OptunaSharp 1.0.0's 14-parameter constructor still exists with its exact metadata signature, so an assembly
    ///     compiled against 1.0.0 binds to it at run time instead of failing with <see cref="MissingMethodException" />,
    ///     and it is Optuna's path: invoked as such a compiled call invokes it (by that signature, every argument
    ///     explicit, 1.0.0's defaults included), it produces the same trials as <c>reuseToldOptimizer: false</c>.
    /// </summary>
    [TestMethod]
    public void TheVersion1_0_0Constructor_StillBinds_AndIsOptunasPath()
    {
        // The parameter types of 1.0.0's public constructor, in order: what a 1.0.0-compiled call site names.
        Type[] signature =
        [
            typeof(IReadOnlyDictionary<string, object?>), typeof(double?), typeof(int), typeof(BaseSampler),
            typeof(bool), typeof(int?), typeof(bool), typeof(string), typeof(int?), typeof(int), typeof(bool),
            typeof(bool), typeof(bool), typeof(List<FrozenTrial>),
        ];
        var constructor = typeof(CmaEsSampler).GetConstructor(signature);
        constructor.Should().NotBeNull("assemblies compiled against OptunaSharp 1.0.0 call this exact signature");

        // A compiled call passes every argument, the defaults it left out included (x0, sigma0, nStartupTrials 1, ...).
        var legacy = (CmaEsSampler)constructor!.Invoke(
            [null, null, 1, null, true, 21, false, null, 6, -1, false, false, false, null]);
        var first = Optuna.CreateStudy(sampler: legacy);
        first.Optimize(Sphere(4, 0.4), nTrials: 120);
        var second = Optuna.CreateStudy(sampler: new CmaEsSampler(seed: 21, popsize: 6, reuseToldOptimizer: false));
        second.Optimize(Sphere(4, 0.4), nTrials: 120);

        second.Trials.Select(t => (t.Params["x0"], t.Params["x3"], t.Value))
            .Should().Equal(first.Trials.Select(t => (t.Params["x0"], t.Params["x3"], t.Value)));
    }

    /// <summary>
    ///     With the opt-in, the told optimizer keeps answering, so a sequential run restores a stored state only once:
    ///     after the first tell, whose optimizer was fresh rather than restored and so is not kept.
    /// </summary>
    [TestMethod]
    public void ReuseToldOptimizer_RestoresOnlyOnce_InASequentialRun()
    {
        var sampler = new CmaEsSampler(seed: 3, popsize: 8, reuseToldOptimizer: true);
        var study = Optuna.CreateStudy(sampler: sampler);
        study.Optimize(Sphere(5, 0.7), nTrials: 200);

        study.Trials.Count(HoldsState).Should().BeGreaterThan(20, "the state is still stored every generation");
        sampler.RestoreCount.Should().Be(1);
    }

    /// <summary>
    ///     With the opt-in, a failed teller never becomes the latest state holder, so its told object (cached under its
    ///     id) never answers: the next ask restores the previous holder's stored state, once per failed teller.
    /// </summary>
    /// <exception cref="InvalidOperationException">
    ///     Thrown by the objective on purpose for every other trial that told a generation, and caught by
    ///     <c>Optimize</c>, so it never leaves the test.
    /// </exception>
    [TestMethod]
    public void ReuseToldOptimizer_AFailedTeller_TheNextAskRestoresThePreviousState()
    {
        var sampler = new CmaEsSampler(seed: 5, popsize: 6, reuseToldOptimizer: true);
        var study = Optuna.CreateStudy(sampler: sampler);
        var sphere = Sphere(4, -1.2);
        int tellersSeen = 0;
        study.Optimize(t =>
        {
            double v = sphere(t);
            bool told = study.Storage.GetTrialSystemAttrs(t.TrialId).ContainsKey("cma:optimizer:0");
            if (told && tellersSeen++ % 2 == 0)
                throw new InvalidOperationException("planned failure");
            return v;
        }, nTrials: 300, @catch: new[] { typeof(InvalidOperationException) });

        // One restore after the first completed tell (its optimizer was fresh, so it was not kept), then one after each
        // later failed teller.
        var trials = study.Trials;
        int expected = 0;
        bool holderExists = false;
        for (int i = 0; i + 1 < trials.Count; i++)
        {
            if (!HoldsState(trials[i]))
                continue;
            if (trials[i].State == TrialState.Complete && !holderExists)
            {
                holderExists = true;
                expected++;
            }
            else if (trials[i].State == TrialState.Fail && holderExists)
            {
                expected++;
            }
        }
        expected.Should().BeGreaterThan(2, "the scenario must exercise failed tellers after the first holder");
        sampler.RestoreCount.Should().Be(expected);
    }

    /// <summary>
    ///     With the opt-in, every generation still stores Optuna's state, so a default sampler (or a Python worker)
    ///     resumes the study from storage: the resumed trials continue the generations where the reusing sampler left
    ///     them.
    /// </summary>
    [TestMethod]
    public void ReuseToldOptimizer_StoresOptunasState_SoADefaultSamplerResumesTheStudy()
    {
        var storage = new InMemoryStorage();
        var first = Optuna.CreateStudy(storage: storage, studyName: "resume",
            sampler: new CmaEsSampler(seed: 8, popsize: 6, reuseToldOptimizer: true));
        first.Optimize(Sphere(3, 1.5), nTrials: 90);
        long lastGeneration = first.Trials.Where(t => t.SystemAttrs.ContainsKey("cma:generation"))
            .Max(t => Convert.ToInt64(t.SystemAttrs["cma:generation"]));

        var resumed = Optuna.LoadStudy("resume", storage, sampler: new CmaEsSampler(seed: 9, popsize: 6));
        resumed.Optimize(Sphere(3, 1.5), nTrials: 60);

        var added = resumed.Trials.Skip(90).ToList();
        added.Should().OnlyContain(t => t.State == TrialState.Complete && t.SystemAttrs.ContainsKey("cma:generation"));
        Convert.ToInt64(added[0].SystemAttrs["cma:generation"]).Should().BeGreaterThanOrEqualTo(lastGeneration,
            "the resumed sampler restores the stored state instead of starting over");
        added.Max(t => Convert.ToInt64(t.SystemAttrs["cma:generation"])).Should().BeGreaterThan(lastGeneration);
    }

    /// <summary>
    ///     The opt-in changes the run but not its quality: on a 10-D sphere both paths reach the optimum to within 1e-6
    ///     for every seed. (Measured on 40-D, 10 seeds: best-value ratio median 0.79, range 0.51-1.48.)
    /// </summary>
    /// <param name="seed">Sampler seed.</param>
    [TestMethod]
    [DataRow(1)]
    [DataRow(2)]
    [DataRow(3)]
    public void ReuseToldOptimizer_ConvergesLikeTheDefault(int seed)
    {
        foreach (bool reuse in new[] { false, true })
        {
            var study = Optuna.CreateStudy(sampler: new CmaEsSampler(seed: seed, reuseToldOptimizer: reuse));
            study.Optimize(Sphere(10, 0.25), nTrials: 1500);
            study.BestValue.Should().BeLessThan(1e-6, $"reuseToldOptimizer: {reuse}, seed {seed}");
        }
    }

    /// <summary>
    ///     Per-worker samplers (each with its own history and cache) and a shared sampler under eight workers both
    ///     finish a study and keep telling generations: the cache adds no state that crosses workers.
    /// </summary>
    /// <param name="concurrency">Shared (one sampler behind its lock) or per-worker instances.</param>
    [TestMethod]
    [DataRow(SamplerConcurrency.Shared)]
    [DataRow(SamplerConcurrency.PerWorker)]
    public void ParallelWorkers_CompleteAndKeepTellingGenerations(SamplerConcurrency concurrency)
    {
        var study = Optuna.CreateStudy(sampler: new CmaEsSampler(seed: 7, popsize: 8));
        study.Optimize(Sphere(6, 0.3), nTrials: 400, nJobs: 8,
            executorOptions: new TrialExecutorOptions { SamplerConcurrency = concurrency });

        study.Trials.Should().HaveCount(400);
        study.Trials.Should().OnlyContain(t => t.State == TrialState.Complete);
        study.Trials.Count(HoldsState).Should().BeGreaterThan(30, "generations keep being told under concurrency");
    }
}
