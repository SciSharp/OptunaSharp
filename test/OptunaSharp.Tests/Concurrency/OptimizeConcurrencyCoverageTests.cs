using System;
using System.Linq;
using System.Threading;
using AwesomeAssertions;
using NumSharp;
using OptunaSharp.Samplers;

namespace OptunaSharp.Tests.Concurrency;

/// <summary>
///     The guarantees of one <see cref="Study" /> object used by many threads at the same instant, through its settable
///     sampler and pruner, its per-thread optimize-loop state and <c>Optimize</c> itself (THREADING.md §3.4, §4, §6.1). A
///     sampler or pruner swapped while others work never splits a call between two instances. The optimize-loop flag is
///     per thread, so concurrent <c>Optimize</c> calls are never nested invocations, and <c>Stop()</c> outside a loop is
///     refused. <c>Optimize(nJobs)</c> called by every thread at once produces one perfect study, and a <c>Stop()</c> from
///     one callback stops every <c>Optimize</c> call running on the object.
/// </summary>
[TestClass]
public class OptimizeConcurrencyCoverageTests
{
    private const int Threads = ConcurrencyGun.DefaultThreads;

    /// <summary>
    ///     Shooter 0 swaps the study's sampler between two probes, without pause, for as long as every other shooter asks,
    ///     suggests and tells. Every study call reads the sampler once and runs entirely on that instance, under that
    ///     instance's monitor, so neither probe ever sees two calls at once. A call that took one sampler's monitor and
    ///     then called the other would.
    /// </summary>
    [TestMethod]
    public void SwappingTheSampler_WhileOthersSuggest_RunsEveryCallOnOneSampler()
    {
        const int rounds = 40, trialsPerShot = 4;
        var first = new OverlapProbeSampler(seed: 41);
        var second = new OverlapProbeSampler(seed: 42);
        var study = Optuna.CreateStudy(sampler: first);
        int workersDone = 0;

        using var quiet = new QuietLogs();
        using var gun = new ConcurrencyGun(Threads);
        GunReport report = gun.Fire(rounds,
            shot =>
            {
                if (shot.Thread == 0)
                {
                    SwapUntil(ref workersDone, Threads - 1, k => study.Sampler = k % 2 == 0 ? second : first);
                    return;
                }
                for (int k = 0; k < trialsPerShot; k++)
                {
                    Trial trial = study.Ask();
                    study.Tell(trial, trial.SuggestFloat("x", 0, 1) + trial.SuggestFloat("y", 0, 1));
                }
                Interlocked.Increment(ref workersDone);
            },
            beforeRound: _ => workersDone = 0);

        report.AssertClean();
        first.SampleIndependentCalls.Should().BePositive("the swaps interleave with the suggestions");
        second.SampleIndependentCalls.Should().BePositive();
        (first.SampleIndependentCalls + second.SampleIndependentCalls).Should().Be((Threads - 1) * rounds * trialsPerShot * 2);
        first.PeakOverlap.Should().Be(1, "every call runs on the sampler it read, under that sampler's monitor");
        second.PeakOverlap.Should().Be(1);
        study.Trials.Should().HaveCount((Threads - 1) * rounds * trialsPerShot).And.OnlyContain(t => t.State == TrialState.Complete);
    }

    /// <summary>
    ///     Runs <paramref name="swap" /> with a growing counter, without pause, until <paramref name="done" /> reaches
    ///     <paramref name="workers" />: the swapping shooter of the swap tests, which must keep swapping for as long as the
    ///     other shooters work.
    /// </summary>
    /// <param name="done">How many workers finished their shot; the other shooters raise it.</param>
    /// <param name="workers">How many workers the swapping shooter waits for.</param>
    /// <param name="swap">One swap, given how many swaps came before it.</param>
    private static void SwapUntil(ref int done, int workers, Action<int> swap)
    {
        for (int k = 0; Volatile.Read(ref done) < workers; k++)
            swap(k);
    }

    /// <summary>
    ///     Shooter 0 swaps the study's pruner between two probes, without pause, for as long as every other shooter reports
    ///     on a trial of its own and asks whether to prune it. Every decision runs on one pruner, under its monitor, and is
    ///     made on that shooter's own trial.
    /// </summary>
    [TestMethod]
    public void SwappingThePruner_WhileOthersAskToPrune_RunsEveryDecisionOnOnePruner()
    {
        const int rounds = 40, decisions = 6;
        var first = new OverlapProbePruner();
        var second = new OverlapProbePruner();
        var study = Optuna.CreateStudy(sampler: new RandomSampler(43), pruner: first);
        var trials = new Trial[Threads];
        int misjudged = 0, workersDone = 0;

        using var quiet = new QuietLogs();
        using var gun = new ConcurrencyGun(Threads);
        GunReport report = gun.Fire(rounds,
            shot =>
            {
                if (shot.Thread == 0)
                {
                    SwapUntil(ref workersDone, Threads - 1, k => study.Pruner = k % 2 == 0 ? second : first);
                    return;
                }
                Trial trial = trials[shot.Thread];
                for (int step = 0; step < decisions; step++)
                {
                    trial.Report(shot.Thread + step, step);
                    // The probe prunes on an odd last value; this shooter's last value is thread + step.
                    if (trial.ShouldPrune() != ((shot.Thread + step) % 2 == 1))
                        Interlocked.Increment(ref misjudged);
                }
                Interlocked.Increment(ref workersDone);
            },
            beforeRound: _ =>
            {
                workersDone = 0;
                for (int t = 1; t < Threads; t++)
                    trials[t] = study.Ask();
            },
            afterRound: _ =>
            {
                for (int t = 1; t < Threads; t++)
                    study.Tell(trials[t], t);
                return false;
            });

        report.AssertClean();
        misjudged.Should().Be(0, "every decision is made on the deciding shooter's own trial");
        first.Calls.Should().BePositive("the swaps interleave with the decisions");
        second.Calls.Should().BePositive();
        (first.Calls + second.Calls).Should().Be((Threads - 1) * rounds * decisions);
        first.PeakOverlap.Should().Be(1);
        second.PeakOverlap.Should().Be(1);
    }

    /// <summary>
    ///     Half the shooters run <c>Optimize</c> (five trials, <c>nJobs: 1</c>, on their own thread) on one study at once,
    ///     while the other half call <c>Stop()</c> outside any optimize loop. The loop flag is per thread (THREADING.md §4):
    ///     no <c>Optimize</c> call is taken for a nested invocation, every <c>Stop()</c> outside a loop is refused with
    ///     Optuna's error and stops nothing, and the study holds every trial.
    /// </summary>
    [TestMethod]
    public void Optimize_FromHalfTheThreads_WhileTheOthersCallStop_IsNeverNested_AndStopOutsideALoopIsRefused()
    {
        const int rounds = 12, trialsPerCall = 5, stopsPerShot = 20;
        var study = Optuna.CreateStudy(sampler: new RandomSampler(44));
        int refused = 0;

        using var quiet = new QuietLogs();
        using var gun = new ConcurrencyGun(Threads);
        GunReport report = gun.Fire(rounds, shot =>
        {
            if (shot.Thread % 2 == 0)
            {
                study.Optimize(t => t.SuggestFloat("x", 0, 1), nTrials: trialsPerCall);
                return;
            }
            for (int k = 0; k < stopsPerShot; k++)
            {
                try
                {
                    study.Stop();
                }
                catch (RuntimeError e) when (e.Message == "`Study.stop` is supposed to be invoked inside an objective function or a callback.")
                {
                    Interlocked.Increment(ref refused);
                }
            }
        });

        report.AssertClean();
        refused.Should().Be(Threads / 2 * rounds * stopsPerShot, "a thread outside an optimize loop may not stop the study");
        var trials = study.Trials;
        trials.Select(t => t.Number).Should().Equal(Enumerable.Range(0, Threads / 2 * rounds * trialsPerCall),
            "no Optimize call was refused as nested or stopped early");
        trials.Should().OnlyContain(t => t.State == TrialState.Complete);
    }

    /// <summary>
    ///     Every shooter calls <c>Optimize(nJobs: 2)</c> on one study at once, so sixteen workers of eight calls share the
    ///     study, its sampler and the worker scheduler: the study holds every trial of every call, numbered without a gap,
    ///     each with its own draw.
    /// </summary>
    [TestMethod]
    public void OptimizeWithNJobs_FromEveryThreadAtOnce_ProducesOnePerfectStudy()
    {
        const int rounds = 6, trialsPerCall = 6;
        var study = Optuna.CreateStudy(sampler: new RandomSampler(45));

        using var quiet = new QuietLogs();
        using var gun = new ConcurrencyGun(Threads);
        GunReport report = gun.Fire(rounds,
            _ => study.Optimize(t => t.SuggestFloat("x", 0, 1), nTrials: trialsPerCall, nJobs: 2));

        report.AssertClean();
        var trials = study.Trials;
        trials.Select(t => t.Number).Should().Equal(Enumerable.Range(0, Threads * rounds * trialsPerCall));
        trials.Should().OnlyContain(t => t.State == TrialState.Complete);
        trials.Select(t => t.Value).Should().OnlyHaveUniqueItems("one generator, drawn one call at a time");
    }

    /// <summary>
    ///     Every shooter runs an effectively endless <c>Optimize</c> on one study at once, and a callback stops the study
    ///     from the twentieth completed trial on. The stop flag belongs to the study object (THREADING.md §3.4), so every
    ///     <c>Optimize</c> call returns soon after: each finishes at most the trial it was running when the flag rose.
    /// </summary>
    /// <remarks>
    ///     The callback stops on every completion from the twentieth on, not only on the twentieth. Each new
    ///     <c>Optimize</c> call clears the flag, as Python's does, so a shooter that the machine started late could clear
    ///     a stop that came before its start, and one stop would not be enough.
    /// </remarks>
    [TestMethod]
    public void StopFromOneCallback_StopsEveryOptimizeCallOnTheStudyObject()
    {
        const int stopAfter = 20;
        var study = Optuna.CreateStudy(sampler: new RandomSampler(46));
        int completed = 0;
        Action<Study, FrozenTrial> stopper = (s, _) =>
        {
            if (Interlocked.Increment(ref completed) >= stopAfter)
                s.Stop();
        };

        using var quiet = new QuietLogs();
        using var gun = new ConcurrencyGun(Threads);
        GunReport report = gun.Fire(1, _ => study.Optimize(t =>
        {
            Thread.Sleep(1);
            return t.SuggestFloat("x", 0, 1);
        }, nTrials: 1_000_000, callbacks: new[] { stopper }));

        report.AssertClean();
        study.Trials.Count.Should().BeInRange(stopAfter, stopAfter + 3 * Threads,
            "one stop ends every Optimize call on the object, each after at most the trial it was running");
    }
}
