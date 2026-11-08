using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using AwesomeAssertions;
using NumSharp;
using OptunaSharp.Collections;
using OptunaSharp.Samplers;

namespace OptunaSharp.Tests;

/// <summary>
///     One study used from several threads in the ways applications do besides <c>Optimize(nJobs)</c>: two
///     <c>Optimize</c> calls at once, ask/tell from many threads, readers during a run, trials enqueued before and
///     during a run, two threads telling one trial, and one sampler shared by two studies. Each expectation was first
///     observed on Optuna 5.0 with the same scenario (quoted per test); Python's threads share the GIL, so where
///     OptunaSharp's run truly in parallel the test also pins that the outcome Python shows is the only outcome.
/// </summary>
[TestClass]
public class StudyConcurrencyEdgeTests
{
    /// <summary>
    ///     Two threads optimizing the same study at once, each with its own worker count: both runs complete and the
    ///     study is one perfect study (Optuna 5.0: two threads × 50 trials, at <c>n_jobs</c> 1 and 4: 100 trials, no
    ///     error, numbered contiguously). The optimize-loop flag is per thread, so neither call is a nested invocation.
    /// </summary>
    /// <param name="nJobs">Each call's worker count.</param>
    [TestMethod]
    [DataRow(1)]
    [DataRow(4)]
    public void TwoThreads_OptimizingOneStudy_ProduceOnePerfectStudy(int nJobs)
    {
        var study = Optuna.CreateStudy(sampler: new RandomSampler(40));
        var errors = new ConcurrentQueue<Exception>();

        RunOnThreads(2, _ => study.Optimize(t => t.SuggestFloat("x", 0, 1), nTrials: 50, nJobs: nJobs), errors);

        errors.Should().BeEmpty();
        ShouldBePerfect(study, 100);
    }

    /// <summary>
    ///     Eight threads each asking, suggesting and telling 50 trials, with no <c>Optimize</c> at all: 400 trials,
    ///     numbered contiguously, each completed with its own value and its own draw (Optuna 5.0: the same, 400
    ///     COMPLETE, 400 distinct values).
    /// </summary>
    [TestMethod]
    public void ManyThreads_AskingAndTelling_ProduceOnePerfectStudy()
    {
        var study = Optuna.CreateStudy(sampler: new RandomSampler(41));
        var errors = new ConcurrentQueue<Exception>();

        RunOnThreads(8, _ =>
        {
            for (int i = 0; i < 50; i++)
            {
                var trial = study.Ask();
                double x = trial.SuggestFloat("x", 0, 1);
                study.Tell(trial, x);
            }
        }, errors);

        errors.Should().BeEmpty();
        ShouldBePerfect(study, 400);
        study.Trials.Should().OnlyContain(t => t.Value == Convert.ToDouble(t.Params["x"]), "each trial was told its own draw");
    }

    /// <summary>
    ///     A reader polling the study while eight workers write it, as a dashboard or a progress reporter does: every
    ///     snapshot is numbered contiguously, every COMPLETE trial in it has its value and completion time, and the
    ///     best value is never worse than any completed trial the reader saw just before (Optuna 5.0: 300 trials at
    ///     <c>n_jobs=8</c> with the same reader: no error).
    /// </summary>
    /// <remarks>
    ///     Trial 0 holds the run open until the reader has completed one full read, and the reader starts once trial 0
    ///     runs, so at least one read overlaps the run while the other seven workers write. Without that handshake the
    ///     300 trivial trials (about 40 ms) could all finish before a starved reader thread was first scheduled: with two
    ///     test processes on 2 cores the reader once made 0 reads, and the test failed on scheduling alone.
    /// </remarks>
    [TestMethod]
    public void AReader_DuringAnNJobsRun_AlwaysSeesAConsistentStudy()
    {
        var study = Optuna.CreateStudy(sampler: new RandomSampler(42));
        var errors = new ConcurrentQueue<Exception>();
        using var finished = new ManualResetEventSlim();
        // The handshake that makes a read overlap the run: trial 0 signals that the run is underway, then waits until the
        // reader has completed a read. Both waits are bounded, so a regression fails the assertions below, not the run.
        using var runStarted = new ManualResetEventSlim();
        using var firstReadDone = new ManualResetEventSlim();
        int reads = 0;

        var reader = new Thread(() =>
        {
            runStarted.Wait(TimeSpan.FromSeconds(30));
            while (!finished.IsSet)
            {
                try
                {
                    var trials = study.Trials;
                    trials.Select(t => t.Number).Should().Equal(Enumerable.Range(0, trials.Count));
                    foreach (var t in trials.Where(t => t.State == TrialState.Complete))
                    {
                        t.Value.Should().NotBeNull();
                        t.DatetimeComplete.Should().NotBeNull();
                    }
                    var complete = study.GetTrials(deepcopy: false, states: new[] { TrialState.Complete });
                    if (complete.Count > 0)
                    {
                        // The best can only improve after this snapshot: it is the minimum over a superset of it.
                        double best = study.BestValue;
                        best.Should().BeLessThanOrEqualTo(complete.Min(t => t.Value!.Value));
                        study.BestTrial.State.Should().Be(TrialState.Complete);
                    }
                    Interlocked.Increment(ref reads);
                    firstReadDone.Set();
                }
                catch (Exception e)
                {
                    errors.Enqueue(e);
                    // Release trial 0 too: the error is asserted after the run, which must not wait out the timeout.
                    firstReadDone.Set();
                    return;
                }
            }
        });
        reader.Start();

        try
        {
            study.Optimize(t =>
            {
                if (t.Number == 0)
                {
                    runStarted.Set();
                    firstReadDone.Wait(TimeSpan.FromSeconds(30));
                }
                return t.SuggestFloat("x", 0, 1);
            }, nTrials: 300, nJobs: 8);
        }
        finally
        {
            finished.Set();
            reader.Join();
        }

        errors.Should().BeEmpty();
        reads.Should().BePositive();
        ShouldBePerfect(study, 300);
    }

    /// <summary>
    ///     Trials enqueued before an nJobs run are each run exactly once, and first: they were numbered 0–19 when they
    ///     were enqueued, and workers pop the queue before creating new trials (Optuna 5.0: 20 enqueued, 40 trials at
    ///     <c>n_jobs=8</c>: trials 0–19 carry the enqueued values, each once).
    /// </summary>
    [TestMethod]
    public void EnqueuedTrials_UnderNJobs_EachRunExactlyOnce_AndFirst()
    {
        var study = Optuna.CreateStudy(sampler: new RandomSampler(43));
        for (int i = 0; i < 20; i++)
            study.EnqueueTrial(new OrderedDict<string, object?> { ["x"] = 100.0 + i });

        study.Optimize(t => t.SuggestFloat("x", 0, 200), nTrials: 40, nJobs: 8);

        var trials = study.Trials;
        ShouldBePerfect(study, 40);
        trials.Take(20).Select(t => Convert.ToDouble(t.Params["x"])).Should().Equal(Enumerable.Range(0, 20).Select(i => 100.0 + i));
    }

    /// <summary>
    ///     Trials enqueued by callbacks during an nJobs run are each popped exactly once by a later trial, and none is
    ///     left waiting (Optuna 5.0: the callbacks of trials 0–9 enqueue one trial each during a 60-trial run at
    ///     <c>n_jobs=8</c>: all 10 run once, 60 trials, none WAITING). Two workers racing to pop one queued trial must
    ///     not both run it: the storage's state change to RUNNING admits one.
    /// </summary>
    [TestMethod]
    public void TrialsEnqueuedByCallbacks_DuringAnNJobsRun_EachRunExactlyOnce()
    {
        var study = Optuna.CreateStudy(sampler: new RandomSampler(44));

        study.Optimize(t => t.SuggestFloat("x", 0, 2000), nTrials: 60, nJobs: 8, callbacks: new Action<Study, FrozenTrial>[]
        {
            (s, frozen) =>
            {
                if (frozen.Number < 10)
                    s.EnqueueTrial(new OrderedDict<string, object?> { ["x"] = 1000.0 + frozen.Number });
            },
        });

        var enqueued = study.Trials.Select(t => Convert.ToDouble(t.Params["x"])).Where(x => x >= 1000 && x < 1010 && x % 1 == 0);
        enqueued.Order().Should().Equal(Enumerable.Range(0, 10).Select(i => 1000.0 + i));
        ShouldBePerfect(study, 60);
    }

    /// <summary>
    ///     Two threads telling one trial at the same instant: exactly one tell stores its value, and the other is
    ///     refused with Python's message, as if it had come second. Optuna 5.0 showed exactly that in 300 of 300 rounds.
    ///     The sampler's <c>AfterTrial</c> runs once, for the stored value.
    /// </summary>
    /// <remarks>
    ///     Python's check ("is the trial still running?") and its write are separate storage calls; the GIL makes two
    ///     threads passing the check together rare enough that 300 rounds never showed it. Here they run in parallel,
    ///     and both used to pass it: the loser then ran the sampler's <c>AfterTrial</c> with its own, unstored value and
    ///     failed in the storage with <c>UpdateFinishedTrialError</c> instead of the refusal.
    /// </remarks>
    [TestMethod]
    public void TwoThreads_TellingOneTrial_OneIsStored_TheOtherIsRefused()
    {
        var outcomes = TellOneTrialFromTwoThreads(rounds: 300, skipIfFinished: false);

        outcomes.Should().OnlyContain(o => o.Successes == 1, "exactly one tell wins each round");
        outcomes.SelectMany(o => o.Errors).Should().OnlyContain(e => e is ValueError && e.Message == "Cannot tell a COMPLETE trial.")
            .And.HaveCount(300);
        outcomes.Should().OnlyContain(o => o.AfterTrialCalls == 1, "the sampler hears of one result per trial");
        outcomes.Should().OnlyContain(o => o.StoredValue == o.WinnerValue, "the stored value is the one whose tell succeeded");
    }

    /// <summary>
    ///     The same race with <c>skipIfFinished: true</c>: neither tell raises; the one that comes second returns the
    ///     stored trial unchanged (Optuna 5.0: 300 of 300 rounds with both tells returning). This is the contract a
    ///     reaper relies on when it fails stale trials while their workers may still be telling them.
    /// </summary>
    [TestMethod]
    public void TwoThreads_TellingOneTrial_WithSkipIfFinished_NeitherRaises()
    {
        var outcomes = TellOneTrialFromTwoThreads(rounds: 300, skipIfFinished: true);

        outcomes.SelectMany(o => o.Errors).Should().BeEmpty();
        outcomes.Should().OnlyContain(o => o.Successes == 2);
        outcomes.Should().OnlyContain(o => o.AfterTrialCalls == 1);
        outcomes.Should().OnlyContain(o => o.ReturnedValues.All(v => v == o.StoredValue),
            "a skipped tell returns the trial as stored, with the other tell's value");
    }

    /// <summary>
    ///     One random sampler shared by two studies that two threads optimize at once: every draw is distinct across
    ///     both studies, since the sampler's one generator serves them in turn (Optuna 5.0: 2 × 200 trials, 400
    ///     distinct values; numpy's <c>RandomState</c> locks each call).
    /// </summary>
    [TestMethod]
    public void OneSampler_SharedByTwoStudiesOnTwoThreads_NeverRepeatsADraw()
    {
        var sampler = new RandomSampler(45);
        var studies = new[] { Optuna.CreateStudy(sampler: sampler), Optuna.CreateStudy(sampler: sampler) };
        var errors = new ConcurrentQueue<Exception>();

        RunOnThreads(2, i => studies[i].Optimize(t => t.SuggestFloat("x", 0, 1), nTrials: 200), errors);

        errors.Should().BeEmpty();
        studies.SelectMany(s => s.Trials).Select(t => Convert.ToDouble(t.Params["x"]))
            .Should().HaveCount(400).And.OnlyHaveUniqueItems();
    }

    /// <summary>What one round of <see cref="TellOneTrialFromTwoThreads" /> produced.</summary>
    /// <param name="Successes">How many of the two tells returned.</param>
    /// <param name="Errors">The exceptions the other tells threw.</param>
    /// <param name="AfterTrialCalls">How many times the sampler's <c>AfterTrial</c> ran for the trial.</param>
    /// <param name="WinnerValue">The value of the tell that stored the trial, or null if none did.</param>
    /// <param name="StoredValue">The trial's value as stored at the end of the round.</param>
    /// <param name="ReturnedValues">The values of the trials the successful tells returned.</param>
    private sealed record TellRound(
        int Successes, IReadOnlyList<Exception> Errors, int AfterTrialCalls, double? WinnerValue, double? StoredValue,
        IReadOnlyList<double?> ReturnedValues);

    /// <summary>
    ///     Runs <paramref name="rounds" /> rounds of one race: a new trial, then two threads released together by a
    ///     barrier, each telling it a different value.
    /// </summary>
    /// <param name="rounds">How many independent trials to race on.</param>
    /// <param name="skipIfFinished">Passed to both tells.</param>
    /// <returns>One record per round.</returns>
    private static List<TellRound> TellOneTrialFromTwoThreads(int rounds, bool skipIfFinished)
    {
        var results = new List<TellRound>();
        for (int round = 0; round < rounds; round++)
        {
            var sampler = new AfterTrialCountingSampler(round);
            var study = Optuna.CreateStudy(sampler: sampler);
            var trial = study.Ask();
            trial.SuggestFloat("x", 0, 1);

            using var barrier = new Barrier(2);
            var errors = new ConcurrentQueue<Exception>();
            var returned = new ConcurrentQueue<FrozenTrial>();
            double? winner = null;
            var threads = new[] { 1.0, 2.0 }.Select(value => new Thread(() =>
            {
                barrier.SignalAndWait();
                try
                {
                    var frozen = study.Tell(trial, value, skipIfFinished);
                    returned.Enqueue(frozen);
                    // The tell that stored the trial is the one whose value the returned trial carries.
                    if (frozen.Value == value)
                        winner = value;
                }
                catch (Exception e)
                {
                    errors.Enqueue(e);
                }
            })).ToList();
            threads.ForEach(t => t.Start());
            threads.ForEach(t => t.Join());

            results.Add(new TellRound(
                returned.Count, errors.ToList(), sampler.AfterTrialCalls, winner, study.Trials[0].Value,
                returned.Select(f => f.Value).ToList()));
        }
        return results;
    }

    /// <summary>A random sampler that counts its <c>AfterTrial</c> calls.</summary>
    /// <param name="seed">The random sampler's seed.</param>
    private sealed class AfterTrialCountingSampler(int seed) : RandomSampler(seed)
    {
        private int _afterTrialCalls;

        /// <summary>How many times <see cref="AfterTrial" /> has run.</summary>
        public int AfterTrialCalls => Volatile.Read(ref _afterTrialCalls);

        /// <summary>Counts the call, then post-processes like the random sampler.</summary>
        /// <param name="study">The study that owns the trial.</param>
        /// <param name="trial">The trial's snapshot before its final state is stored.</param>
        /// <param name="state">The final state about to be stored.</param>
        /// <param name="values">The trial's values, or null.</param>
        public override void AfterTrial(Study study, FrozenTrial trial, TrialState state, IReadOnlyList<double>? values)
        {
            Interlocked.Increment(ref _afterTrialCalls);
            base.AfterTrial(study, trial, state, values);
        }
    }

    /// <summary>
    ///     Runs <paramref name="work" /> on <paramref name="threads" /> dedicated threads released together, and records
    ///     what they throw instead of losing it on a background thread.
    /// </summary>
    /// <param name="threads">How many threads to run.</param>
    /// <param name="work">The work; receives the thread's index.</param>
    /// <param name="errors">Receives every exception the work throws.</param>
    private static void RunOnThreads(int threads, Action<int> work, ConcurrentQueue<Exception> errors)
    {
        using var barrier = new Barrier(threads);
        var list = Enumerable.Range(0, threads).Select(i => new Thread(() =>
        {
            barrier.SignalAndWait();
            try
            {
                work(i);
            }
            catch (Exception e)
            {
                errors.Enqueue(e);
            }
        })).ToList();
        list.ForEach(t => t.Start());
        list.ForEach(t => t.Join());
    }

    /// <summary>
    ///     Asserts that <paramref name="study" /> holds exactly <paramref name="count" /> trials, numbered from 0 without a
    ///     gap, all COMPLETE, with distinct draws of <c>x</c>.
    /// </summary>
    /// <param name="study">The study checked.</param>
    /// <param name="count">The expected trial count.</param>
    private static void ShouldBePerfect(Study study, int count)
    {
        var trials = study.Trials;
        trials.Should().HaveCount(count).And.OnlyContain(t => t.State == TrialState.Complete);
        trials.Select(t => t.Number).Should().Equal(Enumerable.Range(0, count));
        trials.Select(t => Convert.ToDouble(t.Params["x"])).Should().OnlyHaveUniqueItems("no two trials may share a draw");
    }
}
