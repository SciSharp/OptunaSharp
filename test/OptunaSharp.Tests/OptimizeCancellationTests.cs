using System;
using System.Collections.Concurrent;
using System.IO;
using System.Linq;
using System.Threading;
using AwesomeAssertions;
using OptunaSharp.Samplers;
using OptunaSharp.Storages.Journal;

namespace OptunaSharp.Tests;

/// <summary>
///     Cancelling <c>Optimize</c> through its <c>ct</c> argument (plan 10, item 2.4; THREADING.md §3.4), OptunaSharp's
///     form of Python's <c>KeyboardInterrupt</c>. Observed on Optuna 5.0.0 (2026-11-07): at <c>n_jobs=1</c> the interrupt
///     lands in the objective, that trial is stored FAIL (trials 0–3 COMPLETE, 4 FAIL) and the interrupt propagates; at
///     <c>n_jobs=3</c> the trials in flight finish COMPLETE and the interrupt is raised after them.
/// </summary>
/// <remarks>
///     The token is cooperative, so the cases pin both outcomes: an objective that observes
///     <see cref="Trial.CancellationToken" /> is stored FAIL whatever <c>catch</c> holds, and one that ignores it finishes.
///     Either way no trial is left RUNNING, and <c>Optimize</c> throws <see cref="OperationCanceledException" /> carrying the
///     caller's token once every trial in flight is stored.
/// </remarks>
[TestClass]
public class OptimizeCancellationTests
{
    /// <summary>Options for the executor a case runs on.</summary>
    /// <param name="kind">The executor.</param>
    /// <returns>Fresh options selecting <paramref name="kind" />.</returns>
    private static TrialExecutorOptions Executor(TrialExecutorKind kind) => new() { Executor = kind };

    /// <summary>A new in-memory study with a seeded sampler, so a failure reproduces.</summary>
    /// <returns>The study.</returns>
    private static Study NewStudy() => Optuna.CreateStudy(sampler: new RandomSampler(seed: 0));

    /// <summary>An already cancelled token runs no trial and throws at once, carrying the token, on either path.</summary>
    /// <param name="nJobs">1 for the sequential loop, 4 for the worker loop.</param>
    [TestMethod]
    [DataRow(1)]
    [DataRow(4)]
    public void AlreadyCancelledToken_RunsNoTrial_AndThrowsCarryingTheToken(int nJobs)
    {
        var study = NewStudy();
        using var cts = new CancellationTokenSource();
        cts.Cancel();
        int calls = 0;

        var act = () => study.Optimize(t => { Interlocked.Increment(ref calls); return 0.0; }, nTrials: 10, nJobs: nJobs, ct: cts.Token);

        act.Should().Throw<OperationCanceledException>().Which.CancellationToken.Should().Be(cts.Token);
        calls.Should().Be(0);
        study.Trials.Should().BeEmpty();
    }

    /// <summary>
    ///     Cancelling while trials run stops the scheduling: no trial starts afterwards, the trials in flight (which
    ///     ignore the token here) finish COMPLETE, and only then does <c>Optimize</c> throw. At <c>nJobs: 1</c> the
    ///     callback of trial 5 cancels, so exactly trials 0–5 run, as Python's sequential loop would.
    /// </summary>
    /// <remarks>
    ///     With several workers the upper bound is taken from the trials that exist once the token is cancelled, not from
    ///     trial 5's number. Until trial 5's callback runs, the other workers keep finishing trials and starting new ones,
    ///     and trial 5's worker can fall several 10 ms trials behind them on a small or busy runner: a 3-core macOS arm64
    ///     runner ended a run with 18 trials where a bound of <c>6 + nJobs</c> allowed 14 (dispatch 42393897318). What the
    ///     code guarantees is relative to the cancellation: each worker checks the token right before it asks for a
    ///     trial, so after <c>Cancel()</c> only a worker already past that check can still add one, at most one per
    ///     worker other than the cancelling one.
    /// </remarks>
    /// <param name="nJobs">The worker count.</param>
    /// <param name="kind">The executor of the workers.</param>
    [TestMethod]
    [DataRow(1, TrialExecutorKind.DedicatedThreads)]
    [DataRow(8, TrialExecutorKind.DedicatedThreads)]
    [DataRow(8, TrialExecutorKind.ThreadPool)]
    public void CancelMidRun_FinishesTheTrialsInFlight_ThenThrows(int nJobs, TrialExecutorKind kind)
    {
        var study = NewStudy();
        using var cts = new CancellationTokenSource();
        // Trials that exist right after the cancel; written once by the cancelling callback, read after Optimize, which
        // waits for every worker first.
        int existingAfterCancel = -1;
        void CancelAfterTrialFive(Study s, FrozenTrial t)
        {
            if (t.Number != 5)
                return;
            cts.Cancel();
            // Counted after Cancel(), not before: from here on a worker that checks the token sees it cancelled, so the
            // count can only be exceeded by workers that were already between their check and their Ask(). Counting
            // first would let this thread be preempted between the two calls while the others keep adding trials.
            existingAfterCancel = s.GetTrials(deepcopy: false).Count;
        }

        var act = () => study.Optimize(t =>
        {
            Thread.Sleep(10);
            return t.SuggestFloat("x", 0, 1);
        }, nTrials: 200, nJobs: nJobs, callbacks: new Action<Study, FrozenTrial>[] { CancelAfterTrialFive },
            executorOptions: Executor(kind), ct: cts.Token);

        act.Should().Throw<OperationCanceledException>().Which.CancellationToken.Should().Be(cts.Token);
        var trials = study.Trials;
        trials.Should().OnlyContain(t => t.State == TrialState.Complete, "the trials in flight ignore the token and finish");
        if (nJobs == 1)
            trials.Select(t => t.Number).Should().Equal(Enumerable.Range(0, 6));
        else
            trials.Count.Should().BeInRange(6, existingAfterCancel + nJobs - 1,
                "after the cancel only a worker already past its token check can start a trial, at most one per other worker");
    }

    /// <summary>
    ///     An objective that observes the token is stored FAIL, as Python stores the trial its <c>KeyboardInterrupt</c>
    ///     lands in, and the cancellation leaves the call even though <c>catch</c> holds its type; the callbacks are not
    ///     called for that trial, as Python's <c>_run_trial</c> raises before them.
    /// </summary>
    /// <param name="caught">The type the catch list holds.</param>
    [TestMethod]
    [DataRow(typeof(OperationCanceledException))]
    [DataRow(typeof(Exception))]
    public void ObjectiveObservingTheToken_IsStoredFail_WhateverCatchHolds(Type caught)
    {
        var study = NewStudy();
        using var cts = new CancellationTokenSource();
        var calledBack = new ConcurrentBag<int>();

        var act = () => study.Optimize(t =>
        {
            if (t.Number == 3)
            {
                cts.Cancel();
                t.CancellationToken.ThrowIfCancellationRequested();
            }
            return t.SuggestFloat("x", 0, 1);
        }, nTrials: 10, @catch: new[] { caught },
            callbacks: new Action<Study, FrozenTrial>[] { (_, t) => calledBack.Add(t.Number) }, ct: cts.Token);

        act.Should().Throw<OperationCanceledException>().Which.CancellationToken.Should().Be(cts.Token);
        study.Trials.Select(t => t.State).Should().Equal(
            TrialState.Complete, TrialState.Complete, TrialState.Complete, TrialState.Fail);
        calledBack.Should().BeEquivalentTo(new[] { 0, 1, 2 });
    }

    /// <summary>
    ///     With several workers, the objectives that observe the token are stored FAIL, every trial is finished (none is
    ///     left RUNNING), and the call throws carrying the caller's token, on either executor.
    /// </summary>
    /// <param name="kind">The executor of the workers.</param>
    [TestMethod]
    [DataRow(TrialExecutorKind.DedicatedThreads)]
    [DataRow(TrialExecutorKind.ThreadPool)]
    public void WorkersObservingTheToken_AreStoredFail_AndNoneIsLeftRunning(TrialExecutorKind kind)
    {
        var study = NewStudy();
        using var cts = new CancellationTokenSource();

        var act = () => study.Optimize(t =>
        {
            if (t.Number == 6)
                cts.Cancel();
            // A long evaluation that stops at the cancellation instead of running to its end.
            for (int step = 0; step < 50; step++)
            {
                t.CancellationToken.ThrowIfCancellationRequested();
                Thread.Sleep(2);
            }
            return t.SuggestFloat("x", 0, 1);
        }, nTrials: 200, nJobs: 4, executorOptions: Executor(kind), ct: cts.Token);

        act.Should().Throw<OperationCanceledException>().Which.CancellationToken.Should().Be(cts.Token);
        var trials = study.Trials;
        trials.Should().NotContain(t => t.State == TrialState.Running);
        trials.Should().Contain(t => t.Number == 6 && t.State == TrialState.Fail);
        trials.Count.Should().BeLessThan(200);
    }

    /// <summary>
    ///     An objective that ignores the token runs to its end: cancelled in the middle of trial 2, that trial is stored
    ///     COMPLETE, no trial follows, and the call throws.
    /// </summary>
    [TestMethod]
    public void ObjectiveIgnoringTheToken_FinishesAndIsStoredComplete()
    {
        var study = NewStudy();
        using var cts = new CancellationTokenSource();

        var act = () => study.Optimize(t =>
        {
            if (t.Number == 2)
                cts.Cancel();
            return t.SuggestFloat("x", 0, 1);
        }, nTrials: 10, ct: cts.Token);

        act.Should().Throw<OperationCanceledException>().Which.CancellationToken.Should().Be(cts.Token);
        study.Trials.Select(t => t.State).Should().Equal(TrialState.Complete, TrialState.Complete, TrialState.Complete);
    }

    /// <summary>
    ///     A cancellation exception while the caller's token is not cancelled (an HTTP timeout, say) is an ordinary
    ///     failure: caught by <c>catch</c>, every trial runs; not caught, it propagates as the objective threw it.
    /// </summary>
    [TestMethod]
    public void CancellationExceptionOfAnotherToken_IsAnOrdinaryFailure()
    {
        using var cts = new CancellationTokenSource();
        var caughtStudy = NewStudy();
        caughtStudy.Optimize(t => TimedOut(new OperationCanceledException("timed out")), nTrials: 3,
            @catch: new[] { typeof(OperationCanceledException) }, ct: cts.Token);
        caughtStudy.Trials.Should().HaveCount(3).And.OnlyContain(t => t.State == TrialState.Fail);

        var uncaughtStudy = NewStudy();
        var timeout = new OperationCanceledException("timed out");
        var act = () => uncaughtStudy.Optimize(t => TimedOut(timeout), nTrials: 3, ct: cts.Token);
        act.Should().Throw<OperationCanceledException>().Which.Should().BeSameAs(timeout);
        uncaughtStudy.Trials.Should().ContainSingle().Which.State.Should().Be(TrialState.Fail);
    }

    /// <summary>Stands for an objective whose own operation timed out: it throws <paramref name="error" />.</summary>
    /// <param name="error">A cancellation of the objective's own work, unrelated to the caller's token.</param>
    /// <returns>Never returns.</returns>
    /// <exception cref="OperationCanceledException">Always: <paramref name="error" />.</exception>
    private static double TimedOut(OperationCanceledException error) => throw error;

    /// <summary>
    ///     An objective that cancels through a token linked to its trial's carries the linked token; the call reports the
    ///     caller's token instead, the one .NET callers compare, and keeps the objective's exception as the cause.
    /// </summary>
    /// <param name="nJobs">The worker count.</param>
    [TestMethod]
    [DataRow(1)]
    [DataRow(3)]
    public void LinkedTokenCancellation_IsReportedWithTheCallersToken(int nJobs)
    {
        var study = NewStudy();
        using var cts = new CancellationTokenSource();

        var act = () => study.Optimize(t =>
        {
            using var linked = CancellationTokenSource.CreateLinkedTokenSource(t.CancellationToken);
            if (t.Number == 1)
                cts.Cancel();
            linked.Token.ThrowIfCancellationRequested();
            return 0.0;
        }, nTrials: 10, nJobs: nJobs, ct: cts.Token);

        var thrown = act.Should().Throw<OperationCanceledException>().Which;
        thrown.CancellationToken.Should().Be(cts.Token);
        if (nJobs == 1)
            thrown.InnerException.Should().BeOfType<OperationCanceledException>("the objective's own exception is the cause");
    }

    /// <summary>
    ///     <see cref="Trial.CancellationToken" /> is the caller's token for every trial of an <c>Optimize</c> call, on the
    ///     caller's thread and on workers, and <see cref="CancellationToken.None" /> for an asked trial.
    /// </summary>
    [TestMethod]
    public void TrialCancellationToken_IsTheCallersToken_InOptimize_AndNoneForAskedTrials()
    {
        var study = NewStudy();
        using var cts = new CancellationTokenSource();
        var seen = new ConcurrentBag<CancellationToken>();

        study.Optimize(t => { seen.Add(t.CancellationToken); return 0.0; }, nTrials: 3, ct: cts.Token);
        study.Optimize(t => { seen.Add(t.CancellationToken); return 0.0; }, nTrials: 6, nJobs: 3, ct: cts.Token);

        seen.Should().HaveCount(9).And.OnlyContain(token => token == cts.Token);
        study.Ask().CancellationToken.Should().Be(CancellationToken.None);
    }

    /// <summary>
    ///     <c>Stop()</c> keeps Python's meaning when the token is never cancelled: the run ends without an exception. When
    ///     both happen, the cancellation is reported.
    /// </summary>
    [TestMethod]
    public void Stop_StillEndsTheRunQuietly_UnlessTheTokenIsAlsoCancelled()
    {
        using var cts = new CancellationTokenSource();
        var stopped = NewStudy();
        stopped.Optimize(t => 0.0, nTrials: 10, callbacks: new Action<Study, FrozenTrial>[] { (s, _) => s.Stop() }, ct: cts.Token);
        stopped.Trials.Should().ContainSingle();

        var both = NewStudy();
        var act = () => both.Optimize(t => 0.0, nTrials: 10,
            callbacks: new Action<Study, FrozenTrial>[] { (s, _) => { s.Stop(); cts.Cancel(); } }, ct: cts.Token);
        act.Should().Throw<OperationCanceledException>();
        both.Trials.Should().ContainSingle();
    }

    /// <summary>
    ///     The motivation, on a persistent journal: after a cancellation every trial is finished, so a new storage over
    ///     the file sees no RUNNING trial, where a process killed by Ctrl+C would have left the trials in flight RUNNING
    ///     for good (a journal never fails them).
    /// </summary>
    [TestMethod]
    public void CancelledRunOnAJournal_LeavesNoTrialRunning()
    {
        string path = Path.Combine(Path.GetTempPath(), $"optunasharp-cancel-{Guid.NewGuid():N}.log");
        try
        {
            var study = Optuna.CreateStudy(storage: new JournalStorage(new JournalFileBackend(path)),
                sampler: new RandomSampler(seed: 0));
            using var cts = new CancellationTokenSource();
            var act = () => study.Optimize(t =>
            {
                if (t.Number == 9)
                    cts.Cancel();
                Thread.Sleep(5);
                return t.SuggestFloat("x", 0, 1);
            }, nTrials: 100, nJobs: 4, ct: cts.Token);

            act.Should().Throw<OperationCanceledException>();
            var reopened = Optuna.LoadStudy(study.StudyName, new JournalStorage(new JournalFileBackend(path)));
            reopened.Trials.Should().NotBeEmpty().And.NotContain(t => t.State == TrialState.Running);
        }
        finally
        {
            File.Delete(path);
        }
    }
}
