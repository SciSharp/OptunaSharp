using System;
using System.Diagnostics;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using AwesomeAssertions;
using OptunaSharp.Internal;
using OptunaSharp.Samplers;

namespace OptunaSharp.Tests;

/// <summary>
///     The <c>Optimize(nJobs)</c> worker threads (<see cref="TrialWorkerScheduler" />): reused like Python's
///     <c>ThreadPoolExecutor</c> threads instead of one new thread per trial (P3-4), while keeping the properties
///     THREADING.md §3.6 requires of any executor. These are dedicated threads, not pool threads (that one is pinned by
///     <c>OptimizeConcurrencyTests.NJobs_WorkersRunOnDedicatedThreads_NotOnTheThreadPool</c>), with <c>nJobs</c> trials
///     really in flight, no inlining onto the waiting caller, and idle threads that retire.
/// </summary>
[TestClass]
public class TrialWorkerSchedulerTests
{
    /// <summary>
    ///     Threads are reused: 64 trials at <c>nJobs: 4</c> start a handful of new threads at most, not 64. Before the
    ///     scheduler every trial got a thread of its own, which capped cheap studies near 10,000 trials/s.
    /// </summary>
    /// <remarks>
    ///     The check counts threads <i>started</i> during the call, not distinct threads seen: workers are reused across
    ///     calls too, so idle threads left by earlier tests may each take some of these trials without any new thread.
    /// </remarks>
    [TestMethod]
    public void NJobs_ReusesWorkerThreads_InsteadOfOnePerTrial()
    {
        var study = Optuna.CreateStudy(sampler: new RandomSampler(3));
        long before = TrialWorkerScheduler.Shared.ThreadsStarted;

        study.Optimize(t => t.SuggestFloat("x", 0, 1), nTrials: 64, nJobs: 4);

        study.Trials.Should().HaveCount(64);
        // At most four workers, plus a few a submit/finish race may start; never one per trial.
        (TrialWorkerScheduler.Shared.ThreadsStarted - before).Should().BeLessThanOrEqualTo(12, "worker threads are reused across trials");
    }

    /// <summary>
    ///     <c>nJobs</c> trials are really in flight at once: every trial waits on a barrier that only opens when four
    ///     trials stand at it together. Under-starting threads (a queued trial waiting for a busy worker) would time
    ///     the barrier out.
    /// </summary>
    [TestMethod]
    public void NJobs_RunsNJobsTrialsAtOnce()
    {
        var study = Optuna.CreateStudy(sampler: new RandomSampler(4));
        using var barrier = new Barrier(4);
        int timedOut = 0;

        study.Optimize(t =>
        {
            if (!barrier.SignalAndWait(TimeSpan.FromSeconds(30)))
                Interlocked.Increment(ref timedOut);
            return t.SuggestFloat("x", 0, 1);
        }, nTrials: 12, nJobs: 4);

        timedOut.Should().Be(0, "four trials must be able to run at the same time");
        study.Trials.Should().HaveCount(12).And.OnlyContain(t => t.State == TrialState.Complete);
    }

    /// <summary>
    ///     A worker never runs on the thread that waits for it: the TPL may try to inline a queued task into
    ///     <see cref="Task.Wait()" />, and the scheduler refuses, so a trial never runs with the caller's thread-local state.
    /// </summary>
    [TestMethod]
    public void QueuedTask_IsNeverInlinedIntoTheWaitingThread()
    {
        var scheduler = new TrialWorkerScheduler(TimeSpan.FromSeconds(5));
        int caller = Environment.CurrentManagedThreadId;
        int ranOn = -1;

        var task = Task.Factory.StartNew(() => ranOn = Environment.CurrentManagedThreadId,
            CancellationToken.None, TaskCreationOptions.DenyChildAttach, scheduler);
        task.Wait();

        ranOn.Should().NotBe(caller);
    }

    /// <summary>
    ///     Idle worker threads end after the idle timeout, so a burst of <c>nJobs: 32</c> does not leave 32 threads
    ///     behind for the life of the process; a new burst starts threads again.
    /// </summary>
    [TestMethod]
    public void IdleWorkers_RetireAfterTheIdleTimeout_AndRestartOnDemand()
    {
        var scheduler = new TrialWorkerScheduler(TimeSpan.FromMilliseconds(200));
        using var release = new ManualResetEventSlim();
        var tasks = Enumerable.Range(0, 8).Select(_ => Task.Factory.StartNew(() => release.Wait(),
            CancellationToken.None, TaskCreationOptions.DenyChildAttach, scheduler)).ToArray();

        WaitUntil(() => scheduler.ThreadCount == 8, "eight blocked tasks need eight threads");
        release.Set();
        Task.WaitAll(tasks);
        WaitUntil(() => scheduler.ThreadCount == 0, "idle threads retire after the timeout");

        Task.Factory.StartNew(() => { }, CancellationToken.None, TaskCreationOptions.DenyChildAttach, scheduler).Wait();
    }

    /// <summary>Polls <paramref name="condition" /> for up to ten seconds; fails the test with <paramref name="what" /> if it never holds.</summary>
    /// <param name="condition">The state the scheduler must reach.</param>
    /// <param name="what">What the condition means, for the failure message.</param>
    /// <exception cref="AssertFailedException">The condition did not hold within ten seconds.</exception>
    private static void WaitUntil(Func<bool> condition, string what)
    {
        var watch = Stopwatch.StartNew();
        while (!condition())
        {
            if (watch.Elapsed > TimeSpan.FromSeconds(10))
                Assert.Fail($"Timed out waiting until {what}.");
            Thread.Sleep(10);
        }
    }
}
