using System;
using System.Collections.Concurrent;
using System.Globalization;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using AwesomeAssertions;
using Microsoft.Extensions.DependencyInjection;
using NumSharp;
using OptunaSharp.Extensions;
using OptunaSharp.Internal;
using OptunaSharp.Samplers;

namespace OptunaSharp.Tests;

/// <summary>
///     The executor options of <c>Optimize(nJobs)</c> (plan 10, Phase 1; THREADING.md §3.5): which threads run the
///     trials, resolved at the start of each call from four levels (the call's argument, the study's setting, which
///     the DI study factory fills, and the process-wide default), read once, and validated. The pool option is the
///     plain .NET thread pool by the owner's decision; these tests pin that it keeps Python's loop semantics and the
///     thread hygiene the dedicated threads have, and that it never inlines a trial onto the waiting caller.
/// </summary>
/// <remarks>
///     Several tests change the process-wide <see cref="TrialExecutorOptions.Default" />, so the class never runs in
///     parallel with other tests, and every test restores the default it found.
/// </remarks>
[TestClass]
[DoNotParallelize]
public class TrialExecutorOptionsTests
{
    private TrialExecutorOptions _savedDefault = null!;

    /// <summary>Remembers the process-wide default, which some tests replace.</summary>
    [TestInitialize]
    public void SaveDefault() => _savedDefault = TrialExecutorOptions.Default;

    /// <summary>Puts the process-wide default back, so no other test runs on a pool this class chose.</summary>
    [TestCleanup]
    public void RestoreDefault() => TrialExecutorOptions.Default = _savedDefault;

    /// <summary>Fresh options selecting the .NET thread pool; fresh each time, because some tests change them.</summary>
    /// <returns>A new instance with <see cref="TrialExecutorKind.ThreadPool" />.</returns>
    private static TrialExecutorOptions PoolOptions() => new() { Executor = TrialExecutorKind.ThreadPool };

    /// <summary>Fresh options selecting OptunaSharp's dedicated threads explicitly.</summary>
    /// <returns>A new instance with <see cref="TrialExecutorKind.DedicatedThreads" />.</returns>
    private static TrialExecutorOptions DedicatedOptions() => new() { Executor = TrialExecutorKind.DedicatedThreads };

    /// <summary>Runs a cheap study and reports, per trial, whether its objective ran on a thread-pool thread.</summary>
    /// <param name="study">The study to optimize.</param>
    /// <param name="nJobs">The worker count of the call.</param>
    /// <param name="executorOptions">The call's executor argument, or null to let the study and the default decide.</param>
    /// <returns>One entry per trial: true when that trial ran on a pool thread.</returns>
    private static bool[] RunAndRecordPoolThreads(Study study, int nJobs, TrialExecutorOptions? executorOptions = null)
    {
        var onPool = new ConcurrentBag<bool>();
        study.Optimize(t =>
        {
            onPool.Add(Thread.CurrentThread.IsThreadPoolThread);
            return t.SuggestFloat("x", 0, 1);
        }, nTrials: 16, nJobs: nJobs, executorOptions: executorOptions);
        return onPool.ToArray();
    }

    // ------------------------------------------------------------------ the type

    /// <summary>
    ///     A new instance is the behavior of a study with no options anywhere: dedicated threads that retire after
    ///     30 s, the values of the executor that shipped before these options existed.
    /// </summary>
    [TestMethod]
    public void NewOptions_AreDedicatedThreadsWithAThirtySecondIdleTimeout()
    {
        var options = new TrialExecutorOptions();

        options.Executor.Should().Be(TrialExecutorKind.DedicatedThreads);
        options.ThreadIdleTimeout.Should().Be(TimeSpan.FromSeconds(30));
        TrialExecutorOptions.Default.Should().NotBeNull();
    }

    /// <summary>The process-wide default can be replaced, never removed.</summary>
    [TestMethod]
    public void Default_RefusesNull()
    {
        var act = () => TrialExecutorOptions.Default = null!;

        act.Should().Throw<ArgumentNullException>();
        TrialExecutorOptions.Default.Should().BeSameAs(_savedDefault);
    }

    /// <summary>
    ///     The idle timeout must be one a worker can wait for: positive and at most <see cref="int.MaxValue" />
    ///     milliseconds (<c>Monitor.Wait</c>'s limit), or infinite. Zero would retire a worker after every trial and
    ///     bring back the thread-per-trial cost of review item P3-4.
    /// </summary>
    /// <param name="milliseconds">The rejected timeout, in milliseconds.</param>
    [TestMethod]
    [DataRow(0.0)]
    [DataRow(-1000.0)]
    [DataRow(int.MaxValue + 1.0)]
    public void ThreadIdleTimeout_RefusesATimeoutAWorkerCannotWaitFor(double milliseconds)
    {
        var options = new TrialExecutorOptions();

        var act = () => options.ThreadIdleTimeout = TimeSpan.FromMilliseconds(milliseconds);

        act.Should().Throw<ArgumentOutOfRangeException>().Which.ParamName.Should().Be("value");
        options.ThreadIdleTimeout.Should().Be(TimeSpan.FromSeconds(30), "a refused value leaves the old one");
    }

    /// <summary>An infinite idle timeout keeps idle workers; one millisecond is the shortest wait accepted.</summary>
    [TestMethod]
    public void ThreadIdleTimeout_AcceptsInfiniteAndTheShortestPositiveWait()
    {
        var options = new TrialExecutorOptions { ThreadIdleTimeout = Timeout.InfiniteTimeSpan };
        options.ThreadIdleTimeout.Should().Be(Timeout.InfiniteTimeSpan);

        options.ThreadIdleTimeout = TimeSpan.FromMilliseconds(1);
        options.ThreadIdleTimeout.Should().Be(TimeSpan.FromMilliseconds(1));
    }

    /// <summary>Only the two defined executor kinds are accepted.</summary>
    [TestMethod]
    public void Executor_RefusesAnUndefinedKind()
    {
        var options = new TrialExecutorOptions();

        var act = () => options.Executor = (TrialExecutorKind)7;

        act.Should().Throw<ArgumentOutOfRangeException>();
        options.Executor.Should().Be(TrialExecutorKind.DedicatedThreads);
    }

    /// <summary>
    ///     A new instance keeps Optuna's model for the sampler (one shared instance, one call at a time) and leaves
    ///     OpenBLAS to the automatic policy (untouched with <c>nJobs: 1</c>, single-threaded otherwise).
    /// </summary>
    [TestMethod]
    public void NewOptions_ShareTheSampler_AndLeaveOpenBlasToTheAutomaticPolicy()
    {
        var options = new TrialExecutorOptions();

        options.SamplerConcurrency.Should().Be(SamplerConcurrency.Shared);
        options.BlasThreads.Should().BeNull();
    }

    /// <summary>Only the two defined sampler concurrency modes are accepted; a refused value leaves the old one.</summary>
    [TestMethod]
    public void SamplerConcurrency_RefusesAnUndefinedMode()
    {
        var options = new TrialExecutorOptions { SamplerConcurrency = SamplerConcurrency.PerWorker };

        var act = () => options.SamplerConcurrency = (SamplerConcurrency)5;

        act.Should().Throw<ArgumentOutOfRangeException>().Which.ParamName.Should().Be("value");
        options.SamplerConcurrency.Should().Be(SamplerConcurrency.PerWorker);
    }

    /// <summary>
    ///     An OpenBLAS thread count must be positive; zero is refused rather than read as "automatic", which is what
    ///     <see langword="null" /> says. A refused value leaves the old one.
    /// </summary>
    /// <param name="threads">The refused count.</param>
    [TestMethod]
    [DataRow(0)]
    [DataRow(-1)]
    [DataRow(int.MinValue)]
    public void BlasThreads_RefusesZeroAndNegativeCounts(int threads)
    {
        var options = new TrialExecutorOptions { BlasThreads = 3 };

        var act = () => options.BlasThreads = threads;

        act.Should().Throw<ArgumentOutOfRangeException>().Which.ParamName.Should().Be("value");
        options.BlasThreads.Should().Be(3);
    }

    /// <summary>
    ///     Null (the automatic policy) and any positive count round-trip, the largest included: the stored form uses 0
    ///     for null, so no positive count may collide with it.
    /// </summary>
    [TestMethod]
    public void BlasThreads_AcceptsNullAndAnyPositiveCount()
    {
        var options = new TrialExecutorOptions { BlasThreads = 1 };
        options.BlasThreads.Should().Be(1);

        options.BlasThreads = int.MaxValue;
        options.BlasThreads.Should().Be(int.MaxValue);

        options.BlasThreads = null;
        options.BlasThreads.Should().BeNull();
    }

    // ------------------------------------------------------------------ which threads run the trials

    /// <summary>The thread-pool executor runs every trial on a pool thread.</summary>
    [TestMethod]
    public void ThreadPool_RunsEveryTrialOnAPoolThread()
    {
        var study = Optuna.CreateStudy(sampler: new RandomSampler(1));

        RunAndRecordPoolThreads(study, nJobs: 4, PoolOptions())
            .Should().HaveCount(16).And.OnlyContain(p => p);
    }

    /// <summary>Choosing dedicated threads explicitly keeps every trial off the pool, as the default does.</summary>
    [TestMethod]
    public void DedicatedThreads_Explicit_RunsNoTrialOnThePool()
    {
        var study = Optuna.CreateStudy(sampler: new RandomSampler(2));

        RunAndRecordPoolThreads(study, nJobs: 4, DedicatedOptions())
            .Should().HaveCount(16).And.OnlyContain(p => !p);
    }

    /// <summary>
    ///     The dedicated threads reserve 16 MB of stack, CPython's own thread size on macOS (plan 10, item 2.5, the owner's
    ///     choice of 2026-11-07): native code written for Python's threads, such as numba's JIT compiling pymoo's AGE-MOEA
    ///     kernels, overflowed .NET's default 1.5 MB stack. Read from Windows' stack limits of each worker.
    /// </summary>
    [TestMethod]
    [WindowsOnly]
    public void DedicatedThreads_Reserve16MBOfStack()
    {
        if (!OperatingSystem.IsWindows())
            Assert.Inconclusive("The stack limits are read with GetCurrentThreadStackLimits, a Windows API.");
        var study = Optuna.CreateStudy(sampler: new RandomSampler(5));
        var reserved = new ConcurrentBag<long>();

        study.Optimize(t =>
        {
            GetCurrentThreadStackLimits(out nuint low, out nuint high);
            reserved.Add((long)(high - low));
            return t.SuggestFloat("x", 0, 1);
        }, nTrials: 8, nJobs: 4, executorOptions: DedicatedOptions());

        reserved.Should().HaveCount(8).And.OnlyContain(bytes => bytes >= TrialWorkerScheduler.WorkerStackSize);
    }

    /// <summary>Windows' bounds of the calling thread's reserved stack.</summary>
    /// <param name="lowLimit">The lowest address of the stack.</param>
    /// <param name="highLimit">The address just above the stack.</param>
    [System.Runtime.InteropServices.DllImport("kernel32.dll")]
    private static extern void GetCurrentThreadStackLimits(out nuint lowLimit, out nuint highLimit);

    /// <summary>
    ///     <c>nJobs: 1</c> runs every trial on the calling thread, as in Python, whatever executor is in effect; the
    ///     executor only decides where the workers of <c>nJobs</c> other than 1 run.
    /// </summary>
    [TestMethod]
    public void NJobsOne_RunsEveryTrialOnTheCallingThread_WhateverTheExecutor()
    {
        var study = Optuna.CreateStudy(sampler: new RandomSampler(3));
        int caller = Environment.CurrentManagedThreadId;
        var threads = new ConcurrentBag<int>();

        study.Optimize(t =>
        {
            threads.Add(Environment.CurrentManagedThreadId);
            return t.SuggestFloat("x", 0, 1);
        }, nTrials: 10, nJobs: 1, executorOptions: PoolOptions());

        threads.Should().HaveCount(10).And.OnlyContain(id => id == caller);
    }

    /// <summary>
    ///     <c>nJobs</c> trials really run at once on the pool while <c>nJobs</c> stays below the pool's minimum: every
    ///     trial waits on a barrier that opens only when all of them stand at it. Above the minimum the pool may run
    ///     fewer, which <see cref="TrialExecutorKind.ThreadPool" /> documents; this test stays below it.
    /// </summary>
    [TestMethod]
    public void ThreadPool_RunsNJobsTrialsAtOnce_BelowThePoolMinimum()
    {
        ThreadPool.GetMinThreads(out int minWorkers, out _);
        int nJobs = Math.Max(2, Math.Min(4, minWorkers));
        var study = Optuna.CreateStudy(sampler: new RandomSampler(4));
        using var barrier = new Barrier(nJobs);
        int timedOut = 0;

        study.Optimize(t =>
        {
            if (!barrier.SignalAndWait(TimeSpan.FromSeconds(30)))
                Interlocked.Increment(ref timedOut);
            return t.SuggestFloat("x", 0, 1);
        }, nTrials: 3 * nJobs, nJobs: nJobs, executorOptions: PoolOptions());

        timedOut.Should().Be(0, "nJobs trials must be able to run at the same time on the pool");
        study.Trials.Should().HaveCount(3 * nJobs).And.OnlyContain(t => t.State == TrialState.Complete);
    }

    /// <summary>
    ///     An <c>Optimize</c> called on a pool thread never runs one of its own pool trials on that thread. Workers are
    ///     queued with <see cref="TaskCreationOptions.PreferFairness" /> on the pool's global queue, from which the
    ///     caller's final <c>Task.WaitAll</c> cannot pop them. Without it the TPL may inline a not-yet-started trial
    ///     onto the waiting caller, outside the worker accounting and with the caller's thread-local state.
    /// </summary>
    [TestMethod]
    public void ThreadPool_OptimizeCalledFromAPoolThread_NeverRunsATrialOnThatThread()
    {
        var study = Optuna.CreateStudy(sampler: new RandomSampler(5));
        int ranOnCaller = 0;

        Task.Run(() =>
        {
            int caller = Environment.CurrentManagedThreadId;
            study.Optimize(t =>
            {
                if (Environment.CurrentManagedThreadId == caller)
                    Interlocked.Increment(ref ranOnCaller);
                Thread.Sleep(2);
                return t.SuggestFloat("x", 0, 1);
            }, nTrials: 64, nJobs: 4, executorOptions: PoolOptions());
        }).Wait();

        ranOnCaller.Should().Be(0);
        study.Trials.Should().HaveCount(64);
    }

    /// <summary>
    ///     A pool thread that ran a trial leaves no optimize-loop state behind: later single-job <c>Optimize</c> calls on
    ///     pool threads are not mistaken for nested calls. Pool threads are reused by the host, so this is review item
    ///     H5 (61 of 64 such calls failed while workers were pool threads without the restore) on the pool executor.
    /// </summary>
    [TestMethod]
    public void ThreadPool_AfterARun_PoolThreadsCarryNoOptimizeLoopState()
    {
        var study = Optuna.CreateStudy(sampler: new RandomSampler(6));
        study.Optimize(t => t.SuggestFloat("x", 0, 1), nTrials: 64, nJobs: 8, executorOptions: PoolOptions());

        int nested = 0;
        Parallel.For(0, 64, _ =>
        {
            try
            {
                study.Optimize(t => t.SuggestFloat("x", 0, 1), nTrials: 1);
            }
            catch (RuntimeError)
            {
                Interlocked.Increment(ref nested);
            }
        });

        nested.Should().Be(0);
        study.Trials.Should().HaveCount(128);
    }

    // ------------------------------------------------------------------ Python's loop semantics on both executors

    /// <summary>
    ///     An uncaught objective error waits for the trials still running, then raises the original exception, as
    ///     Python's executor block does; the executor does not change that.
    /// </summary>
    /// <param name="kind">The executor that runs the trials.</param>
    /// <exception cref="InvalidOperationException">Thrown by trial 3's objective; the assertion expects Optimize to rethrow it.</exception>
    [TestMethod]
    [DataRow(TrialExecutorKind.DedicatedThreads)]
    [DataRow(TrialExecutorKind.ThreadPool)]
    public void ObjectiveError_WaitsForRunningTrials_ThenRaisesTheOriginalException(TrialExecutorKind kind)
    {
        var study = Optuna.CreateStudy(sampler: new RandomSampler(7));

        var act = () => study.Optimize(t =>
        {
            double x = t.SuggestFloat("x", 0, 1);
            if (t.Number == 3)
                throw new InvalidOperationException("boom");
            Thread.Sleep(100);
            return x;
        }, nTrials: 40, nJobs: 4, executorOptions: new TrialExecutorOptions { Executor = kind });

        act.Should().Throw<InvalidOperationException>().WithMessage("boom");
        study.Trials.Should().NotContain(t => t.State == TrialState.Running, "Optimize returns only once its trials have finished");
    }

    /// <summary>A failure among the trials after the final submission is not raised, as in Python, on both executors.</summary>
    /// <param name="kind">The executor that runs the trials.</param>
    /// <exception cref="InvalidOperationException">Thrown by every objective; Optimize records each trial as FAIL and does not rethrow.</exception>
    [TestMethod]
    [DataRow(TrialExecutorKind.DedicatedThreads)]
    [DataRow(TrialExecutorKind.ThreadPool)]
    public void ErrorInTheFinalBatch_IsNotRaised_AsInPython(TrialExecutorKind kind)
    {
        var study = Optuna.CreateStudy(sampler: new RandomSampler(8));

        study.Optimize((Func<Trial, double>)(_ =>
        {
            Thread.Sleep(50);
            throw new InvalidOperationException("late");
        }), nTrials: 3, nJobs: 4, executorOptions: new TrialExecutorOptions { Executor = kind });

        study.Trials.Should().HaveCount(3).And.OnlyContain(t => t.State == TrialState.Fail);
    }

    /// <summary>
    ///     A <c>Stop()</c> from a callback ends scheduling on both executors: the trials in flight finish, and the run
    ///     ends at most <c>nJobs - 1</c> trials past the stopping trial, as in Python.
    /// </summary>
    /// <param name="kind">The executor that runs the trials.</param>
    [TestMethod]
    [DataRow(TrialExecutorKind.DedicatedThreads)]
    [DataRow(TrialExecutorKind.ThreadPool)]
    public void StopFromACallback_EndsScheduling_OnBothExecutors(TrialExecutorKind kind)
    {
        var study = Optuna.CreateStudy(sampler: new RandomSampler(9));

        study.Optimize(t => t.SuggestFloat("x", 0, 1), nTrials: 1000, nJobs: 4,
            callbacks: new Action<Study, FrozenTrial>[] { (s, frozen) => { if (frozen.Number >= 9) s.Stop(); } },
            executorOptions: new TrialExecutorOptions { Executor = kind });

        study.Trials.Count.Should().BeInRange(10, 10 + 4 + 4, "a Stop() lets only the trials in flight finish");
        study.Trials.Should().NotContain(t => t.State == TrialState.Running);
    }

    /// <summary>
    ///     Workers and callbacks inherit the caller's <see cref="ExecutionContext" /> on both executors: its cultures, so
    ///     an objective formats and parses numbers as it does sequentially, and its <see cref="AsyncLocal{T}" /> values,
    ///     which also carry <c>Activity.Current</c> and logging scopes (THREADING.md §2). Python's executor threads start
    ///     without the caller's <c>contextvars</c>; this is the .NET behavior, pinned because a scheduler that ran its
    ///     tasks outside their captured context would silently change it.
    /// </summary>
    /// <param name="kind">The executor that runs the trials.</param>
    [TestMethod]
    [DataRow(TrialExecutorKind.DedicatedThreads)]
    [DataRow(TrialExecutorKind.ThreadPool)]
    public void Workers_InheritTheCallersCultureAndAsyncLocals_OnBothExecutors(TrialExecutorKind kind)
    {
        var flowing = new AsyncLocal<string?>();
        var seen = new ConcurrentBag<(string Culture, string UiCulture, string? Value, bool OnAWorker)>();
        var study = Optuna.CreateStudy(sampler: new RandomSampler(10));
        int callerThread = Environment.CurrentManagedThreadId;

        // Records what the code running now sees. A worker that ran without the caller's context would report its own
        // thread's cultures and a null value.
        void Record() => seen.Add((CultureInfo.CurrentCulture.Name, CultureInfo.CurrentUICulture.Name, flowing.Value,
            Environment.CurrentManagedThreadId != callerThread));

        CultureInfo savedCulture = CultureInfo.CurrentCulture, savedUiCulture = CultureInfo.CurrentUICulture;
        try
        {
            // Set on the caller only, and different from any test machine's own culture and from the invariant one.
            CultureInfo.CurrentCulture = CultureInfo.GetCultureInfo("de-DE");
            CultureInfo.CurrentUICulture = CultureInfo.GetCultureInfo("fr-FR");
            flowing.Value = "caller";

            study.Optimize(t =>
                {
                    Record();
                    return t.SuggestFloat("x", 0, 1);
                }, nTrials: 16, nJobs: 4,
                callbacks: new Action<Study, FrozenTrial>[] { (_, _) => Record() },
                executorOptions: new TrialExecutorOptions { Executor = kind });
        }
        finally
        {
            // The cultures flow with this thread's context too; put them back so no later test inherits them.
            CultureInfo.CurrentCulture = savedCulture;
            CultureInfo.CurrentUICulture = savedUiCulture;
        }

        seen.Should().HaveCount(32, "16 objectives and 16 callbacks");
        seen.Should().OnlyContain(s => s.OnAWorker, "with nJobs above 1 every trial and callback runs on a worker");
        seen.Should().OnlyContain(s => s.Culture == "de-DE" && s.UiCulture == "fr-FR" && s.Value == "caller");
    }

    // ------------------------------------------------------------------ resolution: argument > study > default

    /// <summary>
    ///     The first level that is set wins: the call's argument over the study's setting, the study's setting over
    ///     the process-wide default, and the default when neither is set.
    /// </summary>
    [TestMethod]
    public void Resolution_TheArgumentBeatsTheStudySetting_WhichBeatsTheDefault()
    {
        TrialExecutorOptions.Default = PoolOptions();
        var study = Optuna.CreateStudy(sampler: new RandomSampler(10));
        study.ExecutorOptions = DedicatedOptions();

        RunAndRecordPoolThreads(study, nJobs: 4, PoolOptions()).Should().OnlyContain(p => p, "the argument wins");
        RunAndRecordPoolThreads(study, nJobs: 4).Should().OnlyContain(p => !p, "the study setting beats the default");

        study.ExecutorOptions = null;
        RunAndRecordPoolThreads(study, nJobs: 4).Should().OnlyContain(p => p, "with neither set, the default applies");
    }

    /// <summary>
    ///     The process-wide default is read at the start of every call, so a new default reaches a study that never
    ///     chose its own, without recreating it.
    /// </summary>
    [TestMethod]
    public void Resolution_ANewDefaultReachesAStudyThatNeverChose()
    {
        var study = Optuna.CreateStudy(sampler: new RandomSampler(11));
        RunAndRecordPoolThreads(study, nJobs: 4).Should().OnlyContain(p => !p);

        TrialExecutorOptions.Default = PoolOptions();

        study.ExecutorOptions.Should().BeNull("the study never chose");
        RunAndRecordPoolThreads(study, nJobs: 4).Should().OnlyContain(p => p);
    }

    /// <summary>Studies from <c>CreateStudy</c> and <c>LoadStudy</c> start with no executor setting of their own.</summary>
    [TestMethod]
    public void CreateAndLoad_LeaveTheStudySettingUnset()
    {
        var storage = new OptunaSharp.Storages.InMemoryStorage();
        var created = Optuna.CreateStudy(storage: storage, studyName: "executor-unset");
        created.ExecutorOptions = PoolOptions();

        var loaded = Optuna.LoadStudy("executor-unset", storage);

        loaded.ExecutorOptions.Should().BeNull("the setting belongs to the study object and is not stored");
        Optuna.CreateStudy().ExecutorOptions.Should().BeNull();
    }

    // ------------------------------------------------------------------ the snapshot

    /// <summary>
    ///     A call reads its options once, when it starts: changing the options object while the call runs leaves the
    ///     call on the executor it started with, and the next call sees the change.
    /// </summary>
    [TestMethod]
    public void Snapshot_ChangingTheOptionsDuringARun_DoesNotChangeThatRun()
    {
        var study = Optuna.CreateStudy(sampler: new RandomSampler(12));
        var options = PoolOptions();
        var onPool = new ConcurrentBag<bool>();

        study.Optimize(t =>
        {
            if (t.Number == 0)
                options.Executor = TrialExecutorKind.DedicatedThreads;
            onPool.Add(Thread.CurrentThread.IsThreadPoolThread);
            return t.SuggestFloat("x", 0, 1);
        }, nTrials: 24, nJobs: 4, executorOptions: options);

        onPool.Should().HaveCount(24).And.OnlyContain(p => p, "the run started on the pool and stays there");
        RunAndRecordPoolThreads(study, nJobs: 4, options).Should().OnlyContain(p => !p, "the next call sees the change");
    }

    /// <summary>Options a derived class computes, counting how often each getter runs.</summary>
    private sealed class CountingOptions : TrialExecutorOptions
    {
        /// <summary>How many times <see cref="Executor" /> was read.</summary>
        public int ExecutorReads;

        /// <summary>How many times <see cref="ThreadIdleTimeout" /> was read.</summary>
        public int TimeoutReads;

        /// <summary>How many times <see cref="SamplerConcurrency" /> was read.</summary>
        public int SamplerConcurrencyReads;

        /// <summary>How many times <see cref="BlasThreads" /> was read.</summary>
        public int BlasThreadsReads;

        /// <summary>Always the shared sampler, the default; counts each read.</summary>
        public override SamplerConcurrency SamplerConcurrency
        {
            get
            {
                Interlocked.Increment(ref SamplerConcurrencyReads);
                return SamplerConcurrency.Shared;
            }
            set { }
        }

        /// <summary>Always the automatic policy, the default; counts each read.</summary>
        public override int? BlasThreads
        {
            get
            {
                Interlocked.Increment(ref BlasThreadsReads);
                return null;
            }
            set { }
        }

        /// <summary>Always the thread pool; counts each read.</summary>
        public override TrialExecutorKind Executor
        {
            get
            {
                Interlocked.Increment(ref ExecutorReads);
                return TrialExecutorKind.ThreadPool;
            }
            set { }
        }

        /// <summary>Always five seconds; counts each read.</summary>
        public override TimeSpan ThreadIdleTimeout
        {
            get
            {
                Interlocked.Increment(ref TimeoutReads);
                return TimeSpan.FromSeconds(5);
            }
            set { }
        }
    }

    /// <summary>
    ///     The class is inheritable, and a derived class's overrides are honored and read exactly once per call, so a
    ///     computed value cannot change in the middle of a run.
    /// </summary>
    [TestMethod]
    public void DerivedOptions_AreHonored_AndEachGetterIsReadOncePerCall()
    {
        var study = Optuna.CreateStudy(sampler: new RandomSampler(13));
        var options = new CountingOptions();

        RunAndRecordPoolThreads(study, nJobs: 4, options).Should().OnlyContain(p => p, "the override selects the pool");

        options.ExecutorReads.Should().Be(1);
        options.TimeoutReads.Should().Be(1);
        options.SamplerConcurrencyReads.Should().Be(1);
        options.BlasThreadsReads.Should().Be(1);
    }

    /// <summary>A derived class that returns values the setters would refuse.</summary>
    private sealed class InvalidOptions : TrialExecutorOptions
    {
        private readonly TrialExecutorKind _executor;
        private readonly TimeSpan _timeout;
        private readonly SamplerConcurrency _samplerConcurrency;
        private readonly int? _blasThreads;

        /// <summary>Creates options whose getters return the given values unchecked.</summary>
        /// <param name="executor">The executor kind the getter returns.</param>
        /// <param name="timeout">The idle timeout the getter returns.</param>
        /// <param name="samplerConcurrency">The sampler concurrency mode the getter returns.</param>
        /// <param name="blasThreads">The OpenBLAS thread count the getter returns.</param>
        public InvalidOptions(
            TrialExecutorKind executor,
            TimeSpan timeout,
            SamplerConcurrency samplerConcurrency = SamplerConcurrency.Shared,
            int? blasThreads = null)
        {
            _executor = executor;
            _timeout = timeout;
            _samplerConcurrency = samplerConcurrency;
            _blasThreads = blasThreads;
        }

        /// <summary>The configured kind, unchecked.</summary>
        public override TrialExecutorKind Executor { get => _executor; set { } }

        /// <summary>The configured timeout, unchecked.</summary>
        public override TimeSpan ThreadIdleTimeout { get => _timeout; set { } }

        /// <summary>The configured sampler concurrency mode, unchecked.</summary>
        public override SamplerConcurrency SamplerConcurrency { get => _samplerConcurrency; set { } }

        /// <summary>The configured OpenBLAS thread count, unchecked.</summary>
        public override int? BlasThreads { get => _blasThreads; set { } }
    }

    /// <summary>
    ///     Values a derived class returns are validated when the call starts, as the setters validate theirs, before
    ///     any trial runs and whatever <c>nJobs</c> is.
    /// </summary>
    /// <param name="nJobs">The worker count of the refused call.</param>
    [TestMethod]
    [DataRow(1)]
    [DataRow(4)]
    public void DerivedOptions_ReturningInvalidValues_AreRefusedBeforeAnyTrial(int nJobs)
    {
        var study = Optuna.CreateStudy(sampler: new RandomSampler(14));

        var badKind = () => study.Optimize(t => t.SuggestFloat("x", 0, 1), nTrials: 4, nJobs: nJobs,
            executorOptions: new InvalidOptions((TrialExecutorKind)9, TimeSpan.FromSeconds(1)));
        var badTimeout = () => study.Optimize(t => t.SuggestFloat("x", 0, 1), nTrials: 4, nJobs: nJobs,
            executorOptions: new InvalidOptions(TrialExecutorKind.DedicatedThreads, TimeSpan.Zero));
        var badConcurrency = () => study.Optimize(t => t.SuggestFloat("x", 0, 1), nTrials: 4, nJobs: nJobs,
            executorOptions: new InvalidOptions(TrialExecutorKind.DedicatedThreads, TimeSpan.FromSeconds(1),
                samplerConcurrency: (SamplerConcurrency)9));
        var badBlasThreads = () => study.Optimize(t => t.SuggestFloat("x", 0, 1), nTrials: 4, nJobs: nJobs,
            executorOptions: new InvalidOptions(TrialExecutorKind.DedicatedThreads, TimeSpan.FromSeconds(1),
                blasThreads: 0));

        badKind.Should().Throw<ArgumentOutOfRangeException>().Which.ParamName.Should().Be(nameof(TrialExecutorOptions.Executor));
        badTimeout.Should().Throw<ArgumentOutOfRangeException>().Which.ParamName.Should().Be(nameof(TrialExecutorOptions.ThreadIdleTimeout));
        badConcurrency.Should().Throw<ArgumentOutOfRangeException>().Which.ParamName.Should().Be(nameof(TrialExecutorOptions.SamplerConcurrency));
        badBlasThreads.Should().Throw<ArgumentOutOfRangeException>().Which.ParamName.Should().Be(nameof(TrialExecutorOptions.BlasThreads));
        study.Trials.Should().BeEmpty();
    }

    // ------------------------------------------------------------------ the idle timeout's thread sets

    /// <summary>
    ///     Calls with equal idle timeouts share one set of worker threads, distinct timeouts get sets of their own, and
    ///     the default timeout is the shared set every call used before the options existed.
    /// </summary>
    [TestMethod]
    public void ThreadIdleTimeout_EqualTimeoutsShareOneThreadSet_DistinctOnesDoNot()
    {
        TrialWorkerScheduler.For(TimeSpan.FromSeconds(30)).Should().BeSameAs(TrialWorkerScheduler.Shared);
        TrialWorkerScheduler.For(TimeSpan.FromSeconds(7)).Should().BeSameAs(TrialWorkerScheduler.For(TimeSpan.FromSeconds(7)));
        TrialWorkerScheduler.For(TimeSpan.FromSeconds(7)).Should().NotBeSameAs(TrialWorkerScheduler.For(TimeSpan.FromSeconds(8)));

        var scheduler = TrialWorkerScheduler.For(TimeSpan.FromSeconds(7));
        long before = scheduler.ThreadsStarted;
        var study = Optuna.CreateStudy(sampler: new RandomSampler(15));

        RunAndRecordPoolThreads(study, nJobs: 4, new TrialExecutorOptions { ThreadIdleTimeout = TimeSpan.FromSeconds(7) })
            .Should().OnlyContain(p => !p);
        (scheduler.ThreadsStarted - before).Should().BeGreaterThan(0, "the 7 s set ran the trials");
        scheduler.IdleTimeout.Should().Be(TimeSpan.FromSeconds(7));
    }

    /// <summary>A short idle timeout from the options retires its workers, which restart for the next call.</summary>
    [TestMethod]
    public void ThreadIdleTimeout_AShortTimeout_RetiresItsWorkers_AndTheyRestart()
    {
        var timeout = TimeSpan.FromMilliseconds(150);
        var scheduler = TrialWorkerScheduler.For(timeout);
        var study = Optuna.CreateStudy(sampler: new RandomSampler(16));
        var options = new TrialExecutorOptions { ThreadIdleTimeout = timeout };

        RunAndRecordPoolThreads(study, nJobs: 4, options);
        WaitUntil(() => scheduler.ThreadCount == 0, "idle workers retire after the options' timeout");

        RunAndRecordPoolThreads(study, nJobs: 4, options).Should().HaveCount(16);
    }

    // ------------------------------------------------------------------ DI

    /// <summary>
    ///     The DI study factory gives every study it builds the configured executor options, the DI level of the
    ///     resolution; the same instance goes to each.
    /// </summary>
    [TestMethod]
    public void StudyFactory_GivesEveryStudyTheConfiguredExecutorOptions()
    {
        var pool = PoolOptions();
        var services = new ServiceCollection();
        services.AddOptunaStudyFactory(o => o.DefaultExecutorOptions = pool);
        using var provider = services.BuildServiceProvider();
        var factory = provider.GetRequiredService<IOptunaStudyFactory>();

        var a = factory.CreateStudy();
        var b = factory.CreateStudy();

        a.ExecutorOptions.Should().BeSameAs(pool);
        b.ExecutorOptions.Should().BeSameAs(pool);
        RunAndRecordPoolThreads(a, nJobs: 4).Should().OnlyContain(p => p);
    }

    /// <summary>Without a configured executor, the factory leaves the study's setting unset, as <c>CreateStudy</c> does.</summary>
    [TestMethod]
    public void StudyFactory_WithoutExecutorOptions_LeavesTheStudySettingUnset()
    {
        var services = new ServiceCollection();
        services.AddOptunaStudyFactory();
        using var provider = services.BuildServiceProvider();

        provider.GetRequiredService<IOptunaStudyFactory>().CreateStudy().ExecutorOptions.Should().BeNull();
    }

    /// <summary>Polls <paramref name="condition" /> for up to ten seconds; fails the test with <paramref name="what" /> if it never holds.</summary>
    /// <param name="condition">The state the scheduler must reach.</param>
    /// <param name="what">What the condition means, for the failure message.</param>
    /// <exception cref="AssertFailedException">The condition did not hold within ten seconds.</exception>
    private static void WaitUntil(Func<bool> condition, string what)
    {
        var watch = System.Diagnostics.Stopwatch.StartNew();
        while (!condition())
        {
            if (watch.Elapsed > TimeSpan.FromSeconds(10))
                Assert.Fail($"Timed out waiting until {what}.");
            Thread.Sleep(10);
        }
    }
}
