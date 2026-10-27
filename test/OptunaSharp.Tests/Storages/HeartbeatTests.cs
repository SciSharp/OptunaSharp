using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using AwesomeAssertions;
using OptunaSharp.Collections;
using OptunaSharp.Distributions;
using OptunaSharp.Internal;
using OptunaSharp.Samplers;
using OptunaSharp.Storages;

namespace OptunaSharp.Tests.Storages;

/// <summary>
///     The heartbeat mechanism (plan 1.3) over a fake heartbeat-capable storage: heartbeat
///     recording during optimize, <c>fail_stale_trials</c> semantics, and the
///     <see cref="RetryHeartbeatStaleTrialCallback" /> retry-history chain.
/// </summary>
[TestClass]
public class HeartbeatTests
{
    /// <summary>An InMemoryStorage that also implements <see cref="IBaseHeartbeat" />.</summary>
    private sealed class HeartbeatStorage : InMemoryStorage, IBaseHeartbeat
    {
        internal readonly List<int> Recorded = new();
        internal readonly List<int> StaleTrialIds = new();
        internal int? Interval = 1;
        internal Action<Study, FrozenTrial>? StaleCallback;

        public void RecordHeartbeat(int trialId)
        {
            lock (Recorded)
                Recorded.Add(trialId);
        }

        public List<int> GetStaleTrialIds(int studyId) => StaleTrialIds.ToList();

        public int? GetHeartbeatInterval() => Interval;

        public Action<Study, FrozenTrial>? GetHeartbeatStaleTrialCallback() => StaleCallback;
    }

    [TestMethod]
    public void IsHeartbeatEnabled_Requires_Interface_And_Interval()
    {
        HeartbeatHelpers.IsHeartbeatEnabled(new InMemoryStorage()).Should().BeFalse();
        var hb = new HeartbeatStorage();
        HeartbeatHelpers.IsHeartbeatEnabled(hb).Should().BeTrue();
        hb.Interval = null;
        HeartbeatHelpers.IsHeartbeatEnabled(hb).Should().BeFalse();
    }

    [TestMethod]
    public void Optimize_Records_Heartbeats_While_Objective_Runs()
    {
        var storage = new HeartbeatStorage();
        var study = Optuna.CreateStudy(storage: storage, sampler: new RandomSampler(1));
        study.Optimize(t =>
        {
            Thread.Sleep(30);  // Give the heartbeat thread at least one beat.
            return t.SuggestFloat("x", 0, 1);
        }, nTrials: 2);

        storage.Recorded.Should().NotBeEmpty("the heartbeat thread must beat during each trial");
        storage.Recorded.Distinct().Should().BeEquivalentTo(
            study.Trials.Select(t => t.TrialId));
    }

    [TestMethod]
    public void FailStaleTrials_Fails_Running_Stale_Trials_And_Invokes_Callback()
    {
        var storage = new HeartbeatStorage();
        var study = Optuna.CreateStudy(storage: storage, sampler: new RandomSampler(1));
        int staleId = storage.CreateNewTrial(study.StudyId);
        int healthyId = storage.CreateNewTrial(study.StudyId);
        storage.StaleTrialIds.Add(staleId);

        var callbackTrials = new List<FrozenTrial>();
        storage.StaleCallback = (s, t) => callbackTrials.Add(t);

        HeartbeatHelpers.FailStaleTrials(study);

        storage.GetTrial(staleId).State.Should().Be(TrialState.Fail);
        storage.GetTrial(healthyId).State.Should().Be(TrialState.Running);
        callbackTrials.Should().ContainSingle(t => t.TrialId == staleId);

        // Already-finished stale ids are tolerated (UpdateFinishedTrialError swallowed).
        HeartbeatHelpers.FailStaleTrials(study);
        storage.GetTrial(staleId).State.Should().Be(TrialState.Fail);
    }

    [TestMethod]
    public void RetryCallback_Requeues_With_History_And_MaxRetry()
    {
        var storage = new HeartbeatStorage();
        var study = Optuna.CreateStudy(storage: storage, sampler: new RandomSampler(1));
        var callback = new RetryHeartbeatStaleTrialCallback(
            maxRetry: 2, inheritIntermediateValues: true);

        var failed = FrozenTrial.CreateTrial(
            state: TrialState.Fail,
            @params: new OrderedDict<string, object?> { ["x"] = 1.0 },
            distributions: new OrderedDict<string, OptunaSharp.Distributions.BaseDistribution>
            {
                ["x"] = new OptunaSharp.Distributions.FloatDistribution(0, 10),
            },
            intermediateValues: new OrderedDict<long, double> { [0] = 0.5 });
        // Materialize as a real failed trial in the study.
        study.AddTrial(failed);
        var storedFailed = study.Trials.Single();

        callback.Invoke(study, storedFailed);

        var requeued = study.GetTrials(deepcopy: false).Single(t => t.State == TrialState.Waiting);
        RetryHeartbeatStaleTrialCallback.RetriedTrialNumber(requeued).Should().Be(storedFailed.Number);
        RetryHeartbeatStaleTrialCallback.RetryHistory(requeued).Should().Equal(storedFailed.Number);
        requeued.Params["x"].Should().Be(1.0);
        requeued.IntermediateValues[0].Should().Be(0.5);

        // Chain a second failure of the retried trial: history grows; third exceeds maxRetry=2.
        requeued.State = TrialState.Fail;
        requeued.DatetimeComplete = DateTime.Now;
        requeued.SetDatetimeStart(DateTime.Now);
        callback.Invoke(study, requeued);
        var secondRetry = study.GetTrials(deepcopy: false).Last(t => t.State == TrialState.Waiting);
        RetryHeartbeatStaleTrialCallback.RetryHistory(secondRetry)
            .Should().Equal(storedFailed.Number, requeued.Number);
        RetryHeartbeatStaleTrialCallback.RetriedTrialNumber(secondRetry)
            .Should().Be(storedFailed.Number, "the original number survives chained retries");

        int before = study.GetTrials(deepcopy: false).Count;
        secondRetry.State = TrialState.Fail;
        secondRetry.DatetimeComplete = DateTime.Now;
        callback.Invoke(study, secondRetry);
        study.GetTrials(deepcopy: false).Should().HaveCount(before, "maxRetry=2 exhausted -> no new trial");
    }

    // ------------------------------------------------------------------ heartbeat threads

    /// <summary>An <see cref="IBaseHeartbeat" /> that only counts beats — for the thread-loop tests.</summary>
    private sealed class CountingHeartbeat : IBaseHeartbeat
    {
        private readonly int _intervalSeconds;
        private int _count;
        private readonly ManualResetEventSlim _twoBeats = new(false);

        public CountingHeartbeat(int intervalSeconds) => _intervalSeconds = intervalSeconds;

        public int Count => Volatile.Read(ref _count);

        public bool WaitForTwoBeats(TimeSpan timeout) => _twoBeats.Wait(timeout);

        public void RecordHeartbeat(int trialId)
        {
            if (Interlocked.Increment(ref _count) >= 2)
                _twoBeats.Set();
        }

        public List<int> GetStaleTrialIds(int studyId) => new();
        public int? GetHeartbeatInterval() => _intervalSeconds;
        public Action<Study, FrozenTrial>? GetHeartbeatStaleTrialCallback() => null;
    }

    [TestMethod]
    public void GetHeartbeatThread_Enabled_Returns_HeartbeatThread_Otherwise_Null()
    {
        HeartbeatHelpers.GetHeartbeatThread(1, new HeartbeatStorage())
            .Should().BeOfType<HeartbeatThread>();
        HeartbeatHelpers.GetHeartbeatThread(1, new InMemoryStorage())
            .Should().BeOfType<NullHeartbeatThread>("a non-heartbeat storage gets the no-op thread");
        HeartbeatHelpers.GetHeartbeatThread(1, new HeartbeatStorage { Interval = null })
            .Should().BeOfType<NullHeartbeatThread>("a heartbeat storage with no interval is disabled");
    }

    [TestMethod]
    public void NullHeartbeatThread_Start_Join_Dispose_AreNoops()
    {
        var thread = new NullHeartbeatThread();
        Action act = () =>
        {
            thread.Start();
            thread.Join();
            thread.Dispose();
        };
        act.Should().NotThrow();
    }

    [TestMethod]
    public void HeartbeatThread_Beats_Repeatedly_Then_Stops_On_Dispose()
    {
        var hb = new CountingHeartbeat(1);
        var thread = new HeartbeatThread(7, hb);
        thread.Start();

        // The loop beats immediately, then again after each interval — proving the while-loop repeats.
        hb.WaitForTwoBeats(TimeSpan.FromSeconds(5)).Should().BeTrue("the thread beats again after the interval");

        thread.Dispose();  // Python __exit__ -> Join: sets the stop event and joins the thread.
        int afterStop = hb.Count;

        Thread.Sleep(1300);  // longer than the 1s interval
        hb.Count.Should().Be(afterStop, "no beats occur after Join/Dispose");
    }

    // ------------------------------------------------------------------ heartbeat failures (review H2)

    /// <summary>
    ///     An <see cref="IBaseHeartbeat" /> whose every heartbeat write fails, the way a dropped database
    ///     connection or a busy SQLite file does. It counts the attempts and signals the first one.
    /// </summary>
    private sealed class FailingHeartbeat : IBaseHeartbeat
    {
        private int _attempts;
        private readonly ManualResetEventSlim _firstAttempt = new(false);

        /// <summary>How many heartbeat writes were attempted.</summary>
        public int Attempts => Volatile.Read(ref _attempts);

        /// <summary>Waits until the heartbeat thread has attempted its first write.</summary>
        /// <param name="timeout">How long to wait before giving up.</param>
        /// <returns>True once an attempt has been made within <paramref name="timeout" />.</returns>
        public bool WaitForFirstAttempt(TimeSpan timeout) => _firstAttempt.Wait(timeout);

        /// <summary>Counts the attempt, then fails it.</summary>
        /// <param name="trialId">Unused; the write fails regardless of the trial.</param>
        /// <exception cref="TimeoutException">Always, standing in for a transient storage failure.</exception>
        public void RecordHeartbeat(int trialId)
        {
            Interlocked.Increment(ref _attempts);
            _firstAttempt.Set();
            throw new TimeoutException("database is locked");
        }

        /// <summary>No trial is ever stale here; the tests only exercise the writing thread.</summary>
        /// <param name="studyId">Unused.</param>
        /// <returns>An empty list.</returns>
        public List<int> GetStaleTrialIds(int studyId) => new();

        /// <summary>A one-second interval, so heartbeats are enabled and the thread is created.</summary>
        /// <returns>Always 1.</returns>
        public int? GetHeartbeatInterval() => 1;

        /// <summary>No stale-trial callback is configured.</summary>
        /// <returns>Always null.</returns>
        public Action<Study, FrozenTrial>? GetHeartbeatStaleTrialCallback() => null;
    }

    /// <summary>An <see cref="InMemoryStorage" /> with heartbeats enabled whose every heartbeat write fails.</summary>
    private sealed class FailingHeartbeatStorage : InMemoryStorage, IBaseHeartbeat
    {
        /// <summary>Fails every heartbeat write, standing in for a storage that lost its connection.</summary>
        /// <param name="trialId">Unused; the write fails regardless of the trial.</param>
        /// <exception cref="TimeoutException">Always.</exception>
        public void RecordHeartbeat(int trialId) => throw new TimeoutException("database is locked");

        /// <summary>No trial is ever stale, so the optimize loop's stale-trial sweep fails nothing.</summary>
        /// <param name="studyId">Unused.</param>
        /// <returns>An empty list.</returns>
        public List<int> GetStaleTrialIds(int studyId) => new();

        /// <summary>A one-second interval, so every trial starts a real heartbeat thread.</summary>
        /// <returns>Always 1.</returns>
        public int? GetHeartbeatInterval() => 1;

        /// <summary>No stale-trial callback is configured.</summary>
        /// <returns>Always null.</returns>
        public Action<Study, FrozenTrial>? GetHeartbeatStaleTrialCallback() => null;
    }

    /// <summary>
    ///     Review item H2: an exception from <see cref="IBaseHeartbeat.RecordHeartbeat" /> used to escape the
    ///     heartbeat thread, and .NET terminates the whole process on an unhandled thread exception. Python
    ///     only loses the heartbeat thread (<c>threading.excepthook</c> prints the error to stderr). The
    ///     thread must now end after reporting the failure in that layout, and <see cref="HeartbeatThread.Join" />
    ///     must return normally.
    /// </summary>
    [TestMethod]
    public void HeartbeatThread_FailedWrite_EndsOnlyTheThread_AndReportsLikePythonExcepthook()
    {
        var hb = new FailingHeartbeat();
        var thread = new HeartbeatThread(7, hb);

        string report = CaptureStderr(() =>
        {
            thread.Start();
            hb.WaitForFirstAttempt(TimeSpan.FromSeconds(5)).Should().BeTrue("the thread beats immediately");
            // Join returns once the thread has exited, so the report is complete before stderr is restored.
            Action join = thread.Join;
            join.Should().NotThrow();
        });

        hb.Attempts.Should().Be(1, "like Python, the heartbeat thread stops at its first failed write");
        report.Should().StartWith("Exception in thread OptunaSharp heartbeat (trial_id=7):\n");
        report.Should().Contain(nameof(TimeoutException)).And.Contain("database is locked");
    }

    /// <summary>
    ///     End to end: a storage whose heartbeat writes all fail no longer kills the process. Each trial keeps
    ///     running without a heartbeat and completes, exactly as under Python Optuna.
    /// </summary>
    [TestMethod]
    public void Optimize_CompletesTrials_WhenEveryHeartbeatWriteFails()
    {
        var storage = new FailingHeartbeatStorage();
        var study = Optuna.CreateStudy(storage: storage, sampler: new RandomSampler(1));

        string report = CaptureStderr(() => study.Optimize(t => t.SuggestFloat("x", 0, 1), nTrials: 3));

        study.Trials.Should().HaveCount(3).And.OnlyContain(t => t.State == TrialState.Complete);
        report.Should().Contain("Exception in thread OptunaSharp heartbeat (trial_id=")
            .And.Contain("database is locked");
    }

    /// <summary>
    ///     Runs <paramref name="action" /> with <see cref="Console.Error" /> redirected and returns what it wrote.
    ///     The library logger is configured first, so the redirect never becomes its default stream (the
    ///     project runs tests sequentially, which makes the process-wide redirect safe).
    /// </summary>
    /// <param name="action">The code whose standard-error output is captured.</param>
    /// <returns>Everything written to standard error while <paramref name="action" /> ran.</returns>
    private static string CaptureStderr(Action action)
    {
        _ = OptunaSharp.Logging.OptunaLogging.GetVerbosity();
        System.IO.TextWriter original = Console.Error;
        using var buffer = new System.IO.StringWriter();
        try
        {
            Console.SetError(buffer);
            action();
        }
        finally
        {
            Console.SetError(original);
        }
        return buffer.ToString();
    }

    // ------------------------------------------------------------------ fail_stale_trials edge cases

    [TestMethod]
    public void FailStaleTrials_Noop_For_NonHeartbeat_Storage()
    {
        var storage = new InMemoryStorage();
        var study = Optuna.CreateStudy(storage: storage, sampler: new RandomSampler(1));
        int trialId = storage.CreateNewTrial(study.StudyId);

        Action act = () => HeartbeatHelpers.FailStaleTrials(study);  // storage is not IBaseHeartbeat
        act.Should().NotThrow();
        storage.GetTrial(trialId).State.Should().Be(TrialState.Running);
    }

    [TestMethod]
    public void FailStaleTrials_Noop_When_Heartbeat_Disabled()
    {
        var storage = new HeartbeatStorage { Interval = null };  // heartbeat disabled
        var study = Optuna.CreateStudy(storage: storage, sampler: new RandomSampler(1));
        int staleId = storage.CreateNewTrial(study.StudyId);
        storage.StaleTrialIds.Add(staleId);

        HeartbeatHelpers.FailStaleTrials(study);

        storage.GetTrial(staleId).State.Should().Be(TrialState.Running, "a disabled heartbeat fails nothing");
    }

    [TestMethod]
    public void FailStaleTrials_Fails_Every_Stale_Trial_With_Null_Callback()
    {
        var storage = new HeartbeatStorage();  // Interval = 1, StaleCallback = null
        var study = Optuna.CreateStudy(storage: storage, sampler: new RandomSampler(1));
        int s1 = storage.CreateNewTrial(study.StudyId);
        int s2 = storage.CreateNewTrial(study.StudyId);
        int healthy = storage.CreateNewTrial(study.StudyId);
        storage.StaleTrialIds.Add(s1);
        storage.StaleTrialIds.Add(s2);

        HeartbeatHelpers.FailStaleTrials(study);  // callback is null -> trials fail, no callback

        storage.GetTrial(s1).State.Should().Be(TrialState.Fail);
        storage.GetTrial(s2).State.Should().Be(TrialState.Fail);
        storage.GetTrial(healthy).State.Should().Be(TrialState.Running);
    }

    // ------------------------------------------------------------------ RetryHeartbeatStaleTrialCallback surface

    [TestMethod]
    public void RetriedTrialNumber_And_RetryHistory_Are_Null_And_Empty_For_A_Fresh_Trial()
    {
        var fresh = FrozenTrial.CreateTrial(state: TrialState.Complete, value: 1.0);
        RetryHeartbeatStaleTrialCallback.RetriedTrialNumber(fresh).Should().BeNull();
        RetryHeartbeatStaleTrialCallback.RetryHistory(fresh).Should().BeEmpty();
    }

    [TestMethod]
    public void RetryCallback_Does_Not_Inherit_IntermediateValues_By_Default()
    {
        var storage = new HeartbeatStorage();
        var study = Optuna.CreateStudy(storage: storage, sampler: new RandomSampler(1));
        var callback = new RetryHeartbeatStaleTrialCallback();  // inheritIntermediateValues defaults to false

        study.AddTrial(FrozenTrial.CreateTrial(
            state: TrialState.Fail,
            @params: new OrderedDict<string, object?> { ["x"] = 1.0 },
            distributions: new OrderedDict<string, BaseDistribution> { ["x"] = new FloatDistribution(0, 10) },
            intermediateValues: new OrderedDict<long, double> { [0] = 0.5 }));
        var stored = study.Trials.Single();

        callback.Invoke(study, stored);

        var requeued = study.GetTrials(deepcopy: false).Single(t => t.State == TrialState.Waiting);
        requeued.IntermediateValues.Should().BeEmpty("intermediate values are not inherited unless requested");
        requeued.Params["x"].Should().Be(1.0, "params are still inherited");
    }

    [TestMethod]
    public void RetryCallback_MaxRetryNull_Retries_Indefinitely()
    {
        var storage = new HeartbeatStorage();
        var study = Optuna.CreateStudy(storage: storage, sampler: new RandomSampler(1));
        var callback = new RetryHeartbeatStaleTrialCallback();  // maxRetry null -> no ceiling

        study.AddTrial(FrozenTrial.CreateTrial(
            state: TrialState.Fail,
            @params: new OrderedDict<string, object?> { ["x"] = 1.0 },
            distributions: new OrderedDict<string, BaseDistribution> { ["x"] = new FloatDistribution(0, 10) }));

        // Chain five failures; every one must be requeued (the retry_history keeps growing).
        for (int i = 0; i < 5; i++)
        {
            var failed = study.GetTrials(deepcopy: false).Last();
            failed.State = TrialState.Fail;
            failed.DatetimeComplete = DateTime.Now;
            callback.Invoke(study, failed);
            study.GetTrials(deepcopy: false).Count(t => t.State == TrialState.Waiting)
                .Should().Be(1, "each stale trial is requeued regardless of history length");
            // Promote the just-created WAITING retry to the tail for the next round.
            var requeued = study.GetTrials(deepcopy: false).Single(t => t.State == TrialState.Waiting);
            RetryHeartbeatStaleTrialCallback.RetryHistory(requeued).Should().HaveCount(i + 1);
            requeued.State = TrialState.Fail;
            requeued.DatetimeComplete = DateTime.Now;
        }
    }

    [TestMethod]
    public void RetryCallback_Inherits_Distributions_UserAttrs_And_Existing_SystemAttrs()
    {
        var storage = new HeartbeatStorage();
        var study = Optuna.CreateStudy(storage: storage, sampler: new RandomSampler(1));
        var dist = new FloatDistribution(0, 10);

        study.AddTrial(FrozenTrial.CreateTrial(
            state: TrialState.Fail,
            @params: new OrderedDict<string, object?> { ["x"] = 1.0 },
            distributions: new OrderedDict<string, BaseDistribution> { ["x"] = dist },
            userAttrs: new OrderedDict<string, object?> { ["owner"] = "eli" },
            systemAttrs: new OrderedDict<string, object?> { ["custom"] = 42L }));
        var stored = study.Trials.Single();

        new RetryHeartbeatStaleTrialCallback().Invoke(study, stored);

        var requeued = study.GetTrials(deepcopy: false).Single(t => t.State == TrialState.Waiting);
        requeued.Distributions["x"].Should().Be(dist);
        requeued.UserAttrs["owner"].Should().Be("eli");
        requeued.SystemAttrs["custom"].Should().Be(42L, "unrelated system attrs survive the requeue");
        RetryHeartbeatStaleTrialCallback.RetriedTrialNumber(requeued).Should().Be(stored.Number);
        RetryHeartbeatStaleTrialCallback.RetryHistory(requeued).Should().Equal(stored.Number);
    }

    [TestMethod]
    public void RetryFailedTrialCallback_DeprecatedAlias_Requeues_Like_The_New_Name()
    {
        var storage = new HeartbeatStorage();
        var study = Optuna.CreateStudy(storage: storage, sampler: new RandomSampler(1));
#pragma warning disable CS0618 // deliberately exercising the deprecated alias
        var callback = new RetryFailedTrialCallback(maxRetry: 2, inheritIntermediateValues: true);
#pragma warning restore CS0618

        study.AddTrial(FrozenTrial.CreateTrial(
            state: TrialState.Fail,
            @params: new OrderedDict<string, object?> { ["x"] = 1.0 },
            distributions: new OrderedDict<string, BaseDistribution> { ["x"] = new FloatDistribution(0, 10) },
            intermediateValues: new OrderedDict<long, double> { [0] = 0.5 }));
        var stored = study.Trials.Single();

        callback.Invoke(study, stored);

        var requeued = study.GetTrials(deepcopy: false).Single(t => t.State == TrialState.Waiting);
        requeued.Params["x"].Should().Be(1.0);
        requeued.IntermediateValues[0].Should().Be(0.5, "the deprecated alias honors inheritIntermediateValues");
        RetryHeartbeatStaleTrialCallback.RetriedTrialNumber(requeued).Should().Be(stored.Number);
    }

    // ------------------------------------------------------------------ Study.Ask heartbeat warning

    [TestMethod]
    public void Ask_OutsideOptimize_With_Heartbeat_Warns()
    {
        var storage = new HeartbeatStorage();
        var study = Optuna.CreateStudy(storage: storage, sampler: new RandomSampler(1));

        var warnings = CaptureWarnings(() => study.Ask());

        warnings.Should().Contain(w => w.Message.Contains("Heartbeat of storage is supposed to be used with Study.optimize"));
    }

    [TestMethod]
    public void Ask_OutsideOptimize_Without_Heartbeat_DoesNotWarn()
    {
        var storage = new HeartbeatStorage { Interval = null };  // heartbeat disabled
        var study = Optuna.CreateStudy(storage: storage, sampler: new RandomSampler(1));

        var warnings = CaptureWarnings(() => study.Ask());

        warnings.Should().NotContain(w => w.Message.Contains("supposed to be used with Study.optimize"));
    }

    [TestMethod]
    public void Optimize_With_Heartbeat_DoesNotWarn_AboutAskUsage()
    {
        var storage = new HeartbeatStorage();
        var study = Optuna.CreateStudy(storage: storage, sampler: new RandomSampler(1));

        var warnings = CaptureWarnings(() =>
            study.Optimize(t => t.SuggestFloat("x", 0, 1), nTrials: 2));

        warnings.Should().NotContain(w => w.Message.Contains("supposed to be used with Study.optimize"),
            "inside the optimize loop the ask-usage warning must not fire");
    }

    // ------------------------------------------------------------------ helpers

    /// <summary>Capture every <see cref="OptunaWarnings" /> raised while <paramref name="action" /> runs.</summary>
    private static List<(string Message, WarningCategory Category)> CaptureWarnings(Action action)
    {
        var captured = new List<(string, WarningCategory)>();
        void Handler(string message, WarningCategory category)
        {
            lock (captured)
                captured.Add((message, category));
        }

        OptunaWarnings.WarningRaised += Handler;
        try
        {
            action();
        }
        finally
        {
            OptunaWarnings.WarningRaised -= Handler;
        }
        return captured;
    }
}
