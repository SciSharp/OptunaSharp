using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using AwesomeAssertions;
using OptunaSharp.Collections;
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
            intermediateValues: new OrderedDict<int, double> { [0] = 0.5 });
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
}
