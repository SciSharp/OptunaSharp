using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using AwesomeAssertions;
using NumSharp;
using OptunaSharp.Distributions;
using OptunaSharp.Internal;
using OptunaSharp.Samplers;
using OptunaSharp.Storages;

namespace OptunaSharp.Tests;

/// <summary>
///     The edges of the optimize loop with <c>nJobs</c>: how a run ends (timeout, <c>Stop</c>,
///     <see cref="MaxTrialsCallback" />), which failures are raised and which only fail a trial (<c>catch</c>, callbacks,
///     NaN and wrong-arity results, sampler hooks, a failed storage write), and the worker-count shapes (-1, more
///     workers than trials, no trials). Every expectation was first observed on Optuna 5.0 with the same scenario; the
///     observed numbers are quoted per test. Counts that depend on thread timing are asserted as the bounds the
///     scheduling allows, never as one run's exact count.
/// </summary>
[TestClass]
public class OptimizeEdgeCaseTests
{
    /// <summary>
    ///     A timeout ends an nJobs run within one trial of the limit, and every trial the run started has finished
    ///     (Optuna 5.0: <c>timeout=1.0</c>, 0.2 s trials, <c>n_jobs=4</c>: 1.01 s, 20 trials, all COMPLETE).
    /// </summary>
    [TestMethod]
    public void NJobs_Timeout_EndsTheRunWithinOneTrialOfTheLimit()
    {
        var study = Optuna.CreateStudy(sampler: new RandomSampler(20));
        var watch = Stopwatch.StartNew();

        study.Optimize(t =>
        {
            Thread.Sleep(100);
            return t.SuggestFloat("x", 0, 1);
        }, timeout: 1.0, nJobs: 4);

        watch.Elapsed.TotalSeconds.Should().BeLessThan(2.5, "the run ends one 0.1 s trial after the 1 s limit, plus scheduling");
        study.Trials.Should().HaveCountGreaterThanOrEqualTo(4, "the first batch starts long before the limit");
        ShouldBeFinishedAndContiguous(study);
    }

    /// <summary>
    ///     A timeout shorter than one trial runs exactly the first batch: the worker the loop starts after its first
    ///     wait sees the limit already passed and runs nothing (Optuna 5.0: <c>timeout=0.3</c>, 1 s trials,
    ///     <c>n_jobs=4</c>: 4 trials in 1.0 s).
    /// </summary>
    [TestMethod]
    public void NJobs_TimeoutShorterThanOneTrial_RunsOnlyTheFirstBatch()
    {
        var study = Optuna.CreateStudy(sampler: new RandomSampler(21));
        var watch = Stopwatch.StartNew();

        study.Optimize(t =>
        {
            Thread.Sleep(1000);
            return t.SuggestFloat("x", 0, 1);
        }, timeout: 0.5, nJobs: 4);

        study.Trials.Should().HaveCount(4, "only the four workers started before the limit run a trial");
        watch.Elapsed.TotalSeconds.Should().BeLessThan(2.5, "one trial's second, not a second batch");
        ShouldBeFinishedAndContiguous(study);
    }

    /// <summary>
    ///     <c>Stop()</c> from an objective ends scheduling: the trials already in flight finish, and no worker starts
    ///     a trial afterwards (Optuna 5.0: stop in trial 10 at <c>n_jobs=4</c>: 14 trials, all COMPLETE). At most
    ///     three other workers can hold a trial when trial 10 stops, so no trial numbered above 13 may exist.
    /// </summary>
    /// <remarks>
    ///     Trials numbered above 10 wait until trial 10 has called <c>Stop()</c>. Without that gate the bound was a
    ///     race: on an oversubscribed runner (macOS, 3 cores) the worker of trial 10 was preempted during its sleep,
    ///     the other three finished trials 11–13 and started 14–15 before the stop existed, and the test saw 16
    ///     trials. With it, no trial above 10 can finish before the stop, so each other worker holds at most one
    ///     trial, and 14 is an exact upper bound on any host.
    /// </remarks>
    [TestMethod]
    public void NJobs_StopFromTheObjective_LetsTheTrialsInFlightFinish_AndStartsNoMore()
    {
        var study = Optuna.CreateStudy(sampler: new RandomSampler(22));
        using var stopped = new ManualResetEventSlim();

        study.Optimize(t =>
        {
            double x = t.SuggestFloat("x", 0, 1);
            Thread.Sleep(20);
            if (t.Number == 10)
            {
                t.Study.Stop();
                stopped.Set();
            }
            else if (t.Number > 10)
            {
                // Bounded so a regression that never runs trial 10 fails the assertions instead of hanging.
                stopped.Wait(TimeSpan.FromSeconds(30));
            }
            return x;
        }, nTrials: 200, nJobs: 4);

        study.Trials.Should().HaveCountGreaterThanOrEqualTo(11).And.HaveCountLessThanOrEqualTo(14);
        ShouldBeFinishedAndContiguous(study);
    }

    /// <summary>
    ///     The same from a callback, which runs on the worker's thread after its trial is stored (Optuna 5.0: stop in
    ///     the callback of trial 10 at <c>n_jobs=4</c>: 14 trials).
    /// </summary>
    /// <remarks>
    ///     Trials numbered above 10 wait until trial 10's callback has called <c>Stop()</c>, for the reason given on
    ///     <see cref="NJobs_StopFromTheObjective_LetsTheTrialsInFlightFinish_AndStartsNoMore" />: it turns the
    ///     timing-dependent bound of 14 into an exact one.
    /// </remarks>
    [TestMethod]
    public void NJobs_StopFromACallback_LetsTheTrialsInFlightFinish_AndStartsNoMore()
    {
        var study = Optuna.CreateStudy(sampler: new RandomSampler(23));
        using var stopped = new ManualResetEventSlim();

        study.Optimize(t =>
        {
            Thread.Sleep(20);
            if (t.Number > 10)
                stopped.Wait(TimeSpan.FromSeconds(30));   // bounded: a regression fails instead of hanging
            return t.SuggestFloat("x", 0, 1);
        }, nTrials: 200, nJobs: 4, callbacks: new Action<Study, FrozenTrial>[]
        {
            (s, frozen) =>
            {
                if (frozen.Number == 10)
                {
                    s.Stop();
                    stopped.Set();
                }
            },
        });

        study.Trials.Should().HaveCountGreaterThanOrEqualTo(11).And.HaveCountLessThanOrEqualTo(14);
        ShouldBeFinishedAndContiguous(study);
    }

    /// <summary>
    ///     <see cref="MaxTrialsCallback" /> under nJobs overshoots by at most the trials still in flight when the limit is
    ///     reached, as in Python (Optuna 5.0: limit 20 COMPLETE at <c>n_jobs=4</c>: 22 trials).
    /// </summary>
    [TestMethod]
    public void NJobs_MaxTrialsCallback_StopsWithinOneBatchOfTheLimit()
    {
        var study = Optuna.CreateStudy(sampler: new RandomSampler(24));

        study.Optimize(t =>
        {
            Thread.Sleep(20);
            return t.SuggestFloat("x", 0, 1);
        }, nTrials: 200, nJobs: 4, callbacks: new[] { (Action<Study, FrozenTrial>)new MaxTrialsCallback(20) });

        study.Trials.Should().HaveCountGreaterThanOrEqualTo(20).And.HaveCountLessThanOrEqualTo(23);
        ShouldBeFinishedAndContiguous(study);
    }

    /// <summary>
    ///     Caught exception types fail their trial and the run continues to the full count, under nJobs too; a
    ///     derived type is caught by its base, as Python's <c>isinstance</c> check catches subclasses (Optuna 5.0:
    ///     <c>ValueError</c> on odd trials, <c>catch=(ValueError,)</c>, <c>n_jobs=4</c>: 40 trials, 20 FAIL).
    /// </summary>
    /// <exception cref="ValueError">Thrown by odd-numbered trials; caught, so each such trial is recorded as FAIL.</exception>
    /// <exception cref="ArgumentOutOfRangeException">Thrown by trials numbered 2 mod 4; caught through its base type.</exception>
    [TestMethod]
    public void NJobs_CaughtExceptions_FailTheirTrials_AndTheRunContinues()
    {
        var study = Optuna.CreateStudy(sampler: new RandomSampler(25));

        study.Optimize(t =>
        {
            double x = t.SuggestFloat("x", 0, 1);
            if (t.Number % 2 == 1)
                throw new ValueError("odd");
            if (t.Number % 4 == 2)
                throw new ArgumentOutOfRangeException(nameof(t), "derived");
            return x;
        }, nTrials: 40, nJobs: 4, @catch: new[] { typeof(ValueError), typeof(ArgumentException) });

        var trials = study.Trials;
        trials.Should().HaveCount(40);
        trials.Where(t => t.State == TrialState.Fail).Select(t => t.Number)
            .Should().Equal(Enumerable.Range(0, 40).Where(n => n % 2 == 1 || n % 4 == 2));
        ShouldBeFinishedAndContiguous(study);
    }

    /// <summary>
    ///     An exception type outside <c>catch</c> is raised even when others are being caught, after the trials in
    ///     flight finish (Optuna 5.0: <c>KeyError</c> in trial 5, <c>ValueError</c> caught on odd trials,
    ///     <c>n_jobs=4</c>: the <c>KeyError</c> is raised, 9 trials, none RUNNING).
    /// </summary>
    /// <exception cref="InvalidOperationException">Thrown by trial 5; not caught, so Optimize rethrows it.</exception>
    /// <exception cref="ValueError">Thrown by the other odd-numbered trials; caught.</exception>
    [TestMethod]
    public void NJobs_AnUncaughtExceptionAmongCaughtOnes_IsRaised()
    {
        var study = Optuna.CreateStudy(sampler: new RandomSampler(26));

        var act = () => study.Optimize(t =>
        {
            double x = t.SuggestFloat("x", 0, 1);
            Thread.Sleep(50);
            if (t.Number == 5)
                throw new InvalidOperationException("five");
            if (t.Number % 2 == 1)
                throw new ValueError("odd");
            return x;
        }, nTrials: 40, nJobs: 4, @catch: new[] { typeof(ValueError) });

        act.Should().Throw<InvalidOperationException>().WithMessage("five");
        study.Trials.Should().HaveCountGreaterThanOrEqualTo(6);
        study.Trials.Single(t => t.Number == 5).State.Should().Be(TrialState.Fail);
        ShouldBeFinishedAndContiguous(study);
    }

    /// <summary>
    ///     A callback's exception is raised whatever <c>catch</c> holds, and the trial it ran for stays COMPLETE: the
    ///     callback runs after the trial is stored (Optuna 5.0: callback raising for trial 5: raised, trial 5 COMPLETE;
    ///     6 trials sequentially, 9 at <c>n_jobs=4</c>).
    /// </summary>
    /// <param name="nJobs">The worker count; 1 is the sequential path, whose trial count is exact.</param>
    /// <exception cref="InvalidOperationException">Thrown by the callback for trial 5; Optimize rethrows it.</exception>
    [TestMethod]
    [DataRow(1)]
    [DataRow(4)]
    public void CallbackError_IsRaised_AndItsTrialStaysComplete(int nJobs)
    {
        var study = Optuna.CreateStudy(sampler: new RandomSampler(27));

        var act = () => study.Optimize(t =>
        {
            Thread.Sleep(nJobs == 1 ? 0 : 50);
            return t.SuggestFloat("x", 0, 1);
        }, nTrials: 40, nJobs: nJobs, @catch: new[] { typeof(Exception) }, callbacks: new Action<Study, FrozenTrial>[]
        {
            (_, frozen) =>
            {
                if (frozen.Number == 5)
                    throw new InvalidOperationException("callback");
            },
        });

        act.Should().Throw<InvalidOperationException>().WithMessage("callback", "catch applies to the objective only");
        study.Trials.Single(t => t.Number == 5).State.Should().Be(TrialState.Complete);
        if (nJobs == 1)
            study.Trials.Should().HaveCount(6);
        else
            study.Trials.Should().HaveCountGreaterThanOrEqualTo(6);
        ShouldBeFinishedAndContiguous(study);
    }

    /// <summary>
    ///     Results the study cannot accept fail their trial without raising, under nJobs as sequentially: NaN, a
    ///     multi-objective study given one value, and no value at all; infinities are acceptable values (Optuna 5.0:
    ///     NaN on even trials: 10 FAIL of 20; one value for a two-objective study on even trials: 10 FAIL of 20;
    ///     <c>None</c>: 8 FAIL of 8; ±inf: 8 COMPLETE, best value -inf).
    /// </summary>
    [TestMethod]
    public void NJobs_UnacceptableResults_FailTheirTrial_WithoutRaising()
    {
        var nan = Optuna.CreateStudy(sampler: new RandomSampler(28));
        nan.Optimize(t => t.Number % 2 == 0 ? double.NaN : t.SuggestFloat("x", 0, 1), nTrials: 20, nJobs: 4);
        nan.Trials.Where(t => t.State == TrialState.Fail).Select(t => t.Number)
            .Should().Equal(Enumerable.Range(0, 20).Where(n => n % 2 == 0));

        var arity = Optuna.CreateStudy(sampler: new RandomSampler(28),
            directions: new[] { StudyDirection.Minimize, StudyDirection.Minimize });
        arity.Optimize(t => t.Number % 2 == 0 ? new[] { t.SuggestFloat("x", 0, 1) } : new[] { 1.0, 2.0 },
            nTrials: 20, nJobs: 4);
        arity.Trials.Where(t => t.State == TrialState.Fail).Select(t => t.Number)
            .Should().Equal(Enumerable.Range(0, 20).Where(n => n % 2 == 0));

        var none = Optuna.CreateStudy(sampler: new RandomSampler(28));
        none.Optimize((Func<Trial, IReadOnlyList<double>>)(_ => null!), nTrials: 8, nJobs: 4);
        none.Trials.Should().HaveCount(8).And.OnlyContain(t => t.State == TrialState.Fail);

        var infinite = Optuna.CreateStudy(sampler: new RandomSampler(28));
        infinite.Optimize(t => t.Number % 2 == 1 ? double.PositiveInfinity : double.NegativeInfinity, nTrials: 8, nJobs: 4);
        infinite.Trials.Should().HaveCount(8).And.OnlyContain(t => t.State == TrialState.Complete);
        infinite.BestValue.Should().Be(double.NegativeInfinity);

        foreach (var study in new[] { nan, arity, none, infinite })
            ShouldBeFinishedAndContiguous(study);
    }

    /// <summary>
    ///     <c>TrialPruned</c> under nJobs records the trial as PRUNED and is never raised (Optuna 5.0: every third
    ///     trial pruned at <c>n_jobs=4</c>: 10 PRUNED and 20 COMPLETE of 30).
    /// </summary>
    /// <exception cref="TrialPruned">Thrown by every third trial; recorded as PRUNED.</exception>
    [TestMethod]
    public void NJobs_PrunedTrials_AreRecordedAsPruned()
    {
        var study = Optuna.CreateStudy(sampler: new RandomSampler(29));

        study.Optimize(t =>
        {
            double x = t.SuggestFloat("x", 0, 1);
            if (t.Number % 3 == 0)
                throw new TrialPruned();
            return x;
        }, nTrials: 30, nJobs: 4);

        study.Trials.Where(t => t.State == TrialState.Pruned).Select(t => t.Number)
            .Should().Equal(Enumerable.Range(0, 30).Where(n => n % 3 == 0));
        ShouldBeFinishedAndContiguous(study);
    }

    /// <summary>
    ///     <c>nJobs: -1</c> runs one worker per processor, as Python's <c>os.cpu_count()</c> (Optuna 5.0: 50 trials,
    ///     all COMPLETE, numbered contiguously). No more trials may ever run at once than there are processors.
    /// </summary>
    [TestMethod]
    public void NJobs_MinusOne_RunsAtMostOneTrialPerProcessor()
    {
        var study = Optuna.CreateStudy(sampler: new RandomSampler(30));
        int running = 0, peak = 0;

        study.Optimize(t =>
        {
            int now = Interlocked.Increment(ref running);
            InterlockedMax(ref peak, now);
            Thread.Sleep(20);
            Interlocked.Decrement(ref running);
            return t.SuggestFloat("x", 0, 1);
        }, nTrials: 50, nJobs: -1);

        study.Trials.Should().HaveCount(50);
        peak.Should().BeLessThanOrEqualTo(Environment.ProcessorCount);
        if (Environment.ProcessorCount > 1)
            peak.Should().BeGreaterThan(1, "-1 means one worker per processor, not a sequential run");
        ShouldBeFinishedAndContiguous(study);
    }

    /// <summary>
    ///     More workers than trials, and no trials at all: the trial count rules (Optuna 5.0: <c>n_jobs=16</c>,
    ///     <c>n_trials=3</c>: 3 trials; <c>n_trials=0</c>: none).
    /// </summary>
    [TestMethod]
    public void NJobs_AboveNTrials_RunsExactlyNTrials_AndZeroTrialsRunsNone()
    {
        var study = Optuna.CreateStudy(sampler: new RandomSampler(31));
        int running = 0, peak = 0;

        study.Optimize(t =>
        {
            InterlockedMax(ref peak, Interlocked.Increment(ref running));
            Thread.Sleep(50);
            Interlocked.Decrement(ref running);
            return t.SuggestFloat("x", 0, 1);
        }, nTrials: 3, nJobs: 16);

        study.Trials.Should().HaveCount(3);
        peak.Should().BeLessThanOrEqualTo(3);
        ShouldBeFinishedAndContiguous(study);

        var empty = Optuna.CreateStudy(sampler: new RandomSampler(31));
        empty.Optimize(t => t.SuggestFloat("x", 0, 1), nTrials: 0, nJobs: 4);
        empty.Trials.Should().BeEmpty();
    }

    /// <summary>
    ///     A failing <c>BeforeTrial</c> is raised whatever <c>catch</c> holds, and leaves its trial RUNNING: it fails
    ///     inside <c>Ask</c>, after the trial is created and before the objective's <c>try</c> (Optuna 5.0: before_trial
    ///     raising for trial 3, <c>catch=(Exception,)</c>: raised, 4 trials, trial 3 RUNNING). Python leaves the trial
    ///     behind; so does OptunaSharp, and a heartbeat-enabled storage is what reclaims such a trial.
    /// </summary>
    /// <exception cref="InvalidOperationException">Thrown by the sampler for trial 3; Optimize rethrows it.</exception>
    [TestMethod]
    public void SamplerBeforeTrialError_IsRaisedEvenWhenCaught_AndLeavesItsTrialRunning()
    {
        var study = Optuna.CreateStudy(sampler: new HookFailingSampler(32, beforeTrialFailsAt: 3));

        var act = () => study.Optimize(t => t.SuggestFloat("x", 0, 1), nTrials: 10, @catch: new[] { typeof(Exception) });

        act.Should().Throw<InvalidOperationException>().WithMessage("before_trial");
        var trials = study.Trials;
        trials.Should().HaveCount(4);
        trials[3].State.Should().Be(TrialState.Running, "Python leaves the trial its sampler failed to start");
        trials.Take(3).Should().OnlyContain(t => t.State == TrialState.Complete);
    }

    /// <summary>
    ///     A failing <c>AfterTrial</c> is raised whatever <c>catch</c> holds, but its trial is still stored COMPLETE:
    ///     the state is written in a <c>finally</c> after the hook (Optuna 5.0: after_trial raising for trial 3,
    ///     <c>catch=(Exception,)</c>: raised, 4 trials, trial 3 COMPLETE).
    /// </summary>
    /// <exception cref="InvalidOperationException">Thrown by the sampler for trial 3; Optimize rethrows it.</exception>
    [TestMethod]
    public void SamplerAfterTrialError_IsRaisedEvenWhenCaught_ButTheTrialIsStored()
    {
        var study = Optuna.CreateStudy(sampler: new HookFailingSampler(33, afterTrialFailsAt: 3));

        var act = () => study.Optimize(t => t.SuggestFloat("x", 0, 1), nTrials: 10, @catch: new[] { typeof(Exception) });

        act.Should().Throw<InvalidOperationException>().WithMessage("after_trial");
        var trials = study.Trials;
        trials.Should().HaveCount(4);
        trials.Should().OnlyContain(t => t.State == TrialState.Complete);
    }

    /// <summary>
    ///     A sampler failing while it samples is an objective error: <c>catch</c> applies, the trial fails, and the
    ///     other workers keep sampling, so the failure released the sampler's monitor (Optuna 5.0: sample_independent
    ///     raising for trial 3, <c>catch=(RuntimeError,)</c>, <c>n_jobs=4</c>: 10 trials, trial 3 FAIL). A monitor left
    ///     held would deadlock the run; the watchdog turns that into a failure.
    /// </summary>
    [TestMethod]
    public void NJobs_SamplerErrorWhileSampling_FailsOnlyItsTrial_AndReleasesTheSampler()
    {
        var study = Optuna.CreateStudy(sampler: new HookFailingSampler(34, sampleFailsAt: 3));

        RunWithin(60, () => study.Optimize(t =>
        {
            double x = t.SuggestFloat("x", 0, 1);
            Thread.Sleep(10);
            return x;
        }, nTrials: 10, nJobs: 4, @catch: new[] { typeof(InvalidOperationException) }));

        var trials = study.Trials;
        trials.Should().HaveCount(10);
        trials.Where(t => t.State == TrialState.Fail).Select(t => t.Number).Should().Equal(3);
        ShouldBeFinishedAndContiguous(study);
    }

    /// <summary>
    ///     <c>Optimize</c> called from inside an nJobs worker's objective is a nested invocation: each such trial fails
    ///     and the first failure is raised (Optuna 5.0: <c>n_jobs=2</c>, 4 trials requested: raised, 2 trials, both
    ///     FAIL). The worker's own loop flag, not the caller's, is what refuses the nested call.
    /// </summary>
    /// <exception cref="RuntimeError">Thrown by the nested call inside each objective; Optimize rethrows the first.</exception>
    [TestMethod]
    public void NJobs_NestedOptimizeInAWorker_IsRejected()
    {
        var study = Optuna.CreateStudy(sampler: new RandomSampler(35));

        var act = () => study.Optimize(t =>
        {
            study.Optimize(_ => 0.0, nTrials: 1);
            return 0.0;
        }, nTrials: 4, nJobs: 2);

        act.Should().Throw<RuntimeError>().WithMessage("Nested invocation of `Study.optimize` method isn't allowed.");
        study.Trials.Should().HaveCount(2).And.OnlyContain(t => t.State == TrialState.Fail);
    }

    /// <summary>
    ///     <c>Stop()</c> works only from an objective or callback: called from any other thread while a run is going
    ///     on, it raises and the run continues, as in Python (Optuna 5.0: a second thread calling <c>study.stop()</c>
    ///     during <c>optimize(timeout=1.0, n_jobs=2)</c> got the RuntimeError). An application that wants to cancel
    ///     from outside sets its own flag and calls <c>Stop()</c> from a callback.
    /// </summary>
    [TestMethod]
    public void Stop_FromAThreadOutsideTheRun_Raises_AndTheRunContinues()
    {
        var study = Optuna.CreateStudy(sampler: new RandomSampler(36));
        Exception? stopError = null;
        var firstTrial = new ManualResetEventSlim();

        var stopper = new Thread(() =>
        {
            firstTrial.Wait(TimeSpan.FromSeconds(30));
            try
            {
                study.Stop();
            }
            catch (Exception e)
            {
                stopError = e;
            }
        });
        stopper.Start();

        var watch = Stopwatch.StartNew();
        study.Optimize(t =>
        {
            firstTrial.Set();
            Thread.Sleep(10);
            return t.SuggestFloat("x", 0, 1);
        }, timeout: 0.8, nJobs: 2);
        stopper.Join();

        stopError.Should().BeOfType<RuntimeError>().Which.Message
            .Should().Be("`Study.stop` is supposed to be invoked inside an objective function or a callback.");
        watch.Elapsed.TotalSeconds.Should().BeGreaterThanOrEqualTo(0.8, "the refused stop did not end the run");
        ShouldBeFinishedAndContiguous(study);
    }

    /// <summary>
    ///     A run ended by <c>Stop()</c> leaves the study usable: the next <c>Optimize</c> clears the stop request and
    ///     runs its full count, under nJobs too (Optuna 5.0: stop in trial 2, then 5 more at <c>n_jobs=2</c>: 3 then 8).
    /// </summary>
    [TestMethod]
    public void Optimize_AfterAStoppedRun_RunsItsFullCount()
    {
        var study = Optuna.CreateStudy(sampler: new RandomSampler(37));
        study.Optimize(t =>
        {
            double x = t.SuggestFloat("x", 0, 1);
            if (t.Number == 2)
                t.Study.Stop();
            return x;
        }, nTrials: 10);
        study.Trials.Should().HaveCount(3);

        study.Optimize(t => t.SuggestFloat("x", 0, 1), nTrials: 5, nJobs: 2);

        study.Trials.Should().HaveCount(8);
        ShouldBeFinishedAndContiguous(study);
    }

    /// <summary>
    ///     A storage that fails to write a trial's final state: OptunaSharp raises the storage's own exception (never
    ///     subject to <c>catch</c>), after the trials in flight finish, and the trial stays RUNNING, since its final
    ///     state was never stored.
    /// </summary>
    /// <remarks>
    ///     [Misaligned] Optuna 5.0 raises <c>AssertionError('Should not reach.')</c> here instead: the logging
    ///     <c>finally</c> of <c>_run_trial</c> finds the trial still RUNNING and hits <c>assert False</c>, which replaces
    ///     the storage's <c>OSError</c> (under <c>python -O</c> the assert is stripped and the <c>OSError</c> propagates,
    ///     as here). Observed: the AssertionError, 6 trials sequentially, trial 5 RUNNING. OptunaSharp keeps the
    ///     original error, so the caller learns that the disk is full.
    /// </remarks>
    /// <param name="nJobs">The worker count; 1 is the sequential path, whose trial count is exact.</param>
    [TestMethod]
    [Misaligned]
    [DataRow(1)]
    [DataRow(4)]
    public void StorageFailureOnTheFinalWrite_RaisesTheStorageError_AndLeavesTheTrialRunning(int nJobs)
    {
        var study = Optuna.CreateStudy(storage: new FinalWriteFailingStorage(failingNumber: 5), sampler: new RandomSampler(38));

        var act = () => study.Optimize(t =>
        {
            Thread.Sleep(nJobs == 1 ? 0 : 20);
            return t.SuggestFloat("x", 0, 1);
        }, nTrials: 20, nJobs: nJobs, @catch: new[] { typeof(IOException) });

        act.Should().Throw<IOException>().WithMessage("disk full");
        var trials = study.Trials;
        trials.Single(t => t.Number == 5).State.Should().Be(TrialState.Running);
        trials.Where(t => t.Number != 5).Should().OnlyContain(t => t.State == TrialState.Complete,
            "Optimize waited for the other workers before raising");
        if (nJobs == 1)
            trials.Should().HaveCount(6);
    }

    /// <summary>
    ///     The run warns and drops the bar when asked for a timeout-only progress bar with nJobs, as Python does
    ///     (Optuna 5.0, verbatim: "The timeout-based progress bar is not supported with n_jobs != 1."), and still runs.
    /// </summary>
    [TestMethod]
    public void NJobs_TimeoutOnlyProgressBar_Warns_AndRunsWithoutTheBar()
    {
        var messages = new List<string>();
        void Handler(string msg, WarningCategory cat) => messages.Add(msg);

        var study = Optuna.CreateStudy(sampler: new RandomSampler(39));
        OptunaWarnings.WarningRaised += Handler;
        try
        {
            study.Optimize(t => t.SuggestFloat("x", 0, 1), timeout: 0.2, nJobs: 2, showProgressBar: true);
        }
        finally
        {
            OptunaWarnings.WarningRaised -= Handler;
        }

        messages.Should().Contain("The timeout-based progress bar is not supported with n_jobs != 1.");
        study.Trials.Should().NotBeEmpty();
    }

    /// <summary>
    ///     Asserts that every trial of <paramref name="study" /> has finished and that the trial numbers run from 0
    ///     without a gap.
    /// </summary>
    /// <param name="study">The study whose trials are checked.</param>
    private static void ShouldBeFinishedAndContiguous(Study study)
    {
        var trials = study.Trials;
        trials.Should().NotContain(t => t.State == TrialState.Running || t.State == TrialState.Waiting,
            "Optimize returns only once every trial it started has finished");
        trials.Select(t => t.Number).OrderBy(n => n).Should().Equal(Enumerable.Range(0, trials.Count));
    }

    /// <summary>Raises <paramref name="location" /> to <paramref name="value" /> if it is lower, atomically.</summary>
    /// <param name="location">The running maximum.</param>
    /// <param name="value">The candidate value.</param>
    private static void InterlockedMax(ref int location, int value)
    {
        int seen;
        while ((seen = Volatile.Read(ref location)) < value && Interlocked.CompareExchange(ref location, value, seen) != seen)
        {
        }
    }

    /// <summary>
    ///     Runs <paramref name="action" /> on a dedicated thread and fails the test if it does not finish within
    ///     <paramref name="seconds" />, so a deadlock reports as a failure instead of hanging the test run.
    /// </summary>
    /// <param name="seconds">The time the action is allowed.</param>
    /// <param name="action">The work to run.</param>
    /// <exception cref="Exception">Whatever <paramref name="action" /> throws, rethrown unwrapped.</exception>
    private static void RunWithin(int seconds, Action action)
    {
        var task = Task.Factory.StartNew(action, CancellationToken.None, TaskCreationOptions.LongRunning, TaskScheduler.Default);
        ((IAsyncResult)task).AsyncWaitHandle.WaitOne(TimeSpan.FromSeconds(seconds))
            .Should().BeTrue($"the run must finish within {seconds} s; a sampler monitor left held deadlocks it");
        task.GetAwaiter().GetResult();
    }

    /// <summary>
    ///     A <see cref="RandomSampler" /> whose hooks fail for one trial number each, as a broken user sampler would:
    ///     <c>BeforeTrial</c> inside <c>Ask</c>, <c>SampleIndependent</c> inside the objective's suggest call, and
    ///     <c>AfterTrial</c> inside <c>Tell</c>.
    /// </summary>
    /// <param name="seed">The random sampler's seed.</param>
    /// <param name="beforeTrialFailsAt">The trial number whose <c>BeforeTrial</c> throws; -1 for none.</param>
    /// <param name="sampleFailsAt">The trial number whose <c>SampleIndependent</c> throws; -1 for none.</param>
    /// <param name="afterTrialFailsAt">The trial number whose <c>AfterTrial</c> throws; -1 for none.</param>
    private sealed class HookFailingSampler(int seed, int beforeTrialFailsAt = -1, int sampleFailsAt = -1, int afterTrialFailsAt = -1)
        : RandomSampler(seed)
    {
        /// <summary>Fails for the configured trial, as a sampler hook raising inside <c>Ask</c>.</summary>
        /// <param name="study">The study that owns the trial.</param>
        /// <param name="trial">The new trial's snapshot.</param>
        /// <exception cref="InvalidOperationException">The trial is the configured one.</exception>
        public override void BeforeTrial(Study study, FrozenTrial trial)
        {
            if (trial.Number == beforeTrialFailsAt)
                throw new InvalidOperationException("before_trial");
            base.BeforeTrial(study, trial);
        }

        /// <summary>Fails for the configured trial; otherwise samples like the random sampler.</summary>
        /// <param name="study">The study that owns the trial.</param>
        /// <param name="trial">The trial being sampled.</param>
        /// <param name="paramName">The parameter's name.</param>
        /// <param name="paramDistribution">The parameter's distribution.</param>
        /// <returns>A uniform draw from <paramref name="paramDistribution" />.</returns>
        /// <exception cref="InvalidOperationException">The trial is the configured one.</exception>
        public override object? SampleIndependent(Study study, FrozenTrial trial, string paramName, BaseDistribution paramDistribution)
        {
            if (trial.Number == sampleFailsAt)
                throw new InvalidOperationException("sample");
            return base.SampleIndependent(study, trial, paramName, paramDistribution);
        }

        /// <summary>Fails for the configured trial, as a sampler hook raising inside <c>Tell</c>.</summary>
        /// <param name="study">The study that owns the trial.</param>
        /// <param name="trial">The trial's snapshot before its final state is stored.</param>
        /// <param name="state">The final state about to be stored.</param>
        /// <param name="values">The trial's values, or null.</param>
        /// <exception cref="InvalidOperationException">The trial is the configured one.</exception>
        public override void AfterTrial(Study study, FrozenTrial trial, TrialState state, IReadOnlyList<double>? values)
        {
            if (trial.Number == afterTrialFailsAt)
                throw new InvalidOperationException("after_trial");
            base.AfterTrial(study, trial, state, values);
        }
    }

    /// <summary>
    ///     An in-memory storage whose final-state write fails for one trial number, as a full disk or a dropped
    ///     connection would fail a persistent storage's.
    /// </summary>
    /// <param name="failingNumber">The trial number whose final write fails.</param>
    private sealed class FinalWriteFailingStorage(int failingNumber) : InMemoryStorage
    {
        /// <summary>Fails the final write of the configured trial; otherwise stores like the in-memory storage.</summary>
        /// <param name="trialId">The trial's storage identifier.</param>
        /// <param name="state">The new state.</param>
        /// <param name="values">The trial's values, or null.</param>
        /// <returns>Whether the state was changed, as the in-memory storage reports it.</returns>
        /// <exception cref="IOException">The write is the configured trial's final one.</exception>
        public override bool SetTrialStateValues(int trialId, TrialState state, IReadOnlyList<double>? values = null)
        {
            if (state != TrialState.Running && GetTrial(trialId).Number == failingNumber)
                throw new IOException("disk full");
            return base.SetTrialStateValues(trialId, state, values);
        }
    }
}
