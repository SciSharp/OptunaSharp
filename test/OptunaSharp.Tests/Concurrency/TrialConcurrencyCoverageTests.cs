using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using AwesomeAssertions;
using NumSharp;
using OptunaSharp.Samplers;
using OptunaSharp.Storages;

namespace OptunaSharp.Tests.Concurrency;

/// <summary>
///     The guarantees of one trial driven by many threads at the same instant (THREADING.md §6.2, §6.4, §6.5), each round
///     fired by a <see cref="ConcurrencyGun" /> so every shooter enters the same trial operation together. A trial
///     serializes its own operations: the first report of a step wins and the trial's view agrees with the storage, a name
///     suggested by every thread is sampled once, and distinct names are each sampled once with distinct draws. Several
///     tells of one trial store one result and refuse the rest, with the sampler's <c>AfterTrial</c> run once. Pruning
///     decisions of concurrent trials are each made on that trial's values, one at a time.
/// </summary>
/// <remarks>
///     The sampler and pruner are probes (<see cref="OverlapProbeSampler" />, <see cref="OverlapProbePruner" />) that record
///     whether two calls ever overlapped inside them, which the monitors must prevent; every test asserts that too. These
///     invariants hold on any machine, so unlike the gun's race-catching tests they need no parallel hardware.
/// </remarks>
[TestClass]
public class TrialConcurrencyCoverageTests
{
    private const int Threads = ConcurrencyGun.DefaultThreads;
    private const int Rounds = 60;

    /// <summary>
    ///     Every shooter reports a different value for step 0 of one trial at once. The step is written to the storage
    ///     exactly once, by the first report: the others are ignored as "already reported". The stored value is one
    ///     shooter's, and the trial's own view holds that same value. Optuna's own <c>Trial</c> races there (on Optuna 5.0
    ///     the last report won in 63% of contended steps); the trial monitor makes the check and the write one step.
    /// </summary>
    /// <remarks>
    ///     The write count is the assertion with teeth. Without the trial monitor several shooters pass the "already
    ///     reported?" check and each overwrites the step, which makes the last report win. Each such shooter still writes
    ///     its own view right after its storage write, so the view and the storage usually still agree: a mutation check
    ///     that removed the monitor passed every other assertion here.
    /// </remarks>
    [TestMethod]
    public void Report_OfOneStepFromEveryThreadAtOnce_WritesItOnce_AndTheTrialSeesThatValue()
    {
        var storage = new WriteCountingStorage();
        var study = Optuna.CreateStudy(storage: storage, sampler: new RandomSampler(1));
        var reported = new double[Threads];
        Trial trial = null!;

        // Every round, seven shooters report a step already reported, and each such report logs a warning.
        using var quiet = new QuietLogs();
        using var gun = new ConcurrencyGun(Threads);
        GunReport report = gun.Fire(Rounds,
            shot =>
            {
                double value = shot.Round * 100 + shot.Thread;
                reported[shot.Thread] = value;
                trial.Report(value, step: 0);
            },
            beforeRound: _ => trial = study.Ask(),
            afterRound: r =>
            {
                storage.IntermediateWritesOf(trial.TrialId).Should().Be(1, "round {0}: only the first report of a step is written", r);
                FrozenTrial stored = study.Storage.GetTrial(trial.TrialId);
                stored.IntermediateValues.Should().ContainSingle("round {0} reported one step", r);
                double kept = stored.IntermediateValues[0];
                reported.Should().Contain(kept, "the stored value is one shooter's report");
                trial.CachedFrozenTrial.IntermediateValues[0].Should().Be(kept, "the trial's view agrees with the storage");
                study.Tell(trial, kept);
                return false;
            });

        report.AssertClean();
    }

    /// <summary>
    ///     Every shooter suggests the same parameter name on one trial at once: the sampler is asked once, every shooter
    ///     receives that one value, and the storage holds it, in every round.
    /// </summary>
    [TestMethod]
    public void Suggest_OfOneNameFromEveryThreadAtOnce_SamplesItOnce_AndEveryThreadGetsThatValue()
    {
        var probe = new OverlapProbeSampler(seed: 2);
        var study = Optuna.CreateStudy(sampler: probe);
        var got = new double[Threads];
        Trial trial = null!;
        int callsBefore = 0;

        using var quiet = new QuietLogs();
        using var gun = new ConcurrencyGun(Threads);
        GunReport report = gun.Fire(Rounds,
            shot => got[shot.Thread] = trial.SuggestFloat("x", 0, 1),
            beforeRound: _ =>
            {
                trial = study.Ask();
                callsBefore = probe.SampleIndependentCalls;
            },
            afterRound: r =>
            {
                got.Distinct().Should().ContainSingle("round {0}: one name is sampled once per trial", r);
                (probe.SampleIndependentCalls - callsBefore).Should().Be(1);
                study.Storage.GetTrial(trial.TrialId).Params["x"].Should().Be(got[0]);
                study.Tell(trial, got[0]);
                return false;
            });

        report.AssertClean();
        probe.PeakOverlap.Should().Be(1, "suggestions on one trial take turns, so the sampler is never entered twice at once");
    }

    /// <summary>
    ///     Every shooter suggests a parameter of its own on one trial at once: each name is sampled once, the storage holds
    ///     all of them with the values the shooters received, and no two draws of the whole volley repeat, because the
    ///     sampler's generator is drawn one call at a time.
    /// </summary>
    [TestMethod]
    public void Suggest_OfDistinctNamesFromEveryThreadAtOnce_StoresEveryName_WithDistinctDraws()
    {
        var probe = new OverlapProbeSampler(seed: 3);
        var study = Optuna.CreateStudy(sampler: probe);
        var got = new double[Threads];
        var everyDraw = new List<double>();
        Trial trial = null!;

        using var quiet = new QuietLogs();
        using var gun = new ConcurrencyGun(Threads);
        GunReport report = gun.Fire(Rounds,
            shot => got[shot.Thread] = trial.SuggestFloat($"x{shot.Thread}", 0, 1),
            beforeRound: _ => trial = study.Ask(),
            afterRound: r =>
            {
                FrozenTrial stored = study.Storage.GetTrial(trial.TrialId);
                stored.Params.Should().HaveCount(Threads, "round {0}: every shooter's name is stored", r);
                for (int t = 0; t < Threads; t++)
                    stored.Params[$"x{t}"].Should().Be(got[t]);
                everyDraw.AddRange(got);
                study.Tell(trial, got.Sum());
                return false;
            });

        report.AssertClean();
        probe.SampleIndependentCalls.Should().Be(Threads * Rounds);
        everyDraw.Should().OnlyHaveUniqueItems("a generator drawn one call at a time never repeats a value");
        probe.PeakOverlap.Should().Be(1);
    }

    /// <summary>
    ///     Every shooter tells the same trial at once, half through the <c>Trial</c> object and half through its number,
    ///     which resolves to the same tell stripe. Exactly one tell is stored and runs the sampler's <c>AfterTrial</c>; every
    ///     other one is refused with Optuna's <c>ValueError: Cannot tell a COMPLETE trial.</c> (THREADING.md §6.5).
    /// </summary>
    [TestMethod]
    public void Tell_OfOneTrialFromEveryThreadAtOnce_StoresOne_AndRefusesEveryOther()
    {
        var probe = new OverlapProbeSampler(seed: 4);
        var study = Optuna.CreateStudy(sampler: probe);
        var stored = new bool[Threads];
        var refusals = new Exception?[Threads];
        Trial trial = null!;

        using var quiet = new QuietLogs();
        using var gun = new ConcurrencyGun(Threads);
        GunReport report = gun.Fire(Rounds,
            shot =>
            {
                try
                {
                    if (shot.Thread % 2 == 0)
                        study.Tell(trial, shot.Thread);
                    else
                        study.Tell(trial.Number, shot.Thread);
                    stored[shot.Thread] = true;
                }
                catch (Exception e)
                {
                    refusals[shot.Thread] = e;
                }
            },
            beforeRound: _ =>
            {
                trial = study.Ask();
                Array.Clear(stored);
                Array.Clear(refusals);
            },
            afterRound: r =>
            {
                stored.Count(s => s).Should().Be(1, "round {0}: exactly one tell of a trial is stored", r);
                int winner = Array.IndexOf(stored, true);
                refusals.Where((_, t) => t != winner).Should()
                    .OnlyContain(e => e is ValueError && e.Message == "Cannot tell a COMPLETE trial.");
                FrozenTrial result = study.Storage.GetTrial(trial.TrialId);
                result.State.Should().Be(TrialState.Complete);
                result.Value.Should().Be(winner, "the stored value is the winning shooter's");
                probe.AfterTrialCallsOf(trial.TrialId).Should().Be(1, "only the stored tell reaches AfterTrial");
                return false;
            });

        report.AssertClean();
        probe.PeakOverlap.Should().Be(1);
    }

    /// <summary>
    ///     Every shooter tells the same trial at once with <c>skipIfFinished</c>: none raises, all of them return the one
    ///     stored result, and <c>AfterTrial</c> runs once.
    /// </summary>
    [TestMethod]
    public void Tell_WithSkipIfFinished_FromEveryThreadAtOnce_ReturnsTheOneStoredResult()
    {
        var probe = new OverlapProbeSampler(seed: 5);
        var study = Optuna.CreateStudy(sampler: probe);
        var results = new FrozenTrial?[Threads];
        Trial trial = null!;

        using var quiet = new QuietLogs();
        using var gun = new ConcurrencyGun(Threads);
        GunReport report = gun.Fire(Rounds,
            shot => results[shot.Thread] = study.Tell(trial, shot.Thread, skipIfFinished: true),
            beforeRound: _ =>
            {
                trial = study.Ask();
                Array.Clear(results);
            },
            afterRound: r =>
            {
                double? storedValue = study.Storage.GetTrial(trial.TrialId).Value;
                results.Should().OnlyContain(x => x != null && x.State == TrialState.Complete && x.Value == storedValue,
                    "round {0}: every tell returns the one stored result", r);
                probe.AfterTrialCallsOf(trial.TrialId).Should().Be(1);
                return false;
            });

        report.AssertClean();
        probe.PeakOverlap.Should().Be(1);
    }

    /// <summary>
    ///     Every shooter reports on a trial of its own and asks the shared pruner at once whether to prune it. Each decision
    ///     is made on that shooter's own trial (the probe prunes on an odd last value, and each shooter reported a different
    ///     parity), and the pruner monitor admits one decision at a time (THREADING.md §6.4).
    /// </summary>
    [TestMethod]
    public void ShouldPrune_OfEveryThreadsOwnTrialAtOnce_DecidesEachOnItsOwnValues_OneAtATime()
    {
        var pruner = new OverlapProbePruner();
        var study = Optuna.CreateStudy(sampler: new RandomSampler(6), pruner: pruner);
        var trials = new Trial[Threads];
        var pruned = new bool[Threads];

        using var quiet = new QuietLogs();
        using var gun = new ConcurrencyGun(Threads);
        GunReport report = gun.Fire(Rounds,
            shot =>
            {
                trials[shot.Thread].Report(shot.Thread + shot.Round, step: 0);
                pruned[shot.Thread] = trials[shot.Thread].ShouldPrune();
            },
            beforeRound: _ =>
            {
                for (int t = 0; t < Threads; t++)
                    trials[t] = study.Ask();
            },
            afterRound: r =>
            {
                for (int t = 0; t < Threads; t++)
                {
                    pruned[t].Should().Be((t + r) % 2 == 1, "round {0}, thread {1}: the decision is on that trial's own value", r, t);
                    study.Tell(trials[t], t);
                }
                return false;
            });

        report.AssertClean();
        pruner.Calls.Should().Be(Threads * Rounds);
        pruner.PeakOverlap.Should().Be(1, "the pruner monitor admits one decision at a time");
    }

    /// <summary>
    ///     Half the shooters write one trial at once (user attributes and new parameters) while the other half read its
    ///     <c>Params</c>, <c>Distributions</c> and <c>UserAttrs</c> over and over. Each read is a whole copy taken under the
    ///     trial monitor (THREADING.md §6.2): enumerating it never throws, and a reader never sees a view smaller than one
    ///     it saw before. Every write lands.
    /// </summary>
    [TestMethod]
    public void TrialViews_ReadWhileOthersWriteTheTrial_AreWholeCopies_ThatNeverShrink()
    {
        const int rounds = 30, writes = 15, reads = 150;
        var study = Optuna.CreateStudy(sampler: new RandomSampler(31));
        Trial trial = null!;
        int shrunk = 0;

        using var quiet = new QuietLogs();
        using var gun = new ConcurrencyGun(Threads);
        GunReport report = gun.Fire(rounds,
            shot =>
            {
                if (shot.Thread % 2 == 0)
                {
                    for (int k = 0; k < writes; k++)
                    {
                        trial.SetUserAttr($"a{shot.Thread}-{k}", k);
                        trial.SuggestFloat($"p{shot.Thread}-{k}", 0, 1);
                    }
                    return;
                }
                int lastAttrs = 0, lastParams = 0, lastDistributions = 0;
                for (int k = 0; k < reads; k++)
                {
                    // Each view is enumerated in full: a live dictionary being written would throw here.
                    int attrs = trial.UserAttrs.Count(kv => kv.Key.Length > 0);
                    int parameters = trial.Params.Count(kv => kv.Key.Length > 0);
                    int distributions = trial.Distributions.Count(kv => kv.Value is not null);
                    if (attrs < lastAttrs || parameters < lastParams || distributions < lastDistributions)
                        Interlocked.Increment(ref shrunk);
                    (lastAttrs, lastParams, lastDistributions) = (attrs, parameters, distributions);
                }
            },
            beforeRound: _ => trial = study.Ask(),
            afterRound: r =>
            {
                int expected = Threads / 2 * writes;
                trial.UserAttrs.Should().HaveCount(expected, "round {0}: every attribute write lands", r);
                trial.Params.Should().HaveCount(expected);
                study.Storage.GetTrial(trial.TrialId).Params.Should().HaveCount(expected);
                study.Tell(trial, 0.0);
                return false;
            });

        report.AssertClean();
        shrunk.Should().Be(0, "a view is a copy of the trial at one instant, and the trial only grows");
    }

    /// <summary>
    ///     Every shooter sets the same user attribute of one trial at once, each to its own value, through a storage that
    ///     holds every writer a little after its write, longer for lower shooter indices. The trial's own view and the
    ///     storage end with the same value: the trial monitor makes the storage write and the view write one step.
    /// </summary>
    /// <remarks>
    ///     The delay is what gives this test its teeth. Without the trial monitor, writers still reach the storage one at
    ///     a time, in some order, but then update the trial's view after delays of different lengths, so the view's last
    ///     writer differs from the storage's. Without the delay that reordering needs a preemption at the wrong instant,
    ///     which a test would rarely see.
    /// </remarks>
    [TestMethod]
    public void SetUserAttr_OfOneKeyFromEveryThreadAtOnce_LeavesTheTrialViewEqualToTheStorage()
    {
        const int rounds = 40;
        var storage = new DelayedAttrStorage();
        var study = Optuna.CreateStudy(storage: storage, sampler: new RandomSampler(32));
        Trial trial = null!;

        using var quiet = new QuietLogs();
        using var gun = new ConcurrencyGun(Threads);
        GunReport report = gun.Fire(rounds,
            shot => trial.SetUserAttr("k", shot.Thread),
            beforeRound: _ => trial = study.Ask(),
            afterRound: r =>
            {
                object? stored = study.Storage.GetTrial(trial.TrialId).UserAttrs["k"];
                trial.UserAttrs["k"].Should().Be(stored, "round {0}: the trial's view holds the value the storage kept", r);
                study.Tell(trial, 0.0);
                return false;
            });

        report.AssertClean();
    }

    /// <summary>
    ///     An in-memory storage that, after storing a trial user attribute whose value is a shooter index, keeps the writer
    ///     busy for a time that shrinks with the index. That reorders whatever a caller does right after the write.
    /// </summary>
    private sealed class DelayedAttrStorage : InMemoryStorage
    {
        /// <summary>Stores the attribute as <see cref="InMemoryStorage" /> does, then delays the writer by its index.</summary>
        /// <param name="trialId">The trial's storage identifier.</param>
        /// <param name="key">The attribute name.</param>
        /// <param name="value">The attribute value; an <see cref="int" /> shooter index sets the delay.</param>
        /// <exception cref="UpdateFinishedTrialError">The trial is already finished.</exception>
        /// <exception cref="KeyError">The trial does not exist.</exception>
        public override void SetTrialUserAttr(int trialId, string key, object? value)
        {
            base.SetTrialUserAttr(trialId, key, value);
            // Lower indices wait longer, so a writer that reached the storage first tends to finish last.
            if (value is int index)
                Thread.SpinWait((Threads - index) * 4_000);
        }
    }

    /// <summary>
    ///     An in-memory storage that counts the intermediate-value writes it receives per trial, so a test can tell one
    ///     report of a step from several that overwrote each other.
    /// </summary>
    private sealed class WriteCountingStorage : InMemoryStorage
    {
        private readonly ConcurrentDictionary<int, int> _writes = new();

        /// <summary>How many intermediate values were written for a trial, whatever their steps.</summary>
        /// <param name="trialId">The trial's storage identifier.</param>
        /// <returns>The write count; 0 for a trial that never reported.</returns>
        internal int IntermediateWritesOf(int trialId) => _writes.TryGetValue(trialId, out int writes) ? writes : 0;

        /// <summary>Counts the write, then stores the value as <see cref="InMemoryStorage" /> does.</summary>
        /// <param name="trialId">The trial's storage identifier.</param>
        /// <param name="step">The reported step.</param>
        /// <param name="intermediateValue">The reported value.</param>
        /// <exception cref="UpdateFinishedTrialError">The trial is already finished.</exception>
        /// <exception cref="KeyError">The trial does not exist.</exception>
        public override void SetTrialIntermediateValue(int trialId, long step, double intermediateValue)
        {
            // Counted before the base write, so a write the storage refuses still shows up as an attempt.
            _writes.AddOrUpdate(trialId, 1, (_, writes) => writes + 1);
            base.SetTrialIntermediateValue(trialId, step, intermediateValue);
        }
    }
}
