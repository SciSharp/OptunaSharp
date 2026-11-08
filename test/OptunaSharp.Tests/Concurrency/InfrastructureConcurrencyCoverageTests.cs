using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using AwesomeAssertions;
using NumSharp;
using OptunaSharp.Distributions;
using OptunaSharp.Internal;
using OptunaSharp.Logging;
using OptunaSharp.Pruners;
using OptunaSharp.Samplers;
using OptunaSharp.Storages;

namespace OptunaSharp.Tests.Concurrency;

/// <summary>
///     The concurrency mechanisms under the study and trial layers, each entered by every shooter of a
///     <see cref="ConcurrencyGun" /> at the same instant (THREADING.md §3.5, §5, §6.3, §6.6). The four monitors before the
///     GIL (a sampler's and a pruner's monitor, a tell stripe, and a <see cref="MonitorGate" /> scope) admit one thread at
///     a time. A sampler's generator stays whole while workers reseed it and others draw, and one sampler serves a study
///     per thread without repeating a draw. A categorical distribution's index, built on first use, maps every choice
///     correctly whoever builds it. The <c>nJobs</c> worker scheduler runs every task queued at once exactly once, all of
///     them in flight together. Log handlers registered and removed while others log never break a record's delivery.
/// </summary>
[TestClass]
public class InfrastructureConcurrencyCoverageTests
{
    private const int Threads = ConcurrencyGun.DefaultThreads;

    /// <summary>
    ///     Every shooter enters the same monitor fifty times a round at once, through the entry OptunaSharp itself uses: no
    ///     two are ever inside together, and every entry happens. The tell stripe is checked for two trials that share a
    ///     stripe (ids 7 and 71 of 64 stripes), since a stripe guards every trial mapped to it.
    /// </summary>
    /// <param name="monitor">
    ///     <c>sampler</c> (<c>BaseSampler.Locked</c>), <c>pruner</c> (<c>BasePruner.Locked</c>), <c>tell stripe</c>
    ///     (<c>BaseStorage.TellLocked</c>) or <c>gate scope</c> (<c>MonitorGate.Scope</c>, the trial monitor's entry).
    /// </param>
    [TestMethod]
    [DataRow("sampler")]
    [DataRow("pruner")]
    [DataRow("tell stripe")]
    [DataRow("gate scope")]
    public void Monitor_EnteredByEveryThreadAtOnce_AdmitsOneThreadAtATime(string monitor)
    {
        const int rounds = 40, entriesPerShot = 50;
        Action<Action, int> enter = EntryFor(monitor);
        int inside = 0, peak = 0, entries = 0;

        using var gun = new ConcurrencyGun(Threads);
        GunReport report = gun.Fire(rounds, shot =>
        {
            for (int k = 0; k < entriesPerShot; k++)
            {
                enter(() =>
                {
                    Watermark.Raise(ref peak, Interlocked.Increment(ref inside));
                    // Stay inside a little, so a second thread let in would be seen.
                    Thread.SpinWait(100);
                    Interlocked.Increment(ref entries);
                    Interlocked.Decrement(ref inside);
                }, shot.Thread);
            }
        });

        report.AssertClean();
        peak.Should().Be(1, "the {0} admits one thread at a time", monitor);
        entries.Should().Be(Threads * rounds * entriesPerShot);
    }

    /// <summary>
    ///     Shooter 0 reseeds a sampler over and over, as every <c>nJobs</c> worker does before its trial, while every other
    ///     shooter draws from the same generator, all through the sampler monitor. No draw throws and no value repeats: a
    ///     reseed never lands between the steps of a draw (THREADING.md §6.3).
    /// </summary>
    [TestMethod]
    public void ReseedRng_WhileEveryOtherThreadDraws_NeverTearsTheGenerator()
    {
        const int rounds = 20, reseeds = 50, draws = 500;
        var sampler = new RandomSampler(8);
        NumPyRandom rng = sampler.Rng.Rng;
        var everyDraw = new List<double>();
        double[][] drawn = Enumerable.Range(0, Threads).Select(_ => new double[draws]).ToArray();

        using var gun = new ConcurrencyGun(Threads);
        GunReport report = gun.Fire(rounds,
            shot =>
            {
                if (shot.Thread == 0)
                {
                    for (int k = 0; k < reseeds; k++)
                        sampler.Locked(sampler.ReseedRng);
                    return;
                }
                double[] mine = drawn[shot.Thread];
                for (int k = 0; k < draws; k++)
                    mine[k] = sampler.Locked(() => (double)rng.rand());
            },
            afterRound: _ =>
            {
                everyDraw.AddRange(drawn.Skip(1).SelectMany(buffer => buffer));
                return false;
            });

        report.AssertClean();
        everyDraw.Should().HaveCount(rounds * (Threads - 1) * draws).And.OnlyHaveUniqueItems();
    }

    /// <summary>
    ///     One sampler serves a study per shooter, each in its own in-memory storage, so every study has id 0: all shooters
    ///     ask, suggest and tell at once. No draw repeats across the studies, and every study is whole (THREADING.md §1.5:
    ///     a stateless sampler can serve several studies at once).
    /// </summary>
    [TestMethod]
    public void OneSampler_ServingAStudyPerThread_NeverRepeatsADraw()
    {
        const int rounds = 40, trialsPerShot = 5;
        var sampler = new RandomSampler(9);
        Study[] studies = Enumerable.Range(0, Threads).Select(_ => Optuna.CreateStudy(sampler: sampler)).ToArray();

        using var quiet = new QuietLogs();
        using var gun = new ConcurrencyGun(Threads);
        GunReport report = gun.Fire(rounds, shot =>
        {
            Study study = studies[shot.Thread];
            for (int k = 0; k < trialsPerShot; k++)
            {
                Trial trial = study.Ask();
                study.Tell(trial, trial.SuggestFloat("x", 0, 1));
            }
        });

        report.AssertClean();
        studies.Should().OnlyContain(s => s.Trials.Select(t => t.Number).SequenceEqual(Enumerable.Range(0, rounds * trialsPerShot)));
        studies.SelectMany(s => s.Trials).Select(t => t.Value).Should().OnlyHaveUniqueItems("one generator, drawn one call at a time");
    }

    /// <summary>
    ///     A fresh categorical distribution each round, converted by every shooter at once. Its choice index is built on
    ///     first use and published whole, so a shooter sees either the finished index or builds an identical one. Each
    ///     shooter starts from a different choice, so the first conversions of a round race, and every choice still maps
    ///     to its index both ways.
    /// </summary>
    /// <remarks>
    ///     This pins the observable contract, correct conversions without an exception, not the publication itself. A
    ///     mutation check that published the index before filling it passed: a lookup that misses falls back to the linear
    ///     scan, which still returns the right index, so the only cost of that bug is the lost fast path.
    /// </remarks>
    [TestMethod]
    public void CategoricalDistribution_ConvertedByEveryThreadAtOnce_OnFirstUse_MapsEveryChoiceToItsIndex()
    {
        const int rounds = 100;
        object?[] values = Enumerable.Range(0, 40).Select(i => (object?)$"s{i}")
            .Concat(Enumerable.Range(0, 20).Select(i => (object?)(1000L + i)))
            .Concat(new object?[] { null, true, 2.5, -7.25 })
            .ToArray();
        CategoricalDistribution distribution = null!;

        using var gun = new ConcurrencyGun(Threads);
        GunReport report = gun.Fire(rounds,
            shot =>
            {
                for (int k = 0; k < values.Length; k++)
                    CheckRoundTrip(distribution, values, (k + shot.Thread * 7) % values.Length);
            },
            beforeRound: _ => distribution = new CategoricalDistribution(values.Select(CategoricalChoice.From)));

        report.AssertClean();
    }

    /// <summary>
    ///     Every shooter queues four tasks on one <c>nJobs</c> worker scheduler at once, and each task blocks until all 32
    ///     of the round have started. That needs one worker per queued task at the same time, the "nJobs trials really in
    ///     flight" property of THREADING.md §3.5: a thread start lost by the scheduler's bookkeeping under simultaneous
    ///     queuing would leave a task queued while every worker waits, and the bounded wait would fail. Every task runs
    ///     exactly once, on a worker thread, never inlined into the shooter waiting for it.
    /// </summary>
    [TestMethod]
    public void TrialWorkerScheduler_TasksQueuedFromEveryThreadAtOnce_RunOnce_AllInFlightTogether()
    {
        const int rounds = 10, tasksPerShooter = 4, total = Threads * tasksPerShooter;
        // Its own scheduler, so this test's worker threads are not shared with optimize runs of other tests; they retire
        // two seconds after the test.
        var scheduler = new TrialWorkerScheduler(TimeSpan.FromSeconds(2));
        var runs = new int[total];
        var onWorker = new bool[total];
        var metEveryone = new bool[total];
        CountdownEvent allStarted = null!;

        using var gun = new ConcurrencyGun(Threads);
        GunReport report = gun.Fire(rounds,
            shot =>
            {
                var tasks = new Task[tasksPerShooter];
                for (int j = 0; j < tasksPerShooter; j++)
                {
                    int slot = shot.Thread * tasksPerShooter + j;
                    tasks[j] = Task.Factory.StartNew(() =>
                    {
                        Interlocked.Increment(ref runs[slot]);
                        onWorker[slot] = Thread.CurrentThread.Name == "OptunaSharp trial worker";
                        allStarted.Signal();
                        metEveryone[slot] = allStarted.Wait(TimeSpan.FromSeconds(20));
                    }, CancellationToken.None, TaskCreationOptions.DenyChildAttach, scheduler);
                }
                Task.WaitAll(tasks);
            },
            beforeRound: _ =>
            {
                allStarted = new CountdownEvent(total);
                Array.Clear(runs);
                Array.Clear(onWorker);
                Array.Clear(metEveryone);
            },
            afterRound: r =>
            {
                runs.Should().OnlyContain(n => n == 1, "round {0}: every queued task runs exactly once", r);
                onWorker.Should().OnlyContain(w => w, "a task runs on a worker thread, never inlined into its waiter");
                metEveryone.Should().OnlyContain(m => m, "all {0} tasks were in flight together", total);
                return false;
            });

        report.AssertClean();
    }

    /// <summary>
    ///     Half the shooters register and remove handlers of their own over and over while the other half log at once.
    ///     <c>Emit</c> calls a snapshot of the handler list taken under the logging lock (THREADING.md §6.6), so no record is
    ///     lost to a concurrent change of the list, and no emit fails: a handler registered throughout receives every
    ///     record.
    /// </summary>
    [TestMethod]
    public void LogHandlers_RegisteredAndRemovedWhileOthersLog_NeverCostARecord()
    {
        const int rounds = 30, operations = 50;
        Logger logger = OptunaLogging.GetLogger("optuna.tests.concurrency_gun");
        var steady = new CountingLogHandler();

        int verbosity = OptunaLogging.GetVerbosity();
        // The default handler would write every record of the volley to stderr; this test only counts them.
        OptunaLogging.DisableDefaultHandler();
        OptunaLogging.SetVerbosity(LogLevel.Warning);
        OptunaLogging.AddHandler(steady);
        try
        {
            using var gun = new ConcurrencyGun(Threads);
            GunReport report = gun.Fire(rounds, shot =>
            {
                if (shot.Thread % 2 == 0)
                {
                    var mine = new CountingLogHandler();
                    for (int k = 0; k < operations; k++)
                    {
                        OptunaLogging.AddHandler(mine);
                        OptunaLogging.RemoveHandler(mine);
                    }
                }
                else
                {
                    for (int k = 0; k < operations; k++)
                        logger.Warning($"shooter {shot.Thread}, round {shot.Round}, record {k}");
                }
            });

            report.AssertClean();
            steady.Count.Should().Be(Threads / 2 * rounds * operations);
        }
        finally
        {
            OptunaLogging.RemoveHandler(steady);
            OptunaLogging.EnableDefaultHandler();
            OptunaLogging.SetVerbosity(verbosity);
        }
    }

    /// <summary>
    ///     How a shooter enters <paramref name="monitor" />: the same entry OptunaSharp's own code takes, wrapped so the
    ///     test body runs inside.
    /// </summary>
    /// <param name="monitor">The monitor's name, as in the test's data rows.</param>
    /// <returns>An action that runs its body inside the monitor; its second argument is the shooter's index.</returns>
    /// <exception cref="ArgumentOutOfRangeException">An unknown monitor name.</exception>
    private static Action<Action, int> EntryFor(string monitor)
    {
        switch (monitor)
        {
            case "sampler":
            {
                var sampler = new RandomSampler(10);
                return (body, _) => sampler.Locked(body);
            }
            case "pruner":
            {
                var pruner = new NopPruner();
                return (body, _) => pruner.Locked(() =>
                {
                    body();
                    return 0;
                });
            }
            case "tell stripe":
            {
                var storage = new InMemoryStorage();
                // Trials 7 and 71 share stripe 7 of 64: half the shooters tell one, half the other.
                return (body, thread) => storage.TellLocked(thread % 2 == 0 ? 7 : 71, () =>
                {
                    body();
                    return 0;
                });
            }
            case "gate scope":
            {
                var gate = new object();
                return (body, _) =>
                {
                    using (MonitorGate.Scope(gate))
                        body();
                };
            }
            default:
                throw new ArgumentOutOfRangeException(nameof(monitor), monitor, "Unknown monitor.");
        }
    }

    /// <summary>
    ///     Checks that the choice at <paramref name="index" /> converts to that index and back, as every shooter of the
    ///     categorical test does for every choice.
    /// </summary>
    /// <param name="distribution">The distribution, shared by the shooters.</param>
    /// <param name="values">The external values its choices were built from, in order.</param>
    /// <param name="index">The choice to check.</param>
    /// <exception cref="AssertFailedException">The choice converts to another index, or its index converts to another choice.</exception>
    private static void CheckRoundTrip(CategoricalDistribution distribution, object?[] values, int index)
    {
        Assert.AreEqual(index, (int)distribution.ToInternalRepr(values[index]), $"choice {index} maps to its index");
        Assert.AreEqual(index, (int)distribution.ToInternalRepr(distribution.ToExternalRepr(index)), $"index {index} maps back");
    }

    /// <summary>A log handler that counts the records it receives, safely from any number of threads.</summary>
    private sealed class CountingLogHandler : ILogHandler
    {
        private int _count;

        /// <summary>How many records the handler received.</summary>
        internal int Count => Volatile.Read(ref _count);

        /// <summary>Counts the record.</summary>
        /// <param name="record">The record; only counted.</param>
        public void Emit(in LogRecord record) => Interlocked.Increment(ref _count);
    }
}
