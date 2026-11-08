using System;
using System.Collections.Generic;
using System.IO;
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
///     them in flight together. Log handlers registered and removed while others log never break a record's delivery, and
///     each handler serves one record at a time under a lock of its own, as Python's <c>Handler.handle</c> does.
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
    ///     Every shooter logs through the same two handlers at once. Each handler serves one record at a time (Python's
    ///     <c>Handler.handle</c> holds the handler's lock around <c>emit</c>, THREADING.md §6.6), and both receive every
    ///     record. Without the lock, eight shooters were inside one handler together.
    /// </summary>
    [TestMethod]
    public void LogHandler_EnteredByEveryThreadAtOnce_ServesOneRecordAtATime()
    {
        const int rounds = 20, records = 50;
        Logger logger = OptunaLogging.GetLogger("optuna.tests.handler_gun");
        var first = new OverlapLogHandler();
        var second = new OverlapLogHandler();

        int verbosity = OptunaLogging.GetVerbosity();
        OptunaLogging.DisableDefaultHandler();
        OptunaLogging.SetVerbosity(LogLevel.Warning);
        OptunaLogging.AddHandler(first);
        OptunaLogging.AddHandler(second);
        try
        {
            using var gun = new ConcurrencyGun(Threads);
            GunReport report = gun.Fire(rounds, shot =>
            {
                for (int k = 0; k < records; k++)
                    logger.Warning($"shooter {shot.Thread}, round {shot.Round}, record {k}");
            });

            report.AssertClean();
            first.PeakInside.Should().Be(1);
            second.PeakInside.Should().Be(1);
            first.Count.Should().Be(Threads * rounds * records);
            second.Count.Should().Be(Threads * rounds * records);
        }
        finally
        {
            OptunaLogging.RemoveHandler(first);
            OptunaLogging.RemoveHandler(second);
            OptunaLogging.EnableDefaultHandler();
            OptunaLogging.SetVerbosity(verbosity);
        }
    }

    /// <summary>
    ///     The lock belongs to each handler, not to logging: while one thread is inside the second handler with its
    ///     record, another thread's record passes through the first handler. A single lock around all handlers would hold
    ///     the second thread out until the first finished, and the first waits for it, up to the timeout.
    /// </summary>
    [TestMethod]
    public void LogHandlers_EachHaveTheirOwnLock_SoDifferentHandlersServeDifferentThreadsAtOnce()
    {
        Logger logger = OptunaLogging.GetLogger("optuna.tests.handler_locks");
        using var insideSecond = new ManualResetEventSlim();
        using var otherPassedFirst = new ManualResetEventSlim();
        bool overlapped = false;
        var first = new DelegateLogHandler(record =>
        {
            if (record.Message == "other")
                otherPassedFirst.Set();
        });
        var second = new DelegateLogHandler(record =>
        {
            if (record.Message != "holder")
                return;
            insideSecond.Set();
            overlapped = otherPassedFirst.Wait(TimeSpan.FromSeconds(10));
        });

        int verbosity = OptunaLogging.GetVerbosity();
        OptunaLogging.DisableDefaultHandler();
        OptunaLogging.SetVerbosity(LogLevel.Warning);
        OptunaLogging.AddHandler(first);
        OptunaLogging.AddHandler(second);
        try
        {
            var holder = new Thread(() => logger.Warning("holder"));
            holder.Start();
            insideSecond.Wait(TimeSpan.FromSeconds(10)).Should().BeTrue("the holder reaches the second handler");
            var other = new Thread(() => logger.Warning("other"));
            other.Start();
            holder.Join();
            other.Join();
        }
        finally
        {
            OptunaLogging.RemoveHandler(first);
            OptunaLogging.RemoveHandler(second);
            OptunaLogging.EnableDefaultHandler();
            OptunaLogging.SetVerbosity(verbosity);
        }

        overlapped.Should().BeTrue("the first handler served another thread while the second held a record");
    }

    /// <summary>
    ///     Every shooter uploads a file of its own to the same trial at once through one file-system artifact store
    ///     (THREADING.md §10: uploads from concurrent trials do not interfere). Every upload gets its own artifact id and
    ///     file, the trial records all of them, and each downloads back to the bytes its shooter uploaded.
    /// </summary>
    [TestMethod]
    public void ArtifactUploads_ToOneTrialFromEveryThreadAtOnce_AreAllStoredAndRecorded()
    {
        const int rounds = 10;
        string root = Path.Combine(Path.GetTempPath(), $"optunasharp-gun-artifacts-{Guid.NewGuid():N}");
        Directory.CreateDirectory(root);
        try
        {
            // The store, like Optuna's, writes into an existing directory and creates none.
            string storePath = Directory.CreateDirectory(Path.Combine(root, "store")).FullName;
            var store = new OptunaSharp.Artifacts.FileSystemArtifactStore(storePath);
            var study = Optuna.CreateStudy(sampler: new RandomSampler(51));
            string[] sources = Enumerable.Range(0, Threads).Select(t => Path.Combine(root, $"source-{t}.txt")).ToArray();
            var ids = new string[Threads];
            Trial trial = null!;

            using var gun = new ConcurrencyGun(Threads);
            GunReport report = gun.Fire(rounds,
                shot => ids[shot.Thread] = OptunaSharp.Artifacts.ArtifactFunctions.UploadArtifact(store, sources[shot.Thread], trial),
                beforeRound: r =>
                {
                    trial = study.Ask();
                    for (int t = 0; t < Threads; t++)
                        File.WriteAllText(sources[t], $"shooter {t}, round {r}");
                },
                afterRound: r =>
                {
                    ids.Should().OnlyHaveUniqueItems("round {0}: every upload gets its own artifact id", r);
                    OptunaSharp.Artifacts.ArtifactFunctions.GetAllArtifactMeta(trial).Select(m => m.ArtifactId).Should().BeEquivalentTo(ids);
                    for (int t = 0; t < Threads; t++)
                    {
                        string copy = Path.Combine(root, $"download-{r}-{t}.txt");
                        OptunaSharp.Artifacts.ArtifactFunctions.DownloadArtifact(store, copy, ids[t]);
                        File.ReadAllText(copy).Should().Be($"shooter {t}, round {r}");
                    }
                    return false;
                });

            report.AssertClean();
        }
        finally
        {
            try
            {
                Directory.Delete(root, recursive: true);
            }
            catch (IOException)
            {
                // A file still held by a straggling handle is left for the OS temp cleanup.
            }
        }
    }

    /// <summary>
    ///     Every shooter invokes one <see cref="OptunaSharp.Terminator.TerminatorCallback" /> at once, twenty times a round, as the
    ///     callbacks of concurrent <c>nJobs</c> trials do. The callback runs its terminator one call at a time (THREADING.md
    ///     §6.6), because the improvement evaluators keep a random generator across calls.
    /// </summary>
    [TestMethod]
    public void TerminatorCallback_InvokedByEveryThreadAtOnce_RunsItsTerminatorOneCallAtATime()
    {
        const int rounds = 30, invocations = 20;
        var terminator = new OverlapProbeTerminator();
        var callback = new OptunaSharp.Terminator.TerminatorCallback(terminator);
        var study = Optuna.CreateStudy(sampler: new RandomSampler(52));
        study.Optimize(t => t.SuggestFloat("x", 0, 1), nTrials: 1);
        FrozenTrial last = study.Trials[0];

        using var quiet = new QuietLogs();
        using var gun = new ConcurrencyGun(Threads);
        GunReport report = gun.Fire(rounds, _ =>
        {
            for (int k = 0; k < invocations; k++)
                callback.Invoke(study, last);
        });

        report.AssertClean();
        terminator.Calls.Should().Be(Threads * rounds * invocations);
        terminator.PeakOverlap.Should().Be(1, "the callback admits one terminator call at a time");
    }

    /// <summary>
    ///     Half the shooters subscribe to and unsubscribe from the warning hook over and over while the other half raise
    ///     warnings. No warning raise fails on a concurrent subscription change, and a subscriber present throughout sees
    ///     every warning.
    /// </summary>
    [TestMethod]
    public void WarningHook_SubscribedAndUnsubscribedWhileOthersWarn_DeliversEveryWarning()
    {
        const int rounds = 30, operations = 50;
        int seen = 0;
        Action<string, OptunaSharp.Internal.WarningCategory> steady = (_, _) => Interlocked.Increment(ref seen);

        // The warnings also go to the log, which would print every one of them.
        using var quiet = new QuietLogs();
        OptunaSharp.Internal.OptunaWarnings.WarningRaised += steady;
        try
        {
            using var gun = new ConcurrencyGun(Threads);
            GunReport report = gun.Fire(rounds, shot =>
            {
                if (shot.Thread % 2 == 0)
                {
                    Action<string, OptunaSharp.Internal.WarningCategory> mine = (_, _) => { };
                    for (int k = 0; k < operations; k++)
                    {
                        OptunaSharp.Internal.OptunaWarnings.WarningRaised += mine;
                        OptunaSharp.Internal.OptunaWarnings.WarningRaised -= mine;
                    }
                }
                else
                {
                    for (int k = 0; k < operations; k++)
                        OptunaSharp.Internal.OptunaWarnings.Warn($"gun warning {shot.Thread}/{k}");
                }
            });

            report.AssertClean();
            seen.Should().Be(Threads / 2 * rounds * operations);
        }
        finally
        {
            OptunaSharp.Internal.OptunaWarnings.WarningRaised -= steady;
        }
    }

    /// <summary>
    ///     Half the shooters ask, suggest and tell while the other half read the whole study in every way an application
    ///     does: its trials, its best trial, its trials table and the storage's study summaries. No read fails, every
    ///     snapshot of the trials is numbered without a gap, and no reader ever sees fewer trials than it saw before.
    /// </summary>
    [TestMethod]
    public void WholeStudyReaders_WhileOthersWrite_NeverFail_AndSeeAGrowingContiguousStudy()
    {
        const int rounds = 20, trialsPerShot = 4, readsPerShot = 10;
        var storage = new InMemoryStorage();
        var study = Optuna.CreateStudy(storage: storage, sampler: new RandomSampler(53));
        // A best trial exists before the first reader asks for one.
        study.Optimize(t => t.SuggestFloat("x", 0, 1), nTrials: 3);
        int inconsistent = 0;

        using var quiet = new QuietLogs();
        using var gun = new ConcurrencyGun(Threads);
        GunReport report = gun.Fire(rounds, shot =>
        {
            if (shot.Thread % 2 == 0)
            {
                for (int k = 0; k < trialsPerShot; k++)
                {
                    Trial trial = study.Ask();
                    study.Tell(trial, trial.SuggestFloat("x", 0, 1));
                }
                return;
            }
            int lastCount = 0;
            for (int k = 0; k < readsPerShot; k++)
            {
                var trials = study.Trials;
                bool contiguous = trials.Select(t => t.Number).SequenceEqual(Enumerable.Range(0, trials.Count));
                bool bestIsComplete = study.BestTrial.State == TrialState.Complete;
                int rows = study.TrialsDataframe().RowCount;
                int summarized = Optuna.GetAllStudySummaries(storage).Single().NTrials;
                if (!contiguous || !bestIsComplete || trials.Count < lastCount || rows < trials.Count || summarized < rows)
                    Interlocked.Increment(ref inconsistent);
                lastCount = trials.Count;
            }
        });

        report.AssertClean();
        inconsistent.Should().Be(0, "every read is a consistent snapshot of a study that only grows");
        study.Trials.Should().HaveCount(3 + Threads / 2 * rounds * trialsPerShot);
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

    /// <summary>
    ///     A terminator that never terminates and records the most calls that were ever inside it at once, which the
    ///     <see cref="OptunaSharp.Terminator.TerminatorCallback" />'s lock keeps at 1.
    /// </summary>
    private sealed class OverlapProbeTerminator : OptunaSharp.Terminator.BaseTerminator
    {
        private int _inside;
        private int _peakOverlap;
        private int _calls;

        /// <summary>The most calls that were inside the terminator at one instant.</summary>
        internal int PeakOverlap => Volatile.Read(ref _peakOverlap);

        /// <summary>How many times the terminator was asked.</summary>
        internal int Calls => Volatile.Read(ref _calls);

        /// <summary>Records the call and answers that the study should go on.</summary>
        /// <param name="study">The study; unused.</param>
        /// <returns>Always false, so the callback never stops the study.</returns>
        public override bool ShouldTerminate(Study study)
        {
            Watermark.Raise(ref _peakOverlap, Interlocked.Increment(ref _inside));
            try
            {
                Interlocked.Increment(ref _calls);
                // Stay inside a little, so a second call let in would be seen.
                Thread.SpinWait(300);
                return false;
            }
            finally
            {
                Interlocked.Decrement(ref _inside);
            }
        }
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

    /// <summary>
    ///     A handler that counts its records and the most emits that were ever inside it at once, staying inside a little
    ///     so a second emit admitted at the same time would be seen.
    /// </summary>
    private sealed class OverlapLogHandler : ILogHandler
    {
        private int _count;
        private int _inside;
        private int _peakInside;

        /// <summary>How many records the handler received.</summary>
        internal int Count => Volatile.Read(ref _count);

        /// <summary>The most emits inside the handler at one instant; 1 when OptunaSharp's handler lock held.</summary>
        internal int PeakInside => Volatile.Read(ref _peakInside);

        /// <summary>Counts the record and the emits inside meanwhile.</summary>
        /// <param name="record">The record; only counted.</param>
        public void Emit(in LogRecord record)
        {
            Watermark.Raise(ref _peakInside, Interlocked.Increment(ref _inside));
            Thread.SpinWait(200);
            Interlocked.Increment(ref _count);
            Interlocked.Decrement(ref _inside);
        }
    }

    /// <summary>A handler that runs a test's delegate for each record.</summary>
    /// <param name="emit">What to do with each record, on the logging thread.</param>
    private sealed class DelegateLogHandler(Action<LogRecord> emit) : ILogHandler
    {
        /// <summary>Runs the delegate.</summary>
        /// <param name="record">The record handed to the delegate.</param>
        public void Emit(in LogRecord record) => emit(record);
    }
}
