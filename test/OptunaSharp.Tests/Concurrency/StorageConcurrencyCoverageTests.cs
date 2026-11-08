using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading;
using AwesomeAssertions;
using OptunaSharp.Samplers;
using OptunaSharp.Storages;
using OptunaSharp.Storages.Journal;

namespace OptunaSharp.Tests.Concurrency;

/// <summary>
///     The storage guarantees under writers and readers that hit the same storage at the same instant (THREADING.md
///     §6.1, §7.1), each on an in-memory storage and on a journal storage over a file. A record read without a deep copy
///     never changes under its reader, because every update replaces the record. Every write of many concurrent writers
///     lands, and a fresh replay of the journal sees exactly what the live storage holds. The best trial is the best of
///     every concurrently told trial. Study attributes set at once are all kept.
/// </summary>
[TestClass]
public class StorageConcurrencyCoverageTests
{
    /// <summary>An <see cref="InMemoryStorage" /> shared by every shooter.</summary>
    private const string Memory = "memory";

    /// <summary>A <see cref="JournalStorage" /> over a file, shared by every shooter.</summary>
    private const string Journal = "journal";

    private const int Threads = ConcurrencyGun.DefaultThreads;

    // A fresh journal per test, so the shooters of one test never see another test's records.
    private string _path = null!;

    /// <summary>Picks a unique journal path in the temp directory; the backend creates the file on first use.</summary>
    [TestInitialize]
    public void SetUp()
        => _path = Path.Combine(Path.GetTempPath(), $"optunasharp-gun-storage-{Guid.NewGuid():N}.log");

    /// <summary>
    ///     Deletes the journal and everything derived from its name. A file still held by a straggling handle is left for
    ///     the OS temp cleanup instead of failing the test.
    /// </summary>
    [TestCleanup]
    public void TearDown()
    {
        foreach (string file in Directory.GetFiles(Path.GetTempPath(), Path.GetFileName(_path) + "*"))
        {
            try
            {
                File.Delete(file);
            }
            catch (IOException)
            {
            }
        }
    }

    /// <summary>
    ///     Shooter 0 reads every trial without a deep copy, fingerprints each record, waits a moment, and fingerprints the
    ///     same record objects again, over and over. Meanwhile every other shooter writes its own trial (attributes,
    ///     reports, parameters). The two fingerprints always match: the storage replaces a record on every update instead
    ///     of changing it, so a record a reader holds is never changed under it, as Optuna documents for
    ///     <c>deepcopy=False</c>.
    /// </summary>
    /// <param name="setup"><c>memory</c> or <c>journal</c>.</param>
    [TestMethod]
    [DataRow(Memory)]
    [DataRow(Journal)]
    public void RecordReadWithoutADeepCopy_NeverChangesUnderItsReader_WhileOthersWriteIt(string setup)
    {
        BaseStorage storage = OpenStorage(setup);
        Study study = Optuna.CreateStudy(storage: storage, sampler: new RandomSampler(21));
        // A journal append waits for the disk, so the journal setup writes far less.
        int rounds = setup == Memory ? 30 : 2, writes = setup == Memory ? 30 : 3;
        var trials = new Trial[Threads];
        int changedUnderReader = 0, reads = 0, firstNumberOfRound = 0;

        using var quiet = new QuietLogs();
        using var gun = new ConcurrencyGun(Threads);
        GunReport report = gun.Fire(rounds,
            shot =>
            {
                if (shot.Thread == 0)
                {
                    // Reads while the writers write; each pass fingerprints this round's records twice, with writes landing
                    // between. Older records are no longer written, so checking them would only cost time.
                    for (int pass = 0; pass < writes * 4; pass++)
                    {
                        List<FrozenTrial> records = storage.GetAllTrials(study.StudyId, deepcopy: false)
                            .Where(r => r.Number >= firstNumberOfRound)
                            .ToList();
                        string[] before = records.Select(Fingerprint).ToArray();
                        Thread.SpinWait(2_000);
                        string[] after = records.Select(Fingerprint).ToArray();
                        if (!before.SequenceEqual(after))
                            Interlocked.Increment(ref changedUnderReader);
                        Interlocked.Increment(ref reads);
                    }
                    return;
                }
                Trial trial = trials[shot.Thread];
                for (int k = 0; k < writes; k++)
                {
                    trial.SetUserAttr($"a{k}", k);
                    trial.Report(k, step: k);
                    trial.SuggestFloat($"p{k}", 0, 1);
                }
            },
            beforeRound: _ =>
            {
                for (int t = 1; t < Threads; t++)
                    trials[t] = study.Ask();
                firstNumberOfRound = trials[1].Number;
            });

        report.AssertClean();
        reads.Should().BeGreaterThan(0);
        changedUnderReader.Should().Be(0, "a record read without a deep copy is never changed under its reader");
    }

    /// <summary>
    ///     Every shooter writes a trial of its own at once (user and system attributes, reports, parameters) and then tells
    ///     it. Every write lands, and for a journal a storage that replays the file from scratch sees exactly what the live
    ///     storage holds.
    /// </summary>
    /// <param name="setup"><c>memory</c> or <c>journal</c>.</param>
    [TestMethod]
    [DataRow(Memory)]
    [DataRow(Journal)]
    public void WritersOfTheirOwnTrials_AtOnce_LandEveryWrite_AndAFreshReplayAgrees(string setup)
    {
        BaseStorage storage = OpenStorage(setup);
        Study study = Optuna.CreateStudy(storage: storage, studyName: "writers", sampler: new RandomSampler(22));
        // A journal append waits for the disk, so the journal setup writes far less.
        int rounds = setup == Memory ? 20 : 2, writes = setup == Memory ? 5 : 2;
        var trials = new Trial[Threads];

        using var quiet = new QuietLogs();
        using var gun = new ConcurrencyGun(Threads);
        GunReport report = gun.Fire(rounds,
            shot =>
            {
                Trial trial = trials[shot.Thread];
                for (int k = 0; k < writes; k++)
                {
                    trial.SetUserAttr($"u{k}", $"{shot.Thread}/{shot.Round}/{k}");
                    storage.SetTrialSystemAttr(trial.TrialId, $"s{k}", k * 10 + shot.Thread);
                    trial.Report(shot.Thread + k, step: k);
                    trial.SuggestInt($"p{k}", 0, 1000);
                }
                study.Tell(trial, shot.Thread);
            },
            beforeRound: _ =>
            {
                for (int t = 0; t < Threads; t++)
                    trials[t] = study.Ask();
            });

        report.AssertClean();
        List<FrozenTrial> live = storage.GetAllTrials(study.StudyId);
        live.Should().HaveCount(Threads * rounds).And.OnlyContain(t => t.State == TrialState.Complete);
        foreach (FrozenTrial trial in live)
        {
            trial.UserAttrs.Should().HaveCount(writes);
            trial.SystemAttrs.Keys.Count(k => k.StartsWith('s')).Should().Be(writes);
            trial.IntermediateValues.Should().HaveCount(writes);
            trial.Params.Should().HaveCount(writes);
        }

        if (setup == Journal)
        {
            var fresh = new JournalStorage(new JournalFileBackend(_path));
            List<FrozenTrial> replayed = fresh.GetAllTrials(fresh.GetStudyIdFromName("writers"));
            replayed.Select(Fingerprint).Should().Equal(live.Select(Fingerprint), "a fresh replay of the journal sees what the live storage holds");
        }
    }

    /// <summary>
    ///     Every shooter tells a trial at once, each with a value of its own; after every round the study's best trial is
    ///     the best of all trials told so far, for a minimized and a maximized study. The storage updates its best-trial
    ///     cache under its lock, so no concurrent tell can overwrite a better best.
    /// </summary>
    /// <param name="setup"><c>memory</c> or <c>journal</c>.</param>
    /// <param name="direction">The study's direction.</param>
    [TestMethod]
    [DataRow(Memory, StudyDirection.Minimize)]
    [DataRow(Memory, StudyDirection.Maximize)]
    [DataRow(Journal, StudyDirection.Minimize)]
    public void TellsAtOnce_KeepTheBestTrialTheBestOfAll(string setup, StudyDirection direction)
    {
        BaseStorage storage = OpenStorage(setup);
        Study study = Optuna.CreateStudy(storage: storage, sampler: new RandomSampler(23), direction: direction);
        int rounds = setup == Memory ? 40 : 4;
        var trials = new Trial[Threads];
        var told = new List<double>();
        var values = new double[Threads];

        using var quiet = new QuietLogs();
        using var gun = new ConcurrencyGun(Threads);
        GunReport report = gun.Fire(rounds,
            shot =>
            {
                // A value per shooter and round that is unique, and whose best is reached by a random shooter each round.
                values[shot.Thread] = ((shot.Thread * 7919 + shot.Round * 104_729) % 1000) + shot.Thread / 100.0;
                study.Tell(trials[shot.Thread], values[shot.Thread]);
            },
            beforeRound: _ =>
            {
                for (int t = 0; t < Threads; t++)
                    trials[t] = study.Ask();
            },
            afterRound: r =>
            {
                told.AddRange(values);
                double best = direction == StudyDirection.Minimize ? told.Min() : told.Max();
                study.BestValue.Should().Be(best, "round {0}: the best trial is the best of every trial told", r);
                study.BestTrial.Value.Should().Be(best);
                return false;
            });

        report.AssertClean();
    }

    /// <summary>
    ///     Every shooter sets study user attributes of its own at once: every key of every round is kept. The storage
    ///     replaces the attribute dictionary under its lock, so two concurrent writers never both start from the old
    ///     dictionary and lose one key.
    /// </summary>
    /// <param name="setup"><c>memory</c> or <c>journal</c>.</param>
    [TestMethod]
    [DataRow(Memory)]
    [DataRow(Journal)]
    public void StudyUserAttrs_SetAtOnce_KeepEveryKey(string setup)
    {
        BaseStorage storage = OpenStorage(setup);
        Study study = Optuna.CreateStudy(storage: storage, sampler: new RandomSampler(24));
        int rounds = setup == Memory ? 50 : 3;
        const int keys = 4;

        using var gun = new ConcurrencyGun(Threads);
        GunReport report = gun.Fire(rounds, shot =>
        {
            for (int k = 0; k < keys; k++)
                study.SetUserAttr($"t{shot.Thread}-r{shot.Round}-k{k}", shot.Thread);
        });

        report.AssertClean();
        study.UserAttrs.Should().HaveCount(Threads * rounds * keys, "no concurrent writer loses another's key");
    }

    /// <summary>Opens the storage a test runs on.</summary>
    /// <param name="setup"><c>memory</c> or <c>journal</c>.</param>
    /// <returns>A new storage; the journal one writes this test's file.</returns>
    /// <exception cref="ArgumentOutOfRangeException">An unknown setup name.</exception>
    private BaseStorage OpenStorage(string setup) => setup switch
    {
        Memory => new InMemoryStorage(),
        Journal => new JournalStorage(new JournalFileBackend(_path)),
        _ => throw new ArgumentOutOfRangeException(nameof(setup), setup, "Unknown setup."),
    };

    /// <summary>
    ///     A text that changes whenever any part of a record that a write can touch changes: state, values, parameters,
    ///     distributions, attributes and reports, in their stored order.
    /// </summary>
    /// <param name="trial">The record.</param>
    /// <returns>The fingerprint.</returns>
    private static string Fingerprint(FrozenTrial trial)
        => string.Join("|",
            trial.Number,
            trial.State,
            trial.Values is null ? "-" : string.Join(",", trial.Values),
            string.Join(",", trial.Params.Select(kv => $"{kv.Key}={kv.Value}")),
            string.Join(",", trial.Distributions.Keys),
            string.Join(",", trial.UserAttrs.Select(kv => $"{kv.Key}={kv.Value}")),
            string.Join(",", trial.SystemAttrs.Select(kv => $"{kv.Key}={kv.Value}")),
            string.Join(",", trial.IntermediateValues.Select(kv => $"{kv.Key}={kv.Value}")));
}
