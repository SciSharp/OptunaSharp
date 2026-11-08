using System;
using System.IO;
using System.Linq;
using AwesomeAssertions;
using OptunaSharp.Collections;
using OptunaSharp.Samplers;
using OptunaSharp.Storages;
using OptunaSharp.Storages.Journal;

namespace OptunaSharp.Tests.Concurrency;

/// <summary>
///     The guarantees of one study used by many threads at the same instant (THREADING.md §6.1, §7.1), each round fired by
///     a <see cref="ConcurrencyGun" />. Every test runs on three setups: one in-memory storage; one journal storage shared
///     by every thread; and one journal storage per thread over the same file, which is how separate processes share a
///     study. Simultaneous asks get distinct trials numbered without a gap. An enqueued trial goes to exactly one asking
///     thread, and so does a waiting trial claimed through the storage. Simultaneous <c>CreateStudy(loadIfExists: true)</c>
///     calls create one study and all load it.
/// </summary>
/// <remarks>
///     The journal file syncs every append to disk, so the journal setups fire fewer rounds than the in-memory one. The
///     per-thread setup also gives every thread its own unseeded sampler, as separate processes would have.
/// </remarks>
[TestClass]
public class StudyConcurrencyCoverageTests
{
    /// <summary>One <see cref="InMemoryStorage" /> and one <see cref="Study" /> object shared by every shooter.</summary>
    private const string Memory = "memory";

    /// <summary>One <see cref="JournalStorage" /> over a file, and one <see cref="Study" /> object, shared by every shooter.</summary>
    private const string Journal = "journal";

    /// <summary>A <see cref="JournalStorage" /> and a <see cref="Study" /> object per shooter, all over the same file.</summary>
    private const string JournalPerThread = "journal-per-thread";

    private const int Threads = ConcurrencyGun.DefaultThreads;

    // A fresh journal per test, so concurrent workers of one test never see another test's records.
    private string _path = null!;

    /// <summary>Picks a unique journal path in the temp directory; the backend creates the file on first use.</summary>
    [TestInitialize]
    public void SetUp()
        => _path = Path.Combine(Path.GetTempPath(), $"optunasharp-gun-{Guid.NewGuid():N}.log");

    /// <summary>
    ///     Deletes the journal and everything derived from its name (the lock, renamed locks, the symlink probe). A file
    ///     still held by a straggling handle is left for the OS temp cleanup instead of failing the test.
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
    ///     Every shooter asks for a trial at once: the round's trials are distinct and take the next numbers without a gap,
    ///     and a fresh read of the storage sees every trial of the volley numbered from 0.
    /// </summary>
    /// <param name="setup">Where the study lives: <c>memory</c>, <c>journal</c> or <c>journal-per-thread</c>.</param>
    [TestMethod]
    [DataRow(Memory)]
    [DataRow(Journal)]
    [DataRow(JournalPerThread)]
    public void Ask_FromEveryThreadAtOnce_GivesEachADistinctTrial_NumberedWithoutAGap(string setup)
    {
        Bench bench = Open(setup);
        var numbers = new int[Threads];
        var ids = new int[Threads];

        using var quiet = new QuietLogs();
        using var gun = new ConcurrencyGun(Threads);
        GunReport report = gun.Fire(bench.Rounds,
            shot =>
            {
                Trial trial = bench.Studies[shot.Thread].Ask();
                numbers[shot.Thread] = trial.Number;
                ids[shot.Thread] = trial.TrialId;
            },
            afterRound: r =>
            {
                numbers.Order().Should().Equal(Enumerable.Range(r * Threads, Threads), "round {0}'s asks take the next {1} numbers", r, Threads);
                ids.Should().OnlyHaveUniqueItems();
                return false;
            });

        report.AssertClean();
        bench.FreshMainStudy().Trials.Select(t => t.Number).Should().Equal(Enumerable.Range(0, Threads * bench.Rounds));
    }

    /// <summary>
    ///     A trial is enqueued before each round and every shooter asks at once: exactly one shooter gets the enqueued
    ///     trial (its fixed value) and the others get new trials. No waiting trial is left, and every enqueued value ran
    ///     exactly once.
    /// </summary>
    /// <param name="setup">Where the study lives: <c>memory</c>, <c>journal</c> or <c>journal-per-thread</c>.</param>
    [TestMethod]
    [DataRow(Memory)]
    [DataRow(Journal)]
    [DataRow(JournalPerThread)]
    public void Ask_WithOneEnqueuedTrial_FromEveryThreadAtOnce_HandsItToExactlyOneThread(string setup)
    {
        Bench bench = Open(setup);
        var xs = new double[Threads];
        var ids = new int[Threads];

        using var quiet = new QuietLogs();
        using var gun = new ConcurrencyGun(Threads);
        GunReport report = gun.Fire(bench.Rounds,
            shot =>
            {
                Study study = bench.Studies[shot.Thread];
                Trial trial = study.Ask();
                xs[shot.Thread] = trial.SuggestFloat("x", 0, 1e6);
                ids[shot.Thread] = trial.TrialId;
                study.Tell(trial, xs[shot.Thread]);
            },
            beforeRound: r => bench.Main.EnqueueTrial(new OrderedDict<string, object?> { ["x"] = EnqueuedX(r) }),
            afterRound: r =>
            {
                xs.Count(x => x == EnqueuedX(r)).Should().Be(1, "round {0}'s enqueued trial goes to exactly one asking thread", r);
                ids.Should().OnlyHaveUniqueItems();
                return false;
            });

        report.AssertClean();
        var trials = bench.FreshMainStudy().Trials;
        trials.Should().OnlyContain(t => t.State == TrialState.Complete, "no enqueued trial is left waiting");
        Enumerable.Range(0, bench.Rounds).Should()
            .OnlyContain(r => trials.Count(t => Convert.ToDouble(t.Params["x"]) == EnqueuedX(r)) == 1);
    }

    /// <summary>
    ///     Every shooter claims the same waiting trial through its storage at once, as <c>Study.Ask</c> does when it pops
    ///     an enqueued trial (<c>SetTrialStateValues(RUNNING)</c>): exactly one claim succeeds. The storage arbitrates, by
    ///     its lock in memory and by the journal's order across journal storages.
    /// </summary>
    /// <param name="setup">Where the study lives: <c>memory</c>, <c>journal</c> or <c>journal-per-thread</c>.</param>
    [TestMethod]
    [DataRow(Memory)]
    [DataRow(Journal)]
    [DataRow(JournalPerThread)]
    public void ClaimingOneWaitingTrial_FromEveryThreadAtOnce_HasExactlyOneWinner(string setup)
    {
        Bench bench = Open(setup);
        var claimed = new bool[Threads];
        int waitingId = -1;

        using var quiet = new QuietLogs();
        using var gun = new ConcurrencyGun(Threads);
        GunReport report = gun.Fire(bench.Rounds,
            shot => claimed[shot.Thread] = TryClaim(bench.Storages[shot.Thread], waitingId),
            beforeRound: r =>
            {
                bench.Main.EnqueueTrial(new OrderedDict<string, object?> { ["x"] = EnqueuedX(r) });
                waitingId = bench.Storages[0]
                    .GetAllTrials(bench.Main.StudyId, deepcopy: false, states: new[] { TrialState.Waiting })
                    .Single().TrialId;
            },
            afterRound: r =>
            {
                claimed.Count(c => c).Should().Be(1, "round {0}: one claim of a waiting trial succeeds", r);
                bench.Main.Tell(bench.Storages[0].GetTrial(waitingId).Number, 0.0);
                return false;
            });

        report.AssertClean();
    }

    /// <summary>
    ///     Every shooter creates the same study name at once with <c>loadIfExists: true</c>: the storage creates the study
    ///     once, the other calls load it, and every shooter ends up with the same study.
    /// </summary>
    /// <param name="setup">Where the studies live: <c>memory</c>, <c>journal</c> or <c>journal-per-thread</c>.</param>
    [TestMethod]
    [DataRow(Memory)]
    [DataRow(Journal)]
    [DataRow(JournalPerThread)]
    public void CreateStudy_LoadIfExists_FromEveryThreadAtOnce_CreatesTheStudyOnce(string setup)
    {
        Bench bench = Open(setup);
        var studyIds = new int[Threads];

        using var quiet = new QuietLogs();
        using var gun = new ConcurrencyGun(Threads);
        GunReport report = gun.Fire(bench.Rounds,
            shot => studyIds[shot.Thread] = Optuna.CreateStudy(
                storage: bench.Storages[shot.Thread], studyName: $"race-{shot.Round}", loadIfExists: true).StudyId,
            afterRound: r =>
            {
                studyIds.Distinct().Should().ContainSingle("round {0}: every shooter has the same study", r);
                bench.Storages[0].GetAllStudies().Count(s => s.StudyName == $"race-{r}").Should().Be(1);
                return false;
            });

        report.AssertClean();
    }

    /// <summary>The fixed value of the trial enqueued before round <paramref name="round" />; no sampler draws it from [0, 1e6).</summary>
    /// <param name="round">The round number.</param>
    /// <returns>A value per round, never a random draw.</returns>
    private static double EnqueuedX(int round) => 1000.5 + round;

    /// <summary>
    ///     Claims a waiting trial as <c>Study</c>'s pop does: through <c>SetTrialStateValues(RUNNING)</c>, treating a trial
    ///     already finished as lost.
    /// </summary>
    /// <param name="storage">The storage to claim through.</param>
    /// <param name="trialId">The waiting trial.</param>
    /// <returns>Whether this claim won the trial.</returns>
    private static bool TryClaim(BaseStorage storage, int trialId)
    {
        try
        {
            return storage.SetTrialStateValues(trialId, TrialState.Running);
        }
        catch (UpdateFinishedTrialError)
        {
            return false;
        }
    }

    /// <summary>
    ///     Opens the setup a test runs on: its storages and study objects, indexed by shooter, and how many rounds to fire.
    /// </summary>
    /// <param name="setup">One of <c>memory</c>, <c>journal</c>, <c>journal-per-thread</c>.</param>
    /// <returns>The setup; its study is named <c>gun</c>.</returns>
    /// <exception cref="ArgumentOutOfRangeException">An unknown setup name.</exception>
    private Bench Open(string setup)
    {
        const string name = "gun";
        switch (setup)
        {
            case Memory:
            {
                var storage = new InMemoryStorage();
                Study study = Optuna.CreateStudy(storage: storage, studyName: name, sampler: new RandomSampler(7));
                return new Bench(Shared<BaseStorage>(storage), Shared(study), rounds: 60, fresh: () => study);
            }
            case Journal:
            {
                var storage = new JournalStorage(new JournalFileBackend(_path));
                Study study = Optuna.CreateStudy(storage: storage, studyName: name, sampler: new RandomSampler(7));
                return new Bench(Shared<BaseStorage>(storage), Shared(study), rounds: 6, fresh: FreshJournalStudy);
            }
            case JournalPerThread:
            {
                Optuna.CreateStudy(storage: new JournalStorage(new JournalFileBackend(_path)), studyName: name);
                var storages = Enumerable.Range(0, Threads)
                    .Select(_ => (BaseStorage)new JournalStorage(new JournalFileBackend(_path)))
                    .ToArray();
                Study[] studies = storages.Select(s => Optuna.LoadStudy(name, s, sampler: new RandomSampler())).ToArray();
                return new Bench(storages, studies, rounds: 6, fresh: FreshJournalStudy);
            }
            default:
                throw new ArgumentOutOfRangeException(nameof(setup), setup, "Unknown setup.");
        }

        // A storage that has never seen the volley: it replays the journal from the file, as a new process would.
        Study FreshJournalStudy() => Optuna.LoadStudy(name, new JournalStorage(new JournalFileBackend(_path)));
    }

    /// <summary>The same object for every shooter.</summary>
    /// <typeparam name="T">The object's type.</typeparam>
    /// <param name="shared">The object.</param>
    /// <returns>An array of <see cref="Threads" /> references to it.</returns>
    private static T[] Shared<T>(T shared) => Enumerable.Repeat(shared, Threads).ToArray();

    /// <summary>A test's storages and study objects, indexed by shooter, and the rounds the setup can afford.</summary>
    /// <param name="storages">Each shooter's storage; the same object for every shooter in the shared setups.</param>
    /// <param name="studies">Each shooter's study object.</param>
    /// <param name="rounds">How many rounds to fire: fewer for a journal, whose appends wait for the disk.</param>
    /// <param name="fresh">Opens the study through a storage that has not been used yet, to check what the volley stored.</param>
    private sealed class Bench(BaseStorage[] storages, Study[] studies, int rounds, Func<Study> fresh)
    {
        /// <summary>Each shooter's storage.</summary>
        internal BaseStorage[] Storages { get; } = storages;

        /// <summary>Each shooter's study object.</summary>
        internal Study[] Studies { get; } = studies;

        /// <summary>Shooter 0's study, through which the callbacks set up and finish rounds.</summary>
        internal Study Main => Studies[0];

        /// <summary>How many rounds to fire.</summary>
        internal int Rounds { get; } = rounds;

        /// <summary>The study as a fresh storage reads it: for journals, a new replay of the file.</summary>
        /// <returns>The study.</returns>
        internal Study FreshMainStudy() => fresh();
    }
}
