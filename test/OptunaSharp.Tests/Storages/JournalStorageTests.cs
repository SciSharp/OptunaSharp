using System;
using System.IO;
using System.Linq;
using AwesomeAssertions;
using NumSharp;
using OptunaSharp.Collections;
using OptunaSharp.Distributions;
using OptunaSharp.Samplers;
using OptunaSharp.Storages.Journal;

namespace OptunaSharp.Tests.Storages;

/// <summary>
///     Journal-specific behavior beyond the conformance suite: the on-disk JSON-lines format,
///     replay-from-scratch equivalence, multi-instance synchronization through the shared file,
///     partial-line tolerance, and the file locks.
/// </summary>
[TestClass]
public class JournalStorageTests
{
    private string _path = null!;

    [TestInitialize]
    public void SetUp()
        => _path = Path.Combine(Path.GetTempPath(), $"optunasharp-journal-{Guid.NewGuid():N}.log");

    [TestCleanup]
    public void TearDown()
    {
        foreach (var f in Directory.GetFiles(Path.GetTempPath(), Path.GetFileName(_path) + "*"))
        {
            try
            {
                File.Delete(f);
            }
            catch (IOException)
            {
            }
        }
    }

    private JournalStorage NewStorage()
        => new(new JournalFileBackend(_path, lockObj: new JournalFileOpenLock(_path)));

    [TestMethod]
    public void File_Format_Is_Compact_JsonLines_With_OpCode_And_WorkerId()
    {
        var storage = NewStorage();
        storage.CreateNewStudy(new[] { StudyDirection.Minimize }, "fmt");

        var lines = File.ReadAllLines(_path);
        lines.Should().HaveCount(1);
        lines[0].Should().StartWith("{\"op_code\":0,\"worker_id\":\"");
        lines[0].Should().EndWith("\"study_name\":\"fmt\",\"directions\":[1]}");
        lines[0].Should().NotContain(": ", "the journal uses compact separators like Python's (',', ':')");
    }

    [TestMethod]
    public void Fresh_Instance_Replays_To_Identical_State()
    {
        var writer = NewStorage();
        int studyId = writer.CreateNewStudy(new[] { StudyDirection.Maximize }, "replay");
        int trialId = writer.CreateNewTrial(studyId);
        writer.SetTrialParam(trialId, "x", 1.25, new FloatDistribution(0, 10));
        writer.SetTrialParam(trialId, "c", 2.0,
            new CategoricalDistribution(new CategoricalChoice[] { "a", "b", "c" }));
        writer.SetTrialIntermediateValue(trialId, 3, double.NaN);
        writer.SetTrialUserAttr(trialId, "note", "héllo");
        writer.SetTrialStateValues(trialId, TrialState.Complete, new[] { 0.5 });
        writer.SetStudyUserAttr(studyId, "tag", 7L);

        var reader = NewStorage();
        int readStudyId = reader.GetStudyIdFromName("replay");
        readStudyId.Should().Be(studyId);
        reader.GetStudyDirections(readStudyId).Should().Equal(StudyDirection.Maximize);
        reader.GetStudyUserAttrs(readStudyId)["tag"].Should().Be(7L);

        var trial = reader.GetAllTrials(readStudyId).Single();
        trial.Params["x"].Should().Be(1.25);
        trial.Params["c"].Should().Be("c");
        double.IsNaN(trial.IntermediateValues[3]).Should().BeTrue("NaN survives the journal via Python's NaN literal");
        trial.UserAttrs["note"].Should().Be("héllo");
        trial.State.Should().Be(TrialState.Complete);
        trial.Value.Should().Be(0.5);
        trial.DatetimeStart.Should().NotBeNull();
        trial.DatetimeComplete.Should().NotBeNull();
    }

    [TestMethod]
    public void Two_Instances_Sync_Through_The_Shared_File()
    {
        var a = NewStorage();
        var b = NewStorage();

        int studyId = a.CreateNewStudy(new[] { StudyDirection.Minimize }, "shared");
        // b sees the study a created.
        b.GetStudyIdFromName("shared").Should().Be(studyId);

        int trialFromB = b.CreateNewTrial(studyId);
        // a sees b's trial on its next read.
        a.GetAllTrials(studyId).Should().ContainSingle(t => t.TrialId == trialFromB);

        // Duplicate study name from the other instance raises for its own worker.
        var act = () => b.CreateNewStudy(new[] { StudyDirection.Minimize }, "shared");
        act.Should().Throw<DuplicatedStudyError>();
    }

    [TestMethod]
    public void Trailing_Partial_Line_Is_Tolerated_Until_Completed()
    {
        var storage = NewStorage();
        int studyId = storage.CreateNewStudy(new[] { StudyDirection.Minimize }, "partial");

        // Simulate a concurrent writer that has flushed only half a line (no trailing \n).
        File.AppendAllText(_path, "{\"op_code\":4,\"worker_id\":\"other\",\"study_id\":0,\"datetime_start\":\"2026-01-01T0");

        // Reads still work; the partial line is simply not visible yet.
        storage.GetAllTrials(studyId).Should().BeEmpty();

        // Complete the line: the trial appears.
        File.AppendAllText(_path, "0:00:00.000000+00:00\"}\n");
        storage.GetAllTrials(studyId).Should().HaveCount(1);
    }

    [TestMethod]
    public void Naive_Datetimes_From_Older_Writers_Are_Accepted()
    {
        // Optuna 4.x wrote naive local datetimes (no offset); the replay treats them as local.
        File.WriteAllText(
            _path,
            "{\"op_code\":0,\"worker_id\":\"w\",\"study_name\":\"legacy\",\"directions\":[1]}\n" +
            "{\"op_code\":4,\"worker_id\":\"w\",\"study_id\":0,\"datetime_start\":\"2026-08-30T08:29:50.875841\"}\n");

        var storage = NewStorage();
        var trial = storage.GetAllTrials(storage.GetStudyIdFromName("legacy")).Single();
        trial.DatetimeStart.Should().Be(new DateTime(2026, 8, 30, 8, 29, 50, 875, 841));
    }

    /// <summary>
    ///     Review item P2-2 (confirmed end to end in pass 2B): a journal record nested 5,000 levels deep (an 11 KB
    ///     line) used to overflow the stack of every C# process that loaded the study, which .NET cannot catch. Now
    ///     the load fails with a catchable error carrying CPython's <c>RecursionError</c> message, which is also what
    ///     Python's journal replay raises on this file (its <c>json.loads</c> stops at 2,998 levels).
    /// </summary>
    [TestMethod]
    public void HostilelyNestedRecord_FailsTheLoadCatchably()
    {
        File.WriteAllText(
            _path,
            "{\"op_code\":0,\"worker_id\":\"w\",\"study_name\":\"deep\",\"directions\":[1]}\n" +
            "{\"op_code\":2,\"worker_id\":\"w\",\"study_id\":0,\"user_attr\":{\"k\":" +
            new string('[', 5_000) + new string(']', 5_000) + "}}\n");

        var load = () => NewStorage().GetAllStudies();

        load.Should().Throw<RuntimeError>()
            .WithMessage("maximum recursion depth exceeded while decoding a JSON array from a unicode string");
    }

    [TestMethod]
    public void Rustuna_DiscardTrials_Op_Is_Skipped()
    {
        File.WriteAllText(
            _path,
            "{\"op_code\":0,\"worker_id\":\"w\",\"study_name\":\"r\",\"directions\":[1]}\n" +
            "{\"op_code\":10,\"worker_id\":\"w\",\"study_id\":0}\n");
        var storage = NewStorage();
        storage.GetStudyIdFromName("r").Should().Be(0);
    }

    [TestMethod]
    public void Optimize_Over_Journal_Works_EndToEnd()
    {
        var storage = NewStorage();
        var study = Optuna.CreateStudy(storage: storage, sampler: new RandomSampler(seed: 11), studyName: "e2e");
        study.Optimize(t => Math.Pow(t.SuggestFloat("x", -5, 5), 2), nTrials: 6);

        var reloaded = Optuna.LoadStudy("e2e", NewStorage(), sampler: new RandomSampler(seed: 12));
        reloaded.Trials.Should().HaveCount(6);
        reloaded.BestValue.Should().Be(study.BestValue);
    }

    [TestMethod]
    public void OpenLock_Is_Exclusive_And_Releases()
    {
        var lockA = new JournalFileOpenLock(_path);
        var lockB = new JournalFileOpenLock(_path, gracePeriod: null);

        lockA.Acquire().Should().BeTrue();
        File.Exists(_path + ".lock").Should().BeTrue();

        // A second acquire would block; release then acquire from B.
        lockA.Release();
        lockB.Acquire().Should().BeTrue();
        lockB.Release();
        File.Exists(_path + ".lock").Should().BeFalse();

        var doubleRelease = () => lockB.Release();
        doubleRelease.Should().Throw<RuntimeError>().WithMessage("Error: did not possess lock");
    }

    [TestMethod]
    public void OpenLock_GracePeriod_Validation()
    {
        var act = () => new JournalFileOpenLock(_path, gracePeriod: 0);
        act.Should().Throw<ValueError>()
            .WithMessage("The value of `grace_period` should be a positive integer.");
    }

    // ------------------------------------------------------------------ default lock

    /// <summary>
    ///     Stands in for <see cref="JournalFileSymlinkLock" /> in a process that cannot create symbolic links.
    ///     By default it throws exactly what <c>File.CreateSymbolicLink</c> throws on Windows without Developer
    ///     Mode (ERROR_PRIVILEGE_NOT_HELD, HRESULT 0x80070522), which the real lock rethrows.
    /// </summary>
    /// <param name="failure">
    ///     The exception to throw instead, e.g. the <see cref="UnauthorizedAccessException" /> .NET raises for
    ///     EPERM on a Unix filesystem without symbolic links; <c>null</c> keeps the Windows failure.
    /// </param>
    private sealed class NoSymlinkLock(Exception? failure = null) : BaseJournalFileLock
    {
        /// <summary>How many times the default lock tried this protocol.</summary>
        public int AcquireCalls;

        /// <summary>Fails like a symbolic-link creation the process or filesystem does not allow.</summary>
        /// <returns>Never returns.</returns>
        /// <exception cref="IOException">Always, unless another failure was configured.</exception>
        public override bool Acquire()
        {
            AcquireCalls++;
            throw failure ?? new IOException(
                "A required privilege is not held by the client.", unchecked((int)0x80070522));
        }

        /// <summary>Never reached: this protocol never acquires.</summary>
        /// <exception cref="InvalidOperationException">Always.</exception>
        public override void Release() => throw new InvalidOperationException("The symlink protocol never acquired.");
    }

    // [Misaligned] Python's JournalFileBackend keeps the symlink lock and raises where symbolic links
    // cannot be created. OptunaSharp's default lock switches to the open-file protocol there instead. Both
    // protocols use the same lock path, so C# and Python workers still exclude each other. Where symlinks
    // work, the behavior is Optuna's.

    /// <summary>
    ///     Regression for the Windows default: without the symlink privilege, the first journal write threw
    ///     "A required privilege is not held by the client". The default lock must switch to the open-file
    ///     protocol, lock with a regular file at the same path, and never retry symbolic links.
    /// </summary>
    [TestMethod]
    [Misaligned]
    public void DefaultLock_SwitchesToOpenLock_WhenSymlinksCannotBeCreated()
    {
        var symlinkLock = new NoSymlinkLock();
        int openLocksBuilt = 0;
        var lockObj = new JournalFileDefaultLock(symlinkLock, () =>
        {
            openLocksBuilt++;
            return new JournalFileOpenLock(_path);
        });

        lockObj.Acquire().Should().BeTrue();
        lockObj.UsesOpenLock.Should().BeTrue();
        File.Exists(_path + ".lock").Should().BeTrue();
        new FileInfo(_path + ".lock").LinkTarget.Should().BeNull("the open-file protocol locks with a regular file");
        lockObj.Release();
        File.Exists(_path + ".lock").Should().BeFalse();

        lockObj.Acquire().Should().BeTrue();
        lockObj.Release();
        symlinkLock.AcquireCalls.Should().Be(1, "the switch is permanent for the instance");
        openLocksBuilt.Should().Be(1);
    }

    /// <summary>
    ///     The Unix shape of the same problem: symlink(2) returns EPERM on a filesystem without symbolic links
    ///     (FAT/exFAT), which .NET raises as <see cref="UnauthorizedAccessException" /> straight through the
    ///     symlink lock. The default lock switches for it too.
    /// </summary>
    [TestMethod]
    [Misaligned]
    public void DefaultLock_SwitchesToOpenLock_WhenTheFilesystemRejectsSymlinks()
    {
        var symlinkLock = new NoSymlinkLock(new UnauthorizedAccessException("Operation not permitted"));
        var lockObj = new JournalFileDefaultLock(symlinkLock, () => new JournalFileOpenLock(_path));

        lockObj.Acquire().Should().BeTrue();
        lockObj.UsesOpenLock.Should().BeTrue();
        lockObj.Release();
        File.Exists(_path + ".lock").Should().BeFalse();
    }

    /// <summary>
    ///     A journal that cannot be locked at all must still fail loudly: when the open-file protocol cannot
    ///     create its file either, its exception propagates instead of being swallowed by the switch.
    /// </summary>
    [TestMethod]
    public void DefaultLock_Propagates_WhenTheOpenLockCannotCreateItsFileEither()
    {
        string unreachable = Path.Combine(Path.GetTempPath(), $"optunasharp-missing-{Guid.NewGuid():N}", "journal.log");
        var lockObj = new JournalFileDefaultLock(new NoSymlinkLock(), () => new JournalFileOpenLock(unreachable));

        var act = () => lockObj.Acquire();
        act.Should().Throw<DirectoryNotFoundException>();
    }

    /// <summary>
    ///     Contention is not a reason to switch: while another process holds the lock (here as a regular
    ///     file, as an open-file or Python worker would), the default lock waits, and takes the lock once it
    ///     is released. Holds whichever protocol this machine supports.
    /// </summary>
    [TestMethod]
    public void DefaultLock_WaitsWhileAnotherProcessHoldsTheLock()
    {
        File.WriteAllText(_path + ".lock", "");
        var lockObj = new JournalFileDefaultLock(_path);

        var acquiring = System.Threading.Tasks.Task.Run(() => lockObj.Acquire());
        acquiring.Wait(TimeSpan.FromMilliseconds(300)).Should().BeFalse("a held lock must block the acquire");

        File.Delete(_path + ".lock");
        acquiring.Wait(TimeSpan.FromSeconds(10)).Should().BeTrue();
        acquiring.Result.Should().BeTrue();
        lockObj.Release();
        File.Exists(_path + ".lock").Should().BeFalse();
    }

    /// <summary>
    ///     A backend built without a lock uses the default lock, an explicit lock is used as given, and a
    ///     journal on the default lock works end to end on this machine, with no lock left behind.
    /// </summary>
    [TestMethod]
    public void Backend_WithoutLock_UsesTheDefaultLock_AndAnExplicitLockAsGiven()
    {
        new JournalFileBackend(_path).LockObj.Should().BeOfType<JournalFileDefaultLock>();
        var explicitLock = new JournalFileOpenLock(_path);
        new JournalFileBackend(_path, explicitLock).LockObj.Should().BeSameAs(explicitLock);

        var study = Optuna.CreateStudy(
            storage: new JournalStorage(new JournalFileBackend(_path)), sampler: new RandomSampler(seed: 3));
        study.Optimize(t => t.SuggestFloat("x", -1, 1), nTrials: 3);
        study.Trials.Should().HaveCount(3);
        File.Exists(_path + ".lock").Should().BeFalse();
    }
}
