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
}
