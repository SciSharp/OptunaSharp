using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using AwesomeAssertions;
using NumSharp;
using OptunaSharp.Collections;
using OptunaSharp.Distributions;
using OptunaSharp.Storages;
using OptunaSharp.Storages.Journal;

namespace OptunaSharp.Tests.Storages;

/// <summary>
///     The storage conformance suite born in plan 0.6: every <see cref="BaseStorage" /> backend
///     must satisfy this contract identically. Runs over <see cref="InMemoryStorage" /> and
///     <see cref="JournalStorage" /> (file backend) via the concrete subclasses at the bottom.
/// </summary>
public abstract class StorageConformanceTests
{
    protected abstract BaseStorage CreateStorage();

    private BaseStorage _storage = null!;

    [TestInitialize]
    public void SetUp() => _storage = CreateStorage();

    [TestCleanup]
    public virtual void TearDown()
    {
    }

    private static readonly StudyDirection[] Minimize = { StudyDirection.Minimize };

    // ------------------------------------------------------------------ studies

    [TestMethod]
    public void CreateNewStudy_Generates_Name_When_Omitted()
    {
        int studyId = _storage.CreateNewStudy(Minimize);
        string name = _storage.GetStudyNameFromId(studyId);
        name.Should().StartWith(BaseStorage.DefaultStudyNamePrefix);
        _storage.GetStudyIdFromName(name).Should().Be(studyId);
    }

    [TestMethod]
    public void CreateNewStudy_Duplicate_Name_Raises()
    {
        _storage.CreateNewStudy(Minimize, "dup");
        var act = () => _storage.CreateNewStudy(Minimize, "dup");
        act.Should().Throw<DuplicatedStudyError>();
    }

    [TestMethod]
    public void DeleteStudy_Removes_Study_And_Its_Trials()
    {
        int studyId = _storage.CreateNewStudy(Minimize, "gone");
        int trialId = _storage.CreateNewTrial(studyId);
        _storage.DeleteStudy(studyId);

        var byName = () => _storage.GetStudyIdFromName("gone");
        byName.Should().Throw<KeyError>();

        // A fresh same-name study must be creatable.
        _storage.CreateNewStudy(Minimize, "gone");
        _ = trialId;
    }

    [TestMethod]
    public void DeleteStudy_Unknown_Id_Raises_KeyError()
    {
        var act = () => _storage.DeleteStudy(12345);
        act.Should().Throw<KeyError>();
    }

    [TestMethod]
    public void Study_Directions_RoundTrip()
    {
        int single = _storage.CreateNewStudy(Minimize, "s1");
        _storage.GetStudyDirections(single).Should().Equal(StudyDirection.Minimize);

        int multi = _storage.CreateNewStudy(
            new[] { StudyDirection.Maximize, StudyDirection.Minimize }, "s2");
        _storage.GetStudyDirections(multi).Should().Equal(StudyDirection.Maximize, StudyDirection.Minimize);
    }

    [TestMethod]
    public void Study_Attrs_RoundTrip_And_Overwrite()
    {
        int studyId = _storage.CreateNewStudy(Minimize, "attrs");
        _storage.SetStudyUserAttr(studyId, "k", 1L);
        _storage.SetStudyUserAttr(studyId, "k", 2L);
        _storage.SetStudyUserAttr(studyId, "list", new List<object?> { 1L, "a" });
        _storage.SetStudySystemAttr(studyId, "sys", "v");

        var userAttrs = _storage.GetStudyUserAttrs(studyId);
        userAttrs["k"].Should().Be(2L);
        ((List<object?>)userAttrs["list"]!).Should().Equal(1L, "a");
        _storage.GetStudySystemAttrs(studyId)["sys"].Should().Be("v");
    }

    [TestMethod]
    public void GetAllStudies_Returns_All_With_Attrs()
    {
        int id1 = _storage.CreateNewStudy(Minimize, "a");
        int id2 = _storage.CreateNewStudy(Minimize, "b");
        _storage.SetStudyUserAttr(id1, "x", 1L);

        var studies = _storage.GetAllStudies();
        studies.Select(s => s.StudyName).Should().Contain(new[] { "a", "b" });
        studies.Single(s => s.StudyName == "a").UserAttrs["x"].Should().Be(1L);
        _ = id2;
    }

    // ------------------------------------------------------------------ trials

    [TestMethod]
    public void CreateNewTrial_Assigns_Sequential_Numbers()
    {
        int studyId = _storage.CreateNewStudy(Minimize, "seq");
        int t0 = _storage.CreateNewTrial(studyId);
        int t1 = _storage.CreateNewTrial(studyId);

        _storage.GetTrial(t0).Number.Should().Be(0);
        _storage.GetTrial(t1).Number.Should().Be(1);
        _storage.GetTrial(t0).State.Should().Be(TrialState.Running);
        _storage.GetTrial(t0).DatetimeStart.Should().NotBeNull();
        _storage.GetTrialNumberFromId(t1).Should().Be(1);
        _storage.GetTrialIdFromStudyIdTrialNumber(studyId, 1).Should().Be(t1);
    }

    [TestMethod]
    public void CreateNewTrial_Unknown_Study_Raises()
    {
        var act = () => _storage.CreateNewTrial(999);
        act.Should().Throw<KeyError>();
    }

    [TestMethod]
    public void CreateNewTrial_From_Template_Copies_Everything()
    {
        int studyId = _storage.CreateNewStudy(Minimize, "tmpl");
        var template = FrozenTrial.CreateTrial(
            state: TrialState.Complete,
            value: 4.0,
            @params: new OrderedDict<string, object?> { ["x"] = 2.0 },
            distributions: new OrderedDict<string, BaseDistribution> { ["x"] = new FloatDistribution(0, 10) },
            userAttrs: new OrderedDict<string, object?> { ["memo"] = "hello" },
            intermediateValues: new OrderedDict<int, double> { [0] = 1.5, [2] = double.NaN });

        int trialId = _storage.CreateNewTrial(studyId, template);
        var stored = _storage.GetTrial(trialId);
        stored.State.Should().Be(TrialState.Complete);
        stored.Value.Should().Be(4.0);
        stored.Params["x"].Should().Be(2.0);
        stored.Distributions["x"].Should().Be(new FloatDistribution(0, 10));
        stored.UserAttrs["memo"].Should().Be("hello");
        stored.IntermediateValues[0].Should().Be(1.5);
        double.IsNaN(stored.IntermediateValues[2]).Should().BeTrue("NaN intermediates persist");
        stored.DatetimeComplete.Should().NotBeNull();
    }

    [TestMethod]
    public void SetTrialParam_Stores_External_Repr_And_Checks_Compat()
    {
        int studyId = _storage.CreateNewStudy(Minimize, "params");
        int trialId = _storage.CreateNewTrial(studyId);

        _storage.SetTrialParam(trialId, "x", 2.5, new FloatDistribution(0, 10));
        _storage.SetTrialParam(trialId, "n", 3.0, new IntDistribution(1, 5));
        _storage.SetTrialParam(trialId, "c", 1.0,
            new CategoricalDistribution(new CategoricalChoice[] { "a", "b" }));

        var trial = _storage.GetTrial(trialId);
        trial.Params["x"].Should().Be(2.5);
        trial.Params["n"].Should().Be(3L, "int external repr");
        trial.Params["c"].Should().Be("b", "categorical external repr");
        _storage.GetTrialParam(trialId, "c").Should().Be(1.0, "internal repr is the index");

        // A second trial with an incompatible categorical distribution for the same name fails.
        int trial2 = _storage.CreateNewTrial(studyId);
        var act = () => _storage.SetTrialParam(trial2, "c", 0.0,
            new CategoricalDistribution(new CategoricalChoice[] { "x", "y" }));
        act.Should().Throw<ValueError>();
    }

    [TestMethod]
    public void SetTrialStateValues_Lifecycle_And_Finished_Guard()
    {
        int studyId = _storage.CreateNewStudy(Minimize, "life");
        int trialId = _storage.CreateNewTrial(studyId);

        // RUNNING -> RUNNING is rejected with false.
        _storage.SetTrialStateValues(trialId, TrialState.Running).Should().BeFalse();

        _storage.SetTrialStateValues(trialId, TrialState.Complete, new[] { 1.5 }).Should().BeTrue();
        var trial = _storage.GetTrial(trialId);
        trial.State.Should().Be(TrialState.Complete);
        trial.Value.Should().Be(1.5);
        trial.DatetimeComplete.Should().NotBeNull();

        // Finished trials cannot be updated — verbatim message.
        var update = () => _storage.SetTrialStateValues(trialId, TrialState.Fail);
        update.Should().Throw<UpdateFinishedTrialError>()
            .WithMessage("Trial#0 has already finished and can not be updated.");
        var setParam = () => _storage.SetTrialParam(trialId, "x", 1.0, new FloatDistribution(0, 1));
        setParam.Should().Throw<UpdateFinishedTrialError>();
        var setAttr = () => _storage.SetTrialUserAttr(trialId, "k", 1L);
        setAttr.Should().Throw<UpdateFinishedTrialError>();
    }

    [TestMethod]
    public void Waiting_Trials_Can_Be_Popped_To_Running()
    {
        int studyId = _storage.CreateNewStudy(Minimize, "waiting");
        var waiting = FrozenTrial.CreateTrial(state: TrialState.Waiting);
        int trialId = _storage.CreateNewTrial(studyId, waiting);

        _storage.GetTrial(trialId).State.Should().Be(TrialState.Waiting);
        _storage.SetTrialStateValues(trialId, TrialState.Running).Should().BeTrue();
        _storage.GetTrial(trialId).State.Should().Be(TrialState.Running);
        _storage.GetTrial(trialId).DatetimeStart.Should().NotBeNull();
    }

    [TestMethod]
    public void Intermediate_Values_And_Attrs_RoundTrip()
    {
        int studyId = _storage.CreateNewStudy(Minimize, "iv");
        int trialId = _storage.CreateNewTrial(studyId);

        _storage.SetTrialIntermediateValue(trialId, 0, 10.0);
        _storage.SetTrialIntermediateValue(trialId, 5, double.NaN);
        _storage.SetTrialIntermediateValue(trialId, 0, 11.0);  // Overwrites.
        _storage.SetTrialUserAttr(trialId, "u", new List<object?> { 1L, 2L });
        _storage.SetTrialSystemAttr(trialId, "s", 1.5);

        var trial = _storage.GetTrial(trialId);
        trial.IntermediateValues[0].Should().Be(11.0);
        double.IsNaN(trial.IntermediateValues[5]).Should().BeTrue();
        ((List<object?>)trial.UserAttrs["u"]!).Should().Equal(1L, 2L);
        trial.SystemAttrs["s"].Should().Be(1.5);
    }

    [TestMethod]
    public void GetAllTrials_Filters_By_State_And_Orders_By_Number()
    {
        int studyId = _storage.CreateNewStudy(Minimize, "filter");
        int t0 = _storage.CreateNewTrial(studyId);
        int t1 = _storage.CreateNewTrial(studyId);
        int t2 = _storage.CreateNewTrial(studyId);
        _storage.SetTrialStateValues(t0, TrialState.Complete, new[] { 1.0 });
        _storage.SetTrialStateValues(t1, TrialState.Fail);

        _storage.GetAllTrials(studyId).Select(t => t.Number).Should().Equal(0, 1, 2);
        _storage.GetAllTrials(studyId, states: new[] { TrialState.Complete })
            .Should().ContainSingle(t => t.TrialId == t0);
        _storage.GetAllTrials(studyId, states: new[] { TrialState.Complete, TrialState.Fail })
            .Should().HaveCount(2);
        _storage.GetNTrials(studyId).Should().Be(3);
        _ = t2;
    }

    [TestMethod]
    public void GetAllTrials_Deepcopy_Isolates_Mutations()
    {
        int studyId = _storage.CreateNewStudy(Minimize, "iso");
        int trialId = _storage.CreateNewTrial(studyId);
        _storage.SetTrialUserAttr(trialId, "k", 1L);

        var copied = _storage.GetAllTrials(studyId, deepcopy: true)[0];
        copied.UserAttrsDict["k"] = 999L;

        _storage.GetTrial(trialId).UserAttrs["k"].Should().Be(1L, "storage state must be isolated from caller mutations");
    }

    [TestMethod]
    public void Returned_Trial_Is_Stable_Against_Subsequent_Writes()
    {
        int studyId = _storage.CreateNewStudy(Minimize, "stable");
        int trialId = _storage.CreateNewTrial(studyId);
        var before = _storage.GetTrial(trialId);
        _storage.SetTrialUserAttr(trialId, "k", 1L);

        before.UserAttrs.Should().NotContainKey("k",
            "a FrozenTrial read from storage must not be modified by subsequent writes (copy-on-write contract)");
    }

    [TestMethod]
    public void GetBestTrial_MinMax_And_Error_Cases()
    {
        int studyId = _storage.CreateNewStudy(Minimize, "best");
        var noneDone = () => _storage.GetBestTrial(studyId);
        noneDone.Should().Throw<ValueError>().WithMessage("No trials are completed yet.");

        int t0 = _storage.CreateNewTrial(studyId);
        _storage.SetTrialStateValues(t0, TrialState.Complete, new[] { 5.0 });
        int t1 = _storage.CreateNewTrial(studyId);
        _storage.SetTrialStateValues(t1, TrialState.Complete, new[] { 2.0 });
        int t2 = _storage.CreateNewTrial(studyId);
        _storage.SetTrialStateValues(t2, TrialState.Complete, new[] { 9.0 });

        _storage.GetBestTrial(studyId).TrialId.Should().Be(t1);

        int multi = _storage.CreateNewStudy(new[] { StudyDirection.Minimize, StudyDirection.Minimize }, "multi");
        int mt = _storage.CreateNewTrial(multi);
        _storage.SetTrialStateValues(mt, TrialState.Complete, new[] { 1.0, 2.0 });
        var multiBest = () => _storage.GetBestTrial(multi);
        multiBest.Should().Throw<RuntimeError>()
            .WithMessage("Best trial can be obtained only for single-objective optimization.");
    }

    [TestMethod]
    public void MultiObjective_Values_RoundTrip()
    {
        int studyId = _storage.CreateNewStudy(
            new[] { StudyDirection.Minimize, StudyDirection.Maximize }, "mo");
        int trialId = _storage.CreateNewTrial(studyId);
        _storage.SetTrialStateValues(trialId, TrialState.Complete, new[] { 1.0, -2.5 });
        _storage.GetTrial(trialId).Values.Should().Equal(1.0, -2.5);
    }

    [TestMethod]
    public void GetTrial_Unknown_Id_Raises_KeyError()
    {
        var act = () => _storage.GetTrial(424242);
        act.Should().Throw<KeyError>();
    }
}

/// <summary>The conformance suite over <see cref="InMemoryStorage" />.</summary>
[TestClass]
public class InMemoryStorageConformanceTests : StorageConformanceTests
{
    protected override BaseStorage CreateStorage() => new InMemoryStorage();
}

/// <summary>The conformance suite over <see cref="JournalStorage" /> with a file backend.</summary>
[TestClass]
public class JournalStorageConformanceTests : StorageConformanceTests
{
    private string _path = null!;

    protected override BaseStorage CreateStorage()
    {
        _path = Path.Combine(Path.GetTempPath(), $"optunasharp-journal-{Guid.NewGuid():N}.log");
        return new JournalStorage(new JournalFileBackend(_path, lockObj: new JournalFileOpenLock(_path)));
    }

    [TestCleanup]
    public override void TearDown()
    {
        try
        {
            File.Delete(_path);
        }
        catch (IOException)
        {
        }
    }
}
