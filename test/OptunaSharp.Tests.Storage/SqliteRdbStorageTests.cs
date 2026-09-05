using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using AwesomeAssertions;
using Microsoft.Data.Sqlite;
using NumSharp;
using OptunaSharp;
using OptunaSharp.Collections;
using OptunaSharp.Distributions;
using OptunaSharp.Samplers;
using OptunaSharp.Storages;

namespace OptunaSharp.Tests.Storage;

/// <summary>
///     Unit tests for the SQLite <see cref="RDBStorage" /> (Phase 4.1). These need no external
///     service — <c>Microsoft.Data.Sqlite</c> bundles the native engine — so they run in the normal
///     unit-test lane. They exercise the full <see cref="BaseStorage" /> contract, the exact
///     Optuna-compatible schema/version pins, and the finite/±inf/NaN and datetime encodings.
/// </summary>
[TestClass]
public class SqliteRdbStorageTests
{
    private readonly List<string> _tempFiles = new();

    [TestCleanup]
    public void Cleanup()
    {
        foreach (var f in _tempFiles)
        {
            try
            {
                if (File.Exists(f))
                    File.Delete(f);
            }
            catch (IOException)
            {
            }
        }
    }

    private string TempDbPath()
    {
        string path = Path.Combine(Path.GetTempPath(), $"optunasharp-rdb-{Guid.NewGuid():N}.db");
        _tempFiles.Add(path);
        return path;
    }

    private RDBStorage NewFileStorage() => new($"sqlite:///{TempDbPath().Replace('\\', '/')}");

    private static RDBStorage NewMemoryStorage() => new("sqlite:///:memory:");

    private static OrderedDict<string, BaseDistribution> QuadDist() =>
        new() { ["x"] = new FloatDistribution(-10, 10) };

    // ------------------------------------------------------------------ schema / version pins

    [TestMethod]
    public void FreshDatabase_StampsOptunaCompatibleSchemaAndVersion()
    {
        string path = TempDbPath();
        using (var storage = new RDBStorage($"sqlite:///{path.Replace('\\', '/')}"))
        {
            storage.CreateNewStudy(new[] { StudyDirection.Minimize }, "s");
        }

        using var conn = new SqliteConnection(new SqliteConnectionStringBuilder { DataSource = path }.ToString());
        conn.Open();

        Scalar(conn, "SELECT version_num FROM alembic_version").Should().Be("v3.2.0.a");
        Scalar(conn, "SELECT schema_version FROM version_info").Should().Be("12");

        // Every Optuna table is present.
        foreach (var table in new[]
        {
            "studies", "study_directions", "study_user_attributes", "study_system_attributes",
            "trials", "trial_params", "trial_values", "trial_intermediate_values",
            "trial_user_attributes", "trial_system_attributes", "trial_heartbeats",
            "version_info", "alembic_version",
        })
        {
            Scalar(conn, $"SELECT name FROM sqlite_master WHERE type='table' AND name='{table}'")
                .Should().Be(table, "table {0} must exist", table);
        }
    }

    [TestMethod]
    public void OpeningExistingDatabase_IsIdempotent_NoDuplicateVersionRows()
    {
        string path = TempDbPath();
        string url = $"sqlite:///{path.Replace('\\', '/')}";
        using (var s1 = new RDBStorage(url))
            s1.CreateNewStudy(new[] { StudyDirection.Minimize }, "s");
        using (var s2 = new RDBStorage(url))
        {
            // Reopening must not raise the compatibility check and must not add version/alembic rows.
            s2.GetAllStudies().Should().ContainSingle();
        }

        using var conn = new SqliteConnection(new SqliteConnectionStringBuilder { DataSource = path }.ToString());
        conn.Open();
        Scalar(conn, "SELECT COUNT(*) FROM version_info").Should().Be("1");
        Scalar(conn, "SELECT COUNT(*) FROM alembic_version").Should().Be("1");
    }

    // ------------------------------------------------------------------ studies

    [TestMethod]
    public void CreateStudy_AssignsIds_AndRoundTripsNameAndDirections()
    {
        using var storage = NewMemoryStorage();
        int id1 = storage.CreateNewStudy(new[] { StudyDirection.Minimize }, "a");
        int id2 = storage.CreateNewStudy(new[] { StudyDirection.Maximize, StudyDirection.Minimize }, "b");

        storage.GetStudyIdFromName("a").Should().Be(id1);
        storage.GetStudyNameFromId(id2).Should().Be("b");
        storage.GetStudyDirections(id1).Should().Equal(StudyDirection.Minimize);
        storage.GetStudyDirections(id2).Should().Equal(StudyDirection.Maximize, StudyDirection.Minimize);
    }

    [TestMethod]
    public void CreateStudy_DuplicateName_Throws_DuplicatedStudyError()
    {
        using var storage = NewMemoryStorage();
        storage.CreateNewStudy(new[] { StudyDirection.Minimize }, "dup");
        Action act = () => storage.CreateNewStudy(new[] { StudyDirection.Minimize }, "dup");
        act.Should().Throw<DuplicatedStudyError>();
    }

    [TestMethod]
    public void CreateStudy_NoName_GeneratesUnique()
    {
        using var storage = NewMemoryStorage();
        int id = storage.CreateNewStudy(new[] { StudyDirection.Minimize });
        storage.GetStudyNameFromId(id).Should().StartWith("no-name-");
    }

    [TestMethod]
    public void StudyAttrs_RoundTrip_UserAndSystem()
    {
        using var storage = NewMemoryStorage();
        int id = storage.CreateNewStudy(new[] { StudyDirection.Minimize }, "s");
        storage.SetStudyUserAttr(id, "owner", "eli");
        storage.SetStudyUserAttr(id, "owner", "eli2"); // overwrite
        storage.SetStudySystemAttr(id, "algo", new OrderedDict<string, object?> { ["k"] = 1L, ["b"] = true });

        storage.GetStudyUserAttrs(id)["owner"].Should().Be("eli2");
        var algo = (OrderedDict<string, object?>)storage.GetStudySystemAttrs(id)["algo"]!;
        algo["k"].Should().Be(1L);
        algo["b"].Should().Be(true);
    }

    [TestMethod]
    public void GetAllStudies_ReturnsAll_SortedById_WithAttrs()
    {
        using var storage = NewMemoryStorage();
        int a = storage.CreateNewStudy(new[] { StudyDirection.Minimize }, "a");
        storage.SetStudyUserAttr(a, "k", "v");
        storage.CreateNewStudy(new[] { StudyDirection.Maximize, StudyDirection.Minimize }, "b");

        var studies = storage.GetAllStudies();
        studies.Select(s => s.StudyName).Should().Equal("a", "b");
        studies[0].UserAttrs["k"].Should().Be("v");
        studies[1].Directions.Should().Equal(StudyDirection.Maximize, StudyDirection.Minimize);
    }

    [TestMethod]
    public void DeleteStudy_RemovesStudyAndCascadesChildren()
    {
        string path = TempDbPath();
        string url = $"sqlite:///{path.Replace('\\', '/')}";
        int studyId;
        using (var storage = new RDBStorage(url))
        {
            studyId = storage.CreateNewStudy(new[] { StudyDirection.Minimize }, "s");
            int trialId = storage.CreateNewTrial(studyId);
            storage.SetTrialParam(trialId, "x", 1.0, new FloatDistribution(-10, 10));
            storage.SetTrialUserAttr(trialId, "k", "v");
            storage.SetTrialIntermediateValue(trialId, 0, 0.5);
            storage.SetTrialStateValues(trialId, TrialState.Complete, new[] { 1.0 });
            storage.DeleteStudy(studyId);
            Action act = () => storage.GetStudyNameFromId(studyId);
            act.Should().Throw<KeyError>();
        }

        // No orphan child rows survive (foreign keys are not enforced, so the cascade is explicit).
        using var conn = new SqliteConnection(new SqliteConnectionStringBuilder { DataSource = path }.ToString());
        conn.Open();
        foreach (var table in new[]
        {
            "trials", "trial_params", "trial_values", "trial_intermediate_values",
            "trial_user_attributes", "study_directions",
        })
        {
            Scalar(conn, $"SELECT COUNT(*) FROM {table}").Should().Be("0", "no rows left in {0}", table);
        }
    }

    [TestMethod]
    public void MissingStudy_Throws_KeyError()
    {
        using var storage = NewMemoryStorage();
        Action act = () => storage.GetStudyDirections(999);
        act.Should().Throw<KeyError>();
    }

    // ------------------------------------------------------------------ trials

    [TestMethod]
    public void CreateTrial_AssignsSequentialNumbers_PerStudy()
    {
        using var storage = NewMemoryStorage();
        int s1 = storage.CreateNewStudy(new[] { StudyDirection.Minimize }, "a");
        int s2 = storage.CreateNewStudy(new[] { StudyDirection.Minimize }, "b");

        int t0 = storage.CreateNewTrial(s1);
        int t1 = storage.CreateNewTrial(s1);
        int t2 = storage.CreateNewTrial(s2);

        storage.GetTrial(t0).Number.Should().Be(0);
        storage.GetTrial(t1).Number.Should().Be(1);
        storage.GetTrial(t2).Number.Should().Be(0);
        storage.GetTrialIdFromStudyIdTrialNumber(s1, 1).Should().Be(t1);
    }

    [TestMethod]
    public void SetAndGetTrialParam_RoundTrips_InternalRepr()
    {
        using var storage = NewMemoryStorage();
        int studyId = storage.CreateNewStudy(new[] { StudyDirection.Minimize }, "s");
        int trialId = storage.CreateNewTrial(studyId);

        var intDist = new IntDistribution(1, 100, log: true);
        storage.SetTrialParam(trialId, "x", 1.5, new FloatDistribution(-10, 10));
        storage.SetTrialParam(trialId, "y", 4.0, intDist);

        storage.GetTrialParam(trialId, "x").Should().Be(1.5);
        storage.GetTrialParam(trialId, "y").Should().Be(4.0);

        var trial = storage.GetTrial(trialId);
        trial.Params["x"].Should().Be(1.5);
        trial.Params["y"].Should().Be(4L);
        trial.Distributions["y"].Should().Be(intDist);
    }

    [TestMethod]
    public void SetTrialParam_IncompatibleDistribution_Throws()
    {
        using var storage = NewMemoryStorage();
        int studyId = storage.CreateNewStudy(new[] { StudyDirection.Minimize }, "s");
        int t0 = storage.CreateNewTrial(studyId);
        storage.SetTrialParam(t0, "x", 1.0, new FloatDistribution(-10, 10));

        int t1 = storage.CreateNewTrial(studyId);
        Action act = () => storage.SetTrialParam(t1, "x", 1.0, new IntDistribution(0, 10));
        act.Should().Throw<ValueError>();
    }

    [TestMethod]
    public void CategoricalParam_MixedTypes_RoundTripsWithPythonEquality()
    {
        using var storage = NewMemoryStorage();
        int studyId = storage.CreateNewStudy(new[] { StudyDirection.Minimize }, "s");
        int trialId = storage.CreateNewTrial(studyId);
        var cat = new CategoricalDistribution(new CategoricalChoice[] { "a", "b", (long)3, true, CategoricalChoice.None });

        storage.SetTrialParam(trialId, "z", 3.0, cat); // index 3 -> true
        var trial = storage.GetTrial(trialId);
        trial.Params["z"].Should().Be(true);
        trial.Distributions["z"].Should().Be(cat);
    }

    [TestMethod]
    public void SetTrialStateValues_CompletesTrial_SetsDatetimeAndValue()
    {
        using var storage = NewMemoryStorage();
        int studyId = storage.CreateNewStudy(new[] { StudyDirection.Minimize }, "s");
        int trialId = storage.CreateNewTrial(studyId);

        storage.GetTrial(trialId).State.Should().Be(TrialState.Running);
        storage.SetTrialStateValues(trialId, TrialState.Complete, new[] { 42.0 }).Should().BeTrue();

        var trial = storage.GetTrial(trialId);
        trial.State.Should().Be(TrialState.Complete);
        trial.Value.Should().Be(42.0);
        trial.DatetimeComplete.Should().NotBeNull();
    }

    [TestMethod]
    public void UpdatingFinishedTrial_Throws_UpdateFinishedTrialError()
    {
        using var storage = NewMemoryStorage();
        int studyId = storage.CreateNewStudy(new[] { StudyDirection.Minimize }, "s");
        int trialId = storage.CreateNewTrial(studyId);
        storage.SetTrialStateValues(trialId, TrialState.Complete, new[] { 1.0 });

        Action act = () => storage.SetTrialUserAttr(trialId, "k", "v");
        act.Should().Throw<UpdateFinishedTrialError>();
    }

    [TestMethod]
    public void MultiObjectiveValues_RoundTrip()
    {
        using var storage = NewMemoryStorage();
        int studyId = storage.CreateNewStudy(new[] { StudyDirection.Minimize, StudyDirection.Maximize }, "s");
        int trialId = storage.CreateNewTrial(studyId);
        storage.SetTrialStateValues(trialId, TrialState.Complete, new[] { 0.3, 10.0 });

        storage.GetTrial(trialId).Values.Should().Equal(0.3, 10.0);
    }

    [TestMethod]
    public void InfiniteObjectiveValues_RoundTrip_ViaTypeTag()
    {
        using var storage = NewMemoryStorage();
        int studyId = storage.CreateNewStudy(new[] { StudyDirection.Minimize }, "s");

        int tPos = storage.CreateNewTrial(studyId);
        storage.SetTrialStateValues(tPos, TrialState.Complete, new[] { double.PositiveInfinity });
        int tNeg = storage.CreateNewTrial(studyId);
        storage.SetTrialStateValues(tNeg, TrialState.Complete, new[] { double.NegativeInfinity });

        storage.GetTrial(tPos).Value.Should().Be(double.PositiveInfinity);
        storage.GetTrial(tNeg).Value.Should().Be(double.NegativeInfinity);
    }

    [TestMethod]
    public void IntermediateValues_RoundTrip_Finite_Inf_Nan()
    {
        using var storage = NewMemoryStorage();
        int studyId = storage.CreateNewStudy(new[] { StudyDirection.Minimize }, "s");
        int trialId = storage.CreateNewTrial(studyId);

        storage.SetTrialIntermediateValue(trialId, 0, 0.5);
        storage.SetTrialIntermediateValue(trialId, 1, double.PositiveInfinity);
        storage.SetTrialIntermediateValue(trialId, 2, double.NaN);
        storage.SetTrialIntermediateValue(trialId, 3, double.NegativeInfinity);
        storage.SetTrialIntermediateValue(trialId, 0, 0.75); // overwrite step 0

        var iv = storage.GetTrial(trialId).IntermediateValues;
        iv[0].Should().Be(0.75);
        iv[1].Should().Be(double.PositiveInfinity);
        double.IsNaN(iv[2]).Should().BeTrue();
        iv[3].Should().Be(double.NegativeInfinity);
    }

    [TestMethod]
    public void GetAllTrials_FiltersByState()
    {
        using var storage = NewMemoryStorage();
        int studyId = storage.CreateNewStudy(new[] { StudyDirection.Minimize }, "s");
        int t0 = storage.CreateNewTrial(studyId);
        storage.SetTrialStateValues(t0, TrialState.Complete, new[] { 1.0 });
        storage.CreateNewTrial(studyId); // running

        storage.GetAllTrials(studyId).Should().HaveCount(2);
        storage.GetAllTrials(studyId, states: new[] { TrialState.Complete }).Should().ContainSingle();
        storage.GetAllTrials(studyId, states: new[] { TrialState.Running }).Should().ContainSingle();
    }

    [TestMethod]
    public void GetBestTrial_HandlesMinMax_AndInfinities()
    {
        using var storage = NewMemoryStorage();
        int studyId = storage.CreateNewStudy(new[] { StudyDirection.Minimize }, "s");
        foreach (double v in new[] { 3.0, 1.0, double.PositiveInfinity, 2.0 })
        {
            int t = storage.CreateNewTrial(studyId);
            storage.SetTrialStateValues(t, TrialState.Complete, new[] { v });
        }
        storage.GetBestTrial(studyId).Value.Should().Be(1.0);

        int maxStudy = storage.CreateNewStudy(new[] { StudyDirection.Maximize }, "m");
        foreach (double v in new[] { 3.0, double.NegativeInfinity, 5.0 })
        {
            int t = storage.CreateNewTrial(maxStudy);
            storage.SetTrialStateValues(t, TrialState.Complete, new[] { v });
        }
        storage.GetBestTrial(maxStudy).Value.Should().Be(5.0);
    }

    [TestMethod]
    public void CreateTrial_FromTemplate_PersistsEverything()
    {
        using var storage = NewMemoryStorage();
        int studyId = storage.CreateNewStudy(new[] { StudyDirection.Minimize }, "s");

        var template = FrozenTrial.CreateTrial(
            state: TrialState.Complete,
            value: 2.25,
            @params: new OrderedDict<string, object?> { ["x"] = 1.5 },
            distributions: QuadDist(),
            userAttrs: new OrderedDict<string, object?> { ["note"] = "hi" },
            intermediateValues: new OrderedDict<int, double> { [0] = 0.5 });

        int trialId = storage.CreateNewTrial(studyId, template);
        var trial = storage.GetTrial(trialId);
        trial.State.Should().Be(TrialState.Complete);
        trial.Value.Should().Be(2.25);
        trial.Params["x"].Should().Be(1.5);
        trial.UserAttrs["note"].Should().Be("hi");
        trial.IntermediateValues[0].Should().Be(0.5);
        trial.Number.Should().Be(0);
    }

    // ------------------------------------------------------------------ persistence / integration

    [TestMethod]
    public void FileStorage_PersistsAcrossReopen()
    {
        string path = TempDbPath();
        string url = $"sqlite:///{path.Replace('\\', '/')}";
        int studyId;
        using (var storage = new RDBStorage(url))
        {
            studyId = storage.CreateNewStudy(new[] { StudyDirection.Minimize }, "s");
            int t = storage.CreateNewTrial(studyId);
            storage.SetTrialParam(t, "x", 1.5, new FloatDistribution(-10, 10));
            storage.SetTrialStateValues(t, TrialState.Complete, new[] { 2.25 });
        }

        using (var reopened = new RDBStorage(url))
        {
            var trials = reopened.GetAllTrials(reopened.GetStudyIdFromName("s"));
            trials.Should().ContainSingle();
            trials[0].Value.Should().Be(2.25);
            trials[0].Params["x"].Should().Be(1.5);
        }
    }

    [TestMethod]
    public void GetStorage_WrapsRdbStorage_InCachedStorage()
    {
        using var rdb = NewMemoryStorage();
        var resolved = StorageFunctions.GetStorage(rdb);
        resolved.Should().BeOfType<CachedStorage>();
        ((CachedStorage)resolved).Backend.Should().BeSameAs(rdb);
    }

    [TestMethod]
    public void EndToEnd_Optimize_ThroughCachedStorage_Wrapping()
    {
        using var rdb = NewMemoryStorage();
        var study = Optuna.CreateStudy(storage: rdb, studyName: "quad", sampler: new RandomSampler(seed: 0));
        study.Optimize(t => { double x = t.SuggestFloat("x", -10, 10); return x * x; }, nTrials: 10);

        study.Trials.Should().HaveCount(10);
        study.BestValue.Should().BeLessThanOrEqualTo(study.Trials.Min(t => t.Value!.Value) + 1e-12);
    }

    // ------------------------------------------------------------------ helpers

    private static string? Scalar(SqliteConnection conn, string sql)
    {
        using var cmd = conn.CreateCommand();
        cmd.CommandText = sql;
        object? value = cmd.ExecuteScalar();
        return value is null or DBNull ? null : Convert.ToString(value, System.Globalization.CultureInfo.InvariantCulture);
    }
}
