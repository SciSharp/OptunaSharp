using System;
using System.Collections.Generic;
using System.Data.Common;
using System.Globalization;
using System.Linq;
using AwesomeAssertions;
using Microsoft.EntityFrameworkCore;
using MySqlConnector;
using Npgsql;
using NumSharp;
using OptunaSharp;
using OptunaSharp.Collections;
using OptunaSharp.Distributions;
using OptunaSharp.Internal;
using OptunaSharp.Samplers;
using OptunaSharp.Storages;
using OptunaSharp.Storages.EFCore;
// Both the SQLite and EF Core packages expose an `RDBStorage`; this suite means the EF Core one.
using RDBStorage = OptunaSharp.Storages.EFCore.RDBStorage;

namespace OptunaSharp.Tests.Storage;

/// <summary>
///     The EF Core RDB conformance suite (Phase 4.2). One shared set of tests run against a live
///     PostgreSQL and MySQL server (Testcontainers) — the full <see cref="BaseStorage" /> contract, the
///     finite/±inf/NaN + datetime encodings, and the Optuna-compatible schema/version pins. It is the
///     EF Core analogue of <c>SqliteRdbStorageTests</c> plus provider-specific schema-shape assertions.
///     <see cref="TestCategoryAttribute">Integration</see>-categorized: self-skips without Docker.
/// </summary>
public abstract class EFCoreRdbStorageTestsBase
{
    protected abstract RdbProvider Provider { get; }

    [TestInitialize]
    public void Setup() => EFCoreDbSession.EnsureAvailable(Provider);

    private RDBStorage NewStorage() => EFCoreDbSession.NewStorage(Provider);

    // Build a storage over a freshly-reset schema with custom options (the golden-standard API tests).
    private RDBStorage NewStorage(EFCoreRdbOptions options)
    {
        EFCoreDbSession.Reset(Provider);
        return new RDBStorage(Provider, EFCoreDbSession.ConnectionString(Provider), options);
    }

    // A second storage over the same database WITHOUT resetting the schema (reopen/persistence tests).
    private RDBStorage ReopenStorage() => new(Provider, EFCoreDbSession.ConnectionString(Provider));

    /// <summary>The connection string for the running container (for native-config-object ctor tests).</summary>
    protected string ConnectionString => EFCoreDbSession.ConnectionString(Provider);

    /// <summary>Options that enable EF Core's retry-on-failure for this provider (see the retry test).</summary>
    protected abstract EFCoreRdbOptions RetryOptions();

    protected DbConnection RawConn()
    {
        DbConnection conn = Provider == RdbProvider.PostgreSql
            ? new NpgsqlConnection(EFCoreDbSession.ConnectionString(Provider))
            : new MySqlConnection(EFCoreDbSession.ConnectionString(Provider));
        conn.Open();
        return conn;
    }

    private static OrderedDict<string, BaseDistribution> QuadDist() =>
        new() { ["x"] = new FloatDistribution(-10, 10) };

    // ------------------------------------------------------------------ schema / version pins

    [TestMethod]
    [TestCategory("Integration")]
    public void FreshDatabase_StampsOptunaCompatibleSchemaAndVersion()
    {
        using (var storage = NewStorage())
            storage.CreateNewStudy(new[] { StudyDirection.Minimize }, "s");

        using var conn = RawConn();
        Scalar(conn, "SELECT version_num FROM alembic_version").Should().Be("v3.2.0.a");
        Scalar(conn, "SELECT schema_version FROM version_info").Should().Be("12");

        // Every Optuna table is present.
        foreach (var table in EFCoreDbSession.Tables)
            TableExists(conn, table).Should().BeTrue("table {0} must exist", table);
    }

    [TestMethod]
    [TestCategory("Integration")]
    public void Schema_MatchesSqlAlchemyShape()
    {
        using (var _ = NewStorage()) { }

        using var conn = RawConn();
        if (Provider == RdbProvider.PostgreSql)
        {
            // Native enum types exist with SQLAlchemy's exact labels (name = lowercased Python class).
            EnumLabels(conn, "studydirection").Should().Equal("NOT_SET", "MINIMIZE", "MAXIMIZE");
            EnumLabels(conn, "trialstate").Should().Equal("RUNNING", "COMPLETE", "PRUNED", "FAIL", "WAITING");
            EnumLabels(conn, "trialvaluetype").Should().Equal("FINITE", "INF_POS", "INF_NEG");
            EnumLabels(conn, "trialintermediatevaluetype").Should().Equal("FINITE", "INF_POS", "INF_NEG", "NAN");

            ColumnType(conn, "trials", "state").Should().Be("USER-DEFINED");            // the enum type
            ColumnType(conn, "trials", "datetime_start").Should().Be("timestamp without time zone");
            ColumnType(conn, "trial_params", "param_value").Should().Be("double precision"); // FLOAT(53)
            ColumnType(conn, "studies", "study_name").Should().Be("character varying");
        }
        else
        {
            ColumnType(conn, "study_directions", "direction")
                .Should().Be("enum('NOT_SET','MINIMIZE','MAXIMIZE')");
            ColumnType(conn, "trials", "state")
                .Should().Be("enum('RUNNING','COMPLETE','PRUNED','FAIL','WAITING')");
            ColumnType(conn, "trial_values", "value_type").Should().Be("enum('FINITE','INF_POS','INF_NEG')");
            ColumnType(conn, "trials", "datetime_start").Should().Be("datetime");
            ColumnType(conn, "trial_params", "param_value").Should().Be("double"); // FLOAT(53) -> DOUBLE
        }
    }

    [TestMethod]
    [TestCategory("Integration")]
    public void OpeningExistingDatabase_IsIdempotent_NoDuplicateVersionRows()
    {
        using (var s1 = NewStorage())
            s1.CreateNewStudy(new[] { StudyDirection.Minimize }, "s");
        using (var s2 = ReopenStorage())
            s2.GetAllStudies().Should().ContainSingle();

        using var conn = RawConn();
        Scalar(conn, "SELECT COUNT(*) FROM version_info").Should().Be("1");
        Scalar(conn, "SELECT COUNT(*) FROM alembic_version").Should().Be("1");
    }

    // ------------------------------------------------------------------ studies

    [TestMethod]
    [TestCategory("Integration")]
    public void CreateStudy_AssignsIds_AndRoundTripsNameAndDirections()
    {
        using var storage = NewStorage();
        int id1 = storage.CreateNewStudy(new[] { StudyDirection.Minimize }, "a");
        int id2 = storage.CreateNewStudy(new[] { StudyDirection.Maximize, StudyDirection.Minimize }, "b");

        storage.GetStudyIdFromName("a").Should().Be(id1);
        storage.GetStudyNameFromId(id2).Should().Be("b");
        storage.GetStudyDirections(id1).Should().Equal(StudyDirection.Minimize);
        storage.GetStudyDirections(id2).Should().Equal(StudyDirection.Maximize, StudyDirection.Minimize);
    }

    [TestMethod]
    [TestCategory("Integration")]
    public void CreateStudy_DuplicateName_Throws_DuplicatedStudyError()
    {
        using var storage = NewStorage();
        storage.CreateNewStudy(new[] { StudyDirection.Minimize }, "dup");
        Action act = () => storage.CreateNewStudy(new[] { StudyDirection.Minimize }, "dup");
        act.Should().Throw<DuplicatedStudyError>();
    }

    [TestMethod]
    [TestCategory("Integration")]
    public void CreateStudy_NoName_GeneratesUnique()
    {
        using var storage = NewStorage();
        int id = storage.CreateNewStudy(new[] { StudyDirection.Minimize });
        storage.GetStudyNameFromId(id).Should().StartWith("no-name-");
    }

    [TestMethod]
    [TestCategory("Integration")]
    public void StudyAttrs_RoundTrip_UserAndSystem()
    {
        using var storage = NewStorage();
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
    [TestCategory("Integration")]
    public void GetAllStudies_ReturnsAll_SortedById_WithAttrs()
    {
        using var storage = NewStorage();
        int a = storage.CreateNewStudy(new[] { StudyDirection.Minimize }, "a");
        storage.SetStudyUserAttr(a, "k", "v");
        storage.CreateNewStudy(new[] { StudyDirection.Maximize, StudyDirection.Minimize }, "b");

        var studies = storage.GetAllStudies();
        studies.Select(s => s.StudyName).Should().Equal("a", "b");
        studies[0].UserAttrs["k"].Should().Be("v");
        studies[1].Directions.Should().Equal(StudyDirection.Maximize, StudyDirection.Minimize);
    }

    [TestMethod]
    [TestCategory("Integration")]
    public void DeleteStudy_RemovesStudyAndCascadesChildren()
    {
        int studyId;
        using (var storage = NewStorage())
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

        using var conn = RawConn();
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
    [TestCategory("Integration")]
    public void MissingStudy_Throws_KeyError()
    {
        using var storage = NewStorage();
        Action act = () => storage.GetStudyDirections(999);
        act.Should().Throw<KeyError>();
    }

    // ------------------------------------------------------------------ trials

    [TestMethod]
    [TestCategory("Integration")]
    public void CreateTrial_AssignsSequentialNumbers_PerStudy()
    {
        using var storage = NewStorage();
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
    [TestCategory("Integration")]
    public void SetAndGetTrialParam_RoundTrips_InternalRepr()
    {
        using var storage = NewStorage();
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
    [TestCategory("Integration")]
    public void SetTrialParam_IncompatibleDistribution_Throws()
    {
        using var storage = NewStorage();
        int studyId = storage.CreateNewStudy(new[] { StudyDirection.Minimize }, "s");
        int t0 = storage.CreateNewTrial(studyId);
        storage.SetTrialParam(t0, "x", 1.0, new FloatDistribution(-10, 10));

        int t1 = storage.CreateNewTrial(studyId);
        Action act = () => storage.SetTrialParam(t1, "x", 1.0, new IntDistribution(0, 10));
        act.Should().Throw<ValueError>();
    }

    [TestMethod]
    [TestCategory("Integration")]
    public void CategoricalParam_MixedTypes_RoundTripsWithPythonEquality()
    {
        using var storage = NewStorage();
        int studyId = storage.CreateNewStudy(new[] { StudyDirection.Minimize }, "s");
        int trialId = storage.CreateNewTrial(studyId);
        var cat = new CategoricalDistribution(new CategoricalChoice[] { "a", "b", (long)3, true, CategoricalChoice.None });

        storage.SetTrialParam(trialId, "z", 3.0, cat); // index 3 -> true
        var trial = storage.GetTrial(trialId);
        trial.Params["z"].Should().Be(true);
        trial.Distributions["z"].Should().Be(cat);
    }

    [TestMethod]
    [TestCategory("Integration")]
    public void SetTrialStateValues_CompletesTrial_SetsDatetimeAndValue()
    {
        using var storage = NewStorage();
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
    [TestCategory("Integration")]
    public void UpdatingFinishedTrial_Throws_UpdateFinishedTrialError()
    {
        using var storage = NewStorage();
        int studyId = storage.CreateNewStudy(new[] { StudyDirection.Minimize }, "s");
        int trialId = storage.CreateNewTrial(studyId);
        storage.SetTrialStateValues(trialId, TrialState.Complete, new[] { 1.0 });

        Action act = () => storage.SetTrialUserAttr(trialId, "k", "v");
        act.Should().Throw<UpdateFinishedTrialError>();
    }

    [TestMethod]
    [TestCategory("Integration")]
    public void MultiObjectiveValues_RoundTrip()
    {
        using var storage = NewStorage();
        int studyId = storage.CreateNewStudy(new[] { StudyDirection.Minimize, StudyDirection.Maximize }, "s");
        int trialId = storage.CreateNewTrial(studyId);
        storage.SetTrialStateValues(trialId, TrialState.Complete, new[] { 0.3, 10.0 });

        storage.GetTrial(trialId).Values.Should().Equal(0.3, 10.0);
    }

    [TestMethod]
    [TestCategory("Integration")]
    public void InfiniteObjectiveValues_RoundTrip_ViaTypeTag()
    {
        using var storage = NewStorage();
        int studyId = storage.CreateNewStudy(new[] { StudyDirection.Minimize }, "s");

        int tPos = storage.CreateNewTrial(studyId);
        storage.SetTrialStateValues(tPos, TrialState.Complete, new[] { double.PositiveInfinity });
        int tNeg = storage.CreateNewTrial(studyId);
        storage.SetTrialStateValues(tNeg, TrialState.Complete, new[] { double.NegativeInfinity });

        storage.GetTrial(tPos).Value.Should().Be(double.PositiveInfinity);
        storage.GetTrial(tNeg).Value.Should().Be(double.NegativeInfinity);
    }

    [TestMethod]
    [TestCategory("Integration")]
    public void IntermediateValues_RoundTrip_Finite_Inf_Nan()
    {
        using var storage = NewStorage();
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
    [TestCategory("Integration")]
    public void GetAllTrials_FiltersByState()
    {
        using var storage = NewStorage();
        int studyId = storage.CreateNewStudy(new[] { StudyDirection.Minimize }, "s");
        int t0 = storage.CreateNewTrial(studyId);
        storage.SetTrialStateValues(t0, TrialState.Complete, new[] { 1.0 });
        storage.CreateNewTrial(studyId); // running

        storage.GetAllTrials(studyId).Should().HaveCount(2);
        storage.GetAllTrials(studyId, states: new[] { TrialState.Complete }).Should().ContainSingle();
        storage.GetAllTrials(studyId, states: new[] { TrialState.Running }).Should().ContainSingle();
    }

    [TestMethod]
    [TestCategory("Integration")]
    public void GetBestTrial_HandlesMinMax_AndInfinities()
    {
        using var storage = NewStorage();
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
    [TestCategory("Integration")]
    public void CreateTrial_FromTemplate_PersistsEverything()
    {
        using var storage = NewStorage();
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
    [TestCategory("Integration")]
    public void FileStorage_PersistsAcrossReopen()
    {
        int studyId;
        using (var storage = NewStorage())
        {
            studyId = storage.CreateNewStudy(new[] { StudyDirection.Minimize }, "s");
            int t = storage.CreateNewTrial(studyId);
            storage.SetTrialParam(t, "x", 1.5, new FloatDistribution(-10, 10));
            storage.SetTrialStateValues(t, TrialState.Complete, new[] { 2.25 });
        }

        using var reopened = ReopenStorage();
        var trials = reopened.GetAllTrials(reopened.GetStudyIdFromName("s"));
        trials.Should().ContainSingle();
        trials[0].Value.Should().Be(2.25);
        trials[0].Params["x"].Should().Be(1.5);
    }

    [TestMethod]
    [TestCategory("Integration")]
    public void GetStorage_WrapsRdbStorage_InCachedStorage()
    {
        using var rdb = NewStorage();
        var resolved = StorageFunctions.GetStorage(rdb);
        resolved.Should().BeOfType<CachedStorage>();
        ((CachedStorage)resolved).Backend.Should().BeSameAs(rdb);
    }

    [TestMethod]
    [TestCategory("Integration")]
    public void EndToEnd_Optimize_ThroughCachedStorage_Wrapping()
    {
        using var rdb = NewStorage();
        var study = Optuna.CreateStudy(storage: rdb, studyName: "quad", sampler: new RandomSampler(seed: 0));
        study.Optimize(t => { double x = t.SuggestFloat("x", -10, 10); return x * x; }, nTrials: 10);

        study.Trials.Should().HaveCount(10);
        study.BestValue.Should().BeLessThanOrEqualTo(study.Trials.Min(t => t.Value!.Value) + 1e-12);
    }

    // ------------------------------------------------------------------ contract coverage

    [TestMethod]
    [TestCategory("Integration")]
    public void GetNTrials_CountsByState()
    {
        using var storage = NewStorage();
        int studyId = storage.CreateNewStudy(new[] { StudyDirection.Minimize }, "s");
        int c0 = storage.CreateNewTrial(studyId);
        storage.SetTrialStateValues(c0, TrialState.Complete, new[] { 1.0 });
        storage.CreateNewTrial(studyId); // running

        storage.GetNTrials(studyId).Should().Be(2);
        storage.GetNTrials(studyId, new[] { TrialState.Complete }).Should().Be(1);
        storage.GetNTrials(studyId, new[] { TrialState.Running }).Should().Be(1);
        storage.GetNTrials(studyId, new[] { TrialState.Pruned }).Should().Be(0);
    }

    [TestMethod]
    [TestCategory("Integration")]
    public void GetTrialNumberFromId_And_ReadHelpers()
    {
        using var storage = NewStorage();
        int studyId = storage.CreateNewStudy(new[] { StudyDirection.Minimize }, "s");
        int t = storage.CreateNewTrial(studyId);
        storage.SetTrialParam(t, "x", 1.5, new FloatDistribution(-10, 10));
        storage.SetTrialUserAttr(t, "u", "uv");
        storage.SetTrialSystemAttr(t, "s", 9L);

        storage.GetTrialNumberFromId(t).Should().Be(0);
        storage.GetTrialParams(t)["x"].Should().Be(1.5);
        storage.GetTrialUserAttrs(t)["u"].Should().Be("uv");
        storage.GetTrialSystemAttrs(t)["s"].Should().Be(9L);
    }

    [TestMethod]
    [TestCategory("Integration")]
    public void SetTrialStateValues_RunningToRunning_ReturnsFalse()
    {
        using var storage = NewStorage();
        int studyId = storage.CreateNewStudy(new[] { StudyDirection.Minimize }, "s");
        int t = storage.CreateNewTrial(studyId);
        storage.SetTrialStateValues(t, TrialState.Running).Should().BeFalse();
        storage.GetTrial(t).State.Should().Be(TrialState.Running);
    }

    [TestMethod]
    [TestCategory("Integration")]
    public void GetBestTrial_NoCompletedTrials_ThrowsValueError()
    {
        using var storage = NewStorage();
        int studyId = storage.CreateNewStudy(new[] { StudyDirection.Minimize }, "s");
        storage.CreateNewTrial(studyId); // running only
        Action act = () => storage.GetBestTrial(studyId);
        act.Should().Throw<ValueError>();
    }

    [TestMethod]
    [TestCategory("Integration")]
    public void GetBestTrial_MultiObjective_ThrowsRuntimeError()
    {
        using var storage = NewStorage();
        int studyId = storage.CreateNewStudy(new[] { StudyDirection.Minimize, StudyDirection.Minimize }, "s");
        int t = storage.CreateNewTrial(studyId);
        storage.SetTrialStateValues(t, TrialState.Complete, new[] { 1.0, 2.0 });
        Action act = () => storage.GetBestTrial(studyId);
        act.Should().Throw<RuntimeError>();
    }

    [TestMethod]
    [TestCategory("Integration")]
    public void MissingRecords_Throw_KeyError()
    {
        using var storage = NewStorage();
        int studyId = storage.CreateNewStudy(new[] { StudyDirection.Minimize }, "s");
        int t = storage.CreateNewTrial(studyId);

        ((Action)(() => storage.GetTrial(999999))).Should().Throw<KeyError>();
        ((Action)(() => storage.GetTrialParam(t, "nope"))).Should().Throw<KeyError>();
        ((Action)(() => storage.GetTrialIdFromStudyIdTrialNumber(studyId, 42))).Should().Throw<KeyError>();
    }

    [TestMethod]
    [TestCategory("Integration")]
    public void Distributions_AllTypes_RoundTrip()
    {
        using var storage = NewStorage();
        int studyId = storage.CreateNewStudy(new[] { StudyDirection.Minimize }, "s");
        int t = storage.CreateNewTrial(studyId);

        var fPlain = new FloatDistribution(-10, 10);
        var fLog = new FloatDistribution(1e-3, 1e3, log: true);
        var fStep = new FloatDistribution(0, 10, step: 0.5);
        var iPlain = new IntDistribution(0, 100);
        var iLog = new IntDistribution(1, 100, log: true);
        var iStep = new IntDistribution(0, 10, step: 2);
        var cat = new CategoricalDistribution(new CategoricalChoice[] { "a", "b", (long)3, true, CategoricalChoice.None });

        storage.SetTrialParam(t, "fPlain", 1.5, fPlain);
        storage.SetTrialParam(t, "fLog", 10.0, fLog);
        storage.SetTrialParam(t, "fStep", 2.5, fStep);
        storage.SetTrialParam(t, "iPlain", 42.0, iPlain);
        storage.SetTrialParam(t, "iLog", 4.0, iLog);
        storage.SetTrialParam(t, "iStep", 4.0, iStep);
        storage.SetTrialParam(t, "cat", 3.0, cat); // index 3 -> true

        var trial = storage.GetTrial(t);
        trial.Distributions["fPlain"].Should().Be(fPlain);
        trial.Distributions["fLog"].Should().Be(fLog);
        trial.Distributions["fStep"].Should().Be(fStep);
        trial.Distributions["iPlain"].Should().Be(iPlain);
        trial.Distributions["iLog"].Should().Be(iLog);
        trial.Distributions["iStep"].Should().Be(iStep);
        trial.Distributions["cat"].Should().Be(cat);
        trial.Params["fPlain"].Should().Be(1.5);
        trial.Params["iLog"].Should().Be(4L);
        trial.Params["cat"].Should().Be(true);
    }

    [TestMethod]
    [TestCategory("Integration")]
    public void AttrJson_ValueTypes_RoundTrip()
    {
        using var storage = NewStorage();
        int studyId = storage.CreateNewStudy(new[] { StudyDirection.Minimize }, "s");
        storage.SetStudySystemAttr(studyId, "str", "hello");
        storage.SetStudySystemAttr(studyId, "int", 42L);
        storage.SetStudySystemAttr(studyId, "float", 3.25);
        storage.SetStudySystemAttr(studyId, "bool", true);
        storage.SetStudySystemAttr(studyId, "null", null);
        storage.SetStudySystemAttr(studyId, "list", new List<object?> { 1L, "two", true });
        storage.SetStudySystemAttr(studyId, "dict", new OrderedDict<string, object?> { ["k"] = 1L, ["v"] = "x" });

        var a = storage.GetStudySystemAttrs(studyId);
        a["str"].Should().Be("hello");
        a["int"].Should().Be(42L);
        a["float"].Should().Be(3.25);
        a["bool"].Should().Be(true);
        a["null"].Should().BeNull();
        ((System.Collections.IEnumerable)a["list"]!).Cast<object?>().Should().Equal(1L, "two", true);
        ((OrderedDict<string, object?>)a["dict"]!)["v"].Should().Be("x");
    }

    [TestMethod]
    [TestCategory("Integration")]
    public void Datetime_StartAndComplete_RoundTrip()
    {
        using var storage = NewStorage();
        int studyId = storage.CreateNewStudy(new[] { StudyDirection.Minimize }, "s");
        int t = storage.CreateNewTrial(studyId);
        storage.GetTrial(t).DatetimeStart.Should().NotBeNull();
        storage.SetTrialStateValues(t, TrialState.Complete, new[] { 1.0 });

        var trial = storage.GetTrial(t);
        trial.DatetimeStart.Should().NotBeNull();
        trial.DatetimeComplete.Should().NotBeNull();
        trial.DatetimeComplete.Should().BeOnOrAfter(trial.DatetimeStart!.Value);
    }

    [TestMethod]
    [TestCategory("Integration")]
    public void ManyTrials_RoundTrip_Stable()
    {
        using var storage = NewStorage();
        int studyId = storage.CreateNewStudy(new[] { StudyDirection.Minimize }, "s");
        const int n = 120;
        for (int i = 0; i < n; i++)
        {
            int t = storage.CreateNewTrial(studyId);
            storage.SetTrialParam(t, "x", i, new IntDistribution(0, n));
            storage.SetTrialStateValues(t, TrialState.Complete, new[] { (double)i });
        }

        var trials = storage.GetAllTrials(studyId);
        trials.Should().HaveCount(n);
        trials[0].Number.Should().Be(0);
        trials[^1].Number.Should().Be(n - 1);
        trials[^1].Value.Should().Be(n - 1);
        storage.GetBestTrial(studyId).Value.Should().Be(0.0);
    }

    [TestMethod]
    [TestCategory("Integration")]
    public void Heartbeat_Interval_And_IsHeartbeatEnabled()
    {
        using var enabled = NewStorage(new EFCoreRdbOptions { HeartbeatInterval = 3 });
        enabled.GetHeartbeatInterval().Should().Be(3);
        HeartbeatHelpers.IsHeartbeatEnabled(enabled).Should().BeTrue();

        using var disabled = NewStorage();
        disabled.GetHeartbeatInterval().Should().BeNull();
        HeartbeatHelpers.IsHeartbeatEnabled(disabled).Should().BeFalse();
    }

    [TestMethod]
    [TestCategory("Integration")]
    [DataRow(0)]
    [DataRow(-1)]
    public void Heartbeat_Interval_MustBePositive(int interval)
    {
        Action act = () => _ = new RDBStorage(Provider, ConnectionString, new EFCoreRdbOptions { HeartbeatInterval = interval });
        act.Should().Throw<ValueError>()
            .WithMessage("The value of `heartbeat_interval` should be a positive integer.");
    }

    [TestMethod]
    [TestCategory("Integration")]
    [DataRow(0)]
    [DataRow(-5)]
    public void Heartbeat_GracePeriod_MustBePositive(int grace)
    {
        Action act = () => _ = new RDBStorage(
            Provider, ConnectionString, new EFCoreRdbOptions { HeartbeatInterval = 1, GracePeriod = grace });
        act.Should().Throw<ValueError>()
            .WithMessage("The value of `grace_period` should be a positive integer.");
    }

    [TestMethod]
    [TestCategory("Integration")]
    public void Heartbeat_SpecifyingBothCallbacks_Throws()
    {
        Action<Study, FrozenTrial> stale = (s, t) => { };
        Action<Study, FrozenTrial> failed = (s, t) => { };
        Action act = () => _ = new RDBStorage(Provider, ConnectionString, new EFCoreRdbOptions
        {
            HeartbeatInterval = 1,
            HeartbeatStaleTrialCallback = stale,
            FailedTrialCallback = failed,
        });
        act.Should().Throw<ValueError>()
            .WithMessage("Specify only one of `heartbeat_stale_trial_callback` and `failed_trial_callback`.");
    }

    // [Misaligned] Optuna raises an ExperimentalWarning (warn_experimental_argument) for
    // heartbeat_interval. OptunaSharp targets the Optuna 5.x surface and treats it as stable, so no
    // ExperimentalWarning is emitted — this guards that deliberate divergence.
    [TestMethod]
    [TestCategory("Integration")]
    [TestCategory("Misaligned")]
    public void Heartbeat_Interval_DoesNotEmitExperimentalWarning()
    {
        var warnings = CaptureWarnings(() =>
        {
            using var _ = new RDBStorage(Provider, ConnectionString, new EFCoreRdbOptions { HeartbeatInterval = 2 });
        });

        warnings.Should().NotContain(w => w.Category == WarningCategory.ExperimentalWarning);
    }

    [TestMethod]
    [TestCategory("Integration")]
    public void Heartbeat_FailedTrialCallback_IsDeprecatedAlias_And_Warns()
    {
        Action<Study, FrozenTrial> cb = (s, t) => { };

        RDBStorage storage = null!;
        var warnings = CaptureWarnings(() =>
        {
            storage = new RDBStorage(Provider, ConnectionString, new EFCoreRdbOptions
            {
                HeartbeatInterval = 1,
                FailedTrialCallback = cb,
            });
        });

        using (storage)
        {
            storage.GetHeartbeatStaleTrialCallback().Should().BeSameAs(cb);
            warnings.Should().Contain(w =>
                w.Category == WarningCategory.FutureWarning &&
                w.Message.Contains("`failed_trial_callback`") && w.Message.Contains("deprecated in v4.9.0"));
        }
    }

    [TestMethod]
    [TestCategory("Integration")]
    public void Heartbeat_RecordHeartbeat_Upserts_SingleRow()
    {
        using var storage = NewStorage(new EFCoreRdbOptions { HeartbeatInterval = 1 });
        int studyId = storage.CreateNewStudy(new[] { StudyDirection.Minimize }, "s");
        int t = storage.CreateNewTrial(studyId);

        HeartbeatRowCount().Should().Be(0);
        storage.RecordHeartbeat(t);
        HeartbeatRowCount().Should().Be(1);

        // A second beat updates the one row (UNIQUE(trial_id) upsert) — no duplicate.
        storage.RecordHeartbeat(t);
        HeartbeatRowCount().Should().Be(1);
    }

    [TestMethod]
    [TestCategory("Integration")]
    public void Heartbeat_GetStaleTrialIds_FreshNotStale_BackdatedStale()
    {
        using var storage = NewStorage(new EFCoreRdbOptions { HeartbeatInterval = 2 });
        storage.GetHeartbeatInterval().Should().Be(2);
        int studyId = storage.CreateNewStudy(new[] { StudyDirection.Minimize }, "s");
        int t = storage.CreateNewTrial(studyId);

        storage.RecordHeartbeat(t);
        storage.GetStaleTrialIds(studyId).Should().BeEmpty("a fresh heartbeat is not stale");

        // Backdate well past the grace period (2 * interval = 4s) to exercise the elapsed-seconds SQL.
        BackdateHeartbeat(t, 3600);
        storage.GetStaleTrialIds(studyId).Should().Contain(t);
    }

    [TestMethod]
    [TestCategory("Integration")]
    public void Heartbeat_GetStaleTrialIds_Ignores_NonRunning_And_TrialsWithoutHeartbeat()
    {
        using var storage = NewStorage(new EFCoreRdbOptions { HeartbeatInterval = 1 });
        int studyId = storage.CreateNewStudy(new[] { StudyDirection.Minimize }, "s");

        int running = storage.CreateNewTrial(studyId);
        storage.RecordHeartbeat(running);
        BackdateHeartbeat(running, 3600);

        int finished = storage.CreateNewTrial(studyId);
        storage.RecordHeartbeat(finished);
        BackdateHeartbeat(finished, 3600);
        storage.SetTrialStateValues(finished, TrialState.Complete, new[] { 1.0 });

        int noBeat = storage.CreateNewTrial(studyId);

        storage.GetStaleTrialIds(studyId).Should().Equal(running);
        _ = noBeat;
    }

    [TestMethod]
    [TestCategory("Integration")]
    public void Heartbeat_GetStaleTrialIds_DefaultGracePeriod_Is_TwiceInterval()
    {
        using var storage = NewStorage(new EFCoreRdbOptions { HeartbeatInterval = 5 });
        int studyId = storage.CreateNewStudy(new[] { StudyDirection.Minimize }, "s");
        int t = storage.CreateNewTrial(studyId);
        storage.RecordHeartbeat(t);

        // Within 2*interval (=10s): not stale.
        BackdateHeartbeat(t, 6);
        storage.GetStaleTrialIds(studyId).Should().BeEmpty("6s < the 10s default grace period");

        // Beyond 2*interval: stale.
        BackdateHeartbeat(t, 15);
        storage.GetStaleTrialIds(studyId).Should().Contain(t);
    }

    [TestMethod]
    [TestCategory("Integration")]
    public void Heartbeat_GetStaleTrialIds_ScopedToStudy()
    {
        using var storage = NewStorage(new EFCoreRdbOptions { HeartbeatInterval = 1 });
        int studyA = storage.CreateNewStudy(new[] { StudyDirection.Minimize }, "a");
        int studyB = storage.CreateNewStudy(new[] { StudyDirection.Minimize }, "b");
        int tA = storage.CreateNewTrial(studyA);
        int tB = storage.CreateNewTrial(studyB);
        storage.RecordHeartbeat(tA);
        storage.RecordHeartbeat(tB);
        BackdateHeartbeat(tA, 3600);
        BackdateHeartbeat(tB, 3600);

        storage.GetStaleTrialIds(studyA).Should().Equal(tA);
        storage.GetStaleTrialIds(studyB).Should().Equal(tB);
    }

    [TestMethod]
    [TestCategory("Integration")]
    public void Heartbeat_FailStaleTrials_EndToEnd_FailsStale_And_InvokesCallback()
    {
        var callbackTrials = new List<FrozenTrial>();
        using var storage = NewStorage(new EFCoreRdbOptions
        {
            HeartbeatInterval = 1,
            HeartbeatStaleTrialCallback = (s, t) => callbackTrials.Add(t),
        });
        var study = Optuna.CreateStudy(storage: storage, studyName: "s", sampler: new RandomSampler(0));

        int staleId = study.Storage.CreateNewTrial(study.StudyId);
        int healthyId = study.Storage.CreateNewTrial(study.StudyId);
        storage.RecordHeartbeat(staleId);
        storage.RecordHeartbeat(healthyId);
        BackdateHeartbeat(staleId, 3600);

        HeartbeatHelpers.FailStaleTrials(study);

        storage.GetTrial(staleId).State.Should().Be(TrialState.Fail);
        storage.GetTrial(healthyId).State.Should().Be(TrialState.Running);
        callbackTrials.Should().ContainSingle(t => t.TrialId == staleId);

        // Already-failed stale ids are tolerated on a second pass (UpdateFinishedTrialError swallowed).
        HeartbeatHelpers.FailStaleTrials(study);
        storage.GetTrial(staleId).State.Should().Be(TrialState.Fail);
    }

    [TestMethod]
    [TestCategory("Integration")]
    public void Heartbeat_RetryCallback_EndToEnd_Requeues_Waiting_Trial()
    {
        using var storage = NewStorage(new EFCoreRdbOptions
        {
            HeartbeatInterval = 1,
            HeartbeatStaleTrialCallback = new RetryHeartbeatStaleTrialCallback(maxRetry: 3),
        });
        var study = Optuna.CreateStudy(storage: storage, studyName: "s", sampler: new RandomSampler(0));

        int staleId = study.Storage.CreateNewTrial(study.StudyId);
        study.Storage.SetTrialParam(staleId, "x", 1.5, new FloatDistribution(-10, 10));
        storage.RecordHeartbeat(staleId);
        BackdateHeartbeat(staleId, 3600);

        HeartbeatHelpers.FailStaleTrials(study);

        storage.GetTrial(staleId).State.Should().Be(TrialState.Fail);
        var requeued = study.GetTrials(deepcopy: false).Single(t => t.State == TrialState.Waiting);
        requeued.Params["x"].Should().Be(1.5);
        RetryHeartbeatStaleTrialCallback.RetriedTrialNumber(requeued).Should().Be(storage.GetTrial(staleId).Number);
        RetryHeartbeatStaleTrialCallback.RetryHistory(requeued).Should().Equal(storage.GetTrial(staleId).Number);
    }

    // ------------------------------------------------------------------ golden-standard API (shared)

    [TestMethod]
    [TestCategory("Integration")]
    public void ConfigureDbContext_LogTo_CapturesSql()
    {
        var log = new System.Collections.Concurrent.ConcurrentBag<string>();
        using var storage = NewStorage(new EFCoreRdbOptions { ConfigureDbContext = b => b.LogTo(log.Add) });
        storage.CreateNewStudy(new[] { StudyDirection.Minimize }, "logged");
        log.Should().NotBeEmpty("the EF Core logger hook must receive events");
    }

    [TestMethod]
    [TestCategory("Integration")]
    public void RetryOnFailure_TransactionsStillWork()
    {
        // EnableRetryOnFailure installs a retrying execution strategy, which forbids user-initiated
        // transactions unless they run inside the strategy — this storage wraps them, so create + tell
        // (both transactional) must still succeed.
        using var storage = NewStorage(RetryOptions());
        var study = Optuna.CreateStudy(storage: storage, studyName: "retry", sampler: new RandomSampler(seed: 0));
        study.Optimize(t => { double x = t.SuggestFloat("x", -10, 10); return x * x; }, nTrials: 5);
        study.Trials.Should().HaveCount(5);
    }

    // ------------------------------------------------------------------ helpers

    private static string? Scalar(DbConnection conn, string sql)
    {
        using var cmd = conn.CreateCommand();
        cmd.CommandText = sql;
        object? value = cmd.ExecuteScalar();
        return value is null or DBNull ? null : Convert.ToString(value, CultureInfo.InvariantCulture);
    }

    /// <summary>Rows currently in <c>trial_heartbeats</c>.</summary>
    private int HeartbeatRowCount()
    {
        using var conn = RawConn();
        return int.Parse(Scalar(conn, "SELECT COUNT(*) FROM trial_heartbeats")!, CultureInfo.InvariantCulture);
    }

    /// <summary>
    ///     Push a trial's heartbeat <paramref name="seconds" /> into the past (on the DB clock) so it is
    ///     older than the grace period, exercising the provider-specific elapsed-seconds SQL in
    ///     <see cref="RDBStorage.GetStaleTrialIds" />.
    /// </summary>
    private void BackdateHeartbeat(int trialId, int seconds)
    {
        using var conn = RawConn();
        using var cmd = conn.CreateCommand();
        cmd.CommandText = Provider == RdbProvider.PostgreSql
            ? $"UPDATE trial_heartbeats SET heartbeat = CURRENT_TIMESTAMP - INTERVAL '{seconds} seconds' WHERE trial_id = {trialId}"
            : $"UPDATE trial_heartbeats SET heartbeat = CURRENT_TIMESTAMP - INTERVAL {seconds} SECOND WHERE trial_id = {trialId}";
        cmd.ExecuteNonQuery();
    }

    /// <summary>Capture every <see cref="OptunaWarnings" /> raised while <paramref name="action" /> runs.</summary>
    private static List<(string Message, WarningCategory Category)> CaptureWarnings(Action action)
    {
        var captured = new List<(string, WarningCategory)>();
        void Handler(string message, WarningCategory category)
        {
            lock (captured)
                captured.Add((message, category));
        }

        OptunaWarnings.WarningRaised += Handler;
        try
        {
            action();
        }
        finally
        {
            OptunaWarnings.WarningRaised -= Handler;
        }
        return captured;
    }

    private bool TableExists(DbConnection conn, string table)
    {
        string sql = Provider == RdbProvider.PostgreSql
            ? "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema='public' AND table_name=@t"
            : "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema=DATABASE() AND table_name=@t";
        using var cmd = conn.CreateCommand();
        cmd.CommandText = sql;
        AddParam(cmd, "@t", table);
        return Convert.ToInt64(cmd.ExecuteScalar(), CultureInfo.InvariantCulture) > 0;
    }

    private string ColumnType(DbConnection conn, string table, string column)
    {
        // PostgreSQL: data_type ('USER-DEFINED' for enums, 'timestamp without time zone', ...).
        // MySQL: column_type ('enum(...)', 'double', 'datetime', ...).
        string col = Provider == RdbProvider.PostgreSql ? "data_type" : "column_type";
        using var cmd = conn.CreateCommand();
        cmd.CommandText =
            $"SELECT {col} FROM information_schema.columns WHERE table_name=@t AND column_name=@c" +
            (Provider == RdbProvider.PostgreSql ? " AND table_schema='public'" : " AND table_schema=DATABASE()");
        AddParam(cmd, "@t", table);
        AddParam(cmd, "@c", column);
        return Convert.ToString(cmd.ExecuteScalar(), CultureInfo.InvariantCulture)!;
    }

    private static List<string> EnumLabels(DbConnection conn, string typeName)
    {
        using var cmd = conn.CreateCommand();
        cmd.CommandText =
            "SELECT e.enumlabel FROM pg_enum e JOIN pg_type t ON t.oid = e.enumtypid " +
            "WHERE t.typname = @n ORDER BY e.enumsortorder";
        AddParam(cmd, "@n", typeName);
        var labels = new List<string>();
        using var r = cmd.ExecuteReader();
        while (r.Read())
            labels.Add(r.GetString(0));
        return labels;
    }

    private static void AddParam(DbCommand cmd, string name, object value)
    {
        var p = cmd.CreateParameter();
        p.ParameterName = name;
        p.Value = value;
        cmd.Parameters.Add(p);
    }
}

/// <summary>The conformance suite against a live PostgreSQL server (Testcontainers).</summary>
[TestClass]
public sealed class PostgreSqlRdbStorageTests : EFCoreRdbStorageTestsBase
{
    protected override RdbProvider Provider => RdbProvider.PostgreSql;

    protected override EFCoreRdbOptions RetryOptions() =>
        new() { ConfigureNpgsql = o => o.EnableRetryOnFailure() };

    [TestMethod]
    [TestCategory("Integration")]
    public void NativeConnectionStringBuilder_Ctor_Works()
    {
        EFCoreDbSession.Reset(RdbProvider.PostgreSql);
        var builder = new NpgsqlConnectionStringBuilder(ConnectionString);
        using var storage = new RDBStorage(builder);
        int id = storage.CreateNewStudy(new[] { StudyDirection.Minimize }, "npgb");
        storage.GetStudyNameFromId(id).Should().Be("npgb");
    }

    [TestMethod]
    [TestCategory("Integration")]
    public void ByoDataSource_WithMapOptunaEnums_Works_AndRespectsOwnership()
    {
        EFCoreDbSession.Reset(RdbProvider.PostgreSql);
        var dsb = new NpgsqlDataSourceBuilder(ConnectionString);
        RDBStorage.MapOptunaEnums(dsb);   // the golden-standard "build your own data source" path
        var dataSource = dsb.Build();
        using (var storage = new RDBStorage(dataSource, ownsDataSource: false))
        {
            var study = Optuna.CreateStudy(storage: storage, studyName: "byods", sampler: new RandomSampler(seed: 0));
            study.Optimize(t => { double x = t.SuggestFloat("x", -5, 5); return x * x; }, nTrials: 4);
            study.Trials.Should().HaveCount(4);
        }

        // ownsDataSource:false ⇒ Dispose left the data source usable.
        using var conn = dataSource.CreateConnection();
        conn.Open();
        conn.State.Should().Be(System.Data.ConnectionState.Open);
        dataSource.Dispose();
    }

    [TestMethod]
    [TestCategory("Integration")]
    public void ConfigureConnectionString_ApplicationName_ReachesTheServer()
    {
        EFCoreDbSession.Reset(RdbProvider.PostgreSql);
        using var storage = new RDBStorage(RdbProvider.PostgreSql, ConnectionString, new EFCoreRdbOptions
        {
            ConfigureConnectionString = b => ((NpgsqlConnectionStringBuilder)b).ApplicationName = "optunasharp-golden",
        });
        storage.CreateNewStudy(new[] { StudyDirection.Minimize }, "app"); // opens a pooled connection

        // The pooled connection stays open with our application_name; a separate raw connection sees it.
        using var conn = RawConn();
        using var cmd = conn.CreateCommand();
        cmd.CommandText = "SELECT COUNT(*) FROM pg_stat_activity WHERE application_name = 'optunasharp-golden'";
        Convert.ToInt64(cmd.ExecuteScalar(), System.Globalization.CultureInfo.InvariantCulture).Should().BeGreaterThan(0);
    }
}

/// <summary>The conformance suite against a live MySQL server (Testcontainers).</summary>
[TestClass]
public sealed class MySqlRdbStorageTests : EFCoreRdbStorageTestsBase
{
    protected override RdbProvider Provider => RdbProvider.MySql;

    protected override EFCoreRdbOptions RetryOptions() =>
        new() { ConfigureMySql = o => o.EnableRetryOnFailure() };

    [TestMethod]
    [TestCategory("Integration")]
    public void NativeConnectionStringBuilder_Ctor_Works()
    {
        EFCoreDbSession.Reset(RdbProvider.MySql);
        var builder = new MySqlConnectionStringBuilder(ConnectionString);
        using var storage = new RDBStorage(builder);
        int id = storage.CreateNewStudy(new[] { StudyDirection.Minimize }, "myb");
        storage.GetStudyNameFromId(id).Should().Be("myb");
    }

    [TestMethod]
    [TestCategory("Integration")]
    public void ExplicitServerVersion_SkipsAutoDetectProbe()
    {
        EFCoreDbSession.Reset(RdbProvider.MySql);
        ServerVersion version = ServerVersion.AutoDetect(ConnectionString); // detect once, then pin it
        using var storage = new RDBStorage(RdbProvider.MySql, ConnectionString,
            new EFCoreRdbOptions { MySqlServerVersion = version });
        int id = storage.CreateNewStudy(new[] { StudyDirection.Minimize }, "ver");
        storage.GetStudyNameFromId(id).Should().Be("ver");
    }
}
