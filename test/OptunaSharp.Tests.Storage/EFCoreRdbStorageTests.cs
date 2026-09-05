using System;
using System.Collections.Generic;
using System.Data.Common;
using System.Globalization;
using System.Linq;
using AwesomeAssertions;
using MySqlConnector;
using Npgsql;
using NumSharp;
using OptunaSharp;
using OptunaSharp.Collections;
using OptunaSharp.Distributions;
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

    // A second storage over the same database WITHOUT resetting the schema (reopen/persistence tests).
    private RDBStorage ReopenStorage() => new(Provider, EFCoreDbSession.ConnectionString(Provider));

    private DbConnection RawConn()
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

    // ------------------------------------------------------------------ helpers

    private static string? Scalar(DbConnection conn, string sql)
    {
        using var cmd = conn.CreateCommand();
        cmd.CommandText = sql;
        object? value = cmd.ExecuteScalar();
        return value is null or DBNull ? null : Convert.ToString(value, CultureInfo.InvariantCulture);
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
}

/// <summary>The conformance suite against a live MySQL server (Testcontainers).</summary>
[TestClass]
public sealed class MySqlRdbStorageTests : EFCoreRdbStorageTestsBase
{
    protected override RdbProvider Provider => RdbProvider.MySql;
}
