using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading;
using AwesomeAssertions;
using Microsoft.Data.Sqlite;
using NumSharp;
using OptunaSharp;
using OptunaSharp.Collections;
using OptunaSharp.Distributions;
using OptunaSharp.Internal;
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

    // ------------------------------------------------------------------ plan 13: bulk attribute writes (S8)

    /// <summary>
    ///     Plan 13, item S8: a batch of trial and study attributes on SQLite leaves exactly what one single write per
    ///     entry leaves (keys, values and their JSON), read back through a fresh storage on the same file.
    /// </summary>
    [TestMethod]
    public void BulkAttributes_LeaveWhatSingleWritesLeave()
    {
        string path = TempDbPath();
        string url = $"sqlite:///{path.Replace('\\', '/')}";
        var attrs = new OrderedDict<string, object?>
        {
            ["seed"] = 7L, ["ratio"] = 0.25, ["name"] = "run", ["flag"] = true, ["none"] = null,
            ["list"] = new List<object?> { 1L, "two" },
        };
        int bulkStudy, singleStudy, bulkTrial, singleTrial;
        using (var storage = new RDBStorage(url))
        {
            bulkStudy = storage.CreateNewStudy(new[] { StudyDirection.Minimize }, "bulk");
            singleStudy = storage.CreateNewStudy(new[] { StudyDirection.Minimize }, "single");
            bulkTrial = storage.CreateNewTrial(bulkStudy);
            singleTrial = storage.CreateNewTrial(singleStudy);
            storage.SetTrialUserAttr(bulkTrial, "name", "earlier");
            storage.SetTrialUserAttr(singleTrial, "name", "earlier");

            storage.SetTrialUserAttrs(bulkTrial, attrs);
            storage.SetTrialSystemAttrs(bulkTrial, attrs);
            storage.SetStudyUserAttrs(bulkStudy, attrs);
            storage.SetStudySystemAttrs(bulkStudy, attrs);
            foreach (var (key, value) in attrs)
            {
                storage.SetTrialUserAttr(singleTrial, key, value);
                storage.SetTrialSystemAttr(singleTrial, key, value);
                storage.SetStudyUserAttr(singleStudy, key, value);
                storage.SetStudySystemAttr(singleStudy, key, value);
            }
        }

        using var reopened = new RDBStorage(url);
        static string Describe(IReadOnlyDictionary<string, object?> a)
            => string.Join("; ", a.OrderBy(kv => kv.Key, StringComparer.Ordinal).Select(kv => $"{kv.Key}={PyJson.Dumps(kv.Value)}"));
        Describe(reopened.GetTrial(bulkTrial).UserAttrs).Should().Be(Describe(reopened.GetTrial(singleTrial).UserAttrs));
        Describe(reopened.GetTrial(bulkTrial).SystemAttrs).Should().Be(Describe(reopened.GetTrial(singleTrial).SystemAttrs));
        Describe(reopened.GetStudyUserAttrs(bulkStudy)).Should().Be(Describe(reopened.GetStudyUserAttrs(singleStudy)));
        Describe(reopened.GetStudySystemAttrs(bulkStudy)).Should().Be(Describe(reopened.GetStudySystemAttrs(singleStudy)));
        reopened.GetTrial(bulkTrial).UserAttrs["name"].Should().Be("run", "the batch overwrites an earlier value");
    }

    /// <summary>
    ///     A batch is one transaction: a value that cannot be serialized, in the middle of the batch, rolls back the
    ///     entries before it; a finished trial and an unknown study refuse the batch with nothing written.
    /// </summary>
    [TestMethod]
    public void BulkAttributes_AreAllOrNothing()
    {
        using RDBStorage storage = NewMemoryStorage();
        int studyId = storage.CreateNewStudy(new[] { StudyDirection.Minimize }, "atomic");
        int trialId = storage.CreateNewTrial(studyId);
        var poisoned = new OrderedDict<string, object?> { ["ok"] = 1L, ["bad"] = new object(), ["after"] = 2L };

        Action trialBatch = () => storage.SetTrialUserAttrs(trialId, poisoned);
        Action studyBatch = () => storage.SetStudyUserAttrs(studyId, poisoned);
        trialBatch.Should().Throw<TypeError>();
        studyBatch.Should().Throw<TypeError>();
        storage.GetTrial(trialId).UserAttrs.Should().BeEmpty("the transaction rolled the first entry back");
        storage.GetStudyUserAttrs(studyId).Should().BeEmpty();

        storage.SetTrialStateValues(trialId, TrialState.Complete, new[] { 1.0 });
        Action finished = () => storage.SetTrialSystemAttrs(trialId, new OrderedDict<string, object?> { ["x"] = 1L });
        Action unknownStudy = () => storage.SetStudySystemAttrs(424242, new OrderedDict<string, object?> { ["x"] = 1L });
        finished.Should().Throw<UpdateFinishedTrialError>();
        unknownStudy.Should().Throw<KeyError>();
        storage.GetTrial(trialId).SystemAttrs.Should().NotContainKey("x");
    }

    /// <summary>
    ///     A study on SQLite runs through <see cref="CachedStorage" />, which forwards a trial's and a study's batch to the
    ///     backend whole; the trial and the study see their attributes, and so does a fresh storage.
    /// </summary>
    [TestMethod]
    public void BulkAttributes_ThroughCachedStorage_ReachTheDatabase()
    {
        string path = TempDbPath();
        string url = $"sqlite:///{path.Replace('\\', '/')}";
        var rdb = new RDBStorage(url);
        Study study = Optuna.CreateStudy(storage: rdb, studyName: "cached");
        study.Storage.Should().BeOfType<CachedStorage>();
        Trial trial = study.Ask();
        trial.SetUserAttrs(new OrderedDict<string, object?> { ["a"] = 1L, ["b"] = "two" });
        study.SetUserAttr(new OrderedDict<string, object?> { ["c"] = 3.5 });
        study.Tell(trial, 1.0);

        trial.UserAttrs.Keys.Should().Contain(new[] { "a", "b" });
        study.UserAttrs["c"].Should().Be(3.5);
        rdb.Dispose();
        using var reopened = new RDBStorage(url);
        int studyId = reopened.GetStudyIdFromName("cached");
        reopened.GetAllTrials(studyId)[0].UserAttrs.Should().Contain(new KeyValuePair<string, object?>("b", "two"));
        reopened.GetStudyUserAttrs(studyId)["c"].Should().Be(3.5);
    }

    // ------------------------------------------------------------------ plan 13: one parse per distribution text

    /// <summary>
    ///     Plan 13 (docs/plans/13-rustuna-derived-performance.md), item S9 as measured: a load parses each distinct
    ///     distribution text once. Reloading a study gives every trial of a parameter the same distribution object, equal
    ///     to a fresh parse of its stored text; a different text (a -0.0 bound, another range, a categorical) is its own
    ///     object, and the deprecated types are never shared, so their warnings fire per use as before.
    /// </summary>
    [TestMethod]
    public void Reload_ParsesEachDistributionTextOnce_SharingOnlyTheCurrentTypes()
    {
        string path = TempDbPath();
        string url = $"sqlite:///{path.Replace('\\', '/')}";
        int studyId;
        using (var storage = new RDBStorage(url))
        {
            studyId = storage.CreateNewStudy(new[] { StudyDirection.Minimize }, "shared-distributions");
            for (int n = 0; n < 6; n++)
            {
                int id = storage.CreateNewTrial(studyId);
                storage.SetTrialParam(id, "x", 0.25 * n, new FloatDistribution(0, 2));
                storage.SetTrialParam(id, "c", n % 2, new CategoricalDistribution(new CategoricalChoice[] { "a", "b" }));
                storage.SetTrialParam(id, "z", 0.5, n < 3 ? new FloatDistribution(0.0, 1) : new FloatDistribution(-0.0, 1));
#pragma warning disable CS0618 // the deprecated family is the point of this case
                storage.SetTrialParam(id, "u", 0.5, new UniformDistribution(0, 1));
#pragma warning restore CS0618
                storage.SetTrialStateValues(id, TrialState.Complete, new[] { (double)n });
            }
        }

        using var reopened = new RDBStorage(url);
        List<FrozenTrial> trials = reopened.GetAllTrials(studyId, deepcopy: false);
        trials.Should().HaveCount(6);
        foreach (FrozenTrial trial in trials)
        {
            trial.Distributions["x"].Should().BeSameAs(trials[0].Distributions["x"]);
            trial.Distributions["c"].Should().BeSameAs(trials[0].Distributions["c"]);
            foreach (var (name, distribution) in trial.Distributions)
                distribution.Should().Be(Distribution.FromJson(Distribution.ToJson(distribution)), name);
        }
        trials[0].Distributions["z"].Should().BeSameAs(trials[2].Distributions["z"]);
        trials[3].Distributions["z"].Should().BeSameAs(trials[5].Distributions["z"]);
        trials[0].Distributions["z"].Should().NotBeSameAs(trials[3].Distributions["z"], "-0.0 is stored as its own text");
        trials[3].Distributions["z"].ToString().Should().Contain("low=-0.0");
        trials[0].Distributions["u"].Should().NotBeSameAs(trials[1].Distributions["u"], "deprecated types are never shared");
        trials.Select(t => t.Params["x"]).Should().Equal(0.0, 0.25, 0.5, 0.75, 1.0, 1.25);
        trials.Select(t => t.Params["c"]).Should().Equal("a", "b", "a", "b", "a", "b");
    }

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
            intermediateValues: new OrderedDict<long, double> { [0] = 0.5 });

        int trialId = storage.CreateNewTrial(studyId, template);
        var trial = storage.GetTrial(trialId);
        trial.State.Should().Be(TrialState.Complete);
        trial.Value.Should().Be(2.25);
        trial.Params["x"].Should().Be(1.5);
        trial.UserAttrs["note"].Should().Be("hi");
        trial.IntermediateValues[0].Should().Be(0.5);
        trial.Number.Should().Be(0);
    }

    /// <summary>
    ///     Review item B-2 (pass 2B): SQLite's <c>INTEGER</c> step column holds 64 bits, and Python stores any int
    ///     step that fits. A row written as Python writes it (step 3,000,000,000, inserted with plain SQL here) used
    ///     to come back wrapped to -1,294,967,296 through an unchecked cast, and a C# process copying the trial would
    ///     have persisted the corruption. It must read back exactly, and C#'s own 64-bit steps must round-trip.
    /// </summary>
    [TestMethod]
    public void IntermediateSteps_PastInt32_ReadAndWriteExactly()
    {
        const long pastInt32 = 3_000_000_000L;
        string path = TempDbPath();
        int studyId, trialId;
        using (var storage = new RDBStorage($"sqlite:///{path.Replace('\\', '/')}"))
        {
            studyId = storage.CreateNewStudy(new[] { StudyDirection.Minimize }, "s");
            trialId = storage.CreateNewTrial(studyId);
            storage.SetTrialIntermediateValue(trialId, long.MaxValue, 0.75);
        }

        using (var conn = new SqliteConnection(new SqliteConnectionStringBuilder { DataSource = path }.ToString()))
        {
            conn.Open();
            using var insert = conn.CreateCommand();
            insert.CommandText =
                "INSERT INTO trial_intermediate_values (trial_id, step, intermediate_value, intermediate_value_type) " +
                "VALUES ($trial, $step, 0.5, 'FINITE')";
            insert.Parameters.AddWithValue("$trial", trialId);
            insert.Parameters.AddWithValue("$step", pastInt32);
            insert.ExecuteNonQuery();
        }
        SqliteConnection.ClearAllPools();

        using var reopened = new RDBStorage($"sqlite:///{path.Replace('\\', '/')}");
        var trial = reopened.GetTrial(trialId);

        trial.IntermediateValues.Should().Equal(new Dictionary<long, double> { [long.MaxValue] = 0.75, [pastInt32] = 0.5 });
        trial.LastStep.Should().Be(long.MaxValue);
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

    /// <summary>
    ///     Regression: <see cref="Optuna.LoadStudy" /> and the <see cref="Study" /> constructor wrap an RDB
    ///     backend in <see cref="CachedStorage" />, as Python's <c>Study.__init__</c> does through
    ///     <c>get_storage</c>. Before, only <see cref="Optuna.CreateStudy" /> wrapped, so a distributed worker
    ///     that loaded the shared study re-read every trial from the database on each sampler call.
    /// </summary>
    [TestMethod]
    public void LoadStudy_And_StudyConstructor_WrapRdbStorage_InCachedStorage()
    {
        using var rdb = NewMemoryStorage();
        Optuna.CreateStudy(storage: rdb, studyName: "s");

        var loaded = Optuna.LoadStudy("s", rdb, sampler: new RandomSampler(seed: 0));
        loaded.Storage.Should().BeOfType<CachedStorage>();
        ((CachedStorage)loaded.Storage).Backend.Should().BeSameAs(rdb);

        var constructed = new Study("s", rdb);
        constructed.Storage.Should().BeOfType<CachedStorage>();
        ((CachedStorage)constructed.Storage).Backend.Should().BeSameAs(rdb);

        // An already-wrapped storage passes through unchanged, which is why CreateStudy never double-wraps.
        var cached = new CachedStorage(rdb);
        new Study("s", cached).Storage.Should().BeSameAs(cached);

        // A loaded study optimizes through its cache, and its trials reach the database.
        loaded.Optimize(t => t.SuggestFloat("x", -1, 1), nTrials: 3);
        rdb.GetAllTrials(rdb.GetStudyIdFromName("s")).Should().HaveCount(3);
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

    // ------------------------------------------------------------------ heartbeat (plan 1.3 over RDB)

    [TestMethod]
    public void Heartbeat_Interval_And_IsHeartbeatEnabled()
    {
        using var enabled = new RDBStorage("sqlite:///:memory:", new SqliteRdbOptions { HeartbeatInterval = 3 });
        enabled.GetHeartbeatInterval().Should().Be(3);
        HeartbeatHelpers.IsHeartbeatEnabled(enabled).Should().BeTrue();

        using var disabled = NewMemoryStorage();
        disabled.GetHeartbeatInterval().Should().BeNull();
        HeartbeatHelpers.IsHeartbeatEnabled(disabled).Should().BeFalse();
    }

    [TestMethod]
    [DataRow(0)]
    [DataRow(-1)]
    public void Heartbeat_Interval_MustBePositive(int interval)
    {
        Action act = () => _ = new RDBStorage("sqlite:///:memory:", new SqliteRdbOptions { HeartbeatInterval = interval });
        act.Should().Throw<ValueError>()
            .WithMessage("The value of `heartbeat_interval` should be a positive integer.");
    }

    [TestMethod]
    [DataRow(0)]
    [DataRow(-5)]
    public void Heartbeat_GracePeriod_MustBePositive(int grace)
    {
        Action act = () => _ = new RDBStorage(
            "sqlite:///:memory:", new SqliteRdbOptions { HeartbeatInterval = 1, GracePeriod = grace });
        act.Should().Throw<ValueError>()
            .WithMessage("The value of `grace_period` should be a positive integer.");
    }

    // [Misaligned] Optuna raises an ExperimentalWarning (warn_experimental_argument) for
    // heartbeat_interval. OptunaSharp targets the Optuna 5.x surface and treats it as stable, so no
    // ExperimentalWarning is emitted — this guards that deliberate divergence.
    [TestMethod]
    [TestCategory("Misaligned")]
    public void Heartbeat_Interval_DoesNotEmitExperimentalWarning()
    {
        var warnings = CaptureWarnings(() =>
        {
            using var _ = new RDBStorage("sqlite:///:memory:", new SqliteRdbOptions { HeartbeatInterval = 2 });
        });

        warnings.Should().NotContain(w => w.Category == WarningCategory.ExperimentalWarning);
    }

    [TestMethod]
    public void Heartbeat_NoInterval_DoesNotWarnExperimental()
    {
        var warnings = CaptureWarnings(() =>
        {
            using var _ = NewMemoryStorage();
        });

        warnings.Should().NotContain(w => w.Category == WarningCategory.ExperimentalWarning);
    }

    [TestMethod]
    public void Heartbeat_SpecifyingBothCallbacks_Throws()
    {
        Action<Study, FrozenTrial> stale = (s, t) => { };
        Action<Study, FrozenTrial> failed = (s, t) => { };
        Action act = () => _ = new RDBStorage("sqlite:///:memory:", new SqliteRdbOptions
        {
            HeartbeatInterval = 1,
            HeartbeatStaleTrialCallback = stale,
            FailedTrialCallback = failed,
        });
        act.Should().Throw<ValueError>()
            .WithMessage("Specify only one of `heartbeat_stale_trial_callback` and `failed_trial_callback`.");
    }

    [TestMethod]
    public void Heartbeat_FailedTrialCallback_IsDeprecatedAlias_And_Warns()
    {
        Action<Study, FrozenTrial> cb = (s, t) => { };

        RDBStorage storage = null!;
        var warnings = CaptureWarnings(() =>
        {
            storage = new RDBStorage("sqlite:///:memory:", new SqliteRdbOptions
            {
                HeartbeatInterval = 1,
                FailedTrialCallback = cb,
            });
        });

        using (storage)
        {
            // The deprecated alias becomes the effective heartbeat-stale callback...
            storage.GetHeartbeatStaleTrialCallback().Should().BeSameAs(cb);
            // ...and its use raised the FutureWarning deprecation.
            warnings.Should().Contain(w =>
                w.Category == WarningCategory.FutureWarning &&
                w.Message.Contains("`failed_trial_callback`") && w.Message.Contains("deprecated in v4.9.0"));
        }
    }

    [TestMethod]
    public void Heartbeat_HeartbeatStaleTrialCallback_TakesPrecedence_NoDeprecationWarning()
    {
        Action<Study, FrozenTrial> primary = (s, t) => { };

        RDBStorage storage = null!;
        var warnings = CaptureWarnings(() =>
        {
            storage = new RDBStorage("sqlite:///:memory:", new SqliteRdbOptions
            {
                HeartbeatInterval = 1,
                HeartbeatStaleTrialCallback = primary,
            });
        });

        using (storage)
        {
            storage.GetHeartbeatStaleTrialCallback().Should().BeSameAs(primary);
            warnings.Should().NotContain(w => w.Category == WarningCategory.FutureWarning);
        }
    }

    [TestMethod]
    public void Heartbeat_GetFailedTrialCallback_Deprecated_ReturnsEffective_And_Warns()
    {
        Action<Study, FrozenTrial> cb = (s, t) => { };
        using var storage = new RDBStorage("sqlite:///:memory:", new SqliteRdbOptions
        {
            HeartbeatInterval = 1,
            HeartbeatStaleTrialCallback = cb,
        });

        Action<Study, FrozenTrial>? returned = null;
        var warnings = CaptureWarnings(() =>
        {
#pragma warning disable CS0618 // deliberately exercising the deprecated shim
            returned = ((IBaseHeartbeat)storage).GetFailedTrialCallback();
#pragma warning restore CS0618
        });

        returned.Should().BeSameAs(cb);
        warnings.Should().Contain(w =>
            w.Category == WarningCategory.FutureWarning && w.Message.Contains("get_failed_trial_callback"));
    }

    [TestMethod]
    public void Heartbeat_RecordHeartbeat_Upserts_SingleRow()
    {
        using var storage = new RDBStorage("sqlite:///:memory:", new SqliteRdbOptions { HeartbeatInterval = 1 });
        int studyId = storage.CreateNewStudy(new[] { StudyDirection.Minimize }, "s");
        int trialId = storage.CreateNewTrial(studyId);

        HeartbeatRowCount(storage).Should().Be(0);
        storage.RecordHeartbeat(trialId);
        HeartbeatRowCount(storage).Should().Be(1);

        // A second beat for the same trial updates the one row (UNIQUE(trial_id)) — no duplicate.
        storage.RecordHeartbeat(trialId);
        HeartbeatRowCount(storage).Should().Be(1);
        Scalar(storage.Connection, $"SELECT COUNT(*) FROM trial_heartbeats WHERE trial_id = {trialId}").Should().Be("1");
    }

    [TestMethod]
    public void Heartbeat_GetStaleTrialIds_FreshNotStale_BackdatedStale()
    {
        using var storage = new RDBStorage("sqlite:///:memory:", new SqliteRdbOptions { HeartbeatInterval = 1 });
        int studyId = storage.CreateNewStudy(new[] { StudyDirection.Minimize }, "s");
        int trialId = storage.CreateNewTrial(studyId);

        storage.RecordHeartbeat(trialId);
        storage.GetStaleTrialIds(studyId).Should().BeEmpty("a fresh heartbeat is within the grace period");

        // Grace period defaults to 2 * interval = 2s; push the beat 60s into the past.
        BackdateHeartbeat(storage, trialId, 60);
        storage.GetStaleTrialIds(studyId).Should().Equal(trialId);
    }

    [TestMethod]
    public void Heartbeat_GetStaleTrialIds_Ignores_NonRunning_And_TrialsWithoutHeartbeat()
    {
        using var storage = new RDBStorage("sqlite:///:memory:", new SqliteRdbOptions { HeartbeatInterval = 1 });
        int studyId = storage.CreateNewStudy(new[] { StudyDirection.Minimize }, "s");

        // (a) running + stale heartbeat -> reported.
        int running = storage.CreateNewTrial(studyId);
        storage.RecordHeartbeat(running);
        BackdateHeartbeat(storage, running, 60);

        // (b) finished trial with an old heartbeat -> NOT reported (only RUNNING trials go stale).
        int finished = storage.CreateNewTrial(studyId);
        storage.RecordHeartbeat(finished);
        BackdateHeartbeat(storage, finished, 60);
        storage.SetTrialStateValues(finished, TrialState.Complete, new[] { 1.0 });

        // (c) running trial that never recorded a heartbeat -> NOT reported (INNER JOIN / Python `continue`).
        int noBeat = storage.CreateNewTrial(studyId);

        storage.GetStaleTrialIds(studyId).Should().Equal(running);
        _ = noBeat;
    }

    [TestMethod]
    public void Heartbeat_GetStaleTrialIds_DefaultGracePeriod_Is_TwiceInterval()
    {
        using var storage = new RDBStorage("sqlite:///:memory:", new SqliteRdbOptions { HeartbeatInterval = 5 });
        int studyId = storage.CreateNewStudy(new[] { StudyDirection.Minimize }, "s");
        int trialId = storage.CreateNewTrial(studyId);
        storage.RecordHeartbeat(trialId);

        // Within 2*interval (=10s): not stale.
        BackdateHeartbeat(storage, trialId, 6);
        storage.GetStaleTrialIds(studyId).Should().BeEmpty("6s < the 10s default grace period");

        // Beyond 2*interval: stale.
        BackdateHeartbeat(storage, trialId, 15);
        storage.GetStaleTrialIds(studyId).Should().Equal(trialId);
    }

    [TestMethod]
    public void Heartbeat_GetStaleTrialIds_ExplicitGracePeriod_Overrides_Default()
    {
        using var storage = new RDBStorage(
            "sqlite:///:memory:", new SqliteRdbOptions { HeartbeatInterval = 1, GracePeriod = 30 });
        int studyId = storage.CreateNewStudy(new[] { StudyDirection.Minimize }, "s");
        int trialId = storage.CreateNewTrial(studyId);
        storage.RecordHeartbeat(trialId);

        // 20s would be stale under the default (2*1=2s) but is within the explicit 30s grace.
        BackdateHeartbeat(storage, trialId, 20);
        storage.GetStaleTrialIds(studyId).Should().BeEmpty("20s < the explicit 30s grace period");

        BackdateHeartbeat(storage, trialId, 40);
        storage.GetStaleTrialIds(studyId).Should().Equal(trialId);
    }

    [TestMethod]
    public void Heartbeat_GetStaleTrialIds_ScopedToStudy()
    {
        using var storage = new RDBStorage("sqlite:///:memory:", new SqliteRdbOptions { HeartbeatInterval = 1 });
        int studyA = storage.CreateNewStudy(new[] { StudyDirection.Minimize }, "a");
        int studyB = storage.CreateNewStudy(new[] { StudyDirection.Minimize }, "b");
        int tA = storage.CreateNewTrial(studyA);
        int tB = storage.CreateNewTrial(studyB);
        storage.RecordHeartbeat(tA);
        storage.RecordHeartbeat(tB);
        BackdateHeartbeat(storage, tA, 60);
        BackdateHeartbeat(storage, tB, 60);

        // Each study reports only its own stale trial.
        storage.GetStaleTrialIds(studyA).Should().Equal(tA);
        storage.GetStaleTrialIds(studyB).Should().Equal(tB);
    }

    [TestMethod]
    public void Heartbeat_FailStaleTrials_EndToEnd_FailsStale_And_InvokesCallback()
    {
        var callbackTrials = new List<FrozenTrial>();
        using var rdb = new RDBStorage("sqlite:///:memory:", new SqliteRdbOptions
        {
            HeartbeatInterval = 1,
            HeartbeatStaleTrialCallback = (s, t) => callbackTrials.Add(t),
        });
        var study = Optuna.CreateStudy(storage: rdb, studyName: "s", sampler: new RandomSampler(0));

        int staleId = study.Storage.CreateNewTrial(study.StudyId);
        int healthyId = study.Storage.CreateNewTrial(study.StudyId);
        rdb.RecordHeartbeat(staleId);
        rdb.RecordHeartbeat(healthyId);
        BackdateHeartbeat(rdb, staleId, 60);  // only the stale one is past the grace period

        HeartbeatHelpers.FailStaleTrials(study);

        rdb.GetTrial(staleId).State.Should().Be(TrialState.Fail);
        rdb.GetTrial(healthyId).State.Should().Be(TrialState.Running);
        callbackTrials.Should().ContainSingle(t => t.TrialId == staleId);

        // Already-failed stale ids are tolerated on a second pass (UpdateFinishedTrialError swallowed).
        HeartbeatHelpers.FailStaleTrials(study);
        rdb.GetTrial(staleId).State.Should().Be(TrialState.Fail);
    }

    [TestMethod]
    public void Heartbeat_RetryCallback_EndToEnd_Requeues_Waiting_Trial()
    {
        using var rdb = new RDBStorage("sqlite:///:memory:", new SqliteRdbOptions
        {
            HeartbeatInterval = 1,
            HeartbeatStaleTrialCallback = new RetryHeartbeatStaleTrialCallback(maxRetry: 3),
        });
        var study = Optuna.CreateStudy(storage: rdb, studyName: "s", sampler: new RandomSampler(0));

        int staleId = study.Storage.CreateNewTrial(study.StudyId);
        study.Storage.SetTrialParam(staleId, "x", 1.5, new FloatDistribution(-10, 10));
        rdb.RecordHeartbeat(staleId);
        BackdateHeartbeat(rdb, staleId, 60);

        HeartbeatHelpers.FailStaleTrials(study);

        rdb.GetTrial(staleId).State.Should().Be(TrialState.Fail);
        var requeued = study.GetTrials(deepcopy: false).Single(t => t.State == TrialState.Waiting);
        requeued.Params["x"].Should().Be(1.5);
        RetryHeartbeatStaleTrialCallback.RetriedTrialNumber(requeued).Should().Be(rdb.GetTrial(staleId).Number);
        RetryHeartbeatStaleTrialCallback.RetryHistory(requeued).Should().Equal(rdb.GetTrial(staleId).Number);
    }

    [TestMethod]
    public void Heartbeat_Optimize_RecordsHeartbeats_For_Each_Trial()
    {
        using var rdb = new RDBStorage("sqlite:///:memory:", new SqliteRdbOptions { HeartbeatInterval = 1 });
        var study = Optuna.CreateStudy(storage: rdb, studyName: "s", sampler: new RandomSampler(0));
        study.Optimize(t =>
        {
            Thread.Sleep(20);  // give the heartbeat thread at least one beat during the trial
            double x = t.SuggestFloat("x", -10, 10);
            return x * x;
        }, nTrials: 3);

        // Every trial has exactly one heartbeat row (UNIQUE(trial_id)); the thread beat during each.
        HeartbeatRowCount(rdb).Should().Be(3);
    }

    [TestMethod]
    public void Heartbeat_CachedStorage_Forwards_Every_Method_To_Backend()
    {
        Action<Study, FrozenTrial> cb = (s, t) => { };
        using var rdb = new RDBStorage("sqlite:///:memory:", new SqliteRdbOptions
        {
            HeartbeatInterval = 7,
            HeartbeatStaleTrialCallback = cb,
        });
        // Optuna's _CachedStorage is BaseStorage + BaseHeartbeat and forwards heartbeat to the backend.
        var cached = new CachedStorage(rdb);

        cached.GetHeartbeatInterval().Should().Be(7);
        cached.GetHeartbeatStaleTrialCallback().Should().BeSameAs(cb);
        HeartbeatHelpers.IsHeartbeatEnabled(cached).Should().BeTrue();

        int studyId = cached.CreateNewStudy(new[] { StudyDirection.Minimize }, "s");
        int trialId = cached.CreateNewTrial(studyId);

        cached.RecordHeartbeat(trialId);  // -> backend
        Scalar(rdb.Connection, "SELECT COUNT(*) FROM trial_heartbeats").Should().Be("1");

        cached.GetStaleTrialIds(studyId).Should().BeEmpty();  // -> backend, fresh
        BackdateHeartbeat(rdb, trialId, 60);
        cached.GetStaleTrialIds(studyId).Should().Equal(trialId);

        // The deprecated default-interface getter routes through GetHeartbeatStaleTrialCallback -> backend.
#pragma warning disable CS0618
        ((IBaseHeartbeat)cached).GetFailedTrialCallback().Should().BeSameAs(cb);
#pragma warning restore CS0618
    }

    // ------------------------------------------------------------------ helpers

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

    /// <summary>Rows currently in <c>trial_heartbeats</c> (read on the storage's own connection).</summary>
    private static int HeartbeatRowCount(RDBStorage storage)
        => int.Parse(Scalar(storage.Connection, "SELECT COUNT(*) FROM trial_heartbeats")!, System.Globalization.CultureInfo.InvariantCulture);

    /// <summary>
    ///     Push a trial's heartbeat <paramref name="seconds" /> into the past (via the storage's own
    ///     connection) so it is older than the grace period. SQLite <c>CURRENT_TIMESTAMP</c> is naive
    ///     UTC in <c>yyyy-MM-dd HH:mm:ss</c> — the exact form <see cref="RDBStorage.GetStaleTrialIds" />
    ///     parses.
    /// </summary>
    private static void BackdateHeartbeat(RDBStorage storage, int trialId, int seconds)
    {
        using var cmd = storage.Connection.CreateCommand();
        cmd.CommandText =
            $"UPDATE trial_heartbeats SET heartbeat = datetime(CURRENT_TIMESTAMP, '-{seconds} seconds') WHERE trial_id = $id";
        var p = cmd.CreateParameter();
        p.ParameterName = "$id";
        p.Value = trialId;
        cmd.Parameters.Add(p);
        cmd.ExecuteNonQuery();
    }

    private static string? Scalar(SqliteConnection conn, string sql)
    {
        using var cmd = conn.CreateCommand();
        cmd.CommandText = sql;
        object? value = cmd.ExecuteScalar();
        return value is null or DBNull ? null : Convert.ToString(value, System.Globalization.CultureInfo.InvariantCulture);
    }
}
