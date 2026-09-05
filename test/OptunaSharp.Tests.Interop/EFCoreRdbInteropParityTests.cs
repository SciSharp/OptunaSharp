using System;
using AwesomeAssertions;
using MySqlConnector;
using Npgsql;
using OptunaSharp.Collections;
using OptunaSharp.Distributions;
using OptunaSharp.Samplers;
using OptunaSharp.Storages;
using OptunaSharp.Storages.EFCore;
using Testcontainers.MySql;
using Testcontainers.PostgreSql;
// The SQLite and EF Core packages both expose an RDBStorage; this gate means the EF Core one.
using RDBStorage = OptunaSharp.Storages.EFCore.RDBStorage;

namespace OptunaSharp.Tests.Interop;

/// <summary>
///     Lazily-started PostgreSQL and MySQL servers (Testcontainers) shared by the EF Core
///     cross-language gate. Containers are reaped by Testcontainers' Ryuk on process exit.
/// </summary>
internal static class EFCoreInteropDb
{
    private sealed class Endpoint
    {
        public string CsConnectionString = "";
        public string PySqlUrl = "";
    }

    private static readonly Lazy<Endpoint?> _postgres = new(StartPostgres);
    private static readonly Lazy<Endpoint?> _mysql = new(StartMySql);

    private static Lazy<Endpoint?> Lazy(RdbProvider p) => p == RdbProvider.PostgreSql ? _postgres : _mysql;

    public static bool Available(RdbProvider p) => Lazy(p).Value is not null;
    public static string CsConnString(RdbProvider p) => Lazy(p).Value!.CsConnectionString;
    public static string PySqlUrl(RdbProvider p) => Lazy(p).Value!.PySqlUrl;

    public static void Reset(RdbProvider provider)
    {
        string[] tables =
        {
            "trial_heartbeats", "trial_intermediate_values", "trial_values", "trial_params",
            "trial_system_attributes", "trial_user_attributes", "trials",
            "study_system_attributes", "study_user_attributes", "study_directions",
            "alembic_version", "version_info", "studies",
        };
        if (provider == RdbProvider.PostgreSql)
        {
            using var c = new NpgsqlConnection(CsConnString(provider));
            c.Open();
            foreach (var t in tables) Exec(c, $"DROP TABLE IF EXISTS {t} CASCADE");
            foreach (var e in new[] { "studydirection", "trialstate", "trialvaluetype", "trialintermediatevaluetype" })
                Exec(c, $"DROP TYPE IF EXISTS {e} CASCADE");
        }
        else
        {
            using var c = new MySqlConnection(CsConnString(provider));
            c.Open();
            Exec(c, "SET FOREIGN_KEY_CHECKS = 0");
            foreach (var t in tables) Exec(c, $"DROP TABLE IF EXISTS {t}");
            Exec(c, "SET FOREIGN_KEY_CHECKS = 1");
        }
    }

    private static void Exec(System.Data.Common.DbConnection c, string sql)
    {
        using var cmd = c.CreateCommand();
        cmd.CommandText = sql;
        cmd.ExecuteNonQuery();
    }

    private static Endpoint? StartPostgres()
    {
        try
        {
            var c = new PostgreSqlBuilder().WithImage("postgres:16-alpine")
                .WithUsername("optuna").WithPassword("optuna").WithDatabase("optuna").Build();
            c.StartAsync().GetAwaiter().GetResult();
            return new Endpoint
            {
                CsConnectionString = c.GetConnectionString(),
                PySqlUrl = $"postgresql://optuna:optuna@{c.Hostname}:{c.GetMappedPublicPort(5432)}/optuna",
            };
        }
        catch { return null; }
    }

    private static Endpoint? StartMySql()
    {
        try
        {
            var c = new MySqlBuilder().WithImage("mysql:8.0")
                .WithUsername("optuna").WithPassword("optuna").WithDatabase("optuna").Build();
            c.StartAsync().GetAwaiter().GetResult();
            // SQLAlchemy needs the driver spelled out for MySQL; pymysql is the pure-Python one.
            return new Endpoint
            {
                CsConnectionString = c.GetConnectionString(),
                PySqlUrl = $"mysql+pymysql://optuna:optuna@{c.Hostname}:{c.GetMappedPublicPort(3306)}/optuna",
            };
        }
        catch { return null; }
    }
}

/// <summary>
///     The Phase 4.2 cross-language gate: <b>Python Optuna and OptunaSharp open the same PostgreSQL /
///     MySQL database</b>. A study written by one side is read back, in full, by the other — the proof
///     that the EF Core <see cref="RDBStorage" /> reproduces Optuna's SQLAlchemy schema and encodings
///     (native/inline <c>ENUM</c>s, alembic <c>v3.2.0.a</c>, naive-UTC datetimes, the finite/±inf/NaN
///     type tags, <c>distribution_json</c> and <c>json.dumps</c> attrs) so both languages interoperate.
///
///     <para>Self-skips (Inconclusive) when Docker, Python/optuna, or the Python DB driver
///     (psycopg2 / pymysql) is absent — like every other <c>RequiresPython</c> test.</para>
/// </summary>
public abstract class EFCoreRdbInteropParityTestsBase : OptunaInteropTestBase
{
    protected abstract RdbProvider Provider { get; }
    protected abstract string PyDriver { get; }

    [TestInitialize]
    public void EFCoreInteropInit()
    {
        if (!EFCoreInteropDb.Available(Provider))
            Assert.Inconclusive($"Docker/{Provider} testcontainer unavailable — skipping the EF Core cross-language gate.");
        EnsurePyDriver();
        EFCoreInteropDb.Reset(Provider);
    }

    private void EnsurePyDriver()
    {
        try
        {
            PyExec($"import {PyDriver}");
        }
        catch (Exception)
        {
            Assert.Inconclusive($"The Python driver '{PyDriver}' is not importable — skipping.");
        }
    }

    private string CsConn => EFCoreInteropDb.CsConnString(Provider);
    private string PyUrl => EFCoreInteropDb.PySqlUrl(Provider);

    private static double MixedObjective(Trial t)
    {
        double x = t.SuggestFloat("x", -10, 10);
        long y = t.SuggestInt("y", 1, 100, log: true);
        _ = t.SuggestCategorical("z", new CategoricalChoice[] { "a", "b", (long)3, true, CategoricalChoice.None });
        t.Report(0.5, 0);
        t.Report(double.PositiveInfinity, 1);
        return x * x + y;
    }

    private const string PyMixedObjective =
        "def _obj(t):\n" +
        "    x = t.suggest_float('x', -10, 10)\n" +
        "    y = t.suggest_int('y', 1, 100, log=True)\n" +
        "    z = t.suggest_categorical('z', ['a', 'b', 3, True, None])\n" +
        "    t.report(0.5, 0)\n" +
        "    t.report(float('inf'), 1)\n" +
        "    return x * x + y\n";

    [TestMethod]
    [TestCategory("RequiresPython")]
    [DataRow(0)]
    [DataRow(42)]
    public void CSharp_Writes_Python_Reads_SameDatabase(int seed)
    {
        using (var rdb = new RDBStorage(Provider, CsConn))
        {
            var csStudy = Optuna.CreateStudy(storage: rdb, studyName: "s", sampler: new RandomSampler(seed));
            csStudy.SetUserAttr("owner", "eli");
            csStudy.SetUserAttr("run", 7L);
            csStudy.Optimize(MixedObjective, nTrials: 12);
            var csState = StudySnapshot.Capture(csStudy);

            PyExec($"study = optuna.load_study(study_name='s', storage=r'{PyUrl}')");
            StudySnapshot.AssertParity(PyStudyState("study"), csState, DatetimeComparison.NullnessOnly);
        }
    }

    [TestMethod]
    [TestCategory("RequiresPython")]
    [DataRow(0)]
    [DataRow(42)]
    public void Python_Writes_CSharp_Reads_SameDatabase(int seed)
    {
        PyExec(
            $"study = optuna.create_study(study_name='s', storage=r'{PyUrl}', " +
            $"sampler=optuna.samplers.RandomSampler({seed}))\n" +
            "study.set_user_attr('owner', 'eli')\n" +
            "study.set_user_attr('run', 7)\n" +
            PyMixedObjective +
            "study.optimize(_obj, n_trials=12)\n");
        var pyState = PyStudyState("study");
        PyExec("study._storage.remove_session()\n");

        using var rdb = new RDBStorage(Provider, CsConn);
        var csStudy = Optuna.LoadStudy("s", rdb);
        var csState = StudySnapshot.Capture(csStudy);
        StudySnapshot.AssertParity(pyState, csState, DatetimeComparison.NullnessOnly);
    }

    [TestMethod]
    [TestCategory("RequiresPython")]
    public void MultiObjective_And_Nonfinite_CSharp_To_Python()
    {
        using var rdb = new RDBStorage(Provider, CsConn);

        var single = Optuna.CreateStudy(storage: rdb, studyName: "single", direction: StudyDirection.Minimize);
        single.AddTrial(FrozenTrial.CreateTrial(
            state: TrialState.Complete,
            value: double.PositiveInfinity,
            @params: new OrderedDict<string, object?> { ["x"] = 1.0 },
            distributions: new OrderedDict<string, BaseDistribution> { ["x"] = new FloatDistribution(-10, 10) },
            intermediateValues: new OrderedDict<int, double>
            {
                [0] = 0.5, [1] = double.PositiveInfinity, [2] = double.NaN, [3] = double.NegativeInfinity,
            }));
        single.AddTrial(FrozenTrial.CreateTrial(
            state: TrialState.Complete,
            value: 3.5,
            @params: new OrderedDict<string, object?> { ["x"] = -2.0 },
            distributions: new OrderedDict<string, BaseDistribution> { ["x"] = new FloatDistribution(-10, 10) }));

        var mo = Optuna.CreateStudy(
            storage: rdb, studyName: "mo",
            directions: new[] { StudyDirection.Minimize, StudyDirection.Maximize });
        mo.AddTrial(FrozenTrial.CreateTrial(
            state: TrialState.Complete,
            values: new[] { 0.3, 10.0 },
            @params: new OrderedDict<string, object?> { ["a"] = 0.3, ["b"] = 5L },
            distributions: new OrderedDict<string, BaseDistribution>
            {
                ["a"] = new FloatDistribution(0, 1), ["b"] = new IntDistribution(1, 10),
            }));

        var csSingle = StudySnapshot.Capture(single);
        var csMo = StudySnapshot.Capture(mo);

        PyExec($"single = optuna.load_study(study_name='single', storage=r'{PyUrl}')");
        PyExec($"mo = optuna.load_study(study_name='mo', storage=r'{PyUrl}')");
        StudySnapshot.AssertParity(PyStudyState("single"), csSingle, DatetimeComparison.NullnessOnly);
        StudySnapshot.AssertParity(PyStudyState("mo"), csMo, DatetimeComparison.NullnessOnly);
    }

    // A dynamic (conditional) search space + intermediate reports + deterministic pruning — exercises
    // trial states (COMPLETE/PRUNED), dynamic distributions, and intermediate_values in one study.
    private static double DynamicPruningObjective(Trial t)
    {
        string kind = (string)t.SuggestCategorical("kind", new CategoricalChoice[] { "linear", "quad" }).Value!;
        double x = t.SuggestFloat("x", -5, 5);
        double value = kind == "quad" ? x * x : Math.Abs(x);
        for (int step = 0; step < 3; step++)
            t.Report(value + step, step);
        if (t.Number % 2 == 1)
            throw new TrialPruned();
        return value;
    }

    private const string PyDynamicPruningObjective =
        "def _obj(t):\n" +
        "    kind = t.suggest_categorical('kind', ['linear', 'quad'])\n" +
        "    x = t.suggest_float('x', -5, 5)\n" +
        "    value = x * x if kind == 'quad' else abs(x)\n" +
        "    for step in range(3):\n" +
        "        t.report(value + step, step)\n" +
        "    if t.number % 2 == 1:\n" +
        "        raise optuna.TrialPruned()\n" +
        "    return value\n";

    [TestMethod]
    [TestCategory("RequiresPython")]
    public void DynamicSpace_Pruning_Intermediates_CSharp_To_Python()
    {
        using var rdb = new RDBStorage(Provider, CsConn);
        var study = Optuna.CreateStudy(storage: rdb, studyName: "dyn", sampler: new RandomSampler(seed: 0));
        study.Optimize(DynamicPruningObjective, nTrials: 8);
        var csState = StudySnapshot.Capture(study);

        PyExec($"study = optuna.load_study(study_name='dyn', storage=r'{PyUrl}')");
        StudySnapshot.AssertParity(PyStudyState("study"), csState, DatetimeComparison.NullnessOnly);
    }

    [TestMethod]
    [TestCategory("RequiresPython")]
    public void MultipleStudies_AndDeleteStudy_CSharp_To_Python()
    {
        using var rdb = new RDBStorage(Provider, CsConn);
        var keep = Optuna.CreateStudy(storage: rdb, studyName: "keep", sampler: new RandomSampler(seed: 0));
        keep.Optimize(t => { double x = t.SuggestFloat("x", -5, 5); return x * x; }, nTrials: 3);

        var drop = Optuna.CreateStudy(storage: rdb, studyName: "drop");
        drop.AddTrial(FrozenTrial.CreateTrial(
            state: TrialState.Complete, value: 1.0,
            @params: new OrderedDict<string, object?> { ["y"] = 2.0 },
            distributions: new OrderedDict<string, BaseDistribution> { ["y"] = new FloatDistribution(0, 10) }));
        rdb.DeleteStudy(rdb.GetStudyIdFromName("drop"));
        var keepState = StudySnapshot.Capture(keep);

        // Python sees exactly the surviving study, and reads it back identically.
        PyExec($"names = sorted(s.study_name for s in optuna.get_all_study_summaries(storage=r'{PyUrl}'))");
        PyStr("names").Should().Be("['keep']");
        PyExec($"study = optuna.load_study(study_name='keep', storage=r'{PyUrl}')");
        StudySnapshot.AssertParity(PyStudyState("study"), keepState, DatetimeComparison.NullnessOnly);
    }
}

/// <summary>The cross-language gate against a live PostgreSQL server.</summary>
[TestClass]
public sealed class PostgreSqlRdbInteropParityTests : EFCoreRdbInteropParityTestsBase
{
    protected override RdbProvider Provider => RdbProvider.PostgreSql;
    protected override string PyDriver => "psycopg2";
}

/// <summary>The cross-language gate against a live MySQL server.</summary>
[TestClass]
public sealed class MySqlRdbInteropParityTests : EFCoreRdbInteropParityTestsBase
{
    protected override RdbProvider Provider => RdbProvider.MySql;
    protected override string PyDriver => "pymysql";
}
