using System;
using System.IO;
using AwesomeAssertions;
using OptunaSharp.Collections;
using OptunaSharp.Distributions;
using OptunaSharp.Samplers;
using OptunaSharp.Storages;

namespace OptunaSharp.Tests.Interop;

/// <summary>
///     The Phase 4.1 cross-language gate: <b>Python Optuna and OptunaSharp open the same SQLite
///     <c>.db</c> file</b>. A study written by one side is read back, in full, by the other — the
///     proof that OptunaSharp's <see cref="RDBStorage" /> reproduces Optuna's schema and encodings
///     byte-for-byte (alembic <c>v3.2.0.a</c>, naive-UTC datetimes, the finite/±inf/NaN type tags,
///     <c>distribution_json</c> and <c>json.dumps</c> attrs).
///
///     <para>No external service is needed — <c>Microsoft.Data.Sqlite</c> bundles the engine — so
///     these self-skip only when Python/optuna is absent, exactly like every other
///     <c>RequiresPython</c> test.</para>
/// </summary>
[TestClass]
public class SqliteRdbInteropParityTests : OptunaInteropTestBase
{
    private string _path = null!;

    [TestInitialize]
    public void SqliteInit()
        => _path = Path.Combine(Path.GetTempPath(), $"optunasharp-rdb-{Guid.NewGuid():N}.db").Replace('\\', '/');

    [TestCleanup]
    public void SqliteCleanup()
    {
        foreach (var f in Directory.GetFiles(Path.GetDirectoryName(_path)!, Path.GetFileName(_path) + "*"))
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

    private string Url => $"sqlite:///{_path}";

    /// <summary>The C# objective — mixed param types, an intermediate report, an ±inf-friendly value.</summary>
    private static double MixedObjective(Trial t)
    {
        double x = t.SuggestFloat("x", -10, 10);
        long y = t.SuggestInt("y", 1, 100, log: true);
        _ = t.SuggestCategorical("z", new CategoricalChoice[] { "a", "b", (long)3, true, CategoricalChoice.None });
        t.Report(0.5, 0);
        t.Report(double.PositiveInfinity, 1);
        return x * x + y;
    }

    /// <summary>The identical objective on the Python side.</summary>
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
    public void CSharp_Writes_Python_Reads_SameFile(int seed)
    {
        PythonSession.EnsureOrInconclusive();

        var rdb = new RDBStorage(Url);
        var csStudy = Optuna.CreateStudy(storage: rdb, studyName: "s", sampler: new RandomSampler(seed));
        csStudy.SetUserAttr("owner", "eli");
        csStudy.SetUserAttr("run", 7L);
        csStudy.Optimize(MixedObjective, nTrials: 12);
        var csState = StudySnapshot.Capture(csStudy);
        rdb.Dispose(); // release the file so Python opens a clean, committed database

        PyExec($"study = optuna.load_study(study_name='s', storage=r'{Url}')");
        StudySnapshot.AssertParity(PyStudyState("study"), csState, DatetimeComparison.NullnessOnly);
    }

    [TestMethod]
    [TestCategory("RequiresPython")]
    [DataRow(0)]
    [DataRow(42)]
    public void Python_Writes_CSharp_Reads_SameFile(int seed)
    {
        PythonSession.EnsureOrInconclusive();

        PyExec(
            $"study = optuna.create_study(study_name='s', storage=r'{Url}', " +
            $"sampler=optuna.samplers.RandomSampler({seed}))\n" +
            "study.set_user_attr('owner', 'eli')\n" +
            "study.set_user_attr('run', 7)\n" +
            PyMixedObjective +
            "study.optimize(_obj, n_trials=12)\n");
        var pyState = PyStudyState("study");
        PyExec("study._storage.remove_session()\n");

        using var rdb = new RDBStorage(Url);
        var csStudy = Optuna.LoadStudy("s", rdb);
        var csState = StudySnapshot.Capture(csStudy);
        StudySnapshot.AssertParity(pyState, csState, DatetimeComparison.NullnessOnly);
    }

    [TestMethod]
    [TestCategory("RequiresPython")]
    public void MultiObjective_And_Nonfinite_CSharp_To_Python()
    {
        PythonSession.EnsureOrInconclusive();

        var rdb = new RDBStorage(Url);

        // A single-objective study whose stored values include +inf, plus NaN/±inf intermediates.
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

        // A multi-objective study.
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
        rdb.Dispose();

        PyExec($"single = optuna.load_study(study_name='single', storage=r'{Url}')");
        PyExec($"mo = optuna.load_study(study_name='mo', storage=r'{Url}')");
        StudySnapshot.AssertParity(PyStudyState("single"), csSingle, DatetimeComparison.NullnessOnly);
        StudySnapshot.AssertParity(PyStudyState("mo"), csMo, DatetimeComparison.NullnessOnly);
    }
}
