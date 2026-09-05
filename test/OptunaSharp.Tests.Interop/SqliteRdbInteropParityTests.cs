using System;
using System.IO;
using System.Linq;
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

    // ------------------------------------------------------------------ heartbeat parity (plan 1.3)
    //
    // The heartbeat table + stale-detection logic must agree byte-for-byte across the two runtimes over
    // the SAME SQLite file: Optuna records the heartbeat as naive-UTC `yyyy-MM-dd HH:mm:ss` TEXT (via
    // `func.now()` == `CURRENT_TIMESTAMP`, no microseconds), which is exactly the form OptunaSharp writes
    // and parses. Both use `heartbeat_interval=1, grace_period=3600` so the few seconds of cross-process
    // latency never flips the fresh trial; the stale trial is backdated 7200s, well past the grace period.

    [TestMethod]
    [TestCategory("RequiresPython")]
    public void Heartbeat_StaleDetection_Python_Writes_CSharp_Reads()
    {
        PythonSession.EnsureOrInconclusive();

        // Python creates a study with one stale (backdated) and one fresh running trial, and computes
        // its own stale set.
        PyExec(
            "import sqlalchemy\n" +
            $"hb = optuna.storages.RDBStorage(r'{Url}', heartbeat_interval=1, grace_period=3600)\n" +
            "hb_study = optuna.create_study(storage=hb, study_name='hb')\n" +
            "hb_sid = hb_study._study_id\n" +
            "hb_stale_tid = hb.create_new_trial(hb_sid)\n" +
            "hb_healthy_tid = hb.create_new_trial(hb_sid)\n" +
            "hb.record_heartbeat(hb_stale_tid)\n" +
            "hb.record_heartbeat(hb_healthy_tid)\n" +
            "with hb.engine.begin() as c:\n" +
            "    c.execute(sqlalchemy.text(\"UPDATE trial_heartbeats SET heartbeat = datetime('now','-7200 seconds') WHERE trial_id = :t\"), {'t': hb_stale_tid})\n" +
            "hb_stale_number = hb.get_trial(hb_stale_tid).number\n" +
            "hb_healthy_number = hb.get_trial(hb_healthy_tid).number\n" +
            "hb_py_stale_numbers = sorted(hb.get_trial(t).number for t in hb._get_stale_trial_ids(hb_sid))\n" +
            "hb.remove_session()\n" +
            "hb.engine.dispose()\n");

        int staleNumber = (int)PyLong("hb_stale_number");
        int healthyNumber = (int)PyLong("hb_healthy_number");

        // OptunaSharp opens the same file and computes its own stale set from Optuna's heartbeat rows.
        using var rdb = new RDBStorage(Url, heartbeatInterval: 1, gracePeriod: 3600);
        int sid = rdb.GetStudyIdFromName("hb");
        var csStaleNumbers = rdb.GetStaleTrialIds(sid).Select(t => rdb.GetTrial(t).Number).OrderBy(n => n).ToList();

        // OptunaSharp flags exactly the backdated trial.
        csStaleNumbers.Should().Equal(staleNumber);
        csStaleNumbers.Should().NotContain(healthyNumber);
        // ...and it agrees, trial-for-trial, with Optuna's own `_get_stale_trial_ids` on the same DB.
        PyLong("len(hb_py_stale_numbers)").Should().Be(1);
        ((int)PyLong("hb_py_stale_numbers[0]")).Should().Be(staleNumber);
    }

    [TestMethod]
    [TestCategory("RequiresPython")]
    public void Heartbeat_StaleDetection_CSharp_Writes_Python_Reads()
    {
        PythonSession.EnsureOrInconclusive();

        int staleNumber;
        int healthyNumber;
        using (var rdb = new RDBStorage(Url, heartbeatInterval: 1, gracePeriod: 3600))
        {
            int sid = rdb.CreateNewStudy(new[] { StudyDirection.Minimize }, "hb");
            int staleId = rdb.CreateNewTrial(sid);
            int healthyId = rdb.CreateNewTrial(sid);
            rdb.RecordHeartbeat(staleId);
            rdb.RecordHeartbeat(healthyId);
            staleNumber = rdb.GetTrial(staleId).Number;
            healthyNumber = rdb.GetTrial(healthyId).Number;
        } // dispose releases the file for Python

        // Optuna opens the same DB and reads OptunaSharp's heartbeat rows: both fresh -> nothing stale.
        PyExec(
            "import sqlalchemy\n" +
            $"hb = optuna.storages.RDBStorage(r'{Url}', heartbeat_interval=1, grace_period=3600)\n" +
            "hb_sid = hb.get_study_id_from_name('hb')\n");
        PyLong("len(hb._get_stale_trial_ids(hb_sid))")
            .Should().Be(0, "OptunaSharp's fresh heartbeats are not stale to Optuna");

        // Backdate the OptunaSharp-written heartbeat; Optuna must now flag exactly that trial.
        PyExec(
            $"hb_stale_tid = hb.get_trial_id_from_study_id_trial_number(hb_sid, {staleNumber})\n" +
            "with hb.engine.begin() as c:\n" +
            "    c.execute(sqlalchemy.text(\"UPDATE trial_heartbeats SET heartbeat = datetime('now','-7200 seconds') WHERE trial_id = :t\"), {'t': hb_stale_tid})\n" +
            "hb_py_stale_numbers = sorted(hb.get_trial(t).number for t in hb._get_stale_trial_ids(hb_sid))\n" +
            "hb.remove_session()\n" +
            "hb.engine.dispose()\n");

        PyLong("len(hb_py_stale_numbers)").Should().Be(1, "Optuna read OptunaSharp's heartbeat row and flags the backdated trial");
        ((int)PyLong("hb_py_stale_numbers[0]")).Should().Be(staleNumber);
        healthyNumber.Should().NotBe(staleNumber);
    }
}
