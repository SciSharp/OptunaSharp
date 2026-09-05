using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using OptunaSharp.Samplers;
using OptunaSharp.Storages.Journal;
using Python.Runtime;

namespace OptunaSharp.Tests.Interop;

/// <summary>
///     Persistence parity over the journal file, in every direction, asserted at FULL study
///     state (bit-exact floats, instant-equal datetimes):
///     <list type="bullet">
///     <item>Python runs a fuzz scenario into a journal → OptunaSharp replays → states equal.</item>
///     <item>OptunaSharp runs the scenario into a journal → Python replays → states equal.</item>
///     <item>OptunaSharp reload self-parity (fresh storage instance over the same file).</item>
///     <item>Alternating writers build one study together; both sides see the identical state.</item>
///     <item>RAW LINE FORMAT parity: the journal C# writes for a scenario is structurally
///           line-by-line identical (op codes, keys, payload values incl. float bytes) to the
///           journal Python writes for the same scenario, after masking worker ids/timestamps.</item>
///     </list>
/// </summary>
[TestClass]
public class JournalStateParityTests : OptunaInteropTestBase
{
    private string _path = null!;

    [TestInitialize]
    public void JournalInit()
        => _path = Path.Combine(Path.GetTempPath(), $"optunasharp-parity-{Guid.NewGuid():N}.log").Replace('\\', '/');

    [TestCleanup]
    public void JournalCleanup()
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

    private JournalStorage NewCSharpStorage(string? path = null)
    {
        path ??= _path;
        return new JournalStorage(new JournalFileBackend(path, lockObj: new JournalFileOpenLock(path)));
    }

    // ------------------------------------------------------------------ direction: Python → C#

    [TestMethod]
    [TestCategory("RequiresPython")]
    [DataRow(101)]
    [DataRow(102)]
    [DataRow(103)]
    [DataRow(104)]
    public void PythonWrites_CSharpReplays_FullState(int seed)
    {
        var scenario = ParityScenario.Generate(seed);

        PySet("spec_json", scenario.ToSpecJson());
        PyExec(ParityScenario.PythonRunnerSource);
        PyExec(
            $"storage = {PyJournalStorageExpr(_path)}\n" +
            "study = optuna.create_study(study_name='p2c', storage=storage, " +
            "sampler=optuna.samplers.RandomSampler(seed=spec['sampler_seed']))\n" +
            "_oss_run(study)\n");
        var pyState = PyStudyState("study");

        var csStudy = Optuna.LoadStudy("p2c", NewCSharpStorage(), sampler: new RandomSampler(0));
        var csState = StudySnapshot.Capture(csStudy);

        // Replayed from the same journal: datetimes must agree as instants too.
        StudySnapshot.AssertParity(pyState, csState, DatetimeComparison.Instant);
    }

    // ------------------------------------------------------------------ direction: C# → Python

    [TestMethod]
    [TestCategory("RequiresPython")]
    [DataRow(201)]
    [DataRow(202)]
    [DataRow(203)]
    [DataRow(204)]
    public void CSharpWrites_PythonReplays_FullState(int seed)
    {
        var scenario = ParityScenario.Generate(seed);

        var csStudy = Optuna.CreateStudy(
            storage: NewCSharpStorage(), sampler: scenario.NewSampler(), studyName: "c2p");
        scenario.RunCSharp(csStudy);
        csStudy.SetUserAttr("writer", "csharp");
        var csState = StudySnapshot.Capture(csStudy);

        PyExec(
            $"storage = {PyJournalStorageExpr(_path)}\n" +
            "study = optuna.load_study(study_name='c2p', storage=storage)\n");
        var pyState = PyStudyState("study");

        StudySnapshot.AssertParity(csState, pyState, DatetimeComparison.Instant);
    }

    // ------------------------------------------------------------------ C#-only reload self-parity

    [TestMethod]
    [DataRow(301)]
    [DataRow(302)]
    [DataRow(303)]
    public void CSharp_Reload_SelfParity(int seed)
    {
        // No Python needed: persist → replay with a FRESH storage over the same file → the
        // full state must survive byte-for-byte (the C#-side persistence completeness claim).
        var scenario = ParityScenario.Generate(seed);

        var written = Optuna.CreateStudy(
            storage: NewCSharpStorage(), sampler: scenario.NewSampler(), studyName: "self");
        scenario.RunCSharp(written);
        var writtenState = StudySnapshot.Capture(written);

        var reloaded = Optuna.LoadStudy("self", NewCSharpStorage(), sampler: new RandomSampler(0));
        var reloadedState = StudySnapshot.Capture(reloaded);

        StudySnapshot.AssertParity(writtenState, reloadedState, DatetimeComparison.Instant);
    }

    [TestMethod]
    [DataRow(311)]
    [DataRow(312)]
    public void CSharp_MultiObjective_Reload_SelfParity(int seed)
    {
        var scenario = ParityScenario.Generate(seed, multiObjective: true);

        var written = Optuna.CreateStudy(
            storage: NewCSharpStorage(), sampler: scenario.NewSampler(), studyName: "self-mo",
            directions: new[] { StudyDirection.Minimize, StudyDirection.Maximize });
        scenario.RunCSharp(written);

        var reloaded = Optuna.LoadStudy("self-mo", NewCSharpStorage(), sampler: new RandomSampler(0));
        StudySnapshot.AssertParity(
            StudySnapshot.Capture(written), StudySnapshot.Capture(reloaded), DatetimeComparison.Instant);
    }

    // ------------------------------------------------------------------ alternating writers

    [TestMethod]
    [TestCategory("RequiresPython")]
    public void Alternating_Writers_Converge_To_Identical_State()
    {
        // C# creates + 3 trials → Python continues 3 → C# 3 more → Python 3 more.
        // At the end BOTH sides replay the shared journal and must see the identical state.
        var storage = NewCSharpStorage();
        var study = Optuna.CreateStudy(
            storage: storage, sampler: new RandomSampler(seed: 31), studyName: "alt");
        double Objective(Trial t)
        {
            double x = t.SuggestFloat("x", -10, 10);
            long n = t.SuggestInt("n", 1, 16, log: true);
            t.Report(x / 2, 0);
            return Math.Pow(x - 2, 2) + n;
        }
        study.Optimize(Objective, nTrials: 3);

        PyExec(
            $"storage = {PyJournalStorageExpr(_path)}\n" +
            "study = optuna.load_study(study_name='alt', storage=storage)\n" +
            "study.sampler = optuna.samplers.RandomSampler(seed=32)\n" +
            "def objective(t):\n" +
            "    x = t.suggest_float('x', -10, 10)\n" +
            "    n = t.suggest_int('n', 1, 16, log=True)\n" +
            "    t.report(x / 2, 0)\n" +
            "    return (x - 2) ** 2 + n\n" +
            "study.optimize(objective, n_trials=3)\n");

        study.Sampler = new RandomSampler(seed: 33);
        study.Optimize(Objective, nTrials: 3);

        PyExec("study.sampler = optuna.samplers.RandomSampler(seed=34)\n" +
               "study.optimize(objective, n_trials=3)\n");

        // Fresh replays on both sides.
        PyExec(
            $"fresh_storage = {PyJournalStorageExpr(_path)}\n" +
            "fresh = optuna.load_study(study_name='alt', storage=fresh_storage)\n");
        var pyState = PyStudyState("fresh");
        var csState = StudySnapshot.Capture(
            Optuna.LoadStudy("alt", NewCSharpStorage(), sampler: new RandomSampler(0)));

        if (csState.Trials.Count != 12)
            throw new AssertFailedException($"Expected 12 trials (4 writers x 3) but got {csState.Trials.Count}.");
        StudySnapshot.AssertParity(pyState, csState, DatetimeComparison.Instant);
    }

    // ------------------------------------------------------------------ raw line-format parity

    [TestMethod]
    [TestCategory("RequiresPython")]
    [DataRow(401)]
    [DataRow(402)]
    [DataRow(403)]
    public void Journal_RawLine_Format_Parity(int seed)
    {
        // Line-by-line float-repr (bit-equal) journal parity — host-pinned to x64 (arm64 libm drifts).
        PythonSession.EnsureBitExactHostOrInconclusive();

        // The SAME deterministic scenario is written to two fresh journals — one by Python,
        // one by OptunaSharp. After masking worker ids and timestamps, the two files must be
        // STRUCTURALLY IDENTICAL line by line: same op sequence, same keys, same payloads
        // (float reprs equal — i.e. bit-equal), same distribution JSON strings.
        var scenario = ParityScenario.Generate(seed);
        string pyPath = _path + ".py.log";
        string csPath = _path + ".cs.log";

        PySet("spec_json", scenario.ToSpecJson());
        PyExec(ParityScenario.PythonRunnerSource);
        PyExec(
            $"storage = {PyJournalStorageExpr(pyPath)}\n" +
            "study = optuna.create_study(study_name='fmt', storage=storage, " +
            "sampler=optuna.samplers.RandomSampler(seed=spec['sampler_seed']))\n" +
            "_oss_run(study)\n");

        var csStudy = Optuna.CreateStudy(
            storage: NewCSharpStorage(csPath), sampler: scenario.NewSampler(), studyName: "fmt");
        scenario.RunCSharp(csStudy);

        // Normalize both files in Python (json.loads → mask volatile → dumps sort_keys).
        PyExec(
            "def _oss_normalize_journal(path):\n" +
            "    lines = []\n" +
            "    with open(path, 'rb') as f:\n" +
            "        for raw in f:\n" +
            "            d = json.loads(raw)\n" +
            "            d.pop('worker_id', None)\n" +
            "            d.pop('datetime_start', None)\n" +
            "            d.pop('datetime_complete', None)\n" +
            "            lines.append(json.dumps(d, sort_keys=True))\n" +
            "    return lines\n" +
            $"py_lines = _oss_normalize_journal(r'{pyPath}')\n" +
            $"cs_lines = _oss_normalize_journal(r'{csPath}')\n" +
            "n_py = len(py_lines)\n" +
            "n_cs = len(cs_lines)\n" +
            "first_diff = next((i for i, (a, b) in enumerate(zip(py_lines, cs_lines)) if a != b), -1)\n" +
            "diff_py = py_lines[first_diff] if first_diff >= 0 else ''\n" +
            "diff_cs = cs_lines[first_diff] if first_diff >= 0 else ''\n");

        long nPy = PyLong("n_py");
        long nCs = PyLong("n_cs");
        long firstDiff = PyLong("first_diff");
        if (nPy != nCs || firstDiff >= 0)
        {
            throw new AssertFailedException(
                $"Journal format parity FAILED for {scenario}:\n" +
                $"  lines: python={nPy}, csharp={nCs}\n" +
                (firstDiff >= 0
                    ? $"  first differing line #{firstDiff}:\n    python: {PyStr("diff_py")}\n    csharp: {PyStr("diff_cs")}"
                    : string.Empty));
        }
    }
}
