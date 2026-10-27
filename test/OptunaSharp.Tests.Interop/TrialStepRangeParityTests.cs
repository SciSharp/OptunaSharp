using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using AwesomeAssertions;
using OptunaSharp.Pruners;
using OptunaSharp.Samplers;
using OptunaSharp.Storages;
using OptunaSharp.Storages.Journal;

namespace OptunaSharp.Tests.Interop;

/// <summary>
///     Review item B-2 (pass 2B), live: trial steps past 2^31 cross between Python Optuna and OptunaSharp unchanged.
///     Python's <c>step</c> is an unbounded <c>int</c> that its journal and SQLite storages keep. With 32-bit steps,
///     a Python-written step of 3,000,000,000 made every C# load of a journal throw <see cref="OverflowException" />
///     and came back from SQLite as -1,294,967,296. Each case writes on one side and reads the same file on the other,
///     with the full study state compared field by field; the pruner case compares live decisions at such steps.
/// </summary>
[TestClass]
public class TrialStepRangeParityTests : OptunaInteropTestBase
{
    private const long PastInt32 = 3_000_000_000L;
    private const long Huge = 1L << 62;

    private string _path = null!;

    /// <summary>A fresh temporary path per test, with forward slashes so Python can embed it in a URL.</summary>
    [TestInitialize]
    public void PathInit()
        => _path = Path.Combine(Path.GetTempPath(), $"optunasharp-steps-{Guid.NewGuid():N}.db").Replace('\\', '/');

    /// <summary>Deletes the test's file and every lock or journal sidecar next to it.</summary>
    [TestCleanup]
    public void PathCleanup()
    {
        foreach (string file in Directory.GetFiles(Path.GetDirectoryName(_path)!, Path.GetFileName(_path) + "*"))
        {
            try
            {
                File.Delete(file);
            }
            catch (IOException)
            {
                // A Python handle can still be closing; the temp folder is cleaned by the OS eventually.
            }
        }
    }

    /// <summary>The Python objective: one parameter, and reports at step 0, past Int32, and at 2^62.</summary>
    private const string PyObjective =
        "def _obj(t):\n" +
        "    x = t.suggest_float('x', 0, 1)\n" +
        "    t.report(0.25, 0)\n" +
        "    t.report(0.5, 3_000_000_000)\n" +
        "    t.report(0.75, 2 ** 62)\n" +
        "    return x\n";

    /// <summary>The same objective in C#.</summary>
    /// <param name="trial">The trial to sample and report on.</param>
    /// <returns>The objective value.</returns>
    private static double Objective(Trial trial)
    {
        double x = trial.SuggestFloat("x", 0, 1);
        trial.Report(0.25, 0);
        trial.Report(0.5, PastInt32);
        trial.Report(0.75, Huge);
        return x;
    }

    /// <summary>The Python storage expression for a storage kind on this test's file.</summary>
    /// <param name="kind"><c>journal</c> or <c>sqlite</c>.</param>
    /// <returns>Python source constructing (or naming) the storage.</returns>
    private string PyStorage(string kind)
        => kind == "journal" ? PyJournalStorageExpr(_path) : $"r'sqlite:///{_path}'";

    /// <summary>Opens this test's file with the OptunaSharp storage of the given kind.</summary>
    /// <param name="kind"><c>journal</c> or <c>sqlite</c>.</param>
    /// <returns>The storage; the caller disposes it when it is disposable.</returns>
    private BaseStorage CSharpStorage(string kind)
        => kind == "journal"
            ? new JournalStorage(new JournalFileBackend(_path, lockObj: new JournalFileOpenLock(_path)))
            : new RDBStorage($"sqlite:///{_path}");

    /// <summary>
    ///     Python writes reports at steps 0, 3,000,000,000 and 2^62; OptunaSharp reads the same journal or SQLite file
    ///     and must reproduce the study exactly, steps included.
    /// </summary>
    /// <param name="kind">The storage both sides share: <c>journal</c> or <c>sqlite</c>.</param>
    [TestMethod]
    [TestCategory("RequiresPython")]
    [DataRow("journal")]
    [DataRow("sqlite")]
    public void PythonWritesStepsPastInt32_CSharpReadsThemExactly(string kind)
    {
        PythonSession.EnsureOrInconclusive();

        PyExec(
            $"study = optuna.create_study(study_name='steps', storage={PyStorage(kind)}, " +
            "sampler=optuna.samplers.RandomSampler(0))\n" +
            PyObjective +
            "study.optimize(_obj, n_trials=3)\n");
        var pyState = PyStudyState("study");
        if (kind == "sqlite")
            PyExec("study._storage.remove_session()\n");

        BaseStorage storage = CSharpStorage(kind);
        try
        {
            var csStudy = Optuna.LoadStudy("steps", storage);
            StudySnapshot.AssertParity(pyState, StudySnapshot.Capture(csStudy), DatetimeComparison.NullnessOnly);
            csStudy.Trials.Should().OnlyContain(t => t.LastStep == Huge && t.IntermediateValues[PastInt32] == 0.5);
        }
        finally
        {
            (storage as IDisposable)?.Dispose();
        }
    }

    /// <summary>
    ///     The reverse direction: OptunaSharp writes reports at steps 0, 3,000,000,000 and 2^62, and Python Optuna reads
    ///     the same journal or SQLite file back to an identical study.
    /// </summary>
    /// <param name="kind">The storage both sides share: <c>journal</c> or <c>sqlite</c>.</param>
    [TestMethod]
    [TestCategory("RequiresPython")]
    [DataRow("journal")]
    [DataRow("sqlite")]
    public void CSharpWritesStepsPastInt32_PythonReadsThemExactly(string kind)
    {
        PythonSession.EnsureOrInconclusive();

        BaseStorage storage = CSharpStorage(kind);
        StudySnapshot csState;
        try
        {
            var csStudy = Optuna.CreateStudy(storage: storage, studyName: "steps", sampler: new RandomSampler(0));
            csStudy.Optimize(Objective, nTrials: 3);
            csState = StudySnapshot.Capture(csStudy);
        }
        finally
        {
            // Release the SQLite file so Python opens a clean, committed database.
            (storage as IDisposable)?.Dispose();
        }

        PyExec($"study = optuna.load_study(study_name='steps', storage={PyStorage(kind)})\n");
        StudySnapshot.AssertParity(PyStudyState("study"), csState, DatetimeComparison.NullnessOnly);
        PyBool("all(t.last_step == 2 ** 62 and t.intermediate_values[3_000_000_000] == 0.5 for t in study.trials)")
            .Should().BeTrue("Python reads the 64-bit steps back exactly");
    }

    /// <summary>
    ///     The pruners at steps past Int32, live: the same deterministic report sequence, starting at step
    ///     3,000,000,000 and crossing several 1,000-step intervals after a warm-up, must give the same
    ///     prune/complete outcome for every trial in OptunaSharp and in Optuna. This exercises the interval arithmetic
    ///     (<c>_is_first_in_interval_step</c>) and the per-step percentile lookup with 64-bit steps.
    /// </summary>
    /// <param name="percentile">The pruner's percentile; 50 uses <c>MedianPruner</c> on both sides.</param>
    [TestMethod]
    [TestCategory("RequiresPython")]
    [DataRow(50.0)]
    [DataRow(25.0)]
    public void PrunerDecisions_AtStepsPastInt32_MatchLiveOptuna(double percentile)
    {
        PythonSession.EnsureOrInconclusive();
        const int nTrials = 10, nSteps = 12, warmup = 1_500, interval = 1_000, stride = 700;

        // Deterministic, RNG-free values: identical integer/IEEE arithmetic on both sides.
        static double Value(int i, int k) => ((i * 31 + k * 17) % 23) / 2.0 - 5.0 + k * 0.05;

        BasePruner pruner = percentile == 50.0
            ? new MedianPruner(nStartupTrials: 2, nWarmupSteps: warmup, intervalSteps: interval)
            : new PercentilePruner(percentile, nStartupTrials: 2, nWarmupSteps: warmup, intervalSteps: interval);
        var study = Optuna.CreateStudy(sampler: new RandomSampler(0), pruner: pruner);
        var outcomes = new List<string>();
        for (int i = 0; i < nTrials; i++)
        {
            var trial = study.Ask();
            trial.SuggestFloat("x", 0, 1);
            string outcome = "complete";
            for (int k = 0; k < nSteps; k++)
            {
                trial.Report(Value(i, k), PastInt32 + (long)k * stride);
                if (trial.ShouldPrune())
                {
                    outcome = $"p@{k}";
                    study.Tell(trial, values: null, state: TrialState.Pruned);
                    break;
                }
            }
            if (outcome == "complete")
                study.Tell(trial, Value(i, nSteps - 1));
            outcomes.Add(outcome);
        }

        string pyPruner = percentile == 50.0
            ? $"optuna.pruners.MedianPruner(n_startup_trials=2, n_warmup_steps={warmup}, interval_steps={interval})"
            : $"optuna.pruners.PercentilePruner(25.0, n_startup_trials=2, n_warmup_steps={warmup}, interval_steps={interval})";
        PyExec(
            "def _oss_value(i, k):\n" +
            "    return ((i * 31 + k * 17) % 23) / 2.0 - 5.0 + k * 0.05\n" +
            $"study = optuna.create_study(sampler=optuna.samplers.RandomSampler(0), pruner={pyPruner})\n" +
            "outcomes = []\n" +
            $"for i in range({nTrials}):\n" +
            "    t = study.ask()\n" +
            "    t.suggest_float('x', 0, 1)\n" +
            "    outcome = 'complete'\n" +
            $"    for k in range({nSteps}):\n" +
            $"        t.report(_oss_value(i, k), 3_000_000_000 + k * {stride})\n" +
            "        if t.should_prune():\n" +
            "            outcome = f'p@{k}'\n" +
            "            study.tell(t, state=optuna.trial.TrialState.PRUNED)\n" +
            "            break\n" +
            "    if outcome == 'complete':\n" +
            $"        study.tell(t, _oss_value(i, {nSteps - 1}))\n" +
            "    outcomes.append(outcome)\n" +
            "result = ','.join(outcomes)\n");

        string.Join(",", outcomes).Should().Be(PyStr("result"), "every prune/complete outcome must match live Optuna");
        outcomes.Should().Contain(o => o.StartsWith("p@", StringComparison.Ordinal), "the sequence must actually prune");
    }
}
