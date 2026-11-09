using System;
using System.Collections.Generic;
using System.Linq;
using AwesomeAssertions;
using OptunaSharp.Distributions;
using OptunaSharp.Samplers;
using OptunaSharp.Storages;

namespace OptunaSharp.Tests.Interop;

/// <summary>
///     Plan 13, item S12, against live Optuna: studies run on <c>InMemoryStorage(compactFinishedTrials: true)</c> give
///     Python's trials bit for bit (TPE to its documented erf tolerance), attributes included, while every finished
///     record stays compact through the run. The compact storage is an opt-in memory layout, so the samplers must not be
///     able to tell it from Optuna's.
/// </summary>
[TestClass]
public class CompactStorageParityTests : OptunaInteropTestBase
{
    /// <summary>The storage under test: Optuna's in-memory storage with finished trials kept compactly.</summary>
    /// <returns>A new compacting storage.</returns>
    private static InMemoryStorage CompactStorage() => new(compactFinishedTrials: true);

    /// <summary>Every finished record of the study is still compact after the run.</summary>
    /// <param name="study">A study on a compacting storage.</param>
    private static void AssertRecordsStayedCompact(Study study)
    {
        var records = study.Storage.GetAllTrials(study.StudyId, deepcopy: false);
        records.Should().NotBeEmpty();
        records.Where(r => r.State.IsFinished()).Should().OnlyContain(r => r.IsCompact,
            "the sampler and the snapshot read finished records through the read-only accessors");
    }

    /// <summary>Runs the Python study and asserts bit-exact full-state parity with the compact C# one.</summary>
    /// <param name="csStudy">The finished OptunaSharp study on a compacting storage.</param>
    /// <param name="pyCreateAndRun">Python that creates and runs the same study as <c>study</c>.</param>
    private void AssertFullParity(Study csStudy, string pyCreateAndRun)
    {
        AssertRecordsStayedCompact(csStudy);
        var csState = StudySnapshot.Capture(csStudy);
        PyExec(pyCreateAndRun);
        StudySnapshot.AssertParity(PyStudyState("study"), csState, DatetimeComparison.NullnessOnly);
        AssertRecordsStayedCompact(csStudy);
    }

    /// <summary>
    ///     Random search over every parameter kind, writing attributes the compact storage stores as bytes (a long ASCII
    ///     string, an odd-length lowercase hexadecimal string), a non-ASCII string it keeps, and a nested list: every
    ///     trial and attribute matches Python's bit for bit.
    /// </summary>
    /// <param name="seed">Sampler seed.</param>
    [TestMethod]
    [TestCategory("RequiresPython")]
    [DataRow(0)]
    [DataRow(17)]
    public void RandomWithAttributes_FullState_IsBitIdentical(int seed)
    {
        PythonSession.EnsureOrInconclusive();

        var csStudy = Optuna.CreateStudy(storage: CompactStorage(), studyName: "compact",
            sampler: new RandomSampler(seed: seed));
        csStudy.Optimize(t =>
        {
            double x = t.SuggestFloat("x", -5, 5);
            t.SuggestFloat("lx", 1e-3, 1e3, log: true);
            t.SuggestFloat("sx", -1, 1, step: 0.25);
            long n = t.SuggestInt("n", -3, 3);
            t.SuggestInt("ln", 1, 64, log: true);
            t.SuggestCategorical("c", new CategoricalChoice[] { "a", CategoricalChoice.None, true, 7L, 2.5 });
            t.SetUserAttr("blob", new string('q', 90) + t.Number);
            t.SetUserAttr("hex", string.Concat(Enumerable.Repeat("0a1b2c3d4e5f", 6)) + "9");
            t.SetUserAttr("unicode", new string('é', 70));
            t.SetUserAttr("nested", new List<object?> { (long)t.Number, "two", 3.5, null });
            return x * x + n;
        }, nTrials: 30);

        AssertFullParity(csStudy,
            $"study = optuna.create_study(study_name='compact', sampler=optuna.samplers.RandomSampler(seed={seed}))\n" +
            "def _obj(t):\n" +
            "    x = t.suggest_float('x', -5, 5)\n" +
            "    t.suggest_float('lx', 1e-3, 1e3, log=True)\n" +
            "    t.suggest_float('sx', -1, 1, step=0.25)\n" +
            "    n = t.suggest_int('n', -3, 3)\n" +
            "    t.suggest_int('ln', 1, 64, log=True)\n" +
            "    t.suggest_categorical('c', ['a', None, True, 7, 2.5])\n" +
            "    t.set_user_attr('blob', 'q' * 90 + str(t.number))\n" +
            "    t.set_user_attr('hex', '0a1b2c3d4e5f' * 6 + '9')\n" +
            "    t.set_user_attr('unicode', '\\u00e9' * 70)\n" +
            "    t.set_user_attr('nested', [t.number, 'two', 3.5, None])\n" +
            "    return x * x + n\n" +
            "study.optimize(_obj, n_trials=30)\n");
    }

    /// <summary>NSGA-II over a mixed space reads its parents from compact records and matches Python bit for bit.</summary>
    /// <param name="seed">Sampler seed.</param>
    [TestMethod]
    [TestCategory("RequiresPython")]
    [DataRow(0)]
    [DataRow(42)]
    public void NsgaiiMixedSpace_FullState_IsBitIdentical(int seed)
    {
        PythonSession.EnsureOrInconclusive();

        var csStudy = Optuna.CreateStudy(storage: CompactStorage(), studyName: "compact",
            sampler: new NSGAIISampler(populationSize: 5, seed: seed), direction: StudyDirection.Minimize);
        csStudy.Optimize(t =>
        {
            double x = t.SuggestFloat("x", -5, 5);
            long y = t.SuggestInt("y", 0, 10);
            t.SuggestCategorical("z", new CategoricalChoice[] { "a", "b", "c" });
            double lx = t.SuggestFloat("lx", 1e-3, 1e3, log: true);
            return Math.Pow(x - 2, 2) + y + lx;
        }, nTrials: 24);

        AssertFullParity(csStudy,
            "study = optuna.create_study(study_name='compact', direction='minimize', " +
            $"sampler=optuna.samplers.NSGAIISampler(population_size=5, seed={seed}))\n" +
            "def _obj(t):\n" +
            "    x = t.suggest_float('x', -5, 5)\n" +
            "    y = t.suggest_int('y', 0, 10)\n" +
            "    t.suggest_categorical('z', ['a', 'b', 'c'])\n" +
            "    lx = t.suggest_float('lx', 1e-3, 1e3, log=True)\n" +
            "    return (x - 2) ** 2 + y + lx\n" +
            "study.optimize(_obj, n_trials=24)\n");
    }

    /// <summary>QMC reads its first finished trial's distributions from a compact record and matches Python bit for bit.</summary>
    [TestMethod]
    [TestCategory("RequiresPython")]
    public void Qmc_FullState_IsBitIdentical()
    {
        PythonSession.EnsureOrInconclusive();

        var csStudy = Optuna.CreateStudy(storage: CompactStorage(), studyName: "compact",
            sampler: new QMCSampler(seed: 3));
        csStudy.Optimize(t => t.SuggestFloat("x", -1, 1) + t.SuggestInt("n", 0, 9), nTrials: 16);

        AssertFullParity(csStudy,
            "import warnings; warnings.simplefilter('ignore')\n" +
            "study = optuna.create_study(study_name='compact', sampler=optuna.samplers.QMCSampler(seed=3))\n" +
            "study.optimize(lambda t: t.suggest_float('x', -1, 1) + t.suggest_int('n', 0, 9), n_trials=16)\n");
    }

    /// <summary>
    ///     CMA-ES restores its optimizer from the hexadecimal chunks the compact storage holds as bytes: every trial
    ///     matches Python bit for bit, and the stored chunks are exactly the strings an uncompacted C# run stores
    ///     (whose pickles <see cref="CmaEsSamplerParityTests" /> already proves equal to Python's).
    /// </summary>
    /// <param name="seed">Sampler seed.</param>
    [TestMethod]
    [TestCategory("RequiresPython")]
    [DataRow(0)]
    [DataRow(7)]
    public void CmaEs_FullState_IsBitIdentical_AndStoresTheSameChunks(int seed)
    {
        PythonSession.EnsureBitExactHostOrInconclusive();

        // Summed left to right on both sides: Python's sum() compensates (Neumaier, CPython 3.12+), which would change
        // the last bit of some values for a reason that has nothing to do with the storage.
        Func<Trial, double> sphere = t =>
        {
            double v = 0;
            for (int i = 0; i < 3; i++)
                v += Math.Pow(t.SuggestFloat($"x{i}", -3, 3) - 1, 2);
            return v;
        };
        var csStudy = Optuna.CreateStudy(storage: CompactStorage(), studyName: "cma", sampler: new CmaEsSampler(seed: seed));
        csStudy.Optimize(sphere, nTrials: 28);
        var plainStudy = Optuna.CreateStudy(storage: new InMemoryStorage(), studyName: "cma", sampler: new CmaEsSampler(seed: seed));
        plainStudy.Optimize(sphere, nTrials: 28);
        AssertRecordsStayedCompact(csStudy);

        var csState = StudySnapshot.Capture(csStudy);
        var plainState = StudySnapshot.Capture(plainStudy);
        int chunks = 0;
        for (int i = 0; i < csState.Trials.Count; i++)
        {
            var compactChunks = csState.Trials[i].SystemAttrs.Where(kv => kv.Key.Contains(":optimizer:")).ToList();
            var plainChunks = plainState.Trials[i].SystemAttrs.Where(kv => kv.Key.Contains(":optimizer:")).ToList();
            compactChunks.Select(kv => kv.Key).Should().Equal(plainChunks.Select(kv => kv.Key));
            compactChunks.Select(kv => (string)kv.Value!).Should().Equal(plainChunks.Select(kv => (string)kv.Value!));
            chunks += compactChunks.Count;
        }
        chunks.Should().BePositive("the study must have stored optimizer states");

        PyExec(
            $"study = optuna.create_study(study_name='cma', sampler=optuna.samplers.CmaEsSampler(seed={seed}))\n" +
            "def _obj(t):\n" +
            "    v = 0.0\n" +
            "    for i in range(3):\n" +
            "        v += (t.suggest_float(f'x{i}', -3, 3) - 1) ** 2\n" +
            "    return v\n" +
            "study.optimize(_obj, n_trials=28)\n");
        var pyState = PyStudyState("study");
        foreach (var snapshot in new[] { pyState, csState })
            foreach (var trial in snapshot.Trials)
                foreach (string key in trial.SystemAttrs.Keys.Where(k => k.Contains(":optimizer:")).ToList())
                    trial.SystemAttrs.Remove(key);
        StudySnapshot.AssertParity(pyState, csState, DatetimeComparison.NullnessOnly);
    }

    /// <summary>
    ///     Multivariate TPE reads every finished trial's parameters on every ask through the compact views and stays in
    ///     lock-step with Python: same trial count, states and parameter sets, continuous values within TPE's documented
    ///     erf tolerance and categorical values exact.
    /// </summary>
    /// <param name="seed">Sampler seed.</param>
    [TestMethod]
    [TestCategory("RequiresPython")]
    [DataRow(0)]
    [DataRow(42)]
    public void TpeMultivariate_TrajectoryMatches(int seed)
    {
        PythonSession.EnsureOrInconclusive();
        const double tolerance = 1e-9;

        var csStudy = Optuna.CreateStudy(storage: CompactStorage(), studyName: "tpe", sampler: new TPESampler(seed: seed));
        csStudy.Optimize(t =>
        {
            double x1 = t.SuggestFloat("x1", -5, 5);
            t.SuggestFloat("x2", -5, 5);
            t.SuggestFloat("lx", 1e-3, 1e3, log: true);
            t.SuggestCategorical("cc", new CategoricalChoice[] { "a", "b", "c" });
            return x1 * x1;
        }, nTrials: 30);
        AssertRecordsStayedCompact(csStudy);
        var csState = StudySnapshot.Capture(csStudy);

        PyExec(
            $"study = optuna.create_study(study_name='tpe', sampler=optuna.samplers.TPESampler(seed={seed}))\n" +
            "def _obj(t):\n" +
            "    x1 = t.suggest_float('x1', -5, 5)\n" +
            "    t.suggest_float('x2', -5, 5)\n" +
            "    t.suggest_float('lx', 1e-3, 1e3, log=True)\n" +
            "    t.suggest_categorical('cc', ['a', 'b', 'c'])\n" +
            "    return x1 ** 2\n" +
            "study.optimize(_obj, n_trials=30)\n");
        var pyState = PyStudyState("study");

        csState.Trials.Count.Should().Be(pyState.Trials.Count);
        for (int i = 0; i < pyState.Trials.Count; i++)
        {
            var expected = pyState.Trials[i];
            var actual = csState.Trials[i];
            actual.State.Should().Be(expected.State, $"trial {i} state");
            actual.Params.Keys.Should().BeEquivalentTo(expected.Params.Keys, $"trial {i} parameter set");
            foreach (string key in expected.Params.Keys)
            {
                if (expected.Params[key] is double e && actual.Params[key] is double a)
                    a.Should().BeApproximately(e, Math.Max(tolerance, tolerance * Math.Abs(e)), $"trial {i} '{key}'");
                else
                    actual.Params[key].Should().Be(expected.Params[key], $"trial {i} '{key}'");
            }
        }
    }
}
