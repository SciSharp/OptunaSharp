using System;
using System.Collections.Generic;
using OptunaSharp.Distributions;
using OptunaSharp.Samplers;
using OptunaSharp.Storages;

namespace OptunaSharp.Tests.Interop;

/// <summary>
///     Full-state bit-parity of the QMCSampler against live Optuna. The Sobol' and Halton engines are
///     deterministic and reproduce <c>scipy.stats.qmc</c> exactly — including <b>scrambled Sobol'</b>
///     (the linear-matrix-scramble + digital shift) and <b>scrambled Halton</b> (Owen's digit
///     permutations, <c>_van_der_corput_permutations</c>), both seeded by
///     <c>numpy.random.default_rng(seed)</c> — the first trial is drawn by the independent
///     <see cref="RandomSampler" /> (MT19937), and the
///     <c>qmc:&lt;sha256&gt;</c> sample-id counter in the study system attrs is byte-identical (its hash
///     folds in <c>scramble</c>/<c>seed</c> when scrambling) — so the whole study state (every trial's
///     params/values/distributions and the system attrs) must agree bit-for-bit.
/// </summary>
[TestClass]
public class QmcSamplerParityTests : OptunaInteropTestBase
{
    private void AssertFullParity(Study csStudy, string pyCreateAndRun)
    {
        var csState = StudySnapshot.Capture(csStudy);
        PyExec(pyCreateAndRun);
        var pyState = PyStudyState("study");
        StudySnapshot.AssertParity(pyState, csState, DatetimeComparison.NullnessOnly);
    }

    [TestMethod]
    [TestCategory("RequiresPython")]
    [DataRow("sobol", 0)]
    [DataRow("sobol", 42)]
    [DataRow("halton", 0)]
    [DataRow("halton", 42)]
    public void MixedSpace_FullState_IsBitIdentical(string qmcType, int seed)
    {
        PythonSession.EnsureOrInconclusive();

        // float / int / categorical / log-float exercise the whole transform + pseudo-categorical
        // mapping, and the first trial uses the independent RandomSampler(seed) → bit-exact.
        var csStudy = Optuna.CreateStudy(
            storage: new InMemoryStorage(), studyName: "qmc",
            sampler: new QMCSampler(qmcType: qmcType, scramble: false, seed: seed));
        csStudy.Optimize(t =>
        {
            double x = t.SuggestFloat("x", -1, 1);
            long y = t.SuggestInt("y", -5, 5);
            t.SuggestCategorical("z", new CategoricalChoice[] { "a", "b", "c" });
            t.SuggestFloat("lx", 1e-3, 1e3, log: true);
            return Math.Pow(x, 2.0) + y;
        }, nTrials: 16);

        AssertFullParity(csStudy,
            "study = optuna.create_study(study_name='qmc', " +
            $"sampler=optuna.samplers.QMCSampler(qmc_type='{qmcType}', scramble=False, seed={seed}))\n" +
            "def _obj(t):\n" +
            "    x = t.suggest_float('x', -1, 1)\n" +
            "    y = t.suggest_int('y', -5, 5)\n" +
            "    t.suggest_categorical('z', ['a', 'b', 'c'])\n" +
            "    t.suggest_float('lx', 1e-3, 1e3, log=True)\n" +
            "    return x ** 2 + y\n" +
            "study.optimize(_obj, n_trials=16)\n");
    }

    [TestMethod]
    [TestCategory("RequiresPython")]
    [DataRow("sobol")]
    [DataRow("halton")]
    public void IntStepAndPowerOfTwo_FullState_IsBitIdentical(string qmcType)
    {
        PythonSession.EnsureOrInconclusive();

        // Stepped int + two floats over a power-of-two trial count (the Sobol' balance regime).
        var csStudy = Optuna.CreateStudy(
            storage: new InMemoryStorage(), studyName: "qmc",
            sampler: new QMCSampler(qmcType: qmcType, scramble: false, seed: 7));
        csStudy.Optimize(t =>
        {
            double a = t.SuggestFloat("a", -10, 10);
            double b = t.SuggestFloat("b", 0, 5, step: 0.5);
            long c = t.SuggestInt("c", 0, 100, step: 10);
            return a + b + c;
        }, nTrials: 32);

        AssertFullParity(csStudy,
            "study = optuna.create_study(study_name='qmc', " +
            $"sampler=optuna.samplers.QMCSampler(qmc_type='{qmcType}', scramble=False, seed=7))\n" +
            "def _obj(t):\n" +
            "    a = t.suggest_float('a', -10, 10)\n" +
            "    b = t.suggest_float('b', 0, 5, step=0.5)\n" +
            "    c = t.suggest_int('c', 0, 100, step=10)\n" +
            "    return a + b + c\n" +
            "study.optimize(_obj, n_trials=32)\n");
    }

    [TestMethod]
    [TestCategory("RequiresPython")]
    [DataRow(0)]
    [DataRow(42)]
    public void ScrambledSobol_MixedSpace_FullState_IsBitIdentical(int seed)
    {
        PythonSession.EnsureOrInconclusive();

        // Scrambled Sobol' over the same mixed space. The scramble is seeded by the explicit `seed`
        // (fixed for the sampler instance), so the whole study state stays bit-identical to Python's.
        var csStudy = Optuna.CreateStudy(
            storage: new InMemoryStorage(), studyName: "qmc",
            sampler: new QMCSampler(qmcType: "sobol", scramble: true, seed: seed));
        csStudy.Optimize(t =>
        {
            double x = t.SuggestFloat("x", -1, 1);
            long y = t.SuggestInt("y", -5, 5);
            t.SuggestCategorical("z", new CategoricalChoice[] { "a", "b", "c" });
            t.SuggestFloat("lx", 1e-3, 1e3, log: true);
            return Math.Pow(x, 2.0) + y;
        }, nTrials: 16);

        AssertFullParity(csStudy,
            "study = optuna.create_study(study_name='qmc', " +
            $"sampler=optuna.samplers.QMCSampler(qmc_type='sobol', scramble=True, seed={seed}))\n" +
            "def _obj(t):\n" +
            "    x = t.suggest_float('x', -1, 1)\n" +
            "    y = t.suggest_int('y', -5, 5)\n" +
            "    t.suggest_categorical('z', ['a', 'b', 'c'])\n" +
            "    t.suggest_float('lx', 1e-3, 1e3, log=True)\n" +
            "    return x ** 2 + y\n" +
            "study.optimize(_obj, n_trials=16)\n");
    }

    [TestMethod]
    [TestCategory("RequiresPython")]
    public void ScrambledSobol_PowerOfTwo_FullState_IsBitIdentical()
    {
        PythonSession.EnsureOrInconclusive();

        // Scrambled Sobol' over a power-of-two trial count (the Sobol' balance regime) + a stepped int.
        var csStudy = Optuna.CreateStudy(
            storage: new InMemoryStorage(), studyName: "qmc",
            sampler: new QMCSampler(qmcType: "sobol", scramble: true, seed: 7));
        csStudy.Optimize(t =>
        {
            double a = t.SuggestFloat("a", -10, 10);
            double b = t.SuggestFloat("b", 0, 5, step: 0.5);
            long c = t.SuggestInt("c", 0, 100, step: 10);
            return a + b + c;
        }, nTrials: 32);

        AssertFullParity(csStudy,
            "study = optuna.create_study(study_name='qmc', " +
            "sampler=optuna.samplers.QMCSampler(qmc_type='sobol', scramble=True, seed=7))\n" +
            "def _obj(t):\n" +
            "    a = t.suggest_float('a', -10, 10)\n" +
            "    b = t.suggest_float('b', 0, 5, step=0.5)\n" +
            "    c = t.suggest_int('c', 0, 100, step=10)\n" +
            "    return a + b + c\n" +
            "study.optimize(_obj, n_trials=32)\n");
    }

    [TestMethod]
    [TestCategory("RequiresPython")]
    [DataRow(0)]
    [DataRow(42)]
    public void ScrambledHalton_MixedSpace_FullState_IsBitIdentical(int seed)
    {
        PythonSession.EnsureOrInconclusive();

        // Scrambled Halton (Owen digit permutations) over the mixed space. The permutations are drawn
        // from default_rng(seed) — fixed for the sampler instance — so the whole study state stays
        // bit-identical to Python's (the last Phase 2.12 combination to reach live parity).
        var csStudy = Optuna.CreateStudy(
            storage: new InMemoryStorage(), studyName: "qmc",
            sampler: new QMCSampler(qmcType: "halton", scramble: true, seed: seed));
        csStudy.Optimize(t =>
        {
            double x = t.SuggestFloat("x", -1, 1);
            long y = t.SuggestInt("y", -5, 5);
            t.SuggestCategorical("z", new CategoricalChoice[] { "a", "b", "c" });
            t.SuggestFloat("lx", 1e-3, 1e3, log: true);
            return Math.Pow(x, 2.0) + y;
        }, nTrials: 16);

        AssertFullParity(csStudy,
            "study = optuna.create_study(study_name='qmc', " +
            $"sampler=optuna.samplers.QMCSampler(qmc_type='halton', scramble=True, seed={seed}))\n" +
            "def _obj(t):\n" +
            "    x = t.suggest_float('x', -1, 1)\n" +
            "    y = t.suggest_int('y', -5, 5)\n" +
            "    t.suggest_categorical('z', ['a', 'b', 'c'])\n" +
            "    t.suggest_float('lx', 1e-3, 1e3, log=True)\n" +
            "    return x ** 2 + y\n" +
            "study.optimize(_obj, n_trials=16)\n");
    }

    [TestMethod]
    [TestCategory("RequiresPython")]
    public void ScrambledHalton_ManyDims_FullState_IsBitIdentical()
    {
        PythonSession.EnsureOrInconclusive();

        // A wider search space exercises higher Halton bases (2, 3, 5, 7, 11, …), so more per-dimension
        // permutation draws are consumed from the shared default_rng(seed) stream in dimension order.
        var csStudy = Optuna.CreateStudy(
            storage: new InMemoryStorage(), studyName: "qmc",
            sampler: new QMCSampler(qmcType: "halton", scramble: true, seed: 7));
        csStudy.Optimize(t =>
        {
            double sum = 0;
            for (int i = 0; i < 6; i++)
                sum += t.SuggestFloat($"x{i}", -1, 1);
            return sum;
        }, nTrials: 20);

        AssertFullParity(csStudy,
            "study = optuna.create_study(study_name='qmc', " +
            "sampler=optuna.samplers.QMCSampler(qmc_type='halton', scramble=True, seed=7))\n" +
            "def _obj(t):\n" +
            // Explicit left-to-right accumulation, NOT the sum() builtin — CPython 3.12+ sums floats with
            // compensated (Neumaier) summation, which would differ from C#'s naive `+=` by up to 1 ULP in
            // the objective value (the params themselves stay bit-identical).
            "    v = 0.0\n" +
            "    for i in range(6):\n" +
            "        v += t.suggest_float(f'x{i}', -1, 1)\n" +
            "    return v\n" +
            "study.optimize(_obj, n_trials=20)\n");
    }

    /// <summary>
    ///     Plan 14 gate: QMC now computes its <c>qmc:&lt;sha256&gt;</c> sample-id key once per search space and reuses it while
    ///     the same distribution objects come back. A 40-parameter study (float, log-float, int and categorical) over 300
    ///     trials, for each engine and scrambling, must keep the whole state, the counter in the study system attrs
    ///     included, bit-identical to live Optuna.
    /// </summary>
    /// <param name="qmcType">The engine: <c>sobol</c> or <c>halton</c>.</param>
    /// <param name="scramble">Whether the engine is scrambled (the seed then enters the key).</param>
    [TestMethod]
    [TestCategory("RequiresPython")]
    [DataRow("sobol", false)]
    [DataRow("halton", false)]
    [DataRow("sobol", true)]
    [DataRow("halton", true)]
    public void LongRun_40Parameters_300Trials_FullState_IsBitIdentical(string qmcType, bool scramble)
    {
        PythonSession.EnsureOrInconclusive();

        var csStudy = Optuna.CreateStudy(
            storage: new InMemoryStorage(), studyName: "qmc",
            sampler: new QMCSampler(qmcType: qmcType, scramble: scramble, seed: 11));
        csStudy.Optimize(t =>
        {
            double v = 0;
            for (int i = 0; i < 37; i++)
                v += t.SuggestFloat($"x{i}", -1, 1);
            v += Math.Log(t.SuggestFloat("lx", 1e-3, 1e3, log: true));
            v += t.SuggestInt("n", -5, 5);
            v += t.SuggestCategorical("c", new[] { "a", "b", "cc" }).Length;
            return v;
        }, nTrials: 300);

        AssertFullParity(csStudy,
            "import math\n" +
            "study = optuna.create_study(study_name='qmc', " +
            $"sampler=optuna.samplers.QMCSampler(qmc_type='{qmcType}', scramble={(scramble ? "True" : "False")}, seed=11))\n" +
            "def _obj(t):\n" +
            "    v = 0.0\n" +
            "    for i in range(37):\n" +
            "        v += t.suggest_float(f'x{i}', -1, 1)\n" +
            "    v += math.log(t.suggest_float('lx', 1e-3, 1e3, log=True))\n" +
            "    v += t.suggest_int('n', -5, 5)\n" +
            "    v += len(t.suggest_categorical('c', ['a', 'b', 'cc']))\n" +
            "    return v\n" +
            "study.optimize(_obj, n_trials=300)\n");
    }
}
