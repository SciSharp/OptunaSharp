using System;
using System.Linq;
using OptunaSharp.Distributions;
using OptunaSharp.Samplers;
using OptunaSharp.Storages;

namespace OptunaSharp.Tests.Interop;

/// <summary>
///     Full-state bit-parity of <see cref="CmaEsSampler" /> against live Optuna (5.0.0.dev + cmaes
///     0.13.0). Every trial's params (external + internal IEEE bits), value, distributions, and the
///     <c>cma:generation</c> attribution must agree byte-for-byte: the CMA optimizer is bit-exact
///     (MT19937 draws, <c>eigh</c> via OpenBLAS, numpy pairwise/sequential reductions, and the
///     upper-triangle covariance symmetry preserved across the per-trial optimizer round-trip), the
///     per-trial reseed <c>randint(1,2**16)+trial.number</c> matches, and the first trial is the
///     independent <see cref="RandomSampler" />.
///
///     <para>The <c>cma:optimizer:*</c> system attrs are excluded from the comparison: Optuna pickles
///     the optimizer, OptunaSharp serializes its own JSON (<c>[Misaligned]</c> — pickle is a
///     non-goal, so no cross-language mid-CMA resume). Everything else, including <c>cma:generation</c>,
///     is compared bit-for-bit.</para>
/// </summary>
[TestClass]
public class CmaEsSamplerParityTests : OptunaInteropTestBase
{
    /// <summary>Drop the pickle-vs-JSON optimizer blob so parity compares the algorithmic state.</summary>
    private static void StripOptimizerAttrs(StudySnapshot s)
    {
        foreach (var t in s.Trials)
            foreach (var key in t.SystemAttrs.Keys.Where(k => k.Contains(":optimizer:")).ToList())
                t.SystemAttrs.Remove(key);
    }

    private void AssertParityExceptOptimizerBlob(Study csStudy, string pyCreateAndRun)
    {
        var csState = StudySnapshot.Capture(csStudy);
        PyExec(pyCreateAndRun);
        var pyState = PyStudyState("study");
        StripOptimizerAttrs(csState);
        StripOptimizerAttrs(pyState);
        StudySnapshot.AssertParity(pyState, csState, DatetimeComparison.NullnessOnly);
    }

    [TestMethod]
    [TestCategory("RequiresPython")]
    [DataRow(2, 0, 30)]
    [DataRow(2, 42, 30)]
    [DataRow(3, 7, 28)]
    [DataRow(5, 13, 26)]
    public void FloatSphere_FullState_IsBitIdentical(int dim, int seed, int nTrials)
    {
        PythonSession.EnsureOrInconclusive();

        var csStudy = Optuna.CreateStudy(
            storage: new InMemoryStorage(), studyName: "cma",
            sampler: new CmaEsSampler(seed: seed));
        csStudy.Optimize(SphereCs(dim), nTrials: nTrials);

        AssertParityExceptOptimizerBlob(csStudy,
            $"study = optuna.create_study(study_name='cma', sampler=optuna.samplers.CmaEsSampler(seed={seed}))\n" +
            SpherePy(dim) +
            $"study.optimize(_obj, n_trials={nTrials})\n");
    }

    [TestMethod]
    [TestCategory("RequiresPython")]
    [DataRow(3, 2)]
    [DataRow(4, 9)]
    public void LrAdapt_FullState_IsBitIdentical(int dim, int seed)
    {
        PythonSession.EnsureOrInconclusive();

        var csStudy = Optuna.CreateStudy(
            storage: new InMemoryStorage(), studyName: "cma",
            sampler: new CmaEsSampler(seed: seed, lrAdapt: true));
        csStudy.Optimize(SphereCs(dim), nTrials: 32);

        AssertParityExceptOptimizerBlob(csStudy,
            $"study = optuna.create_study(study_name='cma', sampler=optuna.samplers.CmaEsSampler(seed={seed}, lr_adapt=True))\n" +
            SpherePy(dim) +
            "study.optimize(_obj, n_trials=32)\n");
    }

    [TestMethod]
    [TestCategory("RequiresPython")]
    [DataRow(10)]
    [DataRow(20)]
    public void ExplicitPopsize_FullState_IsBitIdentical(int popsize)
    {
        PythonSession.EnsureOrInconclusive();

        var csStudy = Optuna.CreateStudy(
            storage: new InMemoryStorage(), studyName: "cma",
            sampler: new CmaEsSampler(seed: popsize, popsize: popsize));
        csStudy.Optimize(SphereCs(3), nTrials: popsize * 2 + 6);

        AssertParityExceptOptimizerBlob(csStudy,
            $"study = optuna.create_study(study_name='cma', sampler=optuna.samplers.CmaEsSampler(seed={popsize}, popsize={popsize}))\n" +
            SpherePy(3) +
            $"study.optimize(_obj, n_trials={popsize * 2 + 6})\n");
    }

    [TestMethod]
    [TestCategory("RequiresPython")]
    public void MixedSpace_FullState_IsBitIdentical()
    {
        PythonSession.EnsureOrInconclusive();

        // float (relative CMA) + int (relative CMA) + categorical (independent RandomSampler).
        var csStudy = Optuna.CreateStudy(
            storage: new InMemoryStorage(), studyName: "cma",
            sampler: new CmaEsSampler(seed: 5));
        csStudy.Optimize(t =>
        {
            double a = t.SuggestFloat("a", -4, 4);
            long b = t.SuggestInt("bb", -5, 5);
            double c = t.SuggestFloat("c", 0.01, 100, log: true);
            t.SuggestCategorical("dd", new CategoricalChoice[] { "p", "q", "r" });
            return Math.Pow(a - 1.5, 2) + Math.Pow(b + 2, 2) + Math.Pow(Math.Log(c) - 1.0, 2);
        }, nTrials: 24);

        AssertParityExceptOptimizerBlob(csStudy,
            "study = optuna.create_study(study_name='cma', sampler=optuna.samplers.CmaEsSampler(seed=5))\n" +
            "import numpy as np\n" +
            "def _obj(t):\n" +
            "    a = t.suggest_float('a', -4, 4)\n" +
            "    b = t.suggest_int('bb', -5, 5)\n" +
            "    c = t.suggest_float('c', 0.01, 100, log=True)\n" +
            "    t.suggest_categorical('dd', ['p', 'q', 'r'])\n" +
            "    return (a - 1.5) ** 2 + (b + 2) ** 2 + (np.log(c) - 1.0) ** 2\n" +
            "study.optimize(_obj, n_trials=24)\n");
    }

    // Deterministic shifted sphere with an explicit left-to-right sum (NOT Python's builtin sum(),
    // which since 3.12 uses Neumaier compensated summation and would drift ~1 ULP from the C# sum).
    private static Func<Trial, double> SphereCs(int dim)
    {
        var sh = Enumerable.Range(0, dim).Select(i => (i - (dim - 1) * 0.5) * 0.4).ToArray();
        return t =>
        {
            double v = 0;
            for (int i = 0; i < dim; i++) v += Math.Pow(t.SuggestFloat($"x{i}", -5, 5) - sh[i], 2);
            return v;
        };
    }

    private static string SpherePy(int dim)
    {
        var sh = string.Join(", ", Enumerable.Range(0, dim).Select(i => ((i - (dim - 1) * 0.5) * 0.4).ToString("R")));
        return
            $"_sh = [{sh}]\n" +
            "def _obj(t):\n" +
            $"    _x = [t.suggest_float(f'x{{i}}', -5, 5) for i in range({dim})]\n" +
            "    _v = 0.0\n" +
            $"    for i in range({dim}): _v = _v + (_x[i] - _sh[i]) ** 2\n" +
            "    return _v\n";
    }
}
