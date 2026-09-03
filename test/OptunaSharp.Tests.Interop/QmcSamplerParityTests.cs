using System;
using System.Collections.Generic;
using OptunaSharp.Distributions;
using OptunaSharp.Samplers;
using OptunaSharp.Storages;

namespace OptunaSharp.Tests.Interop;

/// <summary>
///     Full-state bit-parity of the (non-scrambled) QMCSampler against live Optuna. The Sobol' and
///     Halton engines are deterministic and reproduce <c>scipy.stats.qmc</c> exactly, the first trial
///     is drawn by the independent <see cref="RandomSampler" /> (MT19937), and the <c>qmc:&lt;sha256&gt;</c>
///     sample-id counter in the study system attrs is byte-identical — so the whole study state
///     (every trial's params/values/distributions and the system attrs) must agree bit-for-bit.
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
}
