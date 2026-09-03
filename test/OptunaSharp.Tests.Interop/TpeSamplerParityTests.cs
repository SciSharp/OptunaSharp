using System;
using System.Linq;
using AwesomeAssertions;
using OptunaSharp.Distributions;
using OptunaSharp.Samplers;
using OptunaSharp.Storages;

namespace OptunaSharp.Tests.Interop;

/// <summary>
///     Trajectory parity of the TPE sampler against live Optuna (the vendored 5.0.0.dev reference).
///     Unlike the bit-exact samplers, TPE routes its densities through the FreeBSD <c>erf</c> port
///     (~1-3 ULP from Python's ucrt <c>math.erf</c>), so parity is asserted to a small tolerance on the
///     continuous params and exactly on the int/categorical params — every trial's <c>number</c>,
///     <c>state</c>, param set and values must still line up, proving the whole study stays in RNG
///     lock-step with Python.
///
///     <para>The suites here cover the paths that are exact-to-tolerance against Optuna: multivariate
///     TPE over continuous (float / log-float) and categorical params, and the independent
///     (<c>multivariate=false</c>) path. Studies whose <em>numerical</em> observations repeat (discrete
///     / int params in the multivariate path) are excluded on purpose — there they diverge only through
///     NumPy's x86-simd-sort tie order inside <c>compute_sigmas</c>, a documented divergence (see the
///     remark on <c>ParzenEstimator</c>).</para>
/// </summary>
[TestClass]
public class TpeSamplerParityTests : OptunaInteropTestBase
{
    private const double FloatTol = 1e-9;

    private void AssertTrajectoryParity(Study csStudy, string pyCreateAndRun)
    {
        var csState = StudySnapshot.Capture(csStudy);
        PyExec(pyCreateAndRun);
        var pyState = PyStudyState("study");

        csState.Trials.Count.Should().Be(pyState.Trials.Count, "trajectory length must match");
        for (int i = 0; i < pyState.Trials.Count; i++)
        {
            var e = pyState.Trials[i];
            var a = csState.Trials[i];
            a.Number.Should().Be(e.Number);
            a.State.Should().Be(e.State, $"trial {e.Number} state");
            a.Params.Keys.Should().BeEquivalentTo(e.Params.Keys, $"trial {e.Number} param set");

            foreach (var key in e.Params.Keys)
            {
                object? ev = e.Params[key];
                object? av = a.Params[key];
                if (ev is double ed && av is double ad)
                    ad.Should().BeApproximately(ed, Math.Max(FloatTol, FloatTol * Math.Abs(ed)),
                        $"trial {e.Number} param '{key}'");
                else
                    av.Should().Be(ev, $"trial {e.Number} param '{key}'");
            }

            (a.Values is null).Should().Be(e.Values is null, $"trial {e.Number} values null-ness");
            if (e.Values is not null)
            {
                a.Values!.Count.Should().Be(e.Values.Count);
                for (int k = 0; k < e.Values.Count; k++)
                    a.Values[k].Should().BeApproximately(e.Values[k],
                        Math.Max(FloatTol, FloatTol * Math.Abs(e.Values[k])), $"trial {e.Number} values[{k}]");
            }
        }
    }

    [TestMethod]
    [TestCategory("RequiresPython")]
    [DataRow(0)]
    [DataRow(7)]
    [DataRow(42)]
    public void MultivariateContinuousAndCategorical_TrajectoryMatches(int seed)
    {
        PythonSession.EnsureOrInconclusive();

        // Only x1 drives the objective (so the below/above split is unambiguous); x2 / lx / cc are
        // still sampled and compared — exercising the truncnorm, log-normal and categorical kinds.
        var csStudy = Optuna.CreateStudy(
            storage: new InMemoryStorage(), sampler: new TPESampler(seed: seed), studyName: "tpe");
        csStudy.Optimize(t =>
        {
            var x1 = t.SuggestFloat("x1", -5, 5);
            t.SuggestFloat("x2", -5, 5);
            t.SuggestFloat("lx", 1e-3, 1e3, log: true);
            t.SuggestCategorical("cc", new CategoricalChoice[] { "a", "b", "c" });
            return Math.Pow(x1, 2.0);
        }, nTrials: 30);

        AssertTrajectoryParity(csStudy,
            $"study = optuna.create_study(study_name='tpe', sampler=optuna.samplers.TPESampler(seed={seed}))\n" +
            "def _obj(t):\n" +
            "    x1 = t.suggest_float('x1', -5, 5)\n" +
            "    t.suggest_float('x2', -5, 5)\n" +
            "    t.suggest_float('lx', 1e-3, 1e3, log=True)\n" +
            "    t.suggest_categorical('cc', ['a', 'b', 'c'])\n" +
            "    return x1 ** 2\n" +
            "study.optimize(_obj, n_trials=30)\n");
    }

    [TestMethod]
    [TestCategory("RequiresPython")]
    [DataRow(1)]
    [DataRow(123)]
    public void MultivariateSumOfSquares_TrajectoryMatches(int seed)
    {
        PythonSession.EnsureOrInconclusive();

        var csStudy = Optuna.CreateStudy(
            storage: new InMemoryStorage(), sampler: new TPESampler(seed: seed), studyName: "tpe");
        csStudy.Optimize(t =>
        {
            var x = t.SuggestFloat("x", -10, 10);
            var y = t.SuggestFloat("y", -10, 10);
            return Math.Pow(x, 2.0) + Math.Pow(y, 2.0);
        }, nTrials: 25);

        AssertTrajectoryParity(csStudy,
            $"study = optuna.create_study(study_name='tpe', sampler=optuna.samplers.TPESampler(seed={seed}))\n" +
            "study.optimize(lambda t: t.suggest_float('x', -10, 10) ** 2 + t.suggest_float('y', -10, 10) ** 2," +
            " n_trials=25)\n");
    }

    [TestMethod]
    [TestCategory("RequiresPython")]
    [DataRow(0)]
    [DataRow(42)]
    public void IndependentMode_TrajectoryMatches(int seed)
    {
        PythonSession.EnsureOrInconclusive();

        // multivariate=False -> each parameter is sampled independently via sample_independent.
        var csStudy = Optuna.CreateStudy(
            storage: new InMemoryStorage(), sampler: new TPESampler(seed: seed, multivariate: false), studyName: "tpe");
        csStudy.Optimize(t =>
        {
            var x1 = t.SuggestFloat("x1", -5, 5);
            t.SuggestFloat("lx", 1e-2, 1e2, log: true);
            t.SuggestCategorical("cc", new CategoricalChoice[] { "p", "q" });
            return Math.Pow(x1, 2.0);
        }, nTrials: 28);

        AssertTrajectoryParity(csStudy,
            "study = optuna.create_study(study_name='tpe', " +
            $"sampler=optuna.samplers.TPESampler(seed={seed}, multivariate=False))\n" +
            "def _obj(t):\n" +
            "    x1 = t.suggest_float('x1', -5, 5)\n" +
            "    t.suggest_float('lx', 1e-2, 1e2, log=True)\n" +
            "    t.suggest_categorical('cc', ['p', 'q'])\n" +
            "    return x1 ** 2\n" +
            "study.optimize(_obj, n_trials=28)\n");
    }

    [TestMethod]
    [TestCategory("RequiresPython")]
    public void StartupPhase_UsesRandomSampling_BitExact()
    {
        PythonSession.EnsureOrInconclusive();

        // Below n_startup_trials TPE delegates to RandomSampler(seed): those trials are bit-exact.
        var csStudy = Optuna.CreateStudy(
            storage: new InMemoryStorage(), sampler: new TPESampler(seed: 99), studyName: "tpe");
        csStudy.Optimize(t => Math.Pow(t.SuggestFloat("x", -10, 10), 2.0), nTrials: 10);

        var csState = StudySnapshot.Capture(csStudy);
        PyExec(
            "study = optuna.create_study(study_name='tpe', sampler=optuna.samplers.TPESampler(seed=99))\n" +
            "study.optimize(lambda t: t.suggest_float('x', -10, 10) ** 2, n_trials=10)\n");
        var pyState = PyStudyState("study");

        // The startup trials are pure RandomSampler -> bit-identical internal reprs.
        for (int i = 0; i < 10; i++)
            csState.Trials[i].ParamsInternal["x"].Should().Be(pyState.Trials[i].ParamsInternal["x"]);
    }
}
