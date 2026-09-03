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
    [DataRow(0)]
    [DataRow(7)]
    [DataRow(42)]
    public void GroupDecomposedBranchingSpace_TrajectoryMatches(int seed)
    {
        PythonSession.EnsureOrInconclusive();

        // group=True over a branching space: `x` always occurs; `y`/`z` occur only under their
        // branch, so the search space decomposes into the subspaces {x}, {y}, {z}, each sampled
        // separately by multivariate TPE. The branch decisions must line up exactly and the
        // continuous values to the erf tolerance — proving the group RNG draw order matches Python.
        var csStudy = Optuna.CreateStudy(
            storage: new InMemoryStorage(),
            sampler: new TPESampler(seed: seed, multivariate: true, group: true, nStartupTrials: 5),
            studyName: "tpe");
        csStudy.Optimize(t =>
        {
            var x = (string)t.SuggestCategorical("x", new CategoricalChoice[] { "A", "B" }).Value!;
            return x == "A" ? t.SuggestFloat("y", -10, 10) : t.SuggestFloat("z", -10, 10);
        }, nTrials: 24);

        AssertTrajectoryParity(csStudy,
            "study = optuna.create_study(study_name='tpe', sampler=optuna.samplers.TPESampler(" +
            $"seed={seed}, multivariate=True, group=True, n_startup_trials=5))\n" +
            "def _obj(t):\n" +
            "    x = t.suggest_categorical('x', ['A', 'B'])\n" +
            "    if x == 'A':\n" +
            "        return t.suggest_float('y', -10, 10)\n" +
            "    return t.suggest_float('z', -10, 10)\n" +
            "study.optimize(_obj, n_trials=24)\n");
    }

    [TestMethod]
    [TestCategory("RequiresPython")]
    [DataRow(2, 0)]
    [DataRow(2, 7)]
    [DataRow(2, 42)]
    [DataRow(3, 0)]
    [DataRow(3, 42)]
    public void MultiObjective_TrajectoryMatches(int nObjectives, int seed)
    {
        PythonSession.EnsureOrInconclusive();

        // Multi-objective TPE: below/above split uses non-domination rank + the HSSP tie-break
        // (Phase 2.10). The whole trajectory must stay in RNG lock-step with Python — branch/state
        // exact, continuous values to the erf tolerance. (MO defaults to the independent path.)
        var directions = Enumerable.Repeat(StudyDirection.Minimize, nObjectives).ToArray();
        var csStudy = Optuna.CreateStudy(
            storage: new InMemoryStorage(), directions: directions,
            sampler: new TPESampler(seed: seed, nStartupTrials: 8), studyName: "tpe");
        csStudy.Optimize(t =>
        {
            double x = t.SuggestFloat("x", -5, 5);
            double y = t.SuggestFloat("y", -5, 5);
            var outs = new System.Collections.Generic.List<double> { x * x + y, (x - 2) * (x - 2) + y * y };
            if (nObjectives == 3)
                outs.Add((x + 1) * (x + 1) + (y - 1) * (y - 1));
            return outs;
        }, nTrials: 40);

        string pyDirs = "[" + string.Join(", ", Enumerable.Repeat("'minimize'", nObjectives)) + "]";
        string pyReturn = nObjectives == 3
            ? "    return x * x + y, (x - 2) ** 2 + y * y, (x + 1) ** 2 + (y - 1) ** 2\n"
            : "    return x * x + y, (x - 2) ** 2 + y * y\n";
        AssertTrajectoryParity(csStudy,
            $"study = optuna.create_study(study_name='tpe', directions={pyDirs}, " +
            $"sampler=optuna.samplers.TPESampler(seed={seed}, n_startup_trials=8))\n" +
            "def _obj(t):\n" +
            "    x = t.suggest_float('x', -5, 5)\n" +
            "    y = t.suggest_float('y', -5, 5)\n" +
            pyReturn +
            "study.optimize(_obj, n_trials=40)\n");
    }

    [TestMethod]
    [TestCategory("RequiresPython")]
    [DataRow(0)]
    [DataRow(7)]
    [DataRow(42)]
    public void MultivariateMultiObjective_TrajectoryMatches(int seed)
    {
        PythonSession.EnsureOrInconclusive();

        // multivariate=True on a multi-objective study (MOTPE): the joint Parzen estimator over the
        // below set (uniform l(x) weights) + the non-domination/HSSP split, all in RNG lock-step.
        var csStudy = Optuna.CreateStudy(
            storage: new InMemoryStorage(),
            directions: new[] { StudyDirection.Minimize, StudyDirection.Minimize },
            sampler: new TPESampler(seed: seed, multivariate: true, nStartupTrials: 8), studyName: "tpe");
        csStudy.Optimize(t =>
        {
            double x = t.SuggestFloat("x", -5, 5);
            double y = t.SuggestFloat("y", -5, 5);
            return (System.Collections.Generic.IReadOnlyList<double>)new[] { x * x + y, (x - 2) * (x - 2) + y * y };
        }, nTrials: 35);

        AssertTrajectoryParity(csStudy,
            "study = optuna.create_study(study_name='tpe', directions=['minimize', 'minimize'], " +
            $"sampler=optuna.samplers.TPESampler(seed={seed}, multivariate=True, n_startup_trials=8))\n" +
            "def _obj(t):\n" +
            "    x = t.suggest_float('x', -5, 5)\n" +
            "    y = t.suggest_float('y', -5, 5)\n" +
            "    return x * x + y, (x - 2) ** 2 + y * y\n" +
            "study.optimize(_obj, n_trials=35)\n");
    }

    [TestMethod]
    [TestCategory("RequiresPython")]
    [DataRow(1, 0)]
    [DataRow(1, 42)]
    [DataRow(2, 0)]
    [DataRow(2, 42)]
    public void Constrained_TrajectoryMatches(int nObjectives, int seed)
    {
        PythonSession.EnsureOrInconclusive();

        // Constrained TPE (constraints_func): infeasible trials (x > 3) are split by
        // _split_infeasible_trials before the complete/pruned split. The trajectory staying in
        // lock-step proves the constraint-aware split matches Optuna. Feasible ~80% of the domain,
        // so infeasible trials genuinely occur.
        Func<FrozenTrial, System.Collections.Generic.IEnumerable<double>> cons =
            t => new[] { (double)t.Params["x"] - 3.0 };
        var directions = Enumerable.Repeat(StudyDirection.Minimize, nObjectives).ToArray();
        var csStudy = Optuna.CreateStudy(
            storage: new InMemoryStorage(), directions: directions,
            sampler: new TPESampler(seed: seed, nStartupTrials: 8, constraintsFunc: cons), studyName: "tpe");
        csStudy.Optimize(t =>
        {
            double x = t.SuggestFloat("x", -5, 5);
            double y = t.SuggestFloat("y", -5, 5);
            var outs = new System.Collections.Generic.List<double> { x * x + y };
            if (nObjectives == 2)
                outs.Add((x - 2) * (x - 2) + y * y);
            return outs;
        }, nTrials: 35);

        string pyDirs = "[" + string.Join(", ", Enumerable.Repeat("'minimize'", nObjectives)) + "]";
        string pyReturn = nObjectives == 2
            ? "    return x * x + y, (x - 2) ** 2 + y * y\n"
            : "    return x * x + y\n";
        AssertTrajectoryParity(csStudy,
            $"study = optuna.create_study(study_name='tpe', directions={pyDirs}, " +
            $"sampler=optuna.samplers.TPESampler(seed={seed}, n_startup_trials=8, " +
            "constraints_func=lambda t: (t.params['x'] - 3.0,)))\n" +
            "def _obj(t):\n" +
            "    x = t.suggest_float('x', -5, 5)\n" +
            "    y = t.suggest_float('y', -5, 5)\n" +
            pyReturn +
            "study.optimize(_obj, n_trials=35)\n");
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
