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
        // The startup trials are pure RandomSampler internal reprs (bit-exact) — host-pinned to x64;
        // the density-driven TPE trajectory tests use the erf tolerance and run on every architecture.
        PythonSession.EnsureBitExactHostOrInconclusive();

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

    /// <summary>
    ///     Plan 13 gate (docs/plans/13-rustuna-derived-performance.md, phase 0): the default single-objective TPE
    ///     (multivariate, constant liar) over 40 float parameters for 1,000 trials, the workload whose scoring kernel
    ///     item S6 rewrote. The whole trajectory stays in lock-step with live Optuna, within the erf tolerance of the
    ///     other TPE cases, so the fused kernel, the pooled buffers and the sort that replaced LINQ changed nothing a
    ///     thousand trials of compounding would show.
    /// </summary>
    [TestMethod]
    [TestCategory("RequiresPython")]
    [TestCategory("Slow")]
    public void LongRun_Multivariate40D_1000Trials_TrajectoryMatches()
    {
        PythonSession.EnsureOrInconclusive();

        var csStudy = Optuna.CreateStudy(storage: new InMemoryStorage(), sampler: new TPESampler(seed: 0), studyName: "tpe");
        csStudy.Optimize(t =>
        {
            double s = 0;
            for (int i = 0; i < 40; i++)
            {
                double v = t.SuggestFloat($"x{i}", -10, 10);
                s += (v - 1) * (v - 1);
            }
            return s;
        }, nTrials: 1000);

        AssertTrajectoryParity(csStudy,
            "study = optuna.create_study(study_name='tpe', sampler=optuna.samplers.TPESampler(seed=0))\n" +
            "def _obj(t):\n" +
            "    s = 0.0\n" +
            "    for i in range(40):\n" +
            "        v = t.suggest_float(f'x{i}', -10, 10)\n" +
            "        s += (v - 1) * (v - 1)\n" +
            "    return s\n" +
            "study.optimize(_obj, n_trials=1000)\n");
    }

    /// <summary>
    ///     Plan 13 gate (phase 0): the default two-objective TPE (independent, constant liar) over 40 float parameters
    ///     for 300 trials, the workload of items S3 (the constant-liar read that skips the current trial), S5 (one split per
    ///     trial instead of per parameter) and S6. Every trial stays in lock-step with live Optuna within the erf
    ///     tolerance.
    /// </summary>
    [TestMethod]
    [TestCategory("RequiresPython")]
    [TestCategory("Slow")]
    public void LongRun_MultiObjective40D_300Trials_TrajectoryMatches()
    {
        PythonSession.EnsureOrInconclusive();

        var csStudy = Optuna.CreateStudy(
            storage: new InMemoryStorage(), sampler: new TPESampler(seed: 0), studyName: "tpe",
            directions: new[] { StudyDirection.Minimize, StudyDirection.Minimize });
        csStudy.Optimize(t =>
        {
            double a = 0, b = 0;
            for (int i = 0; i < 40; i++)
            {
                double v = t.SuggestFloat($"x{i}", -10, 10);
                a += v * v;
                b += (v - 2) * (v - 2);
            }
            return new[] { a, b };
        }, nTrials: 300);

        AssertTrajectoryParity(csStudy,
            "study = optuna.create_study(study_name='tpe', directions=['minimize', 'minimize'], " +
            "sampler=optuna.samplers.TPESampler(seed=0))\n" +
            "def _obj(t):\n" +
            "    a = 0.0\n" +
            "    b = 0.0\n" +
            "    for i in range(40):\n" +
            "        v = t.suggest_float(f'x{i}', -10, 10)\n" +
            "        a += v * v\n" +
            "        b += (v - 2) * (v - 2)\n" +
            "    return a, b\n" +
            "study.optimize(_obj, n_trials=300)\n");
    }

    /// <summary>
    ///     Plan 14 gate: the observation cache keys each column by parameter name and distribution, and remembers that a
    ///     finished trial lacks a parameter. This study gives it every case: a float whose range changes with the trial
    ///     number (three distributions under one name), a log float suggested only on one categorical branch (absent from
    ///     the other trials), and pruned trials (finished, so cached, and part of TPE's history), in both TPE modes, for 80
    ///     trials. The trajectory must stay in lock-step with live Optuna within the erf tolerance.
    /// </summary>
    /// <param name="multivariate">Whether TPE models the parameters jointly (the dynamic range is then sampled independently).</param>
    /// <param name="seed">The sampler seed on both sides.</param>
    /// <exception cref="TrialPruned">Never escapes: the objective raises it to prune every seventh trial, and Optimize records that trial as pruned.</exception>
    [TestMethod]
    [TestCategory("RequiresPython")]
    [DataRow(false, 0)]
    [DataRow(false, 42)]
    [DataRow(true, 0)]
    [DataRow(true, 7)]
    public void DynamicRangesConditionalAndPrunedTrials_TrajectoryMatches(bool multivariate, int seed)
    {
        PythonSession.EnsureOrInconclusive();

        var csStudy = Optuna.CreateStudy(
            storage: new InMemoryStorage(), sampler: new TPESampler(seed: seed, multivariate: multivariate), studyName: "tpe");
        csStudy.Optimize(t =>
        {
            double span = 5.0 + t.Number % 3;
            double x = t.SuggestFloat("x", -span, span);
            string c = t.SuggestCategorical("c", new[] { "a", "b", "c" });
            double y = c == "a" ? t.SuggestFloat("y", 1e-3, 10.0, log: true) : 0.0;
            double value = x * x + y + (c == "b" ? 0.0 : 1.0);
            t.Report(value, 0);
            if (t.Number % 7 == 3)
                throw new TrialPruned();
            return value;
        }, nTrials: 80);

        AssertTrajectoryParity(csStudy,
            "study = optuna.create_study(study_name='tpe', " +
            $"sampler=optuna.samplers.TPESampler(seed={seed}, multivariate={(multivariate ? "True" : "False")}))\n" +
            "def _obj(t):\n" +
            "    span = 5.0 + t.number % 3\n" +
            "    x = t.suggest_float('x', -span, span)\n" +
            "    c = t.suggest_categorical('c', ['a', 'b', 'c'])\n" +
            "    y = t.suggest_float('y', 1e-3, 10.0, log=True) if c == 'a' else 0.0\n" +
            "    value = x * x + y + (0.0 if c == 'b' else 1.0)\n" +
            "    t.report(value, 0)\n" +
            "    if t.number % 7 == 3:\n" +
            "        raise optuna.TrialPruned()\n" +
            "    return value\n" +
            "study.optimize(_obj, n_trials=80)\n");
    }

    /// <summary>
    ///     Plan 14 gate: constant liar with trials still running. Trials are asked three at a time and told only after all
    ///     three have sampled, so each later sample sees the earlier ones RUNNING: their relative parameters come from
    ///     system attributes (multivariate) or their suggested values (independent), and the observation cache must leave
    ///     them uncached and read them fresh. Fifteen rounds, both TPE modes, in lock-step with live Optuna within the erf
    ///     tolerance.
    /// </summary>
    /// <param name="multivariate">Whether TPE models the parameters jointly.</param>
    /// <param name="seed">The sampler seed on both sides.</param>
    [TestMethod]
    [TestCategory("RequiresPython")]
    [DataRow(false, 3)]
    [DataRow(true, 3)]
    [DataRow(true, 11)]
    public void ConstantLiarWithRunningTrials_AskTell_TrajectoryMatches(bool multivariate, int seed)
    {
        PythonSession.EnsureOrInconclusive();

        var csStudy = Optuna.CreateStudy(
            storage: new InMemoryStorage(),
            sampler: new TPESampler(seed: seed, multivariate: multivariate, constantLiar: true, nStartupTrials: 5),
            studyName: "tpe");
        for (int round = 0; round < 15; round++)
        {
            var batch = new Trial[3];
            for (int i = 0; i < batch.Length; i++)
            {
                batch[i] = csStudy.Ask();
                batch[i].SuggestFloat("x", -5, 5);
                batch[i].SuggestFloat("y", 1e-2, 1e2, log: true);
                batch[i].SuggestCategorical("c", new[] { "u", "v" });
            }
            foreach (var t in batch)
            {
                var p = csStudy.Storage.GetTrial(t.TrialId).Params;
                double x = (double)p["x"]!, y = (double)p["y"]!;
                csStudy.Tell(t, x * x + Math.Log(y) * Math.Log(y) + ((string)p["c"]! == "u" ? 0.0 : 0.5));
            }
        }

        AssertTrajectoryParity(csStudy,
            "import math\n" +
            "study = optuna.create_study(study_name='tpe', " +
            $"sampler=optuna.samplers.TPESampler(seed={seed}, multivariate={(multivariate ? "True" : "False")}, " +
            "constant_liar=True, n_startup_trials=5))\n" +
            "for _ in range(15):\n" +
            "    batch = [study.ask() for _ in range(3)]\n" +
            "    for t in batch:\n" +
            "        t.suggest_float('x', -5, 5)\n" +
            "        t.suggest_float('y', 1e-2, 1e2, log=True)\n" +
            "        t.suggest_categorical('c', ['u', 'v'])\n" +
            "    for t in batch:\n" +
            "        p = t.params\n" +
            "        study.tell(t, p['x'] ** 2 + math.log(p['y']) ** 2 + (0.0 if p['c'] == 'u' else 0.5))\n");
    }
}
