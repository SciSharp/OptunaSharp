using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Text;
using AwesomeAssertions;
using OptunaSharp.Distributions;
using OptunaSharp.Interop.BoTorch;
using OptunaSharp.Storages;

namespace OptunaSharp.Tests.Interop;

/// <summary>
///     <b>High-level integration</b> parity of the native <see cref="BoTorchSampler" /> — the end-to-end
///     paths the function-level <see cref="BoTorchApiParityTests" /> and the sequential-replay
///     <see cref="BoTorchSamplerParityTests" /> do not exercise, each still asserted <b>bit-for-bit</b>
///     (0-ULP) against the real <c>optuna_integration.BoTorchSampler</c>:
///     <list type="bullet">
///       <item><b>Running trials</b> (<c>considerRunningTrials: true</c>): an <c>Ask</c>/<c>Tell</c>
///       schedule that leaves trials RUNNING so the acquisition conditions on <c>X_pending</c> — the
///       parallel-optimization path (qEI single-objective, qEHVI multi-objective).</item>
///       <item>The <b>deprecated <c>constraintsFunc</c></b> path (constraints computed + stored in
///       <c>AfterTrial</c>, not via <c>Trial.SetConstraint</c>) driving the constrained
///       <c>candidates_func</c> branch.</item>
///       <item>A <b>mixed int + float multi-objective</b> space (the one-hot / int-grid untransform under
///       qEHVI).</item>
///     </list>
///     Same replay methodology as the sibling suites: OptunaSharp runs the schedule (its sampler drives
///     genuine BoTorch via pythonnet), Python replays the identical trajectory one trial at a time —
///     running trials left RUNNING with their shared params, completed trials fed their params/values/
///     constraints — and each BoTorch suggestion is compared to what OptunaSharp produced. Both sides call
///     the SAME botorch under the SAME <c>manual_seed</c>, so the match is exact. Self-skips when botorch /
///     optuna_integration are absent.
/// </summary>
[TestClass]
public class BoTorchIntegrationParityTests : OptunaInteropTestBase
{
    // A unified replay that reproduces the OptunaSharp trajectory through optuna_integration.BoTorchSampler:
    // handles running trials (left RUNNING with params set), constraints (constraints:i system attrs), and
    // any number of objective directions. Bit-exact — this is the bridge, so it agrees with botorch exactly.
    private const string ReplayHelper =
        "import warnings, json\n" +
        "warnings.filterwarnings('ignore')\n" +
        "import optuna_integration as integration\n" +
        "from optuna_integration import BoTorchSampler\n" +
        "from optuna.distributions import FloatDistribution, IntDistribution, CategoricalDistribution\n" +
        "from optuna.trial import TrialState\n" +
        "def _replay(traj_json, space, seed, n_startup, directions, cf, consider_running, has_constraints):\n" +
        "    cs = json.loads(traj_json)\n" +
        "    sampler = BoTorchSampler(n_startup_trials=n_startup, seed=seed, candidates_func=cf,\n" +
        "        consider_running_trials=consider_running)\n" +
        "    study = optuna.create_study(directions=directions, sampler=sampler)\n" +
        "    max_diff = 0.0\n" +
        "    int_mismatches = 0\n" +
        "    n_bo = 0\n" +
        "    n_bo_with_running = 0\n" +
        "    n_running_left = 0\n" +
        "    for k, ct in enumerate(cs):\n" +
        "        tid = study._storage.create_new_trial(study._study_id)\n" +
        "        frozen = study._storage.get_trial(tid)\n" +
        "        ss = sampler.infer_relative_search_space(study, frozen)\n" +
        "        frozen = study._storage.get_trial(tid)\n" +
        "        n_run_present = sum(1 for t in study.get_trials(deepcopy=False, states=(TrialState.RUNNING,))\n" +
        "                            if t.number < k)\n" +
        "        rel = sampler.sample_relative(study, frozen, ss)\n" +
        "        if k >= n_startup and rel:\n" +
        "            n_bo += 1\n" +
        "            if n_run_present > 0:\n" +
        "                n_bo_with_running += 1\n" +
        "            for name, pv in rel.items():\n" +
        "                cv = ct['params'][name]\n" +
        "                d = space[name]\n" +
        "                if isinstance(d, FloatDistribution) and d.step is None and not d.log:\n" +
        "                    max_diff = max(max_diff, abs(pv - cv))\n" +
        "                elif pv != cv:\n" +
        "                    int_mismatches += 1\n" +
        "        for name, val in ct['params'].items():\n" +
        "            d = space[name]\n" +
        "            study._storage.set_trial_param(tid, name, d.to_internal_repr(val), d)\n" +
        "        if ct['running']:\n" +
        "            n_running_left += 1\n" +
        "            continue\n" +
        "        if has_constraints:\n" +
        "            for i, cval in enumerate(ct['constraints']):\n" +
        "                study._storage.set_trial_system_attr(tid, f'constraints:{i}', cval)\n" +
        "        study._storage.set_trial_state_values(tid, TrialState.COMPLETE, ct['values'])\n" +
        "    return max_diff, int_mismatches, n_bo, n_bo_with_running, n_running_left\n";

    // ---- running trials (X_pending) — the parallel-optimization path -----------------------------

    [TestMethod]
    [TestCategory("RequiresPython")]
    [DataRow(0)]
    [DataRow(7)]
    [TestCategory("Slow")]
    public void SingleObjectiveWithRunning_Qei_ByteExact(int seed)
        => AssertReplay(
            seed, new[] { StudyDirection.Minimize }, cfExpr: "None", considerRunning: true, constraintsFunc: null,
            nStartup: 5, nCompletedBo: 3, nRunning: 3,
            spaceExpr: "{'x': FloatDistribution(-10, 10), 'y': FloatDistribution(-10, 10)}",
            objective: t =>
            {
                double x = t.SuggestFloat("x", -10, 10);
                double y = t.SuggestFloat("y", -10, 10);
                return (new() { ["x"] = x, ["y"] = y }, new[] { (x - 2.0) * (x - 2.0) + (y + 1.0) * (y + 1.0) });
            });

    [TestMethod]
    [TestCategory("RequiresPython")]
    [TestCategory("Slow")]
    public void MultiObjectiveWithRunning_Qehvi_ByteExact()
        => AssertReplay(
            0, new[] { StudyDirection.Minimize, StudyDirection.Minimize }, cfExpr: "None",
            considerRunning: true, constraintsFunc: null,
            nStartup: 5, nCompletedBo: 3, nRunning: 3,
            spaceExpr: "{'a': FloatDistribution(-5, 5), 'b': FloatDistribution(-5, 5)}",
            objective: t =>
            {
                double a = t.SuggestFloat("a", -5, 5);
                double b = t.SuggestFloat("b", -5, 5);
                return (new() { ["a"] = a, ["b"] = b }, new[] { a * a + b * b, (a - 2) * (a - 2) + b * b });
            });

    // ---- the deprecated constraintsFunc path (constraints computed + stored in AfterTrial) -------

    [TestMethod]
    [TestCategory("RequiresPython")]
    [TestCategory("Slow")]
    public void ConstraintsFunc_DeprecatedPath_SingleObjective_ByteExact()
    {
        // The sampler's constraintsFunc computes the constraint from the completed trial in AfterTrial
        // (the objective does NOT call SetConstraint) — Auto then resolves to the constrained logEI branch.
        AssertReplay(
            2, new[] { StudyDirection.Minimize }, cfExpr: "None", considerRunning: false,
            constraintsFunc: ft => new[] { (double)ft.ParamsDict["x"]! - 5.0 },   // feasible when x <= 5
            nStartup: 5, nCompletedBo: 4, nRunning: 0,
            spaceExpr: "{'x': FloatDistribution(-10, 10)}",
            objective: t =>
            {
                double x = t.SuggestFloat("x", -10, 10);
                return (new() { ["x"] = x }, new[] { (x - 2.0) * (x - 2.0) });
            },
            afterAssert: study =>
            {
                // Prove AfterTrial actually ran the constraintsFunc and stored a constraint per completed trial.
                foreach (var t in study.Trials)
                    t.Constraints.Count.Should().Be(1, "constraintsFunc must store one constraint per completed trial via AfterTrial");
            });
    }

    // ---- mixed int + float multi-objective (int-grid untransform under qEHVI) --------------------

    [TestMethod]
    [TestCategory("RequiresPython")]
    [TestCategory("Slow")]
    public void MixedIntFloatMultiObjective_Qehvi_ByteExact()
        => AssertReplay(
            4, new[] { StudyDirection.Minimize, StudyDirection.Minimize }, cfExpr: "None",
            considerRunning: false, constraintsFunc: null,
            nStartup: 6, nCompletedBo: 3, nRunning: 0,
            spaceExpr: "{'n': IntDistribution(0, 20), 'x': FloatDistribution(-5, 5)}",
            objective: t =>
            {
                long n = t.SuggestInt("n", 0, 20);
                double x = t.SuggestFloat("x", -5, 5);
                return (new() { ["n"] = n, ["x"] = x },
                    new[] { (n - 8.0) * (n - 8.0) + x * x, (n - 12.0) * (n - 12.0) + (x - 1.0) * (x - 1.0) });
            });

    /// <summary>
    ///     Parity QA run 2 (E63, <c>docs/qa/PARITY_QA_2026-11-10.md</c>): the configuration whose full studies diverged —
    ///     two minimized objectives over <c>n</c> in [0, 4] and <c>x</c> in [0, 1], four startup trials, seed 0. A
    ///     full Python study and the bridge's full study agree on the four startup trials and then differ in the
    ///     eighth significant digit of <c>x</c> from the first BoTorch trial on (both sides deterministic run to run,
    ///     independent of torch's thread count and of whether constraints are present). This replay feeds the bridge's
    ///     own history into real <c>optuna_integration.BoTorchSampler</c> and compares each BoTorch suggestion: QA-82,
    ///     measured 3.4e-8 maximum difference, where every other configuration in this class is 0. The objective
    ///     suggests <c>x</c> before <c>n</c> on purpose: with <c>n</c> suggested first (alphabetical order, as every
    ///     other case here does) the replay is identical, so the divergence depends on the suggestion order, not on
    ///     the ranges (<c>n</c> in [0, 20] and <c>x</c> in [-5, 5] still differ).
    /// </summary>
    /// <exception cref="AssertInconclusiveException">Under CI, where this known-failing repro must not run.</exception>
    [TestMethod]
    [TestCategory("RequiresPython")]
    [TestCategory("Slow")]
    [TestCategory("OpenBugs")]
    public void Run2SmallIntFloatMultiObjective_Qehvi_ByteExact()
    {
        // The interop job's Slow leg filters only `TestCategory=Slow`, not OpenBugs, so this known-failing repro must
        // not run there; locally (no CI variable) it runs and fails until the divergence is fixed.
        if (Environment.GetEnvironmentVariable("CI") == "true")
            Assert.Inconclusive("OpenBugs repro (QA-82); the interop CI job does not filter OpenBugs.");
        AssertReplay(
            0, new[] { StudyDirection.Minimize, StudyDirection.Minimize }, cfExpr: "None",
            considerRunning: false, constraintsFunc: null,
            nStartup: 4, nCompletedBo: 4, nRunning: 0,
            spaceExpr: "{'n': IntDistribution(0, 4), 'x': FloatDistribution(0.0, 1.0)}",
            objective: t =>
            {
                double x = t.SuggestFloat("x", 0.0, 1.0);
                long n = t.SuggestInt("n", 0, 4);
                return (new() { ["x"] = x, ["n"] = n },
                    new[] { x * x + 0.05 * n, Math.Pow(1.0 - x, 2) + 0.05 * (4 - n) });
            });
    }

    // ---- harness --------------------------------------------------------------------------------

    private void AssertReplay(
        int seed, StudyDirection[] directions, string cfExpr, bool considerRunning,
        Func<FrozenTrial, IEnumerable<double>>? constraintsFunc,
        int nStartup, int nCompletedBo, int nRunning, string spaceExpr,
        Func<Trial, (Dictionary<string, object?> Params, double[] Values)> objective,
        Action<Study>? afterAssert = null)
    {
        RequireBoTorchOrInconclusive();

        bool hasConstraints = constraintsFunc != null;
        var study = Optuna.CreateStudy(
            storage: new InMemoryStorage(), studyName: "bibo",
            sampler: new BoTorchSampler(
                constraintsFunc: constraintsFunc, nStartupTrials: nStartup, seed: seed,
                considerRunningTrials: considerRunning),
            directions: directions);

        var traj = new List<string>();

        // Completed trials: nStartup independent + nCompletedBo BoTorch-sampled.
        for (int i = 0; i < nStartup + nCompletedBo; i++)
        {
            Trial t = study.Ask();
            (Dictionary<string, object?> p, double[] v) = objective(t);
            FrozenTrial frozen = study.Tell(t, v);
            double[]? cons = hasConstraints ? frozen.Constraints.Values.ToArray() : null;
            traj.Add(TrajRow(p, v, cons, running: false));
        }

        // Running trials (asked + suggested, never told) — the acquisition sees them as X_pending.
        for (int i = 0; i < nRunning; i++)
        {
            Trial t = study.Ask();
            (Dictionary<string, object?> p, _) = objective(t);
            traj.Add(TrajRow(p, null, null, running: true));
        }

        PyExec(ReplayHelper);
        PySet("_traj", "[" + string.Join(",", traj) + "]");
        string dirs = "[" + string.Join(", ",
            directions.Select(d => d == StudyDirection.Minimize ? "'minimize'" : "'maximize'")) + "]";
        PyExec($"_space = {spaceExpr}\n" +
               $"_md, _im, _nbo, _nbor, _nrl = _replay(_traj, _space, {seed}, {nStartup}, {dirs}, {cfExpr}, " +
               $"{(considerRunning ? "True" : "False")}, {(hasConstraints ? "True" : "False")})\n");

        PyLong("_nbo").Should().BeGreaterThan(0, "there must be BoTorch-sampled trials to compare");
        PyLong("_nrl").Should().Be(nRunning, "the running trials must stay RUNNING in the replay");
        if (nRunning > 1)
            PyLong("_nbor").Should().BeGreaterThan(0, "the X_pending (running-trial) path must be exercised");
        PyLong("_im").Should().Be(0, "int/categorical BoTorch suggestions must land on the identical grid point as real botorch");
        PyFloat("_md").Should().Be(0.0, "BoTorch suggestions must match real optuna_integration.BoTorchSampler bit-for-bit");

        afterAssert?.Invoke(study);
    }

    private static string TrajRow(
        Dictionary<string, object?> paramsDict, double[]? values, double[]? constraints, bool running)
    {
        var sb = new StringBuilder("{\"params\":{");
        bool first = true;
        foreach (var (name, value) in paramsDict)
        {
            if (!first) sb.Append(',');
            first = false;
            sb.Append('"').Append(name).Append("\":");
            sb.Append(value switch
            {
                double d => FmtFloat(d),
                string s => "\"" + s + "\"",
                bool b => b ? "true" : "false",
                _ => Convert.ToInt64(value).ToString(CultureInfo.InvariantCulture),
            });
        }
        sb.Append("},\"running\":").Append(running ? "true" : "false");
        if (!running)
        {
            sb.Append(",\"values\":[").Append(string.Join(",", values!.Select(FmtFloat))).Append(']');
            if (constraints != null)
                sb.Append(",\"constraints\":[").Append(string.Join(",", constraints.Select(FmtFloat))).Append(']');
        }
        return sb.Append('}').ToString();
    }

    private static string FmtFloat(double d) => d.ToString("R", CultureInfo.InvariantCulture);

    private void RequireBoTorchOrInconclusive()
    {
        bool available;
        try
        {
            PyExec("try:\n    import botorch, optuna_integration\n    _have_botorch = True\n" +
                   "except Exception:\n    _have_botorch = False\n");
            available = PyBool("_have_botorch");
        }
        catch
        {
            available = false;
        }
        if (!available)
            Assert.Inconclusive("botorch / optuna_integration are not installed in the interop interpreter.");
    }
}
