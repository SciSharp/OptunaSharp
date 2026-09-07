using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Text;
using AwesomeAssertions;
using OptunaSharp.Distributions;
using OptunaSharp.Interop.BoTorch;

namespace OptunaSharp.Tests.Interop;

/// <summary>
///     Byte-exact parity of the native <see cref="BoTorchSampler" /> against the real
///     <c>optuna_integration.BoTorchSampler</c>. OptunaSharp runs the study (its
///     <see cref="BoTorchSampler" /> drives the genuine BoTorch via pythonnet); Python then
///     <b>replays</b> the identical trial history one trial at a time through
///     <c>optuna_integration.BoTorchSampler</c> and, at each BoTorch trial, its <c>sample_relative</c>
///     is asked for a suggestion given the same completed prefix.
///
///     <para><b>Why this is bit-exact (not tolerance):</b> both sides call the <em>same</em> BoTorch
///     with byte-identical <c>train_x</c>/<c>train_obj</c>/<c>bounds</c> (OptunaSharp's one-hot
///     <c>_SearchSpaceTransform</c> is a bit-exact port) under the <em>same</em> <c>manual_seed</c>, and
///     the startup trials come from a bit-exact <see cref="OptunaSharp.Samplers.RandomSampler" />. So the
///     suggested parameters match to 0 ULP — verified across the default and explicit candidates
///     functions, and across float / int search spaces. This is the whole point of the bridge: it is
///     BoTorch, so it agrees with BoTorch exactly.</para>
///
///     <para>Self-skips (Inconclusive) when <c>botorch</c> / <c>optuna_integration</c> are not installed
///     in the interop interpreter.</para>
/// </summary>
[TestClass]
public class BoTorchSamplerParityTests : OptunaInteropTestBase
{
    private const string ReplayHelper =
        "import warnings, json\n" +
        "warnings.filterwarnings('ignore')\n" +
        "import optuna_integration as integration\n" +
        "from optuna_integration import BoTorchSampler\n" +
        "from optuna.distributions import FloatDistribution, IntDistribution, CategoricalDistribution\n" +
        "from optuna.trial import TrialState\n" +
        "def _replay_botorch(traj_json, space, seed, n_startup, cf):\n" +
        "    cs = json.loads(traj_json)\n" +
        "    sampler = BoTorchSampler(n_startup_trials=n_startup, seed=seed, candidates_func=cf)\n" +
        "    study = optuna.create_study(sampler=sampler)\n" +
        "    max_diff = 0.0\n" +
        "    int_mismatches = 0\n" +
        "    n_bo = 0\n" +
        "    for k, ct in enumerate(cs):\n" +
        "        tid = study._storage.create_new_trial(study._study_id)\n" +
        "        frozen = study._storage.get_trial(tid)\n" +
        "        ss = sampler.infer_relative_search_space(study, frozen)\n" +
        "        frozen = study._storage.get_trial(tid)\n" +
        "        rel = sampler.sample_relative(study, frozen, ss)\n" +
        "        if k >= n_startup and rel:\n" +
        "            n_bo += 1\n" +
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
        "        study._storage.set_trial_state_values(tid, TrialState.COMPLETE, [ct['value']])\n" +
        "    return max_diff, int_mismatches, n_bo\n";

    // ---- single-objective, default candidates_func (log-EI) -------------------------------------

    [TestMethod]
    [TestCategory("RequiresPython")]
    public void SingleFloat_DefaultLogEi_ByteExact()
        => AssertReplayParity(
            objective: t => { double x = t.SuggestFloat("x", -10, 10); return (x - 2.0) * (x - 2.0); },
            spaceExpr: "{'x': FloatDistribution(-10, 10)}",
            candidatesFunc: BoTorchCandidatesFunc.Auto, cfExpr: "None",
            nStartup: 5, nTrials: 12, seed: 0);

    [TestMethod]
    [TestCategory("RequiresPython")]
    public void ThreeFloats_DefaultLogEi_ByteExact()
        => AssertReplayParity(
            objective: t =>
            {
                double a = t.SuggestFloat("a", -5, 5);
                double b = t.SuggestFloat("b", 0, 10);
                double c = t.SuggestFloat("c", -1, 1);
                return (a - 1.0) * (a - 1.0) + (b - 4.0) * (b - 4.0) + c * c;
            },
            spaceExpr: "{'a': FloatDistribution(-5, 5), 'b': FloatDistribution(0, 10), 'c': FloatDistribution(-1, 1)}",
            candidatesFunc: BoTorchCandidatesFunc.Auto, cfExpr: "None",
            nStartup: 6, nTrials: 12, seed: 1);

    // ---- explicit candidates functions ----------------------------------------------------------

    [TestMethod]
    [TestCategory("RequiresPython")]
    public void ExplicitQei_ByteExact()
        => AssertReplayParity(
            objective: t => { double x = t.SuggestFloat("x", -10, 10); return (x - 2.0) * (x - 2.0); },
            spaceExpr: "{'x': FloatDistribution(-10, 10)}",
            candidatesFunc: BoTorchCandidatesFunc.Qei, cfExpr: "integration.botorch.qei_candidates_func",
            nStartup: 5, nTrials: 11, seed: 0);

    [TestMethod]
    [TestCategory("RequiresPython")]
    public void ExplicitQLogEi_ByteExact()
        => AssertReplayParity(
            objective: t => { double x = t.SuggestFloat("x", -10, 10); return (x - 2.0) * (x - 2.0); },
            spaceExpr: "{'x': FloatDistribution(-10, 10)}",
            candidatesFunc: BoTorchCandidatesFunc.QLogEi, cfExpr: "integration.botorch.qlogei_candidates_func",
            nStartup: 5, nTrials: 11, seed: 3);

    [TestMethod]
    [TestCategory("RequiresPython")]
    public void ExplicitQNei_ByteExact()
        => AssertReplayParity(
            objective: t => { double x = t.SuggestFloat("x", -10, 10); return (x - 2.0) * (x - 2.0); },
            spaceExpr: "{'x': FloatDistribution(-10, 10)}",
            candidatesFunc: BoTorchCandidatesFunc.QNei, cfExpr: "integration.botorch.qnei_candidates_func",
            nStartup: 5, nTrials: 11, seed: 7);

    // ---- integer search space (untransform snaps to the identical grid point) -------------------

    [TestMethod]
    [TestCategory("RequiresPython")]
    public void IntParam_ExactGrid()
        => AssertReplayParity(
            objective: t => { long n = t.SuggestInt("n", 0, 50); return (n - 20.0) * (n - 20.0); },
            spaceExpr: "{'n': IntDistribution(0, 50)}",
            candidatesFunc: BoTorchCandidatesFunc.Auto, cfExpr: "None",
            nStartup: 6, nTrials: 12, seed: 2);

    // ---- categorical search space (one-hot transform → botorch) ---------------------------------

    [TestMethod]
    [TestCategory("RequiresPython")]
    public void Categorical_ByteExact()
        => AssertReplayParity(
            objective: t =>
            {
                double x = t.SuggestFloat("x", -5, 5);
                string c = (string)t.SuggestCategorical("c", new CategoricalChoice[] { "a", "b", "c" }).Value!;
                double bonus = c == "a" ? 0.0 : c == "b" ? 1.0 : 2.0;
                return (x - 2.0) * (x - 2.0) + bonus;
            },
            spaceExpr: "{'x': FloatDistribution(-5, 5), 'c': CategoricalDistribution(['a', 'b', 'c'])}",
            candidatesFunc: BoTorchCandidatesFunc.Auto, cfExpr: "None",
            nStartup: 6, nTrials: 13, seed: 4);

    // ---- C# determinism (no Python) -------------------------------------------------------------

    [TestMethod]
    [TestCategory("RequiresPython")]
    public void Reproducible_SameSeed_Identical()
    {
        RequireBoTorchOrInconclusive();
        static double Objective(Trial t) { double x = t.SuggestFloat("x", -10, 10); return (x - 2.0) * (x - 2.0); }

        var first = Optuna.CreateStudy(sampler: new BoTorchSampler(nStartupTrials: 5, seed: 42));
        first.Optimize(Objective, nTrials: 12);
        var second = Optuna.CreateStudy(sampler: new BoTorchSampler(nStartupTrials: 5, seed: 42));
        second.Optimize(Objective, nTrials: 12);

        TrajectoryJson(second).Should().Be(TrajectoryJson(first),
            "the same seed must reproduce the identical trajectory bit-for-bit in C#");
    }

    // ---- multi-objective (default qEHVI / >3-obj EHVI + explicit qNEHVI / qParEGO) ---------------

    [TestMethod]
    [TestCategory("RequiresPython")]
    public void TwoObjectives_DefaultQehvi_ByteExact()
        => AssertMoReplayParity(
            objective: t =>
            {
                double a = t.SuggestFloat("a", -5, 5);
                double b = t.SuggestFloat("b", -5, 5);
                return (IReadOnlyList<double>)new[] { a * a + b * b, (a - 2) * (a - 2) + b * b };
            },
            directions: new[] { StudyDirection.Minimize, StudyDirection.Minimize },
            spaceExpr: "{'a': FloatDistribution(-5, 5), 'b': FloatDistribution(-5, 5)}",
            candidatesFunc: BoTorchCandidatesFunc.Auto, cfExpr: "None",
            nStartup: 5, nTrials: 8, seed: 0, constrained: false);

    [TestMethod]
    [TestCategory("RequiresPython")]
    public void ExplicitQnehvi_ByteExact()
        => AssertMoReplayParity(
            objective: t =>
            {
                double a = t.SuggestFloat("a", -5, 5);
                double b = t.SuggestFloat("b", -5, 5);
                return (IReadOnlyList<double>)new[] { a * a + b * b, (a - 2) * (a - 2) + b * b };
            },
            directions: new[] { StudyDirection.Minimize, StudyDirection.Minimize },
            spaceExpr: "{'a': FloatDistribution(-5, 5), 'b': FloatDistribution(-5, 5)}",
            candidatesFunc: BoTorchCandidatesFunc.Qnehvi, cfExpr: "integration.botorch.qnehvi_candidates_func",
            nStartup: 5, nTrials: 8, seed: 3, constrained: false);

    [TestMethod]
    [TestCategory("RequiresPython")]
    public void FourObjectives_DefaultEhvi_ByteExact()
        => AssertMoReplayParity(
            objective: t =>
            {
                double a = t.SuggestFloat("a", -5, 5);
                double b = t.SuggestFloat("b", -5, 5);
                return (IReadOnlyList<double>)new[]
                    { a * a, b * b, (a - 2) * (a - 2) + b * b, a * a + (b - 2) * (b - 2) };
            },
            directions: new[]
            {
                StudyDirection.Minimize, StudyDirection.Minimize,
                StudyDirection.Minimize, StudyDirection.Minimize,
            },
            spaceExpr: "{'a': FloatDistribution(-5, 5), 'b': FloatDistribution(-5, 5)}",
            candidatesFunc: BoTorchCandidatesFunc.Auto, cfExpr: "None",
            nStartup: 6, nTrials: 8, seed: 1, constrained: false);

    [TestMethod]
    [TestCategory("RequiresPython")]
    public void ExplicitQParEgo_ByteExact()
        => AssertMoReplayParity(
            objective: t =>
            {
                double a = t.SuggestFloat("a", -5, 5);
                double b = t.SuggestFloat("b", -5, 5);
                return (IReadOnlyList<double>)new[] { a * a + b * b, (a - 2) * (a - 2) + b * b };
            },
            directions: new[] { StudyDirection.Minimize, StudyDirection.Minimize },
            spaceExpr: "{'a': FloatDistribution(-5, 5), 'b': FloatDistribution(-5, 5)}",
            candidatesFunc: BoTorchCandidatesFunc.QParEgo, cfExpr: "integration.botorch.qparego_candidates_func",
            nStartup: 5, nTrials: 8, seed: 5, constrained: false);

    // ---- constrained (Trial.SetConstraint → the constrained candidates_func branch) --------------

    [TestMethod]
    [TestCategory("RequiresPython")]
    public void ConstrainedSingleObjective_DefaultLogEi_ByteExact()
        => AssertMoReplayParity(
            objective: t =>
            {
                double x = t.SuggestFloat("x", -10, 10);
                t.SetConstraint("0", x - 5.0);   // feasible when x <= 5
                return (IReadOnlyList<double>)new[] { (x - 2.0) * (x - 2.0) };
            },
            directions: new[] { StudyDirection.Minimize },
            spaceExpr: "{'x': FloatDistribution(-10, 10)}",
            candidatesFunc: BoTorchCandidatesFunc.Auto, cfExpr: "None",
            nStartup: 5, nTrials: 10, seed: 2, constrained: true);

    [TestMethod]
    [TestCategory("RequiresPython")]
    public void ConstrainedMultiObjective_DefaultQehvi_ByteExact()
        => AssertMoReplayParity(
            objective: t =>
            {
                double a = t.SuggestFloat("a", -5, 5);
                double b = t.SuggestFloat("b", -5, 5);
                t.SetConstraint("0", a - 1.0);   // feasible when a <= 1
                return (IReadOnlyList<double>)new[] { a * a + b * b, (a - 2) * (a - 2) + b * b };
            },
            directions: new[] { StudyDirection.Minimize, StudyDirection.Minimize },
            spaceExpr: "{'a': FloatDistribution(-5, 5), 'b': FloatDistribution(-5, 5)}",
            candidatesFunc: BoTorchCandidatesFunc.Auto, cfExpr: "None",
            nStartup: 6, nTrials: 10, seed: 4, constrained: true);

    // ---- device (CPU is the default; CUDA is a smoke test, self-Inconclusive without a device) ----

    [TestMethod]
    [TestCategory("RequiresPython")]
    public void Device_Cuda_Smoke_RunsAndProducesValidCandidates()
    {
        RequireBoTorchOrInconclusive();
        if (!CudaAvailable())
            Assert.Inconclusive("no CUDA device available in the interop interpreter.");

        static double Objective(Trial t) { double x = t.SuggestFloat("x", -10, 10); return (x - 2.0) * (x - 2.0); }

        var study = Optuna.CreateStudy(sampler: new BoTorchSampler(nStartupTrials: 5, seed: 0, device: "cuda"));
        study.Optimize(Objective, nTrials: 8);

        study.Trials.Count.Should().Be(8, "a CUDA study runs the same trial schedule as CPU");
        foreach (var t in study.Trials)
        {
            double x = (double)t.ParamsDict["x"]!;
            double.IsFinite(x).Should().BeTrue("every candidate crosses back from the device as a finite value");
            x.Should().BeInRange(-10.0, 10.0, "the unnormalized candidate must lie inside the bounds");
        }
    }

    // ---- harness --------------------------------------------------------------------------------

    private void AssertReplayParity(
        Func<Trial, double> objective, string spaceExpr,
        BoTorchCandidatesFunc candidatesFunc, string cfExpr,
        int nStartup, int nTrials, int seed)
    {
        RequireBoTorchOrInconclusive();

        var study = Optuna.CreateStudy(
            sampler: new BoTorchSampler(candidatesFunc: candidatesFunc, nStartupTrials: nStartup, seed: seed));
        study.Optimize(objective, nTrials: nTrials);

        PyExec(ReplayHelper);
        PySet("_traj", TrajectoryJson(study));
        PyExec($"_space = {spaceExpr}\n" +
               $"_md, _im, _nbo = _replay_botorch(_traj, _space, {seed}, {nStartup}, {cfExpr})\n");

        PyLong("_nbo").Should().BeGreaterThan(0, "there must be BoTorch-sampled trials to compare");
        PyLong("_im").Should().Be(0, "int/categorical BoTorch suggestions must land on the identical grid point as real botorch");
        PyFloat("_md").Should().Be(0.0, "BoTorch float suggestions must match real optuna_integration.BoTorchSampler bit-for-bit");
    }

    // The multi-objective / constrained replay: create the study with directions, feed each trial its
    // recorded params + all values (+ constraints as `constraints:i` system attrs), and compare each
    // BoTorch suggestion to what OptunaSharp produced — byte-for-byte, exactly like the SO helper.
    private const string GeneralReplayHelper =
        "import warnings, json\n" +
        "warnings.filterwarnings('ignore')\n" +
        "import optuna_integration as integration\n" +
        "from optuna_integration import BoTorchSampler\n" +
        "from optuna.distributions import FloatDistribution, IntDistribution, CategoricalDistribution\n" +
        "from optuna.trial import TrialState\n" +
        "def _replay_botorch_general(traj_json, space, seed, n_startup, cf, directions):\n" +
        "    cs = json.loads(traj_json)\n" +
        "    sampler = BoTorchSampler(n_startup_trials=n_startup, seed=seed, candidates_func=cf)\n" +
        "    study = optuna.create_study(directions=directions, sampler=sampler)\n" +
        "    max_diff = 0.0\n" +
        "    int_mismatches = 0\n" +
        "    n_bo = 0\n" +
        "    for k, ct in enumerate(cs):\n" +
        "        tid = study._storage.create_new_trial(study._study_id)\n" +
        "        frozen = study._storage.get_trial(tid)\n" +
        "        ss = sampler.infer_relative_search_space(study, frozen)\n" +
        "        frozen = study._storage.get_trial(tid)\n" +
        "        rel = sampler.sample_relative(study, frozen, ss)\n" +
        "        if k >= n_startup and rel:\n" +
        "            n_bo += 1\n" +
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
        "        for i, cval in enumerate(ct.get('constraints', [])):\n" +
        "            study._storage.set_trial_system_attr(tid, f'constraints:{i}', cval)\n" +
        "        study._storage.set_trial_state_values(tid, TrialState.COMPLETE, ct['values'])\n" +
        "    return max_diff, int_mismatches, n_bo\n";

    private void AssertMoReplayParity(
        Func<Trial, IReadOnlyList<double>> objective, StudyDirection[] directions, string spaceExpr,
        BoTorchCandidatesFunc candidatesFunc, string cfExpr, int nStartup, int nTrials, int seed, bool constrained)
    {
        RequireBoTorchOrInconclusive();

        var study = Optuna.CreateStudy(
            directions: directions,
            sampler: new BoTorchSampler(candidatesFunc: candidatesFunc, nStartupTrials: nStartup, seed: seed));
        study.Optimize(objective, nTrials: nTrials);

        PyExec(GeneralReplayHelper);
        PySet("_traj", TrajectoryJsonGeneral(study, constrained));
        string dirs = "[" + string.Join(", ",
            directions.Select(d => d == StudyDirection.Minimize ? "'minimize'" : "'maximize'")) + "]";
        PyExec($"_space = {spaceExpr}\n" +
               $"_md, _im, _nbo = _replay_botorch_general(_traj, _space, {seed}, {nStartup}, {cfExpr}, {dirs})\n");

        PyLong("_nbo").Should().BeGreaterThan(0, "there must be BoTorch-sampled trials to compare");
        PyLong("_im").Should().Be(0, "int/categorical BoTorch suggestions must land on the identical grid point as real botorch");
        PyFloat("_md").Should().Be(0.0, "BoTorch suggestions must match real optuna_integration.BoTorchSampler bit-for-bit");
    }

    private bool CudaAvailable()
    {
        try
        {
            PyExec("import torch\n_cuda = torch.cuda.is_available()\n");
            return PyBool("_cuda");
        }
        catch
        {
            return false;
        }
    }

    private void RequireBoTorchOrInconclusive()
    {
        bool available;
        try
        {
            // A direct import is robust (find_spec can misbehave once torch is loaded into the engine).
            PyExec("try:\n" +
                   "    import botorch, optuna_integration\n" +
                   "    _have_botorch = True\n" +
                   "except Exception:\n" +
                   "    _have_botorch = False\n");
            available = PyBool("_have_botorch");
        }
        catch
        {
            available = false;
        }
        if (!available)
            Assert.Inconclusive("botorch / optuna_integration are not installed in the interop interpreter.");
    }

    private static string TrajectoryJson(Study study)
    {
        var sb = new StringBuilder("[");
        bool firstTrial = true;
        foreach (var t in study.Trials)
        {
            if (!firstTrial) sb.Append(',');
            firstTrial = false;
            sb.Append("{\"params\":{");
            bool firstParam = true;
            foreach (var (name, value) in t.ParamsDict)
            {
                if (!firstParam) sb.Append(',');
                firstParam = false;
                sb.Append('"').Append(name).Append("\":");
                sb.Append(value switch
                {
                    double d => FmtFloat(d),
                    string s => "\"" + s + "\"",                // categorical string choice
                    bool b => b ? "true" : "false",
                    _ => Convert.ToInt64(value).ToString(CultureInfo.InvariantCulture),  // int / stepped
                });
            }
            sb.Append("},\"value\":").Append(FmtFloat(t.Value!.Value)).Append('}');
        }
        return sb.Append(']').ToString();
    }

    // Multi-objective / constrained trajectory: a `values` list per trial (+ a `constraints` list when
    // the study is constrained) instead of the single-objective `value` scalar.
    private static string TrajectoryJsonGeneral(Study study, bool constrained)
    {
        var sb = new StringBuilder("[");
        bool firstTrial = true;
        foreach (var t in study.Trials)
        {
            if (!firstTrial) sb.Append(',');
            firstTrial = false;
            sb.Append("{\"params\":{");
            bool firstParam = true;
            foreach (var (name, value) in t.ParamsDict)
            {
                if (!firstParam) sb.Append(',');
                firstParam = false;
                sb.Append('"').Append(name).Append("\":");
                sb.Append(value switch
                {
                    double d => FmtFloat(d),
                    string s => "\"" + s + "\"",
                    bool b => b ? "true" : "false",
                    _ => Convert.ToInt64(value).ToString(CultureInfo.InvariantCulture),
                });
            }
            sb.Append('}');
            if (constrained)
            {
                sb.Append(",\"constraints\":[");
                sb.Append(string.Join(",", t.Constraints.Values.Select(FmtFloat)));
                sb.Append(']');
            }
            sb.Append(",\"values\":[");
            sb.Append(string.Join(",", t.Values!.Select(FmtFloat)));
            sb.Append("]}");
        }
        return sb.Append(']').ToString();
    }

    private static string FmtFloat(double d)
    {
        string s = d.ToString("R", CultureInfo.InvariantCulture);
        // Ensure JSON sees a float (a bare integer-valued double like -10 must round-trip as a number,
        // which it does; NaN/Inf are not expected in these deterministic objectives).
        return s;
    }
}
