using System;
using System.Globalization;
using System.Linq;
using System.Text;
using AwesomeAssertions;
using OptunaSharp.Distributions;
using OptunaSharp.Samplers;
using OptunaSharp.Storages;

namespace OptunaSharp.Tests.Interop;

/// <summary>
///     Per-suggestion parity of the <b>constrained</b> single-objective <see cref="GPSampler" /> path
///     (PORTING_PLAN Phase 3.2 part 3e-2 — the <see cref="LogCEI" /> acquisition function) against live
///     Optuna. OptunaSharp runs a constrained study (a genuine mix of feasible/infeasible trials via
///     <c>Trial.SetConstraint</c>); Optuna then <b>replays</b> the exact history one trial at a time —
///     re-fed OptunaSharp's params, value, <em>and</em> constraint each trial — and its
///     <c>sample_relative</c> is asked for a suggestion under the identical constrained prefix. Because
///     the history is re-synced every trial, each comparison is cascade-free and the sampler RNG stays
///     in lock-step, so every GP suggestion is compared under an identical LogCEI fit input.
///
///     <para>Both the objective GP and the constraint GP are fit by marginal-likelihood maximization,
///     whose kernel hyperparameters drift ~1e-3 from Optuna's (the loose <c>gtol=1e-2</c>, the documented
///     GP boundary), so continuous suggestions are compared within a normalized-coordinate envelope and
///     discrete/int params (which snap to a grid) near-exact. Same replay rationale as the unconstrained
///     <see cref="GpSamplerParityTests" />.</para>
/// </summary>
[TestClass]
public class GpConstrainedSamplerParityTests : OptunaInteropTestBase
{
    private const int NStartup = 10;         // GPSampler default n_startup_trials
    private const double NormDiffTol = 3e-2; // objective + constraint fit both drift → slightly looser than the SO 2e-2

    private const string ReplayHelper =
        "import json, math\n" +
        "from optuna.trial import TrialState\n" +
        "from optuna.distributions import FloatDistribution, IntDistribution, CategoricalDistribution\n" +
        "def _norm_diff(d, pv, cv):\n" +
        "    if d.log:\n" +
        "        return abs(math.log(pv) - math.log(cv)) / (math.log(d.high) - math.log(d.low))\n" +
        "    return abs(pv - cv) / (d.high - d.low)\n" +
        "def _replay_cgp(traj_json, space, seed, n_startup):\n" +
        "    cs = json.loads(traj_json)\n" +
        "    study = optuna.create_study(sampler=optuna.samplers.GPSampler(seed=seed))\n" +
        "    sampler = study.sampler\n" +
        "    max_norm_diff = 0.0\n" +
        "    int_mismatches = 0\n" +
        "    n_gp = 0\n" +
        "    n_feasible = 0\n" +
        "    n_infeasible = 0\n" +
        "    for k, ct in enumerate(cs):\n" +
        "        tid = study._storage.create_new_trial(study._study_id)\n" +
        "        frozen = study._storage.get_trial(tid)\n" +
        "        ss = sampler.infer_relative_search_space(study, frozen)\n" +
        "        frozen = study._storage.get_trial(tid)\n" +
        "        rel = sampler.sample_relative(study, frozen, ss)\n" +
        "        if k >= n_startup and rel:\n" +
        "            n_gp += 1\n" +
        "            for name, pv in rel.items():\n" +
        "                cv = ct['params'][name]\n" +
        "                if isinstance(space[name], FloatDistribution) and space[name].step is None:\n" +
        "                    max_norm_diff = max(max_norm_diff, _norm_diff(space[name], pv, cv))\n" +
        "                elif pv != cv:\n" +
        "                    int_mismatches += 1\n" +
        "        for name, val in ct['params'].items():\n" +
        "            dist = space[name]\n" +
        "            study._storage.set_trial_param(tid, name, dist.to_internal_repr(val), dist)\n" +
        "        for i, cval in enumerate(ct['constraints']):\n" +
        "            study._storage.set_trial_system_attr(tid, f'constraints:{i}', cval)\n" +
        "            if cval <= 0:\n" +
        "                n_feasible += 1\n" +
        "            else:\n" +
        "                n_infeasible += 1\n" +
        "        study._storage.set_trial_state_values(tid, TrialState.COMPLETE, [ct['value']])\n" +
        "    return max_norm_diff, int_mismatches, n_gp, n_feasible, n_infeasible\n";

    [TestMethod]
    [TestCategory("RequiresPython")]
    [DataRow(0, 24)]
    [DataRow(42, 24)]
    public void ConstrainedFloatSphere_PerSuggestion_MatchesLiveOptuna(int seed, int nTrials)
    {
        PythonSession.EnsureOrInconclusive();

        // Constrained shifted sphere: feasible when x0 <= 0.5 (constraint value x0 - 0.5 <= 0).
        var csStudy = Optuna.CreateStudy(
            storage: new InMemoryStorage(), studyName: "cgp", sampler: new GPSampler(seed: seed));
        csStudy.Optimize(t =>
        {
            double x0 = t.SuggestFloat("x0", -5, 5);
            double x1 = t.SuggestFloat("x1", -5, 5);
            t.SetConstraint("0", x0 - 0.5);
            return Math.Pow(x0 + 0.2, 2) + Math.Pow(x1 - 0.2, 2);
        }, nTrials: nTrials);

        AssertConstrainedReplayParity(csStudy, seed,
            "{'x0': FloatDistribution(-5.0, 5.0), 'x1': FloatDistribution(-5.0, 5.0)}");
    }

    [TestMethod]
    [TestCategory("RequiresPython")]
    [DataRow(7, 24)]
    public void ConstrainedMixedSpace_PerSuggestion_MatchesLiveOptuna(int seed, int nTrials)
    {
        PythonSession.EnsureOrInconclusive();

        // float + int (discrete relative dim) + a single constraint.
        var csStudy = Optuna.CreateStudy(
            storage: new InMemoryStorage(), studyName: "cgp", sampler: new GPSampler(seed: seed));
        csStudy.Optimize(t =>
        {
            double a = t.SuggestFloat("a", -4, 4);
            long b = t.SuggestInt("bb", -5, 5);
            t.SetConstraint("0", a + b - 1.0);   // feasible when a + b <= 1
            return Math.Pow(a - 1.0, 2) + Math.Pow(b + 1, 2);
        }, nTrials: nTrials);

        AssertConstrainedReplayParity(csStudy, seed,
            "{'a': FloatDistribution(-4.0, 4.0), 'bb': IntDistribution(-5, 5)}");
    }

    private void AssertConstrainedReplayParity(Study csStudy, int seed, string spaceExpr)
    {
        PyExec(ReplayHelper);
        PySet("_traj", TrajectoryJson(csStudy));
        PyExec(
            $"_space = {spaceExpr}\n" +
            $"_mfd, _imm, _ngp, _nf, _nif = _replay_cgp(_traj, _space, {seed}, {NStartup})\n");

        // The scenario must actually exercise the constrained (LogCEI) path: a mix of feasibility.
        PyLong("_nf").Should().BeGreaterThan(0, "there must be feasible trials");
        PyLong("_nif").Should().BeGreaterThan(0, "there must be infeasible trials (LogCEI is exercised)");
        PyLong("_ngp").Should().BeGreaterThan(0, "there must be GP-sampled trials to compare");
        PyLong("_imm").Should().Be(0,
            "given identical constrained history, discrete/int GP suggestions snap to the same grid point as Optuna");
        PyFloat("_mfd").Should().BeLessThan(NormDiffTol,
            "each continuous constrained GP suggestion matches Optuna's within the fit-drift envelope");
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
                sb.Append(value is double d ? FmtFloat(d) : Convert.ToInt64(value).ToString(CultureInfo.InvariantCulture));
            }
            sb.Append("},\"constraints\":[");
            sb.Append(string.Join(",", t.Constraints.Values.Select(FmtFloat)));
            sb.Append("],\"value\":").Append(FmtFloat(t.Value!.Value)).Append('}');
        }
        return sb.Append(']').ToString();
    }

    private static string FmtFloat(double d)
    {
        string s = d.ToString("R", CultureInfo.InvariantCulture);
        return s.IndexOfAny(new[] { '.', 'e', 'E' }) >= 0 ? s : s + ".0";
    }
}
