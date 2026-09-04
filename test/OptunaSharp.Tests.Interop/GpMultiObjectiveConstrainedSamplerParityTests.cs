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
///     Per-suggestion parity of the <b>multi-objective constrained</b> <see cref="GPSampler" /> path
///     (PORTING_PLAN Phase 3.2 part 3e-5 — the <see cref="LogCEHVI" /> acquisition function) against live
///     Optuna. OptunaSharp runs a 2-objective constrained study (a genuine feasible/infeasible mix via
///     <c>Trial.SetConstraint</c>); Optuna then <b>replays</b> the exact history one trial at a time —
///     re-fed OptunaSharp's params, both values, and the constraint each trial — and its
///     <c>sample_relative</c> is asked for a suggestion under the identical prefix. LogCEHVI is
///     LogEHVI over the feasible trials plus the summed constraint log feasible-probabilities; the
///     sampler RNG stays in lock-step (the EHVI QMC seed, the feasible-Pareto warmstart permutation, and
///     the acqf-optimizer draws are consumed in Optuna's order).
///
///     <para>Three GPs are fit here (two objectives + one constraint), all at the loose <c>gtol=1e-2</c>,
///     so continuous suggestions are compared within a normalized-coordinate envelope and discrete/int
///     params near-exact. Same replay rationale as the single-objective <see cref="GpSamplerParityTests" />.</para>
/// </summary>
[TestClass]
public class GpMultiObjectiveConstrainedSamplerParityTests : OptunaInteropTestBase
{
    private const int NStartup = 10;
    private const double NormDiffTol = 5e-2;  // three-GP fit drift + the EHVI QMC estimate

    private const string ReplayHelper =
        "import json, math\n" +
        "from optuna.trial import TrialState\n" +
        "from optuna.distributions import FloatDistribution, IntDistribution, CategoricalDistribution\n" +
        "def _norm_diff(d, pv, cv):\n" +
        "    if d.log:\n" +
        "        return abs(math.log(pv) - math.log(cv)) / (math.log(d.high) - math.log(d.low))\n" +
        "    return abs(pv - cv) / (d.high - d.low)\n" +
        "def _replay_mocgp(traj_json, space, seed, n_startup, n_dir):\n" +
        "    cs = json.loads(traj_json)\n" +
        "    study = optuna.create_study(directions=['minimize'] * n_dir,\n" +
        "        sampler=optuna.samplers.GPSampler(seed=seed))\n" +
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
        "        study._storage.set_trial_state_values(tid, TrialState.COMPLETE, ct['values'])\n" +
        "    return max_norm_diff, int_mismatches, n_gp, n_feasible, n_infeasible\n";

    [TestMethod]
    [TestCategory("RequiresPython")]
    [DataRow(0, 26)]
    [DataRow(42, 26)]
    public void MultiObjectiveConstrainedFloat_PerSuggestion_MatchesLiveOptuna(int seed, int nTrials)
    {
        PythonSession.EnsureOrInconclusive();

        var csStudy = Optuna.CreateStudy(
            storage: new InMemoryStorage(), studyName: "mocgp",
            sampler: new GPSampler(seed: seed),
            directions: new[] { StudyDirection.Minimize, StudyDirection.Minimize });
        csStudy.Optimize(t =>
        {
            double a = t.SuggestFloat("a", -5, 5);
            double b = t.SuggestFloat("b", -5, 5);
            t.SetConstraint("0", a - 1.0);   // feasible when a <= 1
            return (System.Collections.Generic.IReadOnlyList<double>)
                new[] { a * a + b * b, (a - 2) * (a - 2) + b * b };
        }, nTrials: nTrials);

        PyExec(ReplayHelper);
        PySet("_traj", TrajectoryJson(csStudy));
        PyExec(
            "_space = {'a': FloatDistribution(-5.0, 5.0), 'b': FloatDistribution(-5.0, 5.0)}\n" +
            $"_mfd, _imm, _ngp, _nf, _nif = _replay_mocgp(_traj, _space, {seed}, {NStartup}, 2)\n");

        PyLong("_nf").Should().BeGreaterThan(0, "there must be feasible trials (LogEHVI term present)");
        PyLong("_nif").Should().BeGreaterThan(0, "there must be infeasible trials (constraints active)");
        PyLong("_ngp").Should().BeGreaterThan(0, "there must be GP-sampled trials to compare");
        PyLong("_imm").Should().Be(0, "discrete/int GP suggestions snap to the same grid point as Optuna");
        PyFloat("_mfd").Should().BeLessThan(NormDiffTol,
            "each continuous MO-constrained GP suggestion matches Optuna's within the fit-drift envelope");
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
            sb.Append("],\"values\":[");
            sb.Append(string.Join(",", t.Values!.Select(FmtFloat)));
            sb.Append("]}");
        }
        return sb.Append(']').ToString();
    }

    private static string FmtFloat(double d)
    {
        string s = d.ToString("R", CultureInfo.InvariantCulture);
        return s.IndexOfAny(new[] { '.', 'e', 'E' }) >= 0 ? s : s + ".0";
    }
}
