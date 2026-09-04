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
///     Per-suggestion parity of the <b>multi-objective</b> <see cref="GPSampler" /> path (PORTING_PLAN
///     Phase 3.2 part 3e-4 — the <see cref="LogEHVI" /> acquisition function) against live Optuna.
///     OptunaSharp runs a 2-objective study; Optuna then <b>replays</b> the exact history one trial at a
///     time (re-fed OptunaSharp's params and both values each trial) and its <c>sample_relative</c> is
///     asked for a suggestion under the identical Pareto-front prefix. The sampler RNG stays in
///     lock-step: the EHVI QMC seed (<c>randint</c>), the Pareto warmstart choice (a permutation), and
///     the acqf-optimizer draws are consumed in the same order as Optuna.
///
///     <para>All objectives' GPs are fit by marginal-likelihood maximization (~1e-3 kernel-hyperparameter
///     drift, the loose <c>gtol=1e-2</c>), and the EHVI is a QMC estimate over box decompositions, so
///     continuous suggestions are compared within a normalized-coordinate envelope and discrete/int
///     params near-exact. Same replay rationale as the single-objective <see cref="GpSamplerParityTests" />.</para>
/// </summary>
[TestClass]
public class GpMultiObjectiveSamplerParityTests : OptunaInteropTestBase
{
    private const int NStartup = 10;
    private const double NormDiffTol = 4e-2;  // two-GP fit drift + the EHVI QMC estimate

    private const string ReplayHelper =
        "import json, math\n" +
        "from optuna.trial import TrialState\n" +
        "from optuna.distributions import FloatDistribution, IntDistribution, CategoricalDistribution\n" +
        "def _norm_diff(d, pv, cv):\n" +
        "    if d.log:\n" +
        "        return abs(math.log(pv) - math.log(cv)) / (math.log(d.high) - math.log(d.low))\n" +
        "    return abs(pv - cv) / (d.high - d.low)\n" +
        "def _replay_mogp(traj_json, space, seed, n_startup, n_dir):\n" +
        "    cs = json.loads(traj_json)\n" +
        "    study = optuna.create_study(directions=['minimize'] * n_dir,\n" +
        "        sampler=optuna.samplers.GPSampler(seed=seed))\n" +
        "    sampler = study.sampler\n" +
        "    max_norm_diff = 0.0\n" +
        "    int_mismatches = 0\n" +
        "    n_gp = 0\n" +
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
        "        study._storage.set_trial_state_values(tid, TrialState.COMPLETE, ct['values'])\n" +
        "    return max_norm_diff, int_mismatches, n_gp\n";

    [TestMethod]
    [TestCategory("RequiresPython")]
    [DataRow(0, 24)]
    [DataRow(42, 24)]
    public void MultiObjectiveFloat_PerSuggestion_MatchesLiveOptuna(int seed, int nTrials)
    {
        PythonSession.EnsureOrInconclusive();

        var csStudy = Optuna.CreateStudy(
            storage: new InMemoryStorage(), studyName: "mogp",
            sampler: new GPSampler(seed: seed),
            directions: new[] { StudyDirection.Minimize, StudyDirection.Minimize });
        csStudy.Optimize(t =>
        {
            double a = t.SuggestFloat("a", -5, 5);
            double b = t.SuggestFloat("b", -5, 5);
            return (System.Collections.Generic.IReadOnlyList<double>)
                new[] { a * a + b * b, (a - 2) * (a - 2) + b * b };
        }, nTrials: nTrials);

        AssertReplayParity(csStudy, seed, 2,
            "{'a': FloatDistribution(-5.0, 5.0), 'b': FloatDistribution(-5.0, 5.0)}");
    }

    [TestMethod]
    [TestCategory("RequiresPython")]
    [DataRow(7, 24)]
    public void MultiObjectiveMixed_PerSuggestion_MatchesLiveOptuna(int seed, int nTrials)
    {
        PythonSession.EnsureOrInconclusive();

        // float + int (discrete relative dim), two objectives.
        var csStudy = Optuna.CreateStudy(
            storage: new InMemoryStorage(), studyName: "mogp",
            sampler: new GPSampler(seed: seed),
            directions: new[] { StudyDirection.Minimize, StudyDirection.Minimize });
        csStudy.Optimize(t =>
        {
            double a = t.SuggestFloat("a", -4, 4);
            long b = t.SuggestInt("bb", -5, 5);
            return (System.Collections.Generic.IReadOnlyList<double>)
                new[] { a * a + b * b, (a - 1) * (a - 1) + (b + 1) * (b + 1) };
        }, nTrials: nTrials);

        AssertReplayParity(csStudy, seed, 2,
            "{'a': FloatDistribution(-4.0, 4.0), 'bb': IntDistribution(-5, 5)}");
    }

    private void AssertReplayParity(Study csStudy, int seed, int nDir, string spaceExpr)
    {
        PyExec(ReplayHelper);
        PySet("_traj", TrajectoryJson(csStudy));
        PyExec(
            $"_space = {spaceExpr}\n" +
            $"_mfd, _imm, _ngp = _replay_mogp(_traj, _space, {seed}, {NStartup}, {nDir})\n");

        PyLong("_ngp").Should().BeGreaterThan(0, "there must be GP-sampled trials to compare");
        PyLong("_imm").Should().Be(0,
            "given identical history, discrete/int GP suggestions snap to the same grid point as Optuna");
        PyFloat("_mfd").Should().BeLessThan(NormDiffTol,
            "each continuous multi-objective GP suggestion matches Optuna's within the fit-drift envelope");
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
            sb.Append("},\"values\":[");
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
