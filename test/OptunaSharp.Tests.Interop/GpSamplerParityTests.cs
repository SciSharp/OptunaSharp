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
///     Per-suggestion parity of <see cref="GPSampler" /> (single-objective path, PORTING_PLAN Phase 3.2
///     part 3d) against live Optuna's <c>GPSampler</c>. OptunaSharp runs the study; Optuna then
///     <b>replays</b> the exact same trial history one trial at a time and, at each GP trial, its
///     <c>sample_relative</c> is asked for a suggestion given the identical completed prefix. Because
///     the history is re-synced every trial (Optuna is fed OptunaSharp's actual params + value, never
///     its own), each comparison is cascade-free and the sampler RNG stays in lock-step (the
///     scrambled-Sobol' seed + the roulette <c>rng.choice</c> reproduce numpy's stream), so every GP
///     suggestion is compared under an <em>identical</em> GP fit input.
///
///     <para><b>Why not a raw trajectory compare:</b> the GP fit's kernel hyperparameters drift ~1e-3
///     from Optuna's (the loose <c>gtol=1e-2</c> leaves the argmin in a flat MLL region — the documented
///     GP boundary), which perturbs each suggestion ~1e-2 and, left to run free, occasionally flips a
///     roulette/argmax choice and makes the trajectories diverge outright. Re-syncing isolates the
///     per-suggestion tolerance, which is what the GP boundary actually bounds.</para>
///
///     <para>Continuous params are compared to a small tolerance; discrete/categorical params snap to a
///     grid and are compared as near-exact (a boundary flip under the fit drift is allowed but must be
///     rare). The first <c>n_startup_trials</c> (independent <see cref="RandomSampler" />) are bit-exact
///     and covered by the other sampler parity suites.</para>
/// </summary>
[TestClass]
public class GpSamplerParityTests : OptunaInteropTestBase
{
    private const int NStartup = 10;  // GPSampler default n_startup_trials
    // Per-suggestion drift bound in the GP-normalized [0,1] coordinate. The GP fit's kernel
    // hyperparameters drift ~1e-3 from Optuna's (loose gtol=1e-2 → flat MLL argmin), which moves the
    // LogEI argmax ~1e-2 in the normalized box; observed max across the seeds here is ~9.6e-3.
    private const double NormDiffTol = 2e-2;

    private const string ReplayHelper =
        "import json, math\n" +
        "from optuna.trial import TrialState\n" +
        "from optuna.distributions import FloatDistribution\n" +
        "def _norm_diff(d, pv, cv):\n" +  // drift in the GP-normalized [0,1] coordinate (log-aware)
        "    if d.log:\n" +
        "        return abs(math.log(pv) - math.log(cv)) / (math.log(d.high) - math.log(d.low))\n" +
        "    return abs(pv - cv) / (d.high - d.low)\n" +
        "def _replay_gp(traj_json, space, seed, n_startup, n_dir):\n" +
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
        "                elif pv != cv:\n" +  // int / categorical / stepped-float snap to a grid
        "                    int_mismatches += 1\n" +
        "        for name, val in ct['params'].items():\n" +
        "            dist = space[name]\n" +
        "            study._storage.set_trial_param(tid, name, dist.to_internal_repr(val), dist)\n" +
        "        study._storage.set_trial_state_values(tid, TrialState.COMPLETE, [ct['value']])\n" +
        "    return max_norm_diff, int_mismatches, n_gp\n";

    private void AssertReplayParity(Study csStudy, int seed, string spaceExpr)
    {
        PyExec(ReplayHelper);
        PyExec("from optuna.distributions import FloatDistribution, IntDistribution, CategoricalDistribution\n");
        PySet("_traj", TrajectoryJson(csStudy));
        PyExec(
            $"_space = {spaceExpr}\n" +
            $"_mfd, _imm, _ngp = _replay_gp(_traj, _space, {seed}, {NStartup}, 1)\n");

        PyLong("_ngp").Should().BeGreaterThan(0, "there must be GP-sampled trials to compare");
        PyLong("_imm").Should().Be(0,
            "given identical history, discrete/int/categorical GP suggestions snap to the same grid point as Optuna");
        PyFloat("_mfd").Should().BeLessThan(NormDiffTol,
            "each continuous GP suggestion matches Optuna's within the fit-drift envelope (normalized coordinate)");
    }

    [TestMethod]
    [TestCategory("RequiresPython")]
    [DataRow(2, 0, 20)]
    [DataRow(2, 42, 20)]
    [DataRow(3, 7, 22)]
    public void FloatSphere_PerSuggestion_MatchesLiveOptuna(int dim, int seed, int nTrials)
    {
        PythonSession.EnsureOrInconclusive();

        var csStudy = Optuna.CreateStudy(
            storage: new InMemoryStorage(), studyName: "gp", sampler: new GPSampler(seed: seed));
        csStudy.Optimize(SphereCs(dim), nTrials: nTrials);

        string space = "{" + string.Join(", ",
            Enumerable.Range(0, dim).Select(i => $"'x{i}': FloatDistribution(-5.0, 5.0)")) + "}";
        AssertReplayParity(csStudy, seed, space);
    }

    [TestMethod]
    [TestCategory("RequiresPython")]
    [DataRow(0)]
    [DataRow(13)]
    public void MixedSpace_PerSuggestion_MatchesLiveOptuna(int seed)
    {
        PythonSession.EnsureOrInconclusive();

        // float (relative GP) + int (relative GP, discrete dim) + log-float (relative GP).
        var csStudy = Optuna.CreateStudy(
            storage: new InMemoryStorage(), studyName: "gp", sampler: new GPSampler(seed: seed));
        csStudy.Optimize(t =>
        {
            double a = t.SuggestFloat("a", -4, 4);
            long b = t.SuggestInt("bb", -5, 5);
            double c = t.SuggestFloat("c", 0.01, 100, log: true);
            return Math.Pow(a - 1.5, 2) + Math.Pow(b + 2, 2) + Math.Pow(Math.Log(c) - 1.0, 2);
        }, nTrials: 22);

        AssertReplayParity(csStudy, seed,
            "{'a': FloatDistribution(-4.0, 4.0), 'bb': IntDistribution(-5, 5), "
            + "'c': FloatDistribution(0.01, 100.0, log=True)}");
    }

    /// <summary>End-to-end sanity (no Python): GPSampler drives a shifted sphere close to its optimum.</summary>
    [TestMethod]
    public void FloatSphere_Converges()
    {
        var study = Optuna.CreateStudy(sampler: new GPSampler(seed: 1));
        study.Optimize(SphereCs(2), nTrials: 40);
        study.BestValue.Should().BeLessThan(0.05, "GP-based optimization should approach the sphere optimum");
    }

    // ---- OptunaSharp trajectory → JSON (params keep their float/int type so Python compares in kind) --

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
            sb.Append("},\"value\":").Append(FmtFloat(t.Value!.Value)).Append('}');
        }
        return sb.Append(']').ToString();
    }

    /// <summary>Format a double so it always parses back as a JSON float (a decimal point or exponent).</summary>
    private static string FmtFloat(double d)
    {
        string s = d.ToString("R", CultureInfo.InvariantCulture);
        return s.IndexOfAny(new[] { '.', 'e', 'E' }) >= 0 ? s : s + ".0";
    }

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
}
