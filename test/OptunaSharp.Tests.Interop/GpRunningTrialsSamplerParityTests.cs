using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Text;
using AwesomeAssertions;
using OptunaSharp.Distributions;
using OptunaSharp.Samplers;
using OptunaSharp.Storages;

namespace OptunaSharp.Tests.Interop;

/// <summary>
///     Per-suggestion parity of the <b>running-trial</b> <see cref="GPSampler" /> paths (PORTING_PLAN
///     Phase 3.2 part 3e-6) against live Optuna. When trials are in flight (asked but not yet told),
///     GPSampler switches to the q-batch acquisition functions — <c>qLogEI</c> (single-objective),
///     <c>qLogCEI</c> (single-objective constrained), <c>qLogEHVI</c> (multi-objective), and, since
///     Optuna 5.0.0 (#6804), <c>qLogCEHVI</c> (multi-objective constrained; before 5.0.0 this case was the
///     Kriging-Believer <c>LogCEHVI</c>).
///
///     <para>OptunaSharp drives an <c>Ask</c>/<c>Tell</c> schedule that <b>leaves some trials running</b>
///     (asked + suggested, never told) so the trial under test sees them; Optuna then <b>replays</b> the
///     exact trajectory one trial at a time — re-fed OptunaSharp's params/values/constraints, running
///     trials left RUNNING with their shared params — and its <c>sample_relative</c> is asked for a
///     suggestion under the identical prefix. The sampler RNG stays in lock-step (each trial's
///     <c>sample_relative</c> is invoked in order, re-synced afterward), so continuous suggestions are
///     compared within a normalized-coordinate envelope and discrete/int params near-exact — the same
///     methodology as the sequential <see cref="GpSamplerParityTests" />, extended with running trials.</para>
/// </summary>
[TestClass]
public class GpRunningTrialsSamplerParityTests : OptunaInteropTestBase
{
    private const int NStartup = 10;
    private const double NormDiffTol = 5e-2;   // fit drift + the q-batch QMC estimate

    private const string ReplayHelper =
        "import json, math\n" +
        "from optuna.trial import TrialState\n" +
        "from optuna.distributions import FloatDistribution, IntDistribution, CategoricalDistribution\n" +
        "def _norm_diff(d, pv, cv):\n" +
        "    if getattr(d, 'log', False):\n" +
        "        return abs(math.log(pv) - math.log(cv)) / (math.log(d.high) - math.log(d.low))\n" +
        "    return abs(pv - cv) / (d.high - d.low)\n" +
        "def _replay_running(traj_json, space, seed, n_startup, n_dir, has_constraints):\n" +
        "    cs = json.loads(traj_json)\n" +
        "    if n_dir == 1:\n" +
        "        study = optuna.create_study(sampler=optuna.samplers.GPSampler(seed=seed))\n" +
        "    else:\n" +
        "        study = optuna.create_study(directions=['minimize'] * n_dir,\n" +
        "            sampler=optuna.samplers.GPSampler(seed=seed))\n" +
        "    sampler = study.sampler\n" +
        "    max_norm_diff = 0.0\n" +
        "    int_mismatches = 0\n" +
        "    n_gp = 0\n" +
        "    n_gp_with_running = 0\n" +
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
        "            n_gp += 1\n" +
        "            if n_run_present > 0:\n" +
        "                n_gp_with_running += 1\n" +
        "            for name, pv in rel.items():\n" +
        "                cv = ct['params'][name]\n" +
        "                d = space[name]\n" +
        "                if isinstance(d, FloatDistribution) and d.step is None:\n" +
        "                    max_norm_diff = max(max_norm_diff, _norm_diff(d, pv, cv))\n" +
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
        "    return max_norm_diff, int_mismatches, n_gp, n_gp_with_running, n_running_left\n";

    [TestMethod]
    [TestCategory("RequiresPython")]
    [DataRow(0)]
    [DataRow(42)]
    public void SingleObjectiveWithRunning_PerSuggestion_MatchesLiveOptuna(int seed)
    {
        // qLogEI — single-objective with running trials.
        RunAndAssert(
            seed, nDir: 1, hasConstraints: false, nCompletedGp: 3, nRunning: 3,
            objective: (a, b) => (new[] { a * a + b * b }, null));
    }

    [TestMethod]
    [TestCategory("RequiresPython")]
    [DataRow(0)]
    [DataRow(42)]
    public void MultiObjectiveWithRunning_PerSuggestion_MatchesLiveOptuna(int seed)
    {
        // qLogEHVI — multi-objective with running trials.
        RunAndAssert(
            seed, nDir: 2, hasConstraints: false, nCompletedGp: 3, nRunning: 3,
            objective: (a, b) => (new[] { a * a + b * b, (a - 2) * (a - 2) + b * b }, null));
    }

    [TestMethod]
    [TestCategory("RequiresPython")]
    [DataRow(0)]
    [DataRow(42)]
    public void SingleObjectiveConstrainedWithRunning_PerSuggestion_MatchesLiveOptuna(int seed)
    {
        // qLogCEI — single-objective constrained with running trials.
        RunAndAssert(
            seed, nDir: 1, hasConstraints: true, nCompletedGp: 3, nRunning: 3,
            objective: (a, b) => (new[] { a * a + b * b }, new[] { a - 1.0 }));
    }

    [TestMethod]
    [TestCategory("RequiresPython")]
    [DataRow(0)]
    [DataRow(42)]
    public void MultiObjectiveConstrainedWithRunning_PerSuggestion_MatchesLiveOptuna(int seed)
    {
        // qLogCEHVI (Optuna 5.0.0, #6804) — multi-objective constrained with running trials.
        RunAndAssert(
            seed, nDir: 2, hasConstraints: true, nCompletedGp: 3, nRunning: 3,
            objective: (a, b) => (new[] { a * a + b * b, (a - 2) * (a - 2) + b * b }, new[] { a - 1.0 }));
    }

    /// <summary>
    ///     Drives an ask/tell schedule that leaves <paramref name="nRunning" /> trials running before a
    ///     final trial-under-test, captures the trajectory, and replays it through live Optuna.
    /// </summary>
    private void RunAndAssert(
        int seed, int nDir, bool hasConstraints, int nCompletedGp, int nRunning,
        Func<double, double, (double[] Values, double[]? Constraints)> objective)
    {
        PythonSession.EnsureOrInconclusive();

        var directions = nDir == 1
            ? new[] { StudyDirection.Minimize }
            : new[] { StudyDirection.Minimize, StudyDirection.Minimize };
        var study = Optuna.CreateStudy(
            storage: new InMemoryStorage(), studyName: "rgp",
            sampler: new GPSampler(seed: seed), directions: directions);

        var traj = new List<string>();
        const double Lo = -5, Hi = 5;

        // Completed trials: NStartup independent + nCompletedGp GP-sampled.
        for (int i = 0; i < NStartup + nCompletedGp; i++)
        {
            var t = study.Ask();
            double a = t.SuggestFloat("a", Lo, Hi);
            double b = t.SuggestFloat("b", Lo, Hi);
            (double[] values, double[]? constraints) = objective(a, b);
            if (hasConstraints)
                for (int c = 0; c < constraints!.Length; c++)
                    t.SetConstraint(c.ToString(CultureInfo.InvariantCulture), constraints[c]);
            study.Tell(t, values);
            traj.Add(TrajTrial(a, b, values, constraints, running: false, hasConstraints));
        }

        // Running trials (asked + suggested, never told) plus the trial under test (also left running).
        for (int i = 0; i < nRunning + 1; i++)
        {
            var t = study.Ask();
            double a = t.SuggestFloat("a", Lo, Hi);
            double b = t.SuggestFloat("b", Lo, Hi);
            traj.Add(TrajTrial(a, b, null, null, running: true, hasConstraints));
        }

        PyExec(ReplayHelper);
        PySet("_traj", "[" + string.Join(",", traj) + "]");
        PyExec(
            "_space = {'a': FloatDistribution(-5.0, 5.0), 'b': FloatDistribution(-5.0, 5.0)}\n" +
            $"_mfd, _imm, _ngp, _ngpr, _nrl = _replay_running(_traj, _space, {seed}, {NStartup}, {nDir}, " +
            $"{(hasConstraints ? "True" : "False")})\n");

        PyLong("_ngp").Should().BeGreaterThan(0, "there must be GP-sampled trials to compare");
        PyLong("_ngpr").Should().BeGreaterThan(0, "the running-trial (q-batch/KB) acqf path must be exercised");
        PyLong("_nrl").Should().Be(nRunning + 1, "the running trials must stay RUNNING in the replay");
        PyLong("_imm").Should().Be(0, "discrete/int GP suggestions snap to the same grid point as Optuna");
        PyFloat("_mfd").Should().BeLessThan(NormDiffTol,
            "each continuous running-trial GP suggestion matches Optuna within the fit-drift envelope");
    }

    private static string TrajTrial(
        double a, double b, double[]? values, double[]? constraints, bool running, bool hasConstraints)
    {
        var sb = new StringBuilder("{\"params\":{\"a\":");
        sb.Append(FmtFloat(a)).Append(",\"b\":").Append(FmtFloat(b)).Append("},\"running\":");
        sb.Append(running ? "true" : "false");
        if (!running)
        {
            sb.Append(",\"values\":[").Append(string.Join(",", values!.Select(FmtFloat))).Append(']');
            if (hasConstraints)
                sb.Append(",\"constraints\":[").Append(string.Join(",", constraints!.Select(FmtFloat))).Append(']');
        }
        return sb.Append('}').ToString();
    }

    private static string FmtFloat(double d)
    {
        string s = d.ToString("R", CultureInfo.InvariantCulture);
        return s.IndexOfAny(new[] { '.', 'e', 'E' }) >= 0 ? s : s + ".0";
    }
}
