using System;
using System.Collections.Generic;
using System.Linq;
using AwesomeAssertions;
using OptunaSharp.Pruners;
using OptunaSharp.Samplers;

namespace OptunaSharp.Tests.Interop;

/// <summary>
///     Live decision-parity for the FULL pruner surface (audit follow-up). The existing
///     <see cref="PrunerParityTests" /> covered only Median and SuccessiveHalving live; this feeds
///     an identical deterministic trial history to real Optuna and to OptunaSharp for every other
///     pruner — Percentile, Patient, Threshold, Hyperband, Wilcoxon — and requires the whole
///     prune/complete outcome sequence to match. A fixed <c>study_name</c> is used because
///     HyperbandPruner's bracket routing hashes <c>crc32("{study_name}_{number}")</c> (both Optuna
///     and OptunaSharp — Optuna documents that a fixed name is required for reproducibility).
/// </summary>
[TestClass]
public class PrunerDecisionParityTests : OptunaInteropTestBase
{
    private const int NTrials = 12;
    private const int NSteps = 12;

    // Deterministic, RNG-free value formula — identical integer/IEEE arithmetic on both sides.
    private static double Value(int i, int step) => ((i * 31 + step * 17) % 23) / 2.0 - 5.0 + step * 0.05;

    private static BasePruner MakeCSharpPruner(string key) => key switch
    {
        "median" => new MedianPruner(nStartupTrials: 2, nWarmupSteps: 0, intervalSteps: 1),
        "percentile" => new PercentilePruner(25.0, nStartupTrials: 2, nWarmupSteps: 0, intervalSteps: 1),
        "patient" => new PatientPruner(new MedianPruner(nStartupTrials: 2), 2, 0.0),
        "threshold" => new ThresholdPruner(lower: -3.0, upper: 3.0),
        "hyperband" => new HyperbandPruner(minResource: 1, maxResource: 16, reductionFactor: 2),
        "wilcoxon" => new WilcoxonPruner(pThreshold: 0.15, nStartupSteps: 2),
        _ => throw new ArgumentException($"unknown pruner key {key}"),
    };

    private static string RunCSharp(BasePruner pruner)
    {
        var study = Optuna.CreateStudy(studyName: "prune", sampler: new RandomSampler(0), pruner: pruner);
        var outcomes = new List<string>();
        for (int i = 0; i < NTrials; i++)
        {
            var t = study.Ask();
            t.SuggestFloat("x", 0, 1);
            string outcome = "complete";
            for (int step = 0; step < NSteps; step++)
            {
                t.Report(Value(i, step), step);
                if (t.ShouldPrune())
                {
                    outcome = $"p@{step}";
                    study.Tell(t, values: null, state: TrialState.Pruned);
                    break;
                }
            }
            if (outcome == "complete")
                study.Tell(t, Value(i, NSteps - 1));
            outcomes.Add(outcome);
        }
        return string.Join(",", outcomes);
    }

    [TestMethod]
    [TestCategory("RequiresPython")]
    [DataRow("median", "optuna.pruners.MedianPruner(n_startup_trials=2, n_warmup_steps=0, interval_steps=1)")]
    [DataRow("percentile", "optuna.pruners.PercentilePruner(25.0, n_startup_trials=2, n_warmup_steps=0, interval_steps=1)")]
    [DataRow("patient", "optuna.pruners.PatientPruner(optuna.pruners.MedianPruner(n_startup_trials=2), patience=2, min_delta=0.0)")]
    [DataRow("threshold", "optuna.pruners.ThresholdPruner(lower=-3.0, upper=3.0)")]
    [DataRow("hyperband", "optuna.pruners.HyperbandPruner(min_resource=1, max_resource=16, reduction_factor=2)")]
    [DataRow("wilcoxon", "optuna.pruners.WilcoxonPruner(p_threshold=0.15, n_startup_steps=2)")]
    public void PrunerDecisions_Match_LiveOptuna(string key, string pyPruner)
    {
        PythonSession.EnsureOrInconclusive();

        string csharp = RunCSharp(MakeCSharpPruner(key));

        PyExec(
            "def _oss_value(i, step):\n" +
            "    return ((i * 31 + step * 17) % 23) / 2.0 - 5.0 + step * 0.05\n" +
            "def _oss_run_pruner(pruner):\n" +
            "    study = optuna.create_study(study_name='prune', " +
            "sampler=optuna.samplers.RandomSampler(seed=0), pruner=pruner)\n" +
            "    outcomes = []\n" +
            $"    for i in range({NTrials}):\n" +
            "        t = study.ask()\n" +
            "        t.suggest_float('x', 0, 1)\n" +
            "        outcome = 'complete'\n" +
            $"        for step in range({NSteps}):\n" +
            "            t.report(_oss_value(i, step), step)\n" +
            "            if t.should_prune():\n" +
            "                outcome = f'p@{step}'\n" +
            "                study.tell(t, state=optuna.trial.TrialState.PRUNED)\n" +
            "                break\n" +
            "        if outcome == 'complete':\n" +
            $"            study.tell(t, _oss_value(i, {NSteps - 1}))\n" +
            "        outcomes.append(outcome)\n" +
            "    return ','.join(outcomes)\n" +
            $"result = _oss_run_pruner({pyPruner})\n");
        string python = PyStr("result");

        csharp.Should().Be(python, $"every {key} prune/complete outcome must match live Optuna");
    }

    /// <summary>
    ///     Pre-release review item H4, live: <see cref="PercentilePruner.NanPercentile" /> against the
    ///     <c>numpy.nanpercentile</c> that Optuna's percentile and median pruners call, bit for bit, on 3,000 report
    ///     sets drawn by Python. Values have one to four decimals (the common shape of a reported metric), so
    ///     numpy's <c>b - (b - a) * (1 - t)</c> branch and the textbook formula disagree in the last bit on a good
    ///     share of them; about one set in eight also carries a NaN, +inf or -inf, and an infinity among the two
    ///     interpolated neighbours makes numpy return NaN or an infinity depending on the branch. Inputs and
    ///     results cross the boundary as raw IEEE bits, so no decimal formatting can hide a one-ULP difference.
    /// </summary>
    [TestMethod]
    [TestCategory("RequiresPython")]
    public void NanPercentile_MatchesLiveNumpy_BitForBit()
    {
        PythonSession.EnsureOrInconclusive();

        PyExec(
            "import random, struct, warnings\n" +
            "import numpy as np\n" +
            "def _oss_bits(v):\n" +
            "    return struct.unpack('<q', struct.pack('<d', v))[0]\n" +
            "def _oss_percentile_cases():\n" +
            "    rng = random.Random(20261027)\n" +
            "    specials = [float('nan'), float('inf'), float('-inf')]\n" +
            "    lines = []\n" +
            "    for _ in range(3000):\n" +
            "        n = rng.randint(1, 7)\n" +
            "        values = [round(rng.uniform(-1, 1), rng.randint(1, 4)) for _ in range(n)]\n" +
            "        if rng.random() < 0.125:\n" +
            "            values[rng.randrange(n)] = rng.choice(specials)\n" +
            "        q = rng.choice([0.0, 10.0, 25.0, 33.0, 50.0, 60.0, 66.0, 75.0, 80.0, 90.0, 100.0])\n" +
            "        with warnings.catch_warnings():\n" +
            "            warnings.simplefilter('ignore')\n" +
            "            expected = float(np.nanpercentile(np.array(values, dtype=float), q))\n" +
            "        lines.append(f\"{_oss_bits(q)};{','.join(str(_oss_bits(v)) for v in values)};{_oss_bits(expected)}\")\n" +
            "    return '\\n'.join(lines)\n" +
            "result = _oss_percentile_cases()\n");
        string[] cases = PyStr("result").Split('\n');

        var mismatches = new List<string>();
        foreach (string line in cases)
        {
            string[] parts = line.Split(';');
            double percentile = BitConverter.Int64BitsToDouble(long.Parse(parts[0], System.Globalization.CultureInfo.InvariantCulture));
            double[] values = Array.ConvertAll(parts[1].Split(','),
                s => BitConverter.Int64BitsToDouble(long.Parse(s, System.Globalization.CultureInfo.InvariantCulture)));
            double expected = BitConverter.Int64BitsToDouble(long.Parse(parts[2], System.Globalization.CultureInfo.InvariantCulture));

            double actual = PercentilePruner.NanPercentile(values, percentile);
            bool same = double.IsNaN(expected)
                ? double.IsNaN(actual)
                : BitConverter.DoubleToInt64Bits(actual) == BitConverter.DoubleToInt64Bits(expected);
            if (!same)
                mismatches.Add($"q={percentile:R} [{string.Join(", ", values.Select(v => v.ToString("R", System.Globalization.CultureInfo.InvariantCulture)))}]: numpy {expected:R}, C# {actual:R}");
        }

        cases.Should().HaveCount(3000);
        mismatches.Should().BeEmpty("every cutoff must equal numpy.nanpercentile bit for bit");
    }

    /// <summary>
    ///     Pre-release review item H4 at the pruner level, live: the same history fed to live Optuna's and
    ///     OptunaSharp's median and percentile pruners must give the same <c>should_prune()</c>. Each row puts the
    ///     judged report exactly on, or between, numpy's cutoff and the textbook one, or completes a trial with an
    ///     infinite report; before the fix every row disagreed with Optuna.
    /// </summary>
    /// <param name="percentile">The pruner's percentile; 50 uses <c>MedianPruner</c> on both sides.</param>
    /// <param name="completedReports">The step-0 reports (and final values) of the completed trials, in order.</param>
    /// <param name="currentReport">The step-0 report of the trial being judged.</param>
    [TestMethod]
    [TestCategory("RequiresPython")]
    [DataRow(50.0, new[] { 0.01, 0.03 }, 0.02)]
    [DataRow(50.0, new[] { 1.0, 2.0, double.PositiveInfinity }, 3.0)]
    [DataRow(75.0, new[] { -0.85, -0.536, 0.566 }, 0.015000000000000013)]
    [DataRow(80.0, new[] { 0.82, -0.9 }, 0.4760000000000001)]
    public void PercentilePrunerDecision_AtTheInterpolatedCutoff_MatchesLiveOptuna(
        double percentile, double[] completedReports, double currentReport)
    {
        PythonSession.EnsureOrInconclusive();

        int nStartupTrials = completedReports.Length;
        BasePruner pruner = percentile == 50.0
            ? new MedianPruner(nStartupTrials: nStartupTrials)
            : new PercentilePruner(percentile, nStartupTrials: nStartupTrials);
        var study = Optuna.CreateStudy(sampler: new RandomSampler(0), pruner: pruner);
        foreach (double report in completedReports)
        {
            var done = study.Ask();
            done.Report(report, 0);
            study.Tell(done, report);
        }
        var judged = study.Ask();
        judged.Report(currentReport, 0);
        bool csharp = judged.ShouldPrune();

        // repr() of the doubles round-trips exactly; 'inf' needs float('inf') on the Python side.
        static string Py(double v) => double.IsPositiveInfinity(v) ? "float('inf')"
            : v.ToString("R", System.Globalization.CultureInfo.InvariantCulture);
        string pyPruner = percentile == 50.0
            ? $"optuna.pruners.MedianPruner(n_startup_trials={nStartupTrials})"
            : $"optuna.pruners.PercentilePruner({Py(percentile)}, n_startup_trials={nStartupTrials})";
        PyExec(
            $"study = optuna.create_study(pruner={pyPruner})\n" +
            $"for v in [{string.Join(", ", completedReports.Select(Py))}]:\n" +
            "    t = study.ask()\n" +
            "    t.report(v, 0)\n" +
            "    study.tell(t, v)\n" +
            "judged = study.ask()\n" +
            $"judged.report({Py(currentReport)}, 0)\n" +
            "result = judged.should_prune()\n");
        bool python = PyBool("result");

        csharp.Should().Be(python,
            $"should_prune() for a report of {currentReport:R} after {string.Join(", ", completedReports)} must match live Optuna");
    }
}
