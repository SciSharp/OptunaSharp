using System;
using System.Collections.Generic;
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
}
