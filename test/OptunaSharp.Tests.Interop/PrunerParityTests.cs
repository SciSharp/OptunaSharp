using System;
using System.Collections.Generic;
using System.Linq;
using AwesomeAssertions;
using OptunaSharp.Pruners;
using OptunaSharp.Samplers;
using Python.Runtime;

namespace OptunaSharp.Tests.Interop;

/// <summary>
///     Live decision-parity for the default pruning experience: identical trial histories are
///     fed to real Optuna's MedianPruner and to OptunaSharp's, and every
///     <c>should_prune</c> decision along the reporting sequence must match. This exercises the
///     full infra loop (storage → report → pruner → prune decision) with one algorithm.
/// </summary>
[TestClass]
public class PrunerParityTests
{
    [TestMethod]
    [TestCategory("RequiresPython")]
    [DataRow("minimize", 5)]
    [DataRow("maximize", 5)]
    [DataRow("minimize", 99)]
    public void MedianPruner_Decisions_Match_LiveOptuna(string direction, int seed)
    {
        PythonSession.EnsureOrInconclusive();

        // Deterministic pseudo-random histories: 6 completed trials × 8 steps, then a probe
        // trial reporting 8 values with a should_prune decision after each report.
        var rng = new Random(seed);
        var histories = new double[6][];
        for (int i = 0; i < histories.Length; i++)
            histories[i] = Enumerable.Range(0, 8).Select(_ => Math.Round(rng.NextDouble() * 10, 3)).ToArray();
        var probe = Enumerable.Range(0, 8).Select(_ => Math.Round(rng.NextDouble() * 10, 3)).ToArray();

        // --- C# side -------------------------------------------------------------------
        var study = Optuna.CreateStudy(
            sampler: new RandomSampler(0),
            pruner: new MedianPruner(nStartupTrials: 2, nWarmupSteps: 1, intervalSteps: 2),
            direction: direction == "maximize" ? StudyDirection.Maximize : StudyDirection.Minimize);
        foreach (var history in histories)
        {
            var t = study.Ask();
            t.SuggestFloat("x", 0, 1);
            for (int step = 0; step < history.Length; step++)
                t.Report(history[step], step);
            study.Tell(t, history[^1]);
        }
        var probeTrial = study.Ask();
        probeTrial.SuggestFloat("x", 0, 1);
        var csharpDecisions = new List<bool>();
        for (int step = 0; step < probe.Length; step++)
        {
            probeTrial.Report(probe[step], step);
            csharpDecisions.Add(probeTrial.ShouldPrune());
        }
        study.Tell(probeTrial, probe[^1]);

        // --- Python side ---------------------------------------------------------------
        string historiesPy = "[" + string.Join(",", histories.Select(
            h => "[" + string.Join(",", h.Select(v => v.ToString("R", System.Globalization.CultureInfo.InvariantCulture))) + "]")) + "]";
        string probePy = "[" + string.Join(",", probe.Select(v => v.ToString("R", System.Globalization.CultureInfo.InvariantCulture))) + "]";

        List<bool> pythonDecisions = new();
        using (Py.GIL())
        {
            using PyModule scope = Py.CreateScope();
            scope.Exec(
                "import optuna\n" +
                "optuna.logging.set_verbosity(optuna.logging.WARNING)\n" +
                $"histories = {historiesPy}\n" +
                $"probe = {probePy}\n" +
                "study = optuna.create_study(\n" +
                "    sampler=optuna.samplers.RandomSampler(seed=0),\n" +
                "    pruner=optuna.pruners.MedianPruner(n_startup_trials=2, n_warmup_steps=1, interval_steps=2),\n" +
                $"    direction='{direction}')\n" +
                "for h in histories:\n" +
                "    t = study.ask()\n" +
                "    t.suggest_float('x', 0, 1)\n" +
                "    for i, v in enumerate(h):\n" +
                "        t.report(v, i)\n" +
                "    study.tell(t, h[-1])\n" +
                "t = study.ask()\n" +
                "t.suggest_float('x', 0, 1)\n" +
                "decisions = []\n" +
                "for i, v in enumerate(probe):\n" +
                "    t.report(v, i)\n" +
                "    decisions.append(bool(t.should_prune()))\n" +
                "study.tell(t, probe[-1])\n");
            using var decisions = scope.Get("decisions");
            long count = decisions.Length();
            for (int i = 0; i < count; i++)
                pythonDecisions.Add(decisions[i].As<bool>());
        }

        csharpDecisions.Should().Equal(pythonDecisions,
            $"every should_prune decision must match live Optuna (direction={direction}, seed={seed})");
    }

    [TestMethod]
    [TestCategory("RequiresPython")]
    public void SuccessiveHalving_Decisions_Match_LiveOptuna()
    {
        PythonSession.EnsureOrInconclusive();

        // A fully deterministic ASHA scenario: 10 trials, values fixed by index, sequential
        // pruning decisions after each report. Decision-by-decision parity.
        const int nTrials = 10;
        const int nSteps = 9;
        double Value(int trial, int step) => (trial % 5) + step * 0.01;

        var study = Optuna.CreateStudy(
            sampler: new RandomSampler(0),
            pruner: new SuccessiveHalvingPruner(minResource: 1, reductionFactor: 2));
        var csharpOutcomes = new List<string>();
        for (int i = 0; i < nTrials; i++)
        {
            var t = study.Ask();
            t.SuggestFloat("x", 0, 1);
            string outcome = "complete";
            for (int step = 0; step < nSteps; step++)
            {
                t.Report(Value(i, step), step);
                if (t.ShouldPrune())
                {
                    outcome = $"pruned@{step}";
                    study.Tell(t, values: null, state: TrialState.Pruned);
                    break;
                }
            }
            if (outcome == "complete")
                study.Tell(t, Value(i, nSteps - 1));
            csharpOutcomes.Add(outcome);
        }

        var pythonOutcomes = new List<string>();
        using (Py.GIL())
        {
            using PyModule scope = Py.CreateScope();
            scope.Exec(
                "import optuna\n" +
                "optuna.logging.set_verbosity(optuna.logging.WARNING)\n" +
                "study = optuna.create_study(\n" +
                "    sampler=optuna.samplers.RandomSampler(seed=0),\n" +
                "    pruner=optuna.pruners.SuccessiveHalvingPruner(min_resource=1, reduction_factor=2))\n" +
                "outcomes = []\n" +
                f_value_def() +
                $"for i in range({nTrials}):\n" +
                "    t = study.ask()\n" +
                "    t.suggest_float('x', 0, 1)\n" +
                "    outcome = 'complete'\n" +
                $"    for step in range({nSteps}):\n" +
                "        t.report(value(i, step), step)\n" +
                "        if t.should_prune():\n" +
                "            outcome = f'pruned@{step}'\n" +
                "            study.tell(t, state=optuna.trial.TrialState.PRUNED)\n" +
                "            break\n" +
                "    if outcome == 'complete':\n" +
                $"        study.tell(t, value(i, {nSteps - 1}))\n" +
                "    outcomes.append(outcome)\n");
            using var outcomes = scope.Get("outcomes");
            long count = outcomes.Length();
            for (int i = 0; i < count; i++)
                pythonOutcomes.Add(outcomes[i].As<string>());
        }

        csharpOutcomes.Should().Equal(pythonOutcomes,
            "every ASHA prune/complete outcome must match live Optuna");

        static string f_value_def()
            => "def value(i, step):\n    return (i % 5) + step * 0.01\n";
    }
}
