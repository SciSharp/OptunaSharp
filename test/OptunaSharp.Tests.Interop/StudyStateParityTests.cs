using System;
using OptunaSharp.Storages;
using Python.Runtime;

namespace OptunaSharp.Tests.Interop;

/// <summary>
///     Differential full-state fuzzing: the SAME seeded scenario (mixed search space; reports
///     incl. NaN/±inf; unicode + nested attrs; constraints; complete/fail/prune outcomes;
///     enqueued fixed params; waiting leftovers) runs independently on live Python Optuna and
///     on OptunaSharp, and the complete study states must agree — every trial, every field,
///     floats bit-exact. Datetimes compare by null-ness (independent wall clocks).
/// </summary>
[TestClass]
public class StudyStateParityTests : OptunaInteropTestBase
{
    private void RunDifferential(int seed, bool multiObjective)
    {
        var scenario = ParityScenario.Generate(seed, multiObjective);

        // --- OptunaSharp side (before taking the GIL). -----------------------------------
        var csStudy = Optuna.CreateStudy(
            storage: new InMemoryStorage(),
            sampler: scenario.NewSampler(),
            studyName: "diff",
            directions: multiObjective
                ? new[] { StudyDirection.Minimize, StudyDirection.Minimize }
                : new[] { StudyDirection.Minimize });
        scenario.RunCSharp(csStudy);
        var csState = StudySnapshot.Capture(csStudy);

        // --- live Optuna side. -----------------------------------------------------------
        PySet("spec_json", scenario.ToSpecJson());
        PyExec(ParityScenario.PythonRunnerSource);
        PyExec(
            "study = optuna.create_study(study_name='diff', " +
            "sampler=optuna.samplers.RandomSampler(seed=spec['sampler_seed']), " +
            "directions=(['minimize', 'minimize'] if spec['multi_objective'] else ['minimize']))\n" +
            "_oss_run(study)\n");
        var pyState = PyStudyState("study");

        StudySnapshot.AssertParity(pyState, csState, DatetimeComparison.NullnessOnly);
    }

    [TestMethod]
    [TestCategory("RequiresPython")]
    [DataRow(1)]
    [DataRow(2)]
    [DataRow(3)]
    [DataRow(4)]
    [DataRow(5)]
    [DataRow(6)]
    [DataRow(1337)]
    [DataRow(20260830)]
    public void FullState_Differential_SingleObjective(int seed) => RunDifferential(seed, multiObjective: false);

    [TestMethod]
    [TestCategory("RequiresPython")]
    [DataRow(11)]
    [DataRow(12)]
    [DataRow(13)]
    public void FullState_Differential_MultiObjective(int seed) => RunDifferential(seed, multiObjective: true);

    [TestMethod]
    [TestCategory("RequiresPython")]
    public void BestTrial_Number_Agrees_Across_The_Fuzz_Corpus()
    {
        // Beyond raw state parity: derived best-trial selection agrees for every scenario that
        // produced at least one COMPLETE trial.
        for (int seed = 21; seed < 27; seed++)
        {
            var scenario = ParityScenario.Generate(seed);
            var csStudy = Optuna.CreateStudy(
                storage: new InMemoryStorage(), sampler: scenario.NewSampler(), studyName: $"best{seed}");
            scenario.RunCSharp(csStudy);

            PySet("spec_json", scenario.ToSpecJson());
            PyExec(ParityScenario.PythonRunnerSource);
            PyExec(
                $"study = optuna.create_study(study_name='best{seed}', " +
                "sampler=optuna.samplers.RandomSampler(seed=spec['sampler_seed']))\n" +
                "_oss_run(study)\n" +
                "has_complete = any(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials)\n" +
                "best_number = study.best_trial.number if has_complete else -1\n" +
                "best_value = float(study.best_value) if has_complete else 0.0\n");

            if (PyBool("has_complete"))
            {
                long pyBestNumber = PyLong("best_number");
                if (csStudy.BestTrial.Number != pyBestNumber)
                    throw new AssertFailedException(
                        $"best_trial.number mismatch for {scenario}: C# {csStudy.BestTrial.Number} != Python {pyBestNumber}");
                double pyBest = PyFloat("best_value");
                if (csStudy.BestValue != pyBest)
                    throw new AssertFailedException(
                        $"best_value mismatch for {scenario}: C# {csStudy.BestValue} != Python {pyBest}");
            }
        }
    }
}
