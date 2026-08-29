using AwesomeAssertions;
using Python.Runtime;

namespace OptunaSharp.Tests.Interop;

/// <summary>
///     Interop scaffolding smoke tests. No OptunaSharp algorithms are ported yet, so these prove the
///     <b>live-Optuna</b> harness works end-to-end: the embedded engine starts, real Optuna imports,
///     and a real study runs in-process. This is the template that future byte/behavior parity tests
///     follow — build an input, run it in both stacks, and assert OptunaSharp matches real Optuna.
/// </summary>
[TestClass]
public class OptunaInteropSmokeTests
{
    [TestMethod]
    [TestCategory("RequiresPython")]
    public void Engine_ImportsOptuna_ReportsVersion()
    {
        PythonSession.EnsureOrInconclusive();

        PythonSession.OptunaVersion.Should().NotBeNullOrWhiteSpace(
            "the interop suite must be able to see the installed Optuna's __version__");
    }

    [TestMethod]
    [TestCategory("RequiresPython")]
    public void Optuna_MinimizesSimpleObjective_EndToEnd()
    {
        PythonSession.EnsureOrInconclusive();

        // Drive real Optuna: minimize (x - 2)^2. This exercises the full study/trial/sampler path
        // in-process — exactly the machinery OptunaSharp will re-implement and be checked against.
        using (Py.GIL())
        {
            using PyModule scope = Py.CreateScope();
            scope.Exec(
                "import optuna\n" +
                "optuna.logging.set_verbosity(optuna.logging.WARNING)\n" +
                "def objective(trial):\n" +
                "    x = trial.suggest_float('x', -10, 10)\n" +
                "    return (x - 2) ** 2\n" +
                "study = optuna.create_study()\n" +
                "study.optimize(objective, n_trials=25)\n" +
                "best_value = float(study.best_value)\n" +
                "n_trials = len(study.trials)\n");

            double bestValue = scope.Get("best_value").As<double>();
            int nTrials = scope.Get("n_trials").As<int>();

            nTrials.Should().Be(25);
            bestValue.Should().BeGreaterThanOrEqualTo(0.0, "(x - 2)^2 is non-negative");
        }
    }
}
