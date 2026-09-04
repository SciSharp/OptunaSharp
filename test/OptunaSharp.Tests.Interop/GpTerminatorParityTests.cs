using System.Globalization;
using System.Linq;
using AwesomeAssertions;
using OptunaSharp.Samplers;
using OptunaSharp.Terminator;

namespace OptunaSharp.Tests.Interop;

/// <summary>
///     Live parity for the Phase 3.3 GP-based improvement evaluators —
///     <see cref="RegretBoundEvaluator" /> and <see cref="EMMREvaluator" />. An identical trial
///     history is produced on both sides by the bit-exact <see cref="RandomSampler" /> (seed 0) over a
///     shared <c>x·x</c> objective (kept bit-identical — no scalar <c>**</c>), then each evaluator is
///     run with the same seed so its internal search-space sampling (scrambled Sobol') advances in RNG
///     lock-step. The evaluators ride the native GP fit + posterior, so the criterion matches live
///     Optuna to a <b>tolerance</b>, not bit-for-bit — the documented GP boundary (managed L-BFGS-B and
///     triangular solve drift from scipy/torch by ULPs, moving the fitted hyperparameters ~1e-3).
/// </summary>
[TestClass]
public class GpTerminatorParityTests : OptunaInteropTestBase
{
    // A deterministic study history: RandomSampler(0) params + a bit-identical x·x objective.
    private const string PyBuildStudy =
        "import warnings, numpy as np, optuna, optuna.terminator\n" +
        "optuna.logging.set_verbosity(optuna.logging.WARNING)\n" +
        "def _oss_build(direction, n_trials, dim):\n" +
        "    study = optuna.create_study(direction=direction, sampler=optuna.samplers.RandomSampler(seed=0))\n" +
        "    with warnings.catch_warnings():\n" +
        "        warnings.simplefilter('ignore')\n" +
        "        for _ in range(n_trials):\n" +
        "            t = study.ask()\n" +
        "            xs = [t.suggest_float(f'x{d}', -10.0, 10.0) for d in range(dim)]\n" +
        "            study.tell(t, sum(x * x for x in xs))\n" +
        "    return study\n";

    private static Study BuildStudy(StudyDirection direction, int nTrials, int dim)
    {
        var study = Optuna.CreateStudy(direction: direction, sampler: new RandomSampler(0));
        for (int i = 0; i < nTrials; i++)
        {
            var t = study.Ask();
            double sum = 0.0;
            for (int d = 0; d < dim; d++)
            {
                double x = t.SuggestFloat($"x{d}", -10.0, 10.0);
                sum += x * x;
            }
            study.Tell(t, sum);
        }
        return study;
    }

    private static StudyDirection Dir(string s) => s == "maximize" ? StudyDirection.Maximize : StudyDirection.Minimize;

    [TestMethod]
    [TestCategory("RequiresPython")]
    [DataRow("minimize", 25, 2)]
    [DataRow("maximize", 25, 2)]
    [DataRow("minimize", 30, 3)]
    public void RegretBound_Matches_LiveOptuna(string direction, int nTrials, int dim)
    {
        PythonSession.EnsureOrInconclusive();

        double csharp = new RegretBoundEvaluator(seed: 0)
            .Evaluate(BuildStudy(Dir(direction), nTrials, dim).Trials, Dir(direction));

        PyExec(
            PyBuildStudy +
            "def _oss_regret(direction, n_trials, dim):\n" +
            "    study = _oss_build(direction, n_trials, dim)\n" +
            "    with warnings.catch_warnings():\n" +
            "        warnings.simplefilter('ignore')\n" +
            "        ev = optuna.terminator.RegretBoundEvaluator(seed=0)\n" +
            "        return float(ev.evaluate(study.trials, study.direction))\n" +
            $"result = _oss_regret('{direction}', {nTrials}, {dim})\n");
        double python = PyFloat("result");

        AssertClose(csharp, python, "regret bound");
    }

    [TestMethod]
    [TestCategory("RequiresPython")]
    [DataRow("minimize", 8, 2)]
    [DataRow("maximize", 8, 2)]
    [DataRow("minimize", 12, 3)]
    public void Emmr_Matches_LiveOptuna(string direction, int nTrials, int dim)
    {
        PythonSession.EnsureOrInconclusive();

        double csharp = new EMMREvaluator(seed: 0)
            .Evaluate(BuildStudy(Dir(direction), nTrials, dim).Trials, Dir(direction));

        PyExec(
            PyBuildStudy +
            "def _oss_emmr(direction, n_trials, dim):\n" +
            "    study = _oss_build(direction, n_trials, dim)\n" +
            "    with warnings.catch_warnings():\n" +
            "        warnings.simplefilter('ignore')\n" +
            "        ev = optuna.terminator.EMMREvaluator(seed=0)\n" +
            "        return float(ev.evaluate(study.trials, study.direction))\n" +
            $"result = _oss_emmr('{direction}', {nTrials}, {dim})\n");
        double python = PyFloat("result");

        AssertClose(csharp, python, "EMMR criterion");
    }

    [TestMethod]
    [TestCategory("RequiresPython")]
    [DataRow("minimize", 10, 2)]
    public void ImprovementInfo_Matches_LiveOptuna(string direction, int nTrials, int dim)
    {
        PythonSession.EnsureOrInconclusive();

        // The seeded RegretBoundEvaluator is reused across every completed prefix, so its internal
        // RNG advances in lock-step on both sides — this validates the full _get_improvement_info walk.
        var info = OptunaSharp.Visualization.TerminatorImprovement.GetInfo(
            BuildStudy(Dir(direction), nTrials, dim),
            getError: false,
            improvementEvaluator: new RegretBoundEvaluator(seed: 0));

        PyExec(
            PyBuildStudy +
            "import optuna.visualization._terminator_improvement as _oss_tvi\n" +
            "def _oss_info(direction, n_trials, dim):\n" +
            "    study = _oss_build(direction, n_trials, dim)\n" +
            "    with warnings.catch_warnings():\n" +
            "        warnings.simplefilter('ignore')\n" +
            "        ev = optuna.terminator.RegretBoundEvaluator(seed=0)\n" +
            "        info = _oss_tvi._get_improvement_info(study, False, ev, None)\n" +
            "    return info\n" +
            $"_oss_i = _oss_info('{direction}', {nTrials}, {dim})\n" +
            "result_nums = ','.join(str(n) for n in _oss_i.trial_numbers)\n" +
            "result_imps = ','.join(repr(float(x)) for x in _oss_i.improvements)\n" +
            "result_err = 'none' if _oss_i.errors is None else 'list'\n");

        string pyNums = PyStr("result_nums");
        string pyImps = PyStr("result_imps");
        string pyErr = PyStr("result_err");

        string.Join(",", info.TrialNumbers).Should().Be(pyNums, "the trial-number sequence must match");
        pyErr.Should().Be("none");
        info.Errors.Should().BeNull();

        double[] python = pyImps.Split(',')
            .Select(s => double.Parse(s, CultureInfo.InvariantCulture)).ToArray();
        info.Improvements.Should().HaveCount(python.Length);
        for (int i = 0; i < python.Length; i++)
            AssertClose(info.Improvements[i], python[i], $"improvement[{i}]");
    }

    // Tolerance comparison: relative to the Python magnitude, with a small absolute floor for values
    // near zero (a degenerate regret bound can be ~1e-10). The GP boundary is documented above. In
    // practice these criteria agree to a few 1e-3 relative (the fit's ~1e-3 hyperparameter drift
    // propagated through the posterior); the 3e-2 relative bound leaves ~13x headroom.
    private static void AssertClose(double csharp, double python, string what)
    {
        double tol = 3e-2 * System.Math.Abs(python) + 1e-6;
        System.Math.Abs(csharp - python).Should().BeLessThanOrEqualTo(
            tol,
            $"{what} must match live Optuna within the GP tolerance " +
            $"(C#={csharp.ToString("R", CultureInfo.InvariantCulture)}, " +
            $"Py={python.ToString("R", CultureInfo.InvariantCulture)})");
    }
}
