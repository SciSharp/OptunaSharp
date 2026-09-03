using System;
using System.Collections.Generic;
using System.Linq;
using AwesomeAssertions;
using OptunaSharp.Distributions;
using OptunaSharp.Importance;
using OptunaSharp.Importance.PedAnova;
using OptunaSharp.Samplers;

namespace OptunaSharp.Tests.Interop;

/// <summary>
///     Live parity for Phase 2.7 — <c>get_param_importances</c> with the default
///     <see cref="PedAnovaImportanceEvaluator" />. With a seeded <see cref="RandomSampler" /> both
///     sides produce bit-identical trials (MT19937), so the derived importances must agree with
///     Optuna's. They agree to a few ULP rather than bit-for-bit: the numerical Parzen density
///     routes through <c>erf</c>, and Optuna's <c>math.erf</c> is the platform libm (ucrt), which no
///     managed <c>erf</c> reproduces exactly (documented on <c>TruncNorm</c>). The categorical path,
///     the top-quantile trial selection, and the descending importance ordering are exact.
/// </summary>
[TestClass]
public class ImportanceParityTests : OptunaInteropTestBase
{
    // Generous headroom for the erf/ucrt divergence; still far tighter than any real defect.
    private const double Tolerance = 1e-10;

    private const string PyObjective =
        "def _oss_obj(t):\n" +
        "    x1 = t.suggest_float('x1', -5, 5)\n" +
        "    x2 = t.suggest_float('x2', 1e-3, 1e3, log=True)\n" +
        "    x3 = t.suggest_int('x3', 0, 20)\n" +
        "    x5 = t.suggest_int('x5', 1, 256, log=True)\n" +
        "    x4 = t.suggest_categorical('x4', ['a', 'b', 'c'])\n" +
        "    return x1 * x1 + x2 / 1000 + x3 + x5 / 100 + (1 if x4 == 'a' else 0)\n";

    private static double CsObjective(Trial t)
    {
        double x1 = t.SuggestFloat("x1", -5, 5);
        double x2 = t.SuggestFloat("x2", 1e-3, 1e3, log: true);
        long x3 = t.SuggestInt("x3", 0, 20);
        long x5 = t.SuggestInt("x5", 1, 256, log: true);
        string? x4 = t.SuggestCategorical("x4", new CategoricalChoice[] { "a", "b", "c" }).Value as string;
        return x1 * x1 + x2 / 1000.0 + x3 + x5 / 100.0 + (x4 == "a" ? 1.0 : 0.0);
    }

    [TestMethod]
    [TestCategory("RequiresPython")]
    [DataRow(0, 60)]
    [DataRow(7, 80)]
    [DataRow(2026, 120)]
    public void GetParamImportances_Matches_LiveOptuna(int seed, int nTrials)
    {
        PythonSession.EnsureOrInconclusive();

        var csStudy = Optuna.CreateStudy(sampler: new RandomSampler(seed));
        csStudy.Optimize(CsObjective, nTrials: nTrials);
        var csImp = ImportanceFunctions.GetParamImportances(csStudy);

        PyExec(
            PyObjective +
            $"study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed={seed}))\n" +
            $"study.optimize(_oss_obj, n_trials={nTrials})\n" +
            "imp = optuna.importance.get_param_importances(study)\n" +
            "imp_keys = ','.join(imp.keys())\n");

        // Descending importance ordering must match exactly (the sort is bit-exact).
        string.Join(",", csImp.Keys).Should().Be(PyStr("imp_keys"),
            $"the importance ordering must match live Optuna (seed {seed})");

        foreach (var (name, csValue) in csImp)
        {
            double pyValue = PyFloat($"imp['{name}']");
            csValue.Should().BeApproximately(pyValue, Tolerance,
                $"importance of {name} must match live Optuna (seed {seed})");
        }

        // The sum of the normalized importances is 1.0 on both sides.
        csImp.Values.Sum().Should().BeApproximately(1.0, 1e-9);
    }

    [TestMethod]
    [TestCategory("RequiresPython")]
    [DataRow(11)]
    [DataRow(99)]
    public void GetParamImportances_ParamsSubset_Matches_LiveOptuna(int seed)
    {
        PythonSession.EnsureOrInconclusive();

        var csStudy = Optuna.CreateStudy(sampler: new RandomSampler(seed));
        csStudy.Optimize(CsObjective, nTrials: 80);
        var csImp = ImportanceFunctions.GetParamImportances(
            csStudy, @params: new[] { "x1", "x3", "x4" });

        PyExec(
            PyObjective +
            $"study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed={seed}))\n" +
            "study.optimize(_oss_obj, n_trials=80)\n" +
            "imp = optuna.importance.get_param_importances(study, params=['x1', 'x3', 'x4'])\n" +
            "imp_keys = ','.join(imp.keys())\n");

        string.Join(",", csImp.Keys).Should().Be(PyStr("imp_keys"));
        foreach (var (name, csValue) in csImp)
            csValue.Should().BeApproximately(PyFloat($"imp['{name}']"), Tolerance);
    }

    [TestMethod]
    [TestCategory("RequiresPython")]
    [DataRow(3)]
    [DataRow(2027)]
    public void GetParamImportances_MultiObjectiveWithTarget_Matches_LiveOptuna(int seed)
    {
        PythonSession.EnsureOrInconclusive();

        var csStudy = Optuna.CreateStudy(
            directions: new[] { StudyDirection.Minimize, StudyDirection.Minimize },
            sampler: new RandomSampler(seed));
        csStudy.Optimize((Func<Trial, IReadOnlyList<double>>)(t =>
        {
            double a = t.SuggestFloat("a", -3, 3);
            long b = t.SuggestInt("b", 0, 10);
            return new[] { a * a + b, (a - 1) * (a - 1) - b };
        }), nTrials: 60);

        // target selects a single objective, so the QuantileFilter path (not HSSP) is exercised.
        var csImp = ImportanceFunctions.GetParamImportances(csStudy, target: t => t.Values![0]);

        PyExec(
            "def _oss_mo(t):\n" +
            "    a = t.suggest_float('a', -3, 3)\n" +
            "    b = t.suggest_int('b', 0, 10)\n" +
            "    return a * a + b, (a - 1) * (a - 1) - b\n" +
            "study = optuna.create_study(directions=['minimize', 'minimize'], " +
            $"sampler=optuna.samplers.RandomSampler(seed={seed}))\n" +
            "study.optimize(_oss_mo, n_trials=60)\n" +
            "imp = optuna.importance.get_param_importances(study, target=lambda t: t.values[0])\n" +
            "imp_keys = ','.join(imp.keys())\n");

        string.Join(",", csImp.Keys).Should().Be(PyStr("imp_keys"));
        foreach (var (name, csValue) in csImp)
            csValue.Should().BeApproximately(PyFloat($"imp['{name}']"), Tolerance);
    }

    [TestMethod]
    [TestCategory("RequiresPython")]
    [DataRow(3)]
    [DataRow(2027)]
    public void GetParamImportances_MultiObjectiveTargetNull_Matches_LiveOptuna(int seed)
    {
        PythonSession.EnsureOrInconclusive();

        // target=null on a multi-objective study routes through _split_complete_trials_multi_objective,
        // i.e. the non-domination-rank split with the HSSP tie-break (Phase 2.10 — previously threw).
        var csStudy = Optuna.CreateStudy(
            directions: new[] { StudyDirection.Minimize, StudyDirection.Minimize },
            sampler: new RandomSampler(seed));
        csStudy.Optimize((Func<Trial, IReadOnlyList<double>>)(t =>
        {
            double a = t.SuggestFloat("a", -3, 3);
            long b = t.SuggestInt("b", 0, 10);
            return new[] { a * a + b, (a - 1) * (a - 1) - b };
        }), nTrials: 60);

        var csImp = ImportanceFunctions.GetParamImportances(csStudy);  // target=null → HSSP path.

        PyExec(
            "def _oss_mo_null(t):\n" +
            "    a = t.suggest_float('a', -3, 3)\n" +
            "    b = t.suggest_int('b', 0, 10)\n" +
            "    return a * a + b, (a - 1) * (a - 1) - b\n" +
            "study = optuna.create_study(directions=['minimize', 'minimize'], " +
            $"sampler=optuna.samplers.RandomSampler(seed={seed}))\n" +
            "study.optimize(_oss_mo_null, n_trials=60)\n" +
            "imp = optuna.importance.get_param_importances(study)\n" +
            "imp_keys = ','.join(imp.keys())\n");

        string.Join(",", csImp.Keys).Should().Be(PyStr("imp_keys"));
        foreach (var (name, csValue) in csImp)
            csValue.Should().BeApproximately(PyFloat($"imp['{name}']"), Tolerance);
    }
}
