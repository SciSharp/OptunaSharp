using System;
using System.Linq;
using AwesomeAssertions;
using NumSharp;
using OptunaSharp.Importance.Shap;
using OptunaSharp.Samplers;
using Python.Runtime;

namespace OptunaSharp.Tests.Interop;

/// <summary>
///     Live parity for the SHAP importance evaluator vs Optuna's own
///     <c>optuna_integration.shap.ShapleyImportanceEvaluator</c> (which drives the third-party
///     <c>shap</c> + <c>scikit-learn</c> packages). With a seeded <see cref="RandomSampler" /> both
///     sides observe <b>bit-identical trials</b> (MT19937), so both evaluators fit the identical
///     (X, y). OptunaSharp's port is pure-managed — Core's bit-exact <c>RandomForestRegressor</c>
///     surrogate plus a faithful port of <c>shap/cext/tree_shap.h</c> (tree_path_dependent) — so the
///     importances match Optuna <b>value-for-value</b>, ordering included (asserted at 1e-12; the
///     observed residual against SHAP 0.52.0 on this host is 0 ULP).
///
///     <para>Value parity is host-pinned to x64 (like the fANOVA/MDI native parity): the surrogate's
///     float32 forest accumulation isn't arm64-portable. The tests self-skip (Inconclusive) when
///     <c>shap</c> / <c>scikit-learn</c> / <c>optuna_integration</c> are not importable.</para>
/// </summary>
[TestClass]
public class ShapleyImportanceParityTests : OptunaInteropTestBase
{
    // The optuna_integration.shap test objective (mixed float/int-log/log/categorical — the categorical
    // exercises the one-hot column scatter, np.add.at over encoded_column_to_column).
    private const string PyObjective =
        "def _oss_shap_obj(t):\n" +
        "    x1 = t.suggest_float('x1', 0.1, 3)\n" +
        "    x2 = t.suggest_float('x2', 0.1, 3, log=True)\n" +
        "    x3 = t.suggest_int('x3', 2, 4, log=True)\n" +
        "    x4 = t.suggest_categorical('x4', [0.1, 1.0, 10.0])\n" +
        "    return x1 + x2 * x3 + x4\n";

    private static double CsObjective(Trial t)
    {
        double x1 = t.SuggestFloat("x1", 0.1, 3);
        double x2 = t.SuggestFloat("x2", 0.1, 3, log: true);
        long x3 = t.SuggestInt("x3", 2, 4, log: true);
        double x4 = t.SuggestCategorical("x4", new[] { 0.1, 1.0, 10.0 });
        return x1 + x2 * x3 + x4;
    }

    private const string PyMultiObjective =
        "def _oss_shap_mobj(t):\n" +
        "    x1 = t.suggest_float('x1', 0.1, 3)\n" +
        "    x2 = t.suggest_float('x2', 0.1, 3, log=True)\n" +
        "    x3 = t.suggest_int('x3', 2, 4, log=True)\n" +
        "    x4 = t.suggest_categorical('x4', [0.1, 1.0, 10.0])\n" +
        "    return (x1 + x2 * x3 + x4, x1 * x4)\n";

    private static double[] CsMultiObjective(Trial t)
    {
        double x1 = t.SuggestFloat("x1", 0.1, 3);
        double x2 = t.SuggestFloat("x2", 0.1, 3, log: true);
        long x3 = t.SuggestInt("x3", 2, 4, log: true);
        double x4 = t.SuggestCategorical("x4", new[] { 0.1, 1.0, 10.0 });
        return new[] { x1 + x2 * x3 + x4, x1 * x4 };
    }

    // Python source of a warning-suppressed evaluate() (the class is @experimental_class).
    private const string PyEvalHelper =
        "import warnings\n" +
        "from optuna_integration.shap import ShapleyImportanceEvaluator as _OssShap\n" +
        "def _oss_shap_eval(study, n_trees, max_depth, seed, target=None):\n" +
        "    with warnings.catch_warnings():\n" +
        "        warnings.simplefilter('ignore')\n" +
        "        ev = _OssShap(n_trees=n_trees, max_depth=max_depth, seed=seed)\n" +
        "    return ev.evaluate(study, target=target)\n";

    // The Tree SHAP walk is a faithful port; the observed residual vs SHAP 0.52.0 is 0 ULP on x64.
    // 1e-12 keeps a genuine value-parity assertion (not ranking) with headroom for library drift.
    private const double ValueTolerance = 1e-12;

    private void RequireShapOrInconclusive()
    {
        using (Gil())
        {
            try
            {
                Scope.Exec("import shap, sklearn\nfrom optuna_integration.shap import ShapleyImportanceEvaluator\n");
            }
            catch (PythonException)
            {
                Assert.Inconclusive(
                    "shap / scikit-learn / optuna_integration are not available in the Python env; " +
                    "skipping SHAP-importance parity (Optuna's ShapleyImportanceEvaluator requires them).");
            }
        }
    }

    [TestMethod]
    [TestCategory("RequiresPython")]
    [DataRow(0, 30)]
    [DataRow(7, 60)]
    [DataRow(42, 120)]
    public void ShapleyImportance_MatchesLiveOptunaValues(int seed, int nTrials)
    {
        RunValueParity(seed, nTrials, nTrees: 64, maxDepth: 64);
    }

    [TestMethod]
    [TestCategory("RequiresPython")]
    [DataRow(10, 64)]   // non-power-of-2 tree count exercises values *= (1.0 / n_trees)
    [DataRow(20, 2)]    // shallow trees
    [DataRow(37, 9)]    // odd count + medium depth
    public void ShapleyImportance_VariedForestConfig_MatchesLiveOptuna(int nTrees, int maxDepth)
    {
        RunValueParity(seed: 3, nTrials: 80, nTrees: nTrees, maxDepth: maxDepth);
    }

    private void RunValueParity(int seed, int nTrials, int nTrees, int maxDepth)
    {
        // Value parity is host-pinned to x64: the float32 forest accumulation isn't arm64-portable.
        PythonSession.EnsureBitExactHostOrInconclusive();
        RequireShapOrInconclusive();

        var csStudy = Optuna.CreateStudy(sampler: new RandomSampler(seed));
        csStudy.Optimize(CsObjective, nTrials: nTrials);
        var csImp = new ShapleyImportanceEvaluator(nTrees: nTrees, maxDepth: maxDepth, seed: 0).Evaluate(csStudy);

        PyExec(
            PyObjective + PyEvalHelper +
            $"study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed={seed}))\n" +
            $"study.optimize(_oss_shap_obj, n_trials={nTrials})\n" +
            $"imp = _oss_shap_eval(study, {nTrees}, {maxDepth}, 0)\n" +
            "imp_keys = ','.join(imp.keys())\n");

        // Descending-importance ordering must match exactly (identical forest, identical SHAP, identical sort).
        string.Join(",", csImp.Keys).Should().Be(PyStr("imp_keys"),
            $"the importance ordering must match live Optuna (seed {seed}, n_trees {nTrees}, max_depth {maxDepth})");

        foreach (var (name, csValue) in csImp)
            csValue.Should().BeApproximately(PyFloat($"imp['{name}']"), ValueTolerance,
                $"importance of {name} must match live Optuna (seed {seed}, n_trees {nTrees}, max_depth {maxDepth})");
    }

    [TestMethod]
    [TestCategory("RequiresPython")]
    [DataRow(0, 60)]
    [DataRow(13, 90)]
    public void ShapleyImportance_MultiObjectiveWithTarget_MatchesLiveOptuna(int seed, int nTrials)
    {
        PythonSession.EnsureBitExactHostOrInconclusive();
        RequireShapOrInconclusive();

        var csStudy = Optuna.CreateStudy(
            sampler: new RandomSampler(seed),
            directions: new[] { StudyDirection.Minimize, StudyDirection.Minimize });
        csStudy.Optimize(CsMultiObjective, nTrials: nTrials);
        var csImp = new ShapleyImportanceEvaluator(seed: 0).Evaluate(csStudy, target: t => t.Values![0]);

        PyExec(
            PyMultiObjective + PyEvalHelper +
            $"study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed={seed}), directions=['minimize','minimize'])\n" +
            $"study.optimize(_oss_shap_mobj, n_trials={nTrials})\n" +
            "imp = _oss_shap_eval(study, 64, 64, 0, target=lambda t: t.values[0])\n" +
            "imp_keys = ','.join(imp.keys())\n");

        string.Join(",", csImp.Keys).Should().Be(PyStr("imp_keys"),
            $"multi-objective (target=values[0]) importance ordering must match live Optuna (seed {seed})");
        foreach (var (name, csValue) in csImp)
            csValue.Should().BeApproximately(PyFloat($"imp['{name}']"), ValueTolerance,
                $"multi-objective importance of {name} must match live Optuna (seed {seed})");
    }

    [TestMethod]
    [TestCategory("RequiresPython")]
    public void ShapleyImportance_MultiObjectiveWithoutTarget_RaisesLikeOptuna()
    {
        RequireShapOrInconclusive();

        // C#: no target on a multi-objective study raises ValueError (same message as Optuna).
        var csStudy = Optuna.CreateStudy(
            sampler: new RandomSampler(0),
            directions: new[] { StudyDirection.Minimize, StudyDirection.Minimize });
        csStudy.Optimize(CsMultiObjective, nTrials: 10);
        Action act = () => new ShapleyImportanceEvaluator(seed: 0).Evaluate(csStudy);
        act.Should().Throw<ValueError>().WithMessage("If the `study` is being used for multi-objective*");

        // Python raises ValueError from the same guard.
        PyExec(
            PyMultiObjective + PyEvalHelper +
            "study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=0), directions=['minimize','minimize'])\n" +
            "study.optimize(_oss_shap_mobj, n_trials=10)\n" +
            "raised = False\n" +
            "try:\n" +
            "    _oss_shap_eval(study, 64, 64, 0)\n" +
            "except ValueError:\n" +
            "    raised = True\n");
        PyBool("raised").Should().BeTrue("live Optuna must also raise ValueError for multi-objective without target");
    }
}
