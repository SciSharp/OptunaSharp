using System.Linq;
using AwesomeAssertions;
using OptunaSharp.Importance;
using OptunaSharp.Samplers;
using Python.Runtime;

namespace OptunaSharp.Tests.Interop;

/// <summary>
///     Live parity for Phase 5.1 — the forest-based importance evaluators
///     (<see cref="FanovaImportanceEvaluator" /> + <see cref="MeanDecreaseImpurityImportanceEvaluator" />)
///     vs Optuna's own <c>optuna.importance.FanovaImportanceEvaluator</c> /
///     <c>MeanDecreaseImpurityImportanceEvaluator</c>.
///
///     <para>With a seeded <see cref="RandomSampler" /> both sides observe <b>bit-identical trials</b>
///     (MT19937), so the two evaluators see the identical (X, y). Unlike PED-ANOVA these importances
///     are <b>not</b> asserted equal: OptunaSharp fits the random forest with ML.NET's FastForest and
///     Optuna with scikit-learn's <c>RandomForestRegressor</c>, which are not tree-identical (the
///     plan's Tier X <c>tolerance oracle</c> — see the OptunaSharp.Importance.Forests README). The
///     robust, meaningful parity is <b>ranking agreement on a well-separated problem</b>: the
///     objective <c>10·x² + 0.5·y + 0.01·z</c> makes <c>x</c> dominate the prediction variance, so
///     both forests must rank <c>x</c> first with a large share of the importance.</para>
///
///     <para>Optuna's forest evaluators require scikit-learn; when it is not importable in the host
///     Python the tests self-skip (Inconclusive) rather than fail.</para>
/// </summary>
[TestClass]
public class FanovaMdiImportanceParityTests : OptunaInteropTestBase
{
    private const string PyObjective =
        "def _oss_fmi_obj(t):\n" +
        "    x = t.suggest_float('x', -10, 10)\n" +
        "    y = t.suggest_float('y', -10, 10)\n" +
        "    z = t.suggest_int('z', 0, 10)\n" +
        "    return 10.0 * x * x + 0.5 * y + 0.01 * z\n";

    private static double CsObjective(Trial t)
    {
        double x = t.SuggestFloat("x", -10, 10);
        double y = t.SuggestFloat("y", -10, 10);
        long z = t.SuggestInt("z", 0, 10);
        return 10.0 * x * x + 0.5 * y + 0.01 * z;
    }

    /// <summary>Both forests must identify <c>x</c> as the dominant parameter with a large share.</summary>
    private const double DominantThreshold = 0.7;

    private void RequireSklearnOrInconclusive()
    {
        using (Gil())
        {
            try
            {
                Scope.Exec("import sklearn\n");
            }
            catch (PythonException)
            {
                Assert.Inconclusive("scikit-learn is not available in the Python env; " +
                                    "skipping the forest-importance parity (Optuna's fANOVA/MDI require it).");
            }
        }
    }

    [TestMethod]
    [TestCategory("RequiresPython")]
    [DataRow(0, 80)]
    [DataRow(7, 100)]
    public void Fanova_RankingParity_MatchesLiveOptuna(int seed, int nTrials)
    {
        PythonSession.EnsureOrInconclusive();
        RequireSklearnOrInconclusive();

        var csStudy = Optuna.CreateStudy(sampler: new RandomSampler(seed));
        csStudy.Optimize(CsObjective, nTrials: nTrials);
        var csImp = ImportanceFunctions.GetParamImportances(
            csStudy, evaluator: new FanovaImportanceEvaluator(seed: 0));

        PyExec(
            PyObjective +
            $"study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed={seed}))\n" +
            $"study.optimize(_oss_fmi_obj, n_trials={nTrials})\n" +
            "imp = optuna.importance.get_param_importances(" +
            "study, evaluator=optuna.importance.FanovaImportanceEvaluator(seed=0))\n" +
            "imp_top = list(imp.keys())[0]\n");

        AssertRankingParity(csImp, seed);
    }

    [TestMethod]
    [TestCategory("RequiresPython")]
    [DataRow(0, 80)]
    [DataRow(7, 100)]
    public void Mdi_RankingParity_MatchesLiveOptuna(int seed, int nTrials)
    {
        PythonSession.EnsureOrInconclusive();
        RequireSklearnOrInconclusive();

        var csStudy = Optuna.CreateStudy(sampler: new RandomSampler(seed));
        csStudy.Optimize(CsObjective, nTrials: nTrials);
        var csImp = ImportanceFunctions.GetParamImportances(
            csStudy, evaluator: new MeanDecreaseImpurityImportanceEvaluator(seed: 0));

        PyExec(
            PyObjective +
            $"study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed={seed}))\n" +
            $"study.optimize(_oss_fmi_obj, n_trials={nTrials})\n" +
            "imp = optuna.importance.get_param_importances(" +
            "study, evaluator=optuna.importance.MeanDecreaseImpurityImportanceEvaluator(seed=0))\n" +
            "imp_top = list(imp.keys())[0]\n");

        AssertRankingParity(csImp, seed);
    }

    private void AssertRankingParity(OptunaSharp.Collections.OrderedDict<string, double> csImp, int seed)
    {
        // Both sides assess the same set of parameters.
        csImp.Keys.OrderBy(k => k).Should().Equal("x", "y", "z");

        // The dominant parameter is x on both sides.
        csImp.Keys.First().Should().Be("x", $"x dominates the objective (seed {seed})");
        PyStr("imp_top").Should().Be("x", $"live Optuna must also rank x first (seed {seed})");

        // ...with a large importance share on both sides (the tolerance oracle).
        csImp["x"].Should().BeGreaterThan(DominantThreshold, $"OptunaSharp: x should dominate (seed {seed})");
        PyFloat("imp['x']").Should().BeGreaterThan(DominantThreshold, $"live Optuna: x should dominate (seed {seed})");

        // Both sides are normalized to sum 1.
        csImp.Values.Sum().Should().BeApproximately(1.0, 1e-9);
    }
}
