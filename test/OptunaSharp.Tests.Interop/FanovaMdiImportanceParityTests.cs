using System.Linq;
using AwesomeAssertions;
using OptunaSharp.Distributions;
using OptunaSharp.Importance;
using OptunaSharp.Importance.Forests;
using OptunaSharp.Samplers;
using Python.Runtime;

namespace OptunaSharp.Tests.Interop;

/// <summary>
///     Live parity for Phase 5.1 — the forest-based importance evaluators vs Optuna's own
///     <c>optuna.importance.FanovaImportanceEvaluator</c> / <c>MeanDecreaseImpurityImportanceEvaluator</c>.
///
///     <para>With a seeded <see cref="RandomSampler" /> both sides observe <b>bit-identical trials</b>
///     (MT19937), so both evaluators see the identical (X, y). Two backends, two assertion strengths:</para>
///     <list type="bullet">
///       <item>The <b>native</b> <see cref="FanovaImportanceEvaluator" /> /
///         <see cref="MeanDecreaseImpurityImportanceEvaluator" /> (Core) fit a pure-managed,
///         scikit-learn-bit-exact CART, so their importances match Optuna's <b>value-for-value</b>
///         (asserted at 1e-12 — the actual residual is ~1 ULP from numpy's pairwise float reductions
///         in the fANOVA marginal-variance aggregation), ordering included.</item>
///       <item>The <b>FastForest</b> backend (ML.NET, satellite) is not tree-identical to
///         scikit-learn, so only the <b>ranking</b> is asserted: the dominant parameter matches
///         Optuna's on a well-separated problem.</item>
///     </list>
///
///     <para>Optuna's forest evaluators require scikit-learn; the tests self-skip (Inconclusive) when
///     it is not importable.</para>
/// </summary>
[TestClass]
public class FanovaMdiImportanceParityTests : OptunaInteropTestBase
{
    private const string PyObjective =
        "def _oss_fmi_obj(t):\n" +
        "    x1 = t.suggest_float('x1', -5, 5)\n" +
        "    x2 = t.suggest_int('x2', 0, 20)\n" +
        "    x3 = t.suggest_float('x3', 1e-3, 1e3, log=True)\n" +
        "    x4 = t.suggest_categorical('x4', ['a', 'b', 'c'])\n" +
        "    return x1 * x1 + 0.5 * x2 + x3 / 1000.0 + (1.0 if x4 == 'a' else 0.0)\n";

    private static double CsObjective(Trial t)
    {
        double x1 = t.SuggestFloat("x1", -5, 5);
        long x2 = t.SuggestInt("x2", 0, 20);
        double x3 = t.SuggestFloat("x3", 1e-3, 1e3, log: true);
        string? x4 = t.SuggestCategorical("x4", new CategoricalChoice[] { "a", "b", "c" }).Value as string;
        return x1 * x1 + 0.5 * x2 + x3 / 1000.0 + (x4 == "a" ? 1.0 : 0.0);
    }

    // The fANOVA/MDI forest itself is bit-exact vs sklearn; the only residual vs Optuna is numpy's
    // pairwise float-reduction order in the marginal-variance aggregation (~1 ULP). 1e-12 gives ~1e4x
    // headroom while still being a genuine value-parity assertion (not just ranking).
    private const double ValueTolerance = 1e-12;

    private void RequireSklearnOrInconclusive()
    {
        using (Gil())
        {
            try { Scope.Exec("import sklearn\n"); }
            catch (PythonException)
            {
                Assert.Inconclusive("scikit-learn is not available in the Python env; " +
                                    "skipping forest-importance parity (Optuna's fANOVA/MDI require it).");
            }
        }
    }

    [TestMethod]
    [TestCategory("RequiresPython")]
    [DataRow(0, 60)]
    [DataRow(7, 80)]
    [DataRow(42, 120)]
    public void NativeFanova_MatchesLiveOptunaValues(int seed, int nTrials)
    {
        RunValueParity(seed, nTrials, "FanovaImportanceEvaluator");
    }

    [TestMethod]
    [TestCategory("RequiresPython")]
    [DataRow(0, 60)]
    [DataRow(7, 80)]
    [DataRow(42, 120)]
    public void NativeMdi_MatchesLiveOptunaValues(int seed, int nTrials)
    {
        RunValueParity(seed, nTrials, "MeanDecreaseImpurityImportanceEvaluator");
    }

    private void RunValueParity(int seed, int nTrials, string pyEvaluator)
    {
        PythonSession.EnsureOrInconclusive();
        RequireSklearnOrInconclusive();

        var csStudy = Optuna.CreateStudy(sampler: new RandomSampler(seed));
        csStudy.Optimize(CsObjective, nTrials: nTrials);
        var csImp = pyEvaluator.StartsWith("Fanova")
            ? new FanovaImportanceEvaluator(seed: 0).Evaluate(csStudy)
            : new MeanDecreaseImpurityImportanceEvaluator(seed: 0).Evaluate(csStudy);

        PyExec(
            PyObjective +
            $"study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed={seed}))\n" +
            $"study.optimize(_oss_fmi_obj, n_trials={nTrials})\n" +
            $"imp = optuna.importance.{pyEvaluator}(seed=0).evaluate(study)\n" +
            "imp_keys = ','.join(imp.keys())\n");

        // Descending-importance ordering must match exactly (identical forest, identical sort).
        string.Join(",", csImp.Keys).Should().Be(PyStr("imp_keys"),
            $"the importance ordering must match live Optuna ({pyEvaluator}, seed {seed})");

        foreach (var (name, csValue) in csImp)
            csValue.Should().BeApproximately(PyFloat($"imp['{name}']"), ValueTolerance,
                $"importance of {name} must match live Optuna ({pyEvaluator}, seed {seed})");
    }

    [TestMethod]
    [TestCategory("RequiresPython")]
    [DataRow(0, 80)]
    [DataRow(7, 100)]
    public void FastForestFanova_RankingAgreesWithLiveOptuna(int seed, int nTrials)
    {
        RunRankingParity(seed, nTrials, fanova: true);
    }

    [TestMethod]
    [TestCategory("RequiresPython")]
    [DataRow(0, 80)]
    [DataRow(7, 100)]
    public void FastForestMdi_RankingAgreesWithLiveOptuna(int seed, int nTrials)
    {
        RunRankingParity(seed, nTrials, fanova: false);
    }

    private void RunRankingParity(int seed, int nTrials, bool fanova)
    {
        PythonSession.EnsureOrInconclusive();
        RequireSklearnOrInconclusive();

        var csStudy = Optuna.CreateStudy(sampler: new RandomSampler(seed));
        csStudy.Optimize(CsObjective, nTrials: nTrials);
        var csImp = ImportanceFunctions.GetParamImportances(csStudy, fanova
            ? new FastForestFanovaImportanceEvaluator(seed: 0)
            : new FastForestMeanDecreaseImpurityImportanceEvaluator(seed: 0));

        string pyEvaluator = fanova ? "FanovaImportanceEvaluator" : "MeanDecreaseImpurityImportanceEvaluator";
        PyExec(
            PyObjective +
            $"study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed={seed}))\n" +
            $"study.optimize(_oss_fmi_obj, n_trials={nTrials})\n" +
            $"imp = optuna.importance.get_param_importances(study, evaluator=optuna.importance.{pyEvaluator}(seed=0))\n" +
            "imp_top = list(imp.keys())[0]\n");

        // FastForest importances diverge in value from sklearn, but on this well-separated problem the
        // dominant parameter (x1, via x1*x1) must be ranked first on both sides.
        csImp.Keys.First().Should().Be("x1", $"FastForest must rank x1 first (seed {seed})");
        PyStr("imp_top").Should().Be("x1", $"live Optuna must also rank x1 first (seed {seed})");
        csImp.Values.Sum().Should().BeApproximately(1.0, 1e-9);
    }
}
