using OptunaSharp.Importance;
using OptunaSharp.Samplers;
using OptunaSharp.Terminator;
using OptunaSharp.Visualization.Plotly;
using Plotly.NET;

namespace OptunaSharp.Tests.Interop;

/// <summary>
///     Live render parity for the 🟡 batch — <c>plot_param_importances</c> and
///     <c>plot_terminator_improvement</c>. param-importances rides the native fANOVA (seeded, ~1 ULP);
///     terminator rides the GP improvement evaluator (the documented ~1e-3 fit drift, asserted at 3e-2).
/// </summary>
[TestClass]
public class PlotlyYellowRenderParityTests : PlotlyRenderParityBase
{
    // ---- param importances ----------------------------------------------------------------------

    [TestMethod]
    [TestCategory("RequiresPython")]
    [DataRow(0, 30)]
    [DataRow(4, 40)]
    public void ParamImportances_Render_Matches_LiveOptuna(int seed, int nTrials)
    {
        PythonSession.EnsureBitExactHostOrInconclusive();

        // Default plot_param_importances uses fANOVA with seed=None (non-deterministic) — pin the seed
        // on both sides. fANOVA is value-parity ~1 ULP, so tolerate the bar heights; the raw-float
        // hovertemplate can't absorb a 1-ULP drift, so it is excluded (its number is the bar x anyway).
        var csStudy = Optuna.CreateStudy(sampler: new RandomSampler(seed));
        csStudy.Optimize(t =>
        {
            double x = t.SuggestFloat("x", -10, 10);
            double y = t.SuggestFloat("y", -10, 10);
            long z = t.SuggestInt("z", 0, 10);
            return (x - 2.0) * (x - 2.0) + (y + 3.0) * (y + 3.0) + z;
        }, nTrials: nTrials);
        string csJson = GenericChart.toJson(
            PlotlyVisualization.PlotParamImportances(csStudy, new FanovaImportanceEvaluator(seed: 0)));

        AssertRenderParity(csJson,
            "def _oss_obj_pi(t):\n" +
            "    x = t.suggest_float('x', -10, 10)\n" +
            "    y = t.suggest_float('y', -10, 10)\n" +
            "    z = t.suggest_int('z', 0, 10)\n" +
            "    return (x - 2.0) * (x - 2.0) + (y + 3.0) * (y + 3.0) + z\n" +
            $"study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed={seed}))\n" +
            $"study.optimize(_oss_obj_pi, n_trials={nTrials})\n",
            "optuna.visualization.plot_param_importances(study, evaluator=optuna.importance.FanovaImportanceEvaluator(seed=0))",
            tol: 1e-10,
            ignoreKeys: new[] { "hovertemplate" });
    }

    // ---- terminator improvement -----------------------------------------------------------------

    // The improvement/error MAGNITUDES are GP-posterior-derived — they drift a few % between C# and
    // Python (the documented GP boundary) and are validated in GpTerminatorParityTests. The RENDER test
    // verifies the ASSEMBLY: trace count/order, names, marker/line colors, the FloatRepr opacity string,
    // the min_n_trials segment split, legendgroup/showlegend, and the exact x (trial numbers). So the
    // GP-derived `y` and the derived y-axis `range` are excluded.
    [TestMethod]
    [TestCategory("RequiresPython")]
    [DataRow(0)]
    [DataRow(3)]
    public void TerminatorImprovement_Render_Matches_LiveOptuna(int seed)
    {
        PythonSession.EnsureBitExactHostOrInconclusive();

        var csStudy = Optuna.CreateStudy(sampler: new RandomSampler(seed));
        csStudy.Optimize(t =>
        {
            double x = t.SuggestFloat("x", -10, 10);
            double y = t.SuggestFloat("y", -10, 10);
            return (x - 2.0) * (x - 2.0) + (y + 3.0) * (y + 3.0);
        }, nTrials: 22);
        string csJson = GenericChart.toJson(PlotlyVisualization.PlotTerminatorImprovement(csStudy));

        AssertRenderParity(csJson,
            "def _oss_obj_ti(t):\n" +
            "    x = t.suggest_float('x', -10, 10)\n" +
            "    y = t.suggest_float('y', -10, 10)\n" +
            "    return (x - 2.0) * (x - 2.0) + (y + 3.0) * (y + 3.0)\n" +
            $"study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed={seed}))\n" +
            "study.optimize(_oss_obj_ti, n_trials=22)\n",
            "optuna.visualization.plot_terminator_improvement(study)",
            ignoreKeys: new[] { "y", "range" });
    }

    [TestMethod]
    [TestCategory("RequiresPython")]
    [DataRow(0)]
    public void TerminatorImprovement_PlotError_Render_Matches_LiveOptuna(int seed)
    {
        PythonSession.EnsureBitExactHostOrInconclusive();

        // A StaticErrorEvaluator gives a constant error (no reported CV scores needed) so the red Error
        // trace is populated — exercising the 3rd trace's assembly (name/colors/mode).
        var csStudy = Optuna.CreateStudy(sampler: new RandomSampler(seed));
        csStudy.Optimize(t =>
        {
            double x = t.SuggestFloat("x", -10, 10);
            double y = t.SuggestFloat("y", -10, 10);
            return (x - 2.0) * (x - 2.0) + (y + 3.0) * (y + 3.0);
        }, nTrials: 22);
        string csJson = GenericChart.toJson(PlotlyVisualization.PlotTerminatorImprovement(
            csStudy, plotError: true, errorEvaluator: new StaticErrorEvaluator(0.5)));

        AssertRenderParity(csJson,
            "def _oss_obj_ti(t):\n" +
            "    x = t.suggest_float('x', -10, 10)\n" +
            "    y = t.suggest_float('y', -10, 10)\n" +
            "    return (x - 2.0) * (x - 2.0) + (y + 3.0) * (y + 3.0)\n" +
            $"study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed={seed}))\n" +
            "study.optimize(_oss_obj_ti, n_trials=22)\n",
            "optuna.visualization.plot_terminator_improvement(study, plot_error=True, " +
            "error_evaluator=optuna.terminator.StaticErrorEvaluator(constant=0.5))",
            ignoreKeys: new[] { "y", "range" });
    }
}
