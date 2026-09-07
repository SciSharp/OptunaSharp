using AwesomeAssertions;
using Newtonsoft.Json.Linq;
using OptunaSharp.Importance;
using OptunaSharp.Samplers;
using OptunaSharp.Visualization.Plotly;
using Plotly.NET;

namespace OptunaSharp.Tests.Interop;

/// <summary>
///     Live render parity for the golden-standard customization overloads —
///     <c>PlotlyVisualization.Plot*(study, customization, …)</c> — across <b>all twelve</b> plots. Each
///     test renders the customized figure in C# and diffs it, structurally, against the SAME live Optuna
///     figure with the equivalent customization applied through plotly.py's own
///     <c>fig.update_layout(width, height)</c>. Because the golden bundle's <see cref="Layout" /> adds
///     <c>width</c>/<c>height</c> — keys no Optuna plot authors — the diff proves two things at once:
///     the overload preserved Optuna's canonical figure key-for-key, <b>and</b> its Plotly.NET layout
///     merge lands byte-identically to plotly's. The bundle's <see cref="Config" /> lives outside the
///     <c>{data,layout}</c> figure diff, so it is asserted directly on the C# JSON
///     (<see cref="AssertConfigAndSizeApplied" />). This is the interop-gate complement to the
///     Python-free <c>PlotlyChartCustomizationAllPlotsTests</c>.
/// </summary>
[TestClass]
public class PlotlyCustomizationRenderParityTests : PlotlyRenderParityBase
{
    private const int CustWidth = 960;
    private const int CustHeight = 540;

    /// <summary>The golden bundle applied on the C# side: static-export Config + width/height Layout + DisplayOptions.</summary>
    private static PlotlyChartCustomization Golden()
    {
        var config = new Config();
        config.SetValue("staticPlot", true);
        var layout = new Layout();
        layout.SetValue("width", CustWidth);
        layout.SetValue("height", CustHeight);
        return new PlotlyChartCustomization { Config = config, Layout = layout, DisplayOptions = new DisplayOptions() };
    }

    /// <summary>The Python plot expression with the equivalent size customization applied (update_layout returns the figure).</summary>
    private static string Customized(string pyPlotExpr)
        => $"({pyPlotExpr}).update_layout(width={CustWidth}, height={CustHeight})";

    /// <summary>
    ///     Config is not part of Optuna's <c>figure.to_dict()</c> and never reaches the <c>{data,layout}</c>
    ///     diff, so verify the C# side directly: the static-export Config is attached and the merged
    ///     width/height landed in the layout.
    /// </summary>
    private static void AssertConfigAndSizeApplied(string csJson)
    {
        var fig = JObject.Parse(csJson);
        ((bool)fig["config"]!["staticPlot"]!).Should().BeTrue("the golden Config must be attached to the C# figure");
        ((int)fig["layout"]!["width"]!).Should().Be(CustWidth, "the golden Layout width must be merged");
        ((int)fig["layout"]!["height"]!).Should().Be(CustHeight, "the golden Layout height must be merged");
    }

    // ---- optimization history -------------------------------------------------------------------

    [TestMethod]
    [TestCategory("RequiresPython")]
    [DataRow(0, 30)]
    [DataRow(7, 50)]
    public void OptimizationHistory_Customized_Matches_LiveOptuna(int seed, int nTrials)
    {
        PythonSession.EnsureBitExactHostOrInconclusive();

        var study = BuildQuadraticStudy(seed, nTrials);
        string csJson = GenericChart.toJson(PlotlyVisualization.PlotOptimizationHistory(study, Golden()));
        AssertConfigAndSizeApplied(csJson);

        AssertRenderParity(csJson,
            PyQuadratic +
            $"study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed={seed}))\n" +
            $"study.optimize(_oss_obj, n_trials={nTrials})\n",
            Customized("optuna.visualization.plot_optimization_history(study)"));
    }

    [TestMethod]
    [TestCategory("RequiresPython")]
    [DataRow(0, 30)]
    public void LayoutOverride_Customized_Matches_LiveOptuna(int seed, int nTrials)
    {
        PythonSession.EnsureBitExactHostOrInconclusive();

        // A customization whose Layout OVERRIDES a key the renderer set (the title) must match plotly.py's
        // fig.update_layout(title=...) byte-for-byte — the merge lets the caller win identically on both
        // sides. The title is NOT ignored here: it is the very thing under test.
        const string overrideTitle = "OptunaSharp Custom Title";
        var config = new Config();
        config.SetValue("staticPlot", true);
        var layout = new Layout();
        layout.SetValue("title", overrideTitle);
        layout.SetValue("width", CustWidth);
        layout.SetValue("height", CustHeight);
        var custom = new PlotlyChartCustomization { Config = config, Layout = layout, DisplayOptions = new DisplayOptions() };

        var study = BuildQuadraticStudy(seed, nTrials);
        string csJson = GenericChart.toJson(PlotlyVisualization.PlotOptimizationHistory(study, custom));
        AssertConfigAndSizeApplied(csJson);
        ((string?)JObject.Parse(csJson)["layout"]!["title"]).Should().Be(overrideTitle, "the caller's title must override Optuna's");

        AssertRenderParity(csJson,
            PyQuadratic +
            $"study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed={seed}))\n" +
            $"study.optimize(_oss_obj, n_trials={nTrials})\n",
            $"optuna.visualization.plot_optimization_history(study).update_layout(title='{overrideTitle}', width={CustWidth}, height={CustHeight})");
    }

    // ---- edf ------------------------------------------------------------------------------------

    [TestMethod]
    [TestCategory("RequiresPython")]
    [DataRow(0, 30)]
    [DataRow(7, 50)]
    public void Edf_Customized_Matches_LiveOptuna(int seed, int nTrials)
    {
        PythonSession.EnsureBitExactHostOrInconclusive();

        var study = BuildNamedQuadraticStudy("edf", seed, nTrials);
        string csJson = GenericChart.toJson(PlotlyVisualization.PlotEdf(study, Golden()));
        AssertConfigAndSizeApplied(csJson);

        AssertRenderParity(csJson,
            PyQuadratic +
            $"study = optuna.create_study(study_name='edf', sampler=optuna.samplers.RandomSampler(seed={seed}))\n" +
            $"study.optimize(_oss_obj, n_trials={nTrials})\n",
            Customized("optuna.visualization.plot_edf(study)"));
    }

    // ---- intermediate values --------------------------------------------------------------------

    [TestMethod]
    [TestCategory("RequiresPython")]
    [DataRow(0, 12)]
    [DataRow(5, 20)]
    public void IntermediateValues_Customized_Matches_LiveOptuna(int seed, int nTrials)
    {
        PythonSession.EnsureBitExactHostOrInconclusive();

        var study = Optuna.CreateStudy(sampler: new RandomSampler(seed));
        study.Optimize(t =>
        {
            double x = t.SuggestFloat("x", -10, 10);
            for (int step = 0; step < 5; step++)
                t.Report((x - 2.0) * (x - 2.0) / (step + 1), step);
            return (x - 2.0) * (x - 2.0);
        }, nTrials: nTrials);
        string csJson = GenericChart.toJson(PlotlyVisualization.PlotIntermediateValues(study, Golden()));
        AssertConfigAndSizeApplied(csJson);

        AssertRenderParity(csJson,
            "def _oss_obj_iv(t):\n" +
            "    x = t.suggest_float('x', -10, 10)\n" +
            "    for step in range(5):\n" +
            "        t.report((x - 2.0) * (x - 2.0) / (step + 1), step)\n" +
            "    return (x - 2.0) * (x - 2.0)\n" +
            $"study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed={seed}))\n" +
            $"study.optimize(_oss_obj_iv, n_trials={nTrials})\n",
            Customized("optuna.visualization.plot_intermediate_values(study)"));
    }

    // ---- hypervolume history --------------------------------------------------------------------

    [TestMethod]
    [TestCategory("RequiresPython")]
    [DataRow(0, 30)]
    [DataRow(21, 45)]
    public void HypervolumeHistory_Customized_Matches_LiveOptuna(int seed, int nTrials)
    {
        PythonSession.EnsureBitExactHostOrInconclusive();

        var study = Optuna.CreateStudy(
            directions: new[] { StudyDirection.Minimize, StudyDirection.Minimize },
            sampler: new RandomSampler(seed));
        study.Optimize(t =>
        {
            double x = t.SuggestFloat("x", -10, 10);
            return new[] { (x - 2.0) * (x - 2.0), (x + 3.0) * (x + 3.0) };
        }, nTrials: nTrials);
        string csJson = GenericChart.toJson(
            PlotlyVisualization.PlotHypervolumeHistory(study, new[] { 200.0, 200.0 }, Golden()));
        AssertConfigAndSizeApplied(csJson);

        AssertRenderParity(csJson,
            "def _oss_obj_mo(t):\n" +
            "    x = t.suggest_float('x', -10, 10)\n" +
            "    return [(x - 2.0) * (x - 2.0), (x + 3.0) * (x + 3.0)]\n" +
            $"study = optuna.create_study(directions=['minimize', 'minimize'], sampler=optuna.samplers.RandomSampler(seed={seed}))\n" +
            $"study.optimize(_oss_obj_mo, n_trials={nTrials})\n",
            Customized("optuna.visualization.plot_hypervolume_history(study, [200.0, 200.0])"));
    }

    // ---- timeline -------------------------------------------------------------------------------

    [TestMethod]
    [TestCategory("RequiresPython")]
    [DataRow(0, 20)]
    [DataRow(9, 30)]
    public void Timeline_Customized_Matches_LiveOptuna(int seed, int nTrials)
    {
        PythonSession.EnsureBitExactHostOrInconclusive();

        // start/duration are wall-clock (differ between the two runs) — ignore `base`/`x` as the plain
        // timeline render test does; the customization keys are asserted independently above.
        var study = BuildQuadraticStudy(seed, nTrials);
        string csJson = GenericChart.toJson(PlotlyVisualization.PlotTimeline(study, Golden()));
        AssertConfigAndSizeApplied(csJson);

        AssertRenderParity(csJson,
            PyQuadratic +
            $"study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed={seed}))\n" +
            $"study.optimize(_oss_obj, n_trials={nTrials})\n",
            Customized("optuna.visualization.plot_timeline(study)"),
            ignoreKeys: new[] { "base", "x" });
    }

    // ---- param importances ----------------------------------------------------------------------

    [TestMethod]
    [TestCategory("RequiresPython")]
    [DataRow(0, 30)]
    [DataRow(4, 40)]
    public void ParamImportances_Customized_Matches_LiveOptuna(int seed, int nTrials)
    {
        PythonSession.EnsureBitExactHostOrInconclusive();

        // fANOVA is value-parity ~1 ULP → tolerate the bar heights; the raw-float hovertemplate can't
        // absorb a 1-ULP drift and is excluded (as in the plain param-importances render test).
        var study = Optuna.CreateStudy(sampler: new RandomSampler(seed));
        study.Optimize(t =>
        {
            double x = t.SuggestFloat("x", -10, 10);
            double y = t.SuggestFloat("y", -10, 10);
            long z = t.SuggestInt("z", 0, 10);
            return (x - 2.0) * (x - 2.0) + (y + 3.0) * (y + 3.0) + z;
        }, nTrials: nTrials);
        string csJson = GenericChart.toJson(
            PlotlyVisualization.PlotParamImportances(study, Golden(), new FanovaImportanceEvaluator(seed: 0)));
        AssertConfigAndSizeApplied(csJson);

        AssertRenderParity(csJson,
            "def _oss_obj_pi(t):\n" +
            "    x = t.suggest_float('x', -10, 10)\n" +
            "    y = t.suggest_float('y', -10, 10)\n" +
            "    z = t.suggest_int('z', 0, 10)\n" +
            "    return (x - 2.0) * (x - 2.0) + (y + 3.0) * (y + 3.0) + z\n" +
            $"study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed={seed}))\n" +
            $"study.optimize(_oss_obj_pi, n_trials={nTrials})\n",
            Customized("optuna.visualization.plot_param_importances(study, " +
                       "evaluator=optuna.importance.FanovaImportanceEvaluator(seed=0))"),
            tol: 1e-10,
            ignoreKeys: new[] { "hovertemplate" });
    }

    // ---- terminator improvement -----------------------------------------------------------------

    [TestMethod]
    [TestCategory("RequiresPython")]
    [DataRow(0)]
    public void TerminatorImprovement_Customized_Matches_LiveOptuna(int seed)
    {
        PythonSession.EnsureBitExactHostOrInconclusive();

        // The improvement magnitudes are GP-posterior-derived (validated in GpTerminatorParityTests); the
        // render test verifies the ASSEMBLY, so the GP-derived `y` and derived y-axis `range` are excluded.
        var study = Optuna.CreateStudy(sampler: new RandomSampler(seed));
        study.Optimize(t =>
        {
            double x = t.SuggestFloat("x", -10, 10);
            double y = t.SuggestFloat("y", -10, 10);
            return (x - 2.0) * (x - 2.0) + (y + 3.0) * (y + 3.0);
        }, nTrials: 22);
        string csJson = GenericChart.toJson(PlotlyVisualization.PlotTerminatorImprovement(study, Golden()));
        AssertConfigAndSizeApplied(csJson);

        AssertRenderParity(csJson,
            "def _oss_obj_ti(t):\n" +
            "    x = t.suggest_float('x', -10, 10)\n" +
            "    y = t.suggest_float('y', -10, 10)\n" +
            "    return (x - 2.0) * (x - 2.0) + (y + 3.0) * (y + 3.0)\n" +
            $"study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed={seed}))\n" +
            "study.optimize(_oss_obj_ti, n_trials=22)\n",
            Customized("optuna.visualization.plot_terminator_improvement(study)"),
            ignoreKeys: new[] { "y", "range" });
    }

    // ---- contour --------------------------------------------------------------------------------

    [TestMethod]
    [TestCategory("RequiresPython")]
    [DataRow(0, 25)]
    [DataRow(6, 35)]
    public void Contour_Customized_Matches_LiveOptuna(int seed, int nTrials)
    {
        PythonSession.EnsureBitExactHostOrInconclusive();

        var study = BuildTwoParamStudy(seed, nTrials);
        string csJson = GenericChart.toJson(PlotlyVisualization.PlotContour(study, Golden()));
        AssertConfigAndSizeApplied(csJson);

        AssertRenderParity(csJson, PyTwoParam(seed, nTrials, "_oss_obj_ab"),
            Customized("optuna.visualization.plot_contour(study)"));
    }

    // ---- rank -----------------------------------------------------------------------------------

    [TestMethod]
    [TestCategory("RequiresPython")]
    [DataRow(0, 25)]
    [DataRow(6, 35)]
    public void Rank_Customized_Matches_LiveOptuna(int seed, int nTrials)
    {
        PythonSession.EnsureBitExactHostOrInconclusive();

        var study = BuildTwoParamStudy(seed, nTrials);
        string csJson = GenericChart.toJson(PlotlyVisualization.PlotRank(study, Golden()));
        AssertConfigAndSizeApplied(csJson);

        AssertRenderParity(csJson, PyTwoParam(seed, nTrials, "_oss_obj_ab"),
            Customized("optuna.visualization.plot_rank(study)"));
    }

    // ---- parallel coordinate --------------------------------------------------------------------

    [TestMethod]
    [TestCategory("RequiresPython")]
    [DataRow(0, 25)]
    [DataRow(6, 35)]
    public void ParallelCoordinate_Customized_Matches_LiveOptuna(int seed, int nTrials)
    {
        PythonSession.EnsureBitExactHostOrInconclusive();

        var study = BuildTwoParamStudy(seed, nTrials);
        string csJson = GenericChart.toJson(PlotlyVisualization.PlotParallelCoordinate(study, Golden()));
        AssertConfigAndSizeApplied(csJson);

        AssertRenderParity(csJson, PyTwoParam(seed, nTrials, "_oss_obj_ab"),
            Customized("optuna.visualization.plot_parallel_coordinate(study)"));
    }

    // ---- slice ----------------------------------------------------------------------------------

    [TestMethod]
    [TestCategory("RequiresPython")]
    [DataRow(0, 25)]
    [DataRow(6, 35)]
    public void Slice_Customized_Matches_LiveOptuna(int seed, int nTrials)
    {
        PythonSession.EnsureBitExactHostOrInconclusive();

        // One param → single plot (no make_subplots), byte-exact.
        var study = Optuna.CreateStudy(sampler: new RandomSampler(seed));
        study.Optimize(t =>
        {
            double a = t.SuggestFloat("a", -5, 5);
            return a * a;
        }, nTrials: nTrials);
        string csJson = GenericChart.toJson(PlotlyVisualization.PlotSlice(study, Golden()));
        AssertConfigAndSizeApplied(csJson);

        AssertRenderParity(csJson,
            "def _oss_obj_sl(t):\n" +
            "    a = t.suggest_float('a', -5, 5)\n" +
            "    return a * a\n" +
            $"study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed={seed}))\n" +
            $"study.optimize(_oss_obj_sl, n_trials={nTrials})\n",
            Customized("optuna.visualization.plot_slice(study)"));
    }

    // ---- pareto front ---------------------------------------------------------------------------

    [TestMethod]
    [TestCategory("RequiresPython")]
    [DataRow(0, 20)]
    [DataRow(6, 30)]
    public void ParetoFront_Customized_Matches_LiveOptuna(int seed, int nTrials)
    {
        PythonSession.EnsureBitExactHostOrInconclusive();

        var study = Optuna.CreateStudy(
            directions: new[] { StudyDirection.Minimize, StudyDirection.Minimize },
            sampler: new RandomSampler(seed));
        study.Optimize(t =>
        {
            double x = t.SuggestFloat("x", -5, 5);
            double y = t.SuggestFloat("y", -5, 5);
            return new[] { x, y };
        }, nTrials: nTrials);
        string csJson = GenericChart.toJson(PlotlyVisualization.PlotParetoFront(study, Golden()));
        AssertConfigAndSizeApplied(csJson);

        AssertRenderParity(csJson,
            "def _oss_obj_pf(t):\n" +
            "    x = t.suggest_float('x', -5, 5)\n" +
            "    y = t.suggest_float('y', -5, 5)\n" +
            "    return [x, y]\n" +
            $"study = optuna.create_study(directions=['minimize', 'minimize'], sampler=optuna.samplers.RandomSampler(seed={seed}))\n" +
            $"study.optimize(_oss_obj_pf, n_trials={nTrials})\n",
            Customized("optuna.visualization.plot_pareto_front(study)"));
    }

    // ---- shared two-param (a, b) study for the byte-exact subplot plots -------------------------

    private static Study BuildTwoParamStudy(int seed, int nTrials)
    {
        var study = Optuna.CreateStudy(sampler: new RandomSampler(seed));
        study.Optimize(t =>
        {
            double a = t.SuggestFloat("a", -5, 5);
            double b = t.SuggestFloat("b", -5, 5);
            return a * a + b * b;
        }, nTrials: nTrials);
        return study;
    }

    private static string PyTwoParam(int seed, int nTrials, string fn)
        => $"def {fn}(t):\n" +
           "    a = t.suggest_float('a', -5, 5)\n" +
           "    b = t.suggest_float('b', -5, 5)\n" +
           "    return a * a + b * b\n" +
           $"study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed={seed}))\n" +
           $"study.optimize({fn}, n_trials={nTrials})\n";
}
