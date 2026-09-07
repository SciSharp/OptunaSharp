using OptunaSharp.Samplers;
using OptunaSharp.Visualization.Plotly;
using Plotly.NET;

namespace OptunaSharp.Tests.Interop;

/// <summary>
///     Live render parity for the 🔴 batch — the subplot / colorscale / 3D renderers:
///     <c>plot_contour</c>, <c>plot_rank</c>, <c>plot_parallel_coordinate</c>, <c>plot_pareto_front</c>.
/// </summary>
[TestClass]
public class PlotlyRedRenderParityTests : PlotlyRenderParityBase
{
    // ---- contour --------------------------------------------------------------------------------

    [TestMethod]
    [TestCategory("RequiresPython")]
    [DataRow(0, 25)]
    [DataRow(6, 35)]
    public void Contour_TwoParams_Render_Matches_LiveOptuna(int seed, int nTrials)
    {
        PythonSession.EnsureBitExactHostOrInconclusive();

        var csStudy = Optuna.CreateStudy(sampler: new RandomSampler(seed));
        csStudy.Optimize(t =>
        {
            double a = t.SuggestFloat("a", -5, 5);
            double b = t.SuggestFloat("b", -5, 5);
            return a * a + b * b;
        }, nTrials: nTrials);
        string csJson = GenericChart.toJson(PlotlyVisualization.PlotContour(csStudy));

        AssertRenderParity(csJson,
            "def _oss_obj_ct(t):\n" +
            "    a = t.suggest_float('a', -5, 5)\n" +
            "    b = t.suggest_float('b', -5, 5)\n" +
            "    return a * a + b * b\n" +
            $"study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed={seed}))\n" +
            $"study.optimize(_oss_obj_ct, n_trials={nTrials})\n",
            "optuna.visualization.plot_contour(study)");
    }

    [TestMethod]
    [TestCategory("RequiresPython")]
    [DataRow(0, 30)]
    [DataRow(6, 40)]
    public void Contour_Grid_Render_Matches_LiveOptuna(int seed, int nTrials)
    {
        PythonSession.EnsureBitExactHostOrInconclusive();

        // N×N grid: the make_subplots `domain` geometry is float-op-order boilerplate — excluded.
        var csStudy = Optuna.CreateStudy(sampler: new RandomSampler(seed));
        csStudy.Optimize(t =>
        {
            double a = t.SuggestFloat("a", -5, 5);
            double b = t.SuggestFloat("b", -5, 5);
            double c = t.SuggestFloat("c", -5, 5);
            return a * a + b * b + c * c;
        }, nTrials: nTrials);
        string csJson = GenericChart.toJson(PlotlyVisualization.PlotContour(csStudy));

        AssertRenderParity(csJson,
            "def _oss_obj_ct3(t):\n" +
            "    a = t.suggest_float('a', -5, 5)\n" +
            "    b = t.suggest_float('b', -5, 5)\n" +
            "    c = t.suggest_float('c', -5, 5)\n" +
            "    return a * a + b * b + c * c\n" +
            $"study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed={seed}))\n" +
            $"study.optimize(_oss_obj_ct3, n_trials={nTrials})\n",
            "optuna.visualization.plot_contour(study)",
            ignoreKeys: new[] { "domain" });
    }

    [TestMethod]
    [TestCategory("RequiresPython")]
    [DataRow(0, 30)]
    public void Contour_LogAndCategorical_Render_Matches_LiveOptuna(int seed, int nTrials)
    {
        PythonSession.EnsureBitExactHostOrInconclusive();

        // Exercises a log x-axis and a categorical y-axis (two params => single plot, byte-exact).
        var csStudy = Optuna.CreateStudy(sampler: new RandomSampler(seed));
        csStudy.Optimize(t =>
        {
            double lr = t.SuggestFloat("lr", 1e-5, 1e-1, log: true);
            string opt = t.SuggestCategorical("opt", new[] { "adam", "sgd", "rmsprop" });
            return lr * 100.0 + (opt == "adam" ? 1.0 : 2.0);
        }, nTrials: nTrials);
        string csJson = GenericChart.toJson(PlotlyVisualization.PlotContour(csStudy));

        AssertRenderParity(csJson,
            "def _oss_obj_lc(t):\n" +
            "    lr = t.suggest_float('lr', 1e-5, 1e-1, log=True)\n" +
            "    opt = t.suggest_categorical('opt', ['adam', 'sgd', 'rmsprop'])\n" +
            "    return lr * 100.0 + (1.0 if opt == 'adam' else 2.0)\n" +
            $"study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed={seed}))\n" +
            $"study.optimize(_oss_obj_lc, n_trials={nTrials})\n",
            "optuna.visualization.plot_contour(study)");
    }

    // ---- rank -----------------------------------------------------------------------------------

    [TestMethod]
    [TestCategory("RequiresPython")]
    [DataRow(0, 25)]
    [DataRow(6, 35)]
    public void Rank_TwoParams_Render_Matches_LiveOptuna(int seed, int nTrials)
    {
        PythonSession.EnsureBitExactHostOrInconclusive();

        // Byte-exact: validates the ported RdYlBu_r color interpolation (marker.color), the repr-float
        // hovertext, and the quantile + %g colorbar ticktext.
        var csStudy = Optuna.CreateStudy(sampler: new RandomSampler(seed));
        csStudy.Optimize(t =>
        {
            double a = t.SuggestFloat("a", -5, 5);
            double b = t.SuggestFloat("b", -5, 5);
            return a * a + b * b;
        }, nTrials: nTrials);
        string csJson = GenericChart.toJson(PlotlyVisualization.PlotRank(csStudy));

        AssertRenderParity(csJson,
            "def _oss_obj_rk(t):\n" +
            "    a = t.suggest_float('a', -5, 5)\n" +
            "    b = t.suggest_float('b', -5, 5)\n" +
            "    return a * a + b * b\n" +
            $"study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed={seed}))\n" +
            $"study.optimize(_oss_obj_rk, n_trials={nTrials})\n",
            "optuna.visualization.plot_rank(study)");
    }

    [TestMethod]
    [TestCategory("RequiresPython")]
    [DataRow(0, 30)]
    public void Rank_Grid_Render_Matches_LiveOptuna(int seed, int nTrials)
    {
        PythonSession.EnsureBitExactHostOrInconclusive();

        var csStudy = Optuna.CreateStudy(sampler: new RandomSampler(seed));
        csStudy.Optimize(t =>
        {
            double a = t.SuggestFloat("a", -5, 5);
            double b = t.SuggestFloat("b", -5, 5);
            double c = t.SuggestFloat("c", -5, 5);
            return a * a + b * b + c * c;
        }, nTrials: nTrials);
        string csJson = GenericChart.toJson(PlotlyVisualization.PlotRank(csStudy));

        AssertRenderParity(csJson,
            "def _oss_obj_rk3(t):\n" +
            "    a = t.suggest_float('a', -5, 5)\n" +
            "    b = t.suggest_float('b', -5, 5)\n" +
            "    c = t.suggest_float('c', -5, 5)\n" +
            "    return a * a + b * b + c * c\n" +
            $"study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed={seed}))\n" +
            $"study.optimize(_oss_obj_rk3, n_trials={nTrials})\n",
            "optuna.visualization.plot_rank(study)",
            ignoreKeys: new[] { "domain" });
    }

    // ---- parallel coordinate --------------------------------------------------------------------

    [TestMethod]
    [TestCategory("RequiresPython")]
    [DataRow(0, 25)]
    [DataRow(6, 35)]
    public void ParallelCoordinate_Render_Matches_LiveOptuna(int seed, int nTrials)
    {
        PythonSession.EnsureBitExactHostOrInconclusive();

        // Single-objective: draw_order is range(len) (deterministic). Validates the Blues line-color
        // interpolation, the .3g customdata, the linspace/.3g annotations, shapes and the colorbar.
        var csStudy = Optuna.CreateStudy(sampler: new RandomSampler(seed));
        csStudy.Optimize(t =>
        {
            double a = t.SuggestFloat("a", -5, 5);
            double b = t.SuggestFloat("b", -5, 5);
            return a * a + b * b;
        }, nTrials: nTrials);
        string csJson = GenericChart.toJson(PlotlyVisualization.PlotParallelCoordinate(csStudy));

        AssertRenderParity(csJson,
            "def _oss_obj_pc(t):\n" +
            "    a = t.suggest_float('a', -5, 5)\n" +
            "    b = t.suggest_float('b', -5, 5)\n" +
            "    return a * a + b * b\n" +
            $"study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed={seed}))\n" +
            $"study.optimize(_oss_obj_pc, n_trials={nTrials})\n",
            "optuna.visualization.plot_parallel_coordinate(study)");
    }

    // ---- pareto front ---------------------------------------------------------------------------

    [TestMethod]
    [TestCategory("RequiresPython")]
    [DataRow(0, 20)]
    [DataRow(6, 30)]
    public void ParetoFront_TwoObjectives_Render_Matches_LiveOptuna(int seed, int nTrials)
    {
        PythonSession.EnsureBitExactHostOrInconclusive();

        var csStudy = Optuna.CreateStudy(
            directions: new[] { StudyDirection.Minimize, StudyDirection.Minimize },
            sampler: new RandomSampler(seed));
        csStudy.Optimize(t =>
        {
            double x = t.SuggestFloat("x", -5, 5);
            double y = t.SuggestFloat("y", -5, 5);
            return new double[] { x, y };
        }, nTrials: nTrials);
        string csJson = GenericChart.toJson(PlotlyVisualization.PlotParetoFront(csStudy));

        AssertRenderParity(csJson,
            "def _oss_obj_pf(t):\n" +
            "    x = t.suggest_float('x', -5, 5)\n" +
            "    y = t.suggest_float('y', -5, 5)\n" +
            "    return [x, y]\n" +
            $"study = optuna.create_study(directions=['minimize', 'minimize'], sampler=optuna.samplers.RandomSampler(seed={seed}))\n" +
            $"study.optimize(_oss_obj_pf, n_trials={nTrials})\n",
            "optuna.visualization.plot_pareto_front(study)");
    }

    [TestMethod]
    [TestCategory("RequiresPython")]
    [DataRow(0, 25)]
    public void ParetoFront_ThreeObjectives_Render_Matches_LiveOptuna(int seed, int nTrials)
    {
        PythonSession.EnsureBitExactHostOrInconclusive();

        var csStudy = Optuna.CreateStudy(
            directions: new[] { StudyDirection.Minimize, StudyDirection.Minimize, StudyDirection.Minimize },
            sampler: new RandomSampler(seed));
        csStudy.Optimize(t =>
        {
            double x = t.SuggestFloat("x", -5, 5);
            double y = t.SuggestFloat("y", -5, 5);
            double z = t.SuggestFloat("z", -5, 5);
            return new double[] { x, y, z };
        }, nTrials: nTrials);
        string csJson = GenericChart.toJson(PlotlyVisualization.PlotParetoFront(csStudy));

        AssertRenderParity(csJson,
            "def _oss_obj_pf3(t):\n" +
            "    x = t.suggest_float('x', -5, 5)\n" +
            "    y = t.suggest_float('y', -5, 5)\n" +
            "    z = t.suggest_float('z', -5, 5)\n" +
            "    return [x, y, z]\n" +
            $"study = optuna.create_study(directions=['minimize', 'minimize', 'minimize'], sampler=optuna.samplers.RandomSampler(seed={seed}))\n" +
            $"study.optimize(_oss_obj_pf3, n_trials={nTrials})\n",
            "optuna.visualization.plot_pareto_front(study)");
    }

    // ---- slice ----------------------------------------------------------------------------------

    [TestMethod]
    [TestCategory("RequiresPython")]
    [DataRow(0, 25)]
    [DataRow(6, 35)]
    public void Slice_SingleParam_Render_Matches_LiveOptuna(int seed, int nTrials)
    {
        PythonSession.EnsureBitExactHostOrInconclusive();

        // One param => single plot (no make_subplots), byte-exact incl. the colorbar marker.
        var csStudy = Optuna.CreateStudy(sampler: new RandomSampler(seed));
        csStudy.Optimize(t =>
        {
            double a = t.SuggestFloat("a", -5, 5);
            return a * a;
        }, nTrials: nTrials);
        string csJson = GenericChart.toJson(PlotlyVisualization.PlotSlice(csStudy));

        AssertRenderParity(csJson,
            "def _oss_obj_sl(t):\n" +
            "    a = t.suggest_float('a', -5, 5)\n" +
            "    return a * a\n" +
            $"study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed={seed}))\n" +
            $"study.optimize(_oss_obj_sl, n_trials={nTrials})\n",
            "optuna.visualization.plot_slice(study)");
    }

    [TestMethod]
    [TestCategory("RequiresPython")]
    [DataRow(0, 30)]
    public void Slice_Grid_Render_Matches_LiveOptuna(int seed, int nTrials)
    {
        PythonSession.EnsureBitExactHostOrInconclusive();

        var csStudy = Optuna.CreateStudy(sampler: new RandomSampler(seed));
        csStudy.Optimize(t =>
        {
            double a = t.SuggestFloat("a", -5, 5);
            double b = t.SuggestFloat("b", -5, 5);
            double c = t.SuggestFloat("c", -5, 5);
            return a * a + b * b + c * c;
        }, nTrials: nTrials);
        string csJson = GenericChart.toJson(PlotlyVisualization.PlotSlice(csStudy));

        AssertRenderParity(csJson,
            "def _oss_obj_sl3(t):\n" +
            "    a = t.suggest_float('a', -5, 5)\n" +
            "    b = t.suggest_float('b', -5, 5)\n" +
            "    c = t.suggest_float('c', -5, 5)\n" +
            "    return a * a + b * b + c * c\n" +
            $"study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed={seed}))\n" +
            $"study.optimize(_oss_obj_sl3, n_trials={nTrials})\n",
            "optuna.visualization.plot_slice(study)",
            ignoreKeys: new[] { "domain" });
    }

    [TestMethod]
    [TestCategory("RequiresPython")]
    [DataRow(0, 30)]
    public void Slice_LogAndCategorical_Render_Matches_LiveOptuna(int seed, int nTrials)
    {
        PythonSession.EnsureBitExactHostOrInconclusive();

        // A log axis + a categorical axis (repr labels, categoryarray) — two params => make_subplots.
        var csStudy = Optuna.CreateStudy(sampler: new RandomSampler(seed));
        csStudy.Optimize(t =>
        {
            double lr = t.SuggestFloat("lr", 1e-5, 1e-1, log: true);
            string opt = t.SuggestCategorical("opt", new[] { "adam", "sgd", "rmsprop" });
            return lr * 100.0 + (opt == "adam" ? 1.0 : 2.0);
        }, nTrials: nTrials);
        string csJson = GenericChart.toJson(PlotlyVisualization.PlotSlice(csStudy));

        AssertRenderParity(csJson,
            "def _oss_obj_slc(t):\n" +
            "    lr = t.suggest_float('lr', 1e-5, 1e-1, log=True)\n" +
            "    opt = t.suggest_categorical('opt', ['adam', 'sgd', 'rmsprop'])\n" +
            "    return lr * 100.0 + (1.0 if opt == 'adam' else 2.0)\n" +
            $"study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed={seed}))\n" +
            $"study.optimize(_oss_obj_slc, n_trials={nTrials})\n",
            "optuna.visualization.plot_slice(study)",
            ignoreKeys: new[] { "domain" });
    }
}
