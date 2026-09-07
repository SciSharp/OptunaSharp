using OptunaSharp.Samplers;
using OptunaSharp.Visualization.Plotly;
using Plotly.NET;

namespace OptunaSharp.Tests.Interop;

/// <summary>
///     Live render parity for Phase 5.2 — the plotly rendering package
///     (<see cref="PlotlyVisualization" />). A seeded <see cref="RandomSampler" /> makes the trials
///     bit-identical (MT19937) between C# and Python, and every objective uses multiplication (not
///     <c>**</c>) so the stored values match bit-for-bit. Both sides then render with their own plotly
///     backend and the resulting figure specs are structurally diffed by
///     <see cref="PlotlyFigure.AssertMatches" /> — the same methodology Optuna's own visualization
///     tests use (assert on <c>figure.data</c> + the meaningful <c>figure.layout</c>). The data arrays
///     are bit-exact (no erf / LAPACK on these paths), so the figures must match exactly.
/// </summary>
[TestClass]
public class PlotlyRenderParityTests : OptunaInteropTestBase
{
    /// <summary>
    ///     A Python <c>_oss_fig_json(fig)</c> that returns <c>{data, layout}</c> as a JSON string —
    ///     stripping plotly.py's auto-injected default <c>layout.template</c> and flattening numpy
    ///     arrays / scalars to plain JSON, so it lines up with OptunaSharp's <c>GenericChart.toJson</c>.
    /// </summary>
    private const string FigJsonHelper =
        "import numpy as np, base64\n" +
        "import optuna.visualization\n" +
        "def _oss_fig_json(fig):\n" +
        "    d = fig.to_dict()\n" +
        "    d.get('layout', {}).pop('template', None)\n" +
        "    def clean(o):\n" +
        // plotly>=6 base64-encodes numpy arrays in to_dict() as {'dtype','bdata'[,'shape']} — decode them.
        "        if isinstance(o, dict) and 'bdata' in o and 'dtype' in o:\n" +
        "            arr = np.frombuffer(base64.b64decode(o['bdata']), dtype=o['dtype'])\n" +
        "            shape = o.get('shape')\n" +
        "            if shape is not None:\n" +
        "                arr = arr.reshape([int(s) for s in str(shape).split(',')])\n" +
        "            return clean(arr)\n" +
        "        if isinstance(o, np.ndarray): return [clean(x) for x in o.tolist()]\n" +
        "        if isinstance(o, dict): return {k: clean(v) for k, v in o.items()}\n" +
        "        if isinstance(o, (list, tuple)): return [clean(x) for x in o]\n" +
        "        if isinstance(o, np.generic): return o.item()\n" +
        "        return o\n" +
        "    return json.dumps({'data': clean(d.get('data', [])), 'layout': clean(d.get('layout', {}))})\n";

    [TestMethod]
    [TestCategory("RequiresPython")]
    [DataRow(0, 30)]
    [DataRow(7, 50)]
    [DataRow(2026, 40)]
    public void OptimizationHistory_Render_Matches_LiveOptuna(int seed, int nTrials)
    {
        PythonSession.EnsureBitExactHostOrInconclusive();

        var csStudy = Optuna.CreateStudy(sampler: new RandomSampler(seed));
        csStudy.Optimize(t =>
        {
            double x = t.SuggestFloat("x", -10, 10);
            return (x - 2.0) * (x - 2.0);
        }, nTrials: nTrials);
        string csJson = GenericChart.toJson(PlotlyVisualization.PlotOptimizationHistory(csStudy));

        PyExec(
            "def _oss_obj(t):\n" +
            "    x = t.suggest_float('x', -10, 10)\n" +
            "    return (x - 2.0) * (x - 2.0)\n" +
            $"study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed={seed}))\n" +
            $"study.optimize(_oss_obj, n_trials={nTrials})\n" +
            FigJsonHelper +
            "fig = optuna.visualization.plot_optimization_history(study)\n");
        string pyJson = PyStr("_oss_fig_json(fig)");

        PlotlyFigure.AssertMatches(csJson, pyJson);
    }

    [TestMethod]
    [TestCategory("RequiresPython")]
    [DataRow(0, 40)]
    [DataRow(13, 60)]
    public void OptimizationHistory_WithConstraints_Render_Matches_LiveOptuna(int seed, int nTrials)
    {
        PythonSession.EnsureBitExactHostOrInconclusive();

        var csStudy = Optuna.CreateStudy(sampler: new RandomSampler(seed));
        csStudy.Optimize(t =>
        {
            double x = t.SuggestFloat("x", -10, 10);
            t.SetConstraint("c", x);  // infeasible when x > 0 — exercises the grey "Infeasible Trial" trace.
            return (x - 2.0) * (x - 2.0);
        }, nTrials: nTrials);
        string csJson = GenericChart.toJson(PlotlyVisualization.PlotOptimizationHistory(csStudy));

        PyExec(
            "def _oss_obj_c(t):\n" +
            "    x = t.suggest_float('x', -10, 10)\n" +
            "    t.set_constraint('c', x)\n" +
            "    return (x - 2.0) * (x - 2.0)\n" +
            $"study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed={seed}))\n" +
            $"study.optimize(_oss_obj_c, n_trials={nTrials})\n" +
            FigJsonHelper +
            "fig = optuna.visualization.plot_optimization_history(study)\n");
        string pyJson = PyStr("_oss_fig_json(fig)");

        PlotlyFigure.AssertMatches(csJson, pyJson);
    }

    [TestMethod]
    [TestCategory("RequiresPython")]
    [DataRow(1, 25)]
    [DataRow(99, 35)]
    public void OptimizationHistory_ErrorBar_MultiStudy_Render_Matches_LiveOptuna(int seed, int nTrials)
    {
        PythonSession.EnsureBitExactHostOrInconclusive();

        var csStudies = new[]
        {
            BuildQuadraticStudy(seed, nTrials),
            BuildQuadraticStudy(seed + 100, nTrials),
        };
        string csJson = GenericChart.toJson(
            PlotlyVisualization.PlotOptimizationHistory(csStudies, errorBar: true));

        PyExec(
            "def _oss_obj(t):\n" +
            "    x = t.suggest_float('x', -10, 10)\n" +
            "    return (x - 2.0) * (x - 2.0)\n" +
            "studies = []\n" +
            "for sd in [" + seed + ", " + (seed + 100) + "]:\n" +
            "    st = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=sd))\n" +
            $"    st.optimize(_oss_obj, n_trials={nTrials})\n" +
            "    studies.append(st)\n" +
            FigJsonHelper +
            "fig = optuna.visualization.plot_optimization_history(studies, error_bar=True)\n");
        string pyJson = PyStr("_oss_fig_json(fig)");

        PlotlyFigure.AssertMatches(csJson, pyJson);
    }

    private static Study BuildQuadraticStudy(int seed, int nTrials)
    {
        var study = Optuna.CreateStudy(sampler: new RandomSampler(seed));
        study.Optimize(t =>
        {
            double x = t.SuggestFloat("x", -10, 10);
            return (x - 2.0) * (x - 2.0);
        }, nTrials: nTrials);
        return study;
    }
}
