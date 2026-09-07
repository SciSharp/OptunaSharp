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

    /// <summary>The shared single-objective quadratic — multiplication (not <c>**</c>) for bit-exactness.</summary>
    private const string PyQuadratic =
        "def _oss_obj(t):\n" +
        "    x = t.suggest_float('x', -10, 10)\n" +
        "    return (x - 2.0) * (x - 2.0)\n";

    /// <summary>Render both sides and structurally diff — <paramref name="pySetup" /> binds <c>study</c>/<c>studies</c>, <paramref name="pyPlot" /> is the plot call.</summary>
    private void AssertRenderParity(string csJson, string pySetup, string pyPlot, double tol = 0.0)
    {
        PyExec(pySetup + FigJsonHelper + $"fig = {pyPlot}\n");
        PlotlyFigure.AssertMatches(csJson, PyStr("_oss_fig_json(fig)"), tol);
    }

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

    [TestMethod]
    [TestCategory("RequiresPython")]
    [DataRow(0, 30)]
    [DataRow(5, 45)]
    public void Maximize_Render_Matches_LiveOptuna(int seed, int nTrials)
    {
        PythonSession.EnsureBitExactHostOrInconclusive();

        // Maximize => the best-value line is a running MAX (and leading infeasible best values are -inf).
        var csStudy = Optuna.CreateStudy(direction: StudyDirection.Maximize, sampler: new RandomSampler(seed));
        csStudy.Optimize(t =>
        {
            double x = t.SuggestFloat("x", -10, 10);
            return (x - 2.0) * (x - 2.0);
        }, nTrials: nTrials);
        string csJson = GenericChart.toJson(PlotlyVisualization.PlotOptimizationHistory(csStudy));

        AssertRenderParity(csJson,
            PyQuadratic +
            $"study = optuna.create_study(direction='maximize', sampler=optuna.samplers.RandomSampler(seed={seed}))\n" +
            $"study.optimize(_oss_obj, n_trials={nTrials})\n",
            "optuna.visualization.plot_optimization_history(study)");
    }

    [TestMethod]
    [TestCategory("RequiresPython")]
    [DataRow(0, 30)]
    [DataRow(11, 40)]
    public void CustomTarget_And_TargetName_Render_Matches_LiveOptuna(int seed, int nTrials)
    {
        PythonSession.EnsureBitExactHostOrInconclusive();

        // A custom target => no best-value line (Optuna can't tell which direction is better), a
        // custom target_name on the y-axis + legend.
        var csStudy = BuildQuadraticStudy(seed, nTrials);
        string csJson = GenericChart.toJson(PlotlyVisualization.PlotOptimizationHistory(
            csStudy, target: t => t.Value!.Value, targetName: "my target"));

        AssertRenderParity(csJson,
            PyQuadratic +
            $"study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed={seed}))\n" +
            $"study.optimize(_oss_obj, n_trials={nTrials})\n",
            "optuna.visualization.plot_optimization_history(study, target=lambda t: t.value, target_name='my target')");
    }

    [TestMethod]
    [TestCategory("RequiresPython")]
    [DataRow(3, 25)]
    [DataRow(8, 35)]
    public void MultiStudy_NoErrorBar_Render_Matches_LiveOptuna(int seed, int nTrials)
    {
        PythonSession.EnsureBitExactHostOrInconclusive();

        // Two named studies overlaid => 3 traces each with "<target> of <study_name>" labels.
        var csStudies = new[]
        {
            BuildNamedQuadraticStudy("s0", seed, nTrials),
            BuildNamedQuadraticStudy("s1", seed + 50, nTrials),
        };
        string csJson = GenericChart.toJson(PlotlyVisualization.PlotOptimizationHistory(csStudies));

        AssertRenderParity(csJson,
            PyQuadratic +
            "studies = []\n" +
            "for nm, sd in [('s0', " + seed + "), ('s1', " + (seed + 50) + ")]:\n" +
            "    st = optuna.create_study(study_name=nm, sampler=optuna.samplers.RandomSampler(seed=sd))\n" +
            $"    st.optimize(_oss_obj, n_trials={nTrials})\n" +
            "    studies.append(st)\n",
            "optuna.visualization.plot_optimization_history(studies)");
    }

    [TestMethod]
    [TestCategory("RequiresPython")]
    [DataRow(0, 30)]
    [DataRow(21, 40)]
    public void MultiObjective_Target_Render_Matches_LiveOptuna(int seed, int nTrials)
    {
        PythonSession.EnsureBitExactHostOrInconclusive();

        // Multi-objective => target is REQUIRED; plot the first objective, no best-value line.
        var csStudy = Optuna.CreateStudy(
            directions: new[] { StudyDirection.Minimize, StudyDirection.Minimize },
            sampler: new RandomSampler(seed));
        csStudy.Optimize(t =>
        {
            double x = t.SuggestFloat("x", -10, 10);
            return new double[] { (x - 2.0) * (x - 2.0), (x + 3.0) * (x + 3.0) };
        }, nTrials: nTrials);
        string csJson = GenericChart.toJson(PlotlyVisualization.PlotOptimizationHistory(
            csStudy, target: t => t.Values![0], targetName: "obj0"));

        AssertRenderParity(csJson,
            "def _oss_obj_mo(t):\n" +
            "    x = t.suggest_float('x', -10, 10)\n" +
            "    return [(x - 2.0) * (x - 2.0), (x + 3.0) * (x + 3.0)]\n" +
            $"study = optuna.create_study(directions=['minimize', 'minimize'], sampler=optuna.samplers.RandomSampler(seed={seed}))\n" +
            $"study.optimize(_oss_obj_mo, n_trials={nTrials})\n",
            "optuna.visualization.plot_optimization_history(study, target=lambda t: t.values[0], target_name='obj0')");
    }

    [TestMethod]
    [TestCategory("RequiresPython")]
    [DataRow(2, 30)]
    [DataRow(17, 40)]
    public void ErrorBar_SingleStudy_Render_Matches_LiveOptuna(int seed, int nTrials)
    {
        PythonSession.EnsureBitExactHostOrInconclusive();

        // error_bar over a single study: per-number std is 0, but the band traces still appear.
        var csStudy = BuildQuadraticStudy(seed, nTrials);
        string csJson = GenericChart.toJson(
            PlotlyVisualization.PlotOptimizationHistory(csStudy, errorBar: true));

        AssertRenderParity(csJson,
            PyQuadratic +
            $"study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed={seed}))\n" +
            $"study.optimize(_oss_obj, n_trials={nTrials})\n",
            "optuna.visualization.plot_optimization_history(study, error_bar=True)");
    }

    [TestMethod]
    [TestCategory("RequiresPython")]
    [DataRow(4, 40)]
    [DataRow(29, 55)]
    public void ErrorBar_WithConstraints_Render_Matches_LiveOptuna(int seed, int nTrials)
    {
        PythonSession.EnsureBitExactHostOrInconclusive();

        // error_bar + constraints: aggregate over feasible only (the "infeasible trials" warning path),
        // and the infeasible trace carries x with an empty y — replicated verbatim from Optuna.
        var csStudy = Optuna.CreateStudy(sampler: new RandomSampler(seed));
        csStudy.Optimize(t =>
        {
            double x = t.SuggestFloat("x", -10, 10);
            t.SetConstraint("c", x);
            return (x - 2.0) * (x - 2.0);
        }, nTrials: nTrials);
        string csJson = GenericChart.toJson(
            PlotlyVisualization.PlotOptimizationHistory(csStudy, errorBar: true));

        AssertRenderParity(csJson,
            "def _oss_obj_c(t):\n" +
            "    x = t.suggest_float('x', -10, 10)\n" +
            "    t.set_constraint('c', x)\n" +
            "    return (x - 2.0) * (x - 2.0)\n" +
            $"study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed={seed}))\n" +
            $"study.optimize(_oss_obj_c, n_trials={nTrials})\n",
            "optuna.visualization.plot_optimization_history(study, error_bar=True)");
    }

    [TestMethod]
    [TestCategory("RequiresPython")]
    [DataRow(0, 30)]
    [DataRow(9, 45)]
    public void PrunedTrials_Render_Matches_LiveOptuna(int seed, int nTrials)
    {
        PythonSession.EnsureBitExactHostOrInconclusive();

        // Every 3rd trial is pruned (Incomplete => NaN value): excluded from markers, treated as +inf
        // by the running-best accumulate. suggest is called before the prune so the RNG stays lock-step.
        var csStudy = Optuna.CreateStudy(sampler: new RandomSampler(seed));
        csStudy.Optimize(t =>
        {
            double x = t.SuggestFloat("x", -10, 10);
            if (t.Number % 3 == 0)
                throw new TrialPruned();
            return (x - 2.0) * (x - 2.0);
        }, nTrials: nTrials);
        string csJson = GenericChart.toJson(PlotlyVisualization.PlotOptimizationHistory(csStudy));

        AssertRenderParity(csJson,
            "def _oss_obj_p(t):\n" +
            "    x = t.suggest_float('x', -10, 10)\n" +
            "    if t.number % 3 == 0:\n" +
            "        raise optuna.TrialPruned()\n" +
            "    return (x - 2.0) * (x - 2.0)\n" +
            $"study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed={seed}))\n" +
            $"study.optimize(_oss_obj_p, n_trials={nTrials})\n",
            "optuna.visualization.plot_optimization_history(study)");
    }

    [TestMethod]
    [TestCategory("RequiresPython")]
    public void NoCompleteTrials_Render_Matches_LiveOptuna()
    {
        PythonSession.EnsureBitExactHostOrInconclusive();

        // All trials pruned => "no complete trials": the data layer clears the info list and the figure
        // has an empty data array (just the layout).
        var csStudy = Optuna.CreateStudy(sampler: new RandomSampler(0));
        csStudy.Optimize(t =>
        {
            double x = t.SuggestFloat("x", -10, 10);
            if (t.Number % 1 == 0)   // always true => every trial is pruned; the typed return below
                throw new TrialPruned();   // just disambiguates the Optimize overload (it is unreachable).
            return (x - 2.0) * (x - 2.0);
        }, nTrials: 6);
        string csJson = GenericChart.toJson(PlotlyVisualization.PlotOptimizationHistory(csStudy));

        AssertRenderParity(csJson,
            "def _oss_obj_allp(t):\n" +
            "    t.suggest_float('x', -10, 10)\n" +
            "    raise optuna.TrialPruned()\n" +
            "study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=0))\n" +
            "study.optimize(_oss_obj_allp, n_trials=6)\n",
            "optuna.visualization.plot_optimization_history(study)");
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

    private static Study BuildNamedQuadraticStudy(string name, int seed, int nTrials)
    {
        var study = Optuna.CreateStudy(studyName: name, sampler: new RandomSampler(seed));
        study.Optimize(t =>
        {
            double x = t.SuggestFloat("x", -10, 10);
            return (x - 2.0) * (x - 2.0);
        }, nTrials: nTrials);
        return study;
    }
}
