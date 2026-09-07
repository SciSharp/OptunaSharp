using OptunaSharp.Samplers;

namespace OptunaSharp.Tests.Interop;

/// <summary>
///     Shared plumbing for the plotly render-parity suites (Phase 5.2). A seeded
///     <see cref="RandomSampler" /> makes the trials bit-identical (MT19937) between C# and Python, both
///     sides render with their own plotly backend, and the figure specs are structurally diffed by
///     <see cref="PlotlyFigure.AssertMatches" /> — Optuna's own visualization-test methodology (assert
///     on <c>figure.data</c> + the meaningful <c>figure.layout</c>, not pixels).
/// </summary>
public abstract class PlotlyRenderParityBase : OptunaInteropTestBase
{
    /// <summary>
    ///     A Python <c>_oss_fig_json(fig)</c> that returns <c>{data, layout}</c> as a JSON string —
    ///     stripping plotly.py's auto-injected default <c>layout.template</c>, decoding the
    ///     <c>{dtype,bdata}</c> base64 typed-arrays plotly&gt;=6 emits, and flattening numpy scalars, so
    ///     it lines up with OptunaSharp's <c>GenericChart.toJson</c>.
    /// </summary>
    protected const string FigJsonHelper =
        "import numpy as np, base64\n" +
        "import optuna.visualization\n" +
        "def _oss_fig_json(fig):\n" +
        "    d = fig.to_dict()\n" +
        "    d.get('layout', {}).pop('template', None)\n" +
        "    def clean(o):\n" +
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
    protected const string PyQuadratic =
        "def _oss_obj(t):\n" +
        "    x = t.suggest_float('x', -10, 10)\n" +
        "    return (x - 2.0) * (x - 2.0)\n";

    /// <summary>
    ///     Render both sides and structurally diff — <paramref name="pySetup" /> binds
    ///     <c>study</c>/<c>studies</c>, <paramref name="pyPlot" /> is the plot call. Non-deterministic
    ///     keys (e.g. wall-clock <c>base</c>/<c>x</c>) can be excluded via <paramref name="ignoreKeys" />.
    /// </summary>
    protected void AssertRenderParity(string csJson, string pySetup, string pyPlot, double tol = 0.0,
        string[]? ignoreKeys = null, double rtol = 0.0)
    {
        PyExec(pySetup + FigJsonHelper + $"fig = {pyPlot}\n");
        PlotlyFigure.AssertMatches(csJson, PyStr("_oss_fig_json(fig)"), tol, ignoreKeys, rtol);
    }

    protected static Study BuildQuadraticStudy(int seed, int nTrials)
    {
        var study = Optuna.CreateStudy(sampler: new RandomSampler(seed));
        study.Optimize(t =>
        {
            double x = t.SuggestFloat("x", -10, 10);
            return (x - 2.0) * (x - 2.0);
        }, nTrials: nTrials);
        return study;
    }

    protected static Study BuildNamedQuadraticStudy(string name, int seed, int nTrials)
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
