using OptunaSharp.Samplers;
using OptunaSharp.Visualization.Plotly;
using Plotly.NET;

namespace OptunaSharp.Tests.Interop;

/// <summary>
///     Live render parity for the 🟢 batch — <c>plot_edf</c>, <c>plot_intermediate_values</c>,
///     <c>plot_hypervolume_history</c>, <c>plot_timeline</c>.
/// </summary>
[TestClass]
public class PlotlyGreenRenderParityTests : PlotlyRenderParityBase
{
    // ---- edf ------------------------------------------------------------------------------------

    [TestMethod]
    [TestCategory("RequiresPython")]
    [DataRow(0, 30)]
    [DataRow(7, 50)]
    public void Edf_Render_Matches_LiveOptuna(int seed, int nTrials)
    {
        PythonSession.EnsureBitExactHostOrInconclusive();

        // EDF names each line by study_name, so the study names must match on both sides.
        var csStudy = BuildNamedQuadraticStudy("edf", seed, nTrials);
        string csJson = GenericChart.toJson(PlotlyVisualization.PlotEdf(csStudy));

        AssertRenderParity(csJson,
            PyQuadratic +
            $"study = optuna.create_study(study_name='edf', sampler=optuna.samplers.RandomSampler(seed={seed}))\n" +
            $"study.optimize(_oss_obj, n_trials={nTrials})\n",
            "optuna.visualization.plot_edf(study)");
    }

    [TestMethod]
    [TestCategory("RequiresPython")]
    [DataRow(3, 25)]
    [DataRow(8, 40)]
    public void Edf_MultiStudy_Render_Matches_LiveOptuna(int seed, int nTrials)
    {
        PythonSession.EnsureBitExactHostOrInconclusive();

        var csStudies = new[]
        {
            BuildNamedQuadraticStudy("e0", seed, nTrials),
            BuildNamedQuadraticStudy("e1", seed + 30, nTrials),
        };
        string csJson = GenericChart.toJson(PlotlyVisualization.PlotEdf(csStudies));

        AssertRenderParity(csJson,
            PyQuadratic +
            "studies = []\n" +
            "for nm, sd in [('e0', " + seed + "), ('e1', " + (seed + 30) + ")]:\n" +
            "    st = optuna.create_study(study_name=nm, sampler=optuna.samplers.RandomSampler(seed=sd))\n" +
            $"    st.optimize(_oss_obj, n_trials={nTrials})\n" +
            "    studies.append(st)\n",
            "optuna.visualization.plot_edf(studies)");
    }

    // ---- intermediate values --------------------------------------------------------------------

    [TestMethod]
    [TestCategory("RequiresPython")]
    [DataRow(0, 12)]
    [DataRow(5, 20)]
    public void IntermediateValues_Render_Matches_LiveOptuna(int seed, int nTrials)
    {
        PythonSession.EnsureBitExactHostOrInconclusive();

        var csStudy = Optuna.CreateStudy(sampler: new RandomSampler(seed));
        csStudy.Optimize(t =>
        {
            double x = t.SuggestFloat("x", -10, 10);
            for (int step = 0; step < 5; step++)
                t.Report((x - 2.0) * (x - 2.0) / (step + 1), step);   // exact IEEE division
            return (x - 2.0) * (x - 2.0);
        }, nTrials: nTrials);
        string csJson = GenericChart.toJson(PlotlyVisualization.PlotIntermediateValues(csStudy));

        AssertRenderParity(csJson,
            "def _oss_obj_iv(t):\n" +
            "    x = t.suggest_float('x', -10, 10)\n" +
            "    for step in range(5):\n" +
            "        t.report((x - 2.0) * (x - 2.0) / (step + 1), step)\n" +
            "    return (x - 2.0) * (x - 2.0)\n" +
            $"study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed={seed}))\n" +
            $"study.optimize(_oss_obj_iv, n_trials={nTrials})\n",
            "optuna.visualization.plot_intermediate_values(study)");
    }

    // ---- hypervolume history --------------------------------------------------------------------

    [TestMethod]
    [TestCategory("RequiresPython")]
    [DataRow(0, 30)]
    [DataRow(21, 45)]
    public void HypervolumeHistory_Render_Matches_LiveOptuna(int seed, int nTrials)
    {
        PythonSession.EnsureBitExactHostOrInconclusive();

        var csStudy = Optuna.CreateStudy(
            directions: new[] { StudyDirection.Minimize, StudyDirection.Minimize },
            sampler: new RandomSampler(seed));
        csStudy.Optimize(t =>
        {
            double x = t.SuggestFloat("x", -10, 10);
            return new double[] { (x - 2.0) * (x - 2.0), (x + 3.0) * (x + 3.0) };
        }, nTrials: nTrials);
        string csJson = GenericChart.toJson(
            PlotlyVisualization.PlotHypervolumeHistory(csStudy, new double[] { 200.0, 200.0 }));

        AssertRenderParity(csJson,
            "def _oss_obj_mo(t):\n" +
            "    x = t.suggest_float('x', -10, 10)\n" +
            "    return [(x - 2.0) * (x - 2.0), (x + 3.0) * (x + 3.0)]\n" +
            $"study = optuna.create_study(directions=['minimize', 'minimize'], sampler=optuna.samplers.RandomSampler(seed={seed}))\n" +
            $"study.optimize(_oss_obj_mo, n_trials={nTrials})\n",
            "optuna.visualization.plot_hypervolume_history(study, [200.0, 200.0])");
    }

    // ---- timeline -------------------------------------------------------------------------------

    [TestMethod]
    [TestCategory("RequiresPython")]
    [DataRow(0, 20)]
    [DataRow(9, 30)]
    public void Timeline_Render_Matches_LiveOptuna(int seed, int nTrials)
    {
        PythonSession.EnsureBitExactHostOrInconclusive();

        // start/duration are wall-clock (differ between the two runs) — ignore `base`/`x`; everything
        // else (per-state bars, colors, y=trial numbers, deterministic hovertext) must match.
        var csStudy = BuildQuadraticStudy(seed, nTrials);
        string csJson = GenericChart.toJson(PlotlyVisualization.PlotTimeline(csStudy));

        AssertRenderParity(csJson,
            PyQuadratic +
            $"study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed={seed}))\n" +
            $"study.optimize(_oss_obj, n_trials={nTrials})\n",
            "optuna.visualization.plot_timeline(study)",
            ignoreKeys: new[] { "base", "x" });
    }
}
