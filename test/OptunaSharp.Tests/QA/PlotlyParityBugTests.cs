using System.Linq;
using AwesomeAssertions;
using Newtonsoft.Json.Linq;
using OptunaSharp.Collections;
using OptunaSharp.Distributions;
using OptunaSharp.Visualization.Plotly;
using Plotly.NET;

namespace OptunaSharp.Tests.QA;

/// <summary>
///     Open parity bugs in the Plotly satellite's figures (not the info layer, which <c>VisualizationParityBugTests</c>
///     covers), found by the 2026-11-02 parity QA run (<c>docs/qa/PARITY_QA_2026-11-02.md</c>, D89). Expected values
///     are Optuna 5.0.0 + plotly figures for the same study; each test fails today and passes once fixed, at which
///     point its <c>[OpenBugs]</c> attribute should be removed.
/// </summary>
[TestClass]
public class PlotlyParityBugTests
{
    /// <summary>
    ///     A study with two parameters on <c>[0, 1000]</c> and objective values 1, 2, 3, the second trial's <c>x</c> at
    ///     <c>999.95</c>, so that a parallel-coordinate tick rounds to a power of ten at three significant digits.
    /// </summary>
    /// <returns>The populated in-memory study.</returns>
    private static Study ThreeTrialStudy()
    {
        var study = Optuna.CreateStudy();
        var distribution = new FloatDistribution(0, 1000);
        foreach (var (x, y, v) in new[] { (0.0, 10.0, 1.0), (999.95, 20.0, 2.0), (500.0, 30.0, 3.0) })
        {
            study.AddTrial(FrozenTrial.CreateTrial(value: v,
                @params: new OrderedDict<string, object?> { ["x"] = x, ["y"] = y },
                distributions: new OrderedDict<string, BaseDistribution> { ["x"] = distribution, ["y"] = distribution }));
        }
        return study;
    }

    /// <summary>
    ///     QA-45 (rank colorbar): the colorbar tick text must format each quantile with Optuna's <c>f"{value:3g}"</c>,
    ///     whose <c>3</c> is a minimum width; the port formats with default <c>g</c> and no width, so values shorter
    ///     than three characters lose their padding.
    /// </summary>
    /// <remarks>
    ///     <c>_get_tick_info</c> builds <c>f"{rank_text[i]} ({values[i]:3g})"</c>: for objective values 1, 2, 3 Optuna
    ///     5.0.0 writes <c>['min. (  1)', '25% (1.5)', '50% (  2)', '75% (2.5)', 'max. (  3)']</c>. Integer-valued
    ///     objectives below 100 always hit it. The same helper (<c>PlotlyColors.FormatG</c>) has the rounding defect the
    ///     parallel-coordinate test shows.
    /// </remarks>
    [TestMethod]
    [OpenBugs]
    public void PlotRank_ColorbarTickText_PadsToWidthThreeLikePython()
    {
        var figure = JObject.Parse(GenericChart.toJson(PlotlyVisualization.PlotRank(ThreeTrialStudy())));

        var ticktext = figure["data"]!.Children()
            .Select(trace => trace.SelectToken("marker.colorbar.ticktext"))
            .Single(t => t is not null)!
            .Select(t => (string)t!);

        ticktext.Should().Equal("min. (  1)", "25% (1.5)", "50% (  2)", "75% (2.5)", "max. (  3)");
    }

    /// <summary>
    ///     QA-45 (rounding): tick labels must follow Python's <c>.3g</c>, which rounds to three significant digits
    ///     first and then picks fixed or scientific notation from the rounded exponent; <c>PlotlyColors.FormatG</c>
    ///     takes the exponent from <c>floor(log10(|x|))</c> before rounding, so <c>999.95</c> prints as <c>1000</c>
    ///     (Optuna <c>1e+03</c>), and in scientific notation the mantissa can round up to <c>10</c>.
    /// </summary>
    /// <remarks>
    ///     The parallel-coordinate axis of <c>x</c> spans <c>[0, 999.95]</c>; its five tick annotations are
    ///     <c>0, 250, 500, 750, 1e+03</c> in Optuna 5.0.0 (the hover <c>customdata</c> shows <c>1e+03</c> too). Over
    ///     3,023 sampled values <c>FormatG(v, 3)</c> differs from <c>format(v, '.3g')</c> on 1,201: <c>10e+05</c> for
    ///     <c>999999.5</c> (Optuna <c>1e+06</c>), <c>10e-05</c> for <c>9.99995e-05</c> (Optuna <c>0.0001</c>), fixed
    ///     notation where Python switches to scientific, and <c>Infinitye-324</c> for the subnormal <c>5e-324</c>
    ///     (<c>Math.Pow(10, -324)</c> underflows to 0). The colorbar's <c>QuantileSorted</c> also uses <c>a + t·(b − a)</c>
    ///     where numpy uses <c>b − (b − a)·(1 − t)</c> for <c>t ≥ 0.5</c>, 1 ULP apart in 85 of 2,500 quantiles.
    /// </remarks>
    [TestMethod]
    [OpenBugs]
    public void PlotParallelCoordinate_TickRoundingToPowerOfTen_FormatsLikePythonG()
    {
        var figure = JObject.Parse(GenericChart.toJson(PlotlyVisualization.PlotParallelCoordinate(ThreeTrialStudy())));

        var annotations = figure.SelectToken("layout.annotations")!.Select(a => (string)a["text"]!);

        annotations.Should().Equal("1", "1.5", "2", "2.5", "3", "0", "250", "500", "750", "1e+03", "10", "15", "20", "25", "30");
    }

    /// <summary>
    ///     QA-46: the slice plot draws no infeasible trials on a categorical parameter's subplot in Optuna, because of
    ///     an Optuna defect the port does not reproduce; the port draws them (an extra <c>Infeasible Trial</c> trace).
    /// </summary>
    /// <remarks>
    ///     <c>_generate_slice_subplot</c> appends only <c>x</c> and <c>y</c> for an infeasible trial, never its trial
    ///     number, and for a categorical subplot <c>_get_categorical_plot_values</c> iterates
    ///     <c>zip(values.x, values.y, values.trial_numbers)</c> — empty for the infeasible values — so the infeasible
    ///     list comes back empty and its trace is skipped. Numerical subplots are unaffected. With a feasible trial
    ///     <c>(c='a', x=0.25)</c> and an infeasible one <c>(c='b', x=0.75)</c>, Optuna 5.0.0 draws
    ///     <c>Feasible Trial</c> on <c>x</c> (<c>["'a'"]</c>), then <c>Feasible Trial</c> and <c>Infeasible Trial</c> on
    ///     <c>x2</c>; the port adds an <c>Infeasible Trial</c> trace on <c>x</c>. Copying the defect or keeping the
    ///     port's behaviour (and marking it <c>[Misaligned]</c>) is the owner's decision; this test states Optuna's
    ///     output. Found by the D90 figure fuzz (4 of 60 random studies).
    /// </remarks>
    [TestMethod]
    [OpenBugs]
    public void PlotSlice_InfeasibleCategoricalTrials_OmittedLikeOptuna()
    {
        var study = Optuna.CreateStudy();
        var categorical = new CategoricalDistribution("a", "b");
        var numerical = new FloatDistribution(0, 1);
        foreach (var (c, x, v, g) in new[] { ("a", 0.25, 1.0, -1.0), ("b", 0.75, 2.0, 1.0) })
        {
            study.AddTrial(FrozenTrial.CreateTrial(value: v,
                @params: new OrderedDict<string, object?> { ["c"] = c, ["x"] = x },
                distributions: new OrderedDict<string, BaseDistribution> { ["c"] = categorical, ["x"] = numerical },
                constraints: new OrderedDict<string, double> { ["g"] = g }));
        }

        var figure = JObject.Parse(GenericChart.toJson(PlotlyVisualization.PlotSlice(study)));

        figure["data"]!.Children().Select(t => $"{(string)t["name"]!}@{(string)t["xaxis"]!}")
            .Should().Equal("Feasible Trial@x", "Feasible Trial@x2", "Infeasible Trial@x2");
    }

    /// <summary>
    ///     QA-47: subplot grids must place the bottom row at exactly <c>0.0</c>, as plotly.py's <c>make_subplots</c>
    ///     does; the port's grid arithmetic leaves a rounding residue such as <c>8.326672684688674e-17</c>.
    /// </summary>
    /// <remarks>
    ///     A four-parameter <c>plot_contour</c> (and <c>plot_rank</c>) has a 4×4 grid; plotly.py gives the bottom row
    ///     <c>yaxis13.domain = [0.0, 0.19375]</c>. The port's <c>SubplotGrid</c> produces <c>8.3e-17</c> there (5×5
    ///     grids: <c>2.8e-17</c> / <c>5.6e-17</c>). Invisible in a rendered figure, but the figure JSON is no longer
    ///     equal to plotly.py's, which is what the interop render-parity tests compare. Found by the D90 figure fuzz
    ///     (every contour and rank figure with four or more parameters).
    /// </remarks>
    [TestMethod]
    [OpenBugs]
    public void PlotContour_FourParameterGrid_BottomRowDomainStartsAtZeroLikePlotly()
    {
        var study = Optuna.CreateStudy(sampler: new OptunaSharp.Samplers.RandomSampler(seed: 0));
        study.Optimize(t => t.SuggestFloat("p0", 0, 1) + t.SuggestFloat("p1", 0, 1) + t.SuggestFloat("p2", 0, 1) + t.SuggestFloat("p3", 0, 1), nTrials: 10);

        var figure = JObject.Parse(GenericChart.toJson(PlotlyVisualization.PlotContour(study)));

        ((double)figure.SelectToken("layout.yaxis13.domain[0]")!).Should().Be(0.0);
        ((double)figure.SelectToken("layout.yaxis13.domain[1]")!).Should().Be(0.19375);
    }
}
