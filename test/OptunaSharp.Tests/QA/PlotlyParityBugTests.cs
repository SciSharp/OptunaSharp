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
}
