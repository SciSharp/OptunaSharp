using AwesomeAssertions;
using Newtonsoft.Json.Linq;
using OptunaSharp.Samplers;
using OptunaSharp.Visualization.Plotly;
using Plotly.NET;

namespace OptunaSharp.Tests.Visualization.Plotly;

/// <summary>
///     The golden-standard Plotly.NET customization surface (<see cref="PlotlyChartExtensions" /> +
///     <see cref="PlotlyChartCustomization" /> + the <c>Plot*</c> customization overloads) — fills the
///     <c>.WithConfig</c>/<c>.WithLayout</c>/<c>.WithDisplayOptions</c> gap Plotly.NET.CSharp leaves and
///     lets a caller configure any OptunaSharp chart the idiomatic way.
/// </summary>
[TestClass]
public class PlotlyChartCustomizationTests
{
    private static Study Quadratic()
    {
        var study = Optuna.CreateStudy(sampler: new RandomSampler(0));
        study.Optimize(t =>
        {
            double x = t.SuggestFloat("x", -10, 10);
            return (x - 2.0) * (x - 2.0);
        }, nTrials: 15);
        return study;
    }

    private static Config StaticConfig()
    {
        var config = new Config();
        config.SetValue("staticPlot", true);
        return config;
    }

    private static Layout WidthLayout(int width)
    {
        var layout = new Layout();
        layout.SetValue("width", width);
        return layout;
    }

    [TestMethod]
    public void WithConfig_AttachesPlotlyConfig()
    {
        var chart = PlotlyVisualization.PlotOptimizationHistory(Quadratic()).WithConfig(StaticConfig());
        var fig = JObject.Parse(GenericChart.toJson(chart));

        ((bool)fig["config"]!["staticPlot"]!).Should().BeTrue();
    }

    [TestMethod]
    public void WithLayout_MergesOntoOptunaLayout_PreservingTitle()
    {
        var chart = PlotlyVisualization.PlotOptimizationHistory(Quadratic()).WithLayout(WidthLayout(900));
        var layout = JObject.Parse(GenericChart.toJson(chart))["layout"]!;

        ((int)layout["width"]!).Should().Be(900);                                  // user's key added
        ((string?)layout["title"]).Should().Be("Optimization History Plot");        // Optuna's key preserved
        ((string?)layout["xaxis"]!["title"]).Should().Be("Trial");
    }

    [TestMethod]
    public void WithDisplayOptions_AttachesDisplayOptions()
    {
        // DisplayOptions are HTML-level (not in the figure JSON), so assert the chart carries the exact
        // object we set — WithDisplayOptions fills the gap Plotly.NET's fluent surface leaves.
        var display = new DisplayOptions();
        var chart = PlotlyVisualization.PlotOptimizationHistory(Quadratic()).WithDisplayOptions(display);

        GenericChart.getDisplayOptions(chart).Should().BeSameAs(display);
    }

    [TestMethod]
    public void CustomizationOverload_AppliesConfigAndLayout()
    {
        var chart = PlotlyVisualization.PlotOptimizationHistory(
            Quadratic(),
            new PlotlyChartCustomization { Config = StaticConfig(), Layout = WidthLayout(640) });
        var fig = JObject.Parse(GenericChart.toJson(chart));

        ((bool)fig["config"]!["staticPlot"]!).Should().BeTrue();
        ((int)fig["layout"]!["width"]!).Should().Be(640);
        ((string?)fig["layout"]!["title"]).Should().Be("Optimization History Plot");
    }

    [TestMethod]
    public void Canonical_HasNoCustomConfig()
    {
        // The Optuna-faithful method leaves config empty (default) — customization is opt-in.
        var fig = JObject.Parse(GenericChart.toJson(PlotlyVisualization.PlotOptimizationHistory(Quadratic())));
        fig["config"]!["staticPlot"].Should().BeNull();
    }
}
