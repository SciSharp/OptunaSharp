using System.Linq;
using AwesomeAssertions;
using Newtonsoft.Json.Linq;
using OptunaSharp.Samplers;
using OptunaSharp.Visualization.Plotly;
using Plotly.NET;

namespace OptunaSharp.Tests.Visualization.Plotly;

/// <summary>
///     Python-free structure tests for the plotly <c>optimization_history</c> renderer — a CI guard on
///     the info→figure mapping (the byte-exact-vs-Optuna gate lives in the interop suite,
///     <c>PlotlyRenderParityTests</c>). Builds a deterministic seeded study, renders, and inspects the
///     serialized plotly figure (<c>GenericChart.toJson</c> ⇒ <c>{data,layout}</c>).
/// </summary>
[TestClass]
public class OptimizationHistoryPlotTests
{
    private static JObject Render(Study study, bool errorBar = false)
        => JObject.Parse(GenericChart.toJson(
            PlotlyVisualization.PlotOptimizationHistory(study, errorBar: errorBar)));

    private static Study Quadratic(int seed, int nTrials, bool constrained = false)
    {
        var study = Optuna.CreateStudy(sampler: new RandomSampler(seed));
        study.Optimize(t =>
        {
            double x = t.SuggestFloat("x", -10, 10);
            if (constrained)
                t.SetConstraint("c", x);   // infeasible when x > 0
            return (x - 2.0) * (x - 2.0);
        }, nTrials: nTrials);
        return study;
    }

    [TestMethod]
    public void IsAvailable_IsTrue()
        => PlotlyVisualization.IsAvailable.Should().BeTrue();

    [TestMethod]
    public void Layout_HasTitleAndAxisTitles()
    {
        var fig = Render(Quadratic(0, 20));
        var layout = fig["layout"]!;

        ((string?)layout["title"]).Should().Be("Optimization History Plot");
        ((string?)layout["xaxis"]!["title"]).Should().Be("Trial");
        ((string?)layout["yaxis"]!["title"]).Should().Be("Objective Value");
    }

    [TestMethod]
    public void SingleObjective_NoConstraints_HasFeasibleBestAndEmptyInfeasibleTraces()
    {
        var study = Quadratic(0, 20);
        var data = (JArray)Render(study)["data"]!;

        // feasible markers + best-value line + (empty) infeasible markers
        data.Should().HaveCount(3);
        data.Select(t => (string?)t["name"]).Should()
            .Equal("Objective Value", "Best Value", "Infeasible Trial");
        data.Select(t => (string?)t["type"]).Should().AllBe("scatter");
        ((string?)data[0]["mode"]).Should().Be("markers");
        ((string?)data[1]["mode"]).Should().Be("lines");

        // feasible x = trial numbers 0..N-1; y = the objective values, in order.
        var trials = study.GetTrials(deepcopy: false);
        ((JArray)data[0]["x"]!).Select(v => (int)v).Should().Equal(trials.Select(t => t.Number));
        ((JArray)data[0]["y"]!).Select(v => (double)v).Should()
            .Equal(trials.Select(t => t.Value!.Value));

        // best value is the running minimum (Minimize default).
        double running = double.PositiveInfinity;
        var expectedBest = trials.Select(t => running = System.Math.Min(running, t.Value!.Value));
        ((JArray)data[1]["y"]!).Select(v => (double)v).Should().Equal(expectedBest);

        // no constraints => infeasible trace is empty, grey, hidden from the legend.
        ((JArray)data[2]["x"]!).Should().BeEmpty();
        ((string?)data[2]["marker"]!["color"]).Should().Be("#cccccc");
        ((bool)data[2]["showlegend"]!).Should().BeFalse();

        // no error bars unless requested.
        data[0]["error_y"].Should().BeNull();
    }

    [TestMethod]
    public void WithConstraints_PopulatesInfeasibleTrace()
    {
        var data = (JArray)Render(Quadratic(0, 40, constrained: true))["data"]!;

        // some x>0 draws are infeasible => the grey trace carries points.
        ((JArray)data[2]["x"]!).Count.Should().BeGreaterThan(0);
        ((string?)data[2]["marker"]!["color"]).Should().Be("#cccccc");
    }

    [TestMethod]
    public void ErrorBar_AddsVisibleErrorYAndBand()
    {
        var data = (JArray)Render(Quadratic(7, 30), errorBar: true)["data"]!;

        // feasible markers carry a data error_y; the best-value band adds the width-0.01 upper +
        // tonexty lower fill traces (5 traces total for a single error-bar study).
        var errorY = data[0]["error_y"]!;
        ((string?)errorY["type"]).Should().Be("data");
        ((bool)errorY["visible"]!).Should().BeTrue();

        // order: feasible[0], best-line[1], upper-band[2], lower-band[3], infeasible[4].
        data.Should().HaveCount(5);
        ((string?)data[2]["mode"]).Should().Be("lines");
        ((double)data[2]["line"]!["width"]!).Should().Be(0.01);
        ((string?)data[3]["mode"]).Should().Be("none");
        ((string?)data[3]["fill"]).Should().Be("tonexty");
        ((string?)data[3]["fillcolor"]).Should().Be("rgba(255,0,0,0.2)");
    }
}
