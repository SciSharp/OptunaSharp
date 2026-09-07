using AwesomeAssertions;
using Newtonsoft.Json.Linq;
using OptunaSharp.Importance;
using OptunaSharp.Samplers;
using OptunaSharp.Visualization.Plotly;
using Plotly.NET;

namespace OptunaSharp.Tests.Visualization.Plotly;

/// <summary>
///     The golden-standard customization surface, exercised across <b>every one</b> of the twelve
///     <c>PlotlyVisualization.Plot*(study, customization, …)</c> overloads (Python-free). The single
///     invariant a customization overload must satisfy — and the one this suite pins for all twelve —
///     is that it applies the caller's <see cref="Config" /> / merged <see cref="Layout" /> /
///     <see cref="DisplayOptions" /> <b>without disturbing the Optuna-faithful figure</b>: the
///     <c>data</c> array and every layout key the renderer authored survive byte-for-byte, and only the
///     caller's keys are added on top. (The <i>live</i> figure-diff form of this same guarantee — that
///     the merge matches plotly's own <c>update_layout</c> semantics on a real Optuna figure — is
///     <c>PlotlyCustomizationRenderParityTests</c> in the interop suite.)
/// </summary>
[TestClass]
public class PlotlyChartCustomizationAllPlotsTests
{
    private const int CustWidth = 640;
    private const int CustHeight = 480;

    /// <summary>The golden bundle: a static-export Config, a width/height Layout (keys no Optuna plot sets), and DisplayOptions.</summary>
    private static PlotlyChartCustomization Golden(DisplayOptions display)
    {
        var config = new Config();
        config.SetValue("staticPlot", true);
        var layout = new Layout();
        layout.SetValue("width", CustWidth);
        layout.SetValue("height", CustHeight);
        return new PlotlyChartCustomization { Config = config, Layout = layout, DisplayOptions = display };
    }

    // Rendering is a pure function of a study's completed trials, so two renders of the same study are
    // byte-identical — which is what lets the canonical vs. customized `data`/layout be diffed exactly.

    private static Study MultiParam()
    {
        var s = Optuna.CreateStudy(sampler: new RandomSampler(0));
        s.Optimize(t =>
        {
            double x = t.SuggestFloat("x", -10, 10);
            double y = t.SuggestFloat("y", -10, 10);
            long z = t.SuggestInt("z", 0, 10);
            return (x - 2.0) * (x - 2.0) + (y + 3.0) * (y + 3.0) + z;
        }, nTrials: 30);
        return s;
    }

    private static Study Mo()
    {
        var s = Optuna.CreateStudy(
            directions: new[] { StudyDirection.Minimize, StudyDirection.Minimize },
            sampler: new RandomSampler(0));
        s.Optimize(t =>
        {
            double x = t.SuggestFloat("x", -5, 5);
            double y = t.SuggestFloat("y", -5, 5);
            return new[] { x, y };
        }, nTrials: 24);
        return s;
    }

    private static Study Intermediate()
    {
        var s = Optuna.CreateStudy(sampler: new RandomSampler(0));
        s.Optimize(t =>
        {
            double x = t.SuggestFloat("x", -10, 10);
            for (int step = 0; step < 5; step++)
                t.Report((x - 2.0) * (x - 2.0) / (step + 1), step);
            return (x - 2.0) * (x - 2.0);
        }, nTrials: 12);
        return s;
    }

    private static Study Terminator()
    {
        var s = Optuna.CreateStudy(sampler: new RandomSampler(0));
        s.Optimize(t =>
        {
            double x = t.SuggestFloat("x", -10, 10);
            double y = t.SuggestFloat("y", -10, 10);
            return (x - 2.0) * (x - 2.0) + (y + 3.0) * (y + 3.0);
        }, nTrials: 22);
        return s;
    }

    [TestMethod]
    public void EveryPlotOverload_AppliesGoldenCustomization_PreservingCanonicalFigure()
    {
        var display = new DisplayOptions();
        var cust = Golden(display);

        var mp = MultiParam();
        var mo = Mo();
        var iv = Intermediate();
        var reference = new[] { 200.0, 200.0 };

        // The canonical figure is captured as JSON *before* the customized one is built: Plotly.NET's
        // addLayout mutates a chart's layout in place, so a customized chart derived from a canonical one
        // (the terminator case below) would otherwise retro-add width to the canonical snapshot. C#'s
        // left-to-right argument evaluation guarantees the canonical toJson runs first.
        AssertGolden("optimization_history",
            GenericChart.toJson(PlotlyVisualization.PlotOptimizationHistory(mp)),
            PlotlyVisualization.PlotOptimizationHistory(mp, cust), display);

        AssertGolden("edf",
            GenericChart.toJson(PlotlyVisualization.PlotEdf(mp)),
            PlotlyVisualization.PlotEdf(mp, cust), display);

        AssertGolden("intermediate_values",
            GenericChart.toJson(PlotlyVisualization.PlotIntermediateValues(iv)),
            PlotlyVisualization.PlotIntermediateValues(iv, cust), display);

        AssertGolden("hypervolume_history",
            GenericChart.toJson(PlotlyVisualization.PlotHypervolumeHistory(mo, reference)),
            PlotlyVisualization.PlotHypervolumeHistory(mo, reference, cust), display);

        AssertGolden("timeline",
            GenericChart.toJson(PlotlyVisualization.PlotTimeline(mp)),
            PlotlyVisualization.PlotTimeline(mp, cust), display);

        // A seeded fANOVA (fresh per render) keeps the two importances byte-identical.
        AssertGolden("param_importances",
            GenericChart.toJson(PlotlyVisualization.PlotParamImportances(mp, new FanovaImportanceEvaluator(seed: 0))),
            PlotlyVisualization.PlotParamImportances(mp, cust, new FanovaImportanceEvaluator(seed: 0)), display);

        AssertGolden("contour",
            GenericChart.toJson(PlotlyVisualization.PlotContour(mp)),
            PlotlyVisualization.PlotContour(mp, cust), display);

        AssertGolden("rank",
            GenericChart.toJson(PlotlyVisualization.PlotRank(mp)),
            PlotlyVisualization.PlotRank(mp, cust), display);

        AssertGolden("parallel_coordinate",
            GenericChart.toJson(PlotlyVisualization.PlotParallelCoordinate(mp)),
            PlotlyVisualization.PlotParallelCoordinate(mp, cust), display);

        AssertGolden("slice",
            GenericChart.toJson(PlotlyVisualization.PlotSlice(mp)),
            PlotlyVisualization.PlotSlice(mp, cust), display);

        AssertGolden("pareto_front",
            GenericChart.toJson(PlotlyVisualization.PlotParetoFront(mo)),
            PlotlyVisualization.PlotParetoFront(mo, cust), display);

        // terminator rides the GP posterior — render it once and apply the customization via the shared
        // extension (a second overload render would re-run the GP). Its OVERLOAD is covered live, with
        // GP tolerance, in PlotlyCustomizationRenderParityTests. Same rendered figure ⇒ exact diff.
        var terminatorCanonical = PlotlyVisualization.PlotTerminatorImprovement(Terminator());
        AssertGolden("terminator_improvement",
            GenericChart.toJson(terminatorCanonical),
            terminatorCanonical.WithCustomization(cust), display);
    }

    [TestMethod]
    public void Customization_LayoutConflict_CallerKeyOverridesOptunaKey()
    {
        // When the caller's Layout sets a key the renderer ALSO set (here: the title), the merge must let
        // the caller win — the same semantics as plotly.py's fig.update_layout(title=...). This is what
        // makes the layout genuinely "merged" rather than "ignored where Optuna already spoke".
        var study = MultiParam();
        var canon = JObject.Parse(GenericChart.toJson(PlotlyVisualization.PlotOptimizationHistory(study)));
        ((string?)canon["layout"]!["title"]).Should().Be("Optimization History Plot");  // the renderer's title

        var layout = new Layout();
        layout.SetValue("title", "My Own Title");
        var chart = PlotlyVisualization.PlotOptimizationHistory(
            study, new PlotlyChartCustomization { Layout = layout });
        var fig = JObject.Parse(GenericChart.toJson(chart));

        ((string?)fig["layout"]!["title"]).Should().Be("My Own Title");                  // the caller wins
        ((string?)fig["layout"]!["xaxis"]!["title"]).Should().Be("Trial");               // untouched keys survive
        JToken.DeepEquals(fig["data"], canon["data"]).Should().BeTrue();                  // data still untouched
    }

    [TestMethod]
    public void Customization_AllNullMembers_LeavesFigureUnchanged()
    {
        // An empty bundle (every member null) is a no-op: WithCustomization skips each null, so the figure
        // is byte-identical to the canonical one — customization is strictly additive and opt-in.
        var study = MultiParam();
        string canon = GenericChart.toJson(PlotlyVisualization.PlotOptimizationHistory(study));
        string customized = GenericChart.toJson(
            PlotlyVisualization.PlotOptimizationHistory(study, new PlotlyChartCustomization()));

        JToken.DeepEquals(JObject.Parse(customized), JObject.Parse(canon)).Should().BeTrue();
    }

    /// <summary>
    ///     The customization contract for one plot: the caller's config/size are applied, every
    ///     canonical <c>data</c> trace and <c>layout</c> key is preserved byte-for-byte, and the
    ///     DisplayOptions instance is attached.
    /// </summary>
    private static void AssertGolden(string name, string canonicalJson, GenericChart customized, DisplayOptions display)
    {
        var canon = JObject.Parse(canonicalJson);
        var cust = JObject.Parse(GenericChart.toJson(customized));

        // 1. the Config is applied on the customized figure and absent on the canonical one (opt-in).
        ((bool)cust["config"]!["staticPlot"]!).Should().BeTrue($"{name}: Config.staticPlot must be attached");
        canon["config"]?["staticPlot"].Should().BeNull($"{name}: the canonical figure carries no custom config");

        // 2. the Layout is MERGED — the caller's width/height are added (Optuna sets neither on any plot).
        ((int)cust["layout"]!["width"]!).Should().Be(CustWidth, $"{name}: Layout.width must be merged in");
        ((int)cust["layout"]!["height"]!).Should().Be(CustHeight, $"{name}: Layout.height must be merged in");
        canon["layout"]!["width"].Should().BeNull($"{name}: canonical layout has no width");

        // 3. …without dropping OR mutating a single key the renderer authored (title, axes, shapes, …).
        foreach (var authored in ((JObject)canon["layout"]!).Properties())
            JToken.DeepEquals(cust["layout"]![authored.Name], authored.Value).Should()
                .BeTrue($"{name}: layout.{authored.Name} must survive customization unchanged");

        // 4. the data array (traces) is completely untouched — customization is layout/config/HTML only.
        JToken.DeepEquals(cust["data"], canon["data"]).Should()
            .BeTrue($"{name}: the figure's data array must be preserved verbatim");

        // 5. the HTML DisplayOptions (invisible to the figure JSON) is the exact instance we passed.
        GenericChart.getDisplayOptions(customized).Should().BeSameAs(display, $"{name}: DisplayOptions must be attached");
    }
}
