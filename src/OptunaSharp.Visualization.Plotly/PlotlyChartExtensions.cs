using Plotly.NET;

namespace OptunaSharp.Visualization.Plotly;

/// <summary>
///     Golden-standard Plotly.NET customization for OptunaSharp charts. Every
///     <c>PlotlyVisualization.Plot*</c> returns a <see cref="GenericChart" />, so you extend it the
///     idiomatic Plotly.NET way. Most of the fluent surface already ships in Plotly.NET core (the
///     <c>Plotly.NET</c> namespace) — <c>.WithConfig</c>, <c>.WithLayout</c>, <c>.WithLayoutStyle</c>,
///     <c>.WithSize</c>, <c>.WithTitle</c>, <c>.WithMarginSize</c>, … — so a single
///     <c>using Plotly.NET;</c> gives you them:
///     <code>
///     PlotlyVisualization.PlotOptimizationHistory(study)
///         .WithConfig(config)     // Plotly.NET core: interactivity / static export
///         .WithSize(900, 500)     // Plotly.NET core
///         .WithDisplayOptions(display);   // added here — see below
///     </code>
///     This class adds only what Plotly.NET's fluent surface is <b>missing</b>:
///     <see cref="WithDisplayOptions" /> (Plotly.NET core has no fluent setter for the HTML
///     <see cref="DisplayOptions" />) and <see cref="WithCustomization" /> (apply a
///     <see cref="PlotlyChartCustomization" /> bundle in one call — the same thing the <c>Plot*</c>
///     customization overloads do).
/// </summary>
public static class PlotlyChartExtensions
{
    /// <summary>
    ///     Attach HTML <see cref="DisplayOptions" /> for the rendered page — document title, extra head
    ///     tags, description, chart-config display, … Wraps <c>GenericChart.setDisplayOptions</c>. (Fills
    ///     the gap left by Plotly.NET, whose fluent surface has <c>.WithConfig</c>/<c>.WithLayout</c> but
    ///     no <c>.WithDisplayOptions</c>.)
    /// </summary>
    /// <param name="chart">Chart to which the requested display settings are applied.</param>
    /// <param name="displayOptions">Plotly.NET HTML display settings to attach to the chart.</param>
    /// <returns>The chart returned by Plotly.NET with the requested display options attached.</returns>
    public static GenericChart WithDisplayOptions(this GenericChart chart, DisplayOptions displayOptions)
        => GenericChart.setDisplayOptions(displayOptions, chart);

    /// <summary>
    ///     Apply a <see cref="PlotlyChartCustomization" /> bundle in one call: the <see cref="Layout" />
    ///     is merged first (via <c>GenericChart.addLayout</c>, so Optuna's title/axes survive and
    ///     config/display can still override), then the <see cref="Config" />
    ///     (<c>GenericChart.setConfig</c>), then the <see cref="DisplayOptions" />
    ///     (<c>GenericChart.setDisplayOptions</c>). A <c>null</c> member is skipped. This is exactly what
    ///     the <c>PlotlyVisualization.Plot*(study, customization, …)</c> overloads do.
    /// </summary>
    /// <param name="chart">Chart to which the requested display settings are applied.</param>
    /// <param name="customization">Layout, interaction, and HTML display settings applied after creating the chart.</param>
    /// <returns>The chart after merging the layout and applying non-null config and display settings.</returns>
    public static GenericChart WithCustomization(this GenericChart chart, PlotlyChartCustomization customization)
    {
        if (customization.Layout is { } layout)
            chart = GenericChart.addLayout(layout, chart);
        if (customization.Config is { } config)
            chart = GenericChart.setConfig(config, chart);
        if (customization.DisplayOptions is { } displayOptions)
            chart = GenericChart.setDisplayOptions(displayOptions, chart);
        return chart;
    }
}
