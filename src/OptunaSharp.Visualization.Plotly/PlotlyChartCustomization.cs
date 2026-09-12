using Plotly.NET;

namespace OptunaSharp.Visualization.Plotly;

/// <summary>
///     A bundle of the golden-standard Plotly.NET customization holders, applied on top of the
///     canonical (Optuna-faithful) figure a <c>PlotlyVisualization.Plot*</c> method produces. Every
///     <c>Plot*</c> has a customization overload that takes one of these, and it can be applied to any
///     <see cref="GenericChart" /> after the fact via
///     <see cref="PlotlyChartExtensions.WithCustomization" />.
///     <para>
///     All three are optional and map to the matching Plotly.NET seam:
///     <see cref="Config" /> → <c>GenericChart.setConfig</c>, <see cref="Layout" /> →
///     <c>GenericChart.addLayout</c> (<b>merged</b> onto Optuna's title/axes, not replacing them),
///     <see cref="DisplayOptions" /> → <c>GenericChart.setDisplayOptions</c>.
///     </para>
/// </summary>
/// <example>
///     <code>
///     var config = new Config();  config.SetValue("staticPlot", true);
///     var layout = new Layout();  layout.SetValue("width", 900);   // merged onto the Optuna layout
///     var chart = PlotlyVisualization.PlotOptimizationHistory(
///         study,
///         new PlotlyChartCustomization { Config = config, Layout = layout });
///     </code>
/// </example>
public sealed record PlotlyChartCustomization
{
    /// <summary>
    ///     plotly.js <see cref="global::Plotly.NET.Config" /> — mode bar, static export, responsiveness,
    ///     scroll-zoom, edit mode, image-export options, …
    /// </summary>
    public Config? Config { get; init; }

    /// <summary>
    ///     A plotly <see cref="global::Plotly.NET.Layout" /> <b>merged</b> onto the chart's layout (Optuna's
    ///     title/axis titles are preserved) — size, template, colorway, legend, margins, …
    /// </summary>
    public Layout? Layout { get; init; }

    /// <summary>
    ///     HTML <see cref="global::Plotly.NET.DisplayOptions" /> for the rendered page — document title, extra
    ///     head tags, description, chart-config display, …
    /// </summary>
    public DisplayOptions? DisplayOptions { get; init; }
}
