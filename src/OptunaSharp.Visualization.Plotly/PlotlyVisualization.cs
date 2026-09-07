using System;
using System.Collections.Generic;
using OptunaSharp.Visualization.Plotly.Renderers;
using Plotly.NET;

namespace OptunaSharp.Visualization.Plotly;

/// <summary>
///     The plotly visualization backend for OptunaSharp — the raw-C# port of
///     <c>optuna.visualization</c> (the plotly backend) built on Plotly.NET, over Core's Phase 3.1
///     visualization <b>data</b> layer. Each method mirrors the Optuna function of the same name
///     (parameter names/order/defaults preserved; C# has no keyword-only arguments) and returns a
///     <see cref="GenericChart" /> — a plotly figure whose spec matches live Optuna's
///     <c>figure.to_dict()</c>.
///     <para>
///     Render it with the Plotly.NET fluent extensions: <c>chart.Show()</c> (opens a browser) or
///     <c>chart.SaveHtml("path.html")</c> — the analogs of Optuna's <c>figure.show()</c> /
///     <c>figure.write_html(...)</c>.
///     </para>
/// </summary>
public static class PlotlyVisualization
{
    /// <summary>
    ///     Whether this visualization backend is available. Mirrors
    ///     <c>optuna.visualization.is_available</c>; Plotly.NET is a hard package dependency, so this
    ///     is always <c>true</c> (present for API compatibility).
    /// </summary>
    public static bool IsAvailable => true;

    /// <summary>
    ///     <c>optuna.visualization.plot_optimization_history</c> for a single study. Plots each trial's
    ///     target value and, unless a custom <paramref name="target" /> is given, the running best value.
    /// </summary>
    /// <param name="study">The study whose trials are plotted.</param>
    /// <param name="target">
    ///     Value to display; when <c>null</c> and the study is single-objective the objective values are
    ///     plotted. Required for multi-objective studies.
    /// </param>
    /// <param name="targetName">Target name for the axis label / legend.</param>
    /// <param name="errorBar">Show the error bar (aggregating across the studies).</param>
    public static GenericChart PlotOptimizationHistory(
        Study study,
        Func<FrozenTrial, double>? target = null,
        string targetName = "Objective Value",
        bool errorBar = false)
        => PlotOptimizationHistory(new[] { study }, target, targetName, errorBar);

    /// <summary>
    ///     <c>optuna.visualization.plot_optimization_history</c> for several studies (the
    ///     <c>Sequence[Study]</c> overload) — overlays their histories for comparison.
    /// </summary>
    public static GenericChart PlotOptimizationHistory(
        IReadOnlyList<Study> studies,
        Func<FrozenTrial, double>? target = null,
        string targetName = "Objective Value",
        bool errorBar = false)
    {
        var infoList = OptimizationHistory.GetInfoList(studies, target, targetName, errorBar);
        return OptimizationHistoryPlot.Render(infoList, targetName);
    }

    /// <summary>
    ///     <c>plot_optimization_history</c> for a single study, then apply a golden-standard Plotly.NET
    ///     <paramref name="customization" /> (config / merged layout / display options) — the idiomatic
    ///     way to size the figure, disable the mode bar, make it a static export, swap the template, …
    ///     Equivalent to <c>PlotOptimizationHistory(study, …).WithCustomization(customization)</c>.
    /// </summary>
    public static GenericChart PlotOptimizationHistory(
        Study study,
        PlotlyChartCustomization customization,
        Func<FrozenTrial, double>? target = null,
        string targetName = "Objective Value",
        bool errorBar = false)
        => PlotOptimizationHistory(study, target, targetName, errorBar).WithCustomization(customization);

    /// <summary>
    ///     <c>plot_optimization_history</c> for several studies, then apply a golden-standard Plotly.NET
    ///     <paramref name="customization" />. Equivalent to
    ///     <c>PlotOptimizationHistory(studies, …).WithCustomization(customization)</c>.
    /// </summary>
    public static GenericChart PlotOptimizationHistory(
        IReadOnlyList<Study> studies,
        PlotlyChartCustomization customization,
        Func<FrozenTrial, double>? target = null,
        string targetName = "Objective Value",
        bool errorBar = false)
        => PlotOptimizationHistory(studies, target, targetName, errorBar).WithCustomization(customization);
}
