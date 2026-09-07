using System;
using System.Collections.Generic;
using OptunaSharp.Importance;
using OptunaSharp.Terminator;
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

    // ---- edf ------------------------------------------------------------------------------------

    /// <summary><c>optuna.visualization.plot_edf</c> for a single study — the empirical distribution function.</summary>
    public static GenericChart PlotEdf(
        Study study,
        Func<FrozenTrial, double>? target = null,
        string targetName = "Objective Value")
        => PlotEdf(new[] { study }, target, targetName);

    /// <summary><c>optuna.visualization.plot_edf</c> for several studies (compares their EDFs).</summary>
    public static GenericChart PlotEdf(
        IReadOnlyList<Study> studies,
        Func<FrozenTrial, double>? target = null,
        string targetName = "Objective Value")
        => EdfPlot.Render(Edf.GetInfo(studies, target, targetName), targetName);

    /// <inheritdoc cref="PlotEdf(Study, Func{FrozenTrial, double}, string)" />
    public static GenericChart PlotEdf(
        Study study,
        PlotlyChartCustomization customization,
        Func<FrozenTrial, double>? target = null,
        string targetName = "Objective Value")
        => PlotEdf(study, target, targetName).WithCustomization(customization);

    /// <inheritdoc cref="PlotEdf(IReadOnlyList{Study}, Func{FrozenTrial, double}, string)" />
    public static GenericChart PlotEdf(
        IReadOnlyList<Study> studies,
        PlotlyChartCustomization customization,
        Func<FrozenTrial, double>? target = null,
        string targetName = "Objective Value")
        => PlotEdf(studies, target, targetName).WithCustomization(customization);

    // ---- intermediate values --------------------------------------------------------------------

    /// <summary><c>optuna.visualization.plot_intermediate_values</c> — each trial's reported intermediate values.</summary>
    public static GenericChart PlotIntermediateValues(Study study)
        => IntermediateValuesPlot.Render(IntermediateValues.GetPlotInfo(study));

    /// <inheritdoc cref="PlotIntermediateValues(Study)" />
    public static GenericChart PlotIntermediateValues(Study study, PlotlyChartCustomization customization)
        => PlotIntermediateValues(study).WithCustomization(customization);

    // ---- hypervolume history --------------------------------------------------------------------

    /// <summary>
    ///     <c>optuna.visualization.plot_hypervolume_history</c> — the running hypervolume of a
    ///     multi-objective study relative to <paramref name="referencePoint" />.
    /// </summary>
    public static GenericChart PlotHypervolumeHistory(Study study, IReadOnlyList<double> referencePoint)
        => HypervolumeHistoryPlot.Render(HypervolumeHistory.GetInfo(study, referencePoint));

    /// <inheritdoc cref="PlotHypervolumeHistory(Study, IReadOnlyList{double})" />
    public static GenericChart PlotHypervolumeHistory(
        Study study,
        IReadOnlyList<double> referencePoint,
        PlotlyChartCustomization customization)
        => PlotHypervolumeHistory(study, referencePoint).WithCustomization(customization);

    // ---- timeline -------------------------------------------------------------------------------

    /// <summary>
    ///     <c>optuna.visualization.plot_timeline</c> — each trial's lifetime as a horizontal bar,
    ///     colored by state. <paramref name="nRecentTrials" /> (if given) keeps only the most recent
    ///     trials; it must be a positive integer.
    /// </summary>
    public static GenericChart PlotTimeline(Study study, int? nRecentTrials = null)
        => TimelinePlot.Render(Timeline.GetInfo(study, nRecentTrials));

    /// <inheritdoc cref="PlotTimeline(Study, int?)" />
    public static GenericChart PlotTimeline(
        Study study,
        PlotlyChartCustomization customization,
        int? nRecentTrials = null)
        => PlotTimeline(study, nRecentTrials).WithCustomization(customization);

    // ---- param importances ----------------------------------------------------------------------

    /// <summary>
    ///     <c>optuna.visualization.plot_param_importances</c> — a horizontal bar chart of the
    ///     hyperparameter importances (default evaluator: fANOVA). One bar group per objective.
    /// </summary>
    public static GenericChart PlotParamImportances(
        Study study,
        BaseImportanceEvaluator? evaluator = null,
        IReadOnlyList<string>? @params = null,
        Func<FrozenTrial, double>? target = null,
        string targetName = "Objective Value")
        => ParamImportancesPlot.Render(
            ParamImportances.GetInfos(study, evaluator, @params, target, targetName), study);

    /// <inheritdoc cref="PlotParamImportances(Study, BaseImportanceEvaluator, IReadOnlyList{string}, Func{FrozenTrial, double}, string)" />
    public static GenericChart PlotParamImportances(
        Study study,
        PlotlyChartCustomization customization,
        BaseImportanceEvaluator? evaluator = null,
        IReadOnlyList<string>? @params = null,
        Func<FrozenTrial, double>? target = null,
        string targetName = "Objective Value")
        => PlotParamImportances(study, evaluator, @params, target, targetName).WithCustomization(customization);

    // ---- terminator improvement -----------------------------------------------------------------

    /// <summary>
    ///     <c>optuna.visualization.plot_terminator_improvement</c> — the terminator's improvement
    ///     (and, when <paramref name="plotError" />, the cross-validation error) over the trials.
    /// </summary>
    public static GenericChart PlotTerminatorImprovement(
        Study study,
        bool plotError = false,
        BaseImprovementEvaluator? improvementEvaluator = null,
        BaseErrorEvaluator? errorEvaluator = null,
        int minNTrials = 20)
        => TerminatorImprovementPlot.Render(
            TerminatorImprovement.GetInfo(study, plotError, improvementEvaluator, errorEvaluator), minNTrials);

    /// <inheritdoc cref="PlotTerminatorImprovement(Study, bool, BaseImprovementEvaluator, BaseErrorEvaluator, int)" />
    public static GenericChart PlotTerminatorImprovement(
        Study study,
        PlotlyChartCustomization customization,
        bool plotError = false,
        BaseImprovementEvaluator? improvementEvaluator = null,
        BaseErrorEvaluator? errorEvaluator = null,
        int minNTrials = 20)
        => PlotTerminatorImprovement(study, plotError, improvementEvaluator, errorEvaluator, minNTrials)
            .WithCustomization(customization);
}
