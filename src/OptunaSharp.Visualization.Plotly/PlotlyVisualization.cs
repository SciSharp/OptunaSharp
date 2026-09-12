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
/// <seealso href="https://optuna.readthedocs.io/en/stable/reference/visualization/index.html">Official Optuna visualization API.</seealso>
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
    /// <returns>A Plotly.NET chart that can be customized, displayed, or exported without modifying the study.</returns>
    /// <exception cref="NumSharp.ValueError">A scalar target is required for a multi-objective study, or a requested parameter does not exist.</exception>
    /// <exception cref="System.Exception">A user-supplied selector or evaluator, or the study storage provider, fails; its exception propagates.</exception>
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
    /// <param name="studies">Studies to overlay or aggregate in the figure.</param>
    /// <param name="target">Scalar selector for displayed values; null uses the single objective, and multi-objective plots that require a scalar must provide it.</param>
    /// <param name="targetName">Label used for the selected objective on axes, hover text, or the colorbar.</param>
    /// <param name="errorBar">Whether to aggregate values across studies and show standard-deviation error bars.</param>
    /// <returns>A Plotly.NET chart that can be customized, displayed, or exported without modifying the study.</returns>
    /// <exception cref="NumSharp.ValueError">A scalar target is required for a multi-objective study, or a requested parameter does not exist.</exception>
    /// <exception cref="System.Exception">A user-supplied selector or evaluator, or the study storage provider, fails; its exception propagates.</exception>
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
    /// <param name="study">Study whose recorded trials are visualized.</param>
    /// <param name="customization">Layout, interaction, and HTML display settings applied after creating the chart.</param>
    /// <param name="target">Scalar selector for displayed values; null uses the single objective, and multi-objective plots that require a scalar must provide it.</param>
    /// <param name="targetName">Label used for the selected objective on axes, hover text, or the colorbar.</param>
    /// <param name="errorBar">Whether to aggregate values across studies and show standard-deviation error bars.</param>
    /// <returns>A Plotly.NET chart that can be customized, displayed, or exported without modifying the study.</returns>
    /// <exception cref="NumSharp.ValueError">A scalar target is required for a multi-objective study, or a requested parameter does not exist.</exception>
    /// <exception cref="System.Exception">A user-supplied selector or evaluator, or the study storage provider, fails; its exception propagates.</exception>
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
    /// <param name="studies">Studies to overlay or aggregate in the figure.</param>
    /// <param name="customization">Layout, interaction, and HTML display settings applied after creating the chart.</param>
    /// <param name="target">Scalar selector for displayed values; null uses the single objective, and multi-objective plots that require a scalar must provide it.</param>
    /// <param name="targetName">Label used for the selected objective on axes, hover text, or the colorbar.</param>
    /// <param name="errorBar">Whether to aggregate values across studies and show standard-deviation error bars.</param>
    /// <returns>A Plotly.NET chart that can be customized, displayed, or exported without modifying the study.</returns>
    /// <exception cref="NumSharp.ValueError">A scalar target is required for a multi-objective study, or a requested parameter does not exist.</exception>
    /// <exception cref="System.Exception">A user-supplied selector or evaluator, or the study storage provider, fails; its exception propagates.</exception>
    public static GenericChart PlotOptimizationHistory(
        IReadOnlyList<Study> studies,
        PlotlyChartCustomization customization,
        Func<FrozenTrial, double>? target = null,
        string targetName = "Objective Value",
        bool errorBar = false)
        => PlotOptimizationHistory(studies, target, targetName, errorBar).WithCustomization(customization);

    // ---- edf ------------------------------------------------------------------------------------

    /// <summary><c>optuna.visualization.plot_edf</c> for a single study — the empirical distribution function.</summary>
    /// <param name="study">Study whose recorded trials are visualized.</param>
    /// <param name="target">Scalar selector for displayed values; null uses the single objective, and multi-objective plots that require a scalar must provide it.</param>
    /// <param name="targetName">Label used for the selected objective on axes, hover text, or the colorbar.</param>
    /// <returns>A Plotly.NET chart that can be customized, displayed, or exported without modifying the study.</returns>
    /// <exception cref="NumSharp.ValueError">A scalar target is required for a multi-objective study, or a requested parameter does not exist.</exception>
    /// <exception cref="System.Exception">A user-supplied selector or evaluator, or the study storage provider, fails; its exception propagates.</exception>
    public static GenericChart PlotEdf(
        Study study,
        Func<FrozenTrial, double>? target = null,
        string targetName = "Objective Value")
        => PlotEdf(new[] { study }, target, targetName);

    /// <summary><c>optuna.visualization.plot_edf</c> for several studies (compares their EDFs).</summary>
    /// <param name="studies">Studies to overlay or aggregate in the figure.</param>
    /// <param name="target">Scalar selector for displayed values; null uses the single objective, and multi-objective plots that require a scalar must provide it.</param>
    /// <param name="targetName">Label used for the selected objective on axes, hover text, or the colorbar.</param>
    /// <returns>A Plotly.NET chart that can be customized, displayed, or exported without modifying the study.</returns>
    /// <exception cref="NumSharp.ValueError">A scalar target is required for a multi-objective study, or a requested parameter does not exist.</exception>
    /// <exception cref="System.Exception">A user-supplied selector or evaluator, or the study storage provider, fails; its exception propagates.</exception>
    public static GenericChart PlotEdf(
        IReadOnlyList<Study> studies,
        Func<FrozenTrial, double>? target = null,
        string targetName = "Objective Value")
        => EdfPlot.Render(Edf.GetInfo(studies, target, targetName), targetName);

    /// <inheritdoc cref="PlotEdf(Study, Func{FrozenTrial, double}, string)" />
    /// <param name="customization">Layout, interaction, and HTML display settings applied after creating the chart.</param>
    /// <param name="study">Study whose recorded trials are visualized.</param>
    /// <param name="target">Scalar selector for displayed values; null uses the single objective, and multi-objective plots that require a scalar must provide it.</param>
    /// <param name="targetName">Label used for the selected objective on axes, hover text, or the colorbar.</param>
    public static GenericChart PlotEdf(
        Study study,
        PlotlyChartCustomization customization,
        Func<FrozenTrial, double>? target = null,
        string targetName = "Objective Value")
        => PlotEdf(study, target, targetName).WithCustomization(customization);

    /// <inheritdoc cref="PlotEdf(IReadOnlyList{Study}, Func{FrozenTrial, double}, string)" />
    /// <param name="customization">Layout, interaction, and HTML display settings applied after creating the chart.</param>
    /// <param name="studies">Studies to overlay or aggregate in the figure.</param>
    /// <param name="target">Scalar selector for displayed values; null uses the single objective, and multi-objective plots that require a scalar must provide it.</param>
    /// <param name="targetName">Label used for the selected objective on axes, hover text, or the colorbar.</param>
    public static GenericChart PlotEdf(
        IReadOnlyList<Study> studies,
        PlotlyChartCustomization customization,
        Func<FrozenTrial, double>? target = null,
        string targetName = "Objective Value")
        => PlotEdf(studies, target, targetName).WithCustomization(customization);

    // ---- intermediate values --------------------------------------------------------------------

    /// <summary><c>optuna.visualization.plot_intermediate_values</c> — each trial's reported intermediate values.</summary>
    /// <param name="study">Study whose recorded trials are visualized.</param>
    /// <returns>A Plotly.NET chart that can be customized, displayed, or exported without modifying the study.</returns>
    public static GenericChart PlotIntermediateValues(Study study)
        => IntermediateValuesPlot.Render(IntermediateValues.GetPlotInfo(study));

    /// <inheritdoc cref="PlotIntermediateValues(Study)" />
    /// <param name="customization">Layout, interaction, and HTML display settings applied after creating the chart.</param>
    /// <param name="study">Study whose recorded trials are visualized.</param>
    public static GenericChart PlotIntermediateValues(Study study, PlotlyChartCustomization customization)
        => PlotIntermediateValues(study).WithCustomization(customization);

    // ---- hypervolume history --------------------------------------------------------------------

    /// <summary>
    ///     <c>optuna.visualization.plot_hypervolume_history</c> — the running hypervolume of a
    ///     multi-objective study relative to <paramref name="referencePoint" />.
    /// </summary>
    /// <param name="study">Study whose recorded trials are visualized.</param>
    /// <param name="referencePoint">Reference coordinates in objective order; length must match a multi-objective study.</param>
    /// <returns>A Plotly.NET chart that can be customized, displayed, or exported without modifying the study.</returns>
    /// <exception cref="NumSharp.ValueError">The study is single-objective or the reference point dimension differs from the objective count.</exception>
    public static GenericChart PlotHypervolumeHistory(Study study, IReadOnlyList<double> referencePoint)
        => HypervolumeHistoryPlot.Render(HypervolumeHistory.GetInfo(study, referencePoint));

    /// <inheritdoc cref="PlotHypervolumeHistory(Study, IReadOnlyList{double})" />
    /// <param name="customization">Layout, interaction, and HTML display settings applied after creating the chart.</param>
    /// <param name="study">Study whose recorded trials are visualized.</param>
    /// <param name="referencePoint">Reference coordinates in objective order; length must match a multi-objective study.</param>
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
    /// <param name="study">Study whose recorded trials are visualized.</param>
    /// <param name="nRecentTrials">Positive number of most recent trials to display, or null to show all trials.</param>
    /// <returns>A Plotly.NET chart that can be customized, displayed, or exported without modifying the study.</returns>
    /// <exception cref="NumSharp.ValueError">nRecentTrials is supplied but is not positive.</exception>
    public static GenericChart PlotTimeline(Study study, int? nRecentTrials = null)
        => TimelinePlot.Render(Timeline.GetInfo(study, nRecentTrials));

    /// <inheritdoc cref="PlotTimeline(Study, int?)" />
    /// <param name="customization">Layout, interaction, and HTML display settings applied after creating the chart.</param>
    /// <param name="study">Study whose recorded trials are visualized.</param>
    /// <param name="nRecentTrials">Positive number of most recent trials to display, or null to show all trials.</param>
    public static GenericChart PlotTimeline(
        Study study,
        PlotlyChartCustomization customization,
        int? nRecentTrials = null)
        => PlotTimeline(study, nRecentTrials).WithCustomization(customization);

    // ---- param importances ----------------------------------------------------------------------

    /// <summary>
    ///     <c>optuna.visualization.plot_param_importances</c> — a horizontal bar chart of the
    ///     hyperparameter importances (default evaluator: PED-ANOVA). One bar group per objective.
    /// </summary>
    /// <param name="study">Study whose recorded trials are visualized.</param>
    /// <param name="evaluator">Parameter-importance evaluator; null uses the Core importance default.</param>
    /// <param name="params">Parameter names to visualize; null selects all eligible parameters.</param>
    /// <param name="target">Scalar selector for displayed values; null uses the single objective, and multi-objective plots that require a scalar must provide it.</param>
    /// <param name="targetName">Label used for the selected objective on axes, hover text, or the colorbar.</param>
    /// <returns>A Plotly.NET chart that can be customized, displayed, or exported without modifying the study.</returns>
    /// <exception cref="System.Exception">A user-supplied selector or evaluator, or the study storage provider, fails; its exception propagates.</exception>
    /// <exception cref="NumSharp.ValueError">The importance evaluator cannot evaluate the selected trials or parameter set.</exception>
    public static GenericChart PlotParamImportances(
        Study study,
        BaseImportanceEvaluator? evaluator = null,
        IReadOnlyList<string>? @params = null,
        Func<FrozenTrial, double>? target = null,
        string targetName = "Objective Value")
        => ParamImportancesPlot.Render(
            ParamImportances.GetInfos(study, evaluator, @params, target, targetName), study);

    /// <inheritdoc cref="PlotParamImportances(Study, BaseImportanceEvaluator, IReadOnlyList{string}, Func{FrozenTrial, double}, string)" />
    /// <param name="customization">Layout, interaction, and HTML display settings applied after creating the chart.</param>
    /// <param name="study">Study whose recorded trials are visualized.</param>
    /// <param name="evaluator">Parameter-importance evaluator; null uses the Core importance default.</param>
    /// <param name="params">Parameter names to visualize; null selects all eligible parameters.</param>
    /// <param name="target">Scalar selector for displayed values; null uses the single objective, and multi-objective plots that require a scalar must provide it.</param>
    /// <param name="targetName">Label used for the selected objective on axes, hover text, or the colorbar.</param>
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
    /// <param name="study">Study whose recorded trials are visualized.</param>
    /// <param name="plotError">Whether to compute and plot an error estimate beside the improvement estimate.</param>
    /// <param name="improvementEvaluator">Improvement-bound evaluator; null uses RegretBoundEvaluator.</param>
    /// <param name="errorEvaluator">Error evaluator; null selects the default appropriate to the improvement evaluator.</param>
    /// <param name="minNTrials">Initial trial-count region drawn with reduced opacity and excluded from the later improvement maximum.</param>
    /// <returns>A Plotly.NET chart that can be customized, displayed, or exported without modifying the study.</returns>
    /// <exception cref="System.Exception">A user-supplied selector or evaluator, or the study storage provider, fails; its exception propagates.</exception>
    /// <exception cref="NumSharp.ValueError">The study has multiple objectives or an evaluator rejects its required observations.</exception>
    public static GenericChart PlotTerminatorImprovement(
        Study study,
        bool plotError = false,
        BaseImprovementEvaluator? improvementEvaluator = null,
        BaseErrorEvaluator? errorEvaluator = null,
        int minNTrials = 20)
        => TerminatorImprovementPlot.Render(
            TerminatorImprovement.GetInfo(study, plotError, improvementEvaluator, errorEvaluator), minNTrials);

    /// <inheritdoc cref="PlotTerminatorImprovement(Study, bool, BaseImprovementEvaluator, BaseErrorEvaluator, int)" />
    /// <param name="customization">Layout, interaction, and HTML display settings applied after creating the chart.</param>
    /// <param name="study">Study whose recorded trials are visualized.</param>
    /// <param name="plotError">Whether to compute and plot an error estimate beside the improvement estimate.</param>
    /// <param name="improvementEvaluator">Improvement-bound evaluator; null uses RegretBoundEvaluator.</param>
    /// <param name="errorEvaluator">Error evaluator; null selects the default appropriate to the improvement evaluator.</param>
    /// <param name="minNTrials">Initial trial-count region drawn with reduced opacity and excluded from the later improvement maximum.</param>
    public static GenericChart PlotTerminatorImprovement(
        Study study,
        PlotlyChartCustomization customization,
        bool plotError = false,
        BaseImprovementEvaluator? improvementEvaluator = null,
        BaseErrorEvaluator? errorEvaluator = null,
        int minNTrials = 20)
        => PlotTerminatorImprovement(study, plotError, improvementEvaluator, errorEvaluator, minNTrials)
            .WithCustomization(customization);

    // ---- contour --------------------------------------------------------------------------------

    /// <summary>
    ///     <c>optuna.visualization.plot_contour</c> — the parameter relationships as contour plots
    ///     (a single plot for two params, an N×N matrix for more).
    /// </summary>
    /// <param name="study">Study whose recorded trials are visualized.</param>
    /// <param name="params">Parameter names to visualize; null selects all eligible parameters.</param>
    /// <param name="target">Scalar selector for displayed values; null uses the single objective, and multi-objective plots that require a scalar must provide it.</param>
    /// <param name="targetName">Label used for the selected objective on axes, hover text, or the colorbar.</param>
    /// <returns>A Plotly.NET chart that can be customized, displayed, or exported without modifying the study.</returns>
    /// <exception cref="NumSharp.ValueError">A scalar target is required for a multi-objective study, or a requested parameter does not exist.</exception>
    /// <exception cref="System.Exception">A user-supplied selector or evaluator, or the study storage provider, fails; its exception propagates.</exception>
    public static GenericChart PlotContour(
        Study study,
        IReadOnlyList<string>? @params = null,
        Func<FrozenTrial, double>? target = null,
        string targetName = "Objective Value")
        => ContourPlot.Render(Contour.GetInfo(study, @params, target, targetName));

    /// <inheritdoc cref="PlotContour(Study, IReadOnlyList{string}, Func{FrozenTrial, double}, string)" />
    /// <param name="customization">Layout, interaction, and HTML display settings applied after creating the chart.</param>
    /// <param name="study">Study whose recorded trials are visualized.</param>
    /// <param name="params">Parameter names to visualize; null selects all eligible parameters.</param>
    /// <param name="target">Scalar selector for displayed values; null uses the single objective, and multi-objective plots that require a scalar must provide it.</param>
    /// <param name="targetName">Label used for the selected objective on axes, hover text, or the colorbar.</param>
    public static GenericChart PlotContour(
        Study study,
        PlotlyChartCustomization customization,
        IReadOnlyList<string>? @params = null,
        Func<FrozenTrial, double>? target = null,
        string targetName = "Objective Value")
        => PlotContour(study, @params, target, targetName).WithCustomization(customization);

    // ---- rank -----------------------------------------------------------------------------------

    /// <summary>
    ///     <c>optuna.visualization.plot_rank</c> — the (x-param, y-param) scatters colored by the target
    ///     rank on the <c>RdYlBu_r</c> scale (a single plot for two params, an N×N matrix for more).
    /// </summary>
    /// <param name="study">Study whose recorded trials are visualized.</param>
    /// <param name="params">Parameter names to visualize; null selects all eligible parameters.</param>
    /// <param name="target">Scalar selector for displayed values; null uses the single objective, and multi-objective plots that require a scalar must provide it.</param>
    /// <param name="targetName">Label used for the selected objective on axes, hover text, or the colorbar.</param>
    /// <returns>A Plotly.NET chart that can be customized, displayed, or exported without modifying the study.</returns>
    /// <exception cref="NumSharp.ValueError">A scalar target is required for a multi-objective study, or a requested parameter does not exist.</exception>
    /// <exception cref="System.Exception">A user-supplied selector or evaluator, or the study storage provider, fails; its exception propagates.</exception>
    public static GenericChart PlotRank(
        Study study,
        IReadOnlyList<string>? @params = null,
        Func<FrozenTrial, double>? target = null,
        string targetName = "Objective Value")
        => RankPlot.Render(Rank.GetInfo(study, @params, target, targetName));

    /// <inheritdoc cref="PlotRank(Study, IReadOnlyList{string}, Func{FrozenTrial, double}, string)" />
    /// <param name="customization">Layout, interaction, and HTML display settings applied after creating the chart.</param>
    /// <param name="study">Study whose recorded trials are visualized.</param>
    /// <param name="params">Parameter names to visualize; null selects all eligible parameters.</param>
    /// <param name="target">Scalar selector for displayed values; null uses the single objective, and multi-objective plots that require a scalar must provide it.</param>
    /// <param name="targetName">Label used for the selected objective on axes, hover text, or the colorbar.</param>
    public static GenericChart PlotRank(
        Study study,
        PlotlyChartCustomization customization,
        IReadOnlyList<string>? @params = null,
        Func<FrozenTrial, double>? target = null,
        string targetName = "Objective Value")
        => PlotRank(study, @params, target, targetName).WithCustomization(customization);

    // ---- parallel coordinate --------------------------------------------------------------------

    /// <summary>
    ///     <c>optuna.visualization.plot_parallel_coordinate</c> — the high-dimensional parameter
    ///     relationships as a parallel-coordinate plot, lines colored by target value / Pareto rank.
    /// </summary>
    /// <param name="study">Study whose recorded trials are visualized.</param>
    /// <param name="params">Parameter names to visualize; null selects all eligible parameters.</param>
    /// <param name="target">Scalar selector for displayed values; null uses the single objective, and multi-objective plots that require a scalar must provide it.</param>
    /// <param name="targetName">Label used for the selected objective on axes, hover text, or the colorbar.</param>
    /// <returns>A Plotly.NET chart that can be customized, displayed, or exported without modifying the study.</returns>
    /// <exception cref="System.Exception">A user-supplied selector or evaluator, or the study storage provider, fails; its exception propagates.</exception>
    /// <exception cref="NumSharp.ValueError">A requested parameter does not exist in the study.</exception>
    public static GenericChart PlotParallelCoordinate(
        Study study,
        IReadOnlyList<string>? @params = null,
        Func<FrozenTrial, double>? target = null,
        string targetName = "Objective Value")
        => ParallelCoordinatePlot.Render(ParallelCoordinate.GetInfo(study, @params, target, targetName));

    /// <inheritdoc cref="PlotParallelCoordinate(Study, IReadOnlyList{string}, Func{FrozenTrial, double}, string)" />
    /// <param name="customization">Layout, interaction, and HTML display settings applied after creating the chart.</param>
    /// <param name="study">Study whose recorded trials are visualized.</param>
    /// <param name="params">Parameter names to visualize; null selects all eligible parameters.</param>
    /// <param name="target">Scalar selector for displayed values; null uses the single objective, and multi-objective plots that require a scalar must provide it.</param>
    /// <param name="targetName">Label used for the selected objective on axes, hover text, or the colorbar.</param>
    public static GenericChart PlotParallelCoordinate(
        Study study,
        PlotlyChartCustomization customization,
        IReadOnlyList<string>? @params = null,
        Func<FrozenTrial, double>? target = null,
        string targetName = "Objective Value")
        => PlotParallelCoordinate(study, @params, target, targetName).WithCustomization(customization);

    // ---- slice ----------------------------------------------------------------------------------

    /// <summary>
    ///     <c>optuna.visualization.plot_slice</c> — one subplot per parameter, plotting the target value
    ///     against that parameter (markers colored by trial number).
    /// </summary>
    /// <param name="study">Study whose recorded trials are visualized.</param>
    /// <param name="params">Parameter names to visualize; null selects all eligible parameters.</param>
    /// <param name="target">Scalar selector for displayed values; null uses the single objective, and multi-objective plots that require a scalar must provide it.</param>
    /// <param name="targetName">Label used for the selected objective on axes, hover text, or the colorbar.</param>
    /// <returns>A Plotly.NET chart that can be customized, displayed, or exported without modifying the study.</returns>
    /// <exception cref="NumSharp.ValueError">A scalar target is required for a multi-objective study, or a requested parameter does not exist.</exception>
    /// <exception cref="System.Exception">A user-supplied selector or evaluator, or the study storage provider, fails; its exception propagates.</exception>
    public static GenericChart PlotSlice(
        Study study,
        IReadOnlyList<string>? @params = null,
        Func<FrozenTrial, double>? target = null,
        string targetName = "Objective Value")
        => SlicePlotRenderer.Render(SlicePlot.GetPlotInfo(study, @params, target, targetName));

    /// <inheritdoc cref="PlotSlice(Study, IReadOnlyList{string}, Func{FrozenTrial, double}, string)" />
    /// <param name="customization">Layout, interaction, and HTML display settings applied after creating the chart.</param>
    /// <param name="study">Study whose recorded trials are visualized.</param>
    /// <param name="params">Parameter names to visualize; null selects all eligible parameters.</param>
    /// <param name="target">Scalar selector for displayed values; null uses the single objective, and multi-objective plots that require a scalar must provide it.</param>
    /// <param name="targetName">Label used for the selected objective on axes, hover text, or the colorbar.</param>
    public static GenericChart PlotSlice(
        Study study,
        PlotlyChartCustomization customization,
        IReadOnlyList<string>? @params = null,
        Func<FrozenTrial, double>? target = null,
        string targetName = "Objective Value")
        => PlotSlice(study, @params, target, targetName).WithCustomization(customization);

    // ---- pareto front ---------------------------------------------------------------------------

    /// <summary>
    ///     <c>optuna.visualization.plot_pareto_front</c> — the Pareto front of a 2- or 3-objective study
    ///     (2D scatter or 3D scatter). Provide <paramref name="targets" /> for studies with more
    ///     objectives.
    /// </summary>
    /// <param name="study">Study whose recorded trials are visualized.</param>
    /// <param name="targetNames">Two or three axis labels; required with custom targets when the study has no completed trials.</param>
    /// <param name="includeDominatedTrials">Whether feasible dominated trials are included alongside the Pareto front.</param>
    /// <param name="constraintsFunc">Deprecated constraint callback; positive values indicate violations. Null reads constraints stored on trials.</param>
    /// <param name="targets">Selector returning two or three displayed coordinates; null uses the study's objective values.</param>
    /// <returns>A Plotly.NET chart that can be customized, displayed, or exported without modifying the study.</returns>
    /// <exception cref="System.Exception">A user-supplied selector or evaluator, or the study storage provider, fails; its exception propagates.</exception>
    /// <exception cref="NumSharp.ValueError">There are not two or three displayed coordinates, the target-name count is inconsistent, or an empty custom-target study has no target names.</exception>
    public static GenericChart PlotParetoFront(
        Study study,
        IReadOnlyList<string>? targetNames = null,
        bool includeDominatedTrials = true,
        Func<FrozenTrial, IReadOnlyList<double>>? constraintsFunc = null,
        Func<FrozenTrial, IReadOnlyList<double>>? targets = null)
        => ParetoFrontPlot.Render(
            ParetoFront.GetInfo(study, targetNames, includeDominatedTrials, constraintsFunc, targets));

    /// <inheritdoc cref="PlotParetoFront(Study, IReadOnlyList{string}, bool, Func{FrozenTrial, IReadOnlyList{double}}, Func{FrozenTrial, IReadOnlyList{double}})" />
    /// <param name="customization">Layout, interaction, and HTML display settings applied after creating the chart.</param>
    /// <param name="study">Study whose recorded trials are visualized.</param>
    /// <param name="targetNames">Two or three axis labels; required with custom targets when the study has no completed trials.</param>
    /// <param name="includeDominatedTrials">Whether feasible dominated trials are included alongside the Pareto front.</param>
    /// <param name="constraintsFunc">Deprecated constraint callback; positive values indicate violations. Null reads constraints stored on trials.</param>
    /// <param name="targets">Selector returning two or three displayed coordinates; null uses the study's objective values.</param>
    public static GenericChart PlotParetoFront(
        Study study,
        PlotlyChartCustomization customization,
        IReadOnlyList<string>? targetNames = null,
        bool includeDominatedTrials = true,
        Func<FrozenTrial, IReadOnlyList<double>>? constraintsFunc = null,
        Func<FrozenTrial, IReadOnlyList<double>>? targets = null)
        => PlotParetoFront(study, targetNames, includeDominatedTrials, constraintsFunc, targets)
            .WithCustomization(customization);
}
