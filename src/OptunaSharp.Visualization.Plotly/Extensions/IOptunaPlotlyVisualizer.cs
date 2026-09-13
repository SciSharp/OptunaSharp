using System;
using System.Collections.Generic;
using OptunaSharp.Importance;
using OptunaSharp.Terminator;
using Plotly.NET;

namespace OptunaSharp.Visualization.Plotly.Extensions;

/// <summary>
///     An injectable instance facade over the static <see cref="PlotlyVisualization" /> — the DI-friendly
///     shape of the plotly backend. Each method mirrors the matching <c>PlotlyVisualization.Plot*</c> and
///     automatically applies the visualizer's configured default <see cref="PlotlyChartCustomization" />
///     (the app-wide figure styling registered with <c>AddOptunaPlotlyVisualization</c>), so callers get
///     consistently-styled figures without threading a customization through every call.
///     <para>
///     <b>Why an interface exists at all:</b> the visualization API is static and stateless, so this
///     facade adds no behavior of its own beyond the default-customization convenience — its real value is
///     testability (mock <c>IOptunaPlotlyVisualizer</c> to assert your code requested the right figure) and
///     a single injected place to carry house chart styling. If you do not need either, call
///     <see cref="PlotlyVisualization" /> directly.
///     </para>
/// </summary>
public interface IOptunaPlotlyVisualizer
{
    /// <summary>The default customization applied to every figure this visualizer produces, or <c>null</c> when none was configured.</summary>
    PlotlyChartCustomization? DefaultCustomization { get; }

    /// <summary><c>plot_optimization_history</c> for a single study.</summary>
    /// <param name="study">The study whose trials are plotted.</param>
    /// <param name="target">Scalar value selector; <c>null</c> uses the single objective (required for multi-objective).</param>
    /// <param name="targetName">Axis/legend label for the selected value.</param>
    /// <param name="errorBar">Whether to aggregate across studies and show error bars.</param>
    /// <returns>A styled Plotly.NET chart.</returns>
    GenericChart PlotOptimizationHistory(Study study, Func<FrozenTrial, double>? target = null, string targetName = "Objective Value", bool errorBar = false);

    /// <summary><c>plot_optimization_history</c> for several studies (overlaid/aggregated).</summary>
    /// <param name="studies">Studies to overlay or aggregate.</param>
    /// <param name="target">Scalar value selector; <c>null</c> uses the single objective.</param>
    /// <param name="targetName">Axis/legend label for the selected value.</param>
    /// <param name="errorBar">Whether to aggregate across studies and show error bars.</param>
    /// <returns>A styled Plotly.NET chart.</returns>
    GenericChart PlotOptimizationHistory(IReadOnlyList<Study> studies, Func<FrozenTrial, double>? target = null, string targetName = "Objective Value", bool errorBar = false);

    /// <summary><c>plot_edf</c> for a single study (empirical distribution function).</summary>
    /// <param name="study">The study whose trials are plotted.</param>
    /// <param name="target">Scalar value selector; <c>null</c> uses the single objective.</param>
    /// <param name="targetName">Axis/legend label for the selected value.</param>
    /// <returns>A styled Plotly.NET chart.</returns>
    GenericChart PlotEdf(Study study, Func<FrozenTrial, double>? target = null, string targetName = "Objective Value");

    /// <summary><c>plot_edf</c> for several studies (compares their EDFs).</summary>
    /// <param name="studies">Studies to compare.</param>
    /// <param name="target">Scalar value selector; <c>null</c> uses the single objective.</param>
    /// <param name="targetName">Axis/legend label for the selected value.</param>
    /// <returns>A styled Plotly.NET chart.</returns>
    GenericChart PlotEdf(IReadOnlyList<Study> studies, Func<FrozenTrial, double>? target = null, string targetName = "Objective Value");

    /// <summary><c>plot_intermediate_values</c> — each trial's reported intermediate values.</summary>
    /// <param name="study">The study whose trials are plotted.</param>
    /// <returns>A styled Plotly.NET chart.</returns>
    GenericChart PlotIntermediateValues(Study study);

    /// <summary><c>plot_hypervolume_history</c> — running hypervolume of a multi-objective study.</summary>
    /// <param name="study">The multi-objective study whose trials are plotted.</param>
    /// <param name="referencePoint">Reference coordinates in objective order; length must match the objective count.</param>
    /// <returns>A styled Plotly.NET chart.</returns>
    GenericChart PlotHypervolumeHistory(Study study, IReadOnlyList<double> referencePoint);

    /// <summary><c>plot_timeline</c> — each trial's lifetime as a bar, colored by state.</summary>
    /// <param name="study">The study whose trials are plotted.</param>
    /// <param name="nRecentTrials">Positive count of most recent trials to show, or <c>null</c> for all.</param>
    /// <returns>A styled Plotly.NET chart.</returns>
    GenericChart PlotTimeline(Study study, int? nRecentTrials = null);

    /// <summary><c>plot_param_importances</c> — a bar chart of hyperparameter importances.</summary>
    /// <param name="study">The study whose trials are plotted.</param>
    /// <param name="evaluator">Importance evaluator; <c>null</c> uses the Core default.</param>
    /// <param name="params">Parameter names to include; <c>null</c> selects all eligible.</param>
    /// <param name="target">Scalar value selector; <c>null</c> uses the single objective.</param>
    /// <param name="targetName">Axis/legend label for the selected value.</param>
    /// <returns>A styled Plotly.NET chart.</returns>
    GenericChart PlotParamImportances(Study study, BaseImportanceEvaluator? evaluator = null, IReadOnlyList<string>? @params = null, Func<FrozenTrial, double>? target = null, string targetName = "Objective Value");

    /// <summary><c>plot_terminator_improvement</c> — the terminator's improvement (and optionally CV error) over trials.</summary>
    /// <param name="study">The single-objective study whose trials are plotted.</param>
    /// <param name="plotError">Whether to also compute and plot an error estimate.</param>
    /// <param name="improvementEvaluator">Improvement-bound evaluator; <c>null</c> uses the default.</param>
    /// <param name="errorEvaluator">Error evaluator; <c>null</c> selects the default for the improvement evaluator.</param>
    /// <param name="minNTrials">Initial low-opacity region excluded from the later improvement maximum.</param>
    /// <returns>A styled Plotly.NET chart.</returns>
    GenericChart PlotTerminatorImprovement(Study study, bool plotError = false, BaseImprovementEvaluator? improvementEvaluator = null, BaseErrorEvaluator? errorEvaluator = null, int minNTrials = 20);

    /// <summary><c>plot_contour</c> — parameter relationships as contour plots.</summary>
    /// <param name="study">The study whose trials are plotted.</param>
    /// <param name="params">Parameter names to include; <c>null</c> selects all eligible.</param>
    /// <param name="target">Scalar value selector; <c>null</c> uses the single objective.</param>
    /// <param name="targetName">Axis/legend label for the selected value.</param>
    /// <returns>A styled Plotly.NET chart.</returns>
    GenericChart PlotContour(Study study, IReadOnlyList<string>? @params = null, Func<FrozenTrial, double>? target = null, string targetName = "Objective Value");

    /// <summary><c>plot_rank</c> — (x,y) parameter scatters colored by target rank.</summary>
    /// <param name="study">The study whose trials are plotted.</param>
    /// <param name="params">Parameter names to include; <c>null</c> selects all eligible.</param>
    /// <param name="target">Scalar value selector; <c>null</c> uses the single objective.</param>
    /// <param name="targetName">Axis/legend label for the selected value.</param>
    /// <returns>A styled Plotly.NET chart.</returns>
    GenericChart PlotRank(Study study, IReadOnlyList<string>? @params = null, Func<FrozenTrial, double>? target = null, string targetName = "Objective Value");

    /// <summary><c>plot_parallel_coordinate</c> — high-dimensional parameter relationships.</summary>
    /// <param name="study">The study whose trials are plotted.</param>
    /// <param name="params">Parameter names to include; <c>null</c> selects all eligible.</param>
    /// <param name="target">Scalar value selector; <c>null</c> uses the single objective.</param>
    /// <param name="targetName">Axis/legend label for the selected value.</param>
    /// <returns>A styled Plotly.NET chart.</returns>
    GenericChart PlotParallelCoordinate(Study study, IReadOnlyList<string>? @params = null, Func<FrozenTrial, double>? target = null, string targetName = "Objective Value");

    /// <summary><c>plot_slice</c> — one subplot per parameter vs the target value.</summary>
    /// <param name="study">The study whose trials are plotted.</param>
    /// <param name="params">Parameter names to include; <c>null</c> selects all eligible.</param>
    /// <param name="target">Scalar value selector; <c>null</c> uses the single objective.</param>
    /// <param name="targetName">Axis/legend label for the selected value.</param>
    /// <returns>A styled Plotly.NET chart.</returns>
    GenericChart PlotSlice(Study study, IReadOnlyList<string>? @params = null, Func<FrozenTrial, double>? target = null, string targetName = "Objective Value");

    /// <summary><c>plot_pareto_front</c> — the Pareto front of a 2- or 3-objective study.</summary>
    /// <param name="study">The multi-objective study whose trials are plotted.</param>
    /// <param name="targetNames">Two or three axis labels.</param>
    /// <param name="includeDominatedTrials">Whether feasible dominated trials are shown alongside the front.</param>
    /// <param name="constraintsFunc">Deprecated constraint callback; positive values mark violations. <c>null</c> reads trial constraints.</param>
    /// <param name="targets">Selector returning two or three displayed coordinates; <c>null</c> uses the objective values.</param>
    /// <returns>A styled Plotly.NET chart.</returns>
    GenericChart PlotParetoFront(Study study, IReadOnlyList<string>? targetNames = null, bool includeDominatedTrials = true, Func<FrozenTrial, IReadOnlyList<double>>? constraintsFunc = null, Func<FrozenTrial, IReadOnlyList<double>>? targets = null);
}
