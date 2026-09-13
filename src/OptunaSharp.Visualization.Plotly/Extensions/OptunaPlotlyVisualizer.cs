using System;
using System.Collections.Generic;
using OptunaSharp.Importance;
using OptunaSharp.Terminator;
using Plotly.NET;

namespace OptunaSharp.Visualization.Plotly.Extensions;

/// <summary>
///     The default <see cref="IOptunaPlotlyVisualizer" />: a thin instance facade that delegates to the
///     static <see cref="PlotlyVisualization" /> and, when a default <see cref="PlotlyChartCustomization" />
///     was configured, routes each call through the matching customization overload so every figure picks
///     up the app-wide styling. Registered by
///     <see cref="PlotlyVisualizationServiceCollectionExtensions.AddOptunaPlotlyVisualization" />; construct
///     it directly only when composing a container by hand.
/// </summary>
public sealed class OptunaPlotlyVisualizer : IOptunaPlotlyVisualizer
{
    private readonly PlotlyChartCustomization? _customization;

    /// <summary>Creates a visualizer that applies <paramref name="defaultCustomization" /> to every figure it produces.</summary>
    /// <param name="defaultCustomization">The app-wide figure styling, or <c>null</c> to produce canonical (unstyled) figures.</param>
    public OptunaPlotlyVisualizer(PlotlyChartCustomization? defaultCustomization = null)
        => _customization = defaultCustomization;

    /// <inheritdoc />
    public PlotlyChartCustomization? DefaultCustomization => _customization;

    /// <inheritdoc />
    public GenericChart PlotOptimizationHistory(Study study, Func<FrozenTrial, double>? target = null, string targetName = "Objective Value", bool errorBar = false)
        => _customization is null
            ? PlotlyVisualization.PlotOptimizationHistory(study, target, targetName, errorBar)
            : PlotlyVisualization.PlotOptimizationHistory(study, _customization, target, targetName, errorBar);

    /// <inheritdoc />
    public GenericChart PlotOptimizationHistory(IReadOnlyList<Study> studies, Func<FrozenTrial, double>? target = null, string targetName = "Objective Value", bool errorBar = false)
        => _customization is null
            ? PlotlyVisualization.PlotOptimizationHistory(studies, target, targetName, errorBar)
            : PlotlyVisualization.PlotOptimizationHistory(studies, _customization, target, targetName, errorBar);

    /// <inheritdoc />
    public GenericChart PlotEdf(Study study, Func<FrozenTrial, double>? target = null, string targetName = "Objective Value")
        => _customization is null
            ? PlotlyVisualization.PlotEdf(study, target, targetName)
            : PlotlyVisualization.PlotEdf(study, _customization, target, targetName);

    /// <inheritdoc />
    public GenericChart PlotEdf(IReadOnlyList<Study> studies, Func<FrozenTrial, double>? target = null, string targetName = "Objective Value")
        => _customization is null
            ? PlotlyVisualization.PlotEdf(studies, target, targetName)
            : PlotlyVisualization.PlotEdf(studies, _customization, target, targetName);

    /// <inheritdoc />
    public GenericChart PlotIntermediateValues(Study study)
        => _customization is null
            ? PlotlyVisualization.PlotIntermediateValues(study)
            : PlotlyVisualization.PlotIntermediateValues(study, _customization);

    /// <inheritdoc />
    public GenericChart PlotHypervolumeHistory(Study study, IReadOnlyList<double> referencePoint)
        => _customization is null
            ? PlotlyVisualization.PlotHypervolumeHistory(study, referencePoint)
            : PlotlyVisualization.PlotHypervolumeHistory(study, referencePoint, _customization);

    /// <inheritdoc />
    public GenericChart PlotTimeline(Study study, int? nRecentTrials = null)
        => _customization is null
            ? PlotlyVisualization.PlotTimeline(study, nRecentTrials)
            : PlotlyVisualization.PlotTimeline(study, _customization, nRecentTrials);

    /// <inheritdoc />
    public GenericChart PlotParamImportances(Study study, BaseImportanceEvaluator? evaluator = null, IReadOnlyList<string>? @params = null, Func<FrozenTrial, double>? target = null, string targetName = "Objective Value")
        => _customization is null
            ? PlotlyVisualization.PlotParamImportances(study, evaluator, @params, target, targetName)
            : PlotlyVisualization.PlotParamImportances(study, _customization, evaluator, @params, target, targetName);

    /// <inheritdoc />
    public GenericChart PlotTerminatorImprovement(Study study, bool plotError = false, BaseImprovementEvaluator? improvementEvaluator = null, BaseErrorEvaluator? errorEvaluator = null, int minNTrials = 20)
        => _customization is null
            ? PlotlyVisualization.PlotTerminatorImprovement(study, plotError, improvementEvaluator, errorEvaluator, minNTrials)
            : PlotlyVisualization.PlotTerminatorImprovement(study, _customization, plotError, improvementEvaluator, errorEvaluator, minNTrials);

    /// <inheritdoc />
    public GenericChart PlotContour(Study study, IReadOnlyList<string>? @params = null, Func<FrozenTrial, double>? target = null, string targetName = "Objective Value")
        => _customization is null
            ? PlotlyVisualization.PlotContour(study, @params, target, targetName)
            : PlotlyVisualization.PlotContour(study, _customization, @params, target, targetName);

    /// <inheritdoc />
    public GenericChart PlotRank(Study study, IReadOnlyList<string>? @params = null, Func<FrozenTrial, double>? target = null, string targetName = "Objective Value")
        => _customization is null
            ? PlotlyVisualization.PlotRank(study, @params, target, targetName)
            : PlotlyVisualization.PlotRank(study, _customization, @params, target, targetName);

    /// <inheritdoc />
    public GenericChart PlotParallelCoordinate(Study study, IReadOnlyList<string>? @params = null, Func<FrozenTrial, double>? target = null, string targetName = "Objective Value")
        => _customization is null
            ? PlotlyVisualization.PlotParallelCoordinate(study, @params, target, targetName)
            : PlotlyVisualization.PlotParallelCoordinate(study, _customization, @params, target, targetName);

    /// <inheritdoc />
    public GenericChart PlotSlice(Study study, IReadOnlyList<string>? @params = null, Func<FrozenTrial, double>? target = null, string targetName = "Objective Value")
        => _customization is null
            ? PlotlyVisualization.PlotSlice(study, @params, target, targetName)
            : PlotlyVisualization.PlotSlice(study, _customization, @params, target, targetName);

    /// <inheritdoc />
    public GenericChart PlotParetoFront(Study study, IReadOnlyList<string>? targetNames = null, bool includeDominatedTrials = true, Func<FrozenTrial, IReadOnlyList<double>>? constraintsFunc = null, Func<FrozenTrial, IReadOnlyList<double>>? targets = null)
        => _customization is null
            ? PlotlyVisualization.PlotParetoFront(study, targetNames, includeDominatedTrials, constraintsFunc, targets)
            : PlotlyVisualization.PlotParetoFront(study, _customization, targetNames, includeDominatedTrials, constraintsFunc, targets);
}
