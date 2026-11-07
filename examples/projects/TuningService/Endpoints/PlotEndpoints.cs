using Microsoft.AspNetCore.Http.HttpResults;
using OptunaSharp;
using OptunaSharp.Storages;
using OptunaSharp.Visualization.Plotly.Extensions;
using Plotly.NET;

namespace TuningService.Endpoints;

/// <summary>
///     The chart endpoints: <c>GET /studies/{name}/plots/{kind}</c> returns an interactive HTML page, and
///     <c>GET /api/studies/{name}/plots/{kind}</c> returns the same figure as Plotly JSON for a front end to
///     render with plotly.js.
/// </summary>
/// <remarks>
///     The charts come from <see cref="IOptunaPlotlyVisualizer" />, the injectable face of OptunaSharp's
///     Plotly backend. It is registered with the app's house style (see Program.cs), so every chart gets it
///     without the endpoints knowing about it. The figures match what Optuna's <c>optuna.visualization</c>
///     draws for the same study.
/// </remarks>
public static class PlotEndpoints
{
    /// <summary>The chart kinds that apply to a single-objective study, in the order the dashboard lists them.</summary>
    public static readonly IReadOnlyList<string> SingleObjectiveKinds =
        ["history", "intermediate", "slice", "parallel-coordinate", "contour", "importance", "edf", "rank", "timeline"];

    /// <summary>The chart kinds that apply to a multi-objective study.</summary>
    public static readonly IReadOnlyList<string> MultiObjectiveKinds = ["pareto-front", "timeline"];

    /// <summary>Maps the chart endpoints.</summary>
    /// <param name="endpoints">The app's endpoint builder.</param>
    /// <returns>The same <paramref name="endpoints" />, for chaining.</returns>
    public static IEndpointRouteBuilder MapPlotEndpoints(this IEndpointRouteBuilder endpoints)
    {
        endpoints.MapGet("/studies/{name}/plots/{kind}", PlotHtml).WithTags("Charts");
        endpoints.MapGet("/api/studies/{name}/plots/{kind}", PlotJson).WithTags("Charts");
        return endpoints;
    }

    /// <summary><c>GET /studies/{name}/plots/{kind}</c>: a self-contained HTML page with one interactive chart.</summary>
    /// <param name="name">The study name.</param>
    /// <param name="kind">The chart kind, one of <see cref="SingleObjectiveKinds" /> or <see cref="MultiObjectiveKinds" />.</param>
    /// <param name="storage">The application's storage.</param>
    /// <param name="plots">The Plotly visualizer.</param>
    /// <returns>200 with the page, or 404 for an unknown kind.</returns>
    /// <exception cref="NumSharp.KeyError">No study has that name (answered 404).</exception>
    /// <exception cref="NumSharp.ValueError">The chart does not apply to this study, e.g. a Pareto front of one objective (answered 400).</exception>
    /// <remarks>The page loads plotly.js from Plotly's CDN, so the browser needs internet access to draw it.</remarks>
    private static Results<ContentHttpResult, NotFound> PlotHtml(
        string name, string kind, BaseStorage storage, IOptunaPlotlyVisualizer plots)
    {
        GenericChart? chart = Build(kind, Optuna.LoadStudy(StudyRoute.Name(name), storage), plots);
        return chart is null
            ? TypedResults.NotFound()
            : TypedResults.Content(GenericChart.toEmbeddedHTML(chart), "text/html; charset=utf-8");
    }

    /// <summary><c>GET /api/studies/{name}/plots/{kind}</c>: one chart as a Plotly figure (<c>{"data": …, "layout": …}</c>).</summary>
    /// <param name="name">The study name.</param>
    /// <param name="kind">The chart kind.</param>
    /// <param name="storage">The application's storage.</param>
    /// <param name="plots">The Plotly visualizer.</param>
    /// <returns>200 with the figure JSON, or 404 for an unknown kind.</returns>
    /// <exception cref="NumSharp.KeyError">No study has that name (answered 404).</exception>
    /// <exception cref="NumSharp.ValueError">The chart does not apply to this study (answered 400).</exception>
    private static Results<ContentHttpResult, NotFound> PlotJson(
        string name, string kind, BaseStorage storage, IOptunaPlotlyVisualizer plots)
    {
        GenericChart? chart = Build(kind, Optuna.LoadStudy(StudyRoute.Name(name), storage), plots);
        return chart is null
            ? TypedResults.NotFound()
            : TypedResults.Content(GenericChart.toFigureJson(chart), "application/json");
    }

    /// <summary>Builds one chart of a study.</summary>
    /// <param name="kind">The chart kind.</param>
    /// <param name="study">The study to chart.</param>
    /// <param name="plots">The Plotly visualizer.</param>
    /// <returns>The chart, or <c>null</c> for an unknown kind.</returns>
    /// <exception cref="NumSharp.ValueError">The chart does not apply to this study.</exception>
    private static GenericChart? Build(string kind, Study study, IOptunaPlotlyVisualizer plots) => kind switch
    {
        "history" => plots.PlotOptimizationHistory(study),
        "intermediate" => plots.PlotIntermediateValues(study),
        "slice" => plots.PlotSlice(study),
        "parallel-coordinate" => plots.PlotParallelCoordinate(study),
        "contour" => plots.PlotContour(study),
        "importance" => plots.PlotParamImportances(study),
        "edf" => plots.PlotEdf(study),
        "rank" => plots.PlotRank(study),
        "timeline" => plots.PlotTimeline(study),
        "pareto-front" => plots.PlotParetoFront(study),
        _ => null,
    };
}
