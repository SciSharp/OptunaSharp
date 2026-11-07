using System.Globalization;
using System.Text;
using System.Text.Encodings.Web;
using Microsoft.AspNetCore.Http.HttpResults;
using Microsoft.Extensions.Options;
using OptunaSharp;
using OptunaSharp.Storages;
using TuningService.Contracts;
using TuningService.Jobs;

namespace TuningService.Endpoints;

/// <summary>
///     <c>GET /</c>: a small dashboard that lists every study with its best value and links to its charts, and
///     every background job with its progress. It reloads itself every five seconds, so a running job, a
///     TuningWorker or a Python worker can be watched live.
/// </summary>
/// <remarks>
///     The page is plain server-built HTML, to keep the example free of front-end tooling. Study names come
///     from clients, so every value is HTML-encoded and every link URL-escaped.
/// </remarks>
public static class DashboardEndpoints
{
    /// <summary>Maps the dashboard.</summary>
    /// <param name="endpoints">The app's endpoint builder.</param>
    /// <returns>The same <paramref name="endpoints" />, for chaining.</returns>
    public static IEndpointRouteBuilder MapDashboard(this IEndpointRouteBuilder endpoints)
    {
        endpoints.MapGet("/", Dashboard).ExcludeFromDescription();
        return endpoints;
    }

    /// <summary>Renders the dashboard.</summary>
    /// <param name="storage">The application's storage.</param>
    /// <param name="queue">The job queue.</param>
    /// <param name="options">The service options; only the storage URL's scheme is shown.</param>
    /// <returns>200 with the HTML page.</returns>
    private static ContentHttpResult Dashboard(
        BaseStorage storage, OptimizationJobQueue queue, IOptions<TuningServiceOptions> options)
    {
        HtmlEncoder html = HtmlEncoder.Default;
        var page = new StringBuilder();
        page.Append("""
            <!doctype html>
            <html lang="en">
            <head>
              <meta charset="utf-8">
              <meta http-equiv="refresh" content="5">
              <title>OptunaSharp tuning service</title>
              <style>
                body { font-family: system-ui, sans-serif; margin: 2rem; color: #1f2328; }
                table { border-collapse: collapse; margin-bottom: 2rem; }
                th, td { border-bottom: 1px solid #d0d7de; padding: .4rem .8rem; text-align: left; }
                th { background: #f6f8fa; }
                .muted { color: #59636e; }
                a { color: #0969da; }
              </style>
            </head>
            <body>
              <h1>OptunaSharp tuning service</h1>
            """);

        // only the scheme: a database URL can carry a password, which must never reach a web page
        string scheme = options.Value.Storage.Split("://", 2)[0];
        page.Append(CultureInfo.InvariantCulture, $"""
              <p class="muted">Storage: {html.Encode(scheme)} · gRPC storage proxy on port 13000 by default ·
                <a href="/api/studies">/api/studies</a> · <a href="/api/jobs">/api/jobs</a> · <a href="/health">/health</a></p>
              <h2>Studies</h2>
            """);
        AppendStudies(page, storage, html);

        page.Append("<h2>Background jobs</h2>");
        AppendJobs(page, queue, html);

        page.Append("</body></html>");
        return TypedResults.Content(page.ToString(), "text/html; charset=utf-8");
    }

    /// <summary>Appends the studies table, or a hint when there are no studies yet.</summary>
    /// <param name="page">The page being built.</param>
    /// <param name="storage">The application's storage.</param>
    /// <param name="html">The HTML encoder for client-supplied text.</param>
    private static void AppendStudies(StringBuilder page, BaseStorage storage, HtmlEncoder html)
    {
        List<StudySummary> studies = Optuna.GetAllStudySummaries(storage);
        if (studies.Count == 0)
        {
            page.Append("""
                <p>No studies yet. Start one with
                <code>curl -X POST http://localhost:5080/api/studies/demo/optimize -H "Content-Type: application/json" -d "{\"nTrials\": 40, \"nJobs\": 4}"</code>
                or run a TuningWorker.</p>
                """);
            return;
        }

        page.Append("<table><tr><th>Study</th><th>Directions</th><th>Trials</th><th>Best value</th><th>Charts</th></tr>");
        foreach (StudySummary study in studies)
        {
            string escaped = StudyRoute.Escape(study.StudyName);
            IReadOnlyList<string> kinds = study.Directions.Count == 1
                ? PlotEndpoints.SingleObjectiveKinds
                : PlotEndpoints.MultiObjectiveKinds;
            string charts = string.Join(" · ", kinds.Select(k => $"<a href=\"/studies/{escaped}/plots/{k}\">{k}</a>"));
            string best = study.BestTrial?.Value?.ToString("G6", CultureInfo.InvariantCulture) ?? "–";

            page.Append(CultureInfo.InvariantCulture, $"""
                <tr>
                  <td><a href="/api/studies/{escaped}">{html.Encode(study.StudyName)}</a>
                      (<a href="/api/studies/{escaped}/trials">trials</a>)</td>
                  <td>{string.Join(", ", study.Directions)}</td>
                  <td>{study.NTrials}</td>
                  <td>{best}</td>
                  <td>{charts}</td>
                </tr>
                """);
        }

        page.Append("</table>");
    }

    /// <summary>Appends the jobs table, or a note when no job ran yet.</summary>
    /// <param name="page">The page being built.</param>
    /// <param name="queue">The job queue.</param>
    /// <param name="html">The HTML encoder for client-supplied text.</param>
    private static void AppendJobs(StringBuilder page, OptimizationJobQueue queue, HtmlEncoder html)
    {
        IReadOnlyList<OptimizationJob> jobs = queue.Snapshot();
        if (jobs.Count == 0)
        {
            page.Append("<p class=\"muted\">No jobs since the service started.</p>");
            return;
        }

        page.Append("<table><tr><th>Job</th><th>Study</th><th>State</th><th>Complete</th><th>Pruned</th><th>Failed</th><th>Trials</th></tr>");
        // newest first; OrderByDescending rather than Reverse(), which C# 14 binds to the in-place Span
        // overload whenever the source is an array
        foreach (OptimizationJob job in jobs.OrderByDescending(j => j.QueuedAt))
        {
            JobDto dto = job.ToDto();
            page.Append(CultureInfo.InvariantCulture, $"""
                <tr>
                  <td><a href="/api/jobs/{dto.Id}">{dto.Id.ToString()[..8]}</a></td>
                  <td>{html.Encode(dto.StudyName)}</td>
                  <td>{dto.State}</td>
                  <td>{dto.Complete}</td>
                  <td>{dto.Pruned}</td>
                  <td>{dto.Failed}</td>
                  <td>{dto.NTrials} ({dto.NJobs} in parallel)</td>
                </tr>
                """);
        }

        page.Append("</table>");
    }
}
