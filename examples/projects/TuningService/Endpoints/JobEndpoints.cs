using Microsoft.AspNetCore.Http.HttpResults;
using Microsoft.Extensions.Options;
using OptunaSharp;
using OptunaSharp.Storages;
using TuningService.Contracts;
using TuningService.Jobs;

namespace TuningService.Endpoints;

/// <summary>
///     The background-job endpoints: <c>POST /api/studies/{name}/optimize</c> queues a run of the built-in
///     objective, and <c>/api/jobs</c> lists, reads and cancels jobs. <see cref="OptimizationJobRunner" />
///     does the work.
/// </summary>
public static class JobEndpoints
{
    /// <summary>Maps the job endpoints.</summary>
    /// <param name="endpoints">The app's endpoint builder.</param>
    /// <returns>The same <paramref name="endpoints" />, for chaining.</returns>
    public static IEndpointRouteBuilder MapJobEndpoints(this IEndpointRouteBuilder endpoints)
    {
        endpoints.MapPost("/api/studies/{name}/optimize", Enqueue).WithTags("Jobs");

        RouteGroupBuilder jobs = endpoints.MapGroup("/api/jobs").WithTags("Jobs");
        jobs.MapGet("/", ListJobs);
        jobs.MapGet("/{id:guid}", GetJob);
        jobs.MapDelete("/{id:guid}", CancelJob);
        return endpoints;
    }

    /// <summary>
    ///     <c>POST /api/studies/{name}/optimize</c>: queues a job that optimizes the study with the built-in
    ///     objective. The study is created on first use.
    /// </summary>
    /// <param name="name">The study name.</param>
    /// <param name="request">The number of trials and the parallelism; an empty body takes the defaults (40 trials, 1 at a time).</param>
    /// <param name="queue">The job queue.</param>
    /// <param name="storage">The application's storage, to check an existing study's directions.</param>
    /// <param name="options">The service options with the per-job limits.</param>
    /// <returns>
    ///     202 with the queued job and its location; 400 for limits out of range or a multi-objective study;
    ///     429 when the queue is full.
    /// </returns>
    private static Results<Accepted<JobDto>, ValidationProblem, ProblemHttpResult> Enqueue(
        string name,
        OptimizeRequest? request,
        OptimizationJobQueue queue,
        BaseStorage storage,
        IOptions<TuningServiceOptions> options)
    {
        request ??= new OptimizeRequest();
        TuningServiceOptions limits = options.Value;
        string studyName = StudyRoute.Name(name);

        // validate here, while the client is still waiting: an error inside the job would only show up later
        var errors = new Dictionary<string, string[]>();
        if (request.NTrials < 1 || request.NTrials > limits.MaxTrialsPerJob)
            errors["nTrials"] = [$"Must be between 1 and {limits.MaxTrialsPerJob}."];
        if (request.NJobs < 1 || request.NJobs > limits.MaxParallelTrials)
            errors["nJobs"] = [$"Must be between 1 and {limits.MaxParallelTrials}."];
        if (errors.Count > 0)
            return TypedResults.ValidationProblem(errors);

        // the built-in objective returns one value, which a multi-objective study would store as a FAIL trial
        // on every trial; refuse such a study up front. GetAllStudies reads only the study table.
        List<StudyDirection>? directions = storage.GetAllStudies()
            .FirstOrDefault(s => s.StudyName == studyName)?.Directions;
        if (directions is { Count: > 1 })
        {
            return TypedResults.Problem(
                statusCode: StatusCodes.Status400BadRequest,
                title: "The study has several objectives",
                detail: $"The built-in objective is single-objective, and study '{studyName}' has {directions.Count}.");
        }

        var job = new OptimizationJob(studyName, request.NTrials, request.NJobs);
        if (!queue.TryEnqueue(job))
        {
            return TypedResults.Problem(
                statusCode: StatusCodes.Status429TooManyRequests,
                title: "Too many queued jobs",
                detail: $"{limits.MaxQueuedJobs} jobs are already waiting; try again when one has started.");
        }

        return TypedResults.Accepted(JobUri(job.Id), job.ToDto());
    }

    /// <summary><c>GET /api/jobs</c>: every job since the service started, oldest first.</summary>
    /// <param name="queue">The job queue.</param>
    /// <returns>200 with the jobs.</returns>
    private static Ok<List<JobDto>> ListJobs(OptimizationJobQueue queue)
        => TypedResults.Ok(queue.Snapshot().Select(j => j.ToDto()).ToList());

    /// <summary><c>GET /api/jobs/{id}</c>: one job's state and progress.</summary>
    /// <param name="id">The job identifier.</param>
    /// <param name="queue">The job queue.</param>
    /// <returns>200 with the job, or 404.</returns>
    private static Results<Ok<JobDto>, NotFound> GetJob(Guid id, OptimizationJobQueue queue)
        => queue.TryGet(id, out OptimizationJob? job) ? TypedResults.Ok(job.ToDto()) : TypedResults.NotFound();

    /// <summary>
    ///     <c>DELETE /api/jobs/{id}</c>: cancels a job. A queued job never starts; a running one stops after its
    ///     trials in flight finish.
    /// </summary>
    /// <param name="id">The job identifier.</param>
    /// <param name="queue">The job queue.</param>
    /// <returns>202 with the job; 404 for an unknown job; 409 for a job that already finished.</returns>
    private static Results<Accepted<JobDto>, NotFound, Conflict<JobDto>> CancelJob(Guid id, OptimizationJobQueue queue)
    {
        if (!queue.TryGet(id, out OptimizationJob? job))
            return TypedResults.NotFound();
        return job.Cancel()
            ? TypedResults.Accepted(JobUri(id), job.ToDto())
            : TypedResults.Conflict(job.ToDto());
    }

    /// <summary>The API location of a job.</summary>
    /// <param name="id">The job identifier.</param>
    /// <returns>The relative URI <c>/api/jobs/{id}</c>.</returns>
    private static string JobUri(Guid id) => $"/api/jobs/{id}";
}
