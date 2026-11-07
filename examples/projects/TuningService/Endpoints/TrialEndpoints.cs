using System.Text.Json;
using Microsoft.AspNetCore.Http.HttpResults;
using NumSharp;
using OptunaSharp;
using OptunaSharp.Collections;
using OptunaSharp.Distributions;
using OptunaSharp.Pruners;
using OptunaSharp.Samplers;
using OptunaSharp.Storages;
using TuningCommon;
using TuningService.Contracts;

namespace TuningService.Endpoints;

/// <summary>
///     The trial endpoints under <c>/api/studies/{name}/trials</c>: Optuna's ask-and-tell interface over HTTP,
///     so a program in any language can let the service choose its parameters, report its progress, and have
///     the service's pruner decide whether to stop early.
/// </summary>
/// <remarks>
///     <para>
///         A client calls <c>POST …/trials</c> (ask) for parameters, evaluates them, optionally calls
///         <c>POST …/trials/{number}/report</c> after each step and stops if the answer says to prune, then calls
///         <c>POST …/trials/{number}/tell</c> with the result. Each call is a separate request, so each one loads
///         the study again: the storage holds the state between calls, exactly as between two
///         <c>optuna ask</c> and <c>optuna tell</c> command-line runs.
///     </para>
///     <para>
///         The handlers that sample, prune or finish a trial take a <see cref="BaseSampler" /> and a
///         <see cref="BasePruner" /> parameter. Both are registered transient, so every request gets fresh
///         instances of the configured algorithms. Samplers rebuild their model from the stored trials on each
///         call, so nothing is lost between requests; see <see cref="TuningServiceOptions.SamplerSeed" /> for the
///         one thing to avoid.
///     </para>
/// </remarks>
public static class TrialEndpoints
{
    /// <summary>Maps the trial endpoints.</summary>
    /// <param name="endpoints">The app's endpoint builder.</param>
    /// <returns>The same <paramref name="endpoints" />, for chaining.</returns>
    public static IEndpointRouteBuilder MapTrialEndpoints(this IEndpointRouteBuilder endpoints)
    {
        RouteGroupBuilder trials = endpoints.MapGroup("/api/studies/{name}/trials").WithTags("Trials");
        trials.MapGet("/", ListTrials);
        trials.MapPost("/", Ask);
        trials.MapGet("/{number:int}", GetTrial);
        trials.MapPost("/{number:int}/report", Report);
        trials.MapPost("/{number:int}/tell", Tell);
        return endpoints;
    }

    /// <summary><c>GET /api/studies/{name}/trials?state=…</c>: the study's trials, optionally of one state.</summary>
    /// <param name="name">The study name.</param>
    /// <param name="state">
    ///     Only trials in this state (<c>complete</c>, <c>pruned</c>, <c>fail</c>, <c>running</c> or
    ///     <c>waiting</c>, in any letter case), or all trials when omitted.
    /// </param>
    /// <param name="storage">The application's storage.</param>
    /// <returns>200 with the trials in number order; 400 for an unknown state.</returns>
    /// <exception cref="KeyError">No study has that name (answered 404).</exception>
    /// <remarks>
    ///     The state is bound as a string and parsed here because minimal APIs parse an enum in the query
    ///     string case-sensitively, and answer a mistyped <c>?state=pruned</c> with an empty 400. JSON bodies
    ///     do not have that problem: <c>JsonStringEnumConverter</c> reads enum names in any case.
    /// </remarks>
    private static Results<Ok<List<TrialDto>>, ValidationProblem> ListTrials(
        string name, string? state, BaseStorage storage)
    {
        TrialState[]? states = null;
        if (state is not null)
        {
            if (!Enum.TryParse(state, ignoreCase: true, out TrialState parsed) || !Enum.IsDefined(parsed))
            {
                return TypedResults.ValidationProblem(new Dictionary<string, string[]>
                {
                    ["state"] = [$"Expected one of {string.Join(", ", Enum.GetNames<TrialState>())}."],
                });
            }

            states = [parsed];
        }

        Study study = Optuna.LoadStudy(StudyRoute.Name(name), storage);
        return TypedResults.Ok(study.GetTrials(deepcopy: false, states: states).Select(TrialDto.From).ToList());
    }

    /// <summary><c>GET /api/studies/{name}/trials/{number}</c>: one trial.</summary>
    /// <param name="name">The study name.</param>
    /// <param name="number">The trial number.</param>
    /// <param name="storage">The application's storage.</param>
    /// <returns>200 with the trial.</returns>
    /// <exception cref="KeyError">No such study or trial (answered 404).</exception>
    private static Ok<TrialDto> GetTrial(string name, int number, BaseStorage storage)
        => TypedResults.Ok(TrialDto.From(storage.GetTrial(TrialId(storage, StudyRoute.Name(name), number))));

    /// <summary>
    ///     <c>POST /api/studies/{name}/trials</c>: asks the study for a new trial and samples the search space
    ///     the client sent, with the configured sampler.
    /// </summary>
    /// <param name="name">The study name.</param>
    /// <param name="request">The search space and an optional worker name; an empty body asks a trial without parameters.</param>
    /// <param name="storage">The application's storage.</param>
    /// <param name="sampler">This request's sampler, built by the container.</param>
    /// <param name="pruner">This request's pruner, built by the container.</param>
    /// <returns>201 with the trial number and its parameters; 400 naming each malformed distribution.</returns>
    /// <exception cref="KeyError">No study has that name (answered 404).</exception>
    private static Results<Created<AskResponse>, ValidationProblem> Ask(
        string name, AskRequest? request, BaseStorage storage, BaseSampler sampler, BasePruner pruner)
    {
        // parse every distribution before touching the study, so a bad request creates no trial
        var searchSpace = new OrderedDict<string, BaseDistribution>();
        var errors = new Dictionary<string, string[]>();
        foreach ((string param, JsonElement json) in request?.SearchSpace ?? new Dictionary<string, JsonElement>())
        {
            try
            {
                searchSpace[param] = Distribution.FromJson(json.GetRawText());
            }
            catch (Exception e) when (e is ValueError or TypeError or InvalidCastException
                                          or KeyNotFoundException or FormatException or OverflowException)
            {
                // FromJson's documented failures; caught here so that a missing "attributes" key (a
                // KeyNotFoundException) reads as a bad request, not as a missing study
                errors[$"searchSpace.{param}"] = [e.Message];
            }
        }

        if (errors.Count > 0)
            return TypedResults.ValidationProblem(errors);

        Study study = Optuna.LoadStudy(StudyRoute.Name(name), storage, sampler, pruner);
        Trial trial = study.Ask(searchSpace);   // samples every parameter of the search space now
        if (!string.IsNullOrWhiteSpace(request?.Worker))
            trial.SetUserAttr(SimulatedTraining.WorkerAttribute, request.Worker);

        return TypedResults.Created(
            $"{StudyRoute.Uri(study.StudyName)}/trials/{trial.Number}",
            new AskResponse(trial.Number, trial.Params));
    }

    /// <summary>
    ///     <c>POST /api/studies/{name}/trials/{number}/report</c>: records an intermediate value of a running
    ///     trial and asks the configured pruner whether to stop it.
    /// </summary>
    /// <param name="name">The study name.</param>
    /// <param name="number">The trial number from the ask call.</param>
    /// <param name="request">The step and its intermediate value.</param>
    /// <param name="storage">The application's storage.</param>
    /// <param name="sampler">This request's sampler; constructing a trial runs its before-trial hook.</param>
    /// <param name="pruner">This request's pruner, which decides.</param>
    /// <returns>200 with the pruner's decision.</returns>
    /// <exception cref="KeyError">No such study or trial (answered 404).</exception>
    /// <exception cref="UpdateFinishedTrialError">The trial is already finished (answered 409).</exception>
    /// <remarks>
    ///     A trial asked by an earlier request is reopened as Optuna's <c>Trial(study, trial_id)</c>, which
    ///     takes the storage's trial id. The id is looked up from the number, which is what clients know.
    /// </remarks>
    private static Ok<ReportResponse> Report(
        string name, int number, ReportRequest request, BaseStorage storage, BaseSampler sampler, BasePruner pruner)
    {
        Study study = Optuna.LoadStudy(StudyRoute.Name(name), storage, sampler, pruner);
        var trial = new Trial(study, TrialId(storage, StudyRoute.Name(name), number));
        trial.Report(request.Value, request.Step);
        return TypedResults.Ok(new ReportResponse(number, request.Step, trial.ShouldPrune()));
    }

    /// <summary>
    ///     <c>POST /api/studies/{name}/trials/{number}/tell</c>: finishes a trial with its objective value(s), or
    ///     as pruned or failed.
    /// </summary>
    /// <param name="name">The study name.</param>
    /// <param name="number">The trial number from the ask call.</param>
    /// <param name="request">The values, the state, or both.</param>
    /// <param name="storage">The application's storage.</param>
    /// <param name="sampler">This request's sampler; Tell runs its after-trial hook.</param>
    /// <param name="pruner">This request's pruner.</param>
    /// <returns>200 with the finished trial.</returns>
    /// <exception cref="KeyError">No such study or trial (answered 404).</exception>
    /// <exception cref="ValueError">
    ///     The trial is not running, or the values do not fit the state or the number of objectives (answered 400
    ///     with Optuna's message, such as <c>Cannot tell a COMPLETE trial.</c>).
    /// </exception>
    /// <remarks>
    ///     A trial told <c>Pruned</c> without values gets its last reported intermediate value as its value, as
    ///     in Optuna.
    /// </remarks>
    private static Ok<TrialDto> Tell(
        string name, int number, TellRequest request, BaseStorage storage, BaseSampler sampler, BasePruner pruner)
    {
        Study study = Optuna.LoadStudy(StudyRoute.Name(name), storage, sampler, pruner);
        FrozenTrial told = study.Tell(number, request.Values, request.State);
        return TypedResults.Ok(TrialDto.From(told));
    }

    /// <summary>Finds the storage's id of a study's trial from its number.</summary>
    /// <param name="storage">The application's storage.</param>
    /// <param name="name">The study name.</param>
    /// <param name="number">The trial number.</param>
    /// <returns>The trial id the storage uses internally.</returns>
    /// <exception cref="KeyError">No such study or trial.</exception>
    private static int TrialId(BaseStorage storage, string name, int number)
        => storage.GetTrialIdFromStudyIdTrialNumber(storage.GetStudyIdFromName(name), number);
}
