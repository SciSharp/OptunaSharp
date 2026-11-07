using Microsoft.AspNetCore.Http.HttpResults;
using OptunaSharp;
using OptunaSharp.Extensions;
using OptunaSharp.Importance;
using OptunaSharp.Importance.PedAnova;
using OptunaSharp.Importance.Shap;
using OptunaSharp.Storages;
using TuningService.Contracts;

namespace TuningService.Endpoints;

/// <summary>
///     The study endpoints under <c>/api/studies</c>: list, create, read and delete studies, and compute
///     parameter importance.
/// </summary>
/// <remarks>
///     Each handler takes the OptunaSharp services it needs as parameters, and minimal APIs inject them from
///     the request's scope: the singleton <see cref="BaseStorage" /> to read, the transient
///     <see cref="IOptunaStudyFactory" /> to create. Errors are not caught here: OptunaSharp raises Optuna's
///     exceptions (a missing study is a <c>KeyError</c>, a duplicate name a <c>DuplicatedStudyError</c>), and
///     <c>OptunaExceptionHandler</c> turns them into 404, 409 and 400 problem responses.
/// </remarks>
public static class StudyEndpoints
{
    /// <summary>Maps the study endpoints.</summary>
    /// <param name="endpoints">The app's endpoint builder.</param>
    /// <returns>The same <paramref name="endpoints" />, for chaining.</returns>
    public static IEndpointRouteBuilder MapStudyEndpoints(this IEndpointRouteBuilder endpoints)
    {
        RouteGroupBuilder studies = endpoints.MapGroup("/api/studies").WithTags("Studies");
        studies.MapGet("/", ListStudies);
        studies.MapPost("/", CreateStudy);
        studies.MapGet("/{name}", GetStudy);
        studies.MapDelete("/{name}", DeleteStudy);
        studies.MapGet("/{name}/importance", GetImportance);
        return endpoints;
    }

    /// <summary><c>GET /api/studies</c>: every study in the storage, with its best value.</summary>
    /// <param name="storage">The application's storage.</param>
    /// <returns>200 with one summary per study.</returns>
    private static Ok<List<StudySummaryDto>> ListStudies(BaseStorage storage)
        => TypedResults.Ok(Optuna.GetAllStudySummaries(storage).Select(StudySummaryDto.From).ToList());

    /// <summary>
    ///     <c>POST /api/studies</c>: creates a study through the container's <see cref="IOptunaStudyFactory" />,
    ///     so it lives in the configured storage.
    /// </summary>
    /// <param name="request">The name and direction(s) of the new study.</param>
    /// <param name="studies">The study factory.</param>
    /// <returns>201 with the new study and its location.</returns>
    /// <exception cref="DuplicatedStudyError">A study with that name exists (answered 409).</exception>
    /// <exception cref="NumSharp.ValueError">
    ///     Both <c>direction</c> and <c>directions</c> are set, or <c>directions</c> is empty (answered 400 with
    ///     Optuna's message).
    /// </exception>
    private static Created<StudyDto> CreateStudy(CreateStudyRequest request, IOptunaStudyFactory studies)
    {
        // loadIfExists: false makes a duplicate name an error, as POST should; the background jobs pass true
        Study study = studies.CreateStudy(
            studyName: request.Name,
            direction: request.Direction,
            loadIfExists: false,
            directions: request.Directions);
        return TypedResults.Created(StudyRoute.Uri(study.StudyName), StudyDto.From(study));
    }

    /// <summary><c>GET /api/studies/{name}</c>: one study, with its trial counts and best trial(s).</summary>
    /// <param name="name">The study name.</param>
    /// <param name="storage">The application's storage.</param>
    /// <returns>200 with the study.</returns>
    /// <exception cref="NumSharp.KeyError">No study has that name (answered 404).</exception>
    private static Ok<StudyDto> GetStudy(string name, BaseStorage storage)
        => TypedResults.Ok(StudyDto.From(Optuna.LoadStudy(StudyRoute.Name(name), storage)));

    /// <summary><c>DELETE /api/studies/{name}</c>: deletes a study and all its trials.</summary>
    /// <param name="name">The study name.</param>
    /// <param name="storage">The application's storage.</param>
    /// <returns>204.</returns>
    /// <exception cref="NumSharp.KeyError">No study has that name (answered 404).</exception>
    private static NoContent DeleteStudy(string name, BaseStorage storage)
    {
        Optuna.DeleteStudy(StudyRoute.Name(name), storage);
        return TypedResults.NoContent();
    }

    /// <summary>
    ///     <c>GET /api/studies/{name}/importance?evaluator=…</c>: how much each parameter matters to the
    ///     objective, from the study's complete trials.
    /// </summary>
    /// <param name="name">The study name.</param>
    /// <param name="evaluator">
    ///     <c>ped-anova</c> (Optuna's default when omitted), <c>fanova</c>, <c>mdi</c> or <c>shap</c>.
    /// </param>
    /// <param name="storage">The application's storage.</param>
    /// <returns>200 with the importances, most important first; 400 for an unknown evaluator.</returns>
    /// <exception cref="NumSharp.KeyError">No study has that name (answered 404).</exception>
    /// <exception cref="NumSharp.ValueError">
    ///     The study has too few complete trials, or several objectives (answered 400 with Optuna's message).
    /// </exception>
    /// <remarks>
    ///     The evaluators are created per request with <c>new</c> rather than injected: they are configured by
    ///     values (a tree count, a seed), not by other services, which is the rule OptunaSharp's DI layer follows
    ///     for what belongs in the container. The seeded ones give the same answer for the same trials.
    /// </remarks>
    private static Results<Ok<List<ParamImportanceDto>>, ValidationProblem> GetImportance(
        string name, string? evaluator, BaseStorage storage)
    {
        BaseImportanceEvaluator? chosen;
        switch (evaluator?.ToLowerInvariant())
        {
            case null or "ped-anova":
                chosen = new PedAnovaImportanceEvaluator();
                break;
            case "fanova":
                chosen = new FanovaImportanceEvaluator(seed: 0);
                break;
            case "mdi":
                chosen = new MeanDecreaseImpurityImportanceEvaluator(seed: 0);
                break;
            case "shap":
                chosen = new ShapleyImportanceEvaluator(seed: 0);
                break;
            default:
                return TypedResults.ValidationProblem(new Dictionary<string, string[]>
                {
                    ["evaluator"] = ["Expected ped-anova, fanova, mdi or shap."],
                });
        }

        Study study = Optuna.LoadStudy(StudyRoute.Name(name), storage);
        List<ParamImportanceDto> importances = ImportanceFunctions.GetParamImportances(study, chosen)
            .Select(kv => new ParamImportanceDto(kv.Key, kv.Value))
            .ToList();
        return TypedResults.Ok(importances);
    }
}
