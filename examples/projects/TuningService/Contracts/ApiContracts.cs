using System.Text.Json;
using OptunaSharp;
using TuningService.Jobs;

namespace TuningService.Contracts;

/// <summary>The body of <c>POST /api/studies</c>: the study to create.</summary>
/// <param name="Name">The study name, or <c>null</c> to let the storage generate a unique one.</param>
/// <param name="Direction">The direction of a single-objective study; <c>null</c> means Optuna's default, minimize.</param>
/// <param name="Directions">The directions of a multi-objective study. Set it or <paramref name="Direction" />, not both.</param>
public sealed record CreateStudyRequest(
    string? Name,
    StudyDirection? Direction,
    IReadOnlyList<StudyDirection>? Directions);

/// <summary>One study in the <c>GET /api/studies</c> list.</summary>
/// <param name="Name">The study name.</param>
/// <param name="Directions">One direction per objective.</param>
/// <param name="Trials">The number of trials in any state.</param>
/// <param name="BestValue">The best objective value of a single-objective study, or <c>null</c> before the first complete trial.</param>
/// <param name="BestParams">The parameters of that best trial, or <c>null</c>.</param>
/// <param name="DatetimeStart">When the first trial started, or <c>null</c> for a study without trials.</param>
public sealed record StudySummaryDto(
    string Name,
    IReadOnlyList<StudyDirection> Directions,
    int Trials,
    double? BestValue,
    IReadOnlyDictionary<string, object?>? BestParams,
    DateTime? DatetimeStart)
{
    /// <summary>Converts OptunaSharp's <see cref="StudySummary" /> into the API shape.</summary>
    /// <param name="summary">A summary from <c>Optuna.GetAllStudySummaries</c>.</param>
    /// <returns>The API representation of the summary.</returns>
    public static StudySummaryDto From(StudySummary summary)
        => new(
            summary.StudyName,
            summary.Directions,
            summary.NTrials,
            summary.BestTrial?.Value,
            summary.BestTrial?.Params,
            summary.DatetimeStart);
}

/// <summary>The body of <c>GET /api/studies/{name}</c>: one study in detail.</summary>
/// <param name="Name">The study name.</param>
/// <param name="Directions">One direction per objective.</param>
/// <param name="TrialCounts">How many trials are in each state; states without trials are left out.</param>
/// <param name="BestTrial">The best trial of a single-objective study, or <c>null</c> before the first complete trial (and always for a multi-objective study).</param>
/// <param name="BestTrials">The Pareto front of a multi-objective study, or the best trial of a single-objective one; empty before the first complete trial.</param>
/// <param name="UserAttrs">The study's user attributes.</param>
public sealed record StudyDto(
    string Name,
    IReadOnlyList<StudyDirection> Directions,
    IReadOnlyDictionary<TrialState, int> TrialCounts,
    TrialDto? BestTrial,
    IReadOnlyList<TrialDto> BestTrials,
    IReadOnlyDictionary<string, object?> UserAttrs)
{
    /// <summary>Reads a study's trials once and summarizes them.</summary>
    /// <param name="study">The study to describe.</param>
    /// <returns>The API representation of the study.</returns>
    /// <remarks>
    ///     <c>Study.BestTrial</c> raises Optuna's <c>ValueError</c> when no trial is complete yet, so the best
    ///     trial is read only after the counts show a complete one.
    /// </remarks>
    public static StudyDto From(Study study)
    {
        // deepcopy: false reads the storage's own records without copying them; this method only reads them
        List<FrozenTrial> trials = study.GetTrials(deepcopy: false);
        Dictionary<TrialState, int> counts = trials
            .GroupBy(t => t.State)
            .ToDictionary(g => g.Key, g => g.Count());

        bool anyComplete = counts.ContainsKey(TrialState.Complete);
        bool singleObjective = study.Directions.Count == 1;
        TrialDto? best = anyComplete && singleObjective ? TrialDto.From(study.BestTrial) : null;
        IReadOnlyList<TrialDto> front = anyComplete ? study.BestTrials.Select(TrialDto.From).ToList() : [];
        return new StudyDto(study.StudyName, study.Directions, counts, best, front, study.UserAttrs);
    }
}

/// <summary>One trial, as the trials and tell endpoints return it.</summary>
/// <param name="Number">The trial number, which counts from 0 within its study.</param>
/// <param name="State">Running, Complete, Pruned, Fail or Waiting.</param>
/// <param name="Values">The objective values of a complete trial (one per direction), or <c>null</c>.</param>
/// <param name="Params">The suggested parameters, by name.</param>
/// <param name="IntermediateValues">The reported intermediate values, by step.</param>
/// <param name="UserAttrs">The trial's user attributes; the built-in objective records <c>worker</c> here.</param>
/// <param name="DatetimeStart">When the trial started, in the server's local time, or <c>null</c> for a waiting trial.</param>
/// <param name="DatetimeComplete">When the trial finished, or <c>null</c> while it runs.</param>
/// <param name="DurationSeconds">How long the trial ran, or <c>null</c> while it runs.</param>
public sealed record TrialDto(
    int Number,
    TrialState State,
    IReadOnlyList<double>? Values,
    IReadOnlyDictionary<string, object?> Params,
    IReadOnlyDictionary<long, double> IntermediateValues,
    IReadOnlyDictionary<string, object?> UserAttrs,
    DateTime? DatetimeStart,
    DateTime? DatetimeComplete,
    double? DurationSeconds)
{
    /// <summary>Converts OptunaSharp's <see cref="FrozenTrial" /> into the API shape.</summary>
    /// <param name="trial">The trial record to convert.</param>
    /// <returns>The API representation of the trial.</returns>
    public static TrialDto From(FrozenTrial trial)
        => new(
            trial.Number,
            trial.State,
            trial.Values,
            trial.Params,
            // copied into a plain dictionary, which System.Text.Json writes as {"1": 0.93, "2": 0.88, …}
            new Dictionary<long, double>(trial.IntermediateValues),
            trial.UserAttrs,
            trial.DatetimeStart,
            trial.DatetimeComplete,
            trial.Duration?.TotalSeconds);
}

/// <summary>The body of <c>POST /api/studies/{name}/trials</c>: ask for a new trial.</summary>
/// <param name="SearchSpace">
///     The parameters to suggest, each in Optuna's distribution JSON, the format of
///     <c>optuna ask --search-space</c> and <c>optuna.distributions.distribution_to_json</c>, for example
///     <c>{"x": {"name": "FloatDistribution", "attributes": {"low": -10, "high": 10, "log": false, "step": null}}}</c>.
///     <c>null</c> or empty asks a trial without parameters.
/// </param>
/// <param name="Worker">An optional name for the caller, recorded as the trial's <c>worker</c> user attribute.</param>
public sealed record AskRequest(
    IReadOnlyDictionary<string, JsonElement>? SearchSpace,
    string? Worker);

/// <summary>The body the ask endpoint returns: the new trial and the parameters to evaluate.</summary>
/// <param name="Number">The trial number; use it in the report and tell calls.</param>
/// <param name="Params">The sampled value of each parameter of the search space.</param>
public sealed record AskResponse(int Number, IReadOnlyDictionary<string, object?> Params);

/// <summary>The body of <c>POST /api/studies/{name}/trials/{number}/report</c>: one intermediate value.</summary>
/// <param name="Step">The step, typically an epoch number; the first report of a step wins, as in Optuna.</param>
/// <param name="Value">The intermediate objective value at that step.</param>
public sealed record ReportRequest(long Step, double Value);

/// <summary>The body the report endpoint returns: whether the study's pruner wants the trial stopped.</summary>
/// <param name="Number">The trial number.</param>
/// <param name="Step">The step that was reported.</param>
/// <param name="ShouldPrune">
///     <c>true</c> when the pruner decided to stop the trial. The caller should then stop evaluating and tell
///     the trial with the state <c>Pruned</c>.
/// </param>
public sealed record ReportResponse(int Number, long Step, bool ShouldPrune);

/// <summary>The body of <c>POST /api/studies/{name}/trials/{number}/tell</c>: finish a trial.</summary>
/// <param name="Values">The objective values, one per direction. Required for a complete trial; leave it out for a pruned or failed one.</param>
/// <param name="State"><c>Complete</c>, <c>Pruned</c> or <c>Fail</c>; <c>null</c> lets Optuna infer Complete or Fail from the values.</param>
public sealed record TellRequest(IReadOnlyList<double>? Values, TrialState? State);

/// <summary>The body of <c>POST /api/studies/{name}/optimize</c>: run the built-in objective in the background.</summary>
/// <param name="NTrials">How many trials to run.</param>
/// <param name="NJobs">How many trials to run in parallel, on OptunaSharp's own worker threads.</param>
public sealed record OptimizeRequest(int NTrials = 40, int NJobs = 1);

/// <summary>One optimization job, as the job endpoints return it.</summary>
/// <param name="Id">The job's identifier.</param>
/// <param name="StudyName">The study the job optimizes.</param>
/// <param name="State">Queued, Running, Completed, Canceled or Failed.</param>
/// <param name="NTrials">The number of trials the job was asked to run.</param>
/// <param name="NJobs">The number of trials it runs in parallel.</param>
/// <param name="Complete">How many of its trials completed so far.</param>
/// <param name="Pruned">How many of its trials the pruner stopped so far.</param>
/// <param name="Failed">How many of its trials failed so far.</param>
/// <param name="QueuedAt">When the job was queued.</param>
/// <param name="StartedAt">When it started, or <c>null</c> while queued.</param>
/// <param name="FinishedAt">When it finished, or <c>null</c> while queued or running.</param>
/// <param name="Error">The error that ended a failed job, or <c>null</c>.</param>
public sealed record JobDto(
    Guid Id,
    string StudyName,
    OptimizationJobState State,
    int NTrials,
    int NJobs,
    int Complete,
    int Pruned,
    int Failed,
    DateTimeOffset QueuedAt,
    DateTimeOffset? StartedAt,
    DateTimeOffset? FinishedAt,
    string? Error);

/// <summary>One parameter's importance, as <c>GET /api/studies/{name}/importance</c> returns it.</summary>
/// <param name="Param">The parameter name.</param>
/// <param name="Importance">Its importance; the values sum to 1 and the list is sorted from most to least important.</param>
public sealed record ParamImportanceDto(string Param, double Importance);
