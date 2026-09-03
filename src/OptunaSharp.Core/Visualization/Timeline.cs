using System;
using System.Collections.Generic;
using System.Linq;
using NumSharp;
using OptunaSharp.Logging;

namespace OptunaSharp.Visualization;

/// <summary>
///     One trial's lifetime bar. Port of <c>_TimelineBarInfo</c> in
///     <c>optuna/visualization/_timeline.py</c>.
/// </summary>
public sealed record TimelineBarInfo(
    int Number,
    DateTime Start,
    DateTime Complete,
    TrialState State,
    string Hovertext,
    bool Infeasible);

/// <summary>The data backing a timeline plot. Port of <c>_TimelineInfo</c>.</summary>
public sealed record TimelineInfo(IReadOnlyList<TimelineBarInfo> Bars);

/// <summary>
///     Data layer for <c>optuna.visualization.plot_timeline</c> — the pure <c>_get_timeline_info</c>
///     half of <c>_timeline.py</c> (with the <c>n_recent_trials</c> validation from
///     <c>plot_timeline</c> folded into <see cref="GetInfo" />).
/// </summary>
public static class Timeline
{
    private static readonly Logger Logger = OptunaLogging.GetLogger("optuna.visualization._timeline");

    /// <summary>Python <c>_get_timeline_info(study, n_recent_trials)</c>.</summary>
    /// <exception cref="ValueError">If <paramref name="nRecentTrials" /> is 0 or negative.</exception>
    public static TimelineInfo GetInfo(Study study, int? nRecentTrials = null)
    {
        if (nRecentTrials is not null && nRecentTrials <= 0)
            throw new ValueError("n_recent_trials must be a positive integer or None.");

        var bars = new List<TimelineBarInfo>();

        var maxDatetime = GetMaxDatetimeComplete(study);
        var timedeltaForSmallBar = TimeSpan.FromSeconds(1);

        var trials = study.GetTrials(deepcopy: false);
        if (nRecentTrials is not null)
            trials = trials.Skip(Math.Max(0, trials.Count - nRecentTrials.Value)).ToList();

        foreach (var trial in trials)
        {
            var datetimeStart = trial.DatetimeStart ?? maxDatetime;
            var datetimeComplete = trial.State == TrialState.Running
                ? maxDatetime + timedeltaForSmallBar
                : trial.DatetimeComplete ?? datetimeStart + timedeltaForSmallBar;
            bool infeasible = trial.Constraints.Values.Any(x => x > 0);
            if (datetimeComplete < datetimeStart)
                Logger.Warning(
                    $"The start and end times for Trial {trial.Number} seem to be reversed. " +
                    $"The start time is {datetimeStart} and the end time is {datetimeComplete}.");
            bars.Add(new TimelineBarInfo(
                trial.Number, datetimeStart, datetimeComplete, trial.State,
                VisualizationUtils.MakeHovertext(trial), infeasible));
        }

        if (bars.Count == 0)
            Logger.Warning("Your study does not have any trials.");

        return new TimelineInfo(bars);
    }

    /// <summary>Python <c>_get_max_datetime_complete(study)</c>.</summary>
    private static DateTime GetMaxDatetimeComplete(Study study)
    {
        var trials = study.GetTrials(deepcopy: false);

        TimeSpan? maxRunDuration = null;
        foreach (var t in trials)
            if (t.DatetimeComplete is not null && t.DatetimeStart is not null)
            {
                var duration = t.DatetimeComplete.Value - t.DatetimeStart.Value;
                if (maxRunDuration is null || duration > maxRunDuration.Value)
                    maxRunDuration = duration;
            }

        if (IsRunningTrialsInStudy(study, maxRunDuration))
            return DateTime.Now;

        DateTime? maxComplete = null;
        foreach (var t in trials)
            if (t.DatetimeComplete is not null && (maxComplete is null || t.DatetimeComplete.Value > maxComplete.Value))
                maxComplete = t.DatetimeComplete;

        return maxComplete ?? DateTime.Now;
    }

    /// <summary>Python <c>_is_running_trials_in_study(study, max_run_duration)</c>.</summary>
    private static bool IsRunningTrialsInStudy(Study study, TimeSpan? maxRunDuration)
    {
        var running = study.GetTrials(deepcopy: false, states: new[] { TrialState.Running });
        if (maxRunDuration is null)
            return running.Count > 0;

        var now = DateTime.Now;
        // This heuristic is to check whether we have trials that were somehow killed,
        // still remain as `RUNNING` in `study`.
        return running.Any(t =>
            t.DatetimeStart is not null && (now - t.DatetimeStart.Value) < maxRunDuration.Value * 5.0);
    }
}
