using System;
using System.Collections.Generic;
using System.Linq;
using OptunaSharp.Collections;

namespace OptunaSharp.Storages;

/// <summary>
///     Retry a heartbeat-stale trial up to a maximum number of times.
///     Port of <c>optuna/storages/_callbacks.py: RetryHeartbeatStaleTrialCallback</c>
///     (experimental since 2.8.0).
///
///     When a running trial becomes stale due to the heartbeat mechanism, this callback can be
///     used with a storage class to recreate the trial in <see cref="TrialState.Waiting" /> to
///     queue up the trial to be run again.
///
///     The original stale trial can be identified by <see cref="RetriedTrialNumber" />. Even if
///     repetitive failure occurs (a retried trial becomes stale again), this method returns the
///     number of the original trial. To get a full list including the numbers of the retried
///     trials as well as their original trial, call <see cref="RetryHistory" />.
/// </summary>
public class RetryHeartbeatStaleTrialCallback
{
    private readonly int? _maxRetry;
    private readonly bool _inheritIntermediateValues;

    /// <param name="maxRetry">
    ///     The max number of times a trial can be retried. If <c>null</c> (default), retry
    ///     indefinitely; if an integer, only retry that many times.
    /// </param>
    /// <param name="inheritIntermediateValues">
    ///     Option to inherit <c>trial.intermediate_values</c> reported by
    ///     <see cref="Trial.Report" /> from the failed trial.
    /// </param>
    public RetryHeartbeatStaleTrialCallback(int? maxRetry = null, bool inheritIntermediateValues = false)
    {
        _maxRetry = maxRetry;
        _inheritIntermediateValues = inheritIntermediateValues;
    }

    /// <summary>Python <c>__call__(study, trial)</c>.</summary>
    public void Invoke(Study study, FrozenTrial trial)
    {
        // {"failed_trial": n, "retry_history": [], **trial.system_attrs} — existing attrs win.
        var systemAttrs = new OrderedDict<string, object?>
        {
            ["failed_trial"] = (long)trial.Number,
            ["retry_history"] = new List<object?>(),
        };
        foreach (var (key, value) in trial.SystemAttrsDict)
            systemAttrs[key] = value;

        var retryHistory = (List<object?>)systemAttrs["retry_history"]!;
        retryHistory.Add((long)trial.Number);
        if (_maxRetry is not null && _maxRetry.Value < retryHistory.Count)
            return;

        study.AddTrial(FrozenTrial.CreateTrial(
            state: TrialState.Waiting,
            @params: trial.ParamsDict.ShallowClone(),
            distributions: trial.DistributionsDict.ShallowClone(),
            userAttrs: trial.UserAttrsDict.ShallowClone(),
            systemAttrs: systemAttrs,
            intermediateValues: _inheritIntermediateValues ? trial.IntermediateValues.ShallowClone() : null));
    }

    /// <summary>Allows passing the callback where an <see cref="Action{Study, FrozenTrial}" /> is expected.</summary>
    public static implicit operator Action<Study, FrozenTrial>(RetryHeartbeatStaleTrialCallback callback)
        => callback.Invoke;

    /// <summary>
    ///     Return the number of the original trial being retried, or <c>null</c> if the trial is
    ///     not a retry of a previous trial. (Experimental since 2.8.0.)
    /// </summary>
    public static int? RetriedTrialNumber(FrozenTrial trial)
        => trial.SystemAttrsDict.GetValueOrDefault("failed_trial") is { } value
            ? Convert.ToInt32(value, System.Globalization.CultureInfo.InvariantCulture)
            : null;

    /// <summary>
    ///     Return the list of retried trial numbers with respect to the specified trial, in
    ///     ascending order of the series of retried trials; empty if the trial is not a retry.
    ///     (Experimental since 3.0.0.)
    /// </summary>
    public static List<int> RetryHistory(FrozenTrial trial)
        => trial.SystemAttrsDict.GetValueOrDefault("retry_history") is List<object?> history
            ? history.Select(n => Convert.ToInt32(n, System.Globalization.CultureInfo.InvariantCulture)).ToList()
            : new List<int>();
}

/// <summary>
///     Deprecated alias of <see cref="RetryHeartbeatStaleTrialCallback" />.
/// </summary>
[Obsolete("Deprecated in v4.9.0. This feature will be removed in v6.0.0. Use RetryHeartbeatStaleTrialCallback instead.")]
public class RetryFailedTrialCallback : RetryHeartbeatStaleTrialCallback
{
    public RetryFailedTrialCallback(int? maxRetry = null, bool inheritIntermediateValues = false)
        : base(maxRetry, inheritIntermediateValues)
    {
    }
}
