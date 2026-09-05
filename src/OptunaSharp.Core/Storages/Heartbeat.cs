using System;
using System.Collections.Generic;
using System.Threading;
using OptunaSharp.Internal;

namespace OptunaSharp.Storages;

/// <summary>
///     Base interface for heartbeat. Port of <c>optuna/storages/_heartbeat.py: BaseHeartbeat</c>
///     (a mixin class in Python; an interface in C# so storage backends can combine it with
///     <see cref="BaseStorage" />).
///
///     The heartbeat mechanism periodically checks whether each trial process is alive during an
///     optimization loop.
/// </summary>
public interface IBaseHeartbeat
{
    /// <summary>Record the heartbeat of the trial.</summary>
    void RecordHeartbeat(int trialId);

    /// <summary>Get the stale trial ids of the study (Python: <c>_get_stale_trial_ids</c>).</summary>
    List<int> GetStaleTrialIds(int studyId);

    /// <summary>Get the heartbeat interval if it is set, otherwise <c>null</c>.</summary>
    int? GetHeartbeatInterval();

    /// <summary>Get the heartbeat-stale trial callback function if it is set, otherwise <c>null</c>.</summary>
    Action<Study, FrozenTrial>? GetHeartbeatStaleTrialCallback();

    /// <summary>
    ///     Deprecated. Get the failed-trial callback function — returns the same value as
    ///     <see cref="GetHeartbeatStaleTrialCallback" />. Port of the concrete, deprecated
    ///     <c>BaseHeartbeat.get_failed_trial_callback</c> (deprecated in v4.9.0, removed in v6.0.0);
    ///     a default interface method so every heartbeat backend inherits it, exactly as the Python
    ///     base class does.
    /// </summary>
    [Obsolete("Deprecated in v4.9.0. This feature will be removed in v6.0.0. " +
              "Use GetHeartbeatStaleTrialCallback instead.")]
    Action<Study, FrozenTrial>? GetFailedTrialCallback()
    {
        OptunaWarnings.Warn(
            "get_failed_trial_callback has been deprecated in v4.9.0. This feature will be removed in " +
            "v6.0.0. See https://github.com/optuna/optuna/releases/tag/v4.9.0. " +
            "Use `get_heartbeat_stale_trial_callback` instead.",
            WarningCategory.FutureWarning);
        return GetHeartbeatStaleTrialCallback();
    }
}

/// <summary>Python <c>BaseHeartbeatThread</c> (context manager → <see cref="IDisposable" />).</summary>
public abstract class BaseHeartbeatThread : IDisposable
{
    public abstract void Start();

    public abstract void Join();

    /// <summary>Python <c>__exit__</c>: join on scope exit.</summary>
    public void Dispose() => Join();
}

public sealed class NullHeartbeatThread : BaseHeartbeatThread
{
    public override void Start()
    {
    }

    public override void Join()
    {
    }
}

public sealed class HeartbeatThread : BaseHeartbeatThread
{
    private readonly int _trialId;
    private readonly IBaseHeartbeat _heartbeat;
    private Thread? _thread;
    private ManualResetEventSlim? _stopEvent;

    public HeartbeatThread(int trialId, IBaseHeartbeat heartbeat)
    {
        _trialId = trialId;
        _heartbeat = heartbeat;
    }

    public override void Start()
    {
        _stopEvent = new ManualResetEventSlim(false);
        var stopEvent = _stopEvent;
        _thread = new Thread(() => RecordHeartbeat(_trialId, _heartbeat, stopEvent)) { IsBackground = true };
        _thread.Start();
    }

    public override void Join()
    {
        System.Diagnostics.Debug.Assert(_stopEvent is not null);
        System.Diagnostics.Debug.Assert(_thread is not null);
        _stopEvent!.Set();
        _thread!.Join();
    }

    private static void RecordHeartbeat(int trialId, IBaseHeartbeat heartbeat, ManualResetEventSlim stopEvent)
    {
        int? heartbeatInterval = heartbeat.GetHeartbeatInterval();
        System.Diagnostics.Debug.Assert(heartbeatInterval is not null);
        while (true)
        {
            heartbeat.RecordHeartbeat(trialId);
            if (stopEvent.Wait(TimeSpan.FromSeconds(heartbeatInterval!.Value)))
                return;
        }
    }
}

/// <summary>Module-level functions of <c>optuna/storages/_heartbeat.py</c>.</summary>
public static class HeartbeatHelpers
{
    /// <summary>Python <c>get_heartbeat_thread(trial_id, storage)</c>.</summary>
    public static BaseHeartbeatThread GetHeartbeatThread(int trialId, BaseStorage storage)
    {
        if (IsHeartbeatEnabled(storage))
            return new HeartbeatThread(trialId, (IBaseHeartbeat)storage);
        return new NullHeartbeatThread();
    }

    /// <summary>
    ///     Fail stale trials and run their failure callbacks (Python <c>fail_stale_trials</c>,
    ///     experimental since 2.9.0).
    ///
    ///     The running trials whose heartbeat has not been updated for a long time will be
    ///     failed, that is, those states will be changed to <see cref="TrialState.Fail" />.
    /// </summary>
    public static void FailStaleTrials(Study study)
    {
        var storage = study.Storage;

        if (storage is not IBaseHeartbeat heartbeat)
            return;

        if (!IsHeartbeatEnabled(storage))
            return;

        var failedTrialIds = new List<int>();
        foreach (int trialId in heartbeat.GetStaleTrialIds(study.StudyId))
        {
            try
            {
                if (storage.SetTrialStateValues(trialId, state: TrialState.Fail))
                    failedTrialIds.Add(trialId);
            }
            catch (UpdateFinishedTrialError)
            {
                // If another process fails the trial, the storage raises UpdateFinishedTrialError.
            }
        }

        var heartbeatStaleTrialCallback = heartbeat.GetHeartbeatStaleTrialCallback();
        if (heartbeatStaleTrialCallback is not null)
        {
            foreach (int trialId in failedTrialIds)
            {
                var failedTrial = storage.GetTrial(trialId).DeepCopy();
                heartbeatStaleTrialCallback(study, failedTrial);
            }
        }
    }

    /// <summary>
    ///     Check whether the storage enables the heartbeat: the storage implements
    ///     <see cref="IBaseHeartbeat" /> and <see cref="IBaseHeartbeat.GetHeartbeatInterval" />
    ///     returns an integer.
    /// </summary>
    public static bool IsHeartbeatEnabled(BaseStorage storage)
        => storage is IBaseHeartbeat heartbeat && heartbeat.GetHeartbeatInterval() is not null;
}
