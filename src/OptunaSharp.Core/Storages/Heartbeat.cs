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
    /// <param name="trialId">Storage identifier of the trial; distinct from its per-study trial number.</param>
    /// <exception cref="System.Exception">The heartbeat storage provider fails.</exception>
    void RecordHeartbeat(int trialId);

    /// <summary>Get the stale trial ids of the study (Python: <c>_get_stale_trial_ids</c>).</summary>
    /// <param name="studyId">Storage identifier of the study; distinct from its name.</param>
    /// <returns>Identifiers of running trials considered stale by the backend.</returns>
    List<int> GetStaleTrialIds(int studyId);

    /// <summary>Get the heartbeat interval if it is set, otherwise <c>null</c>.</summary>
    /// <returns>Heartbeat interval in seconds, or null when disabled.</returns>
    int? GetHeartbeatInterval();

    /// <summary>Get the heartbeat-stale trial callback function if it is set, otherwise <c>null</c>.</summary>
    /// <returns>Configured callback, or null if no stale-trial callback is registered.</returns>
    Action<Study, FrozenTrial>? GetHeartbeatStaleTrialCallback();

    /// <summary>
    ///     Deprecated. Get the failed-trial callback function — returns the same value as
    ///     <see cref="GetHeartbeatStaleTrialCallback" />. Port of the concrete, deprecated
    ///     <c>BaseHeartbeat.get_failed_trial_callback</c> (deprecated in v4.9.0, removed in v6.0.0);
    ///     a default interface method so every heartbeat backend inherits it, exactly as the Python
    ///     base class does.
    /// </summary>
    /// <returns>The same callback returned by GetHeartbeatStaleTrialCallback.</returns>
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
    /// <summary>Starts the periodic heartbeat worker; the null implementation performs no work.</summary>
    public abstract void Start();

    /// <summary>Signals the worker to stop and waits for it; the null implementation performs no work.</summary>
    public abstract void Join();

    /// <summary>Python <c>__exit__</c>: join on scope exit.</summary>
    public void Dispose() => Join();
}

/// <summary>No-op heartbeat worker used when the storage has heartbeats disabled.</summary>
public sealed class NullHeartbeatThread : BaseHeartbeatThread
{
    /// <summary>Starts the periodic heartbeat worker; the null implementation performs no work.</summary>
    public override void Start()
    {
    }

    /// <summary>Signals the worker to stop and waits for it; the null implementation performs no work.</summary>
    public override void Join()
    {
    }
}

/// <summary>Background thread that records liveness until the owning optimization scope ends.</summary>
public sealed class HeartbeatThread : BaseHeartbeatThread
{
    private readonly int _trialId;
    private readonly IBaseHeartbeat _heartbeat;
    private Thread? _thread;
    private ManualResetEventSlim? _stopEvent;

    /// <param name="trialId">Storage identifier of the trial; distinct from its per-study trial number.</param>
    /// <summary>Binds a heartbeat worker to an existing trial.</summary>
    /// <param name="heartbeat">Shared heartbeat backend with a non-null interval; this worker does not own the backend.</param>
    public HeartbeatThread(int trialId, IBaseHeartbeat heartbeat)
    {
        _trialId = trialId;
        _heartbeat = heartbeat;
    }

    /// <summary>Starts the periodic heartbeat worker; the null implementation performs no work.</summary>
    public override void Start()
    {
        _stopEvent = new ManualResetEventSlim(false);
        var stopEvent = _stopEvent;
        _thread = new Thread(() => RecordHeartbeat(_trialId, _heartbeat, stopEvent)) { IsBackground = true };
        _thread.Start();
    }

    /// <summary>Stops a started worker and blocks until its thread exits.</summary>
    /// <exception cref="System.NullReferenceException">Start has not initialized the worker.</exception>
    /// <exception cref="System.Threading.ThreadInterruptedException">The calling thread is interrupted while joining.</exception>
    public override void Join()
    {
        System.Diagnostics.Debug.Assert(_stopEvent is not null);
        System.Diagnostics.Debug.Assert(_thread is not null);
        _stopEvent!.Set();
        _thread!.Join();
    }

    /// <param name="trialId">Storage identifier of the trial; distinct from its per-study trial number.</param>
    /// <summary>Records liveness repeatedly, using an interruptible wait so shutdown need not wait a full interval.</summary>
    /// <param name="heartbeat">Backend used to persist each heartbeat.</param>
    /// <param name="stopEvent">Signal owned by this worker; setting it ends the recording loop.</param>
    /// <exception cref="System.Exception">The backend rejects a heartbeat or the configured interval is invalid; this executes on the heartbeat thread.</exception>
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
    /// <param name="trialId">Storage identifier of the trial; distinct from its per-study trial number.</param>
    /// <param name="storage">Storage backend holding study and trial records.</param>
    /// <returns>An unstarted real or no-op worker; call Start before disposing the returned scope.</returns>
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
    /// <param name="study">Study whose history and storage are used by the operation.</param>
    /// <exception cref="System.Exception">The storage or stale-trial callback fails; already-finished races are ignored.</exception>
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
    /// <param name="storage">Storage backend holding study and trial records.</param>
    /// <returns>True when storage implements the heartbeat protocol and has a non-null interval.</returns>
    public static bool IsHeartbeatEnabled(BaseStorage storage)
        => storage is IBaseHeartbeat heartbeat && heartbeat.GetHeartbeatInterval() is not null;
}
