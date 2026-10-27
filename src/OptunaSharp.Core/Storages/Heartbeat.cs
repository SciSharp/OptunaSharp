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
/// <remarks>
///     <para><b>A failed heartbeat write ends only this thread.</b> Python's <c>_record_heartbeat</c> runs on a
///     <c>threading.Thread</c>; when <c>record_heartbeat</c> raises (a dropped database connection, a busy
///     SQLite file), the thread dies, <c>threading.excepthook</c> prints the error to stderr, and the trial keeps
///     running without heartbeats. .NET instead terminates the whole process on an unhandled thread exception,
///     so this port catches every exception the loop raises, reports it in the excepthook layout
///     (<c>Exception in thread &lt;name&gt;:</c> followed by the exception) on the current standard error, and
///     lets the thread end.</para>
///     <para>Consequence, as in Optuna: once the heartbeat stops, the trial's heartbeat ages, and any worker's
///     <see cref="HeartbeatHelpers.FailStaleTrials" /> may fail it as stale while its objective is still running.
///     The thread does not retry, because Optuna does not.</para>
/// </remarks>
public sealed class HeartbeatThread : BaseHeartbeatThread
{
    private readonly int _trialId;
    private readonly IBaseHeartbeat _heartbeat;
    private Thread? _thread;
    private ManualResetEventSlim? _stopEvent;

    /// <summary>Binds a heartbeat worker to an existing trial; nothing runs until <see cref="Start" />.</summary>
    /// <param name="trialId">Storage identifier of the trial; distinct from its per-study trial number.</param>
    /// <param name="heartbeat">Shared heartbeat backend with a non-null interval; this worker does not own the backend.</param>
    public HeartbeatThread(int trialId, IBaseHeartbeat heartbeat)
    {
        _trialId = trialId;
        _heartbeat = heartbeat;
    }

    /// <summary>
    ///     Starts a background thread that beats immediately and then once per interval until
    ///     <see cref="Join" />. It is a background thread, so it never keeps the process alive on its own.
    /// </summary>
    /// <exception cref="System.OutOfMemoryException">There is not enough memory to start the thread.</exception>
    public override void Start()
    {
        _stopEvent = new ManualResetEventSlim(false);
        var stopEvent = _stopEvent;
        // The name is what a failure report and a debugger show; it identifies the trial the thread served.
        string name = $"OptunaSharp heartbeat (trial_id={_trialId})";
        _thread = new Thread(() => RecordHeartbeat(_trialId, _heartbeat, stopEvent, name))
        {
            IsBackground = true,
            Name = name,
        };
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

    /// <summary>
    ///     Records liveness repeatedly, using an interruptible wait so shutdown need not wait a full interval.
    ///     Never throws: this is the heartbeat thread's entry point, and an exception escaping it would
    ///     terminate the process.
    /// </summary>
    /// <param name="trialId">Storage identifier of the trial; distinct from its per-study trial number.</param>
    /// <param name="heartbeat">Backend used to persist each heartbeat.</param>
    /// <param name="stopEvent">Signal owned by this worker; setting it ends the recording loop.</param>
    /// <param name="threadName">The thread's name, quoted in the failure report.</param>
    private static void RecordHeartbeat(
        int trialId, IBaseHeartbeat heartbeat, ManualResetEventSlim stopEvent, string threadName)
    {
        try
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
        catch (Exception e)
        {
            // Python: the exception ends the heartbeat thread and threading.excepthook reports it; the trial
            // and the process keep running. Catching here is what keeps .NET from killing the process.
            ReportUnhandledException(threadName, e);
        }
    }

    /// <summary>
    ///     Port of Python's default <c>threading.excepthook</c>: writes <c>Exception in thread &lt;name&gt;:</c>
    ///     and the exception (its type, message and stack trace) to the <i>current</i> standard error.
    /// </summary>
    /// <remarks>
    ///     The report is best-effort and never throws. Python's hook does nothing when <c>sys.stderr</c> is
    ///     <c>None</c>, and a failure inside the hook still ends only the thread. Here every failure of the
    ///     write (a disposed <see cref="Console.Error" />, or any exception a host's replacement writer throws)
    ///     is swallowed: this runs in the heartbeat thread's last catch block, where an escaping exception would
    ///     turn the lost heartbeat back into a process crash.
    /// </remarks>
    /// <param name="threadName">The name of the thread whose loop failed.</param>
    /// <param name="error">The exception that ended the loop.</param>
    private static void ReportUnhandledException(string threadName, Exception error)
    {
        try
        {
            Console.Error.Write($"Exception in thread {threadName}:\n{error}\n");
        }
        catch (Exception)
        {
            // Nothing else can be done for a report that has nowhere to go.
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
