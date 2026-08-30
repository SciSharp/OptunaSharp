namespace OptunaSharp;

/// <summary>
///     State of a <see cref="Trial" />. Port of <c>optuna/trial/_state.py</c>; the numeric
///     values are Optuna's and are persisted by storages.
/// </summary>
public enum TrialState
{
    /// <summary>The <see cref="Trial" /> is running.</summary>
    Running = 0,

    /// <summary>The <see cref="Trial" /> has been finished without any error.</summary>
    Complete = 1,

    /// <summary>The <see cref="Trial" /> has been pruned with <see cref="TrialPruned" />.</summary>
    Pruned = 2,

    /// <summary>The <see cref="Trial" /> has failed due to an uncaught error.</summary>
    Fail = 3,

    /// <summary>The <see cref="Trial" /> is waiting and unfinished.</summary>
    Waiting = 4,
}

public static class TrialStateExtensions
{
    /// <summary>
    ///     Return a bool value to represent whether the trial state is unfinished or not.
    ///
    ///     The unfinished state is either <see cref="TrialState.Running" /> or
    ///     <see cref="TrialState.Waiting" />.
    /// </summary>
    public static bool IsFinished(this TrialState state)
        => state != TrialState.Running && state != TrialState.Waiting;

    /// <summary>Python <c>state.name</c> (e.g. <c>"COMPLETE"</c>) — used in verbatim messages.</summary>
    internal static string PythonName(this TrialState state) => state switch
    {
        TrialState.Running => "RUNNING",
        TrialState.Complete => "COMPLETE",
        TrialState.Pruned => "PRUNED",
        TrialState.Fail => "FAIL",
        _ => "WAITING",
    };

    /// <summary>Python <c>str(state)</c> (e.g. <c>"TrialState.COMPLETE"</c>).</summary>
    internal static string PythonStr(this TrialState state) => "TrialState." + state.PythonName();
}
