using System;
using OptunaSharp.Internal;
using OptunaSharp.Logging;

namespace OptunaSharp.Terminator;

/// <summary>
///     A callback that terminates the optimization using <see cref="Terminator" />.
///     Port of <c>optuna/terminator/callback.py: TerminatorCallback</c>.
///
///     This class wraps a <see cref="Terminator" /> so that it can be used with
///     <see cref="Study.Optimize(System.Func{Trial, double}, int?, double?, int, System.Type[], System.Collections.Generic.IEnumerable{System.Action{Study, FrozenTrial}}, bool, bool, TrialExecutorOptions, System.Threading.CancellationToken)" />.
///     Pass it via the <c>callbacks</c> argument; the implicit conversion to
///     <see cref="Action{Study, FrozenTrial}" /> lets it be handed over directly.
/// </summary>
public class TerminatorCallback
{
    private static readonly Logger _logger = OptunaLogging.GetLogger("optuna.terminator.callback");

    private readonly BaseTerminator _terminator;

    // Serializes concurrent calls: under Optimize(nJobs > 1) the callbacks of different trials run at once, and the
    // default terminator's evaluators (RegretBoundEvaluator, EMMREvaluator) keep one LazyRandomState across calls.
    // NumSharp's MT19937 takes no lock, so unguarded concurrent draws repeat values and can throw; Python is safe
    // only because numpy's RandomState locks each draw (THREADING.md §6.6).
    private readonly object _invokeGate = new();

    /// <param name="terminator">
    ///     A terminator object which determines whether to terminate the optimization by assessing
    ///     the room for optimization and statistical error. Defaults to a <see cref="Terminator" />
    ///     object with default <c>improvementEvaluator</c> and <c>errorEvaluator</c>.
    /// </param>
    /// <summary>Adapts a terminator into an after-trial callback that can stop the active optimization loop.</summary>
    public TerminatorCallback(BaseTerminator? terminator = null)
    {
        OptunaWarnings.Warn(TerminatorModule.DeprecationWarningMessage, WarningCategory.FutureWarning);
        _terminator = terminator ?? new Terminator();
    }

    /// <summary>Python <c>__call__(study, trial)</c>.</summary>
    /// <param name="study">The study supplying trial history, objective directions, and storage.</param>
    /// <param name="trial">The trial whose recorded state and values are used.</param>
    /// <remarks>
    ///     Safe to call from several threads at once, as <c>Optimize(nJobs &gt; 1)</c> does: the terminator's
    ///     decision runs under one lock per callback, so concurrent trials take turns. A terminator that takes the
    ///     Python GIL inside <see cref="BaseTerminator.ShouldTerminate" /> is safe too: a caller that holds the GIL
    ///     releases it while it waits for that lock, as for every lock that comes before the GIL (THREADING.md §8).
    ///     <see cref="Study.Stop" /> is called outside the lock.
    /// </remarks>
    /// <exception cref="Exception">The terminator's own evaluation fails; the lock is released first.</exception>
    public void Invoke(Study study, FrozenTrial trial)
    {
        bool shouldTerminate;
        // MonitorGate, not a plain lock: this lock comes before the GIL in the lock order (THREADING.md §5.2), so a
        // thread that holds the GIL must release it while it waits, or a GIL-taking terminator inside could never
        // finish (review item P3-1).
        using (MonitorGate.Scope(_invokeGate))
            shouldTerminate = _terminator.ShouldTerminate(study);

        if (shouldTerminate)
        {
            _logger.Info("The study has been stopped by the terminator.");
            study.Stop();
        }
    }

    /// <summary>Allows passing the callback where an <see cref="Action{Study, FrozenTrial}" /> is expected.</summary>
    /// <param name="callback">The callback object to adapt to an optimization-completion delegate.</param>
    /// <returns>A delegate that invokes this callback after each completed trial.</returns>
    public static implicit operator Action<Study, FrozenTrial>(TerminatorCallback callback)
        => callback.Invoke;
}
