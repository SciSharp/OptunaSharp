using System;
using OptunaSharp.Internal;
using OptunaSharp.Logging;

namespace OptunaSharp.Terminator;

/// <summary>
///     A callback that terminates the optimization using <see cref="Terminator" />.
///     Port of <c>optuna/terminator/callback.py: TerminatorCallback</c>.
///
///     This class wraps a <see cref="Terminator" /> so that it can be used with
///     <see cref="Study.Optimize(System.Func{Trial, double}, int?, double?, int, System.Type[], System.Collections.Generic.IEnumerable{System.Action{Study, FrozenTrial}}, bool, bool, TrialExecutorOptions)" />.
///     Pass it via the <c>callbacks</c> argument; the implicit conversion to
///     <see cref="Action{Study, FrozenTrial}" /> lets it be handed over directly.
/// </summary>
public class TerminatorCallback
{
    private static readonly Logger _logger = OptunaLogging.GetLogger("optuna.terminator.callback");

    private readonly BaseTerminator _terminator;

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
    public void Invoke(Study study, FrozenTrial trial)
    {
        bool shouldTerminate = _terminator.ShouldTerminate(study);

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
