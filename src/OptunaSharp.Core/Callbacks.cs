using System;
using System.Collections.Generic;

namespace OptunaSharp;

/// <summary>
///     Set a maximum number of trials before ending the study.
///     Port of <c>optuna/_callbacks.py: MaxTrialsCallback</c> (exposed as
///     <c>optuna.study.MaxTrialsCallback</c>).
///
///     While the <c>nTrials</c> argument of <c>Study.Optimize</c> sets the number of trials that
///     will be run, you may want to continue running until you have a certain number of
///     successfully completed trials, or stop the study when you have a certain number of trials
///     that fail. This <see cref="MaxTrialsCallback" /> class allows you to set a maximum number of
///     trials for a particular <see cref="TrialState" /> before stopping the study.
/// </summary>
/// <example>
///     <code>
///     var study = Optuna.CreateStudy();
///     study.Optimize(
///         trial => { var x = trial.SuggestFloat("x", -1, 1); return x * x; },
///         callbacks: new[]
///         {
///             (Action&lt;Study, FrozenTrial&gt;)new MaxTrialsCallback(10, new[] { TrialState.Complete }),
///         });
///     </code>
/// </example>
public class MaxTrialsCallback
{
    private readonly int _nTrials;
    private readonly IReadOnlyCollection<TrialState>? _states;

    /// <summary>
    ///     Count only trials in <see cref="TrialState.Complete" /> towards the limit — Optuna's
    ///     default <c>states=(TrialState.COMPLETE,)</c>.
    /// </summary>
    /// <param name="nTrials">The max number of trials. Must be set to an integer.</param>
    public MaxTrialsCallback(int nTrials)
        : this(nTrials, new[] { TrialState.Complete })
    {
    }

    /// <param name="nTrials">The max number of trials. Must be set to an integer.</param>
    /// <param name="states">
    ///     The set of <see cref="TrialState" /> counted towards the max-trials limit. Optuna's
    ///     default is <c>(TrialState.COMPLETE,)</c> — use the single-argument constructor for that.
    ///     If <c>null</c>, count trials in all states.
    /// </param>
    public MaxTrialsCallback(int nTrials, IReadOnlyCollection<TrialState>? states)
    {
        _nTrials = nTrials;
        _states = states;
    }

    /// <summary>Python <c>__call__(study, trial)</c>.</summary>
    public void Invoke(Study study, FrozenTrial trial)
    {
        var trials = study.GetTrials(deepcopy: false, states: _states);
        int nComplete = trials.Count;
        if (nComplete >= _nTrials)
            study.Stop();
    }

    /// <summary>Allows passing the callback where an <see cref="Action{Study, FrozenTrial}" /> is expected.</summary>
    public static implicit operator Action<Study, FrozenTrial>(MaxTrialsCallback callback)
        => callback.Invoke;
}
