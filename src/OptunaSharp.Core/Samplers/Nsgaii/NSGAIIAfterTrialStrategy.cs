using System.Collections.Generic;

namespace OptunaSharp.Samplers.Nsgaii;

/// <summary>
///     The default after-trial strategy of <see cref="NSGAIISampler" />. Port of
///     <c>optuna/samplers/nsgaii/_after_trial_strategy.py: NSGAIIAfterTrialStrategy</c>.
///
///     Examines whether the trial result is valid in terms of constraints and stores the results in
///     the study's system attributes (only when a <c>constraints_func</c> is configured).
/// </summary>
public class NSGAIIAfterTrialStrategy
{
    private readonly System.Func<FrozenTrial, IEnumerable<double>>? _constraintsFunc;

    public NSGAIIAfterTrialStrategy(System.Func<FrozenTrial, IEnumerable<double>>? constraintsFunc = null)
    {
        _constraintsFunc = constraintsFunc;
    }

    /// <summary>Carry out the after-trial process of default NSGA-II (Python <c>__call__</c>).</summary>
    public void Call(Study study, FrozenTrial trial, TrialState state, IReadOnlyList<double>? values = null)
    {
        if (_constraintsFunc is not null)
            SamplerHelpers.ProcessConstraintsAfterTrial(_constraintsFunc, study, trial, state);
    }
}
