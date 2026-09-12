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

    /// <summary>Configures optional legacy constraint evaluation after an individual finishes.</summary>
    /// <param name="constraintsFunc">Optional legacy callback returning constraint values; positive values violate a constraint.</param>
    public NSGAIIAfterTrialStrategy(System.Func<FrozenTrial, IEnumerable<double>>? constraintsFunc = null)
    {
        _constraintsFunc = constraintsFunc;
    }

    /// <summary>Carry out the after-trial process of default NSGA-II (Python <c>__call__</c>).</summary>
    /// <param name="study">Study supplying trial history, objective directions, and storage.</param>
    /// <param name="trial">Trial snapshot being examined; copy its contents before modifying them.</param>
    /// <param name="state">Terminal trial state being recorded.</param>
    /// <param name="values">Objective values in study order, or null when no final values are available.</param>
    /// <exception cref="NumSharp.ValueError">The configured constraint callback returns NaN.</exception>
    /// <exception cref="System.Exception">The configured constraint callback or storage provider fails; its exception propagates.</exception>
    public void Call(Study study, FrozenTrial trial, TrialState state, IReadOnlyList<double>? values = null)
    {
        if (_constraintsFunc is not null)
            SamplerHelpers.ProcessConstraintsAfterTrial(_constraintsFunc, study, trial, state);
    }
}
