namespace OptunaSharp.Pruners;

/// <summary>
///     Base class for pruners. Port of <c>optuna/pruners/_base.py: BasePruner</c>.
/// </summary>
public abstract class BasePruner
{
    /// <summary>
    ///     Judge whether the trial should be pruned based on the reported values.
    ///
    ///     Note that this method is not supposed to be called by library users. Instead,
    ///     <see cref="Trial.Report" /> and <see cref="Trial.ShouldPrune" /> provide user
    ///     interfaces to implement pruning mechanism in an objective function.
    /// </summary>
    /// <param name="study">Study object of the target study.</param>
    /// <param name="trial">
    ///     FrozenTrial object of the target trial. Take a copy before modifying this object.
    /// </param>
    /// <returns>A boolean value representing whether the trial should be pruned.</returns>
    public abstract bool Prune(Study study, FrozenTrial trial);

    /// <summary>
    ///     Python <c>optuna.pruners._filter_study</c> seam: <see cref="HyperbandPruner" />
    ///     overrides this to substitute the bracket study; every other pruner returns the study
    ///     unchanged.
    /// </summary>
    /// <param name="study">Study whose history is made available to the sampler.</param>
    /// <param name="trial">Current trial used to select any pruner-specific history partition.</param>
    /// <returns>The original study, or an override's view restricted to relevant trials.</returns>
    internal virtual Study FilterStudy(Study study, FrozenTrial trial) => study;
}

/// <summary>Module-level helpers of <c>optuna/pruners/__init__.py</c>.</summary>
internal static class PrunerHelpers
{
    /// <summary>Python <c>optuna.pruners._filter_study(study, trial)</c>.</summary>
    /// <param name="study">Study whose configured pruner selects the history view.</param>
    /// <param name="trial">Current trial used to select any history partition.</param>
    /// <returns>The study view supplied by the configured pruner.</returns>
    internal static Study FilterStudy(Study study, FrozenTrial trial)
        => study.Pruner.FilterStudy(study, trial);
}
