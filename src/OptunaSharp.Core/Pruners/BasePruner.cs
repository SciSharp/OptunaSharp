namespace OptunaSharp.Pruners;

/// <summary>
///     Base class for pruners. Port of <c>optuna/pruners/_base.py: BasePruner</c>.
/// </summary>
public abstract class BasePruner
{
    /// <summary>
    ///     The monitor that every study-driven call into this pruner runs under: <see cref="Prune" /> from
    ///     <see cref="Trial.ShouldPrune" />, and <see cref="FilterStudy" /> from the trial and tell paths.
    /// </summary>
    /// <remarks>
    ///     One pruner instance serves every thread of a study. Python relies on the GIL; here, lazily built state
    ///     such as <see cref="HyperbandPruner" />'s brackets was appended from several threads at once, which
    ///     crashed <c>Optimize</c> at <c>n_jobs</c> above 1. Pruning decisions of concurrent trials are therefore
    ///     serial per pruner. A wrapping pruner calls its inner pruner directly under this monitor.
    /// </remarks>
    internal object SyncRoot { get; } = new();

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
    /// <remarks>Runs under the pruner's <see cref="BasePruner.SyncRoot" />, so concurrent trials never see its state half-built.</remarks>
    internal static Study FilterStudy(Study study, FrozenTrial trial)
    {
        // Read the property once: Study.Pruner is settable, and the monitor must belong to the pruner that runs.
        var pruner = study.Pruner;
        lock (pruner.SyncRoot)
            return pruner.FilterStudy(study, trial);
    }
}
