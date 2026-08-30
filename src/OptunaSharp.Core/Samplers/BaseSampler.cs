using System;
using System.Collections.Generic;
using System.Linq;
using NumSharp;
using OptunaSharp.Collections;
using OptunaSharp.Distributions;
using OptunaSharp.Internal;

namespace OptunaSharp.Samplers;

/// <summary>
///     Base class for samplers. Port of <c>optuna/samplers/_base.py: BaseSampler</c>.
///
///     Optuna combines two types of sampling strategies, which are called <i>relative
///     sampling</i> and <i>independent sampling</i>.
///
///     The relative sampling determines values of multiple parameters simultaneously so that
///     sampling algorithms can use relationship between parameters (e.g., correlation). Target
///     parameters of the relative sampling are described in a relative search space, which is
///     determined by <see cref="InferRelativeSearchSpace" />.
///
///     The independent sampling determines a value of a single parameter without considering any
///     relationship between parameters. Target parameters of the independent sampling are the
///     parameters not described in the relative search space.
/// </summary>
public abstract class BaseSampler
{
    internal const string IndependentSamplingWarningTemplate =
        "The parameter `{0}` in Trial#{1} is sampled independently " +
        "using `{2}` instead of `{3}`, potentially " +
        "degrading the optimization performance. This fallback happened because " +
        "{4}. You can suppress this warning by setting " +
        "`warn_independent_sampling` to `False` in the constructor of `{3}` if " +
        "this independent sampling is intended behavior.";

    public override string ToString() => GetType().Name;

    /// <summary>
    ///     Infer the search space that will be used by relative sampling in the target trial.
    ///
    ///     This method is called right before <see cref="SampleRelative" />, and the search space
    ///     returned by this method is passed to it. The parameters not contained in the search
    ///     space will be sampled by using <see cref="SampleIndependent" />.
    /// </summary>
    public abstract OrderedDict<string, BaseDistribution> InferRelativeSearchSpace(Study study, FrozenTrial trial);

    /// <summary>
    ///     Sample parameters in a given search space.
    ///
    ///     This method is called once at the beginning of each trial, i.e., right before the
    ///     evaluation of the objective function. This method is suitable for sampling algorithms
    ///     that use relationship between parameters such as Gaussian Process and CMA-ES.
    /// </summary>
    public abstract OrderedDict<string, object?> SampleRelative(
        Study study, FrozenTrial trial, OrderedDict<string, BaseDistribution> searchSpace);

    /// <summary>
    ///     Sample a parameter for a given distribution.
    ///
    ///     This method is called only for the parameters not contained in the search space
    ///     returned by <see cref="SampleRelative" />. This method is suitable for sampling
    ///     algorithms that do not use relationship between parameters such as random sampling
    ///     and TPE.
    /// </summary>
    public abstract object? SampleIndependent(
        Study study, FrozenTrial trial, string paramName, BaseDistribution paramDistribution);

    /// <summary>
    ///     Trial pre-processing: called during trial initialization, just before the
    ///     <see cref="InferRelativeSearchSpace" /> call.
    /// </summary>
    public virtual void BeforeTrial(Study study, FrozenTrial trial)
    {
    }

    /// <summary>
    ///     Trial post-processing: called after the objective function returns and right before
    ///     the trial is finished and its state is stored.
    /// </summary>
    /// <param name="values">Resulting trial values. Guaranteed to not be <c>null</c> if trial succeeded.</param>
    public virtual void AfterTrial(Study study, FrozenTrial trial, TrialState state, IReadOnlyList<double>? values)
    {
    }

    /// <summary>
    ///     Reseed sampler's random number generator.
    ///
    ///     This method is called by the <see cref="Study" /> instance if trials are executed in
    ///     parallel with the option <c>nJobs &gt; 1</c>. In that case, the sampler instance will
    ///     be replicated including the state of the random number generator, and they may suggest
    ///     the same values. To prevent this issue, this method assigns a different seed to each
    ///     random number generator.
    /// </summary>
    public virtual void ReseedRng()
    {
    }

    /// <summary>Python <c>_raise_error_if_multi_objective</c>.</summary>
    internal void RaiseErrorIfMultiObjective(Study study)
    {
        if (study.IsMultiObjective())
            throw new ValueError(
                "If the study is being used for multi-objective optimization, " +
                $"{GetType().Name} cannot be used.");
    }
}

/// <summary>Module-level helpers of <c>optuna/samplers/_base.py</c>.</summary>
internal static class SamplerHelpers
{
    /// <summary>Python <c>_process_constraints_after_trial</c>.</summary>
    internal static void ProcessConstraintsAfterTrial(
        Func<FrozenTrial, IEnumerable<double>> constraintsFunc, Study study, FrozenTrial trial, TrialState state)
    {
        if (state != TrialState.Complete && state != TrialState.Pruned)
            return;

        List<double>? constraints = null;
        try
        {
            var con = constraintsFunc(trial).ToList();
            if (con.Any(double.IsNaN))
                throw new ValueError("Constraint values cannot be NaN.");
            constraints = con;
        }
        finally
        {
            study.Storage.SetTrialSystemAttr(
                trial.TrialId,
                ConstrainedOptimization.ConstraintsKey,
                constraints?.Cast<object?>().ToList());
        }
    }
}
