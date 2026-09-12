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
    /// <summary>Message shared by samplers when a parameter falls back to independent sampling.</summary>
    internal const string IndependentSamplingWarningTemplate =
        "The parameter `{0}` in Trial#{1} is sampled independently " +
        "using `{2}` instead of `{3}`, potentially " +
        "degrading the optimization performance. This fallback happened because " +
        "{4}. You can suppress this warning by setting " +
        "`warn_independent_sampling` to `False` in the constructor of `{3}` if " +
        "this independent sampling is intended behavior.";

    /// <summary>Returns the concrete sampler name for logs and diagnostics.</summary>
    /// <returns>The runtime type name, without a namespace.</returns>
    public override string ToString() => GetType().Name;

    /// <summary>
    ///     Infer the search space that will be used by relative sampling in the target trial.
    ///
    ///     This method is called right before <see cref="SampleRelative" />, and the search space
    ///     returned by this method is passed to it. The parameters not contained in the search
    ///     space will be sampled by using <see cref="SampleIndependent" />.
    /// </summary>
    /// <param name="study">Study supplying trial history, directions, and storage.</param>
    /// <param name="trial">Current trial snapshot; copy it before modifying its contents.</param>
    /// <returns>Parameter distributions eligible for joint sampling, in sampler-defined order.</returns>
    public abstract OrderedDict<string, BaseDistribution> InferRelativeSearchSpace(Study study, FrozenTrial trial);

    /// <summary>
    ///     Sample parameters in a given search space.
    ///
    ///     The trial invokes this method lazily when it first needs relative parameters and caches
    ///     the result for its remaining suggestions. This method is suitable for sampling algorithms
    ///     that use relationship between parameters such as Gaussian Process and CMA-ES.
    /// </summary>
    /// <param name="study">Study supplying trial history, directions, and storage.</param>
    /// <param name="trial">Current trial snapshot; copy it before modifying its contents.</param>
    /// <param name="searchSpace">Distributions returned by <see cref="InferRelativeSearchSpace" />.</param>
    /// <returns>Jointly sampled external parameter values; omitted parameters use independent sampling.</returns>
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
    /// <param name="study">Study supplying trial history, directions, and storage.</param>
    /// <param name="trial">Current trial snapshot; copy it before modifying its contents.</param>
    /// <param name="paramName">Name of the parameter being suggested.</param>
    /// <param name="paramDistribution">Distribution defining the parameter's domain and scale.</param>
    /// <returns>A value in the distribution's external representation, including null categorical choices.</returns>
    public abstract object? SampleIndependent(
        Study study, FrozenTrial trial, string paramName, BaseDistribution paramDistribution);

    /// <summary>
    ///     Trial pre-processing: called during trial initialization, just before the
    ///     <see cref="InferRelativeSearchSpace" /> call.
    /// </summary>
    /// <param name="study">Study that owns the newly initialized trial.</param>
    /// <param name="trial">New trial snapshot available before search-space inference.</param>
    public virtual void BeforeTrial(Study study, FrozenTrial trial)
    {
    }

    /// <summary>
    ///     Trial post-processing: called after the objective function returns and right before
    ///     the trial is finished and its state is stored.
    /// </summary>
    /// <param name="study">Study that owns the trial being finished.</param>
    /// <param name="trial">Trial snapshot before its terminal state is written to storage.</param>
    /// <param name="state">Terminal state being recorded: complete, pruned, or failed.</param>
    /// <param name="values">Resulting trial values in objective order; non-null for a successful trial.</param>
    public virtual void AfterTrial(Study study, FrozenTrial trial, TrialState state, IReadOnlyList<double>? values)
    {
    }

    /// <summary>
    ///     Reseed sampler's random number generator.
    ///
    ///     Parallel optimization calls this hook before a worker's sequential trial loop. Concrete
    ///     samplers refresh their random streams so workers do not deliberately reuse a seeded
    ///     sequence. OptunaSharp workers share the study's sampler instance; this hook does not clone
    ///     it or make concurrent sampling reproducible.
    /// </summary>
    public virtual void ReseedRng()
    {
    }

    /// <summary>Rejects a multi-objective study for samplers that require a scalar objective.</summary>
    /// <param name="study">Study whose objective count is checked.</param>
    /// <exception cref="ValueError">The study has more than one objective.</exception>
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
    /// <summary>Evaluates and persists legacy constraints for completed or pruned trials.</summary>
    /// <param name="constraintsFunc">Callback returning constraint values; positive values indicate violations.</param>
    /// <param name="study">Study whose storage receives the constraint attribute.</param>
    /// <param name="trial">Trial snapshot passed to the callback.</param>
    /// <param name="state">Terminal trial state; failed trials are ignored.</param>
    /// <exception cref="ValueError">The callback returns a NaN constraint.</exception>
    /// <exception cref="Exception">The user callback or storage provider fails; its exception propagates.</exception>
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
            // Persist null even after callback failure so missing/invalid constraints cannot be
            // mistaken for a feasible trial by a later constrained sampler.
            study.Storage.SetTrialSystemAttr(
                trial.TrialId,
                ConstrainedOptimization.ConstraintsKey,
                constraints?.Cast<object?>().ToList());
        }
    }
}
