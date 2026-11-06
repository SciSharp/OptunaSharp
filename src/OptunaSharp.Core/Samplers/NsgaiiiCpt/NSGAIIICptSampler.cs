using System;
using System.Collections.Generic;
using System.Security.Cryptography;
using NumSharp;
using OptunaSharp.Collections;
using OptunaSharp.Distributions;
using OptunaSharp.Internal;
using OptunaSharp.Samplers.Nsgaii;
using OptunaSharp.Samplers.Nsgaiii;
using OptunaSharp.Samplers.NsgaiiiCpt;
using OptunaSharp.SearchSpace;

namespace OptunaSharp.Samplers;

/// <summary>
///     NSGA-III-CPT ("NSGA-III with <b>C</b>ategorical <b>P</b>arameter <b>T</b>rees") — an OptunaSharp original
///     that extends NSGA-III to optimize <i>conditional</i> (define-by-run, branch-defined) hyperparameters. It
///     is a standalone copy of <see cref="NSGAIIISampler" />: same many-objective genetic core (reference-point
///     niching via <see cref="NSGAIIIElitePopulationSelectionStrategy" />, the shared crossovers, mutations and
///     ranking), plus the conditional-parameter machinery that <see cref="NSGAIIISampler" /> deliberately no
///     longer carries.
///
///     <para><b>Why it exists.</b> Optuna's GA samplers evolve only the <i>intersection</i> search space, so a
///     parameter suggested under one branch (<c>if optimizer == "adam": beta1 = suggest(...)</c>) drops out of
///     the evolved space after the first trial that lacks it and is resampled at random ever after — it gets no
///     crossover, mutation or inheritance. NSGA-III-CPT takes a declarative <see cref="ConditionalParams" /> rule
///     set that says which parameters are conditional on which activator values; those parameters then stay in
///     the relative search space (<see cref="NsgaiiiCptSearchSpace" />), are crossed over, mutated and inherited
///     whenever active (<see cref="NsgaiiiCptChildGenerationStrategy" /> /
///     <see cref="NsgaiiiCptCrossover" />), and are simply absent whenever not. This is the tree of
///     categorically-gated parameters the "CPT" name denotes. See the NSGA-III-CPT documentation page
///     (<c>docs/website-src/docs/our-methods/nsga-iii-cpt.md</c>).</para>
///
///     <para><b>Relationship to NSGA-III.</b> Constructed with <c>conditionalParams: null</c> this sampler is
///     behaviorally identical to <see cref="NSGAIIISampler" /> (the CPT search space and child generation both
///     fall back to the shared Optuna-parity fold), so it is a strict superset. The conditional behavior is a
///     deliberate divergence from Optuna and has no upstream analog; the underlying NSGA-III algorithm is the
///     bit-exact port documented on <see cref="NSGAIIISampler" />.</para>
/// </summary>
/// <seealso cref="NSGAIIISampler" />
/// <seealso cref="ConditionalParams" />
public class NSGAIIICptSampler : BaseGASampler
{
    private readonly RandomSampler _randomSampler;
    private readonly LazyRandomState _rng;
    private readonly NsgaiiiCptSearchSpace _searchSpace;
    private readonly ConditionalParams? _conditionalParams;

    private readonly Func<Study, List<FrozenTrial>, List<FrozenTrial>> _elitePopulationSelectionStrategy;
    private readonly Func<Study, OrderedDict<string, BaseDistribution>, List<FrozenTrial>, OrderedDict<string, object?>>
        _childGenerationStrategy;
    private readonly Action<Study, FrozenTrial, TrialState, IReadOnlyList<double>?> _afterTrialStrategy;

    /// <param name="populationSize">
    ///     Number of individuals (trials) in a generation. Must be &gt;= <c>crossover.NParents</c>
    ///     (3 for <see cref="UNDXCrossover" />/<see cref="SPXCrossover" />, else 2).
    /// </param>
    /// <param name="mutation">
    ///     Mutation applied when creating child individuals. If <c>null</c>, the parameter selected
    ///     for mutation is instead resampled independently using <see cref="RandomSampler" />.
    /// </param>
    /// <param name="mutationProb">
    ///     Probability of mutating each parameter. If <c>null</c>, <c>1 / len(parent.params)</c> is used.
    /// </param>
    /// <param name="crossover">
    ///     Crossover applied when creating child individuals. <see cref="UniformCrossover" /> is
    ///     always applied to categorical parameters, and by default to the others.
    /// </param>
    /// <param name="crossoverProb">Probability that a crossover occurs when creating a new individual.</param>
    /// <param name="swappingProb">Probability of swapping each parameter of the parents during crossover.</param>
    /// <param name="seed">Seed for random number generator.</param>
    /// <param name="constraintsFunc">
    ///     Deprecated (v5.0.0). Optional function computing objective constraints (a value strictly
    ///     larger than 0 means a constraint is violated). Handled by constrained domination.
    /// </param>
    /// <param name="referencePoints">
    ///     A 2-D array with <c>n_objectives</c> columns: the reference points that determine who
    ///     survives from the borderline front (survival favors sparsely populated reference points).
    ///     When <c>null</c> (default), `uniformly` spread points are generated from
    ///     <paramref name="dividingParameter" />.
    /// </param>
    /// <param name="dividingParameter">
    ///     Determines the density of the default reference points — how many divisions are made
    ///     between reference points on each axis. Smaller means fewer reference points. Unused when
    ///     <paramref name="referencePoints" /> is not <c>null</c>.
    /// </param>
    /// <param name="elitePopulationSelectionStrategy">Overrides elite selection.</param>
    /// <param name="childGenerationStrategy">
    ///     Overrides child generation. A custom strategy receives the (conditional-widened) search space and must
    ///     itself cope with parents that lack conditional parameters; the built-in
    ///     <see cref="NsgaiiiCptChildGenerationStrategy" /> is the one that honors
    ///     <paramref name="conditionalParams" />.
    /// </param>
    /// <param name="afterTrialStrategy">Overrides the after-trial procedure.</param>
    /// <param name="conditionalParams">
    ///     The activation rules for parameters that only exist under some values of other parameters — the reason
    ///     to reach for this sampler over <see cref="NSGAIIISampler" />. Declared parameters are evolved
    ///     (crossed over, mutated, inherited) whenever active instead of falling out of the intersection search
    ///     space and being resampled at random on every trial. <c>null</c> makes this sampler behave exactly like
    ///     <see cref="NSGAIIISampler" />.
    /// </param>
    /// <summary>Configures NSGA-III-CPT reference-point selection, conditional-aware reproduction, and trial-completion hooks.</summary>
    /// <exception cref="NumSharp.ValueError">Population size is below two or the crossover parent count, or a configured reproduction probability is invalid.</exception>
    public NSGAIIICptSampler(
        int populationSize = 50,
        BaseMutation? mutation = null,
        double? mutationProb = null,
        BaseCrossover? crossover = null,
        double crossoverProb = 0.9,
        double swappingProb = 0.5,
        int? seed = null,
        Func<FrozenTrial, IEnumerable<double>>? constraintsFunc = null,
        double[,]? referencePoints = null,
        int dividingParameter = 3,
        Func<Study, List<FrozenTrial>, List<FrozenTrial>>? elitePopulationSelectionStrategy = null,
        Func<Study, OrderedDict<string, BaseDistribution>, List<FrozenTrial>, OrderedDict<string, object?>>?
            childGenerationStrategy = null,
        Action<Study, FrozenTrial, TrialState, IReadOnlyList<double>?>? afterTrialStrategy = null,
        ConditionalParams? conditionalParams = null)
        : base(populationSize)
    {
        // NSGA-III-CPT is an OptunaSharp original; its conditional behavior has no Optuna analog and the
        // underlying NSGA-III algorithm is treated as stable, so no ExperimentalWarning is emitted.

        // TODO(ohta): Reconsider the default value of each parameter.
        if (populationSize < 2)
            throw new ValueError("`population_size` must be greater than or equal to 2.");

        if (constraintsFunc is not null)
        {
            string msg = string.Format(Deprecations.Template, "`constraints_func`", "5.0.0", "7.0.0");
            OptunaWarnings.Warn(
                $"{msg} Use `optuna.trial.Trial.set_constraint` instead.", WarningCategory.FutureWarning);
        }

        crossover ??= new UniformCrossover(swappingProb);

        if (populationSize < crossover.NParents)
            throw new ValueError(
                $"Using {crossover}," +
                $" the population size should be greater than or equal to {crossover.NParents}." +
                $" The specified `population_size` is {populationSize}.");

        _randomSampler = new RandomSampler(seed);
        _rng = new LazyRandomState(seed);
        _searchSpace = new NsgaiiiCptSearchSpace(conditionalParams);
        _conditionalParams = conditionalParams;

        _elitePopulationSelectionStrategy = elitePopulationSelectionStrategy
            ?? new NSGAIIIElitePopulationSelectionStrategy(
                populationSize, _rng, constraintsFunc, referencePoints, dividingParameter).Select;
        _childGenerationStrategy = childGenerationStrategy
            ?? new NsgaiiiCptChildGenerationStrategy(
                mutation, mutationProb, crossover, crossoverProb, swappingProb, constraintsFunc, _rng,
                conditionalParams).Generate;
        _afterTrialStrategy = afterTrialStrategy
            ?? new NSGAIIAfterTrialStrategy(constraintsFunc).Call;
    }

    /// <inheritdoc />
    public override void ReseedRng()
    {
        _randomSampler.ReseedRng();
        // Python: self._rng.rng.seed() — reseed from OS entropy.
        Span<byte> buffer = stackalloc byte[4];
        RandomNumberGenerator.Fill(buffer);
        _rng.Rng.seed(BitConverter.ToUInt32(buffer));
    }

    /// <inheritdoc />
    public override OrderedDict<string, BaseDistribution> InferRelativeSearchSpace(Study study, FrozenTrial trial)
    {
        var searchSpace = new OrderedDict<string, BaseDistribution>();
        foreach (var (name, distribution) in _searchSpace.Calculate(study))
        {
            // The untransform of _SearchSpaceTransform does not assume a single value, so single
            // value objects are sampled via sample_independent, not sample_relative.
            if (distribution.Single())
                continue;
            searchSpace[name] = distribution;
        }
        return searchSpace;
    }

    /// <inheritdoc />
    public override List<FrozenTrial> SelectParent(Study study, int generation)
    {
        var combined = GetPopulation(study, generation - 1);
        combined.AddRange(GetParentPopulation(study, generation - 1));
        return _elitePopulationSelectionStrategy(study, combined);
    }

    /// <inheritdoc />
    public override OrderedDict<string, object?> SampleRelative(
        Study study, FrozenTrial trial, OrderedDict<string, BaseDistribution> searchSpace)
    {
        int generation = GetTrialGeneration(study, trial);
        var parentPopulation = GetParentPopulation(study, generation);
        if (parentPopulation.Count == 0)
            return new OrderedDict<string, object?>();
        var child = _childGenerationStrategy(study, searchSpace, parentPopulation);
        // Off unless ConditionalParams.PreserveDormantValues is set. Parks inactive conditional values out of the
        // equations and resumes reactivating ones from their latent value instead of a random draw; no-op otherwise.
        _conditionalParams?.ApplyDormantCarry(study, trial, child, parentPopulation, searchSpace);
        return child;
    }

    /// <inheritdoc />
    public override object? SampleIndependent(
        Study study, FrozenTrial trial, string paramName, BaseDistribution paramDistribution)
    {
        // Randomly sampled here: (1) a parameter in the initial/first generation, (2) a parameter to
        // mutate, (3) a parameter excluded from the intersection search space.
        return _randomSampler.SampleIndependent(study, trial, paramName, paramDistribution);
    }

    /// <inheritdoc />
    public override void BeforeTrial(Study study, FrozenTrial trial)
    {
        _randomSampler.BeforeTrial(study, trial);
    }

    /// <inheritdoc />
    /// <exception cref="System.Exception">A configured after-trial or constraint callback fails; its exception propagates.</exception>
    public override void AfterTrial(Study study, FrozenTrial trial, TrialState state, IReadOnlyList<double>? values)
    {
        System.Diagnostics.Debug.Assert(
            state is TrialState.Complete or TrialState.Fail or TrialState.Pruned);
        _afterTrialStrategy(study, trial, state, values);
        _randomSampler.AfterTrial(study, trial, state, values);
    }
}
