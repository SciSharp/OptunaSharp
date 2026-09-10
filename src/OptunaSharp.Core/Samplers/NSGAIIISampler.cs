using System;
using System.Collections.Generic;
using System.Security.Cryptography;
using NumSharp;
using OptunaSharp.Collections;
using OptunaSharp.Distributions;
using OptunaSharp.Internal;
using OptunaSharp.Samplers.Nsgaii;
using OptunaSharp.Samplers.Nsgaiii;
using OptunaSharp.SearchSpace;

namespace OptunaSharp.Samplers;

/// <summary>
///     Multi-objective sampler using the NSGA-III algorithm. Port of
///     <c>optuna/samplers/_nsgaiii/_sampler.py: NSGAIIISampler</c> (experimental since Optuna 3.2.0).
///
///     NSGA-III ("Nondominated Sorting Genetic Algorithm III") is a modification of
///     <see cref="NSGAIISampler" /> for many-objective optimization. It reuses the whole NSGA-II
///     genetic machinery (<see cref="BaseGASampler" />, child generation, crossovers, mutations) and
///     only swaps the elite-selection strategy: instead of crowding distance, the borderline front is
///     truncated by <b>reference-point niching</b> (see
///     <see cref="NSGAIIIElitePopulationSelectionStrategy" />). See
///     <a href="https://doi.org/10.1109/TEVC.2013.2281535">Deb &amp; Jain (2014)</a>.
///
///     <para><b>Parity.</b> The default path stays in exact MT19937 lock-step with Optuna (generation
///     attribution, parent cache, tournament/crossover/mutation draws, and the niching
///     <c>rng.shuffle</c>). Objective-value niching is bit-exact except for a ~1-ULP managed-vs-LAPACK
///     drift in the hyperplane-fit solve, which only changes selection when it flips a niche decision
///     within ULPs of a tie — see the <c>[Misaligned]</c> note on
///     <see cref="NSGAIIIElitePopulationSelectionStrategy" />.</para>
///
///     <para><b>Conditional parameters (OptunaSharp extension).</b> Optuna's GA samplers evolve only the
///     intersection search space, so a define-by-run parameter suggested under a branch drops out after
///     the first trial that lacks it and is resampled at random ever after. Passing
///     <see cref="ConditionalParams" /> declares which parameters are conditional on which activator
///     values; they then stay in the relative search space and are crossed over, mutated and inherited
///     whenever active, and are simply absent whenever not. Off by default (parity untouched).</para>
/// </summary>
public class NSGAIIISampler : BaseGASampler
{
    private readonly RandomSampler _randomSampler;
    private readonly LazyRandomState _rng;
    private readonly IntersectionSearchSpace _searchSpace;
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
    /// <param name="childGenerationStrategy">Overrides child generation.</param>
    /// <param name="afterTrialStrategy">Overrides the after-trial procedure.</param>
    /// <param name="conditionalParams">
    ///     OptunaSharp extension (no Optuna analog): activation rules for parameters that only exist under
    ///     some values of other parameters — see <see cref="ConditionalParams" />. Declared parameters are
    ///     evolved (crossed over, mutated, inherited) whenever active instead of falling out of the
    ///     intersection search space and being resampled at random on every trial. <c>null</c> (default)
    ///     keeps the Optuna-parity behavior. Only the built-in child generation honors it; a custom
    ///     <paramref name="childGenerationStrategy" /> receives the widened search space and must cope with
    ///     parents that lack conditional parameters itself.
    /// </param>
    public NSGAIIISampler(
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
        // @experimental_class("3.2.0") wraps __init__ and warns before any validation runs.
        OptunaWarnings.Warn(
            "NSGAIIISampler is experimental (supported from v3.2.0). The interface can change in the future.",
            WarningCategory.ExperimentalWarning);

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
        _searchSpace = new IntersectionSearchSpace(conditionalParams: conditionalParams);
        _conditionalParams = conditionalParams;

        _elitePopulationSelectionStrategy = elitePopulationSelectionStrategy
            ?? new NSGAIIIElitePopulationSelectionStrategy(
                populationSize, _rng, constraintsFunc, referencePoints, dividingParameter).Select;
        _childGenerationStrategy = childGenerationStrategy
            ?? new NSGAIIChildGenerationStrategy(
                mutation, mutationProb, crossover, crossoverProb, swappingProb, constraintsFunc, _rng,
                conditionalParams).Generate;
        _afterTrialStrategy = afterTrialStrategy
            ?? new NSGAIIAfterTrialStrategy(constraintsFunc).Call;
    }

    public override void ReseedRng()
    {
        _randomSampler.ReseedRng();
        // Python: self._rng.rng.seed() — reseed from OS entropy.
        Span<byte> buffer = stackalloc byte[4];
        RandomNumberGenerator.Fill(buffer);
        _rng.Rng.seed(BitConverter.ToUInt32(buffer));
    }

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

    public override List<FrozenTrial> SelectParent(Study study, int generation)
    {
        var combined = GetPopulation(study, generation - 1);
        combined.AddRange(GetParentPopulation(study, generation - 1));
        return _elitePopulationSelectionStrategy(study, combined);
    }

    public override OrderedDict<string, object?> SampleRelative(
        Study study, FrozenTrial trial, OrderedDict<string, BaseDistribution> searchSpace)
    {
        int generation = GetTrialGeneration(study, trial);
        var parentPopulation = GetParentPopulation(study, generation);
        if (parentPopulation.Count == 0)
            return new OrderedDict<string, object?>();
        var child = _childGenerationStrategy(study, searchSpace, parentPopulation);
        // OptunaSharp extension: opt-in, off unless ConditionalParams.PreserveDormantValues is set. Parks
        // inactive conditional values out of the equations and resumes reactivating ones from their latent
        // value instead of a random draw. No-op otherwise, so the parity path is untouched.
        _conditionalParams?.ApplyDormantCarry(study, trial, child, parentPopulation, searchSpace);
        return child;
    }

    public override object? SampleIndependent(
        Study study, FrozenTrial trial, string paramName, BaseDistribution paramDistribution)
    {
        // Randomly sampled here: (1) a parameter in the initial/first generation, (2) a parameter to
        // mutate, (3) a parameter excluded from the intersection search space.
        return _randomSampler.SampleIndependent(study, trial, paramName, paramDistribution);
    }

    public override void BeforeTrial(Study study, FrozenTrial trial)
    {
        _randomSampler.BeforeTrial(study, trial);
    }

    public override void AfterTrial(Study study, FrozenTrial trial, TrialState state, IReadOnlyList<double>? values)
    {
        System.Diagnostics.Debug.Assert(
            state is TrialState.Complete or TrialState.Fail or TrialState.Pruned);
        _afterTrialStrategy(study, trial, state, values);
        _randomSampler.AfterTrial(study, trial, state, values);
    }
}
