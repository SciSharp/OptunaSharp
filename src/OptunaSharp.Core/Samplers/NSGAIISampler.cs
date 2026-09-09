using System;
using System.Collections.Generic;
using System.Linq;
using System.Security.Cryptography;
using NumSharp;
using OptunaSharp.Collections;
using OptunaSharp.Distributions;
using OptunaSharp.Internal;
using OptunaSharp.Samplers.Nsgaii;
using OptunaSharp.SearchSpace;

namespace OptunaSharp.Samplers;

/// <summary>
///     Multi-objective sampler using the NSGA-II algorithm. Port of
///     <c>optuna/samplers/nsgaii/_sampler.py: NSGAIISampler</c>.
///
///     NSGA-II ("Nondominated Sorting Genetic Algorithm II") is a fast, elitist multi-objective
///     genetic algorithm. See <a href="https://doi.org/10.1109/4235.996017">Deb et al. (2002)</a>.
///
///     <para><b>Parity.</b> The default (<see cref="UniformCrossover" /> + no mutation object) path is
///     bit-exact against Optuna: uniform crossover copies parent values verbatim and mutated-away
///     parameters are resampled by the independent <see cref="RandomSampler" /> (both MT19937). Every
///     RNG draw runs through <see cref="LazyRandomState" />, and the generation / parent-cache system
///     attributes are byte-identical, so full study state matches. The arithmetic crossovers
///     (BLX-α/SBX/vSBX/SPX) stay bit-exact; <see cref="UNDXCrossover" /> is tolerance-bounded for
///     ≥2 numerical parameters (managed QR vs LAPACK — see its note) but keeps RNG lock-step.</para>
///
///     <para><b>Conditional parameters (OptunaSharp extension).</b> Optuna's GA samplers evolve only the
///     intersection search space, so a define-by-run parameter suggested under a branch drops out after
///     the first trial that lacks it and is resampled at random ever after. Passing
///     <see cref="ConditionalParams" /> declares which parameters are conditional on which activator
///     values; they then stay in the relative search space and are crossed over, mutated and inherited
///     whenever active, and are simply absent whenever not. Off by default (parity untouched).</para>
/// </summary>
public class NSGAIISampler : BaseGASampler
{
    private readonly RandomSampler _randomSampler;
    private readonly LazyRandomState _rng;
    private readonly IntersectionSearchSpace _searchSpace;

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
    /// <param name="elitePopulationSelectionStrategy">Experimental (v3.3.0). Overrides elite selection.</param>
    /// <param name="childGenerationStrategy">Experimental (v3.3.0). Overrides child generation.</param>
    /// <param name="afterTrialStrategy">Experimental (v3.3.0). Overrides the after-trial procedure.</param>
    /// <param name="conditionalParams">
    ///     OptunaSharp extension (no Optuna analog): activation rules for parameters that only exist under
    ///     some values of other parameters — see <see cref="ConditionalParams" />. Declared parameters are
    ///     evolved (crossed over, mutated, inherited) whenever active instead of falling out of the
    ///     intersection search space and being resampled at random on every trial. <c>null</c> (default)
    ///     keeps the Optuna-parity behavior. Only the built-in child generation honors it; a custom
    ///     <paramref name="childGenerationStrategy" /> receives the widened search space and must cope with
    ///     parents that lack conditional parameters itself.
    /// </param>
    public NSGAIISampler(
        int populationSize = 50,
        BaseMutation? mutation = null,
        double? mutationProb = null,
        BaseCrossover? crossover = null,
        double crossoverProb = 0.9,
        double swappingProb = 0.5,
        int? seed = null,
        Func<FrozenTrial, IEnumerable<double>>? constraintsFunc = null,
        Func<Study, List<FrozenTrial>, List<FrozenTrial>>? elitePopulationSelectionStrategy = null,
        Func<Study, OrderedDict<string, BaseDistribution>, List<FrozenTrial>, OrderedDict<string, object?>>?
            childGenerationStrategy = null,
        Action<Study, FrozenTrial, TrialState, IReadOnlyList<double>?>? afterTrialStrategy = null,
        ConditionalParams? conditionalParams = null)
        : base(populationSize)
    {
        // TODO(ohta): Reconsider the default value of each parameter.
        if (populationSize < 2)
            throw new ValueError("`population_size` must be greater than or equal to 2.");

        if (constraintsFunc is not null)
        {
            string msg = string.Format(Deprecations.Template, "`constraints_func`", "5.0.0", "7.0.0");
            OptunaWarnings.Warn(
                $"{msg} Use `optuna.trial.Trial.set_constraint` instead.", WarningCategory.FutureWarning);
        }
        if (afterTrialStrategy is not null)
            WarnExperimentalArgument("after_trial_strategy");
        if (childGenerationStrategy is not null)
            WarnExperimentalArgument("child_generation_strategy");
        if (elitePopulationSelectionStrategy is not null)
            WarnExperimentalArgument("elite_population_selection_strategy");

        crossover ??= new UniformCrossover(swappingProb);

        if (populationSize < crossover.NParents)
            throw new ValueError(
                $"Using {crossover}," +
                $" the population size should be greater than or equal to {crossover.NParents}." +
                $" The specified `population_size` is {populationSize}.");

        _randomSampler = new RandomSampler(seed);
        _rng = new LazyRandomState(seed);
        _searchSpace = new IntersectionSearchSpace(conditionalParams: conditionalParams);

        _elitePopulationSelectionStrategy = elitePopulationSelectionStrategy
            ?? new NSGAIIElitePopulationSelectionStrategy(populationSize, constraintsFunc).Select;
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
        return _childGenerationStrategy(study, searchSpace, parentPopulation);
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

    /// <summary>Python <c>optuna._experimental.warn_experimental_argument</c>.</summary>
    private static void WarnExperimentalArgument(string name)
        => OptunaWarnings.Warn(
            $"Argument ``{name}`` is an experimental feature. The interface can change in the future.",
            WarningCategory.ExperimentalWarning);
}
