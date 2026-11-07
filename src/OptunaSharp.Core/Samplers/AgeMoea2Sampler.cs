using System;
using System.Collections.Generic;
using OptunaSharp.Collections;
using OptunaSharp.Distributions;
using OptunaSharp.Internal;
using OptunaSharp.Samplers.AgeMoea;
using OptunaSharp.Samplers.Nsgaii;
using OptunaSharp.SearchSpace;

namespace OptunaSharp.Samplers;

/// <summary>
///     Multi-objective sampler using AGE-MOEA-2 (Layer 2 of the AGE-MOEA-2 port). The Optuna-idiomatic
///     face of the algorithm — an ask/tell <see cref="BaseGASampler" /> that reconstructs the
///     population from completed trials exactly like <see cref="NSGAIISampler" />, but replaces
///     NSGA-II's rank + crowding-distance elite selection with AGE-MOEA-2's geometry-based
///     environmental selection (<see cref="AgeMoea2ElitePopulationSelectionStrategy" /> over
///     <see cref="AgeMoea.AgeMoea2Survival" />). Children are generated with SBX + polynomial mutation
///     over Optuna's transformed search space, so the sampler works across Optuna's full parameter
///     space (float / int / categorical). For <i>conditional</i> (branch-defined) parameters use
///     <see cref="NSGAIIICptSampler" /> (NSGA-III-CPT); this sampler has no such extension.
///
///     AGE-MOEA-2 (Panichella, GECCO 2022, doi:10.1145/3512290.3528732) is absent from Optuna core, so
///     — like LM-CMA and JANUS — this is a beyond-core sampler. <b>Scope of the parity guarantee:</b>
///     the byte-perfect, pymoo-faithful AGE-MOEA-2 lives in Layer 1
///     (<see cref="AgeMoea.AgeMoea2Optimizer" />, generation-by-generation bit-exact vs
///     <c>pymoo.minimize</c>). This Layer-2 sampler is validated <b>behaviorally</b>: it uses Optuna's
///     operators (a different RNG stream than pymoo's), so it is not pymoo-bit-exact — it brings
///     AGE-MOEA-2's survival to Optuna's define-by-run workflow. Unconstrained multi-objective.
/// </summary>
public class AgeMoea2Sampler : BaseGASampler
{
    private readonly RandomSampler _randomSampler;
    private readonly LazyRandomState _rng;
    private readonly IntersectionSearchSpace _searchSpace;

    private readonly AgeMoea2ElitePopulationSelectionStrategy _eliteStrategy;
    private readonly Func<Study, OrderedDict<string, BaseDistribution>, List<FrozenTrial>, OrderedDict<string, object?>>
        _childGenerationStrategy;
    private readonly Action<Study, FrozenTrial, TrialState, IReadOnlyList<double>?> _afterTrialStrategy;

    /// <param name="populationSize">Number of individuals (trials) per generation. Must be &gt;= 2.</param>
    /// <param name="mutation">
    ///     Mutation applied when creating children. Null selects <see cref="Nsgaii.PolynomialMutation" />
    ///     with <c>eta = 20</c>, AGE-MOEA-2's default mutation.
    /// </param>
    /// <param name="mutationProb">Per-parameter mutation probability. <c>null</c> ⇒ <c>1 / len(params)</c>.</param>
    /// <param name="crossover">Crossover applied when creating children. Defaults to <see cref="SBXCrossover" /> (<c>eta = 15</c>).</param>
    /// <param name="crossoverProb">Probability that a crossover occurs.</param>
    /// <param name="swappingProb">Probability of swapping each parameter during crossover.</param>
    /// <param name="seed">Seed for the random number generator.</param>
    /// <summary>Configures AGE-MOEA-2 survival and the genetic operators used to produce each generation.</summary>
    /// <exception cref="NumSharp.ValueError">Population size is below two or the crossover parent count, or a configured reproduction probability is invalid.</exception>
    public AgeMoea2Sampler(
        int populationSize = 50,
        BaseMutation? mutation = null,
        double? mutationProb = null,
        BaseCrossover? crossover = null,
        double crossoverProb = 0.9,
        double swappingProb = 0.5,
        int? seed = null)
        : base(populationSize)
    {
        if (populationSize < 2)
            throw new NumSharp.ValueError("`population_size` must be greater than or equal to 2.");

        mutation ??= new Nsgaii.PolynomialMutation(eta: 20.0);
        crossover ??= new SBXCrossover(eta: 15.0);

        if (populationSize < crossover.NParents)
            throw new NumSharp.ValueError(
                $"Using {crossover}," +
                $" the population size should be greater than or equal to {crossover.NParents}." +
                $" The specified `population_size` is {populationSize}.");

        _randomSampler = new RandomSampler(seed);
        _rng = new LazyRandomState(seed);
        _searchSpace = new IntersectionSearchSpace();

        _eliteStrategy = new AgeMoea2ElitePopulationSelectionStrategy(populationSize);
        _childGenerationStrategy = new NSGAIIChildGenerationStrategy(
            mutation, mutationProb, crossover, crossoverProb, swappingProb, null, _rng).Generate;
        _afterTrialStrategy = new NSGAIIAfterTrialStrategy(null).Call;
    }

    /// <inheritdoc />
    public override void ReseedRng()
    {
        _randomSampler.ReseedRng();
        // Python: self._rng.rng.seed(), fresh OS entropy: 128 bits through numpy's own key construction,
        // so concurrent workers never share a stream (THREADING.md §6.3).
        OsEntropySeed.Reseed(_rng.Rng);
    }

    /// <inheritdoc />
    public override OrderedDict<string, BaseDistribution> InferRelativeSearchSpace(Study study, FrozenTrial trial)
    {
        var searchSpace = new OrderedDict<string, BaseDistribution>();
        foreach (var (name, distribution) in _searchSpace.Calculate(study))
        {
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
        return _eliteStrategy.Select(study, combined);
    }

    /// <inheritdoc />
    public override OrderedDict<string, object?> SampleRelative(
        Study study, FrozenTrial trial, OrderedDict<string, BaseDistribution> searchSpace)
    {
        int generation = GetTrialGeneration(study, trial);
        var parentPopulation = GetParentPopulation(study, generation);
        if (parentPopulation.Count == 0)
            return new OrderedDict<string, object?>();
        return _childGenerationStrategy(study, searchSpace, parentPopulation);
    }

    /// <inheritdoc />
    public override object? SampleIndependent(
        Study study, FrozenTrial trial, string paramName, BaseDistribution paramDistribution)
        => _randomSampler.SampleIndependent(study, trial, paramName, paramDistribution);

    /// <inheritdoc />
    public override void BeforeTrial(Study study, FrozenTrial trial)
        => _randomSampler.BeforeTrial(study, trial);

    /// <inheritdoc />
    /// <exception cref="System.Exception">A configured after-trial or constraint callback fails; its exception propagates.</exception>
    public override void AfterTrial(Study study, FrozenTrial trial, TrialState state, IReadOnlyList<double>? values)
    {
        _afterTrialStrategy(study, trial, state, values);
        _randomSampler.AfterTrial(study, trial, state, values);
    }
}
