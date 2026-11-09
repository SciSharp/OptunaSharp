using System;
using System.Collections.Generic;
using NumSharp;
using OptunaSharp.Collections;
using OptunaSharp.Distributions;
using OptunaSharp.Internal;

namespace OptunaSharp.Samplers.Nsgaii;

/// <summary>
///     The default child-generation strategy of <see cref="NSGAIISampler" />. Port of
///     <c>optuna/samplers/nsgaii/_child_generation_strategy.py: NSGAIIChildGenerationStrategy</c>.
///
///     With probability <c>crossover_prob</c> a child is produced by crossover of tournament-selected
///     parents; otherwise a single parent is copied. Each parameter is then mutated with probability
///     <c>mutation_prob</c> (defaulting to <c>1 / len(params)</c>): if a mutation object is set the
///     value is mutated in place, otherwise the parameter is dropped and later resampled independently.
///
///     <para>This is a bit-exact Optuna port. The conditional-parameter ("Categorical Parameter Trees")
///     counterpart that evolves branch-defined parameters is
///     <see cref="OptunaSharp.Samplers.NsgaiiiCpt.NsgaiiiCptChildGenerationStrategy" />, used by
///     <see cref="OptunaSharp.Samplers.NSGAIIICptSampler" />.</para>
/// </summary>
public class NSGAIIChildGenerationStrategy
{
    private readonly double _crossoverProb;
    private readonly double? _mutationProb;
    private readonly double _swappingProb;
    private readonly BaseMutation? _mutation;
    private readonly BaseCrossover _crossover;
    private readonly Func<FrozenTrial, IEnumerable<double>>? _constraintsFunc;
    private readonly LazyRandomState _rng;

    /// <summary>Configures the genetic operators and selection randomness used for offspring.</summary>
    /// <param name="mutation">Numerical mutation operator used when an offspring coordinate is selected for mutation.</param>
    /// <param name="mutationProb">Per-parameter mutation probability; null selects the operator default.</param>
    /// <param name="crossover">Operator combining the selected numerical parent vectors.</param>
    /// <param name="crossoverProb">Probability of applying crossover to a mating.</param>
    /// <param name="swappingProb">Probability of taking a coordinate from the alternate parent.</param>
    /// <param name="constraintsFunc">Optional legacy callback returning constraint values; positive values violate a constraint.</param>
    /// <param name="rng">Random stream whose draw order is shared with the reference algorithm.</param>
    /// <exception cref="NumSharp.ValueError">A mutation, crossover, or swapping probability is outside its supported interval.</exception>
    public NSGAIIChildGenerationStrategy(
        BaseMutation? mutation,
        double? mutationProb,
        BaseCrossover crossover,
        double crossoverProb,
        double swappingProb,
        Func<FrozenTrial, IEnumerable<double>>? constraintsFunc,
        LazyRandomState rng)
    {
        if (!(mutationProb is null || (0.0 <= mutationProb && mutationProb <= 1.0)))
            throw new ValueError("`mutation_prob` must be None or a float value within the range [0.0, 1.0].");

        if (!(0.0 <= crossoverProb && crossoverProb <= 1.0))
            throw new ValueError("`crossover_prob` must be a float value within the range [0.0, 1.0].");

        if (!(0.0 <= swappingProb && swappingProb <= 1.0))
            throw new ValueError("`swapping_prob` must be a float value within the range [0.0, 1.0].");

        // In C# the crossover/mutation are strongly typed, so the isinstance guards are enforced
        // structurally; Optuna's ValueError for an invalid crossover has no analog to reach here.
        _crossoverProb = crossoverProb;
        _mutationProb = mutationProb;
        _swappingProb = swappingProb;
        _mutation = mutation;
        _crossover = crossover;
        _constraintsFunc = constraintsFunc;
        _rng = rng;
    }

    /// <summary>Generate a child parameter dict from the given parent population (Python <c>__call__</c>).</summary>
    /// <param name="study">Study supplying trial history, objective directions, and storage.</param>
    /// <param name="searchSpace">Ordered parameter distributions defining the coordinates used by this operation.</param>
    /// <param name="parentPopulation">Eligible parent trials from the current generation.</param>
    /// <returns>External offspring parameter values produced by the configured reproduction strategy.</returns>
    public OrderedDict<string, object?> Generate(
        Study study,
        OrderedDict<string, BaseDistribution> searchSpace,
        List<FrozenTrial> parentPopulation)
    {
        Func<FrozenTrial, FrozenTrial, IReadOnlyList<StudyDirection>, bool> dominates =
            ConstrainedOptimization.IsConstrainedOptimization(parentPopulation)
                ? NsgaiiConstraintsEvaluation.ConstrainedDominates
                : MultiObjective.Dominates;

        // We choose a child based on the specified crossover method.
        OrderedDict<string, object?> childParams;
        if ((double)_rng.Rng.rand() < _crossoverProb)
        {
            childParams = NsgaiiCrossover.PerformCrossover(
                _crossover, study, parentPopulation, searchSpace, _rng.Rng, _swappingProb, dominates);
        }
        else
        {
            int parentPopulationSize = parentPopulation.Count;
            var parentParams = parentPopulation[(int)_rng.Rng.randint(0, parentPopulationSize)].Params;
            childParams = new OrderedDict<string, object?>();
            foreach (var name in searchSpace.Keys)
                childParams[name] = parentParams[name];
        }

        int nParams = childParams.Count;
        double mutationProb = _mutationProb ?? 1.0 / Math.Max(1.0, nParams);

        var @params = new OrderedDict<string, object?>();
        foreach (var paramName in childParams.Keys)
        {
            if ((double)_rng.Rng.rand() >= mutationProb)
            {
                @params[paramName] = childParams[paramName];
            }
            else if (_mutation is not null)
            {
                var mutationValue = NsgaiiMutation.PerformMutation(
                    _mutation, _rng.Rng, study, searchSpace[paramName], childParams[paramName]);
                if (mutationValue is not null)
                    @params[paramName] = mutationValue;
            }
        }
        return @params;
    }
}
