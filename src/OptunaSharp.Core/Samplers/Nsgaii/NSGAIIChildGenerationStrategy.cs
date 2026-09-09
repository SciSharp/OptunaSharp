using System;
using System.Collections.Generic;
using NumSharp;
using OptunaSharp.Collections;
using OptunaSharp.Distributions;
using OptunaSharp.Internal;
using OptunaSharp.SearchSpace;

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
///     <para><b>Conditional parameters (OptunaSharp extension).</b> With <see cref="ConditionalParams" />
///     declared, the search space may hold parameters a parent lacks (inactive for it). Crossover then
///     goes through <see cref="NsgaiiCrossover.PerformConditionalCrossover" />, the parent copy takes
///     what the parent carries, and every conditional parameter inactive for the child is pruned before
///     the mutation pass — so it is neither mutated nor counted in the default <c>1 / n_params</c>. The
///     mutation pass itself is unchanged. Without rules the strategy is byte-for-byte the Optuna port.</para>
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
    private readonly ConditionalParams? _conditionalParams;

    /// <param name="conditionalParams">
    ///     OptunaSharp extension: activation rules for parameters that only exist under some values of
    ///     other parameters. <c>null</c> (default) keeps the Optuna-parity behavior.
    /// </param>
    public NSGAIIChildGenerationStrategy(
        BaseMutation? mutation,
        double? mutationProb,
        BaseCrossover crossover,
        double crossoverProb,
        double swappingProb,
        Func<FrozenTrial, IEnumerable<double>>? constraintsFunc,
        LazyRandomState rng,
        ConditionalParams? conditionalParams = null)
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
        _conditionalParams = conditionalParams;
    }

    /// <summary>Generate a child parameter dict from the given parent population (Python <c>__call__</c>).</summary>
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
            childParams = _conditionalParams is null
                ? NsgaiiCrossover.PerformCrossover(
                    _crossover, study, parentPopulation, searchSpace, _rng.Rng, _swappingProb, dominates)
                : NsgaiiCrossover.PerformConditionalCrossover(
                    _crossover, study, parentPopulation, searchSpace, _rng.Rng, _swappingProb, dominates,
                    _conditionalParams);
        }
        else
        {
            int parentPopulationSize = parentPopulation.Count;
            var parentParams = parentPopulation[(int)_rng.Rng.randint(0, parentPopulationSize)].ParamsDict;
            childParams = new OrderedDict<string, object?>();
            if (_conditionalParams is null)
            {
                foreach (var name in searchSpace.Keys)
                    childParams[name] = parentParams[name];
            }
            else
            {
                // The parent lacks exactly the conditional parameters that were inactive for it; copy what
                // it carries, then prune in case the declared rules and the parent's own branches disagree.
                foreach (var name in searchSpace.Keys)
                {
                    if (parentParams.TryGetValue(name, out var value))
                        childParams[name] = value;
                }
                _conditionalParams.Prune(childParams, searchSpace);
            }
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
