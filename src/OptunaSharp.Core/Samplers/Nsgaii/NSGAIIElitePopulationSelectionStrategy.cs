using System;
using System.Collections.Generic;
using System.Linq;
using NumSharp;
using OptunaSharp.Internal;

namespace OptunaSharp.Samplers.Nsgaii;

/// <summary>
///     The default elite-population selection strategy of <see cref="NSGAIISampler" />. Port of
///     <c>optuna/samplers/nsgaii/_elite_population_selection_strategy.py</c>.
///
///     Ranks the combined population by (constrained) non-domination, then fills the elite
///     population rank by rank; the rank that overflows <c>population_size</c> is truncated by
///     crowding distance (descending). The resulting order is load-bearing — it seeds the parent
///     population that later generations index into with <c>rng.choice</c>.
/// </summary>
public class NSGAIIElitePopulationSelectionStrategy
{
    /// <summary>Python <c>&lt;</c> on floats: a &lt; b is the only ordering primitive; -0.0 == 0.0 and NaN compares equal (kept stable).</summary>
    private sealed class PyLessComparer : IComparer<double>
    {
        /// <summary>Shared stateless instance used by the containing algorithm.</summary>
        internal static readonly PyLessComparer Instance = new();

        /// <summary>Orders objective values using Python less-than semantics, leaving unordered NaNs tied.</summary>
        /// <param name="a">Scalar operand used in the comparison or numerical transformation.</param>
        /// <param name="b">Scalar operand used in the comparison or numerical transformation.</param>
        /// <returns>Negative, zero, or positive according to the Python-style ordering.</returns>
        public int Compare(double a, double b) => a < b ? -1 : (b < a ? 1 : 0);
    }

    private readonly int _populationSize;
    private readonly Func<FrozenTrial, IEnumerable<double>>? _constraintsFunc;

    /// <summary>Configures nondomination ranking and crowding-distance truncation to a fixed population size.</summary>
    /// <param name="populationSize">Number of individuals retained or evaluated per generation.</param>
    /// <param name="constraintsFunc">Optional legacy callback returning constraint values; positive values violate a constraint.</param>
    /// <exception cref="NumSharp.ValueError">Population size is below two.</exception>
    public NSGAIIElitePopulationSelectionStrategy(
        int populationSize, Func<FrozenTrial, IEnumerable<double>>? constraintsFunc = null)
    {
        if (populationSize < 2)
            throw new ValueError("`population_size` must be greater than or equal to 2.");

        _populationSize = populationSize;
        _constraintsFunc = constraintsFunc;
    }

    /// <summary>Select the elite population from the given trials (Python <c>__call__</c>).</summary>
    /// <param name="study">Study supplying trial history, objective directions, and storage.</param>
    /// <param name="population">Individuals available for survival or reproduction.</param>
    /// <returns>The individuals retained by the configured survival rule, in selection order.</returns>
    public List<FrozenTrial> Select(Study study, List<FrozenTrial> population)
    {
        bool isConstrained = ConstrainedOptimization.IsConstrainedOptimization(population);
        NsgaiiConstraintsEvaluation.ValidateConstraints(population, isConstrained);
        var populationPerRank = RankPopulation(population, study.Directions, isConstrained);

        var elitePopulation = new List<FrozenTrial>();
        foreach (var individuals in populationPerRank)
        {
            if (elitePopulation.Count + individuals.Count < _populationSize)
            {
                elitePopulation.AddRange(individuals);
            }
            else
            {
                int n = _populationSize - elitePopulation.Count;
                CrowdingDistanceSort(individuals);
                elitePopulation.AddRange(individuals.Take(n));
                break;
            }
        }

        return elitePopulation;
    }

    /// <summary>
    ///     Python <c>_rank_population</c>. Module-level in Optuna's
    ///     <c>nsgaii/_elite_population_selection_strategy.py</c> and imported by the NSGA-III strategy,
    ///     so it is <c>internal static</c> here and shared with
    ///     <see cref="OptunaSharp.Samplers.Nsgaiii.NSGAIIIElitePopulationSelectionStrategy" />.
    /// </summary>
    /// <param name="population">Individuals available for survival or reproduction.</param>
    /// <param name="directions">One optimization direction per objective.</param>
    /// <param name="isConstrained">Whether constraint feasibility participates in ranking.</param>
    /// <returns>Trial fronts ordered by nondomination rank.</returns>
    internal static List<List<FrozenTrial>> RankPopulation(
        IReadOnlyList<FrozenTrial> population, IReadOnlyList<StudyDirection> directions, bool isConstrained)
    {
        if (population.Count == 0)
            return new List<List<FrozenTrial>>();

        var objectiveValues = new double[population.Count][];
        for (int i = 0; i < population.Count; i++)
        {
            var values = population[i].Values!;
            var row = new double[values.Count];
            for (int k = 0; k < values.Count; k++)
            {
                double sign = directions[k] == StudyDirection.Maximize ? -1.0 : 1.0;
                row[k] = values[k] * sign;
            }
            objectiveValues[i] = row;
        }

        double[]? penalty = isConstrained ? NsgaiiConstraintsEvaluation.EvaluatePenalty(population) : null;
        var dominationRanks = MultiObjective.FastNonDominationRank(objectiveValues, penalty: penalty);

        int maxRank = dominationRanks.Max();
        var populationPerRank = new List<List<FrozenTrial>>(maxRank + 1);
        for (int r = 0; r <= maxRank; r++)
            populationPerRank.Add(new List<FrozenTrial>());
        for (int i = 0; i < population.Count; i++)
        {
            int rank = dominationRanks[i];
            if (rank == -1)
                continue;
            populationPerRank[rank].Add(population[i]);
        }

        return populationPerRank;
    }

    /// <summary>Python <c>_crowding_distance_sort</c>: sort by crowding distance descending (in place).</summary>
    /// <param name="population">Individuals available for survival or reproduction.</param>
    private static void CrowdingDistanceSort(List<FrozenTrial> population)
    {
        var manhattanDistances = CalcCrowdingDistance(population);
        StableSortInPlace(population, x => manhattanDistances.GetValueOrDefault(x.Number, 0.0));
        population.Reverse();
    }

    /// <summary>Python <c>_calc_crowding_distance</c>.</summary>
    /// <param name="population">Individuals available for survival or reproduction.</param>
    /// <returns>Diversity distances indexed by trial number.</returns>
    private static Dictionary<int, double> CalcCrowdingDistance(List<FrozenTrial> population)
    {
        var manhattanDistances = new Dictionary<int, double>();
        if (population.Count == 0)
            return manhattanDistances;

        int nObjectives = population[0].Values!.Count;
        for (int i = 0; i < nObjectives; i++)
        {
            int obj = i;
            StableSortInPlace(population, x => x.Values![obj]);

            // If all trials have the same value in this dimension, it makes no difference.
            if (population[0].Values![i] == population[^1].Values![i])
                continue;

            // vs = [-inf] + values + [inf].
            int m = population.Count;
            var vs = new double[m + 2];
            vs[0] = double.NegativeInfinity;
            for (int j = 0; j < m; j++)
                vs[j + 1] = population[j].Values![i];
            vs[m + 1] = double.PositiveInfinity;

            // Smallest / largest finite value: v_min = next(x for x in vs if x != -inf);
            // v_max = next(x for x in reversed(vs) if x != inf).
            double vMin = double.NegativeInfinity;
            for (int k = 0; k < vs.Length; k++)
            {
                if (vs[k] != double.NegativeInfinity)
                {
                    vMin = vs[k];
                    break;
                }
            }
            double vMax = double.PositiveInfinity;
            for (int k = vs.Length - 1; k >= 0; k--)
            {
                if (vs[k] != double.PositiveInfinity)
                {
                    vMax = vs[k];
                    break;
                }
            }

            double width = vMax - vMin;
            if (width <= 0)  // width == 0 or width == -inf.
                width = 1.0;

            for (int j = 0; j < m; j++)
            {
                // inf - inf and (-inf) - (-inf) is considered to be zero.
                double gap = vs[j] == vs[j + 2] ? 0.0 : vs[j + 2] - vs[j];
                int number = population[j].Number;
                manhattanDistances[number] = manhattanDistances.GetValueOrDefault(number, 0.0) + gap / width;
            }
        }

        return manhattanDistances;
    }

    /// <summary>Python <c>list.sort(key=...)</c> — a stable ascending sort under Python's <c>&lt;</c>, in place.</summary>
    /// <param name="population">Individuals available for survival or reproduction.</param>
    /// <param name="key">Callback extracting the scalar sort key for each trial.</param>
    private static void StableSortInPlace(List<FrozenTrial> population, Func<FrozenTrial, double> key)
    {
        var sorted = population.OrderBy(key, PyLessComparer.Instance).ToList();
        population.Clear();
        population.AddRange(sorted);
    }
}
