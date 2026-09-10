using System.Collections.Generic;
using System.Linq;
using OptunaSharp.Internal;

namespace OptunaSharp.Samplers.AgeMoea;

/// <summary>
///     Elite-population selection for <see cref="AgeMoea2Sampler" />: the AGE-MOEA-2 environmental
///     selection (<see cref="AgeMoea2Survival" />) applied to Optuna trials. Objective values are
///     sign-adjusted to minimization (like <c>_rank_population</c>), fed through the survival
///     geometry, and the survivors are returned ordered by front rank (ascending) then AGE-MOEA-2
///     crowding / survival score (descending) — so the extreme and most-spread solutions seed the
///     parent population first.
///
///     This is the AGE-MOEA-2-specific piece of Layer 2. Child generation stays Optuna-idiomatic
///     (SBX/PM over the transformed search space), so the sampler handles Optuna's full parameter
///     space; the byte-perfect pymoo-faithful AGE-MOEA-2 is Layer 1 (<see cref="AgeMoea2Optimizer" />).
///     Constrained optimization is not yet modeled here (unconstrained multi-objective only).
/// </summary>
public sealed class AgeMoea2ElitePopulationSelectionStrategy
{
    private readonly int _populationSize;

    public AgeMoea2ElitePopulationSelectionStrategy(int populationSize)
    {
        if (populationSize < 2)
            throw new NumSharp.ValueError("`population_size` must be greater than or equal to 2.");
        _populationSize = populationSize;
    }

    /// <summary>Select the elite population from the combined trials (Python <c>__call__</c>).</summary>
    public List<FrozenTrial> Select(Study study, List<FrozenTrial> population)
    {
        if (population.Count == 0)
            return new List<FrozenTrial>();

        var directions = study.Directions;
        int n = population.Count;
        int nObj = population[0].Values!.Count;

        var f = new double[n][];
        for (int i = 0; i < n; i++)
        {
            var values = population[i].Values!;
            var row = new double[nObj];
            for (int k = 0; k < nObj; k++)
            {
                double sign = directions[k] == StudyDirection.Maximize ? -1.0 : 1.0;
                row[k] = values[k] * sign;
            }
            f[i] = row;
        }

        var pop = new Population(f) { F = f, CV = new double[n] };
        var survival = new AgeMoea2Survival();
        bool[] mask = survival.SelectMask(pop, _populationSize);

        return Enumerable.Range(0, n)
            .Where(i => mask[i])
            .OrderBy(i => pop.Rank![i])
            .ThenByDescending(i => pop.Crowding![i])
            .ThenBy(i => i)
            .Select(i => population[i])
            .ToList();
    }
}
