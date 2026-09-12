// MOEA/D offspring generation — mate a pair of parents drawn from a subproblem's mating pool
// (neighborhood or whole population) into one child using the shared NSGA operators.
//
// Reference: Zhang & Li 2007; the operator plumbing mirrors pymoo's genetic mating (SBX + polynomial
// mutation, `pymoo/algorithms/moo/moead.py` uses SBX(prob=1, eta=20) + PM(eta=20)). MOEA/D is
// beyond-core; the operators are the same ones the NSGA-II family uses, so nothing is re-implemented.

using System;
using System.Collections.Generic;
using System.Linq;
using NumSharp;
using OptunaSharp.Collections;
using OptunaSharp.Distributions;
using OptunaSharp.Internal;
using OptunaSharp.Samplers.Nsgaii;

namespace OptunaSharp.Samplers.Moead;

/// <summary>
///     Produces one MOEA/D child from a mating pool: it draws <c>crossover.NParents</c> distinct parents
///     uniformly at random from the pool, applies the (numerical) <see cref="BaseCrossover" /> in the
///     transformed space and uniform crossover to categorical parameters, then mutates each numerical
///     parameter with probability <c>mutationProb</c> via the <see cref="BaseMutation" />. This reuses the
///     exact operator classes and transform the NSGA-II family uses; only the parent selection differs
///     (random from the neighborhood, as MOEA/D specifies, rather than binary tournament).
/// </summary>
internal static class MoeadOffspring
{
    internal static OrderedDict<string, object?> Generate(
        Study study,
        OrderedDict<string, BaseDistribution> searchSpace,
        IReadOnlyList<FrozenTrial> matingPool,
        BaseCrossover crossover,
        BaseMutation mutation,
        double? mutationProb,
        double swappingProb,
        NumPyRandom rng)
    {
        var parents = SelectParents(matingPool, crossover.NParents, rng);

        var numericalSearchSpace = new OrderedDict<string, BaseDistribution>();
        var categoricalSearchSpace = new OrderedDict<string, BaseDistribution>();
        foreach (var (key, value) in searchSpace)
        {
            if (value is FloatDistribution or IntDistribution)
                numericalSearchSpace[key] = value;
            else
                categoricalSearchSpace[key] = value;
        }

        var childParams = new OrderedDict<string, object?>();

        // Categorical: uniform crossover between the first and last selected parent (same rule the
        // NSGA-II crossover dispatcher uses for categorical parameters).
        if (categoricalSearchSpace.Count > 0)
        {
            var catKeys = categoricalSearchSpace.Keys.ToList();
            var parent0Cat = catKeys.Select(p => parents[0].ParamsDict[p]).ToArray();
            var parentLastCat = catKeys.Select(p => parents[^1].ParamsDict[p]).ToArray();
            var r = rng.rand(catKeys.Count).ToArray<double>();
            for (int j = 0; j < catKeys.Count; j++)
                childParams[catKeys[j]] = r[j] >= swappingProb ? parentLastCat[j] : parent0Cat[j];
        }

        // Numerical: transform each parent into the numerical search space, run the crossover operator
        // (bounded SBX by default → child stays in range), and untransform.
        if (numericalSearchSpace.Count > 0)
        {
            var transform = new SearchSpaceTransform(numericalSearchSpace);
            var numKeys = numericalSearchSpace.Keys.ToList();
            var parentsNumerical = new double[parents.Count][];
            for (int i = 0; i < parents.Count; i++)
            {
                var row = new OrderedDict<string, object?>();
                foreach (var k in numKeys)
                    row[k] = parents[i].ParamsDict[k];
                parentsNumerical[i] = transform.Transform(row).ToArray<double>();
            }

            var childNumericalArray = crossover.Crossover(parentsNumerical, rng, study, transform.RawBounds);
            var childNumericalParams = transform.Untransform(np.array(childNumericalArray));
            foreach (var (k, v) in childNumericalParams)
                childParams[k] = v;
        }

        // Mutation: polynomial mutation on the numerical parameters (categoricals were set by uniform
        // crossover). Default per-parameter probability is 1 / len(params), as in the NSGA-II family.
        int nParams = childParams.Count;
        double prob = mutationProb ?? 1.0 / Math.Max(1.0, nParams);
        foreach (var key in numericalSearchSpace.Keys)
        {
            if ((double)rng.rand() >= prob)
                continue;
            var mutated = NsgaiiMutation.PerformMutation(
                mutation, rng, study, searchSpace[key], childParams[key]);
            if (mutated is not null)
                childParams[key] = mutated;
        }

        return childParams;
    }

    /// <summary>Draw <paramref name="nParents" /> distinct members of <paramref name="pool" /> uniformly at
    ///     random (repeats only if the pool is smaller than the requested count).</summary>
    private static List<FrozenTrial> SelectParents(IReadOnlyList<FrozenTrial> pool, int nParents, NumPyRandom rng)
    {
        int n = pool.Count;
        var parents = new List<FrozenTrial>(nParents);
        var used = new HashSet<int>();
        for (int i = 0; i < nParents; i++)
        {
            int idx;
            if (used.Count >= n)
            {
                idx = (int)rng.randint(0, n);   // pool exhausted: allow a repeat rather than loop forever
            }
            else
            {
                do { idx = (int)rng.randint(0, n); } while (!used.Add(idx));
            }
            parents.Add(pool[idx]);
        }
        return parents;
    }
}
