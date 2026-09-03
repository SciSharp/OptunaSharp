using System;
using System.Collections.Generic;
using System.Linq;
using NumSharp;
using OptunaSharp.Collections;
using OptunaSharp.Distributions;
using OptunaSharp.Internal;

namespace OptunaSharp.Samplers.Nsgaii;

/// <summary>
///     Port of <c>optuna/samplers/nsgaii/_crossover.py</c> — the crossover dispatcher that splits
///     the search space into numerical (fed to the <see cref="BaseCrossover" />) and categorical
///     (uniform crossover) parts, selects parents by binary tournament, and re-draws until the child
///     lands inside the search space. RNG draw order matches Optuna exactly.
/// </summary>
internal static class NsgaiiCrossover
{
    internal static OrderedDict<string, object?> PerformCrossover(
        BaseCrossover crossover,
        Study study,
        IReadOnlyList<FrozenTrial> parentPopulation,
        OrderedDict<string, BaseDistribution> searchSpace,
        NumPyRandom rng,
        double swappingProb,
        Func<FrozenTrial, FrozenTrial, IReadOnlyList<StudyDirection>, bool> dominates)
    {
        var numericalSearchSpace = new OrderedDict<string, BaseDistribution>();
        var categoricalSearchSpace = new OrderedDict<string, BaseDistribution>();
        foreach (var (key, value) in searchSpace)
        {
            if (value is FloatDistribution or IntDistribution)
                numericalSearchSpace[key] = value;
            else
                categoricalSearchSpace[key] = value;
        }

        SearchSpaceTransform? numericalTransform = null;
        if (numericalSearchSpace.Count != 0)
            numericalTransform = new SearchSpaceTransform(numericalSearchSpace);

        while (true)  // Repeat while parameters lie outside search space boundaries.
        {
            var parents = SelectParents(crossover, study, parentPopulation, rng, dominates);
            var childParams = TryCrossover(
                parents, crossover, study, rng, swappingProb,
                categoricalSearchSpace, numericalSearchSpace, numericalTransform);

            if (IsContained(childParams, searchSpace))
                return childParams;
        }
    }

    private static OrderedDict<string, object?> TryCrossover(
        List<FrozenTrial> parents,
        BaseCrossover crossover,
        Study study,
        NumPyRandom rng,
        double swappingProb,
        OrderedDict<string, BaseDistribution> categoricalSearchSpace,
        OrderedDict<string, BaseDistribution> numericalSearchSpace,
        SearchSpaceTransform? numericalTransform)
    {
        var childParams = new OrderedDict<string, object?>();

        if (categoricalSearchSpace.Count > 0)
        {
            var catKeys = categoricalSearchSpace.Keys.ToList();
            // parents_categorical_params = [[parents[0][p]...], [parents[-1][p]...]]  (2 rows).
            var parent0Cat = catKeys.Select(p => parents[0].ParamsDict[p]).ToArray();
            var parentLastCat = catKeys.Select(p => parents[^1].ParamsDict[p]).ToArray();

            var childCategorical = InlinedCategoricalUniformCrossover(
                parent0Cat, parentLastCat, rng, swappingProb, catKeys.Count);
            for (int j = 0; j < catKeys.Count; j++)
                childParams[catKeys[j]] = childCategorical[j];
        }

        if (numericalTransform is null)
            return childParams;

        // The following is applied only for numerical parameters.
        var numKeys = numericalSearchSpace.Keys.ToList();
        var parentsNumerical = new double[parents.Count][];
        for (int i = 0; i < parents.Count; i++)
        {
            var row = new OrderedDict<string, object?>();
            foreach (var k in numKeys)
                row[k] = parents[i].ParamsDict[k];
            parentsNumerical[i] = numericalTransform.Transform(row).ToArray<double>();
        }

        var childNumericalArray = crossover.Crossover(
            parentsNumerical, rng, study, numericalTransform.RawBounds);
        var childNumericalParams = numericalTransform.Untransform(np.array(childNumericalArray));
        foreach (var (k, v) in childNumericalParams)
            childParams[k] = v;

        return childParams;
    }

    private static List<FrozenTrial> SelectParents(
        BaseCrossover crossover,
        Study study,
        IReadOnlyList<FrozenTrial> parentPopulation,
        NumPyRandom rng,
        Func<FrozenTrial, FrozenTrial, IReadOnlyList<StudyDirection>, bool> dominates)
    {
        var parents = new List<FrozenTrial>();
        var selectedIds = new HashSet<int>();
        for (int i = 0; i < crossover.NParents; i++)
        {
            var remaining = parentPopulation.Where(t => !selectedIds.Contains(t.TrialId)).ToList();
            var parent = SelectParent(study, remaining, rng, dominates);
            parents.Add(parent);
            selectedIds.Add(parent.TrialId);
        }
        return parents;
    }

    private static FrozenTrial SelectParent(
        Study study,
        IReadOnlyList<FrozenTrial> parentPopulation,
        NumPyRandom rng,
        Func<FrozenTrial, FrozenTrial, IReadOnlyList<StudyDirection>, bool> dominates)
    {
        int populationSize = parentPopulation.Count;
        var candidate0 = parentPopulation[(int)rng.randint(0, populationSize)];
        var candidate1 = parentPopulation[(int)rng.randint(0, populationSize)];

        // TODO(ohta): Consider crowding distance.
        return dominates(candidate0, candidate1, study.Directions) ? candidate0 : candidate1;
    }

    private static bool IsContained(OrderedDict<string, object?> @params, OrderedDict<string, BaseDistribution> searchSpace)
    {
        foreach (var (paramName, param) in @params)
        {
            var distribution = searchSpace[paramName];
            if (!distribution.Contains(distribution.ToInternalRepr(param)))
                return false;
        }
        return true;
    }

    /// <summary>
    ///     Python <c>_inlined_categorical_uniform_crossover</c>. Categorical params are passed
    ///     untransformed, so uniform crossover is done here directly against the two parents'
    ///     external values.
    /// </summary>
    private static object?[] InlinedCategoricalUniformCrossover(
        object?[] parent0, object?[] parentLast, NumPyRandom rng, double swappingProb, int nCategorical)
    {
        // masks = (rng.rand(n) >= swapping_prob).astype(int); child = parent_params[masks, range(n)].
        var r = rng.rand(nCategorical).ToArray<double>();
        var child = new object?[nCategorical];
        for (int j = 0; j < nCategorical; j++)
            child[j] = r[j] >= swappingProb ? parentLast[j] : parent0[j];
        return child;
    }
}
