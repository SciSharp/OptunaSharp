using System;
using System.Collections.Generic;
using System.Linq;
using NumSharp;
using OptunaSharp.Collections;
using OptunaSharp.Distributions;
using OptunaSharp.Internal;
using OptunaSharp.SearchSpace;

namespace OptunaSharp.Samplers.Nsgaii;

/// <summary>
///     Port of <c>optuna/samplers/nsgaii/_crossover.py</c> — the crossover dispatcher that splits
///     the search space into numerical (fed to the <see cref="BaseCrossover" />) and categorical
///     (uniform crossover) parts, selects parents by binary tournament, and re-draws until the child
///     lands inside the search space. RNG draw order matches Optuna exactly.
/// </summary>
internal static class NsgaiiCrossover
{
    /// <summary>Selects parent trials and retries reproduction until the child is inside its distributions.</summary>
    /// <param name="crossover">Operator combining the selected numerical parent vectors.</param>
    /// <param name="study">Study supplying trial history, objective directions, and storage.</param>
    /// <param name="parentPopulation">Eligible parent trials from the current generation.</param>
    /// <param name="searchSpace">Ordered parameter distributions defining the coordinates used by this operation.</param>
    /// <param name="rng">Random stream whose draw order is shared with the reference algorithm.</param>
    /// <param name="swappingProb">Probability of taking a coordinate from the alternate parent.</param>
    /// <param name="dominates">Direction-aware dominance predicate used by parent selection.</param>
    /// <returns>External child parameter values inside the declared distributions.</returns>
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

    /// <summary>Combines numerical and categorical parent coordinates into one candidate trial.</summary>
    /// <param name="parents">Selected parent individuals in crossover order.</param>
    /// <param name="crossover">Operator combining the selected numerical parent vectors.</param>
    /// <param name="study">Study supplying trial history, objective directions, and storage.</param>
    /// <param name="rng">Random stream whose draw order is shared with the reference algorithm.</param>
    /// <param name="swappingProb">Probability of taking a coordinate from the alternate parent.</param>
    /// <param name="categoricalSearchSpace">Categorical coordinates handled by uniform parent-value inheritance.</param>
    /// <param name="numericalSearchSpace">Numerical coordinates handled by the configured crossover.</param>
    /// <param name="numericalTransform">Transform for numerical parent vectors, or null when no numerical coordinates exist.</param>
    /// <returns>One candidate child in external parameter representation.</returns>
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
        var childParams = new OrderedDict<string, object?>(categoricalSearchSpace.Count + numericalSearchSpace.Count);

        if (categoricalSearchSpace.Count > 0)
        {
            var catKeys = categoricalSearchSpace.Keys.ToList();
            // parents_categorical_params = [[parents[0][p]...], [parents[-1][p]...]]  (2 rows).
            var parent0Cat = catKeys.Select(p => parents[0].Params[p]).ToArray();
            var parentLastCat = catKeys.Select(p => parents[^1].Params[p]).ToArray();

            var childCategorical = InlinedCategoricalUniformCrossover(
                parent0Cat, parentLastCat, rng, swappingProb, catKeys.Count);
            for (int j = 0; j < catKeys.Count; j++)
                childParams[catKeys[j]] = childCategorical[j];
        }

        if (numericalTransform is null)
            return childParams;

        // The following is applied only for numerical parameters.
        // Optuna transforms each parent's numerical params ({p: parent.params[p] for p in numerical_search_space}). The
        // transform reads exactly those names from the parent's own params, so it is given them directly: no per-parent
        // dictionary copy and no NumSharp array per parent (plan 14), the same values in the same column order.
        var parentsNumerical = new double[parents.Count][];
        for (int i = 0; i < parents.Count; i++)
            parentsNumerical[i] = numericalTransform.TransformValues(parents[i].Params);

        var childNumericalArray = crossover.Crossover(
            parentsNumerical, rng, study, numericalTransform.RawBounds);
        // Read the crossover child as it is: the untransform Optuna applies, without wrapping the managed child in a
        // NumSharp array first.
        var childNumericalParams = numericalTransform.UntransformValues(childNumericalArray);
        foreach (var (k, v) in childNumericalParams)
            childParams[k] = v;

        return childParams;
    }

    /// <summary>Chooses the distinct parents required by the crossover using dominance tournaments.</summary>
    /// <param name="crossover">Operator combining the selected numerical parent vectors.</param>
    /// <param name="study">Study supplying trial history, objective directions, and storage.</param>
    /// <param name="parentPopulation">Eligible parent trials from the current generation.</param>
    /// <param name="rng">Random stream whose draw order is shared with the reference algorithm.</param>
    /// <param name="dominates">Direction-aware dominance predicate used by parent selection.</param>
    /// <returns>The distinct trial parents required by the configured crossover.</returns>
    private static List<FrozenTrial> SelectParents(
        BaseCrossover crossover,
        Study study,
        IReadOnlyList<FrozenTrial> parentPopulation,
        NumPyRandom rng,
        Func<FrozenTrial, FrozenTrial, IReadOnlyList<StudyDirection>, bool> dominates)
    {
        var parents = new List<FrozenTrial>(crossover.NParents);
        var selectedIds = new HashSet<int>();
        for (int i = 0; i < crossover.NParents; i++)
        {
            // Optuna's `[t for t in parent_population if t._trial_id not in parent_ids]`. Before the first pick nothing is
            // excluded, so that list is the population itself, in its order: it is used as it is instead of being copied.
            IReadOnlyList<FrozenTrial> remaining = selectedIds.Count == 0
                ? parentPopulation
                : parentPopulation.Where(t => !selectedIds.Contains(t.TrialId)).ToList();
            var parent = SelectParent(study, remaining, rng, dominates);
            parents.Add(parent);
            selectedIds.Add(parent.TrialId);
        }
        return parents;
    }

    /// <summary>Chooses a parent using a two-trial dominance tournament.</summary>
    /// <param name="study">Study supplying trial history, objective directions, and storage.</param>
    /// <param name="parentPopulation">Eligible parent trials from the current generation.</param>
    /// <param name="rng">Random stream whose draw order is shared with the reference algorithm.</param>
    /// <param name="dominates">Direction-aware dominance predicate used by parent selection.</param>
    /// <returns>The tournament-winning trial.</returns>
    private static FrozenTrial SelectParent(
        Study study,
        IReadOnlyList<FrozenTrial> parentPopulation,
        NumPyRandom rng,
        Func<FrozenTrial, FrozenTrial, IReadOnlyList<StudyDirection>, bool> dominates)
    {
        int populationSize = parentPopulation.Count;
        var candidate0 = parentPopulation[(int)NumPyDraws.RandInt(rng, 0, populationSize)];
        var candidate1 = parentPopulation[(int)NumPyDraws.RandInt(rng, 0, populationSize)];

        // TODO(ohta): Consider crowding distance.
        return dominates(candidate0, candidate1, study.Directions) ? candidate0 : candidate1;
    }

    /// <summary>Checks whether every proposed external value belongs to its declared distribution.</summary>
    /// <param name="params">External parameter values recorded along the objective path.</param>
    /// <param name="searchSpace">Ordered parameter distributions defining the coordinates used by this operation.</param>
    /// <returns>True if every proposed value belongs to its distribution.</returns>
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
    /// <param name="parent0">First parent trial supplying inherited categorical coordinates.</param>
    /// <param name="parentLast">Last selected parent used by the crossover.</param>
    /// <param name="rng">Random stream whose draw order is shared with the reference algorithm.</param>
    /// <param name="swappingProb">Probability of taking a coordinate from the alternate parent.</param>
    /// <param name="nCategorical">Number of categorical coordinates inherited from the parent arrays.</param>
    /// <returns>Inherited categorical values in search-space order.</returns>
    private static object?[] InlinedCategoricalUniformCrossover(
        object?[] parent0, object?[] parentLast, NumPyRandom rng, double swappingProb, int nCategorical)
    {
        // masks = (rng.rand(n) >= swapping_prob).astype(int); child = parent_params[masks, range(n)].
        var r = NumPyDraws.Rand(rng, nCategorical);
        var child = new object?[nCategorical];
        for (int j = 0; j < nCategorical; j++)
            child[j] = r[j] >= swappingProb ? parentLast[j] : parent0[j];
        return child;
    }
}
