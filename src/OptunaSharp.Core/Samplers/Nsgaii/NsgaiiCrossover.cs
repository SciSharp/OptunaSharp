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

    /// <summary>
    ///     The <see cref="ConditionalParams" /> variant of <see cref="PerformCrossover" /> — an OptunaSharp
    ///     extension with no Optuna analog, reached only when rules are declared, so the parity path above
    ///     is untouched. Parents may lack conditional parameters: the operators run over what the selected
    ///     parents share, the rest is inherited from a carrier, and parameters inactive for the child are
    ///     pruned before the containment check (so an inactive value can neither fail it nor reach the
    ///     trial). Parent selection is the same binary tournament as the parity path.
    /// </summary>
    /// <param name="crossover">Operator combining the selected numerical parent vectors.</param>
    /// <param name="study">Study supplying trial history, objective directions, and storage.</param>
    /// <param name="parentPopulation">Eligible parent trials from the current generation.</param>
    /// <param name="searchSpace">Ordered parameter distributions defining the coordinates used by this operation.</param>
    /// <param name="rng">Random stream whose draw order is shared with the reference algorithm.</param>
    /// <param name="swappingProb">Probability of taking a coordinate from the alternate parent.</param>
    /// <param name="dominates">Direction-aware dominance predicate used by parent selection.</param>
    /// <param name="conditionalParams">Optional activation and dormant-value policy for conditional parameters.</param>
    /// <returns>A valid offspring containing only currently active conditional coordinates.</returns>
    internal static OrderedDict<string, object?> PerformConditionalCrossover(
        BaseCrossover crossover,
        Study study,
        IReadOnlyList<FrozenTrial> parentPopulation,
        OrderedDict<string, BaseDistribution> searchSpace,
        NumPyRandom rng,
        double swappingProb,
        Func<FrozenTrial, FrozenTrial, IReadOnlyList<StudyDirection>, bool> dominates,
        ConditionalParams conditionalParams)
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

        while (true)  // Repeat while parameters lie outside search space boundaries.
        {
            var parents = SelectParents(crossover, study, parentPopulation, rng, dominates);
            var childParams = TryConditionalCrossover(
                parents, crossover, study, rng, swappingProb, categoricalSearchSpace, numericalSearchSpace,
                conditionalParams);
            conditionalParams.Prune(childParams, searchSpace);

            if (IsContained(childParams, searchSpace))
                return childParams;
        }
    }

    /// <summary>
    ///     <see cref="TryCrossover" /> for parents that may lack conditional parameters. Categorical:
    ///     uniform crossover between the first and last parent over the keys both carry (one mask draw per
    ///     shared key, in search-space order — the same draws as the parity path when nothing is missing);
    ///     a key only one of them carries is inherited from it. Numerical: the crossover operator sees the
    ///     keys every parent carries (a well-formed parents × dims matrix in the transformed space, with
    ///     the transform built per parent set); a key only some parents carry is inherited from one of
    ///     them, drawn uniformly. A key no parent carries stays absent and is sampled independently later
    ///     if the objective asks for it. Child keys keep search-space order, as in the parity path.
    ///
    ///     <para><b>Deactivation is math-neutral.</b> A numerical parameter that is already <i>definitely
    ///     inactive</i> for this child (its activators are categorical, crossed in the block above, and they
    ///     exclude it) is dropped here <b>before</b> any crossover work: it never enters the operator's
    ///     input (so it cannot change the dimensionality or the operator's internal draws for the other
    ///     parameters) and never triggers the partial-carrier <c>rng.randint</c>. A multi-parent crossover
    ///     therefore consumes exactly the RNG it would in a world where the deactivated parameter did not
    ///     exist. Parameters whose activation cannot yet be decided (an activator still to be drawn, e.g. a
    ///     numerical/int activator) are kept and handled by <see cref="ConditionalParams.Prune" /> as before.</para>
    /// </summary>
    /// <param name="parents">Selected parent individuals in crossover order.</param>
    /// <param name="crossover">Operator combining the selected numerical parent vectors.</param>
    /// <param name="study">Study supplying trial history, objective directions, and storage.</param>
    /// <param name="rng">Random stream whose draw order is shared with the reference algorithm.</param>
    /// <param name="swappingProb">Probability of taking a coordinate from the alternate parent.</param>
    /// <param name="categoricalSearchSpace">Categorical coordinates handled by uniform parent-value inheritance.</param>
    /// <param name="numericalSearchSpace">Numerical coordinates handled by the configured crossover.</param>
    /// <param name="conditionalParams">Optional activation and dormant-value policy for conditional parameters.</param>
    /// <returns>One conditional offspring candidate before distribution containment checks.</returns>
    private static OrderedDict<string, object?> TryConditionalCrossover(
        List<FrozenTrial> parents,
        BaseCrossover crossover,
        Study study,
        NumPyRandom rng,
        double swappingProb,
        OrderedDict<string, BaseDistribution> categoricalSearchSpace,
        OrderedDict<string, BaseDistribution> numericalSearchSpace,
        ConditionalParams conditionalParams)
    {
        var childParams = new OrderedDict<string, object?>();

        if (categoricalSearchSpace.Count > 0)
        {
            var first = parents[0].ParamsDict;
            var last = parents[^1].ParamsDict;
            var shared = categoricalSearchSpace.Keys.Where(k => first.ContainsKey(k) && last.ContainsKey(k)).ToList();
            var crossed = shared.Count == 0
                ? Array.Empty<object?>()
                : InlinedCategoricalUniformCrossover(
                    shared.Select(k => first[k]).ToArray(), shared.Select(k => last[k]).ToArray(),
                    rng, swappingProb, shared.Count);

            int j = 0;
            foreach (var key in categoricalSearchSpace.Keys)
            {
                bool inFirst = first.ContainsKey(key), inLast = last.ContainsKey(key);
                if (inFirst && inLast)
                    childParams[key] = crossed[j++];
                else if (inFirst)
                    childParams[key] = first[key];
                else if (inLast)
                    childParams[key] = last[key];
            }
        }

        if (numericalSearchSpace.Count == 0)
            return childParams;

        var common = new OrderedDict<string, BaseDistribution>();
        var partialCarriers = new Dictionary<string, List<FrozenTrial>>(StringComparer.Ordinal);
        foreach (var (key, distribution) in numericalSearchSpace)
        {
            // Keep a parameter already known to be inactive for this child out of every crossover
            // equation and out of the partial-carrier draw below — it would only be pruned afterwards,
            // so letting it consume RNG would make deactivation perturb the other parameters' math.
            if (conditionalParams.ResolveActive(key, childParams) == false)
                continue;
            var carriers = parents.Where(p => p.ParamsDict.ContainsKey(key)).ToList();
            if (carriers.Count == parents.Count)
                common[key] = distribution;
            else if (carriers.Count > 0)
                partialCarriers[key] = carriers;
        }

        var crossedNumerical = new OrderedDict<string, object?>();
        if (common.Count > 0)
        {
            var transform = new SearchSpaceTransform(common);
            var parentsNumerical = new double[parents.Count][];
            for (int i = 0; i < parents.Count; i++)
            {
                var row = new OrderedDict<string, object?>();
                foreach (var key in common.Keys)
                    row[key] = parents[i].ParamsDict[key];
                parentsNumerical[i] = transform.Transform(row).ToArray<double>();
            }

            var childNumericalArray = crossover.Crossover(parentsNumerical, rng, study, transform.RawBounds);
            crossedNumerical = transform.Untransform(np.array(childNumericalArray));
        }

        foreach (var key in numericalSearchSpace.Keys)
        {
            if (crossedNumerical.TryGetValue(key, out var value))
            {
                childParams[key] = value;
            }
            else if (partialCarriers.TryGetValue(key, out var carriers))
            {
                var carrier = carriers.Count == 1 ? carriers[0] : carriers[(int)rng.randint(0, carriers.Count)];
                childParams[key] = carrier.ParamsDict[key];
            }
        }

        return childParams;
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
        var candidate0 = parentPopulation[(int)rng.randint(0, populationSize)];
        var candidate1 = parentPopulation[(int)rng.randint(0, populationSize)];

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
        var r = rng.rand(nCategorical).ToArray<double>();
        var child = new object?[nCategorical];
        for (int j = 0; j < nCategorical; j++)
            child[j] = r[j] >= swappingProb ? parentLast[j] : parent0[j];
        return child;
    }
}
