using System;
using System.Collections.Generic;
using System.Linq;
using NumSharp;
using OptunaSharp.Internal;

namespace OptunaSharp.Internal;

/// <summary>
///     Port of <c>optuna/study/_multi_objective.py</c>. The NumPy row-wise operations
///     (<c>np.unique(axis=0)</c> lexsort/dedupe) are reproduced in plain C# with identical
///     semantics — everything here is deterministic comparison logic, no RNG.
/// </summary>
internal static class MultiObjective
{
    /// <summary>Python <c>_get_pareto_front_trials_by_trials</c>.</summary>
    internal static List<FrozenTrial> GetParetoFrontTrialsByTrials(
        IEnumerable<FrozenTrial> trials,
        IReadOnlyList<StudyDirection> directions,
        bool considerConstraint = false)
    {
        // NOTE(nabenabe0928): Vectorization relies on all the trials being complete.
        var completeTrials = trials.Where(t => t.State == TrialState.Complete).ToList();
        if (considerConstraint)
            completeTrials = ConstrainedOptimization.GetFeasibleTrials(completeTrials);
        if (completeTrials.Count == 0)
            return new List<FrozenTrial>();

        if (completeTrials.Any(t => t.Values!.Count != directions.Count))
            throw new ValueError("The number of the values and the number of the objectives must be identical.");

        var lossValues = completeTrials
            .Select(t => t.Values!.Select((v, i) => NormalizeValue(v, directions[i])).ToArray())
            .ToArray();
        var onFront = IsParetoFront(lossValues, assumeUniqueLexsorted: false);
        var result = new List<FrozenTrial>();
        for (int i = 0; i < completeTrials.Count; i++)
        {
            if (onFront[i])
                result.Add(completeTrials[i]);
        }
        return result;
    }

    /// <summary>Python <c>_get_pareto_front_trials</c>.</summary>
    internal static List<FrozenTrial> GetParetoFrontTrials(Study study, bool considerConstraint = false)
        => GetParetoFrontTrialsByTrials(study.Trials, study.Directions, considerConstraint);

    /// <summary>
    ///     Python <c>_fast_non_domination_rank</c>: non-domination rank based on the fast
    ///     non-dominated sort algorithm of constrained NSGA-II. Correct only up to the
    ///     top-<paramref name="nBelow" /> solutions.
    /// </summary>
    internal static int[] FastNonDominationRank(double[][] lossValues, double[]? penalty = null, int? nBelow = null)
    {
        if (lossValues.Length == 0)
            return Array.Empty<int>();

        int below = nBelow ?? lossValues.Length;
        System.Diagnostics.Debug.Assert(below > 0, "n_below must be a positive integer.");

        if (penalty is null)
            return CalculateNondominationRank(lossValues, below);

        if (penalty.Length != lossValues.Length)
            throw new ValueError(
                "The length of penalty and loss_values must be same, but got " +
                $"len(penalty)={penalty.Length} and len(loss_values)={lossValues.Length}.");

        var ranks = Enumerable.Repeat(-1, lossValues.Length).ToArray();
        var feasibleIndices = new List<int>();
        var infeasibleIndices = new List<int>();
        for (int i = 0; i < penalty.Length; i++)
        {
            // NaN penalties fall into neither bucket in NumPy (`penalty <= 0` and `penalty > 0`
            // are both false) — but Optuna guards NaN out upstream, and the final assert would
            // fire; mirror by treating NaN as infeasible-last like the Python assert expects.
            if (penalty[i] <= 0)
                feasibleIndices.Add(i);
            else
                infeasibleIndices.Add(i);
        }

        // First, we calculate the domination rank for feasible trials.
        var feasibleLoss = feasibleIndices.Select(i => lossValues[i]).ToArray();
        var feasibleRanks = CalculateNondominationRank(feasibleLoss, below);
        for (int i = 0; i < feasibleIndices.Count; i++)
            ranks[feasibleIndices[i]] = feasibleRanks[i];
        below -= feasibleIndices.Count;

        // Second, we calculate the domination rank for infeasible trials.
        int topRankInfeasible = feasibleRanks.Length == 0 ? 0 : feasibleRanks.Max() + 1;
        var infeasiblePenalty = infeasibleIndices.Select(i => new[] { penalty[i] }).ToArray();
        var infeasibleRanks = CalculateNondominationRank(infeasiblePenalty, below);
        for (int i = 0; i < infeasibleIndices.Count; i++)
            ranks[infeasibleIndices[i]] = topRankInfeasible + infeasibleRanks[i];

        System.Diagnostics.Debug.Assert(ranks.All(r => r != -1), "All the rank must be updated.");
        return ranks;
    }

    /// <summary>Python <c>_is_pareto_front_nd</c>.</summary>
    private static bool[] IsParetoFrontNd(double[][] uniqueLexsortedLossValues)
    {
        // As column 0 is sorted, we do not need it to judge dominance.
        int nTrials = uniqueLexsortedLossValues.Length;
        int nObjectives = nTrials == 0 ? 0 : uniqueLexsortedLossValues[0].Length;
        var onFront = new bool[nTrials];
        var remainingIndices = Enumerable.Range(0, nTrials).ToList();

        while (remainingIndices.Count > 0)
        {
            // NOTE: trials[j] cannot dominate trials[i] for i < j because of lexsort.
            // Therefore, remaining_indices[0] is always non-dominated.
            int newNondominatedIndex = remainingIndices[0];
            onFront[newNondominatedIndex] = true;
            var nondominated = uniqueLexsortedLossValues[newNondominatedIndex];

            var next = new List<int>();
            foreach (int idx in remainingIndices)
            {
                // np.any(loss_values[idx, 1:] < nondominated[1:]).
                bool anyLess = false;
                for (int k = 1; k < nObjectives; k++)
                {
                    if (uniqueLexsortedLossValues[idx][k] < nondominated[k])
                    {
                        anyLess = true;
                        break;
                    }
                }
                if (anyLess)
                    next.Add(idx);
            }
            remainingIndices = next;
        }

        return onFront;
    }

    /// <summary>Python <c>_is_pareto_front_2d</c>.</summary>
    private static bool[] IsParetoFront2d(double[][] uniqueLexsortedLossValues)
    {
        int nTrials = uniqueLexsortedLossValues.Length;
        var onFront = new bool[nTrials];
        if (nTrials == 0)
            return onFront;
        onFront[0] = true;
        double cummin = uniqueLexsortedLossValues[0][1];
        for (int i = 1; i < nTrials; i++)
        {
            double value1 = uniqueLexsortedLossValues[i][1];
            onFront[i] = value1 < cummin;  // True if cummin value1 is new minimum.
            cummin = Math.Min(cummin, value1);
        }
        return onFront;
    }

    /// <summary>Python <c>_is_pareto_front_for_unique_sorted</c>.</summary>
    private static bool[] IsParetoFrontForUniqueSorted(double[][] uniqueLexsortedLossValues)
    {
        int nObjectives = uniqueLexsortedLossValues.Length == 0 ? 0 : uniqueLexsortedLossValues[0].Length;
        if (nObjectives == 1)
        {
            var onFront = new bool[uniqueLexsortedLossValues.Length];
            if (onFront.Length > 0)
                onFront[0] = true;  // Only the first element is Pareto optimal.
            return onFront;
        }
        if (nObjectives == 2)
            return IsParetoFront2d(uniqueLexsortedLossValues);
        return IsParetoFrontNd(uniqueLexsortedLossValues);
    }

    /// <summary>Python <c>_is_pareto_front(loss_values, assume_unique_lexsorted)</c>.</summary>
    internal static bool[] IsParetoFront(double[][] lossValues, bool assumeUniqueLexsorted)
    {
        if (assumeUniqueLexsorted)
            return IsParetoFrontForUniqueSorted(lossValues);

        var (uniqueRows, orderInv) = UniqueRows(lossValues);
        var onFrontUnique = IsParetoFrontForUniqueSorted(uniqueRows);
        var onFront = new bool[lossValues.Length];
        for (int i = 0; i < lossValues.Length; i++)
            onFront[i] = onFrontUnique[orderInv[i]];
        return onFront;
    }

    /// <summary>Python <c>_calculate_nondomination_rank</c>.</summary>
    internal static int[] CalculateNondominationRank(double[][] lossValues, int? nBelow = null)
    {
        if (lossValues.Length == 0 || (nBelow is not null && nBelow <= 0))
            return new int[lossValues.Length];

        int nObjectives = lossValues[0].Length;
        if (nObjectives == 1)
        {
            // np.unique(loss_values[:, 0], return_inverse=True): rank = index into sorted uniques.
            var uniqueSorted = lossValues.Select(r => r[0]).Distinct().OrderBy(v => v).ToArray();
            var ranks1 = new int[lossValues.Length];
            for (int i = 0; i < lossValues.Length; i++)
                ranks1[i] = Array.BinarySearch(uniqueSorted, lossValues[i][0]);
            return ranks1;
        }

        // It ensures that trials[j] will not dominate trials[i] for i < j (np.unique lexsorts).
        var (uniqueRows, orderInv) = UniqueRows(lossValues);
        int nUnique = uniqueRows.Length;
        int below = Math.Min(nBelow ?? nUnique, nUnique);
        var ranks = new int[nUnique];
        int rank = 0;
        var indices = Enumerable.Range(0, nUnique).ToArray();
        var remaining = uniqueRows;
        while (nUnique - indices.Length < below)
        {
            var onFront = IsParetoFront(remaining, assumeUniqueLexsorted: true);
            var nextIndices = new List<int>();
            var nextRemaining = new List<double[]>();
            for (int i = 0; i < indices.Length; i++)
            {
                if (onFront[i])
                    ranks[indices[i]] = rank;
                else
                {
                    nextIndices.Add(indices[i]);
                    nextRemaining.Add(remaining[i]);
                }
            }
            indices = nextIndices.ToArray();
            remaining = nextRemaining.ToArray();
            rank += 1;
        }

        // Rank worse than the top n_below is defined as the worst rank.
        foreach (int idx in indices)
            ranks[idx] = rank;

        var result = new int[lossValues.Length];
        for (int i = 0; i < lossValues.Length; i++)
            result[i] = ranks[orderInv[i]];
        return result;
    }

    /// <summary>Python <c>_dominates(trial0, trial1, directions)</c>.</summary>
    internal static bool Dominates(FrozenTrial trial0, FrozenTrial trial1, IReadOnlyList<StudyDirection> directions)
    {
        var values0 = trial0.Values;
        var values1 = trial1.Values;

        if (trial0.State != TrialState.Complete)
            return false;

        if (trial1.State != TrialState.Complete)
            return true;

        System.Diagnostics.Debug.Assert(values0 is not null);
        System.Diagnostics.Debug.Assert(values1 is not null);

        if (values0!.Count != values1!.Count)
            throw new ValueError("Trials with different numbers of objectives cannot be compared.");

        if (values0.Count != directions.Count)
            throw new ValueError("The number of the values and the number of the objectives are mismatched.");

        var normalized0 = values0.Select((v, i) => NormalizeValue(v, directions[i])).ToArray();
        var normalized1 = values1.Select((v, i) => NormalizeValue(v, directions[i])).ToArray();

        if (normalized0.SequenceEqual(normalized1))
            return false;

        return normalized0.Zip(normalized1, (v0, v1) => v0 <= v1).All(x => x);
    }

    /// <summary>Python <c>_normalize_value</c>.</summary>
    internal static double NormalizeValue(double? value, StudyDirection direction)
    {
        if (value is null)
            return double.PositiveInfinity;
        return direction == StudyDirection.Maximize ? -value.Value : value.Value;
    }

    /// <summary>
    ///     NumPy <c>np.unique(a, axis=0, return_inverse=True)</c>: rows sorted lexicographically
    ///     (leading column first), deduplicated, plus the inverse mapping original-row → unique-row.
    /// </summary>
    internal static (double[][] UniqueRows, int[] OrderInverse) UniqueRows(double[][] rows)
    {
        int n = rows.Length;
        var order = Enumerable.Range(0, n).ToArray();
        Array.Sort(order, (a, b) => CompareRows(rows[a], rows[b]));

        var uniqueRows = new List<double[]>();
        var orderInv = new int[n];
        for (int i = 0; i < n; i++)
        {
            int rowIdx = order[i];
            if (uniqueRows.Count == 0 || CompareRows(uniqueRows[^1], rows[rowIdx]) != 0)
                uniqueRows.Add(rows[rowIdx]);
            orderInv[rowIdx] = uniqueRows.Count - 1;
        }
        return (uniqueRows.ToArray(), orderInv);
    }

    private static int CompareRows(double[] a, double[] b)
    {
        for (int i = 0; i < a.Length; i++)
        {
            // Plain comparisons (not CompareTo) so that -0.0 == 0.0, matching NumPy's sort.
            if (a[i] < b[i])
                return -1;
            if (a[i] > b[i])
                return 1;
        }
        return 0;
    }
}
