using System;
using System.Collections.Generic;
using System.Linq;
using NumSharp;

namespace OptunaSharp.Internal;

/// <summary>
///     Port of <c>optuna/_hypervolume/*</c> — the exact hypervolume calculator (<c>wfg.py</c>) and
///     the greedy Hypervolume Subset Selection Problem solver (<c>hssp.py</c>). Consumed by TPE's
///     multi-objective trial split (<see cref="TpeTrialSplit" />). <c>box_decomposition.py</c>
///     (<c>get_non_dominated_box_bounds</c>) is deferred with its only consumer, the GP acquisition
///     function (PORTING_PLAN Phase 3.2).
///
///     <para><b>Numeric strategy.</b> NumPy <c>@</c>/<c>np.dot</c> is reproduced as a sequential dot
///     and <c>.prod(axis=-1)</c> as a sequential product. <see cref="ComputeHypervolume" /> agrees
///     with Optuna to ~1 ULP (verified bit-exact-to-tolerance over 120 mixed 2–6D cases, max relative
///     error 2e-16). The HSSP greedy selection is a pure sequence of <c>np.argmax</c> (first-max)
///     choices, so the resulting subset is deterministic and order-independent: the lazy-contribution
///     skipping in <see cref="LazyContribsUpdate" /> only prunes strictly-smaller upper bounds, so
///     every candidate tied at the maximum is evaluated regardless of the <c>argsort</c> tie order.</para>
///
///     <para><b>[Misaligned] HSSP ≥3-objective zero-contribution ties.</b> The two-objective solver
///     (<see cref="SolveHssp2d" />) is dot-free and matches Optuna exactly. For three or more
///     objectives, once the greedily-selected points dominate the rest, every remaining candidate has
///     a true marginal hypervolume of 0, computed as <c>H(S ∪ {i}) − H(S)</c>. Optuna's <c>np.dot</c>
///     (BLAS) leaves a sub-ULP residual there (e.g. <c>-1.8e-15</c>) that the sequential dot cancels
///     to an exact <c>0</c>, so <c>np.argmax</c> can pick a different — but hypervolume-equivalent —
///     zero-contribution point. The selected subset is equally optimal (identical total hypervolume,
///     same 1−1/e guarantee); only the tie resolution differs, and only in this degenerate case. It
///     does not surface in realistic multi-objective TPE studies (2- and 3-objective trajectories
///     match live Optuna to the erf tolerance in full lock-step). Same class as the TPE argsort and
///     erf/ucrt notes.</para>
/// </summary>
internal static class Hypervolume
{
    private const double Eps = 1e-12;

    // ===========================================================================================
    // wfg.py — compute_hypervolume
    // ===========================================================================================

    /// <summary>Python <c>compute_hypervolume(loss_vals, reference_point, assume_pareto)</c>.</summary>
    internal static double ComputeHypervolume(double[][] lossVals, double[] referencePoint, bool assumePareto = false)
    {
        int m = referencePoint.Length;

        // if not np.all(loss_vals <= reference_point): raise. NaN fails the `<=`, so it raises too.
        foreach (var row in lossVals)
            for (int k = 0; k < m; k++)
                if (!(row[k] <= referencePoint[k]))
                    throw new ValueError(
                        "All points must dominate or equal the reference point. " +
                        "That is, for all points in the loss_vals and the coordinate `i`, " +
                        "`loss_vals[i] <= reference_point[i]`.");

        // reference_point has no NaN thanks to the check above; a non-finite bound means inf HV.
        foreach (var r in referencePoint)
            if (!double.IsFinite(r))
                return double.PositiveInfinity;

        if (lossVals.Length == 0)
            return 0.0;

        double[][] sortedParetoSols;
        if (!assumePareto)
        {
            var (unique, _) = MultiObjective.UniqueRows(lossVals);
            var onFront = MultiObjective.IsParetoFront(unique, assumeUniqueLexsorted: true);
            var list = new List<double[]>();
            for (int i = 0; i < unique.Length; i++)
                if (onFront[i])
                    list.Add(unique[i]);
            sortedParetoSols = list.ToArray();
        }
        else
        {
            // loss_vals[loss_vals[:, 0].argsort()].
            var order = StableArgsort(lossVals.Select(r => r[0]).ToArray());
            sortedParetoSols = order.Select(i => lossVals[i]).ToArray();
        }

        double hv = m switch
        {
            2 => Compute2d(sortedParetoSols, referencePoint),
            3 => Compute3d(sortedParetoSols, referencePoint),
            _ => ComputeHv(sortedParetoSols, referencePoint),
        };

        // `nan` happens on inf - inf, but that is inf in hypervolume due to submodularity.
        return double.IsFinite(hv) ? hv : double.PositiveInfinity;
    }

    /// <summary>Python <c>wfg._compute_2d</c>.</summary>
    private static double Compute2d(double[][] sortedParetoSols, double[] referencePoint)
    {
        // rect_diag_y = concat([ref[1:], sols[:-1, 1]]); edge_x = ref[0]-sols[:,0];
        // edge_y = rect_diag_y - sols[:,1]; return edge_x @ edge_y.
        int n = sortedParetoSols.Length;
        double dot = 0.0;
        for (int i = 0; i < n; i++)
        {
            double rectDiagY = i == 0 ? referencePoint[1] : sortedParetoSols[i - 1][1];
            double edgeX = referencePoint[0] - sortedParetoSols[i][0];
            double edgeY = rectDiagY - sortedParetoSols[i][1];
            dot += edgeX * edgeY;
        }
        return dot;
    }

    /// <summary>Python <c>wfg._compute_3d</c> (O(N^2) cumulative-max voxel count).</summary>
    private static double Compute3d(double[][] sortedParetoSols, double[] referencePoint)
    {
        int n = sortedParetoSols.Length;
        var yOrder = StableArgsort(sortedParetoSols.Select(s => s[1]).ToArray());

        var zDelta = new double[n][];
        for (int r = 0; r < n; r++)
            zDelta[r] = new double[n];
        for (int k = 0; k < n; k++)
            zDelta[yOrder[k]][k] = referencePoint[2] - sortedParetoSols[yOrder[k]][2];

        // np.maximum.accumulate axis=0 then axis=1.
        for (int c = 0; c < n; c++)
            for (int r = 1; r < n; r++)
                zDelta[r][c] = Math.Max(zDelta[r][c], zDelta[r - 1][c]);
        for (int r = 0; r < n; r++)
            for (int c = 1; c < n; c++)
                zDelta[r][c] = Math.Max(zDelta[r][c], zDelta[r][c - 1]);

        var xVals = sortedParetoSols.Select(s => s[0]).ToArray();
        var yVals = yOrder.Select(idx => sortedParetoSols[idx][1]).ToArray();
        var xDelta = new double[n];
        var yDelta = new double[n];
        for (int i = 0; i < n; i++)
        {
            xDelta[i] = (i < n - 1 ? xVals[i + 1] : referencePoint[0]) - xVals[i];
            yDelta[i] = (i < n - 1 ? yVals[i + 1] : referencePoint[1]) - yVals[i];
        }

        // np.dot(np.dot(z_delta, y_delta), x_delta).
        double result = 0.0;
        for (int r = 0; r < n; r++)
        {
            double inner = 0.0;
            for (int c = 0; c < n; c++)
                inner += zDelta[r][c] * yDelta[c];
            result += inner * xDelta[r];
        }
        return result;
    }

    /// <summary>Python <c>wfg._compute_hv</c> — the WFG recursion for any dimension.</summary>
    private static double ComputeHv(double[][] sortedLossVals, double[] referencePoint)
    {
        int n = sortedLossVals.Length;
        int m = referencePoint.Length;

        if (n == 1)
        {
            double inclusive = 1.0;
            for (int k = 0; k < m; k++)
                inclusive *= referencePoint[k] - sortedLossVals[0][k];
            return inclusive;
        }
        if (n == 2)
        {
            double hv1 = 1.0, hv2 = 1.0, intersec = 1.0;
            for (int k = 0; k < m; k++)
            {
                hv1 *= referencePoint[k] - sortedLossVals[0][k];
                hv2 *= referencePoint[k] - sortedLossVals[1][k];
                intersec *= referencePoint[k] - Math.Max(sortedLossVals[0][k], sortedLossVals[1][k]);
            }
            return hv1 + hv2 - intersec;
        }

        var inclusiveHvs = new double[n];
        for (int i = 0; i < n; i++)
            inclusiveHvs[i] = ProdRefMinus(referencePoint, sortedLossVals[i]);

        // inclusive_hvs[-1] + sum_i exclusive_hv(limited_sols[i, i+1:], inclusive_hvs[i], ref).
        double acc = 0.0;
        for (int i = 0; i < n - 1; i++)
        {
            var limited = new double[n - 1 - i][];
            for (int j = i + 1; j < n; j++)
            {
                var rowMax = new double[m];
                for (int k = 0; k < m; k++)
                    rowMax[k] = Math.Max(sortedLossVals[i][k], sortedLossVals[j][k]);
                limited[j - i - 1] = rowMax;
            }
            acc += ComputeExclusiveHv(limited, inclusiveHvs[i], referencePoint);
        }
        return inclusiveHvs[n - 1] + acc;
    }

    /// <summary>Python <c>wfg._compute_exclusive_hv</c>.</summary>
    private static double ComputeExclusiveHv(double[][] limitedSols, double inclusiveHv, double[] referencePoint)
    {
        if (limitedSols.Length <= 3)
            return inclusiveHv - ComputeHv(limitedSols, referencePoint);

        var onFront = MultiObjective.IsParetoFront(limitedSols, assumeUniqueLexsorted: true);
        var filtered = new List<double[]>();
        for (int i = 0; i < limitedSols.Length; i++)
            if (onFront[i])
                filtered.Add(limitedSols[i]);
        return inclusiveHv - ComputeHv(filtered.ToArray(), referencePoint);
    }

    // ===========================================================================================
    // hssp.py — _solve_hssp
    // ===========================================================================================

    /// <summary>Python <c>hssp._solve_hssp(rank_i_loss_vals, rank_i_indices, subset_size, reference_point)</c>.</summary>
    internal static int[] SolveHssp(
        double[][] rankILossVals, int[] rankIIndices, int subsetSize, double[] referencePoint)
    {
        if (subsetSize == rankIIndices.Length)
            return rankIIndices;

        // np.unique(rank_i_loss_vals, return_index=True, axis=0).
        var (uniqueLossVals, indicesOfUnique) = MultiObjective.UniqueRowsFirstIndex(rankILossVals);
        int nUnique = indicesOfUnique.Length;
        if (nUnique < subsetSize)
        {
            // Not enough unique points: take all unique plus arbitrary duplicates to reach the size.
            var chosen = new bool[rankIIndices.Length];
            foreach (int idx in indicesOfUnique)
                chosen[idx] = true;
            var duplicated = Enumerable.Range(0, rankIIndices.Length).Where(i => !chosen[i]).ToArray();
            for (int i = 0; i < subsetSize - nUnique; i++)
                chosen[duplicated[i]] = true;
            return Enumerable.Range(0, rankIIndices.Length)
                .Where(i => chosen[i]).Select(i => rankIIndices[i]).ToArray();
        }

        var selected = SolveHsspOnUniqueLossVals(uniqueLossVals, indicesOfUnique, subsetSize, referencePoint);
        return selected.Select(i => rankIIndices[i]).ToArray();
    }

    /// <summary>Python <c>hssp._solve_hssp_on_unique_loss_vals</c>.</summary>
    private static int[] SolveHsspOnUniqueLossVals(
        double[][] rankILossVals, int[] mappedIndices, int subsetSize, double[] referencePoint)
    {
        foreach (var r in referencePoint)
            if (!double.IsFinite(r))
                return mappedIndices.Take(subsetSize).ToArray();
        if (mappedIndices.Length == subsetSize)
            return mappedIndices;
        if (referencePoint.Length == 2)
            return SolveHssp2d(rankILossVals, mappedIndices, subsetSize, referencePoint);

        int nSolutions = rankILossVals.Length;
        var contribs = new List<double>(nSolutions);
        for (int i = 0; i < nSolutions; i++)
            contribs.Add(ProdRefMinus(referencePoint, rankILossVals[i]));

        var selectedIndices = new int[subsetSize];
        var selectedVecs = new double[subsetSize][];
        var indices = Enumerable.Range(0, nSolutions).ToList();
        var lossVals = rankILossVals.Select(r => (double[])r.Clone()).ToList();
        double hv = 0.0;
        for (int k = 0; k < subsetSize; k++)
        {
            int maxIndex = ArgMax(contribs);
            hv += contribs[maxIndex];
            selectedIndices[k] = indices[maxIndex];
            selectedVecs[k] = (double[])lossVals[maxIndex].Clone();

            contribs.RemoveAt(maxIndex);
            indices.RemoveAt(maxIndex);
            lossVals.RemoveAt(maxIndex);
            if (k == subsetSize - 1)
                break;

            contribs = LazyContribsUpdate(
                contribs.ToArray(), lossVals.ToArray(), selectedVecs, k + 2, referencePoint, hv).ToList();
        }

        return selectedIndices.Select(i => mappedIndices[i]).ToArray();
    }

    /// <summary>Python <c>hssp._solve_hssp_2d</c> (O(subset_size * n) for two objectives).</summary>
    private static int[] SolveHssp2d(
        double[][] rankILossVals, int[] mappedIndices, int subsetSize, double[] referencePoint)
    {
        int nTrials = rankILossVals.Length;
        var sortedIndices = Enumerable.Range(0, nTrials).ToList();
        var sortedLossVals = rankILossVals.Select(r => (double[])r.Clone()).ToList();
        var rectDiags = new List<double[]>(nTrials);
        for (int i = 0; i < nTrials; i++)
            rectDiags.Add((double[])referencePoint.Clone());

        var selectedIndices = new int[subsetSize];
        for (int i = 0; i < subsetSize; i++)
        {
            int cur = sortedLossVals.Count;
            var contribs = new double[cur];
            for (int t = 0; t < cur; t++)
                contribs[t] = (rectDiags[t][0] - sortedLossVals[t][0]) * (rectDiags[t][1] - sortedLossVals[t][1]);

            int maxIndex = ArgMax(contribs);
            selectedIndices[i] = mappedIndices[sortedIndices[maxIndex]];
            var lossVals = (double[])sortedLossVals[maxIndex].Clone();

            sortedIndices.RemoveAt(maxIndex);
            rectDiags.RemoveAt(maxIndex);
            sortedLossVals.RemoveAt(maxIndex);

            // Update the rectangle diagonals for the remaining points (see hssp.py).
            for (int t = 0; t < maxIndex; t++)
                rectDiags[t][0] = Math.Min(lossVals[0], rectDiags[t][0]);
            for (int t = maxIndex; t < rectDiags.Count; t++)
                rectDiags[t][1] = Math.Min(lossVals[1], rectDiags[t][1]);
        }

        return selectedIndices;
    }

    /// <summary>Python <c>hssp._lazy_contribs_update</c>.</summary>
    private static double[] LazyContribsUpdate(
        double[] contribs, double[][] paretoLossValues, double[][] selectedVecs, int selectedCount,
        double[] referencePoint, double hvSelected)
    {
        int nRemaining = paretoLossValues.Length;
        if (double.IsInfinity(hvSelected))
        {
            var full = new double[nRemaining];
            for (int i = 0; i < nRemaining; i++)
                full[i] = double.PositiveInfinity;
            return full;
        }

        int m = referencePoint.Length;
        // The last selected point (selected_vecs[:-1][-1]) drives the cheap contribution bound.
        double[] lastSelected = selectedVecs[selectedCount - 2];

        var inclusiveHvs = new double[nRemaining];
        var isContribInf = new bool[nRemaining];
        var updated = new double[nRemaining];
        for (int i = 0; i < nRemaining; i++)
        {
            inclusiveHvs[i] = ProdRefMinus(referencePoint, paretoLossValues[i]);
            isContribInf[i] = double.IsInfinity(inclusiveHvs[i]);
            // contribs = min(contribs, inclusive_hvs - prod(ref - max(pareto, last_selected))).
            double prodIntersecLast = 1.0;
            for (int k = 0; k < m; k++)
                prodIntersecLast *= referencePoint[k] - Math.Max(paretoLossValues[i][k], lastSelected[k]);
            updated[i] = Math.Min(contribs[i], inclusiveHvs[i] - prodIntersecLast);
        }

        double maxContrib = 0.0;
        bool isHvCalcFast = m <= 3;
        foreach (int i in ArgsortDescendingNanLast(updated))
        {
            if (isContribInf[i])
            {
                maxContrib = updated[i] = double.PositiveInfinity;
                continue;
            }
            if (updated[i] < maxContrib)
                continue;

            if (isHvCalcFast)
            {
                // Incremental: contribs[i] = H(S v {i}) - H(S). S is the selectedCount-1 selected
                // points (selected_vecs[:-1]); the last row is the candidate (selected_vecs[-1]).
                var vecs = new double[selectedCount][];
                for (int r = 0; r < selectedCount - 1; r++)
                    vecs[r] = selectedVecs[r];
                vecs[selectedCount - 1] = paretoLossValues[i];
                double hvPlus = ComputeHypervolume(vecs, referencePoint, assumePareto: true);
                updated[i] = hvPlus - hvSelected;
            }
            else
            {
                // Decremental: contribs[i] = H({i}) - H(S ^ {i}).
                var intersec = new double[selectedCount - 1][];
                for (int s = 0; s < selectedCount - 1; s++)
                {
                    var rowMax = new double[m];
                    for (int k = 0; k < m; k++)
                        rowMax[k] = Math.Max(paretoLossValues[i][k], selectedVecs[s][k]);
                    intersec[s] = rowMax;
                }
                updated[i] = inclusiveHvs[i] - ComputeHypervolume(intersec, referencePoint);
            }
            maxContrib = Math.Max(updated[i], maxContrib);
        }

        return updated;
    }

    // ===========================================================================================
    // helpers
    // ===========================================================================================

    /// <summary>Sequential <c>np.prod(reference_point - row)</c>.</summary>
    private static double ProdRefMinus(double[] referencePoint, double[] row)
    {
        double p = 1.0;
        for (int k = 0; k < referencePoint.Length; k++)
            p *= referencePoint[k] - row[k];
        return p;
    }

    /// <summary>NumPy <c>np.argmax</c>: the first index of the maximum (values here are non-negative, finite or +inf).</summary>
    private static int ArgMax(IReadOnlyList<double> values)
    {
        int best = 0;
        double bestVal = values[0];
        for (int i = 1; i < values.Count; i++)
        {
            if (values[i] > bestVal)
            {
                bestVal = values[i];
                best = i;
            }
        }
        return best;
    }

    /// <summary>Stable ascending argsort (NaN sorts last, mirroring NumPy).</summary>
    private static int[] StableArgsort(double[] values)
        => Enumerable.Range(0, values.Length)
            .OrderBy(i => values[i], AscNanLastComparer.Instance).ToArray();

    /// <summary>
    ///     Stable descending argsort with NaN last — the iteration order of
    ///     <c>np.argsort(-contribs)</c>. The result is tie-order-independent (see class remark), so a
    ///     stable order suffices.
    /// </summary>
    private static int[] ArgsortDescendingNanLast(double[] values)
        => Enumerable.Range(0, values.Length)
            .OrderBy(i => values[i], DescNanLastComparer.Instance).ToArray();

    private sealed class AscNanLastComparer : IComparer<double>
    {
        internal static readonly AscNanLastComparer Instance = new();
        public int Compare(double a, double b)
        {
            bool na = double.IsNaN(a), nb = double.IsNaN(b);
            if (na || nb)
                return na == nb ? 0 : na ? 1 : -1;
            return a.CompareTo(b);
        }
    }

    private sealed class DescNanLastComparer : IComparer<double>
    {
        internal static readonly DescNanLastComparer Instance = new();
        public int Compare(double a, double b)
        {
            bool na = double.IsNaN(a), nb = double.IsNaN(b);
            if (na || nb)
                return na == nb ? 0 : na ? 1 : -1;
            return b.CompareTo(a);
        }
    }
}
