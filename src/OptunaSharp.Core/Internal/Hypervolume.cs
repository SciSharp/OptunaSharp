using System;
using System.Collections.Generic;
using System.Linq;
using NumSharp;

namespace OptunaSharp.Internal;

/// <summary>
///     Port of <c>optuna/_hypervolume/*</c> — the exact hypervolume calculator (<c>wfg.py</c>), the
///     greedy Hypervolume Subset Selection Problem solver (<c>hssp.py</c>), and the box decomposition
///     of the non-dominated region (<c>box_decomposition.py</c>: <see cref="GetNonDominatedBoxBounds" />,
///     Phase 3.2 part 3e). Consumed by TPE's multi-objective trial split
///     (<see cref="TpeTrialSplit" />) and the GP log-EHVI acquisition function.
///
///     <para><b>Numeric strategy.</b> Optuna's NumPy <c>@</c>/<c>np.dot</c> products (the two- and
///     three-objective volumes) run through NumSharp's <c>np.dot</c> via <see cref="LinAlg.Dot(double[], double[])" />
///     and <see cref="LinAlg.Dot(double[,], double[])" />, i.e. the same OpenBLAS <c>ddot</c>/<c>dgemv</c> binary NumPy
///     calls, so their sums round exactly as Optuna's do. <c>.prod(axis=-1)</c> stays a sequential product: NumPy
///     multiplies along that axis in order, which a plain loop reproduces and NumSharp's <c>np.prod</c> does not promise
///     for floats. Measured live against Optuna 5.0.0 (<c>HypervolumeParityTests</c>, 240 random 2- to 5-objective
///     fronts): every <see cref="ComputeHypervolume" /> result is bit-identical (the former sequential dot differed in 37
///     of them, by an ulp or two).</para>
///
///     <para><b>Subset selection.</b> The HSSP greedy selection is a pure sequence of <c>np.argmax</c> (first-max)
///     choices, so the subset is deterministic, and the lazy-contribution skipping in
///     <see cref="LazyContribsUpdate" /> only prunes strictly-smaller upper bounds, so every candidate tied at the
///     maximum is evaluated whatever the <c>argsort</c> tie order. With three or more objectives a candidate's true
///     marginal hypervolume can be exactly 0 (a box of zero volume, or one the selected points cover), computed as
///     <c>H(S ∪ {i}) − H(S)</c>; Optuna's BLAS products leave a sub-ulp residual there (e.g. <c>-1.8e-15</c>) and
///     <c>np.argmax</c> picks by it. The former sequential dot cancelled those residuals to an exact 0 and could pick a
///     different, hypervolume-equivalent point (a documented [Misaligned] case); with NumPy's own products the
///     residuals, and so the picks, are Optuna's (240 of 240 selections identical in <c>HypervolumeParityTests</c>,
///     zero-volume ties included, against 237 before). The two-objective solver (<see cref="SolveHssp2d" />) is
///     dot-free.</para>
/// </summary>
internal static class Hypervolume
{
    private const double Eps = 1e-12;

    // ===========================================================================================
    // wfg.py — compute_hypervolume
    // ===========================================================================================

    /// <summary>Python <c>compute_hypervolume(loss_vals, reference_point, assume_pareto)</c>.</summary>
    /// <param name="lossVals">Objective rows converted to minimization, one solution per row.</param>
    /// <param name="referencePoint">The reference objective vector, in the same coordinate system and order as the supplied objective rows.</param>
    /// <param name="assumePareto">Whether the caller guarantees that the input rows are already nondominated.</param>
    /// <returns>The volume dominated by the solutions and bounded by the reference point.</returns>
    /// <exception cref="NumSharp.ValueError">At least one minimization objective exceeds the corresponding reference-point coordinate.</exception>
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

    /// <summary>
    ///     Python <c>wfg._compute_2d</c>: the staircase area as one inner product, <c>edge_x @ edge_y</c>, taken through
    ///     NumSharp's <c>np.dot</c> (BLAS <c>ddot</c>) so the sum rounds exactly as Optuna's does.
    /// </summary>
    /// <param name="sortedParetoSols">Unique nondominated minimization rows in lexicographic order.</param>
    /// <param name="referencePoint">The reference objective vector, in the same coordinate system and order as the supplied objective rows.</param>
    /// <returns>The dominated area between the Pareto front and the reference point, Optuna's bits.</returns>
    /// <remarks>
    ///     The edges are single subtractions, exact as NumPy's elementwise ones. Their products were once summed left to
    ///     right; BLAS's <c>ddot</c> accumulates in several registers, so that loop differed from Optuna in the last bits on
    ///     fronts of more than a handful of points (measured live: 37 of 240 random 2- to 5-objective volumes before the
    ///     switch, 0 after).
    /// </remarks>
    private static double Compute2d(double[][] sortedParetoSols, double[] referencePoint)
    {
        // rect_diag_y = concat([ref[1:], sols[:-1, 1]]); edge_x = ref[0]-sols[:,0];
        // edge_y = rect_diag_y - sols[:,1]; return edge_x @ edge_y.
        int n = sortedParetoSols.Length;
        var edgeX = new double[n];
        var edgeY = new double[n];
        for (int i = 0; i < n; i++)
        {
            double rectDiagY = i == 0 ? referencePoint[1] : sortedParetoSols[i - 1][1];
            edgeX[i] = referencePoint[0] - sortedParetoSols[i][0];
            edgeY[i] = rectDiagY - sortedParetoSols[i][1];
        }
        return LinAlg.Dot(edgeX, edgeY);
    }

    /// <summary>
    ///     Python <c>wfg._compute_3d</c> (O(N^2) cumulative-max voxel count): the two products
    ///     <c>np.dot(np.dot(z_delta, y_delta), x_delta)</c> go through NumSharp's <c>np.dot</c> (BLAS <c>dgemv</c>, then
    ///     <c>ddot</c>) so the volume rounds exactly as Optuna's does.
    /// </summary>
    /// <param name="sortedParetoSols">Unique nondominated minimization rows in lexicographic order.</param>
    /// <param name="referencePoint">The reference objective vector, in the same coordinate system and order as the supplied objective rows.</param>
    /// <returns>The dominated three-dimensional volume relative to the reference point, Optuna's bits.</returns>
    /// <remarks>
    ///     <c>z_delta</c> is a C-contiguous <c>double[,]</c> because that is the layout NumPy hands to BLAS after its two
    ///     <c>np.maximum.accumulate</c> passes; a different layout would take another <c>gemv</c> route. Optuna orders the
    ///     columns with <c>np.argsort</c> on the y coordinates, NumPy's default (unstable, SIMD) sort, and
    ///     <see cref="StableArgsort" /> reproduces it only when those coordinates are distinct; tied y values can move a
    ///     product to another BLAS accumulator, the documented argsort tie class.
    /// </remarks>
    private static double Compute3d(double[][] sortedParetoSols, double[] referencePoint)
    {
        int n = sortedParetoSols.Length;
        var yOrder = StableArgsort(sortedParetoSols.Select(s => s[1]).ToArray());

        var zDelta = new double[n, n];
        for (int k = 0; k < n; k++)
            zDelta[yOrder[k], k] = referencePoint[2] - sortedParetoSols[yOrder[k]][2];

        // np.maximum.accumulate axis=0 then axis=1 (max is exact, so only the final values matter).
        for (int c = 0; c < n; c++)
            for (int r = 1; r < n; r++)
                zDelta[r, c] = Math.Max(zDelta[r, c], zDelta[r - 1, c]);
        for (int r = 0; r < n; r++)
            for (int c = 1; c < n; c++)
                zDelta[r, c] = Math.Max(zDelta[r, c], zDelta[r, c - 1]);

        var xVals = sortedParetoSols.Select(s => s[0]).ToArray();
        var yVals = yOrder.Select(idx => sortedParetoSols[idx][1]).ToArray();
        var xDelta = new double[n];
        var yDelta = new double[n];
        for (int i = 0; i < n; i++)
        {
            xDelta[i] = (i < n - 1 ? xVals[i + 1] : referencePoint[0]) - xVals[i];
            yDelta[i] = (i < n - 1 ? yVals[i + 1] : referencePoint[1]) - yVals[i];
        }

        // np.dot(np.dot(z_delta, y_delta), x_delta): a matrix-vector product, then an inner product.
        return LinAlg.Dot(LinAlg.Dot(zDelta, yDelta), xDelta);
    }

    /// <summary>Python <c>wfg._compute_hv</c> — the WFG recursion for any dimension.</summary>
    /// <param name="sortedLossVals">Minimization objective rows sorted lexicographically.</param>
    /// <param name="referencePoint">The reference objective vector, in the same coordinate system and order as the supplied objective rows.</param>
    /// <returns>The union volume of the dominated objective boxes.</returns>
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
    /// <param name="limitedSols">The competing solutions clipped to the candidate's dominated box.</param>
    /// <param name="inclusiveHv">The candidate box volume before subtracting overlap with competing boxes.</param>
    /// <param name="referencePoint">The reference objective vector, in the same coordinate system and order as the supplied objective rows.</param>
    /// <returns>The candidate's dominated volume after removing overlap with the other boxes.</returns>
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
    /// <param name="rankILossVals">The minimization objective rows in the partially selected Pareto rank.</param>
    /// <param name="rankIIndices">Original trial indices corresponding to the rows of the partially selected rank.</param>
    /// <param name="subsetSize">The number of solutions to select from the candidate rank.</param>
    /// <param name="referencePoint">The reference objective vector, in the same coordinate system and order as the supplied objective rows.</param>
    /// <returns>The original indices of the greedily selected hypervolume-maximizing subset.</returns>
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
    /// <param name="rankILossVals">The minimization objective rows in the partially selected Pareto rank.</param>
    /// <param name="mappedIndices">Original row indices to return after selecting from the deduplicated rows.</param>
    /// <param name="subsetSize">The number of solutions to select from the candidate rank.</param>
    /// <param name="referencePoint">The reference objective vector, in the same coordinate system and order as the supplied objective rows.</param>
    /// <returns>The mapped original indices selected from the distinct candidate rows.</returns>
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
    /// <param name="rankILossVals">The minimization objective rows in the partially selected Pareto rank.</param>
    /// <param name="mappedIndices">Original row indices to return after selecting from the deduplicated rows.</param>
    /// <param name="subsetSize">The number of solutions to select from the candidate rank.</param>
    /// <param name="referencePoint">The reference objective vector, in the same coordinate system and order as the supplied objective rows.</param>
    /// <returns>The mapped original indices selected by the two-objective subset routine.</returns>
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
    /// <param name="contribs">Cached upper estimates of each candidate's hypervolume contribution.</param>
    /// <param name="paretoLossValues">The nondominated objective rows from which the subset is selected.</param>
    /// <param name="selectedVecs">The workspace containing objective rows already selected.</param>
    /// <param name="selectedCount">The number of populated rows in the selected-solution workspace.</param>
    /// <param name="referencePoint">The reference objective vector, in the same coordinate system and order as the supplied objective rows.</param>
    /// <param name="hvSelected">The hypervolume already covered by the selected solutions.</param>
    /// <returns>The updated cached hypervolume-contribution estimates.</returns>
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
    /// <param name="referencePoint">The reference objective vector, in the same coordinate system and order as the supplied objective rows.</param>
    /// <param name="row">The objective or feature row used by the operation.</param>
    /// <returns>The product of coordinate distances from the solution to the reference point.</returns>
    private static double ProdRefMinus(double[] referencePoint, double[] row)
    {
        double p = 1.0;
        for (int k = 0; k < referencePoint.Length; k++)
            p *= referencePoint[k] - row[k];
        return p;
    }

    /// <summary>
    ///     NumPy <c>np.argmax</c> over the hypervolume contributions: the first index of the maximum, or of the first NaN
    ///     when there is one (NumPy treats a NaN as maximal and stops at it).
    /// </summary>
    /// <param name="values">The remaining candidates' hypervolume contributions; at least one.</param>
    /// <returns>The index of the first NaN if any contribution is NaN, else the first index holding the largest one.</returns>
    /// <remarks>
    ///     Contributions are non-negative, finite or <c>+inf</c>, until a loss value of <c>-inf</c> makes the running
    ///     hypervolume infinite: <c>_lazy_contribs_update</c>'s <c>inclusive_hvs - np.prod(...)</c> is then <c>inf - inf</c>
    ///     and <c>np.minimum</c> carries the NaN into the contributions. Optuna's <c>np.argmax</c> selects that candidate;
    ///     the former strict <c>&gt;</c> loop skipped it. The statements are NumPy's argmax loop (<c>!(v &lt;= best)</c>,
    ///     stop at a NaN). NumSharp's <c>np.argmax</c> gives the same index, but this runs once per greedy selection step,
    ///     where a NumSharp call (about 2.3 µs) would cost more than the step's own arithmetic on small fronts.
    /// </remarks>
    private static int ArgMax(IReadOnlyList<double> values)
    {
        int best = 0;
        double bestVal = values[0];
        // NumPy checks the first element on its own: a leading NaN is the answer.
        if (double.IsNaN(bestVal))
            return 0;
        for (int i = 1; i < values.Count; i++)
        {
            double value = values[i];
            // !(value <= best) is true for a larger value and for a NaN; a NaN ends the search, as in NumPy.
            if (!(value <= bestVal))
            {
                bestVal = value;
                best = i;
                if (double.IsNaN(value))
                    break;
            }
        }
        return best;
    }

    // ===========================================================================================
    // box_decomposition.py — get_non_dominated_box_bounds (PORTING_PLAN Phase 3.2 part 3e)
    // ===========================================================================================

    /// <summary>
    ///     Port of <c>optuna/_hypervolume/box_decomposition.py: get_non_dominated_box_bounds</c> —
    ///     decompose the non-dominated region below the Pareto front (and dominated by
    ///     <paramref name="refPoint" />) into axis-aligned boxes, the box tiling the log-EHVI
    ///     acquisition function (<c>acqf.logehvi</c>, Phase 3.2 part 3e) integrates over. The boxes
    ///     extend to <c>-inf</c> along the improving directions (the non-dominated region is
    ///     unbounded below), so lower bounds carry <c>-∞</c> entries and intervals <c>+∞</c> — the
    ///     acqf clamps them. Returns the <c>(n_boxes, n_objectives)</c> lower and upper bounds.
    ///
    ///     <para>Follows Algorithm 2 of Lacour, Klamroth &amp; Fonseca (arxiv 1510.01963), refactored
    ///     from BoTorch v0.13.0 exactly as Optuna's port is. <b>Bit-exact vs Optuna:</b> every value is
    ///     a coordinate of <paramref name="lossVals" />/<paramref name="refPoint" /> (or ±∞) moved
    ///     through negation, elementwise <c>max</c>, comparisons, and lexicographic <c>np.unique</c>
    ///     sort — no summation, dot, or transcendental, so there is no float-order drift.</para>
    /// </summary>
    /// <param name="lossVals">Objective rows converted to minimization, one solution per row.</param>
    /// <param name="refPoint">The reference objective vector bounding the dominated region.</param>
    /// <returns>Paired lower and upper rows describing the nondominated boxes.</returns>
    /// <exception cref="System.ArgumentException">An objective row has not been clipped to the reference-point boundary required by the decomposition.</exception>
    internal static (double[][] Lower, double[][] Upper) GetNonDominatedBoxBounds(
        double[][] lossVals, double[] refPoint)
    {
        // assert np.all(np.isfinite(loss_vals)) — the caller clips before box decomposition.
        foreach (double[] row in lossVals)
            foreach (double v in row)
                if (!double.IsFinite(v))
                    throw new ArgumentException("loss_vals must be clipped before box decomposition.");

        // Remove duplications and lexsort the solutions by ``np.unique``.
        double[][] uniqueLexsorted = MultiObjective.UniqueRows(lossVals).UniqueRows;
        bool[] onFront = MultiObjective.IsParetoFront(uniqueLexsorted, assumeUniqueLexsorted: true);
        double[][] sortedParetoSols = Mask(uniqueLexsorted, onFront);

        int nObjectives = refPoint.Length;  // loss_vals.shape[-1]
        // The condition here follows BoTorch — box decomposition is used only for MO problems.
        if (nObjectives > 4)
            OptunaWarnings.Warn(
                "Box decomposition (typically used by `GPSampler`) might be significantly slow for "
                + "n_objectives > 4. Please consider using another sampler instead.");

        return GetNonDominatedBoxBoundsInner(sortedParetoSols, refPoint);
    }

    /// <summary>Python <c>_get_non_dominated_box_bounds</c>. See <see cref="GetNonDominatedBoxBounds" />.</summary>
    /// <param name="sortedParetoSols">Unique nondominated minimization rows in lexicographic order.</param>
    /// <param name="refPoint">The reference objective vector bounding the dominated region.</param>
    /// <returns>Paired lower and upper rows describing the nondominated boxes.</returns>
    private static (double[][] Lower, double[][] Upper) GetNonDominatedBoxBoundsInner(
        double[][] sortedParetoSols, double[] refPoint)
    {
        int m = refPoint.Length;
        // neg_upper_bound_set = -_get_upper_bound_set(sorted_pareto_sols, ref_point)[0]
        (List<double[]> upperBoundSet, _) = GetUpperBoundSet(sortedParetoSols, refPoint);
        var negUpperBoundSet = new double[upperBoundSet.Count][];
        for (int i = 0; i < upperBoundSet.Count; i++)
            negUpperBoundSet[i] = Neg(upperBoundSet[i]);

        // np.unique(..., axis=0) lexsorts by np.unique.
        double[][] sortedNegUbs = MultiObjective.UniqueRows(negUpperBoundSet).UniqueRows;
        var pointAtInfinity = new double[m];
        Array.Fill(pointAtInfinity, double.PositiveInfinity);

        // Only the Pareto solutions in the neg_upper_bound_set are needed.
        bool[] front = MultiObjective.IsParetoFront(sortedNegUbs, assumeUniqueLexsorted: true);
        double[][] filtered = Mask(sortedNegUbs, front);
        (List<double[]> negLowerBoundSet, List<double[,]> negDefPoints) =
            GetUpperBoundSet(filtered, pointAtInfinity);

        // box_upper_bounds, box_lower_bounds = -_get_box_bounds(neg_lower_bound_set, neg_def_points, inf)
        (double[][] bounds0, double[][] bounds1) =
            GetBoxBounds(negLowerBoundSet, negDefPoints, pointAtInfinity);
        var boxUpper = new double[bounds0.Length][];
        var boxLower = new double[bounds1.Length][];
        for (int i = 0; i < bounds0.Length; i++)
        {
            boxUpper[i] = Neg(bounds0[i]);
            boxLower[i] = Neg(bounds1[i]);
        }
        return (boxLower, boxUpper);
    }

    /// <summary>
    ///     Python <c>_get_upper_bound_set</c> (Algorithm 2 of Lacour17): the upper-bound set
    ///     <c>U(N)</c> for the Pareto solutions (sorted w.r.t. the first objective) and their defining
    ///     points. Returns <c>upper_bound_set</c> (each <c>(n_objectives,)</c>) and <c>def_points</c>
    ///     (each <c>(n_objectives, n_objectives)</c>, indexed <c>[defining-dim, coordinate]</c>).
    /// </summary>
    /// <param name="sortedParetoSols">Unique nondominated minimization rows in lexicographic order.</param>
    /// <param name="refPoint">The reference objective vector bounding the dominated region.</param>
    /// <returns>The local upper-bound vectors and their corresponding defining-point matrices.</returns>
    private static (List<double[]> UpperBoundSet, List<double[,]> DefPoints) GetUpperBoundSet(
        double[][] sortedParetoSols, double[] refPoint)
    {
        int m = refPoint.Length;
        // upper_bound_set = [ref_point]  (Line 1 of Alg. 2).
        var ubs = new List<double[]> { (double[])refPoint.Clone() };
        // def_points[0] = -inf everywhere, with the diagonal set to ref_point (\hat{z}^k dummy point).
        var dps = new List<double[,]>();
        var dp0 = new double[m, m];
        for (int k = 0; k < m; k++)
            for (int j = 0; j < m; j++)
                dp0[k, j] = k == j ? refPoint[k] : double.NegativeInfinity;
        dps.Add(dp0);

        foreach (double[] solution in sortedParetoSols)  // Sorted must be fulfilled.
            (ubs, dps) = UpperBoundSetUpdate(solution, ubs, dps, m);

        return (ubs, dps);
    }

    /// <summary>
    ///     Python <c>_get_upper_bound_set.update</c> — the update rule (Section 2.2 of Lacour17) that
    ///     folds one Pareto solution into the running upper-bound set / defining points. Replaces each
    ///     dominated bound with the boxes it splits into, keeping the C-order enumeration of the
    ///     <c>(bound, dimension)</c> updates so the output row order matches NumPy's boolean-mask
    ///     flatten and <c>np.vstack</c> exactly.
    /// </summary>
    /// <param name="sol">The new objective row to incorporate into the upper-bound decomposition.</param>
    /// <param name="ubs">The current local upper bounds of the uncovered region.</param>
    /// <param name="dps">The defining points associated with each local upper bound.</param>
    /// <param name="m">The number of objective dimensions.</param>
    /// <returns>The updated local upper bounds and corresponding defining points.</returns>
    private static (List<double[]>, List<double[,]>) UpperBoundSetUpdate(
        double[] sol, List<double[]> ubs, List<double[,]> dps, int m)
    {
        int nb = ubs.Count;
        // is_dominated = np.all(sol < ubs, axis=-1)
        var isDominated = new bool[nb];
        bool anyDominated = false;
        for (int i = 0; i < nb; i++)
        {
            bool all = true;
            for (int d = 0; d < m; d++)
                if (!(sol[d] < ubs[i][d])) { all = false; break; }
            isDominated[i] = all;
            anyDominated |= all;
        }
        if (!anyDominated)
            return (ubs, dps);

        // dominated_dps = dps[is_dominated]; local indices into the dominated subset.
        var domIdx = new List<int>();
        for (int i = 0; i < nb; i++)
            if (isDominated[i]) domIdx.Add(i);

        // For each dominated bound (C-order outer) and each dimension j (C-order inner):
        //   maxval[j] = max over k (k != j and j != 0) of dominated_dps[i, k, j], else -inf,
        //   and if sol[j] >= maxval[j], emit a new (ubs, dps) entry — Lines 2,3 of Alg. 2.
        var newUbs = new List<double[]>();
        var newDps = new List<double[,]>();
        foreach (int i in domIdx)
        {
            double[,] dpi = dps[i];
            double[] ubi = ubs[i];
            for (int j = 0; j < m; j++)
            {
                double maxval = double.NegativeInfinity;
                if (j != 0)  // j == 0 => whole column skipped (skip_ineq_judge[:, 0] = True) => -inf.
                    for (int k = 0; k < m; k++)
                        if (k != j && dpi[k, j] > maxval)  // skip the diagonal k == j.
                            maxval = dpi[k, j];
                if (sol[j] >= maxval)
                {
                    // new_dps[..] = dominated_dps[i] with row j (the defining dim) set to sol.
                    var nd = (double[,])dpi.Clone();
                    for (int c = 0; c < m; c++)
                        nd[j, c] = sol[c];
                    newDps.Add(nd);
                    // new_ubs[..] = ubs[i] with coordinate j set to sol[j] (\bar{z}_j).
                    var nu = (double[])ubi.Clone();
                    nu[j] = sol[j];
                    newUbs.Add(nu);
                }
            }
        }

        // np.vstack([ubs[~is_dominated], new_ubs]) — kept bounds (original order) then the new ones.
        var resUbs = new List<double[]>();
        var resDps = new List<double[,]>();
        for (int i = 0; i < nb; i++)
            if (!isDominated[i])
            {
                resUbs.Add(ubs[i]);
                resDps.Add(dps[i]);
            }
        resUbs.AddRange(newUbs);
        resDps.AddRange(newDps);
        return (resUbs, resDps);
    }

    /// <summary>
    ///     Python <c>_get_box_bounds</c> (Eq. (2) of Lacour17) — turn the upper-bound set and defining
    ///     points into box <c>[lower, upper]</c> bounds, dropping empty boxes (<c>upper &lt;= lower</c>
    ///     in any coordinate). Returns the two stacked halves the caller negates/swaps.
    /// </summary>
    /// <param name="upperBoundSet">The local upper bounds defining nondominated boxes.</param>
    /// <param name="defPoints">The defining points associated with each local upper bound.</param>
    /// <param name="refPoint">The reference objective vector bounding the dominated region.</param>
    /// <returns>Paired lower and upper corner rows for the resulting boxes.</returns>
    private static (double[][] Bounds0, double[][] Bounds1) GetBoxBounds(
        List<double[]> upperBoundSet, List<double[,]> defPoints, double[] refPoint)
    {
        int m = refPoint.Length;
        int n = upperBoundSet.Count;
        var keptBounds0 = new List<double[]>();
        var keptBounds1 = new List<double[]>();
        for (int i = 0; i < n; i++)
        {
            double[,] dpi = defPoints[i];
            double[] ubi = upperBoundSet[i];
            var b0 = new double[m];
            var b1 = new double[m];
            // bounds[0, :, 0] = def_points[:, 0, 0]; bounds[1, :, 0] = ref_point[0].
            b0[0] = dpi[0, 0];
            b1[0] = refPoint[0];
            // bounds[0, :, 1:] = np.maximum.accumulate(def_points, axis=-2)[:, r, r+1]
            //   => b0[j] = max_{k=0..j-1} def_points[i, k, j].  bounds[1, :, 1:] = upper_bound_set[:, 1:].
            for (int j = 1; j < m; j++)
            {
                double acc = dpi[0, j];
                for (int k = 1; k < j; k++)
                    if (dpi[k, j] > acc) acc = dpi[k, j];
                b0[j] = acc;
                b1[j] = ubi[j];
            }
            // not_empty = ~np.any(bounds[1] <= bounds[0], axis=-1).
            bool empty = false;
            for (int j = 0; j < m; j++)
                if (b1[j] <= b0[j]) { empty = true; break; }
            if (!empty)
            {
                keptBounds0.Add(b0);
                keptBounds1.Add(b1);
            }
        }
        return (keptBounds0.ToArray(), keptBounds1.ToArray());
    }

    /// <summary>Returns a new vector with every input coordinate negated.</summary>
    /// <param name="a">The first numerical operand; its dimensions must agree with the operation described in the summary.</param>
    /// <returns>A new vector with every input coordinate negated.</returns>
    private static double[] Neg(double[] a)
    {
        var r = new double[a.Length];
        for (int i = 0; i < a.Length; i++)
            r[i] = -a[i];
        return r;
    }

    /// <summary>Returns the retained row references in their original order.</summary>
    /// <param name="rows">The matrix rows to select or process.</param>
    /// <param name="keep">One flag per row; only rows whose flag is true are retained.</param>
    /// <returns>The retained row references in their original order.</returns>
    private static double[][] Mask(double[][] rows, bool[] keep)
    {
        var result = new List<double[]>();
        for (int i = 0; i < rows.Length; i++)
            if (keep[i]) result.Add(rows[i]);
        return result.ToArray();
    }

    /// <summary>Stable ascending argsort (NaN sorts last, mirroring NumPy).</summary>
    /// <param name="values">The ordered values to aggregate or format.</param>
    /// <returns>Indices sorting values in ascending order while preserving equal-value order.</returns>
    private static int[] StableArgsort(double[] values)
        => Enumerable.Range(0, values.Length)
            .OrderBy(i => values[i], AscNanLastComparer.Instance).ToArray();

    /// <summary>
    ///     Stable descending argsort with NaN last — the iteration order of
    ///     <c>np.argsort(-contribs)</c>. The result is tie-order-independent (see class remark), so a
    ///     stable order suffices.
    /// </summary>
    /// <param name="values">The ordered values to aggregate or format.</param>
    /// <returns>Indices sorting finite values descending and placing NaNs last.</returns>
    private static int[] ArgsortDescendingNanLast(double[] values)
        => Enumerable.Range(0, values.Length)
            .OrderBy(i => values[i], DescNanLastComparer.Instance).ToArray();

    private sealed class AscNanLastComparer : IComparer<double>
    {
        /// <summary>The shared stateless comparer used for deterministic ordering.</summary>
        internal static readonly AscNanLastComparer Instance = new();
        /// <summary>Returns a negative, zero, or positive value for ascending order, with NaNs after numeric values.</summary>
        /// <param name="a">The first numerical operand; its dimensions must agree with the operation described in the summary.</param>
        /// <param name="b">The second numerical operand; its dimensions must agree with the first operand.</param>
        /// <returns>A negative, zero, or positive value for ascending order, with NaNs after numeric values.</returns>
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
        /// <summary>The shared stateless comparer used for deterministic ordering.</summary>
        internal static readonly DescNanLastComparer Instance = new();
        /// <summary>Returns a negative, zero, or positive value for descending order, with NaNs after numeric values.</summary>
        /// <param name="a">The first numerical operand; its dimensions must agree with the operation described in the summary.</param>
        /// <param name="b">The second numerical operand; its dimensions must agree with the first operand.</param>
        /// <returns>A negative, zero, or positive value for descending order, with NaNs after numeric values.</returns>
        public int Compare(double a, double b)
        {
            bool na = double.IsNaN(a), nb = double.IsNaN(b);
            if (na || nb)
                return na == nb ? 0 : na ? 1 : -1;
            return b.CompareTo(a);
        }
    }
}
