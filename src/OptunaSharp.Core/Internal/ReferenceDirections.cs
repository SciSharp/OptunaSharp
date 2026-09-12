// Das–Dennis structured reference directions — the weight-vector / reference-vector simplex lattice
// shared by the decomposition- and reference-vector-guided MOEAs (MOEA/D and RVEA).
//
// Reference: pymoo's `pymoo/util/reference_direction.py` (Apache-2.0) — `das_dennis`,
// `das_dennis_recursion`, `get_number_of_uniform_points` and `get_partition_closest_to_points`. The
// original construction is Das & Dennis, "Normal-Boundary Intersection: A New Method for Generating
// the Pareto Surface in Nonlinear Multicriteria Optimization Problems", SIAM J. Optim. 8(3), 1998.
//
// MOEA/D and RVEA are beyond-core samplers (no Optuna counterpart), so this helper is validated by the
// samplers' convergence/hypervolume gates, not bit-matched to an upstream trajectory. Only the SET of
// generated directions matters (each is an independent subproblem / niche), so the natural recursion
// order is kept as-is rather than replicating pymoo's post-hoc lexsort.

using System;
using System.Collections.Generic;

namespace OptunaSharp.Internal;

/// <summary>
///     Generates the Das–Dennis structured simplex lattice of reference directions — every point on
///     the unit simplex whose coordinates are non-negative multiples of <c>1/H</c> summing to one
///     (<c>H</c> = the number of partitions). Used as MOEA/D's weight vectors and RVEA's reference
///     vectors. Port of pymoo's <c>das_dennis</c> family.
/// </summary>
internal static class ReferenceDirections
{
    /// <summary>
    ///     The Das–Dennis lattice for <paramref name="nDim" /> objectives divided into
    ///     <paramref name="nPartitions" /> parts per axis, as a jagged <c>n × nDim</c> array. With
    ///     <c>H = 0</c> pymoo returns the single centroid direction <c>[1/m, …, 1/m]</c>.
    /// </summary>
    /// <param name="nPartitions">The number of equally spaced simplex partitions.</param>
    /// <param name="nDim">The number of coordinates in a reference direction.</param>
    /// <returns>The reference directions on the unit simplex.</returns>
    /// <exception cref="System.ArgumentOutOfRangeException">The number of objectives is not positive.</exception>
    internal static double[][] DasDennis(int nPartitions, int nDim)
    {
        if (nDim <= 0)
            throw new ArgumentOutOfRangeException(nameof(nDim), "The number of objectives must be positive.");

        if (nPartitions == 0)
        {
            var single = new double[nDim];
            for (int j = 0; j < nDim; j++) single[j] = 1.0 / nDim;
            return new[] { single };
        }

        var refDirs = new List<double[]>();
        Recurse(refDirs, new double[nDim], nPartitions, nPartitions, 0);
        return refDirs.ToArray();
    }

    /// <summary>
    ///     pymoo <c>das_dennis_recursion</c>: at the last coordinate the remaining budget <paramref name="beta" />
    ///     is placed; otherwise each admissible share of the budget is tried for the current coordinate and
    ///     the rest is recursed. <paramref name="refDir" /> is copied at each branch so completed rows are
    ///     independent.
    /// </summary>
    /// <param name="refDirs">The destination collection receiving completed reference-direction vectors.</param>
    /// <param name="refDir">The mutable reference-direction vector being filled recursively.</param>
    /// <param name="nPartitions">The number of equally spaced simplex partitions.</param>
    /// <param name="beta">The remaining integer partition mass to distribute among the unfinished coordinates.</param>
    /// <param name="depth">The coordinate or tree depth currently being processed.</param>
    private static void Recurse(List<double[]> refDirs, double[] refDir, int nPartitions, int beta, int depth)
    {
        if (depth == refDir.Length - 1)
        {
            refDir[depth] = beta / (double)nPartitions;
            refDirs.Add((double[])refDir.Clone());
            return;
        }

        for (int i = 0; i <= beta; i++)
        {
            refDir[depth] = i / (double)nPartitions;
            Recurse(refDirs, (double[])refDir.Clone(), nPartitions, beta - i, depth + 1);
        }
    }

    /// <summary>
    ///     pymoo <c>get_number_of_uniform_points</c>: the size of the Das–Dennis lattice for
    ///     <paramref name="nPartitions" /> partitions and <paramref name="nDim" /> objectives —
    ///     <c>C(nDim + nPartitions − 1, nPartitions)</c>.
    /// </summary>
    /// <param name="nPartitions">The number of equally spaced simplex partitions.</param>
    /// <param name="nDim">The number of coordinates in a reference direction.</param>
    /// <returns>The number of simplex lattice directions for the given partition count.</returns>
    internal static long NumberOfUniformPoints(int nPartitions, int nDim)
        => Binomial(nDim + nPartitions - 1, nPartitions);

    /// <summary>
    ///     pymoo <c>get_partition_closest_to_points</c>: the largest number of partitions whose lattice
    ///     size does not exceed <paramref name="nPoints" /> (so callers can request a target population
    ///     size and get the closest structured lattice at or below it). Always at least 1.
    /// </summary>
    /// <param name="nPoints">The requested approximate number of uniform reference directions.</param>
    /// <param name="nDim">The number of coordinates in a reference direction.</param>
    /// <returns>The partition count chosen to approximate the requested number of directions.</returns>
    internal static int PartitionClosestToPoints(long nPoints, int nDim)
    {
        if (nDim == 1)
            return 0;

        int nPartitions = 1;
        long count = NumberOfUniformPoints(nPartitions, nDim);
        while (count <= nPoints)
        {
            nPartitions += 1;
            count = NumberOfUniformPoints(nPartitions, nDim);
        }
        return Math.Max(1, nPartitions - 1);
    }

    /// <summary>Binomial coefficient <c>C(n, k)</c> via the multiplicative formula (exact for the modest
    ///     lattice sizes used here; guards against overflow by working in <see cref="long" />).</summary>
    /// <param name="n">The number of elements or points to process.</param>
    /// <param name="k">The integer index or combinatorial selection count used by this operation.</param>
    /// <returns>The binomial coefficient for the supplied nonnegative integers.</returns>
    private static long Binomial(int n, int k)
    {
        if (k < 0 || k > n) return 0;
        k = Math.Min(k, n - k);
        long result = 1;
        for (int i = 1; i <= k; i++)
        {
            result = result * (n - k + i) / i;
        }
        return result;
    }
}
