using System;
using System.Linq;

namespace OptunaSharp.Internal;

/// <summary>
///     The Wilcoxon signed-rank test as <c>scipy.stats.wilcoxon(d, alternative=...,
///     zero_method="zsplit")</c> computes it (scipy 1.16 semantics: <c>method="auto"</c> picks
///     exact when there are no ties/zeros, an exhaustive sign-flip permutation test when ties or
///     zeros exist and <c>n &lt;= 13</c>, and the normal approximation otherwise;
///     <c>correction=False</c>). This is the self-contained port the plan's item 2.3 calls for —
///     no scipy dependency; verified differentially against scipy in tests.
/// </summary>
internal static class SignedRankTest
{
    /// <summary>The direction of the one-sided signed-rank alternative hypothesis.</summary>
    internal enum Alternative
    {
        /// <summary>Tests whether paired differences tend to be negative.</summary>
        Less,
        /// <summary>Tests whether paired differences tend to be positive.</summary>
        Greater,
    }

    /// <summary>P-value of the one-sample test on differences <paramref name="d" /> with <c>zero_method="zsplit"</c>.</summary>
    /// <param name="d">The signed paired differences being tested.</param>
    /// <param name="alternative">The one-sided alternative hypothesis for the signed differences.</param>
    /// <returns>The one-sided signed-rank p-value using the zsplit treatment of zero differences.</returns>
    internal static double PValueZsplit(double[] d, Alternative alternative)
    {
        int n = d.Length;
        int nZero = d.Count(v => v == 0);

        var (ranks, tieCounts, hasTies) = AverageRanks(d.Select(Math.Abs).ToArray());

        double rPlus = 0;
        double rZero = 0;
        for (int i = 0; i < n; i++)
        {
            if (d[i] > 0)
                rPlus += ranks[i];
            else if (d[i] == 0)
                rZero += ranks[i];
        }
        // The "zero-split" method: add half the zeros' contribution to r_plus (and half to r_minus).
        rPlus += rZero / 2;

        // Method selection (scipy method="auto").
        if (!hasTies && nZero == 0)
            return ExactPValue(n, rPlus, alternative);
        if (n <= 13)
            return PermutationPValue(d, ranks, rZero, rPlus, alternative);
        return AsymptoticPValue(n, rPlus, tieCounts, alternative);
    }

    /// <summary>
    ///     scipy <c>_rankdata(x, "average", return_ties=True)</c>: average ranks plus per-group
    ///     tie counts (count at each sorted group's first element, 0 elsewhere in the group).
    /// </summary>
    /// <param name="x">The numerical input values, in the order used by the operation.</param>
    /// <returns>Average ranks, tie-group sizes, and whether any ties were present.</returns>
    private static (double[] Ranks, long[] TieCounts, bool HasTies) AverageRanks(double[] x)
    {
        int n = x.Length;
        var order = Enumerable.Range(0, n).ToArray();
        Array.Sort(order, (a, b) => x[a].CompareTo(x[b]));

        var ranks = new double[n];
        var tieCounts = new long[n];
        bool hasTies = false;
        int i = 0;
        while (i < n)
        {
            int j = i;
            while (j + 1 < n && x[order[j + 1]] == x[order[i]])
                j++;
            int count = j - i + 1;
            double avgRank = (i + 1 + j + 1) / 2.0;
            for (int k = i; k <= j; k++)
                ranks[order[k]] = avgRank;
            tieCounts[i] = count;  // Position-in-sorted-order semantics; zeros elsewhere.
            if (count > 1)
                hasTies = true;
            i = j + 1;
        }
        return (ranks, tieCounts, hasTies);
    }

    /// <summary>The normal approximation with tie correction (scipy <c>method="asymptotic"</c>, <c>correction=False</c>).</summary>
    /// <param name="n">The number of elements or points to process.</param>
    /// <param name="rPlus">The sum of ranks assigned to positive differences.</param>
    /// <param name="tieCounts">The size of each tied rank group used in the variance correction.</param>
    /// <param name="alternative">The one-sided alternative hypothesis for the signed differences.</param>
    /// <returns>The normal-approximation one-sided p-value with the tie correction.</returns>
    private static double AsymptoticPValue(int n, double rPlus, long[] tieCounts, Alternative alternative)
    {
        double mn = n * (n + 1.0) * 0.25;
        double se = n * (n + 1.0) * (2.0 * n + 1.0);

        double tieCorrect = 0;
        foreach (long t in tieCounts)
            tieCorrect += (double)t * t * t - t;
        se -= tieCorrect / 2;
        se = Math.Sqrt(se / 24);

        double z = (rPlus - mn) / se;
        return alternative == Alternative.Less ? Erf.NormCdf(z) : Erf.NormSf(z);
    }

    /// <summary>The exact null distribution of <c>r_plus</c> (no ties/zeros; scipy <c>WilcoxonDistribution</c>).</summary>
    /// <param name="n">The number of elements or points to process.</param>
    /// <param name="rPlus">The sum of ranks assigned to positive differences.</param>
    /// <param name="alternative">The one-sided alternative hypothesis for the signed differences.</param>
    /// <returns>The exact one-sided p-value from the signed-rank null distribution.</returns>
    private static double ExactPValue(int n, double rPlus, Alternative alternative)
    {
        // pmf of r_plus over 0..n(n+1)/2: DP over rank subsets, each outcome weight 2^-n.
        int maxSum = n * (n + 1) / 2;
        var counts = new double[maxSum + 1];
        counts[0] = 1;
        for (int rank = 1; rank <= n; rank++)
        {
            for (int s = maxSum; s >= rank; s--)
                counts[s] += counts[s - rank];
        }
        double total = Math.Pow(2, n);
        double mean = n * (n + 1) / 4.0;

        // scipy is conservative with non-integral statistics: ceil before CDF, floor before SF.
        if (alternative == Alternative.Less)
        {
            long k = (long)Math.Ceiling(rPlus);
            return Cdf(counts, total, maxSum, mean, k);
        }
        else
        {
            long k = (long)Math.Floor(rPlus);
            return Sf(counts, total, maxSum, mean, k);
        }
    }

    /// <summary>Returns the lower-tail probability at the requested rank sum.</summary>
    /// <param name="counts">The number or probability mass of rank-sum outcomes at each integer sum.</param>
    /// <param name="total">The total mass of all possible rank-sum outcomes.</param>
    /// <param name="maxSum">The largest attainable sum of positive ranks.</param>
    /// <param name="mean">The predictive mean in the standardized objective coordinate system.</param>
    /// <param name="k">The integer index or combinatorial selection count used by this operation.</param>
    /// <returns>The lower-tail probability at the requested rank sum.</returns>
    private static double Cdf(double[] counts, double total, int maxSum, double mean, long k)
    {
        if (k < 0)
            return 0;
        if (k >= maxSum)
            return 1;
        if (k <= mean)
        {
            double sum = 0;
            for (long s = 0; s <= k; s++)
                sum += counts[s];
            return sum / total;
        }
        return 1 - Sf(counts, total, maxSum, mean, k + 1);
    }

    /// <summary>Returns the upper-tail probability at the requested rank sum.</summary>
    /// <param name="counts">The number or probability mass of rank-sum outcomes at each integer sum.</param>
    /// <param name="total">The total mass of all possible rank-sum outcomes.</param>
    /// <param name="maxSum">The largest attainable sum of positive ranks.</param>
    /// <param name="mean">The predictive mean in the standardized objective coordinate system.</param>
    /// <param name="k">The integer index or combinatorial selection count used by this operation.</param>
    /// <returns>The upper-tail probability at the requested rank sum.</returns>
    private static double Sf(double[] counts, double total, int maxSum, double mean, long k)
    {
        if (k <= 0)
            return 1;
        if (k > maxSum)
            return 0;
        if (k <= mean)
            return 1 - Cdf(counts, total, maxSum, mean, k - 1);
        double sum = 0;
        for (long s = k; s <= maxSum; s++)
            sum += counts[s];
        return sum / total;
    }

    /// <summary>
    ///     scipy's deterministic sign-flip permutation test (<c>PermutationMethod</c> with
    ///     <c>2^n &lt;= 9999</c> outcomes): the exact distribution of the zsplit statistic over
    ///     all sign assignments; p-values compare with scipy's relative tolerance
    ///     <c>gamma = 100·eps·|observed|</c> and no +1 adjustment (exhaustive enumeration).
    /// </summary>
    /// <param name="d">The signed paired differences being tested.</param>
    /// <param name="ranks">The average rank assigned to each absolute nonzero difference.</param>
    /// <param name="rZeroHalfTotal">Half the rank mass of zero differences, assigned to each side by the zsplit convention.</param>
    /// <param name="observed">The observed positive-rank sum compared with the null distribution.</param>
    /// <param name="alternative">The one-sided alternative hypothesis for the signed differences.</param>
    /// <returns>The one-sided p-value from exhaustive sign permutations.</returns>
    private static double PermutationPValue(
        double[] d, double[] ranks, double rZeroHalfTotal, double observed, Alternative alternative)
    {
        // Zero entries contribute rZero/2 regardless of sign, so enumerate only nonzero indices;
        // each pattern then implicitly stands for 2^nZero equal outcomes.
        var nonzeroRanks = new System.Collections.Generic.List<double>();
        for (int i = 0; i < d.Length; i++)
        {
            if (d[i] != 0)
                nonzeroRanks.Add(ranks[i]);
        }
        int m = nonzeroRanks.Count;
        double baseline = rZeroHalfTotal / 2;

        double gamma = Math.Abs(2.220446049250313e-16 * 100 * observed);
        long countCmp = 0;
        long totalOutcomes = 1L << m;
        for (long mask = 0; mask < totalOutcomes; mask++)
        {
            double stat = baseline;
            for (int bit = 0; bit < m; bit++)
            {
                if ((mask & (1L << bit)) != 0)
                    stat += nonzeroRanks[bit];
            }
            if (alternative == Alternative.Less ? stat <= observed + gamma : stat >= observed - gamma)
                countCmp++;
        }
        return (double)countCmp / totalOutcomes;
    }
}
