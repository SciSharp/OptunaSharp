// Rank-based fitness shaping shared by the Natural-Evolution-Strategy family (OpenAI-ES / SNES / xNES).
//
// References:
//   Tim Salimans, Jonathan Ho, Xi Chen, Szymon Sidor, Ilya Sutskever.
//     "Evolution Strategies as a Scalable Alternative to Reinforcement Learning." arXiv:1703.03864 (2017).
//     §2.1 "fitness shaping" — a rank transformation applied to the returns before the update.
//   Daan Wierstra, Tom Schaul, Tobias Glasmachers, Yi Sun, Jan Peters, Jürgen Schmidhuber.
//     "Natural Evolution Strategies." JMLR 15 (2014) 949–980. §3.1 fitness shaping + the utility
//     function (eq. after Algorithm 3); Table 1 pins it as a function of the population size.
//
// Both families make the update invariant to order-preserving transformations of the objective by
// replacing raw fitness with a rank-based surrogate. They use DIFFERENT surrogates: OpenAI-ES uses
// centered ranks in [-0.5, 0.5] (the OpenAI `evolution-strategies-starter` / `estool` recipe), while
// SNES/xNES use the CMA-style utility weights shifted to sum to zero (a fitness baseline). Both are
// kept here so the three optimizers share one audited, byte-reproducible implementation.

using System;
using System.Linq;

namespace OptunaSharp.Samplers.EvolutionStrategy;

/// <summary>
///     Rank-based fitness-shaping surrogates for the Natural-Evolution-Strategy samplers. All methods
///     are pure and deterministic (LINQ <c>OrderBy</c> is a stable sort, so ties resolve by original
///     index) so a seeded study reproduces bit-for-bit. <b>Orientation:</b> every method here treats a
///     <i>larger</i> score as <i>better</i> (the papers maximize). OptunaSharp minimizes, so callers pass
///     the negated objective as the score — the sole place the minimize/maximize convention is bridged.
/// </summary>
internal static class EsFitnessShaping
{
    /// <summary>
    ///     Ascending competition ranks: <c>result[i]</c> is the number of scores strictly smaller than
    ///     <paramref name="scores" /><c>[i]</c> (ties broken by index, so the rank set is exactly
    ///     <c>{0, …, n−1}</c>). The smallest score gets rank <c>0</c>, the largest rank <c>n−1</c> — the
    ///     building block of both shaping schemes.
    /// </summary>
    /// <param name="scores">Per-individual scores (larger = better); NaN sorts as the smallest (worst) value.</param>
    /// <returns>A per-individual rank in <c>[0, n−1]</c>, aligned with <paramref name="scores" />.</returns>
    internal static int[] ComputeRanks(double[] scores)
    {
        int n = scores.Length;
        // Stable ascending order of indices; NaN is coerced to −∞ so it always sorts to rank 0 (worst),
        // never poisoning the comparison the way a raw NaN would (NaN compares false against everything).
        int[] order = Enumerable.Range(0, n)
            .OrderBy(i => double.IsNaN(scores[i]) ? double.NegativeInfinity : scores[i])
            .ToArray();
        var ranks = new int[n];
        for (int r = 0; r < n; r++)
            ranks[order[r]] = r;
        return ranks;
    }

    /// <summary>
    ///     OpenAI-ES centered ranks (Salimans 2017 §2.1, the <c>estool</c>/<c>evolution-strategies-starter</c>
    ///     recipe): map ascending ranks onto the interval <c>[-0.5, 0.5]</c> via <c>rank/(n−1) − 0.5</c>. The
    ///     best individual weighs <c>+0.5</c>, the worst <c>−0.5</c>, and the weights sum to zero (a built-in
    ///     baseline). This removes outlier influence and makes the gradient estimate scale-free.
    /// </summary>
    /// <param name="scores">Per-individual scores (larger = better) — pass the negated objective when minimizing.</param>
    /// <returns>Centered rank weights in <c>[-0.5, 0.5]</c>, aligned with <paramref name="scores" />.</returns>
    /// <exception cref="System.ArgumentException"><paramref name="scores" /> is empty (a rank scale needs at least one point).</exception>
    /// <remarks>
    ///     A population of one degenerates to a single <c>0.0</c> weight (the <c>n−1</c> denominator would
    ///     divide by zero), which is the correct "no signal" outcome; this is handled explicitly so the
    ///     optimizer never divides by zero.
    /// </remarks>
    internal static double[] CenteredRanks(double[] scores)
    {
        int n = scores.Length;
        if (n == 0)
            throw new ArgumentException("Cannot shape an empty population.", nameof(scores));
        // With a single individual the ranks carry no ordering information and (n-1) = 0 would divide by
        // zero; emit the neutral, zero-sum weight 0.0 (the general formula is zero-sum for every n > 1).
        if (n == 1)
            return new[] { 0.0 };
        int[] ranks = ComputeRanks(scores);
        var y = new double[n];
        double denom = n - 1;
        for (int i = 0; i < n; i++)
            y[i] = ranks[i] / denom - 0.5;
        return y;
    }

    /// <summary>
    ///     NES utility weights (Wierstra 2014, the CMA-derived choice under Algorithm 3, pinned in Table 1):
    ///     <c>u_k = max(0, ln(λ/2 + 1) − ln k) / Σ_j max(0, ln(λ/2 + 1) − ln j) − 1/λ</c> for rank
    ///     <c>k = 1…λ</c> (rank 1 = best). The raw weights are normalized to sum to one, then shifted by
    ///     <c>−1/λ</c> so the utilities sum to <b>zero</b> — the fitness baseline that keeps the natural
    ///     gradient centered. The result is monotonically decreasing: only the better half of the population
    ///     carries positive weight.
    /// </summary>
    /// <param name="lambda">Population size λ (must be ≥ 1); indices map to ranks <c>1…λ</c>.</param>
    /// <returns>Utility weights indexed by <b>rank − 1</b> (so <c>result[0]</c> is the best individual's weight); sums to zero.</returns>
    /// <exception cref="System.ArgumentOutOfRangeException"><paramref name="lambda" /> is below one.</exception>
    internal static double[] NesUtilities(int lambda)
    {
        if (lambda < 1)
            throw new ArgumentOutOfRangeException(nameof(lambda), lambda, "Population size must be at least 1.");
        var raw = new double[lambda];
        double sum = 0;
        double cap = Math.Log(lambda / 2.0 + 1);
        for (int k = 1; k <= lambda; k++)
        {
            // max(0, …) zeroes out the worse-than-median ranks; the survivors form the positive weights.
            double w = Math.Max(0.0, cap - Math.Log(k));
            raw[k - 1] = w;
            sum += w;
        }
        var u = new double[lambda];
        for (int i = 0; i < lambda; i++)
            // Normalize to a unit-sum distribution, then subtract 1/λ so the utilities sum to zero.
            u[i] = raw[i] / sum - 1.0 / lambda;
        return u;
    }
}
