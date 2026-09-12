// MOEA/D scalarizing functions (decomposition) — the aggregation that turns the m-objective vector of
// a candidate into a single subproblem fitness, given that subproblem's weight vector and the current
// ideal point z*.
//
// Reference: Zhang & Li, "MOEA/D: A Multiobjective Evolutionary Algorithm Based on Decomposition", IEEE
// TEVC 11(6), 2007; the Tchebycheff and PBI forms follow pymoo's `decomposition/tchebicheff.py` and
// `decomposition/{pbi,util}.py` (Apache-2.0). MOEA/D has no Optuna counterpart → beyond-core, validated
// by convergence/hypervolume, not bit-matched to an upstream.

using System;

namespace OptunaSharp.Samplers.Moead;

/// <summary>The scalarizing function MOEA/D uses to decompose the multi-objective problem into
///     single-objective subproblems.</summary>
public enum MoeadScalarization
{
    /// <summary>Tchebycheff (weighted-Chebyshev): <c>max_j w_j · |F_j − z*_j|</c>. The MOEA/D default,
    ///     works well on convex and non-convex fronts.</summary>
    Tchebycheff,

    /// <summary>Penalty-based boundary intersection: <c>d1 + θ·d2</c>, the projection distance along the
    ///     weight direction plus a penalty <c>θ</c> (5) times the perpendicular distance.</summary>
    Pbi,

    /// <summary>Weighted sum: <c>Σ_j w_j · (F_j − z*_j)</c>. Cheapest, but cannot reach the concave part
    ///     of a non-convex front.</summary>
    WeightedSum,
}

/// <summary>
///     Evaluates a MOEA/D scalarizing function. Every objective vector <c>f</c> and weight
///     vector are expressed in the <b>minimization</b> convention (lower is better on every objective),
///     with <c>z*</c> the running ideal point; a smaller returned value is a better subproblem fitness.
/// </summary>
internal static class MoeadDecomposition
{
    /// <summary>pymoo's PBI penalty <c>θ</c> (see <c>decomposition/pbi.py</c>, default 5).</summary>
    internal const double PbiTheta = 5.0;

    /// <summary>The scalarized fitness of objective vector <paramref name="f" /> for the subproblem with
    ///     weight vector <paramref name="weight" /> and ideal point <paramref name="ideal" /> (all length m,
    ///     minimization convention).</summary>
    /// <param name="scalarization">Scalar decomposition function used to compare candidates for a subproblem.</param>
    /// <param name="f">Objective vectors in minimization orientation.</param>
    /// <param name="weight">Subproblem weights aligned with objective order.</param>
    /// <param name="ideal">Ideal objective vector used as the normalization origin.</param>
    /// <returns>The decomposition-specific scalar loss for the weight vector.</returns>
    /// <exception cref="System.ArgumentOutOfRangeException">The scalarization enum value is unsupported.</exception>
    internal static double Scalarize(
        MoeadScalarization scalarization, double[] f, double[] weight, double[] ideal)
    {
        int m = f.Length;
        switch (scalarization)
        {
            case MoeadScalarization.Tchebycheff:
            {
                // max_j weight_j * |F_j - z*_j|  (pymoo uses raw weights, so a zero weight drops that axis
                // — that is how boundary weight vectors reduce to single-objective subproblems).
                double best = double.NegativeInfinity;
                for (int j = 0; j < m; j++)
                {
                    double v = weight[j] * Math.Abs(f[j] - ideal[j]);
                    if (v > best) best = v;
                }
                return best;
            }
            case MoeadScalarization.WeightedSum:
            {
                double s = 0.0;
                for (int j = 0; j < m; j++) s += weight[j] * (f[j] - ideal[j]);
                return s;
            }
            case MoeadScalarization.Pbi:
            {
                // d1 = (F' · w) / ||w|| ; d2 = || F' - d1 * w/||w|| || ; g = d1 + theta * d2, F' = F - z*.
                double norm = 0.0;
                for (int j = 0; j < m; j++) norm += weight[j] * weight[j];
                norm = Math.Sqrt(norm);
                if (norm <= 0.0)
                {
                    // Degenerate all-zero weight: fall back to the plain distance to the ideal point.
                    double d = 0.0;
                    for (int j = 0; j < m; j++) { double diff = f[j] - ideal[j]; d += diff * diff; }
                    return Math.Sqrt(d);
                }

                double d1 = 0.0;
                for (int j = 0; j < m; j++) d1 += (f[j] - ideal[j]) * weight[j];
                d1 /= norm;

                double d2 = 0.0;
                for (int j = 0; j < m; j++)
                {
                    double proj = d1 * weight[j] / norm;
                    double perp = (f[j] - ideal[j]) - proj;
                    d2 += perp * perp;
                }
                d2 = Math.Sqrt(d2);

                return d1 + PbiTheta * d2;
            }
            default:
                throw new ArgumentOutOfRangeException(nameof(scalarization), scalarization, null);
        }
    }
}
