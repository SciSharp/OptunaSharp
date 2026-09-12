using System;

namespace OptunaSharp.Samplers.AgeMoea;

/// <summary>
///     AGE-MOEA-2 (2022) survival — port of <c>pymoo.algorithms.moo.age2.AGEMOEA2Survival</c>
///     (<c>@0.6.1.3</c>). It inherits the whole environmental selection / greedy score from
///     <see cref="AgeMoeaSurvival" /> and overrides only <see cref="ComputeGeometry" /> (the
///     Newton-solved geometry exponent) and <see cref="PairwiseDistances" /> (the manifold-projected
///     geodesic distances).
///
///     Panichella, A. (2022), <i>An Improved Pareto Front Modeling Algorithm for Large-scale
///     Many-Objective Optimization</i>, GECCO 2022, doi:10.1145/3512290.3528732.
/// </summary>
internal sealed class AgeMoea2Survival : AgeMoeaSurvival
{
    /// <summary>
    ///     <c>project_on_manifold(point, p)</c>: scale the whole point by <c>1 / ‖point⁺‖_p</c>, where
    ///     the p-norm is taken over the strictly-positive components only.
    /// </summary>
    /// <param name="point">Objective-space point used for projection or distance measurement.</param>
    /// <param name="p">Minkowski norm exponent describing the front geometry.</param>
    /// <returns>The point scaled onto the positive-coordinate unit p-norm manifold.</returns>
    internal static double[] ProjectOnManifold(double[] point, double p)
    {
        double acc = 0.0;
        foreach (double v in point)
            if (v > 0.0)
                acc += Math.Pow(v, p);
        double dist = Math.Pow(acc, 1.0 / p);

        double inv = 1.0 / dist;
        var result = new double[point.Length];
        for (int k = 0; k < point.Length; k++)
            result[k] = point[k] * inv;
        return result;
    }

    /// <summary>
    ///     <c>find_zero(point, n, precision)</c>: Newton's method (in log-space) for the exponent
    ///     <c>p</c> solving <c>Σ pointᵢ^p = 1</c> over positive components, ≤ 100 iterations.
    ///
    ///     <para><b>pymoo bug preserved verbatim:</b> the reference misspells the loop's update as
    ///     <c>paste_value = x</c>, so <c>past_value</c> stays at its initial <c>1</c> forever and the
    ///     break test is effectively <c>|x − 1| ≤ precision</c> (not "change since the last iterate").
    ///     Reproducing this is required for byte-exactness.</para>
    /// </summary>
    /// <param name="point">Objective-space point used for projection or distance measurement.</param>
    /// <param name="n">Number of numerical coordinates.</param>
    /// <param name="precision">Tolerance used by the geometry-exponent iteration.</param>
    /// <returns>The geometry exponent found by the bounded Newton iteration.</returns>
    internal static double FindZero(double[] point, int n, double precision)
    {
        double x = 1.0;
        double pastValue = x; // NB: pymoo never updates this (typo 'paste_value = x') — do not "fix" it.

        for (int i = 0; i < 100; i++)
        {
            double f = 0.0;
            for (int obj = 0; obj < n; obj++)
                if (point[obj] > 0.0)
                    f += Math.Pow(point[obj], x);
            f = Math.Log(f);

            double numerator = 0.0;
            double denominator = 0.0;
            for (int obj = 0; obj < n; obj++)
                if (point[obj] > 0.0)
                {
                    numerator += Math.Pow(point[obj], x) * Math.Log(point[obj]);
                    denominator += Math.Pow(point[obj], x);
                }

            if (denominator == 0.0)
                return 1.0;

            double ff = numerator / denominator;
            x -= f / ff;

            if (Math.Abs(x - pastValue) <= precision)
                break;
            // else: pymoo's 'paste_value = x' is a no-op typo; pastValue stays 1.
        }

        return x;
    }

    /// <summary>
    ///     AGE-MOEA-2 <c>compute_geometry</c>: pick the interior point of smallest Euclidean norm (not
    ///     an extreme) and solve <c>p = find_zero(point, n, 1e-3)</c>; clamp <c>p ≤ 0.1 / NaN → 1</c>,
    ///     <c>p &gt; 20 → 20</c>.
    /// </summary>
    /// <inheritdoc />
    internal override double ComputeGeometry(double[][] front, int[] extreme, int n)
    {
        int m = front.Length;
        var d = new double[m];
        for (int i = 0; i < m; i++)
        {
            double s = 0.0;
            for (int k = 0; k < n; k++)
                s += front[i][k] * front[i][k]; // front[i] ** 2 (element square)
            d[i] = Math.Pow(s, 0.5); // ** 0.5 (scalar power, matches numpy's pow — not sqrt)
        }
        foreach (int e in extreme) d[e] = double.PositiveInfinity;

        int index = 0;
        double best = d[0];
        for (int i = 1; i < m; i++)
            if (d[i] < best) { best = d[i]; index = i; } // argmin, first occurrence

        double p = FindZero(front[index], n, 0.001);

        if (double.IsNaN(p) || p <= 0.1) p = 1.0;
        else if (p > 20) p = 20.0;
        return p;
    }

    /// <summary>
    ///     AGE-MOEA-2 <c>pairwise_distances</c>: project every point onto the manifold, then take the
    ///     plain Euclidean distance when <c>0.95 &lt; p &lt; 1.05</c>, otherwise the two-segment geodesic
    ///     through the projected midpoint. Returned symmetric (<c>distances + distancesᵀ</c>).
    /// </summary>
    /// <inheritdoc />
    internal override double[][] PairwiseDistances(double[][] front, double p)
    {
        int m = front.Length;
        int n = front[0].Length;

        var projected = new double[m][];
        for (int i = 0; i < m; i++)
            projected[i] = ProjectOnManifold(front[i], p);

        var upper = new double[m][];
        for (int i = 0; i < m; i++) upper[i] = new double[m];

        bool euclidean = 0.95 < p && p < 1.05;
        for (int row = 0; row < m - 1; row++)
            for (int col = row + 1; col < m; col++)
            {
                if (euclidean)
                {
                    double s = 0.0;
                    for (int k = 0; k < n; k++)
                    {
                        double v = Math.Abs(projected[row][k] - projected[col][k]);
                        s += v * v;
                    }
                    upper[row][col] = Math.Pow(s, 0.5);
                }
                else
                {
                    var mid = new double[n];
                    for (int k = 0; k < n; k++)
                        mid[k] = projected[row][k] * 0.5 + projected[col][k] * 0.5;
                    mid = ProjectOnManifold(mid, p);

                    double s1 = 0.0, s2 = 0.0;
                    for (int k = 0; k < n; k++)
                    {
                        double a = Math.Abs(projected[row][k] - mid[k]);
                        double b = Math.Abs(projected[col][k] - mid[k]);
                        s1 += a * a;
                        s2 += b * b;
                    }
                    upper[row][col] = Math.Pow(s1, 0.5) + Math.Pow(s2, 0.5);
                }
            }

        // distances + distances.T
        var sym = new double[m][];
        for (int i = 0; i < m; i++)
        {
            sym[i] = new double[m];
            for (int j = 0; j < m; j++)
                sym[i][j] = upper[i][j] + upper[j][i];
        }
        return sym;
    }
}
