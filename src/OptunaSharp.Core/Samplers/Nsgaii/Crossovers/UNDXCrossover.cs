using System;
using NumSharp;

namespace OptunaSharp.Samplers.Nsgaii;

/// <summary>
///     Unimodal Normal Distribution Crossover used by <see cref="NSGAIISampler" />. Port of
///     <c>optuna/samplers/nsgaii/_crossovers/_undx.py: UNDXCrossover</c> (experimental v3.0.0).
///
///     Generates a child from three parents using a multivariate normal distribution.
///
///     <para><b>[Misaligned] managed QR (n &gt; 1 only).</b> The orthonormal basis orthogonal to the
///     primary search line comes from <c>np.linalg.qr</c>, i.e. LAPACK <c>dgeqrf</c>/<c>dorgqr</c>.
///     OptunaSharp uses a managed Householder QR that reproduces numpy's basis vectors to ~1e-14 in
///     magnitude but <b>not their per-column signs</b> (LAPACK's column-sign bookkeeping is internal
///     and not portably reproducible — the same managed-vs-LAPACK class the plan excuses in §10).
///     Because <c>etas ~ N(0, σ²)</c> is symmetric, a sign-flipped basis is still a correct UNDX
///     sampler, but with two or more numerical parameters the child values differ from Optuna by
///     O(1), and that difference cascades through the crowding-distance/non-domination selection —
///     so the study trajectory diverges from Optuna's for such spaces. The Gaussian draw count/order
///     is identical regardless (the study stays in RNG lock-step within OptunaSharp, so it is
///     reproducible), and a single numerical parameter is bit-exact (the basis is unused).</para>
/// </summary>
public class UNDXCrossover : BaseCrossover
{
    private readonly double _sigmaXi;
    private readonly double? _sigmaEta;

    /// <param name="sigmaXi">Parametrizes the normal distribution from which <c>xi</c> is drawn.</param>
    /// <param name="sigmaEta">Parametrizes the normal distribution from which <c>etas</c> are drawn. Defaults to <c>0.35 / sqrt(len(search_space))</c>.</param>
    /// <summary>Configures normal offspring variation along and perpendicular to the primary parent axis.</summary>
    public UNDXCrossover(double sigmaXi = 0.5, double? sigmaEta = null)
    {
        _sigmaXi = sigmaXi;
        _sigmaEta = sigmaEta;
    }

    /// <inheritdoc />
    public override int NParents => 3;

    /// <inheritdoc />
    public override double[] Crossover(
        double[][] parentsParams, NumPyRandom rng, Study study, double[,] searchSpaceBounds)
    {
        int n = searchSpaceBounds.GetLength(0);
        var p0 = parentsParams[0];
        var p1 = parentsParams[1];

        var xp = new double[n];
        var d = new double[n];
        for (int i = 0; i < n; i++)
        {
            xp[i] = (p0[i] + p1[i]) / 2;  // Section 2 (2).
            d[i] = p0[i] - p1[i];         // Section 2 (3).
        }

        double sigmaEta = _sigmaEta ?? 0.35 / Math.Sqrt(n);

        // Draw order: etas (n gaussians) then xi (1 gaussian) — Optuna passes the *variance* as the
        // normal `scale` argument, replicated verbatim.
        var etas = rng.normal(0.0, Math.Pow(sigmaEta, 2.0), new Shape(n)).ToArray<double>();
        double xi = (double)rng.normal(0.0, Math.Pow(_sigmaXi, 2.0));

        var es = OrthonormalBasisVectorToPsl(parentsParams, n);  // Orthonormal basis orthogonal to psl.

        var child = new double[n];
        for (int i = 0; i < n; i++)
            child[i] = xp[i] + xi * d[i];  // one + two (Section 2 (5)).

        if (n > 1)  // When n == 1 there is no sub-search component.
        {
            double bigD = DistanceFromXToPsl(parentsParams);  // Section 2 (4).
            var three = new double[n];
            for (int i = 0; i < n - 1; i++)
            {
                for (int c = 0; c < n; c++)
                    three[c] += etas[i] * es[i][c];
            }
            for (int c = 0; c < n; c++)
                child[c] += three[c] * bigD;
        }

        return child;
    }

    /// <summary>Python <c>_normalized_x1_to_x2</c>: normalized vector from x1 to x2.</summary>
    /// <param name="parentsParams">Numerical parent vectors, one parent per row and one parameter per column.</param>
    /// <returns>A unit vector along the primary parent axis.</returns>
    private static double[] NormalizedX1ToX2(double[][] parentsParams)
    {
        int n = parentsParams[0].Length;
        var v12 = new double[n];
        double sumSq = 0.0;
        for (int i = 0; i < n; i++)
        {
            v12[i] = parentsParams[1][i] - parentsParams[0][i];
            sumSq += v12[i] * v12[i];
        }
        double m12 = Math.Sqrt(sumSq);
        double denom = Math.Max(m12, 1e-10);  // np.clip(m_12, 1e-10, None).
        var e12 = new double[n];
        for (int i = 0; i < n; i++)
            e12[i] = v12[i] / denom;
        return e12;
    }

    /// <summary>Python <c>_distance_from_x_to_psl</c>.</summary>
    /// <param name="parentsParams">Numerical parent vectors, one parent per row and one parameter per column.</param>
    /// <returns>The perpendicular distance to the primary parent line.</returns>
    private static double DistanceFromXToPsl(double[][] parentsParams)
    {
        int n = parentsParams[0].Length;
        var e12 = NormalizedX1ToX2(parentsParams);
        var v13 = new double[n];
        double dot = 0.0;
        for (int i = 0; i < n; i++)
        {
            v13[i] = parentsParams[2][i] - parentsParams[0][i];
            dot += v13[i] * e12[i];
        }
        double sumSq = 0.0;
        for (int i = 0; i < n; i++)
        {
            double v = v13[i] - dot * e12[i];  // orthogonal component.
            sumSq += v * v;
        }
        return Math.Sqrt(sumSq);
    }

    /// <summary>Python <c>_orthonormal_basis_vector_to_psl</c> → returns <c>Q.T[1:]</c> (n-1 rows of length n).</summary>
    /// <param name="parentsParams">Numerical parent vectors, one parent per row and one parameter per column.</param>
    /// <param name="n">Number of numerical coordinates.</param>
    /// <returns>Basis vectors perpendicular to the primary parent line.</returns>
    private static double[][] OrthonormalBasisVectorToPsl(double[][] parentsParams, int n)
    {
        var e12 = NormalizedX1ToX2(parentsParams);
        bool anyNonZero = false;
        for (int i = 0; i < n; i++)
        {
            if (e12[i] != 0.0)
            {
                anyNonZero = true;
                break;
            }
        }

        // basis_matrix = identity(n); if e_12 != 0: basis_matrix[0] = e_12. Then qr(basis_matrix.T).
        var a = new double[n, n];
        for (int i = 0; i < n; i++)
            a[i, i] = 1.0;
        if (anyNonZero)
        {
            for (int c = 0; c < n; c++)
                a[0, c] = e12[c];
        }
        // basis_matrix.T
        var at = new double[n, n];
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                at[i, j] = a[j, i];

        var q = HouseholderQ(at, n);
        // return Q.T[1:] — rows 1..n-1 of Q^T = columns 1..n-1 of Q.
        var es = new double[n - 1][];
        for (int col = 1; col < n; col++)
        {
            var row = new double[n];
            for (int r = 0; r < n; r++)
                row[r] = q[r, col];
            es[col - 1] = row;
        }
        return es;
    }

    /// <summary>
    ///     Householder QR of the <paramref name="n" />×<paramref name="n" /> matrix
    ///     <paramref name="a" />; returns the orthogonal factor <c>Q</c> such that <c>A = Q R</c>,
    ///     using the LAPACK sign convention (<c>alpha = -sign(x0)·‖x‖</c>).
    /// </summary>
    /// <param name="a">First numerical array operand.</param>
    /// <param name="n">Number of numerical coordinates.</param>
    /// <returns>The orthogonal factor produced by Householder reflections.</returns>
    private static double[,] HouseholderQ(double[,] a, int n)
    {
        var r = (double[,])a.Clone();
        var q = new double[n, n];
        for (int i = 0; i < n; i++)
            q[i, i] = 1.0;

        var v = new double[n];
        for (int k = 0; k < n; k++)
        {
            double normX = 0.0;
            for (int i = k; i < n; i++)
                normX += r[i, k] * r[i, k];
            normX = Math.Sqrt(normX);
            if (normX == 0.0)
                continue;

            double alpha = r[k, k] >= 0 ? -normX : normX;
            for (int i = 0; i < n; i++)
                v[i] = 0.0;
            v[k] = r[k, k] - alpha;
            for (int i = k + 1; i < n; i++)
                v[i] = r[i, k];

            double vNorm2 = 0.0;
            for (int i = k; i < n; i++)
                vNorm2 += v[i] * v[i];
            if (vNorm2 == 0.0)
                continue;
            double beta = 2.0 / vNorm2;

            // R = H_k R : for each column j, R[k:,j] -= beta * v * (v . R[k:,j]).
            for (int j = 0; j < n; j++)
            {
                double s = 0.0;
                for (int i = k; i < n; i++)
                    s += v[i] * r[i, j];
                s *= beta;
                for (int i = k; i < n; i++)
                    r[i, j] -= s * v[i];
            }

            // Q = Q H_k : for each row rr, Q[rr,k:] -= beta * (Q[rr,k:] . v) * v.
            for (int rr = 0; rr < n; rr++)
            {
                double s = 0.0;
                for (int i = k; i < n; i++)
                    s += q[rr, i] * v[i];
                s *= beta;
                for (int i = k; i < n; i++)
                    q[rr, i] -= s * v[i];
            }
        }

        return q;
    }
}
