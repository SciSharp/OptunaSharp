using System;

namespace OptunaSharp.Samplers.Nsgaiii;

/// <summary>
///     The small managed dense-linear-algebra primitives the NSGA-III hyperplane fit needs —
///     <c>np.linalg.solve</c> and <c>np.linalg.matrix_rank</c> — over the tiny
///     (<c>n_objectives</c> × <c>n_objectives</c>) extreme-point systems. NumSharp.Core ships no
///     managed SVD (so no <c>matrix_rank</c>) without the optional OpenBLAS backend, and OptunaSharp
///     keeps its numeric primitives in-repo (the same reason <see cref="Crossovers.UNDXCrossover" />
///     carries a managed Householder QR); these are that preview of the plan's 2.15 managed linalg.
///
///     <para><b>allclose, not bit-exact — like every managed-vs-LAPACK primitive.</b>
///     <see cref="Solve" /> is the classic unblocked reciprocal-multiplier LU (LAPACK <c>dgetf2</c>'s
///     <c>DSCAL(1/pivot)</c> + <c>DGER</c> shape), the closest managed match to numpy's
///     <c>np.linalg.solve</c>: bit-for-bit on the overwhelming majority of the small systems that
///     arise, a ~1-ULP drift on a few percent (numpy's LAPACK rides an arch-specific FMA <c>gemm</c>
///     that managed code cannot reproduce). See the <c>[Misaligned]</c> note on
///     <see cref="NSGAIIIElitePopulationSelectionStrategy" />.</para>
/// </summary>
internal static class NsgaiiiLinAlg
{
    // np.finfo(np.float64).eps.
    private const double Eps = 2.220446049250313e-16;

    /// <summary>
    ///     Solve <c>a x = b</c> for a square <paramref name="a" /> via Gaussian elimination with
    ///     partial pivoting (unblocked reciprocal-multiplier LU + unit-lower/upper substitution).
    ///     Does not mutate its arguments. Assumes <paramref name="a" /> is non-singular (the caller
    ///     guards with <see cref="MatrixRank" />, mirroring Optuna).
    /// </summary>
    internal static double[] Solve(double[,] a, double[] b)
    {
        int n = b.Length;
        var m = (double[,])a.Clone();
        var x = (double[])b.Clone();

        for (int k = 0; k < n; k++)
        {
            // Partial pivot: largest-magnitude entry in the column (LAPACK idamax, first on ties).
            int pivot = k;
            for (int i = k + 1; i < n; i++)
                if (Math.Abs(m[i, k]) > Math.Abs(m[pivot, k]))
                    pivot = i;
            if (pivot != k)
            {
                for (int j = 0; j < n; j++)
                    (m[k, j], m[pivot, j]) = (m[pivot, j], m[k, j]);
                (x[k], x[pivot]) = (x[pivot], x[k]);
            }

            // DSCAL: store the column multipliers as col * (1 / pivot).
            double inv = 1.0 / m[k, k];
            for (int i = k + 1; i < n; i++)
                m[i, k] *= inv;

            // DGER: rank-1 update of the trailing submatrix.
            for (int i = k + 1; i < n; i++)
            {
                double mik = m[i, k];
                for (int j = k + 1; j < n; j++)
                    m[i, j] -= mik * m[k, j];
            }
        }

        // Forward substitution through the unit-lower L (multipliers live in m[i, j], j < i).
        for (int i = 0; i < n; i++)
        {
            double s = x[i];
            for (int j = 0; j < i; j++)
                s -= m[i, j] * x[j];
            x[i] = s;
        }
        // Back substitution through the upper U.
        for (int i = n - 1; i >= 0; i--)
        {
            double s = x[i];
            for (int j = i + 1; j < n; j++)
                s -= m[i, j] * x[j];
            x[i] = s / m[i, i];
        }
        return x;
    }

    /// <summary>
    ///     The number of linearly independent rows/columns of <paramref name="a" /> — numpy's
    ///     <c>np.linalg.matrix_rank</c>. Row-echelon via partial-pivot elimination, counting pivots
    ///     above <c>tol = max|a| · max(rows, cols) · eps</c> (numpy's SVD-based tolerance uses the
    ///     largest singular value in place of <c>max|a|</c>; the two agree except at the degeneracy
    ///     boundary). Used only as the degeneracy guard on the extreme-point matrix. Does not mutate
    ///     its argument.
    /// </summary>
    internal static int MatrixRank(double[,] a)
    {
        int rows = a.GetLength(0);
        int cols = a.GetLength(1);
        var m = (double[,])a.Clone();

        double maxAbs = 0.0;
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                maxAbs = Math.Max(maxAbs, Math.Abs(m[i, j]));
        double tol = maxAbs * Math.Max(rows, cols) * Eps;

        int rank = 0;
        int row = 0;
        for (int col = 0; col < cols && row < rows; col++)
        {
            int pivot = row;
            for (int i = row + 1; i < rows; i++)
                if (Math.Abs(m[i, col]) > Math.Abs(m[pivot, col]))
                    pivot = i;
            if (Math.Abs(m[pivot, col]) <= tol)
                continue;

            if (pivot != row)
                for (int j = 0; j < cols; j++)
                    (m[row, j], m[pivot, j]) = (m[pivot, j], m[row, j]);

            for (int i = row + 1; i < rows; i++)
            {
                double f = m[i, col] / m[row, col];
                for (int j = col; j < cols; j++)
                    m[i, j] -= f * m[row, j];
            }
            row++;
            rank++;
        }
        return rank;
    }
}
