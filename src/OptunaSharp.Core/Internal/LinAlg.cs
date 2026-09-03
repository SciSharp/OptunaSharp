using System;
using NumSharp;

namespace OptunaSharp.Internal;

/// <summary>
///     Dense linear-algebra helpers over plain managed arrays (<c>double[,]</c> / <c>double[]</c>)
///     for the samplers that work outside NumSharp's <c>NDArray</c> world. Every factorisation
///     delegates to NumSharp's <c>np.linalg</c>, which — with <c>NumSharp.Interop.OpenBLAS</c>
///     referenced by OptunaSharp.Core and force-loaded by <see cref="OpenBlasActivation" /> — runs
///     through LAPACK and is <b>byte-identical to NumPy</b> (verified: <c>cholesky</c>,
///     <c>eigh</c> values+vectors, and <c>solve</c> match numpy 2.4.2 bit-for-bit). This is what
///     keeps the linalg-backed samplers byte-exact rather than managed-tolerance: NSGA-III's
///     hyperplane <see cref="Solve" />/<see cref="MatrixRank" /> (2.14, now byte-exact), CMA-ES's
///     <see cref="Eigh" /> (2.16), and the GP stack's <see cref="Cholesky" /> (3.2).
///
///     <para><see cref="SolveTriangular(double[,],double[],bool)" /> is the one primitive still
///     computed in managed code: NumSharp exposes no LAPACK triangular solve (<c>dtrtrs</c>/
///     <c>dtrsm</c>), so it is a plain forward/back substitution here — exact to a rounding, and to
///     be routed through LAPACK once NumSharp gains the binding (with the GP consumer in 3.2).</para>
/// </summary>
internal static class LinAlg
{
    /// <summary>Solve <c>a x = b</c> for a square <paramref name="a" /> — <c>np.linalg.solve</c> (LAPACK <c>gesv</c>).</summary>
    internal static double[] Solve(double[,] a, double[] b)
        => np.linalg.solve(np.array(a), np.array(b)).ToArray<double>();

    /// <summary>Matrix product <c>a · b</c> — <c>np.dot</c> (CBLAS <c>gemm</c>, byte-identical to NumPy).</summary>
    internal static double[,] Dot(double[,] a, double[,] b)
        => To2D(np.dot(np.array(a), np.array(b)), a.GetLength(0), b.GetLength(1));

    /// <summary>Matrix-vector product <c>a · b</c> — <c>np.dot</c> (byte-identical to NumPy).</summary>
    internal static double[] Dot(double[,] a, double[] b)
        => np.dot(np.array(a), np.array(b)).ToArray<double>();

    /// <summary>Outer product <c>a ⊗ b</c> — <c>np.outer</c> (an elementwise product, exact).</summary>
    internal static double[,] Outer(double[] a, double[] b)
        => To2D(np.outer(np.array(a), np.array(b)), a.Length, b.Length);

    /// <summary>Euclidean norm <c>‖a‖₂</c> — <c>np.linalg.norm</c>.</summary>
    internal static double Norm(double[] a)
        => (double)np.linalg.norm(np.array(a));

    /// <summary>The rank of <paramref name="a" /> — <c>np.linalg.matrix_rank</c> (SVD-based tolerance).</summary>
    internal static int MatrixRank(double[,] a)
        => (int)Convert.ToInt64(np.linalg.matrix_rank(np.array(a)).GetValue(0));

    /// <summary>
    ///     The lower-triangular Cholesky factor <c>L</c> of a symmetric positive-definite
    ///     <paramref name="a" /> (<c>a = L Lᵀ</c>) — <c>np.linalg.cholesky</c> (LAPACK <c>potrf</c>).
    ///     Raises <see cref="LinAlgError" /> when <paramref name="a" /> is not positive definite.
    /// </summary>
    internal static double[,] Cholesky(double[,] a)
    {
        int n = a.GetLength(0);
        return To2D(np.linalg.cholesky(np.array(a)), n, n);
    }

    /// <summary>
    ///     Eigenvalues (ascending) and orthonormal eigenvectors (as columns) of a real symmetric
    ///     <paramref name="a" /> — <c>np.linalg.eigh</c> (LAPACK <c>syevd</c>, <c>UPLO='L'</c>), so
    ///     <c>a = V · diag(w) · Vᵀ</c>. Byte-identical to NumPy, signs included.
    /// </summary>
    internal static (double[] Eigenvalues, double[,] Eigenvectors) Eigh(double[,] a)
    {
        int n = a.GetLength(0);
        var (w, v) = np.linalg.eigh(np.array(a));
        return (w.ToArray<double>(), To2D(v, n, n));
    }

    /// <summary>
    ///     Solve the triangular system <c>a x = b</c> for a vector. <paramref name="upper" /> selects
    ///     back substitution (upper <paramref name="a" />) vs forward (lower). Managed substitution —
    ///     see the type remarks.
    /// </summary>
    internal static double[] SolveTriangular(double[,] a, double[] b, bool upper)
        => TriSolveVec(a, b, upper, transpose: false);

    /// <summary>
    ///     Triangular solve with a matrix right-hand side — torch's
    ///     <c>solve_triangular(a, b, upper, left)</c>. <paramref name="left" /> <c>true</c> solves
    ///     <c>a X = b</c> (column by column); <c>false</c> solves <c>X a = b</c> (row by row).
    /// </summary>
    internal static double[,] SolveTriangular(double[,] a, double[,] b, bool upper, bool left = true)
    {
        int rows = b.GetLength(0);
        int cols = b.GetLength(1);
        int n = a.GetLength(0);
        var result = new double[rows, cols];

        if (left)
        {
            var column = new double[n];
            for (int c = 0; c < cols; c++)
            {
                for (int i = 0; i < n; i++)
                    column[i] = b[i, c];
                double[] solved = TriSolveVec(a, column, upper, transpose: false);
                for (int i = 0; i < n; i++)
                    result[i, c] = solved[i];
            }
        }
        else
        {
            var rowVec = new double[n];
            for (int r = 0; r < rows; r++)
            {
                for (int j = 0; j < n; j++)
                    rowVec[j] = b[r, j];
                double[] solved = TriSolveVec(a, rowVec, upper, transpose: true);
                for (int j = 0; j < n; j++)
                    result[r, j] = solved[j];
            }
        }
        return result;
    }

    /// <summary>
    ///     Solve <c>op(a) x = b</c> where <c>op(a)</c> is <c>a</c> (<paramref name="transpose" />
    ///     <c>false</c>) or <c>aᵀ</c> (<c>true</c>), <paramref name="a" /> triangular. The effective
    ///     system is upper-triangular (back substitution) iff <paramref name="upper" /> XOR
    ///     <paramref name="transpose" />.
    /// </summary>
    private static double[] TriSolveVec(double[,] a, double[] b, bool upper, bool transpose)
    {
        int n = b.Length;
        var x = new double[n];
        bool effectiveUpper = upper ^ transpose;

        double Get(int i, int j) => transpose ? a[j, i] : a[i, j];

        if (!effectiveUpper)
        {
            for (int i = 0; i < n; i++)
            {
                double s = b[i];
                for (int j = 0; j < i; j++)
                    s -= Get(i, j) * x[j];
                x[i] = s / Get(i, i);
            }
        }
        else
        {
            for (int i = n - 1; i >= 0; i--)
            {
                double s = b[i];
                for (int j = i + 1; j < n; j++)
                    s -= Get(i, j) * x[j];
                x[i] = s / Get(i, i);
            }
        }
        return x;
    }

    /// <summary>Reshape a row-major flattened <c>NDArray</c> back into a <c>double[rows, cols]</c>.</summary>
    private static double[,] To2D(NDArray nd, int rows, int cols)
    {
        double[] flat = nd.ToArray<double>();
        var result = new double[rows, cols];
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                result[i, j] = flat[i * cols + j];
        return result;
    }
}
