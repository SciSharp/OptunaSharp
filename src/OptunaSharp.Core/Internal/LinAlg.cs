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
///     computed in managed code — a plain forward/back substitution, exact to a rounding. NumSharp
///     surfaces no <em>public</em> triangular solve to route it through (no <c>np.linalg</c> entry,
///     no <c>IBlasBackend.TryTrsm</c>); its OpenBLAS interop does now bind <c>cblas_dtrsm</c> at the
///     native layer (<c>OpenBlasNative.Dtrsm</c>), but that binding is <c>internal</c> and
///     unreachable from OptunaSharp — so this awaits a public NumSharp triangular-solve op. It is one
///     of the GP fit's documented tolerance sources (alongside the hand-derived MLL gradient and the
///     managed L-BFGS-B, both a different float-op sequence than torch/scipy), which is why the GP
///     oracle is tolerance-exact, not bit-exact — a boundary no linalg-backend switch can remove.
///     By contrast CMA-ES's <see cref="Eigh" /> and the GP's <see cref="Cholesky" /> already route
///     through OpenBLAS and are byte-identical to NumPy (the "5.5 byte-exactness switch," folded into
///     Core when OpenBLAS became a Core dependency — verified bit-for-bit vs numpy 2.4.2).</para>
/// </summary>
internal static class LinAlg
{
    /// <summary>Solve <c>a x = b</c> for a square <paramref name="a" /> — <c>np.linalg.solve</c> (LAPACK <c>gesv</c>).</summary>
    /// <param name="a">The square coefficient matrix of the linear system.</param>
    /// <param name="b">The right-hand-side vector, with one value per coefficient row.</param>
    /// <returns>The solution vector of the square linear system.</returns>
    /// <exception cref="NumSharp.LinAlgError">The coefficient matrix is singular or incompatible with the right-hand side.</exception>
    internal static double[] Solve(double[,] a, double[] b)
        => np.linalg.solve(np.array(a), np.array(b)).ToArray<double>();

    /// <summary>Matrix product <c>a · b</c> — <c>np.dot</c> (CBLAS <c>gemm</c>, byte-identical to NumPy).</summary>
    /// <param name="a">The first numerical operand; its dimensions must agree with the operation described in the summary.</param>
    /// <param name="b">The second numerical operand; its dimensions must agree with the first operand.</param>
    /// <returns>The matrix or vector product with rows determined by the first operand.</returns>
    internal static double[,] Dot(double[,] a, double[,] b)
        => To2D(np.dot(np.array(a), np.array(b)), a.GetLength(0), b.GetLength(1));

    /// <summary>Matrix-vector product <c>a · b</c> — <c>np.dot</c> (byte-identical to NumPy).</summary>
    /// <param name="a">The first numerical operand; its dimensions must agree with the operation described in the summary.</param>
    /// <param name="b">The second numerical operand; its dimensions must agree with the first operand.</param>
    /// <returns>The matrix or vector product with rows determined by the first operand.</returns>
    internal static double[] Dot(double[,] a, double[] b)
        => np.dot(np.array(a), np.array(b)).ToArray<double>();

    /// <summary>Outer product <c>a ⊗ b</c> — <c>np.outer</c> (an elementwise product, exact).</summary>
    /// <param name="a">The first numerical operand; its dimensions must agree with the operation described in the summary.</param>
    /// <param name="b">The second numerical operand; its dimensions must agree with the first operand.</param>
    /// <returns>The matrix whose (i,j) entry is the product of the corresponding vector elements.</returns>
    internal static double[,] Outer(double[] a, double[] b)
        => To2D(np.outer(np.array(a), np.array(b)), a.Length, b.Length);

    /// <summary>Euclidean norm <c>‖a‖₂</c> — <c>np.linalg.norm</c>.</summary>
    /// <param name="a">The vector whose Euclidean norm is computed.</param>
    /// <returns>The Euclidean length of the vector.</returns>
    internal static double Norm(double[] a)
        => (double)np.linalg.norm(np.array(a));

    /// <summary>The rank of <paramref name="a" /> — <c>np.linalg.matrix_rank</c> (SVD-based tolerance).</summary>
    /// <param name="a">The rectangular matrix whose numerical rank is computed.</param>
    /// <returns>The number of singular values exceeding NumSharp's rank tolerance.</returns>
    internal static int MatrixRank(double[,] a)
        => (int)Convert.ToInt64(np.linalg.matrix_rank(np.array(a)).GetValue(0));

    /// <summary>The determinant of <paramref name="a" /> — <c>np.linalg.det</c> (LAPACK LU).</summary>
    /// <param name="a">The square matrix whose determinant is computed.</param>
    /// <returns>The determinant of the square input matrix.</returns>
    internal static double Det(double[,] a)
        => (double)np.linalg.det(np.array(a));

    /// <summary>
    ///     The lower-triangular Cholesky factor <c>L</c> of a symmetric positive-definite
    ///     <paramref name="a" /> (<c>a = L Lᵀ</c>) — <c>np.linalg.cholesky</c> (LAPACK <c>potrf</c>).
    ///     Raises <see cref="LinAlgError" /> when <paramref name="a" /> is not positive definite.
    /// </summary>
    /// <param name="a">The square symmetric positive-definite matrix to factor.</param>
    /// <returns>The lower-triangular factor L satisfying A = L * transpose(L).</returns>
    /// <exception cref="NumSharp.LinAlgError">The covariance matrix is not numerically positive definite and cannot be Cholesky-factored.</exception>
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
    /// <param name="a">The square symmetric matrix; the lower triangle determines the eigensystem.</param>
    /// <returns>Ascending eigenvalues and the corresponding orthonormal eigenvectors as columns.</returns>
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
    /// <param name="a">The triangular coefficient matrix; only the selected triangle participates in the solve.</param>
    /// <param name="b">The compatible vector or matrix of right-hand sides.</param>
    /// <param name="upper">Whether the supplied coefficient matrix is upper triangular.</param>
    /// <returns>The solution of the requested left- or right-sided triangular system.</returns>
    internal static double[] SolveTriangular(double[,] a, double[] b, bool upper)
        => TriSolveVec(a, b, upper, transpose: false);

    /// <summary>
    ///     Triangular solve with a matrix right-hand side — torch's
    ///     <c>solve_triangular(a, b, upper, left)</c>. <paramref name="left" /> <c>true</c> solves
    ///     <c>a X = b</c> (column by column); <c>false</c> solves <c>X a = b</c> (row by row).
    /// </summary>
    /// <param name="a">The triangular coefficient matrix; only the selected triangle participates in the solve.</param>
    /// <param name="b">The compatible vector or matrix of right-hand sides.</param>
    /// <param name="upper">Whether the supplied coefficient matrix is upper triangular.</param>
    /// <param name="left">True solves with the coefficient matrix on the left; false solves with it on the right.</param>
    /// <returns>The solution of the requested left- or right-sided triangular system.</returns>
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
    /// <param name="a">The triangular coefficient matrix in its stored orientation.</param>
    /// <param name="b">The right-hand-side vector; its length determines the system size.</param>
    /// <param name="upper">Whether the supplied coefficient matrix is upper triangular.</param>
    /// <param name="transpose">Whether to solve using the transposed coefficient matrix.</param>
    /// <returns>The vector solving the triangular system in the requested orientation.</returns>
    private static double[] TriSolveVec(double[,] a, double[] b, bool upper, bool transpose)
    {
        int n = b.Length;
        var x = new double[n];
        // Transposition reverses which triangle contains the dependency graph; XOR chooses the
        // substitution direction without materializing a transposed coefficient matrix.
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
    /// <param name="nd">The NumSharp array whose row-major values are copied.</param>
    /// <param name="rows">The number of matrix rows to allocate or copy.</param>
    /// <param name="cols">The number of columns in the resulting matrix.</param>
    /// <returns>A new rectangular matrix filled from the source's row-major values.</returns>
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
