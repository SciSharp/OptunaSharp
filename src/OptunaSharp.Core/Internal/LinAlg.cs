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
    {
        // View both operands zero-copy (copy: false pins the managed arrays instead of
        // cloning them) and dispose the transients here — none escape. np.linalg.solve
        // copies its inputs into internal LAPACK work buffers (it never writes a/b), so a
        // view is read-only-safe; the returned solution is a fresh owned NDArray.
        using NDArray na = np.array(a, copy: false);
        using NDArray nb = np.array(b, copy: false);
        using NDArray solution = np.linalg.solve(na, nb);
        return solution.ToArray<double>();
    }

    /// <summary>Matrix product <c>a · b</c> — <c>np.dot</c> (CBLAS <c>gemm</c>, byte-identical to NumPy).</summary>
    /// <param name="a">The first numerical operand; its dimensions must agree with the operation described in the summary.</param>
    /// <param name="b">The second numerical operand; its dimensions must agree with the first operand.</param>
    /// <returns>The matrix or vector product with rows determined by the first operand.</returns>
    internal static double[,] Dot(double[,] a, double[,] b)
    {
        // Zero-copy views of the operands; np.dot reads both (BLAS gemm) and returns a
        // fresh product. Dispose all three transients — To2D has copied the values out.
        using NDArray na = np.array(a, copy: false);
        using NDArray nb = np.array(b, copy: false);
        using NDArray product = np.dot(na, nb);
        return To2D(product, a.GetLength(0), b.GetLength(1));
    }

    /// <summary>Matrix-vector product <c>a · b</c> — <c>np.dot</c> (byte-identical to NumPy).</summary>
    /// <param name="a">The first numerical operand; its dimensions must agree with the operation described in the summary.</param>
    /// <param name="b">The second numerical operand; its dimensions must agree with the first operand.</param>
    /// <returns>The matrix or vector product with rows determined by the first operand.</returns>
    internal static double[] Dot(double[,] a, double[] b)
    {
        // Zero-copy views; np.dot reads both operands and returns a fresh vector.
        using NDArray na = np.array(a, copy: false);
        using NDArray nb = np.array(b, copy: false);
        using NDArray product = np.dot(na, nb);
        return product.ToArray<double>();
    }

    /// <summary>Outer product <c>a ⊗ b</c> — <c>np.outer</c> (an elementwise product, exact).</summary>
    /// <param name="a">The first numerical operand; its dimensions must agree with the operation described in the summary.</param>
    /// <param name="b">The second numerical operand; its dimensions must agree with the first operand.</param>
    /// <returns>The matrix whose (i,j) entry is the product of the corresponding vector elements.</returns>
    internal static double[,] Outer(double[] a, double[] b)
    {
        // Zero-copy views; np.outer reads both operands and returns a fresh matrix.
        using NDArray na = np.array(a, copy: false);
        using NDArray nb = np.array(b, copy: false);
        using NDArray product = np.outer(na, nb);
        return To2D(product, a.Length, b.Length);
    }

    /// <summary>Euclidean norm <c>‖a‖₂</c> — <c>np.linalg.norm</c>.</summary>
    /// <param name="a">The vector whose Euclidean norm is computed.</param>
    /// <returns>The Euclidean length of the vector.</returns>
    internal static double Norm(double[] a)
    {
        // Zero-copy view; norm reads it and returns a fresh scalar NDArray.
        using NDArray na = np.array(a, copy: false);
        using NDArray norm = np.linalg.norm(na);
        return (double)norm;
    }

    /// <summary>The rank of <paramref name="a" /> — <c>np.linalg.matrix_rank</c> (SVD-based tolerance).</summary>
    /// <param name="a">The rectangular matrix whose numerical rank is computed.</param>
    /// <returns>The number of singular values exceeding NumSharp's rank tolerance.</returns>
    internal static int MatrixRank(double[,] a)
    {
        // Zero-copy view; matrix_rank reads it (SVD) and returns a fresh scalar NDArray.
        using NDArray na = np.array(a, copy: false);
        using NDArray rank = np.linalg.matrix_rank(na);
        return (int)Convert.ToInt64(rank.GetValue(0));
    }

    /// <summary>The determinant of <paramref name="a" /> — <c>np.linalg.det</c> (LAPACK LU).</summary>
    /// <param name="a">The square matrix whose determinant is computed.</param>
    /// <returns>The determinant of the square input matrix.</returns>
    internal static double Det(double[,] a)
    {
        // Zero-copy view; det reads it (LU) and returns a fresh scalar NDArray.
        using NDArray na = np.array(a, copy: false);
        using NDArray det = np.linalg.det(na);
        return (double)det;
    }

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
        // np.linalg.cholesky reads a into an internal LAPACK buffer (potrf) — a view is
        // read-only-safe. Dispose both transients; To2D has copied the factor out.
        using NDArray na = np.array(a, copy: false);
        using NDArray factor = np.linalg.cholesky(na);
        return To2D(factor, n, n);
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
        // eigh reads a (syevd) and returns two fresh arrays; view the input, dispose it
        // and both outputs once their values have been copied into managed results.
        using NDArray na = np.array(a, copy: false);
        var (w, v) = np.linalg.eigh(na);
        using (w)
        using (v)
            return (w.ToArray<double>(), To2D(v, n, n));
    }

    /// <summary>
    ///     The reduced orthonormal factor <c>Q</c> of <c>a = Q R</c> — <c>np.linalg.qr</c> (LAPACK
    ///     <c>geqrf</c>/<c>orgqr</c>). <b>Use this instead of a hand-rolled Householder QR whenever the
    ///     per-column signs must match Optuna:</b> QR is unique only up to the signs of R's diagonal, and
    ///     LAPACK's sign convention — reproduced here byte-for-byte with NumPy — is what a managed
    ///     Householder cannot portably match (its accumulated <c>Q = Q·Hₖ</c> flips column signs vs
    ///     <c>orgqr</c>). Only the factor <c>Q</c> is returned because that is all the sign-sensitive
    ///     callers (UNDX crossover's PSL basis) need.
    /// </summary>
    /// <param name="a">The matrix to factor, <c>rows ≥ cols</c>; Q comes back <c>rows × min(rows, cols)</c>.</param>
    /// <returns>The reduced orthonormal factor Q, with LAPACK/NumPy column signs.</returns>
    /// <exception cref="NumSharp.Interop.OpenBLAS.OpenBlasMissingBackendException">The native OpenBLAS/LAPACK backend is not loaded; gate on <see cref="NumSharp.Interop.OpenBLAS.OpenBlasEngine.LapackAvailable" /> before calling on a no-native (Tier N) build.</exception>
    internal static double[,] QrQ(double[,] a)
    {
        int rows = a.GetLength(0);
        // Reduced mode: Q is rows×k where k = min(rows, cols); for the square UNDX matrix k == rows.
        int k = Math.Min(rows, a.GetLength(1));
        // View the input (qr reads it, geqrf/orgqr on internal buffers). Dispose the input
        // AND both factors — the R factor is unused here but is still a fresh owned array
        // that would otherwise leak to the finalizer (the old `_` discard did exactly that).
        using NDArray na = np.array(a, copy: false);
        var (q, r) = np.linalg.qr(na);
        using (q)
        using (r)
            return To2D(q, rows, k);
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
