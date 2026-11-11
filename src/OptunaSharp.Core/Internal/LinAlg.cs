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
    /// <exception cref="NumSharp.OpenBlasMissingBackendException">The native OpenBLAS/LAPACK backend is not loaded; gate on <see cref="NumSharp.Interop.OpenBLAS.OpenBlasEngine.LapackAvailable" /> before calling on a no-native (Tier N) build.</exception>
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
    ///     Triangular solve with a vector right-hand side against <paramref name="a" />'s transpose, read in place:
    ///     <c>SolveTriangular(a, b, upper, transposeA: true)</c> computes exactly what
    ///     <c>SolveTriangular(Transpose(a), b, !upper)</c> does, without materializing the transpose.
    /// </summary>
    /// <param name="a">The triangular coefficient matrix as stored; with <paramref name="transposeA" />, its transpose is solved with.</param>
    /// <param name="b">The right-hand side; its length is the system size.</param>
    /// <param name="upper">Whether <paramref name="a" /> is upper triangular as stored.</param>
    /// <param name="transposeA">Whether to solve with <paramref name="a" />'s transpose.</param>
    /// <returns>The solution vector.</returns>
    /// <exception cref="IndexOutOfRangeException"><paramref name="a" /> is smaller than the system <paramref name="b" /> defines.</exception>
    /// <remarks>
    ///     Plan 15: the GP's per-point posterior and its covariance cache solved <c>Lᵀ v = w</c> by transposing the
    ///     Cholesky factor first, an n × n allocation per call (on the large-object heap from about a hundred training
    ///     points) inside the acquisition's local search. Reading <c>a[j, i]</c> for <c>op(a)[i, j]</c> yields the same
    ///     coefficients, and the substitution direction follows from <paramref name="upper" /> XOR
    ///     <paramref name="transposeA" />, so the operations and their order are unchanged.
    /// </remarks>
    internal static double[] SolveTriangular(double[,] a, double[] b, bool upper, bool transposeA)
        => TriSolveVec(a, b, upper, transposeA);

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
    /// <remarks>
    ///     <para>
    ///         Every right-hand side is solved with <see cref="TriSolveVec" />'s operations in its order: for each
    ///         unknown, <c>s = b[i]</c>, then <c>s -= op(a)[i, j] * x[j]</c> for the solved unknowns <c>j</c> in
    ///         ascending order, then <c>x[i] = s / op(a)[i, i]</c>.
    ///     </para>
    ///     <para>
    ///         <b>All right-hand sides at once (plan 15).</b> The right-hand sides are independent, so they are swept
    ///         together, one unknown at a time, by <see cref="SweepTriangular" /> over a row-major block holding one
    ///         right-hand side per column: each update is a row of the block minus a coefficient times an earlier row,
    ///         contiguous and vectorized across the right-hand sides, and each element still receives its
    ///         right-hand side's own sequence above. A left solve's right-hand sides are already <paramref name="b" />'s
    ///         columns; a right solve's are <paramref name="b" />'s rows, so the block is <paramref name="b" />
    ///         transposed and the solution is transposed back. Before, each right-hand side was copied out, solved
    ///         through a captured accessor with two-dimensional bounds checks, and copied back, with two arrays
    ///         allocated per right-hand side: the GP sampler spent 43% of its time here (its kernel fit inverts the
    ///         Cholesky factor, its posterior solves thousands of candidate rows per call).
    ///     </para>
    /// </remarks>
    internal static double[,] SolveTriangular(double[,] a, double[,] b, bool upper, bool left = true)
    {
        int rows = b.GetLength(0);
        int cols = b.GetLength(1);
        var result = new double[rows, cols];

        if (left)
        {
            // a X = b: the right-hand sides are b's columns, so b itself is the block (n unknowns × cols sides). The
            // system's size is a's, as TriSolveVec took it.
            int n = a.GetLength(0);
            double[] block = System.Buffers.ArrayPool<double>.Shared.Rent(Math.Max(n * cols, 1));
            try
            {
                var y = new Span<double>(block, 0, n * cols);
                for (int i = 0; i < n; i++)
                    for (int c = 0; c < cols; c++)
                        y[i * cols + c] = b[i, c];
                SweepTriangular(a, upper, transpose: false, y, n, cols);
                for (int i = 0; i < n; i++)
                    for (int c = 0; c < cols; c++)
                        result[i, c] = y[i * cols + c];
            }
            finally
            {
                System.Buffers.ArrayPool<double>.Shared.Return(block);
            }
        }
        else
        {
            // X a = b, i.e. aᵀ Xᵀ = bᵀ: the right-hand sides are b's rows, so the block is bᵀ (n unknowns × rows sides).
            int n = a.GetLength(0);
            double[] block = System.Buffers.ArrayPool<double>.Shared.Rent(Math.Max(n * rows, 1));
            try
            {
                var y = new Span<double>(block, 0, n * rows);
                for (int r = 0; r < rows; r++)
                    for (int j = 0; j < n; j++)
                        y[j * rows + r] = b[r, j];
                SweepTriangular(a, upper, transpose: true, y, n, rows);
                for (int r = 0; r < rows; r++)
                    for (int j = 0; j < n; j++)
                        result[r, j] = y[j * rows + r];
            }
            finally
            {
                System.Buffers.ArrayPool<double>.Shared.Return(block);
            }
        }
        return result;
    }

    /// <summary>
    ///     Solves with a Cholesky factorization <c>C = L Lᵀ</c>: <c>C⁻¹ B</c> when <paramref name="left" />, else
    ///     <c>B C⁻¹</c>; exactly the values of the two triangular solves the GP stack spelled out:
    ///     <c>SolveTriangular(Lᵀ, SolveTriangular(L, B, upper: false), upper: true)</c> on the left,
    ///     <c>SolveTriangular(L, SolveTriangular(Lᵀ, B, upper: true, left: false), upper: false, left: false)</c> on the
    ///     right.
    /// </summary>
    /// <param name="l">The lower-triangular Cholesky factor L; its upper triangle is not read.</param>
    /// <param name="b">The right-hand sides: n × k on the left, k × n on the right.</param>
    /// <param name="left">True for <c>C⁻¹ B</c>, false for <c>B C⁻¹</c>.</param>
    /// <returns>A new matrix of <paramref name="b" />'s shape holding the solution.</returns>
    /// <exception cref="IndexOutOfRangeException"><paramref name="b" />'s solved dimension is smaller than <paramref name="l" />'s.</exception>
    /// <remarks>
    ///     <para>
    ///         Plan 15: both triangular solves run on one block of right-hand sides (<paramref name="b" />, or its
    ///         transpose on the right), back to back, through <see cref="SweepTriangular" />: no intermediate matrix, no
    ///         materialized <c>Lᵀ</c>, two copies instead of four. The values are the same because each step reads the
    ///         same coefficients in the same order: solving with <c>Lᵀ</c> stored as an upper-triangular matrix reads
    ///         <c>Lᵀ[i, j]</c>, which is what sweeping <c>L</c> read transposed reads, and the substitution direction is
    ///         the same (back substitution). On the left the steps are L (forward) then <c>Lᵀ</c> (backward); on the
    ///         right, the first solve <c>X Lᵀ = B</c> is a forward sweep with <c>L</c> as stored and the second
    ///         <c>X L = …</c> a backward sweep with <c>L</c> read transposed: the same two sweeps over the transposed
    ///         block.
    ///     </para>
    /// </remarks>
    internal static double[,] CholeskySolve(double[,] l, double[,] b, bool left)
    {
        int rows = b.GetLength(0);
        int cols = b.GetLength(1);
        int n = l.GetLength(0);
        var result = new double[rows, cols];
        // The block holds one right-hand side per column: b's columns on the left, b's rows (b transposed) on the right.
        int width = left ? cols : rows;
        double[] blockBuffer = System.Buffers.ArrayPool<double>.Shared.Rent(Math.Max(n * width, 1));
        try
        {
            var y = new Span<double>(blockBuffer, 0, n * width);
            if (left)
            {
                for (int i = 0; i < n; i++)
                    for (int c = 0; c < cols; c++)
                        y[i * cols + c] = b[i, c];
            }
            else
            {
                for (int r = 0; r < rows; r++)
                    for (int j = 0; j < n; j++)
                        y[j * rows + r] = b[r, j];
            }

            SweepTriangular(l, upper: false, transpose: false, y, n, width);   // L, forward
            SweepTriangular(l, upper: false, transpose: true, y, n, width);    // Lᵀ, backward

            if (left)
            {
                for (int i = 0; i < n; i++)
                    for (int c = 0; c < cols; c++)
                        result[i, c] = y[i * cols + c];
            }
            else
            {
                for (int r = 0; r < rows; r++)
                    for (int j = 0; j < n; j++)
                        result[r, j] = y[j * rows + r];
            }
        }
        finally
        {
            System.Buffers.ArrayPool<double>.Shared.Return(blockBuffer);
        }
        return result;
    }

    /// <summary>
    ///     Solves <c>op(a) Y = B</c> in place for a block of right-hand sides, one per column of the row-major block
    ///     <paramref name="y" />, which holds <c>B</c> on entry and <c>Y</c> on return; <c>op(a)</c> is <paramref name="a" />
    ///     or, with <paramref name="transpose" />, its transpose. Forward substitution when the effective system is lower
    ///     triangular (<paramref name="upper" /> XOR <paramref name="transpose" /> is false), back substitution otherwise.
    /// </summary>
    /// <param name="a">The triangular coefficient matrix in its stored orientation; only the effective triangle is read.</param>
    /// <param name="upper">Whether <paramref name="a" /> is upper triangular as stored.</param>
    /// <param name="transpose">Whether to solve with <paramref name="a" />'s transpose.</param>
    /// <param name="y">The block, <paramref name="n" /> rows (unknowns) of <paramref name="width" /> right-hand sides, row-major.</param>
    /// <param name="n">The number of unknowns per right-hand side; <paramref name="a" /> is at least n × n.</param>
    /// <param name="width">The number of right-hand sides.</param>
    /// <exception cref="IndexOutOfRangeException"><paramref name="a" /> is smaller than n × n or <paramref name="y" /> is shorter than n × width.</exception>
    /// <remarks>
    ///     Row <c>i</c> starts as <c>b[i]</c>, then for each solved row <c>j</c> in ascending order becomes
    ///     <c>row_i - op(a)[i, j] * row_j</c>, then <c>row_i / op(a)[i, i]</c>: per element, exactly
    ///     <see cref="TriSolveVec" />'s <c>s -= op(a)[i, j] * x[j]</c> sequence and final division. Elements of a row
    ///     never mix, so the vector path computes each with the scalar path's operations (.NET never fuses the
    ///     multiplication and the subtraction on its own); the elements after the last whole vector take the scalar path.
    /// </remarks>
    private static void SweepTriangular(double[,] a, bool upper, bool transpose, Span<double> y, int n, int width)
    {
        // Transposition reverses which triangle holds the dependencies; XOR picks the substitution direction without
        // materializing the transposed coefficient matrix, as TriSolveVec does.
        bool effectiveUpper = upper ^ transpose;
        if (!effectiveUpper)
        {
            for (int i = 0; i < n; i++)
            {
                Span<double> rowI = y.Slice(i * width, width);
                for (int j = 0; j < i; j++)
                    SubtractScaledRow(rowI, transpose ? a[j, i] : a[i, j], y.Slice(j * width, width));
                DivideRow(rowI, a[i, i]);
            }
        }
        else
        {
            for (int i = n - 1; i >= 0; i--)
            {
                Span<double> rowI = y.Slice(i * width, width);
                for (int j = i + 1; j < n; j++)
                    SubtractScaledRow(rowI, transpose ? a[j, i] : a[i, j], y.Slice(j * width, width));
                DivideRow(rowI, a[i, i]);
            }
        }
    }

    /// <summary><c>row[k] = row[k] - coefficient * other[k]</c> for every element, the substitution update of one unknown.</summary>
    /// <param name="row">The row being solved; updated in place.</param>
    /// <param name="coefficient">The coefficient of the already-solved unknown <paramref name="other" /> holds.</param>
    /// <param name="other">An already-solved row, as long as <paramref name="row" />.</param>
    /// <exception cref="IndexOutOfRangeException"><paramref name="other" /> is shorter than <paramref name="row" />.</exception>
    private static void SubtractScaledRow(Span<double> row, double coefficient, ReadOnlySpan<double> other)
    {
        int width = row.Length;
        if (System.Numerics.Vector.IsHardwareAccelerated && width >= System.Numerics.Vector<double>.Count)
        {
            // Vector path: whole vectors here, the elements after the last one on the scalar path below.
            int vectorWidth = System.Numerics.Vector<double>.Count;
            var coefficients = new System.Numerics.Vector<double>(coefficient);
            int k = 0;
            for (; k <= width - vectorWidth; k += vectorWidth)
            {
                var updated = new System.Numerics.Vector<double>(row[k..]) - coefficients * new System.Numerics.Vector<double>(other[k..]);
                updated.CopyTo(row[k..]);
            }
            for (; k < width; k++)
                row[k] -= coefficient * other[k];
        }
        else
        {
            // Scalar path: no vector hardware, or a row narrower than one vector.
            for (int k = 0; k < width; k++)
                row[k] -= coefficient * other[k];
        }
    }

    /// <summary><c>row[k] = row[k] / divisor</c> for every element: the final division by the unknown's diagonal coefficient.</summary>
    /// <param name="row">The row being solved; updated in place.</param>
    /// <param name="divisor">The diagonal coefficient.</param>
    /// <remarks>A division, never a multiplication by the reciprocal, which would round differently.</remarks>
    private static void DivideRow(Span<double> row, double divisor)
    {
        int width = row.Length;
        if (System.Numerics.Vector.IsHardwareAccelerated && width >= System.Numerics.Vector<double>.Count)
        {
            // Vector path: whole vectors here, the elements after the last one on the scalar path below.
            int vectorWidth = System.Numerics.Vector<double>.Count;
            var divisors = new System.Numerics.Vector<double>(divisor);
            int k = 0;
            for (; k <= width - vectorWidth; k += vectorWidth)
                (new System.Numerics.Vector<double>(row[k..]) / divisors).CopyTo(row[k..]);
            for (; k < width; k++)
                row[k] /= divisor;
        }
        else
        {
            // Scalar path: no vector hardware, or a row narrower than one vector.
            for (int k = 0; k < width; k++)
                row[k] /= divisor;
        }
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
