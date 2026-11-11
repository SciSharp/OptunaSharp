using System;
using AwesomeAssertions;
using NumSharp;
using OptunaSharp.Internal;

namespace OptunaSharp.Tests;

/// <summary>
///     Unit coverage for the <see cref="LinAlg" /> dense linear-algebra primitives (PORTING_PLAN
///     Phase 2.15): <c>Solve</c>/<c>MatrixRank</c>/<c>Cholesky</c>/<c>Eigh</c> (delegated to NumSharp's
///     <c>np.linalg</c> over the OpenBLAS LAPACK backend) plus the managed <c>SolveTriangular</c>, for
///     NSGA-III (2.14), CMA-ES (2.16) and the GP stack (3.2). Known small cases pin the exact math;
///     the self-consistency sweep asserts reconstruction / residual / orthonormality to machine
///     precision. (Byte-exact parity vs numpy/LAPACK lives in the interop <c>LinAlgParityTests</c>.)
/// </summary>
[TestClass]
public class LinAlgTests
{
    private const double Eps = 1e-12;

    // ---- Cholesky -----------------------------------------------------------------------------

    [TestMethod]
    public void Cholesky_KnownMatrix()
    {
        // [[4,2],[2,3]] -> L = [[2,0],[1,sqrt(2)]].
        var l = LinAlg.Cholesky(new[,] { { 4.0, 2.0 }, { 2.0, 3.0 } });
        l[0, 0].Should().BeApproximately(2.0, Eps);
        l[0, 1].Should().Be(0.0);
        l[1, 0].Should().BeApproximately(1.0, Eps);
        l[1, 1].Should().BeApproximately(Math.Sqrt(2.0), Eps);
    }

    [TestMethod]
    public void Cholesky_NotPositiveDefinite_Throws()
        // [[1,2],[2,1]] has eigenvalues -1, 3 -> not PD.
        => ((Action)(() => LinAlg.Cholesky(new[,] { { 1.0, 2.0 }, { 2.0, 1.0 } })))
            .Should().Throw<LinAlgError>().WithMessage("*positive definite*");

    // ---- SolveTriangular ----------------------------------------------------------------------

    [TestMethod]
    public void SolveTriangular_Lower_ForwardSubstitution()
    {
        // [[2,0],[1,3]] x = [2,7] -> x = [1,2].
        var x = LinAlg.SolveTriangular(new[,] { { 2.0, 0.0 }, { 1.0, 3.0 } }, new[] { 2.0, 7.0 }, upper: false);
        x[0].Should().BeApproximately(1.0, Eps);
        x[1].Should().BeApproximately(2.0, Eps);
    }

    [TestMethod]
    public void SolveTriangular_Upper_BackSubstitution()
    {
        // [[3,1],[0,2]] x = [5,4] -> x = [1,2].
        var x = LinAlg.SolveTriangular(new[,] { { 3.0, 1.0 }, { 0.0, 2.0 } }, new[] { 5.0, 4.0 }, upper: true);
        x[0].Should().BeApproximately(1.0, Eps);
        x[1].Should().BeApproximately(2.0, Eps);
    }

    [TestMethod]
    public void SolveTriangular_Matrix_Left()
    {
        // L X = B, L=[[2,0],[1,3]], B=[[2,4],[7,9]] -> X columns solve the two systems.
        var l = new[,] { { 2.0, 0.0 }, { 1.0, 3.0 } };
        var b = new[,] { { 2.0, 4.0 }, { 7.0, 9.0 } };
        var x = LinAlg.SolveTriangular(l, b, upper: false, left: true);
        // Column 0: [1,2]; column 1: [2, (9-2)/3].
        x[0, 0].Should().BeApproximately(1.0, Eps);
        x[1, 0].Should().BeApproximately(2.0, Eps);
        x[0, 1].Should().BeApproximately(2.0, Eps);
        x[1, 1].Should().BeApproximately(7.0 / 3.0, Eps);
    }

    [TestMethod]
    public void SolveTriangular_Matrix_Right_ResidualIsZero()
    {
        // X L = B (right solve): verify X L == B.
        var l = new[,] { { 2.0, 0.0 }, { 1.0, 3.0 } };
        var b = new[,] { { 4.0, 9.0 }, { 6.0, 12.0 } };
        var x = LinAlg.SolveTriangular(l, b, upper: false, left: false);
        for (int r = 0; r < 2; r++)
            for (int c = 0; c < 2; c++)
            {
                double v = x[r, 0] * l[0, c] + x[r, 1] * l[1, c];
                v.Should().BeApproximately(b[r, c], Eps);
            }
    }

    // ---- Eigh ---------------------------------------------------------------------------------

    [TestMethod]
    public void Eigh_KnownEigenvalues_Ascending()
    {
        // [[2,1],[1,2]] -> eigenvalues 1, 3 (ascending).
        var (w, v) = LinAlg.Eigh(new[,] { { 2.0, 1.0 }, { 1.0, 2.0 } });
        w[0].Should().BeApproximately(1.0, Eps);
        w[1].Should().BeApproximately(3.0, Eps);
        // V diag(w) Vᵀ reconstructs A.
        for (int i = 0; i < 2; i++)
            for (int j = 0; j < 2; j++)
            {
                double recon = v[i, 0] * w[0] * v[j, 0] + v[i, 1] * w[1] * v[j, 1];
                recon.Should().BeApproximately(i == j ? 2.0 : 1.0, Eps);
            }
    }

    [TestMethod]
    public void Eigh_DiagonalMatrix_SortsAscending()
    {
        var (w, _) = LinAlg.Eigh(new[,] { { 5.0, 0.0 }, { 0.0, 1.0 } });
        w[0].Should().BeApproximately(1.0, Eps);
        w[1].Should().BeApproximately(5.0, Eps);
    }

    [TestMethod]
    public void Eigh_OneByOne()
    {
        var (w, v) = LinAlg.Eigh(new[,] { { 3.5 } });
        w[0].Should().Be(3.5);
        v[0, 0].Should().Be(1.0);
    }

    // ---- graduated Solve / MatrixRank ---------------------------------------------------------

    [TestMethod]
    public void Solve_KnownSystem()
    {
        // [[3,1],[1,3]] x = [1,1] -> x = [0.25, 0.25].
        var x = LinAlg.Solve(new[,] { { 3.0, 1.0 }, { 1.0, 3.0 } }, new[] { 1.0, 1.0 });
        x[0].Should().BeApproximately(0.25, Eps);
        x[1].Should().BeApproximately(0.25, Eps);
    }

    [TestMethod]
    public void MatrixRank_FullAndDeficient()
    {
        LinAlg.MatrixRank(new[,] { { 3.0, 0.0 }, { 0.0, 3.0 } }).Should().Be(2);
        LinAlg.MatrixRank(new[,] { { 1.0, 2.0 }, { 2.0, 4.0 } }).Should().Be(1);  // rank-deficient
    }

    // ---- self-consistency sweeps (machine precision) ------------------------------------------

    [TestMethod]
    public void SelfConsistency_Cholesky_SolveTriangular_Eigh()
    {
        var rng = new Random(1);
        double maxChol = 0, maxTri = 0, maxRecon = 0, maxOrtho = 0;
        int ascFail = 0;

        for (int trial = 0; trial < 2000; trial++)
        {
            int n = 2 + rng.Next(5);  // 2..6
            var sym = new double[n, n];
            for (int i = 0; i < n; i++)
                for (int j = i; j < n; j++) { double val = rng.NextDouble() * 4 - 2; sym[i, j] = val; sym[j, i] = val; }

            // SPD = sym·symᵀ + n·I.
            var spd = new double[n, n];
            for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++)
                {
                    double s = 0; for (int k = 0; k < n; k++) s += sym[i, k] * sym[j, k];
                    spd[i, j] = s + (i == j ? n : 0.0);
                }

            var l = LinAlg.Cholesky(spd);
            for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++)
                {
                    double s = 0; for (int k = 0; k <= Math.Min(i, j); k++) s += l[i, k] * l[j, k];
                    maxChol = Math.Max(maxChol, Math.Abs(s - spd[i, j]));
                }

            var b = new double[n]; for (int i = 0; i < n; i++) b[i] = rng.NextDouble() * 2 - 1;
            var x = LinAlg.SolveTriangular(l, b, upper: false);
            for (int i = 0; i < n; i++)
            {
                double s = 0; for (int j = 0; j <= i; j++) s += l[i, j] * x[j];
                maxTri = Math.Max(maxTri, Math.Abs(s - b[i]));
            }

            var (w, v) = LinAlg.Eigh(sym);
            for (int i = 1; i < n; i++) if (w[i] < w[i - 1]) ascFail++;
            for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++)
                {
                    double recon = 0; for (int k = 0; k < n; k++) recon += v[i, k] * w[k] * v[j, k];
                    maxRecon = Math.Max(maxRecon, Math.Abs(recon - sym[i, j]));
                    double dot = 0; for (int k = 0; k < n; k++) dot += v[k, i] * v[k, j];
                    maxOrtho = Math.Max(maxOrtho, Math.Abs(dot - (i == j ? 1.0 : 0.0)));
                }
        }

        maxChol.Should().BeLessThan(1e-11);
        maxTri.Should().BeLessThan(1e-11);
        maxRecon.Should().BeLessThan(1e-11);
        maxOrtho.Should().BeLessThan(1e-11);
        ascFail.Should().Be(0);
    }

    // ---- SolveTriangular with a block of right-hand sides (plan 15) --------------------------------

    /// <summary>
    ///     Plan 15: the matrix <c>SolveTriangular</c> sweeps all right-hand sides at once, vectorized across them, instead
    ///     of solving them one by one. Every combination of triangle (lower, upper), side (left, right), system size (1 to
    ///     17) and number of right-hand sides (1 to 33, below, at and past whole vectors) against the replaced
    ///     one-at-a-time code, written out below: every element has the same bits. The coefficient matrices have random
    ///     off-diagonal values in the other triangle too, which must be ignored as before.
    /// </summary>
    [TestMethod]
    public void SolveTriangular_Matrix_AllRightHandSidesAtOnce_MatchesTheOneAtATimeCodeToTheBit()
    {
        var rng = new Random(15);
        foreach (int n in new[] { 1, 2, 3, 5, 8, 17 })
        {
            foreach (int sides in new[] { 1, 2, 3, 4, 5, 7, 8, 13, 33 })
            {
                foreach (bool upper in new[] { false, true })
                {
                    var a = new double[n, n];
                    for (int i = 0; i < n; i++)
                        for (int j = 0; j < n; j++)
                            a[i, j] = i == j ? 0.5 + rng.NextDouble() * 3 : rng.NextDouble() * 4 - 2;
                    foreach (bool left in new[] { true, false })
                    {
                        var b = left ? new double[n, sides] : new double[sides, n];
                        for (int i = 0; i < b.GetLength(0); i++)
                            for (int j = 0; j < b.GetLength(1); j++)
                                b[i, j] = rng.NextDouble() * 20 - 10;

                        double[,] actual = LinAlg.SolveTriangular(a, b, upper, left);
                        double[,] expected = OneAtATimeSolveTriangular(a, b, upper, left);
                        actual.GetLength(0).Should().Be(expected.GetLength(0));
                        actual.GetLength(1).Should().Be(expected.GetLength(1));
                        for (int i = 0; i < expected.GetLength(0); i++)
                            for (int j = 0; j < expected.GetLength(1); j++)
                                BitConverter.DoubleToInt64Bits(actual[i, j]).Should().Be(BitConverter.DoubleToInt64Bits(expected[i, j]),
                                    $"n={n}, sides={sides}, upper={upper}, left={left}, element [{i},{j}]");
                    }
                }
            }
        }
    }

    /// <summary>
    ///     Plan 15: <c>CholeskySolve</c> runs both triangular solves of a Cholesky solve on one block. On both sides, for
    ///     sizes 1 to 17 and 1 to 33 right-hand sides, it equals the two solves the GP stack spelled out (L then a
    ///     materialized Lᵀ on the left; Lᵀ then L on the right), each computed with the replaced one-at-a-time code: same
    ///     bits. The factor's upper triangle holds noise, which must be ignored as before.
    /// </summary>
    [TestMethod]
    public void CholeskySolve_FusedOnOneBlock_MatchesTheTwoSpelledOutSolvesToTheBit()
    {
        var rng = new Random(16);
        foreach (int n in new[] { 1, 2, 3, 5, 8, 17 })
        {
            foreach (int sides in new[] { 1, 3, 4, 5, 9, 33 })
            {
                var l = new double[n, n];
                for (int i = 0; i < n; i++)
                    for (int j = 0; j < n; j++)
                        l[i, j] = i == j ? 0.5 + rng.NextDouble() * 3 : rng.NextDouble() * 4 - 2;
                var lt = new double[n, n];
                for (int i = 0; i < n; i++)
                    for (int j = 0; j < n; j++)
                        lt[i, j] = l[j, i];

                var bLeft = new double[n, sides];
                for (int i = 0; i < n; i++)
                    for (int c = 0; c < sides; c++)
                        bLeft[i, c] = rng.NextDouble() * 20 - 10;
                double[,] expectedLeft = OneAtATimeSolveTriangular(lt, OneAtATimeSolveTriangular(l, bLeft, upper: false, left: true), upper: true, left: true);
                AssertSameBits(LinAlg.CholeskySolve(l, bLeft, left: true), expectedLeft, $"left, n={n}, sides={sides}");

                var bRight = new double[sides, n];
                for (int r = 0; r < sides; r++)
                    for (int j = 0; j < n; j++)
                        bRight[r, j] = rng.NextDouble() * 20 - 10;
                double[,] expectedRight = OneAtATimeSolveTriangular(l, OneAtATimeSolveTriangular(lt, bRight, upper: true, left: false), upper: false, left: false);
                AssertSameBits(LinAlg.CholeskySolve(l, bRight, left: false), expectedRight, $"right, n={n}, sides={sides}");
            }
        }
    }

    /// <summary>
    ///     Plan 15: the vector <c>SolveTriangular(a, b, upper, transposeA: true)</c> reads <c>a</c> transposed in place;
    ///     it equals solving with a materialized transpose and the opposite triangle, for both triangles and sizes 1 to
    ///     40: same bits.
    /// </summary>
    [TestMethod]
    public void SolveTriangular_Vector_TransposedInPlace_MatchesTheMaterializedTransposeToTheBit()
    {
        var rng = new Random(17);
        for (int n = 1; n <= 40; n++)
        {
            foreach (bool upper in new[] { false, true })
            {
                var a = new double[n, n];
                for (int i = 0; i < n; i++)
                    for (int j = 0; j < n; j++)
                        a[i, j] = i == j ? 0.5 + rng.NextDouble() * 3 : rng.NextDouble() * 4 - 2;
                var at = new double[n, n];
                for (int i = 0; i < n; i++)
                    for (int j = 0; j < n; j++)
                        at[i, j] = a[j, i];
                var b = new double[n];
                for (int i = 0; i < n; i++)
                    b[i] = rng.NextDouble() * 20 - 10;

                double[] actual = LinAlg.SolveTriangular(a, b, upper, transposeA: true);
                double[] expected = OneTriSolve(at, b, !upper, transpose: false);
                for (int i = 0; i < n; i++)
                    BitConverter.DoubleToInt64Bits(actual[i]).Should().Be(BitConverter.DoubleToInt64Bits(expected[i]),
                        $"n={n}, upper={upper}, element {i}");
            }
        }
    }

    /// <summary>Asserts two matrices have the same shape and the same bits in every element.</summary>
    /// <param name="actual">The matrix under test.</param>
    /// <param name="expected">The reference matrix.</param>
    /// <param name="what">Names the case in failure messages.</param>
    private static void AssertSameBits(double[,] actual, double[,] expected, string what)
    {
        actual.GetLength(0).Should().Be(expected.GetLength(0), what);
        actual.GetLength(1).Should().Be(expected.GetLength(1), what);
        for (int i = 0; i < expected.GetLength(0); i++)
            for (int j = 0; j < expected.GetLength(1); j++)
                BitConverter.DoubleToInt64Bits(actual[i, j]).Should().Be(BitConverter.DoubleToInt64Bits(expected[i, j]),
                    $"{what}, element [{i},{j}]");
    }

    /// <summary>
    ///     The matrix <c>SolveTriangular</c> before plan 15: each right-hand side copied out, solved by substitution
    ///     through an accessor that reads the coefficient matrix transposed for a right solve, and copied back.
    /// </summary>
    /// <param name="a">The triangular coefficient matrix.</param>
    /// <param name="b">The right-hand sides: columns for a left solve, rows for a right solve.</param>
    /// <param name="upper">Whether <paramref name="a" /> is upper triangular.</param>
    /// <param name="left">Whether to solve <c>a X = b</c> (else <c>X a = b</c>).</param>
    /// <returns>The solution.</returns>
    private static double[,] OneAtATimeSolveTriangular(double[,] a, double[,] b, bool upper, bool left)
    {
        int rows = b.GetLength(0), cols = b.GetLength(1), n = a.GetLength(0);
        var result = new double[rows, cols];
        if (left)
        {
            var column = new double[n];
            for (int c = 0; c < cols; c++)
            {
                for (int i = 0; i < n; i++)
                    column[i] = b[i, c];
                double[] solved = OneTriSolve(a, column, upper, transpose: false);
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
                double[] solved = OneTriSolve(a, rowVec, upper, transpose: true);
                for (int j = 0; j < n; j++)
                    result[r, j] = solved[j];
            }
        }
        return result;
    }

    /// <summary>The replaced per-right-hand-side substitution, <c>op(a) x = b</c>.</summary>
    /// <param name="a">The triangular coefficient matrix in its stored orientation.</param>
    /// <param name="b">The right-hand side.</param>
    /// <param name="upper">Whether <paramref name="a" /> is upper triangular as stored.</param>
    /// <param name="transpose">Whether to solve with the transpose of <paramref name="a" />.</param>
    /// <returns>The solution vector.</returns>
    private static double[] OneTriSolve(double[,] a, double[] b, bool upper, bool transpose)
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
}
