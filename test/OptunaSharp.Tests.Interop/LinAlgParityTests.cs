using System;
using System.Globalization;
using System.Text;
using AwesomeAssertions;
using OptunaSharp.Internal;

namespace OptunaSharp.Tests.Interop;

/// <summary>
///     Parity of the <see cref="LinAlg" /> primitives (PORTING_PLAN Phase 2.15) against numpy / scipy.
///     <c>Cholesky</c>/<c>Eigh</c>/<c>Solve</c>/<c>MatrixRank</c> delegate to NumSharp's
///     <c>np.linalg</c> over the <c>NumSharp.Interop.OpenBLAS</c> LAPACK backend — the same
///     scipy-openblas 0.3.31 build NumPy 2.4.2 bundles — so they are <b>byte-identical to NumPy</b>
///     (values and, for <c>eigh</c>, eigenvectors incl. signs). Those assertions are exact bit
///     compares. Only <c>SolveTriangular</c> is managed (NumSharp exposes no LAPACK <c>dtrtrs</c>),
///     so it is a tolerance <c>allclose</c> vs scipy.
/// </summary>
[TestClass]
public class LinAlgParityTests : OptunaInteropTestBase
{
    private const double Tol = 1e-9;

    // ---- byte-exact: cholesky / eigh / solve via OpenBLAS LAPACK == numpy ----------------------

    [TestMethod]
    [TestCategory("RequiresPython")]
    public void FixedMatrix_ByteExact_CholeskyEighSolve()
    {
        PythonSession.EnsureOrInconclusive();
        PyExec("import numpy as np");

        var a = new[,] { { 4.0, 1.0, 2.0 }, { 1.0, 3.0, 0.0 }, { 2.0, 0.0, 5.0 } };
        var b = new[] { 1.0, 2.0, 3.0 };
        var l = LinAlg.Cholesky(a);
        var (w, v) = LinAlg.Eigh(a);
        var x = LinAlg.Solve(a, b);

        PyExec(
            $"A = np.array({Lit(a)})\n" +
            $"b = np.array({Lit(b)})\n" +
            $"L = np.array({Lit(l)})\n" +
            $"w = np.array({Lit(w)})\n" +
            $"V = np.array({Lit(v)})\n" +
            $"x = np.array({Lit(x)})\n" +
            "wn, Vn = np.linalg.eigh(A)\n" +
            "d_chol = float(np.max(np.abs(np.linalg.cholesky(A) - L)))\n" +
            "d_w = float(np.max(np.abs(wn - w)))\n" +
            "d_v = float(np.max(np.abs(Vn - V)))\n" +   // eigenvectors incl. sign
            "d_solve = float(np.max(np.abs(np.linalg.solve(A, b) - x)))\n");

        PyFloat("d_chol").Should().Be(0.0, "cholesky is byte-identical to numpy");
        PyFloat("d_w").Should().Be(0.0, "eigenvalues are byte-identical to numpy");
        PyFloat("d_v").Should().Be(0.0, "eigenvectors (signs included) are byte-identical to numpy");
        PyFloat("d_solve").Should().Be(0.0, "solve is byte-identical to numpy");
    }

    [TestMethod]
    [TestCategory("RequiresPython")]
    public void Cholesky_ByteExact_Sweep()
    {
        PythonSession.EnsureOrInconclusive();
        PyExec("import numpy as np");
        var rng = new Random(7);
        for (int trial = 0; trial < 30; trial++)
        {
            int n = 2 + rng.Next(6);  // 2..7
            var s = RandomSpd(rng, n);
            var l = LinAlg.Cholesky(s);
            PyExec(
                $"S = np.array({Lit(s)})\n" +
                $"Lcs = np.array({Lit(l)})\n" +
                "diff = float(np.max(np.abs(np.linalg.cholesky(S) - Lcs)))\n");
            PyFloat("diff").Should().Be(0.0, $"cholesky n={n} trial={trial}");
        }
    }

    [TestMethod]
    [TestCategory("RequiresPython")]
    public void Solve_ByteExact_Sweep()
    {
        PythonSession.EnsureOrInconclusive();
        PyExec("import numpy as np");
        var rng = new Random(23);
        for (int trial = 0; trial < 30; trial++)
        {
            int n = 2 + rng.Next(6);
            var a = RandomWellConditioned(rng, n);
            var b = new double[n];
            for (int i = 0; i < n; i++) b[i] = rng.NextDouble() * 2 - 1;
            var x = LinAlg.Solve(a, b);
            PyExec(
                $"A = np.array({Lit(a)})\n" +
                $"b = np.array({Lit(b)})\n" +
                $"xcs = np.array({Lit(x)})\n" +
                "diff = float(np.max(np.abs(np.linalg.solve(A, b) - xcs)))\n");
            PyFloat("diff").Should().Be(0.0, $"solve n={n} trial={trial}");
        }
    }

    [TestMethod]
    [TestCategory("RequiresPython")]
    public void Eigh_ByteExactEigenvalues_MatchNumpy()
    {
        PythonSession.EnsureOrInconclusive();
        PyExec("import numpy as np");
        var rng = new Random(17);
        for (int trial = 0; trial < 30; trial++)
        {
            int n = 2 + rng.Next(6);
            var a = RandomSymmetric(rng, n);
            var (w, v) = LinAlg.Eigh(a);
            PyExec(
                $"A = np.array({Lit(a)})\n" +
                $"w_cs = np.array({Lit(w)})\n" +
                $"V_cs = np.array({Lit(v)})\n" +
                "d_w = float(np.max(np.abs(np.linalg.eigh(A)[0] - w_cs)))\n" +
                "d_recon = float(np.max(np.abs((V_cs * w_cs) @ V_cs.T - A)))\n" +
                "d_ortho = float(np.max(np.abs(V_cs.T @ V_cs - np.eye(A.shape[0]))))\n");
            PyFloat("d_w").Should().Be(0.0, $"eigenvalues n={n} trial={trial}");
            PyFloat("d_recon").Should().BeLessThan(Tol, $"reconstruction n={n} trial={trial}");
            PyFloat("d_ortho").Should().BeLessThan(Tol, $"orthonormality n={n} trial={trial}");
        }
    }

    // ---- managed solve_triangular: tolerance vs scipy ------------------------------------------

    [TestMethod]
    [TestCategory("RequiresPython")]
    public void SolveTriangular_Vector_MatchesScipy()
    {
        PythonSession.EnsureOrInconclusive();
        PyExec("import numpy as np\nimport scipy.linalg as sla");
        var rng = new Random(11);
        for (int trial = 0; trial < 30; trial++)
        {
            int n = 2 + rng.Next(6);
            var l = LinAlg.Cholesky(RandomSpd(rng, n));
            var b = new double[n];
            for (int i = 0; i < n; i++) b[i] = rng.NextDouble() * 2 - 1;

            var xLower = LinAlg.SolveTriangular(l, b, upper: false);
            var lt = Transpose(l);
            var xUpper = LinAlg.SolveTriangular(lt, b, upper: true);

            PyExec(
                $"L = np.array({Lit(l)})\n" +
                $"LT = np.array({Lit(lt)})\n" +
                $"b = np.array({Lit(b)})\n" +
                $"xl = np.array({Lit(xLower)})\n" +
                $"xu = np.array({Lit(xUpper)})\n" +
                "d_lower = float(np.max(np.abs(sla.solve_triangular(L, b, lower=True) - xl)))\n" +
                "d_upper = float(np.max(np.abs(sla.solve_triangular(LT, b, lower=False) - xu)))\n");
            PyFloat("d_lower").Should().BeLessThan(Tol, $"lower n={n} trial={trial}");
            PyFloat("d_upper").Should().BeLessThan(Tol, $"upper n={n} trial={trial}");
        }
    }

    [TestMethod]
    [TestCategory("RequiresPython")]
    public void SolveTriangular_Matrix_LeftAndRight_MatchScipy()
    {
        PythonSession.EnsureOrInconclusive();
        PyExec("import numpy as np\nimport scipy.linalg as sla");
        var rng = new Random(13);
        for (int trial = 0; trial < 25; trial++)
        {
            int n = 2 + rng.Next(5);
            int m = 1 + rng.Next(4);
            var l = LinAlg.Cholesky(RandomSpd(rng, n));

            var bLeft = RandomMatrix(rng, n, m);
            var xLeft = LinAlg.SolveTriangular(l, bLeft, upper: false, left: true);
            var bRight = RandomMatrix(rng, m, n);
            var xRight = LinAlg.SolveTriangular(l, bRight, upper: false, left: false);

            PyExec(
                $"L = np.array({Lit(l)})\n" +
                $"BL = np.array({Lit(bLeft)})\n" +
                $"XL = np.array({Lit(xLeft)})\n" +
                $"BR = np.array({Lit(bRight)})\n" +
                $"XR = np.array({Lit(xRight)})\n" +
                "d_left = float(np.max(np.abs(sla.solve_triangular(L, BL, lower=True) - XL)))\n" +
                "d_right = float(np.max(np.abs(XR @ L - BR)))\n");
            PyFloat("d_left").Should().BeLessThan(Tol, $"left n={n} m={m} trial={trial}");
            PyFloat("d_right").Should().BeLessThan(Tol, $"right n={n} m={m} trial={trial}");
        }
    }

    // ---- literals + random generators ---------------------------------------------------------

    private static string Lit(double[] a)
    {
        var sb = new StringBuilder("[");
        for (int i = 0; i < a.Length; i++)
        {
            if (i > 0) sb.Append(", ");
            sb.Append(a[i].ToString("R", CultureInfo.InvariantCulture));
        }
        return sb.Append(']').ToString();
    }

    private static string Lit(double[,] a)
    {
        int rows = a.GetLength(0), cols = a.GetLength(1);
        var sb = new StringBuilder("[");
        for (int i = 0; i < rows; i++)
        {
            if (i > 0) sb.Append(", ");
            sb.Append('[');
            for (int j = 0; j < cols; j++)
            {
                if (j > 0) sb.Append(", ");
                sb.Append(a[i, j].ToString("R", CultureInfo.InvariantCulture));
            }
            sb.Append(']');
        }
        return sb.Append(']').ToString();
    }

    private static double[,] RandomSymmetric(Random rng, int n)
    {
        var a = new double[n, n];
        for (int i = 0; i < n; i++)
            for (int j = i; j < n; j++)
            {
                double v = rng.NextDouble() * 4 - 2;
                a[i, j] = v;
                a[j, i] = v;
            }
        return a;
    }

    private static double[,] RandomSpd(Random rng, int n)
    {
        var a = RandomSymmetric(rng, n);
        var s = new double[n, n];
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
            {
                double sum = 0;
                for (int k = 0; k < n; k++) sum += a[i, k] * a[j, k];
                s[i, j] = sum + (i == j ? n : 0.0);  // A Aᵀ + n I is SPD
            }
        return s;
    }

    private static double[,] RandomWellConditioned(Random rng, int n)
    {
        var a = RandomMatrix(rng, n, n);
        for (int i = 0; i < n; i++) a[i, i] += n;  // diagonally boost for a non-singular solve
        return a;
    }

    private static double[,] RandomMatrix(Random rng, int rows, int cols)
    {
        var a = new double[rows, cols];
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                a[i, j] = rng.NextDouble() * 2 - 1;
        return a;
    }

    private static double[,] Transpose(double[,] a)
    {
        int rows = a.GetLength(0), cols = a.GetLength(1);
        var t = new double[cols, rows];
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                t[j, i] = a[i, j];
        return t;
    }
}
