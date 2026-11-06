using System;
using System.Globalization;
using System.Text;
using AwesomeAssertions;
using OptunaSharp.Samplers.Janus;
using Python.Runtime;

namespace OptunaSharp.Tests.Interop;

/// <summary>
///     Byte-to-byte parity of the multi-objective JANUS CMIX curvature fits
///     (<see cref="JanusMooMetric.FitComponentMetric" /> / <see cref="JanusMooMetric.FitDiagQuadraticMetric" />)
///     against the authors' <b>live reference</b> module <c>janus_optuna.janus_moo_cmix</c>
///     (arXiv:2608.22862; https://github.com/hongyuanyu/JANUS, Apache-2.0), driven in-process via pythonnet.
///
///     <para><b>Why the MOO file is fully byte-comparable.</b> Unlike the single-objective sampler,
///     <c>janus_moo_cmix.py</c> is <i>torch-free</i> — it subclasses Optuna's NSGA-II and its metric fits are
///     pure numpy (<c>np.linalg.solve</c>, standardize, <c>@</c>), which OptunaSharp routes through the SAME
///     OpenBLAS/LAPACK. So the module imports and runs without any torch, and these two module-level
///     functions — the geometric heart of JANUS-MOO — are directly comparable to their C# port.</para>
///
///     <para><b>Tolerance.</b> ~1 ULP (gated at 1e-11·(1+scale)), the same float64-determinism tier as
///     <c>JanusMetricParityTests</c> / the fANOVA-SHAP ports; the only residual is numpy's pairwise vs the
///     port's sequential reductions in <c>std</c>/<c>mean</c>. Host-pinned to x64. Self-skips (Inconclusive)
///     when Python/Optuna or the JANUS reference clone is unavailable (a clone of
///     <c>https://github.com/hongyuanyu/JANUS</c> in a <c>JANUS</c> folder beside the repository; override with
///     <c>JANUS_REF_PATH</c>, see <see cref="JanusReference.Folder" />).</para>
/// </summary>
[TestClass]
public class JanusMooMetricParityTests : OptunaInteropTestBase
{
    private const double Tol = 1e-11;

    private void ImportMooReferenceOrInconclusive()
    {
        string refPath = JanusReference.Folder();
        try
        {
            PyExec(
                "import sys\n" +
                $"_p = r'{refPath}'\n" +
                "if _p not in sys.path: sys.path.insert(0, _p)\n" +
                "import numpy as np\n" +
                "import janus_optuna.janus_moo_cmix as _moo\n");
        }
        catch (PythonException ex)
        {
            Assert.Inconclusive(
                $"JANUS MOO reference not importable from '{refPath}' (set JANUS_REF_PATH). {ex.Message}");
        }
    }

    // ---- component metric byte-parity ------------------------------------------------------------

    [TestMethod]
    [TestCategory("RequiresPython")]
    public void FitComponentMetric_ByteParity_AcrossShapesRidges()
    {
        PythonSession.EnsureBitExactHostOrInconclusive();
        ImportMooReferenceOrInconclusive();

        var rng = new Random(20260911);
        int compared = 0;
        foreach (int d in new[] { 3, 6 })
        foreach (int n in new[] { 50, 130 })
        foreach (int m in new[] { d, 2 })
        foreach (double ridge in new[] { 1.0, 0.1 })
        {
            var (x, s) = MakeComponentData(rng, n, d, m);
            double[,]? metric = JanusMooMetric.FitComponentMetric(x, s, ridge);
            metric.Should().NotBeNull($"the component fit succeeds on well-conditioned data (n={n} d={d} m={m})");

            PyExec(
                $"X = np.array({Lit(x)})\n" +
                $"S = np.array({Lit(s)})\n" +
                $"M_cs = np.array({Lit(metric!)})\n" +
                $"ridge = {ridge.ToString("R", CultureInfo.InvariantCulture)}\n" +
                "r = _moo._fit_component_metric(X, S, ridge)\n" +
                "py_none = r is None\n");
            PyBool("py_none").Should().BeFalse($"reference produced a metric too (n={n} d={d} m={m})");

            PyExec("diff = float(np.max(np.abs(r - M_cs)))\nscale = float(np.max(np.abs(r)))\n");
            double diff = PyFloat("diff"), scale = PyFloat("scale");
            diff.Should().BeLessThan(Tol * (1.0 + scale),
                $"component metric byte-parity to ~1 ULP (n={n} d={d} m={m} ridge={ridge}, scale={scale:g3}, diff={diff:g3})");
            compared++;
        }

        compared.Should().Be(2 * 2 * 2 * 2);
    }

    // ---- diagonal-quadratic scalar-fallback metric byte-parity -----------------------------------

    [TestMethod]
    [TestCategory("RequiresPython")]
    public void FitDiagQuadraticMetric_ByteParity_AcrossShapesRidges()
    {
        PythonSession.EnsureBitExactHostOrInconclusive();
        ImportMooReferenceOrInconclusive();

        var rng = new Random(7654321);
        int compared = 0;
        foreach (int d in new[] { 3, 5 })
        foreach (int n in new[] { 40, 120 })
        foreach (double ridge in new[] { 1.0, 0.1 })
        {
            var (x, y) = MakeScalarData(rng, n, d);
            double[,]? metric = JanusMooMetric.FitDiagQuadraticMetric(x, y, ridge);
            metric.Should().NotBeNull($"the diag-quadratic fit succeeds (n={n} d={d})");

            PyExec(
                $"X = np.array({Lit(x)})\n" +
                $"y = np.array({Lit(y)})\n" +
                $"M_cs = np.array({Lit(metric!)})\n" +
                $"ridge = {ridge.ToString("R", CultureInfo.InvariantCulture)}\n" +
                "r = _moo._fit_diag_quadratic_metric(X, y, ridge)\n" +
                "py_none = r is None\n");
            PyBool("py_none").Should().BeFalse($"reference produced a metric too (n={n} d={d})");

            PyExec("diff = float(np.max(np.abs(r - M_cs)))\nscale = float(np.max(np.abs(r)))\n");
            double diff = PyFloat("diff"), scale = PyFloat("scale");
            diff.Should().BeLessThan(Tol * (1.0 + scale),
                $"diag-quadratic metric byte-parity to ~1 ULP (n={n} d={d} ridge={ridge}, scale={scale:g3}, diff={diff:g3})");
            compared++;
        }

        compared.Should().Be(2 * 2 * 2);
    }

    // ---- null agreement on too-few-rows ----------------------------------------------------------

    [TestMethod]
    [TestCategory("RequiresPython")]
    public void Metrics_NullAgreement_OnTooFewRows()
    {
        PythonSession.EnsureBitExactHostOrInconclusive();
        ImportMooReferenceOrInconclusive();

        var rng = new Random(99);
        var (x, s) = MakeComponentData(rng, 3, 4, 4);   // < 4 rows ⇒ both null
        var (x2, y) = MakeScalarData(rng, 3, 4);

        JanusMooMetric.FitComponentMetric(x, s, 1.0).Should().BeNull("< 4 rows: C# component fit is null");
        JanusMooMetric.FitDiagQuadraticMetric(x2, y, 1.0).Should().BeNull("< 4 rows: C# diag fit is null");

        PyExec(
            $"X = np.array({Lit(x)})\nS = np.array({Lit(s)})\n" +
            $"X2 = np.array({Lit(x2)})\ny = np.array({Lit(y)})\n" +
            "cn = _moo._fit_component_metric(X, S, 1.0) is None\n" +
            "dn = _moo._fit_diag_quadratic_metric(X2, y, 1.0) is None\n");
        PyBool("cn").Should().BeTrue("< 4 rows: reference component fit is None");
        PyBool("dn").Should().BeTrue("< 4 rows: reference diag fit is None");
    }

    // ---- deterministic data (passed identically to both sides as float literals) -----------------

    private static (double[,] X, double[,] S) MakeComponentData(Random rng, int n, int d, int m)
    {
        var x = new double[n, d];
        var s = new double[n, m];
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < d; j++) x[i, j] = rng.NextDouble() * 4 - 2;
            for (int c = 0; c < m; c++)
            {
                double v = x[i, c % d] - 0.3 * (c + 1);
                s[i, c] = v * v + 0.05 * rng.NextDouble();   // nonnegative residuals, as JANUS expects
            }
        }
        return (x, s);
    }

    private static (double[,] X, double[] Y) MakeScalarData(Random rng, int n, int d)
    {
        var x = new double[n, d];
        var y = new double[n];
        for (int i = 0; i < n; i++)
        {
            double val = 0;
            for (int j = 0; j < d; j++)
            {
                x[i, j] = rng.NextDouble() * 4 - 2;
                val += Math.Pow(10.0, j / (double)Math.Max(1, d - 1)) * x[i, j] * x[i, j];
            }
            y[i] = val;
        }
        return (x, y);
    }

    // ---- python float literals (round-trip exact) ------------------------------------------------

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
}
