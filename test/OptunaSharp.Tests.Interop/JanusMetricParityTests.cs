using System;
using System.Globalization;
using System.Text;
using AwesomeAssertions;
using OptunaSharp.Samplers.Janus;
using Python.Runtime;

namespace OptunaSharp.Tests.Interop;

/// <summary>
///     Byte-to-byte parity of JANUS's residual-coordinate curvature fits (<see cref="JanusMetric" />)
///     against the authors' <b>live reference</b> <c>janus_optuna.janus.pseudo_metric</c>
///     (arXiv:2608.22862; https://github.com/hongyuanyu/JANUS, Apache-2.0), driven in-process via
///     pythonnet.
///
///     <para><b>Why only the metric fits.</b> The reference's single-objective sampler runs a
///     <i>torch/evotorch</i> CMA-ES host with a numpy-PCG64 CMIX stream, so an end-to-end study can
///     never be bit-identical to OptunaSharp's MT19937 <c>Cma</c> port (the same reason PyCma and
///     BoTorch ship as bridges, not native ports). JANUS's <b>novel math</b> — the scalar
///     linear/quadratic residual-coordinate metric <c>G ≈ JᵀJ</c> — is pure numpy
///     (<c>np.linalg.solve</c>/<c>eigh</c>/<c>outer</c>), which OptunaSharp routes through the SAME
///     OpenBLAS/LAPACK. That surface IS byte-comparable, and this is where JANUS's contribution lives.
///     The reference's metric functions import torch-free (their torch import is <c>try_import</c>-guarded),
///     so no CUDA/torch is needed here.</para>
///
///     <para><b>Tolerance.</b> Agreement is to ~1 ULP (observed ≤ 1e-14 on O(1–100) scales) — the
///     float64-determinism limit, since numpy's 1-D <c>mean</c> uses pairwise summation while the port
///     sums sequentially. This is the same "value-parity ~1 ULP" tier the native fANOVA/SHAP ports are
///     held to; the gate at 1e-11 sits ~1000× above that floor and ~1e5× below any real algorithmic
///     divergence. Host-pinned to x64 (OpenBLAS DYNAMIC_ARCH + summation order), like
///     <c>LinAlgParityTests</c>.</para>
///
///     <para>Self-skips (Inconclusive) when Python/Optuna or the JANUS reference clone is unavailable.
///     The reference is expected at <c>K:/source/JANUS</c> (override with <c>JANUS_REF_PATH</c>).</para>
/// </summary>
[TestClass]
public class JanusMetricParityTests : OptunaInteropTestBase
{
    private const double Tol = 1e-11;

    private static readonly string[] Modes =
        { "linear", "quadratic", "diag_quadratic", "pca_quadratic", "diag_pca_quadratic" };

    /// <summary>Add the reference clone to sys.path and import it; Inconclusive if it isn't present.</summary>
    private void ImportJanusReferenceOrInconclusive()
    {
        string refPath = Environment.GetEnvironmentVariable("JANUS_REF_PATH") ?? "K:/source/JANUS";
        try
        {
            PyExec(
                "import sys\n" +
                $"_p = r'{refPath}'\n" +
                "if _p not in sys.path: sys.path.insert(0, _p)\n" +
                "import numpy as np\n" +
                "import janus_optuna.janus as _janus\n");
        }
        catch (PythonException ex)
        {
            Assert.Inconclusive(
                $"JANUS reference not importable from '{refPath}' (set JANUS_REF_PATH). {ex.Message}");
        }
    }

    // ---- byte-parity of pseudo_metric across modes / objectives / targets ------------------------

    [TestMethod]
    [TestCategory("RequiresPython")]
    public void PseudoMetric_ByteParity_AcrossModesObjectivesTargets()
    {
        PythonSession.EnsureBitExactHostOrInconclusive();
        ImportJanusReferenceOrInconclusive();

        var rng = new Random(20260910);
        int scenarios = 0, compared = 0, bothNull = 0;
        foreach (int d in new[] { 3, 5 })
        foreach (int n in new[] { 60, 140 })
        foreach (string objective in new[] { "linear", "quadratic", "rosenbrock" })
        foreach (string target in new[] { "raw", "log" })
        foreach (string mode in Modes)
        {
            var (x, f) = MakeData(rng, n, d, objective);
            CheckOne(x, f, mode, target, 1.0, ref compared, ref bothNull);
            scenarios++;
        }

        // Sanity: the sweep actually exercised real comparisons (not all self-skipped as null).
        scenarios.Should().Be(2 * 2 * 3 * 2 * 5);
        compared.Should().BeGreaterThan(40, "most scenarios yield a fittable metric on both sides");
    }

    // ---- null-agreement on degenerate inputs -----------------------------------------------------

    [TestMethod]
    [TestCategory("RequiresPython")]
    public void PseudoMetric_NullAgreement_OnDegenerateInputs()
    {
        PythonSession.EnsureBitExactHostOrInconclusive();
        ImportJanusReferenceOrInconclusive();

        var rng = new Random(11);
        int compared = 0, bothNull = 0;

        // Too few rows (< 4): both must return null/None.
        var (x3, f3) = MakeData(rng, 3, 4, "quadratic");
        CheckOne(x3, f3, "linear", "raw", 1.0, ref compared, ref bothNull);
        CheckOne(x3, f3, "quadratic", "raw", 1.0, ref compared, ref bothNull);

        // Quadratic mode with too few rows to fit the full quadratic design (falls back to linear on
        // both sides — a non-null agreement).
        var (x8, f8) = MakeData(rng, 8, 6, "quadratic");
        CheckOne(x8, f8, "quadratic", "raw", 1.0, ref compared, ref bothNull);

        bothNull.Should().BeGreaterThan(0, "the < 4-row cases return null on both sides");
    }

    // ---- component-based residual Jacobian fit (the paper's primary / recommended mode) ----------

    [TestMethod]
    [TestCategory("RequiresPython")]
    public void ComponentJacobian_ByteParity_MatchesReferenceAssembleGn()
    {
        PythonSession.EnsureBitExactHostOrInconclusive();
        ImportJanusReferenceOrInconclusive();

        // The reference's component fit is inline in _assemble_gn (janus.py) and is unreachable live
        // without the torch/evotorch host, so we byte-compare against its VERBATIM numpy (cited below).
        // Our FitComponentJacobian is the single source of truth shared with the live sampler path.
        var rng = new Random(4242);
        int compared = 0;
        foreach (int d in new[] { 3, 6 })
        foreach (int n in new[] { 50, 130 })
        foreach (int m in new[] { d, 2 })   // square (m=d) and rectangular (m=2) residual maps
        foreach (double ridge in new[] { 1.0, 0.1 })
        {
            var (x, r) = MakeComponentData(rng, n, d, m);
            var fit = JanusOptimizer.FitComponentJacobian(x, r, ridge);
            fit.Should().NotBeNull($"the ridge solve succeeds on well-conditioned data (n={n} d={d} m={m})");
            var (j, gnJtJ) = fit!.Value;

            PyExec(
                $"X = np.array({Lit(x)})\n" +
                $"R = np.array({Lit(r)})\n" +
                $"J_cs = np.array({Lit(j)})\n" +
                $"G_cs = np.array({Lit(gnJtJ)})\n" +
                $"ridge = {ridge.ToString("R", CultureInfo.InvariantCulture)}\n" +
                "d = X.shape[1]\n" +
                // verbatim _assemble_gn component lines (janus.py):
                "mu = X.mean(0)\n" +
                "sd = X.std(0) + 1e-9\n" +
                "Z = (X - mu) / sd\n" +
                "S = np.sqrt(np.maximum(R, 0.0))\n" +
                "AinvZt = np.linalg.solve(Z.T @ Z + ridge * np.eye(d), Z.T)\n" +
                "Sc = S - S.mean(0)\n" +
                "W = (AinvZt @ Sc).T\n" +
                "J = W / sd[None, :]\n" +
                "gn_JtJ = J.T @ J\n" +
                "dJ = float(np.max(np.abs(J - J_cs)))\n" +
                "dG = float(np.max(np.abs(gn_JtJ - G_cs)))\n" +
                "scaleG = float(np.max(np.abs(gn_JtJ)))\n" +
                "scaleJ = float(np.max(np.abs(J)))\n");

            double dj = PyFloat("dJ"), dg = PyFloat("dG");
            double sj = PyFloat("scaleJ"), sg = PyFloat("scaleG");
            dj.Should().BeLessThan(Tol * (1.0 + sj), $"J byte-parity (n={n} d={d} m={m} ridge={ridge})");
            dg.Should().BeLessThan(Tol * (1.0 + sg), $"JᵀJ byte-parity (n={n} d={d} m={m} ridge={ridge})");
            compared++;
        }

        compared.Should().Be(2 * 2 * 2 * 2);
    }

    private static (double[,] X, double[,] R) MakeComponentData(Random rng, int n, int d, int m)
    {
        var x = new double[n, d];
        var r = new double[n, m];
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < d; j++) x[i, j] = rng.NextDouble() * 4 - 2;
            // nonnegative per-component residuals (e.g. squared shifted coordinates), as JANUS expects.
            for (int c = 0; c < m; c++)
            {
                double v = x[i, c % d] - 0.3 * (c + 1);
                r[i, c] = v * v + 0.05 * rng.NextDouble();
            }
        }
        return (x, r);
    }

    // ---- one scenario: compute in C#, compute the reference in Python, compare -------------------

    private void CheckOne(double[,] x, double[] f, string mode, string target, double ridge,
        ref int compared, ref int bothNull)
    {
        JanusMetricResult? m = JanusMetric.Compute(x, f, mode, ridge, "auto", target);

        PyExec(
            $"X = np.array({Lit(x)})\n" +
            $"f = np.array({Lit(f)})\n" +
            $"r = _janus.pseudo_metric(X, f, mode='{mode}', ridge={ridge.ToString("R", CultureInfo.InvariantCulture)}, pca_k='auto', target='{target}')\n" +
            "py_none = r is None\n");

        bool pyNone = PyBool("py_none");
        string ctx = $"mode={mode} target={target} shape={x.GetLength(0)}x{x.GetLength(1)}";

        if (m is null)
        {
            pyNone.Should().BeTrue($"C# metric is null ⇒ reference must be None ({ctx})");
            if (pyNone) bothNull++;
            return;
        }

        pyNone.Should().BeFalse($"C# produced a metric ⇒ reference must too ({ctx})");
        PyExec(
            $"JtJ = np.array({Lit(m.JtJ)})\n" +
            "diff = float(np.max(np.abs(r['JtJ'] - JtJ)))\n" +
            "scale = float(np.max(np.abs(r['JtJ'])))\n" +
            "ref_mode = str(r['mode'])\n");

        PyStr("ref_mode").Should().Be(m.Mode, $"the fit mode must agree ({ctx})");
        double diff = PyFloat("diff");
        double scale = PyFloat("scale");
        diff.Should().BeLessThan(Tol * (1.0 + scale),
            $"JtJ byte-parity to ~1 ULP ({ctx}, scale={scale:g3}, diff={diff:g3})");
        compared++;
    }

    // ---- deterministic data (passed identically to both sides as float literals) -----------------

    private static (double[,] X, double[] F) MakeData(Random rng, int n, int d, string objective)
    {
        var x = new double[n, d];
        var f = new double[n];
        for (int i = 0; i < n; i++)
        {
            var row = new double[d];
            for (int j = 0; j < d; j++) { row[j] = rng.NextDouble() * 4 - 2; x[i, j] = row[j]; }
            f[i] = objective switch
            {
                "linear" => Linear(row),
                "quadratic" => Quadratic(row),
                _ => Rosenbrock(row),
            };
        }
        return (x, f);
    }

    private static double Linear(double[] x)
    {
        double s = 0;
        for (int j = 0; j < x.Length; j++) s += (j + 1) * 0.5 * x[j] - 0.3 * x[j];
        return s + 1.7;
    }

    private static double Quadratic(double[] x)
    {
        double s = 0;
        for (int j = 0; j < x.Length; j++) s += Math.Pow(10.0, j / (double)Math.Max(1, x.Length - 1)) * x[j] * x[j];
        return s;
    }

    private static double Rosenbrock(double[] x)
    {
        double s = 0;
        for (int j = 0; j < x.Length - 1; j++)
        {
            double a = x[j + 1] - x[j] * x[j], b = 1 - x[j];
            s += 100 * a * a + b * b;
        }
        return s;
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
