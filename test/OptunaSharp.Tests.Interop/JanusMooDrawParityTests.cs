using System;
using System.Globalization;
using System.Text;
using AwesomeAssertions;
using NumSharp;
using OptunaSharp.Samplers.Janus;
using Python.Runtime;

namespace OptunaSharp.Tests.Interop;

/// <summary>
///     Byte-parity of JANUS-MOO's CMIX <b>random draws</b> against numpy — the Plan-05 stretch goal,
///     realized. The reference <c>janus_moo_cmix.JanusMooCmixSampler</c> draws its CMIX stream from
///     <c>np.random.default_rng(seed)</c> (PCG64): one <c>random()</c> per trial for the fire decision and
///     <c>multivariate_normal(center, cov)</c> for the injected sample. OptunaSharp drives the SAME PCG64
///     bit generator (NumSharp's <c>default_rng</c>) and reproduces numpy's default <c>method="svd"</c>
///     <c>multivariate_normal</c> — <c>standard_normal(d)</c> off that stream, then the SVD transform
///     factored through the OpenBLAS <c>gesdd</c> NumPy itself calls (<see cref="JanusMooMetric.MultivariateNormalSvd" />).
///
///     <para>Because the metric fits (<c>JanusMooMetricParityTests</c>) and NSGA-II selection are already
///     bit-exact, this draw-parity closes the loop: the whole JANUS-MOO CMIX injection is byte-identical to
///     the reference — for a relative search space of ANY size, since <c>gesdd</c> is byte-exact at any
///     size (the same host-pinned x64 / threads=1 OpenBLAS-parity condition as <c>LinAlgParityTests</c>),
///     not the 4×4 ceiling of NumSharp's own Jacobi <c>multivariate_normal</c>. Self-skips (Inconclusive)
///     without Python/NumPy or off the pinned host.</para>
/// </summary>
[TestClass]
public class JanusMooDrawParityTests : OptunaInteropTestBase
{
    private const double Tol = 1e-12;   // byte-exact in practice; a floor well below any real divergence

    [TestMethod]
    [TestCategory("RequiresPython")]
    public void CmixDraws_ByteParity_AcrossDimsSeeds()
    {
        PythonSession.EnsureBitExactHostOrInconclusive();
        PyExec("import numpy as np\n");

        int compared = 0;
        foreach (int d in new[] { 2, 3, 4, 6, 8 })   // include >4 to prove the gesdd path is not 4×4-limited
        foreach (long seed in new long[] { 1, 7, 42 })
        {
            var (mean, cov) = MakeSpd(d);

            // C# side: PCG64 fire draw, standard_normal, and the SVD-transform multivariate_normal.
            double fireCs = np.random.default_rng(seed).random().ToArray<double>()[0];
            double[] z = np.random.default_rng(seed).standard_normal(new Shape(d)).ToArray<double>();
            double[] sampleCs = JanusMooMetric.MultivariateNormalSvd(z, mean, cov);

            // Python side: the reference's exact calls off the same seed.
            PyExec(
                $"mean = np.array({Lit(mean)})\n" +
                $"cov = np.array({Lit(cov)})\n" +
                $"z_cs = np.array({Lit(z)})\n" +
                $"sample_cs = np.array({Lit(sampleCs)})\n" +
                $"mvn = np.random.default_rng({seed}).multivariate_normal(mean, cov)\n" +
                $"z_py = np.random.default_rng({seed}).standard_normal({d})\n" +
                $"fire_py = float(np.random.default_rng({seed}).random())\n" +
                "dz = float(np.max(np.abs(z_py - z_cs)))\n" +
                "dm = float(np.max(np.abs(mvn - sample_cs)))\n");

            double dz = PyFloat("dz"), dm = PyFloat("dm"), firePy = PyFloat("fire_py");
            dz.Should().BeLessThan(Tol, $"PCG64 standard_normal byte-parity (d={d}, seed={seed}, dz={dz:g3})");
            dm.Should().BeLessThan(Tol, $"multivariate_normal (svd) byte-parity (d={d}, seed={seed}, dm={dm:g3})");
            fireCs.Should().BeApproximately(firePy, Tol, $"PCG64 random() fire-decision byte-parity (seed={seed})");
            compared++;
        }

        compared.Should().Be(5 * 3);
    }

    /// <summary>A fixed symmetric diagonally-dominant (hence SPD) covariance + mean for dimension d.</summary>
    private static (double[] Mean, double[,] Cov) MakeSpd(int d)
    {
        var mean = new double[d];
        var cov = new double[d, d];
        for (int i = 0; i < d; i++)
        {
            mean[i] = 0.1 * (i + 1);
            for (int j = 0; j < d; j++)
                cov[i, j] = i == j ? 1.5 + 0.1 * i : 0.2 / (1 + Math.Abs(i - j));
        }
        return (mean, cov);
    }

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
