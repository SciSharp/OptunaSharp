using System;
using AwesomeAssertions;
using OptunaSharp.Gp;

namespace OptunaSharp.Tests.Interop;

/// <summary>
///     Parity of the GP QMC normal Sobol' sampler (PORTING_PLAN Phase 3.2 part 3e-3) against
///     <c>optuna/_gp/qmc.py: sample_from_normal_sobol</c>, which draws torch's scrambled Sobol'
///     sequence (<c>torch.quasirandom.SobolEngine(scramble=True, seed)</c>) and maps it through
///     <c>erfinv</c>. <see cref="Qmc.SampleFromNormalSobol" /> reproduces torch's scramble bit-for-bit
///     (classic MT19937 + the unit-diagonal LMS on the scipy-identical unscrambled direction numbers)
///     and applies cephes <c>ndtri</c>; torch's <c>erfinv</c> is machine-accurate too, so the samples
///     agree to ~1e-15 (the tiny residual is the ndtri-vs-erfinv algorithm difference, not the Sobol'
///     bits — those are exact).
/// </summary>
[TestClass]
public class GpNormalSobolParityTests : OptunaInteropTestBase
{
    [TestMethod]
    [TestCategory("RequiresPython")]
    public void SampleFromNormalSobol_MatchesTorch_Sweep()
    {
        PythonSession.EnsureOrInconclusive();
        PyExec(
            "import numpy as np\n" +
            "from optuna._gp.qmc import sample_from_normal_sobol\n");

        var rng = new Random(31337);
        double worst = 0.0;
        for (int trial = 0; trial < 24; trial++)
        {
            int dim = 1 + rng.Next(4);            // 1..4
            int n = 1 << (3 + rng.Next(5));       // 8..128 (power of two)
            int seed = rng.Next(1, 2_000_000_000);

            double[,] cs = Qmc.SampleFromNormalSobol(dim, n, seed);
            PyExec(
                $"cs = np.array({Lit(cs)})\n" +
                $"py = sample_from_normal_sobol(dim={dim}, n_samples={n}, seed={seed}).numpy()\n" +
                "diff = float(np.max(np.abs(py - cs)))\n" +
                "shape_ok = bool(py.shape == cs.shape)\n");
            PyBool("shape_ok").Should().BeTrue($"shape dim={dim} n={n} seed={seed}");
            double d = PyFloat("diff");
            worst = Math.Max(worst, d);
            d.Should().BeLessThan(1e-12, $"normal Sobol' dim={dim} n={n} seed={seed}");
        }
        // The Sobol' bits are exact; the only gap is cephes ndtri vs torch erfinv (~1e-15).
        worst.Should().BeLessThan(1e-12);
    }

    private static string Lit(double[,] a)
    {
        int rows = a.GetLength(0), cols = a.GetLength(1);
        var sb = new System.Text.StringBuilder("[");
        for (int i = 0; i < rows; i++)
        {
            if (i > 0) sb.Append(", ");
            sb.Append('[');
            for (int j = 0; j < cols; j++)
            {
                if (j > 0) sb.Append(", ");
                sb.Append(a[i, j].ToString("R", System.Globalization.CultureInfo.InvariantCulture));
            }
            sb.Append(']');
        }
        return sb.Append(']').ToString();
    }
}
