using System;
using System.Globalization;
using System.Linq;
using AwesomeAssertions;
using OptunaSharp.Internal;

namespace OptunaSharp.Tests.Interop;

/// <summary>
///     Byte-exact parity of the non-dominated box decomposition (PORTING_PLAN Phase 3.2 part 3e — the
///     2.10 remainder) against Optuna's <c>optuna._hypervolume.get_non_dominated_box_bounds</c>. The
///     decomposition moves only coordinates of the inputs through negation, elementwise <c>max</c>,
///     comparisons, and the lexicographic <c>np.unique</c> sort — no summation or transcendental — so
///     the lower/upper bound arrays (including their <c>±∞</c> entries and their row order) match
///     Optuna bit-for-bit. The box tiling is what the GP log-EHVI acquisition function integrates over.
/// </summary>
[TestClass]
public class GpBoxDecompositionParityTests : OptunaInteropTestBase
{
    [TestMethod]
    [TestCategory("RequiresPython")]
    public void GetNonDominatedBoxBounds_ByteExact_Sweep()
    {
        PythonSession.EnsureOrInconclusive();
        PyExec(
            "import warnings\n" +
            "import numpy as np\n" +
            "from optuna._hypervolume import get_non_dominated_box_bounds\n");

        var rng = new Random(20240904);
        for (int trial = 0; trial < 40; trial++)
        {
            int m = 2 + rng.Next(3);         // 2, 3, 4 objectives.
            int n = m + 1 + rng.Next(8);     // a handful of loss vectors.
            var loss = new double[n][];
            for (int i = 0; i < n; i++)
            {
                loss[i] = new double[m];
                for (int d = 0; d < m; d++)
                    loss[i][d] = Math.Round(rng.NextDouble() * 10.0, 4);  // finite, clean literal.
            }
            // Reference point strictly above every loss vector so the front dominates it.
            var reff = new double[m];
            for (int d = 0; d < m; d++)
            {
                double mx = double.NegativeInfinity;
                for (int i = 0; i < n; i++)
                    mx = Math.Max(mx, loss[i][d]);
                reff[d] = mx + 1.0;
            }

            (double[][] lower, double[][] upper) = Hypervolume.GetNonDominatedBoxBounds(loss, reff);

            PyExec(
                $"loss = np.array({Lit(loss)})\n" +
                $"ref = np.array({Lit(reff)})\n" +
                "with warnings.catch_warnings():\n" +
                "    warnings.simplefilter('ignore')\n" +
                "    pl, pu = get_non_dominated_box_bounds(loss, ref)\n" +
                $"cl = np.array({Lit(lower)}).reshape(-1, {m})\n" +
                $"cu = np.array({Lit(upper)}).reshape(-1, {m})\n" +
                "ok = bool(pl.shape == cl.shape and pu.shape == cu.shape "
                + "and np.array_equal(pl, cl) and np.array_equal(pu, cu))\n" +
                "nboxes = int(pl.shape[0])\n");
            PyLong("nboxes").Should().BeGreaterThan(0, $"box decomposition m={m} n={n} trial={trial}");
            PyBool("ok").Should().BeTrue($"box decomposition m={m} n={n} trial={trial}");
        }
    }

    private static string Lit(double[][] a)
        => "[" + string.Join(", ", a.Select(Lit)) + "]";

    private static string Lit(double[] a)
        => "[" + string.Join(", ", a.Select(LitNum)) + "]";

    private static string LitNum(double x)
        => double.IsPositiveInfinity(x) ? "float('inf')"
            : double.IsNegativeInfinity(x) ? "float('-inf')"
            : double.IsNaN(x) ? "float('nan')"
            : x.ToString("R", CultureInfo.InvariantCulture);
}
