using System;
using System.Globalization;
using AwesomeAssertions;
using OptunaSharp.Internal;

namespace OptunaSharp.Tests.Interop;

/// <summary>
///     Live parity of the managed <see cref="LBfgsB" /> optimizer (PORTING_PLAN Phase 3.2 part 2)
///     against scipy's <c>scipy.optimize.minimize(method="l-bfgs-b")</c> — the same routine
///     OptunaSharp ports (scipy's <c>__lbfgsb.c</c>). On identical test functions (bit-identical
///     value+gradient in both languages) the port reproduces scipy's iteration and evaluation counts
///     and converges to the same optimum.
///
///     <para><b>Tolerance-faithful.</b> The internal reductions and system LAPACK scipy links differ
///     from these hand-written kernels by ULPs, so the iterate path tracks scipy's bit-for-bit for
///     the first several iterations then drifts within tolerance — both reaching the same optimum.
///     These tests therefore assert equal iteration/evaluation counts and a close final result.</para>
/// </summary>
[TestClass]
public class LBfgsBParityTests : OptunaInteropTestBase
{
    private const string Helper =
        "import numpy as np\n" +
        "from scipy.optimize import minimize\n" +
        "def _lbfgsb_run(kind, x0, bounds):\n" +
        "    def fg(x):\n" +
        "        if kind == 'quad':\n" +
        "            a = np.array([1.0, -2.0, 3.5]); d = x - a\n" +
        "            return float((d*d).sum()), 2*d\n" +
        "        if kind == 'rosen':\n" +
        "            a = 1 - x[0]; b = x[1] - x[0]*x[0]\n" +
        "            return float(a*a + 100*b*b), np.array([-2*a - 400*x[0]*b, 200*b])\n" +
        "        a = np.array([1.0, 2.0]); d = x - a\n" +
        "        return float((d*d).sum()), 2*d\n" +
        "    res = minimize(fg, np.array(x0, float), jac=True, method='l-bfgs-b', bounds=bounds,\n" +
        "                   options={'gtol':1e-5, 'ftol':1e7*np.finfo(float).eps})\n" +
        "    return res.x.tolist(), float(res.fun), int(res.nit), int(res.nfev)\n";

    [TestMethod]
    [TestCategory("RequiresPython")]
    public void Quadratic_MatchesScipy()
    {
        var target = new[] { 1.0, -2.0, 3.5 };
        AssertParity("quad", new double[] { 0, 0, 0 }, null, x =>
        {
            double f = 0; var g = new double[3];
            for (int i = 0; i < 3; i++) { double d = x[i] - target[i]; f += d * d; g[i] = 2 * d; }
            return (f, g);
        });
    }

    [TestMethod]
    [TestCategory("RequiresPython")]
    public void Rosenbrock_MatchesScipy()
    {
        AssertParity("rosen", new double[] { -1.2, 1.0 }, null, x =>
        {
            double a = 1 - x[0], b = x[1] - x[0] * x[0];
            return (a * a + 100 * b * b, new[] { -2 * a - 400 * x[0] * b, 200 * b });
        });
    }

    [TestMethod]
    [TestCategory("RequiresPython")]
    public void Bounded_MatchesScipy()
    {
        AssertParity("bound", new double[] { 0, 0 }, new[] { (-5.0, 5.0), (-5.0, 1.5) }, x =>
        {
            double f = 0; var g = new double[2];
            double[] a = { 1.0, 2.0 };
            for (int i = 0; i < 2; i++) { double d = x[i] - a[i]; f += d * d; g[i] = 2 * d; }
            return (f, g);
        });
    }

    private void AssertParity(string kind, double[] x0, (double, double)[]? bounds,
        Func<double[], (double, double[])> fg)
    {
        PythonSession.EnsureOrInconclusive();
        PyExec(Helper);
        LBfgsB.Result res = LBfgsB.Minimize(fg, x0, bounds);

        string boundsPy = bounds == null
            ? "None"
            : "[" + string.Join(", ", Array.ConvertAll(bounds, b =>
                $"({b.Item1.ToString("R", CultureInfo.InvariantCulture)}, {b.Item2.ToString("R", CultureInfo.InvariantCulture)})")) + "]";
        string x0Py = "[" + string.Join(", ", Array.ConvertAll(x0, v => v.ToString("R", CultureInfo.InvariantCulture))) + "]";

        PyExec($"xopt, fopt, nit, nfev = _lbfgsb_run('{kind}', {x0Py}, {boundsPy})\n" +
               $"xcs = np.array({Lit(res.X)})\n" +
               $"d_x = float(np.max(np.abs(np.array(xopt) - xcs)))\n");

        PyLong("nit").Should().Be(res.NumIterations, $"{kind}: iteration count matches scipy");
        PyLong("nfev").Should().Be(res.NumFuncEvals, $"{kind}: evaluation count matches scipy");
        PyFloat("d_x").Should().BeLessThan(1e-6, $"{kind}: optimum matches scipy");
        PyFloat("abs(fopt - " + res.F.ToString("R", CultureInfo.InvariantCulture) + ")")
            .Should().BeLessThan(1e-8, $"{kind}: optimal value matches scipy");
    }

    private static string Lit(double[] a)
    {
        return "[" + string.Join(", ", Array.ConvertAll(a, v => v.ToString("R", CultureInfo.InvariantCulture))) + "]";
    }
}
