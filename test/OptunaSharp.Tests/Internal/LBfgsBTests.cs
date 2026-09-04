using System;
using AwesomeAssertions;
using OptunaSharp.Internal;

namespace OptunaSharp.Tests.Internal;

/// <summary>
///     Unit coverage for the managed <see cref="LBfgsB" /> optimizer (PORTING_PLAN Phase 3.2 part 2 —
///     the port of scipy's <c>__lbfgsb.c</c> that the GP kernel-parameter fit drives). These are
///     oracle-free convergence checks on standard test functions; bit-tolerance parity of the
///     iterate trajectory vs scipy lives in the interop <c>LBfgsBParityTests</c>.
/// </summary>
[TestClass]
public class LBfgsBTests
{
    [TestMethod]
    public void Minimize_SeparableQuadratic_FindsMinimum()
    {
        var target = new[] { 1.0, -2.0, 3.5 };
        LBfgsB.Result res = LBfgsB.Minimize(x =>
        {
            double f = 0;
            var g = new double[3];
            for (int i = 0; i < 3; i++)
            {
                double d = x[i] - target[i];
                f += d * d;
                g[i] = 2 * d;
            }
            return (f, g);
        }, new double[] { 0, 0, 0 });

        res.WarnFlag.Should().Be(0, "the quadratic converges cleanly");
        for (int i = 0; i < 3; i++)
            res.X[i].Should().BeApproximately(target[i], 1e-6);
        res.F.Should().BeLessThan(1e-12);
    }

    [TestMethod]
    public void Minimize_Rosenbrock_FindsGlobalMinimum()
    {
        LBfgsB.Result res = LBfgsB.Minimize(x =>
        {
            double a = 1 - x[0], b = x[1] - x[0] * x[0];
            double f = a * a + 100 * b * b;
            var g = new[] { -2 * a - 400 * x[0] * b, 200 * b };
            return (f, g);
        }, new double[] { -1.2, 1.0 });

        res.WarnFlag.Should().Be(0);
        res.X[0].Should().BeApproximately(1.0, 1e-5);
        res.X[1].Should().BeApproximately(1.0, 1e-5);
    }

    [TestMethod]
    public void Minimize_WithBounds_RespectsAndHitsBound()
    {
        // Unconstrained minimum is (1, 2); the second coordinate is capped at 1.5.
        LBfgsB.Result res = LBfgsB.Minimize(x =>
        {
            double f = 0;
            var g = new double[2];
            double[] a = { 1.0, 2.0 };
            for (int i = 0; i < 2; i++)
            {
                double d = x[i] - a[i];
                f += d * d;
                g[i] = 2 * d;
            }
            return (f, g);
        }, new double[] { 0, 0 }, new[] { (-5.0, 5.0), (-5.0, 1.5) });

        res.WarnFlag.Should().Be(0);
        res.X[0].Should().BeApproximately(1.0, 1e-6);
        res.X[1].Should().BeApproximately(1.5, 1e-9, "the second coordinate is pinned to its upper bound");
    }

    [TestMethod]
    public void Minimize_IsDeterministic()
    {
        Func<double[], (double, double[])> f = x =>
            (x[0] * x[0] + 3 * x[1] * x[1], new[] { 2 * x[0], 6 * x[1] });
        LBfgsB.Result a = LBfgsB.Minimize(f, new double[] { 5, -4 });
        LBfgsB.Result b = LBfgsB.Minimize(f, new double[] { 5, -4 });
        a.X.Should().BeEquivalentTo(b.X);
        a.F.Should().Be(b.F);
        a.NumFuncEvals.Should().Be(b.NumFuncEvals);
    }
}
