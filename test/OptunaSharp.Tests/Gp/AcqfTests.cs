using System;
using AwesomeAssertions;
using OptunaSharp.Distributions;
using OptunaSharp.Gp;
using OptunaSharp.Internal;

namespace OptunaSharp.Tests.Gp;

/// <summary>
///     Unit coverage for the GP LogEI acquisition function (PORTING_PLAN Phase 3.2 part 3b —
///     <c>optuna/_gp/acqf.py</c>): <c>standard_logei</c>/<c>logei</c>, <see cref="Erf.Erfcx" />, and
///     <see cref="LogEI" />'s batched evaluation plus the hand-derived gradient that replaces
///     autograd. Scalar values are pinned to live-Optuna float64 figures; the gradient is checked
///     against a central finite difference (oracle-free). Byte-tolerance parity vs live autograd lives
///     in the interop <c>GpAcqfParityTests</c>.
/// </summary>
[TestClass]
public class AcqfTests
{
    [TestMethod]
    public void Erfcx_MatchesReference()
    {
        Erf.Erfcx(2.5).Should().BeApproximately(0.2108063640611436, 1e-13);
        Erf.Erfcx(20.0).Should().BeApproximately(0.028174348741051323, 1e-15);
        // erfcx(-x) = 2 exp(x²) - erfcx(x).
        Erf.Erfcx(-1.5).Should().BeApproximately(2.0 * Math.Exp(2.25) - Erf.Erfcx(1.5), 1e-12);
    }

    [TestMethod]
    public void StandardLogEi_MatchesOptuna()
    {
        Acqf.StandardLogEi(3.0).Should().BeApproximately(1.0987396653277077, 1e-13);
        Acqf.StandardLogEi(0.5).Should().BeApproximately(-0.35982768374506374, 1e-13);
        Acqf.StandardLogEi(-1.0).Should().BeApproximately(-2.4851210257126413, 1e-13);
        Acqf.StandardLogEi(-5.0).Should().BeApproximately(-16.744301162660967, 1e-13);
        // z < -25 branch (erfcx form).
        Acqf.StandardLogEi(-30.0).Should().BeApproximately(-457.7246537605979, 1e-9);
        Acqf.StandardLogEi(-100.0).Should().BeApproximately(-5010.129578800247, 1e-8);
    }

    [TestMethod]
    public void LogEi_MatchesOptuna()
        => Acqf.LogEi(1.2, 0.3, 0.5).Should().BeApproximately(-0.3200331084864203, 1e-13);

    private static (GPRegressor Gpr, LogEI Acqf) FitScenario()
    {
        var x = new double[,] { { 0.1 }, { 0.35 }, { 0.5 }, { 0.72 }, { 0.9 }, { 0.25 }, { 0.6 }, { 0.8 } };
        var y = new[]
        {
            0.7306317300743783, 1.268428212469242, 0.2936730029150271, -1.1446831397092652,
            -0.849223481972128, 1.418560466127369, -0.5033619930308116, -1.2140247968738114
        };
        GPRegressor gpr = GPRegressor.FitKernelParams(x, y, new[] { false }, 1e-6, false, null, 1e-2);
        var space = new System.Collections.Generic.Dictionary<string, BaseDistribution> { ["x"] = new FloatDistribution(0.0, 1.0) };
        return (gpr, new LogEI(gpr, new GpSearchSpace(space), threshold: 1.418560466127369));
    }

    [TestMethod]
    public void EvalAcqfNoGrad_MatchesOptuna()
    {
        (_, LogEI acqf) = FitScenario();
        double[] vals = acqf.EvalAcqfNoGrad(new double[,] { { 0.15 }, { 0.45 } });
        // From live Optuna (the C# fit matches Optuna to ~1e-14, so eval matches to a tolerance).
        vals[0].Should().BeApproximately(-39.457491396443004, 1e-5);
        vals[1].Should().BeApproximately(-233.17606833707896, 1e-4);
    }

    [TestMethod]
    public void EvalAcqfWithGrad_Gradient_MatchesFiniteDifference()
    {
        (_, LogEI acqf) = FitScenario();
        const double eps = 1e-7;
        foreach (double xv in new[] { 0.15, 0.3, 0.45, 0.7 })
        {
            (double _, double[] grad) = acqf.EvalAcqfWithGrad(new[] { xv });
            double fPlus = acqf.EvalAcqfNoGrad(new double[,] { { xv + eps } })[0];
            double fMinus = acqf.EvalAcqfNoGrad(new double[,] { { xv - eps } })[0];
            double fd = (fPlus - fMinus) / (2 * eps);
            // The acqf is steep here, so compare relative to the magnitude.
            grad[0].Should().BeApproximately(fd, 1e-4 * (1 + Math.Abs(fd)),
                $"analytic ∂LogEI/∂x at {xv} must match the central difference");
        }
    }

    [TestMethod]
    public void EvalAcqf_NegInfThreshold_IsZero()
    {
        (GPRegressor gpr, _) = FitScenario();
        var space = new System.Collections.Generic.Dictionary<string, BaseDistribution> { ["x"] = new FloatDistribution(0.0, 1.0) };
        var acqf = new LogEI(gpr, new GpSearchSpace(space), threshold: double.NegativeInfinity);
        acqf.EvalAcqfNoGrad(new double[,] { { 0.2 }, { 0.6 } }).Should().Equal(0.0, 0.0);
        (double val, double[] grad) = acqf.EvalAcqfWithGrad(new[] { 0.4 });
        val.Should().Be(0.0);
        grad.Should().Equal(0.0);
    }
}
