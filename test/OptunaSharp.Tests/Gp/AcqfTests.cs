using System;
using AwesomeAssertions;
using OptunaSharp.Distributions;
using OptunaSharp.Gp;
using OptunaSharp.Internal;

namespace OptunaSharp.Tests.Gp;

/// <summary>
///     Unit coverage for the GP acquisition functions (PORTING_PLAN Phase 3.2 parts 3b/3e —
///     <c>optuna/_gp/acqf.py</c>): <c>standard_logei</c>/<c>logei</c>, <see cref="Erf.Erfcx" />, and
///     <see cref="LogEI" />'s batched evaluation plus the hand-derived gradient that replaces
///     autograd (part 3b); and the constrained single-objective acqfs <see cref="LogPI" /> and
///     <see cref="LogCEI" /> = LogEI + Σ LogPI (part 3e-2). Scalar values are pinned to live-Optuna
///     float64 figures; gradients are checked against a central finite difference (oracle-free), and
///     the LogCEI composition is asserted exact. Byte-tolerance parity vs live autograd lives in the
///     interop <c>GpAcqfParityTests</c>.
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

    // ---- LogPI / LogCEI (Phase 3.2 part 3e-2 — constrained single-objective) -------------------

    [TestMethod]
    public void LogPI_BatchMatchesSinglePoint()
    {
        (GPRegressor gpr, LogEI ei) = FitScenario();
        var pi = new LogPI(gpr, ei.SearchSpace, threshold: 0.3);
        double[] batch = pi.EvalAcqfNoGrad(new double[,] { { 0.15 }, { 0.45 }, { 0.7 } });
        batch[0].Should().Be(pi.EvalAcqfNoGrad(new[] { 0.15 }));
        batch[1].Should().Be(pi.EvalAcqfNoGrad(new[] { 0.45 }));
        batch[2].Should().Be(pi.EvalAcqfNoGrad(new[] { 0.7 }));
    }

    [TestMethod]
    public void LogPI_Gradient_MatchesFiniteDifference()
    {
        (GPRegressor gpr, LogEI ei) = FitScenario();
        var pi = new LogPI(gpr, ei.SearchSpace, threshold: 0.3);
        const double eps = 1e-7;
        foreach (double xv in new[] { 0.15, 0.3, 0.45, 0.7 })
        {
            (double _, double[] grad) = pi.EvalAcqfWithGrad(new[] { xv });
            double fPlus = pi.EvalAcqfNoGrad(new double[,] { { xv + eps } })[0];
            double fMinus = pi.EvalAcqfNoGrad(new double[,] { { xv - eps } })[0];
            double fd = (fPlus - fMinus) / (2 * eps);
            grad[0].Should().BeApproximately(fd, 1e-4 * (1 + Math.Abs(fd)),
                $"analytic ∂LogPI/∂x at {xv} must match the central difference");
        }
    }

    [TestMethod]
    public void LogCEI_EqualsLogEiPlusSumLogPi()
    {
        // LogCEI(x) = LogEI(x) + Σ LogPI_c(x) — the composition must be exact (same float ops).
        (GPRegressor gpr, LogEI ei) = FitScenario();
        GpSearchSpace space = ei.SearchSpace;
        var pi = new LogPI(gpr, space, threshold: 0.2);
        var cei = new LogCEI(gpr, space, threshold: 1.418560466127369, new[] { gpr }, new[] { 0.2 });
        foreach (double xv in new[] { 0.15, 0.45, 0.7 })
        {
            cei.EvalAcqfNoGrad(new[] { xv }).Should()
                .Be(ei.EvalAcqfNoGrad(new[] { xv }) + pi.EvalAcqfNoGrad(new[] { xv }));

            (double vEi, double[] gEi) = ei.EvalAcqfWithGrad(new[] { xv });
            (double vPi, double[] gPi) = pi.EvalAcqfWithGrad(new[] { xv });
            (double vCei, double[] gCei) = cei.EvalAcqfWithGrad(new[] { xv });
            vCei.Should().Be(vEi + vPi);
            gCei[0].Should().Be(gEi[0] + gPi[0]);
        }
    }

    [TestMethod]
    public void LogCEI_NegInfThreshold_IgnoresObjectiveTerm()
    {
        // No feasible trial ⇒ objective threshold −∞ ⇒ the LogEI term is 0, leaving only constraints.
        (GPRegressor gpr, LogEI ei) = FitScenario();
        GpSearchSpace space = ei.SearchSpace;
        var pi = new LogPI(gpr, space, threshold: 0.2);
        var cei = new LogCEI(gpr, space, threshold: double.NegativeInfinity, new[] { gpr }, new[] { 0.2 });
        cei.EvalAcqfNoGrad(new[] { 0.4 }).Should().Be(pi.EvalAcqfNoGrad(new[] { 0.4 }));
    }

    [TestMethod]
    public void LogCEI_EmptyConstraints_Throws()
    {
        (GPRegressor gpr, LogEI ei) = FitScenario();
        var act = () => new LogCEI(gpr, ei.SearchSpace, 0.0,
            System.Array.Empty<GPRegressor>(), System.Array.Empty<double>());
        act.Should().Throw<System.ArgumentException>();
    }

    // ---- LogEHVI (Phase 3.2 part 3e-4 — multi-objective) ---------------------------------------

    private static (LogEHVI Acqf, GpSearchSpace Space) FitEhviScenario()
    {
        var x = new double[,] { { 0.1 }, { 0.35 }, { 0.5 }, { 0.72 }, { 0.9 }, { 0.25 }, { 0.6 }, { 0.8 } };
        var y0 = new[] { 0.73, 1.26, 0.29, -1.14, -0.84, 1.41, -0.50, -1.21 };
        var y1 = new[] { -0.90, 0.40, 1.10, 0.20, -1.30, -0.10, 1.35, 0.55 };
        var gpr0 = GPRegressor.FitKernelParams(x, y0, new[] { false }, 1e-6, false, null, 1e-2);
        var gpr1 = GPRegressor.FitKernelParams(x, y1, new[] { false }, 1e-6, false, null, 1e-2);
        var space = new GpSearchSpace(new System.Collections.Generic.Dictionary<string, BaseDistribution>
            { ["x"] = new FloatDistribution(0.0, 1.0) });
        var yTrain = new double[8, 2];
        for (int i = 0; i < 8; i++) { yTrain[i, 0] = y0[i]; yTrain[i, 1] = y1[i]; }
        return (new LogEHVI(new[] { gpr0, gpr1 }, space, yTrain, nQmcSamples: 128, qmcSeed: 42), space);
    }

    [TestMethod]
    public void LogEHVI_BatchMatchesSinglePoint()
    {
        (LogEHVI acqf, _) = FitEhviScenario();
        double[] batch = acqf.EvalAcqfNoGrad(new double[,] { { 0.15 }, { 0.45 }, { 0.7 } });
        batch[0].Should().Be(acqf.EvalAcqfNoGrad(new[] { 0.15 }));
        batch[1].Should().Be(acqf.EvalAcqfNoGrad(new[] { 0.45 }));
        batch[2].Should().Be(acqf.EvalAcqfNoGrad(new[] { 0.7 }));
    }

    [TestMethod]
    public void LogEHVI_Gradient_MatchesFiniteDifference()
    {
        (LogEHVI acqf, _) = FitEhviScenario();
        const double eps = 1e-7;
        foreach (double xv in new[] { 0.15, 0.3, 0.45, 0.7, 0.85 })
        {
            (double _, double[] grad) = acqf.EvalAcqfWithGrad(new[] { xv });
            double fPlus = acqf.EvalAcqfNoGrad(new double[,] { { xv + eps } })[0];
            double fMinus = acqf.EvalAcqfNoGrad(new double[,] { { xv - eps } })[0];
            double fd = (fPlus - fMinus) / (2 * eps);
            grad[0].Should().BeApproximately(fd, 1e-4 * (1 + Math.Abs(fd)),
                $"analytic ∂LogEHVI/∂x at {xv} must match the central difference");
        }
    }

    [TestMethod]
    public void LogEHVI_ValueAndGrad_ForwardsAgree()
    {
        // The value returned by EvalAcqfWithGrad must equal EvalAcqfNoGrad at the same point.
        (LogEHVI acqf, _) = FitEhviScenario();
        foreach (double xv in new[] { 0.2, 0.55, 0.9 })
        {
            (double val, _) = acqf.EvalAcqfWithGrad(new[] { xv });
            val.Should().BeApproximately(acqf.EvalAcqfNoGrad(new[] { xv }), 1e-12);
        }
    }

    // ---- LogCEHVI (Phase 3.2 part 3e-5 — multi-objective constrained) --------------------------

    private static (GPRegressor[] Gprs, GpSearchSpace Space, double[,] YTrain, GPRegressor ConstraintGpr) FitCehviScenario()
    {
        var x = new double[,] { { 0.1 }, { 0.35 }, { 0.5 }, { 0.72 }, { 0.9 }, { 0.25 }, { 0.6 }, { 0.8 } };
        var y0 = new[] { 0.73, 1.26, 0.29, -1.14, -0.84, 1.41, -0.50, -1.21 };
        var y1 = new[] { -0.90, 0.40, 1.10, 0.20, -1.30, -0.10, 1.35, 0.55 };
        var yc = new[] { 0.30, -0.70, 0.90, -0.20, 1.10, -0.40, 0.15, -0.95 };
        var gpr0 = GPRegressor.FitKernelParams(x, y0, new[] { false }, 1e-6, false, null, 1e-2);
        var gpr1 = GPRegressor.FitKernelParams(x, y1, new[] { false }, 1e-6, false, null, 1e-2);
        var cgpr = GPRegressor.FitKernelParams(x, yc, new[] { false }, 1e-6, false, null, 1e-2);
        var space = new GpSearchSpace(new System.Collections.Generic.Dictionary<string, BaseDistribution>
            { ["x"] = new FloatDistribution(0.0, 1.0) });
        var yTrain = new double[8, 2];
        for (int i = 0; i < 8; i++) { yTrain[i, 0] = y0[i]; yTrain[i, 1] = y1[i]; }
        return (new[] { gpr0, gpr1 }, space, yTrain, cgpr);
    }

    [TestMethod]
    public void LogCEHVI_EqualsLogEhviPlusSumLogPi()
    {
        (GPRegressor[] gprs, GpSearchSpace space, double[,] yTrain, GPRegressor cgpr) = FitCehviScenario();
        var ehvi = new LogEHVI(gprs, space, yTrain, nQmcSamples: 128, qmcSeed: 42);
        var pi = new LogPI(cgpr, space, threshold: 0.1);
        var cehvi = new LogCEHVI(gprs, space, yTrain, 128, 42, new[] { cgpr }, new[] { 0.1 });
        foreach (double xv in new[] { 0.2, 0.55, 0.9 })
        {
            cehvi.EvalAcqfNoGrad(new[] { xv }).Should()
                .BeApproximately(ehvi.EvalAcqfNoGrad(new[] { xv }) + pi.EvalAcqfNoGrad(new[] { xv }), 1e-12);
        }
    }

    [TestMethod]
    public void LogCEHVI_NoFeasible_IsConstraintsOnly()
    {
        // Y_feasible = null (all infeasible) ⇒ the LogEHVI term is dropped, leaving only the constraints.
        (GPRegressor[] gprs, GpSearchSpace space, _, GPRegressor cgpr) = FitCehviScenario();
        var pi = new LogPI(cgpr, space, threshold: 0.1);
        var cehvi = new LogCEHVI(gprs, space, yFeasible: null, 128, 42, new[] { cgpr }, new[] { 0.1 });
        foreach (double xv in new[] { 0.3, 0.7 })
            cehvi.EvalAcqfNoGrad(new[] { xv }).Should().Be(pi.EvalAcqfNoGrad(new[] { xv }));
    }
}
